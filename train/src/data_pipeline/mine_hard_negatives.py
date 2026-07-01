#!/usr/bin/env python3
"""Mine hard negatives using the local search API.

For each (query, positive_chunk) training pair, queries the search API to find
top-K retrieval results. Non-positive results become hard negatives — these are
the actual mistakes the retrieval system makes in production.

Usage:
    python mine_hard_negatives.py \
        --input training/data/train.jsonl \
        --output training/data/train_hn.jsonl \
        --num-negatives 7

    # With more candidates (slower but better negatives)
    python mine_hard_negatives.py \
        --input training/data/train.jsonl \
        --output training/data/train_hn.jsonl \
        --num-negatives 7 --n-docs 50
"""

import argparse
import json
import logging
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

SEARCH_URL = "http://localhost:30888/search"


def search_batch(
    queries: list[str], n_docs: int = 20, nprobe: int = 128
) -> list[list[dict]]:
    """Query the search API with a batch of text queries."""
    payload = {
        "queries": [{"text": q} for q in queries],
        "n_docs": n_docs,
        "nprobe": nprobe,
    }
    resp = requests.post(SEARCH_URL, json=payload, timeout=120)
    resp.raise_for_status()
    results = resp.json()["results"]
    return [r["hits"] for r in results]


def mine_from_search(
    pairs: list[dict],
    num_negatives: int = 7,
    n_docs: int = 20,
    batch_size: int = 64,
    nprobe: int = 128,
    filter_mode: str = "none",
    margin: float | None = None,
) -> list[dict]:
    """Mine hard negatives by querying the search API.

    For each pair, the search API returns top-K results. Any result whose path
    differs from the positive chunk_path is a hard negative.

    Args:
        filter_mode: How to filter likely false negatives:
            - "article": Skip chunks from the same article (by article_id).
                         Keeps hard negatives that score higher than positive.
            - "margin": Skip negatives where neg_score > pos_score * margin.
                        Conservative, but may discard useful hard cases.
            - "skip_top1": Skip the #1 non-positive result (likely false negative),
                          keep #2-#K. Compromise between article and margin.
            - "none": No filtering, keep all non-positive hits.
        margin: Only used when filter_mode="margin". Typical: 0.95-0.98.
    """
    # Deduplicate queries for efficiency
    unique_queries = list(dict.fromkeys(p["query"] for p in pairs))
    query_to_idx = {q: i for i, q in enumerate(unique_queries)}
    logger.info(f"{len(unique_queries)} unique queries (from {len(pairs)} pairs)")

    # Collect all positive paths per unique query
    query_positives = {}
    for pair in pairs:
        q = pair["query"]
        if q not in query_positives:
            query_positives[q] = set()
        query_positives[q].add(pair["chunk_path"])

    # Query search API in batches
    all_hits = [None] * len(unique_queries)
    n_batches = (len(unique_queries) + batch_size - 1) // batch_size

    for i in range(0, len(unique_queries), batch_size):
        batch_queries = unique_queries[i : i + batch_size]
        batch_idx = i // batch_size + 1

        try:
            batch_hits = search_batch(batch_queries, n_docs=n_docs, nprobe=nprobe)
            for j, hits in enumerate(batch_hits):
                all_hits[i + j] = hits
        except Exception as e:
            logger.warning(f"Batch {batch_idx}/{n_batches} failed: {e}")
            for j in range(len(batch_queries)):
                all_hits[i + j] = []

        done = min(i + batch_size, len(unique_queries))
        if batch_idx % 10 == 0 or done == len(unique_queries):
            logger.info(f"  Searched: {done}/{len(unique_queries)}")

    # Build set of positive article_ids per query for false-negative filtering
    query_pos_articles = {}
    for q in unique_queries:
        pos_paths = query_positives[q]
        # Extract article_id from hits that match positive paths
        hits = all_hits[query_to_idx[q]] or []
        article_ids = set()
        for hit in hits:
            if hit.get("path", "") in pos_paths:
                article_ids.add(hit.get("article_id"))
        query_pos_articles[q] = article_ids

    # Extract hard negatives per unique query
    query_negatives = {}
    query_metadata = {}
    stats = {
        "total": 0,
        "with_negs": 0,
        "avg_negs": 0,
        "avg_pos_rank": 0,
        "same_article_filtered": 0,
        "margin_filtered": 0,
        "skip_top1_filtered": 0,
        "harder_than_pos": 0,
    }
    pos_ranks = []
    pos_rank_distribution = {str(i): 0 for i in range(1, n_docs + 1)}
    pos_rank_distribution[f">{n_docs}"] = 0

    for q in unique_queries:
        positives = query_positives[q]
        pos_article_ids = query_pos_articles[q]
        hits = all_hits[query_to_idx[q]] or []

        # Find positive score for stats
        pos_score = None
        for hit in hits:
            if hit.get("path", "") in positives:
                pos_score = hit["score"]
                break

        # Find negatives: exclude positives and same-article chunks (likely false negatives)
        neg_paths = []
        pos_rank = None
        for rank, hit in enumerate(hits):
            hit_path = hit.get("path", "")
            if hit_path in positives:
                if pos_rank is None:
                    pos_rank = rank
            else:
                # Filter likely false negatives
                if filter_mode == "article":
                    if hit.get("article_id") in pos_article_ids:
                        stats["same_article_filtered"] += 1
                        continue
                elif (
                    filter_mode == "margin"
                    and margin is not None
                    and pos_score is not None
                ):
                    if hit["score"] > pos_score * margin:
                        stats["margin_filtered"] += 1
                        continue
                elif filter_mode == "skip_top1":
                    if len(neg_paths) == 0 and rank < 5:
                        # Skip the first non-positive hit (likely false negative)
                        stats["skip_top1_filtered"] += 1
                        continue
                # filter_mode == "none": no filtering

                if len(neg_paths) < num_negatives:
                    neg_paths.append(hit_path)
                    if pos_score is not None and hit["score"] >= pos_score:
                        stats["harder_than_pos"] += 1

        query_negatives[q] = neg_paths
        query_metadata[q] = {
            "retrieve_top20": [
                {
                    "rank": rank + 1,
                    "path": hit.get("path", ""),
                    "score": hit.get("score", 0.0),
                    "article_id": hit.get("article_id"),
                }
                for rank, hit in enumerate(hits)
            ],
            "positive_score": pos_score if pos_score is not None else 0.0,
            "positive_rank": pos_rank + 1 if pos_rank is not None else 0,
        }
        stats["total"] += 1
        if neg_paths:
            stats["with_negs"] += 1
        if pos_rank is not None:
            pos_ranks.append(pos_rank)
            pos_rank_distribution[str(pos_rank + 1)] += 1
        else:
            pos_rank_distribution[f">{n_docs}"] += 1

    # Write output
    output_pairs = []
    for pair in pairs:
        neg_paths = query_negatives.get(pair["query"], [])
        meta = query_metadata.get(pair["query"], {})
        output_pairs.append(
            {
                **pair,
                "neg_chunk_paths": neg_paths,
                "retrieve_top20": meta.get("retrieve_top20", []),
                "positive_score": meta.get("positive_score", 0.0),
                "positive_rank": meta.get("positive_rank", 0),
            }
        )

    # Stats
    total_queries = len(unique_queries)
    stats["avg_negs"] = (
        sum(len(query_negatives[q]) for q in unique_queries) / total_queries
    )
    if pos_ranks:
        stats["avg_pos_rank"] = sum(pos_ranks) / len(pos_ranks)
    stats["pos_found_rate"] = len(pos_ranks) / total_queries
    stats["pos_recall@1"] = sum(1 for r in pos_ranks if r == 0) / total_queries
    stats["pos_recall@10"] = sum(1 for r in pos_ranks if r < 10) / total_queries
    stats["pos_recall@20"] = sum(1 for r in pos_ranks if r < 20) / total_queries
    stats["pos_rank_distribution"] = pos_rank_distribution

    return output_pairs, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input JSONL with {query, chunk_path}"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL with added neg_chunk_paths"
    )
    parser.add_argument("--num-negatives", type=int, default=7)
    parser.add_argument(
        "--n-docs",
        type=int,
        default=20,
        help="Number of docs to retrieve per query from search API",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["article", "margin", "skip_top1", "none"],
        default="none",
        help="False-negative filter: 'article' (skip same article_id), "
        "'margin' (skip by score margin), 'none' (no filter)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.95,
        help="Margin threshold (only for --filter-mode margin). Typical: 0.95-0.98",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for search API queries"
    )
    parser.add_argument(
        "--chunk-path-prefix",
        type=str,
        default="/opt/dlami/nvme/kiwix_tiles/",
        help="Prefix to prepend to relative chunk_path for matching search API results",
    )
    parser.add_argument(
        "--nprobe", type=int, default=128, help="FAISS nprobe for search API"
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Optional JSON path to write mining stats",
    )
    args = parser.parse_args()

    # Check search API
    try:
        resp = requests.get("http://localhost:30888/health", timeout=5)
        resp.raise_for_status()
        logger.info("Search API is healthy")
    except Exception as e:
        logger.error(f"Search API not available: {e}")
        sys.exit(1)

    # Load data
    pairs = []
    with open(args.input) as f:
        for line in f:
            pair = json.loads(line)
            # Normalize chunk_path to absolute path for matching search API results
            if args.chunk_path_prefix and not pair["chunk_path"].startswith("/"):
                pair["chunk_path"] = args.chunk_path_prefix + pair["chunk_path"]
            pairs.append(pair)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Mine
    t0 = time.time()
    output_pairs, stats = mine_from_search(
        pairs,
        num_negatives=args.num_negatives,
        n_docs=args.n_docs,
        batch_size=args.batch_size,
        nprobe=args.nprobe,
        filter_mode=args.filter_mode,
        margin=args.margin,
    )
    elapsed = time.time() - t0

    # Write
    with open(args.output, "w") as f:
        for pair in output_pairs:
            f.write(json.dumps(pair) + "\n")

    n_with_negs = sum(1 for p in output_pairs if p["neg_chunk_paths"])
    logger.info(f"Wrote {len(output_pairs)} pairs to {args.output}")
    logger.info(
        f"  {n_with_negs} with negatives ({n_with_negs / len(output_pairs):.1%})"
    )
    logger.info(f"  Avg negatives per query: {stats['avg_negs']:.1f}")
    logger.info(f"  Avg positive rank: {stats.get('avg_pos_rank', 'N/A')}")
    if "pos_recall@1" in stats:
        logger.info(f"  Search API recall@1: {stats['pos_recall@1']:.3f}")
        logger.info(f"  Search API recall@10: {stats['pos_recall@10']:.3f}")
        logger.info(f"  Search API recall@20: {stats['pos_recall@20']:.3f}")
    if stats.get("same_article_filtered"):
        logger.info(f"  Same-article filtered: {stats['same_article_filtered']}")
    if stats.get("margin_filtered"):
        logger.info(f"  Margin-filtered: {stats['margin_filtered']}")
    if stats.get("skip_top1_filtered"):
        logger.info(f"  Skip-top1 filtered: {stats['skip_top1_filtered']}")
    if stats.get("harder_than_pos"):
        logger.info(f"  Negatives harder than positive: {stats['harder_than_pos']}")
    logger.info(f"  Positive rank distribution: {stats['pos_rank_distribution']}")
    logger.info(f"  Time: {elapsed:.0f}s")

    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        logger.info(f"Wrote stats to {args.stats_output}")


if __name__ == "__main__":
    main()
