"""Strict multi-pass filter for chunk-discrimination training data.

Strategy:
  Pass 1 (rule-based):
    - Auto-KEEP queries whose positive chunk is deep in the article (c5+ or page 1+)
    - Auto-REMOVE queries that are obviously generic ("What is X?", "X is known for")
  Pass 2 (LLM, strict):
    - Classify remaining queries with a very strict prompt
    - Only keep queries that require reading a SPECIFIC section/table/list
  Pass 3 (visual sample):
    - Spot-check images of kept entries

Target: ~20% of original (~20K from 104K)
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Pass 1: rule-based ----

GENERIC_PATTERNS = [
    r"^what is [\w\s]+\??$",
    r"^who is [\w\s]+\??$",
    r"^what does [\w\s]+ mean\??$",
    r"^describe ",
    r"^tell me about ",
    r"what is .+ known for",
    r"what type of .+ is",
    r"what kind of .+ is",
    r"^where is [\w\s]+( located)?\??$",
    r"^when was [\w\s]+ (born|founded|created|established)\??$",
]
GENERIC_RE = [re.compile(p, re.IGNORECASE) for p in GENERIC_PATTERNS]


def parse_chunk_position(chunk_path):
    """Extract (page, chunk_idx) from chunk path."""
    match = re.search(r"chunk_(\d+)_(\d+)\.png", chunk_path)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def rule_filter(record):
    """Returns 'keep', 'remove', or 'classify'."""
    page, chunk = parse_chunk_position(record["chunk_path"])
    query = record["query"].strip()

    # Deep chunks → auto-keep (answer must be in specific section)
    if page >= 1 or chunk >= 5:
        return "keep"

    # Obviously generic queries → auto-remove
    for pat in GENERIC_RE:
        if pat.search(query):
            return "remove"

    # Very short queries are usually generic
    if len(query.split()) < 5:
        return "remove"

    return "classify"


# ---- Pass 2: strict LLM classification ----

STRICT_SYSTEM = """You are a strict data quality filter. We want ONLY the most specific, fact-dense queries.

Context: We train a visual document retrieval model on (query, Wikipedia_chunk_image) pairs.
Our #1 problem: the model retrieves the intro/overview chunk instead of the chunk containing the answer.
We need training data where the query FORCES the model to find a specific chunk deep in the article.

## KEEP (mark as 1) — ONLY if ALL of these are true:
- The answer is a SPECIFIC fact: a number, date, name from a list, statistic, episode number, score
- The answer would NOT be in the first paragraph or infobox of the article
- The query references a specific context that narrows to a particular section
  (e.g., "in the 2014 season", "episode 7", "the 1951 recipient", "during World War II")
- A reader would need to scan through tables, lists, or deep sections to find the answer

## REMOVE (mark as 0) — if ANY of these are true:
- The query asks about the main topic/subject of an article (answerable from intro)
- The query asks for basic biographical info (birth/death, nationality, occupation)
- The query asks "what is X" or "who is X" style overview questions
- The answer is likely in the first few sentences or the infobox
- The query asks about the founding, origin, or general description of something
- The query could be answered by someone who only read the Wikipedia abstract

Be VERY strict. When in doubt, mark as 0. We want to keep only ~25% of queries.

Respond with a JSON array of 0s and 1s, one per query.
Example for 5 queries: [0, 1, 0, 0, 1]"""

USER_TEMPLATE = """Rate these {n} queries (0=remove, 1=keep). Respond with JSON array only.

{queries_block}"""


async def classify_batch(client, queries, batch_idx, semaphore, model="gpt-4.1-mini"):
    """Classify a batch, returns list of 0/1."""
    async with semaphore:
        queries_block = "\n".join(f"[{i}] {q}" for i, q in enumerate(queries))
        user_msg = USER_TEMPLATE.format(n=len(queries), queries_block=queries_block)

        for attempt in range(3):
            try:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": STRICT_SYSTEM},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0,
                        "max_tokens": 300,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                labels = json.loads(content)
                if isinstance(labels, list) and len(labels) == len(queries):
                    return batch_idx, [int(bool(x)) for x in labels]
                # Fallback: if wrong length, try to pad/truncate
                if isinstance(labels, list):
                    labels = labels[: len(queries)]
                    while len(labels) < len(queries):
                        labels.append(0)
                    return batch_idx, [int(bool(x)) for x in labels]
                return batch_idx, [0] * len(queries)
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Batch {batch_idx} failed: {e}")
                    return batch_idx, [0] * len(queries)
                await asyncio.sleep(2**attempt)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://us.api.openai.com/v1")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Need --api-key or OPENAI_API_KEY")

    # Load
    logger.info(f"Loading {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records")

    # ---- Pass 1: Rule-based ----
    auto_keep = []
    auto_remove = []
    to_classify = []

    for i, r in enumerate(records):
        decision = rule_filter(r)
        if decision == "keep":
            auto_keep.append(i)
        elif decision == "remove":
            auto_remove.append(i)
        else:
            to_classify.append(i)

    logger.info(
        f"Pass 1: auto-keep={len(auto_keep)}, auto-remove={len(auto_remove)}, "
        f"to-classify={len(to_classify)}"
    )

    # ---- Pass 2: LLM classification ----
    classify_queries = [(idx, records[idx]["query"]) for idx in to_classify]

    batches = []
    for i in range(0, len(classify_queries), args.batch_size):
        batch = classify_queries[i : i + args.batch_size]
        batches.append(batch)
    logger.info(f"Pass 2: {len(batches)} batches of ~{args.batch_size}")

    llm_keep = set()
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=90,
    ) as client:
        tasks = []
        for batch_idx, batch in enumerate(batches):
            queries_only = [q for _, q in batch]
            tasks.append(classify_batch(client, queries_only, batch_idx, semaphore))

        t0 = time.time()
        done = 0
        for coro in asyncio.as_completed(tasks):
            batch_idx, labels = await coro
            batch = batches[batch_idx]
            for (orig_idx, _), label in zip(batch, labels):
                if label == 1:
                    llm_keep.add(orig_idx)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                logger.info(
                    f"Pass 2: {done}/{len(tasks)} batches, "
                    f"{len(llm_keep)} LLM-kept ({len(llm_keep) / len(to_classify) * 100:.1f}%), "
                    f"ETA: {eta:.0f}s"
                )

    # Combine
    all_keep = set(auto_keep) | llm_keep
    logger.info(
        f"\nFinal: {len(all_keep)}/{len(records)} "
        f"= {len(all_keep) / len(records) * 100:.1f}%"
    )
    logger.info(f"  auto-keep (deep chunks): {len(auto_keep)}")
    logger.info(f"  LLM-kept: {len(llm_keep)}")

    # Chunk position breakdown of kept records
    from collections import Counter

    pos_counts = Counter()
    for i in all_keep:
        p, c = parse_chunk_position(records[i]["chunk_path"])
        pos_counts[f"p{p}_c{c}"] += 1
    logger.info("Kept records by chunk position (top 15):")
    for pos, cnt in sorted(pos_counts.items(), key=lambda x: -x[1])[:15]:
        logger.info(f"  {pos}: {cnt}")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kept = 0
    with open(args.output, "w") as f:
        for i, rec in enumerate(records):
            if i in all_keep:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    logger.info(f"Wrote {kept} records to {args.output}")

    # Also save a sample for visual inspection
    import random

    random.seed(42)
    sample_indices = random.sample(sorted(all_keep), min(200, len(all_keep)))
    sample_file = args.output.replace(".jsonl", "_sample200.jsonl")
    with open(sample_file, "w") as f:
        for i in sample_indices:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    logger.info(f"Saved 200 sample records to {sample_file}")


if __name__ == "__main__":
    asyncio.run(main())
