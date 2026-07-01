"""Filter training data to keep only entity-locating queries.

Entity-locating queries target specific details within an article chunk:
  - specific dates, years, episode numbers
  - table rows, statistics, award recipients by year
  - specific prices, vote counts, measurements
  - details that live in a particular section/table, not the intro

These are the queries that train the embedding model to discriminate
between chunks of the same article, which is the #1 failure mode
(46% of eval failures are "right article, wrong chunk").

Usage:
    python filter_entity_queries.py \
        --input training/data/natrual_filtered_v2/split/train_hn.jsonl \
        --output training/data/entity_filtered/train_hn.jsonl \
        --target-ratio 0.20
"""

import argparse
import asyncio
import json
import logging
import os
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a data quality filter for a visual document retrieval system.

We train an embedding model on (query, document_chunk) pairs from Wikipedia screenshots.
Our biggest failure mode is retrieving the RIGHT article but WRONG chunk — the overview/intro
chunk ranks highest, but the answer is buried in a specific table, list, or section.

Your job: classify each query as SPECIFIC or GENERAL.

**SPECIFIC** (KEEP) — queries whose answer lives in a particular chunk, NOT the intro:
- Asks for a specific year, date, episode number, season
- Asks for a value from a table (award recipient in year X, stat, score, price)
- Asks for details from a specific section (e.g., "in season 3...", "during the 2014 match...")
- Requires finding a particular row in a list or table
- Asks about a specific event with a date/number anchor
- Asks "who won X award in YEAR", "what was the score in MATCH", "which episode..."

**GENERAL** (REMOVE) — queries answerable from the intro/overview chunk:
- "What is X?" / "What is X known for?"
- Biographical basics (birth, death, nationality, occupation)
- General descriptions of a topic
- Questions about the main subject of the article that would be in the first paragraph
- "Who founded X?" / "When was X established?" if it's THE main topic of the article

Respond with ONLY a JSON array of indices (0-based) of the SPECIFIC queries.
Example: [0, 3, 7, 12]
If none are specific, respond: []"""

USER_TEMPLATE = """Classify these {n} queries. Return JSON array of SPECIFIC query indices only.

{queries_block}"""


async def classify_batch(client, queries, batch_idx, semaphore):
    """Send a batch of queries to GPT-4.1-mini for classification."""
    async with semaphore:
        queries_block = "\n".join(f"[{i}] {q}" for i, q in enumerate(queries))
        user_msg = USER_TEMPLATE.format(n=len(queries), queries_block=queries_block)

        for attempt in range(3):
            try:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": "gpt-4.1-mini",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0,
                        "max_tokens": 500,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Parse JSON array from response
                # Handle cases where model wraps in markdown code block
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                indices = json.loads(content)
                if not isinstance(indices, list):
                    indices = []
                return batch_idx, [
                    i for i in indices if isinstance(i, int) and 0 <= i < len(queries)
                ]
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Batch {batch_idx} failed after 3 attempts: {e}")
                    return batch_idx, []
                await asyncio.sleep(2**attempt)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.20,
        help="Target fraction to keep (approximate)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Queries per API call"
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Max concurrent API calls"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default="https://us.api.openai.com/v1")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Need --api-key or OPENAI_API_KEY env var")

    # Load data
    logger.info(f"Loading {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records")

    queries = [r["query"] for r in records]

    # Batch queries
    batches = []
    for i in range(0, len(queries), args.batch_size):
        batches.append((i, queries[i : i + args.batch_size]))
    logger.info(f"Created {len(batches)} batches of ~{args.batch_size}")

    # Run classification
    semaphore = asyncio.Semaphore(args.concurrency)
    keep_indices = set()

    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=90,
    ) as client:
        tasks = []
        for batch_idx, (start_idx, batch_queries) in enumerate(batches):
            tasks.append(classify_batch(client, batch_queries, batch_idx, semaphore))

        t0 = time.time()
        done = 0
        for coro in asyncio.as_completed(tasks):
            batch_idx, local_indices = await coro
            start_idx = batches[batch_idx][0]
            for li in local_indices:
                keep_indices.add(start_idx + li)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {done}/{len(tasks)} batches "
                    f"({len(keep_indices)} kept so far, "
                    f"{len(keep_indices) / len(records) * 100:.1f}%) "
                    f"ETA: {eta:.0f}s"
                )

    logger.info(
        f"Classification done: {len(keep_indices)}/{len(records)} "
        f"= {len(keep_indices) / len(records) * 100:.1f}% marked SPECIFIC"
    )

    # Write filtered output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kept = 0
    with open(args.output, "w") as f:
        for i, rec in enumerate(records):
            if i in keep_indices:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    logger.info(f"Wrote {kept} records to {args.output}")
    logger.info(f"Ratio: {kept / len(records) * 100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
