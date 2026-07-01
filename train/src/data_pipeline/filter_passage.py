"""Filter for passage-buried answers: answers hidden in dense prose text.

Removes queries whose answers likely come from:
- Tables (statistics, scores, dates in columns)
- Infoboxes (birth/death, structured sidebar data)
- Lists (bullet points, award lists, episode lists)

Keeps queries whose answers require reading through paragraph text:
- Narrative details ("who did X do after Y?")
- Descriptions ("what nickname does...")
- Explanations ("why did X...", "according to...")
- Events embedded in prose ("what happened when...")

Uses GPT-4.1 with a passage-vs-structured classification prompt.
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

SYSTEM = """You classify queries for a visual document retrieval dataset.

Task: For each query, predict whether the answer is most likely found in:
  P = Dense paragraph/prose text (requires reading comprehension)
  S = Structured data: table, list, infobox, sidebar, statistics

Examples:
  "Who was demoted by the Emperor after the battle?" → P (narrative in prose)
  "What nickname does Wendy call Misha?" → P (character description paragraph)
  "According to the report, why was the project cancelled?" → P (explanation in text)
  "What role did she play in establishing the university?" → P (narrative)
  "What caused the collapse of the bridge?" → P (explanation)
  "What was the population in the 2016 census?" → S (infobox/table)
  "How many goals did X score?" → S (statistics table)
  "What date was X released?" → S (infobox field)
  "Who won the award in 2014?" → S (awards list)
  "What score did the film get on Rotten Tomatoes?" → S (review aggregator number)
  "On what date did X happen?" → S (usually a date field, not prose)
  "How much money/revenue/population..." → S (structured number)

Borderline cases:
  "On what date did X marry Y?" → could be either. If it's a notable person's bio, likely P (marriage described in narrative). Mark P.
  "What year did X join the company?" → likely S if infobox, but P if in career section narrative. When unsure, mark P.
  "Which award did X win for Y?" → S if awards table, but P if mentioned in career narrative. When unsure, mark S.

Be thoughtful. Respond with ONLY a JSON array of "P" or "S" strings, one per query."""

USER_TPL = """Classify {n} queries as P (passage) or S (structured). JSON array only.

{block}"""


async def classify_batch(client, queries, batch_idx, sem):
    async with sem:
        block = "\n".join(f"[{i}] {q}" for i, q in enumerate(queries))
        for attempt in range(3):
            try:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": "gpt-4.1-mini",
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": USER_TPL.format(n=len(queries), block=block),
                            },
                        ],
                        "temperature": 0,
                        "max_tokens": 400,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                labels = json.loads(content)
                if isinstance(labels, list):
                    while len(labels) < len(queries):
                        labels.append("S")
                    return batch_idx, [
                        str(x).upper().strip('"') for x in labels[: len(queries)]
                    ]
                return batch_idx, ["S"] * len(queries)
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Batch {batch_idx}: {e}")
                    return batch_idx, ["S"] * len(queries)
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

    logger.info(f"Loading {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    N = len(records)
    logger.info(f"Loaded {N} records")

    # Classify all queries
    queries = [r["query"] for r in records]
    batches = []
    for i in range(0, len(queries), args.batch_size):
        batches.append(queries[i : i + args.batch_size])
    logger.info(f"{len(batches)} batches")

    labels_all = ["S"] * N
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=90,
    ) as client:
        tasks = [classify_batch(client, b, bi, sem) for bi, b in enumerate(batches)]
        done = 0
        t0 = time.time()
        for coro in asyncio.as_completed(tasks):
            bi, labels = await coro
            start = bi * args.batch_size
            for j, lbl in enumerate(labels):
                if start + j < N:
                    labels_all[start + j] = lbl
            done += 1
            if done % 100 == 0 or done == len(tasks):
                p_count = sum(1 for x in labels_all if x.startswith("P"))
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 1
                eta = (len(tasks) - done) / rate
                logger.info(
                    f"{done}/{len(tasks)}, P={p_count} ({p_count / N * 100:.1f}%), ETA {eta:.0f}s"
                )

    p_count = sum(1 for x in labels_all if x.startswith("P"))
    s_count = N - p_count
    logger.info(
        f"\nClassification: P={p_count} ({p_count / N * 100:.1f}%), S={s_count} ({s_count / N * 100:.1f}%)"
    )

    # Write passage-only
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kept = 0
    with open(args.output, "w") as f:
        for i, rec in enumerate(records):
            if labels_all[i].startswith("P"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    logger.info(f"Wrote {kept} passage records to {args.output}")

    # Chunk position breakdown
    from collections import Counter

    def parse_pos(path):
        m = re.search(r"chunk_(\d+)_(\d+)\.png", path)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    p_pos = Counter()
    s_pos = Counter()
    for i in range(N):
        p, c = parse_pos(records[i]["chunk_path"])
        bucket = (
            "c0" if c == 0 else ("c1" if c == 1 else ("c2-3" if c <= 3 else "c4+/p1+"))
        )
        if p >= 1:
            bucket = "c4+/p1+"
        if labels_all[i].startswith("P"):
            p_pos[bucket] += 1
        else:
            s_pos[bucket] += 1

    logger.info("Passage (P) by position:")
    for k in ["c0", "c1", "c2-3", "c4+/p1+"]:
        logger.info(f"  {k}: {p_pos.get(k, 0)}")
    logger.info("Structured (S) by position:")
    for k in ["c0", "c1", "c2-3", "c4+/p1+"]:
        logger.info(f"  {k}: {s_pos.get(k, 0)}")

    # Save sample of each for visual verification
    import random

    random.seed(42)
    p_indices = [i for i in range(N) if labels_all[i].startswith("P")]
    s_indices = [i for i in range(N) if labels_all[i].startswith("S")]

    for tag, indices in [("passage", p_indices), ("structured", s_indices)]:
        sample = random.sample(indices, min(100, len(indices)))
        sf = args.output.replace(".jsonl", f"_sample_{tag}.jsonl")
        with open(sf, "w") as f:
            for i in sample:
                f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
        logger.info(f"Sample: {sf}")


if __name__ == "__main__":
    asyncio.run(main())
