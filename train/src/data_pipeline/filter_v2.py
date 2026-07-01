"""V2 filter: chunk position + keyword rules + GPT-4.1 for hard cases.

Tiers:
  Tier 1 (auto-keep): chunk c3+ or page 1+  → ~27K
  Tier 2 (keyword-keep): c0-c2 queries with strong specificity signals → ~15K
  Tier 3 (GPT-4.1 strict): c0-c2 borderline queries → ~5-10K from ~60K
  Tier 4 (auto-remove): c0-c2 generic queries

Target: ~45-55K high-quality records (40-50% of original)
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


def parse_chunk_pos(path):
    m = re.search(r"chunk_(\d+)_(\d+)\.png", path)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# ---- Keyword signals ----

SPECIFIC_KEYWORDS = [
    # Temporal anchors
    r"\b(in |during |as of )\d{4}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d",
    r"\b\d{1,2}(st|nd|rd|th)\s+(of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)",
    # Episode/season
    r"\b(episode|season|series|chapter|volume|issue|part)\s+\d",
    r"\bseason\s+\d+\s+(episode|ep)\s+\d",
    # Rankings/ordinals
    r"\b(first|second|third|fourth|fifth|\d+(st|nd|rd|th))\s+(person|place|team|winner|recipient|player|entry)",
    r"\b(rank|ranked|ranking)\s+\d",
    # Statistics/measurements
    r"\bhow (many|much|long|far|old|tall|high|fast|deep)\b",
    r"\b(score|scored|goals|points|votes|runs|wickets|fouls|assists)\b",
    r"\b(percentage|rate|ratio|proportion|frequency)\b",
    r"\b(population|elevation|area|height|weight|length|distance|speed|temperature)\b",
    r"\b(salary|price|cost|revenue|budget|fine|award|prize)\s+(of|for|in|was)",
    # Specific entities in context
    r"\b(award|prize|medal|trophy|honour)\s+(in|for|winner|recipient)\s+\d{4}",
    r"\b(won|received|awarded|earned)\s+(the|a)\s+\w+\s+(award|prize|medal)",
    r"\bwho (won|received|got|earned)\b.*\b\d{4}\b",
    r"\bwhat (year|date|day|month)\b",
    r"\bwhen did\b.*\b(first|last|final)\b",
    # Table/list signals
    r"\baccording to\b",
    r"\bwhat is the name of the\b.*\b(who|that|which)\b",
    r"\bhex(adecimal)?\s*(code|color|value)\b",
    r"\bboiling point|melting point|molar\b",
    r"\bstrike rate|batting average|bowling\b",
]
SPECIFIC_RE = [re.compile(p, re.IGNORECASE) for p in SPECIFIC_KEYWORDS]

GENERIC_STRONG = [
    r"^what is [\w\s]{3,30}\??$",
    r"^who is [\w\s]{3,30}\??$",
    r"^where is [\w\s]{3,30}\??$",
    r"\bknown for\b",
    r"\bfamous for\b",
    r"\bwhat (type|kind|genre|style) of\b",
    r"^describe\b",
    r"^explain\b",
    r"\bwhat is the (main|primary|official) (language|religion|currency)\b",
    r"\bwhat is the capital of\b",
]
GENERIC_STRONG_RE = [re.compile(p, re.IGNORECASE) for p in GENERIC_STRONG]


def has_specific_signal(query):
    return any(p.search(query) for p in SPECIFIC_RE)


def has_generic_signal(query):
    return any(p.search(query) for p in GENERIC_STRONG_RE)


# ---- GPT-4.1 classification ----

GPT41_SYSTEM = """You filter queries for a visual retrieval training dataset.
We need queries whose answers are in SPECIFIC sections of Wikipedia articles, NOT the introduction.

Rate each query: 1 = answer requires finding a specific fact in a table/list/deep section, 0 = answerable from intro.

STRICT RULES — mark 0 unless the query clearly needs a deep section:
- "What is X?" → 0 (intro)
- "When was X born/founded?" → 0 (infobox)
- "Who is the CEO of X?" → 0 (intro/infobox)
- "What award did X win in 2014?" → 1 (specific year in awards section)
- "How many goals in the 2018 final?" → 1 (match statistics)
- "In episode 7, who..." → 1 (specific episode)
- "What is the boiling point of..." → 1 (properties table)

Respond ONLY with a JSON array of 0s and 1s."""

GPT41_USER = """Rate {n} queries. JSON array of 0/1 only.

{block}"""


async def gpt41_batch(client, queries, batch_idx, semaphore):
    async with semaphore:
        block = "\n".join(f"[{i}] {q}" for i, q in enumerate(queries))
        for attempt in range(3):
            try:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": "gpt-4.1-mini",
                        "messages": [
                            {"role": "system", "content": GPT41_SYSTEM},
                            {
                                "role": "user",
                                "content": GPT41_USER.format(
                                    n=len(queries), block=block
                                ),
                            },
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
                if isinstance(labels, list):
                    while len(labels) < len(queries):
                        labels.append(0)
                    return batch_idx, [int(bool(x)) for x in labels[: len(queries)]]
                return batch_idx, [0] * len(queries)
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Batch {batch_idx}: {e}")
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

    logger.info(f"Loading {args.input}")
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    N = len(records)
    logger.info(f"Loaded {N} records")

    # ---- Tier assignments ----
    tier1_keep = set()  # deep chunks
    tier2_keep = set()  # keyword-specific in c0-c2
    tier4_remove = set()  # clearly generic
    tier3_classify = []  # borderline → LLM

    for i, r in enumerate(records):
        page, chunk = parse_chunk_pos(r["chunk_path"])
        q = r["query"].strip()

        if page >= 1 or chunk >= 3:
            tier1_keep.add(i)
        elif has_generic_signal(q) and not has_specific_signal(q):
            tier4_remove.add(i)
        elif has_specific_signal(q):
            tier2_keep.add(i)
        elif len(q.split()) < 6:
            tier4_remove.add(i)
        else:
            tier3_classify.append(i)

    logger.info(
        f"Tier 1 (deep chunk auto-keep): {len(tier1_keep)} ({len(tier1_keep) / N * 100:.1f}%)"
    )
    logger.info(
        f"Tier 2 (keyword-specific keep): {len(tier2_keep)} ({len(tier2_keep) / N * 100:.1f}%)"
    )
    logger.info(
        f"Tier 3 (LLM classify): {len(tier3_classify)} ({len(tier3_classify) / N * 100:.1f}%)"
    )
    logger.info(
        f"Tier 4 (generic remove): {len(tier4_remove)} ({len(tier4_remove) / N * 100:.1f}%)"
    )

    # ---- LLM on tier 3 ----
    classify_pairs = [(idx, records[idx]["query"]) for idx in tier3_classify]
    batches = []
    for i in range(0, len(classify_pairs), args.batch_size):
        batches.append(classify_pairs[i : i + args.batch_size])

    logger.info(f"Running LLM on {len(batches)} batches...")
    llm_keep = set()
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=90,
    ) as client:
        tasks = [
            gpt41_batch(client, [q for _, q in b], bi, semaphore)
            for bi, b in enumerate(batches)
        ]
        done = 0
        t0 = time.time()
        for coro in asyncio.as_completed(tasks):
            bi, labels = await coro
            for (orig_idx, _), label in zip(batches[bi], labels):
                if label == 1:
                    llm_keep.add(orig_idx)
            done += 1
            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 1
                eta = (len(tasks) - done) / rate
                logger.info(
                    f"  LLM: {done}/{len(tasks)}, kept {len(llm_keep)} "
                    f"({len(llm_keep) / max(1, done * args.batch_size) * 100:.0f}%), "
                    f"ETA {eta:.0f}s"
                )

    # ---- Combine ----
    all_keep = tier1_keep | tier2_keep | llm_keep
    logger.info("\n=== FINAL RESULTS ===")
    logger.info(f"Total kept: {len(all_keep)}/{N} = {len(all_keep) / N * 100:.1f}%")
    logger.info(f"  Tier 1 (deep chunks):    {len(tier1_keep)}")
    logger.info(f"  Tier 2 (keyword):        {len(tier2_keep)}")
    logger.info(f"  Tier 3 (LLM approved):   {len(llm_keep)}")
    logger.info(f"  Removed (generic+LLM):   {N - len(all_keep)}")

    # Position breakdown
    from collections import Counter

    pos = Counter()
    for i in all_keep:
        p, c = parse_chunk_pos(records[i]["chunk_path"])
        pos[f"p{p}_c{c}"] += 1
    logger.info("Chunk position breakdown:")
    for k, v in sorted(pos.items(), key=lambda x: -x[1])[:12]:
        logger.info(f"  {k}: {v}")

    # Write
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kept = 0
    with open(args.output, "w") as f:
        for i, rec in enumerate(records):
            if i in all_keep:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    logger.info(f"Wrote {kept} to {args.output}")

    # Sample for inspection
    import random

    random.seed(42)
    sample_file = args.output.replace(".jsonl", "_sample200.jsonl")
    sample = random.sample(sorted(all_keep), min(200, len(all_keep)))
    with open(sample_file, "w") as f:
        for i in sample:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    logger.info(f"Sample saved to {sample_file}")

    # Also save REMOVED sample for comparison
    removed = set(range(N)) - all_keep
    removed_sample = random.sample(sorted(removed), min(200, len(removed)))
    removed_file = args.output.replace(".jsonl", "_removed_sample200.jsonl")
    with open(removed_file, "w") as f:
        for i in removed_sample:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    logger.info(f"Removed sample saved to {removed_file}")


if __name__ == "__main__":
    asyncio.run(main())
