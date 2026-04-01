"""
Build a calibration set for the merger agent.

Samples 1 paper per score bin, runs sub-agents (critic, neutral, spark,
related work) but NOT the merger, then pairs the outputs with real human
scores and decisions. Saves as calibration.md for few-shot injection.

Usage:
  python build_calibration.py --data-dir iclr2025_data --parallel
  python build_calibration.py --data-dir iclr2025_data --no-spark --no-related-work
"""

import asyncio
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from paper_reviewer import (
    HARSH_CRITIC_PROMPT,
    MODEL_HARSH,
    MODEL_NEUTRAL,
    MODEL_RELATED_WORK,
    MODEL_SPARK,
    NEUTRAL_REVIEWER_PROMPT,
    SPARK_FINDER_PROMPT,
    _get_client,
    run_related_work_search,
    run_reviewer_claude,
    run_reviewer_openrouter,
    sanitize_text,
)

# Reuse the same GT loader from bench
sys.path.insert(0, str(Path(__file__).parent))
from run_iclr_bench import load_ground_truth, DEFAULT_BENCH_DIR


def sample_one_per_bin(papers: list[dict], seed: int) -> list[dict]:
    """Sample exactly 1 paper per score bin (rounded to int)."""
    rng = random.Random(seed)
    bins = defaultdict(list)
    for p in papers:
        bins[round(p["avg_score"])].append(p)
    for k in bins:
        rng.shuffle(bins[k])

    samples = []
    for k in sorted(bins.keys()):
        if bins[k]:
            samples.append(bins[k][0])
            print(f"  Bin ~{k}: picked {bins[k][0]['paper_id']} (avg={bins[k][0]['avg_score']:.1f}, {bins[k][0]['gt_binary']})")
    print(f"  Total: {len(samples)} calibration papers\n")
    return samples


async def run_sub_agents(
    paper_info: dict, paper_path: Path,
    parallel: bool = False, skip_spark: bool = False, skip_related_work: bool = False,
) -> dict:
    """Run sub-agents only (no merger). Return all outputs."""
    paper_content = paper_path.read_text(encoding="utf-8", errors="replace")
    paper_content = sanitize_text(paper_content)

    client = _get_client()
    pp = str(paper_path)

    # Stage 1: Critic (Claude) + Neutral + Related Work (OpenRouter)
    if parallel:
        tasks = [
            run_reviewer_claude("harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue="ICLR"),
            run_reviewer_openrouter(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue="ICLR"),
        ]
        if not skip_related_work:
            tasks.append(run_related_work_search(client, paper_content))
            harsh_review, neutral_review, related_work = await asyncio.gather(*tasks)
        else:
            harsh_review, neutral_review = await asyncio.gather(*tasks)
            related_work = ""
    else:
        harsh_review = await run_reviewer_claude("harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue="ICLR")
        neutral_review = await run_reviewer_openrouter(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue="ICLR")
        if not skip_related_work:
            related_work = await run_related_work_search(client, paper_content)
        else:
            related_work = ""

    # Stage 2: Spark (Claude) — sequential after stage 1
    if not skip_spark:
        spark_review = await run_reviewer_claude(
            "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue="ICLR",
        )
    else:
        spark_review = ""

    return {
        "harsh_review": harsh_review,
        "neutral_review": neutral_review,
        "spark_review": spark_review,
        "related_work": related_work,
    }


def build_calibration_md(results: list[dict]) -> str:
    """Build calibration markdown from sub-agent outputs + human scores."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"=== CALIBRATION EXAMPLE {i} ===\n")
        parts.append(f"# Review 1: Harsh Critic\n{r['harsh_review']}\n")
        parts.append(f"# Review 2: Neutral Reviewer\n{r['neutral_review']}\n")
        if r["spark_review"]:
            parts.append(f"# Review 3: Spark Finder\n{r['spark_review']}\n")
        if r["related_work"]:
            parts.append(f"# Report: Potentially Missed Related Work\n{r['related_work']}\n")
        parts.append(f"# ACTUAL HUMAN SCORES AND DECISION")
        parts.append(f"Individual reviewer scores: {r['scores']}")
        parts.append(f"Average score: {r['avg_score']:.1f}")
        parts.append(f"Decision: {r['decision']}")
        parts.append(f"Binary: {r['gt_binary']}\n")
    return "\n".join(parts)


async def main(
    data_dir: str | None = None, seed: int = 42,
    parallel: bool = False, skip_spark: bool = False, skip_related_work: bool = False,
):
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print("Building Calibration Set (sub-agents only, no merger)")
    print(f"Data: {bench_dir}")
    print("=" * 72)

    gt_data, papers_dir = load_ground_truth(bench_dir)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    print(f"Loaded {len(available)} papers with text.\n")

    print("Sampling 1 paper per score bin...")
    samples = sample_one_per_bin(available, seed)

    results = []
    for i, paper_info in enumerate(samples, 1):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"

        print(f"\n{'─' * 72}")
        print(f"[{i}/{len(samples)}] {paper_info.get('title', pid)}")
        print(f"  GT: {paper_info['decision']} | Avg: {paper_info['avg_score']:.1f} | Scores: {paper_info['scores']}")
        print(f"{'─' * 72}")

        start = time.time()
        outputs = await run_sub_agents(
            paper_info, paper_path,
            parallel=parallel, skip_spark=skip_spark, skip_related_work=skip_related_work,
        )
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")

        results.append({
            **outputs,
            "paper_id": pid,
            "scores": paper_info["scores"],
            "avg_score": paper_info["avg_score"],
            "decision": paper_info["decision"],
            "gt_binary": paper_info["gt_binary"],
        })

    # Save calibration.md
    cal_md = build_calibration_md(results)
    cal_path = Path(__file__).parent / "calibration.md"
    cal_path.write_text(cal_md, encoding="utf-8")
    print(f"\nCalibration doc saved to: {cal_path} ({len(cal_md):,} chars)")

    # Save calibration_ids.json
    ids = [r["paper_id"] for r in results]
    ids_path = Path(__file__).parent / "calibration_ids.json"
    ids_path.write_text(json.dumps(ids, indent=2))
    print(f"Calibration IDs saved to: {ids_path} ({len(ids)} papers)")

    # Summary
    print(f"\n{'=' * 72}")
    print("Calibration set built:")
    for r in results:
        print(f"  {r['paper_id']}: avg={r['avg_score']:.1f} scores={r['scores']} dec={r['gt_binary']}")
    print(f"\nTo use in benchmark:")
    print(f"  python run_iclr_bench.py 10 42 --parallel --data-dir {data_dir or 'AI-Scientist/...'} --calibration calibration.md")


if __name__ == "__main__":
    parallel = "--parallel" in sys.argv
    skip_spark = "--no-spark" in sys.argv
    skip_related = "--no-related-work" in sys.argv
    data_dir = None
    if "--data-dir" in sys.argv:
        idx = sys.argv.index("--data-dir")
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != data_dir]
    seed = int(args[0]) if args else 42
    asyncio.run(main(data_dir=data_dir, seed=seed, parallel=parallel, skip_spark=skip_spark, skip_related_work=skip_related))
