"""
Build a calibration set for the score predictor.

Samples papers across score bins, runs the review stack, then pairs the
outputs with real human scores and decisions. Saves as calibration.md for
few-shot injection.

Usage:
  python build_calibration.py --data-dir iclr2026_data --parallel
  python build_calibration.py --data-dir iclr2026_data --no-spark --no-related-work
"""

import asyncio
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from paper_reviewer import (
    HARSH_CRITIC_PROMPT,
    MODEL_HARSH,
    MODEL_MERGER,
    MODEL_NEUTRAL,
    MODEL_RELATED_WORK,
    MODEL_SPARK,
    MERGER_PROMPT,
    NEUTRAL_REVIEWER_PROMPT,
    SPARK_FINDER_PROMPT,
    _get_client,
    run_merge,
    run_related_work_search,
    run_reviewer,
    sanitize_text,
)

# Reuse the same GT loader from bench
sys.path.insert(0, str(Path(__file__).parent))
from run_iclr_bench import load_ground_truth, DEFAULT_BENCH_DIR


BORDERLINE_BINS = {5, 6}  # bins where accept/reject is hardest to distinguish
BORDERLINE_EXTRA = 0      # extra papers per borderline bin
CONCURRENCY = 14

def sample_one_per_bin(papers: list[dict], seed: int) -> list[dict]:
    """Sample 1 paper per score bin, with extra papers in borderline bins (5, 6)."""
    rng = random.Random(seed)
    bins = defaultdict(list)
    for p in papers:
        bins[round(p["avg_score"])].append(p)
    for k in bins:
        rng.shuffle(bins[k])

    samples = []
    for k in sorted(bins.keys()):
        if not bins[k]:
            continue
        # Take more from borderline bins
        n_take = 10 # + BORDERLINE_EXTRA if k in BORDERLINE_BINS else 1
        n_take = min(n_take, len(bins[k]))
        for j in range(n_take):
            samples.append(bins[k][j])
            tag = " (borderline)" if k in BORDERLINE_BINS else ""
            print(f"  Bin ~{k}: picked {bins[k][j]['paper_id']} (avg={bins[k][j]['avg_score']:.1f}, {bins[k][j]['gt_binary']}){tag}")
    print(f"  Total: {len(samples)} calibration papers ({sum(1 for s in samples if round(s['avg_score']) in BORDERLINE_BINS)} borderline)\n")
    return samples


async def run_sub_agents_and_merger(
    paper_info: dict, paper_path: Path,
    parallel: bool = False, skip_spark: bool = False, skip_related_work: bool = False,
    skip_neutral: bool = False,
) -> dict:
    """Run sub-agents + merger (no score). Return all outputs."""
    paper_content = paper_path.read_text(encoding="utf-8", errors="replace")
    paper_content = sanitize_text(paper_content)

    client = _get_client()
    pp = str(paper_path)

    # Phase 1: All reviewers (parallel or sequential) — all via OpenAI
    if parallel:
        tasks = [
            run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue="ICLR"),
        ]
        if not skip_neutral:
            tasks.append(run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue="ICLR"))
        if not skip_spark:
            tasks.append(run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue="ICLR"))
        if not skip_related_work:
            tasks.append(run_related_work_search(client, paper_content))

        results_list = await asyncio.gather(*tasks)
        idx = 0
        harsh_review, _ = results_list[idx]; idx += 1
        if not skip_neutral:
            neutral_review, _ = results_list[idx]; idx += 1
        else:
            neutral_review = ""
        if not skip_spark:
            spark_review, _ = results_list[idx]; idx += 1
        else:
            spark_review = ""
        if not skip_related_work:
            related_work, _ = results_list[idx]
        else:
            related_work = ""
    else:
        harsh_review, _ = await run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue="ICLR")
        if not skip_neutral:
            neutral_review, _ = await run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue="ICLR")
        else:
            neutral_review = ""
        if not skip_spark:
            spark_review, _ = await run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue="ICLR")
        else:
            spark_review = ""
        if not skip_related_work:
            related_work, _ = await run_related_work_search(client, paper_content)
        else:
            related_work = ""

    # Phase 2: Merger only (no scoring — calibration reviews must not be scored)
    print("  Running merger ...")
    merged_review, _ = await run_merge(
        client, harsh_review, neutral_review,
        spark_review, related_work, paper_content,
        skip_neutral=skip_neutral,
        skip_spark=skip_spark,
        skip_related_work=skip_related_work,
    )

    return {
        "harsh_review": harsh_review,
        "neutral_review": neutral_review,
        "spark_review": spark_review,
        "related_work": related_work,
        "merged_review": merged_review,
    }


def shorten_title(title: str, max_len: int = 60) -> str:
    """Turn a paper title into a safe, short filename (no extension)."""
    # Lowercase, keep only alphanumeric and spaces, collapse whitespace
    name = re.sub(r"[^a-z0-9 ]", "", title.lower())
    name = re.sub(r"\s+", "_", name.strip())
    if len(name) > max_len:
        name = name[:max_len].rstrip("_")
    return name or "untitled"


def build_calibration_md(results: list[dict]) -> str:
    """Build compact calibration markdown from final review + human scores."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"=== CALIBRATION EXAMPLE {i} ===\n")
        parts.append("# Final Consolidated Review")
        parts.append(r["merged_review"])
        parts.append("")
        parts.append("# Actual Human Scores")
        parts.append(f"Individual reviewer scores: {r['scores']}")
        parts.append(f"Average score: {r['avg_score']:.1f}")
        parts.append(f"Binary outcome: {r['gt_binary']}\n")
    return "\n".join(parts)


def save_calibration_files(results: list[dict], cal_dir: Path, papers_dir: Path) -> None:
    """Save each calibration example as {name}_review.md + {name}_paper.md in cal_dir."""
    cal_dir.mkdir(exist_ok=True)
    for i, r in enumerate(results, 1):
        title = r.get("title", r["paper_id"])
        base = shorten_title(title)

        # Save review file
        parts = []
        parts.append(f"=== CALIBRATION EXAMPLE {i} ===\n")
        if r.get("harsh_review"):
            parts.append("# Harsh Critic Review")
            parts.append(r["harsh_review"])
            parts.append("")
        if r.get("neutral_review"):
            parts.append("# Neutral Reviewer")
            parts.append(r["neutral_review"])
            parts.append("")
        if r.get("spark_review"):
            parts.append("# Spark Finder Review")
            parts.append(r["spark_review"])
            parts.append("")
        if r.get("related_work"):
            parts.append("# Related Work Analysis")
            parts.append(r["related_work"])
            parts.append("")
        parts.append("# Final Consolidated Review")
        parts.append(r["merged_review"])
        parts.append("")
        parts.append("# Actual Human Scores")
        parts.append(f"Individual reviewer scores: {r['scores']}")
        parts.append(f"Average score: {r['avg_score']:.1f}")
        parts.append(f"Binary outcome: {r['gt_binary']}\n")
        (cal_dir / f"{base}_review.md").write_text("\n".join(parts), encoding="utf-8")

        # Copy paper text
        paper_src = papers_dir / f"{r['paper_id']}.txt"
        if paper_src.exists():
            paper_text = paper_src.read_text(encoding="utf-8", errors="replace")
            (cal_dir / f"{base}_paper.md").write_text(paper_text, encoding="utf-8")

    print(f"Saved {len(results)} calibration file pairs to: {cal_dir}")


async def main(
    data_dir: str | None = None, seed: int = 42,
    parallel: bool = False, skip_spark: bool = False, skip_related_work: bool = False,
    skip_neutral: bool = False,
):
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print("Building Calibration Set")
    print(f"Data: {bench_dir}")
    print("=" * 72)

    gt_data, papers_dir = load_ground_truth(bench_dir)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    print(f"Loaded {len(available)} papers with text.\n")

    print("Sampling 1 paper per score bin...")
    samples = sample_one_per_bin(available, seed)

    results = []

    async def process_one(i, paper_info):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"

        print(f"\n{'─' * 72}")
        print(f"[{i}/{len(samples)}] {paper_info.get('title', pid)}")
        print(f"  GT: {paper_info['decision']} | Avg: {paper_info['avg_score']:.1f} | Scores: {paper_info['scores']}")
        print(f"{'─' * 72}")

        attempt = 0
        while True:
            attempt += 1
            start = time.time()
            try:
                outputs = await run_sub_agents_and_merger(
                    paper_info, paper_path,
                    parallel=parallel, skip_spark=skip_spark, skip_related_work=skip_related_work,
                    skip_neutral=skip_neutral,
                )
                elapsed = time.time() - start
                print(f"  [{pid}] Done in {elapsed:.1f}s (attempt {attempt})")

                return {
                    **outputs,
                    "paper_id": pid,
                    "title": paper_info.get("title", pid),
                    "scores": paper_info["scores"],
                    "avg_score": paper_info["avg_score"],
                    "decision": paper_info["decision"],
                    "gt_binary": paper_info["gt_binary"],
                    "error": None,
                }
            except Exception as e:
                elapsed = time.time() - start
                wait = min(30 * attempt, 120)
                print(f"  [{pid}] ERROR (attempt {attempt}) after {elapsed:.1f}s: {e}")
                print(f"  [{pid}] Retrying in {wait}s ...")
                await asyncio.sleep(wait)

    # Run calibration papers concurrently
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def limited(i, paper_info):
        async with semaphore:
            return await process_one(i, paper_info)

    print(f"Running {len(samples)} calibration papers (concurrency={CONCURRENCY}) ...")
    all_results = await asyncio.gather(*(
        limited(i, p) for i, p in enumerate(samples, 1)
    ))

    results = [r for r in all_results if not r.get("error")]
    failures = [r for r in all_results if r.get("error")]

    if failures:
        print(f"\nSkipped {len(failures)} failed calibration papers:")
        for r in failures:
            print(f"  {r['paper_id']}: {r['error']}")
    if not results:
        raise RuntimeError("No calibration papers completed successfully.")

    # Save calibration.md (single file, kept for backward compat)
    cal_md = build_calibration_md(results)
    cal_path = Path(__file__).parent / "calibration.md"
    cal_path.write_text(cal_md, encoding="utf-8")
    print(f"\nCalibration doc saved to: {cal_path} ({len(cal_md):,} chars)")

    # Save individual calibration files to cal/
    cal_dir = Path(__file__).parent / "cal"
    save_calibration_files(results, cal_dir, papers_dir)

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
    if failures:
        print(f"Failed: {len(failures)}")
    print(f"\nTo use in benchmark:")
    print(f"  python run_iclr_bench.py 10 42 --parallel --data-dir {data_dir or 'AI-Scientist/...'} --calibration calibration.md")


if __name__ == "__main__":
    parallel = "--parallel" in sys.argv
    skip_spark = "--no-spark" in sys.argv
    skip_related = "--no-related-work" in sys.argv
    skip_neutral = "--no-neutral" in sys.argv
    data_dir = None
    if "--data-dir" in sys.argv:
        idx = sys.argv.index("--data-dir")
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != data_dir]
    seed = int(args[0]) if args else 42
    asyncio.run(main(data_dir=data_dir, seed=seed, parallel=parallel, skip_spark=skip_spark, skip_related_work=skip_related, skip_neutral=skip_neutral))
