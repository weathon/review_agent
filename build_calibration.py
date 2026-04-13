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
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from baselines.structured_review.build_calibration import BORDERLINE_EXTRA
from paper_reviewer import (
    _get_client,
    run_pipeline,
    sanitize_text,
)

# Reuse the same GT loader from bench
sys.path.insert(0, str(Path(__file__).parent))
from run_iclr_bench import load_ground_truth, DEFAULT_BENCH_DIR


BORDERLINE_BINS = {4, 5, 6}  # bins where accept/reject is hardest to distinguish
GARBAGE_BINS = {0, 1, 2}
CONCURRENCY = 3


def get_count(k):
    if k in GARBAGE_BINS:
        return 10  # just a few in the garbage bins to set standard
    elif k in BORDERLINE_BINS:
        return 10  # more from borderline bins
    else: 
        return 10 # a moderate number from the clear bins



def sample_one_per_bin(papers: list[dict], seed: int) -> list[dict]:
    # sam
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
        n_take = get_count(k)
        n_take = min(n_take, len(bins[k]))
        for j in range(n_take):
            samples.append(bins[k][j])
            tag = " (borderline)" if k in BORDERLINE_BINS else ""
            print(f"  Bin ~{k}: picked {bins[k][j]['paper_id']} (avg={bins[k][j]['avg_score']:.1f}, {bins[k][j]['gt_binary']}){tag}")
    print(f"  Total: {len(samples)} calibration papers ({sum(1 for s in samples if round(s['avg_score']) in BORDERLINE_BINS)} borderline)\n")
    random.shuffle(samples)
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

    result = await run_pipeline(
        paper_path=str(paper_path),
        paper_content=paper_content,
        client=client,
        parallel=parallel,
        skip_related_work=skip_related_work,
        skip_spark=skip_spark,
        skip_neutral=skip_neutral,
        skip_score=True,
        venue="ICLR",
    )

    return {
        "harsh_review": result["harsh_review"],
        "neutral_review": result["neutral_review"],
        "spark_review": result["spark_review"],
        "related_work": result["related_work"],
        "merged_review": result["merged_review"],
        "cost": result["cost"],
        "sdk_savings": result["sdk_savings"],
    }


def shorten_title(title: str, max_len: int = 60) -> str:
    """Turn a paper title into a safe, short filename (no extension)."""
    # Lowercase, keep only alphanumeric and spaces, collapse whitespace
    name = re.sub(r"[^a-z0-9 ]", "", title.lower())
    name = re.sub(r"\s+", "_", name.strip())
    if len(name) > max_len:
        name = name[:max_len].rstrip("_")
    return name or "untitled"



def save_calibration_files(results: list[dict], cal_dir: Path, papers_dir: Path) -> None:
    """Save each calibration example as {name}_review.md + {name}_paper.md in cal_dir."""
    cal_dir.mkdir(exist_ok=True)
    for i, r in enumerate(results, 1):
        title = r.get("title", r["paper_id"])
        base = shorten_title(title)

        # Save review file
        parts = []
        parts.append(f"=== CALIBRATION EXAMPLE {i} ===\n")
        # if r.get("harsh_review"):
        #     parts.append("# Harsh Critic Review")
        #     parts.append(r["harsh_review"])
        #     parts.append("")
        # if r.get("neutral_review"):
        #     parts.append("# Neutral Reviewer")
        #     parts.append(r["neutral_review"])
        #     parts.append("")
        # if r.get("spark_review"):
        #     parts.append("# Spark Finder Review")
        #     parts.append(r["spark_review"])
        #     parts.append("")
        # if r.get("related_work"):
        #     parts.append("# Related Work Analysis")
        #     parts.append(r["related_work"])
        #     parts.append("")
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

    samples = sample_one_per_bin(available, seed)

    results = []
    results_lock = asyncio.Lock()
    cal_dir = Path(__file__).parent / "cal"
    ids_path = Path(__file__).parent / "calibration_ids.json"

    # Check for already-completed papers
    done_ids = set()
    for s in samples:
        title = s.get("title", s["paper_id"])
        base = shorten_title(title)
        if (cal_dir / f"{base}_review.md").exists():
            done_ids.add(s["paper_id"])

    if done_ids:
        print(f"\nFound {len(done_ids)}/{len(samples)} papers already completed in {cal_dir}.")
        choice = input("Continue (skip done papers) or overwrite cal? [c/o]: ").strip().lower()
        if choice == "o":
            import shutil
            if cal_dir.exists():
                shutil.rmtree(cal_dir)
                print(f"Deleted {cal_dir}")
            if ids_path.exists():
                ids_path.unlink()
                print(f"Deleted {ids_path}")
            print("Overwriting: cleared calibration outputs and will re-run all papers.")
        else:
            print(f"Continuing: skipping {len(done_ids)} done papers.")
            samples = [s for s in samples if s["paper_id"] not in done_ids]
            if not samples:
                print("All papers already done. Nothing to do.")
                return

    def _save_all(results_snapshot: list[dict]) -> None:
        """Save all accumulated results to disk."""
        save_calibration_files(results_snapshot, cal_dir, papers_dir)
        ids = [r["paper_id"] for r in results_snapshot]
        ids_path.write_text(json.dumps(ids, indent=2))

    async def process_one(i, paper_info):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"

        print(f"\n{'─' * 72}")
        print(f"[{i}/{len(samples)}] {paper_info.get('title', pid)}")
        print(f"  GT: {paper_info['decision']} | Avg: {paper_info['avg_score']:.1f} | Scores: {paper_info['scores']}")
        print(f"{'─' * 72}")

        attempt = 0
        for i in range(3): 
            attempt += 1
            start = time.time()
            try:
                outputs = await run_sub_agents_and_merger(
                    paper_info, paper_path,
                    parallel=parallel, skip_spark=skip_spark, skip_related_work=skip_related_work,
                    skip_neutral=skip_neutral,
                )
                elapsed = time.time() - start
                print(f"  [{pid}] Done in {elapsed:.1f}s (attempt {attempt}) — cost: ${outputs.get('cost', 0):.4f}, sdk_savings: ${outputs.get('sdk_savings', 0):.4f}")

                result = {
                    **outputs,
                    "paper_id": pid,
                    "title": paper_info.get("title", pid),
                    "scores": paper_info["scores"],
                    "avg_score": paper_info["avg_score"],
                    "decision": paper_info["decision"],
                    "gt_binary": paper_info["gt_binary"],
                    "error": None,
                }

                # Save incrementally after each paper completes
                async with results_lock:
                    results.append(result)
                    _save_all(list(results))
                    print(f"  [{pid}] Saved ({len(results)} papers so far)")

                return result
            except Exception as e:
                elapsed = time.time() - start
                wait = min(30 * attempt, 120)
                print(f"  [{pid}] ERROR (attempt {attempt}) after {elapsed:.1f}s: {e}")
                print(f"  [{pid}] Retrying in {wait}s ...")
                await asyncio.sleep(wait)

    # Run calibration papers concurrently via worker pool with staggered start
    queue: asyncio.Queue[tuple[int, dict] | None] = asyncio.Queue()
    for i, p in enumerate(samples, 1):
        queue.put_nowait((i, p))
    for _ in range(CONCURRENCY):
        queue.put_nowait(None)  # sentinel

    async def worker(worker_id: int):
        jitter = random.uniform(0, 60)
        print(f"  [worker-{worker_id}] stagger wait {jitter:.1f}s ...")
        await asyncio.sleep(jitter)
        while True:
            item = await queue.get()
            if item is None:
                break
            i, paper_info = item
            await process_one(i, paper_info)

    print(f"Running {len(samples)} calibration papers (concurrency={CONCURRENCY}) ...")
    await asyncio.gather(*(worker(w) for w in range(CONCURRENCY)))

    failures = [r for r in results if r.get("error")]

    if failures:
        print(f"\nSkipped {len(failures)} failed calibration papers:")
        for r in failures:
            print(f"  {r['paper_id']}: {r['error']}")
    if not results:
        raise RuntimeError("No calibration papers completed successfully.")

    print(f"\nCalibration files saved to: {cal_dir}")
    print(f"Calibration IDs saved to: {ids_path} ({len(results)} papers)")

    # Summary
    total_cost = sum(r.get("cost", 0) for r in results)
    total_sdk_savings = sum(r.get("sdk_savings", 0) for r in results)
    print(f"\n{'=' * 72}")
    print("Calibration set built:")
    for r in results:
        print(f"  {r['paper_id']}: avg={r['avg_score']:.1f} scores={r['scores']} dec={r['gt_binary']}")
    if failures:
        print(f"Failed: {len(failures)}")
    print(f"\nTotal cost:        ${total_cost:.4f}")
    print(f"SDK savings:       ${total_sdk_savings:.4f}")
    print(f"Net cost:          ${total_cost - total_sdk_savings:.4f}")


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
