"""
ICLR Benchmark Runner using OpenRouter chat completions.

Usage:
  python run_iclr_bench.py                          # 10 papers, sequential
  python run_iclr_bench.py 5                        # 5 papers
  python run_iclr_bench.py 10 42 --parallel         # parallel agents
"""

import asyncio
import csv
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Tee stdout+stderr to a log file so all intermediate output is saved
_log_path = Path(__file__).parent / "bench_run.log"
_log_file = open(_log_path, "w")
CONCURRENCY = 5

class _Tee:
    """Write to both the original stream and a log file."""
    def __init__(self, stream, log):
        self._stream = stream
        self._log = log
    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._log.write(data)
        self._log.flush()
    def flush(self):
        self._stream.flush()
        self._log.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)

sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
import dotenv
dotenv.load_dotenv()

from paper_reviewer import (
    MODEL_HARSH,
    MODEL_NEUTRAL,
    MODEL_RELATED_WORK,
    get_client,
    decision_match,
    match_label,
    run_pipeline,
    sanitize_text,
)

# ── Paths (defaults to AI-Scientist, overridable with --data-dir) ─────

DEFAULT_BENCH_DIR = Path(__file__).parent / "AI-Scientist" / "review_iclr_bench"

# ── Helpers ───────────────────────────────────────────────────────────


def load_ground_truth(bench_dir: Path) -> tuple[list[dict], Path]:
    """Load GT from either AI-Scientist TSV or iclr2026_data CSV format."""
    # Try iclr2026_data format first (CSV with paper_id, title, decision, gt_binary, avg_score, score_0..5)
    csv_file = bench_dir / "ratings.csv"
    tsv_file = bench_dir / "ratings_subset.tsv"

    if csv_file.exists():
        papers_dir = bench_dir / "papers"
        rows = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores = []
                for i in range(6):
                    val = row.get(f"score_{i}", "").strip()
                    if val:
                        scores.append(float(val))
                decision = row.get("decision", "").strip()
                gt_binary = row.get("gt_binary", "").strip()
                if not gt_binary:
                    gt_binary = "Accept" if "Accept" in decision else "Reject"
                rows.append({
                    "paper_id": row["paper_id"].strip(),
                    "title": row.get("title", "").strip(),
                    "scores": scores,
                    "avg_score": float(row.get("avg_score", 0)),
                    "decision": decision,
                    "gt_binary": gt_binary,
                })
        return rows, papers_dir

    elif tsv_file.exists():
        papers_dir = bench_dir / "iclr_parsed"
        rows = []
        with open(tsv_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                scores = []
                for i in range(7):
                    val = row.get(str(i), "").strip()
                    if val:
                        scores.append(float(val))
                decision = row["decision"].strip()
                gt_binary = "Accept" if "Accept" in decision else "Reject"
                rows.append({
                    "paper_id": row["paper_id"].strip(),
                    "title": "",
                    "scores": scores,
                    "avg_score": sum(scores) / len(scores) if scores else 0,
                    "decision": decision,
                    "gt_binary": gt_binary,
                })
        return rows, papers_dir

    else:
        raise FileNotFoundError(f"No ratings file found in {bench_dir}")


VALID_SCORES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def _snap_score(raw: float) -> float:
    """Snap a raw score to the nearest valid ICLR score."""
    return min(VALID_SCORES, key=lambda v: abs(v - raw))


def _shorten_title(title: str, max_len: int = 60) -> str:
    name = re.sub(r"[^a-z0-9 ]", "", title.lower())
    name = re.sub(r"\s+", "_", name.strip())
    if len(name) > max_len:
        name = name[:max_len].rstrip("_")
    return name or "untitled"


def _load_calibration_ids_from_cal_dir(cal_dir: Path, gt_data: list[dict]) -> set[str]:
    review_files = sorted(cal_dir.glob("*_review.md"))
    review_bases = [review_file.name.removesuffix("_review.md") for review_file in review_files]
    title_to_id = {}
    for row in gt_data:
        title = row.get("title", "").strip()
        if not title:
            continue
        shortened = _shorten_title(title)
        if shortened in title_to_id and title_to_id[shortened] != row["paper_id"]:
            raise ValueError(f"Duplicate shortened calibration title mapping for '{shortened}'")
        title_to_id[shortened] = row["paper_id"]
    calibration_ids = [title_to_id[base] for base in review_bases if base in title_to_id]
    assert len(calibration_ids) == len(review_bases), (
        f"Calibration title->id mapping mismatch: mapped {len(calibration_ids)} ids "
        f"for {len(review_bases)} calibration review files in {cal_dir}"
    )
    return set(calibration_ids)


async def review_single_paper(
    paper_id: str, paper_path: Path, parallel: bool = False, skip_related_work: bool = False, skip_spark: bool = False, skip_neutral: bool = False, calibration_context: str = "", cal_dir: str = "", gt_score: float | None = None, merger_output_score: bool = False
) -> dict:
    """Run the full pipeline on one paper."""
    paper_content = paper_path.read_text(encoding="utf-8", errors="replace")
    paper_content = sanitize_text(paper_content)

    print(f"  Paper length: {len(paper_content):,} chars")

    client = get_client()

    result = await run_pipeline(
        paper_path=str(paper_path),
        paper_content=paper_content,
        client=client,
        parallel=parallel,
        skip_related_work=skip_related_work,
        skip_spark=skip_spark,
        skip_neutral=skip_neutral,
        skip_score=False,
        merger_output_score=merger_output_score,
        venue="ICLR",
        calibration_context=calibration_context,
        cal_dir=cal_dir,
        gt_score=gt_score,
    )

    print(f"  [merger_score] structured score: {result['score']}")
    print(f"  Total cost: ${result['cost']:.4f}")
    print(f"  SDK savings: ${result['sdk_savings']:.4f}")

    return {
        "final_review": result["merged_review"],
        "predicted_score": result["score"],
        "predicted_decision": result["decision"],
        "cost": result["cost"],
        "sdk_savings": result["sdk_savings"],
    }


# ── Main ──────────────────────────────────────────────────────────────


def stratified_sample(papers: list[dict], n: int, seed: int) -> list[dict]:
    """Sample equally from each score bin (rounded to int)."""
    from collections import defaultdict
    rng = random.Random(seed)
    bins = defaultdict(list)
    for p in papers:
        bins[round(p["avg_score"])].append(p)
    for k in bins:
        rng.shuffle(bins[k])
    sorted_bins = sorted(bins.keys())
    n_bins = len(sorted_bins)
    per_bin = n // n_bins
    remainder = n % n_bins
    print(f"Stratified sampling: {n_bins} bins, {per_bin} per bin (+{remainder} extra)")
    print(f"Bins: {', '.join(f'{k}({len(bins[k])})' for k in sorted_bins)}")
    samples = []
    for i, k in enumerate(sorted_bins):
        take = per_bin + (1 if i < remainder else 0)
        take = min(take, len(bins[k]))
        samples.extend(bins[k][:take])
    rng.shuffle(samples)
    print(f"Total sampled: {len(samples)}\n")
    return samples


async def main(n_samples: int = 10, seed: int = 42, parallel: bool = False, skip_related_work: bool = False, skip_spark: bool = False, skip_neutral: bool = False, balanced: bool = False, data_dir: str | None = None, calibration_path: str | None = None, merger_output_score: bool = False, csv_name: str = "bench_scores.csv") -> list[dict]:
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print("ICLR Benchmark: Multi-Agent Paper Reviewer")
    print(f"  Data: {bench_dir}")
    print("  OpenRouter chat completions for all agents")
    print("=" * 72)
    print(f"Mode: {'parallel' if parallel else 'sequential'}")
    print(f"Sampling: {'balanced (stratified)' if balanced else 'random'}")
    print(f"Models:")
    print(f"  Critic/Spark/Merger:             {MODEL_HARSH}")
    print(f"  Neutral:                         {MODEL_NEUTRAL}")
    print(f"  Related Work:                    {MODEL_RELATED_WORK}")

    gt_data, papers_dir = load_ground_truth(bench_dir)
    print(f"\nLoaded {len(gt_data)} papers from ground truth.")

    # Load calibration if provided
    calibration_context = ""
    cal_dir = ""
    calibration_ids = set()
    if calibration_path:
        cal_path = Path(calibration_path)
        # Check for cal/ directory (RAG mode) next to calibration_path
        cal_dir_candidate = cal_path.parent / "human_reviews"
        if cal_dir_candidate.is_dir():
            cal_dir = str(cal_dir_candidate)
            print(f"\nUsing RAG calibration: {cal_dir} (Agent SDK scorer)")
            calibration_ids = _load_calibration_ids_from_cal_dir(cal_dir_candidate, gt_data)
            print(f"Excluding {len(calibration_ids)} calibration papers based on titles in {cal_dir_candidate}")
        elif cal_path.exists():
            calibration_context = cal_path.read_text(encoding="utf-8")
            print(f"\nLoaded calibration: {cal_path} ({len(calibration_context):,} chars)")
        else:
            print(f"\nWARNING: calibration file not found: {cal_path}")

    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    if calibration_ids:
        available = [r for r in available if r["paper_id"] not in calibration_ids]
    print(f"Papers with parsed text (after exclusions): {len(available)}")

    if balanced:
        samples = stratified_sample(available, n_samples, seed)
    else:
        random.seed(seed)
        samples = random.sample(available, min(n_samples, len(available)))
        print(f"Selected {len(samples)} papers (seed={seed}).\n")
    random.shuffle(samples)
    results = []
    total_start = time.time()

    output_path = Path(__file__).parent / "bench_results.md"
    csv_path = Path(__file__).parent / csv_name
    reviews_dir = Path(__file__).parent / csv_name.replace(".csv", "_reviews")
    reviews_dir.mkdir(exist_ok=True)

    # Check for existing results and ask user whether to continue or overwrite
    finished_ids: set[str] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        import pandas as pd
        existing_df = pd.read_csv(csv_path)
        existing_count = len(existing_df)
        print(f"\nFound existing {csv_path.name} with {existing_count} results.")
        choice = input("  [C]ontinue (skip finished papers) or [O]verwrite? [C/o]: ").strip().lower()
        if choice in ("o", "overwrite"):
            print("  Overwriting existing results.\n")
        else:
            finished_ids = set(existing_df["paper_id"].astype(str))
            print(f"  Continuing — will skip {len(finished_ids)} already-finished papers.\n")

    if not finished_ids:
        # Fresh start — write headers
        with open(output_path, "w") as f:
            f.write(f"# ICLR Benchmark Results\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Critic/Merger: {MODEL_HARSH} (OpenRouter)\n")
            f.write(f"Neutral: {MODEL_NEUTRAL}, ")
            f.write(f"Related Work: {MODEL_RELATED_WORK} (OpenRouter)\n\n")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["paper_id", "pred_score", "pred_decision", "gt_avg_score", "gt_decision", "gt_binary", "match", "cost", "sdk_savings",
                         "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3", "gt_score_4", "gt_score_5", "gt_score_6"])

    # Run papers concurrently (up to CONCURRENCY at a time)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    file_lock = asyncio.Lock()
    completed = [0]  # mutable counter

    async def process_paper(i: int, paper_info: dict):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"

        if pid in finished_ids:
            print(f"  [{i}/{len(samples)}] Skipping {pid} (already finished)")
            return

        async with semaphore:
            print(f"\n{'─' * 72}")
            print(f"[{i}/{len(samples)}] Paper: {pid}")
            print(f"  GT Decision: {paper_info['decision']}  |  GT Avg Score: {paper_info['avg_score']:.1f}")
            print(f"  GT Reviewer Scores: {paper_info['scores']}")
            print(f"{'─' * 72}")

            max_paper_retries = 3
            for attempt in range(1, max_paper_retries + 1):
                start = time.time()
                try:
                    review_result = await review_single_paper(pid, paper_path, parallel=parallel, skip_related_work=skip_related_work, skip_spark=skip_spark, skip_neutral=skip_neutral, calibration_context=calibration_context, cal_dir=cal_dir, gt_score=paper_info["avg_score"], merger_output_score=merger_output_score)
                    elapsed = time.time() - start

                    pred_score = review_result["predicted_score"]
                    pred_dec = review_result["predicted_decision"]

                    match = decision_match(pred_dec, paper_info["gt_binary"])
                    marker = "MATCH" if match is True else ("MISMATCH" if match is False else "N/A")

                    print(f"\n  [{pid}] Predicted Score: {pred_score}/10  |  Predicted Decision: {pred_dec}")
                    print(f"  [{pid}] GT Binary: {paper_info['gt_binary']}  |  Result: *** {marker} ***")
                    print(f"  [{pid}] Time: {elapsed:.1f}s")

                    r = {
                        "paper_id": pid,
                        "gt_decision": paper_info["decision"],
                        "gt_binary": paper_info["gt_binary"],
                        "gt_avg_score": paper_info["avg_score"],
                        "gt_scores": paper_info["scores"],
                        "predicted_score": pred_score,
                        "predicted_decision": pred_dec,
                        "match": match,
                        "cost": review_result.get("cost", 0.0),
                        "sdk_savings": review_result.get("sdk_savings", 0.0),
                        "time_s": elapsed,
                        "final_review": review_result["final_review"],
                    }
                    break  # success, exit retry loop

                except Exception as e:
                    elapsed = time.time() - start
                    if attempt < max_paper_retries:
                        wait = 15 * attempt
                        print(f"\n  [{pid}] ERROR (attempt {attempt}/{max_paper_retries}): {e}")
                        print(f"  [{pid}] Retrying in {wait}s ...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"\n  [{pid}] ERROR (attempt {attempt}/{max_paper_retries}, giving up): {e}")
                        print(f"  [{pid}] Time: {elapsed:.1f}s")
                        r = {
                            "paper_id": pid,
                            "gt_decision": paper_info["decision"],
                            "gt_binary": paper_info["gt_binary"],
                            "gt_avg_score": paper_info["avg_score"],
                            "gt_scores": paper_info["scores"],
                            "predicted_score": None,
                            "predicted_decision": "N/A",
                            "match": None,
                            "cost": 0.0,
                            "sdk_savings": 0.0,
                            "time_s": elapsed,
                            "final_review": f"ERROR: {e}",
                        }

            # Thread-safe file writes + results append
            async with file_lock:
                results.append(r)
                completed[0] += 1
                print(f"  [{pid}] *** Completed {completed[0]}/{len(samples)} ***")

                with open(output_path, "a") as f:
                    f.write(f"## {r['paper_id']}\n\n")
                    f.write(f"- GT: {r['gt_decision']} (avg {r['gt_avg_score']:.1f})\n")
                    f.write(f"- Predicted: {r['predicted_decision']} ({r['predicted_score']}/10)\n")
                    f.write(f"- Match: {match_label(r['match'])}\n\n")
                    f.write(f"### Final Review\n\n{r['final_review']}\n\n---\n\n")
                # Save individual review file
                review_file = reviews_dir / f"{r['paper_id']}.md"
                review_file.write_text(r["final_review"], encoding="utf-8")
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    gt_scores_padded = r["gt_scores"] + [""] * (7 - len(r["gt_scores"]))
                    match_str = match_label(r["match"])
                    w.writerow([
                        r["paper_id"],
                        r["predicted_score"],
                        r["predicted_decision"],
                        f"{r['gt_avg_score']:.2f}",
                        r["gt_decision"],
                        r["gt_binary"],
                        match_str,
                        f"{r['cost']:.4f}",
                        f"{r['sdk_savings']:.4f}",
                        *gt_scores_padded,
                    ])

    # Launch all papers via worker pool with staggered start
    to_run = len(samples) - len(finished_ids & {p["paper_id"] for p in samples})
    print(f"\nRunning {to_run} papers ({len(finished_ids)} skipped) with concurrency={CONCURRENCY}...")

    queue: asyncio.Queue[tuple[int, dict] | None] = asyncio.Queue()
    for i, paper_info in enumerate(samples, 1):
        queue.put_nowait((i, paper_info))
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
            await process_paper(i, paper_info)

    await asyncio.gather(*(worker(w) for w in range(CONCURRENCY)))

    total_elapsed = time.time() - total_start

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 72)

    successful = [r for r in results if r["predicted_score"] is not None]
    valid = [r for r in results if r["match"] is not None]
    matches = sum(1 for r in valid if r["match"])
    accuracy = matches / len(valid) if valid else 0

    print(f"\nPapers reviewed:  {len(results)}")
    print(f"Successful:       {len(successful)}")
    print(f"Decision eval:    {len(valid)}")
    if valid:
        print(f"Correct:          {matches}/{len(valid)}")
        print(f"Accuracy:         {accuracy:.1%}")
    else:
        print("Correct:          N/A")
        print("Accuracy:         N/A (decision labels disabled)")
    total_cost = sum(r.get("cost", 0.0) for r in results)
    total_sdk_savings = sum(r.get("sdk_savings", 0.0) for r in results)
    print(f"Total time:       {total_elapsed:.1f}s")
    print(f"Avg time/paper:   {total_elapsed / len(results):.1f}s")
    print(f"Total cost:       ${total_cost:.4f}")
    print(f"Avg cost/paper:   ${total_cost / max(len(results), 1):.4f}")
    print(f"SDK savings:      ${total_sdk_savings:.4f}")
    print(f"Net cost:         ${total_cost - total_sdk_savings:.4f}")

    print(f"\n{'Paper ID':<20} {'GT':>10} {'Predicted':>10} {'GT Score':>10} {'Pred Score':>11} {'Match':>7}")
    print("─" * 72)
    for r in results:
        gt = r["gt_binary"]
        pred = r["predicted_decision"] or "N/A"
        gt_sc = f"{r['gt_avg_score']:.1f}"
        pred_sc = f"{r['predicted_score']:.1f}" if r["predicted_score"] else "N/A"
        match_str = match_label(r["match"])
        print(f"{r['paper_id']:<20} {gt:>10} {pred:>10} {gt_sc:>10} {pred_sc:>11} {match_str:>7}")

    paired = [(r["gt_avg_score"], r["predicted_score"]) for r in results if r["predicted_score"] is not None]
    if len(paired) >= 2:
        gt_scores = [p[0] for p in paired]
        pred_scores = [p[1] for p in paired]
        mean_gt = sum(gt_scores) / len(gt_scores)
        mean_pred = sum(pred_scores) / len(pred_scores)
        print(f"\nMean GT Score:      {mean_gt:.2f}")
        print(f"Mean Pred Score:    {mean_pred:.2f}")
        print(f"Score diff (avg):   {sum(abs(g - p) for g, p in paired) / len(paired):.2f}")

    # Append summary to the file
    with open(output_path, "a") as f:
        accuracy_str = f"{accuracy:.1%}" if valid else "N/A"
        f.write(f"\n# Summary\n\nPapers: {len(results)} | Accuracy: {accuracy_str}\n")

    print(f"\nDetailed results saved to: {output_path}")

    return results


if __name__ == "__main__":
    parallel = "--parallel" in sys.argv
    skip_related = "--no-related-work" in sys.argv
    skip_spark = "--no-spark" in sys.argv
    skip_neutral = "--no-neutral" in sys.argv
    balanced = "--balanced" in sys.argv
    merger_output_score = "--merger-output-score" in sys.argv

    data_dir = None
    calibration_path = None
    if "--data-dir" in sys.argv:
        idx = sys.argv.index("--data-dir")
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    if "--calibration" in sys.argv:
        idx = sys.argv.index("--calibration")
        if idx + 1 < len(sys.argv):
            calibration_path = sys.argv[idx + 1]

    if "--save_path" in sys.argv:
        idx = sys.argv.index("--save_path")
        if idx + 1 < len(sys.argv):
            csv_name = sys.argv[idx + 1]
    else:
        csv_name = "bench_scores.csv"
        
    flag_values = {data_dir, calibration_path} - {None}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in flag_values]
    n = int(args[0]) if len(args) > 0 else 10
    seed = int(args[1]) if len(args) > 1 else 42
    asyncio.run(main(n_samples=n, seed=seed, parallel=parallel, skip_related_work=skip_related, skip_spark=skip_spark, skip_neutral=skip_neutral, balanced=balanced, data_dir=data_dir, calibration_path=calibration_path, merger_output_score=merger_output_score, csv_name=csv_name))
