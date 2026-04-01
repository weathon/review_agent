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
import sys
import time
from datetime import datetime
from pathlib import Path

# Tee stdout+stderr to a log file so all intermediate output is saved
_log_path = Path(__file__).parent / "bench_run.log"
_log_file = open(_log_path, "w")

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


from paper_reviewer import (
    HARSH_CRITIC_PROMPT,
    MODEL_HARSH,
    MODEL_MERGER,
    MODEL_NEUTRAL,
    MODEL_RELATED_WORK,
    MODEL_SPARK,
    NEUTRAL_REVIEWER_PROMPT,
    SPARK_FINDER_PROMPT,
    _get_client,
    run_merger,
    run_related_work_search,
    run_reviewer,
    run_score_predictor,
    sanitize_text,
    score_to_decision,
)

# ── Paths (defaults to AI-Scientist, overridable with --data-dir) ─────

DEFAULT_BENCH_DIR = Path(__file__).parent / "AI-Scientist" / "review_iclr_bench"

# ── Helpers ───────────────────────────────────────────────────────────


def load_ground_truth(bench_dir: Path) -> tuple[list[dict], Path]:
    """Load GT from either AI-Scientist TSV or iclr2025_data CSV format."""
    # Try iclr2025_data format first (CSV with paper_id, title, decision, gt_binary, avg_score, score_0..5)
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


VALID_SCORES = [1.0, 3.0, 5.0, 6.0, 8.0, 10.0]


def _snap_score(raw: float) -> float:
    """Snap a raw score to the nearest valid ICLR score."""
    return min(VALID_SCORES, key=lambda v: abs(v - raw))


async def review_single_paper(
    paper_id: str, paper_path: Path, parallel: bool = False, skip_related_work: bool = False, skip_spark: bool = False, calibration_context: str = "",
) -> dict:
    """Run the full pipeline on one paper."""
    paper_content = paper_path.read_text(encoding="utf-8", errors="replace")
    paper_content = sanitize_text(paper_content)

    print(f"  Paper length: {len(paper_content):,} chars")

    client = _get_client()
    pp = str(paper_path)

    sep = "~" * 60

    # Phase 1: All reviewers (parallel or sequential)
    if parallel:
        tasks = [
            run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue="ICLR"),
            run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue="ICLR"),
        ]
        if not skip_spark:
            tasks.append(run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue="ICLR"))
        if not skip_related_work:
            tasks.append(run_related_work_search(client, paper_content))

        print("  Phase 1: All reviewers in parallel ...")
        results_list = await asyncio.gather(*tasks)

        idx = 0
        harsh_review = results_list[idx]; idx += 1
        neutral_review = results_list[idx]; idx += 1
        spark_review = results_list[idx] if not skip_spark else "Spark finder was skipped."; idx += (0 if skip_spark else 1)
        related_work = results_list[idx] if not skip_related_work else "Related work search was skipped."
    else:
        print("  Phase 1: Reviewers sequentially ...")
        harsh_review = await run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue="ICLR")
        neutral_review = await run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue="ICLR")
        if not skip_spark:
            spark_review = await run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue="ICLR")
        else:
            spark_review = "Spark finder was skipped."
        if not skip_related_work:
            related_work = await run_related_work_search(client, paper_content)
        else:
            related_work = "Related work search was skipped."

    for label, text in [("harsh_critic", harsh_review), ("neutral", neutral_review)]:
        print(f"\n  {sep}\n  [{label} output] ({len(text)} chars)\n  {sep}\n{text}\n")
        if not text.strip():
            print(f"  *** WARNING: {label} returned empty output ***")
    if not skip_spark:
        print(f"\n  {sep}\n  [spark_finder output] ({len(spark_review)} chars)\n  {sep}\n{spark_review}\n")
    if not skip_related_work:
        print(f"\n  {sep}\n  [related_work output] ({len(related_work)} chars)\n  {sep}\n{related_work}\n")

    # Phase 2: Merger (review only, no score)
    print("  Phase 2: Merger ...")
    final_review = await run_merger(
        client, harsh_review, neutral_review,
        spark_review, related_work, paper_content,
    )
    print(f"\n  {sep}\n  [merger output] ({len(final_review)} chars)\n  {sep}\n{final_review}\n")

    # Phase 3: Score predictor (with calibration)
    print("  Phase 3: Score predictor ...")
    score = await run_score_predictor(
        client, harsh_review, neutral_review,
        spark_review, related_work, final_review,
        calibration_context=calibration_context,
    )
    score = round(float(score), 1)
    decision = score_to_decision(score)
    print(f"  [score_predictor] structured score: {score}")

    return {
        "final_review": final_review,
        "predicted_score": score,
        "predicted_decision": decision,
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


async def main(n_samples: int = 10, seed: int = 42, parallel: bool = False, skip_related_work: bool = False, skip_spark: bool = False, balanced: bool = False, data_dir: str | None = None, calibration_path: str | None = None):
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

    # Load calibration if provided
    calibration_context = ""
    calibration_ids = set()
    if calibration_path:
        cal_path = Path(calibration_path)
        if cal_path.exists():
            calibration_context = cal_path.read_text(encoding="utf-8")
            print(f"\nLoaded calibration: {cal_path} ({len(calibration_context):,} chars)")
            # Load excluded IDs
            ids_path = cal_path.parent / "calibration_ids.json"
            if ids_path.exists():
                calibration_ids = set(json.load(open(ids_path)))
                print(f"Excluding {len(calibration_ids)} calibration papers from sampling")
        else:
            print(f"\nWARNING: calibration file not found: {cal_path}")

    gt_data, papers_dir = load_ground_truth(bench_dir)
    print(f"\nLoaded {len(gt_data)} papers from ground truth.")

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

    results = []
    total_start = time.time()

    # Write results incrementally — header now, append after each paper
    output_path = Path(__file__).parent / "bench_results.md"
    csv_path = Path(__file__).parent / "bench_scores.csv"
    with open(output_path, "w") as f:
        f.write(f"# ICLR Benchmark Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Critic/Merger: {MODEL_HARSH} (OpenRouter)\n")
        f.write(f"Neutral: {MODEL_NEUTRAL}, ")
        f.write(f"Related Work: {MODEL_RELATED_WORK} (OpenRouter)\n\n")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "pred_score", "pred_decision", "gt_avg_score", "gt_decision", "gt_binary", "match",
                     "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3", "gt_score_4", "gt_score_5", "gt_score_6"])

    # Run papers concurrently (up to CONCURRENCY at a time)
    CONCURRENCY = 10
    semaphore = asyncio.Semaphore(CONCURRENCY)
    file_lock = asyncio.Lock()
    completed = [0]  # mutable counter

    async def process_paper(i: int, paper_info: dict):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"

        async with semaphore:
            print(f"\n{'─' * 72}")
            print(f"[{i}/{len(samples)}] Paper: {pid}")
            print(f"  GT Decision: {paper_info['decision']}  |  GT Avg Score: {paper_info['avg_score']:.1f}")
            print(f"  GT Reviewer Scores: {paper_info['scores']}")
            print(f"{'─' * 72}")

            start = time.time()
            try:
                review_result = await review_single_paper(pid, paper_path, parallel=parallel, skip_related_work=skip_related_work, skip_spark=skip_spark, calibration_context=calibration_context)
                elapsed = time.time() - start

                pred_score = review_result["predicted_score"]
                pred_dec = review_result["predicted_decision"]

                match = pred_dec == paper_info["gt_binary"] if pred_dec else None
                marker = "MATCH" if match else ("MISMATCH" if match is not None else "PARSE_FAIL")

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
                    "time_s": elapsed,
                    "final_review": review_result["final_review"],
                }

            except Exception as e:
                elapsed = time.time() - start
                print(f"\n  [{pid}] ERROR: {e}")
                print(f"  [{pid}] Time: {elapsed:.1f}s")
                r = {
                    "paper_id": pid,
                    "gt_decision": paper_info["decision"],
                    "gt_binary": paper_info["gt_binary"],
                    "gt_avg_score": paper_info["avg_score"],
                    "gt_scores": paper_info["scores"],
                    "predicted_score": None,
                    "predicted_decision": None,
                    "match": None,
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
                    f.write(f"- Match: {'Yes' if r['match'] else ('No' if r['match'] is not None else 'Parse fail')}\n\n")
                    f.write(f"### Final Review\n\n{r['final_review']}\n\n---\n\n")
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    gt_scores_padded = r["gt_scores"] + [""] * (7 - len(r["gt_scores"]))
                    match_str = "YES" if r["match"] else ("NO" if r["match"] is not None else "PARSE_FAIL")
                    w.writerow([
                        r["paper_id"],
                        r["predicted_score"],
                        r["predicted_decision"],
                        f"{r['gt_avg_score']:.2f}",
                        r["gt_decision"],
                        r["gt_binary"],
                        match_str,
                        *gt_scores_padded,
                    ])

    # Launch all papers concurrently (semaphore limits to CONCURRENCY)
    print(f"\nRunning {len(samples)} papers with concurrency={CONCURRENCY}...")
    await asyncio.gather(*(
        process_paper(i, paper_info)
        for i, paper_info in enumerate(samples, 1)
    ))

    total_elapsed = time.time() - total_start

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 72)

    valid = [r for r in results if r["match"] is not None]
    matches = sum(1 for r in valid if r["match"])
    accuracy = matches / len(valid) if valid else 0

    print(f"\nPapers reviewed:  {len(results)}")
    print(f"Successful:       {len(valid)}")
    print(f"Correct:          {matches}/{len(valid)}")
    print(f"Accuracy:         {accuracy:.1%}")
    print(f"Total time:       {total_elapsed:.1f}s")
    print(f"Avg time/paper:   {total_elapsed / len(results):.1f}s")

    print(f"\n{'Paper ID':<20} {'GT':>10} {'Predicted':>10} {'GT Score':>10} {'Pred Score':>11} {'Match':>7}")
    print("─" * 72)
    for r in results:
        gt = r["gt_binary"]
        pred = r["predicted_decision"] or "N/A"
        gt_sc = f"{r['gt_avg_score']:.1f}"
        pred_sc = f"{r['predicted_score']:.1f}" if r["predicted_score"] else "N/A"
        match_str = "YES" if r["match"] else ("NO" if r["match"] is not None else "ERR")
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
        f.write(f"\n# Summary\n\nPapers: {len(results)} | Accuracy: {accuracy:.1%}\n")

    print(f"\nDetailed results saved to: {output_path}")

    return results


if __name__ == "__main__":
    parallel = "--parallel" in sys.argv
    skip_related = "--no-related-work" in sys.argv
    skip_spark = "--no-spark" in sys.argv
    balanced = "--balanced" in sys.argv
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
    flag_values = {data_dir, calibration_path} - {None}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in flag_values]
    n = int(args[0]) if len(args) > 0 else 10
    seed = int(args[1]) if len(args) > 1 else 42
    asyncio.run(main(n_samples=n, seed=seed, parallel=parallel, skip_related_work=skip_related, skip_spark=skip_spark, balanced=balanced, data_dir=data_dir, calibration_path=calibration_path))
