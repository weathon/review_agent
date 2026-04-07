"""
ICLR Benchmark Runner.

Usage:
  python run_iclr_bench.py                          # 10 papers, sequential
  python run_iclr_bench.py 5                        # 5 papers
  python run_iclr_bench.py 10 42 --parallel         # parallel agents
"""

import csv
import json
import random
import re
import sys
import time
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from queue import Queue

# Tee stdout+stderr to a log file
_log_path = Path(__file__).parent / "bench_run.log"
_log_file = open(_log_path, "w")

class _Tee:
    def __init__(self, stream, log):
        self._stream = stream
        self._log = log
    def write(self, data):
        self._stream.write(data); self._stream.flush()
        self._log.write(data); self._log.flush()
    def flush(self):
        self._stream.flush(); self._log.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)

sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

from paper_reviewer import (
    MODEL_HARSH, MODEL_NEUTRAL, MODEL_RELATED_WORK, MODEL_MERGER,
    decision_match, match_label, score_to_decision, review_paper,
)

DEFAULT_BENCH_DIR = Path(__file__).parent / "AI-Scientist" / "review_iclr_bench"


def load_ground_truth(bench_dir: Path) -> tuple[list[dict], Path]:
    csv_file = bench_dir / "ratings.csv"
    tsv_file = bench_dir / "ratings_subset.tsv"

    if csv_file.exists():
        papers_dir = bench_dir / "papers"
        rows = []
        with open(csv_file, "r") as f:
            for row in csv.DictReader(f):
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
            for row in csv.DictReader(f, delimiter="\t"):
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


def _extract_score_from_review(review_path: Path) -> float | None:
    """Extract the score line from a review file."""
    if not review_path.exists():
        return None
    text = review_path.read_text(encoding="utf-8")
    m = re.search(r"^Score:\s*([\d.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def stratified_sample(papers: list[dict], n: int, seed: int) -> list[dict]:
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
    samples = []
    for i, k in enumerate(sorted_bins):
        take = per_bin + (1 if i < remainder else 0)
        take = min(take, len(bins[k]))
        samples.extend(bins[k][:take])
    rng.shuffle(samples)
    print(f"Total sampled: {len(samples)}\n")
    return samples


def main(n_samples=10, seed=42, parallel=False, skip_related_work=False, skip_spark=False, skip_neutral=False, balanced=False, data_dir=None, calibration_path=None):
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print("ICLR Benchmark: Multi-Agent Paper Reviewer")
    print(f"  Data: {bench_dir}")
    print("=" * 72)

    cal_dir = ""
    calibration_ids = set()
    if calibration_path:
        cal_path = Path(calibration_path)
        cal_candidate = cal_path.parent / "cal"
        if cal_candidate.is_dir():
            cal_dir = str(cal_candidate)
            print(f"\nUsing RAG calibration: {cal_dir}")
        ids_path = cal_path.parent / "calibration_ids.json"
        if ids_path.exists():
            calibration_ids = set(json.load(open(ids_path)))
            print(f"Excluding {len(calibration_ids)} calibration papers")

    gt_data, papers_dir = load_ground_truth(bench_dir)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    if calibration_ids:
        available = [r for r in available if r["paper_id"] not in calibration_ids]
    print(f"Papers with text: {len(available)}")

    if balanced:
        samples = stratified_sample(available, n_samples, seed)
    else:
        random.seed(seed)
        samples = random.sample(available, min(n_samples, len(available)))
        print(f"Selected {len(samples)} papers (seed={seed}).\n")

    results = []
    results_lock = threading.Lock()
    total_start = time.time()

    output_path = Path(__file__).parent / "bench_results.md"
    csv_path = Path(__file__).parent / "bench_scores.csv"
    reviews_dir = Path(__file__).parent / "bench_reviews"
    reviews_dir.mkdir(exist_ok=True)

    finished_ids: set[str] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        import pandas as pd
        existing_df = pd.read_csv(csv_path)
        print(f"\nFound existing bench_scores.csv with {len(existing_df)} results.")
        choice = input("  [C]ontinue or [O]verwrite? [C/o]: ").strip().lower()
        if choice in ("o", "overwrite"):
            print("  Overwriting.\n")
        else:
            finished_ids = set(existing_df["paper_id"].astype(str))
            print(f"  Skipping {len(finished_ids)} finished papers.\n")

    if not finished_ids:
        with open(output_path, "w") as f:
            f.write(f"# ICLR Benchmark Results\n\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["paper_id", "pred_score", "pred_decision", "gt_avg_score", "gt_decision", "gt_binary", "match", "cost",
                         "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3", "gt_score_4", "gt_score_5", "gt_score_6"])

    CONCURRENCY = 10
    completed = [0]

    queue: Queue[tuple[int, dict] | None] = Queue()
    for i, p in enumerate(samples, 1):
        queue.put((i, p))
    for _ in range(CONCURRENCY):
        queue.put(None)

    def worker(worker_id):
        jitter = random.uniform(0, 60)
        print(f"  [worker-{worker_id}] stagger wait {jitter:.1f}s ...")
        time.sleep(jitter)

        while True:
            item = queue.get()
            if item is None:
                break
            i, paper_info = item
            pid = paper_info["paper_id"]
            paper_path = papers_dir / f"{pid}.txt"

            if pid in finished_ids:
                print(f"  [{i}/{len(samples)}] Skipping {pid}")
                continue

            print(f"\n{'─' * 72}")
            print(f"[{i}/{len(samples)}] Paper: {pid}")
            print(f"  GT: {paper_info['decision']} | Avg: {paper_info['avg_score']:.1f}")
            print(f"{'─' * 72}")

            for attempt in range(1, 4):
                start = time.time()
                try:
                    review_text, _ = review_paper(
                        str(paper_path), parallel=parallel,
                        skip_related_work=skip_related_work, skip_spark=skip_spark,
                        skip_neutral=skip_neutral, cal_dir=cal_dir,
                    )
                    elapsed = time.time() - start

                    review_file = paper_path.parent / f"{pid}_review.md"
                    score = _extract_score_from_review(review_file)
                    pred_score = round(score, 1) if score is not None else None
                    pred_dec = score_to_decision(pred_score)
                    match = decision_match(pred_dec, paper_info["gt_binary"])

                    print(f"  [{pid}] Score: {pred_score} | Time: {elapsed:.1f}s")

                    r = {
                        "paper_id": pid,
                        "gt_decision": paper_info["decision"],
                        "gt_binary": paper_info["gt_binary"],
                        "gt_avg_score": paper_info["avg_score"],
                        "gt_scores": paper_info["scores"],
                        "predicted_score": pred_score,
                        "predicted_decision": pred_dec,
                        "match": match,
                        "cost": 0.0,
                        "time_s": elapsed,
                        "final_review": review_text,
                    }
                    break
                except Exception as e:
                    elapsed = time.time() - start
                    if attempt < 3:
                        wait = 15 * attempt
                        print(f"  [{pid}] ERROR (attempt {attempt}): {e}")
                        print(f"  [{pid}] Retrying in {wait}s ...")
                        time.sleep(wait)
                    else:
                        print(f"  [{pid}] FAILED after 3 attempts: {e}")
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
                            "time_s": elapsed,
                            "final_review": f"ERROR: {e}",
                        }

            with results_lock:
                results.append(r)
                completed[0] += 1
                print(f"  [{pid}] *** Completed {completed[0]}/{len(samples)} ***")

                with open(output_path, "a") as f:
                    f.write(f"## {r['paper_id']}\n\n")
                    f.write(f"- GT: {r['gt_decision']} (avg {r['gt_avg_score']:.1f})\n")
                    f.write(f"- Predicted: {r['predicted_decision']} ({r['predicted_score']})\n")
                    f.write(f"- Match: {match_label(r['match'])}\n\n---\n\n")
                review_out = reviews_dir / f"{r['paper_id']}.md"
                review_out.write_text(r["final_review"], encoding="utf-8")
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    gt_scores_padded = r["gt_scores"] + [""] * (7 - len(r["gt_scores"]))
                    w.writerow([r["paper_id"], r["predicted_score"], r["predicted_decision"],
                                f"{r['gt_avg_score']:.2f}", r["gt_decision"], r["gt_binary"],
                                match_label(r["match"]), f"{r['cost']:.4f}", *gt_scores_padded])

    to_run = len(samples) - len(finished_ids & {p["paper_id"] for p in samples})
    print(f"\nRunning {to_run} papers (concurrency={CONCURRENCY})...")

    threads = []
    for w in range(CONCURRENCY):
        t = threading.Thread(target=worker, args=(w,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 72)

    successful = [r for r in results if r["predicted_score"] is not None]
    valid = [r for r in results if r["match"] is not None]
    matches = sum(1 for r in valid if r["match"])
    accuracy = matches / len(valid) if valid else 0

    print(f"\nPapers: {len(results)} | Successful: {len(successful)}")
    if valid:
        print(f"Accuracy: {matches}/{len(valid)} = {accuracy:.1%}")
    print(f"Time: {total_elapsed:.1f}s ({total_elapsed / max(len(results), 1):.1f}s/paper)")

    paired = [(r["gt_avg_score"], r["predicted_score"]) for r in results if r["predicted_score"] is not None]
    if len(paired) >= 2:
        print(f"Mean GT: {sum(p[0] for p in paired) / len(paired):.2f}")
        print(f"Mean Pred: {sum(p[1] for p in paired) / len(paired):.2f}")
        print(f"MAE: {sum(abs(g - p) for g, p in paired) / len(paired):.2f}")

    return results


if __name__ == "__main__":
    parallel = "--parallel" in sys.argv
    skip_related = "--no-related-work" in sys.argv
    skip_spark = "--no-spark" in sys.argv
    skip_neutral = "--no-neutral" in sys.argv
    balanced = "--balanced" in sys.argv
    data_dir = None
    calibration_path = None
    if "--data-dir" in sys.argv:
        idx = sys.argv.index("--data-dir")
        if idx + 1 < len(sys.argv): data_dir = sys.argv[idx + 1]
    if "--calibration" in sys.argv:
        idx = sys.argv.index("--calibration")
        if idx + 1 < len(sys.argv): calibration_path = sys.argv[idx + 1]
    flag_values = {data_dir, calibration_path} - {None}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in flag_values]
    n = int(args[0]) if len(args) > 0 else 10
    seed = int(args[1]) if len(args) > 1 else 42
    main(n_samples=n, seed=seed, parallel=parallel, skip_related_work=skip_related, skip_spark=skip_spark, skip_neutral=skip_neutral, balanced=balanced, data_dir=data_dir, calibration_path=calibration_path)
