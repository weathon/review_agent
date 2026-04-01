"""
Baseline: always predict score=6, decision=Accept.

Uses the same sampling as run_iclr_bench.py for fair comparison.

Usage:
  python run_baseline.py             # 10 papers, seed=42
  python run_baseline.py 100 4112
  python run_baseline.py 100 4112 --balanced
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_BENCH_DIR = Path(__file__).parent / "AI-Scientist" / "review_iclr_bench"

VALID_SCORES = [1.0, 3.0, 5.0, 6.0, 8.0, 10.0]
BASELINE_SCORE = 6.0
BASELINE_DECISION = "Accept"


def load_ground_truth(bench_dir: Path) -> tuple[list[dict], Path]:
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
                    "scores": scores,
                    "avg_score": float(row.get("avg_score", 0)),
                    "decision": decision,
                    "gt_binary": gt_binary,
                })
        return rows, papers_dir

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
                "scores": scores,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "decision": decision,
                "gt_binary": gt_binary,
            })
    return rows, papers_dir


def stratified_sample(papers, n, seed):
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
    print(f"Stratified: {n_bins} bins, {per_bin}/bin (+{remainder} extra)")
    print(f"Bins: {', '.join(f'{k}({len(bins[k])})' for k in sorted_bins)}")
    samples = []
    for i, k in enumerate(sorted_bins):
        take = min(per_bin + (1 if i < remainder else 0), len(bins[k]))
        samples.extend(bins[k][:take])
    rng.shuffle(samples)
    print(f"Total: {len(samples)}\n")
    return samples


def main(
    n_samples: int = 10,
    seed: int = 42,
    balanced: bool = False,
    data_dir: str | None = None,
    calibration_path: str | None = None,
):
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print(f"BASELINE: Always predict score=6, decision=Accept")
    print(f"Data: {bench_dir}")
    print(f"Sampling: {'balanced' if balanced else 'random'}")
    print("=" * 72)

    calibration_ids = set()
    if calibration_path:
        ids_path = Path(calibration_path).parent / "calibration_ids.json"
        if ids_path.exists():
            calibration_ids = set(json.loads(ids_path.read_text()))
            print(f"Excluding {len(calibration_ids)} calibration papers")

    gt_data, papers_dir = load_ground_truth(bench_dir)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    if calibration_ids:
        available = [r for r in available if r["paper_id"] not in calibration_ids]
    print(f"Loaded {len(gt_data)} papers, {len(available)} with parsed text.")

    if balanced:
        samples = stratified_sample(available, n_samples, seed)
    else:
        random.seed(seed)
        samples = random.sample(available, min(n_samples, len(available)))
        print(f"Selected {len(samples)} papers (seed={seed}).\n")

    results = []
    csv_path = Path(__file__).parent / "baseline_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "pred_score", "pred_decision", "gt_avg_score", "gt_decision", "gt_binary", "match",
                     "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3", "gt_score_4", "gt_score_5", "gt_score_6"])

    for i, paper_info in enumerate(samples, 1):
        pid = paper_info["paper_id"]
        match = BASELINE_DECISION == paper_info["gt_binary"]
        marker = "MATCH" if match else "MISMATCH"

        print(f"[{i}/{len(samples)}] {pid}  GT={paper_info['gt_binary']}({paper_info['avg_score']:.1f})  Pred={BASELINE_DECISION}({BASELINE_SCORE})  {marker}")

        r = {
            "paper_id": pid,
            "gt_decision": paper_info["decision"],
            "gt_binary": paper_info["gt_binary"],
            "gt_avg_score": paper_info["avg_score"],
            "gt_scores": paper_info["scores"],
            "predicted_score": BASELINE_SCORE,
            "predicted_decision": BASELINE_DECISION,
            "match": match,
        }
        results.append(r)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            gt_padded = r["gt_scores"] + [""] * (7 - len(r["gt_scores"]))
            w.writerow([
                r["paper_id"],
                r["predicted_score"],
                r["predicted_decision"],
                f"{r['gt_avg_score']:.2f}",
                r["gt_decision"],
                r["gt_binary"],
                "YES" if r["match"] else "NO",
                *gt_padded,
            ])

    # Summary
    matches = sum(1 for r in results if r["match"])
    accuracy = matches / len(results) if results else 0

    print(f"\n{'=' * 72}")
    print("BASELINE RESULTS")
    print(f"{'=' * 72}")
    print(f"Papers:    {len(results)}")
    print(f"Correct:   {matches}/{len(results)}")
    print(f"Accuracy:  {accuracy:.1%}")

    gt_accepts = sum(1 for r in results if r["gt_binary"] == "Accept")
    gt_rejects = sum(1 for r in results if r["gt_binary"] == "Reject")
    print(f"GT split:  {gt_accepts} Accept / {gt_rejects} Reject")

    paired = [(r["gt_avg_score"], r["predicted_score"]) for r in results]
    mean_gt = sum(p[0] for p in paired) / len(paired)
    mean_pred = sum(p[1] for p in paired) / len(paired)
    print(f"\nMean GT Score:    {mean_gt:.2f}")
    print(f"Mean Pred Score:  {mean_pred:.2f}")
    print(f"Score diff (avg): {sum(abs(g - p) for g, p in paired) / len(paired):.2f}")

    print(f"\n{'Paper ID':<20} {'GT':>10} {'Predicted':>10} {'GT Score':>10} {'Pred Score':>11} {'Match':>7}")
    print("─" * 72)
    for r in results:
        match_str = "YES" if r["match"] else "NO"
        print(f"{r['paper_id']:<20} {r['gt_binary']:>10} {r['predicted_decision']:>10} {r['gt_avg_score']:>10.1f} {r['predicted_score']:>11.1f} {match_str:>7}")

    print(f"\nCSV saved to: {csv_path}")


    # Per-bin accuracy
    bin_c, bin_t = Counter(), Counter()
    for r in results:
        b = round(r["gt_avg_score"])
        bin_t[b] += 1
        if r["match"]:
            bin_c[b] += 1
    print("\nPer-bin:")
    for b in sorted(bin_t):
        print(f"  ~{b}: {bin_c[b]}/{bin_t[b]} ({bin_c[b]/bin_t[b]:.0%})")

    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python run_baseline.py [n] [seed] [options]")
        print()
        print("Options:")
        print("  --balanced              Stratified sampling across score bins")
        print("  --data-dir <path>       Dataset directory")
        print("  --calibration <path>    Calibration file; excludes calibration_ids.json")
        sys.exit(0)

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
    main(
        n_samples=n,
        seed=seed,
        balanced=balanced,
        data_dir=data_dir,
        calibration_path=calibration_path,
    )
