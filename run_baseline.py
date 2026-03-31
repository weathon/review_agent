"""
Baseline: always predict score=6, decision=Accept.

Uses the same sampling as run_iclr_bench.py for fair comparison.

Usage:
  python run_baseline.py             # 10 papers, seed=42
  python run_baseline.py 10 42
"""

import csv
import random
import re
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent / "AI-Scientist" / "review_iclr_bench"
RATINGS_FILE = BENCH_DIR / "ratings_subset.tsv"
PAPERS_DIR = BENCH_DIR / "iclr_parsed"

VALID_SCORES = [1.0, 3.0, 5.0, 6.0, 8.0, 10.0]
BASELINE_SCORE = 6.0
BASELINE_DECISION = "Accept"


def load_ground_truth() -> list[dict]:
    rows = []
    with open(RATINGS_FILE, "r") as f:
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
    return rows


def main(n_samples: int = 10, seed: int = 42):
    print("=" * 72)
    print("BASELINE: Always predict score=6, decision=Accept")
    print("=" * 72)

    gt_data = load_ground_truth()
    available = [r for r in gt_data if (PAPERS_DIR / f"{r['paper_id']}.txt").exists()]
    print(f"Loaded {len(gt_data)} papers, {len(available)} with parsed text.")

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


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if len(args) > 0 else 10
    seed = int(args[1]) if len(args) > 1 else 42
    main(n_samples=n, seed=seed)
