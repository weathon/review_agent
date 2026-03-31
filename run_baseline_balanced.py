"""
Baseline with balanced sampling across score bins.

Stratified sampling: equal papers from each score bin (rounded to nearest int).
Then always predicts score=6, decision=Accept as baseline.

Usage:
  python run_baseline_balanced.py             # 100 papers, seed=42
  python run_baseline_balanced.py 100 4112
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path(__file__).parent / "AI-Scientist" / "review_iclr_bench"
RATINGS_FILE = BENCH_DIR / "ratings_subset.tsv"
PAPERS_DIR = BENCH_DIR / "iclr_parsed"

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


def stratified_sample(papers: list[dict], n: int, seed: int) -> list[dict]:
    """Sample equally from each score bin (rounded to int)."""
    rng = random.Random(seed)

    # Group by rounded avg score
    bins = defaultdict(list)
    for p in papers:
        bin_key = round(p["avg_score"])
        bins[bin_key].append(p)

    # Shuffle within each bin
    for k in bins:
        rng.shuffle(bins[k])

    sorted_bins = sorted(bins.keys())
    n_bins = len(sorted_bins)
    per_bin = n // n_bins
    remainder = n % n_bins

    print(f"Stratified sampling: {n_bins} bins, {per_bin} per bin (+{remainder} extra)")
    print(f"Bins: {', '.join(f'{k}({len(bins[k])})' for k in sorted_bins)}\n")

    samples = []
    for i, k in enumerate(sorted_bins):
        take = per_bin + (1 if i < remainder else 0)
        take = min(take, len(bins[k]))
        samples.extend(bins[k][:take])
        print(f"  Bin {k}: taking {take}/{len(bins[k])}")

    rng.shuffle(samples)
    print(f"\nTotal sampled: {len(samples)}\n")
    return samples


def main(n_samples: int = 100, seed: int = 42):
    print("=" * 72)
    print("BALANCED BASELINE: Stratified sampling, always predict score=6")
    print("=" * 72)

    gt_data = load_ground_truth()
    available = [r for r in gt_data if (PAPERS_DIR / f"{r['paper_id']}.txt").exists()]
    print(f"Loaded {len(gt_data)} papers, {len(available)} with parsed text.\n")

    samples = stratified_sample(available, n_samples, seed)

    results = []
    csv_path = Path(__file__).parent / "baseline_balanced_scores.csv"
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
                r["paper_id"], r["predicted_score"], r["predicted_decision"],
                f"{r['gt_avg_score']:.2f}", r["gt_decision"], r["gt_binary"],
                "YES" if r["match"] else "NO", *gt_padded,
            ])

    # Summary
    matches = sum(1 for r in results if r["match"])
    accuracy = matches / len(results) if results else 0

    print(f"\n{'=' * 72}")
    print("BALANCED BASELINE RESULTS")
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

    # Per-bin accuracy
    from collections import Counter
    bin_correct = Counter()
    bin_total = Counter()
    for r in results:
        b = round(r["gt_avg_score"])
        bin_total[b] += 1
        if r["match"]:
            bin_correct[b] += 1

    print(f"\nPer-bin accuracy:")
    for b in sorted(bin_total):
        acc = bin_correct[b] / bin_total[b] if bin_total[b] else 0
        print(f"  Score ~{b}: {bin_correct[b]}/{bin_total[b]} ({acc:.0%})")

    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if len(args) > 0 else 100
    seed = int(args[1]) if len(args) > 1 else 42
    main(n_samples=n, seed=seed)
