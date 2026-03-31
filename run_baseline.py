"""
Baseline: always predict score=6, decision=Accept.
Uses davidheineman/iclr-2026 dataset.

Usage:
  python run_baseline.py             # 10 papers, seed=42
  python run_baseline.py 100 4112
  python run_baseline.py 100 4112 --balanced
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ACCEPT_THRESHOLD = 5.5
BASELINE_SCORE = 6.0
BASELINE_DECISION = "Accept"


def load_iclr2026() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("davidheineman/iclr-2026")["main"]
    rows = []
    for item in ds:
        if "Withdrawn" in item["venue"]:
            continue
        if not item["reviews"]:
            continue
        scores = [r["rating"] for r in item["reviews"] if r["rating"] and r["rating"] > 0]
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        gt_binary = "Accept" if avg_score >= ACCEPT_THRESHOLD else "Reject"
        rows.append({
            "paper_id": item["url"].split("=")[-1] if item["url"] else item["title"][:20],
            "title": item["title"],
            "scores": scores,
            "avg_score": avg_score,
            "decision": f"Inferred ({'Accept' if avg_score >= ACCEPT_THRESHOLD else 'Reject'})",
            "gt_binary": gt_binary,
        })
    return rows


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


def main(n_samples=10, seed=42, balanced=False):
    print("=" * 72)
    print(f"BASELINE (ICLR 2026): Always predict score={BASELINE_SCORE}, decision={BASELINE_DECISION}")
    print(f"Sampling: {'balanced' if balanced else 'random'}")
    print("=" * 72)

    print("Loading dataset...")
    gt_data = load_iclr2026()
    print(f"Loaded {len(gt_data)} papers.\n")

    if balanced:
        samples = stratified_sample(gt_data, n_samples, seed)
    else:
        random.seed(seed)
        samples = random.sample(gt_data, min(n_samples, len(gt_data)))
        print(f"Selected {len(samples)} papers (seed={seed}).\n")

    results = []
    csv_path = Path(__file__).parent / "baseline_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "title", "pred_score", "pred_decision",
                     "gt_avg_score", "gt_decision", "gt_binary", "match",
                     "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3",
                     "gt_score_4", "gt_score_5", "gt_score_6"])

    for i, p in enumerate(samples, 1):
        match = BASELINE_DECISION == p["gt_binary"]
        marker = "MATCH" if match else "MISMATCH"
        print(f"[{i}/{len(samples)}] {p['title'][:55]}...  GT={p['gt_binary']}({p['avg_score']:.1f})  {marker}")

        r = {**p, "predicted_score": BASELINE_SCORE, "predicted_decision": BASELINE_DECISION, "match": match}
        results.append(r)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            gt_padded = r["scores"] + [""] * (7 - len(r["scores"]))
            w.writerow([r["paper_id"], r["title"], r["predicted_score"], r["predicted_decision"],
                         f"{r['avg_score']:.2f}", r["decision"], r["gt_binary"],
                         "YES" if match else "NO", *gt_padded[:7]])

    matches = sum(1 for r in results if r["match"])
    accuracy = matches / len(results) if results else 0
    gt_accepts = sum(1 for r in results if r["gt_binary"] == "Accept")
    gt_rejects = len(results) - gt_accepts
    paired = [(r["avg_score"], r["predicted_score"]) for r in results]
    mean_gt = sum(p[0] for p in paired) / len(paired)

    print(f"\n{'=' * 72}")
    print(f"Papers: {len(results)} | Correct: {matches}/{len(results)} | Accuracy: {accuracy:.1%}")
    print(f"GT split: {gt_accepts} Accept / {gt_rejects} Reject")
    print(f"Mean GT: {mean_gt:.2f} | Mean Pred: {BASELINE_SCORE:.2f} | Avg diff: {sum(abs(g-p) for g,p in paired)/len(paired):.2f}")

    # Per-bin
    bin_c, bin_t = Counter(), Counter()
    for r in results:
        b = round(r["avg_score"])
        bin_t[b] += 1
        if r["match"]:
            bin_c[b] += 1
    print("\nPer-bin:")
    for b in sorted(bin_t):
        print(f"  ~{b}: {bin_c[b]}/{bin_t[b]} ({bin_c[b]/bin_t[b]:.0%})")

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    balanced = "--balanced" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if len(args) > 0 else 10
    seed = int(args[1]) if len(args) > 1 else 42
    main(n_samples=n, seed=seed, balanced=balanced)
