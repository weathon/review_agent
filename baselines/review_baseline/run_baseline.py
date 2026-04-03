"""
Review-then-score baseline: a single model generates a simple review first,
then scores using calibration context in a second turn.

Two-turn flow per paper:
  Turn 1: MODEL reads the paper → writes a review (no score)
  Turn 2: Given the review + calibration examples → produces a score

This is a single-model ablation of the full pipeline: same model does
both reviewing and scoring, without the multi-agent sub-reviews.

Calibration: use build_calibration.py in this directory.

Usage:
  python baselines/review_baseline/run_baseline.py 50 3112 --data-dir iclr2025_data --calibration baselines/review_baseline/calibration.md
  python baselines/review_baseline/run_baseline.py 50 3112 --balanced --calibration baselines/review_baseline/calibration.md
"""

import asyncio
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path so we can import from paper_reviewer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from paper_reviewer import (
    _get_client,
    _parse_score,
    _call_openai,
    sanitize_text,
    score_to_decision,
)

# Baseline uses its own model — not the pipeline's MODEL_SCORER
BASELINE_MODEL = "z-ai/glm-5"

DEFAULT_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "iclr2025_data"
CONCURRENCY = 5

REVIEW_PROMPT = """\
You are an experienced academic reviewer for a top ML venue (ICLR).

You will be given a paper. Read it carefully and write a detailed review covering:
1. **Summary** — What the paper does and its main contribution.
2. **Strengths** — What is done well (novelty, clarity, experiments, etc.).
3. **Weaknesses** — Genuine concerns (methodology, evaluation, missing baselines, etc.).
4. **Questions** — Key clarifications needed from the authors.

Do NOT include a numerical score — just the review text.
"""

SCORE_PROMPT = """\
You are an experienced academic reviewer. You have just written a review of a paper.
Now assign a single overall score from 1.0 to 10.0 based on your review.

Use the FULL scoring range — do NOT cluster around 5-6. Be discriminative:
- 9.0-10.0: Strong accept. Exceptional, field-advancing contribution.
- 7.0-8.9:  Accept. Clear contribution, solid execution, minor issues.
- 5.0-5.9:  Borderline reject. Has some merit but weaknesses outweigh.
- 3.0-4.9:  Reject. Significant issues with claims, method, or evaluation.
- 1.0-2.9:  Strong reject. Fundamental flaws, unclear contribution, or wrong.

NEVER give exactly 6.0. If slightly positive give 7, if slightly negative give 5.
Commit to your assessment — do not hedge toward the middle.
"""


def load_ground_truth(bench_dir: Path) -> tuple[list[dict], Path]:
    csv_file = bench_dir / "ratings.csv"
    if csv_file.exists():
        papers_dir = bench_dir / "papers"
        rows = []
        with open(csv_file) as f:
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
    raise FileNotFoundError(f"No ratings.csv in {bench_dir}")


def stratified_sample(papers, n, seed):
    rng = random.Random(seed)
    bins = defaultdict(list)
    for p in papers:
        bins[round(p["avg_score"])].append(p)
    for k in bins:
        rng.shuffle(bins[k])
    sorted_bins = sorted(bins.keys())
    per_bin = n // len(sorted_bins)
    remainder = n % len(sorted_bins)
    samples = []
    for i, k in enumerate(sorted_bins):
        take = min(per_bin + (1 if i < remainder else 0), len(bins[k]))
        samples.extend(bins[k][:take])
    rng.shuffle(samples)
    print(f"Stratified: {len(sorted_bins)} bins, {per_bin}/bin (+{remainder} extra), total={len(samples)}")
    return samples


async def review_then_score(
    client, paper_content: str, calibration_context: str = "",
) -> tuple[str, float, float]:
    """
    Turn 1: generate review. Turn 2: score with calibration.
    Returns (review_text, score, total_cost).
    """
    # ── Turn 1: Review ───────────────────────────────────────────
    review_user = (
        f"Here is the paper to review:\n\n"
        f"--- PAPER START ---\n{paper_content}\n--- PAPER END ---\n\n"
        f"Write your detailed review now."
    )

    print(f"    [turn1_review] generating review ({BASELINE_MODEL}) ...")
    review_text, cost_review = await _call_openai(
        client, "review_baseline_t1", REVIEW_PROMPT, review_user, BASELINE_MODEL,
    )
    print(f"    [turn1_review] done ({len(review_text)} chars, ${cost_review:.4f})")

    # ── Turn 2: Score ────────────────────────────────────────────
    score_user = (
        f"Here is the review you wrote:\n\n"
        f"--- YOUR REVIEW ---\n{review_text}\n--- END YOUR REVIEW ---\n\n"
    )
    if calibration_context:
        score_user += (
            f"Here are calibration examples — reviews of other papers paired with "
            f"ACTUAL human reviewer scores. Use these as your primary scoring anchor:\n\n"
            f"--- CALIBRATION EXAMPLES ---\n{calibration_context}\n"
            f"--- END CALIBRATION EXAMPLES ---\n\n"
        )
    score_user += (
        "Based on the review above, assign a single overall score from 1.0 to 10.0.\n"
        "Use the FULL range. Commit to your assessment."
    )

    print(f"    [turn2_score] scoring ({BASELINE_MODEL}) ...")
    score_text, cost_score = await _call_openai(
        client, "review_baseline_t2", SCORE_PROMPT, score_user, BASELINE_MODEL,
    )
    print(f"    [turn2_score] done (${cost_score:.4f})")

    score, cost_parse = await _parse_score(client, score_text)
    print(f"    [score_parser] parsed score: {score} — ${cost_parse:.4f}")

    return review_text, score, cost_review + cost_score + cost_parse


async def main(
    n_samples: int = 10,
    seed: int = 42,
    balanced: bool = False,
    data_dir: str | None = None,
    calibration_path: str | None = None,
):
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print("REVIEW-THEN-SCORE BASELINE")
    print(f"  Model: {BASELINE_MODEL}")
    print(f"  Data:  {bench_dir}")
    print(f"  Mode:  {'balanced' if balanced else 'random'}")
    print("=" * 72)

    calibration_context = ""
    calibration_ids = set()
    if calibration_path:
        cal_path = Path(calibration_path)
        if cal_path.exists():
            calibration_context = cal_path.read_text(encoding="utf-8")
            print(f"Loaded calibration: {cal_path} ({len(calibration_context):,} chars)")
            ids_path = cal_path.parent / "calibration_ids.json"
            if ids_path.exists():
                calibration_ids = set(json.load(open(ids_path)))
                print(f"Excluding {len(calibration_ids)} calibration papers")

    gt_data, papers_dir = load_ground_truth(bench_dir)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    if calibration_ids:
        available = [r for r in available if r["paper_id"] not in calibration_ids]
    print(f"Loaded {len(gt_data)} papers, {len(available)} available.\n")

    if balanced:
        samples = stratified_sample(available, n_samples, seed)
    else:
        random.seed(seed)
        samples = random.sample(available, min(n_samples, len(available)))
        print(f"Selected {len(samples)} papers (seed={seed}).\n")

    client = _get_client()
    results = []
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "scores.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "pred_score", "pred_decision", "gt_avg_score",
                     "gt_decision", "gt_binary", "match", "cost",
                     "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3", "gt_score_4", "gt_score_5", "gt_score_6"])

    semaphore = asyncio.Semaphore(CONCURRENCY)
    file_lock = asyncio.Lock()
    completed = [0]
    total_start = time.time()

    async def process_paper(i: int, paper_info: dict):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"
        paper_content = paper_path.read_text(encoding="utf-8", errors="replace")
        paper_content = sanitize_text(paper_content)

        async with semaphore:
            print(f"\n  [{i}/{len(samples)}] {pid}  GT={paper_info['gt_binary']}({paper_info['avg_score']:.1f})")
            start = time.time()
            try:
                review, score, cost = await review_then_score(client, paper_content, calibration_context)
                score = round(float(score), 1)
                decision = score_to_decision(score)
                elapsed = time.time() - start
                match = decision == paper_info["gt_binary"]
                marker = "MATCH" if match else "MISMATCH"
                print(f"    [{pid}] score={score} dec={decision} {marker} ({elapsed:.1f}s, ${cost:.4f})")
            except Exception as e:
                elapsed = time.time() - start
                print(f"    [{pid}] ERROR: {e} ({elapsed:.1f}s)")
                score, decision, match, cost = None, None, None, 0.0

            r = {
                "paper_id": pid,
                "gt_decision": paper_info["decision"],
                "gt_binary": paper_info["gt_binary"],
                "gt_avg_score": paper_info["avg_score"],
                "gt_scores": paper_info["scores"],
                "predicted_score": score,
                "predicted_decision": decision,
                "match": match,
                "cost": cost,
            }

            async with file_lock:
                results.append(r)
                completed[0] += 1
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    match_str = "YES" if r["match"] else ("NO" if r["match"] is not None else "ERROR")
                    gt_padded = r["gt_scores"] + [""] * (7 - len(r["gt_scores"]))
                    w.writerow([
                        r["paper_id"],
                        r["predicted_score"],
                        r["predicted_decision"],
                        f"{r['gt_avg_score']:.2f}",
                        r["gt_decision"],
                        r["gt_binary"],
                        match_str,
                        f"{r['cost']:.4f}",
                        *gt_padded,
                    ])

    await asyncio.gather(*(
        process_paper(i, info) for i, info in enumerate(samples, 1)
    ))

    total_elapsed = time.time() - total_start

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("REVIEW-THEN-SCORE BASELINE RESULTS")
    print(f"{'=' * 72}")

    valid = [r for r in results if r["match"] is not None]
    matches = sum(1 for r in valid if r["match"])
    accuracy = matches / len(valid) if valid else 0
    total_cost = sum(r.get("cost", 0.0) for r in results)

    print(f"Papers:    {len(results)}")
    print(f"Valid:     {len(valid)}")
    print(f"Correct:   {matches}/{len(valid)}")
    print(f"Accuracy:  {accuracy:.1%}")
    print(f"Time:      {total_elapsed:.1f}s")
    print(f"Cost:      ${total_cost:.4f}")

    paired = [(r["gt_avg_score"], r["predicted_score"]) for r in results if r["predicted_score"] is not None]
    if paired:
        mean_gt = sum(p[0] for p in paired) / len(paired)
        mean_pred = sum(p[1] for p in paired) / len(paired)
        mae = sum(abs(g - p) for g, p in paired) / len(paired)
        print(f"\nMean GT:    {mean_gt:.2f}")
        print(f"Mean Pred:  {mean_pred:.2f}")
        print(f"MAE:        {mae:.2f}")

    print(f"\n{'Paper ID':<20} {'GT':>10} {'Predicted':>10} {'GT Score':>10} {'Pred Score':>11} {'Match':>7}")
    print("─" * 72)
    for r in results:
        pred = r["predicted_decision"] or "N/A"
        pred_sc = f"{r['predicted_score']:.1f}" if r["predicted_score"] else "N/A"
        match_str = "YES" if r["match"] else ("NO" if r["match"] is not None else "ERR")
        print(f"{r['paper_id']:<20} {r['gt_binary']:>10} {pred:>10} {r['gt_avg_score']:>10.1f} {pred_sc:>11} {match_str:>7}")

    bin_c, bin_t = Counter(), Counter()
    for r in valid:
        b = round(r["gt_avg_score"])
        bin_t[b] += 1
        if r["match"]:
            bin_c[b] += 1
    if bin_t:
        print("\nPer-bin accuracy:")
        for b in sorted(bin_t):
            print(f"  ~{b}: {bin_c[b]}/{bin_t[b]} ({bin_c[b]/bin_t[b]:.0%})")

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
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
    asyncio.run(main(n_samples=n, seed=seed, balanced=balanced, data_dir=data_dir, calibration_path=calibration_path))
