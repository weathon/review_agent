"""
Direct-scoring baseline: send the paper directly to MODEL_SCORER and ask it
to score without any sub-agent reviews.

This measures how well the scoring model performs when it reads the paper
itself, vs. the full multi-agent pipeline where specialized reviewers feed
into a merger+scorer.

Usage:
  python baselines/direct_review/run_direct_baseline.py                     # 10 papers, seed=42
  python baselines/direct_review/run_direct_baseline.py 50 3112 --parallel
  python baselines/direct_review/run_direct_baseline.py --run-name my_experiment
"""

# note: the downloaded files are already blanced

import asyncio
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from paper_reviewer import (
    decision_match,
    _get_client,
    _parse_score,
    _call_openai,
    match_label,
    sanitize_text,
    score_to_decision,
)

# Baseline uses its own model — not the pipeline's MODEL_SCORER
BASELINE_MODEL = "qwen/qwen3.6-plus:free"

# DEFAULT_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "iclr2025_data.old"
DEFAULT_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "iclr2025_data_v2"

VALID_SCORES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
CONCURRENCY = 5


def _snap_score(raw: float) -> float:
    return min(VALID_SCORES, key=lambda v: abs(v - raw))


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


DIRECT_SCORE_PROMPT = """\


Output your final review in this markdown format:

## Summary
2-3 sentence summary of the paper's contribution.

## Strengths
- strength 1 with evidence
- strength 2 with evidence

## Weaknesses
- weakness 1 — why it matters
- weakness 2 — why it matters

## Nice-to-Haves
- suggestion that would improve but is not a core flaw

## Novel Insights
One paragraph synthesizing genuinely novel observations.

## Potentially Missed Related Work
- paper — why relevant (or "None identified")

## Suggestions
- specific actionable suggestion


DO differentiate between papers of varying quality clearly: the content of the review should make it clear whether the paper is strong or weak.


Then please rate the paper:

Use the FULL scoring range — do NOT cluster around 5-6. Be discriminative:

Then assign an overall score from 0.0 to 10.0.

This is for a top-tier venue (ICLR, ~29% acceptance rate), most papers are scored lower than 6. \

Score based on these:
- novelty
- technical soundness
- empirical support
- significance
- clarity

Do NOT be afraid to give very high (>8) or very low (<4) scores! Good papers should be praised and bad paper should be found out.

Score continuously (e.g. 3.5, 4.7, 8.1). Use the full range — do not cluster \
around 5-6. Do not round to .5 or .0. give scores in x.2, 2.8, 7.3, etc. 

The score should be DISCRIMINATIVE. A weak paper is weak — \
give it a low score (1-3). A strong paper is strong — give it a high score (7-9). \
Do not cluster everything around 5. The quality difference between papers is real \
and your scores should reflect it.

## Scoring guide
- 10: Strong accept. Exceptional, field-advancing contribution.
- 8:  Accept.
- 6:  Borderline accept.
- 4:  Borderline reject.
- 2:  Reject.
- 0:  Strong reject. 
"""


async def score_paper_directly(
    client, paper_content: str,
) -> tuple[str, float, float]:
    """Send paper directly to BASELINE_MODEL, parse score. Returns (review_text, score, cost)."""

    user_prompt = (
        f"Here is the paper to review and score:\n\n"
        f"--- PAPER START ---\n{paper_content}\n--- PAPER END ---\n\n"
    )
    user_prompt += "Now read the paper carefully, assess it, and provide your score."

    review_text, cost_score = await _call_openai(
        client, "direct_score", DIRECT_SCORE_PROMPT, user_prompt, BASELINE_MODEL,
    )

    score, cost_parse = await _parse_score(client, review_text)
    print(f"    [direct_score] parsed score: {score} — ${cost_parse:.4f}")
    return review_text, score, cost_score + cost_parse


async def main(
    n_samples: int = 10,
    seed: int = 42,
    parallel: bool = False,
    balanced: bool = False,
    data_dir: str | None = None,
    run_name: str = "direct_baseline",
):
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print("DIRECT-SCORING BASELINE")
    print(f"  Scorer: {BASELINE_MODEL}")
    print(f"  Data:   {bench_dir}")
    print(f"  Mode:   {'balanced' if balanced else 'random'}")
    print("=" * 72)

    gt_data, papers_dir = load_ground_truth(bench_dir)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
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
    csv_path = out_dir / f"{run_name}_scores.csv"
    results_path = out_dir / f"{run_name}_results.md"

    with open(results_path, "w") as f:
        f.write("# Direct-Scoring Baseline Results\n\n")
        f.write(f"Model: {BASELINE_MODEL}\n\n")

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
            print(f"  [{i}/{len(samples)}] {pid}  GT={paper_info['gt_binary']}({paper_info['avg_score']:.1f})")
            start = time.time()
            try:
                review, score, cost = await score_paper_directly(client, paper_content)
                score = round(float(score), 1)
                decision = score_to_decision(score)
                elapsed = time.time() - start
                match = decision_match(decision, paper_info["gt_binary"])
                marker = "MATCH" if match is True else ("MISMATCH" if match is False else "N/A")
                print(f"    [{pid}] score={score} dec={decision} {marker} ({elapsed:.1f}s, ${cost:.4f})")
            except Exception as e:
                elapsed = time.time() - start
                print(f"    [{pid}] ERROR: {e} ({elapsed:.1f}s)")
                review = f"ERROR: {e}"
                score, decision, match, cost = None, "N/A", None, 0.0

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
                "review": review,
            }

            async with file_lock:
                results.append(r)
                completed[0] += 1
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    match_str = match_label(r["match"])
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
                with open(results_path, "a") as f:
                    f.write(f"## {r['paper_id']}\n\n")
                    f.write(f"- GT: {r['gt_decision']} (avg {r['gt_avg_score']:.1f})\n")
                    f.write(f"- Predicted: {r['predicted_decision']} ({r['predicted_score']}/10)\n")
                    match_str2 = "Yes" if r["match"] else ("No" if r["match"] is not None else "N/A")
                    f.write(f"- Match: {match_str2}\n\n")
                    f.write(f"### Review\n\n{r['review']}\n\n---\n\n")

    await asyncio.gather(*(
        process_paper(i, info) for i, info in enumerate(samples, 1)
    ))

    total_elapsed = time.time() - total_start

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("DIRECT-SCORING BASELINE RESULTS")
    print(f"{'=' * 72}")

    successful = [r for r in results if r["predicted_score"] is not None]
    valid = [r for r in results if r["match"] is not None]
    matches = sum(1 for r in valid if r["match"])
    accuracy = matches / len(valid) if valid else 0
    total_cost = sum(r.get("cost", 0.0) for r in results)

    print(f"Papers:    {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Decision eval: {len(valid)}")
    if valid:
        print(f"Correct:   {matches}/{len(valid)}")
        print(f"Accuracy:  {accuracy:.1%}")
    else:
        print("Correct:   N/A")
        print("Accuracy:  N/A (decision labels disabled)")
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
        match_str = match_label(r["match"])
        print(f"{r['paper_id']:<20} {r['gt_binary']:>10} {pred:>10} {r['gt_avg_score']:>10.1f} {pred_sc:>11} {match_str:>7}")

    # Per-bin accuracy
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

    if "--data-dir" in sys.argv:
        idx = sys.argv.index("--data-dir")
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]

    run_name = "direct_baseline"
    if "--run-name" in sys.argv:
        idx = sys.argv.index("--run-name")
        if idx + 1 < len(sys.argv):
            run_name = sys.argv[idx + 1]

    flag_values = {data_dir, run_name} - {None}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in flag_values]
    n = int(args[0]) if len(args) > 0 else 10
    seed = int(args[1]) if len(args) > 1 else 42
    asyncio.run(main(n_samples=n, seed=seed, balanced=balanced, data_dir=data_dir, run_name=run_name))
