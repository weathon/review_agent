"""
Structured-review baseline: a single model produces a review matching the
exact same sections as the multi-agent merger, then scores using calibration
context in a second turn.

This is a STRONGER single-model baseline than the simple review baseline:
it outputs the same structured sections (Summary, Strengths, Weaknesses,
Nice-to-Haves, Novel Insights, Potentially Missed Related Work, Suggestions)
but without the multi-agent pipeline (no harsh/neutral/spark/related-work
sub-agents, no merger synthesis).

Two-turn flow per paper:
  Turn 1: MODEL reads the paper → writes a structured review (no score)
  Turn 2: Given the review + calibration examples → comparative scoring

Calibration: use build_calibration.py in this directory.

Usage:
  python baselines/structured_review/run_baseline.py 50 3112 --data-dir iclr2025_data --calibration baselines/structured_review/calibration.md
  python baselines/structured_review/run_baseline.py 50 3112 --balanced --calibration baselines/structured_review/calibration.md
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
    decision_match,
    _get_client,
    _parse_score,
    _call_openai,
    match_label,
    sanitize_text,
    score_to_decision,
)

BASELINE_MODEL = "z-ai/glm-5"

DEFAULT_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "iclr2025_data"
CONCURRENCY = 5

# ── Review prompt: same sections as the merger output ────────────────

REVIEW_PROMPT = """\
You are a senior academic reviewer / area chair for a top ML venue (ICLR). \
You will be given a paper. Read it carefully and write a thorough, \
expert-level review.

Your review must cover ALL of the following sections in the EXACT format \
below. Be thoughtful, balanced, and evidence-based. Cite specific sections, \
equations, figures, tables, or claims when making points.

Cross-check your own criticisms against the paper content. Remove criticisms \
that are factually wrong or misunderstand the contribution. Remove pure \
formatting/style nitpicks. Keep criticisms that are factually correct AND \
substantive — a single valid concern still counts.

Rules:
- Strengths must be backed by evidence from the paper.
- Weaknesses must be real, substantive issues — not nitpicks or formatting \
  complaints. Missing baselines, unsupported claims, and flawed experiments \
  are real weaknesses, not nice-to-haves.
- Nice-to-Haves are improvements that would help but are NOT core flaws.
- Novel Insights should highlight genuinely novel observations about the work.
- Potentially Missed Related Work should list real, existing papers the \
  authors may have missed (especially recent ones, 2022-2026). Present as \
  suggestions only — do not penalize the paper for these.
- Suggestions should be specific and actionable.

Output format (strictly follow):

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
One paragraph synthesizing genuinely novel observations about this work.

## Potentially Missed Related Work
- paper — why relevant (or "None identified")

## Suggestions
- specific actionable suggestion

Do NOT output any numerical scores or subscores. Do NOT output an \
accept/reject decision. Your job is ONLY to produce the qualitative review. \
Scoring will be done separately.
"""

# ── Score prompt: same comparative scoring as the main pipeline ──────

SCORE_PROMPT = """\
You previously wrote a consolidated review of a paper. Now you must assign \
a calibrated overall score from 1.0 to 10.0 using COMPARATIVE SCORING \
against calibration examples. Note that the review should be written to be very harsh, \
but that doesn't mean the score should be low. Compare with similar level paper in the calibration \
set to determine the score.

Base your score on YOUR OWN review above — the strengths, weaknesses, \
nice-to-haves, and suggestions you already identified. Do not re-evaluate \
from scratch.

## Comparative Scoring Procedure (MANDATORY when calibration examples exist)

You MUST follow these steps in order:

**Step 1 — Identify the LOWER BOUND paper.**
Find the calibration example that is clearly WORSE than or roughly equal to \
the current paper. This is the paper whose quality is just below the current \
paper's level. Note its human average score — this is your score floor.

**Step 2 — Identify the UPPER BOUND paper.**
Find the calibration example that is clearly BETTER than or roughly equal to \
the current paper. This is the paper whose quality is just above the current \
paper's level. Note its human average score — this is your score ceiling.

**Step 3 — Compare dimension by dimension.**
For BOTH the lower and upper bound papers, compare against the current paper on:
- novelty
- technical soundness
- empirical support
- significance
- clarity

For each dimension, determine: is the current paper closer to the lower bound \
or the upper bound?

**Step 4 — Interpolate the final score.**
Place the current paper's score BETWEEN the lower bound and upper bound scores \
based on the dimension-by-dimension comparison. If the current paper is closer \
to the upper bound on most dimensions, score it closer to the upper bound's \
human score; if closer to the lower bound, score it closer to that.

**Edge cases:**
- If the current paper is WORSE than ALL calibration examples, score it below \
  the lowest calibration score.
- If the current paper is BETTER than ALL calibration examples, score it above \
  the highest calibration score.
- If there is only one calibration example nearby, use it as a single anchor \
  and adjust up or down based on the comparison.

## Scoring Constraints

The "score" field is a CONTINUOUS value from 1.0 to 10.0 (e.g. 3.5, 4.7, 6.2, 8.1). \
Use the full range — do NOT cluster around 5-6. Be DISCRIMINATIVE:
- A truly bad paper deserves a 2.0, not a 4.5.
- A truly great paper deserves a 9.0, not a 7.0.
- Do NOT hedge toward the middle. Commit to your assessment.
- However, do not over-penalize papers for a long list of minor points if the core contribution is sound.
- A few serious flaws matter more than many small nitpicks.
- NEVER give a 6.0 score, no matter what. A score of 6 is a non-committal fence-sit. If the \
  paper is even slightly positive, give 6.5 or 7. If it is even slightly negative, \
  give 5.5 or 5. You must decide which side of the borderline the paper falls on.

Scoring guide (use only when NO calibration examples are available):
- 9.0-10.0: Strong accept. Exceptional, field-advancing contribution.
- 7.0-8.9:  Accept. Clear contribution, solid execution, minor issues.
- 5.0-5.9:  Borderline reject. Has some merit but weaknesses outweigh.
- 3.0-4.9:  Reject. Significant issues with claims, method, or evaluation.
- 1.0-2.9:  Strong reject. Fundamental flaws, unclear contribution, or wrong.

Real flaws go in "weaknesses" and hurt the score. \
Nice-to-haves could affect the scores but not significantly as weaknesses. \
But be honest — missing baselines, unsupported claims, and flawed experiments \
are real weaknesses, not nice-to-haves.

## Fairness Check

Before deciding on the final score, ask:
- Are the main concerns actually central to the paper's claims?
- Would these concerns realistically cause rejection at the target venue?
- Is the paper still a meaningful contribution despite its weaknesses?

Return the score using the provided structured response schema.
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
    Turn 1: generate structured review (same sections as merger).
    Turn 2: comparative score with calibration.
    Returns (review_text, score, total_cost).
    """
    # ── Turn 1: Structured Review ───────────────────────────────
    review_user = (
        f"NOTE: This paper was extracted from PDF by an automated parser. "
        f"There may be formatting artifacts such as broken equations, garbled "
        f"tables, misplaced figure references, or OCR errors. These are parser "
        f"issues, NOT problems with the paper itself. Do NOT treat formatting "
        f"artifacts as weaknesses.\n\n"
        f"Here is the paper to review:\n\n"
        f"--- PAPER START ---\n{paper_content}\n--- PAPER END ---\n\n"
        f"Write your structured review now, following the exact output format."
    )

    print(f"    [turn1_review] generating structured review ({BASELINE_MODEL}) ...")
    review_text, cost_review = await _call_openai(
        client, "structured_baseline_t1", REVIEW_PROMPT, review_user, BASELINE_MODEL,
    )
    print(f"    [turn1_review] done ({len(review_text)} chars, ${cost_review:.4f})")

    # ── Turn 2: Comparative Score ───────────────────────────────
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
        "Follow the Comparative Scoring Procedure. Use the FULL range. "
        "Commit to your assessment."
    )

    print(f"    [turn2_score] scoring ({BASELINE_MODEL}) ...")
    score_text, cost_score = await _call_openai(
        client, "structured_baseline_t2", SCORE_PROMPT, score_user, BASELINE_MODEL,
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
    print("STRUCTURED-REVIEW BASELINE (merger-format, single model)")
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
                match = decision_match(decision, paper_info["gt_binary"])
                marker = "MATCH" if match is True else ("MISMATCH" if match is False else "N/A")
                print(f"    [{pid}] score={score} dec={decision} {marker} ({elapsed:.1f}s, ${cost:.4f})")
            except Exception as e:
                elapsed = time.time() - start
                print(f"    [{pid}] ERROR: {e} ({elapsed:.1f}s)")
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

    await asyncio.gather(*(
        process_paper(i, info) for i, info in enumerate(samples, 1)
    ))

    total_elapsed = time.time() - total_start

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("STRUCTURED-REVIEW BASELINE RESULTS")
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
