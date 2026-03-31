"""
ICLR Benchmark Runner — hybrid Claude Code SDK + OpenRouter.

Claude calls (critic + merger) are free via Claude Code SDK.
All other calls go through OpenRouter.

Usage:
  python run_iclr_bench.py                          # 10 papers, sequential
  python run_iclr_bench.py 5                        # 5 papers
  python run_iclr_bench.py 10 42 --parallel         # parallel OpenRouter agents
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
    run_reviewer_claude,
    run_reviewer_openrouter,
    sanitize_text,
)

# ── Paths ─────────────────────────────────────────────────────────────

BENCH_DIR = Path(__file__).parent / "AI-Scientist" / "review_iclr_bench"
RATINGS_FILE = BENCH_DIR / "ratings_subset.tsv"
PAPERS_DIR = BENCH_DIR / "iclr_parsed"

# ── Helpers ───────────────────────────────────────────────────────────


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


VALID_SCORES = [1.0, 3.0, 5.0, 6.0, 8.0, 10.0]


def _snap_score(raw: float) -> float:
    """Snap a raw score to the nearest valid ICLR score."""
    return min(VALID_SCORES, key=lambda v: abs(v - raw))


def parse_score_and_decision(review_text: str) -> tuple[float | None, str | None]:
    """Parse JSON output from merger. Falls back to regex if JSON fails."""
    score = None
    decision = None

    # Try JSON parse first — extract from ```json block or raw JSON
    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", review_text)
    if not json_match:
        json_match = re.search(r"```\s*(\{[\s\S]*?\})\s*```", review_text)
    if not json_match:
        json_match = re.search(r"(\{[\s\S]*\})", review_text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            raw_score = data.get("score")
            if raw_score is not None:
                score = round(float(raw_score), 1)
            dec = data.get("decision", "")
            if dec.lower() in ("accept", "reject"):
                decision = dec.capitalize()
            if decision is None and score is not None:
                decision = "Accept" if score >= 5.5 else "Reject"
            return score, decision
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Regex fallback for non-JSON output
    score_match = re.search(r"\*{0,2}(\d+(?:\.\d+)?)\*{0,2}\s*/\s*\*{0,2}10\*{0,2}", review_text)
    if not score_match:
        score_match = re.search(r"Final Score[:\s\n]*\*{0,2}(\d+(?:\.\d+)?)\*{0,2}", review_text, re.IGNORECASE)
    if score_match:
        score = round(float(score_match.group(1)), 1)

    dec_match = re.search(r"###\s*Decision\s*\n+\s*\*{0,2}(Accept|Reject)\*{0,2}", review_text, re.IGNORECASE)
    if not dec_match:
        dec_match = re.search(r"Decision[:\s]*\*{0,2}(Accept|Reject)\*{0,2}", review_text, re.IGNORECASE)
    if dec_match:
        decision = dec_match.group(1).capitalize()

    if decision is None and score is not None:
        decision = "Accept" if score >= 5.5 else "Reject"

    return score, decision


async def review_single_paper(
    paper_id: str, paper_path: Path, parallel: bool = False, skip_related_work: bool = False, skip_spark: bool = False,
) -> dict:
    """Run the full pipeline on one paper."""
    paper_content = paper_path.read_text(encoding="utf-8", errors="replace")
    paper_content = sanitize_text(paper_content)

    max_chars = 60_000
    if len(paper_content) > max_chars:
        paper_content = paper_content[:max_chars] + "\n\n[... truncated for length ...]"

    print(f"  Paper length: {len(paper_content):,} chars")

    client = _get_client()
    pp = str(paper_path)

    sep = "~" * 60

    # Stage 1: Critic (Claude SDK) || Neutral + Related Work (OpenRouter)
    if parallel:
        tasks = [
            run_reviewer_claude("harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue="ICLR"),
            run_reviewer_openrouter(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue="ICLR"),
        ]
        if not skip_related_work:
            print("  Stage 1: Critic (Claude) || Neutral + Related Work (OpenRouter) ...")
            tasks.append(run_related_work_search(client, paper_content))
            harsh_review, neutral_review, related_work = await asyncio.gather(*tasks)
        else:
            print("  Stage 1: Critic (Claude) || Neutral (OpenRouter) ...")
            harsh_review, neutral_review = await asyncio.gather(*tasks)
            related_work = "Related work search was skipped."
    else:
        print("  Stage 1: Critic + Neutral (sequential) ...")
        harsh_review = await run_reviewer_claude("harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue="ICLR")
        neutral_review = await run_reviewer_openrouter(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue="ICLR")
        if not skip_related_work:
            related_work = await run_related_work_search(client, paper_content)
        else:
            related_work = "Related work search was skipped."

    for label, text in [("harsh_critic", harsh_review), ("neutral", neutral_review)]:
        print(f"\n  {sep}\n  [{label} output] ({len(text)} chars)\n  {sep}\n{text}\n")
        if not text.strip():
            print(f"  *** WARNING: {label} returned empty output ***")
    if not skip_related_work:
        print(f"\n  {sep}\n  [related_work output] ({len(related_work)} chars)\n  {sep}\n{related_work}\n")

    # Stage 2: Spark Finder (Claude SDK — waits for stage 1)
    if not skip_spark:
        print("  Stage 2: Spark Finder (Claude SDK) ...")
        spark_review = await run_reviewer_claude(
            "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue="ICLR",
        )
        print(f"\n  {sep}\n  [spark_finder output]\n  {sep}\n{spark_review}\n")
    else:
        spark_review = "Spark finder was skipped."

    # Stage 3: Merger (Claude SDK — waits for everything)
    print("  Stage 3: Merger (Claude SDK) ...")
    final_review = await run_merger(
        harsh_review, neutral_review,
        spark_review, related_work, paper_content,
    )
    print(f"\n  {sep}\n  [merger output]\n  {sep}\n{final_review}\n")

    score, decision = parse_score_and_decision(final_review)

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


async def main(n_samples: int = 10, seed: int = 42, parallel: bool = False, skip_related_work: bool = False, skip_spark: bool = False, balanced: bool = False):
    print("=" * 72)
    print("ICLR Benchmark: Multi-Agent Paper Reviewer")
    print("  Claude SDK (free): critic + merger")
    print("  OpenRouter (paid): neutral + related work")
    print("=" * 72)
    print(f"Mode: {'parallel' if parallel else 'sequential'}")
    print(f"Sampling: {'balanced (stratified)' if balanced else 'random'}")
    print(f"Models:")
    print(f"  Critic/Spark/Merger (Claude SDK): {MODEL_HARSH}")
    print(f"  Neutral (OpenRouter):            {MODEL_NEUTRAL}")
    print(f"  Related Work (OpenRouter):       {MODEL_RELATED_WORK}")

    gt_data = load_ground_truth()
    print(f"\nLoaded {len(gt_data)} papers from ground truth.")

    available = [r for r in gt_data if (PAPERS_DIR / f"{r['paper_id']}.txt").exists()]
    print(f"Papers with parsed text: {len(available)}")

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
        f.write(f"Critic/Merger: {MODEL_HARSH} (Claude SDK, free)\n")
        f.write(f"Neutral: {MODEL_NEUTRAL}, ")
        f.write(f"Related Work: {MODEL_RELATED_WORK} (OpenRouter)\n\n")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "pred_score", "pred_decision", "gt_avg_score", "gt_decision", "gt_binary", "match",
                     "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3", "gt_score_4", "gt_score_5", "gt_score_6"])

    for i, paper_info in enumerate(samples, 1):
        pid = paper_info["paper_id"]
        paper_path = PAPERS_DIR / f"{pid}.txt"

        print(f"\n{'─' * 72}")
        print(f"[{i}/{len(samples)}] Paper: {pid}")
        print(f"  GT Decision: {paper_info['decision']}  |  GT Avg Score: {paper_info['avg_score']:.1f}")
        print(f"  GT Reviewer Scores: {paper_info['scores']}")
        print(f"{'─' * 72}")

        start = time.time()
        try:
            review_result = await review_single_paper(pid, paper_path, parallel=parallel, skip_related_work=skip_related_work, skip_spark=skip_spark)
            elapsed = time.time() - start

            pred_score = review_result["predicted_score"]
            pred_dec = review_result["predicted_decision"]

            match = pred_dec == paper_info["gt_binary"] if pred_dec else None
            marker = "MATCH" if match else ("MISMATCH" if match is not None else "PARSE_FAIL")

            print(f"\n  Predicted Score: {pred_score}/10  |  Predicted Decision: {pred_dec}")
            print(f"  GT Binary: {paper_info['gt_binary']}  |  Result: *** {marker} ***")
            print(f"  Time: {elapsed:.1f}s")

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
            results.append(r)

        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  ERROR: {e}")
            print(f"  Time: {elapsed:.1f}s")
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
            results.append(r)

        # Append this paper's result to files immediately
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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if len(args) > 0 else 10
    seed = int(args[1]) if len(args) > 1 else 42
    asyncio.run(main(n_samples=n, seed=seed, parallel=parallel, skip_related_work=skip_related, skip_spark=skip_spark, balanced=balanced))
