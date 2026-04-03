"""
Build calibration set for the simple review-then-score baseline.

Uses BASELINE_MODEL to generate a review for each calibration paper, then
pairs it with the real human score. Output is used as few-shot context
when scoring new papers in run_baseline.py.

Usage:
  python baselines/review_baseline/build_calibration.py --data-dir iclr2025_data
  python baselines/review_baseline/build_calibration.py --data-dir iclr2025_data --seed 42
"""

import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path so we can import from paper_reviewer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from paper_reviewer import (
    _get_client,
    _call_openai,
    sanitize_text,
)
from run_iclr_bench import load_ground_truth, DEFAULT_BENCH_DIR

# Baseline uses its own model — not the pipeline's MODEL_SCORER
BASELINE_MODEL = "z-ai/glm-5"

BORDERLINE_BINS = {5, 6}
BORDERLINE_EXTRA = 1
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


def sample_one_per_bin(papers: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    bins = defaultdict(list)
    for p in papers:
        bins[round(p["avg_score"])].append(p)
    for k in bins:
        rng.shuffle(bins[k])

    samples = []
    for k in sorted(bins.keys()):
        if not bins[k]:
            continue
        n_take = 2 + BORDERLINE_EXTRA if k in BORDERLINE_BINS else 1
        n_take = min(n_take, len(bins[k]))
        for j in range(n_take):
            samples.append(bins[k][j])
            tag = " (borderline)" if k in BORDERLINE_BINS else ""
            print(f"  Bin ~{k}: picked {bins[k][j]['paper_id']} (avg={bins[k][j]['avg_score']:.1f}, {bins[k][j]['gt_binary']}){tag}")
    print(f"  Total: {len(samples)} calibration papers\n")
    return samples


async def generate_review(client, paper_content: str) -> tuple[str, float]:
    """Generate a review for the paper using BASELINE_MODEL."""
    user_prompt = (
        f"Here is the paper to review:\n\n"
        f"--- PAPER START ---\n{paper_content}\n--- PAPER END ---\n\n"
        f"Write your detailed review now."
    )

    return await _call_openai(
        client, "calibration_review", REVIEW_PROMPT, user_prompt, BASELINE_MODEL,
    )


def build_calibration_md(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"=== CALIBRATION EXAMPLE {i} ===\n")
        parts.append("# Review")
        parts.append(r["review"])
        parts.append("")
        parts.append("# Actual Human Scores")
        parts.append(f"Individual reviewer scores: {r['scores']}")
        parts.append(f"Average score: {r['avg_score']:.1f}")
        parts.append(f"Binary outcome: {r['gt_binary']}\n")
    return "\n".join(parts)


async def main(data_dir: str | None = None, seed: int = 42):
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print("Building Calibration Set (review-then-score baseline)")
    print(f"  Scorer/Reviewer: {BASELINE_MODEL}")
    print(f"  Data: {bench_dir}")
    print("=" * 72)

    gt_data, papers_dir = load_ground_truth(bench_dir)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    print(f"Loaded {len(available)} papers with text.\n")

    print("Sampling calibration papers...")
    samples = sample_one_per_bin(available, seed)

    client = _get_client()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def process_one(i, paper_info):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"
        paper_content = paper_path.read_text(encoding="utf-8", errors="replace")
        paper_content = sanitize_text(paper_content)

        async with semaphore:
            print(f"  [{i}/{len(samples)}] {pid} (GT avg={paper_info['avg_score']:.1f}, {paper_info['gt_binary']})")
            attempt = 0
            while True:
                attempt += 1
                start = time.time()
                try:
                    review, cost = await generate_review(client, paper_content)
                    elapsed = time.time() - start
                    print(f"    [{pid}] done ({len(review)} chars, {elapsed:.1f}s, ${cost:.4f})")
                    return {
                        "paper_id": pid,
                        "review": review,
                        "scores": paper_info["scores"],
                        "avg_score": paper_info["avg_score"],
                        "decision": paper_info["decision"],
                        "gt_binary": paper_info["gt_binary"],
                        "cost": cost,
                    }
                except Exception as e:
                    elapsed = time.time() - start
                    wait = min(30 * attempt, 120)
                    print(f"    [{pid}] ERROR (attempt {attempt}) {elapsed:.1f}s: {e}")
                    print(f"    [{pid}] Retrying in {wait}s ...")
                    await asyncio.sleep(wait)

    all_results = await asyncio.gather(*(
        process_one(i, p) for i, p in enumerate(samples, 1)
    ))

    results = [r for r in all_results if r is not None]

    # Save calibration
    out_dir = Path(__file__).resolve().parent
    cal_md = build_calibration_md(results)
    cal_path = out_dir / "calibration.md"
    cal_path.write_text(cal_md, encoding="utf-8")
    print(f"\nCalibration saved to: {cal_path} ({len(cal_md):,} chars)")

    ids = [r["paper_id"] for r in results]
    ids_path = out_dir / "calibration_ids.json"
    ids_path.write_text(json.dumps(ids, indent=2))
    print(f"Calibration IDs saved to: {ids_path} ({len(ids)} papers)")

    total_cost = sum(r["cost"] for r in results)
    print(f"Total cost: ${total_cost:.4f}")

    print(f"\n{'=' * 72}")
    print("Calibration set built:")
    for r in results:
        print(f"  {r['paper_id']}: avg={r['avg_score']:.1f} scores={r['scores']} dec={r['gt_binary']}")
    print(f"\nTo use:")
    print(f"  python baselines/review_baseline/run_baseline.py 50 3112 --data-dir {data_dir or 'iclr2025_data'} --calibration baselines/review_baseline/calibration.md")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    data_dir = None
    if "--data-dir" in sys.argv:
        idx = sys.argv.index("--data-dir")
        if idx + 1 < len(sys.argv):
            data_dir = sys.argv[idx + 1]
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != data_dir]
    seed = int(args[0]) if args else 42
    asyncio.run(main(data_dir=data_dir, seed=seed))
