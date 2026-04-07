"""
Build a calibration set for the score predictor.

Usage:
  python build_calibration.py --data-dir iclr2026_data --parallel
  python build_calibration.py --data-dir iclr2026_data --no-spark --no-related-work
"""

import json
import os
import random
import re
import subprocess
import sys
import time
import threading
from collections import defaultdict
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).parent))
from run_iclr_bench import load_ground_truth, DEFAULT_BENCH_DIR

SCRIPT_DIR = Path(__file__).parent
REVIEW_SCRIPT = SCRIPT_DIR / "review_paper.sh"
BORDERLINE_BINS = {5, 6}
CONCURRENCY = 14


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
        n_take = min(10, len(bins[k]))
        for j in range(n_take):
            samples.append(bins[k][j])
            tag = " (borderline)" if k in BORDERLINE_BINS else ""
            print(f"  Bin ~{k}: picked {bins[k][j]['paper_id']} (avg={bins[k][j]['avg_score']:.1f}){tag}")
    print(f"  Total: {len(samples)} calibration papers\n")
    return samples


def run_review(paper_path: Path, parallel: bool, skip_spark: bool, skip_related_work: bool, skip_neutral: bool) -> str:
    """Run review_paper.sh on a single paper. Returns merged review text (no score)."""
    # For calibration we run the sub-agents + merger but the score doesn't matter.
    # We run the full pipeline and just extract the merged review section.
    output_path = paper_path.parent / f"{paper_path.stem}_review.md"
    cmd = ["bash", str(REVIEW_SCRIPT), str(paper_path), "--venue", "ICLR"]
    if skip_neutral:
        cmd.append("--skip-neutral")
    if skip_spark:
        cmd.append("--skip-spark")
    if skip_related_work:
        cmd.append("--skip-related-work")
    else:
        cmd.append("--with-related-work")
    if not parallel:
        cmd.append("--sequential")
    cmd += ["--output", str(output_path)]

    subprocess.run(cmd, check=True)
    return output_path.read_text(encoding="utf-8") if output_path.exists() else ""


def _extract_section(review_text: str, header: str) -> str:
    """Extract text between a header line and the next separator."""
    lines = review_text.split("\n")
    collecting = False
    result = []
    for line in lines:
        if header in line:
            collecting = True
            continue
        if collecting:
            if line.startswith("=" * 10):
                break
            result.append(line)
    return "\n".join(result).strip()


def shorten_title(title: str, max_len: int = 60) -> str:
    name = re.sub(r"[^a-z0-9 ]", "", title.lower())
    name = re.sub(r"\s+", "_", name.strip())
    if len(name) > max_len:
        name = name[:max_len].rstrip("_")
    return name or "untitled"


def save_calibration_files(results: list[dict], cal_dir: Path, papers_dir: Path) -> None:
    cal_dir.mkdir(exist_ok=True)
    for i, r in enumerate(results, 1):
        title = r.get("title", r["paper_id"])
        base = shorten_title(title)

        parts = [f"=== CALIBRATION EXAMPLE {i} ===\n"]
        parts.append("# Final Consolidated Review")
        parts.append(r["merged_review"])
        parts.append("")
        parts.append("# Actual Human Scores")
        parts.append(f"Individual reviewer scores: {r['scores']}")
        parts.append(f"Average score: {r['avg_score']:.1f}")
        parts.append(f"Binary outcome: {r['gt_binary']}\n")
        (cal_dir / f"{base}_review.md").write_text("\n".join(parts), encoding="utf-8")

        paper_src = papers_dir / f"{r['paper_id']}.txt"
        if paper_src.exists():
            (cal_dir / f"{base}_paper.md").write_text(
                paper_src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    print(f"Saved {len(results)} calibration file pairs to: {cal_dir}")


def main(data_dir=None, seed=42, parallel=False, skip_spark=False, skip_related_work=False, skip_neutral=False):
    bench_dir = Path(data_dir) if data_dir else DEFAULT_BENCH_DIR

    print("=" * 72)
    print("Building Calibration Set")
    print(f"Data: {bench_dir}")
    print("=" * 72)

    gt_data, papers_dir = load_ground_truth(bench_dir)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    print(f"Loaded {len(available)} papers with text.\n")

    samples = sample_one_per_bin(available, seed)

    results = []
    results_lock = threading.Lock()
    cal_dir = SCRIPT_DIR / "cal"
    ids_path = SCRIPT_DIR / "calibration_ids.json"

    def _save_all(snapshot):
        save_calibration_files(snapshot, cal_dir, papers_dir)
        ids_path.write_text(json.dumps([r["paper_id"] for r in snapshot], indent=2))

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

            print(f"\n{'─' * 72}")
            print(f"[{i}/{len(samples)}] {paper_info.get('title', pid)}")
            print(f"  GT: {paper_info['decision']} | Avg: {paper_info['avg_score']:.1f}")
            print(f"{'─' * 72}")

            attempt = 0
            while True:
                attempt += 1
                start = time.time()
                try:
                    review_text = run_review(paper_path, parallel, skip_spark, skip_related_work, skip_neutral)
                    elapsed = time.time() - start
                    print(f"  [{pid}] Done in {elapsed:.1f}s (attempt {attempt})")

                    result = {
                        "paper_id": pid,
                        "title": paper_info.get("title", pid),
                        "scores": paper_info["scores"],
                        "avg_score": paper_info["avg_score"],
                        "decision": paper_info["decision"],
                        "gt_binary": paper_info["gt_binary"],
                        "merged_review": review_text,
                        "error": None,
                    }

                    with results_lock:
                        results.append(result)
                        _save_all(list(results))
                        print(f"  [{pid}] Saved ({len(results)} so far)")
                    break
                except Exception as e:
                    elapsed = time.time() - start
                    wait = min(30 * attempt, 120)
                    print(f"  [{pid}] ERROR (attempt {attempt}) after {elapsed:.1f}s: {e}")
                    print(f"  [{pid}] Retrying in {wait}s ...")
                    time.sleep(wait)

    print(f"Running {len(samples)} papers (concurrency={CONCURRENCY}) ...")
    threads = []
    for w in range(CONCURRENCY):
        t = threading.Thread(target=worker, args=(w,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if not results:
        raise RuntimeError("No calibration papers completed successfully.")

    print(f"\n{'=' * 72}")
    print("Calibration set built:")
    for r in results:
        print(f"  {r['paper_id']}: avg={r['avg_score']:.1f} dec={r['gt_binary']}")


if __name__ == "__main__":
    parallel = "--parallel" in sys.argv
    skip_spark = "--no-spark" in sys.argv
    skip_related = "--no-related-work" in sys.argv
    skip_neutral = "--no-neutral" in sys.argv
    data_dir = None
    if "--data-dir" in sys.argv:
        idx = sys.argv.index("--data-dir")
        if idx + 1 < len(sys.argv): data_dir = sys.argv[idx + 1]
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a != data_dir]
    seed = int(args[0]) if args else 42
    main(data_dir=data_dir, seed=seed, parallel=parallel, skip_spark=skip_spark, skip_related_work=skip_related, skip_neutral=skip_neutral)
