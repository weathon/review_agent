import argparse
import asyncio
import csv
import json
import random
import re
import logging
import time
from collections import defaultdict
from pathlib import Path

from agents import Agent, Runner, function_tool
import dotenv
dotenv.load_dotenv()
import os
os.environ["OPENAI_DEFAULT_MODEL"] = "deepseek/deepseek-v3.2"
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled

custom_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
set_default_openai_client(custom_client)
set_tracing_disabled(True)

# ── Leakage detection ────────────────────────────────────────────────
LEAKAGE_WARNING_PATTERNS = [
    r"\bsame paper\b",
    r"\bexact same paper\b",
    r"\bthis exact paper\b",
    r"\bcontains this exact paper\b",
    r"\bthe exact same paper\b",
    r"\bcalibration copy\b",
]

_error_log_path = Path(__file__).parent / "error.log"
_error_logger = logging.getLogger("gpt_agent_sdk.errors")
_error_logger.setLevel(logging.ERROR)
_error_handler = logging.FileHandler(_error_log_path, mode="a")
_error_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_error_logger.addHandler(_error_handler)


def _detect_leakage(text: str) -> list[str]:
    matches = []
    for pattern in LEAKAGE_WARNING_PATTERNS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            matches.append(found.group(0))
    return matches


# ── Agent-level retry ────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_DELAY = 10


async def run_agent_with_retry(agent, prompt: str, max_turns: int = 30) -> str:
    agent_name = agent.name
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await Runner.run(agent, prompt, max_turns=max_turns)
            output = result.final_output
            if not output or not output.strip():
                if attempt < MAX_RETRIES:
                    print(f"  [{agent_name}] empty response (attempt {attempt}/{MAX_RETRIES}), retrying ...")
                    await asyncio.sleep(RETRY_DELAY + attempt * 5)
                    continue
                raise RuntimeError(f"[{agent_name}] empty response after {MAX_RETRIES} attempts")
            print(f"  [{agent_name}] done")
            return output
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{agent_name}] error (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ... {e}")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(f"[{agent_name}] failed after {MAX_RETRIES} attempts")


# ── Prompt loading ───────────────────────────────────────────────────
with open("prompts/timeline.txt", "r") as f:
    timeline = f.read().replace("{{CURRENT_DATE}}", time.strftime("%Y-%m-%d"))


def load_prompts(path):
    with open("prompts/" + path, "r") as f:
        return f.read() + "\n\n" + timeline


# ── Tools ────────────────────────────────────────────────────────────
@function_tool
def read_file(abs_path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read lines from a file. Returns lines numbered start_line to end_line (inclusive, 1-based).
    If end_line is 0, reads to end of file."""
    if ("/papers/" in abs_path or abs_path.endswith("_paper.md")) and end_line == 0:
        return "ERROR: Full paper reads blocked. Use grep_files first, then read_file with start_line/end_line."
    with open(abs_path, "r") as f:
        lines = f.readlines()
    selected = lines[max(0, start_line - 1):end_line if end_line > 0 else len(lines)]
    return "".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))

@function_tool
def read_file_full(abs_path: str) -> str:
    """Read an entire file. Only use this inside the Summarizer agent."""
    print(abs_path)
    with open(abs_path, "r") as f:
        return f.read()

@function_tool
def glob_files(pattern: str, directory: str = ".") -> str:
    """Find files matching a glob pattern (e.g. '**/*.md', '*.txt') under a directory. Returns one path per line."""
    import glob as _glob
    matches = sorted(_glob.glob(pattern, root_dir=directory, recursive=True))
    return "\n".join(os.path.join(directory, m) for m in matches) if matches else "No files matched."

@function_tool
def grep_files(pattern: str, directory: str = ".", file_glob: str = "*") -> str:
    """Search file contents for a regex pattern. Returns matching lines with file paths and line numbers."""
    import glob as _glob
    import re
    matches = []
    files = sorted(_glob.glob(file_glob, root_dir=directory, recursive=True))
    for f in files[:500]:
        fpath = os.path.join(directory, f)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if re.search(pattern, line):
                        matches.append(f"{fpath}:{i}: {line.rstrip()}")
        except Exception:
            continue
        if len(matches) >= 200:
            break
    return "\n".join(matches) if matches else "No matches found."


# ── Agent definitions ────────────────────────────────────────────────
summarizer = Agent(
    name="Summarizer",
    instructions="You are a subagent that summarizes files or answers questions about them. Read the file using read_file_full, then respond.",
    tools=[read_file_full],
)

_tool_agents = [read_file, glob_files, grep_files, summarizer.as_tool(
    tool_name="summarization", tool_description="Summarizing or answering questions about a file given **its absolute path** and question.",
)]
harsh = Agent(name="Harsh Critic", instructions=load_prompts("harsh_critic.txt"))
neutral_reviewer = Agent(name="Neutral Reviewer", instructions=load_prompts("neutral_reviewer.txt"))
merger = Agent(name="Merger", instructions=load_prompts("merger.txt"), tools=[read_file_full] + _tool_agents)
spark = Agent(name="Spark", instructions=load_prompts("spark_finder.txt"))

human_finder = Agent(name="Human Finder", instructions=load_prompts("find_human_match.txt"), tools=_tool_agents)
scorer = Agent(name="Scorer", instructions=load_prompts("scorer_agent_gpt.txt"), tools=_tool_agents)


# ── Constants ────────────────────────────────────────────────────────
HUMAN_REVIEW_DIR = os.path.abspath("iclr2025_data")
CONCURRENCY = 1

REVIEW_PROMPT = """Review the following paper thoroughly.

NOTE: This paper was extracted from PDF by an automated parser. There may be formatting artifacts such as broken equations, garbled tables, misplaced figure references, or OCR errors. These are parser issues, NOT problems with the paper itself. Do NOT treat formatting artifacts as weaknesses.

{paper_path}
--- PAPER CONTENT START ---
{paper_content}
--- PAPER CONTENT END (EVERYTHING AFTER REFERENCE IS REMOVED) ---"""


# ── Core pipeline ────────────────────────────────────────────────────

async def run_pipeline(paper_path: str, skip_scoring: bool = False) -> dict:
    paper_path_abs = os.path.abspath(paper_path)
    with open(paper_path, "r") as f:
        paper_content = f.read()
    paper_content = paper_content.split("REFERENCES")[0]

    review_prompt = REVIEW_PROMPT.format(paper_path=paper_path_abs, paper_content=paper_content)
    
    find_human_prompt = f"Paper file path: {paper_path_abs}\nHuman reviews directory: {HUMAN_REVIEW_DIR}\n"

    agents_and_prompts = [
        (harsh, review_prompt), (neutral_reviewer, review_prompt),
        (spark, review_prompt), (human_finder, find_human_prompt),
    ]

    print(f"  Phase 1: Running {len(agents_and_prompts)} agents in parallel ...")
    responses = await asyncio.gather(
        *(run_agent_with_retry(a, p) for a, p in agents_and_prompts)
    )

    labeled = [f"### {a.name}\n{out}" for (a, _), out in zip(agents_and_prompts, responses)]
    merger_prompt = (
        f"The paper file is at: {paper_path_abs}\n"
        f"Use grep_files and read_file to verify claims from the reviews against the actual paper.\n\n"
        f"Here are the inputs:\n\n{chr(10).join(labeled)}\n\n"
        f"Now produce the final consolidated review following your instructions. "
        f"Remember: many of the harsh critic's points may be nonsensical or overly "
        f"picky — cross-check everything against the actual paper before including it."
    )

    print("  Phase 2: Merger ...")
    merged_review = await run_agent_with_retry(merger, merger_prompt)

    scorer_output = None
    if not skip_scoring:
        print("  Phase 3: Scorer ...")
        scorer_output = await run_agent_with_retry(
            scorer, f"Review: \n\n{merged_review}\n\ncal_dir_abs:{os.path.abspath('cal/')}")
        leakage = _detect_leakage(scorer_output)
        if leakage:
            print(f"  [scorer] WARNING: Potential leakage: {', '.join(leakage)}")
            _error_logger.error(f"Potential calibration leakage: {', '.join(leakage)}")

    return {"merged_review": merged_review, "scorer_output": scorer_output}


# ── Helpers ──────────────────────────────────────────────────────────

def load_ground_truth(data_dir: Path) -> tuple[list[dict], Path]:
    csv_file = data_dir / "ratings.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"No ratings.csv found in {data_dir}")
    papers_dir = data_dir / "papers"
    rows = []
    with open(csv_file, "r") as f:
        for row in csv.DictReader(f):
            scores = [float(row[f"score_{i}"]) for i in range(7) if row.get(f"score_{i}", "").strip()]
            decision = row.get("decision", "").strip()
            gt_binary = row.get("gt_binary", "").strip() or ("Accept" if "Accept" in decision else "Reject")
            rows.append({
                "paper_id": row["paper_id"].strip(),
                "title": row.get("title", "").strip(),
                "scores": scores,
                "avg_score": float(row.get("avg_score", 0)),
                "decision": decision,
                "gt_binary": gt_binary,
            })
    return rows, papers_dir


def shorten_title(title: str, max_len: int = 60) -> str:
    name = re.sub(r"[^a-z0-9 ]", "", title.lower())
    name = re.sub(r"\s+", "_", name.strip())
    return (name[:max_len].rstrip("_") if len(name) > max_len else name) or "untitled"


def stratified_sample(papers: list[dict], n_per_bin: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    bins = defaultdict(list)
    for p in papers:
        bins[round(p["avg_score"])].append(p)
    for k in bins:
        rng.shuffle(bins[k])
    samples = []
    for k in sorted(bins.keys()):
        samples.extend(bins[k][:n_per_bin])
    rng.shuffle(samples)
    print(f"  Stratified sample: {len(samples)} papers from {len(bins)} bins ({n_per_bin}/bin)")
    return samples


def parse_score(text: str) -> float | None:
    match = re.search(r"<pineapple>([\d.]+)</pineapple>", text)
    return float(match.group(1)) if match else None


async def process_papers(papers: list[dict], papers_dir: Path, skip_scoring: bool, callback):
    """Run pipeline on a list of papers with CONCURRENCY concurrent tasks."""
    sem = asyncio.Semaphore(CONCURRENCY)

    async def process_one(i, paper_info):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"
        print(f"\n[{i}/{len(papers)}] {paper_info.get('title', pid)} (avg={paper_info['avg_score']:.1f})")
        async with sem:
            try:
                result = await run_pipeline(str(paper_path), skip_scoring=skip_scoring)
            except Exception as e:
                raise RuntimeError(f"[{pid}] pipeline failed: {e}") from e
            callback(paper_info, result)

    await asyncio.gather(*(process_one(i, p) for i, p in enumerate(papers, 1)))


# ── Calibration ──────────────────────────────────────────────────────

async def run_calibration(data_dir: str, seed: int = 42):
    data_path = Path(data_dir)
    cal_dir = Path(__file__).parent / "cal"
    ids_path = Path(__file__).parent / "calibration_ids.json"
    cal_dir.mkdir(exist_ok=True)

    gt_data, papers_dir = load_ground_truth(data_path)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists()]
    print(f"Loaded {len(available)} papers with text.")

    samples = stratified_sample(available, n_per_bin=10, seed=seed)

    # Skip already done
    done = {s["paper_id"] for s in samples if (cal_dir / f"{shorten_title(s.get('title', s['paper_id']))}_review.md").exists()}
    samples = [s for s in samples if s["paper_id"] not in done]
    if done:
        print(f"Skipping {len(done)} already-completed papers.")
    if not samples:
        print("All done."); return

    existing_ids = set(json.loads(ids_path.read_text())) if ids_path.exists() else set()
    count = [0]

    def on_complete(paper_info, result):
        count[0] += 1
        base = shorten_title(paper_info.get("title", paper_info["paper_id"]))
        # Save review
        review_text = (
            f"=== CALIBRATION EXAMPLE {count[0]} ===\n\n"
            f"# Final Consolidated Review\n{result['merged_review']}\n\n"
            f"# Actual Human Scores\n"
            f"Individual reviewer scores: {paper_info['scores']}\n"
            f"Average score: {paper_info['avg_score']:.1f}\n"
            f"Binary outcome: {paper_info['gt_binary']}\n"
        )
        (cal_dir / f"{base}_review.md").write_text(review_text, encoding="utf-8")
        # Copy paper
        src = papers_dir / f"{paper_info['paper_id']}.txt"
        if src.exists():
            (cal_dir / f"{base}_paper.md").write_text(src.read_text(errors="replace"), encoding="utf-8")
        # Update IDs
        existing_ids.add(paper_info["paper_id"])
        ids_path.write_text(json.dumps(sorted(existing_ids), indent=2))
        print(f"  [{paper_info['paper_id']}] Saved ({count[0]} so far)")

    print(f"Running {len(samples)} calibration papers (concurrency={CONCURRENCY}) ...")
    await process_papers(samples, papers_dir, skip_scoring=True, callback=on_complete)
    print(f"\nCalibration complete: {count[0]} papers saved to {cal_dir}")


# ── Benchmark ────────────────────────────────────────────────────────

async def run_benchmark(data_dir: str, n_samples: int = 10, seed: int = 42, balanced: bool = False):
    data_path = Path(data_dir)
    ids_path = Path(__file__).parent / "calibration_ids.json"
    cal_ids = set(json.loads(ids_path.read_text())) if ids_path.exists() else set()

    gt_data, papers_dir = load_ground_truth(data_path)
    available = [r for r in gt_data if (papers_dir / f"{r['paper_id']}.txt").exists() and r["paper_id"] not in cal_ids]
    print(f"Available papers: {len(available)} (excluded {len(cal_ids)} calibration)")

    if balanced:
        samples = stratified_sample(available, n_per_bin=max(1, n_samples // 10), seed=seed)
    else:
        samples = random.Random(seed).sample(available, min(n_samples, len(available)))
        print(f"Random sample: {len(samples)} papers")

    out_dir = Path(__file__).parent
    csv_path = out_dir / "bench_scores.csv"
    reviews_dir = out_dir / "bench_reviews"
    reviews_dir.mkdir(exist_ok=True)

    # Check existing
    finished = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path) as f:
            finished = {row["paper_id"] for row in csv.DictReader(f)}
        print(f"Skipping {len(finished)} already-finished papers.")
    samples = [s for s in samples if s["paper_id"] not in finished]

    if not finished:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["paper_id", "pred_score", "gt_avg_score", "gt_binary", "gt_decision"])

    results = []

    def on_complete(paper_info, result):
        pred_score = parse_score(result["scorer_output"] or "")
        print(f"  [{paper_info['paper_id']}] predicted={pred_score} gt={paper_info['avg_score']:.1f}")
        results.append({"pred_score": pred_score, "gt_avg_score": paper_info["avg_score"]})
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([paper_info["paper_id"], pred_score, paper_info["avg_score"],
                                    paper_info["gt_binary"], paper_info["decision"]])
        (reviews_dir / f"{paper_info['paper_id']}.md").write_text(result["merged_review"], encoding="utf-8")

    if not samples:
        print("Nothing to run."); return
    print(f"Running {len(samples)} benchmark papers (concurrency={CONCURRENCY}) ...")
    await process_papers(samples, papers_dir, skip_scoring=False, callback=on_complete)

    scored = [r for r in results if r["pred_score"] is not None]
    if scored:
        mae = sum(abs(r["pred_score"] - r["gt_avg_score"]) for r in scored) / len(scored)
        print(f"\nResults: {len(scored)} scored, MAE={mae:.2f}")


# ── Single paper ─────────────────────────────────────────────────────

async def run_single_paper(paper_path: str):
    print(f"Reviewing: {paper_path}")
    result = await run_pipeline(paper_path)
    print(f"\n{'=' * 72}\nFINAL REVIEW\n{'=' * 72}\n{result['merged_review']}")
    score = parse_score(result["scorer_output"] or "")
    if score is not None:
        print(f"\nPredicted score: {score}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent paper reviewer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single_paper", type=str)
    group.add_argument("--benchmark", type=str, metavar="DATA_DIR")
    group.add_argument("--calibration", type=str, metavar="DATA_DIR")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()

    if args.single_paper:
        asyncio.run(run_single_paper(args.single_paper))
    elif args.calibration:
        asyncio.run(run_calibration(args.calibration, seed=args.seed))
    elif args.benchmark:
        asyncio.run(run_benchmark(args.benchmark, n_samples=args.n_samples, seed=args.seed, balanced=args.balanced))
