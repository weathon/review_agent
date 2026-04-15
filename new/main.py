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
from tools import read_file, read_file_full, grep_files, search_file  # glob_files removed (unused)

from agents import Agent, Runner, function_tool
import dotenv
dotenv.load_dotenv()
import os
os.environ["OPENAI_DEFAULT_MODEL"] = "z-ai/glm-5.1"
HARSH_MODEL = "gpt-5.4"
SCORER_MODEL = "gpt-5.4-mini" 
MODEL_FIND_HUMAN = None
MERGER_MODEL = "z-ai/glm-5.1"
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_export_api_key

custom_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
set_default_openai_client(custom_client)
tracing_api_key = os.environ["OPENAI_API_KEY"]
set_tracing_export_api_key(tracing_api_key)

# Suppress SDK's internal error logging — we handle errors in run_agent_with_retry
# logging.getLogger("openai.agents").setLevel(logging.CRITICAL) # this should be commented out in production to handle unexpected errors
from helpers import _detect_leakage


_error_log_path = Path(__file__).parent / "error.log"
_error_logger = logging.getLogger("gpt_agent_sdk.errors")
_error_logger.setLevel(logging.ERROR)
_error_handler = logging.FileHandler(_error_log_path, mode="a")
_error_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_error_logger.addHandler(_error_handler)

HUMAN_REVIEW_DIR = os.path.abspath("../human_reviews/")
CONCURRENCY = 1

# ── Agent-level retry ────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_DELAY = 10


async def run_agent_with_retry(agent, prompt: str, max_turns: int = 30) -> str:
    agent_name = agent.name
    print(f"  [{agent_name}] starting ...")
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
                raise RuntimeError(f"[{agent_name}] {e}") from e
    raise RuntimeError(f"[{agent_name}] failed after {MAX_RETRIES} attempts")


# ── Prompt loading ───────────────────────────────────────────────────
with open("prompts/timeline.md", "r") as f:
    timeline = f.read().replace("{{CURRENT_DATE}}", time.strftime("%Y-%m-%d"))


def load_prompts(path):
    with open("prompts/" + path, "r") as f:
        return f.read() + "\n\n" + timeline

# ── Agent definitions ────────────────────────────────────────────────
summarizer = Agent(
    name="Summarizer",
    instructions="You are a subagent that summarizes files or answers questions about them. Read the file using read_file_full, then respond. You are only able to do specific files, deny other requests.",
    tools=[read_file_full],
)

_tool_agents = [read_file, summarizer.as_tool(
    tool_name="summarization", tool_description="Summarizing or answering questions about a specific file given **its absolute path** and question.",
)] 

harsh = Agent(name="Harsh Critic", instructions=load_prompts("harsh_critic.md"), model=HARSH_MODEL)
neutral_reviewer = Agent(name="Neutral Reviewer", instructions=load_prompts("neutral_reviewer.md"))
merger = Agent(name="Merger", instructions=load_prompts("merger.md"), model=MERGER_MODEL, tools=_tool_agents)
spark = Agent(name="Spark", instructions=load_prompts("spark_finder.md"))

human_finder = Agent(name="Human Finder", instructions=load_prompts("find_human_match.md"), tools=_tool_agents + [search_file, grep_files])
# scorer = Agent(name="Scorer", instructions=load_prompts("scorer_agent_gpt.txt"), tools=_tool_agents, model=SCORER_MODEL)


# ── Constants ────────────────────────────────────────────────────────
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
    
    find_human_prompt = (
        f"Paper file path: {paper_path_abs}\n"
        f"Human reviews directory: {HUMAN_REVIEW_DIR}\n\n"
        f"--- PAPER CONTENT START ---\n{paper_content}\n--- PAPER CONTENT END ---\n"
    )

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
        f"Here is the paper being reviewed (extracted from PDF — formatting "
        f"artifacts are parser issues, not paper problems):\n\n"
        f"--- PAPER CONTENT START ---\n{paper_content}--- PAPER CONTENT END ---\n\n"
        f"Here are the inputs:\n\n{chr(10).join(labeled)}\n\n"
        f"Now produce the final consolidated review following your instructions. "
        f"Remember: many of the harsh critic's points may be nonsensical or overly "
        f"picky — cross-check everything against the actual paper before including it."
    )

    print("  Phase 2: Merger ...")
    merged_review = await run_agent_with_retry(merger, merger_prompt) 
    scorer_output = float(merged_review.split("<pineapple>")[1].split("</pineapple>")[0]) if "<pineapple>" in merged_review else -1
    decision = (merged_review.split("<orange>")[1].split("</orange>")[0]) if "<orange>" in merged_review else "N/A"

    log_path = Path(__file__).parent / "pipeline.log"
    with open(log_path, "a") as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Paper: {paper_path}\n")
        log_f.write(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n")
        log_f.write(f"\n--- Merged Review ---\n{merged_review}\n")
        log_f.write(f"\n--- Scorer Output ---\n{scorer_output}\n")
        log_f.write(f"\n--- Decision ---\n{decision}\n")

    return {"merged_review": merged_review, "scorer_output": scorer_output, "decision": decision}


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


def decision_match(predicted: str | None, gt_binary: str) -> bool | None:
    if predicted in (None, "", "N/A"):
        return None
    return predicted == gt_binary


def match_label(match: bool | None) -> str:
    if match is None:
        return "N/A"
    return "YES" if match else "NO"


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


# ── Benchmark ────────────────────────────────────────────────────────

async def run_benchmark(data_dir: str, n_samples: int = 10, seed: int = 42, balanced: bool = False):
    data_path = Path(data_dir)
    cal_ids = [i.split(".")[0] for i in os.listdir(HUMAN_REVIEW_DIR) if i.endswith(".md")]

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

    # Check for existing results and ask user whether to continue or overwrite
    finished = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        import pandas as pd
        existing_df = pd.read_csv(csv_path)
        existing_count = len(existing_df)
        print(f"\nFound existing bench_scores.csv with {existing_count} results.")
        choice = input("  [C]ontinue (skip finished papers) or [O]verwrite? [C/o]: ").strip().lower()
        if choice in ("o", "overwrite"):
            print("  Overwriting existing results.\n")
        else:
            finished = set(existing_df["paper_id"].astype(str))
            print(f"  Continuing — will skip {len(finished)} already-finished papers.\n")
    samples = [s for s in samples if s["paper_id"] not in finished]

    if not finished:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["paper_id", "pred_score", "pred_decision", "gt_avg_score", "gt_decision", "gt_binary", "match", "cost", "sdk_savings",
                                    "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3", "gt_score_4", "gt_score_5", "gt_score_6"])

    results = []

    def on_complete(paper_info, result):
        pred_score = result["scorer_output"]
        pred_decision = result["decision"]
        match = decision_match(pred_decision, paper_info["gt_binary"])
        match_str = match_label(match)
        print(f"  [{paper_info['paper_id']}] predicted={pred_score} gt={paper_info['avg_score']:.1f} match={match_str}")
        results.append({"pred_score": pred_score, "gt_avg_score": paper_info["avg_score"], "pred_decision": pred_decision, "gt_binary": paper_info["gt_binary"], "match": match})
        gt_scores_padded = paper_info["scores"] + [""] * (7 - len(paper_info["scores"]))
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                paper_info["paper_id"],
                pred_score,
                pred_decision,
                f"{paper_info['avg_score']:.2f}",
                paper_info["decision"],
                paper_info["gt_binary"],
                match_str,
                "0.0000",
                "0.0000",
                *gt_scores_padded,
            ])
        (reviews_dir / f"{paper_info['paper_id']}.md").write_text(result["merged_review"], encoding="utf-8")

    if not samples:
        print("Nothing to run."); return
    print(f"Running {len(samples)} benchmark papers (concurrency={CONCURRENCY}) ...")
    await process_papers(samples, papers_dir, skip_scoring=False, callback=on_complete)

    scored = [r for r in results if r["pred_score"] != -1]
    if scored:
        mae = sum(abs(r["pred_score"] - r["gt_avg_score"]) for r in scored) / len(scored)
        print(f"\nResults: {len(scored)} scored, MAE={mae:.2f}")


# ── Single paper ─────────────────────────────────────────────────────

async def run_single_paper(paper_path: str):
    print(f"Reviewing: {paper_path}")
    result = await run_pipeline(paper_path)
    print(f"\n{'=' * 72}\nFINAL REVIEW\n{'=' * 72}\n{result['merged_review']}")
    score = result["scorer_output"]
    if score != -1:
        print(f"\nPredicted score: {score}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent paper reviewer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single_paper", type=str)
    group.add_argument("--benchmark", type=str, metavar="DATA_DIR")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()

    if args.single_paper:
        asyncio.run(run_single_paper(args.single_paper))
    elif args.benchmark:
        asyncio.run(run_benchmark(args.benchmark, n_samples=args.n_samples, seed=args.seed, balanced=args.balanced))
