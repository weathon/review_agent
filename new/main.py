import argparse
import asyncio
import csv
import json
import random
import re
import logging
import sys
import time
import os
from collections import defaultdict
from pathlib import Path

from tools import read_file, read_file_full, grep_file, search_file, allow_path, HUMAN_REVIEW_DIR  # glob_files removed (unused)
import weave
weave.init("openai-agents")

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
import dotenv
dotenv.load_dotenv()
os.environ["OPENAI_DEFAULT_MODEL"] = os.getenv("OPENAI_DEFAULT_MODEL", "z-ai/glm-5.1")
HARSH_MODEL = os.environ.get("HARSH_MODEL", "gpt-5.4")
# HUMAN_FINDER = "kimi-k2.5"
HUMAN_FINDER = os.environ.get("HUMAN_FINDER", "ollama:glm-5.1:cloud")
MERGER_MODEL = os.environ.get("MERGER_MODEL", "ollama:glm-5.1:cloud")
# MERGER_MODEL = "claude_sdk:claude-sonnet-4-6" # use dash instead of dot in claude sdk
# MERGER_MODEL = "claude-sonnet-4.6"
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

CONCURRENCY = 5

# ── Agent-level retry ────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_DELAY = 10


async def run_agent_with_retry(agent, prompt: str, max_turns: int = 30) -> tuple[str, object]:
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
            return output, result.context_wrapper.usage
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


PAPER_ACCESS_INJECTION = "The full paper text is included in the user message. Use it to verify reviewer claims directly."
PAPER_ACCESS_FILE = "The paper path is provided in the user message. Use read_file to read the paper and verify reviewer claims directly."

CAL_INSTRUCTION_WITH = """Use comparative scoring to calibrate your final score. You have access to human reviews of other papers through the review finder and search/grep tools. Search tool supports both bm25 and vector search.

Your calibration process:

1. **Topic-based anchors**: Use the review finder to retrieve papers with similar topics. Note their human scores.

2. **Quality-based anchors**: This is critical. Do NOT only search by topic. Search for papers that share similar strength/weakness patterns with the paper under review:
   - If this paper has strong empirical results but overclaims, search for reviews mentioning "overclaim" "strong experiments" and note how humans scored those.
   - If this paper has a novel framing but weak baselines, search for reviews mentioning "novel framing" "missing baselines" and note those scores.

3. **Deliberate range anchoring**: Actively seek out both HIGH-scoring and LOW-scoring papers to anchor the extremes of your scale, **even if there is a topic mismatch**:
   - Search for reviews of papers that were scored ~7+ by humans. Read what made them strong.
   - Search for reviews of papers that were scored ~3 or below by humans. Read what made them weak.
   - Compare the paper under review against BOTH ends, not just the middle.

   Examples: if reviewing a paper about privacy attacks on face recognition, search for:
   - "privacy attack face recognition strong paper" → find high-scored papers in the same area

   If no papers are found with the same topic, you can use more general queries.

4. **Score relative to anchors**: Your final score should be positioned relative to the retrieved examples. If retrieved papers with similar strengths got 7s from humans, and papers with similar weaknesses got 3s, use that range. Do not compress everything into 4-6.

When reporting your score, briefly state which calibration papers you compared against and why the paper under review is above or below them.

You can use read_file to read these files. List the papers you compared and the reasoning.

Let the score distribution follow the actual quality of the paper relative to the calibration examples.
The samples could be concentrated in the middle, that does not mean you have to score it in the middle as well.
List all papers you compared against and their human scores, and explain how you positioned the current paper relative to them.
You HAVE TO include a few extream high and low end samples. 

There are less papers with extreme scores, so if the paper is truly exceptional or truly weak, it is okay to give it an extreme score even if most found papers are in the middle. You HAVE TO also try to find papers with extreme scores to see what made a paper really good/bad, it doesn't need to be the same topic for these extream score queries. 

Do NOT be afraid to give extreme scores if justified.
"""

CAL_INSTRUCTION_WITHOUT = """Assign a score based solely on your assessment of the paper's quality. Do NOT use the search or review finder tools for calibration — score directly from the paper's merits and weaknesses as identified in the review above."""


def load_prompts(path, paper_access: str = PAPER_ACCESS_INJECTION, no_cal: bool = False):
    with open("prompts/" + path, "r") as f:
        content = f.read()
    content = content.replace("{{PAPER_ACCESS_INSTRUCTION}}", paper_access)
    cal_instruction = CAL_INSTRUCTION_WITHOUT if no_cal else CAL_INSTRUCTION_WITH
    content = content.replace("{{CALIBRATION_INSTRUCTION}}", cal_instruction)
    return content + "\n\n" + timeline

# ── Agent definitions ────────────────────────────────────────────────
# summarizer = Agent(
#     name="Summarizer",
#     instructions="You are a subagent that summarizes files or answers questions about them. Read the file using read_file_full, then respond. You are only able to do specific files, deny other requests. If there is no file path given, return the error message.",
#     tools=[read_file_full],
# )

_tool_agents = [read_file, search_file, grep_file] 
# summarizer.as_tool(
    # tool_name="summarization", tool_description="Summarizing or answering questions about a specific file given **its absolute path** and question.",

    
harsh = Agent(name="Harsh Critic", instructions=load_prompts("harsh_critic.md"), model=HARSH_MODEL)
neutral_reviewer = Agent(name="Neutral Reviewer", instructions=load_prompts("neutral_reviewer.md"))
if not HUMAN_FINDER.startswith("ollama:"):
    human_finder = Agent(name="Human Finder", instructions=load_prompts("find_human_match.md"), tools=_tool_agents, model=HUMAN_FINDER)
else:
    model = HUMAN_FINDER.replace("ollama:", "")
    client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1/")
    model = OpenAIChatCompletionsModel(model=model, openai_client=client)
    human_finder = Agent(name="Human Finder", instructions=load_prompts("find_human_match.md"), tools=_tool_agents, model=model)

_NO_CAL = "--no_cal" in __import__("sys").argv
_merger_instructions = load_prompts("merger.md", paper_access=PAPER_ACCESS_FILE, no_cal=_NO_CAL)
_merger_tools = [read_file, grep_file] if _NO_CAL else _tool_agents

if MERGER_MODEL.startswith("claude_sdk:"):
    merger = None  # Claude SDK merger — created per-call in run_pipeline
    _MERGER_SDK_MODEL = MERGER_MODEL[len("claude_sdk:"):]
elif MERGER_MODEL.startswith("ollama:"):
    merger_model = MERGER_MODEL.replace("ollama:", "")
    client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1/")
    merger_model = OpenAIChatCompletionsModel(model=merger_model, openai_client=client)
    merger = Agent(name="Merger", instructions=_merger_instructions, model=merger_model, tools=_merger_tools)
    _MERGER_SDK_MODEL = None
else:
    merger = Agent(name="Merger", instructions=_merger_instructions, model=MERGER_MODEL, tools=_merger_tools)
    _MERGER_SDK_MODEL = None
     
spark = Agent(name="Spark", instructions=load_prompts("spark_finder.md"))

# scorer = Agent(name="Scorer", instructions=load_prompts("scorer_agent_gpt.txt"), tools=_tool_agents, model=SCORER_MODEL)


# ── Constants ────────────────────────────────────────────────────────
REVIEW_PROMPT = """Review the following paper thoroughly.

NOTE: This paper was extracted from PDF by an automated parser. There may be formatting artifacts such as broken equations, garbled tables, misplaced figure references, or OCR errors. These are parser issues, NOT problems with the paper itself. Do NOT treat formatting artifacts as weaknesses.

The full paper text is included below. Do NOT attempt to read the paper from disk — use the inline content.

--- PAPER CONTENT START ---
{paper_content}
--- PAPER CONTENT END (EVERYTHING AFTER REFERENCE IS REMOVED) ---"""


# ── Core pipeline ────────────────────────────────────────────────────

async def run_pipeline(paper_path: str, skip_scoring: bool = False, no_cal: bool = False) -> dict:
    paper_path_abs = os.path.abspath(paper_path)
    allow_path(paper_path_abs)
    with open(paper_path, "r") as f:
        paper_content = f.read()
    paper_content = paper_content

    review_prompt = REVIEW_PROMPT.format(paper_content=paper_content)
    
    find_human_prompt = (
        f"Human reviews directory: {HUMAN_REVIEW_DIR}\n\n"
        f"--- PAPER CONTENT START ---\n{paper_content}\n--- PAPER CONTENT END ---\n"
    )

    agents_and_prompts = [
        (harsh, review_prompt), (neutral_reviewer, review_prompt),
        (spark, review_prompt), (human_finder, find_human_prompt),
    ]

    print(f"  Phase 1: Running {len(agents_and_prompts)} agents in parallel ...")
    responses = await asyncio.gather(
        *(run_agent_with_retry(a, p) for a, p in agents_and_prompts),
        return_exceptions=True,
    )
    for (a, _), r in zip(agents_and_prompts, responses):
        if isinstance(r, Exception):
            print(f"  🔥ERROR: {paper_path} — agent '{a.name}' raised {type(r).__name__}: {r}")
            return None

    agent_usages = {}
    outputs = []
    for (a, _), (out, usage) in zip(agents_and_prompts, responses):
        outputs.append(out)
        agent_usages[a.name] = usage
    labeled = [f"### {a.name}\n{out}" for (a, _), out in zip(agents_and_prompts, outputs)]


    print("  Phase 2: Merger ...")
    if _MERGER_SDK_MODEL is not None:
        from claude_merger import run_merger_claude_sdk
        merger_prompt = (
            f"Here is the paper being reviewed (extracted from PDF — formatting "
            f"artifacts are parser issues, not paper problems):\n\n"
            f"Paper path: {paper_path_abs}, read it in chunks.\n\n"
            f"Here are the inputs:\n\n{chr(10).join(labeled)}\n\n"
            f"Now produce the final consolidated review following your instructions. "
            f"Remember: many of the harsh critic's points may be nonsensical or overly "
            f"picky — cross-check everything against the actual paper before including it."
        )
        paper_dir = str(Path(paper_path_abs).parent)
        merged_review = await run_merger_claude_sdk(_MERGER_SDK_MODEL, merger_prompt, paper_dir, no_cal=no_cal)
        agent_usages["Merger"] = None
    else:
        merger_prompt = (
            f"Paper being reviewed (extracted from PDF — formatting artifacts are parser "
            f"issues, not paper problems).\n\n"
            f"Paper path: {paper_path_abs}\n"
            f"Use read_file(abs_path, start_line, end_line) and grep_file(pattern, abs_path) "
            f"to inspect the paper in chunks. Full-file reads of the paper are blocked — "
            f"pass explicit line ranges.\n\n"
            f"Here are the inputs:\n\n{chr(10).join(labeled)}\n\n"
            f"Now produce the final consolidated review following your instructions. "
            f"Remember: many of the harsh critic's points may be nonsensical or overly "
            f"picky — cross-check everything against the actual paper before including it."
        )
        merged_review, merger_usage = await run_agent_with_retry(merger, merger_prompt)
        agent_usages["Merger"] = merger_usage
    scorer_output = float(merged_review.split("<pineapple>")[1].split("</pineapple>")[0]) if "<pineapple>" in merged_review else -1
    decision = (merged_review.split("<orange>")[1].split("</orange>")[0]) if "<orange>" in merged_review else "N/A"

    total_input = total_output = total_tokens = 0
    token_lines = []
    for agent_name, usage in agent_usages.items():
        if usage is None:
            token_lines.append(f"  {agent_name}: N/A (claude_sdk path)")
        else:
            token_lines.append(
                f"  {agent_name}: input={usage.input_tokens} output={usage.output_tokens} total={usage.total_tokens} requests={usage.requests}"
            )
            total_input += usage.input_tokens
            total_output += usage.output_tokens
            total_tokens += usage.total_tokens
    token_lines.append(f"  TOTAL: input={total_input} output={total_output} total={total_tokens}")

    log_path = Path(__file__).parent / os.environ.get("MERGE_LOG", "pipeline.log")
    with open(log_path, "a") as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Paper: {paper_path}\n")
        log_f.write(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n")
        log_f.write(f"\n--- Token Usage ---\n" + "\n".join(token_lines) + "\n")
        log_f.write(f"\n--- Merged Inputs ---\n\n{chr(10).join(labeled)}\n")
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


async def process_papers(papers: list[dict], papers_dir: Path, skip_scoring: bool, callback, no_cal: bool = False):
    """Run pipeline on a list of papers with CONCURRENCY concurrent tasks."""
    sem = asyncio.Semaphore(CONCURRENCY)

    async def process_one(i, paper_info):
        pid = paper_info["paper_id"]
        paper_path = papers_dir / f"{pid}.txt"
        print(f"\n[{i}/{len(papers)}] {paper_info.get('title', pid)} (avg={paper_info['avg_score']:.1f})")
        async with sem:
            try:
                result = await run_pipeline(str(paper_path), skip_scoring=skip_scoring, no_cal=no_cal)
            except Exception as e:
                raise RuntimeError(f"[{pid}] pipeline failed: {e}") from e
            if result is None:
                return
            callback(paper_info, result)

    await asyncio.gather(*(process_one(i, p) for i, p in enumerate(papers, 1)))


# ── Benchmark ────────────────────────────────────────────────────────

async def run_benchmark(data_dir: str, n_samples: int = 10, seed: int = 42, balanced: bool = False, no_cal: bool = False):
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
    csv_path = out_dir / os.getenv("OUTPUT_CSV", "bench_scores.csv")
    reviews_dir = out_dir / "bench_reviews"
    reviews_dir.mkdir(exist_ok=True)

    # Check for existing results and ask user whether to continue or overwrite
    finished = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        import pandas as pd
        existing_df = pd.read_csv(csv_path)
        existing_count = len(existing_df)
        print(f"\nFound existing {csv_path} with {existing_count} results.")
        choice = input("  [C]ontinue (skip finished papers) or [O]verwrite? [C/o]: ").strip().lower()
        if choice in ("o", "overwrite"):
            for review_file in reviews_dir.iterdir():
                if review_file.is_file():
                    review_file.unlink()
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
    await process_papers(samples, papers_dir, skip_scoring=False, callback=on_complete, no_cal=no_cal)

    scored = [r for r in results if r["pred_score"] != -1]
    if scored:
        mae = sum(abs(r["pred_score"] - r["gt_avg_score"]) for r in scored) / len(scored)
        print(f"\nResults: {len(scored)} scored, MAE={mae:.2f}")


# ── Single paper ─────────────────────────────────────────────────────

import datetime

async def run_single_paper(paper_path: str, no_cal: bool = False):
    print(f"Reviewing: {paper_path}")
    result = await run_pipeline(paper_path, no_cal=no_cal)
    print(f"\n{'=' * 72}\nFINAL REVIEW\n{'=' * 72}\n{result['merged_review']}")
    score = result["scorer_output"]
    if score != -1:
        print(f"\nPredicted score: {score}")
    os.makedirs(os.path.join(Path(__file__).parent, "reviews"), exist_ok=True)
    with open(os.path.join(Path(__file__).parent, "reviews", os.path.basename(paper_path).split(".")[0] + f"_review_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"), "w", encoding="utf-8") as f:
        f.write(f"# Review of {paper_path}\n\n")
        f.write(result["merged_review"])
        f.write(f"\n\n**Predicted score: {score}**\n" if score != -1 else "") 


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent paper reviewer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single_paper", type=str)
    group.add_argument("--benchmark", type=str, metavar="DATA_DIR")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--no_cal", action="store_true", help="Skip calibration sample search; score based on paper merits alone")
    args = parser.parse_args()

    if args.single_paper:
        asyncio.run(run_single_paper(args.single_paper, no_cal=args.no_cal))
    elif args.benchmark:
        asyncio.run(run_benchmark(args.benchmark, n_samples=args.n_samples, seed=args.seed, balanced=args.balanced, no_cal=args.no_cal))
