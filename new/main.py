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
from tools import read_file, read_file_full, grep_file, search_file, IN_SUBAGENT, new_tool_use_counts, _count_tool  # glob_files removed (unused)
import weave
weave.init("openai-agents")

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
import dotenv
dotenv.load_dotenv()
import os
os.environ["OPENAI_DEFAULT_MODEL"] = os.getenv("OPENAI_DEFAULT_MODEL", "z-ai/glm-5.1")
HARSH_MODEL = os.environ.get("HARSH_MODEL", "gpt-5.4")
NEUTRAL_MODEL = os.environ.get("NEUTRAL_MODEL")
MERGER_MODEL = os.environ.get("MERGER_MODEL", "ollama:glm-5.1:cloud")
SUBAGENT_MODEL = os.environ.get("SUBAGENT_MODEL", MERGER_MODEL)  # calibration_search subagent (OpenAI merger path)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
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

HUMAN_REVIEW_DIR = os.path.abspath("../human_reviews/")
CONCURRENCY = 10

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

CAL_INSTRUCTION_WITH = """Use comparative scoring to calibrate your final score.

How retrieval works: you do not have direct search tools for the human-review corpus. Use the `calibration_search` tool for every retrieval. It runs BM25 / vector search / grep internally and returns a list of paper paths with one-sentence summaries. You decide what to look for; it does the looking.

How to phrase the query to `calibration_search`: the query you pass is a complete instruction to a separate subagent, not a raw search phrase typed into a search box. Write it as a directive you would give to a helper: state what the subagent should do, what it should look for, what score range or topic/weakness pattern to target, and what you want back. Do not pass a bare keyword string or a lone phrase like "privacy attack face recognition" — those are search queries, not instructions. Instead, wrap the intent in an imperative instruction.

Good example (instruction): "Find 3-5 papers in the corpus whose reviews mention strong empirical results but overclaimed contributions, where the human score was 6 or higher. Return the paper paths with a one-sentence summary of each paper's main strengths and the overclaim."

Bad example (raw query phrase): "strong empirical results overclaim score 6+".

Workflow for every calibration step below:
1. Decide what you want to retrieve (topic, weakness pattern, strength pattern, score range, etc.).
2. Call `calibration_search` with a complete natural-language instruction (as described above) telling it what to retrieve and how to report back.
3. Read the returned paper list. If you want more detail on a specific anchor, use your own read_file tool on the returned absolute path.

Do not try to call search_file, grep_file, or the BM25/vector index directly — those tools are only available to the subagent. If you want more or different anchors, call `calibration_search` again with a refined instruction.

Your calibration process:

1. Topic-based anchors: ask `calibration_search` for papers with similar topics. Note their human scores.

2. Quality-based anchors: this is critical. Do not only search by topic. Instruct `calibration_search` to find papers that share similar strength/weakness patterns with the paper under review. Phrase each call as a full instruction, e.g.:
   - "Find papers in the corpus whose reviews flag overclaimed contributions alongside strong empirical results. Return 3-5 paper paths with their human scores and a one-sentence summary of the overclaim."
   - "Find papers whose reviews praise novel framing but flag missing or weak baselines. Return 3-5 paper paths with their human scores and a one-sentence summary of the baseline issue."

3. Deliberate range anchoring: seek out both high-scoring and low-scoring papers to anchor the extremes of your scale. Retrieve multiple (ideally 2-4) papers per score range, not just one — a single anchor is too noisy to rely on. Phrase each request as an instruction to `calibration_search`:
   - "Find 3-4 papers scored 7 or higher by humans. Return paths and one-sentence summaries of what made each strong."
   - "Find 3-4 papers scored 4-6 by humans (borderline anchors). Return paths and one-sentence summaries."
   - "Find 3-4 papers scored 3 or below by humans. Return paths and one-sentence summaries of what made each weak."
   - Compare the paper under review against all ranges, not just whichever came back in retrieval.

   Example instructions for a paper about privacy attacks on face recognition:
   - "Find high-scored (7+) papers on privacy attacks against face recognition. Return 3-5 paths with one-sentence summaries."
   - "Find low-scored (≤3) papers on privacy attacks against face recognition. Return 3-5 paths with one-sentence summaries of their weaknesses."
   - "Broaden to face recognition evaluation more generally: find high-scored anchors. Return 3-5 paths with summaries."
   - "Find rejected/low-scored papers on privacy evaluation with flaws similar to the paper under review. Return 3-5 paths with summaries."

   If no papers are found with the same topic, relax the instruction to a more general one and call `calibration_search` again.

4. Score relative to anchors: your final score should be positioned relative to the retrieved examples. If retrieved papers with similar strengths got 7s from humans, and papers with similar weaknesses got 3s, use that range. Do not compress everything into 4-6.

5. Score from the anchors, not from how the merged review reads. Papers with many listed weaknesses can still score high if their anchors did. Lean on the anchor range when your gut disagrees with it.

Retrieval is noisy — a single 8 or 3 doesn't pin your score. Use the center of the anchor cluster, weighted by topical similarity, and move outside that range only if the paper clearly beats or falls below most of the anchors.

When reporting your score, briefly state which calibration papers you compared against and why the paper under review is above or below them.

You can use read_file to read the returned anchor files for more detail. List the papers you compared and the reasoning.

Let the score distribution follow the actual quality of the paper relative to the calibration examples.
The samples could be concentrated in the middle, that does not mean you have to score it in the middle as well.

There are less papers with extreme scores, so if the paper is truly exceptional or truly weak, it is okay to give it an extreme score even if most found papers are in the middle. You can also try to ask `calibration_search` for more papers with extreme scores to see what made a paper really good/bad.

Limit your `calibration_search` invocations to 3–5 rounds, do NOT dig too deep into retrieval."""

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

    
def resolve_model(spec: str | None):
    """Return a model arg for Agent(...). Supports 'ollama:<name>' for local Ollama backend."""
    if spec is None:
        return None
    if spec.startswith("ollama:"):
        name = spec[len("ollama:"):]
        client = AsyncOpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
        return OpenAIChatCompletionsModel(model=name, openai_client=client)
    return spec

if HARSH_MODEL.startswith("claude_sdk:"):
    harsh = None  # Claude SDK Harsh Critic — invoked per-call in run_pipeline
    _HARSH_SDK_MODEL = HARSH_MODEL[len("claude_sdk:"):]
    _harsh_sdk_system_prompt = load_prompts("harsh_critic.md", paper_access=PAPER_ACCESS_FILE)
else:
    harsh = Agent(name="Harsh Critic", instructions=load_prompts("harsh_critic.md"), model=resolve_model(HARSH_MODEL))
    _HARSH_SDK_MODEL = None
    _harsh_sdk_system_prompt = None
neutral_reviewer = Agent(name="Strength Finder", instructions=load_prompts("neutral_reviewer.md"), model=resolve_model(NEUTRAL_MODEL))

_NO_CAL = "--no_cal" in __import__("sys").argv

CALIBRATION_SUBAGENT_INSTRUCTIONS = """You are a retrieval helper for the main merger agent. The main agent sends you a retrieval request (e.g. "find papers on face recognition privacy with high scores" or "find papers with weakness: unfair baseline comparison"), and you return a concise list of matching paper reviews.

You have these tools:
- search_file(query, n, mode): BM25 or vector search over human reviews. mode='bm25' or 'vector'.
- read_file(abs_path, start_line, end_line): read lines from a human review file.
- grep_file(pattern, abs_path): substring search inside a single file.

Workflow:
1. Run 1-3 search_file calls to find candidate reviews matching the request. Use vector search for semantic queries and bm25 for literal keyword matches.
2. Optionally skim promising candidates with read_file to confirm they match (especially to verify score/decision if the request specifies a score range).
3. Return a list of matching papers. For each: the absolute file path and ONE sentence describing why it matches (the key weakness/strength/topic/score that makes it relevant).

Output format (strict):
- <abs_path>: <one-sentence reason, mentioning human score and decision if known>
- <abs_path>: <one-sentence reason>
...

Constraints:
- Return 3-8 papers, not more.
- Do not produce a review, do not give calibration advice, do not compare the retrieved papers to the paper under review. The main agent handles all reasoning — you just retrieve.
- Keep the whole response under 300 words.
- Aim for 2–3 tool calls total; stop as soon as you have enough matches.
- Search exactly for what the main agent asked. Do not broaden or narrow the request on your own.
"""

if MERGER_MODEL.startswith("claude_sdk:"):
    merger = None  # Claude SDK merger — created per-call in run_pipeline
    _MERGER_SDK_MODEL = MERGER_MODEL[len("claude_sdk:"):]
else:
    _merger_instructions = load_prompts("merger.md", paper_access=PAPER_ACCESS_FILE, no_cal=_NO_CAL)
    if _NO_CAL:
        _merger_tools = [read_file, grep_file]
    else:
        _calibration_subagent = Agent(
            name="Calibration Search",
            instructions=CALIBRATION_SUBAGENT_INSTRUCTIONS,
            tools=[search_file, read_file, grep_file],
            model=resolve_model(SUBAGENT_MODEL),
        )
        _calibration_tool = _calibration_subagent.as_tool(
            tool_name="calibration_search",
            tool_description="Retrieve calibration anchors from the human-review corpus. Send a short retrieval request (topic / weakness / strength / score range). Returns a list of paper paths each with a one-sentence summary. Does not do calibration reasoning — just retrieves.",
            max_turns=12,
        )
        _orig_calibration_invoke = _calibration_tool.on_invoke_tool

        async def _counted_calibration_invoke(ctx, args):
            _count_tool("calibration_search")
            token = IN_SUBAGENT.set(True)
            try:
                return await _orig_calibration_invoke(ctx, args)
            finally:
                IN_SUBAGENT.reset(token)

        _calibration_tool.on_invoke_tool = _counted_calibration_invoke
        _merger_tools = [read_file, grep_file, _calibration_tool]
    merger = Agent(
        name="Merger",
        instructions=_merger_instructions,
        model=resolve_model(MERGER_MODEL),
        tools=_merger_tools,
    )
    _MERGER_SDK_MODEL = None

# scorer = Agent(name="Scorer", instructions=load_prompts("scorer_agent_gpt.txt"), tools=_tool_agents, model=SCORER_MODEL)


# ── Constants ────────────────────────────────────────────────────────
REVIEW_PROMPT = """Review the following paper thoroughly.

The paper was extracted from PDF by an automated parser. Treat formatting artifacts (broken equations, garbled tables, OCR errors) as parser issues, not paper flaws. The appendix and references were stripped by the parser; assume they exist in the original submission and don't flag them as missing.

{paper_path}
--- PAPER CONTENT START ---
{paper_content}
--- PAPER CONTENT END (everything after references stripped by parser) ---"""


# ── Core pipeline ────────────────────────────────────────────────────

async def run_pipeline(paper_path: str, skip_scoring: bool = False, no_cal: bool = False) -> dict:
    tool_counts = new_tool_use_counts()
    paper_path_abs = os.path.abspath(paper_path)
    with open(paper_path, "r") as f:
        paper_content = f.read()
    paper_content = paper_content

    review_prompt = REVIEW_PROMPT.format(paper_path=paper_path_abs, paper_content=paper_content)

    # Harsh Critic input: when running via Claude SDK, replace the inline paper
    # content with a directive to read it from disk (SDK has a CLI input length
    # limit). Otherwise, pass the full paper inline as before.
    sdk_harsh_user_prompt = (
        f"The paper was extracted from PDF by an automated parser. Treat formatting artifacts (broken equations, garbled tables, OCR errors) as parser issues, not paper flaws. The appendix and references were stripped by the parser; assume they exist in the original submission and don't flag them as missing.\n\n"
        f"Paper path: {paper_path_abs}. Use read_file (in chunks) to read the paper end-to-end before reviewing."
    )

    async def _run_harsh():
        if _HARSH_SDK_MODEL is not None:
            from claude_merger import run_harsh_claude_sdk
            paper_dir = str(Path(paper_path_abs).parent)
            text, usage = await run_harsh_claude_sdk(
                _HARSH_SDK_MODEL, sdk_harsh_user_prompt, paper_dir, _harsh_sdk_system_prompt
            )
            return ("Harsh Critic", text, None, usage)
        text, usage = await run_agent_with_retry(harsh, review_prompt)
        return ("Harsh Critic", text, usage, None)

    async def _run_neutral():
        text, usage = await run_agent_with_retry(neutral_reviewer, review_prompt)
        return ("Strength Finder", text, usage, None)

    print(f"  Phase 1: Running 2 agents in parallel ...")
    phase1_results = await asyncio.gather(_run_harsh(), _run_neutral(), return_exceptions=True)
    for r in phase1_results:
        if isinstance(r, Exception):
            print(f"  🔥ERROR: {paper_path} — phase 1 agent raised {type(r).__name__}: {r}")
            return None

    agent_usages: dict = {}
    sdk_usages: dict = {}
    outputs = []
    names = []
    for name, out, openai_usage, sdk_usage_phase1 in phase1_results:
        names.append(name)
        outputs.append(out)
        agent_usages[name] = openai_usage  # may be None for SDK path
        if sdk_usage_phase1 is not None:
            sdk_usages[name] = sdk_usage_phase1
    labeled = [f"### {n}\n{o}" for n, o in zip(names, outputs)]


    print("  Phase 2: Merger ...")
    merger_start = time.monotonic()
    if _MERGER_SDK_MODEL is not None:
        from claude_merger import run_merger_claude_sdk
        merger_prompt = (
            f"Here is the paper being reviewed (extracted from PDF — formatting "
            f"artifacts are parser issues, not paper problems):\n\n"
            f"Paper path: {paper_path_abs}, read it in chunks.\n\n"
            f"Human reviews directory (for calibration): {HUMAN_REVIEW_DIR}\n\n"
            f"Here are the inputs:\n\n{chr(10).join(labeled)}\n\n"
            f"Now produce the final consolidated review following your instructions. "
            f"Remember: many of the harsh critic's points may be nonsensical or overly "
            f"picky — cross-check everything against the actual paper before including it."
        )
        paper_dir = str(Path(paper_path_abs).parent)
        merged_review, merger_sdk_usage = await run_merger_claude_sdk(_MERGER_SDK_MODEL, merger_prompt, paper_dir, no_cal=no_cal)
        sdk_usages["Merger"] = merger_sdk_usage
        agent_usages["Merger"] = None  # SDK usage tracked separately below
    else:
        # OpenAI Agent SDK merger: grant read/grep access to the paper's dir and
        # point the merger at the paper via path (not inline).
        from tools import allow_path
        allow_path(str(Path(paper_path_abs).parent))
        merger_prompt = (
            f"Here is the paper being reviewed (extracted from PDF — formatting "
            f"artifacts are parser issues, not paper problems).\n\n"
            f"Paper path: {paper_path_abs} — use read_file (in chunks) or grep_file to read it.\n\n"
            f"Human reviews directory (for calibration): {HUMAN_REVIEW_DIR}\n\n"
            f"Here are the inputs:\n\n{chr(10).join(labeled)}\n\n"
            f"Now produce the final consolidated review following your instructions. "
            f"Remember: many of the harsh critic's points may be nonsensical or overly "
            f"picky — cross-check everything against the actual paper before including it."
        )
        merged_review, merger_usage = await run_agent_with_retry(merger, merger_prompt)
        agent_usages["Merger"] = merger_usage
    merger_elapsed = time.monotonic() - merger_start
    scorer_output = float(merged_review.split("<pineapple>")[1].split("</pineapple>")[0]) if "<pineapple>" in merged_review else -1
    decision = (merged_review.split("<orange>")[1].split("</orange>")[0]) if "<orange>" in merged_review else "N/A"

    total_input = total_output = total_tokens = 0
    token_lines = []
    for agent_name, usage in agent_usages.items():
        if usage is None:
            token_lines.append(f"  {agent_name}: N/A (claude_sdk path)")
        else:
            cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
            reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None)
            token_lines.append(
                f"  {agent_name}: input={usage.input_tokens} (cached={cached}) "
                f"output={usage.output_tokens} (reasoning={reasoning}) "
                f"total={usage.total_tokens} requests={usage.requests}"
            )
            total_input += usage.input_tokens
            total_output += usage.output_tokens
            total_tokens += usage.total_tokens
    token_lines.append(f"  TOTAL: input={total_input} output={total_output} total={total_tokens}")

    merger_output_tokens = 0
    _m_usage = agent_usages.get("Merger")
    if _m_usage is not None:
        merger_output_tokens = getattr(_m_usage, "output_tokens", 0) or 0
    _m_sdk = sdk_usages.get("Merger")
    if _m_sdk is not None:
        _u = (_m_sdk or {}).get("usage") or {}
        merger_output_tokens += (_u.get("output_tokens") or 0)
    merger_tps = (merger_output_tokens / merger_elapsed) if merger_elapsed > 0 else 0.0
    token_lines.append(
        f"  Merger throughput: {merger_output_tokens} output tokens / {merger_elapsed:.1f}s "
        f"= {merger_tps:.1f} tok/s"
    )

    sdk_lines = []
    sdk_total_cost = 0.0
    for sdk_name, su in sdk_usages.items():
        u = (su or {}).get("usage") or {}
        sdk_lines.append(f"  [{sdk_name}]")
        sdk_lines.append(f"    Model: {su.get('model')}")
        sdk_lines.append(f"    Session ID: {su.get('session_id')}")
        sdk_lines.append(f"    Cost (USD): {su.get('total_cost_usd')}")
        sdk_lines.append(f"    Turns: {su.get('num_turns')}")
        sdk_lines.append(f"    Duration: total={su.get('duration_ms')}ms api={su.get('duration_api_ms')}ms")
        sdk_lines.append(
            f"    Tokens: input={u.get('input_tokens')} output={u.get('output_tokens')} "
            f"cache_read={u.get('cache_read_input_tokens')} cache_creation={u.get('cache_creation_input_tokens')}"
        )
        rl = (su or {}).get("rate_limit")
        if rl:
            util = rl.get("utilization")
            util_str = f"{util*100:.1f}%" if util is not None else "n/a"
            sdk_lines.append(
                f"    Plan usage: type={rl.get('type')} util={util_str} "
                f"status={rl.get('status')} overage={rl.get('overage_status')}"
            )
        if su.get("total_cost_usd"):
            sdk_total_cost += su["total_cost_usd"]
    if sdk_lines:
        sdk_lines.append(f"  TOTAL Claude SDK cost (USD): {sdk_total_cost:.4f}")

    tool_lines = []
    if _MERGER_SDK_MODEL is None:
        main_counts = dict(tool_counts["main"])
        sub_counts = dict(tool_counts["subagent"])
        main_total = sum(main_counts.values())
        sub_total = sum(sub_counts.values())
        tool_lines.append(f"  Main merger: total={main_total} " + (", ".join(f"{k}={v}" for k, v in sorted(main_counts.items())) or "(none)"))
        tool_lines.append(f"  Subagent:    total={sub_total} " + (", ".join(f"{k}={v}" for k, v in sorted(sub_counts.items())) or "(none)"))

    log_path = Path(__file__).parent / os.environ.get("MERGE_LOG", "pipeline.log")
    with open(log_path, "a") as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Paper: {paper_path}\n")
        log_f.write(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}\n")
        log_f.write(f"\n--- Token Usage ---\n" + "\n".join(token_lines) + "\n")
        if tool_lines:
            log_f.write(f"\n--- Tool Use Counts (OpenAI Agent SDK) ---\n" + "\n".join(tool_lines) + "\n")
        if sdk_lines:
            log_f.write(f"\n--- Claude SDK Usage ---\n" + "\n".join(sdk_lines) + "\n")
        log_f.write(f"\n--- Merged Inputs ---\n\n{chr(10).join(labeled)}\n")
        log_f.write(f"\n--- Merged Review ---\n{merged_review}\n")
        log_f.write(f"\n--- Scorer Output ---\n{scorer_output}\n")
        log_f.write(f"\n--- Decision ---\n{decision}\n")

    return {
        "merged_review": merged_review,
        "scorer_output": scorer_output,
        "decision": decision,
        "sdk_usages": sdk_usages,
        "tool_use_counts": {"main": dict(tool_counts["main"]), "subagent": dict(tool_counts["subagent"])},
        "merger_elapsed_s": merger_elapsed,
        "merger_output_tokens": merger_output_tokens,
        "merger_tokens_per_s": merger_tps,
    }


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

def predict_acceptance_rate(csv_path: str, score: float, window: float = 0.5):
    if not os.path.exists(csv_path):
        print(f"  Acceptance CSV not found: {csv_path}")
        return None
    exact_total = exact_acc = win_total = win_acc = 0
    all_scores = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            try:
                s = float(row["pred_score"])
            except (ValueError, KeyError, TypeError):
                continue
            all_scores.append(s)
            gt = row.get("gt_binary", "").strip()
            if gt not in ("Accept", "Reject"):
                continue
            is_acc = gt == "Accept"
            if abs(s - score) < 1e-9:
                exact_total += 1
                exact_acc += is_acc
            if abs(s - score) <= window:
                win_total += 1
                win_acc += is_acc
    exact_rate = (exact_acc / exact_total) if exact_total else float("nan")
    win_rate = (win_acc / win_total) if win_total else float("nan")
    if all_scores:
        below = sum(1 for s in all_scores if s < score)
        equal = sum(1 for s in all_scores if abs(s - score) < 1e-9)
        percentile = (below + 0.5 * equal) / len(all_scores) * 100
        pct_n = len(all_scores)
    else:
        percentile = float("nan")
        pct_n = 0
    return exact_rate, exact_total, win_rate, win_total, percentile, pct_n


from datalab_sdk import AsyncDatalabClient, ConvertOptions

async def pdf_to_markdown(pdf_path: Path) -> str:
    options = ConvertOptions(
        output_format="markdown",  # "markdown", "html", "json", "chunks"
        mode="fast",           # "fast", "balanced", "accurate"
        paginate=True,             # Add page delimiters
        page_range="0-9",         # Process specific pages (0-indexed)
        token_efficient_markdown=True,  # Optimize markdown output for LLM token usage
    )

    async with AsyncDatalabClient() as client:
        result = await client.convert(pdf_path, options=options)
    return result.markdown + "\n\n Rest of paper (reference and Appendix) is removed."

import re
async def run_single_paper(paper_path: str, no_cal: bool = False, accept_csv: str | None = None):
    print(f"Reviewing: {paper_path}")

    if paper_path.endswith(".pdf"):
        md = await pdf_to_markdown(Path(paper_path))
        md = re.sub(r"Published as a conference paper at ICLR \d{4}\s*\n?", "", md)
        md_path = Path(paper_path).with_suffix(".md")
        md_path.write_text(md, encoding="utf-8")
        paper_path = str(md_path)
        print(f"Converted PDF to markdown: {paper_path}")

    result = await run_pipeline(paper_path, no_cal=no_cal)
    print(f"\n{'=' * 72}\nFINAL REVIEW\n{'=' * 72}\n{result['merged_review']}")
    score = result["scorer_output"]
    accept_info = None
    if score != -1:
        print(f"\nPredicted score: {score}")
        if accept_csv:
            accept_info = predict_acceptance_rate(accept_csv, score)
            if accept_info is not None:
                exact_rate, exact_n, win_rate, win_n, percentile, pct_n = accept_info
                print(f"Acceptance rate @ score={score}: {exact_rate:.2%} (n={exact_n})")
                print(f"Acceptance rate @ score={score}±0.5: {win_rate:.2%} (n={win_n})")
                print(f"Percentile of score={score}: {percentile:.1f}% (n={pct_n})")

    if result.get("merger_elapsed_s") is not None:
        print(
            f"\nMerger throughput: {result['merger_output_tokens']} output tokens / "
            f"{result['merger_elapsed_s']:.1f}s = {result['merger_tokens_per_s']:.1f} tok/s "
            f"(output-only; input processing ~instant, main agent waits on subagents, tools ~instant)"
        )

    tuc = result.get("tool_use_counts") or {}
    if tuc and (tuc.get("main") or tuc.get("subagent")):
        main_c = tuc.get("main", {})
        sub_c = tuc.get("subagent", {})
        print(f"\n{'=' * 72}\nOpenAI Agent SDK Tool Use\n{'=' * 72}")
        print(f"  Main merger: total={sum(main_c.values())} " + (", ".join(f"{k}={v}" for k, v in sorted(main_c.items())) or "(none)"))
        print(f"  Subagent:    total={sum(sub_c.values())} " + (", ".join(f"{k}={v}" for k, v in sorted(sub_c.items())) or "(none)"))

    sdk_usages = result.get("sdk_usages") or {}
    if sdk_usages:
        print(f"\n{'=' * 72}\nClaude SDK Usage\n{'=' * 72}")
        total_cost = 0.0
        for name, su in sdk_usages.items():
            u = (su or {}).get("usage") or {}
            print(f"  [{name}]")
            print(f"    Model:         {su.get('model')}")
            print(f"    Session ID:    {su.get('session_id')}")
            print(f"    Cost (USD):    ${su.get('total_cost_usd')}")
            print(f"    Turns:         {su.get('num_turns')}")
            print(f"    Duration:      total={su.get('duration_ms')}ms api={su.get('duration_api_ms')}ms")
            print(f"    Input tokens:  {u.get('input_tokens')}")
            print(f"    Output tokens: {u.get('output_tokens')}")
            print(f"    Cache read:    {u.get('cache_read_input_tokens')}")
            print(f"    Cache create:  {u.get('cache_creation_input_tokens')}")
            rl = (su or {}).get("rate_limit")
            if rl:
                util = rl.get("utilization")
                util_str = f"{util*100:.1f}%" if util is not None else "n/a"
                print(f"    Plan usage:    type={rl.get('type')} util={util_str} status={rl.get('status')} overage={rl.get('overage_status')}")
            if su.get("total_cost_usd"):
                total_cost += su["total_cost_usd"]
        print(f"  TOTAL Claude SDK cost (USD): ${total_cost:.4f}")
    os.makedirs(os.path.join(Path(__file__).parent, "reviews"), exist_ok=True)
    with open(os.path.join(Path(__file__).parent, "reviews", os.path.basename(paper_path).split(".")[0] + f"_review_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"), "w", encoding="utf-8") as f:
        f.write(f"# Review of {paper_path}\n\n")
        f.write(result["merged_review"])
        f.write(f"\n\n**Predicted score: {score}**\n" if score != -1 else "")
        if accept_info is not None:
            exact_rate, exact_n, win_rate, win_n, percentile, pct_n = accept_info
            f.write(f"\n**Acceptance rate @ score={score}: {exact_rate:.2%} (n={exact_n})**\n")
            f.write(f"\n**Acceptance rate @ score={score}±0.5: {win_rate:.2%} (n={win_n})**\n")
            f.write(f"\n**Percentile of score={score}: {percentile:.1f}% (n={pct_n})**\n")


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
    parser.add_argument("--accept_csv", type=str, default=None, help="Path to bench CSV; predict acceptance rate at predicted score and ±0.5")
    args = parser.parse_args()

    if args.single_paper:
        asyncio.run(run_single_paper(args.single_paper, no_cal=args.no_cal, accept_csv=args.accept_csv))
    elif args.benchmark:
        asyncio.run(run_benchmark(args.benchmark, n_samples=args.n_samples, seed=args.seed, balanced=args.balanced, no_cal=args.no_cal))
