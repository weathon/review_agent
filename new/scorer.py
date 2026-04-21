"""
Scorer: second-stage agent that takes the Merger's final review + compacted
context block and assigns a numeric score + accept/reject decision.

Two backends:
- OpenAI Agents SDK (`build_scorer_agent`, `run_scorer_openai`), mirrors how
  the merger was set up in main.py.
- Claude Agent SDK (`run_scorer_claude_sdk`), mirrors `run_merger_claude_sdk`.

When calibration is enabled, the scorer has access to the calibration_search
subagent (SDK path) / calibration_search tool (OpenAI path) for anchor
retrieval. When `no_cal=True`, the scorer just reads the review + context and
scores directly.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from claude_merger import _make_merger_mcp_server, _run_claude_sdk_query


with open("prompts/cal_with.md", "r") as _f:
    CAL_INSTRUCTION_WITH = _f.read()

with open("prompts/cal_without.md", "r") as _f:
    CAL_INSTRUCTION_WITHOUT = _f.read()


SCORER_USER_PROMPT_TEMPLATE = """You are scoring a paper given the Merger's final review plus the Merger's compacted context block.

Human reviews directory (for calibration anchors, if enabled): {human_review_dir}

--- MERGER OUTPUT START ---
{merged_review}
--- MERGER OUTPUT END ---

Follow your instructions to produce the final score and decision."""


CALIBRATION_SUBAGENT_PROMPT = """You are a retrieval helper for the Scorer agent. The Scorer sends you a retrieval request (e.g. "find papers on face recognition privacy with high scores" or "find papers with weakness: unfair baseline comparison"), and you return a concise list of matching paper reviews.

You have these tools (all under the mcp__merger_fs__ namespace):
- search_file(query, n, mode, low_score=0, high_score=10): BM25 or vector search over human reviews, pre-filtered by the reviewer avg-score range. mode='vector' or 'bm25'. Set low_score/high_score to anchor to a band (e.g. low_score=7 for strong papers, high_score=3 for weak ones).
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
- Do not produce a review, do not give calibration advice, do not compare the retrieved papers to the paper under review. The Scorer handles all reasoning — you just retrieve.
- Keep the whole response under 300 words.
- Cap yourself at 6 tool calls total.
- Search exactly for what the Scorer asked. Do not broaden or narrow the request on your own.
"""


def _load_scorer_system_prompt(no_cal: bool) -> str:
    with open("prompts/scorer.md", "r") as f:
        prompt = f.read()
    cal_instruction = CAL_INSTRUCTION_WITHOUT if no_cal else CAL_INSTRUCTION_WITH
    return prompt.replace("{{CALIBRATION_INSTRUCTION}}", cal_instruction)


def build_scorer_user_prompt(merged_review: str, human_review_dir: str) -> str:
    return SCORER_USER_PROMPT_TEMPLATE.format(
        merged_review=merged_review,
        human_review_dir=human_review_dir,
    )


# ── OpenAI Agents SDK path ──────────────────────────────────────────────

def build_scorer_agent(resolve_model, scorer_model: str, subagent_model: str, no_cal: bool):
    """
    Construct an OpenAI-Agents-SDK Scorer agent.

    resolve_model: main.py's resolve_model helper (callable).
    scorer_model / subagent_model: model specs (same semantics as MERGER_MODEL).
    no_cal: if True, the scorer gets no calibration tool.
    """
    from agents import Agent
    from tools import read_file, grep_file, search_file

    instructions = _load_scorer_system_prompt(no_cal)

    if no_cal:
        tools = []
    else:
        calibration_subagent = Agent(
            name="Calibration Search",
            instructions=CALIBRATION_SUBAGENT_PROMPT,
            tools=[search_file, read_file, grep_file],
            model=resolve_model(subagent_model),
        )
        calibration_tool = calibration_subagent.as_tool(
            tool_name="calibration_search",
            tool_description="Retrieve calibration anchors from the human-review corpus. Send a short retrieval request (topic / weakness / strength / score range). Returns a list of paper paths each with a one-sentence summary. Does not do calibration reasoning — just retrieves.",
            max_turns=12,
        )
        # Scorer also needs to be able to read the anchor files it gets back.
        tools = [read_file, grep_file, calibration_tool]

    return Agent(
        name="Scorer",
        instructions=instructions,
        model=resolve_model(scorer_model),
        tools=tools,
    )


async def run_scorer_openai(scorer_agent, merged_review: str, human_review_dir: str, run_agent_with_retry):
    """
    Run an OpenAI-Agents-SDK Scorer. Returns (text, usage).
    """
    user_prompt = build_scorer_user_prompt(merged_review, human_review_dir)
    print("  [Scorer] starting OpenAI Agents SDK ...")
    _wall_start = time.monotonic()
    out, usage = await run_agent_with_retry(scorer_agent, user_prompt)
    wall_ms = int((time.monotonic() - _wall_start) * 1000)
    print(f"  [Scorer] done (OpenAI Agents SDK) — total time {wall_ms/1000:.1f}s")
    return out, usage


# ── Claude Agent SDK path ───────────────────────────────────────────────

def _make_calibration_subagent():
    from claude_agent_sdk import AgentDefinition

    return AgentDefinition(
        description="Retrieval helper for calibration anchors. Accepts a free-form retrieval request (e.g. 'find papers with weakness X' or 'find papers scored 7+ on topic Y') and returns 3-8 paper paths each with a one-sentence summary. Does not do calibration reasoning — just retrieves.",
        prompt=CALIBRATION_SUBAGENT_PROMPT,
        tools=[
            "mcp__merger_fs__search_file",
            "mcp__merger_fs__read_file",
            "mcp__merger_fs__grep_file",
        ],
        model="haiku",
    )


async def run_scorer_claude_sdk(
    model_id: str,
    merged_review: str,
    human_review_dir: str,
    paper_dir: str,
    no_cal: bool = False,
) -> tuple[str, dict]:
    """
    Run the Scorer via Claude Agent SDK. Returns (text, usage dict).

    The scorer has read_file/grep_file for reading human-review anchor files
    returned by calibration_search. It does NOT read the paper — that is the
    merger's job and the context block is the scorer's window into the paper.
    """
    system_prompt = _load_scorer_system_prompt(no_cal)

    mcp_server = _make_merger_mcp_server(paper_dir, no_cal=no_cal)

    allowed_tools = [
        "mcp__merger_fs__read_file",
        "mcp__merger_fs__grep_file",
    ]
    agents = None
    if not no_cal:
        allowed_tools.append("Task")
        agents = {"calibration_search": _make_calibration_subagent()}

    user_prompt = build_scorer_user_prompt(merged_review, human_review_dir)

    return await _run_claude_sdk_query(
        label="Scorer",
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        allowed_tools=allowed_tools,
        mcp_servers={"merger_fs": mcp_server},
        agents=agents,
        max_turns=30,
    )
