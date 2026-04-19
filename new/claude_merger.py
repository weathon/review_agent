"""
Claude Agent SDK merger for main.py.
Used when MERGER_MODEL starts with 'claude_sdk:'.
"""
from __future__ import annotations

import os
import numpy as np
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

HUMAN_REVIEW_DIR = os.path.abspath("../human_reviews/")

# ── Build indexes (mirrors tools.py) ──────────────────────────────────
_bm25_db: dict = {}
_or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

def _ensure_indexes():
    global _bm25_db, _vectors, _filenames
    if _bm25_db:
        return
    all_files = []
    all_file_paths = []
    for root, _, files in os.walk(HUMAN_REVIEW_DIR):
        for fname in files:
            if fname.endswith(".txt") or fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                with open(fpath, "r", errors="replace") as fh:
                    content = fh.read()
                if content.strip():
                    all_files.append(content)
                    all_file_paths.append(fpath)
    tokenized = [doc.split(" ") for doc in all_files]
    _bm25_db["bm25"] = BM25Okapi(tokenized)
    _bm25_db["files"] = all_file_paths

    with open("./human_reviews_embeddings.pkl", "rb") as f:
        db = pickle.load(f)
    _bm25_db["filenames"] = list(db.keys())
    _bm25_db["vectors"] = np.array(list(db.values()))


def _make_merger_mcp_server(paper_dir: str, no_cal: bool = False):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    if not no_cal:
        _ensure_indexes()
    allowed_paths = [paper_dir, HUMAN_REVIEW_DIR]

    def _check_path(path: str) -> str | None:
        resolved = os.path.abspath(path)
        if any(resolved.startswith(ap) for ap in allowed_paths):
            return None
        return f"ERROR: Access denied. Path '{resolved}' is not under any allowed directory: {allowed_paths}"

    @tool(
        "read_file",
        "Read lines from a file. Returns lines numbered start_line to end_line (1-based). If end_line is 0, reads to EOF.",
        {"abs_path": str, "start_line": int, "end_line": int},
    )
    async def _read_file(args: dict) -> dict:
        abs_path = args["abs_path"]
        start_line = args.get("start_line", 1) or 1
        end_line = args.get("end_line", 0) or 0
        print(f"  [claude:read_file] {abs_path} lines {start_line}-{end_line or 'EOF'}")
        err = _check_path(abs_path)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        try:
            with open(abs_path, "r", errors="replace") as fh:
                lines = fh.readlines()
            selected = lines[max(0, start_line - 1):end_line if end_line > 0 else len(lines)]
            text = "".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))
            return {"content": [{"type": "text", "text": text}]}
        except FileNotFoundError:
            return {"content": [{"type": "text", "text": f"ERROR: File not found: {abs_path}"}], "is_error": True}

    @tool(
        "grep_file",
        "Search a single file for a substring pattern. Returns matching lines with line numbers.",
        {"pattern": str, "abs_path": str},
    )
    async def _grep_file(args: dict) -> dict:
        import re as _re
        pattern = args["pattern"]
        abs_path = args["abs_path"]
        print(f"  [merger:grep_file] pattern='{pattern}' in '{abs_path}'")
        err = _check_path(abs_path)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        if not os.path.isfile(abs_path):
            return {"content": [{"type": "text", "text": f"ERROR: '{abs_path}' is not a file."}], "is_error": True}
        matches = []
        try:
            with open(abs_path, "r", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if _re.search(pattern, line):
                        matches.append(f"{i}: {line.rstrip()}")
        except Exception as e:
            return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}
        text = "\n".join(matches) if matches else "No matches found."
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "search_file",
        "Search for a query in human reviews using BM25 or vector embeddings. Returns top n matching files. Set mode='bm25' or 'vector'.",
        {"query": str, "n": int, "mode": str},
    )
    async def _search_file(args: dict) -> dict:
        query = args["query"]
        n = args.get("n", 5)
        mode = args.get("mode", "vector")
        print(f"  [merger:search_file] query='{query}' n={n} mode='{mode}'")
        if mode == "bm25":
            bm25 = _bm25_db["bm25"]
            files = _bm25_db["files"]
            tokenized_query = query.split(" ")
            doc_scores = bm25.get_scores(tokenized_query)
            top_indices = doc_scores.argsort()[-n:][::-1]
            results = []
            for idx in top_indices:
                fpath = os.path.abspath(files[idx])
                score = doc_scores[idx]
                with open(fpath, "r", errors="replace") as fh:
                    content = fh.read()
                results.append(f"{fpath}\nscore: {score:.2f}\nfirst 1000 chars:\n{content[:1000]}\n")
            text = "\n---\n".join(results) if results else "No relevant files found."
        else:
            query_embedding = _or_client.embeddings.create(
                model="google/gemini-embedding-001",
                input=query,
                encoding_format="float",
            )
            query_vector = np.array(query_embedding.data[0].embedding)
            vectors = _bm25_db["vectors"]
            filenames = _bm25_db["filenames"]
            similarities = vectors @ query_vector.T
            top_indices = similarities.argsort()[-n:][::-1]
            results = []
            for idx in top_indices:
                fpath = os.path.abspath(f"../human_reviews/{filenames[idx]}")
                score = similarities[idx]
                with open(fpath, "r", errors="replace") as fh:
                    content = fh.read()
                results.append(f"{fpath}\nscore: {score:.2f}\nfirst 1000 chars:\n{content[:1000]}\n")
            text = "\n---\n".join(results) if results else "No relevant files found."

        return {"content": [{"type": "text", "text": text}]}

    tools = [_read_file, _grep_file]
    if not no_cal:
        tools.append(_search_file)
    return create_sdk_mcp_server(
        name="merger_fs",
        version="1.0.0",
        tools=tools,
    )


CAL_INSTRUCTION_WITH = """Use comparative scoring to calibrate your final score.

How retrieval works: you do not have direct search tools for the human-review corpus. Use the `calibration_search` subagent (invoked via the Task tool with subagent_type='calibration_search') for every retrieval. The subagent runs BM25 / vector search / grep internally and returns a list of paper paths with one-sentence summaries. You decide what to look for; it does the looking.

Workflow for every calibration step below:
1. Decide what you want to retrieve (topic, weakness pattern, strength pattern, score range, etc.).
2. Invoke the `calibration_search` subagent with a short natural-language request describing what you want.
3. Read the returned paper list. If you want more detail on a specific anchor, use your own read_file tool on the returned absolute path.

Do not try to call search_file, grep_file, or the BM25/vector index directly — those tools are only available to the subagent. If you want more or different anchors, invoke the subagent again with a refined request.

Your calibration process:

1. Topic-based anchors: ask the subagent to retrieve papers with similar topics. Note their human scores.

2. Quality-based anchors: this is critical. Do not only search by topic. Ask the subagent for papers that share similar strength/weakness patterns with the paper under review:
   - If this paper has strong empirical results but overclaims, ask for reviews mentioning "overclaim" "strong experiments" and note how humans scored those.
   - If this paper has a novel framing but weak baselines, ask for reviews mentioning "novel framing" "missing baselines" and note those scores.

3. Deliberate range anchoring: seek out both high-scoring and low-scoring papers to anchor the extremes of your scale. Retrieve multiple (ideally 2-4) papers per score range, not just one — a single anchor is too noisy to rely on:
   - Ask for reviews of papers that were scored ~7+ by humans. Read a few of them to see what made them strong.
   - Ask for reviews of papers that were scored ~4-6 by humans. These are your borderline anchors.
   - Ask for reviews of papers that were scored ~3 or below by humans. Read a few to see what made them weak.
   - Compare the paper under review against all ranges, not just whichever came back in retrieval.

   Examples: if reviewing a paper about privacy attacks on face recognition, ask for:
   - "privacy attack face recognition strong paper" → find high-scored papers in the same area
   - "privacy attack face recognition weak paper" → find low-scored papers in the same area
   - "face recognition evaluation paper high score" → broaden to related topics at the high end
   - "privacy evaluation rejected" → find low-end anchors with similar flaws

   If no papers are found with the same topic, you can use more general queries.

4. Score relative to anchors: your final score should be positioned relative to the retrieved examples. If retrieved papers with similar strengths got 7s from humans, and papers with similar weaknesses got 3s, use that range. Do not compress everything into 4-6.

5. Score from the anchors, not from how the merged review reads. Papers with many listed weaknesses can still score high if their anchors did. Lean on the anchor range when your gut disagrees with it.

Retrieval is noisy — a single 8 or 3 doesn't pin your score. Use the center of the anchor cluster, weighted by topical similarity, and move outside that range only if the paper clearly beats or falls below most of the anchors.

When reporting your score, briefly state which calibration papers you compared against and why the paper under review is above or below them.

You can use read_file to read the returned anchor files for more detail. List the papers you compared and the reasoning.

Let the score distribution follow the actual quality of the paper relative to the calibration examples.
The samples could be concentrated in the middle, that does not mean you have to score it in the middle as well.

There are less papers with extreme scores, so if the paper is truly exceptional or truly weak, it is okay to give it an extreme score even if most found papers are in the middle. You can also try to ask the subagent for more papers with extreme scores to see what made a paper really good/bad.

Limit your calibration_search invocations to less than 20 rounds, do not dig too deep into retrieval."""

CAL_INSTRUCTION_WITHOUT = """Assign a score based solely on your assessment of the paper's quality. Do NOT use the search or review finder tools for calibration — score directly from the paper's merits and weaknesses as identified in the review above."""


CALIBRATION_SUBAGENT_PROMPT = """You are a retrieval helper for the main merger agent. The main agent sends you a retrieval request (e.g. "find papers on face recognition privacy with high scores" or "find papers with weakness: unfair baseline comparison"), and you return a concise list of matching paper reviews.

You have these tools (all under the mcp__merger_fs__ namespace):
- search_file(query, n, mode): BM25 or vector search over human reviews. mode='vector' (default) or 'bm25'.
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
- Cap yourself at 6 tool calls total.
- Search exactly for what the main agent asked. Do not broaden or narrow the request on your own.
"""


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


async def _run_claude_sdk_query(
    *,
    label: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    allowed_tools: list[str],
    mcp_servers: dict | None = None,
    agents: dict | None = None,
    max_turns: int = 30,
) -> tuple[str, dict]:
    """
    Generic single-turn-style Claude SDK runner. Captures cost/usage from
    ResultMessage and returns (text, usage dict).
    """
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        AssistantMessage,
        TextBlock,
        ResultMessage,
        RateLimitEvent,
    )

    print(f"  [{label}] starting Claude Agent SDK ({model_id}) ...")

    options = ClaudeAgentOptions(
        model=model_id,
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
        disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write"],
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        cwd="/tmp",
        agents=agents,
    )

    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    result_text = ""
    sdk_usage: dict = {
        "model": model_id,
        "session_id": None,
        "total_cost_usd": None,
        "num_turns": None,
        "duration_ms": None,
        "duration_api_ms": None,
        "usage": None,
        "rate_limit": None,
    }
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(full_prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text
            elif isinstance(message, ResultMessage):
                sdk_usage["session_id"] = message.session_id
                sdk_usage["total_cost_usd"] = message.total_cost_usd
                sdk_usage["num_turns"] = message.num_turns
                sdk_usage["duration_ms"] = message.duration_ms
                sdk_usage["duration_api_ms"] = message.duration_api_ms
                sdk_usage["usage"] = message.usage
            elif isinstance(message, RateLimitEvent):
                info = message.rate_limit_info
                sdk_usage["rate_limit"] = {
                    "status": info.status,
                    "type": info.rate_limit_type,
                    "utilization": info.utilization,
                    "resets_at": info.resets_at,
                    "overage_status": info.overage_status,
                    "overage_resets_at": info.overage_resets_at,
                }

    if not result_text.strip():
        raise RuntimeError(f"[{label}] Claude Agent SDK returned empty output")

    print(f"  [{label}] done — {model_id} (Claude Agent SDK)")
    return result_text, sdk_usage


async def run_harsh_claude_sdk(model_id: str, harsh_prompt_user: str, paper_dir: str, system_prompt: str) -> tuple[str, dict]:
    """
    Run the Harsh Critic via Claude Agent SDK with only read_file (so it can
    read the paper from disk instead of receiving it inline).
    """
    mcp_server = _make_merger_mcp_server(paper_dir, no_cal=True)
    return await _run_claude_sdk_query(
        label="Harsh Critic",
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=harsh_prompt_user,
        allowed_tools=["mcp__merger_fs__read_file"],
        mcp_servers={"merger_fs": mcp_server},
        max_turns=15,
    )


async def run_merger_claude_sdk(model_id: str, merger_prompt: str, paper_dir: str, no_cal: bool = False) -> tuple[str, dict]:
    """
    Run the merger agent via Claude Agent SDK.
    Returns (final merged review text, usage dict with cost/tokens/turns).
    """
    with open("prompts/merger.md", "r") as f:
        system_prompt = f.read()
    system_prompt = system_prompt.replace(
        "{{PAPER_ACCESS_INSTRUCTION}}",
        "The paper path is provided in the user message. Use read_file to read the paper and verify reviewer claims directly.",
    )
    cal_instruction = CAL_INSTRUCTION_WITHOUT if no_cal else CAL_INSTRUCTION_WITH
    system_prompt = system_prompt.replace("{{CALIBRATION_INSTRUCTION}}", cal_instruction)

    mcp_server = _make_merger_mcp_server(paper_dir, no_cal=no_cal)

    # Main merger only gets read_file/grep_file. Calibration retrieval is
    # delegated to the calibration_search subagent (invoked via Task) so its
    # many search/read tool results don't accumulate in the main merger's
    # context — only the subagent's short paper-list response does.
    allowed_tools = [
        "mcp__merger_fs__read_file",
        "mcp__merger_fs__grep_file",
    ]
    agents = None
    if not no_cal:
        allowed_tools.append("Task")
        agents = {"calibration_search": _make_calibration_subagent()}

    return await _run_claude_sdk_query(
        label="Merger",
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=merger_prompt,
        allowed_tools=allowed_tools,
        mcp_servers={"merger_fs": mcp_server},
        agents=agents,
        max_turns=30,
    )
