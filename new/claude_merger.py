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


CAL_INSTRUCTION_WITH = """Use comparative scoring to calibrate your final score. You have access to human reviews of other papers via the `calibration_search` subagent (invoked through the Task tool).

How to use the subagent:
- Send it a SHORT, focused retrieval request describing ONE attribute (a topic, a specific weakness, or a specific strength).
- For EACH request, the subagent will internally run THREE score-bin retrievals (low / medium / high) and return a combined anchor list covering the full score spectrum. You do not need to ask for score ranges yourself — the subagent handles that.
- The subagent returns paper paths each with a one-sentence summary + score bin. It does NOT do calibration reasoning — that is your job.
- After you receive each list, use your own read_file tool to read the FULL reviews of the most relevant anchors. Then judge how the paper under review compares.

You MUST issue exactly these THREE calibration_search calls (run them in parallel where possible):

1. **Topic** — dispatch a request naming the paper's topic area.
   Example: "topic: <topic of paper under review>"

2. **Weakness** — dispatch a request naming ONE specific weakness of the paper under review (ignore topic).
   Example: "weakness: <specific weakness from this paper>"

3. **Strength** — dispatch a request naming ONE specific strength of the paper under review (ignore topic).
   Example: "strength: <specific strength from this paper>"

Do NOT collapse multiple attributes into one call. Do NOT skip any of the three calls. Do NOT specify score ranges — the subagent does that itself.

After all three calls return:
- Read the FULL review of 1-2 most relevant anchors per bucket and per score bin (so roughly 6-9 read_file calls total on human review files).
- Position your final score relative to the retrieved examples. If similar-topic/weakness/strength HIGH-bin anchors got 7s from humans, and LOW-bin anchors got 3s, use that as your range. Do not compress everything into 4-6.

When reporting your score, briefly state which calibration papers you compared against (cite each of the three buckets) and why the paper under review is above or below them.

Let the score distribution follow the actual quality of the paper relative to the calibration examples. The samples could be concentrated in the middle, that does not mean you have to score it in the middle as well."""

CAL_INSTRUCTION_WITHOUT = """Assign a score based solely on your assessment of the paper's quality. Do NOT use the search or review finder tools for calibration — score directly from the paper's merits and weaknesses as identified in the review above."""


CALIBRATION_SUBAGENT_PROMPT = """You are a retrieval helper for the main merger agent. The main agent gives you ONE attribute (a topic, a specific weakness, or a specific strength) and you return anchor papers spanning LOW / MEDIUM / HIGH human-score bins for that attribute.

You have these tools (all under the mcp__merger_fs__ namespace):
- search_file(query, n, mode): BM25 or vector search over human reviews. mode='vector' (default) or 'bm25'.
- read_file(abs_path, start_line, end_line): read lines from a human review file.
- grep_file(pattern, abs_path): substring search inside a single file.

Workflow (run these three searches SEQUENTIALLY, one per score bin):

1. **LOW bin (~3 or below)**: Run search_file with a query combining the attribute with low-score signals (e.g. "<attribute> reject weak paper" or "<attribute> score 3 below"). Skim 1-2 top hits with read_file to confirm the human score is actually low; discard candidates that do not match the score bin.

2. **MEDIUM bin (~4-6)**: Run search_file with a query combining the attribute with medium/borderline signals (e.g. "<attribute> borderline poster" or "<attribute> score 5"). Skim 1-2 top hits with read_file to confirm.

3. **HIGH bin (~7 or above)**: Run search_file with a query combining the attribute with high-score signals (e.g. "<attribute> strong accept oral spotlight" or "<attribute> score 7 above"). Skim 1-2 top hits with read_file to confirm.

For each bin, return 2-3 anchor papers that actually land in that bin based on what you read.

Output format (strict):
### LOW (~3 or below)
- <abs_path>: <one-sentence reason + human score + decision>
- <abs_path>: ...

### MEDIUM (~4-6)
- <abs_path>: ...

### HIGH (~7 or above)
- <abs_path>: ...

Constraints:
- Return 2-3 anchors per bin (6-9 total).
- Do NOT produce a review, do NOT give calibration advice, do NOT compare the retrieved papers to the paper under review. The main agent handles all reasoning — you just retrieve.
- Keep the whole response under 400 words.
- Cap yourself at ~10 tool calls total (3 search_file + up to ~6 read_file for verification).
- If one bin has no solid match after searching, write "(none found)" for that bin rather than forcing a weak match.
- For the TOPIC attribute, treat the topic as LOOSELY RELATED, not exact-match, so the anchor set stays diverse.
"""


def _make_calibration_subagent():
    from claude_agent_sdk import AgentDefinition

    return AgentDefinition(
        description="Retrieval helper for calibration anchors. Accepts ONE attribute (topic, weakness, or strength) and internally runs three sequential score-bin retrievals (low/medium/high), returning 2-3 anchor papers per bin with one-sentence summaries. Does not do calibration reasoning.",
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
        "total_cost_usd": None,
        "num_turns": None,
        "duration_ms": None,
        "duration_api_ms": None,
        "usage": None,
    }
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(full_prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text
            elif isinstance(message, ResultMessage):
                sdk_usage["total_cost_usd"] = message.total_cost_usd
                sdk_usage["num_turns"] = message.num_turns
                sdk_usage["duration_ms"] = message.duration_ms
                sdk_usage["duration_api_ms"] = message.duration_api_ms
                sdk_usage["usage"] = message.usage

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
