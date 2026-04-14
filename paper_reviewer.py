from __future__ import annotations

"""
Multi-Agent Paper Reviewer using OpenRouter chat completions.

Usage:
  python paper_reviewer.py <paper.txt>                 # sequential (default)
  python paper_reviewer.py <paper.txt> --parallel      # parallel agents
"""

import json
import logging
import os
import sys
import asyncio
from pathlib import Path


from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from review_agents.openrouter_utils import (
    call_openai,
    extract_cost,
    get_client,
    ollama_client,
    resolve_openai_client_and_model,
)
from review_agents.paper_reviewer_helpers import (
    HARSH_CRITIC_PROMPT,
    MERGER_PROMPT,
    NEUTRAL_REVIEWER_PROMPT,
    RELATED_WORK_FILTER_PROMPT,
    RELATED_WORK_PROMPT,
    SCORE_PROMPT,
    SPARK_FINDER_PROMPT,
    build_merger_prompt,
    decision_match,
    detect_leakage_warning_phrases,
    load_prompt,
    match_label,
    sanitize_text,
    score_to_decision,
)

load_dotenv()  # loads .env from cwd or parent dirs

# ── Config ────────────────────────────────────────────────────────────
PROVIDER = "zai" 


#base_model = "qwen/qwen3.6-plus:free" #用限时免费模型白嫖
base_model = "qwen/qwen3.5-flash-02-23"
MODEL_HARSH = f"claude:claude-sonnet-4-6" #用claude subscription白嫖
# MODEL_NEUTRAL = "ollama:qwen3.5:397b-cloud"
# MODEL_SPARK = "ollama:qwen3.5:397b-cloud"
MODEL_NEUTRAL = "qwen/qwen3.5-plus-02-15"
MODEL_SPARK = "qwen/qwen3.5-plus-02-15"
MODEL_RELATED_WORK = f"{base_model}:online" 
MODEL_FILTER = f"{base_model}"
# MODEL_MERGER = f"zai:glm-5.1" #用zai coding plan白嫖
# MODEL_MERGER = f"ollama:glm-5:cloud" 
MODEL_HUMAN_FINDER = f"claude:claude-haiku-4-5"
MODEL_HUMAN_MERGER = f"claude:claude-sonnet-4-6"
MODEL_MERGER = f"z-ai/glm-5" 
MODEL_PARSER = "openai/gpt-5.4-nano"

DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "calibration.md"

# ── Error logging ────────────────────────────────────────────────────
_error_log_path = Path(__file__).parent / "error.log"
_error_logger = logging.getLogger("paper_reviewer.errors")
_error_logger.setLevel(logging.ERROR)
_error_handler = logging.FileHandler(_error_log_path, mode="a")
_error_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_error_logger.addHandler(_error_handler)

class ScoreSchema(BaseModel):
    score: float


def _make_search_mcp_server(search_path: str):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    abs_search_path = os.path.abspath(search_path)
    all_files: list[str] = []
    all_file_paths: list[str] = []
    for root, _dirs, files in os.walk(abs_search_path):
        for file_name in files:
            if file_name.endswith(".txt") or file_name.endswith(".md"):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", errors="replace") as file_handle:
                    all_files.append(file_handle.read())
                    all_file_paths.append(file_path)

    tokenized = [doc.split(" ") for doc in all_files if doc.strip()]
    if not tokenized:
        return None

    bm25 = BM25Okapi(tokenized)

    @tool(
        "search_file",
        "Search for a pattern in a directory using the BM25 index. Returns the top n matching files with their first 1000 chars.",
        {"query": str, "path": str, "n": int},
    )
    async def _search_file_tool(args: dict) -> dict:
        query = args["query"]
        path = os.path.abspath(args["path"])
        n = args.get("n", 5)
        if path != abs_search_path:
            print(f"  [search_file] WARNING: Attempt to search path '{path}' which is outside the indexed path '{abs_search_path}'")
            return {
                "content": [{"type": "text", "text": f"ERROR: Path '{path}' is not indexed. Expected: {abs_search_path}"}],
                "is_error": True,
            }
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = doc_scores.argsort()[-n:][::-1]
        results = []
        for idx in top_indices:
            file_path = os.path.abspath(all_file_paths[idx])
            score = doc_scores[idx]
            with open(file_path, "r", errors="replace") as file_handle:
                content = file_handle.read()
            results.append(f"{file_path}\nscore: {score:.2f}\n first 1000 chars:\n{content[:1000]}\n")
        text = "\n---\n".join(results) if results else "No relevant files found."
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(
        name="search",
        version="1.0.0",
        tools=[_search_file_tool],
    )


def _make_sandboxed_tools(allowed_paths: list[str]):
    from claude_agent_sdk import tool
    import glob as _glob_mod

    resolved_allowed = [os.path.abspath(path) for path in allowed_paths]

    def _check_path(path: str) -> str | None:
        resolved = os.path.abspath(path)
        if any(resolved.startswith(allowed_path) for allowed_path in resolved_allowed):
            return None
        return f"ERROR: Access denied. Path '{resolved}' is not under any allowed directory: {resolved_allowed}"

    @tool(
        "read_file",
        "Read a file. Returns the full content with line numbers. Restricted to allowed directories only.",
        {"abs_path": str, "start_line": int, "end_line": int},
    )
    async def _read_file(args: dict) -> dict:
        print(f"  [read_file] requested: {args['abs_path']} (lines {args.get('start_line', 1)}-{args.get('end_line', 'end')})")
        abs_path = args["abs_path"]
        start_line = args.get("start_line", 1) or 1
        end_line = args.get("end_line", 0) or 0
        err = _check_path(abs_path)
        if err:
            print(f"  [read_file] ERROR: {err}")
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        try:
            with open(abs_path, "r", errors="replace") as file_handle:
                lines = file_handle.readlines()
            selected = lines[max(0, start_line - 1):end_line if end_line > 0 else len(lines)]
            text = "".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))
            return {"content": [{"type": "text", "text": text}]}
        except FileNotFoundError:
            print(f"  [read_file] ERROR: File not found: {abs_path}")
            return {"content": [{"type": "text", "text": f"ERROR: File not found: {abs_path}"}], "is_error": True}

    @tool(
        "grep_files",
        "Search file contents for a regex pattern in a directory. Returns matching lines with file paths and line numbers. Restricted to allowed directories only.",
        {"pattern": str, "directory": str, "file_glob": str},
    )
    async def _grep_files(args: dict) -> dict:
        print(f"  [grep_files] requested: pattern='{args['pattern']}' in directory='{args.get('directory', '.')}' with glob='{args.get('file_glob', '**/*')}'")
        pattern = args["pattern"]
        directory = args.get("directory", ".")
        file_glob = args.get("file_glob", "**/*")
        err = _check_path(directory)
        if err:
            print(f"  [grep_files] ERROR: {err}")
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        matches = []
        files = sorted(_glob_mod.glob(file_glob, root_dir=directory, recursive=True))
        for filename in files[:500]:
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, "r", errors="replace") as file_handle:
                    for line_number, line in enumerate(file_handle, 1):
                        if re.search(pattern, line):
                            matches.append(f"{file_path}:{line_number}: {line.rstrip()}")
            except Exception:
                continue
            if len(matches) >= 200:
                break
        text = "\n".join(matches) if matches else "No matches found."
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "glob_files",
        "Find files matching a glob pattern under a directory. Returns one path per line. Restricted to allowed directories only.",
        {"pattern": str, "directory": str},
    )
    async def _glob_files(args: dict) -> dict:
        print(f"  [glob_files] requested: pattern='{args['pattern']}' in directory='{args.get('directory', '.')}'")
        pattern = args["pattern"]
        directory = args.get("directory", ".")
        err = _check_path(directory)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        matches = sorted(_glob_mod.glob(pattern, root_dir=directory, recursive=True))
        text = "\n".join(os.path.join(directory, match) for match in matches) if matches else "No files matched."
        return {"content": [{"type": "text", "text": text}]}

    return [_read_file, _grep_files, _glob_files]


def _make_sandboxed_mcp_server(name: str, allowed_paths: list[str]):
    from claude_agent_sdk import create_sdk_mcp_server

    return create_sdk_mcp_server(
        name=name,
        version="1.0.0",
        tools=_make_sandboxed_tools(allowed_paths),
    )

print("Testing ZAI client with a simple call ...")

# ans = asyncio.run(call_openai(get_client(), "test", "You are a helpful assistant.", "What is the capital of France?", "glm-5.1", _error_logger))
# if not "paris" in ans[0].lower():
#     print(ans[0])
#     print("🔥ZAI client test failed: unexpected answer")

# ── Agent runners ─────────────────────────────────────────────────────

async def _run_reviewer_claude_sdk(
    name: str,
    system_prompt: str,
    paper_path: str,
    model_id: str,
    venue: str = "",
) -> tuple[str, float]:
    """Run a reviewer via Claude Agent SDK. The agent reads the paper file itself.
    Returns (review, cost=0.0) — SDK does not expose cost."""
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

    paper_abs = str(Path(paper_path).resolve())
    paper_dir = str(Path(paper_abs).parent)
    venue_line = (
        f"This paper was submitted to **{venue}**. "
        f"You MUST evaluate it against {venue}'s specific standards, acceptance bar, "
        f"and expectations. Consider what {venue} reviewers typically look for.\n\n"
    ) if venue else ""

    prompt = (
        f"{system_prompt}\n\n"
        f"---\n\n"
        f"{venue_line}"
        f"Review the following paper thoroughly.\n\n"
        f"NOTE: This paper was extracted from PDF by an automated parser. "
        f"There may be formatting artifacts such as broken equations, garbled "
        f"tables, misplaced figure references, or OCR errors. These are parser "
        f"issues, NOT problems with the paper itself. Do NOT treat formatting "
        f"artifacts as weaknesses.\n\n"
        f"The paper is located at: {paper_abs}\n"
        f"Use the read_file tool to read the paper file, then produce your review."
    )

    print(f"  [{name}] starting Claude Agent SDK ({model_id}) ...")

    result_text = ""
    reviewer_fs = _make_sandboxed_mcp_server("reviewer_fs", [paper_dir])
    options = ClaudeAgentOptions(
        model=model_id,
        cwd=paper_dir,
        allowed_tools=[
            "mcp__reviewer_fs__read_file",
            "mcp__reviewer_fs__grep_files",
            "mcp__reviewer_fs__glob_files",
        ],
        permission_mode="bypassPermissions",
        disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Agent"],
        mcp_servers={"reviewer_fs": reviewer_fs},
        max_turns=30,
    )
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text
                        print(block.text)
            

    print(f"  [{name}] done — {model_id} (Claude Agent SDK)")
    return result_text, 0.0


async def run_reviewer(
    client: AsyncOpenAI,
    name: str,
    system_prompt: str,
    paper_path: str,
    paper_content: str,
    model: str,
    venue: str = "",
) -> tuple[str, float]:
    """Run a reviewer. Dispatches to Claude Agent SDK if model starts with 'claude:',
    otherwise uses OpenRouter chat completions. Returns (review, cost)."""
    if model.startswith("claude:"):
        model_id = model[len("claude:"):]
        return await _run_reviewer_claude_sdk(
            name, system_prompt, paper_path, model_id, venue=venue,
        )

    print(f"  [{name}] started ({model}) ...")
    venue_line = (
        f"This paper was submitted to **{venue}**. "
        f"You MUST evaluate it against {venue}'s specific standards, acceptance bar, "
        f"and expectations. Consider what {venue} reviewers typically look for.\n\n"
    ) if venue else ""
    user_prompt = (
        f"{venue_line}"
        f"Review the following paper thoroughly.\n\n"
        f"NOTE: This paper was extracted from PDF by an automated parser. "
        f"There may be formatting artifacts such as broken equations, garbled "
        f"tables, misplaced figure references, or OCR errors. These are parser "
        f"issues, NOT problems with the paper itself. Do NOT treat formatting "
        f"artifacts as weaknesses.\n\n"
        # f"Paper file: {paper_path}\n\n"
        f"--- PAPER CONTENT START ---\n"
        f"{paper_content}\n"
        f"--- PAPER CONTENT END ---"
    )
    return await call_openai(client, name, system_prompt, user_prompt, model, _error_logger)


async def run_related_work_search(
    client: AsyncOpenAI,
    paper_content: str,
) -> tuple[str, float]:
    """
    Two-step related work pipeline via OpenRouter. Returns (filtered_results, total_cost).
    """
    abstract_section = paper_content[:3000]

    print("  [related_work_search] started (OpenRouter online) ...")
    raw_results, cost1 = await call_openai(
        client,
        "related_work_search",
        RELATED_WORK_PROMPT,
        (
            f"Find related work for this paper. Here is the title and abstract:\n\n"
            f"{abstract_section}\n\n"
            f"Search for real, published papers that are closely related."
        ),
        MODEL_RELATED_WORK,
        _error_logger,
    )

    print("  [related_work_filter] started (OpenRouter) ...")
    filtered, cost2 = await call_openai(
        client,
        "related_work_filter",
        RELATED_WORK_FILTER_PROMPT,
        (
            f"Here is the FULL PAPER (extracted from PDF — ignore formatting artifacts). "
            f"Check references and citations carefully:\n\n"
            f"--- PAPER CONTENT START ---\n"
            f"{paper_content}\n"
            f"--- PAPER CONTENT END ---\n\n"
            f"Here are the related works found by the search agent:\n\n"
            f"{raw_results}\n\n"
            f"Filter out already-cited and loosely related works."
        ),
        MODEL_FILTER,
        _error_logger,
    )

    return filtered, cost1 + cost2



async def _parse_score(client: AsyncOpenAI, text: str) -> tuple[float, float]:
    """Extract a score using an OpenAI-compatible endpoint. Returns (score, cost)."""
    resolved_client, resolved_model, provider_name = resolve_openai_client_and_model(client, MODEL_PARSER)
    if provider_name == "Ollama":
        parser_prompt = (
            f"{load_prompt('parse_score.txt')}\n\n"
            f"Return valid JSON with exactly this shape: {{\"score\": 0.0}}"
        )
        parser_text, cost = await call_openai(
            resolved_client,
            "score_parser",
            parser_prompt,
            text,
            MODEL_PARSER,
            _error_logger,
        )
        parsed = ScoreSchema.model_validate(json.loads(parser_text))
        return parsed.score, cost

    response = await resolved_client.beta.chat.completions.parse(
        model=resolved_model,
        messages=[
            {"role": "system", "content": load_prompt("parse_score.txt")},
            {"role": "user", "content": text},
        ],
        response_format=ScoreSchema,
        timeout=30,
    )
    parsed = response.choices[0].message.parsed
    cost = 0.0 if provider_name == "Ollama" else extract_cost(response)
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    tokens = f"{input_tokens}in/{output_tokens}out" if input_tokens and output_tokens else "n/a"
    print(f"  [score_parser] done — {MODEL_PARSER} ({provider_name}) — {tokens} tokens — ${cost:.4f}")
    if parsed is None:
        raise ValueError("score_parser returned no parsed output")
    return parsed.score, cost


async def run_merge(
    client: AsyncOpenAI,
    harsh_review: str,
    neutral_review: str,
    spark_review: str,
    related_work: str,
    paper_path: str,
    skip_neutral: bool = False,
    skip_spark: bool = False,
    skip_related_work: bool = False,
    pred_score: bool = False,
) -> tuple[str, float] | tuple[str, float, float]:
    """
    Merger only — synthesize sub-agent reviews into a consolidated review.
    Uses Claude Agent SDK with sandboxed file tools so the model reads the
    paper directly from disk instead of receiving it in the prompt.
    Returns `(review_text, cost)` by default, or
    `(review_text, score, cost)` when `pred_score=True`.
    """
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

    print(f"  [merger] started (claude-sonnet-4-6 agent) ...")

    merger_system_prompt = build_merger_prompt(
        skip_neutral=skip_neutral,
        skip_spark=skip_spark,
        skip_related_work=skip_related_work,
    )

    if pred_score:
        with open("prompts/merger_score.txt", "r", encoding="utf-8") as f:
            merger_system_prompt += "\n\n" + f.read()

    review_num = 1
    reviews_section = f"# Review {review_num}: Harsh Critic\n{harsh_review}\n\n"
    if not skip_neutral:
        review_num += 1
        reviews_section += f"# Review {review_num}: Positive-Leaning Reviewer\n{neutral_review}\n\n"
    if not skip_spark:
        review_num += 1
        reviews_section += f"# Review {review_num}: Spark Finder\n{spark_review}\n\n"
    if not skip_related_work:
        review_num += 1
        reviews_section += (
            f"# Report {review_num}: Potentially Missed Related Work\n"
            f"(NOTE: These are SUGGESTIONS only. The search agent may have found \n"
            f"works that are not truly missed or are only tangentially related.)\n"
            f"{related_work}\n\n"
        )

    abs_paper_path = str(Path(paper_path).resolve())
    paper_dir = str(Path(abs_paper_path).parent)

    merger_agent_template = load_prompt("merger_agent.txt")
    agent_prompt = merger_agent_template.format(
        merger_system_prompt=merger_system_prompt,
        paper_path=abs_paper_path,
    )

    user_message = (
        f"Here are the sub-reviews to synthesize:\n\n"
        f"{reviews_section}"
        f"Now produce the final consolidated review following your instructions. "
        f"Remember: many of the harsh critic's points may be nonsensical or overly "
        f"picky — cross-check everything against the actual paper before including it."
    )

    merger_fs = _make_sandboxed_mcp_server("merger_fs", [paper_dir])

    options = ClaudeAgentOptions(
        model=MODEL_HUMAN_MERGER,
        allowed_tools=[
            "mcp__merger_fs__read_file",
            "mcp__merger_fs__glob_files",
            "mcp__scorer_fs__grep_files",
        ],
        permission_mode="bypassPermissions",
        disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write"],
        mcp_servers={"merger_fs": merger_fs},
        effort="medium",
        max_turns=15,
    )

    review_text = ""
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(f"{agent_prompt}\n\n{user_message}")
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        review_text += block.text
                        print(block.text)

    if not review_text.strip():
        raise ValueError("merger agent returned empty output")

    # Cost is not tracked for Claude SDK calls (subscription-based)
    cost = 0.0

    if not pred_score:
        return review_text, cost
    else:
        score, parser_cost = await _parse_score(client, review_text)
        return review_text, score, cost + parser_cost


async def run_scorer(
    client: AsyncOpenAI,
    review_text: str,
    paper_content: str,
    calibration_context: str = "",
    cal_dir: str = "",
    gt_score: float | None = None,
) -> tuple[float, float]:
    """
    Scorer — uses Claude Agent SDK (claude-sonnet-4-6) to search calibration
    examples via Grep/Read, then scores the paper. Returns (score, cost).
    """
    import tempfile
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

    cal_dir_abs = str(Path(cal_dir).resolve()) if cal_dir else ""
    search_mcp = _make_search_mcp_server(cal_dir_abs) if cal_dir_abs else None

    # Write review and paper to temp files so the agent can Read them
    # (avoids CLI character limit on the prompt)
    tmp_dir = Path(tempfile.mkdtemp(prefix="scorer_"))
    review_path = tmp_dir / "review.txt"
    paper_path = tmp_dir / "paper.txt"
    review_path.write_text(review_text, encoding="utf-8")
    paper_path.write_text(paper_content, encoding="utf-8")
    scorer_fs = _make_sandboxed_mcp_server(
        "scorer_fs",
        [cal_dir_abs, str(tmp_dir)] if cal_dir_abs else [str(tmp_dir)],
    )

    scorer_agent_template = load_prompt("scorer_agent.txt")
    prompt = scorer_agent_template.format(
        score_prompt=SCORE_PROMPT,
        review_path=review_path,
        paper_path=paper_path,
        cal_dir_abs=cal_dir_abs,
    )

    print(f"  [scorer-agent] starting RAG scorer (claude-sonnet-4-6, cal={cal_dir_abs}) ...")

    result_text = ""
    options = ClaudeAgentOptions(
        model=MODEL_HUMAN_MERGER.split(":")[1],
        cwd=cal_dir_abs or None,
        allowed_tools=[
            "mcp__scorer_fs__read_file",
            "mcp__scorer_fs__grep_files",
            "mcp__scorer_fs__glob_files",
            "Agent",
        ] + (["mcp__search__search_file"] if search_mcp else []),
        permission_mode="bypassPermissions",
        disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write"],
        mcp_servers={
            "scorer_fs": scorer_fs,
            **({"search": search_mcp} if search_mcp else {}),
        },
        effort="medium",
        max_turns=30,
    )
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text

    # Clean up temp files
    review_path.unlink(missing_ok=True)
    paper_path.unlink(missing_ok=True)
    tmp_dir.rmdir()


    # Log full scorer output to file for debugging
    scorer_log_path = Path(__file__).parent / "scorer_debug.log"
    with open(scorer_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 72}\n")
        f.write(f"cal_dir: {cal_dir_abs}\n")
        f.write(f"{'─' * 72}\n")
        f.write(result_text)
        f.write(f"\nGT Score: {gt_score}\n")
        f.write(f"\n{'=' * 72}\n\n")

    leakage_matches = detect_leakage_warning_phrases(result_text)
    if leakage_matches:
        matched_text = ", ".join(sorted(set(leakage_matches), key=str.lower))
        warning_msg = (
            f"🚨 Potential calibration leakage warning: scorer output contains "
            f"suspicious phrase(s): {matched_text}"
        )
        print(f"  [scorer-agent] WARNING: {warning_msg}")
        _error_logger.error(warning_msg)
        raise ValueError(warning_msg)

    # Use _parse_score to extract the numerical score
    score, cost_parse = await _parse_score(client, result_text)
    print(f"  [scorer-agent] parsed score: {score}")
    return score, cost_parse


async def run_merger(
    client: AsyncOpenAI,
    harsh_review: str,
    neutral_review: str,
    spark_review: str,
    related_work: str,
    paper_path: str,
    paper_content: str,
    calibration_context: str = "",
    cal_dir: str = "",
    skip_neutral: bool = False,
    skip_spark: bool = False,
    skip_related_work: bool = False,
    gt_score: float | None = None,
) -> tuple[str, float, float]:
    """
    Merger + Scorer (two separate calls).
    Returns (review_text, score, total_cost).
    """
    review_text, cost_merge = await run_merge(
        client, harsh_review, neutral_review,
        spark_review, related_work, paper_path,
        skip_neutral=skip_neutral,
        skip_spark=skip_spark,
        skip_related_work=skip_related_work,
    )

    score, cost_score = await run_scorer(
        client, review_text, paper_content,
        calibration_context=calibration_context,
        cal_dir=cal_dir,
        gt_score=gt_score
    )
    return review_text, score, cost_merge + cost_score


async def run_pipeline(
    paper_path: str,
    paper_content: str,
    client: AsyncOpenAI,
    parallel: bool = True,
    skip_related_work: bool = True,
    skip_spark: bool = False,
    skip_neutral: bool = False,
    skip_score: bool = False,
    venue: str = "ICLR",
    calibration_context: str = "",
    cal_dir: str = "",
    gt_score: float | None = None,
    merger_output_score: bool = False,
) -> dict:
    """Compatibility wrapper used by calibration and benchmark scripts."""
    pp = str(Path(paper_path).expanduser().resolve())
    cleaned_paper_content = sanitize_text(paper_content)

    total_cost = 0.0
    if parallel:
        if MODEL_HARSH.startswith("ollama:"):
            tasks = [
                run_reviewer(ollama_client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, cleaned_paper_content, MODEL_HARSH.replace("ollama:", "", 1), venue=venue),
            ]
        else:
            tasks = [
                run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, cleaned_paper_content, MODEL_HARSH, venue=venue),
            ]
        if not skip_neutral:
            if MODEL_NEUTRAL.startswith("ollama:"):
                tasks.append(run_reviewer(ollama_client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, cleaned_paper_content, MODEL_NEUTRAL.replace("ollama:", "", 1), venue=venue))
            else:
                tasks.append(run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, cleaned_paper_content, MODEL_NEUTRAL, venue=venue))
        if not skip_spark:
            if MODEL_SPARK.startswith("ollama:"):
                tasks.append(run_reviewer(ollama_client, "spark_finder", SPARK_FINDER_PROMPT, pp, cleaned_paper_content, MODEL_SPARK.replace("ollama:", "", 1), venue=venue))
            else:
                tasks.append(run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, cleaned_paper_content, MODEL_SPARK, venue=venue))
        if not skip_related_work:
            tasks.append(run_related_work_search(client, cleaned_paper_content))

        
        results_list = await asyncio.gather(*tasks)
        idx = 0
        harsh_review, c = results_list[idx]; total_cost += c; idx += 1
        if not skip_neutral:
            neutral_review, c = results_list[idx]; total_cost += c; idx += 1
        else:
            neutral_review = "Neutral reviewer was skipped."
        if not skip_spark:
            spark_review, c = results_list[idx]; total_cost += c; idx += 1
        else:
            spark_review = "Spark finder was skipped."
        if not skip_related_work:
            related_work, c = results_list[idx]; total_cost += c
        else:
            related_work = "Related work search was skipped."
    else:
        if MODEL_HARSH.startswith("ollama:"):
            harsh_review, c = await run_reviewer(ollama_client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, cleaned_paper_content, MODEL_HARSH.replace("ollama:", "", 1), venue=venue)
        else:
            harsh_review, c = await run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, cleaned_paper_content, MODEL_HARSH, venue=venue)
        total_cost += c
        if not skip_neutral:
            if MODEL_NEUTRAL.startswith("ollama:"):
                neutral_review, c = await run_reviewer(ollama_client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, cleaned_paper_content, MODEL_NEUTRAL.replace("ollama:", "", 1), venue=venue)
            else:
                neutral_review, c = await run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, cleaned_paper_content, MODEL_NEUTRAL, venue=venue)
            total_cost += c
        else:
            neutral_review = "Neutral reviewer was skipped."
        if not skip_spark:
            if MODEL_SPARK.startswith("ollama:"):
                spark_review, c = await run_reviewer(ollama_client, "spark_finder", SPARK_FINDER_PROMPT, pp, cleaned_paper_content, MODEL_SPARK.replace("ollama:", "", 1), venue=venue)
            else:
                spark_review, c = await run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, cleaned_paper_content, MODEL_SPARK, venue=venue)
            total_cost += c
        else:
            spark_review = "Spark finder was skipped."
        if not skip_related_work:
            related_work, c = await run_related_work_search(client, cleaned_paper_content)
            total_cost += c
        else:
            related_work = "Related work search was skipped."

    if skip_score:
        merged_review, merge_cost = await run_merge(
            client,
            harsh_review,
            neutral_review,
            spark_review,
            related_work,
            pp,
            skip_neutral=skip_neutral,
            skip_spark=skip_spark,
            skip_related_work=skip_related_work,
        )
        total_cost += merge_cost
        score = None
        decision = None
    elif merger_output_score:
        merged_review, score, merge_cost = await run_merge(
            client,
            harsh_review,
            neutral_review,
            spark_review,
            related_work,
            pp,
            skip_neutral=skip_neutral,
            skip_spark=skip_spark,
            skip_related_work=skip_related_work,
            pred_score=True,
        )
        total_cost += merge_cost
        score = round(float(score), 1)
        decision = score_to_decision(score)
    else:
        merged_review, score, merge_cost = await run_merger(
            client,
            harsh_review,
            neutral_review,
            spark_review,
            related_work,
            paper_path=pp,
            paper_content=cleaned_paper_content,
            calibration_context=calibration_context,
            cal_dir=cal_dir,
            skip_neutral=skip_neutral,
            skip_spark=skip_spark,
            skip_related_work=skip_related_work,
            gt_score=gt_score,
        )
        total_cost += merge_cost
        score = round(float(score), 1)
        decision = score_to_decision(score)

    return {
        "harsh_review": harsh_review,
        "neutral_review": neutral_review,
        "spark_review": spark_review,
        "related_work": related_work,
        "merged_review": merged_review,
        "score": score,
        "decision": decision,
        "cost": total_cost,
        "sdk_savings": 0.0,
    }


# ── Main orchestration ────────────────────────────────────────────────

def _resolve_calibration_inputs(
    calibration_context: str = "",
    cal_dir: str = "",
    calibration_path: str | None = None,
) -> tuple[str, str]:
    """
    Resolve calibration inputs the same way as the benchmark runner.
    Prefer a sibling cal/ directory (RAG mode); otherwise fall back to the
    calibration markdown file content.
    """
    if cal_dir:
        return calibration_context, cal_dir
    if calibration_context:
        return calibration_context, cal_dir

    resolved_path: Path | None = None
    if calibration_path:
        resolved_path = Path(calibration_path).expanduser().resolve()
    elif DEFAULT_CALIBRATION_PATH.exists():
        resolved_path = DEFAULT_CALIBRATION_PATH.resolve()

    if resolved_path is None:
        return calibration_context, cal_dir

    cal_dir_candidate = resolved_path.parent / "cal"
    if cal_dir_candidate.is_dir():
        return "", str(cal_dir_candidate)
    if resolved_path.exists():
        return resolved_path.read_text(encoding="utf-8", errors="replace"), ""
    return calibration_context, cal_dir

async def review_paper(
    paper_path: str,
    parallel: bool = True,
    skip_related_work: bool = True,
    skip_spark: bool = False,
    skip_neutral: bool = False,
    venue: str = "ICLR",
    calibration_context: str = "",
    cal_dir: str = "",
    calibration_path: str | None = None,
    api_key: str | None = None,
    merger_output_score: bool = False,
) -> tuple[str, float]:
    """
    Main entry point. All agents via OpenRouter chat completions — can fully parallelize.

    Phase 1 (parallel if --parallel):
      Critic + Neutral + Spark + Related Work — all at once

    Phase 2:
      Merger — waits for all reviewers
    """

    path = Path(paper_path).expanduser().resolve()
    print("Reviewing:", paper_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper not found: {path}")

    paper_content = path.read_text(encoding="utf-8", errors="replace")
    paper_content = sanitize_text(paper_content)
    calibration_context, cal_dir = _resolve_calibration_inputs(
        calibration_context=calibration_context,
        cal_dir=cal_dir,
        calibration_path=calibration_path,
    )
    print(f"Loaded paper: {path.name} ({len(paper_content):,} chars)")
    print(f"Mode: {'parallel' if parallel else 'sequential'}")
    print(f"Related work: {'disabled' if skip_related_work else 'enabled'}")
    print(f"Spark finder: {'disabled' if skip_spark else 'enabled'}")
    print(f"Neutral reviewer: {'disabled' if skip_neutral else 'enabled'}")
    if venue:
        print(f"Venue: {venue}")
    print(f"Models:")
    print(f"  Harsh Critic:   {MODEL_HARSH}")
    if not skip_neutral:
        print(f"  Neutral:        {MODEL_NEUTRAL}")
    if not skip_spark:
        print(f"  Spark Finder:   {MODEL_SPARK}")
    if not skip_related_work:
        print(f"  Related Work:   {MODEL_RELATED_WORK}")
    print(f"  Merger:         {MODEL_MERGER}")
    print(f"  Scorer:         claude-sonnet-4-6 (Agent SDK)\n")
    print(f"Score source: {'merger output' if merger_output_score else 'scorer'}")

    client = get_client(api_key=api_key)
    pp = str(path)

    print("Phase 1 + 2: Delegating to run_pipeline ...")
    pipeline_result = await run_pipeline(
        paper_path=pp,
        paper_content=paper_content,
        client=client,
        parallel=parallel,
        skip_related_work=skip_related_work,
        skip_spark=skip_spark,
        skip_neutral=skip_neutral,
        venue=venue,
        calibration_context=calibration_context,
        cal_dir=cal_dir,
        merger_output_score=merger_output_score,
    )
    harsh_review = pipeline_result["harsh_review"]
    neutral_review = pipeline_result["neutral_review"]
    spark_review = pipeline_result["spark_review"]
    related_work = pipeline_result["related_work"]
    final_review = pipeline_result["merged_review"]
    final_score = pipeline_result["score"]
    total_cost = pipeline_result["cost"]
    print(f"Total cost for this paper: ${total_cost:.4f}")
    final_decision = score_to_decision(final_score)

    # ── Output ────────────────────────────────────────────────────
    separator = "=" * 72
    full_output = (
        f"\n{separator}\n"
        f"INDIVIDUAL REVIEWS\n"
        f"{separator}\n\n"
        f"{'─' * 40}\n"
        f"HARSH CRITIC ({MODEL_HARSH} via {'Ollama' if MODEL_HARSH.startswith('ollama:') else 'Claude Agent SDK' if MODEL_HARSH.startswith('claude:') else 'OpenRouter'})\n"
        f"{'─' * 40}\n"
        f"{harsh_review}\n\n"
        f"{'─' * 40}\n"
        f"NEUTRAL REVIEWER ({MODEL_NEUTRAL} via {'Ollama' if MODEL_NEUTRAL.startswith('ollama:') else 'Claude Agent SDK' if MODEL_NEUTRAL.startswith('claude:') else 'OpenRouter'})\n"
        f"{'─' * 40}\n"
        f"{neutral_review}\n\n"
        f"{'─' * 40}\n"
        f"SPARK FINDER ({MODEL_SPARK} via {'Ollama' if MODEL_SPARK.startswith('ollama:') else 'Claude Agent SDK' if MODEL_SPARK.startswith('claude:') else 'OpenRouter'})\n"
        f"{'─' * 40}\n"
        f"{spark_review}\n\n"
        f"{'─' * 40}\n"
        f"POTENTIALLY MISSED RELATED WORK ({MODEL_RELATED_WORK} via OpenRouter)\n"
        f"{'─' * 40}\n"
        f"{related_work}\n\n"
        f"{separator}\n"
        f"FINAL CONSOLIDATED REVIEW ({MODEL_MERGER} via OpenRouter)\n"
        f"{separator}\n\n"
        f"{final_review}\n\n"
        f"{separator}\n"
        f"PREDICTED SCORE\n"
        f"{separator}\n\n"
        f"Score: {final_score}\n"
        f"Decision: {final_decision or 'N/A'}\n"
        f"Total Cost: ${total_cost:.4f}\n"
    )

    output_path = path.parent / f"{path.stem}_review.md"
    output_path.write_text(full_output, encoding="utf-8")
    print(f"\nReview saved to: {output_path}")

    return full_output, total_cost


async def review_paper_text(
    paper_text: str,
    source_name: str = "paper.txt",
    parallel: bool = True,
    skip_related_work: bool = True,
    skip_spark: bool = False,
    skip_neutral: bool = False,
    venue: str = "ICLR",
    calibration_context: str = "",
    cal_dir: str = "",
    calibration_path: str | None = None,
    api_key: str | None = None,
    output_dir: str | None = None,
    merger_output_score: bool = False,
) -> tuple[str, str]:
    """Review paper content provided directly as text."""
    cleaned_text = sanitize_text(paper_text)
    if not cleaned_text.strip():
        raise ValueError("Paper content is empty.")

    target_dir = Path(output_dir or "webui_runs").expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(source_name).name or "paper.txt"
    if not Path(safe_name).suffix:
        safe_name = f"{safe_name}.txt"

    input_path = target_dir / safe_name
    input_path.write_text(cleaned_text, encoding="utf-8")

    result, total_cost = await review_paper(
        str(input_path),
        parallel=parallel,
        skip_related_work=skip_related_work,
        skip_spark=skip_spark,
        skip_neutral=skip_neutral,
        venue=venue,
        calibration_context=calibration_context,
        cal_dir=cal_dir,
        calibration_path=calibration_path,
        api_key=api_key,
        merger_output_score=merger_output_score,
    )
    return result, str(input_path.with_name(f"{input_path.stem}_review.md"))


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python paper_reviewer.py <paper.txt> [options]")
        print()
        print("Flags:")
        print("  --sequential        Run agents sequentially")
        print("  --with-related-work Enable related work search & filter")
        print("  --no-spark          Skip spark finder agent")
        print("  --no-neutral        Skip neutral reviewer agent")
        print("  --merger-output-score Use merger predicted score instead of scorer")
        print("  --venue <name>      Set venue (e.g. ICLR, NeurIPS, ICML)")
        print("  --calibration <p>   Calibration file/path (default: calibration.md if present)")
        print()
        print("Environment variables (or set in .env):")
        print("  OPENROUTER_API_KEY   (required) Your OpenRouter API key")
        print()
        print("Models per stage:")
        print(f"  Harsh Critic ({'Ollama' if MODEL_HARSH.startswith('ollama:') else 'Claude Agent SDK' if MODEL_HARSH.startswith('claude:') else 'OpenRouter'}):      {MODEL_HARSH}")
        print(f"  Neutral ({'Ollama' if MODEL_NEUTRAL.startswith('ollama:') else 'Claude Agent SDK' if MODEL_NEUTRAL.startswith('claude:') else 'OpenRouter'}):           {MODEL_NEUTRAL}")
        print(f"  Spark Finder ({'Ollama' if MODEL_SPARK.startswith('ollama:') else 'Claude Agent SDK' if MODEL_SPARK.startswith('claude:') else 'OpenRouter'}):      {MODEL_SPARK}")
        print(f"  Related Work (OpenRouter):      {MODEL_RELATED_WORK}")
        print(f"  Merger (OpenRouter):            {MODEL_MERGER}")
        print(f"  Scorer:                         claude-sonnet-4-6 (Agent SDK)")
        sys.exit(0 if "--help" in sys.argv else 1)

    parallel = "--sequential" not in sys.argv
    skip_related = "--with-related-work" not in sys.argv
    skip_spark = "--no-spark" in sys.argv
    skip_neutral = "--no-neutral" in sys.argv
    merger_output_score = "--merger-output-score" in sys.argv
    venue = "ICLR"
    calibration_path = None
    if "--venue" in sys.argv:
        idx = sys.argv.index("--venue")
        if idx + 1 < len(sys.argv):
            venue = sys.argv[idx + 1]
    if "--calibration" in sys.argv:
        idx = sys.argv.index("--calibration")
        if idx + 1 < len(sys.argv):
            calibration_path = sys.argv[idx + 1]
    flag_values = {venue, calibration_path} - {None}
    paper_file = [a for a in sys.argv[1:] if not a.startswith("--") and a not in flag_values][0]
    result, total_cost = asyncio.run(
        review_paper(
            paper_file,
            parallel=parallel,
            skip_related_work=skip_related,
            skip_spark=skip_spark,
            skip_neutral=skip_neutral,
            venue=venue,
            calibration_path=calibration_path,
            merger_output_score=merger_output_score,
        )
    )
    print(result)
