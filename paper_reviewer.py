from __future__ import annotations

""" 
Multi-Agent Paper Reviewer using OpenRouter chat completions.

Usage:
  python paper_reviewer.py <paper.txt>                 # sequential (default)
  python paper_reviewer.py <paper.txt> --parallel      # parallel agents
"""

import asyncio
import json
import logging
import os
import random as _random
import re
import sys
import traceback
from pathlib import Path


from dotenv import load_dotenv 
from openai import APITimeoutError, AsyncOpenAI
from pydantic import BaseModel
from review_agents.claude_backend import (
    make_claude_file_mcp_server,
    run_claude_agent_task,
    run_claude_reviewer,
)
from review_agents.openai_backend import (
    build_openai_file_tools,
    run_openai_agent_task,
    run_openai_reviewer,
)

load_dotenv()  # loads .env from cwd or parent dirs

# ── Config ────────────────────────────────────────────────────────────
PROVIDER = "zai" 


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4/"

base_model = "qwen/qwen3.5-flash-02-23" 
# Reviewer models can use one of three backends:
# - plain OpenRouter chat model id, e.g. "deepseek/deepseek-v3.2"
# - Claude Agent SDK, e.g. "claude:claude-sonnet-4-5"
# - OpenAI Agent SDK via OpenRouter, e.g. "openai_agent:deepseek/deepseek-v3.2"
MODEL_HARSH = f"qwen/qwen3.6-plus" 
MODEL_NEUTRAL = f"{base_model}"
MODEL_SPARK = f"qwen/qwen3.6-plus" 
MODEL_RELATED_WORK = f"{base_model}:online" 
MODEL_FILTER = f"{base_model}"
# MODEL_MERGER = f"zai:glm-5.1" #用zai coding plan白嫖
MODEL_MERGER = "gpt-5.4"
MODEL_PARSER = "openai/gpt-5.4-nano" 
MODEL_FIND_HUMAN = "openai_agent:minimax-m2.7"
MODEL_SCORER = "openai_agent:gpt-5.4"

human_review_dir = "/home/wg25r/review_agent/iclr2025_data"
MODEL_QA = "minimax-m2.7"

MAX_RETRIES = 10
RETRY_DELAY = 10 
REQUEST_TIMEOUT = 120
DEFAULT_CALIBRATION_PATH = Path(__file__).parent / "calibration.md"

# ── Error logging ────────────────────────────────────────────────────
_error_log_path = Path(__file__).parent / "error.log"
_error_logger = logging.getLogger("paper_reviewer.errors")
_error_logger.setLevel(logging.ERROR)
_error_handler = logging.FileHandler(_error_log_path, mode="a")
_error_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_error_logger.addHandler(_error_handler)

LEAKAGE_WARNING_PATTERNS = [
    r"\bsame paper\b",
    r"\bexact same paper\b",
    r"\bthis exact paper\b",
    r"\bcontains this exact paper\b",
    r"\bthe exact same paper\b",
    r"\bcalibration copy\b",
]

_total_sdk_savings: float = 0.0

def _add_sdk_savings(amount: float) -> None:
    global _total_sdk_savings
    _total_sdk_savings += amount

def get_sdk_savings() -> float:
    return _total_sdk_savings

def reset_sdk_savings() -> None:
    global _total_sdk_savings
    _total_sdk_savings = 0.0


class ScoreSchema(BaseModel):
    score: float


def score_to_decision(score: float | None) -> str | None:
    return "N/A"


def decision_match(predicted: str | None, gt_binary: str) -> bool | None:
    if predicted in (None, "", "N/A"):
        return None
    return predicted == gt_binary


def match_label(match: bool | None) -> str:
    if match is None:
        return "N/A"
    return "YES" if match else "NO"


def _detect_leakage_warning_phrases(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in LEAKAGE_WARNING_PATTERNS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            matches.append(found.group(0))
    return matches


def _extract_final_review_block(text: str) -> str | None:
    match = re.search(r"<final_review>\s*(.*?)\s*</final_review>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    return content

# ── Prompt loading ────────────────────────────────────────────────────
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts/ directory."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# ── Agent system prompts ──────────────────────────────────────────────
import time
timeline_prompt = "\n" + _load_prompt("timeline.txt").replace("{{CURRENT_DATE}}", time.strftime("%B %d, %Y"))

HARSH_CRITIC_PROMPT = _load_prompt("harsh_critic.txt") + timeline_prompt
NEUTRAL_REVIEWER_PROMPT = _load_prompt("neutral_reviewer.txt") + timeline_prompt
SPARK_FINDER_PROMPT = _load_prompt("spark_finder.txt") + timeline_prompt
RELATED_WORK_PROMPT = _load_prompt("related_work.txt") + timeline_prompt
RELATED_WORK_FILTER_PROMPT = _load_prompt("related_work_filter.txt") + timeline_prompt
_MERGER_PROMPT_TEMPLATE = _load_prompt("merger.txt") + timeline_prompt
HUMAN_FINDER_PROMPT = _load_prompt("find_human_match.txt") + timeline_prompt



def _build_merger_prompt(skip_neutral: bool = False, skip_spark: bool = False, skip_related_work: bool = False) -> str:
    num = 1
    neutral_line = ""
    spark_line = ""
    related_work_line = ""
    if not skip_neutral:
        num += 1
        neutral_line = f"{num}. A **neutral/balanced** review\n"
    if not skip_spark:
        num += 1
        spark_line = f"{num}. A **spark finder** report (focuses on insights, not flaws)\n"
    if not skip_related_work:
        num += 1
        related_work_line = (
            f"{num}. A **potentially missed related work** report (these are SUGGESTIONS, not "
            f"definitive omissions — the authors may have good reasons for not citing them)\n"
        )
    return _MERGER_PROMPT_TEMPLATE.format(
        input_count=num,
        neutral_line=neutral_line,
        spark_line=spark_line,
        related_work_line=related_work_line,
    )


# Default for backward compat
MERGER_PROMPT = _build_merger_prompt()


# ── Core logic ────────────────────────────────────────────────────────

def sanitize_text(text: str) -> str:
    """Remove null bytes and other problematic characters from text."""
    return text.replace("\x00", "")


def _get_client(api_key: str | None = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at OpenRouter."""
    resolved_api_key = api_key or OPENROUTER_API_KEY
    if not resolved_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set.\n"
            "Set it in .env or export it."
        )
    return AsyncOpenAI(api_key=resolved_api_key, base_url=OPENROUTER_BASE_URL)



def _get_zai_client(api_key: str | None = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at OpenRouter."""
    resolved_api_key = api_key
    if not resolved_api_key:
        raise ValueError(
            "ZAI_API_KEY environment variable not set.\n"
            "Set it in .env or export it."
        )
    return AsyncOpenAI(api_key=resolved_api_key, base_url=ZAI_BASE_URL)

zai_client = _get_zai_client(os.environ["ZAI_API_KEY"])



# ── OpenRouter calls ───────────────────────────────────────────────────

# Models that support OpenRouter reasoning config
REASONING_MODELS = {"z-ai/glm-5", "minimax/minimax-m2.7", "deepseek/deepseek-v3.2", "minimax/minimax-m2.5:free", "stepfun/step-3.5-flash:free"}

# Model → official provider mapping (for OpenRouter provider pinning)
PROVIDER_MAP = {
    "z-ai/glm-5": ["deepinfra/fp4"],
    "z-ai/glm-5:online": ["deepinfra/fp4"],
    "minimax/minimax-m2.7": ["minimax/fp8"],
    "deepseek/deepseek-v3.2": ["parasail/fp8"],
}


def _build_extra_body(model: str, reasoning_effort: str = "high") -> dict | None:
    """Build extra_body with reasoning and/or provider config for OpenRouter."""
    extra = {}
    if model in REASONING_MODELS:
        extra["reasoning"] = {"effort": reasoning_effort}
    if model in PROVIDER_MAP:
        extra["provider"] = {"only": PROVIDER_MAP[model]}
    return extra or None


def _extract_cost(response) -> float:
    """Extract cost from OpenRouter response usage object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        raise ValueError("Response has no usage object — cannot extract cost")
    cost = getattr(usage, "cost", None)
    if cost is not None:
        return float(cost)
    if isinstance(usage, dict) and "cost" in usage:
        return float(usage["cost"])
    raise ValueError(f"Usage object has no cost field: {usage}")


async def _call_openai(
    client: AsyncOpenAI,
    name: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    tools: list[dict] | None = None,
) -> tuple[str, float]:
    """Call OpenRouter chat completions with retry logic. Returns (result, cost)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=REQUEST_TIMEOUT,
            )
            extra = _build_extra_body(model, reasoning_effort="medium")
            if extra:
                kwargs["extra_body"] = extra
            response = await client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content
            cost = _extract_cost(response)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            output_tokens = getattr(usage, "completion_tokens", None) if usage else None
            if input_tokens is not None and output_tokens is not None:
                tokens = f"{input_tokens}in/{output_tokens}out"
            else:
                tokens = "n/a"
            if not result.strip():
                if attempt < MAX_RETRIES:
                    _error_logger.error(f"[{name}] empty response (attempt {attempt}/{MAX_RETRIES}), model={model}")
                    print(f"  [{name}] empty response (attempt {attempt}/{MAX_RETRIES}), retrying ...")
                    await asyncio.sleep(RETRY_DELAY + _random.uniform(0, 5))
                    continue
                raise RuntimeError(f"[{name}] empty response after {MAX_RETRIES} attempts, model={model}")
            print(f"  [{name}] done — {model} (OpenRouter) — {tokens} tokens — ${cost:.4f}")
            return result, cost
        except APITimeoutError as e:
            _error_logger.error(f"[{name}] timeout (attempt {attempt}/{MAX_RETRIES}), model={model}\n{traceback.format_exc()}")
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{name}] timeout (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ...")
                await asyncio.sleep(wait)
                continue
            raise
        except Exception as e:
            _error_logger.error(f"[{name}] error (attempt {attempt}/{MAX_RETRIES}), model={model}: {e}\n{traceback.format_exc()}")
            err_str = str(e).lower()
            is_retryable = any(
                kw in err_str for kw in ["rate_limit", "overloaded", "429", "529", "timeout", "gateway", "502", "503", "504"]
            )
            if is_retryable and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{name}] transient error (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ...", e)
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(f"[{name}] failed after {MAX_RETRIES} attempts, model={model}")

print("Testing ZAI client with a simple call ...")

# ans = asyncio.run(_call_openai(zai_client, "test", "You are a helpful assistant.", "What is the capital of France?", "glm-5.1"))
# if not "paris" in ans[0].lower():
#     print(ans[0])
#     print("🔥ZAI client test failed: unexpected answer")

# ── Agent runners ─────────────────────────────────────────────────────
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage, tool, create_sdk_mcp_server

# ── BM25 index for human review search ──────────────────────────────
from rank_bm25 import BM25Okapi

_SEARCH_PATHS = [os.path.abspath(human_review_dir)]
_bm25_database: dict = {}

print("Indexing human reviews ...")
_idx_start = time.time()
for _sp in _SEARCH_PATHS:
    _all_files = []
    _all_file_paths = []
    for _root, _dirs, _files in os.walk(_sp):
        for _f in _files:
            if _f.endswith(".txt") or _f.endswith(".md"):
                with open(os.path.join(_root, _f), "r", errors="replace") as _fh:
                    _all_files.append(_fh.read())
                    _all_file_paths.append(os.path.join(_root, _f))
    _tokenized = [doc.split(" ") for doc in _all_files if doc.strip()]
    if not _tokenized:
        print(f"  Skipping {_sp} (no files found)")
        continue
    _bm25_database[_sp] = {"files": _all_file_paths, "bm25": BM25Okapi(_tokenized)}
print(f"Indexing complete. Time taken: {time.time() - _idx_start:.2f}s")


@tool(
    "search_file",
    "Search for a pattern in a directory using the BM25 index. Returns the top n matching files with their first 1000 chars.",
    {"query": str, "path": str, "n": int},
)
async def _search_file_tool(args: dict) -> dict:
    query = args["query"]
    path = os.path.abspath(args["path"])
    n = args.get("n", 5)
    if path not in _bm25_database:
        return {"content": [{"type": "text", "text": f"ERROR: Path '{path}' is not indexed or allowed for searching."}], "is_error": True}
    bm25 = _bm25_database[path]["bm25"]
    files = _bm25_database[path]["files"]
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = doc_scores.argsort()[-n:][::-1]
    results = []
    for idx in top_indices:
        file_path = os.path.abspath(files[idx])
        score = doc_scores[idx]
        with open(file_path, "r", errors="replace") as f:
            content = f.read()
        results.append(f"{file_path}\nscore: {score:.2f}\n first 1000 chars:\n{content[:1000]}\n")
    text = "\n---\n".join(results) if results else "No relevant files found."
    return {"content": [{"type": "text", "text": text}]}


_search_mcp_server = create_sdk_mcp_server(
    name="search",
    version="1.0.0",
    tools=[_search_file_tool],
)


# ── Sandboxed file tools (path-restricted) ───────────────────────────
# These replace built-in Read/Grep/Glob for Claude SDK agents to prevent
# agents from reading files outside their allowed directories.

def _make_sandboxed_tools(allowed_paths: list[str]):
    """Create path-restricted read_file, grep_files, glob_files MCP tools.

    Each tool checks that the resolved path starts with one of the allowed
    directories before performing any I/O.
    """
    import glob as _glob_mod

    resolved_allowed = [os.path.abspath(p) for p in allowed_paths]

    def _check_path(path: str) -> str | None:
        """Return resolved path if allowed, else return error string."""
        resolved = os.path.abspath(path)
        if any(resolved.startswith(ap) for ap in resolved_allowed):
            return None  # allowed
        return f"ERROR: Access denied. Path '{resolved}' is not under any allowed directory: {resolved_allowed}"

    @tool(
        "read_file",
        "Read a file. Returns the full content with line numbers. "
        "Restricted to allowed directories only.",
        {"abs_path": str, "start_line": int, "end_line": int},
    )
    async def _read_file(args: dict) -> dict:
        abs_path = args["abs_path"]
        start_line = args.get("start_line", 1) or 1
        end_line = args.get("end_line", 0) or 0
        err = _check_path(abs_path)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        try:
            with open(abs_path, "r", errors="replace") as f:
                lines = f.readlines()
            selected = lines[max(0, start_line - 1):end_line if end_line > 0 else len(lines)]
            text = "".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))
            return {"content": [{"type": "text", "text": text}]}
        except FileNotFoundError:
            return {"content": [{"type": "text", "text": f"ERROR: File not found: {abs_path}"}], "is_error": True}

    @tool(
        "grep_files",
        "Search file contents for a regex pattern in a directory. "
        "Returns matching lines with file paths and line numbers. "
        "Restricted to allowed directories only.",
        {"pattern": str, "directory": str, "file_glob": str},
    )
    async def _grep_files(args: dict) -> dict:
        pattern = args["pattern"]
        directory = args.get("directory", ".")
        file_glob = args.get("file_glob", "**/*")
        err = _check_path(directory)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        matches = []
        files = sorted(_glob_mod.glob(file_glob, root_dir=directory, recursive=True))
        for fname in files[:500]:
            fpath = os.path.join(directory, fname)
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
        text = "\n".join(matches) if matches else "No matches found."
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "glob_files",
        "Find files matching a glob pattern under a directory. "
        "Returns one path per line. Restricted to allowed directories only.",
        {"pattern": str, "directory": str},
    )
    async def _glob_files(args: dict) -> dict:
        pattern = args["pattern"]
        directory = args.get("directory", ".")
        err = _check_path(directory)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        matches = sorted(_glob_mod.glob(pattern, root_dir=directory, recursive=True))
        text = "\n".join(os.path.join(directory, m) for m in matches) if matches else "No files matched."
        return {"content": [{"type": "text", "text": text}]}



    @tool(
        "file_qa",
        "Answer questions about a file's content. "
        "Restricted to allowed directories only.",
        {"abs_path": str, "question": str},
    )
    async def _file_qa(args: dict) -> dict:
        abs_path = args["abs_path"]
        question = args["question"]
        err = _check_path(abs_path)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        try:
            print(f"  [file_qa] reading file for QA: {abs_path} ... and answering question: {question}")
            with open(abs_path, "r", errors="replace") as f:
                file = f.read()
            if len(file) > 200_000:
                file = file[:200_000] + "\n\n[... truncated]"
            answer, _cost = await _call_openai(
                _get_client(OPENROUTER_API_KEY),
                "file_qa",
                "You are a helpful assistant that answers questions about the content of a file.",
                f"Here is the content of the file:\n\n{file}\n\nQuestion: {question}",
                MODEL_QA,
            )

            return {"content": [{"type": "text", "text": answer}]}
            
        except FileNotFoundError:
            return {"content": [{"type": "text", "text": f"ERROR: File not found: {abs_path}"}], "is_error": True}

    return [_read_file, _grep_files, _glob_files, _file_qa]


def _make_sandboxed_mcp_server(name: str, allowed_paths: list[str]):
    """Create an MCP server with path-restricted file tools."""
    tools = _make_sandboxed_tools(allowed_paths)
    return create_sdk_mcp_server(name=name, version="1.0.0", tools=tools)


# async def _run_reviewer_claude_sdk(
#     name: str,
#     system_prompt: str,
#     paper_path: str,
#     model_id: str,
#     venue: str = "",
# ) -> tuple[str, float]:
#     """Run a reviewer via Claude Agent SDK. The agent reads the paper file itself.
#     Returns (review, cost=0.0) — SDK does not expose cost."""

#     paper_abs = str(Path(paper_path).resolve())
#     paper_dir = str(Path(paper_abs).parent)
#     venue_line = (
#         f"This paper was submitted to **{venue}**. "
#         f"You MUST evaluate it against {venue}'s specific standards, acceptance bar, "
#         f"and expectations. Consider what {venue} reviewers typically look for.\n\n"
#     ) if venue else ""

#     prompt = (
#         f"{system_prompt}\n\n"
#         f"---\n\n"
#         f"{venue_line}"
#         f"Review the following paper thoroughly.\n\n"
#         f"NOTE: This paper was extracted from PDF by an automated parser. "
#         f"There may be formatting artifacts such as broken equations, garbled "
#         f"tables, misplaced figure references, or OCR errors. These are parser "
#         f"issues, NOT problems with the paper itself. Do NOT treat formatting "
#         f"artifacts as weaknesses.\n\n"
#         f"The paper is located at: {paper_abs}\n"
#         f"Use the read_file tool to read the paper file, then produce your review."
#     )

#     print(f"  [{name}] starting Claude Agent SDK ({model_id}) ...")

#     reviewer_mcp = _make_sandboxed_mcp_server("reviewer_fs", [paper_dir])
#     result_text = ""
#     options = ClaudeAgentOptions(
#         model=model_id,
#         cwd="./tmp",
#         allowed_tools=[
#             "mcp__reviewer_fs__read_file",
#             "mcp__reviewer_fs__grep_files",
#             "mcp__reviewer_fs__glob_files",
#         ],
#         disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Agent"],
#         mcp_servers={"reviewer_fs": reviewer_mcp},
#         max_turns=30,
#     )
#     cost = 0
#     async with ClaudeSDKClient(options=options) as sdk_client:
#         await sdk_client.query(prompt)
#         async for message in sdk_client.receive_response():
#             if isinstance(message, AssistantMessage):
#                 for block in message.content:
#                     if isinstance(block, TextBlock):
#                         result_text += block.text
#             if isinstance(message, ResultMessage):
#                 cost += message.total_cost_usd

#     print(f"  [{name}] done — {model_id} (Claude Agent SDK) — saved ${cost:.4f}")
#     _add_sdk_savings(cost)
#     return result_text, 0.0


async def run_reviewer(
    client: AsyncOpenAI,
    name: str,
    system_prompt: str,
    paper_path: str,
    paper_content: str,
    model: str,
    venue: str = "",
) -> tuple[str, float]:
    """Run a reviewer via plain OpenRouter chat completions. Returns (review, cost)."""
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
    return await _call_openai(client, name, system_prompt, user_prompt, model)


async def run_related_work_search(
    client: AsyncOpenAI,
    paper_content: str,
) -> tuple[str, float]:
    """
    Two-step related work pipeline via OpenRouter. Returns (filtered_results, total_cost).
    """
    abstract_section = paper_content[:3000]

    print("  [related_work_search] started (OpenRouter online) ...")
    raw_results, cost1 = await _call_openai(
        client,
        "related_work_search",
        RELATED_WORK_PROMPT,
        (
            f"Find related work for this paper. Here is the title and abstract:\n\n"
            f"{abstract_section}\n\n"
            f"Search for real, published papers that are closely related."
        ),
        MODEL_RELATED_WORK,
    )

    print("  [related_work_filter] started (OpenRouter) ...")
    filtered, cost2 = await _call_openai(
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
    )

    return filtered, cost1 + cost2



async def _parse_score(client: AsyncOpenAI, text: str) -> tuple[float, float]:
    """Use GPT-5.4-nano with structured output to extract a score. Returns (score, cost)."""
    response = await client.beta.chat.completions.parse(
        model=MODEL_PARSER,
        messages=[
            {"role": "system", "content": _load_prompt("parse_score.txt")},
            {"role": "user", "content": text},
        ],
        response_format=ScoreSchema,
        timeout=30,
    )
    parsed = response.choices[0].message.parsed
    cost = _extract_cost(response)
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    tokens = f"{input_tokens}in/{output_tokens}out" if input_tokens and output_tokens else "n/a"
    print(f"  [score_parser] done — {MODEL_PARSER} — {tokens} tokens — ${cost:.4f}")
    if parsed is None:
        raise ValueError("score_parser returned no parsed output")
    return parsed.score, cost


async def _answer_file_question(abs_path: str, question: str, allowed_paths: list[str]) -> str:
    resolved = os.path.abspath(abs_path)
    resolved_allowed = [os.path.abspath(path) for path in allowed_paths]
    if not any(resolved.startswith(allowed_path) for allowed_path in resolved_allowed):
        return (
            f"ERROR: Access denied. Path '{resolved}' is not under any allowed directory: "
            f"{resolved_allowed}"
        )
    try:
        with open(resolved, "r", errors="replace") as file_handle:
            file_content = file_handle.read()
    except FileNotFoundError:
        return f"ERROR: File not found: {resolved}"
    if len(file_content) > 200_000:
        file_content = file_content[:200_000] + "\n\n[... truncated]"
    answer, _cost = await _call_openai(
        _get_client(OPENROUTER_API_KEY),
        "file_qa",
        "You are a helpful assistant that answers questions about the content of a file.",
        f"Here is the content of the file:\n\n{file_content}\n\nQuestion: {question}",
        MODEL_QA,
    )
    return answer


async def run_merge(
    client: AsyncOpenAI,
    harsh_review: str,
    neutral_review: str,
    spark_review: str,
    related_work: str,
    paper_content: str,
    skip_neutral: bool = False,
    skip_spark: bool = False,
    skip_related_work: bool = False,
) -> tuple[str, float]:
    """
    Merger only — synthesize sub-agent reviews into a consolidated review.
    Returns (review_text, cost).
    """
    merger_prompt = _build_merger_prompt(
        skip_neutral=skip_neutral,
        skip_spark=skip_spark,
        skip_related_work=skip_related_work,
    )

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

    user_prompt_review = (
        f"Here is the paper being reviewed (extracted from PDF — formatting "
        f"artifacts are parser issues, not paper problems):\n\n"
        f"--- PAPER CONTENT START ---\n"
        f"{paper_content}\n"
        f"--- PAPER CONTENT END ---\n\n"
        f"Here are the inputs:\n\n"
        f"{reviews_section}" 
        f"Now produce the final consolidated review following your instructions. "
        f"Remember: many of the harsh critic's points may be nonsensical or overly "
        f"picky — cross-check everything against the actual paper before including it."
    )
    if MODEL_MERGER.startswith("claude:"):
        review_text, cost = await run_claude_agent_task(
            name="merger",
            instructions=merger_prompt,
            user_prompt=user_prompt_review,
            model_id=MODEL_MERGER.split(":", 1)[1],
        )
    elif MODEL_MERGER.startswith("openai_agent:"):
        review_text, cost = await run_openai_agent_task(
            name="merger",
            instructions=merger_prompt,
            user_prompt=user_prompt_review,
            model_id=MODEL_MERGER[len("openai_agent:"):],
        )
    elif MODEL_MERGER.startswith("zai:"):
        review_text, cost = await _call_openai(
            zai_client, "merger", merger_prompt, user_prompt_review, MODEL_MERGER.split(":", 1)[1]
        )
    else:
        review_text, cost = await _call_openai(
            client, "merger", merger_prompt, user_prompt_review, MODEL_MERGER,
        )
    return review_text, cost


async def run_scorer(
    client: AsyncOpenAI,
    review_text: str,
    paper_path: str,
    calibration_context: str = "",
    cal_dir: str = "",
    gt_score: float | None = None,
) -> tuple[float, float]:
    """
    Scorer — uses an SDK backend to search calibration examples via
    Grep/Read, then scores the paper. Returns (score, cost).
    """
    cal_dir_abs = str(Path(cal_dir).resolve()) if cal_dir else ""
    paper_abs = str(Path(paper_path).resolve())
    paper_dir_abs = str(Path(paper_abs).parent)
    paper_file_name = Path(paper_abs).name
    scorer_allowed_paths = [path for path in [cal_dir_abs, paper_dir_abs] if path]

    scorer_agent_template = _load_prompt("scorer_agent.txt")
    scorer_instructions = scorer_agent_template.format(
        cal_dir_abs=cal_dir_abs, 
        paper_abs=paper_abs,
        paper_dir_abs=paper_dir_abs,
        paper_file_name=paper_file_name,
    )
    if MODEL_SCORER.startswith("claude:"):
        scorer_mcp = make_claude_file_mcp_server(
            "scorer_fs",
            scorer_allowed_paths,
            file_qa_callback=_answer_file_question,
        ) if scorer_allowed_paths else None
        scorer_mcp_tools = [
            "mcp__scorer_fs__read_file",
            "mcp__scorer_fs__grep_files",
            "mcp__scorer_fs__glob_files",
            "mcp__scorer_fs__file_qa",
        ] if scorer_mcp else []
        scorer_mcp_servers = {"scorer_fs": scorer_mcp} if scorer_mcp else {}
        result_text, total_cost = await run_claude_agent_task(
            name="scorer-agent",
            instructions=scorer_instructions,
            user_prompt=review_text,
            model_id=MODEL_SCORER.split(":", 1)[1],
            allowed_tools=scorer_mcp_tools,
            mcp_servers=scorer_mcp_servers,
            cwd=cal_dir_abs or paper_dir_abs or None,
        )
        print(f"  [scorer-agent] Claude Agent SDK savings: ${total_cost:.4f}")
        _add_sdk_savings(total_cost)
    elif MODEL_SCORER.startswith("openai_agent:"):
        openai_tools = build_openai_file_tools(
            scorer_allowed_paths,
            file_qa_callback=_answer_file_question,
        ) if scorer_allowed_paths else []
        result_text, _agent_cost = await run_openai_agent_task(
            name="scorer-agent",
            instructions=scorer_instructions,
            user_prompt=review_text,
            model_id=MODEL_SCORER[len("openai_agent:"):],
            tools=openai_tools,
        )
    else:
        raise ValueError(
            f"MODEL_SCORER must use an SDK backend prefix. Got: {MODEL_SCORER}"
        )


    # Log full scorer output to file for debugging
    scorer_log_path = Path(__file__).parent / "scorer_debug.log"
    with open(scorer_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 72}\n")
        f.write(f"cal_dir: {cal_dir_abs}\n")
        f.write(f"{'─' * 72}\n")
        f.write(result_text)
        f.write(f"\nGT Score: {gt_score}\n")
        f.write(f"\n{'=' * 72}\n\n")

    leakage_matches = _detect_leakage_warning_phrases(result_text)
    if leakage_matches:
        matched_text = ", ".join(sorted(set(leakage_matches), key=str.lower))
        warning_msg = (
            f"Potential calibration leakage warning: scorer output contains "
            f"suspicious phrase(s): {matched_text}"
        )
        print(f"  [scorer-agent] WARNING: {warning_msg}")
        _error_logger.error(warning_msg)

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
    paper_content: str,
    paper_path: str,
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
        spark_review, related_work, paper_content,
        skip_neutral=skip_neutral,
        skip_spark=skip_spark,
        skip_related_work=skip_related_work,
    )
    score, cost_score = await run_scorer(
        client, review_text, 
        paper_path=paper_path,
        calibration_context=calibration_context,
        cal_dir=cal_dir,
        gt_score=gt_score
    )
    return review_text, score, cost_merge + cost_score


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
        raise FileNotFoundError(
            "No calibration source found: no cal_dir, no calibration_context, "
            "no calibration_path provided, and default calibration.md does not exist."
        )

    cal_dir_candidate = resolved_path.parent / "cal"
    if cal_dir_candidate.is_dir():
        return "", str(cal_dir_candidate)
    if resolved_path.exists():
        return resolved_path.read_text(encoding="utf-8", errors="replace"), ""
    raise FileNotFoundError(f"Calibration path does not exist: {resolved_path}")



async def run_human_finder(pp, human_review_dir):
    paper_abs = str(Path(pp).resolve())
    human_review_abs = str(Path(human_review_dir).resolve())
    query = (
        f"{HUMAN_FINDER_PROMPT}\n\n"
        f"Paper file path: {paper_abs}\n"
        f"Human reviews directory: {human_review_abs}\n"
    )
    paper_dir = str(Path(paper_abs).parent)
    raw_result_text = ""
    last_message_text = ""

    if MODEL_FIND_HUMAN.startswith("claude:"):
        hf_mcp = _make_sandboxed_mcp_server("hf_fs", [human_review_abs, paper_dir])
        options = ClaudeAgentOptions(
            model=MODEL_FIND_HUMAN.split(":", 1)[1],
            cwd=human_review_abs,
            allowed_tools=[
                "mcp__hf_fs__read_file",
                "mcp__hf_fs__grep_files",
                "mcp__hf_fs__glob_files",
                "mcp__hf_fs__file_qa",
                "mcp__search__search_file",
            ],
            disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Agent"],
            mcp_servers={"search": _search_mcp_server, "hf_fs": hf_mcp},
            max_turns=30,
        )
        print(f"  [Human Finder] starting Claude Agent SDK ({MODEL_FIND_HUMAN}) ...")
        sdk_savings = 0.0
        async with ClaudeSDKClient(options=options) as sdk_client:
            await sdk_client.query(query)
            async for message in sdk_client.receive_response():
                if isinstance(message, AssistantMessage):
                    current_message_text = ""
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            current_message_text += block.text
                    if current_message_text:
                        raw_result_text += current_message_text
                        last_message_text = current_message_text
                if isinstance(message, ResultMessage):
                    sdk_savings += message.total_cost_usd

        _add_sdk_savings(sdk_savings)
        print(f"  [Human Finder] done (Claude Agent SDK) — saved ${sdk_savings:.4f}")
    elif MODEL_FIND_HUMAN.startswith("openai_agent:"):
        from agents import function_tool

        @function_tool
        def search_file(query: str, path: str, n: int = 5) -> str:
            """Search for a pattern in a directory using the BM25 index."""
            path = os.path.abspath(path)
            if path not in _bm25_database:
                raise ValueError(f"Path '{path}' is not indexed or allowed for searching.")
            bm25 = _bm25_database[path]["bm25"]
            files = _bm25_database[path]["files"]
            tokenized_query = query.split(" ")
            doc_scores = bm25.get_scores(tokenized_query)
            top_indices = doc_scores.argsort()[-n:][::-1]
            results = []
            for idx in top_indices:
                file_path = os.path.abspath(files[idx])
                score = doc_scores[idx]
                with open(file_path, "r", errors="replace") as file_handle:
                    content = file_handle.read()
                results.append(f"{file_path}\nscore: {score:.2f}\n first 1000 chars:\n{content[:1000]}\n")
            if not results:
                return "No relevant files found."
            return "\n---\n".join(results)

        hf_tools = build_openai_file_tools(
            [human_review_abs, paper_dir],
            file_qa_callback=_answer_file_question,
        )
        hf_tools.append(search_file)
        print(f"  [Human Finder] starting OpenAI Agent SDK via OpenRouter ({MODEL_FIND_HUMAN}) ...")
        result_text, _sdk_cost = await run_openai_agent_task(
            name="Human Finder",
            instructions=HUMAN_FINDER_PROMPT,
            user_prompt=(
                f"Paper file path: {paper_abs}\n"
                f"Human reviews directory: {human_review_abs}\n"
            ),
            model_id=MODEL_FIND_HUMAN[len("openai_agent:"):],
            tools=hf_tools,
        )
        raw_result_text = result_text
        last_message_text = result_text
    else:
        raise ValueError(f"MODEL_FIND_HUMAN must use an SDK backend prefix. Got: {MODEL_FIND_HUMAN}")

    if not last_message_text.strip():
        raise ValueError("Human finder returned no final assistant message")
    parsed_result_text = _extract_final_review_block(last_message_text)
    if parsed_result_text is None:
        parsed_result_text = last_message_text.strip()
    with open(Path(__file__).parent / "human_finder_debug.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 72}\n")
        f.write(f"Paper path: {pp}\n")
        f.write(f"Human review dir: {human_review_dir}\n")
        f.write(f"{'-' * 72}\n")
        f.write("RAW LAST MESSAGE\n")
        f.write(f"{'-' * 72}\n")
        f.write(last_message_text)
        f.write(f"\n{'-' * 72}\n")
        f.write("RAW TOTAL OUTPUT\n")
        f.write(f"{'-' * 72}\n")
        f.write(raw_result_text)
        f.write(f"\n{'-' * 72}\n")
        f.write("PARSED FINAL REVIEW\n")
        f.write(f"{'-' * 72}\n")
        f.write(parsed_result_text)
        f.write(f"\n{'=' * 72}\n\n")
        
    return parsed_result_text, 0.0

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
) -> dict:
    """
    Core review pipeline: Phase 1 (reviewers) + Phase 2 (merger + optional scorer).

    Returns a dict with keys:
      harsh_review, neutral_review, spark_review, related_work,
      merged_review, cost, sdk_savings,
      score (float or None if skip_score), decision (str or None if skip_score).
    """
    pp = paper_path

    # Guard against data leakage: paper ID must not exist in the human review set
    paper_id = Path(pp).stem
    human_review_path = Path(human_review_dir) / "human_reviews" / f"{paper_id}.md"
    if human_review_path.exists():
        raise ValueError(
            f"Data leakage: paper {paper_id} has a human review in "
            f"{human_review_dir}/human_reviews/. The input dataset and human "
            f"review set must not share paper IDs."
        )

    savings_before = get_sdk_savings()

    # ── Phase 1: All reviewers (parallel or sequential) ───────────
    total_cost = 0.0
    if parallel:
        tasks = [
            run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue=venue),
        ]
        if not skip_neutral:
            tasks.append(run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue=venue))
        if not skip_spark:
            tasks.append(run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue=venue))
        if not skip_related_work:
            tasks.append(run_related_work_search(client, paper_content))
        tasks.append(run_human_finder(pp, human_review_dir))

        print("  Phase 1: All reviewers in parallel ...")
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
            related_work, c = results_list[idx]; total_cost += c; idx += 1
        else:
            related_work = "Related work search was skipped."
        human_match_review, c = results_list[idx]; total_cost += c
        harsh_review = (
            f"{harsh_review.rstrip()}\n\n"
            f"Additional transferable weaknesses from matched human reviews:\n"
            f"{human_match_review.lstrip()}"
        )
    else:
        raise NotImplementedError("Sequential mode is not implemented in this version.")

    # ── Phase 2: Merger (+ optional scorer) ───────────────────────
    if skip_score:
        print("  Phase 2: Merger (no scoring) ...")
        merged_review, merger_cost = await run_merge(
            client, harsh_review, neutral_review,
            spark_review, related_work, paper_content,
            skip_neutral=skip_neutral,
            skip_spark=skip_spark,
            skip_related_work=skip_related_work,
        )
        total_cost += merger_cost
        score = None
        decision = None
    else:
        print("  Phase 2: Merger + Scorer ...")
        merged_review, score, merger_cost = await run_merger(
            client, harsh_review, neutral_review,
            spark_review, related_work, paper_content, paper_path,
            calibration_context=calibration_context,
            cal_dir=cal_dir,
            skip_neutral=skip_neutral,
            skip_spark=skip_spark,
            skip_related_work=skip_related_work,
            gt_score=gt_score,
        )
        total_cost += merger_cost
        score = round(float(score), 1)
        decision = score_to_decision(score)

    sdk_savings = get_sdk_savings() - savings_before

    return {
        "harsh_review": harsh_review,
        "neutral_review": neutral_review,
        "spark_review": spark_review,
        "related_work": related_work,
        "merged_review": merged_review,
        "score": score,
        "decision": decision,
        "cost": total_cost,
        "sdk_savings": sdk_savings,
    }


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
    pp = str(path)
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
    print(f"  Scorer:         claude-haiku-4-5 (Agent SDK)\n")

    client = _get_client(api_key=api_key)

    result = await run_pipeline(
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
    )

    total_cost = result["cost"]
    final_review = result["merged_review"]
    final_score = result["score"]
    final_decision = result["decision"]
    harsh_review = result["harsh_review"]
    neutral_review = result["neutral_review"]
    spark_review = result["spark_review"]
    related_work = result["related_work"]

    print(f"Total cost for this paper: ${total_cost:.4f}")

    # ── Output ────────────────────────────────────────────────────
    separator = "=" * 72
    full_output = (
        f"\n{separator}\n"
        f"INDIVIDUAL REVIEWS\n"
        f"{separator}\n\n"
        f"{'─' * 40}\n"
        f"HARSH CRITIC ({MODEL_HARSH} via OpenRouter)\n"
        f"{'─' * 40}\n"
        f"{harsh_review}\n\n"
        f"{'─' * 40}\n"
        f"NEUTRAL REVIEWER ({MODEL_NEUTRAL} via OpenRouter)\n"
        f"{'─' * 40}\n"
        f"{neutral_review}\n\n"
        f"{'─' * 40}\n"
        f"SPARK FINDER ({MODEL_SPARK} via OpenRouter)\n"
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
        print("  --venue <name>      Set venue (e.g. ICLR, NeurIPS, ICML)")
        print("  --calibration <p>   Calibration file/path (default: calibration.md if present)")
        print()
        print("Environment variables (or set in .env):")
        print("  OPENROUTER_API_KEY   (required) Your OpenRouter API key")
        print()
        print("Models per stage:")
        print(f"  Harsh Critic (OpenRouter):      {MODEL_HARSH}")
        print(f"  Neutral (OpenRouter):           {MODEL_NEUTRAL}")
        print(f"  Spark Finder (OpenRouter):      {MODEL_SPARK}")
        print(f"  Related Work (OpenRouter):      {MODEL_RELATED_WORK}")
        print(f"  Merger (OpenRouter):            {MODEL_MERGER}")
        print(f"  Scorer:                         claude-haiku-4-5 (Agent SDK)")
        sys.exit(0 if "--help" in sys.argv else 1)

    parallel = "--sequential" not in sys.argv
    skip_related = "--with-related-work" not in sys.argv
    skip_spark = "--no-spark" in sys.argv
    skip_neutral = "--no-neutral" in sys.argv
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
        )
    )
    print(result)
