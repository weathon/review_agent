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

load_dotenv()  # loads .env from cwd or parent dirs

# ── Config ────────────────────────────────────────────────────────────
PROVIDER = "zai" 


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_CLOUD_BASE_URL = "http://localhost:11434/v1/"
ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4/"

#base_model = "qwen/qwen3.6-plus:free" #用限时免费模型白嫖
base_model = "qwen/qwen3.5-flash-02-23"
MODEL_HARSH = f"claude:claude-sonnet-4-6" #用claude subscription白嫖
MODEL_NEUTRAL = f"ollama:{base_model}"
MODEL_SPARK = "ollama:qwen/qwen3.5-plus-02-15"
MODEL_RELATED_WORK = f"{base_model}:online" 
MODEL_FILTER = f"{base_model}"
# MODEL_MERGER = f"zai:glm-5.1" #用zai coding plan白嫖
MODEL_MERGER = f"z-ai/glm-5" #rate limit is very tight on zai coding plan, switch back to openrouter if needed
MODEL_PARSER = "openai/gpt-5.4-nano"

MAX_RETRIES = 5
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

# ── Prompt loading ────────────────────────────────────────────────────
_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts/ directory."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# ── Agent system prompts ──────────────────────────────────────────────

HARSH_CRITIC_PROMPT = _load_prompt("harsh_critic.txt")
NEUTRAL_REVIEWER_PROMPT = _load_prompt("neutral_reviewer.txt")
SPARK_FINDER_PROMPT = _load_prompt("spark_finder.txt")
RELATED_WORK_PROMPT = _load_prompt("related_work.txt")
RELATED_WORK_FILTER_PROMPT = _load_prompt("related_work_filter.txt")
_MERGER_PROMPT_TEMPLATE = _load_prompt("merger.txt")


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


SCORE_PROMPT = _load_prompt("scorer.txt")
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

zai_client = _get_zai_client(os.environ.get("ZAI_API_KEY", ""))


def _get_ollama_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key="ollama", base_url=OLLAMA_CLOUD_BASE_URL)


ollama_client = _get_ollama_client()


def _resolve_openai_client_and_model(
    client: AsyncOpenAI,
    model: str,
) -> tuple[AsyncOpenAI, str, str]:
    if model.startswith("ollama:"):
        return ollama_client, model.split(":", 1)[1], "Ollama"
    if model.startswith("zai:"):
        return zai_client, model.split(":", 1)[1], "ZAI"
    return client, model, "OpenRouter"



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
        return 0.0
    cost = getattr(usage, "cost", None)
    if cost is not None:
        return float(cost)
    if isinstance(usage, dict):
        return float(usage.get("cost", 0.0))
    return 0.0


async def _call_openai(
    client: AsyncOpenAI,
    name: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    tools: list[dict] | None = None,
) -> tuple[str, float]:
    """Call an OpenAI-compatible chat completion endpoint with retry logic."""
    resolved_client, resolved_model, provider_name = _resolve_openai_client_and_model(client, model)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=REQUEST_TIMEOUT,
            )
            extra = None
            if provider_name == "OpenRouter":
                extra = _build_extra_body(resolved_model, reasoning_effort="medium")
            if extra:
                kwargs["extra_body"] = extra
            response = await resolved_client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content or ""
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
                _error_logger.error(f"[{name}] empty response after {MAX_RETRIES} attempts, model={model}")
                print(f"  [{name}] empty response after {MAX_RETRIES} attempts")
            print(f"  [{name}] done — {model} ({provider_name}) — {tokens} tokens — ${cost:.4f}")
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
    return "", 0.0

print("Testing ZAI client with a simple call ...")

# ans = asyncio.run(_call_openai(zai_client, "test", "You are a helpful assistant.", "What is the capital of France?", "glm-5.1"))
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
        f"Use the Read tool to read the paper file, then produce your review."
    )

    print(f"  [{name}] starting Claude Agent SDK ({model_id}) ...")

    result_text = ""
    options = ClaudeAgentOptions(
        model=model_id,
        cwd="./tmp",
        allowed_tools=["Read", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        max_turns=30,
    )
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text

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
    """Extract a score using an OpenAI-compatible endpoint. Returns (score, cost)."""
    resolved_client, resolved_model, provider_name = _resolve_openai_client_and_model(client, MODEL_PARSER)
    if provider_name == "Ollama":
        parser_prompt = (
            f"{_load_prompt('parse_score.txt')}\n\n"
            f"Return valid JSON with exactly this shape: {{\"score\": 0.0}}"
        )
        parser_text, cost = await _call_openai(
            resolved_client,
            "score_parser",
            parser_prompt,
            text,
            MODEL_PARSER,
        )
        parsed = ScoreSchema.model_validate(json.loads(parser_text))
        return parsed.score, cost

    response = await resolved_client.beta.chat.completions.parse(
        model=resolved_model,
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
    paper_content: str,
    skip_neutral: bool = False,
    skip_spark: bool = False,
    skip_related_work: bool = False,
) -> tuple[str, float]:
    """
    Merger only — synthesize sub-agent reviews into a consolidated review.
    Returns (review_text, cost).
    """
    print(f"  [merger] started ({MODEL_MERGER}) ...")

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
    review_text, cost = await _call_openai(
        client, "merger", merger_prompt, user_prompt_review, MODEL_MERGER,
    )
    return review_text, cost


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

    # Write review and paper to temp files so the agent can Read them
    # (avoids CLI character limit on the prompt)
    tmp_dir = Path(tempfile.mkdtemp(prefix="scorer_"))
    review_path = tmp_dir / "review.txt"
    paper_path = tmp_dir / "paper.txt"
    review_path.write_text(review_text, encoding="utf-8")
    paper_path.write_text(paper_content, encoding="utf-8")

    scorer_agent_template = _load_prompt("scorer_agent.txt")
    prompt = scorer_agent_template.format(
        score_prompt=SCORE_PROMPT,
        review_path=review_path,
        paper_path=paper_path,
        cal_dir_abs=cal_dir_abs,
    )

    print(f"  [scorer-agent] starting RAG scorer (claude-sonnet-4-6, cal={cal_dir_abs}) ...")

    result_text = ""
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        cwd=cal_dir_abs or None,
        allowed_tools=["Grep", "Read", "Glob", "Agent"],
        permission_mode="bypassPermissions",
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
        client, review_text, paper_content,
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

    client = _get_client(api_key=api_key)
    pp = str(path)

    # ── Phase 1: All reviewers (parallel or sequential) ───────────
    total_cost = 0.0
    if parallel:
        if MODEL_HARSH.startswith("ollama:"):
            tasks = [
                run_reviewer(ollama_client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH.replace("ollama:", "", 1), venue=venue),
            ]
        else:
            tasks = [
                run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue=venue),
            ]
        if not skip_neutral:
            if MODEL_NEUTRAL.startswith("ollama:"):
                tasks.append(run_reviewer(ollama_client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL.replace("ollama:", "", 1), venue=venue))
            else:
                tasks.append(run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue=venue))
        if not skip_spark:
            if MODEL_SPARK.startswith("ollama:"):
                tasks.append(run_reviewer(ollama_client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK.replace("ollama:", "", 1), venue=venue))
            else:
                tasks.append(run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue=venue))
        if not skip_related_work:
            tasks.append(run_related_work_search(client, paper_content))

        print("Phase 1: All reviewers in parallel ...")
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
        print("Phase 1: Reviewers sequentially ...")
        if MODEL_HARSH.startswith("ollama:"):
            harsh_review, c = await run_reviewer(ollama_client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH.replace("ollama:", "", 1), venue=venue)
        else:
            harsh_review, c = await run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue=venue)
        total_cost += c
        if not skip_neutral:
            if MODEL_NEUTRAL.startswith("ollama:"):
                neutral_review, c = await run_reviewer(ollama_client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL.replace("ollama:", "", 1), venue=venue)
            else:
                neutral_review, c = await run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue=venue)
            total_cost += c
        else:
            neutral_review = "Neutral reviewer was skipped."
        if not skip_spark:
            if MODEL_SPARK.startswith("ollama:"):
                spark_review, c = await run_reviewer(ollama_client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK.replace("ollama:", "", 1), venue=venue)
            else:
                spark_review, c = await run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue=venue)
            total_cost += c
        else:
            spark_review = "Spark finder was skipped."
        if not skip_related_work:
            related_work, c = await run_related_work_search(client, paper_content)
            total_cost += c
        else:
            related_work = "Related work search was skipped."

    # ── Phase 2: Merger + Score (same conversation) ───────────────
    print("\nPhase 2: Merger ...")
    final_review, final_score, merger_cost = await run_merger(
        client, harsh_review, neutral_review,
        spark_review, related_work, paper_content,
        calibration_context=calibration_context,
        cal_dir=cal_dir,
        skip_neutral=skip_neutral,
        skip_spark=skip_spark,
        skip_related_work=skip_related_work,
    )
    total_cost += merger_cost
    final_score = round(float(final_score), 1)
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
