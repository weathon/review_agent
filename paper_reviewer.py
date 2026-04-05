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

# Per-stage model assignments — all via OpenRouter
base_model = "qwen/qwen3.6-plus:free"
MODEL_HARSH = f"{base_model}"
MODEL_NEUTRAL = f"{base_model}"
MODEL_SPARK = f"{base_model}"
MODEL_RELATED_WORK = f"{base_model}:online" 
MODEL_FILTER = f"{base_model}"
MODEL_MERGER = f"{base_model}"
MODEL_SCORER = f"{base_model}"
MODEL_PARSER = "openai/gpt-5.4-nano"

MAX_RETRIES = 3
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

# ── Agent system prompts ──────────────────────────────────────────────

HARSH_CRITIC_PROMPT = """\
You are a deeply thoughtful, experienced academic reviewer. You are NOT trying \
to be picky or find fault for its own sake. Instead, you engage seriously with \
the paper and raise genuine questions, concerns, and insights that the authors \
need to address.

You MUST evaluate EACH section of the paper individually using the rubric below. \
Do not give vague general comments — cite specific sections, equations, figures, \
tables, or claims when raising concerns.

Do NOT nitpick formatting, style, or minor phrasing. Do NOT invent problems \
that aren't there. Do NOT penalize intentional scope decisions. Focus on what \
actually matters for the paper's contribution.

Output format:

## Section-by-Section Critical Review

Go through the paper's sections and evaluate each one on its own merits. \
You are not limited to the categories below — adapt your review to the \
paper's actual structure and content. The following are examples of the \
kinds of questions you might consider, but raise whatever concerns are \
genuinely important for this specific paper:

Example sections and questions (use as inspiration, not a checklist):
- Title & Abstract: Does the title reflect the contribution? Are abstract \
claims supported?
- Introduction & Motivation: Is the problem well-motivated? Are contributions \
clearly stated?
- Method / Approach: Is it reproducible? Are assumptions justified? Any \
logical gaps or missing edge cases? Are proofs correct?
- Experiments & Results: Do experiments test the claims? Are baselines fair? \
Missing ablations? Statistical significance? Cherry-picked results?
- Writing & Clarity: Any sections confusing enough to impede understanding? \
(Do NOT nitpick grammar or formatting.)
- Limitations & Broader Impact: Are key limitations acknowledged? Any missed \
failure modes or negative societal impacts?

Focus on whatever matters most for THIS paper. Skip sections that are fine \
and spend more time on sections with real issues.

### Overall Assessment
One paragraph. Summarize the most important concerns and whether the \
contribution stands despite them. Be honest, direct, and calibrated.
"""


NEUTRAL_REVIEWER_PROMPT = """\
You are a fair, balanced academic reviewer. You give credit where due and \
critique where warranted, without bias in either direction.

Your job:
- Summarize the paper's main contribution in 2-3 sentences.
- List concrete strengths (with evidence from the paper).
- List concrete weaknesses (with evidence from the paper).
- Assess novelty, clarity, reproducibility, and significance.
- Suggest specific improvements.

Output format (strictly follow):
## Balanced Review

### Summary
...

### Strengths
1. ...

### Weaknesses
1. ...

### Novelty & Significance
...

### Suggestions for Improvement
1. ...
"""

SPARK_FINDER_PROMPT = """\
You are a senior research advisor. Your job is to identify the MOST CRITICAL \
gaps in this paper — what is missing or incomplete that directly undermines \
the paper's claims or contribution.

Be focused and prioritized. List only the TOP 3-5 most important issues per \
category. Do NOT write an exhaustive wishlist of everything that could \
possibly be done — that dilutes the signal. Every item you list should pass \
this test: "Would addressing this meaningfully change whether the paper's \
core claims are believable?" If not, leave it out.

Do not praise the paper. Do not pad suggestions with compliments. State \
what's needed directly: "Add X because without it, claim Y is not convincing."

Be concise. Each item should be 1-3 sentences, not a paragraph.

Your job:
- What experiments are missing that directly undermine the paper's claims? \
  Be specific about datasets, baselines, or ablations.
- What analysis is absent that would be needed to trust the method or results?
- What visualizations or case studies would expose whether the method \
  actually works vs. fails?
- What are the most obvious next steps the authors should have done?

Output format (strictly follow):
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. ... (what, why it matters for the claims — keep it concise)

### Deeper Analysis Needed (top 3-5 only)
1. ... (what insight is missing and why it matters)

### Visualizations & Case Studies
1. ... (what would reveal whether the method works)

### Obvious Next Steps
1. ... (what should have been in this paper)
"""

RELATED_WORK_PROMPT = """\
You are a research librarian and literature scout. Given a paper's title, \
abstract, and key contributions, your job is to find REAL, EXISTING related \
work that the authors should be aware of.

Your job:
- Search for closely related papers, especially recent ones (2022-2026).
- Include papers that use similar methods on different problems.
- Include papers that tackle the same problem with different methods.
- Include foundational/seminal work that should be cited.
- For each paper, provide: title, authors, year, venue, and a 1-sentence \
  explanation of why it's relevant.

Output format (strictly follow):
## Related Work Search Results

### Closely Related (same problem + similar approach)
1. **Title** — Authors (Year, Venue). Why relevant: ...

### Same Problem, Different Approach
1. **Title** — Authors (Year, Venue). Why relevant: ...

### Same Method, Different Problem
1. **Title** — Authors (Year, Venue). Why relevant: ...

### Foundational Work
1. **Title** — Authors (Year, Venue). Why relevant: ...
"""

RELATED_WORK_FILTER_PROMPT = """\
You are given a paper and a list of potentially related works found by a \
search agent. Your job is to FILTER this list:

1. REMOVE any work that is ALREADY CITED in the paper. Check the references \
   section and in-text citations carefully. If the paper already mentions it \
   (even by a different abbreviation or partial title), remove it.
2. REMOVE any work that is only TANGENTIALLY related — if the connection is \
   a stretch or requires multiple leaps of logic, drop it.
3. KEEP only works that are genuinely relevant AND not already cited.

For each kept work, briefly explain why it's a potentially missed reference.

Output format (strictly follow):
## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have \
intentionally excluded them or been unaware of them.)

1. **Title** — Authors (Year, Venue).
   Why potentially missed: ...

If all works are already cited or not relevant, say:
"No significant potentially missed related work identified."
"""



_MERGER_PROMPT_TEMPLATE = """\
You are a senior meta-reviewer / area chair. You have received {input_count} inputs \
about the same paper:

1. A **harsh critic** review (may be overly critical)
{neutral_line}\
{spark_line}\
{related_work_line}\


Your job is to synthesize these into ONE authoritative final review. \
Be honest and unsparing about real problems, but do not manufacture or inflate weaknesses. \
It is for ICLR — standards are high, but a strong paper should read as strong.

Before including any weakness, verify: (1) does the paper actually have this problem, or did the reviewer \
misread a section? (2) if the paper partially addresses this concern, is the addressal unreasonable or is \
the reviewer ignoring it? Quote the relevant section if needed to justify keeping or removing the criticism.

Rules:
- REMOVE criticisms that are factually wrong or misunderstand the paper.
- REMOVE pure formatting/style nitpicks.
- REMOVE or WEAKEN criticisms that demand the paper address problems outside its stated scope \
and contributions. A paper about X should be evaluated on whether it does X well, not on whether \
it also does Y. If the paper explicitly scopes out a direction, criticizing its absence is scope creep, \
not a weakness.
- REMOVE or WEAKEN weaknesses (or put them into nice-to-have) if they are generic or one-weakness-suit-all type \
and does not harm the core claim of the paper.
- REMOVE weaknesses that authors already address in the paper, even if imperfectly and the addressal is reasonable.
- REMOVE or WEAKEN weaknesses where it is not what the paper intended to address \
- REMOVE criticisms that claim a cited reference does not exist, a method or model is not yet released, \
or a benchmark is unavailable. These are due to the lack of knowledge of reviewer, not author mis-claiming. \
If the paper cites it, assume it exists unless proven otherwise.
or where no reasonable revision could address the concern.
- MOVE TO NICE TO HAVE for weaknesses that demand methodological practices that are not standard or expected \
in the paper's field or setting. \
Examples: requesting confidence intervals or multiple-run statistics for large scale benchmarks where \
single-run evaluation is the norm, demanding theoretical proofs for an empirical systems paper, or \
requiring user studies for a purely algorithmic contribution. \
The review should evaluate the paper against the standards of its own community, not impose arbitrary \
rigor requirements.
- KEEP criticisms that are factually correct AND substantive, even if only \
  one reviewer raised them.
- KEEP genuine strengths backed by evidence.
- DO NOT mention missing related works, as you do not have external sources to confirm their existence and could be making things up.
- Do NOT pad Strengths or Weaknesses to appear balanced or make the list "more thorough". \
If the paper has only one (or none) genuine strength, list only one (or none). \
If a weakness is minor or redundant, omit it. Quality over quantity.
- REMOVE or WEAKEN strengths that are generic or would apply to any paper. \
Examples: "the paper is well-written," "the topic is important," "the experiments are extensive." \
A strength must identify something specific this paper does well that most papers in the area do not.
- If the weaknesses identified would, if true, invalidate or severely undermine the paper's core \
contribution, the review should reflect that clearly — do not soften the overall tone to appear balanced.

Output your final review in this markdown format:

## Summary
2-3 sentence summary of the paper's contribution.

## Strengths
- strength 1 with evidence
- strength 2 with evidence

## Weaknesses
- weakness 1 — why it matters
- weakness 2 — why it matters

## Nice-to-Haves
- suggestion that would improve but is not a core flaw

## Novel Insights
One paragraph synthesizing genuinely novel observations. \
If no genuinely novel insight emerges from the reviews beyond the paper's own contributions, write \
"None beyond the paper's own contributions."


## Suggestions
- specific actionable suggestion

Do NOT output any numerical scores, subscores, or accept/reject decisions. \

DO differentiate between papers of varying quality clearly: the content of the review \
should make it clear whether the paper is strong or weak, without using numerical scores.
"""


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


SCORE_PROMPT = """\
You previously wrote a consolidated review of a paper. Now assign an overall \
score from 0.0 to 10.0.

This is for a top-tier venue (ICLR, ~29% acceptance rate), most papers are scored lower than 6. \


## Comparative Scoring

Select a few calibration papers that are closest in quality to the current \
paper. Compare them against the current paper on these dimensions:
- novelty
- technical soundness
- empirical support
- significance
- clarity

Then set your score relative to the human scores of those selected papers.

Do NOT be afraid to give very high (>8) or very low (<4) scores! Good papers should be praised and bad paper should be found out.

Score continuously (e.g. 3.5, 4.7, 8.1). Use the full range — do not cluster \
around 5-6. Do not round to .5 or .0. give scores in x.2, 2.8, 7.3, etc. 

The score should be DISCRIMINATIVE. A weak paper is weak — \
give it a low score (1-3). A strong paper is strong — give it a high score (7-9). \
Do not cluster everything around 5. The quality difference between papers is real \
and your scores should reflect it.

## Scoring guide
- 10: Strong accept. Exceptional, field-advancing contribution.
- 8:  Accept.
- 6:  Borderline accept.
- 4:  Borderline reject.
- 2:  Reject.
- 0:  Strong reject. 
"""
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


# ── OpenRouter calls ───────────────────────────────────────────────────

# Models that support OpenRouter reasoning config
REASONING_MODELS = {"z-ai/glm-5", "minimax/minimax-m2.7", "deepseek/deepseek-v3.2", "minimax/minimax-m2.5:free", "stepfun/step-3.5-flash:free"}

# Model → official provider mapping (for OpenRouter provider pinning)
PROVIDER_MAP = {
    "z-ai/glm-5": ["deepinfra/fp4"],
    "z-ai/glm-5:online": ["deepinfra/fp4"],
    "minimax/minimax-m2.7": ["minimax/fp8"],
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
    return "", 0.0



# ── Agent runners ─────────────────────────────────────────────────────

async def run_reviewer(
    client: AsyncOpenAI,
    name: str,
    system_prompt: str,
    paper_path: str,
    paper_content: str,
    model: str,
    venue: str = "",
) -> tuple[str, float]:
    """Run a reviewer via OpenRouter chat completions. Returns (review, cost)."""
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
        f"Paper file: {paper_path}\n\n"
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
            {"role": "system", "content": (
                "The text contains a paper scoring analysis that references calibration examples "
                "with their own scores. Ignore all calibration/reference scores. "
                "Extract ONLY the final score the author assigned to the paper being reviewed. "
                "Look for 'MY FINAL SCORE:' at the end of the text."
            )},
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

    prompt = f"""\
{SCORE_PROMPT}

You are a paper scoring agent. Your job:

1. Read the consolidated review at {review_path} and the paper at {paper_path}.

2. Use Grep to search the calibration directory at {cal_dir_abs} for relevant
   calibration examples. The directory contains pairs of files:
   - *_paper.md  — the original paper text
   - *_review.md — the review + human scores
   Search BOTH *_paper.md and *_review.md files for keywords related to the
   paper's topic, methodology, strengths, and weaknesses. This helps you find
   calibration papers that are similar in content or quality.

3. Once you identify relevant matches, ONLY Read the *_review.md files (NOT the
   *_paper.md files) to save context. Aim for 5-7 relevant calibration reviews.
   You can also grep the score part of the review, usually in this format, to select a wide range of papers.
   ```
# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0]
Average score: 0.0
Binary outcome: Reject
```

4. Compare the paper's quality against those calibration examples on:
   novelty, technical soundness, empirical support, significance, clarity.

5. Assign a single overall score from 0.0 to 10.0.

Based on the calibration examples you found, assign a score. Explain your reasoning.

IMPORTANT: At the very end of your response, you MUST write exactly this line:
MY FINAL SCORE: <number>
This must be the LAST line of your output. Do NOT repeat calibration scores here — only YOUR score for THIS paper.
"""

    print(f"  [scorer-agent] starting RAG scorer (claude-sonnet-4-6, cal={cal_dir_abs}) ...")

    result_text = ""
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        cwd=cal_dir_abs or None,
        allowed_tools=["Grep", "Read", "Glob"],
        permission_mode="bypassPermissions",
        max_turns=15,
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
        f.write(f"GT Score: {gt_score}\n")
        f.write(f"\n{'=' * 72}\n\n")
        

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
        tasks = [
            run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue=venue),
        ]
        if not skip_neutral:
            tasks.append(run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue=venue))
        if not skip_spark:
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
        harsh_review, c = await run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue=venue)
        total_cost += c
        if not skip_neutral:
            neutral_review, c = await run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue=venue)
            total_cost += c
        else:
            neutral_review = "Neutral reviewer was skipped."
        if not skip_spark:
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
