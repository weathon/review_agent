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

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-stage model assignments — all via OpenRouter
MODEL_HARSH = "z-ai/glm-5"
MODEL_NEUTRAL = "z-ai/glm-5"
MODEL_SPARK = "z-ai/glm-5"
MODEL_RELATED_WORK = "z-ai/glm-5:online" 
MODEL_FILTER = "z-ai/glm-5"
MODEL_MERGER = "z-ai/glm-5"
MODEL_SCORER = "claude-sdk:claude-sonnet-4-6"
MODEL_PARSER = "openai/gpt-5.4-nano"

MAX_RETRIES = 3
RETRY_DELAY = 10
REQUEST_TIMEOUT = 120

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
    if score is None:
        return None
    return "Accept" if score >= 5.5 else "Reject"

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

Output format (strictly follow — evaluate EVERY section):

## Section-by-Section Critical Review

### Title & Abstract
- Does the title accurately reflect the contribution?
- Does the abstract clearly state the problem, method, and key results?
- Are any claims in the abstract unsupported by the paper?

### Introduction & Motivation
- Is the problem well-motivated? Is the gap in prior work clearly identified?
- Are the contributions clearly stated and accurate?
- Does the introduction over-claim or under-sell?

### Method / Approach
- Is the method clearly described and reproducible?
- Are key assumptions stated and justified?
- Are there logical gaps in the derivation or reasoning?
- Are there edge cases or failure modes not discussed?
- For theoretical claims: are proofs correct and complete?

### Experiments & Results
- Do the experiments actually test the paper's claims?
- Are baselines appropriate and fairly compared?
- Are there missing ablations that would materially change conclusions?
- Are error bars / statistical significance reported?
- Do the results support the claims made, or are they cherry-picked?
- Are datasets and evaluation metrics appropriate?

### Writing & Clarity
- Are there sections that are confusing or poorly explained?
- Are figures and tables clear and informative?
(Do NOT nitpick grammar or formatting — only flag clarity issues that \
impede understanding of the contribution.)

### Limitations & Broader Impact
- Do the authors acknowledge the key limitations?
- Are there fundamental limitations they missed?
- Are there failure modes or negative societal impacts not discussed?

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



MERGER_PROMPT = """\
You are a senior meta-reviewer / area chair. You have received four inputs \
about the same paper:

1. A **harsh critic** review (may be overly critical)
2. A **neutral/balanced** review
3. A **spark finder** report (focuses on insights, not flaws)
4. A **potentially missed related work** report (these are SUGGESTIONS, not \
   definitive omissions — the authors may have good reasons for not citing them)

Your job is to synthesize these into ONE authoritative final review.

Cross-check every criticism against the actual paper content and the other \
reviews. Remove criticisms that are factually wrong about the paper, that \
misunderstand the contribution, or that are pure formatting/style nitpicks. \
But do NOT excuse real problems — if the critic and neutral reviewer both \
flag the same issue, it's real.

Rules:
- REMOVE criticisms that are factually wrong or misunderstand the paper.
- REMOVE pure formatting/style nitpicks.
- KEEP criticisms that are factually correct AND substantive, even if only \
  one reviewer raised them — a single valid concern still counts.
- KEEP genuine strengths backed by evidence.
- For potentially missed related work: present as suggestions, do not penalize.

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
One paragraph synthesizing genuinely novel observations.

## Potentially Missed Related Work
- paper — why relevant (or "None identified")

## Suggestions
- specific actionable suggestion

Do NOT output any numerical scores or subscores. Do NOT output an accept/reject \
decision. Your job is ONLY to produce the qualitative review. Scoring will be \
done separately.

When you are later asked to score: be DISCRIMINATIVE. A weak paper is weak — \
give it a low score (1-3). A strong paper is strong — give it a high score (7-9). \
Do not cluster everything around 5. The quality difference between papers is real \
and your scores should reflect it.
"""



SCORE_PROMPT = """\
You previously wrote a consolidated review of a paper. Now you must assign \
a calibrated overall score from 1.0 to 10.0 using COMPARATIVE SCORING \
against calibration examples. Note that the review should be written to be very harsh, \
but that doesn't mean the score should be low. Compare with similar level paper in the calibration \
set to determine the score.

Base your score on YOUR OWN review above — the strengths, weaknesses, \
nice-to-haves, and suggestions you already identified. Do not re-evaluate \
from scratch.

## Comparative Scoring Procedure (MANDATORY when calibration examples exist)

You MUST follow these steps in order:

**Step 1 — Identify the LOWER BOUND paper.**
Find the calibration example that is clearly WORSE than or roughly equal to \
the current paper. This is the paper whose quality is just below the current \
paper's level. Note its human average score — this is your score floor.

**Step 2 — Identify the UPPER BOUND paper.**
Find the calibration example that is clearly BETTER than or roughly equal to \
the current paper. This is the paper whose quality is just above the current \
paper's level. Note its human average score — this is your score ceiling.

**Step 3 — Compare dimension by dimension.**
For BOTH the lower and upper bound papers, compare against the current paper on:
- novelty
- technical soundness
- empirical support
- significance
- clarity

For each dimension, determine: is the current paper closer to the lower bound \
or the upper bound?

**Step 4 — Interpolate the final score.**
Place the current paper's score BETWEEN the lower bound and upper bound scores \
based on the dimension-by-dimension comparison. If the current paper is closer \
to the upper bound on most dimensions, score it closer to the upper bound's \
human score; if closer to the lower bound, score it closer to that.

**Edge cases:**
- If the current paper is WORSE than ALL calibration examples, score it below \
  the lowest calibration score.
- If the current paper is BETTER than ALL calibration examples, score it above \
  the highest calibration score.
- If there is only one calibration example nearby, use it as a single anchor \
  and adjust up or down based on the comparison.

## Scoring Constraints

The "score" field is a CONTINUOUS value from 1.0 to 10.0 (e.g. 3.5, 4.7, 6.2, 8.1). \
Use the full range — do NOT cluster around 5-6. Be DISCRIMINATIVE:
- A truly bad paper deserves a 2.0, not a 4.5.
- A truly great paper deserves a 9.0, not a 7.0.
- Do NOT hedge toward the middle. Commit to your assessment.
- However, do not over-penalize papers for a long list of minor points if the core contribution is sound.
- A few serious flaws matter more than many small nitpicks.
- NEVER give a 6.0 score, no matter what. A score of 6 is a non-committal fence-sit. If the \
  paper is even slightly positive, give 6.5 or 7. If it is even slightly negative, \
  give 5.5 or 5. You must decide which side of the borderline the paper falls on. 

Scoring guide (use only when NO calibration examples are available):
- 9.0-10.0: Strong accept. Exceptional, field-advancing contribution.
- 7.0-8.9:  Accept. Clear contribution, solid execution, minor issues.
- 5.0-5.9:  Borderline reject. Has some merit but weaknesses outweigh.
- 3.0-4.9:  Reject. Significant issues with claims, method, or evaluation.
- 1.0-2.9:  Strong reject. Fundamental flaws, unclear contribution, or wrong.

Real flaws go in "weaknesses" and hurt the score. \
Nice-to-haves could affect the scores but not significantly as weaknesses. \
But be honest — missing baselines, unsupported claims, and flawed experiments \
are real weaknesses, not nice-to-haves.

## Fairness Check

Before deciding on the final score, ask:
- Are the main concerns actually central to the paper's claims?
- Would these concerns realistically cause rejection at the target venue?
- Is the paper still a meaningful contribution despite its weaknesses?

Return the score using the provided structured response schema.
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
REASONING_MODELS = {"z-ai/glm-5", "minimax/minimax-m2.7"}

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
            extra = _build_extra_body(model, reasoning_effort="low")
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
            {"role": "system", "content": "Extract the numerical score from the text."},
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


async def _score_with_agent_sdk(
    system_prompt: str,
    review_text: str,
    score_user: str,
    model: str = "claude-opus-4-6",
) -> tuple[str, float]:
    """Use Claude Agent SDK (agent loop, no tools, max_turns=1) to score a paper."""
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock

    prompt = (
        "Here is a consolidated review of a paper (you do NOT have the full paper):\n\n"
        "--- CONSOLIDATED REVIEW ---\n"
        f"{review_text}\n"
        "--- END CONSOLIDATED REVIEW ---\n\n"
        f"{score_user}"
    )

    options = ClaudeAgentOptions(
        model=model,
        allowed_tools=[],
        max_turns=1,
        system_prompt=system_prompt,
    )

    score_text = ""
    cost = 0.0

    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        score_text += block.text
            if isinstance(message, ResultMessage):
                if message.total_cost_usd is not None:
                    cost = message.total_cost_usd

    if not score_text.strip():
        raise ValueError("Agent SDK scorer returned empty response")

    print(f"  [merger_score] done — {model} (Agent SDK) — ${cost:.4f}")
    return score_text, cost


async def run_merger(
    client: AsyncOpenAI,
    harsh_review: str,
    neutral_review: str,
    spark_review: str,
    related_work: str,
    paper_content: str,
    calibration_context: str = "",
) -> tuple[str, float, float]:
    """
    Two-turn merger:
      Turn 1 — produce the consolidated review (markdown, no scores).
      Turn 2 — score the paper with calibration context.
      Then parse the score with a lightweight model.

    Returns (review_text, score, total_cost).
    """
    print(f"  [merger] started ({MODEL_MERGER}) ...")

    # ── Turn 1: Review only (free text) ──────────────────────────────
    user_prompt_review = (
        f"Here is the paper being reviewed (extracted from PDF — formatting "
        f"artifacts are parser issues, not paper problems):\n\n"
        f"--- PAPER CONTENT START ---\n"
        f"{paper_content}\n"
        f"--- PAPER CONTENT END ---\n\n"
        f"Here are the four inputs:\n\n"
        f"# Review 1: Harsh Critic\n{harsh_review}\n\n"
        f"# Review 2: Positive-Leaning Reviewer\n{neutral_review}\n\n"
        f"# Review 3: Spark Finder\n{spark_review}\n\n"
        f"# Report 4: Potentially Missed Related Work\n"
        f"(NOTE: These are SUGGESTIONS only. The search agent may have found \n"
        f"works that are not truly missed or are only tangentially related.)\n"
        f"{related_work}\n\n"
        f"Now produce the final consolidated review following your instructions. "
        f"Remember: many of the harsh critic's points may be nonsensical or overly "
        f"picky — cross-check everything against the actual paper before including it."
    )
    review_text, cost_review = await _call_openai(
        client, "merger", MERGER_PROMPT, user_prompt_review, MODEL_MERGER,
    )

    # ── Turn 2: Score (same conversation, calibration injected) ──────
    print(f"  [merger_score] scoring with calibration ({MODEL_SCORER}) ...")

    score_user = ""
    if calibration_context:
        score_user += (
            f"Now score this paper. Here are calibration examples — reviews of \n"
            f"other papers paired with ACTUAL human reviewer scores. Use these as \n"
            f"your primary scoring anchor:\n\n"
            f"--- CALIBRATION EXAMPLES ---\n"
            f"{calibration_context}\n"
            f"--- END CALIBRATION EXAMPLES ---\n\n"
        )
    else:
        score_user += "Now score this paper.\n\n"

    score_user += (
        "Based on the consolidated review above (you do NOT have the full paper, "
        "only the review), assign a single overall score.\n\n"
        "IMPORTANT — use the FULL range from 1.0 to 10.0. Do NOT compress "
        "scores into 4-6. A paper with fundamental flaws deserves 1-3. "
        "A strong paper with clear contributions deserves 7-9. "
        "Commit to your assessment — do not hedge toward the middle."
    )



    # Strip full paper from scoring context — the review already covers it
    # and sending it wastes tokens / distracts the scorer.
    if "--- PAPER CONTENT END ---" in user_prompt_review:
        user_prompt_review = user_prompt_review.split("--- PAPER CONTENT END ---\n\n", 1)[1]

    NO_SIX_NUDGE = (
        "You gave a score of exactly 6.0. A score of 6 is a non-committal fence-sit. "
        "You MUST pick a side. If the paper is even slightly above average, give 6.5 or 7. "
        "If it is even slightly below, give 5 or 5.5. Re-read your review and commit to "
        "a non-6.0 score now."
    )

    # ── Agent SDK scoring path (MODEL_SCORER = "claude-sdk:<model>") ──
    if MODEL_SCORER.startswith("claude-sdk:"):
        sdk_model = MODEL_SCORER.split(":", 1)[1]
        print(f"  [merger_score] scoring with Agent SDK ({sdk_model}) ...")
        score_text, cost_score = await _score_with_agent_sdk(
            MERGER_PROMPT, review_text, score_user, sdk_model
        )
        score, cost_parse = await _parse_score(client, score_text)
        total_cost = cost_review + cost_score + cost_parse
        print(f"  [score_parser] parsed score: {score} — ${cost_parse:.4f}")

        # Re-score if exactly 6.0
        if abs(score - 6.0) < 0.01:
            print(f"  [merger_score] score is 6.0, re-scoring with nudge ...")
            rescore_prompt = (
                f"Your previous scoring response was:\n{score_text}\n\n{NO_SIX_NUDGE}"
            )
            rescore_text, rescore_cost = await _score_with_agent_sdk(
                MERGER_PROMPT, review_text, rescore_prompt, sdk_model
            )
            score, parse_cost2 = await _parse_score(client, rescore_text)
            total_cost += rescore_cost + parse_cost2
            print(f"  [score_parser] re-scored: {score} — ${parse_cost2:.4f}")

        return review_text, score, total_cost

    # ── OpenRouter scoring path ───────────────────────────────────
    # Multi-turn: system + turn1 + assistant + turn2
    messages = [
        {"role": "system", "content": MERGER_PROMPT},
        {"role": "user", "content": user_prompt_review},
        {"role": "assistant", "content": review_text},
        {"role": "user", "content": score_user},
    ]

    # Call for score (free text)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=MODEL_SCORER,
                messages=messages,
                timeout=REQUEST_TIMEOUT,
            )
            extra = _build_extra_body(MODEL_SCORER, reasoning_effort="high")
            if extra:
                kwargs["extra_body"] = extra
            response = await client.chat.completions.create(**kwargs)
            score_text = response.choices[0].message.content or ""
            cost_score = _extract_cost(response)
            if not score_text.strip():
                _error_logger.error(f"[merger_score] empty response (attempt {attempt}/{MAX_RETRIES}), model={MODEL_SCORER}")
                if attempt < MAX_RETRIES:
                    print(f"  [merger_score] empty response (attempt {attempt}/{MAX_RETRIES}), retrying ...")
                    await asyncio.sleep(RETRY_DELAY + _random.uniform(0, 5))
                    continue
                raise ValueError("merger_score returned empty response")
            # Parse score using lightweight model
            score, cost_parse = await _parse_score(client, score_text)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            output_tokens = getattr(usage, "completion_tokens", None) if usage else None
            tokens = f"{input_tokens}in/{output_tokens}out" if input_tokens and output_tokens else "n/a"
            print(f"  [merger_score] done — {MODEL_SCORER} — {tokens} tokens — ${cost_score:.4f}")
            print(f"  [score_parser] parsed score: {score} — ${cost_parse:.4f}")

            # Re-score if exactly 6.0
            if abs(score - 6.0) < 0.01:
                print(f"  [merger_score] score is 6.0, re-scoring with nudge ...")
                messages.append({"role": "assistant", "content": score_text})
                messages.append({"role": "user", "content": NO_SIX_NUDGE})
                kwargs2 = dict(model=MODEL_SCORER, messages=messages, timeout=REQUEST_TIMEOUT)
                extra2 = _build_extra_body(MODEL_SCORER, reasoning_effort="high")
                if extra2:
                    kwargs2["extra_body"] = extra2
                response2 = await client.chat.completions.create(**kwargs2)
                rescore_text = response2.choices[0].message.content or ""
                cost_score += _extract_cost(response2)
                if rescore_text.strip():
                    score, cost_parse2 = await _parse_score(client, rescore_text)
                    cost_parse += cost_parse2
                    print(f"  [score_parser] re-scored: {score} — ${cost_parse2:.4f}")

            return review_text, score, cost_review + cost_score + cost_parse
        except APITimeoutError as e:
            _error_logger.error(f"[merger_score] timeout (attempt {attempt}/{MAX_RETRIES}), model={MODEL_SCORER}\n{traceback.format_exc()}")
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [merger_score] timeout (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ...")
                await asyncio.sleep(wait)
                continue
            raise
        except Exception as e:
            _error_logger.error(f"[merger_score] error (attempt {attempt}/{MAX_RETRIES}), model={MODEL_SCORER}: {e}\n{traceback.format_exc()}")
            err_str = str(e).lower()
            is_retryable = any(
                kw in err_str for kw in ["rate_limit", "overloaded", "429", "529", "timeout", "gateway", "502", "503", "504"]
            )
            if is_retryable and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [merger_score] transient error (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ...")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("merger_score failed after all retries")


# ── Main orchestration ────────────────────────────────────────────────

async def review_paper(
    paper_path: str,
    parallel: bool = False,
    skip_related_work: bool = False,
    skip_spark: bool = False,
    venue: str = "",
    calibration_context: str = "",
    api_key: str | None = None,
) -> str:
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
    print(f"Loaded paper: {path.name} ({len(paper_content):,} chars)")
    print(f"Mode: {'parallel' if parallel else 'sequential'}")
    print(f"Related work: {'disabled' if skip_related_work else 'enabled'}")
    print(f"Spark finder: {'disabled' if skip_spark else 'enabled'}")
    if venue:
        print(f"Venue: {venue}")
    print(f"Models:")
    print(f"  Harsh Critic:   {MODEL_HARSH}")
    print(f"  Neutral:        {MODEL_NEUTRAL}")
    if not skip_spark:
        print(f"  Spark Finder:   {MODEL_SPARK}")
    if not skip_related_work:
        print(f"  Related Work:   {MODEL_RELATED_WORK}")
    print(f"  Merger:         {MODEL_MERGER}")
    print(f"  Scorer:         {MODEL_SCORER}\n")

    client = _get_client(api_key=api_key)
    pp = str(path)

    # ── Phase 1: All reviewers (parallel or sequential) ───────────
    total_cost = 0.0
    if parallel:
        tasks = [
            run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue=venue),
            run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue=venue),
        ]
        if not skip_spark:
            tasks.append(run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue=venue))
        if not skip_related_work:
            tasks.append(run_related_work_search(client, paper_content))

        print("Phase 1: All reviewers in parallel ...")
        results_list = await asyncio.gather(*tasks)

        idx = 0
        harsh_review, c = results_list[idx]; total_cost += c; idx += 1
        neutral_review, c = results_list[idx]; total_cost += c; idx += 1
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
        neutral_review, c = await run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue=venue)
        total_cost += c
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
    parallel: bool = False,
    skip_related_work: bool = False,
    skip_spark: bool = False,
    venue: str = "",
    calibration_context: str = "",
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
        venue=venue,
        calibration_context=calibration_context,
        api_key=api_key,
    )
    return result, str(input_path.with_name(f"{input_path.stem}_review.md"))


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python paper_reviewer.py <paper.txt> [options]")
        print()
        print("Flags:")
        print("  --parallel          Run agents in parallel")
        print("  --no-related-work   Skip related work search & filter")
        print("  --no-spark          Skip spark finder agent")
        print("  --venue <name>      Set venue (e.g. ICLR, NeurIPS, ICML)")
        print()
        print("Environment variables (or set in .env):")
        print("  OPENROUTER_API_KEY   (required) Your OpenRouter API key")
        print("  ANTHROPIC_API_KEY    (required if MODEL_SCORER uses claude-sdk:)")
        print()
        print("Models per stage:")
        print(f"  Harsh Critic (OpenRouter):      {MODEL_HARSH}")
        print(f"  Neutral (OpenRouter):           {MODEL_NEUTRAL}")
        print(f"  Spark Finder (OpenRouter):      {MODEL_SPARK}")
        print(f"  Related Work (OpenRouter):      {MODEL_RELATED_WORK}")
        print(f"  Merger (OpenRouter):            {MODEL_MERGER}")
        print(f"  Scorer:                         {MODEL_SCORER}")
        print()
        print("To use Claude Agent SDK for scoring, set MODEL_SCORER to")
        print("  claude-sdk:<model>  e.g. claude-sdk:claude-opus-4-6")
        sys.exit(0 if "--help" in sys.argv else 1)

    parallel = "--parallel" in sys.argv
    skip_related = "--no-related-work" in sys.argv
    skip_spark = "--no-spark" in sys.argv
    venue = ""
    if "--venue" in sys.argv:
        idx = sys.argv.index("--venue")
        if idx + 1 < len(sys.argv):
            venue = sys.argv[idx + 1]
    paper_file = [a for a in sys.argv[1:] if not a.startswith("--") and a != venue][0]
    result, total_cost = asyncio.run(review_paper(paper_file, parallel=parallel, skip_related_work=skip_related, skip_spark=skip_spark, venue=venue))
    print(result)
