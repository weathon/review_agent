"""
Multi-Agent Paper Reviewer — hybrid Claude Code SDK + OpenRouter.

Pipeline (5 agents, each with its own model):
  1. Harsh Critic (claude-opus-4.6 via Claude Code SDK — free)
  2. Neutral Reviewer (glm-5 via OpenRouter)
  3. Spark Finder (claude-opus-4.6 via Claude Code SDK — free)
  4. Related Work Scout (perplexity/sonar-pro via OpenRouter)
     → filtered by glm-5 to remove already-cited & loosely related work
  5. Merger (claude-opus-4.6 via Claude Code SDK — free)

Claude calls (critic + merger) go through Claude Code SDK (no API cost).
All other calls go through OpenRouter.

Usage:
  python paper_reviewer.py <paper.txt>                 # sequential (default)
  python paper_reviewer.py <paper.txt> --parallel      # parallel non-Claude agents
"""

import asyncio
import os
import random as _random
import sys
from pathlib import Path


from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()  # loads .env from cwd or parent dirs

# ── Config ────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Per-stage model assignments — all via OpenRouter
MODEL_HARSH = "openai/gpt-5.4"
MODEL_NEUTRAL = "z-ai/glm-5"
MODEL_SPARK = "openai/gpt-5.4"
MODEL_RELATED_WORK = "perplexity/sonar-pro"
MODEL_FILTER = "z-ai/glm-5"
MODEL_MERGER = "openai/gpt-5.4"

MAX_RETRIES = 3
RETRY_DELAY = 10

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

### Related Work
- Are the most relevant baselines and prior work cited?
- Is the positioning against prior work fair and accurate?
- Are there obvious missing references?

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
You are a constructive research advisor, NOT a critic. Your job is to identify \
what is MISSING or INCOMPLETE in this paper — not as weaknesses, but as \
opportunities that would make the work significantly stronger.

Think of yourself as a senior collaborator who has read the paper and is now \
brainstorming with the authors about what to add next.

Your job:
- What experiments are missing that would strengthen the claims? Be specific \
  about datasets, baselines, ablations, or analysis types.
- What theoretical insights or analysis could deepen the contribution? \
  (e.g., convergence guarantees, complexity analysis, connections to known results)
- What additional applications or domains could this method be tested on?
- What visualizations, case studies, or qualitative analysis would help the \
  reader understand when/why the method works or fails?
- What are natural next steps that build directly on this work?

Frame everything POSITIVELY — "the paper would be stronger with X" not \
"the paper is weak because it lacks X". These are suggestions to help \
the authors, not ammunition for rejection.

Output format (strictly follow):
## Strengthening Opportunities

### Missing Experiments
1. ... (be specific: what dataset, what baseline, what would it show)

### Deeper Analysis Needed
1. ... (theoretical insights, ablations, or understanding that's absent)

### Untapped Applications
1. ... (domains or use cases the authors haven't explored)

### Visualizations & Case Studies
1. ... (what would help the reader understand the method better)

### Natural Next Steps
1. ... (what should the authors work on next, building on this paper)
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

You MUST output your final review as a single JSON object (no markdown, no \
extra text before or after). Use this exact schema:

{
  "summary": "2-3 sentence paper summary",
  "strengths": ["strength 1 with evidence", "strength 2 with evidence"],
  "weaknesses": ["REAL weakness 1 — affects the core contribution", "REAL weakness 2"],
  "nice_to_haves": ["would improve but not required 1", "out-of-scope suggestion 2"],
  "novel_insights": "synthesized from spark finder, grounded observations only",
  "missed_related_work": ["paper 1 — why relevant", "paper 2 — why relevant"],
  "suggestions": ["suggestion 1", "suggestion 2"],
}
"""



SCORE_PROMPT = """\
You are a paper review agent. You have received four inputs \
about the same paper:

1. A **harsh critic** review (may be overly critical)
2. A **neutral/balanced** review
3. A **spark finder** report (focuses on insights, not flaws)
4. A **potentially missed related work** report (these are SUGGESTIONS, not \
   definitive omissions — the authors may have good reasons for not citing them)
5. A final consolidated review


CALIBRATION — you MUST be realistic about quality. Most papers submitted to \
top venues are NOT strong accepts. The score distribution at ICLR is roughly:
- ~5% score 8+ (strong accept)
- ~25% score 6 (borderline accept)
- ~40% score 5 (borderline reject)
- ~30% score 3 or below (clear reject)
If a paper has weak experiments, unclear contributions, or incremental novelty, \
give it a LOW score. Do NOT inflate scores out of politeness. A 5.0 is not a \
bad score — it means the paper has some merit but real issues.


The "score" field is a CONTINUOUS value from 1.0 to 10.0 (e.g. 3.5, 4.7, 6.2, 8.1). \
Use the full range — do NOT cluster around 5-6. Be DISCRIMINATIVE:
- A truly bad paper deserves a 2.0, not a 4.5.
- A truly great paper deserves a 9.0, not a 7.0.
- Do NOT hedge toward the middle. Commit to your assessment.

Scoring guide:
- 9.0-10.0: Strong accept. Exceptional, field-advancing contribution.
- 7.0-8.9:  Accept. Clear contribution, solid execution, minor issues.
- 5.5-6.9:  Borderline. Has merit but real weaknesses hold it back.
- 3.5-5.4:  Reject. Significant issues with claims, method, or evaluation.
- 1.0-3.4:  Strong reject. Fundamental flaws, unclear contribution, or wrong.

The "decision" field MUST be exactly "Accept" or "Reject".

SCOPE CHECK — for each weakness, ask: is this a real flaw in the paper's \
claims, or just something extra that would be nice to have? \
Real flaws go in "weaknesses" and hurt the score. \
Nice-to-haves go in "nice_to_haves" and it could affect the scores but not significantly as weaknesses. \
But be honest — missing baselines, unsupported claims, and flawed experiments \
are real weaknesses, not nice-to-haves.


You will also be given a set of calibration examples (if available) — these are examples \
with same structure as above, but with a list of actual human scores and decisions for each paper. \
Use these to calibrate your scoring — they show what real scores look like for different quality levels.

Output format: a single float only for the score. 
"""

# ── Core logic ────────────────────────────────────────────────────────

def sanitize_text(text: str) -> str:
    """Remove null bytes and other problematic characters from text."""
    return text.replace("\x00", "")


def _get_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set.\n"
            "Set it in .env or export it. Get one at https://openrouter.ai/keys"
        )
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


# ── OpenRouter call (all models) ──────────────────────────────────────

# Models that support reasoning effort parameter
REASONING_MODELS = {"openai/gpt-5.4", "openai/o3-mini", "openai/o3", "openai/o4-mini"}

async def _call_openrouter(
    client: AsyncOpenAI,
    name: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> str:
    """Call OpenRouter with retry logic for rate limits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
            )
            # Add reasoning effort for supported models
            if model in REASONING_MODELS:
                kwargs["extra_body"] = {"reasoning": {"effort": "high"}}
            response = await client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content or ""
            usage = response.usage
            tokens = f"{usage.prompt_tokens}in/{usage.completion_tokens}out" if usage else "n/a"
            model_short = model.split("/")[-1]
            if not result.strip():
                if attempt < MAX_RETRIES:
                    print(f"  [{name}] empty response (attempt {attempt}/{MAX_RETRIES}), retrying ...")
                    await asyncio.sleep(RETRY_DELAY + _random.uniform(0, 5))
                    continue
                print(f"  [{name}] empty response after {MAX_RETRIES} attempts")
            print(f"  [{name}] done — {model_short} (OpenRouter) — {tokens} tokens")
            return result
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(
                kw in err_str for kw in ["rate_limit", "overloaded", "429", "529", "timeout"]
            )
            if is_retryable and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{name}] rate limited (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ...")
                await asyncio.sleep(wait)
            else:
                raise
    return ""


# ── Agent runners ─────────────────────────────────────────────────────

async def run_reviewer(
    client: AsyncOpenAI,
    name: str,
    system_prompt: str,
    paper_path: str,
    paper_content: str,
    model: str,
    venue: str = "",
) -> str:
    """Run a reviewer via OpenRouter."""
    print(f"  [{name}] started ({model.split('/')[-1]}) ...")
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
    return await _call_openrouter(client, name, system_prompt, user_prompt, model)


async def run_related_work_search(
    client: AsyncOpenAI,
    paper_content: str,
) -> str:
    """
    Two-step related work pipeline (both via OpenRouter):
    1. Perplexity sonar-pro searches for related papers
    2. kimi-k2.5 filters already-cited and loosely related ones
    """
    abstract_section = paper_content[:3000]

    print("  [related_work_search] started (OpenRouter) ...")
    raw_results = await _call_openrouter(
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
    filtered = await _call_openrouter(
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

    return filtered


async def run_merger(
    client: AsyncOpenAI,
    harsh_review: str,
    neutral_review: str,
    spark_review: str,
    related_work: str,
    paper_content: str,
    calibration_context: str = "",
) -> str:
    """Run the merger via OpenRouter."""
    print(f"  [merger] started ({MODEL_MERGER.split('/')[-1]}) ...")

    calibration_block = ""
    # if calibration_context:
    #     calibration_block = (
    #         f"Here are examples of sub-agent reviews for other papers, paired with\n"
    #         f"the ACTUAL human reviewer scores and decisions. Use these to calibrate\n"
    #         f"your scoring — they show what real scores look like for different\n"
    #         f"quality levels:\n\n"
    #         f"--- CALIBRATION EXAMPLES ---\n"
    #         f"{calibration_context}\n"
    #         f"--- END CALIBRATION EXAMPLES ---\n\n"
    #         f"Now review the current paper:\n\n"
    #     )

    user_prompt = (
        # f"{calibration_block}"
        f"Here is the paper being reviewed (extracted from PDF — formatting "
        f"artifacts are parser issues, not paper problems):\n\n"
        f"--- PAPER CONTENT START ---\n"
        f"{paper_content}\n"
        f"--- PAPER CONTENT END ---\n\n"
        f"Here are the four inputs:\n\n"
        f"# Review 1: Harsh Critic\n{harsh_review}\n\n"
        f"# Review 2: Neutral Reviewer\n{neutral_review}\n\n"
        f"# Review 3: Spark Finder\n{spark_review}\n\n"
        f"# Report 4: Potentially Missed Related Work\n"
        f"(NOTE: These are SUGGESTIONS only. The search agent may have found \n"
        f"works that are not truly missed or are only tangentially related. \n"
        # f"Do NOT penalize the paper's score for these.)\n"
        f"{related_work}\n\n"
        f"Now produce the final consolidated review following your instructions. "
        f"Remember: many of the harsh critic's points may be nonsensical or overly "
        f"picky — cross-check everything against the actual paper before including it."
    )
    return await _call_openrouter(client, "merger", MERGER_PROMPT, user_prompt, MODEL_MERGER)



async def run_score_predictor(
    client: AsyncOpenAI,
    harsh_review: str,
    neutral_review: str,
    spark_review: str,
    related_work: str,
    final_review: str,
    calibration_context: str = "",
) -> float:
    """Run a score predictor (optional, can be used for calibration)."""
    
    calibration_block = ""
    if calibration_context:
        calibration_block = (
            f"Here are examples of reviews for other papers, paired with\n"
            f"the ACTUAL human reviewer scores and decisions. Use these to calibrate\n"
            f"your scoring — they show what real scores look like for different\n"
            f"quality levels:\n\n"
            f"--- CALIBRATION EXAMPLES ---\n" 
            f"{calibration_context}\n"
            f"--- END CALIBRATION EXAMPLES ---\n\n"
            f"Now review the current paper:\n\n"
            f"Harsh Critic Review:\n{harsh_review}\n\n"
            f"Neutral Review:\n{neutral_review}\n\n"
            f"Spark Finder Review:\n{spark_review}\n\n"
            f"Related Work Review:\n{related_work}\n\n"
            f"Final Review:\n{final_review}\n\n"
        )
    return await _call_openrouter(client, "score_predictor", SCORE_PROMPT, calibration_block, MODEL_MERGER)
    

    

# ── Main orchestration ────────────────────────────────────────────────

async def review_paper(
    paper_path: str,
    parallel: bool = False,
    skip_related_work: bool = False,
    skip_spark: bool = False,
    venue: str = "",
    calibration_context: str = "",
) -> str:
    """
    Main entry point. All agents via OpenRouter — can fully parallelize.

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
    print(f"  Merger:         {MODEL_MERGER}\n")

    client = _get_client()
    pp = str(path)

    # ── Phase 1: All reviewers (parallel or sequential) ───────────
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
        harsh_review = results_list[idx]; idx += 1
        neutral_review = results_list[idx]; idx += 1
        spark_review = results_list[idx] if not skip_spark else "Spark finder was skipped."; idx += (0 if skip_spark else 1)
        related_work = results_list[idx] if not skip_related_work else "Related work search was skipped."
    else:
        print("Phase 1: Reviewers sequentially ...")
        harsh_review = await run_reviewer(client, "harsh_critic", HARSH_CRITIC_PROMPT, pp, paper_content, MODEL_HARSH, venue=venue)
        neutral_review = await run_reviewer(client, "neutral", NEUTRAL_REVIEWER_PROMPT, pp, paper_content, MODEL_NEUTRAL, venue=venue)
        if not skip_spark:
            spark_review = await run_reviewer(client, "spark_finder", SPARK_FINDER_PROMPT, pp, paper_content, MODEL_SPARK, venue=venue)
        else:
            spark_review = "Spark finder was skipped."
        if not skip_related_work:
            related_work = await run_related_work_search(client, paper_content)
        else:
            related_work = "Related work search was skipped."

    # ── Phase 2: Merger (waits for all reviewers) ─────────────────
    print("\nPhase 2: Merger ...")
    final_review = await run_merger(
        client, harsh_review, neutral_review,
        spark_review, related_work, paper_content,
        calibration_context=calibration_context,
    )


    final_score = await run_score_predictor(client, harsh_review, neutral_review,
        spark_review, related_work, final_review, calibration_context=calibration_context) 

    # ── Output ────────────────────────────────────────────────────
    separator = "=" * 72
    full_output = (
        f"\n{separator}\n"
        f"INDIVIDUAL REVIEWS\n"
        f"{separator}\n\n"
        f"{'─' * 40}\n"
        f"HARSH CRITIC ({MODEL_HARSH} via Claude SDK)\n"
        f"{'─' * 40}\n"
        f"{harsh_review}\n\n"
        f"{'─' * 40}\n"
        f"NEUTRAL REVIEWER ({MODEL_NEUTRAL} via OpenRouter)\n"
        f"{'─' * 40}\n"
        f"{neutral_review}\n\n"
        f"{'─' * 40}\n"
        f"SPARK FINDER ({MODEL_SPARK} via Claude SDK)\n"
        f"{'─' * 40}\n"
        f"{spark_review}\n\n"
        f"{'─' * 40}\n"
        f"POTENTIALLY MISSED RELATED WORK ({MODEL_RELATED_WORK} via OpenRouter)\n"
        f"{'─' * 40}\n"
        f"{related_work}\n\n"
        f"{separator}\n"
        f"FINAL CONSOLIDATED REVIEW ({MODEL_MERGER} via Claude SDK)\n"
        f"{separator}\n\n"
        f"{final_review}\n"
    )

    output_path = path.parent / f"{path.stem}_review.md"
    output_path.write_text(full_output, encoding="utf-8")
    print(f"\nReview saved to: {output_path}")

    return full_output


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python paper_reviewer.py <paper.txt> [options]")
        print()
        print("Flags:")
        print("  --parallel          Run OpenRouter agents in parallel")
        print("  --no-related-work   Skip related work search & filter")
        print("  --no-spark          Skip spark finder agent")
        print("  --venue <name>      Set venue (e.g. ICLR, NeurIPS, ICML)")
        print()
        print("Environment variables (or set in .env):")
        print("  OPENROUTER_API_KEY   (required) Your OpenRouter API key")
        print()
        print("Models per stage:")
        print(f"  Harsh Critic (Claude SDK):  {MODEL_HARSH}")
        print(f"  Neutral (OpenRouter):       {MODEL_NEUTRAL}")
        print(f"  Spark Finder (Claude SDK):  {MODEL_SPARK}")
        print(f"  Related Work (OpenRouter):  {MODEL_RELATED_WORK}")
        print(f"  Merger (Claude SDK):        {MODEL_MERGER}")
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
    result = asyncio.run(review_paper(paper_file, parallel=parallel, skip_related_work=skip_related, skip_spark=skip_spark, venue=venue))
    print(result)
