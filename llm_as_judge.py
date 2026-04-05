system_prompt = """
# LLM-as-Judge Prompt for AI Review Agent Evaluation

You are evaluating the quality of an AI-generated peer review. You will receive the original paper and the generated review. Your job is to check the review for common failure modes, not to judge the paper itself.

For each failure mode, check whether it occurs anywhere in the review. If it does, flag it and quote the offending part(s).

## Failure Modes to Check

### 1. Misunderstanding
Does the review mischaracterize, misunderstand, or strawman what the paper actually does? Compare the review's description of the method/contribution against what the paper states. A review that critiques something the paper doesn't claim is a misunderstanding.

### 2. Generic Comments
Apply this test to each strength and weakness: remove the paper title and all specific references (method names, dataset names, table/figure numbers). If the comment still reads as a valid critique of a random paper, it is generic. Examples: "the evaluation could be more extensive," "the paper makes an important contribution to the field," "the related work section is missing some references."

### 3. Scope Creep
Does any criticism ask the paper to be a different paper? First identify the paper's stated scope and contributions from its introduction/abstract. Then check if any weakness asks the authors to address something outside that scope. Example: a paper about indoor scene recognition being criticized for not handling outdoor scenes.

### 4. Non-Actionable Criticism
Does any weakness describe a problem the authors cannot meaningfully address without changing the paper's fundamental research direction? This includes requesting entirely new experiments on different domains, asking for a different methodology, or vague complaints with no identifiable fix. Note: the review does NOT need to provide explicit solutions — the criticism just needs to be something the authors could reasonably act on.

### 5. Missing Real Problems
Does the review only address surface-level issues (formatting, wording, minor presentation) while ignoring substantive concerns like missing baselines, unjustified claims, flawed experimental design, or logical gaps in the argument? If the review's weaknesses are all things that wouldn't change the paper's conclusions even if fixed, flag this.

### 6. Severity Miscalibration
Does the review treat minor issues as major flaws, or dismiss significant problems as minor? A missing ablation that could invalidate the main claim is not a "minor weakness." A typo in a table caption is not a "significant concern."

### 7. Score Consistency
Two separate checks:
#### 7a. Internal Consistency: Does the review's own content justify the score it gives? A review that lists mostly strengths with minor weaknesses but gives a low score is internally inconsistent. Similarly, a review that identifies fundamental flaws but gives a high score is inconsistent. Ignore the GT score for this check — only look at whether the review's arguments match its own conclusion.
#### 7b. GT Deviation: Compare the review's score against the ground truth score. If the deviation is large (3+ points on a 1-10 scale), check whether the review's content explains the gap. A large deviation is not automatically a failure — the review might have identified real issues that human reviewers missed, or vice versa. Flag this only if the deviation is large AND the review's reasoning does not provide a plausible explanation for it. Also consider whether the deviation stems from other failure modes (misunderstanding, scope creep, generic criticism).


`failure_count` is the number of failure modes flagged as true.
"""



user_prompt = """
Here is the paper:
<paper>
{paper_content}
</paper>

Here is the AI-generated review to evaluate:
<review>
{review_content}
</review>

Check this review against all failure modes. Output JSON only."""

import os

from typing import Optional
from pydantic import BaseModel
from openai import APITimeoutError, OpenAI
from paper_reviewer import ScoreSchema
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"



judge_model = "openai/gpt-5.4"
resolved_api_key = OPENROUTER_API_KEY
if not resolved_api_key:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable not set.\n"
        "Set it in .env or export it."
    )
client = OpenAI(api_key=resolved_api_key, base_url=OPENROUTER_BASE_URL)
import time


class SingleEvidence(BaseModel):
    flagged: bool
    evidence: Optional[str] = None

class ListEvidence(BaseModel):
    flagged: bool
    evidence: list[str] = []

class ScoreInternalConsistency(BaseModel):
    flagged: bool
    agent_score: float
    evidence: Optional[str] = None

class ScoreGTDeviation(BaseModel):
    flagged: bool
    agent_score: float
    gt_score: float
    deviation: float
    evidence: Optional[str] = None

class JudgeResult(BaseModel):
    misunderstanding: SingleEvidence
    generic_comments: ListEvidence
    scope_creep: ListEvidence
    non_actionable: ListEvidence
    missing_real_problems: SingleEvidence
    severity_miscalibration: ListEvidence
    score_internal_consistency: ScoreInternalConsistency
    score_gt_deviation: ScoreGTDeviation
    summary: str
    failure_count: int

class JudgeFormat(BaseModel):
    result: JudgeResult


    
def judge_review(paper_content, review_content):
    while 1:
        try:
            kwargs = dict(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(paper_content=paper_content, review_content=review_content)},
                ],
                timeout=120,
                response_format=JudgeFormat,
            )
            
            response = client.beta.chat.completions.parse(**kwargs)
            return response.choices[0].message.content
        except APITimeoutError:
            print("API call timed out. Retrying...")
            time.sleep(5)  # Wait before retrying

import json
# baseline
if __name__ == "__main__":
    print("Running Baseline Evaluation...")
    with open("baselines/direct_review/direct_baseline_results.md", "r") as f:
        content = f.read()
    
    reviews = content.split("---")[:-1]
    for review in reviews:
        paper_id = review.split("- GT: ")[0].strip().split("##")[1].strip()
        with open("iclr2026_unbalanced/papers/{}.txt".format(paper_id), "r") as f:
            paper_content = f.read()

        review_content = review
        result = judge_review(paper_content, review_content)
        print(f"Paper ID: {paper_id}")

        with open("baselines/direct_review/judge.md", "a") as f:
            f.write(f"## Paper ID: {paper_id}\n")
            f.write(f"### Judge Result:\n")
            json.dump(json.loads(result), f, indent=4)
            f.write("---\n")

