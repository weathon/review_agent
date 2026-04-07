"""
LLM-as-Judge: judge whether each weakness point in a review is valid and reasonable.

Given a review file and a paper file, uses Claude Agent SDK to evaluate
each individual weakness point in the review against the actual paper content.

Usage:
  python llm_as_judge_valid.py <review_path> <paper_path>
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


async def judge_weaknesses(review_path: str, paper_path: str) -> str:
    """Judge each weakness point in the review against the paper."""
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

    review_abs = str(Path(review_path).resolve())
    paper_abs = str(Path(paper_path).resolve())

    prompt = f"""\
You are an expert meta-reviewer. Your task is to judge whether each weakness point
in a review is valid and reasonable, given the actual paper.

You have two files to read:
1. The review: {review_abs}
2. The paper: {paper_abs}

Steps:
1. Read the paper thoroughly.
2. Read the review and identify every individual weakness point.
3. For EACH weakness point, judge whether it is:
   - **Reasonable**: The criticism is accurate and well-supported by what the paper actually says/does.
   - **Not so reasonable**: The criticism is factually wrong, misunderstands the paper, or is unreasonable or the criticism has some merit but is exaggerated or partly inaccurate. Or the demanding is not reasonable.

For each weakness point, output:
- The weakness point (quoted or summarized)
- Your verdict
- A brief justification referencing the paper content.


Pay special attention to directionality with upper/lower bounds and performance claims:
- If the reviewer says "the reported accuracy is an upper bound and thus misleading," but the experimental setup actually gives a lower bound (e.g., the authors used a harder split, fewer training samples, or no data augmentation), the criticism is reversed in direction and invalid.
- If the reviewer says "the method's advantage is overstated because the evaluation is too easy," but the benchmark is actually known to be challenging or the authors' setup makes it harder (e.g., zero-shot instead of few-shot), then the result is a lower bound on performance and the criticism is wrong.
- If the reviewer claims "the authors' ablation unfairly favors the proposed component," but removing that component actually makes the task easier for the model (i.e., the ablation gives an upper bound for the baseline), then the comparison is conservative and the criticism is invalid.
- If the reviewer says "using oracle labels makes the evaluation unrealistic and inflates results," but the oracle is applied to the baseline (not the proposed method), then it actually strengthens the baseline and makes the comparison harder for the authors — the criticism is directionally wrong.
- If the reviewer claims "the proposed method has an unfair advantage due to extra training data," but the extra data was also provided to all baselines, or the proposed method was tested without it and still outperformed, then the criticism does not hold.

At the end, output a summary line:
**Summary**: X out of Y weakness points are valid.
"""

    result_text = ""
    options = ClaudeAgentOptions(
        model="claude-opus-4-6",
        allowed_tools=["Read"],
        permission_mode="bypassPermissions",
        effort="high",
        max_turns=20,
    )
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text

    print(result_text)
    return result_text


def main():
    if len(sys.argv) < 3:
        print("Usage: python llm_as_judge_valid.py <review_path> <paper_path>")
        sys.exit(1)

    review_path = sys.argv[1]
    paper_path = sys.argv[2]

    asyncio.run(judge_weaknesses(review_path, paper_path))


if __name__ == "__main__":
    main()

# do not compare with human because authors could already fixed them in camera ready 