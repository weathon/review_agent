#!/usr/bin/env python3
"""Scorer agent: reads review + paper + calibration dir, outputs score.

Usage:
  python agents/scorer.py --review work/merged.txt --paper paper.txt \
      --cal-dir cal/ --output work/score.txt
"""
import asyncio
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class ScoreSchema(BaseModel):
    score: float


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--review", required=True)
    p.add_argument("--paper", required=True)
    p.add_argument("--cal-dir", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="claude-opus-4-6")
    args = p.parse_args()

    review_path = str(Path(args.review).resolve())
    paper_path = str(Path(args.paper).resolve())
    cal_dir = str(Path(args.cal_dir).resolve()) if args.cal_dir else ""

    # Build scorer prompt
    score_prompt = PROMPTS_DIR.joinpath("scorer.txt").read_text(encoding="utf-8")
    template = PROMPTS_DIR.joinpath("scorer_agent.txt").read_text(encoding="utf-8")
    prompt = template.format(score_prompt=score_prompt, review_path=review_path, paper_path=paper_path, cal_dir_abs=cal_dir)

    # Run Claude Agent SDK
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

    async def _run():
        result = ""
        opts = ClaudeAgentOptions(
            model=args.model, cwd=cal_dir or None,
            allowed_tools=["Grep", "Read", "Glob", "Agent"],
            permission_mode="bypassPermissions", effort="medium", max_turns=30,
        )
        async with ClaudeSDKClient(options=opts) as c:
            await c.query(prompt)
            async for msg in c.receive_response():
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock):
                            result += b.text
        return result

    print(f"  [scorer] starting ({args.model}, cal={cal_dir}) ...")
    scorer_output = asyncio.run(_run())

    # Parse score via GPT-5.4-nano
    parse_prompt = PROMPTS_DIR.joinpath("parse_score.txt").read_text(encoding="utf-8")
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    response = client.beta.chat.completions.parse(
        model="openai/gpt-5.4-nano",
        messages=[{"role": "system", "content": parse_prompt}, {"role": "user", "content": scorer_output}],
        response_format=ScoreSchema, timeout=30,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        print("ERROR: score parser returned nothing", file=sys.stderr)
        sys.exit(1)

    Path(args.output).write_text(str(parsed.score), encoding="utf-8")
    print(f"  [scorer] score: {parsed.score}")


if __name__ == "__main__":
    main()
