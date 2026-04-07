#!/usr/bin/env python3
"""Extract score from scorer output using GPT-5.4-nano structured output.

Usage:
  python agents/parse_score.py --input /tmp/work/scorer_output.txt --output /tmp/work/score.txt
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_PARSER = "openai/gpt-5.4-nano"


class ScoreSchema(BaseModel):
    score: float


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parse-prompt", default=None, help="Path to parse prompt file")
    args = p.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    prompts_dir = Path(__file__).parent.parent / "prompts"
    parse_prompt = Path(args.parse_prompt).read_text(encoding="utf-8") if args.parse_prompt else (prompts_dir / "parse_score.txt").read_text(encoding="utf-8")

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    response = client.beta.chat.completions.parse(
        model=MODEL_PARSER,
        messages=[
            {"role": "system", "content": parse_prompt},
            {"role": "user", "content": text},
        ],
        response_format=ScoreSchema,
        timeout=30,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        print("ERROR: score_parser returned no parsed output", file=sys.stderr)
        sys.exit(1)

    Path(args.output).write_text(str(parsed.score), encoding="utf-8")
    print(f"  [parse_score] score: {parsed.score}")


if __name__ == "__main__":
    main()
