#!/usr/bin/env python3
"""Merger agent: combines review files into one consolidated review.

Usage:
  python agents/merger.py --paper paper.txt --harsh work/harsh.txt \
      --output work/merged.txt --model deepseek/deepseek-v3.2 \
      [--neutral work/neutral.txt] [--spark work/spark.txt] [--related work/related.txt]
"""
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5
RETRY_DELAY = 10
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

REASONING_MODELS = {"z-ai/glm-5", "minimax/minimax-m2.7", "deepseek/deepseek-v3.2", "minimax/minimax-m2.5:free", "stepfun/step-3.5-flash:free"}
PROVIDER_MAP = {
    "z-ai/glm-5": ["deepinfra/fp4"],
    "z-ai/glm-5:online": ["deepinfra/fp4"],
    "minimax/minimax-m2.7": ["minimax/fp8"],
    "deepseek/deepseek-v3.2": ["parasail/fp8"],
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--paper", required=True)
    p.add_argument("--harsh", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--neutral", default=None)
    p.add_argument("--spark", default=None)
    p.add_argument("--related", default=None)
    args = p.parse_args()

    # Build system prompt from template
    template = PROMPTS_DIR.joinpath("merger.txt").read_text(encoding="utf-8")
    num = 1
    neutral_line = spark_line = related_work_line = ""
    if args.neutral:
        num += 1; neutral_line = f"{num}. A **neutral/balanced** review\n"
    if args.spark:
        num += 1; spark_line = f"{num}. A **spark finder** report (focuses on insights, not flaws)\n"
    if args.related:
        num += 1; related_work_line = f"{num}. A **potentially missed related work** report (these are SUGGESTIONS, not definitive omissions — the authors may have good reasons for not citing them)\n"
    system_prompt = template.format(input_count=num, neutral_line=neutral_line, spark_line=spark_line, related_work_line=related_work_line)

    # Build user prompt from files
    paper_content = Path(args.paper).read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    parts = []
    parts.append(f"Here is the paper being reviewed (extracted from PDF — formatting artifacts are parser issues, not paper problems):\n\n--- PAPER CONTENT START ---\n{paper_content}\n--- PAPER CONTENT END ---\n\nHere are the inputs:\n")

    review_num = 1
    parts.append(f"\n# Review {review_num}: Harsh Critic\n{Path(args.harsh).read_text(encoding='utf-8')}\n")
    if args.neutral:
        review_num += 1
        parts.append(f"\n# Review {review_num}: Positive-Leaning Reviewer\n{Path(args.neutral).read_text(encoding='utf-8')}\n")
    if args.spark:
        review_num += 1
        parts.append(f"\n# Review {review_num}: Spark Finder\n{Path(args.spark).read_text(encoding='utf-8')}\n")
    if args.related:
        review_num += 1
        parts.append(f"\n# Report {review_num}: Potentially Missed Related Work\n(NOTE: These are SUGGESTIONS only.)\n{Path(args.related).read_text(encoding='utf-8')}\n")

    parts.append("\nNow produce the final consolidated review following your instructions. Remember: many of the harsh critic's points may be nonsensical or overly picky — cross-check everything against the actual paper before including it.")
    user_prompt = "".join(parts)

    # Call OpenRouter
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(model=args.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], timeout=120)
            extra = {}
            if args.model in REASONING_MODELS:
                extra["reasoning"] = {"effort": "medium"}
            if args.model in PROVIDER_MAP:
                extra["provider"] = {"only": PROVIDER_MAP[args.model]}
            if extra:
                kwargs["extra_body"] = extra
            result = client.chat.completions.create(**kwargs).choices[0].message.content or ""
            if not result.strip() and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY + random.uniform(0, 5)); continue
            Path(args.output).write_text(result, encoding="utf-8")
            print(f"  [merger] done — {args.model}")
            return
        except APITimeoutError:
            if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY * attempt); continue
            raise
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ["rate_limit", "429", "529", "502", "503", "504", "overloaded"]) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
    sys.exit(1)


if __name__ == "__main__":
    main()
