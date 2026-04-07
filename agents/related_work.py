#!/usr/bin/env python3
"""Two-step related work agent: search then filter.

Usage:
  python agents/related_work.py --paper paper.txt --output work/related.txt \
      [--search-model deepseek/deepseek-v3.2:online] [--filter-model deepseek/deepseek-v3.2]
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


def call(client, system_prompt, user_prompt, model, name):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], timeout=120)
            extra = {}
            if model in REASONING_MODELS:
                extra["reasoning"] = {"effort": "medium"}
            if model in PROVIDER_MAP:
                extra["provider"] = {"only": PROVIDER_MAP[model]}
            if extra:
                kwargs["extra_body"] = extra
            result = client.chat.completions.create(**kwargs).choices[0].message.content or ""
            if not result.strip() and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY + random.uniform(0, 5)); continue
            print(f"  [{name}] done — {model}")
            return result
        except APITimeoutError:
            if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY * attempt); continue
            raise
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ["rate_limit", "429", "529", "502", "503", "504", "overloaded"]) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
    return ""


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--paper", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--search-model", default="deepseek/deepseek-v3.2:online")
    p.add_argument("--filter-model", default="deepseek/deepseek-v3.2")
    args = p.parse_args()

    paper_content = Path(args.paper).read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    # Step 1: search
    raw = call(client,
        PROMPTS_DIR.joinpath("related_work.txt").read_text(encoding="utf-8"),
        f"Find related work for this paper. Here is the title and abstract:\n\n{paper_content[:3000]}\n\nSearch for real, published papers that are closely related.",
        args.search_model, "related_work_search")

    # Step 2: filter
    filtered = call(client,
        PROMPTS_DIR.joinpath("related_work_filter.txt").read_text(encoding="utf-8"),
        f"Here is the FULL PAPER:\n\n--- PAPER CONTENT START ---\n{paper_content}\n--- PAPER CONTENT END ---\n\nHere are the related works found:\n\n{raw}\n\nFilter out already-cited and loosely related works.",
        args.filter_model, "related_work_filter")

    Path(args.output).write_text(filtered, encoding="utf-8")


if __name__ == "__main__":
    main()
