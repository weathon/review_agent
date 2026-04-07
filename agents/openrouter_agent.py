#!/usr/bin/env python3
"""Generic OpenRouter agent. Reads prompts from files, writes result to file.

Usage:
  python agents/openrouter_agent.py --system-prompt prompts/harsh_critic.txt \
      --user-prompt /tmp/work/harsh_user.txt --model deepseek/deepseek-v3.2 \
      --output /tmp/work/harsh_review.txt
"""
import os
import random
import sys
import time
import traceback
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5
RETRY_DELAY = 10
REQUEST_TIMEOUT = 120

REASONING_MODELS = {"z-ai/glm-5", "minimax/minimax-m2.7", "deepseek/deepseek-v3.2", "minimax/minimax-m2.5:free", "stepfun/step-3.5-flash:free"}
PROVIDER_MAP = {
    "z-ai/glm-5": ["deepinfra/fp4"],
    "z-ai/glm-5:online": ["deepinfra/fp4"],
    "minimax/minimax-m2.7": ["minimax/fp8"],
    "deepseek/deepseek-v3.2": ["parasail/fp8"],
}

_error_logger = logging.getLogger("openrouter_agent")
_error_logger.setLevel(logging.ERROR)
_handler = logging.FileHandler(Path(__file__).parent.parent / "error.log", mode="a")
_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_error_logger.addHandler(_handler)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--system-prompt", required=True, help="Path to system prompt file")
    p.add_argument("--user-prompt", required=True, help="Path to user prompt file")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, help="Path to write result")
    p.add_argument("--name", default="agent", help="Name for logging")
    args = p.parse_args()

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    user_prompt = Path(args.user_prompt).read_text(encoding="utf-8")

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=REQUEST_TIMEOUT,
            )
            extra = {}
            if args.model in REASONING_MODELS:
                extra["reasoning"] = {"effort": "medium"}
            if args.model in PROVIDER_MAP:
                extra["provider"] = {"only": PROVIDER_MAP[args.model]}
            if extra:
                kwargs["extra_body"] = extra

            response = client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content or ""

            if not result.strip():
                if attempt < MAX_RETRIES:
                    print(f"  [{args.name}] empty response (attempt {attempt}/{MAX_RETRIES}), retrying ...", file=sys.stderr)
                    time.sleep(RETRY_DELAY + random.uniform(0, 5))
                    continue
                print(f"  [{args.name}] empty response after {MAX_RETRIES} attempts", file=sys.stderr)

            Path(args.output).write_text(result, encoding="utf-8")
            print(f"  [{args.name}] done — {args.model}")
            sys.exit(0)

        except APITimeoutError:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{args.name}] timeout (attempt {attempt}), waiting {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(kw in err_str for kw in ["rate_limit", "overloaded", "429", "529", "timeout", "gateway", "502", "503", "504"])
            if is_retryable and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{args.name}] transient error (attempt {attempt}), waiting {wait}s ...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

    print(f"  [{args.name}] all retries exhausted", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
