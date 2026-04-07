#!/usr/bin/env python3
"""Run a single reviewer agent (harsh/neutral/spark).

Usage:
  python agents/reviewer.py --paper paper.txt --prompt prompts/harsh_critic.txt \
      --model deepseek/deepseek-v3.2 --output work/harsh.txt [--venue ICLR]

If --model starts with "claude:", uses Claude Agent SDK (paper read via tool).
Otherwise uses OpenRouter (paper content inlined in prompt).
"""
import os
import random
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5
RETRY_DELAY = 10

REASONING_MODELS = {"z-ai/glm-5", "minimax/minimax-m2.7", "deepseek/deepseek-v3.2", "minimax/minimax-m2.5:free", "stepfun/step-3.5-flash:free"}
PROVIDER_MAP = {
    "z-ai/glm-5": ["deepinfra/fp4"],
    "z-ai/glm-5:online": ["deepinfra/fp4"],
    "minimax/minimax-m2.7": ["minimax/fp8"],
    "deepseek/deepseek-v3.2": ["parasail/fp8"],
}

VENUE_LINE = "This paper was submitted to **{venue}**. You MUST evaluate it against {venue}'s specific standards, acceptance bar, and expectations. Consider what {venue} reviewers typically look for.\n\n"
PARSER_NOTE = "NOTE: This paper was extracted from PDF by an automated parser. There may be formatting artifacts such as broken equations, garbled tables, misplaced figure references, or OCR errors. These are parser issues, NOT problems with the paper itself. Do NOT treat formatting artifacts as weaknesses.\n\n"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--paper", required=True)
    p.add_argument("--prompt", required=True, help="System prompt file")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--venue", default="")
    p.add_argument("--name", default="reviewer")
    args = p.parse_args()

    paper_path = str(Path(args.paper).resolve())
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")
    venue_line = VENUE_LINE.format(venue=args.venue) if args.venue else ""

    if args.model.startswith("claude:"):
        model_id = args.model[len("claude:"):]
        prompt = (
            f"{system_prompt}\n\n---\n\n"
            f"{venue_line}"
            f"Review the following paper thoroughly.\n\n"
            f"{PARSER_NOTE}"
            f"The paper is located at: {paper_path}\n"
            f"Use the Read tool to read the paper file, then produce your review."
        )
        _run_claude(prompt, model_id, args.output, args.name)
    else:
        paper_content = Path(args.paper).read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        user_prompt = (
            f"{venue_line}"
            f"Review the following paper thoroughly.\n\n"
            f"{PARSER_NOTE}"
            f"Paper file: {paper_path}\n\n"
            f"--- PAPER CONTENT START ---\n{paper_content}\n--- PAPER CONTENT END ---"
        )
        _run_openrouter(system_prompt, user_prompt, args.model, args.output, args.name)


def _run_openrouter(system_prompt, user_prompt, model, output, name):
    from openai import APITimeoutError, OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                timeout=120,
            )
            extra = {}
            if model in REASONING_MODELS:
                extra["reasoning"] = {"effort": "medium"}
            if model in PROVIDER_MAP:
                extra["provider"] = {"only": PROVIDER_MAP[model]}
            if extra:
                kwargs["extra_body"] = extra

            result = client.chat.completions.create(**kwargs).choices[0].message.content or ""
            if not result.strip() and attempt < MAX_RETRIES:
                print(f"  [{name}] empty (attempt {attempt}), retrying...", file=sys.stderr)
                time.sleep(RETRY_DELAY + random.uniform(0, 5))
                continue

            Path(output).write_text(result, encoding="utf-8")
            print(f"  [{name}] done — {model}")
            return
        except APITimeoutError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                continue
            raise
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ["rate_limit", "429", "529", "502", "503", "504", "overloaded"]) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
    sys.exit(1)


def _run_claude(prompt, model_id, output, name):
    import asyncio
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

    async def _go():
        result = ""
        opts = ClaudeAgentOptions(model=model_id, allowed_tools=["Read", "Glob", "Grep"], permission_mode="bypassPermissions", max_turns=30)
        async with ClaudeSDKClient(options=opts) as c:
            await c.query(prompt)
            async for msg in c.receive_response():
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock):
                            result += b.text
        return result

    Path(output).write_text(asyncio.run(_go()), encoding="utf-8")
    print(f"  [{name}] done — {model_id} (Claude SDK)")


if __name__ == "__main__":
    main()
