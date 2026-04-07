#!/usr/bin/env python3
"""Generic Claude Agent SDK agent. Reads prompt from file, writes result to file.
The agent gets Read/Glob/Grep tools so it can read files referenced in the prompt.

Usage:
  python agents/claude_agent.py --prompt /tmp/work/prompt.txt \
      --model claude-opus-4-6 --output /tmp/work/result.txt
"""
import asyncio
import sys
from pathlib import Path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, help="Path to prompt file")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name", default="claude-agent")
    p.add_argument("--cwd", default=None)
    p.add_argument("--effort", default="high")
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--tools", default="Read,Glob,Grep", help="Comma-separated tool list")
    args = p.parse_args()

    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    allowed_tools = [t.strip() for t in args.tools.split(",")]

    options = ClaudeAgentOptions(
        model=args.model,
        cwd=args.cwd,
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
        effort=args.effort,
        max_turns=args.max_turns,
    )

    async def _run():
        result = ""
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result += block.text
        return result

    result = asyncio.run(_run())
    Path(args.output).write_text(result, encoding="utf-8")
    print(f"  [{args.name}] done — {args.model} (Claude Agent SDK)")


if __name__ == "__main__":
    main()
