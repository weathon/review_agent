from .common import build_reviewer_query
from .shared_tools import glob_files_text, grep_files_text, read_file_text


def make_claude_file_mcp_server(name: str, allowed_paths: list[str], file_qa_callback=None):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool(
        "read_file",
        "Read a file. Returns the content with line numbers.",
        {"abs_path": str, "start_line": int, "end_line": int},
    )
    async def read_file_tool(args: dict) -> dict:
        text = read_file_text(
            abs_path=args["abs_path"],
            start_line=args.get("start_line", 1) or 1,
            end_line=args.get("end_line", 0) or 0,
            allowed_paths=allowed_paths,
        )
        return {
            "content": [{"type": "text", "text": text}],
            "is_error": text.startswith("ERROR:"),
        }

    @tool(
        "grep_files",
        "Search file contents for a regex pattern.",
        {"pattern": str, "directory": str, "file_glob": str},
    )
    async def grep_files_tool(args: dict) -> dict:
        text = grep_files_text(
            pattern=args["pattern"],
            directory=args.get("directory", "."),
            file_glob=args.get("file_glob", "**/*"),
            allowed_paths=allowed_paths,
        )
        return {
            "content": [{"type": "text", "text": text}],
            "is_error": text.startswith("ERROR:"),
        }

    @tool(
        "glob_files",
        "Find files matching a glob pattern under a directory.",
        {"pattern": str, "directory": str},
    )
    async def glob_files_tool(args: dict) -> dict:
        text = glob_files_text(
            pattern=args["pattern"],
            directory=args.get("directory", "."),
            allowed_paths=allowed_paths,
        )
        return {
            "content": [{"type": "text", "text": text}],
            "is_error": text.startswith("ERROR:"),
        }

    tools = [read_file_tool, grep_files_tool, glob_files_tool]

    if file_qa_callback is not None:
        @tool(
            "file_qa",
            "Answer a question about a file's content.",
            {"abs_path": str, "question": str},
        )
        async def file_qa_tool(args: dict) -> dict:
            text = await file_qa_callback(
                abs_path=args["abs_path"],
                question=args["question"],
                allowed_paths=allowed_paths,
            )
            return {
                "content": [{"type": "text", "text": text}],
                "is_error": text.startswith("ERROR:"),
            }

        tools.append(file_qa_tool)

    return create_sdk_mcp_server(
        name=name,
        version="1.0.0",
        tools=tools,
    )


async def run_claude_agent_task(
    name: str,
    instructions: str,
    user_prompt: str,
    model_id: str,
    allowed_tools=None,
    mcp_servers=None,
    cwd: str | None = None,
    max_turns: int = 30,
) -> tuple[str, float]:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
    )

    print(f"  [{name}] starting Claude Agent SDK ({model_id}) ...")
    result_text = ""
    total_cost = 0.0
    query = f"{instructions}\n\n---\n\n{user_prompt}"
    options = ClaudeAgentOptions(
        model=model_id,
        cwd=cwd,
        allowed_tools=allowed_tools or [],
        disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Agent"],
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
    )
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(query)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text
            if isinstance(message, ResultMessage):
                total_cost += message.total_cost_usd
    if not result_text.strip():
        raise ValueError(f"{name} returned empty output")
    print(f"  [{name}] done — {model_id} (Claude Agent SDK) — saved ${total_cost:.4f}")
    return result_text, total_cost


async def run_claude_reviewer(
    name: str,
    system_prompt: str,
    paper_path: str,
    model_id: str,
    venue: str = "",
    max_turns: int = 30,
) -> tuple[str, float]:
    query, _paper_abs, paper_dir = build_reviewer_query(system_prompt, paper_path, venue)
    reviewer_mcp = make_claude_file_mcp_server("reviewer_fs", [paper_dir])
    return await run_claude_agent_task(
        name=name,
        instructions=system_prompt,
        user_prompt=query,
        model_id=model_id,
        allowed_tools=[
            "mcp__reviewer_fs__read_file",
            "mcp__reviewer_fs__grep_files",
            "mcp__reviewer_fs__glob_files",
        ],
        mcp_servers={"reviewer_fs": reviewer_mcp},
        cwd=paper_dir,
        max_turns=max_turns,
    )
