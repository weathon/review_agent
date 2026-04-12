import os

from .common import build_reviewer_query
from .shared_tools import glob_files_text, grep_files_text, read_file_text


_openai_agent_client_configured = False


def _configure_openai_agent_client() -> None:
    global _openai_agent_client_configured
    if _openai_agent_client_configured:
        return

    from agents import set_default_openai_client, set_tracing_disabled
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    set_default_openai_client(client)
    set_tracing_disabled(True)
    _openai_agent_client_configured = True


def build_openai_file_tools(allowed_paths: list[str], file_qa_callback=None):
    from agents import function_tool

    @function_tool
    def read_file(abs_path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file. Returns the content with line numbers."""
        return read_file_text(
            abs_path=abs_path,
            start_line=start_line,
            end_line=end_line,
            allowed_paths=allowed_paths,
        )

    @function_tool
    def grep_files(pattern: str, directory: str = ".", file_glob: str = "**/*") -> str:
        """Search file contents for a regex pattern."""
        return grep_files_text(
            pattern=pattern,
            directory=directory,
            file_glob=file_glob,
            allowed_paths=allowed_paths,
        )

    @function_tool
    def glob_files(pattern: str, directory: str = ".") -> str:
        """Find files matching a glob pattern under a directory."""
        return glob_files_text(
            pattern=pattern,
            directory=directory,
            allowed_paths=allowed_paths,
        )

    tools = [read_file, grep_files, glob_files]

    if file_qa_callback is not None:
        @function_tool
        async def file_qa(abs_path: str, question: str) -> str:
            """Answer a question about a file's content."""
            return await file_qa_callback(
                abs_path=abs_path,
                question=question,
                allowed_paths=allowed_paths,
            )

        tools.append(file_qa)

    return tools


async def run_openai_agent_task(
    name: str,
    instructions: str,
    user_prompt: str,
    model_id: str,
    tools=None,
    max_turns: int = 30,
) -> tuple[str, float]:
    from agents import Agent, Runner

    _configure_openai_agent_client()
    print(f"  [{name}] starting OpenAI Agent SDK via OpenRouter ({model_id}) ...")
    agent = Agent(
        name=name,
        instructions=instructions,
        tools=tools or [],
        model=model_id,
    )
    result = await Runner.run(agent, user_prompt, max_turns=max_turns)
    result_text = result.final_output
    if not isinstance(result_text, str):
        result_text = str(result_text)
    if not result_text.strip():
        raise ValueError(f"{name} returned empty output")
    print(f"  [{name}] done — {model_id} (OpenAI Agent SDK via OpenRouter)")
    return result_text, 0.0


async def run_openai_reviewer(
    name: str,
    system_prompt: str,
    paper_path: str,
    model_id: str,
    venue: str = "",
    max_turns: int = 30,
) -> tuple[str, float]:
    query, _paper_abs, paper_dir = build_reviewer_query(system_prompt, paper_path, venue)
    return await run_openai_agent_task(
        name=name,
        instructions=system_prompt,
        user_prompt=query,
        model_id=model_id,
        tools=build_openai_file_tools([paper_dir]),
        max_turns=max_turns,
    )
