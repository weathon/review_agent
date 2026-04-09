import asyncio
import re
import logging
from pathlib import Path
from agents import (
    Agent,
    Runner,
    function_tool
)
import dotenv
dotenv.load_dotenv()
import weave
import os
os.environ["OPENAI_DEFAULT_MODEL"] = "gpt-5.4"
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_tracing_disabled

custom_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
set_default_openai_client(custom_client)
set_tracing_disabled(True)


import time

# ── Leakage detection ────────────────────────────────────────────────
LEAKAGE_WARNING_PATTERNS = [
    r"\bsame paper\b",
    r"\bexact same paper\b",
    r"\bthis exact paper\b",
    r"\bcontains this exact paper\b",
    r"\bthe exact same paper\b",
    r"\bcalibration copy\b",
]

_error_log_path = Path(__file__).parent / "error.log"
_error_logger = logging.getLogger("gpt_agent_sdk.errors")
_error_logger.setLevel(logging.ERROR)
_error_handler = logging.FileHandler(_error_log_path, mode="a")
_error_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_error_logger.addHandler(_error_handler)


def _detect_leakage_warning_phrases(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in LEAKAGE_WARNING_PATTERNS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            matches.append(found.group(0))
    return matches


# ── Agent-level retry ────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_DELAY = 10


async def run_agent_with_retry(agent, prompt: str, max_turns: int = 30) -> str:
    """Run an agent with retry on transient failures. Returns final_output."""
    agent_name = agent.name
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await Runner.run(agent, prompt, max_turns=max_turns)
            output = result.final_output
            if not output or not output.strip():
                if attempt < MAX_RETRIES:
                    print(f"  [{agent_name}] empty response (attempt {attempt}/{MAX_RETRIES}), retrying ...")
                    _error_logger.error(f"[{agent_name}] empty response (attempt {attempt}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY + attempt * 5)
                    continue
                raise RuntimeError(f"[{agent_name}] empty response after {MAX_RETRIES} attempts")
            print(f"  [{agent_name}] done")
            return output
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(
                kw in err_str for kw in ["rate_limit", "overloaded", "429", "529", "timeout", "gateway", "502", "503", "504"]
            )
            if is_retryable and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{agent_name}] transient error (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ... {e}")
                _error_logger.error(f"[{agent_name}] transient error (attempt {attempt}/{MAX_RETRIES}): {e}")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(f"[{agent_name}] failed after {MAX_RETRIES} attempts")


with open("prompts/timeline.txt", "r") as f:
    timeline = f.read().replace("{{CURRENT_DATE}}", time.strftime("%Y-%m-%d"))


def load_prompts(path):
    with open("prompts/" + path, "r") as f:
        return f.read() + "\n\n" + timeline

@function_tool
def read_file(abs_path: str) -> str:
    """Reads the content of a file and returns it as a string."""
    with open(abs_path, "r") as f:
        return f.read()

@function_tool
def glob_files(pattern: str, directory: str = ".") -> str:
    """Find files matching a glob pattern (e.g. '**/*.md', '*.txt') under a directory. Returns one path per line."""
    import glob as _glob
    matches = sorted(_glob.glob(pattern, root_dir=directory, recursive=True))
    return "\n".join(os.path.join(directory, m) for m in matches) if matches else "No files matched."

@function_tool
def grep_files(pattern: str, directory: str = ".", file_glob: str = "*") -> str:
    """Search file contents for a regex pattern. Returns matching lines with file paths and line numbers.

    Args:
        pattern: Regex pattern to search for.
        directory: Directory to search in.
        file_glob: Glob to filter which files to search (e.g. '*.md', '**/*.txt').
    """
    import glob as _glob
    import re
    matches = []
    files = sorted(_glob.glob(file_glob, root_dir=directory, recursive=True))
    for f in files[:500]:  # cap to avoid runaway searches
        fpath = os.path.join(directory, f)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if re.search(pattern, line):
                        matches.append(f"{fpath}:{i}: {line.rstrip()}")
        except Exception:
            continue
        if len(matches) >= 200:  # cap output
            break
    return "\n".join(matches) if matches else "No matches found."


summarizer = Agent(
    name="Summarizer",
    instructions="You are an subagent that specializes in summarizing files. ",
    tools=[read_file], 
)

harsh = Agent(
    name="Harsh Critic",
    instructions=load_prompts("harsh_critic.txt"),
)

neutral_reviewer = Agent(
    name="Neutral Reviewer",
    instructions=load_prompts("neutral_reviewer.txt"),
)

merger = Agent(
    name="Merger",
    instructions=load_prompts("merger.txt"),
)

spark = Agent(
    name="Spark",
    instructions=load_prompts("spark_finder.txt"),
)


human_finder = Agent(
    name="Human Finder",
    instructions=load_prompts("find_human_match.txt"),
    tools=[read_file, glob_files, grep_files, summarizer.as_tool(
            tool_name="summarization",
            tool_description="Summarize a file given its absolute path.",
        ),
    ]
)

scorer = Agent(
    name="Scorer",
    instructions=load_prompts("scorer_agent_gpt.txt"),
    tools=[read_file, glob_files, grep_files, summarizer.as_tool(
            tool_name="summarization",
            tool_description="Summarize a file given its absolute path.",
        ),
    ]
)


starts, ends = [], []
async def run_agent(agent, review_text: str):
    agent_name = agent.name

    start = time.time()
    starts.append((agent_name, start))

    result = await run_agent_with_retry(agent, review_text, max_turns=30)

    end = time.time()
    ends.append((agent_name, end))

    return result

import os

human_review_dir = os.path.abspath("iclr2025_data")
user_prompts = {
    "review": """Review the following paper thoroughly.

NOTE: This paper was extracted from PDF by an automated parser. There may be formatting artifacts such as broken equations, garbled tables, misplaced figure references, or OCR errors. These are parser issues, NOT problems with the paper itself. Do NOT treat formatting artifacts as weaknesses.

{paper_path}
--- PAPER CONTENT START ---
{paper_content}
--- PAPER CONTENT END ---""",
    "human_finder": "Paper file path: {pp}\nHuman reviews directory: {human_review_dir}\n"
}



def main():
    run_agents("paper.md")

from agents import (
    Agent,
    Runner,
)




async def run_agents(paper_path):
    paper_path_abs = os.path.abspath(paper_path)
    print(paper_path_abs)
    with open(paper_path, "r") as f:
        paper_content = f.read()
    parallel_agents = [harsh, neutral_reviewer, spark, human_finder]
    review_prompt = user_prompts["review"].format(paper_path=paper_path_abs, paper_content=paper_content)
    find_human_prompt = user_prompts["human_finder"].format(pp=paper_path_abs, human_review_dir=human_review_dir)
    prompts = [review_prompt, review_prompt, review_prompt, find_human_prompt]
    print(find_human_prompt)

    responses = await asyncio.gather(
        *(run_agent(agent, prompt) for prompt, agent in zip(prompts, parallel_agents))
    )

    agent_names = [a.name for a in parallel_agents]
    labeled_summaries = [
        f"### {name}\n{output}"
        for name, output in zip(agent_names, responses)
    ]

    inputs_block = '\n\n'.join(labeled_summaries)
    merger_user_prompt = f"Here is the paper being reviewed (extracted from PDF — formatting artifacts are parser issues, not paper problems):\n\n--- PAPER CONTENT START ---\n{paper_content}--- PAPER CONTENT END ---\n\nHere are the inputs:\n\n{inputs_block}\n\nNow produce the final consolidated review following your instructions. Remember: many of the harsh critic's points may be nonsensical or overly picky — cross-check everything against the actual paper before including it."
    merged_review = await run_agent_with_retry(merger, merger_user_prompt, max_turns=30)

    scorer_user_prompt = f"Review: \n\n{merged_review}\n\ncal_dir_abs:{os.path.abspath('cal/')}"
    scorer_output = await run_agent_with_retry(scorer, scorer_user_prompt, max_turns=30)

    # Leakage detection on scorer output
    leakage_matches = _detect_leakage_warning_phrases(scorer_output)
    if leakage_matches:
        matched_text = ", ".join(sorted(set(leakage_matches), key=str.lower))
        warning_msg = (
            f"Potential calibration leakage: scorer output contains "
            f"suspicious phrase(s): {matched_text}"
        )
        print(f"  [scorer] WARNING: {warning_msg}")
        _error_logger.error(warning_msg)

    print('Final summary:', scorer_output)

    return scorer_output

async def main():
    result = await run_agents("paper.md")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())