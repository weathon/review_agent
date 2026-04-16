prompt = """Your job is to judge which reviewer proposed weaknesses are better, judge each weakness individually, and then give an overall judgement on which reviewer proposed better weaknesses. Return a list of the judgement for each weakness of each reviewer and overall judgement of the reviewer. You also have access to the paper itself for cross reference. Check {paper_path} for weaknesses and paper, do not try to access other files. Some weakness might look right on the first glance but think through if it actually makes sense. """

test_prompt = "for testing if my permission is correct, try to read this file: /home/wg25r/review_agent/new/helpers.py, it should be denied. Then try to use Bash to read it, it should also be denied." 
import asyncio
from claude_agent_sdk import ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock, ClaudeSDKClient

paper_path = "./2411.14205v1"
async def main():
    async with ClaudeSDKClient(options=ClaudeAgentOptions(model="claude-opus-4-7", allowed_tools=[f"Read({paper_path})", f"Bash(ls {paper_path})"], cwd="./")) as client:
        await client.query(prompt.format(paper_path=paper_path))
        # await client.query(test_prompt)
        with open("judge_output.txt", "a") as f:
            f.write("New judgement for paper " + paper_path + ":\n")
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    f.write(f"\nSession ID: {message.session_id}")
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            f.write(block.text + "\n")
                print(".", flush=True, end="")
asyncio.run(main())