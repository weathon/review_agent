prompt = """Your job is to judge which reviewer proposed weaknesses are better, judge each weakness individually, and then give an overall judgement on which reviewer proposed better weaknesses. Return a list of the judgement for each weakness of each reviewer and overall judgement of the reviewer. You also have access to the paper itself for cross reference. Check {paper_ids} for weaknesses and paper, do not try to access other files. Some weakness might look right on the first glance but think through if it actually makes sense. 

Check for these things in the weaknesses:
- Nonsense/invalid: how many weaknesses are just nonsense or incorrect applying to the paper
- Excessive demands: if the weaknesses are just asking for excessive things that are not necessary for a good paper. 
- Actionable: how many weaknesses are actionable for the authors to improve the paper, instead of just comments.
- Generic comment rate: how many weaknesses are just generic comments that can apply to any paper, without really pointing out the specific problems of the paper.
- Value/Quality of the weaknesses: how valuable the weaknesses are for improving the paper or pointing out the problems of the paper.
"""

test_prompt = "for testing if my permission is correct, try to read this file: /home/wg25r/review_agent/new/helpers.py, it should be denied. Then try to use Bash to read it, it should also be denied." 

import asyncio
from claude_agent_sdk import ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock, ClaudeSDKClient

import os
paper_path = "./results"
paper_id = "c7OsKOOZo8.md"
paper_ids = [os.path.join(paper_path, dirname, paper_id) for dirname in os.listdir(paper_path)]
assert all(os.path.exists(pid) for pid in paper_ids), f"Not all paper paths exist: {paper_ids}"
paper_ids = " \n".join(paper_ids)
print(paper_ids)
async def main():
    async with ClaudeSDKClient(options=ClaudeAgentOptions(model="claude-opus-4-7", allowed_tools=[f"Read({paper_path})"], cwd="./")) as client:
        await client.query(prompt.format(paper_ids=paper_ids))
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