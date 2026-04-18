
prompt = """Looking for red flags in the two review provided, cross compare with the paper. 
| Error Type | Explanation |
|---|---|
| Misunderstanding | The reviewer misinterprets claims or ideas presented in the paper, leading to inaccurate or irrelevant comments. |
| Neglect | The reviewer overlooks important details explicitly stated in the paper, resulting in unwarranted questions or critiques. |
| Vague Critique | The review lacks specificity, claiming missing components without clearly identifying what is missing. |
| Out-of-scope | The reviewer suggests additional methods, experiments, or analyses that are beyond the intended scope of the paper. |
| Invalid Criticism | The reviewer's criticism is considered invalid, especially when suggesting impractical experiments or trivializing results. |
| Superficial Review | The reviewer appears to have only skimmed the paper, providing generic or unsupported comments about the presence or absence of weaknesses. |
| Unstated statement | Statements made in the review are not supported by content in the paper. |
| Excessive demands | if the weaknesses are just asking for excessive things that are not necessary for a good paper. |
| Generic comment | weaknesses are just generic comments that can apply to any paper, without really pointing out the specific problems of the paper. | 

Then conclude each has fewer red flags. 

Review and Paper:
{paper_ids}
"""

test_prompt = "for testing if my permission is correct, try to read this file: /home/wg25r/review_agent/new/helpers.py, it should be denied. Then try to use Bash to read it, it should also be denied." 

import asyncio
from claude_agent_sdk import ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock, ClaudeSDKClient

import os
# paper_path = "./results"
# paper_id = "0JLUFJMo5p.md"
# paper_ids = [os.path.join(paper_path, dirname, paper_id) for dirname in os.listdir(paper_path) if dirname not in ["JE87"]]
# assert all(os.path.exists(pid) for pid in paper_ids), f"Not all paper paths exist: {paper_ids}"
# paper_ids = " \n".join(paper_ids)
# print(paper_ids)


paper_path = "./human_eval"
paper_id = "2602.10095"
paper_ids = [os.path.abspath(os.path.join(paper_path, i)) for i in os.listdir(paper_path) if i.startswith(paper_id) and i.endswith(".md")]
print(paper_ids)
import random 
emojies = ["🚀", "🌟", "🔥", "🎉", "🍕", "🌈", "⚡", "🎸", "🌮", "🦄", "🌻", "🍜", "🎨", "🚲", "🌊", "🍩", "🎯", "🧩", "🌵", "🍔", "🎻", "🚁", "🌸", "🍎", "🎲", "🧊", "🌴", "🍇", "🎹", "🚂"]
async def main():
    async with ClaudeSDKClient(options=ClaudeAgentOptions(model="claude-opus-4-6", effort="high", allowed_tools=[f"Read({paper_path})"], cwd="./")) as client:
        with open("judge_output.txt", "a") as f:
            await client.query(prompt.format(paper_ids=paper_ids))
            f.write("New judgement for paper " + paper_path + ":\n")
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    f.write(f"\nSession ID: {message.session_id}")
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            f.write(block.text + "\n")
                            print(block.text)
                print(random.choice(emojies), flush=True, end="")
        
asyncio.run(main())


# prompt = """Your job is to judge which reviewer proposed weaknesses are better, judge each weakness individually, and then give an overall judgement on which reviewer proposed better weaknesses. Return a list of the judgement for each weakness of each reviewer and overall judgement of the reviewer. You also have access to the paper itself for cross reference. Check {paper_ids} for weaknesses and paper, do not try to access other files. Some weakness might look right on the first glance but think through if it actually makes sense, exam them critically.

# **CRITICAL: Default stance is INVALID.** A weakness must be marked INVALID unless you can explicitly cite concrete reasons grounded in the paper's content that prove it is valid. Do NOT mark a weakness valid just because it sounds plausible, reasonable, or well-written. Do NOT assume the reviewer is correct. For every weakness you mark valid, you MUST:
# 1. Quote or cite the specific part of the paper the weakness refers to.
# 2. Explain why the weakness genuinely applies to that content.
# 3. Confirm the reviewer is not misreading, over-generalizing, or fabricating.

# If you cannot do all three, mark it INVALID. Being unsure = INVALID. Sounding reasonable = INVALID. The burden of proof is on validity, not invalidity.

# Check for these things in the weaknesses:
# - Nonsense/invalid: how many weaknesses are just nonsense or incorrect applying to the paper
# - Actionable: how many weaknesses are actionable for the authors to improve the paper, instead of just comments.
# - How good it is at surface level features, like length, how well the weakness sounds, formating, how formal they look, etc. **penalize** surface level features at the final evaluation (Note: Not just discounting them — actively penalizing it.) if the point.

# Nonsense types: if any of the following issues exist in the weaknesses, mark them as nonsense/invalid:
# | Error Type | Explanation |
# |---|---|
# | Misunderstanding | The reviewer misinterprets claims or ideas presented in the paper, leading to inaccurate or irrelevant comments. |
# | Neglect | The reviewer overlooks important details explicitly stated in the paper, resulting in unwarranted questions or critiques. |
# | Vague Critique | The review lacks specificity, claiming missing components without clearly identifying what is missing. |
# | Out-of-scope | The reviewer suggests additional methods, experiments, or analyses that are beyond the intended scope of the paper. |
# | Invalid Criticism | The reviewer's criticism is considered invalid, especially when suggesting impractical experiments or trivializing results. |
# | Superficial Review | The reviewer appears to have only skimmed the paper, providing generic or unsupported comments about the presence or absence of weaknesses. |
# | Unstated statement | Statements made in the review are not supported by content in the paper. |
# | Excessive demands | if the weaknesses are just asking for excessive things that are not necessary for a good paper. |
# | Generic comment | weaknesses are just generic comments that can apply to any paper, without really pointing out the specific problems of the paper. | 

# Note that an review could be right but still nonsense, for example: "The paper only used GC-MS and not GC-MS/MS" Which could be correct - the paper did noy do that -  but is non sense because it is asking for something that is not necessary for a good paper.

# Note that some weaknesses might have multiple issues, so mark all that apply. The weaknesses might be written in a way that looks legitimate on the first glance, but they actually have problems, be critical about that, cross verify it carefully. 
# """