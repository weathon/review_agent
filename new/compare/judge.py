
prompt = """Your job is to judge which reviewer proposed weaknesses are better, judge each weakness individually, and then give an overall judgement on which reviewer proposed better weaknesses. Return a list of the judgement for each weakness of each reviewer and overall judgement of the reviewer. You also have access to the paper itself for cross reference. Check {paper_ids} for weaknesses and paper, do not try to access other files. Some weakness might look right on the first glance but think through if it actually makes sense. 

Check for these things in the weaknesses:
- Nonsense/invalid: how many weaknesses are just nonsense or incorrect applying to the paper
- Excessive demands: if the weaknesses are just asking for excessive things that are not necessary for a good paper. 
- Actionable: how many weaknesses are actionable for the authors to improve the paper, instead of just comments.
- Generic comment rate: how many weaknesses are just generic comments that can apply to any paper, without really pointing out the specific problems of the paper.
- Value/Quality of the weaknesses: how valuable the weaknesses are for improving the paper or pointing out the problems of the paper.

Common red flags to look for in the weaknesses, mark each weakness with the corresponding error type if you find any of these issues:
| Error Type | Explanation |
|---|---|
| Misunderstanding | The reviewer misinterprets claims or ideas presented in the paper, leading to inaccurate or irrelevant comments. |
| Neglect | The reviewer overlooks important details explicitly stated in the paper, resulting in unwarranted questions or critiques. |
| Vague Critique | The review lacks specificity, claiming missing components without clearly identifying what is missing. |
| Inaccurate Summary | The summary in the review misrepresents the main content or contributions of the paper. |
| Out-of-scope | The reviewer suggests additional methods, experiments, or analyses that are beyond the intended scope of the paper. |
| Misunderstanding of the Submission Rule | The reviewer believes the submission format violates conference rules, but this is not actually the case. |
| Subjective | The review makes assertions about the paper's clarity or quality without providing sufficient justification or evidence. |
| Invalid Criticism | The reviewer's criticism is considered invalid, especially when suggesting impractical experiments or trivializing results. |
| Misinterpret Novelty | The reviewer questions the novelty of the work without substantiating their claims with relevant references. |
| Superficial Review | The reviewer appears to have only skimmed the paper, providing generic or unsupported comments about the presence or absence of weaknesses. |
| Writing | Discrepancies arise when the reviewer praises the writing, while you suggest it needs more clarity or explicitness. |
| Inexpert Statement | The reviewer exhibits a lack of domain knowledge, leading to unnecessary or irrelevant concerns. |
| Missing Reference | The reviewer proposes alternative frameworks or methods without providing justification or citing relevant references. |
| Experiment | Conflicting opinions about the design of experiments; the reviewer praises them while you suggest adding more baselines or tests. |
| Misplaced attributes | Strengths are incorrectly listed as weaknesses or vice versa. |
| Invalid Reference | The reviewer cites non-peer-reviewed sources or blogs, which is not appropriate for academic validation. |
| Unstated statement | Statements made in the review are not supported by content in the paper. |
| Summary Too Short | The provided summary is excessively brief, offering little to no insight into the actual content of the paper. |
| Contradiction | The reviewer contradicts themselves within the review, such as criticizing the paper's experiments while later stating that the experiments are comprehensive. |
| Typo | The review contains typographical errors that may affect clarity or understanding. |
| Copy-pasted Summary | The summary is directly copied from the submission. |
| Concurrent work | The reviewer requests comparisons with work conducted concurrently, which may not have been considered by the authors. |
| Duplication | The review segment is a repetition or duplication of a previous segment within the same review. |

Note that some weaknesses might have multiple issues, so mark all that apply. The weaknesses might be written in a way that looks legitimate on the first glance, but they actually have problems, be critical about that, cross verify it carefully. 
"""

test_prompt = "for testing if my permission is correct, try to read this file: /home/wg25r/review_agent/new/helpers.py, it should be denied. Then try to use Bash to read it, it should also be denied." 

import asyncio
from claude_agent_sdk import ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock, ClaudeSDKClient

import os
paper_path = "./results"
paper_id = "0JLUFJMo5p.md"
paper_ids = [os.path.join(paper_path, dirname, paper_id) for dirname in os.listdir(paper_path) if dirname not in ["JE87"]]
assert all(os.path.exists(pid) for pid in paper_ids), f"Not all paper paths exist: {paper_ids}"
paper_ids = " \n".join(paper_ids)
print(paper_ids)

import random
emojies = ["🚀", "🌟", "🔥", "🎉", "🍕", "🌈", "⚡", "🎸", "🌮", "🦄", "🌻", "🍜", "🎨", "🚲", "🌊", "🍩", "🎯", "🧩", "🌵", "🍔", "🎻", "🚁", "🌸", "🍎", "🎲", "🧊", "🌴", "🍇", "🎹", "🚂"]
async def main():
    async with ClaudeSDKClient(options=ClaudeAgentOptions(model="claude-opus-4-7", effort="low", allowed_tools=[f"Read({paper_path})"], cwd="./")) as client:
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
                            print(block.text)
                print(random.choice(emojies), flush=True, end="")
asyncio.run(main())