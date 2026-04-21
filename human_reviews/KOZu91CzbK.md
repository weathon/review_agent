# Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization

- Avg Score: 5.67
- Decision: Accept (spotlight)
- Scores: 6, 8, 3

## Abstract
Recent months have seen the emergence of a powerful new trend in which large language models (LLMs) are augmented to become autonomous language agents capable of performing objective oriented multi-step tasks on their own, rather than merely responding to queries from human users. Most existing language agents, however, are not optimized using environment-specific rewards. Although some agents enable iterative refinement through verbal feedback, they do not reason and plan in ways that are compatible with gradient-based learning from rewards. This paper introduces a principled framework for reinforcing large language agents by learning a retrospective model, which automatically tunes the language agent prompts from environment feedback through policy gradient. Specifically, our proposed agent architecture learns from rewards across multiple environments and tasks, for fine-tuning a pre-trained language model which refines the language agent prompt by summarizing the root cause of prior failed attempts and proposing action plans. Experimental results on various tasks demonstrate that the language agents improve over time and that our approach considerably outperforms baselines that do not properly leverage gradients from the environment.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper propose a RLHF way to tune the prompt for the LLM agent.

### Strengths
The authors design a RLHF framework to tuning the prompt for the agents and have better scores on multiple tasks.

### Weaknesses
I think the main weakness is that this kind of prompt tuning may not be necessary for a well trained agent. The agent should be well tuned to understand all kinds of different prompts. So the problem itself may not be very significant. Instead of fixing the agents and tuning the prompts, tuning the agents from RLHF feedback may have a more significant effect for the LLM. But this is just my opinion. The authors can discuss whether it's correct or not.

### Questions
See the weakness section

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Summary: the paper introduces Retroformer, an algorithm for training a LLM to optimize a retrospective model which provides feedback on another "agent" model's trajectory, incorporating it into a prompt which the agent LLM can condition on for its next trial at the task.

They do this by rolling out an agent LLM in the environment (where observations and actions are all text), prompting the retrospective model on the trajectory and the final reward (which is computed heuristically), then prompting the retrospective model to output text which reflects on what went wrong and what the agent can do better next time. The actor model then conditions on this text.

They create many rollouts using this procedure, score them, and finetune the retrospective agent using PPO.

On HotPotQA, Alfworld, and Webshop, Retroformer shows modest success improvements over Reflexion, the previous SOTA.

### Strengths
- The paper is easy to read and positions itself clearly with respect to related work. 
- It addresses a challenge which has not been solved yet - the problem of how to do RL finetuning with language agents.
- The proposed algorithm can be used in conjunction with a hosted actor LLM (e.g. GPT4) which cannot be finetuned. This seems important for making this algorithm useful for users.
- The results show consistent improvements over Reflexion.

### Weaknesses
- The paper's examples of bad/good prompts do not obviously show that the finetuned reflection model produces better prompts. For instance, in Figure 5, the Reflexion plan is better formatted but still doesn't point out that it was incorrect to return two series rather than one. It would be useful to see an analysis of how often the plan correctly identifies an improved plan (e.g. have a human hand-label 100 prompts for Reflexion and 100 from the frozen LM and record their success rates.)
-
- See my questions/suggestions below

### Questions
- I'm confused how the "f1 score" reward works.
- I'd like to see the following additional curves added to Fig 4 and 6 (possibly in an appendix), which might make it clearer how the finetuning contributes:
  - Retroformer, rolling out the base policy (before RL finetuning).  This would make it easier to see how much of the improvement above Reflexion is due to finetuning vs other algorithmic details.
  - Retroformer, rolling out the base policy, but with GPT-3 as the reflection model. This would answer the question of whether it's typically a better strategy to finetune a small model or just prompt a more capable cloud model.
- I'd recommend writing out the exact algorithm you used for PPO finetuning, especially since when PPO is used with language models slight details from the original algo are changed. Also, it's typically an online algorithm, so it would be helpful to detail any changes you made to make it work well in the offline setting.
- Equation 6 seems to be optimizing for each (x, y) pair individually (i.e. not treating a sequence of (x1, y1), (x2, y2) (x3, y3)... as a trajectory. Is this correct? If so, I'd recommend making this clearer in the text.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces "Retroformer", a framework designed to enhance LLM-assited agents. The Retroformer system comprises two language models, the actor and the retrospective model. The actor model executes tasks while the retrospective model provides feedback to the actor model, allowing it to self-improve. Retroformer employs both short-term and long-term memory to shape the rewards. Short-term memory is represented by the actor model's action history while long-term memory is created by appending summaries of previous failures to the task prompt. Experimental results demonstrates the effectiveness of Retroformer in various tasks.

### Strengths
- The overall framework designed by Retroformer is interesting and alleviates some of the shortcomings in previous works.
- The paper is well-structured with clear explanations of terminology and content, aiding in readability.

### Weaknesses
- The improvements brought by Retroformer are limited. There are no significant improvements in HotPotQA and WebShop, only meaningful improvement is observed in AlfWorld.
- The experiments are not solid enough. It lacks comparisons with RL methods that have been recognized to perform well within the same framework of interaction-feedback-learning. Additionally, there is no comparison with the currently acknowledged GPT-4 model, which has impressive decision-making capabilities, making it insufficient to demonstrate the contribution of this work.
- Only the prompt used in the HotPotQA environment is displayed, and it is difficult to determine whether special prompt engineering is needed in different environments. Therefore, it is insufficient to verify the claim of 'automatically tunes the language agent prompts'.

### Questions
- The feedback obtained from the interaction between the agent and the environment is usually sparse, and the computed rating $r=G_{k,i+1}-G_{k,i}$ represents the difference in return between two consecutive episodes. This means that the data used for finetuning retrospect models are not significantly different within a certain period. How does Retroformer address the overfitting issue caused by fine-tuning on a large amount of repetitive data?
- Are there any differences in the prompts used by Retroformer and baselines methods, and are there experimental results to analyze this part through ablation analysis?
- What are the possible reasons for limited performance improvements in HotPotQA and WebShop?
- How much efficiency is gained by using a retrospective model to tune prompts, rather than directly tuning the LLMs?
- Are there any details about the hyperparameters of PPO?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
