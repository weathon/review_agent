# Prompt reinforcing for long-term planning of large language models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Large language models (LLMs) have achieved remarkable success in a wide range of natural language processing tasks and can be adapted through prompting. 
However, they remain suboptimal in multi-turn interactions, often relying on incorrect early assumptions and failing to track user goals over time, which makes such tasks particularly challenging.
Prior works in dialogue systems have shown that long-term planning is essential for handling interactive tasks. 
In this work, we propose a prompt optimisation framework inspired by reinforcement learning, which enables such planning to take place by only modifying the task instruction prompt of the LLM-based agent. 
By generating turn-by-turn feedback and leveraging experience replay for prompt rewriting, our proposed method shows significant improvement in multi-turn tasks such as text-to-SQL and task-oriented dialogue. 
Moreover, it generalises across different LLM-based agents and can leverage diverse LLMs as meta-prompting agents.
This warrants future research in reinforcement learning-inspired parameter-free optimisation methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper generalizes the idea of self-refinement to the instruction prompt and conditions on step-level feedback from the model at intermediate steps. They demonstrate that this prompt learning approach significantly outperforms other single-turn approaches for optimizing prompts in accordance with feedback.

### Strengths
1. The paper demonstrate sthat this approach is more cost-effective and performant than existing prompt optimization approaches 
2. The idea proposed in the paper is simple, elegant, and well formalized, such that any researcher could implement this on their own tasks 
3. The paper clearly explains with mathematics why this approach for fine-grained feedback might be more appropriate than single-turn feedback 
4. Using this method is more cost-effective for optimization than training-based approaches

### Weaknesses
1. Prompt optimization is useful for improving performance but struggles when LLMs or humans can give useful feedback, but what should practictioners do if the feedback is not useful
2. The paper shows improvement, but this reviewer doesn't find any qualitative analysis of the feedback generated 
3. It is unclear how this might scale as the trajectory gets longer. If feedback is applied to every single step this could become very expensive in terms of API costs.

### Questions
1. How do we ensure that the feedback given to the models is properly aligned and accurate? The fact that the reward improves is perhaps enough evidence, but do you have qualitative examples? 
2. How many trajectories are taken into consideration when doing this? Are the prompts optimized within trajectories or across trajectories? E.g., do we take feedback from multiple trajectories on the same prompt and use this or is it only within the same trajectory that we optimize? 
3. Is the feedback incorporated into the instruction prompt in-flight or only at the end of the trajectory? For example, if the model makes a mistake on step x and gets feedback, is the instruction immediately changed to address this or is it only at the end of the turn? 
4. From this reviewer’s perspective, prompt-based optimization has fundamental limitations in regards to scaling or teaching models wholly new ideas. Do the authors have any ideas on how this type of approach could be combined with parameter updates to do this sort of optimization, or be used in a domain where an LLM or human might not be suitable to give good feedback?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Reinforced Prompt Optimisation (RPO), a  reinforcement learning–inspired framework for improving long-term planning and consistency of large language models (LLMs) in multi-turn interactions.

### Strengths
* Treating prompts as *optimizable parameters* under a reinforcement learning paradigm is both interesting and practical, especially for API-constrained models. Unlike prior prompt optimisation work, RPO explicitly models *temporal feedback* across multiple dialogue turns, addressing a critical weakness of current LLMs.
* The experiments span diverse tasks (SQL, dialogue, medical QA) and model families, demonstrating robustness and generalization.

### Weaknesses
* One limitation is the limited theoretical analysis. While the method draws inspiration from RL, there is no formal definition and analysis on how to relate the concepts of RPO to state, action and value functions. 	
* Most results rely on task success or human preference; it might be interesting to study more on the roles of different LLM players (rewriter and feedbacker). For example, what is the quality of the feedbacker? Are rewards and values generated in reasonable ways?

### Questions
1. Equation 3, is the value function actually optimal value function $V^*$?
2. Figure 4, the text is too small to read;
3. Figure 10, is the turn-level feedback still generated given the full dialogue? Why not generate feedback per turn? 
4. How sensitive is RPO to the feedback quality or the LLM backbone used as the feedbacker? Would weaker models still yield improvement?
5. How do you prevent prompt degradation over iterations? Is there a stopping criterion or performance monitor to avoid over-editing?
6. What are the failure modes*observed during optimisation? Are there cases where RPO overfits to user trajectories or amplifies LLM hallucinations?
7. Could future work explore numerical reward modeling (e.g., via LLM-generated scores) to combine textual feedback with scalar signals?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a prompt optimization technique designed to mitigate performance degradation of large language models (LLMs) in multi-turn dialogue. The proposed method improves textual prompts without using gradients, making it cost-efficient and applicable to API-only models. The experiments demonstrate strong performance across multiple aspects.

### Strengths
The proposed method is easy to follow and can be applied quickly.

The experiments are solid and well support the authors’ claims.

The writing quality is good.

### Weaknesses
Lack of comparison with RL-based prompt generation methods.
The paper compares only against text-gradient-based approaches, but there also exist fully RL-based methods such as StablePrompt [1], which trains a small rewriter model using reinforcement learning. Such methods are generally reported to outperform text-gradient approaches and offer more stable training. Moreover, StablePrompt can be applied to API-based models since it does not directly optimize the system agent. The paper should at least demonstrate that the proposed method outperforms RL-based methods like StablePrompt in some scenarios.

Lack of quantitative comparison on API call costs.
Because RPO uses a replay buffer to reuse prompt–feedback pairs over multiple epochs, it could lead to increased API calls and token usage. While the authors claim efficiency, they do not provide quantitative evidence showing how many more API calls or tokens RPO consumes compared to APO or GPO.

Insufficient statistical reliability of results.
Figure 4 shows high variance in RPO’s performance. Moreover, text-gradient-based methods are known for their instability and high variance during training. Reporting standard deviations in Tables 1 and 2 would help alleviate concerns regarding reliability.

Missing ablation study for the Feedbacker and Rewriter components.
Although the paper highlights the importance of the Feedbacker and Rewriter, no experiments are conducted to replace or analyze them individually. The paper provides insufficient explanation of the prompts used for generating text gradients.

Reference
[1] Kwon et al. "StablePrompt: Automatic Prompt Tuning using Reinforcement Learning for Large Language Models." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024).

### Questions
The proposed RPO does not include gradient descent or a policy improvement step like reinforcement learning. Does the method theoretically (even if not rigorously) guarantee convergence to an optimal policy? My intuition is that it may diverge rather than converge.

How exactly is the text gradient generation process conducted? What kinds of prompts are used for feedback generation and rewriting? Are different prompts used for different Rewriters?

Aside from being TD-style, what distinguishes this method from ORPO? As far as I understand, ORPO and MC-style RPO are essentially identical—is that correct? Why are there no comparisons with ORPO? It would help if the paper clarified which components differ from ORPO.

If I understand correctly,  V(s_t) seems to represent feedback rather than a numeric value. How is the discount factor gamma applied? Is it explicitly written in the LLM prompt as an instruction (e.g., “use this discount rate”)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper describes a prompt optimization method for improving multi-turn interactive tasks with LLMs. Unlike prior work which focuses on single-turn prompt optimization or self-feedback of outputs, the authors propose a framework called Reinforced Prompt Optimization (RPO) that treats the instruction prompt itself as a parameter to be refined via turn-level feedback and prompt rewriting. The model interacts over multiple turns in tasks such as text-to-SQL, task-oriented dialogue and medical QA; a “feedbacker” LLM or human expert gives incremental feedback and a “rewriter” LLM updates the prompt to improve long-term planning. Experiments on three multi-turn tasks show that RPO improves success rates and generalizes across different LLM base models.

### Strengths
1. The idea of treating the instruction prompt as a dynamic, optimizable parameter in multi‐turn settings offers a different angle to the other approaches that require tuning of model weights.
2. The experimental setup is broad: covering multiple tasks (SQL generation, dialogue, medical QA) and a mixture of open and closed source LLMs, which supports generalizability.
3. The experiments uses a clear setup with ablations, e.g. showing turn-level feedback vs full-trajectory feedback, to highlight which components contribute to improvement.

### Weaknesses
1. The idea of using LLM to provide feedback and adjust system prompt has been widely discussed, e.g. [1] and [2]
2. It's unclear how the choice of the base model for the feedbacker and rewritter would affect the final system performance. What's the principles or best practice for selecting the underlying models for this modules?
3. Practical aspect, it’s unclear how many epochs, how much interaction data, or how timely prompt updates can be in a live system.


[1] Self-Refine: Iterative Refinement with Self-Feedback
[2] Reflexion: Language Agents with Verbal Reinforcement Learning

### Questions
1. Any principle or best practice one should follow to select the underlying models for the feedbacker and the rewritter?
2. What happens if the feedbacker LLM is of lower quality or the feedback is noisy, how robust is RPO in that case?

### Soundness
2

### Presentation
3

### Contribution
2
