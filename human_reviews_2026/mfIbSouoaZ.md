# Reinforcement Learning for Machine Learning Engineering Agents

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Machine learning engineering (MLE) has a clear objective: Given an MLE task and a verifier (e.g., performance on some held-out data), what is the most effective way to utilize compute to achieve the best performance for the given task? Existing language model (LM) agents rely on prompting frontier LMs and accumulating experience non-parametrically by storing and retrieving experience through agent scaffolds and test-time compute. In this paper, we show that in environments such as MLE where a good verifier is available, adapting the LM parameters through gradient updates can be more effective in utilizing compute and agent’s experience. Specifically, we show that agents backed by weaker models that improve via reinforcement learning (RL) can eventually outperform agents backed by much larger, but static models for a given MLE task. We identify two major challenges with RL in this setting. First, actions can take a variable amount of time (e.g., executing code for different solutions), which leads to asynchronous policy gradient updates that favor faster but suboptimal solutions. We propose duration-aware gradient updates in a distributed asynchronous RL framework to amplify high-cost but high-reward actions. Second, using performance on the held-out data as a reward for MLE provides limited feedback. A program that’s nearly correct is treated the same as one that fails entirely (e.g., during data loading). We propose environment instrumentation to offer verifiable partial credit, using a separate, static language model to insert print statement to an existing program. Our experiments suggest that a small LM (Qwen2.5-3B) adapted with RL, when given enough compute, can solve an MLE task better than prompting a frontier model (Claude-3.5-Sonnet) with the state-of-the-art agent scaffold (AIDE) by an average of 22% across 12 Kaggle tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores a RL framework for code generation that incorporates environment instrumentation to provide more informative rewards. By measuring both execution time and partial correctness of intermediate steps, the approach aims to make RL-based fine-tuning of code models more efficient and stable. The method is implemented on code generation benchmarks, and experimental results show that a small model trained with RL can outperform larger models or those using scaffolding, demonstrating the practical effectiveness of the proposed approach.

### Strengths
1. The idea of integrating execution time and partial step validity into the RL framework is innovative and interesting, providing a richer and more informative learning signal.
2. The paper is well-organized and clearly written, with logical flow and easy readability.
3. Experimental results convincingly show that a small model enhanced with RL can outperform larger models with scaffolding, validating the method’s efficiency and effectiveness.

### Weaknesses
1. The proposed Environment Instrumentation is conceptually interesting but appears to be an engineering optimization rather than a fundamentally new RL formulation. It would be helpful to explain why simpler signals, such as the error line number from runtime exceptions, could not serve as reward feedback.
2. The experiments lack comparisons with other RL-based code generation methods, focusing only on commercial or scaffolding-based baselines. This limits the clarity of how much the proposed method improves over existing RL techniques.
3. Minor issues: Line 245 contains a possible error, and Figure 4 seems mislabeled (the prompt content is missing and should likely correspond to Table 4).

### Questions
Please check my comments in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores how to best utilize compute in machine learning engineering (MLE) tasks. The paper proposes to adapt a small LM through reinforcement learning (RL), leveraging duration-aware gradient updates from the task’s verifier signal. Empirical results on 12 Kaggle-style MLE tasks show that an RL-adapted Qwen2.5-3B model outperforms a much larger static model (Claude 3.5 Sonnet) using the state-of-the-art agent scaffold (AIDE) by an average of 22%。

### Strengths
1. Technical contributions: Both duration-aware gradient updates and partial-credit instrumentation are conceptually clean, easy to implement, and empirically validated.
2. Strong empirical evidence: Demonstrates small adaptive models outperforming much larger static ones, which is an impactful and counterintuitive result.

### Weaknesses
1. Lack of detailed ablations: The effect of partial-credit feedback is not isolated in the reported experiments.
2. Generalization and robustness unclear. It is not evident whether the adapted model transfers well to unseen MLE tasks or remains stable under changing task distributions.

### Questions
1. Could you report performance with only duration-aware updates, and only partial-credit instrumentation, to isolate their contributions?
2. Does the RL-adapted agent transfer to unseen Kaggle tasks without additional fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This is a great paper. It does RL to solve machine learning engineering tasks. From what I understood, the problems with those tasks is that when formulated as MDPs, credit assignment is difficult because the only non-trivial reward is given at the end of a trajectory when the final engineering task is solved or not. To get intermediate rewards and mitigate this problem, authors propose environment instrumentation (adding prints). They also propose duration aware gradient to favor high-cost but high-reward actions.

### Strengths
This paper's ideas are great and could make it a seminal paper for anyone interesting in RL for machine learning engineering. 
The experiments are amazing with comprehensive ablations and multiple seeds of RL training which is rare in nowadays research.
The problem inherent to RL for MLE are well presented.

### Weaknesses
Here is the thing, the paper is clear as is and has very few objective weaknesses. While the propose duration aware gradients and environment instrumentation are objectively (supported by strong experiments) good contributions, I also believe that objectively the experiments do not allow to determine if RL for MLE is a real advantage. Correct me I am wrong, and I am not an expert in LLMs, but it is semms that authors only use RL for the qwen model and compare the latter to e.g. claude. To really show RL benefits for MLE, wouldn't it be better to show for a set of K models the performance differences between k_1 k_2 k_3... and k_1+RL, k_2+RL ... ?


Then, from an RL perspective, it seems to me that both duration awareness and environment instrumentation could be presented, not algorithm tricks but as MDP formulation changes: duration awareness could be added to the reward function, and prints could be added to the states and transitions? Also the asynchronous setting is not clearly motivated. It would better to focus on the fully synchronous setting for now.

### Questions
Could you try GAE estimatation a.k.a elligibility traces in your RL experiments for even better credit assignment? 
Could you explain why asynchronicity is needed? And if not needed, would duration aware gradients still be needed?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzes the best way to improve performance on a MLE task with a well-defined metric and grader, specifically focusing on the benefits of prompting/scaffolding versus incremental RL. The authors explore duration-aware gradient updates and environment instrumentation / partial credit and ultimately find that a small model with RL performs best in this setting.

### Strengths
The authors focus on a realistic problem (machine learning engineering) and ask a very timely and relevant question (whether it's better to prompt/scaffold vs train with RL to improve performance on a task). This is a question many application businesses are asking themselves right now. The authors prove their points with practical and sound experiments that are clearly laid out. They test a variety of models, scaffolds and settings and lay out results clearly.

### Weaknesses
The authors focus on 12 tasks; it would be interesting to increase size (and specifically, diversity and generality) of the task set.

### Questions
How do these results generalize to settings beyond this 12-task set?

### Soundness
3

### Presentation
3

### Contribution
3
