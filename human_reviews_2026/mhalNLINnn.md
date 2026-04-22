# Dynamic Optimizations of LLM Ensembles with Two-Stage Reinforcement Learning Agents

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
The advancement of LLMs and their accessibility have triggered renewed interest in multi-agent reinforcement learning as robust and adaptive frameworks for dynamically changing environments. This paper introduces RL-Focal, a two-stage RL agent framework that routes and ensembles LLMs. \textit{First}, we develop the Decider RL-agent, which learns to dynamically select an ensemble of small size ($m_i$) among $N$ LLMs ($m_i \ll N$) for incoming queries from a user-defined downstream task $i$, by maximizing both error-diversity and reasoning-performance of the selected ensemble through iterative updates of task-adaptive rewards and policy. \textit{Second}, to enable effective fusion of dynamically selected LLMs, we develop the stage-2 Fusion RL-agent, which learns to resolve reasoning conflicts from different LLMs and dynamically adapts to different ensemble teams composed by the Decider Agent for different downstream tasks. {\em Third}, we introduce the focal diversity metric to better model the error correlations among multiple LLMs further improving the generalization performance of the Decider Agent, which actively prunes the ensemble combinations. By focal diversity, we enhance performance across tasks by effectively promoting reward-aware and policy-adaptive ensemble selection and inference fusion. 
Extensive evaluations on five benchmarks show that RL-Focal achieves the performance improvement of 8.48\% with an ensemble of small size 
compared to the best individual LLM in a pool and offers stronger robustness. Code is available at  \url{https://anonymous.4open.science/r/rl-focal-8DCF/}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RL-Focal, a novel two-stage Reinforcement Learning (RL) agent framework designed for robust and adaptive multi-agent systems built upon existing LLMs. Extensive evaluations on five benchmarks demonstrate that RL-Focal achieves an 8.48% performance improvement with a small ensemble compared to the best individual LLM, while also offering stronger robustness.

### Strengths
* Well motivated.
* Improvement on BBH seems significant.
* Methods are described in detail

### Weaknesses
The manuscript has substantial room for improvement, particularly in the representation and experimental design.
* The manuscript's structure seems unbalanced. Only two of the nine pages are dedicated to describing experimental results. Given the apparent lack of a theoretical contribution, the content devoted to methods and general descriptions should be significantly compressed to allow for a deeper discussion of the findings and ablations.
* The current experiments and ablations are limited. I recommend performing a detailed analysis of model selection and usage. Specifically the activation frequency or utilization ratio of each individual model within the agent system across different tasks.
* Since models like Llama 3 and Mixtral show strong inherent performance (e.g., on MMLU), are these models frequently selected for the ensemble within the proposed agent system?

To better understand the system's core capabilities, please consider the following baselines:
* Evaluate the ensemble performance of only the top two or three best-performing models on a specific task to establish a powerful baseline.
* Evaluate the baseline that combining the top $k$ outputs (e.g., $k=5$ outputs) from the best models (best of N).
* Computational cost and improvement compared to these baselines.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RL-Focal, a two-stage multi-agent RL framework that dynamically routes and ensembles LLMs. The stage-1 Decider agent learns to select a small subset of models per query by optimizing both error diversity and reasoning performance with task-adaptive rewards and policies. The stage-2 Fusion agent learns to resolve conflicts and fuse the selected models outputs, and a new focal diversity metric models error correlations to improve generalization in selection and fusion.

### Strengths
1.	Tackles an important, practical problem: adaptive, query-wise ensembling/routing among LLMs rather than static majority voting.
2.	Ablations and sensitivity analyses help understand behavior.

### Weaknesses
1.	**Poor writing/formatting.**  1) Lines 50–51 contain content that should not appear in the paper; please remove or rewrite appropriately. 2) The captions/layout for Figure 3 and Figure 4 have almost no spacing, which hurts readability. Please increase the vertical spacing and ensure consistent caption styling.
2.	**Overstated novelty in the problem formulation.** The paper claims to be the first to formulate LLM ensembling as a POMDP, yet prior work (e.g., RLAE[A], DER [B]) already models LLM-ensemble reasoning as an MDP. Please clarify the substantive differences between your method and prior work.
3.	**Incomplete reporting in Table 1.** Several entries are missing (marked “–”), preventing a complete comparison across datasets. Please fill in the absent results or justify why they are unavailable.
4.	**Lack of same-setting SOTA baselines in the main table.** Table 1 compares RL-Focal primarily against base models; strong ensemble/router baselines are not included there under the same pool and evaluation protocol. For fairness, include leading SOTA methods in Table 1 (or provide a unified main results table) under identical settings.
5.	**Metric inconsistency for Qwen2.5-72B on MMLU.** The paper reports 75.01, whereas widely cited numbers are around 86.1 [C].
6.	**Minor issues.** Occasionally inconsistent capitalization (“LLama” vs “LLaMA”).


[A] RLAE: Reinforcement Learning-Assisted Ensemble for LLMs, arxiv 2025.

[B] Efficient Dynamic Ensembling for Multiple LLM Experts, IJCAI 2025.

[B] Qwen2.5 Technical Report, 2024.

### Questions
see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Introduces RL-Focal. This uses RL to route queries to the best subset of LLMs from a pool. Then another agent fuses the ensembles responses together. They also introduce a new focal diversity metric to improve pruning performance. The paper demonstrates an effective performance increase against popular benchmarks and baseline methods.

### Strengths
Interesting two-stage formulation separating selection (Decider) and combination (Fusion), with a multi-agent RL formulation and a centralized critic to stabilize training. Algorithms and training loops are clearly described (Algorithm 1 and 2). Furthermore, the paper attempts cost accounting and shows wall-clock/param comparisons in Appendix E (encouraging effort to quantify cost).

### Weaknesses
There are some similar RL ensemble approaches which limit the novelty (i.e. RLAE can in effect prune LLMs by lowering weights near zero), although they are formulated differently. The paper motivates RL via online adaptivity, but an explicit demonstration of that advantage would clarify necessity. Furthermore, training which uses two RL policies and a centralized critic adds significant computational overhead over supervised learning methods, though this is perhaps offset by the lower inference cost. It would be nice to have some further details regarding the impact of warm starting as well as experiments task distribution shifts (i.e. start with maths, end with reasoning to highlight the strengths of RL). Furthermore, I am surprised such small networks are able learn complexities of routing and combining LLM outputs. Lastly, it would be nice to see the RL training curves (performance over episodes).

### Questions
Do the advantages of online RL here justify the training costs of over supervised or simpler RL frameworks? What differences in approaches leads to the difference in performance between LLM-TOPLA and RL-Focal on GSM8k?
Do you have any results for performance of the difference in performance between the warm start and final RL tuned model?
Did you experiment with higher parameter counts? Are there any intuitions behind why it was effective with so few?
Are there any patterns or reasons why certain LLMs are chosen by the policy? Does this change over training time?

### Soundness
3

### Presentation
3

### Contribution
3
