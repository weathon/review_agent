# EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Large language models (LLMs) have recently advanced in reasoning when optimized with reinforcement learning (RL) under verifiable rewards. Existing methods primarily rely on outcome-based supervision to strengthen internal LLM reasoning, often leading to inefficient exploration and sparse rewards. To mitigate this issue, we propose Expert-Assisted Policy Optimization (EAPO), a novel RL framework that enhances exploration by incorporating multi-turn interactions with external experts during training. Unlike prior methods, where policies reason in isolation, EAPO incentivizes the policy to adaptively determine when and how to consult experts, yielding richer reward signals and more reliable reasoning trajectories. External assistance ultimately internalizes expert knowledge into the policy model, amplifying the model’s inherent reasoning capabilities. During evaluation, the policy model has been well-optimized to solve questions independently, producing improved reasoning paths and more accurate solutions. Experiments on mathematical reasoning benchmarks, including AIME 2024, AIME 2025, and AIMO 2025, show that EAPO consistently outperforms expert-assisted workflow, expert-distilled models, and RL baselines, with an average gain of 5 points over self-exploratory models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Expert-Assisted Policy Optimization (EAPO), a reinforcement learning (RL) framework designed to enhance the reasoning capabilities of large language models (LLMs) in complex tasks, particularly mathematical reasoning. EAPO augments the policy model's action space with a "consult experts" action, allowing the model to seek assistance from external experts (stronger LLMs) during training on-demand. This is optimized end-to-end with verifiable rewards, incorporating mechanisms like parallel expert querying, acceptance rate annealing, and consultation penalties to internalize knowledge and reduce reliance on experts over time. During evaluation, the policy reasons independently. Experiments on AIME 2024/2025 and AIMO 2025 show EAPO outperforming baselines like self-exploratory RL, expert-assisted workflows, and distillation methods, with gains of ~5 points on average accuracy and improved stability.

### Strengths
1.	EAPO creatively treats expert consultation as a learnable action within the RL policy, enabling adaptive, on-demand guidance during training. This addresses key RL challenges in reasoning tasks: sparse rewards and inefficient exploration.
2.	The experiments are comprehensive, evaluating on challenging math benchmarks with Pass@32 and variance metrics. EAPO consistently outperforms baselines (e.g., +4.91% over self-exploratory RL on average). Ablations on expert parallelism (up to K=3), expert size (14B vs. 32B), and policy scaling (7B to 14B) provide insightful evidence of benefits.
3.	The paper emphasizes reproducibility with detailed setups (e.g., models like DeepSeek-R1-Distill-Qwen-7B as policy, QwQ-32B as expert), hyperparameters (Table 4), prompts (Appendix C), and vLLM deployment (Appendix D).

### Weaknesses
1.	While effective on AIME/AIMO, the evaluation is confined to mathematical tasks with verifiable rewards. Generalization to other reasoning domains (e.g., code, commonsense, or multi-modal) is not explored, despite claims of "complex reasoning." The paper acknowledges this in limitations but lacks even preliminary cross-task experiments. This narrows the claimed impact on "large reasoning models (LRMs)."
2.	EAPO's gains hinge on high-quality experts (e.g., 32B > 14B in Table 2). If experts are noisy or biased, the policy might internalize errors, exacerbating reward hacking— a known RLHF issue cited but not deeply mitigated here. Ablations on noisy experts or homogeneous vs. heterogeneous pools could strengthen robustness claims. Additionally, the fixed expert pool (up to 3 replicas) limits diversity; integrating more varied experts (e.g., specialized for sub-tasks) could be discussed.
3.	Training involves multi-turn interactions with experts, potentially increasing costs (e.g., parallel queries but up to T=10 turns). While annealing reduces this, no direct comparison of training efficiency (e.g., wall-time or FLOPs) vs. baselines is provided. For larger policies/experts, scalability might be challenging, especially since vLLM is used but not benchmarked against alternatives.
4.	The mechanistic intuitions (Section 2.2) on alleviating sparse rewards and information gain are intuitive but lack formal proofs or bounds (e.g., on variance reduction or convergence). Related work connections (e.g., to HRL or self-distillation) are solid, but EAPO's novelty could be sharper by quantifying how it improves over them (e.g., via regret bounds).

### Questions
1.	Could you provide results on non-math tasks (e.g., coding or planning) to assess generalization? The limitations mention cross-task studies—any preliminary data?
2.	How sensitive is EAPO to expert quality? E.g., what if experts are weaker than the policy or provide conflicting advice?
3.	Did you experiment with dynamic expert pools (e.g., routing to specialized experts) or more than K=3 parallelism?
4.	The reward function includes a 0.1 partial credit for correct format but wrong answer—how does this affect optimization vs. pure 0/1 rewards?
5.	In Appendix E cases, how were examples selected? Are there failure modes where EAPO over-consults or under-internalizes?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This article proposes a novel reinforcement learning framework called EAPO (Expert-Assisted Policy Optimization) that allows policy models to dynamically request the assistance of more powerful expert models during training, integrating expert feedback into their sampling trajectory. The empirical studies show that EAPO consistently outperforms strong baseline methods, including self-exploratory reinforcement learning, expert distillation, and multi-agent collaboration in math problems, demonstrating its effective knowledge transfer and inference capabilities.

### Strengths
1. Formulating expert consultation as an optimizable and annealing action in reinforcement learning is a novel idea. It goes beyond static workflows and distillation.
2. Experiments are comprehensive. This paper compares extensive baseline methods including self-exploratory methods and distillation methods on three complex reasoning benchmarks. Results are statistically and qualitatively convincing.
3. The presentation of this paper is clear, with intuitive explanations of complex mechanisms.

### Weaknesses
1. EAPO mainly aims to use powerful models for inference in RL processes. In this case, why not directly use large models for inference, which would consume fewer resources.
2. EAPO provides access to significantly stronger expert models. It’s unclear how EAPO performs when the expert is only marginally better or when experts are noisy. An ablation on expert quality would strengthen robustness claims.
3. Parallel expert querying during training increases inference and memory overhead. The paper lacks an analysis of training-time cost against gain trade-offs. The article lacks an analysis of the training process time, such as the inference time for each training step.
4. While promising, all experiments are in mathematical reasoning. Demonstrating EAPO on non-math domains (e.g., code, science) would bolster its generality.
5. EAPO provides a form of intermediate guidance, but the paper does not discuss how this compares to the more established concept of process supervision. (e.g., PRMs)

### Questions
1. How does EAPO performance change if the expert model is weaker, for example, the same size as policy or only slightly better? Could EAPO still learn useful strategies for exploration?
2. Do the authors test EAPO on non-math tasks such as HumanEval? Does EAPO generalize to other structured reasoning domains?
3. How does EAPO compare to methods that use learned reward models trained on step-level annotations (e.g., PRMs)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a reinforcement learning framework called Expert-Assisted Policy Optimization (EAPO), which integrates external expert models (e.g., larger LLMs) during policy training. The goal is to leverage expert feedback to provide richer rewards, accelerate convergence, and improve learning stability compared to standard RL optimization.

### Strengths
1. Paper is well written and motivation is clearly stated.
2. Experiments support the effectiveness of their proposed RL framework.

### Weaknesses
1. The "expert" idea is not novel. Many prior works have explored using larger or teacher models to guide smaller student models. Similar approaches have also been studied, such as using supervised data (human or model-generated) to improve RL. This paper lacks a clear discussion and comparison with those related methods.
2. The paper does not explicitly state: how the framework decides whether experts is required at time t; how to determine which instances are simple, hard, or complex at the beginning of training
3. Notation typos. The subscript in Line 121-123 is not consistent with Line120; Line 124 should use index t-1 based on the equation in Line 126; the meaning of q_t in Line 235 is unclear.

### Questions
1. What is the Expert-Assisted Workflow method mentioned, and how does it differ from your proposed EAPO?
2. Have you considered the compute cost of using experts? Calling larger models can be expensive, and standard RL might generate more trajectories within the same compute budget.
3. It seems the main benefit of using experts is to make the initial reward signals denser so that more meaningful trajectories are used for model updates. Could you show how the initial rewards change after introducing experts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Expert-Assisted Policy Optimization (EAPO), a framework that integrates multi-turn interactions with external experts throughout the training process. Unlike prior methods where policy models reason in isolation, EAPO enables the policy to adaptively determine when and how to consult experts, thus obtaining richer reward signals and more reliable reasoning trajectories. Experiments on mathematical reasoning benchmarks like AIME 2024, AIME 2025, and AIMO 2025 demonstrate that EAPO outperforms baselines.

### Strengths
1. The paper is well-organized and easy to follow, and the figures are clear and helpful.

2. The paper’s ablation studies, focusing on expert parallelism and expert model size, validate EAPO’s key designs: parallel expert queries outperform sequential ones and self-exploratory RL, while larger experts (e.g., QwQ-32B) outperform smaller ones.

### Weaknesses
1. This paper only validates EAPO’s performance on mathematical reasoning benchmarks, including AIME and AIMO, while excluding other domains’ datasets like HLE and GPQA.

2. The expert model-related overhead is relatively high. EAPO adopts a multi-expert parallel reasoning mechanism to improve information coverage and relies on large-scale expert models to ensure the quality of guidance, which undoubtedly increases the consumption of computing resources during model training and deployment, limiting its practicality in resource-constrained scenarios.

3. EAPO is highly dependent on the capabilities of expert models. Experiments show that using smaller-scale expert models will lead to a significant decline in EAPO's performance. Moreover, when the expert is just a single model or is identical to the policy model, it may be difficult to effectively assist policy model optimization due to the lack of diverse perspectives.

4. In terms of feedback mechanism design, EAPO is similar in idea to related works that rely on reward models to provide feedback. It does not form a significant difference in core feedback logic or framework design, resulting in insufficient novelty compared with existing reinforcement learning-assisted optimization methods.

5. Minor issues in Section 2: In Line 122, how to generate the expert assistance $o_i$
is unclear, in Line 126, $\pi_\theta(\tau_t, \alpha_t | H_{t-1}) = \pi_\theta^\tau(\tau_t | H_{t-1}) \cdot \pi_\theta^\alpha(\alpha_t | H_{t-1}, \tau_t)$ is problematic, since the formula is given $H_{t-1}$ instead of $H_t$, and in Line235, the $q_i$ is undefined.

### Questions
1. How does EAPO perform on other domains’ datasets like HLE and GPQA?

2. How robust is EAPO to low-quality or noisy expert models? For example, when experts provide incorrect or meaningless guidance, does the framework’s performance degrade gracefully? And how could you solve this problem?

3. How dependent is EAPO on the expert models? What happens if the experts are fewer, weaker, or partially misaligned? Specifically, if the expert model is identical to the policy model, how do performance and stability change?

### Soundness
2

### Presentation
2

### Contribution
2
