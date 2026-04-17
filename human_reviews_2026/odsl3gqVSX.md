# Towards Optimism-Pessimism Trade-off in Model-based Offline-to-Online Reinforcement Learning

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Model-based offline-to-online reinforcement learning (RL) provides a sample-efficient framework by pre-training environment models and control policies using offline data, followed by fine-tuning through limited online interactions. However, the distribution shifts between offline and online stages often hinders fine-tuning performance. Existing methods approach this problem by adjusting the trade-off between optimism and pessimism using a single-objective formulation, which requires online evaluation across tasks. This results in an expensive bi-level optimization procedure. In this work, we identify this optimism-pessimism trade-off during offline training as a key challenge: optimistic policies tend to generalize better to novel online tasks by exploring out-of-distribution states and actions, while pessimistic policies remain constrained to the offline data distribution and perform better on tasks that are similar to the offline tasks. To address this challenge, we propose a bi-objective formulation that captures this trade-off and yields a pool of Pareto policies during offline training. These policies reflect varying levels of trade-offs, enabling flexible selection of policies for various online tasks. To produce these policies, we introduce Multiple-Objective Soft Actor-critIC (MOSAIC), which solves multiple bi-objective optimization problems guided by reference vectors and refines the Pareto policy pool through neighborhood search. After offline training, a contextual bandit algorithm hierarchically selects the most suitable policy for fine-tuning at each online interaction step. Empirically, our pipeline,**Hi**erarchical **P**areto **P**olicy **P**ool (**HiP3**), achieves state-of-the-art performance on offline-to-online RL benchmarks with diverse online tasks. Comprehensive ablation studies are conducted to further elucidate the mechanisms behind HiP3.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces HiP3 (Hierarchical Pareto Policy Pool), a model-based offline-to-online reinforcement learning framework. It utilizes MOSAIC (Multiple-Objective Soft Actor-critIC) to generate a pool of Pareto policies that capture different optimism–pessimism trade-offs during offline training, and then adopts a contextual bandit algorithm to adaptively select the most suitable policy for online fine-tuning. Experimental results demonstrate that HiP3 achieves state-of-the-art performance on multiple D4RL benchmarks, particularly showing superior adaptability to novel online tasks. Empirical evaluations confirm its superior performance compared to prior state-of-the-art methods.

### Strengths
1.By formulating the optimism–pessimism trade-off as a bi-objective optimization problem, the authors provide empirical evidence supporting a balanced trade-off between exploration and conservatism, as illustrated in Figure 1, where different Pareto policies exhibit varying optimism–pessimism trade-offs and corresponding online adaptation behaviors.
2. The HiP3 framework integrates MOSAIC with a contextual bandit (LinUCB) for adaptive online policy selection, enabling state-dependent switching between optimistic and pessimistic policies.
3. Extensive experiments on D4RL show consistent state-of-the-art results, particularly in novel online tasks.

### Weaknesses
1. The HiP3 pipeline introduces complexity in both design and computation due to the multi-stage process of generating and selecting from a diverse Pareto policy pool, which increases both algorithmic complexity and the difficulty of implementation.
2. The hierarchical selection strategy appears effective empirically but is justified mainly through intuition and experiments.

### Questions
1. How large is the gap between novel online tasks and the original tasks in terms of task distribution shift? How do the authors propose to measure the difference between the two tasks, and is the novel online task inherently more difficult or easier than the original tasks? Additionally, how does HiP3 maintain performance across such variations?
2. How computationally expensive is maintaining and updating the Pareto policy pool, especially in high-dimensional continuous action spaces?
3. What is the computational overhead (training time and memory) introduced by maintaining multiple Pareto policies compared to other baselines?

### Soundness
3

### Presentation
2

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
The Hierarchical Pareto Policy Pool (HiP3) method was proposed to alleviate the distribution shift in offline-to-online reinforcement learning based on models and to improve the fine-tuning performance. This method mainly consists of the following three modules:

1. A bi-objective formulation was proposed to balance the optimistic (model-predicted rewards) and pessimistic (model uncertainty) two indicators.

2. Multiple-Objective Soft Actor-critic (MOSAIC) extends Soft Actor-Critic (SAC), solves the problem of multi-objective optimization, and combines the neighborhood search method to discover the Pareto optimal strategy.
3. Utilize a hierarchical reinforcement learning approach, and combine a contextual bandit algorithm (LinUCB) as an advanced strategy to reduce the number of interactions with the environment and more efficiently select the most suitable strategy for fine-tuning.

### Strengths
1.	The "Hierarchical Pareto Strategy Pool" (HiP3) method was proposed. It alleviated the distribution bias in the process of offline-to-online reinforcement learning based on models, improved the fine-tuning performance, and achieved efficiency and low consumption.

2.	The theoretical explanations and formula derivations are very thorough.

3.	The experimental verification is very thorough.

### Weaknesses
1.	During the process of generating the strategy pool in the offline stage, the initial and final values of the reference vector, 0.1 and 0.9, were directly set without any related analysis, and there is a possibility that the actual situation may deviate from this.

2.	The values of the reference vectors in the paper are uniform. However, in reality, non-uniform distribution may occur, which could lead to inefficiency.

### Questions
1.	Have you considered using an LLM?

2.	Currently, the experiments are conducted in a fully observable environment. Have we considered exploring the adaptability in more complex environments (such as unobservable environments, etc.)?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Authors address the optimism-pessimism trade-off in model-based offline-to-online RL. They introduce an algorithm (MOSAIC / HiP3) that seeks to get the best of both worlds by learning a pool of policies with different levels of optimism/pessimism, and learning how to switch between them.

### Strengths
- MOSAIC doesn’t require solving prohibitively expensive bi-level optimization problems.
- Authors provide theoretical convergence statement for MOSAIC.
- Authors provide code to reproduce their results.

### Weaknesses
The paper has two key weaknesses: the evaluation environments, baselines, and results are underwhelming; and the motivation is weak, given that the method introduces a lot of complexity that does not seem to yield very significant improvements.

- Very limited evaluation domains: the evaluation suite consists only of 3 different environments (with 4 datasets each) and only of locomotion domains. It is unclear how MOSAIC would perform in other domains.
- The baseline comparisons are insufficient. The only other model-based method included is MBPO, which is 6 years old. Why were other newer model-based RL methods not included, e.g. TD-MPC2 (ICLR 2024 spotlight)?
- Results across the paper use 3 seeds and 10 evaluations each. This is not enough (e.g. in table 1 MBPO and HiP3 have overlapping confidence intervals for total mean).
- The method is significantly more complex than MBPO, but doesn’t get statistically significantly better performance (Tab. 1).
- The explanation in the second paragraph for the proof-of-concept experiment is very unclear and hard to follow. It is unexplained what “pool of Pareto policies” means. It is also unclear what optimism and pessimism mean here. When authors mention fine-tuning these policies, it is unclear what the fine-tuning method they used was. Why dies (a) use unnormalized returns and (b, c) use normalized score?

### Questions
- Figure 1: what is “negative uncertainty predicted by the Env model”?
- How were the environments shown in Figure 3 selected? Can you add another plot with average performance over every environment?

### Soundness
4

### Presentation
3

### Contribution
1
