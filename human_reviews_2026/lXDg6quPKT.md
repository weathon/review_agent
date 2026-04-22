# Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has emerged as the leading approach for enhancing reasoning capabilities in large language models. However, it faces a fundamental compute and memory asymmetry: rollout generation is embarrassingly parallel and memory-light, whereas policy updates are communication-heavy and memory-intensive. To address this, we introduce **PODS** (**P**olicy **O**ptimization with **D**own-**S**ampling), which decouples rollout generation from policy updates by training only on a strategically selected subset of rollouts, maintaining learning quality while dramatically reducing update costs. We propose a principled subset selection criterion—*max-variance down-sampling*—that maximizes reward diversity, and provide an efficient $O(n\log n)$ implementation. Empirically, Group Relative Policy Optimization (GRPO) with PODS achieves the peak test accuracy of vanilla GRPO at least $\mathbf{1.7\times}$ **faster** across the different reasoning benchmarks and hardware configurations we tested.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces PODS, a data selection framework designed to improve the efficiency of GRPO training. The key idea is that rollout generation can be efficiently parallelized via batching, whereas policy updates become costly with large rollout sizes. To address this, PODS downsamples generated rollouts by selecting those with the lowest and highest rewards, thereby maximizing reward variance within each training batch.

### Strengths
+ GRPO efficiency is an important and timely problem in RLHF.
+ The paper is overall well-written and well-structured.
+ Empirical results show encouraging performance improvements.

### Weaknesses
- The contribution is relatively incremental, both in the data selection strategy and in algorithmic design.
- The evaluation is limited in scope: benchmarks focus only on GSM8K and MATH, and baselines are restricted to the naive GRPO implementation. Including code generation or reasoning benchmarks would strengthen the results. Also, comparative analysis could be more comprehensive using relevant baselines, such as GRESO [1].

### Questions
Thank you for submitting this work. The paper tackles an important problem in RLHF training efficiency, but I am concerned that the novelty is limited and that key design choices are not sufficiently justified. Below are specific comments and questions:

- Q1: he complexity of Algorithm 1 should be O(mlogn), since maintaining the m lowest and highest rollouts in a priority queue avoids sorting the entire list. Could the authors clarify?

- Q2: How does PODS perform on more diverse tasks and benchmarks, such as code generation or reasoning datasets? Including stronger baselines (e.g., GRESO [1]) would make the results more convincing.

- Q3: Could the authors elaborate on why PODS leads to better final model performance? This improvement and claim seem highly sensitive to the choice of the hyperparameter m.

Reference:

[1] Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts, NeurIPS 2025 / arXiv:2506.02177.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This work introduces PODS (Parallel Optimized Down-Sampling), a lightweight and algorithm-agnostic framework designed to tackle a core inefficiency in modern Reinforcement Learning with Verifiable Rewards (RLVR): the mismatch between highly parallelizable rollout generation and memory-constrained, sequential policy updates. PODS addresses this by generating large batches of rollouts in parallel and selectively updating the policy on a small, informative subset chosen via a max-variance selection rule. 

Despite its simplicity, PODS delivers consistent improvements: under identical wall-clock time budgets, it outperforms standard GRPO, achieves at least a 1.7× training speedup, and attains higher final accuracy across diverse model architectures, scales, and deployment settings.

### Strengths
(1) Well-written
(2) Detailed experiment
(3) The problems related to training efficiency that have been solved are distinctive and seem valuable to the industrial sector

### Weaknesses
N/A

### Questions
I do not know this specialized research field very well. I will adjust my score and optimize my review document based on the evaluations of other expert reviewers and my performance during the rebuttal period.

### Soundness
3

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
3

### Summary
This paper aims at the computational inefficency RLVR for large languange models. This paper identify a key asymmetry between the paralleizable rollout generation phase and memory-intensive policy update phase. To mitigate this, this paper proposes PODS, a general framework that generates rollouts but selective update the policy with a subset of the rollouts. Meanwhile, thie paper introduces a max-variance down-sampling method to select rollout with maximum reward diversity. Experiments with GRPO on GSM8k and MATH shows that PODS maintains comparable accuracy performance and more than 1.7x faster convergence.

### Strengths
1. Originality: This paper tackles an underexplored issue in LLM RL. The proposed PODS framework and max-variance down-sampling criterion represent an interesting aspect of efficiency optimization. This work distincts from existing prompt-selection or gradient-accumulation approaches. The idea of selective subset section is conceptually simple and original.
2. Quality: The methodology is clearly formalized with simple mathematical justification. Theoretical analysis provides clarity and computational guarantees for the proposed down-sampling rule. The experiments are comprehensive, spanning different model sizes, datasets, and hardware setups.
3. Clarity: The paper is well-written and easy to follow. The motivation for the problem is clearly illustrated. The presentation of algorithms and visual comparisons are well-structured. 
4. Significance: The work addresses a bottleneck in scaling reinforcement learning for LLMs, reducing memory and communication overhead during policy updates. Given the rising computational costs in reasoning-focused RL for LLMs, the proposed method is timely and practically impactful. This framework can be integrated with other RL variants, increasing its potential influence and relevance for large-scale LLM training.

### Weaknesses
1. LImited scope empirical validation. As mentioned in the limitation section, the evaluation is conducted on mathematical reasoning tasks (GSM8K, MATH) with rule-based reward models. These tasks have well-defined, verifiable rewards, which may overstate the benefits of the method. The paper would be significantly strengthened by including results on other multiple tasks where reward distributions are noisier and less binary.
2. Dependence on single baseline RLVR algorithm. Although the method is claimed to be algorithm-agnostic, all experiments and analyses are built around GRPO. It remains unclear whether the same variance-based selection criterion would hold for other RL frameworks.  Demonstrating adaptability across multiple RL algorithms would make the contribution more convincing.
3. Lack of deeper theoretical justification of learning dynamics. The paper provides a sound combinatorial analysis for max-variance selection but does not connect this formally to expected policy improvement or gradient variance reduction. Without a theoretical link between rollout diversity and learning efficiency, the variance criterion, while intuitive, remains heuristic.
4. Limited discussion of potential trade-offs: While the paper reports 1.7×–3× speedups, it provides little detail about wall-clock breakdowns (e.g., inference vs. update time) or the communication cost savings. Furthermore, potential drawbacks such as off-policy bias or degradation of gradient fidelity with extreme down-sampling ratios are acknowledged but not empirically investigated.
5. Meanwhile, the LLM used in this paper is simply 3B and 7B, of which the inference time consumption and training time is fair. However for larger LLMs, the inference time and training time increase significantly. The time complexity of subset selection, nlogn, is not that important, especially for rollout number of 64 and subset size 16. The reduction of average training seconds per training step is mainly caused by the reduction of training data. Therefore, the time complexity of this subset selection is fair theoretically but limited in actual training.

### Questions
1. The evaluation focuses exclusively on GSM8K and MATH which have verifiable and relatively noise-free rewards. Can the authors discuss any preliminary evidence or theoretical reasoning suggesting that the max-variance down-sampling rule remains effective under noisy or sparse reward distributions?
2. While PODS is described as algorithm-agnostic, could the authors explain what modifications, if any, would be required to adapt PODS to reinforce++ or to reinforcement learning from human feedback (RLHF) setups? Please provide one of the demonstration results.
3. Since PODS alters the effective training distribution by selective sampling, does this introduce off-policy bias relative to standard on-policy GRPO?
4. Have the authors considered or tested any correction mechanisms (e.g., importance weighting) to mitigate this potential bias? A short theoretical justification or empirical comparison could help assess the trade-off between efficiency and policy fidelity.
5. The proposed max-variance rule is intuitive and empirically effective, but the paper does not formally connect it to expected policy improvement. Could the authors provide analytical or empirical arguments showing that higher reward variance indeed correlates with more informative gradients or faster convergence in GRPO-like updates?
6. The paper reports wall-clock speedups, but it would be helpful to see a breakdown of inference time, policy-update time, subset data selection time, especially in distributed setups.
7. How does max-variance down-sampling compare to other principled criteria such as entropy-based selection, uncertainty sampling, or advantage-based weighting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets a very concrete bottleneck in reinforcement learning for LLM reasoning with verifiable rewards: rollout generation is cheap and highly parallelizable, but policy updates (GRPO/PPO-style) are memory- and communication-bound, so we can’t just “generate more rollouts” to improve sample quality. The authors propose PODS (Policy Optimization with Down-Sampling): for each prompt, generate a large pool of rollouts, score them with the verifiable reward, and then select only a small, most informative subset for the policy update. The key technical piece is a max-variance selection rule: choose the size-𝑚 subset whose rewards have the largest variance, which they show always corresponds to “take some of the lowest and some of the highest” rollouts; this yields an efficient $O(nlogn)$ algorithm. Plugged into GRPO, PODS achieves 1.7×–3× faster wall-clock to the same or better accuracy on GSM8K/MATH and across Qwen2.5 and Llama 3.2 models, on both single-GPU (LoRA) and multi-GPU (full-param) setups.

### Strengths
1. Very well-motivated problem. The paper identifies a real training-systems bottleneck: inference scales, but updates don’t.
2. Simple, plug-in idea. PODS is architecturally lightweight: keep your GRPO pipeline, just over-generate and then down-sample. This makes adoption easy.
3. Principled selection rule. Instead of just “pick top-k,” they argue that we want reward diversity. The max-variance formulation + structure theorem is a nice, clean piece of theory that justifies the heuristic.
4. Strong empirical evidence. Multiple models, multiple hardware regimes , and realistic tasks. The improvement is in wall-clock, not just final accuracy, which matters in practice.
5. Clear ablations. They study both the number of generated rollouts n and the number of kept rollouts m, and compare to random and max-reward selection. The proposed max-variance rule consistently wins or ties.

### Weaknesses
1. Task scope is narrow. All experiments are on verifiable math-style rewards. These are low-noise, almost binary rewards. It’s not fully clear that the same max-variance rule is optimal when rewards are noisy, delayed, or dense (e.g., coding with partial credit, preference RL, or tool-use tasks).
2. Mild off-policy effect not deeply analyzed. By selecting only a subset of generated samples, the method introduces some off-policy bias relative to pure on-policy GRPO. In practice it seems fine (results look good), but the paper could say more about stability under very aggressive down-sampling (e.g. keeping 2 out of 64).
3. Assumes you can cheaply over-generate. The whole story relies on the common RLVR setup where generation is the easy part. In settings where inference is also bottlenecked (long contexts, tools, multi-turn), the benefit may shrink.
4. No non-verifiable / preference benchmark. Even one experiment on a noisier or non-binary task would make the generality claim stronger.

### Questions
1. Reward noise: How sensitive is max-variance selection when reward signals have small stochastic noise (e.g., randomized unit tests for code)? Does the bottom-and-top structure still hold in practice?
2. Extremely unbalanced batches: If almost all rollouts succeed (or all fail), does the algorithm degrade gracefully to a reasonable selection (e.g. pick the rare failures/successes)?
3. Generalization to other objectives: You show PODS with GRPO. Would you expect the same variance criterion to work for PPO-like preference RL where the “contrast” is not purely on reward but on pairwise preferences?

### Soundness
3

### Presentation
3

### Contribution
3
