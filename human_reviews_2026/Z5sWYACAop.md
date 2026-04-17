# Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Sampling efficiency is a key bottleneck in reinforcement learning with verifiable rewards. Existing group-based policy optimization methods, such as GRPO, allocate a fixed number of rollouts for all training prompts. This uniform allocation implicitly treats all prompts as equally informative, and could lead to inefficient computational budget usage and impede training progress. We introduce VIP, a Variance-Informed Predictive allocation strategy that allocates a given rollout budget to the prompts in the incumbent batch to minimize the expected gradient variance of the policy update. At each iteration, VIP uses a lightweight Gaussian process model to predict per-prompt success probabilities based on recent rollouts. These probability predictions are translated into variance estimates, which are then fed into a convex optimization problem to determine the optimal rollout allocations under a hard compute budget constraint. Empirical results show that VIP consistently improves sampling efficiency and achieves higher performance than uniform or heuristic allocation strategies in multiple benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies adaptive selection for rollout trajectories for online reinforcement learning (RL)-based language model (LM) training. Specifically, algorithms like GRPO need a batch of rollouts to calculate the (batch normalized) value function, and there are acquisition cost vs. performance tradeoff depending on the number of sample. The paper discuss how to improve the data efficiency of trajectory retrieval by adaptively allocating the rollout budget in a way to minimize the variance of the policy gradient. In particular, the proposed method adapts Baysian approach to estimate the probability of the outcome becomes positive. The experiment results indicates that adding the adaptive rollout components, the algorithm improves the performance compared to uniform rollout allocation.

### Strengths
- The motivation for the rollout budget cost and performance tradeoff is well-explained.

- The proposed method can be integrated with off-the-shelf learning algorithms, and the experiment results show an improvement in the result.

- The parameter update for adaptive allocation can be done in an online learning approach using Bayesian updates, and does not seem to require huge computation when new data is added.

### Weaknesses
- One concern I have is whether the Bayesian estimation results in biased estimation of the success probability. If I understand correctly, the probability $p$ measures the probability that the outcome sampled on some query results in a positive reward. This probability should depend on the policy's sentence generation probability, and given that the policy is updated during the training phase, the algorithm should be able to handle the non-stationarity of the reward. However, this discussion is not provided, and I wonder whether the estimation is biased towards past policy or the results can change depending on the choice of initial policy.

- Application may be limited to the binary outcome case. It is not obvious how to generalize the idea to the continuous outcome case, especially, how to track the distribution of rewards in such cases.

### Questions
- How does the proposed method track the non-stationarity of the expected reward (occurring by the policy's generation probability)?

- How to generalize the idea to the continuous outcome case?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The submitted manuscript introduces VIP (Verifiable Importance-based Policy), a method for adaptive rollout allocation in reinforcement learning with verifiable rewards.
Instead of assigning a fixed number of rollouts per prompt, the proposed scheme dynamically allocates rollout budgets based on estimated gradient variance, aiming to improve sample efficiency. This allocation is based on simplified variance computations and the computation of a budget constraint optimization problem. Experiments on math reasoning and question-answering tasks show that VIP achieves better accuracy and reward efficiency compared to uniform rollout allocation.

### Strengths
- The paper addresses a relevant and timely problem in reinforcement learning for language models, focusing on efficient rollout allocation under limited budgets.
- The proposed VIP framework is clearly motivated and presented as a practical enhancement to existing RL with verifiable rewards (RLVR) methods.
- The topic is well aligned with current interest in improving training efficiency for large models.
- The proposed idea of allocating samples based on variance estimates, where the unknown dependencies are modeled through a predictive Gaussian process approximation, is both interesting and promising.

### Weaknesses
The theoretical analysis is developed under very restrictive assumptions. It sometimes feels as though several of the key challenges of the original setting have been simplified away in order to make the variance computations tractable. While this makes the analysis cleaner, it also raises concerns about the realism and relevance of the resulting conclusions and the potential for counterintuitive sample allocations.

Concern about Assumption 3.1: The assumption $\pi_{old} = \pi_\theta$ effectively removes the off-policy nature of the considered algorithms. Under this setting, the approach seems to reduce to a REINFORCE-style algorithm (with a baseline), where the importance weights are omitted. This simplification has a substantial impact on the resulting variance expressions, since dropping (even clipped) importance weights can alter the underlying variance structure. Consequently, the proposed allocation behaves as if the algorithm were purely on-policy, which does not reflect how these methods are typically used in practice.

There also appears to be a conceptual inconsistency in the motivation. The paper begins by discussing algorithms such as PPO and GRPO, which rely on partially off-policy training through clipped importance weights. Later, however, by adopting Assumption 3.1, the analysis assumes all weights are one and argues that purely on-policy training is preferable. This shift in perspective makes it unclear why the analysis focuses on those off-policy algorithms rather than directly studying the on-policy methods.

Concern about Assumption 4.1: Both parts of this assumption seem rather strong. The independence between $\tilde R$ and $\tilde Z$ is understandable as a simplifying step, but the second part appears difficult to justify in practice. The statistical tests in Appendix B do not provide convincing evidence for these assumptions, they only show that there is insufficient evidence to reject $H_0$. Similar concerns apply to the assumption that all $\tilde Z_q$ share the same variance.

Taken together, Assumptions 3.1 and 4.1 make the variance derivations in Propositions 4.2 and 4.3 relatively straightforward. However, this comes at the cost of removing much of the complexity inherent in the real algorithms. As a result, the computed variances differ substantially from those of the actual gradient estimators used in the experiments.

Finally, while the empirical results show that VIP yields consistent improvements over uniform rollout allocation on several reasoning benchmarks, the gains are difficult to interpret without additional context. In particular, computational costs are not reported (solving the allocation optimization and fitting the Gaussian process introduce extra overhead), and the experiments lack statistical validation such as error bars or multiple runs, making it difficult to assess the robustness of the observed improvements.

### Questions
- Under Assumption 3.1, what is the difference between the proposed method and the REINFORCE algorithm?
- In equation (3), where does $\tau$ appear? Shouldn’t $A_j$ depend on $\tau$?
- Why is it feasible to project the vector $H$? While projecting the gradient of $J$ would correspond to a projected gradient method, projecting $H$ itself appears to introduce a bias.
- How do the results in Table 1 change when considering computational cost? Does the budget-allocation procedure introduce significant additional runtime?
- Are the variances of $Z_q$ assumed to be known? If so, why is this assumption considered feasible in practice?

Typos:
- Line 50: …ha[ve] emerged…
- Line 97: Section title should be Related Work[]
- Line 171: However, [o]ur paper…
- Line 266: …denote… denoted…
- In general, I suggest to work over the reference list. In some references there are publisher information missing (Liao et al. 2025, Lin et al. 2025, Sun et al. 2025).
- Some abbreviations are multiple times defined.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes adaptive rollout allocation for group-based RL with verifiable rewards: instead of using a fixed number of rollouts per prompt, it predicts each prompt’s success probability and allocates the rollout budget to minimize expected gradient variance. Concretely, the method (“VIP”) fits a lightweight Gaussian Process over prompt embeddings to estimate per-prompt success probabilities, translates them into variance proxies, and solves a convex budget-allocation problem (with an integer rounding heuristic) under a hard rollout budget. Experiments on math reasoning and tool-augmented reasoning show higher accuracy than uniform or heuristic allocation at the same total rollout budget.

### Strengths
he paper derives per-prompt gradient-variance formulas for Dr.GRPO and RLOO, motivating variance-aware allocation rather than uniform rollouts.

On AIME24/25 and tool-augmented retrieval, VIP consistently improves accuracy/quality over uniform allocation while keeping the same total number of rollouts.

Replacing either the GP predictor or the optimizer degrades performance, suggesting both components contribute meaningfully.

### Weaknesses
Results are reported under the same rollout budget; however, adaptive per-prompt rollout counts can change the number of optimizer steps and total time. The paper should add (a) wall-clock time vs. accuracy and (b) equal-step comparisons to isolate compute-efficiency.

Training a GP and solving the allocation each iteration adds overhead; the paper does not provide detailed runtime/memory profiling vs. uniform baselines.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
