# Policy Optimization with $f$-Divergence Regularization

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Policy iteration is a common algorithm framework in reinforcement learning (RL) to find the optimal policy for a Markov decision process (MDP). To improve training stability and prevent catastrophic failure, researchers have developed several policy iteration algorithms based on the Kullback-Leibler (KL) divergence, such as the well-known trust region policy optimization (TRPO) and proximal policy optimization (PPO). However, these methods are limited to the KL divergence, which may not be the best choice for all environments. In this work, we generalize previous work using a more general form of divergence, the $f$-divergence, and design a new family of algorithms that can improve learning policy with theoretical improvement guarantees. Our method, $f$-divergence-regularized policy optimization ($f$RPO), can be applied to both online and offline RL settings. Empirical studies show that $f$RPO can outperform existing methods, including the commonly used KL divergence, on common benchmark problems in RL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a general framework for policy optimization in reinforcement learning based on f-divergence regularization. The authors generalize this idea by introducing f-divergence-regularized Policy Optimization (fRPO), which replaces KL with a broad family of f-divergences (including reverse-KL, Pearson, Neyman, and Hellinger) and provides both theoretical guarantees and potential empirical benefits of doing so.

### Strengths
1. **Theoretical soundness**. The derivation of the proposed fRPO framework is theoretically sound. The paper not only provides a rigorous proof of policy improvement but also reveals the underlying connections among many existing methods, offering a unified view of regularized policy iteration.
2. **Evaluation across online and offline settings.** The paper conducts experiments in both online and offline RL settings, demonstrating that the proposed approach can adapt to most mainstream RL paradigms.

### Weaknesses
Although the paper presents a clear theoretical motivation and rigorous proofs, the main weaknesses lie in the experimental section.

1. **Limited experiments.**
The experiments are primarily conducted on MuJoCo control tasks, which are relatively simple and may not fully demonstrate the generality of the proposed method. For a framework that claims to be broadly applicable, evaluation on a wider and more diverse set of environments (e.g., Atari) would provide stronger empirical support.
In addition, the number of random seeds appears to be quite limited. Given that reinforcement learning performance can vary significantly across seeds, the reported results—both in online and offline settings—are not entirely convincing without variance measures or statistical significance tests.

2. **Choice and performance of baselines.**
Some baseline methods, such as PPO in the online experiments, exhibit unexpectedly low performance, which raises concerns about the fairness or correctness of the comparison. 

3. **Algorithmic empirical performance.**
Moreover, in several tasks the performance improvements from replacing KL with other f-divergences are relatively marginal (even though the paper does not explicitly claim large gains). This observation partially undermines the practical motivation for adopting fRPO: **if the performance difference is small, not guaranteed, and not clearly explained**, it remains unclear why one should prefer fRPO over existing KL-based approaches.

### Questions
1. Could the authors provide insights—perhaps in terms of exploration–exploitation balance, policy conservatism, or distributional asymmetry—that explain **how the choice of f affects the behavior and performance of the algorithm across different environments**?

I believe this is likely the core issue of the paper. If the authors could provide a clear explanation—and ideally, supporting experiments—the work would become much more complete and compelling.

### Soundness
2

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
4

### Summary
The paper proposes f-divergence–regularized policy optimization (fRPO), a policy-iteration framework that uses general f-divergences as the regularizer in the policy-improvement step. It derives a closed-form target policy and a practical neural-network training recipe, and proves a monotonic-improvement bound for the theoretical update. Experiments in onlineand offline settings show competitive and in some cases superior performance compared to KL-based baselines.

### Strengths
The paper provides sufficient theoretical analysis, including a clear derivation and improvement guarantee.

The experimental results are rich, evaluating several f-divergence choices against competitive baselines.

### Weaknesses
1. **Motivation feels weak**:
While I understand that different environments may benefit from different regularizers, the paper does not clearly articulate the specific problem fRPO solves. For example, TRPO addresses large on-policy update steps that make fresh rollouts unreliable; PPO offers a simplified, widely usable alternative. In contrast, this paper doesn’t pinpoint a concrete failure mode or practitioner pain point that fRPO uniquely resolves. Given that PPO and SAC are standard in robotics and GRPO is popular in LLM training, it would be more informative to explain and demonstrate how fRPO helps in practice beyond reporting score deltas.

2. **Formulation and hyperparameter selection**: 
The empirical results suggest mixed outcomes across different f-divergences, and choosing among them feels similar to tuning PPO/SAC hyperparameters, and fRPO also introduces additional hyperparameters. From users' perspective, it’s unclear why one would invest extra tuning effort here rather than further tuning PPO or SAC. It would be much more useful if the paper provided guidance for selecting the divergence and defaults, such as simple heuristics tied to application and recommended hyperparameter ranges.

### Questions
My questions are link to the weaknesses I raise, i.e.

1. What concrete problem does the proposed method solve? In other words, what specific performance capability or apllication does fRPO enable that existing methods do not?

2. How should practitioners choose the f-divergence for a given use case? Is there a principled selection rule or practical guidance with defaults for different regimes?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an $f$-divergence–regularized policy-optimization framework (“fRPO”), where each iteration computes a closed-form, non-parametric target policy by maximizing a standard advantage surrogate penalized by a general $f$-divergence to the current policy, then projects to a neural policy by a forward-KL fit. Theoretically, it gives a TRPO-style monotonic improvement inequality for arbitrary $f$-divergences by relating total variation to $f$-divergences. Empirically, it shows comparable performance to KL-based methods (TRPO/PPO/AWR, etc.) with improvements on certain choices of $f$.

### Strengths
1. The paper generalizes KL-regularized policy improvement to arbitrary f-divergences and derives a closed-form target policy.
2. Broadening the choice of f-divergence yields some empirical gains.

### Weaknesses
1. While this work extends regularization to arbitrary f-divergences directly in policy space, givent that Belousov & Peters already generalized trust regions to f-divergences at the occupancy (state–action distribution) level and highlighted the benefits of $f$-divergence. Hence, the novelty feels limited.
2. As the main theoretical motivation and contrast, Belousov & Peters is not included as an experimental baseline, leaving the advantage of deriving in policy space (vs. occupancy space) unclear.
3. The lower-bound, TRPO-style improvement guarantee is standard for this family of methods and appears relatively straightforward and incremental.
4. Other related f-divergence based RL works [1-2] are not discussed in the paper.

[1] Agarwal et al. f-Policy Gradients: A General Framework for Goal-Conditioned RL using f-Divergences. NeurIPS 2023

[2] Gong et al. The f-Divergence Reinforcement Learning Framework. 2022

### Questions
How does the computational cost and complexity of the closed-form f-divergence–regularized update followed by a forward-KL projection compare with the baselines?

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
Overall, the paper provides a well-structured and mathematically consistent formulation of policy-regularized reinforcement learning that generalizes TRPO to an arbitrary f--divergence. The authors derive a monotonic policy improvement bound under standard convex optimization assumptions, propose a closed-form target policy using convex conjugates, and develop a practical algorithm (fRPO) that is applicable to both online and offline settings.

Empirically, the method is evaluated on MuJoCo and D4RL benchmarks, demonstrating that non-KL divergences (e.g., Pearson, Neyman) can achieve competitive or superior performance to KL-based baselines such as TRPO, PPO, and AWR.

However, while the theoretical development is sound and the presentation clear, the theoretical novelty appears limited and the empirical gains seem relatively marginal.

### Strengths
The paper offers a coherent and mathematically sound formulation of a "general" policy-regularized reinforcement learning framework, though certain aspects could be further clarified or strengthened. The main strengths of the paper are summarized below, and I will ask for specific clarifications in the question section later.

* It derives the monotonic improvement bound following the TRPO [1] framework, and the extension to an arbitrary $f$-divergence is logically valid under the standard convex optimization framework.

* The use of convex conjugate analysis and the resulting dual representation of the policy update are clearly written, showing solid understanding of policy regularization through convex optimization.

* The overall theoretical exposition is clean. The notation and derivations align with standard results in regularized RL (e.g., online: TRPO [1], and offline: SQL [2]).

*Reference*

[1] John Schulman "Trust Region Policy Optimization" Proceedings of the 32nd International Conference on Machine Learning.

[2] Xu, Haoran, et al. "Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization." The Eleventh International Conference on Learning Representations.

### Weaknesses
* Lack of Novel Theoretical Contribution (Online RL)

   * The paper extends the monotonic improvement guarantee of TRPO [1] by replacing the KL divergence with a general $f$-divergence. While this extension is mathematically correct, it represents a straightforward generalization rather than a fundamentally new theoretical insight. The proof closely follows the original TRPO derivation, with no substantial change in assumptions, bounding techniques, or underlying theoretical implications.

* Similarity to SQL [2] (Offline RL)

   * The paper applies a behavior-regularized reinforcement learning framework to offline RL, but the resulting convex optimization formulation and loss function appear almost identical to those used in Sparse Q-Learning (SQL) [2].

* Limited Empirical Significance

   * Although the paper reports performance improvements in both online (Table 2) and offline (Table 3) settings, the reported gains over strong baselines such as TRPO, PPO, SAC, and IQL appear relatively marginal. The differences seems to be within the range of variance typically observed in these benchmarks, making it unclear whether the improvements are statistically significant or practically meaningful.
   * In addition, the paper does not provide confidence intervals or any measure of statistical significance for the reported results. This omission makes it difficult to assess whether the observed performance differences are consistent or simply due to random fluctuations across seeds.

### Questions
* Q1. Effect of Different *$f$*-Divergences on Policy Characteristics

The paper replaces the KL divergence with a general f-divergence, which is theoretically correct. However, it remains unclear how this substitution affects the qualitative behavior or characteristics of the learned policy. For instance, do certain divergences encourage more conservative, exploratory, or stochastic policies compared to KL? In addition, could the authors elaborate on how these differences translate into practical advantages in (online/offline) RL settings?

* Q2. Could the authors clarify how the proposed formulation fundamentally differs from SQL[2], both in terms of theoretical motivation and algorithmic structure?

* Q3. Statistical Significance of Reported Results

The experimental results in Tables 2 and 3 are presented without any indication of statistical significance, such as confidence intervals or standard deviations across random seeds. Could the authors provide such information? Including these measures would strengthen the empirical claims and help assess whether the proposed method’s improvements are statistically meaningful.

* Q4. Minor Question Regarding Mathematical Expression

In the paragraph right above Section 3.3, could the authors clarify what it means to “approximate” *$D_f(\pi_k|\pi_k)$*? As currently written, the expression of it seems to evaluate to zero in that context, so it is unclear why an approximation is needed. This may be a typographical or notational issue, and a clarification would be appreciated.

### Soundness
2

### Presentation
2

### Contribution
1
