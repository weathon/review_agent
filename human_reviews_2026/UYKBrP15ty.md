# Variance-Reduced Reinforcement Learning for Large Reasoning Models via James-Stein Baselines

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 8, 4, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) is becoming an impactful paradigm in large reasoning model (LRM) post-training.
To stabilize training, control variates (baselines) are commonly introduced, canonically chosen to approximate the value function. Popular approaches such as RLOO and GRPO estimate baselines with per-prompt empirical
averages of generated response, which can exhibit high variance under limited rollout budgets. Recognizing that value functions must be estimated simultaneously across all prompts in a batch, we propose a James–Stein estimator as the baseline. This approach leverages statistical shrinkage to reduce the mean squared error in the overall value function estimation, without additional computational overhead while maintaining the unbiasedness of the policy gradient estimator. We provide theoretical justification for James-Stein baselines and validate it empirically. Across diverse models, tasks, and rollout budgets, our approach consistently outperforms existing baselines, demonstrating robust variance reduction and improved training stability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a new variance-reduced technique for reinforcement learning with verifiable rewards (RLVR), inspired by the James-Stein shrinkage principle. The proposed method, called James-Stein Policy Optimization (JSPO), aims to reduce the variance of policy gradient estimates in RLVR settings without introducing additional computational overhead. By leveraging statistical shrinkage, the JSPO baseline trades a small amount of bias for a significant reduction in mean squared error (MSE), leading to improved training stability and efficiency.

### Strengths
The introduction of the James-Stein estimator as a baseline for variance reduction is novel, and the approach is easy to implement. The authors provide theoretical justification for the use of the James-Stein estimator, demonstrating its effectiveness in reducing policy gradient variance theoretically and empirically.

### Weaknesses
The notation in the paper is overwhelming and can be confusing. It is not always clear which random variable the expectations and variances are being taken with respect to. A table summarizing the key notations and their contexts would be helpful for clarity. For other questions please refer to the following. I would be willing to raise  my  score if the authors address my confusion regarding these points.

### Questions
1. What is the exact definition of $b_i^j$ in equ (4) and elsewhere in the paper? It would be beneficial to give a clear expression.
2. In Line 185, is $\mathbb{E}_{Y}[b_i^j]=\mathbb{E}[b_i^j]$ satisfied? In my opinion, the LHS take expectation with respect to the response $y_j$ following the $\pi(\cdot|x_i)$, while the RHS appears to take expectation over both the response $y^j_i$ and the prompt $x_i$. Could you clarify if this equality holds?
3. The James-Stein estimator is derived under assumptions that are closely tied to **Gaussian distributions**. How does this approach behave when reward distributions are skewed or multimodal, which are common in complex reasoning tasks? Would the shrinkage still yield a reduction in mean squared error (MSE) under these conditions ?
4. RLVR typically involves sparse rewards, with feedback provided at the end of a reasoning process. How do the James-Stein estimator, address the challenges posed by sparse rewards, or how does this impact the learning efficiency and stability of the model in sparse reward settings?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes James–Stein Policy Optimization (JSPO): a new baseline (control-variate) for RL with verifiable rewards (RLVR) used to fine-tune large reasoning models. The key idea is to estimate each prompt’s value not in isolation (per-prompt mean, as in RLOO/GRPO), but jointly across the whole batch via James–Stein shrinkage toward the batch mean, implemented with a two-level leave-one-out construction to keep the policy-gradient estimator unbiased. Theoretical claims include: (i) Proposition 1—unbiasedness of the gradient when the per-sample baseline is independent of the reward; (ii) a bias–variance decomposition showing why prompt-only means are inadmissible under batch MSE; (iii) Proposition 2/Theorem 1—a closed-form optimal shrinkage coefficient that trades off within-prompt noise vs. across-prompt variability and reduces policy-gradient variance while preserving unbiasedness. Empirically, JSPO improves Pass@1 on math/logic benchmarks over RLOO/GRPO and reduces estimated gradient-variance by ~11–32% across several models and rollout budgets (2/4/8 generations). The method is drop-in, has negligible overhead, and no extra hyperparameters.

### Strengths
Clear, principled objective: Reduces policy-gradient variance by reframing baseline design as multi-task value estimation with an MSE criterion; the James–Stein angle is elegant and well-motivated. 
Unbiasedness preserved: The two-level leave-one-out construction (per-prompt and per-batch) keeps baselines independent of the held-out reward, matching the REINFORCE unbiasedness requirement. 
Consistent empirical gains: Improvements across tasks (MATH500, OlympiadBench, AMC/logic puzzles; also GSM8K with 2/4/8 rollouts), plus direct gradient-variance tracking that aligns with the theory. 
Practicality: Critic-free, tiny code change (a few lines), batch-only statistics, compatible with existing RLVR stacks

### Weaknesses
Ablations could be deeper: How sensitive are outcomes to batch size n and rollouts m beyond the tested grid? What happens with non-binary, dense, or scale-shifted rewards?

Distributional assumptions not stress-tested: James–Stein style shrinkage helps when prompts share a latent mean structure. If batch prompts are intentionally heterogeneous (mixed tasks/difficulties), shrinkage could over-bias the baseline; the paper largely uses in-distribution batches.

Generalization beyond reasoning tasks: All experiments target math/logic puzzles; evidence on coding, tool-use, or longer-horizon tasks would strengthen claims of generality.

### Questions
Robustness of λ̂: Do you use any clipping or shrinkage-to-zero floors/ceilings for λ̂ to avoid instability with small n or heavy-tailed rewards?

In Figure 3, JSPO’s improvement over RLOO on Countdown is noticeably smaller than the other three tasks. Could you diagnose why?

Table 1 (fairness across methods): Are decoding/hyperparameters (temperature, top-p, token limits) identical for ReMax, REINFORCE++, GRPO, RLOO, BLOO, and JSPO? If any differ, please list them, since small decoding shifts can change GSM8K accuracy.

### Soundness
3

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
2

### Summary
The paper proposes a James–Stein–based baseline for policy-gradient training of large reasoning models (RLVR). By shrinking per-prompt reward means toward a batch mean—via a leave-one-out construction to preserve unbiasedness—the method aims to reduce gradient variance and thereby stabilize training. The authors provide a derivation linking variance of the policy-gradient estimator to MSE of the baseline, prove an optimal shrinkage coefficient, and give an implementation that adds negligible overhead. Experiments on math and logic-puzzle benchmarks show consistent accuracy gains and lower estimated gradient variance compared to RLOO/GRPO-style baselines under multiple rollout budgets.

### Strengths
The paper cleanly reduces policy-gradient variance control to estimating a value-function baseline, motivating James–Stein shrinkage and proving an optimal (data-driven) coefficient with an unbiased leave-one-out construction.

The baseline is easy to add to existing critic-free RLVR pipelines and is presented with concise pseudo-code.

### Weaknesses
While well-executed, the paper extends a long line of “better baseline” work; the novelty is primarily in bringing James–Stein shrinkage to RLVR with careful LOO plumbing, rather than introducing a new learning paradigm.

Evidence is confined to RLVR-style reasoning tasks (math/puzzles); there is no evaluation on broader RL settings where baseline design has also been heavily studied, which limits generality claims.

### Questions
When batch prompts are heterogeneous, performance may revert toward LOO means. Can you provide a runtime diagnostic and an automatic rule for disabling or annealing shrinkage?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a variance-reduced baseline for policy gradient methods in reinforcement learning with verifiable rewards (RLVR) on large reasoning language models. The central idea is to use a James-Stein-type shrinkage estimator as a control variate in policy gradient updates, adaptively blending per-prompt and global-batch mean reward estimates. The method, termed JSPO, is theoretically motivated to reduce mean squared error and, via a carefully constructed leave-one-out strategy, maintains unbiased gradients. The paper offers both theoretical justification and extensive experiments showing JSPO’s benefits over existing baselines in reducing gradient variance and improving downstream performance on mathematical and logical reasoning tasks.

### Strengths
1 The utilization of the James-Stein estimator builds directly on well-understood results in statistics, and the paper provides thorough derivations connecting batch-level value function estimation to baseline variance reduction (see especially Section 3.3 and Theorem 1, page 6).

2 The leave-one-out adaptation of the shrinkage estimator to RLVR is both elegant and practically impactful, ensuring unbiased gradients while achieving variance benefits. The mathematical treatment is careful, with Proofs of Proposition 1 and Theorem 1 provided in Appendix B.

3 SPO outperforms RLOO, GRPO, ReMax, and BLOO across all tested rollout regimes, including challenging low-rollout settings where baseline variance is most limiting

### Weaknesses
1 The methodology, though thoughtfully adapted, is essentially a direct application of classical James-Stein shrinkage (as in James et al., 1961 and Stein et al., 1956) to RLVR policy gradient baselines. While the adaptation—especially the unbiased leave-one-out variant—is useful, the paper could more rigorously discuss the theoretical/empirical boundaries between what is gained by the JSPO version versus more general empirical Bayes or shrinkage estimators (see Feldman et al., 2014; Efron & Morris, 1973; Brown, 1971).
The related work section does not discuss these key foundational works adequately, which reduces the clarity on conceptual originality (see the "Potentially Missing Related Work" section below).

2 While the experiments are broad, the scope is still largely limited to math and logic reasoning tasks. There is insufficient evidence to claim generality across different RLVR domains (e.g., for instructions, summarization, or control tasks).

3 There is no ablation studying what happens if rollouts are highly non-i.i.d. across prompts, nor is there an analysis of the estimator’s robustness to reward sparsity or distributional shift.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
