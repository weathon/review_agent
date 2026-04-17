# Kalman Filter Enhanced Group Relative Policy Optimization for Language Model Reasoning

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
The advantage function is a central concept in RL that helps reduce variance in policy gradient estimates. Recently, for language modeling, Group Relative Policy Optimization (GRPO) was proposed to compute the advantage for each output by subtracting the mean reward, as the baseline, for all outputs in the group. However, it can lead to high variance when the reward advantage is inaccurately predicted. In this work, we propose Kalman Filter Enhanced Group Relative Policy Optimization (KRPO) model, by using lightweight Kalman filtering to dynamically estimate the latent reward baseline and uncertainty. This filtering technique replaces the naive group mean, enabling more adaptive advantage normalization. Our method does not require additional learned parameters over GRPO. This approach offers a simple yet effective way to incorporate multiple outputs of GRPO into advantage estimation, improving policy optimization in settings where highly dynamic reward signals are difficult to model for language models. Through the accuracies and rewards obtained from math question answering and reasoning, we show that using a more adaptive advantage estimation model, KRPO can improve the stability and performance of GRPO. The code is available at https://anonymous.4open.science/r/KRPO-E1D3.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes KRPO, a GRPO-style policy-optimization method that replaces the group-mean baseline with a Kalman filter (KF) to estimate a per-prompt latent reward mean and its uncertainty. Concretely, the authors model the reward samples for a prompt as noisy observations of a 1D latent state and compute an advantage by normalizing the innovation with the (reported) posterior variance. The aim is to obtain an uncertainty-aware advantage that stabilizes updates and improves sample-efficiency. Experiments on math reasoning benchmarks (e.g., GSM8K/MATH) report consistent gains over GRPO and related baselines.

Contributions: (i) framing GRPO’s baseline estimation as Bayesian state estimation; (ii) a simple KF-based advantage that (purportedly) down-weights noisy samples; (iii) empirical evidence of improved stability and accuracy.

### Strengths
- Simple, potentially impactful idea: Re-casting the group baseline as a KF offers a principled path to uncertainty-aware advantages with minimal code changes.

- Empirical promise: Reported improvements on reasoning benchmarks; qualitative plots suggest smoother training dynamics.

- Practicality: Method is lightweight, compatible with existing GRPO pipelines, and conceptually extensible (e.g., robust KF/UKF/particle variants).

### Weaknesses
The main methodological concern is that, in its current form, the estimator for the advantage appears to be biased and order-sensitive, which directly affects the validity of the reported gains.These issues are critical to address.

- Advantage uses posterior baseline/variance conditioned on the same reward sample, risking biased gradients.
- Normalization by `P_{i|i}` (posterior) instead of the innovation variance `S_i = P_{i|i-1} + R` mis-scales residuals and creates order-dependent weights.
- Permutation invariance claim unsubstantiated: With sequential updates and `Q > 0`, Kalman Filter estimates are intrinsically order-dependent; no ablation quantifies the effect.
- Assumption gaps: Stationarity/CLT justifications are not validated; rewards are discrete and may be non-Gaussian/heavy-tailed.
- Reporting gaps: Incomplete details for `(Q, R)` selection, KL weight notation, significance testing (test type, effect sizes, multiple comparisons).
- Connection to GRPO as a “special case” is imprecise: Batch (simultaneous) updates with `Q = 0` differ from sequential updates with `Q > 0`; the equivalence needs formalization.

### Questions
1. Why is the posterior baseline `x̂_{i|i}` and variance `P_{i|i}` used rather than the prior `x̂_{i|i-1}` and innovation variance `S_i`? Please provide derivations or justify unbiasedness under your policy-gradient estimator.

2. Please run a permutation test: fix reward multisets per prompt, evaluate 50 random permutations under (a) sequential `Q > 0`, (b) sequential `Q = 0`, (c) batch KF `Q = 0`. Please report `Var_π[Acc]`, `Var_π[||g||]`, ICC/Kendall’s `W`, and Friedman tests.

3. Test controlled non-stationary settings (e.g., drift or mixture rewards within a prompt) and compare KF vs robust KF/UKF/particle filters. Do your gains persist?

4. What search ranges or criteria were used for `(Q, R)`? Which statistical tests produced your p-values? Please include effect sizes and multiple-comparison corrections.

5. Under which precise assumptions does your method reduce to GRPO (e.g., `Q → 0` and batch updates)? A short derivation would help.

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
2

### Summary
The paper introduces Kalman Filter Enhanced Group Relative Policy Optimization (KRPO), a simple modification to GRPO for reinforcement learning fine-tuning of large language models. Instead of using the group mean reward as a fixed baseline, KRPO applies a one-dimensional Kalman filter to estimate a smoothed reward baseline and its uncertainty. The method aims to stabilize advantage estimation without adding new parameters or changing the policy network. Experiments on several math reasoning datasets show consistent accuracy and reward improvements compared with GRPO and PPO.

### Strengths
1.The idea is straightforward and easy to integrate into existing GRPO training as a simple plug-in.
2. Experiments covers multiple math-related datasets, model sizes, and hyper parameter analyses, and  the results consistently show stable improvements in both accuracy and training reward.
3.The paper is clearly written and easy to follow, the overall idea is simple and easy to understand, with reasonable motivation and transparent reporting.

### Weaknesses
1. All experiments are restricted to math reasoning with discrete rewards, so generalization to open-ended RLHF tasks is unclear.
2. The method depends on the order of sampled rewards within a group, which contradicts the permutation-invariant nature of GRPO and introduces inconsistency. Because responses within a group are independent samples, the temporal assumption of the Kalman filter is not fully justified.
3.The paper  does not systematically study how different rollout sizes affect KRPO’s behavior. Since the Kalman filter operates within each group, its smoothing and variance‐reduction effect may depend strongly on the number of samples per prompt. Without such analysis, it is unclear whether the observed improvements hold under smaller or larger rollout settings.

### Questions
Please make explanations for the points listed in the weaknesses. 
How sensitive is KRPO to the order of rewards within each group?
Can the approach handle continuous or human-preference rewards where variance is not binary?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose Kalman Filter Enhanced Group Relative Policy Optimization (KRPO), a modification of GRPO that estimates advantage mean and variance using a Kalman filter. They compare KRPO to PPO and GRPO on 4 math datasets with 0.5-1.5B models and report improved performance.

### Strengths
The authors present a simple, easy to implement, and computationally efficient modification to the GRPO algorithm and analyze it empirically. The provided code is cleanly written and helped with understanding the methods and experiment details.

### Weaknesses
1. I am not convinced that the current set of experiments are sufficient to demonstrate that KRPO is superior to GRPO. Hyperparameters do not seem to be tuned for each algorithm and experiments are small scale (0.5-1.5B parameter models) and seem to be run with only one seed.

2. Confidence intervals would be more useful than p-values in the results tables. The paper does not describe what test is used to generate the p-values (so it is difficult to interpret) and the p-values do not provide an interpretable summary of the spread for each metric.

3. Experiments with popular variants of GRPO (e.g. Dr. GRPO, which removes the standard normalization and sequence length normalization) would help build confidence that the Kalman filter approach is generally useful. Additional baselines would also be helpful.

4. Presentation of the algorithm is a bit confusing: subscript i is used to describe the Kalman filter before defining what it is indexing.

### Questions
See weaknesses. My biggest concern is conducting sufficiently thorough experiments to support the conclusion that KRPO is generally superior to GRPO (or showing when it is superior and when it is not).

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
3

### Summary
This paper proposes Kalman Filter Enhanced Group Relative Policy Optimization (KRPO), to improve the stability and performance of the Group Relative Policy Optimization (GRPO) algorithm. Instead of subtracting the simple mean reward of all outputs in a group as the baseline to obtain advantages, KRPO uses a one-dimensional Kalman filter, which dynamically estimates a latent reward baseline and its associated uncertainty. The advantage is calculate in Eq. (6).

### Strengths
Using Kalman filter to estimate the mean and variance in GRPO seems a novel and reasonable idea. The resulting KRPO algorithm seems cost-effective. The experimental results are promising.

### Weaknesses
The base models used are relatively small (Llama-3.2-1B-Instruct, Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct). Experiments on ~7B scales would justify the effectiveness of the method more convincingly.

### Questions
Although the authors noted, but what wouldn't the reward be viewed as controlled by a slowly-changing latent variable across the training process, such that you don't reset the filter for each prompt? If that is not a good idea, do you see any evidences or have any explanations (if that doesn't perform well)?

### Soundness
3

### Presentation
3

### Contribution
3
