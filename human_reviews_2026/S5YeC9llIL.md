# Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Reinforcement Learning (RL) has proven highly effective for autoregressive language models, but adapting these methods to diffusion large language models (dLLMs) presents fundamental challenges. The core difficulty lies in likelihood approximation: while autoregressive models naturally provide token-level conditional probabilities essential for token-level RL objectives (e.g., GRPO), dLLMs generate sequences through iterative non-autoregressive denoising steps that lack this factorization. To address this fundamental mismatch, we propose ELBO-based Sequence-level Policy Optimization (ESPO), a principled RL framework that treats entire sequence generation as a single action and uses the ELBO as a tractable sequence-level likelihood proxy. Our method incorporates per-token normalization of importance ratios and robust KL-divergence estimation to ensure stable large-scale training.  Extensive experiments on mathematical reasoning, coding, and planning tasks demonstrate that ESPO significantly outperforms token-level baselines, achieving dramatic improvements of 20-40 points on the Countdown task, while maintaining consistent gains on math and coding benchmarks. Our approach establishes sequence-level optimization as a principled and empirically effective paradigm for RL in dLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes ELBO-based Sequence-level Policy Optimization (ESPO), a principled framework tailored for dLLMs. ESPO defines the entire sequence generation as an atomic action (avoiding token-level decomposition), employs the evidence lower bound (ELBO) as a tractable proxy for sequence-level likelihood, and utilizes a robust $k_2$ estimator for KL divergence (free from exponential instability).

Extensive experiments on models like LLaDA-8B-Instruct and Dream-7B-Instruct across tasks (mathematical reasoning, coding, planning) demonstrate ESPO’s superiority: it achieves up to 60-point absolute improvements in planning tasks (e.g., Sudoku), stable gains in knowledge-intensive tasks (e.g., GSM8K, HumanEval), and superior training efficiency (only 47% cost increase when MC samples double).

### Strengths
1. Clearly identifies the mismatch issue under the dLLM framework, highlighting that existing RL algorithms struggle to compute token-level importance sampling, and proposes a sequence-level alternative.

2. The experiments are thorough, including extensive validation and ablation studies across both math reasoning and agent tasks, demonstrating the effectiveness of the proposed method.

3. The paper is well-organized, with clear presentation and coherent argumentation.

### Weaknesses
1. For the math reasoning experiments, could the authors provide training curves? The improvement over the baseline appears limited.

2. Why are there no experiments on Dream-7B-Instruct with the d1 baseline?

3. Why are the results on coding benchmark such as HumanEval and MBPP missing baseline comparisons?

4. The performance on Sudoku shows a sudden change. Could the authors provide a deeper explanation or analysis?

5. From Figure 2, the observed improvements seem mainly driven by $k_2$ estimator. Could the authors include wd1 and d1 with $k_2$ estimator on Sudoku as additional comparisons?

I am willing to raise my score should the authors provide satisfactory responses and clarifications.

### Questions
Please see questions in Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a reinforcement learning framework for training diffusion language models. The approach builds upon group sequence-level importance sampling ratios for diffusion language models and incorporates an additional KL regularization term (referred to as K2-type regularization by the authors). The method is evaluated empirically on several tasks: Countdown, mathematical reasoning, and coding problems, demonstrating performance improvements over baseline approaches.

### Strengths
- The paper is well-written with a logical structure that makes the technical content accessible. The progression from problem formulation to methodology to experimental validation is easy to follow.

- The experimental evaluation demonstrates notable improvements on both the Countdown and math coding tasks, suggesting the proposed approach is effective for the target applications.

### Weaknesses
-  The proposed method largely combines existing techniques from prior work (Zheng et al., 2025; Tang & Munos, 2025b) without introducing new algorithmic components or theoretical insights. The contribution appears primarily incremental, adapting established methods to the diffusion language model setting rather than developing new approaches tailored to the unique characteristics of these models.
- While the empirical improvements are encouraging, the paper does not address the fundamental question of what constitutes an appropriate RL formulation for diffusion language models. The current approach treats the problem as one of better approximating log-probabilities, but a more principled direction would be to re-derive the policy gradient theorem specifically for diffusion language models from first principles. Critical questions remain unanswered: What is the proper form of policy gradients for diffusion language models? Does the "log-probability" term even appear in such gradients, or should the formulation take a fundamentally different form? Without addressing these foundational issues, the work risks building on potentially unsuitable assumptions.

### Questions
**Justification for sequence-level formulation:** The paper does not provide clear insight into why the proposed sequence-level formulation outperforms the token-level formulation. Is the advantage due to reduced bias, reduced variance in the policy gradient estimates, or both? A quantitative analysis comparing the bias-variance tradeoffs of both formulations would strengthen the paper's claims.

**Hyperparameter selection:** How was the clipping parameter $\epsilon$ in Proximal Policy Optimization chosen? Were separate hyperparameter searches conducted for the token-level and sequence-level formulations, or was the same value used for both? This is important for ensuring fair comparison between the two approaches.

**Role of KL regularization:** The results in Figure 2 raise several questions about KL regularization:

The figure suggests that KL regularization enables a significant performance breakthrough in later training stages, which is unexpected. Typically, KL regularization is understood to improve training stability rather than drive performance gains. Can the authors explain this phenomenon?
- In standard language model fine-tuning on reasoning tasks, KL regularization is often minimal or omitted entirely. Why is it critical in this setting?
- For clarity: which type of KL regularization (forward, reverse, or the K2-type mentioned) is used in Figure 1?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes ESPO, an RL framework tailored to diffusion LLMs which treats the whole completion as a single action. Experiments on LLaDA-8B-Instruct and Dream-7B-Instruct across math (GSM8K, MATH), coding (HumanEval/MBPP and EvalPlus), and planning (Countdown, Sudoku) show consistent gains versus token-level d1/wd1 baselines.

### Strengths
- Using a sequence-level action space makes the method very simple
- Experiments show strong improvements over the baselines which do not treat the entire sequence as an action

### Weaknesses
- Seems like a straightforward application of GSPO to diffusion models novelty-wise
- Not clear why per-token evaluation is necessarily bad for diffusion LLMs specifically - is it true for all LLMs (as GSPO claims) or just diffusion LLMs?

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies the problem of reinforcement learning (RL) to diffusion LLMs. Token-level objectives for autoregressive model (e.g., GRPO) require token log-likelihoods that are intractable to compute for diffusion models. The paper proposes ESPO, a sequence-level policy optimization method that treats generating the entire completion as one action and replaces the intractable sequence log-likelihood with an ELBO proxy. ESPO stabilizes training by (i) normalizing the ELBO log-ratio by sequence length and (ii) using a k2 (quadratic) KL estimator instead of the unstable exponential k3 estimator, alongside variance-reduction tricks (antithetic/coupled masking). ESPO is empirically evaluated with LLaDA-8B-Instruct and Dream-7B-Instruct on math (GSM8K, MATH), coding (HumanEval/MBPP), and planning (Countdown, Sudoku). ESPO consistently beats diffu-GRPO and wd1. Ablations show sequence-level+ELBO is the only stable formulation among tested variants and a training-cost analysis shows FLOPs/time grow mildly with Monte-Carlo samples because generation dominates compute.

### Strengths
* The paper explains the problem setup and why token-level importance ratios lack a valid probabilistic interpretation for dLLMs fairly well. The shortcomings of existing methods also makes the motivations quite clear.

* I like the idea of moving to a sequence level objective and using an ELBO-based ratio avoiding heuristic token surrogates.

* The performance of the method is tested across a variety of tasks and two different base models and shows consistent improvement over the baselines.

* The paper is also generally quite well written.

### Weaknesses
* ESPO optimizes an ELBO difference, not the true sequence likelihood ratio but the paper does not quantify how ELBO tightness affects policy improvement or bias across tasks.
* The paper misses some closely related prior work on RL fine-tuning of diffusion language models [1, 2]. I believe a comparison to these baselines would be critical.

[1] Venkatraman et al., 2024. Amortizing intractable inference in diffusion models for vision, language, and control.

[2] Zekri and Boullé, 2025. Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods.

### Questions
* Length normalization is shown to have unintended consequences on the reasoning performance for AR LLMs. I am curious if it impacts the long reasoning behaviors in your experiments?

### Soundness
3

### Presentation
3

### Contribution
3
