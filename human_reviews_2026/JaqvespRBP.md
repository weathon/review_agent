# Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Diffusion language models (DLMs) enable parallel, order-agnostic generation with iterative refinement, offering a flexible alternative to autoregressive large language models (LLMs). However, adapting reinforcement learning (RL) fine-tuning to DLMs remains an open challenge because of the intractable likelihood. Pioneering work such as diffu-GRPO estimated token-level likelihoods via one-step unmasking. While computationally efficient, this approach is severely biased. A more principled foundation lies in sequence-level likelihoods, where the evidence lower bound (ELBO) serves as a surrogate. Yet, despite this clean mathematical connection, ELBO-based methods have seen limited adoption due to the prohibitive cost of likelihood evaluation. In this work, we revisit ELBO estimation and disentangle its sources of variance. This decomposition motivates reducing variance through fast, deterministic integral approximations along a few pivotal dimensions. Building on this insight, we introduce **Group Diffusion Policy Optimization (GDPO)**, a new RL algorithm tailored for DLMs. GDPO leverages simple yet effective *Semi-deterministic Monte Carlo* schemes to mitigate the variance explosion of ELBO estimators under vanilla double Monte Carlo sampling, yielding a provably lower-variance estimator under tight evaluation budgets. Empirically, GDPO achieves consistent gains over pretrained checkpoints and outperforms diffu-GRPO, one of the state-of-the-art baselines, on the majority of math, reasoning, and coding benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper suggests that the direct multiplying token-wise probabilities (used in diffu-GRPO) is not a good estimation of the probability of the whole sequence. Considering the expensive cost strictly following the ELBO, the authors analyze the main cause of the estimation variance, and find that the randomness of sampling $t$ is the main cause. Thus, GDPO uses deterministic time to consider the estimation as a time integral to reduce variance. Experiments indicate that it outperforms the original LLaDA and diffu-GRPO on multiple benchmarks.

### Strengths
- This paper provides a detailed analysis on the origin of the variance in diffu-GRPO, taking it as the motivation for deterministic $t$ decisions. This analysis is valuable for DLM RL algorithm designs.
- The comparison between GDPO and diffu-GRPO is comprehensive, including multiple domains (math, planning, coding).
- The paper is well written and easy to follow.

### Weaknesses
- I think the most severe problem is that the authors perform too few comparisons, only with LLaDA and diffu-GRPO. Actually, there have been much more related works on DLM RL, such as [1,2]. It is necessary to compare them directly.
- I do not consider diffu-GRPO as a token-level RL method. Actually, it also follows the ELBO in L 195 to provide an approximation for the probability to generate the certain sequence, but with an extremely small number of sample times (N=1). Increasing this could also make the algorithm more stable. We agree that use deterministic $t$ may further reduce the variance. However, other than comparing with diffu-GRPO, the paper does not provide a comparison between this two methods under the same number of MC sample times.
- There also lacks ablation studies to validate the effectiveness of the quadrature rule.

### Questions
I think the main weakness is the lack of experiments, see weaknesses for details.

### Soundness
1

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
The paper studies the problem reinforcement learning with diffusion language models. A key challenge in RL with DLMs is that likelihoods are hard to estimate. Prior methods either used biased token-level proxies or high-variance sequence-level estimates. As an alternative the authors propose Group Diffusion Policy Optimization (GDPO), which keeps a GRPO-style group advantage but replaces token-level terms with the sequence-level ELBO estimated by a semi-deterministic Monte-Carlo, This consists of fixing a few diffusion timesteps via Gaussian quadrature (2–3 points) and using a single inner MC sample, yielding a provably lower-variance estimator than standard double-MC under tight evaluation budgets. The paper analyzes ELBO variance, showing timestep sampling dominates and  provides asymptotic error bounds/bias discussion for SDMC. Empirically, using LLaDA-8B as the base model, GDPO improves over the base models and over diffu-GRPO across math (GSM8K, MATH500), planning (Countdown, Sudoku), and coding (HumanEval, MBPP), including better length generalization to 512 tokens.

### Strengths
* The paper overall is quite well written. The problem is defined quite clearly and approach explained clearly. 
* The variance analysis in Section 3.2 is quite useful and clear motivation for the problem.
* The variance reduction approach is explained clearly and coupled with useful theoretical results establishing lower variance. The authors also validate this empirically. 
* The resulting algorithm is fairly efficient and simple of implement.
* The empirical results show consistent improvements over some baselines across a variety of different tasks (math, coding, planning).

### Weaknesses
* The paper misses some closely related prior work on RL fine-tuning of diffusion language models [1, 2, 3]. In particular, I would like to see a comparison on how the SDMC compares with the variance reduction scheme from [2]. I believe a comparison to these baselines would be critical. 
* I also think the empirical results would benefit from having some simple test-time baselines like majority voting. 
* The experiments are limited to a single base model. I think it would benefit the paper to have some experiments with another base model (e.g. https://github.com/DreamLM/Dream)


[1] Venkatraman et al., 2024. Amortizing intractable inference in diffusion models for vision, language, and control.

[2] Zhu et al. 2025. LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models.

[3] Zekri and Boullé, 2025. Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods.

### Questions
* Are the results in Table 2 and 3 single seed for training and inference? If so I think it would be better to report mean@k. 
* The authors mention that their approach doesn't require too many compute resources, but there is not much detail about how much time training takes on the said resources and how it compares to say diffu-GRPO. Could you please clarify that?

### Soundness
3

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
4

### Summary
This paper introduces Group Diffusion Policy Optimization (GDPO), a novel reinforcement learning algorithm tailored to enhance the reasoning capabilities of Diffusion Language Models (DLMs). At its heart, the work tackles a central challenge in applying RL to DLMs: the difficulty of performing tractable likelihood estimation, which has long obstructed conventional policy optimization. The authors pinpoint the high variance and computational burden of existing sequence-level ELBO estimators as the key bottleneck. To overcome this, GDPO pioneers a new, low-variance ELBO estimator built on a Semi-deterministic Monte Carlo scheme—motivated by a fine-grained breakdown of the underlying variance sources. Empirically, GDPO consistently outperforms the strong baseline diffu-GRPO across a variety of reasoning, mathematics, and coding benchmarks, demonstrating its practical strength in fine-tuning DLMs.

### Strengths
1. The paper presents a novel idea and introduces a more principled approach for applying GRPO to diffusion large language models. This perspective is insightful and offers valuable inspiration for the research community.
2. GDPO demonstrates substantially better performance than the baseline diffu-GRPO on planning-intensive tasks such as Countdown and Sudoku, highlighting its advantage in structured reasoning and planning.
3. The analysis of variance sources in ELBO approximation is well-motivated, and the proposed semi-deterministic Monte Carlo method appears effective in significantly reducing variance.

### Weaknesses
It remains unclear why GDPO underperforms diffu-GRPO on certain mathematical reasoning benchmarks (e.g., GSM8K and MATH 500). The authors are encouraged to provide deeper insight into the potential reasons behind this performance gap.

### Questions
1. How was the KL term in GDPO estimated? A more detailed explanation in the paper would be helpful, as this aspect currently seems underexplored.
2. Have the authors experimented with techniques similar to those in GSPO, such as normalizing the reward by the length of the generated sequence?
3. Is the variance reduction technique introduced in this method orthogonal to the approach mentioned in LLaDA 1.5? If so, could they be combined to further enhance performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper deal with RLHF with diffusion language models (dLLM). RL methods for fine-tuning LLMs need likelihoods, however DLMs do not provide generation likelihoods. Thus current RL methods either (a) use cheap approximate likelihoods or (b) use the expensive to compute ELBO (which also is approximate but consistent). This work proposes Group Diffusion Policy Optimization (GDPO), which uses ELBO for the likelihood but instead of the. standard Monte-Carlo ELBO, uses low-order quadrature integration to produce a lower variance estimate. Their results show improved performance over both the LLADA and the more recent d1/diffu-GRPO model on a variety of tasks.

### Strengths
The problem of inefficient likelihoods for dLLMs is an interesting problem. The semi-deterministic Monte Carlo (SDMC) using quadrature integration is a neat, targeted variance-reduction idea and new to the best of my knowledge.
Experiments are not just limited to more standard tasks like Countdown and GSM used in dLLM literature, but also code related ones.
The paper also provides error analysis for the SDMC estimator, providing insight for determining when their method might give better results.

### Weaknesses
Theoretically, the claims rely on smoothness and convex/monotone properties of the integrand tied to KL forms and forward-process assumptions. It’s not fully clear these conditions generally hold.

There are other methods which have been proposed to address the likelihood problem for dLLM tuning. Some of these even focus specifically on variance reduction of the ELBO in dLLMs (for eg diffuCoder). These have not been compared against or even discussed.  

The improvements on GSM are minor (and are in fact worse on MATH). They are also worse than wd1, which is a missing baseline on multiple tasks. Similarly coding results are not compared against diffuCoder.

[1] wd1 Weighted Policy Optimization for Reasoning in Diffusion Language Models
[2]  PADRE: Pseudo-Likelihood based Alignment of Diffusion Language Models
[3] Test-Time Alignment of Discrete Diffusion Models with Sequential Monte Carlo 
[4] DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation

### Questions
Some questions have been mentioned in the previous section.Additional to those:

I would like to see some more ablations with varying N (quadrature), not just the accuracy.

 Since the experiments are only with small N (upto 4), what happens if we use a larger N. Since the estimate should get better with higher N, I am also confused why you have small optimal N, instead of as high an N as computationally feasible.

Since there is non-trivial amount of additional computations to compute their approximation to the ELBO, a comparison with baseline models under same computational cost (FLOPS, wall-clock or any other) is important.

Is there some easy metric or other diagnostics to determine if the used N is too small.

### Soundness
3

### Presentation
2

### Contribution
2
