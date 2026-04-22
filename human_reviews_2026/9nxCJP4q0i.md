# Consolidating Reinforcement Learning for Multimodal Discrete Diffusion Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 2, 6

## Abstract
Optimizing discrete diffusion model (DDM) with rewards remains a challenge—the non-autoregressive paradigm makes importance sampling intractable and rollout complex, puzzling reinforcement learning methods such as Group Relative Policy
Optimization (GRPO). In this study, we introduce **MaskGRPO**, the first viable approach to enable scalable multimodal reinforcement learning in discrete diffusion with effective importance sampling and modality-specific adaptations. To this end,
we first clarify the theoretical foundation for DDMs, which facilitates building an importance estimator that captures valuable token fluctuation for gradient updates. We then delicately tailored the rollout method for visual sequences, which yields diverse completions and reliable optimization gradients. Across math reasoning, coding, and visual generation benchmarks, MaskGRPO brings more stable and efficient updates, **doubling** reinforcement learning gains while speeding up training by up to **30%**. This study establishes MaskGRPO as a systematic policy optimization approach and the first practical way for discretized visual diffusion. The code is available at https://github.com/martian422/MaskGRPO.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose maskGRPO, a GRPO-style RL method for discrete diffusion models developed and shown to work on multimodal DDMs, both text and images. To achieve this, they propose a tractable importance estimator, and an approximated surrogate for importance weighting and KL in GRPO for DDMs. The authors claim more stable and efficient updates while showing improved performance on a set of both visual and language tasks.

### Strengths
The paper is generally well written and easy to follow. The results are compelling enough and it's nice to see GRPO working on image DDMs. The formulations of the importance estimator for DDMs is also useful.

### Weaknesses
A cornerstone of the narrative of the paper seems to revolve around compute/sample efficiency, but I did not see a detailed tokens-to-gain or wall-clock comparison against the strongest off-policy GRPO / surrogate-policy approaches, or a theoretical note explaining why the method is expected to work more efficiently in practice.

### Questions
Can you provide either empirical or theoretical grounding in support of markGRPO being more efficient than similar methods?

### Soundness
3

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
3

### Summary
The paper proposes MaskGRPO, an RL framework extending GRPO to multimodal DDMs. It introduces modality specific rollout strategies with AR-like reversing for text and probabilistic emerging for vision, along with a low-variance importance estimator for stable and scalable optimization across discrete diffusion. Experiments show consistent gains in reasoning, coding, and visual generation.

### Strengths
1. The problem is clearly defined — the paper directly targets the limitation of static embeddings in collaborative code completion by proposing a dynamic, incremental embedding approach.

2. The overall framework is well-structured and conceptually coherent, with a clear system design.

3. The method improves both language and vision generation, demonstrating strong empirical utility.

### Weaknesses
The paper lacks discussion of training cost or comparison of resource consumption with baseline methods.

### Questions
1. How is the mask ratio in the Rev(·, t) operator scheduled? Is it fixed or dynamically adjusted during RL training?

2. How does the training cost of MaskGRPO compare with other diffusion-RL methods such as Diffu-GRPO?

3. The paper introduces modality-specific rollout strategies, but it is unclear how sensitive the performance gains are to these design choices. Could the authors provide ablation or analysis showing how each component (AR-like reversing, emerging sampler) contributes to the overall gains?

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
The work proposes a method adapting GRPO for multimodal discrete diffusion models, by using modality specific sampling algorithms for rollouts (semi-autoregressive for text, and random probabilistic unmasking for images), as well as modality specific remasking strategies to use for updates in the GRPO objective (autoregressive-like remasking for text and standard random remasking for images). This is done along with limiting GRPO updates to trajectories in the time interval $(\gamma, 1)$. The modified GRPO method is used to fine-tune LLaDA for text reasoning tasks as well as MMaDA (a multimodal discrete diffusion model) for text-image alignment and aesthetic quality, where it is shown to outperform alternate RL methods.

### Strengths
1. A comprehensive set of baselines are evaluated for experiments.
2. The generated image samples in Figures 4, 7 are convincing in terms of improvement.
3. The ablations with truncation ratio $\gamma$ are helpful in establishing its impact (Figure 5).

### Weaknesses
1. The writing needs polishing and proofreading
    - There are numerous grammatical mistakes that interfere with the clarity of presentation
    - The citation format is incorrect (using `\citet` rather than `\citep` )
    - The Emerge sampler is described unclearly in text (lines 238-247)

2. The Emerge sampler (algorithm 4) doesn’t seem novel, and appears to be the same as the usual random unmasking from masked diffusion models (eg. without confidence based heuristics for unmasking). This is discussed in the original MDLM paper (Sahoo et al., 2024) as well as other early work on masked diffusion (eg. (Shi et al., 2024)). Despite this, the text appears to frame the sampler as a novel contribution (eg. line 244 ‘we refer to MDLM … and propose the … sampling strategy’ and line 247, ‘our sampler’). This should be clarified.

3. A number of claims are made which require more detail or justification:
    - The motivation for the autoregressive-like remasking invokes an observation that tokens with high entropy provide more informative signal for training, and that later tokens (in AR generation) tend to diverge more. This should be supported by some evidence from rollouts
    - Remasking with the $\mathrm{Rev}(\cdot, t)$ operator is asserted to be more stable and have low-variance (line 192, and also on line 189, the estimator is asserted to be “low-discrepancy”)  (I am assuming, compared to random remasking) - but this statement should be verified explicitly.
    - TraceRL is claimed to induce “biased estimation of sequence-level importance” - why is the method biased, compared to remasking with $\mathrm{Rev}(\cdot,t)$? This seems important since the introduction of a new remasking strategy is a core aspect of the method, and the straightforward thing appears to be reusing the partially masked completion obtained during rollouts.

4. It would be helpful to list the time taken to achieve the reward improvement (or some proxy), since the importance weight computation in this method appears to be more computationally intensive than the mean-field approximation for token level likelihood used in diffu-GRPO 

I recommend for a reject, mainly due to point 2 above, which I view as critically important.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and
generalized masked diffusion for discrete data. arXiv preprint arXiv:2406.04329, 2024.

### Questions
1. For text, if tokens later on in generation (more towards the end of generation) are more useful for training, in what sense does the AR-like process assign them “higher attention”. 
    - This appears to conflict with the truncation ratio being set at $\gamma > 0$  since it will exclude a group of tokens near the end of the (block-autoregressive) generation.
    
2. Is the masking rate multiplier ($\frac1t$ for linear) used in the likelihood computation at the token-level (for instance Equation 8)? The notation in Equation (2) implies it is.
    - If it is, is the masking rate adjusted for the alternate remasking strategies considered (namely autoregressive-like remasking)?

3. The mechanism behind why smaller truncation ratios lead to training failure (for image fine-tuning) is unclear. An explanation is given in terms of samples having stronger correlations between image patches, but its unclear to me why this results in collapse.

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
This paper tackles reinforcement learning (policy-based methods such as GRPO, PPO) for discrete diffusion models (dLLMs for text generation, bidirectional discrete diffusion models for visual generation). The authors propose to improve GRPO with effective importance sampling and modality-specific adaptations to the text and visual generation domains. For text generation, it proposes a semi-autoregressive reverse masking strategy to align better with the inherent causality in reasoning tasks. For visual generation, it adopts a probability-based sampler for better visual textures. The method was shown to improve LLaDA and MMaDA on math, coding, and compositional visual generation.

### Strengths
+ The AR-ness introduced back to the masking scheme to improve text-based reasoning tasks looks convincing.
+ Evaluation on several benchmarks shows promising results.

### Weaknesses
+ The overall novelty is a bit limited. The AR-ness that could help dLLMs with reasoning tasks has been noted in previous work. While there are improvements over MaskGIT sampler for visual decoding and sampling, the latter is a relatively old method. Visual generation is evaluated only on text-to-image generation, which is not the biggest advantage of discrete diffusion models. There could be an evaluation of image editing to make the results more significant (however, recent approaches like Transfusion [1] and BAGEL [2] do explore continuous diffusion heads instead of discrete diffusion models for the visual generation part).
+ Minor point: “3.2 ROLLOUT ADAPTION” was placed in the wrong place. Shouldn’t it be right before “Let visual tokens emerge from masks”?

[1] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

[2] Emerging Properties in Unified Multimodal Pretraining

### Questions
See "weaknesses"

### Soundness
3

### Presentation
3

### Contribution
2
