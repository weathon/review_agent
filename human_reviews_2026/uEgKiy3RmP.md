# Mitigating Reward Hacking in Inference-Time Alignment of T2I Diffusion Models via Distributional Regularization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Diffusion models excel at generating images conditioned on text prompts, but the resulting images often do not satisfy user-specific criteria measured by scalar rewards such as Aesthetic Scores. This alignment typically requires fine-tuning, which is computationally demanding. Recently, inference-time alignment via noise optimization has emerged as an efficient alternative, modifying initial input noise to steer the diffusion denoising process towards generating high-reward images. However, this approach suffers from reward hacking, where the model produces images that score highly, yet deviate significantly from the original prompt. We show that noise-space regularization is insufficient and that preventing reward hacking requires an explicit image-space constraint. To this end, we propose MIRA (MItigating Reward hAcking), a training-free, inference-time alignment method. MIRA introduces an image-space, score-based KL surrogate that regularizes the sampling trajectory with a frozen backbone, constraining the output distribution so reward can increase without off-distribution drift (reward hacking). We derive a tractable approximation to KL using diffusion scores. Across SDv1.5 and SDXL, rewards (Aesthetic, HPSv2, PickScore), and public datasets (e.g., Animal-Animal, HPDv2), MIRA achieves >60% win rate vs. strong baselines while preserving prompt adherence; mechanism plots show reward gains with near-zero drift, whereas DNO drifts as compute increases. We further introduce MIRA-DPO, mapping preference optimization to inference time with a frozen backbone, extending MIRA to non-differentiable rewards without fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a simple regularization technique for inference-time alignment through input noise optimization and further proposes a MIRA-DPO algorithm for black-box input noise optimization. The authors present both qualitative and quantitative analyses demonstrating consistent improvements over the previous DNO method.

### Strengths
1. The proposed method is straightforward, simple to implement, and conceptually clear.

2. The paper demonstrates both quantitative and qualitative improvements over previous state-of-the-art approaches.

### Weaknesses
1. The original DNO paper explicitly tackles reward hacking by constraining optimization to high-probability regions of the noise space and introducing concrete regularization terms. It is unclear whether the experiments in this paper were conducted with the same PRNO hyperparameters, as the DNO authors report strong results for their regularized variant, which already mitigates reward hacking through a relatively lightweight approach. While optimizing the KL divergence could, in principle, provide stronger regularization, the baseline method already requires several minutes to improve a single image, raising concerns about efficiency and practical gains.

2. The paper offers limited novelty. KL constraints are well-established, DPO is applied in a standard way, and the idea of transferring optimization from network parameters to input noise is a relatively direct conceptual extension rather than a fundamentally new contribution.

### Questions
1. Could you elaborate on the fairness of your comparison with PRNO (the regularized DNO method)? The original DNO paper reports qualitative examples of reward hacking but addresses them through a more lightweight regularization approach. Clarifying whether your experiments used comparable settings would help ensure a fair evaluation.

2. Can you report the reward dynamics with respect to optimization time? You mention that your method requires around five minutes and shows reduced reward hacking, while DNO reports up to ten minutes with plausible results. A direct comparison of reward progression over time would clarify the efficiency–performance trade-off.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a method for inference-time alignment of diffusion models with external rewards. Their key concept is to utilize initial noise optimization formulation to optimize either a differentiable reward directly, or optimize the initial noise with direct preference optimization for non-differentiable rewards. In terms of technical contributions, the primary contribution of the paper is to introduce a surrogate KL regularization on the score as opposed to the regularization done on the noise space in previous works. Results on a few different rewards indicate that the proposed method is able to achieve superior results compared to prior noise optimization formulations.

### Strengths
The paper demonstrates strong empirical results in terms of aligning text-to-image models with rewards directly at inference without any fine-tuning, even when compared to prior noise optimization techniques

The paper also does a good job of presenting the ideas quite clearly, especially motivating the theoretical ideas.

### Weaknesses
[Major]
Fundamentally, the main contribution of the paper is the regularization method (which applied the KL surrogate on the score as opposed to regularizing the norm of the initial noise as was done before). While this does seem sound for the most part, I am also a bit uncertain whether this on its own serves as sufficient contributions for acceptance. 

A slightly lesser concern is the hyperparameter configurations, since reward hacking can be mitigated/increased by either controlling the weight of the regularization term, or the learning rate for the updates, or the number of optimization steps (i.e early stopping). Here, I think Fig. 6 does a good job of showing the difference in optimization trajectories and the effect of regularization weight. However, with Fig. 5 the initial results (step 0) itself appears to be different which makes it tricky to make the comparison. In Fig. 7, it's somewhat unclear what were the configurations of the other methods used to generate the final sample used for comparison. While the original configurations of these methods might have been suited for their original tasks, it might perhaps be appropriate to have a hyperparameter sweep for all methods and especially have a held-out reward (perhaps even with just GPT-4o) to validate the results and even implement early stopping. 

[Minor]
The results in the paper are primarily with older, smaller (<3B) models. It would be valuable to see the effect of the proposed regularization on larger models (e.g. Flux among others).

### Questions
I have a minor question regarding Proposition 1. The final result states that in the case of noise regularization (with the norm), the KL divergence in the image space is the product of the L2 distance in the noise space and some term with the norm of the model weights. Given that we have frozen the base model which is also typically trained with weight decay (i.e. applying L2 regularization of the model weights), isn't this sufficiently bounded? If one was also fine-tuning the weights, it's evident that it could diverge arbitrarily, but with the noise optimization on a frozen model, it appears like regularizing the noise would still bound the KL divergence (even if it's not as tight as the proposed objective)?  

Overall, while I do like the paper, I would primarily like to see a more compelling justification for why this regularization alone makes for sufficient contribution before recommending acceptance.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **MIRA**, a method for inference-time alignment of diffusion models that mitigates reward hacking by directly regularizing the output image distribution. Its main contribution is a practical, score-based surrogate loss that prevents semantic drift while optimizing for rewards, applicable to both differentiable and black-box objectives via MIRA-DPO. Extensive experiments show that MIRA achieves superior win rates against state-of-the-art baselines across multiple models and datasets, demonstrating a better trade-off between reward maximization and prompt faithfulness.

### Strengths
**Quality & Clarity:** The work is thorough, with rigorous experiments across multiple models, rewards, and datasets. The mechanism analysis (distributional drift via KL surrogate and CMMD) provides quantifiable evidence for its claims. The paper is well-structured, with clear motivations, derivations, and accessible visuals.

**Significance:** This work is important for addressing the critical, unsolved problem of reward hacking in DNO. By providing a practical and effective inference-time solution, it enhances the **reliability and real-world usability** of text-to-image generation, representing a substantial advance in the quest for robustly aligned AI systems.

### Weaknesses
**Limited Baseline Comparison Scope:** The experimental evaluation focuses predominantly on noise optimization methods like DNO, while giving less attention to other competitive inference-time alignment approaches, like Golden Noise[1]. Besides, although DNO mainly archive their experiments on SD1.5 and SDXL, authors are encoughed to extend their mothod to better base model with DiT architecture, at least to SD3.5.

**Insufficient Theoretical Analysis of the Surrogate Objective:** While the score-based KL surrogate is pragmatically motivated, the paper lacks a rigorous theoretical characterization of its approximation quality. A formal analysis of the tightness of this upper bound, or at least an empirical ablation studying the gap between the true KL and its surrogate across different noise schedules, would strengthen the methodological foundation.

1. Zhou, Zikai, et al. "Golden noise for diffusion models: A learning framework." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

### Questions
You can see Weakness for specific questions. My main concern is the scope of this method.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces MIRA, an inference-time alignment method for diffusion models that aims to mitigate reward hacking. The core idea is to regularize the whole sampling trajectory (instead of regularizing the noise by itself) by penalizing deviations from the base model's output distribution, which is implemented via a tractable, score-based surrogate for KL divergence. The authors demonstrate that this regularization is more effective at preventing semantic drift than prior noise-space methods, i.e. preventing reward-hacking. They also introduce MIRA-DPO, an extension for handling non-differentiable rewards. The empirical results, particularly the qualitative experiments and win-rate comparisons, suggest the method is effective.

### Strengths
- The paper is well-motivated, tackling the important and challenging problem of reward hacking in inference-time optimization by proposing a new regularization.
- The experiments using simple rewards like brightness/darkness provide a very clear and illustrative visualization of reward hacking and MIRA's ability to mitigate it. The quantitative analysis of distributional drift in Figure 6 is a strong piece of supporting evidence.
- The introduction of MIRA-DPO to handle non-differentiable rewards is a valuable contribution, extending the applicability of the framework to more realistic scenarios.
- Convincing ablation studies are included, e.g. on the hyperparameter β (Appendix A.5) is well-executed and provides useful insights into the trade-off between reward optimization and prompt fidelity.

### Weaknesses
- **Insufficient acknowledgment of Adjoint Matching**: The paper does not discuss its relationship to Adjoint Matching [1], despite significant overlap. Adjoint Matching was proposed for reward fine-tuning with what appears to be an identical regularization approach:
	- **MIRA's regularizer (Eq. 4):** L2 loss on score difference between base and optimized models over the denoising trajectory: $\sum_t \sigma^2_t \|s_{\text{base}}(x_t) - s_{\text{opt}}(x_t)\|^2$
     - **Adjoint Matching's loss (Eq. 42):** L2 loss on velocity difference: $\sum_t \|v_{\text{finetune}}(X_t) - v_{\text{base}}(X_t)\|^2 2 /\sigma^2(t)$: (equivalent formulation for scores/velocities)
	- **Key questions:**
		- Is MIRA's KL regularization mathematically identical to Adjoint Matching's?
		- Could Adjoint Matching's efficient adjoint-based gradient computation (O(1) memory vs. O(T) backpropagation) be applied here?
	- **Impact on novelty:** The derivation in Appendix A.9 (KL → score-based surrogate) appears to reproduce Adjoint Matching's framework. If so, the contribution would more accurately be framed as applying this regularization to inference-time noise optimization rather than presenting the score-based KL surrogate as a novel derivation. The current framing in the abstract ("we derive a tractable approximation to KL") may overstate the theoretical novelty.
- **Clarification on Pointwise Insufficiency**: The paper motivates its regularizer by arguing that noise-space regularization is "fundamentally insufficient" (Proposition 1), using a proof that demonstrates a pointwise property where two close noise vectors can map to distant image distributions. This argument, however, may not fully address the core principle of existing noise-space methods, which are often viewed from a distributional perspective, aiming to keep the optimized noise `z` within the high-probability regions of the prior `N(0, I)`, rather than strictly bounding pointwise distances. The motivation could be strengthened by either showing that this distributional view also fails or by framing MIRA as a complementary approach. I would be happy to discuss this point further, as it seems central to the paper's framing.
- **Scope of theoretical framework regarding ODE sampling**: It is unclear how MIRA extends to deterministic ODE samplers (e.g., DDIM with η=0 or modern ODE solvers), which are widely used in practice. In the deterministic case, the output is a single point, making KL divergence ill-defined (0 or ∞). Consequently, MIRA's objective may no longer serve as a direct KL surrogate. **Questions:**
	- Does the objective still act as an effective regularizer in the deterministic case? If so, what is its theoretical justification outside the stochastic KL framework?
	- This is particularly relevant for the SDXL-Turbo experiments, i.e. how should we interpret MIRA's regularization for single-step deterministic sampling?
- **Evaluation Based on Non-Standard Benchmarks:** While GPT-4o win rates are interesting, they are not a standard, reproducible benchmark for evaluating text-to-image models. The evaluation could be substantially strengthened by including results on established benchmarks like GenEval, DPG-Bench, or T2I-CompBench. This would allow for a more direct and fair comparison with the broader literature. Additionally, it would be more convincing to include quantitative results for the non-differentiable rewards instead of relying on purely qualitative. (Section 5.3)
- **Missing Implementation Details:** The main experiments in Section 5.2 do not state which reward function was used to generate the images for the GPT-4o win-rate comparison in Table 1. How many inference steps are used for SDXL-Turbo? If only one step is used (as is common), how should MIRA's multi-step regularization term be interpreted?

Minor: The GPT-4o winrate table is hard to read; some highlighting would benefit readability.

[1] Domingo-Enrich et al. "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control". ICLR 2025.

### Questions
- Would using MIRA to reward fine-tune a model be equivalent to Adjoint Matching? (without the added adjoint state efficiency)
- See Weaknesses

### Soundness
1

### Presentation
2

### Contribution
3
