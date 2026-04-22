# Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4

## Abstract
Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning of large pre-trained models. Yet LoRA can face generalization challenges.
One promising way to improve the generalization is Sharpness-Aware Minimization (SAM), which has proven effective for small-scale training scenarios. In this paper, we propose **Bi**-directional **Lo**w-**R**ank **A**daptation (Bi-LoRA), which introduces an auxiliary adversarial LoRA module. This design explicitly decouples sharpness optimization, handled by the auxiliary module, from task adaptation, performed by the primary module. Such a separation yields two key benefits. First, it transforms the sequential computation of primary LoRA update and adversarial perturbation into a parallel form, which roughly halves the time and conquers the main obstacle of applying SAM in LoRA. Second, it provides perturbations from the auxiliary module that do not collapse into the restricted optimization subspace of the primary module, enabling broader sharpness exploration and flatter minima. Bi-LoRA simultaneously achieves both efficiency and effectiveness within a single framework, as verified by extensive experiments across diverse architectures and tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the generalization issue of LoRA, this paper proposes a method called Bi-LoRA. Specifically, besides the standard LoRA module trained via gradient descent, the method introduces an auxiliary LoRA module trained via gradient ascent. By training these two modules in parallel, the approach achieves an effect analogous to sharpness-aware minimization (SAM). Extensive experiments demonstrate that the proposed method attains superior or comparable performance across various settings.

### Strengths
1. The proposed method is simple and easy to implement, while effectively reducing computational overhead compared to traditional SAM.

2. The experiments are comprehensive, covering both image generation and language tasks.

3. The paper is clearly written and easy to follow.

### Weaknesses
1. The motivation of the paper is confusing. It is unclear why the computation of perturbations needs to consider a broader space. Moreover, the proposed method does not seem to alleviate this issue, as the perturbation space it explores is still constrained.
2. Although the proposed method is theoretically shown in Proposition 3 to be equivalent to SAM under the corresponding assumptions, this equivalence does not hold in practice due to the conflict between its assumptions and the observed experimental behavior.
3. Similarly, the argument presented in Proposition 2 is not convincing, as the update direction of the auxiliary module at iteration t is applied to the optimization of iteration t+1, making its alignment with the SAM perturbation direction at iteration t conceptually questionable.
4. The improvement brought by the proposed method is quite limited; however, this is understandable.

### Questions
1. In Figure 3, the convergence curve of the auxiliary module appears to converge rapidly after the primary module has already converged. Providing a reasonable explanation for this phenomenon would help deepen the understanding of the role and effect of the auxiliary module.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes Bi-LoRA, introducing a *bi-directional* (dual-module) structure for integrating Sharpness-Aware Minimization (SAM) into Low-Rank Adaptation (LoRA). The method adds an auxiliary LoRA module that performs gradient ascent to model adversarial perturbations, while the primary LoRA module performs gradient descent for task adaptation.

This design allows both updates to be computed simultaneously in one forward/backward pass, avoiding the doubled training cost of standard SAM, while also decoupling sharpness optimization from task adaptation.

Empirical results on a broad range of tasks (LLMs like LLaMA-2/3.1, Qwen-14B, and SDXL diffusion models) show that Bi-LoRA improves generalization performance over LoRA and LoRA-SAM with only minor additional memory and time cost.

### Strengths
1. **Clear Motivation.** The paper clearly identifies the inefficiency and subspace limitation of LoRA-SAM and logically motivates the need for decoupling SAM’s sharpness optimization from LoRA’s adaptation.
2. **Elegant Design.** The proposed bi-directional scheme (gradient descent for main LoRA, ascent for auxiliary LoRA) is simple yet conceptually neat. It transforms SAM’s two sequential steps into a single parallel step without additional training time.
3. **Theoretical Analysis.** The paper provides analytical support (Propositions 1–3), clarifying LoRA-SAM’s subspace restriction and proving that Bi-LoRA’s ascent direction aligns with SAM’s perturbation direction.
4. **Strong Empirical Validation.** Comprehensive experiments across architectures (transformers, diffusion models) and tasks (math reasoning, code, chat, instruction following, and image generation) show consistent improvement. The results are credible and diverse.
5. **Efficiency Retained.** The method achieves near-LoRA efficiency (≈1.09× training time, +0.7 GB memory) while roughly halving SAM’s cost, making it practical for large-scale LLM fine-tuning.
6. **Compatibility with Existing Methods.** The paper demonstrates that Bi-LoRA can be seamlessly integrated with LoRA variants such as LoRA-GA, PiSSA, and DoRA, yielding further gains.

### Weaknesses
1. **Overclaim**.  Bi-LoRA has a higher loss in Figure 4(a) than the LoRA-SAM, thus, it is a overclaim that Bi-LoRA achieves a significantly flatter loss landscape in LoRA parameter space in Line 252-253. 
2. **Ablation Clarity**. The effect of each component (e.g., gradient clipping in Eqn. (10)) is not fully isolated. It’s unclear how much each contributes to final performance.

### Questions
1. How does the choice of the neighborhood radius (ρ) and the auxiliary LoRA rank (r_2)? Is there a scaling rule or heuristic for choosing these jointly?
2. Why you apply the normalization only when the total Frobenius norm over $N$ auxiliary LoRA modules instead of do normalization for each auxiliary LoRA modules?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Bi-LoRA, a dual-adapter fine-tuning scheme in which a primary LoRA module is optimized by gradient descent for task adaptation while an auxiliary LoRA module is optimized by gradient ascent to approximate SAM-style perturbations in a single forward/backward pass. The decoupled design aims to mitigate the perturbation-subspace restriction observed when applying SAM directly to LoRA (LoRA-SAM) and to promote flatter minima in the full parameter space with minimal extra memory and near-LoRA time cost. The authors provide analytical support (e.g., alignment of the auxiliary ascent with the full gradient and a connection to a low-rank gradient-norm/Ky Fan regularizer under ideal inner maximization) and report consistent empirical improvements over LoRA and efficient SAM variants across LLaMA-2/3.1, Qwen-2.5-14B, T5-base (GLUE/SuperGLUE), and SDXL DreamBooth. At inference, the auxiliary adapter is discarded and only the primary LoRA is merged into the base model.

### Strengths
1) The method is simple and practical: a bi-directional LoRA scheme that achieves SAM-like sharpness-aware training in a single forward/backward pass with near-LoRA compute and modest memory overhead.
2) The paper offers clear analytical insights, including a derivation of LoRA-SAM’s perturbation subspace limitation and supporting theory for the auxiliary ascent direction (e.g., gradient alignment and a Ky Fan–style regularization view).
3) Empirical results are broad and consistent across LLMs and diffusion models, showing gains over LoRA and efficient SAM baselines, compatibility with other LoRA variants, and robustness under quantization.

### Weaknesses
1) The causal source of gains is under-isolated: without a “dual-LoRA (both descent)” control, it remains uncertain how much improvement comes from adversarial ascent versus added optimization degrees of freedom during training.
2) Theoretical grounding for the practical single-step inner ascent is limited; results rely on an idealized inner maximization, leaving a gap between the theory and the implemented algorithm.
3) Ablations and baselines are incomplete: the choice of global vs. per-layer clipping, sensitivity to ρ/rank/step sizes, and comparisons to broader single-pass SAM approximations / LoRA variants are not fully explored.

### Questions
1) Can you report a dual-LoRA (both descent) control with the same rank and clipping as Bi-LoRA, to isolate the effect of adversarial ascent?
2) How sensitive are results to ρ, (r1, r2), and the ascent/descent step sizes, and is there a regime where multiple inner ascent steps improve performance at acceptable compute cost?
3) Did you evaluate per-layer clipping (vs. global) and can you provide compute-parity details (hardware, precision, gradient checkpointing) to ensure fair efficiency comparisons across baselines?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new method for LoRA training. In particular, the paper focuses on incorporating Sharpness-Aware Minimization (SAM) into LoRA training to improve the generalization performance of LoRA. The paper notes that direct naive application of SAM to LoRA limits the performance and incurs overhead. As such, the paper proposes to introduce independent perturbation via an auxiliary LoRA module, decoupling the sharpness optimization and task adaptation. As a result, the proposed method, Bi-LoRA, significantly reduces the training cost and prevents the perturbations from collapsing into the restricted optimization subspace. The experiemntal results demonstrate the strong performance of Bi-LoRA.

### Strengths
- The proposed idea of decoupling the perturbation from the optimization has clear motivations.
- The proposed idea is simple and effective.
- The proposed method does not bring significant overhead and yet brings performance improvement.

### Weaknesses
- The paper does not compare against other LoRA variants, such as PiSSA and DoRA on the entire benchmark.
- The paper notes that Bi-LoRA shares similar perspective with WSAM (Yue et al., 2023) that views SAM as regularization. However, the paper does not experimentally compare against WSAM.
- The paper does not provide theoretical justification as to how Bi-LoRA achieves better generalization compared to LoRA and LoRA-SAM.
- Bi-LoRA introduces additional hyperparameter that seems to need to be tuned for each dataset.

### Questions
- Could authors provide an intuitive explanation as to how the Bi-LoRA is able to give appropriate aligned perturbations, despited the fact that optimization and perturbation are decoupled?

### Soundness
3

### Presentation
3

### Contribution
3
