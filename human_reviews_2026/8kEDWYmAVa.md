# VISOR++ : VISUAL INPUT BASED STEERING FOR LARGE VISION LANGUAGE MODELS

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
As Vision Language Models (VLMs) are deployed across safety-critical applications, understanding and controlling their behavioral patterns has become increasingly important. Existing behavioral control methods face significant limitations: system prompting is a popular approach but could easily be overridden by user instructions, while applying activation-based steering vectors requires invasive runtime access to model internals, precluding deployment with API-based services and closed-source models. Finding steering methods that transfer across multiple VLMs is still an open area of research. To this end, we introduce universal visual input based steering for output redirection (VISOR++), a novel approach that achieves behavioral control through optimized visual inputs alone. We demonstrate that a single VISOR++ image can be generated for an ensemble of VLMs that by itself can emulate each of their steering vectors. By crafting universal visual inputs that induce target activation patterns, VISOR++ eliminates the need for runtime model access while remaining deployment-agnostic. This means that when an underlying model supports multimodal capability, model behaviors can be steered by inserting an image input completely replacing runtime steering vector based interventions. We first demonstrate the effectiveness of the VISOR++ images on open-access models such as LLaVA-1.5-7B and IDEFICS2-8B along three alignment directions: refusal, sycophancy and survival instinct. Both the model-specific steering images and the jointly optimized images achieve performance parity closely following that of steering vectors for both positive and negative steering tasks. We also show the promise of VISOR++ images in achieving directional behavioral shifts for unseen models that include both open-access and closed-access models. At the same time, VISOR++ images are able to preserve 99.9\% performance on 14,000 unrelated MMLU evaluation samples highlighting their specificity to inducing only behavioral shifts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
VISOR++ introduces a novel VLM behavioral control method that uses optimized, universal visual inputs to replace traditional activation-based steering vectors. This approach overcomes the limitations of existing techniques (e.g., system prompts being easily overridden, activation steering requiring white-box model access), making behavioral alignment feasible for closed-source and API-served VLMs.Leveraging a differentiable preprocessing pipeline and an adversarial optimization algorithm (CWA-SSA), VISOR++ crafts a single image that effectively steers VLMs like LLaVA and IDEFICS2 across behaviors such as refusal, sycophancy, and survival instinct, all while preserving 99.9% of performance on unrelated MMLU tasks. The research also demonstrates promising directional transferability of these steering images to unseen models (including GPT-4 variants), particularly for negative steering.This work establishes a new, deployment-agnostic paradigm for VLM safety and control, offering a solution with significant universal potential.

### Strengths
1. Clear and important motivation. The paper tackles the practical challenge of behavior control for VLMs in safety-critical settings. Prior approaches either rely on brittle system prompts that are easily overridden or on activation-space guidance that requires white-box access. By steering behavior purely through visual input, VISOR++ sidesteps white-box assumptions and is directly applicable to closed-source/API-only models.

2. Methodological novelty. The work shifts guidance from the activation space to the input space by generating adversarially optimized “guidance images.” The differentiable preprocessing pipeline is a key technical contribution, mitigating gradient discontinuities arising from heterogeneous model preprocessing and thereby enabling cross-model optimization.

3. Generalization and transfer. Universal VISOR++ images perform strongly on the integrated model and show preliminary transfer to unseen models (including GPT-4 variants), with particularly consistent directional effects under negative guidance.

### Weaknesses
1. Evidence for universal are limited.     While the title/abstract foreground “universality,” the experiments chiefly optimize over a small two-model ensemble (LLaVA-1.5-7B, IDEFICS2-8B).     Transfer to unseen models yields modest Δ effects, particularly weak for positive guidance, with limited impact on Qwen2-VL-7B and Claude Sonnet 3.5.     The universality claim would be more convincing with a larger, more heterogeneous optimization ensemble and stronger transfer results across a broader set of models.     Scaling curves (ensemble size vs. transfer), per-model effect sizes with confidence intervals, and task/prompt diversity would help substantiate the claim.

2. Optimization cost.     Universal image generation relies on double-layer momentum, spectral enhancement, and nontrivial hyperparameter schedules.     Although the method removes runtime access, producing these images still requires gradient access to the integrated model and appears computationally heavy.     For practitioners, the cost and complexity of generating/maintaining universal images may become the new bottleneck.     Reporting wall-clock/GPU-hour budgets, sensitivity to hyperparameters, and simpler ablations (or black-box approximations) would clarify practicality.

3. Case for perturbation results.    The paper does not show representative VISOR++ guidance images or analyze their human perceptibility.     It is unclear whether they are near-imperceptible perturbations or conspicuous textures.     Providing qualitative examples, basic perceptual metrics (e.g., LPIPS/SSIM/PSNR), frequency-domain analyses, and a small human study would illuminate usability, potential covert-attack risks, and deployment considerations.

### Questions
1.  Transferability：
Could the authors elaborate on why the transferability for negative steering appears consistently stronger than for positive steering, especially when transferred to unseen models?   Is this asymmetry indicative of fundamental differences in how specific behaviors (e.g., refusal vs. compliance) are represented or aligned within the models' internal states?

2.  Clarifying the mechanism:
CWA-SSA for behavioral steering vs. general adversarial attacks, while the paper leverages adversarial optimization techniques (specifically CWA-SSA), it is crucial to clearly differentiate how VISOR++'s approach uniquely serves 'behavioral steering' rather than merely resulting in simple 'jailbreaks' or 'misclassification' (as often seen in traditional adversarial attacks).   Specifically, given that steering vectors are integrated into the target activations for optimization, how does the underlying mechanism ensure that the generated image steers a broad behavioral tendency across diverse prompts, as opposed to simply maximizing the likelihood of a specific output string or misclassifying a single input?

3.  Visualization : Could the authors please include some typical VISOR++ images in the main paper or appendix?   Furthermore, a discussion on their perceptual properties is needed: are these images subtly perturbed and imperceptible to the human eye, similar to many adversarial examples?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes VISOR++, a method for steering the behavior of Vision-Language Models through optimized visual inputs instead of activation-based vectors. By leveraging adversarial optimization with spectral augmentation and dual-momentum updates, VISOR++ can generate universal images that induce specific behavioral shifts (e.g., refusal, sycophancy) across multiple models without accessing their internals. Experiments on LLaVA-1.5 and IDEFICS2 demonstrate comparable steering performance to activation-based methods while maintaining normal task accuracy.

### Strengths
1.The paper introduces a novel concept of visual input-based behavioral steering, shifting control from activation space to input space for VLMs.

2.VISOR++ offers a practical, deployment-agnostic approach for behavioral control, potentially useful for closed-source or API-based multimodal systems.

### Weaknesses
1.The paper introduces several algorithmic components (e.g., spectral augmentation, dual-momentum optimization) but provides no ablation to isolate their contributions.

2.The transferability study (Table 2) only compares VISOR++ with random images, making it hard to judge relative effectiveness.

3.Baseline comparison is limited to steering vectors and system prompts, omitting relevant visual or multimodal steering methods such as ASTRA or SteerVLM.

3.The abstract incorrectly describes MMLU as containing “14,000 tasks,” while it actually has 57 tasks and about 14 k samples.

4.The paper inconsistently mixes the singular and plural forms of “VLM”/“VLMs” and “LLM”/“LLMs,” which may confuse readers and reflects insufficient attention to writing consistency.

5.Several standard components and models (e.g.,LLaVA, Mistral, Qwen2-VL, LLaVA-NeXT, Llama, etc.) are mentioned without proper citations.

### Questions
1.Could you provide an ablation showing VISOR++ performance without the spectral-domain augmentation?

2.How much improvement does the dual-momentum optimization provide over single-momentum or PGD-only optimization?

3.What experimental evidence supports the claim that “PGD is very effective” (line 225)?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VISOR++, a visual input–based steering method for large vision–language models (VLMs). Instead of modifying internal activations, VISOR++ learns a single optimized image that reproduces the effects of steering vectors, steering model behavior along safety dimensions like refusal, sycophancy, and survival instinct. The approach shows strong parity with activation-based steering while being compatible with API-based, closed-source models.

### Strengths
(1) Clever idea — turning activation steering into an input-space optimization makes the method deployable even when model internals are inaccessible.

(2) The algorithm is well-engineered, using differentiable preprocessing and spectral augmentation for cross-model robustness.

(3) Experiments are thorough, including both open-source and API-based models, and show that VISOR++ doesn’t degrade unrelated task performance.

### Weaknesses
(1) While the concept is novel, the empirical gains over activation steering are modest; VISOR++ mostly matches, not surpasses, existing methods.

(2) The paper treats the steering image as universal but doesn’t explore why it generalizes. There’s no mechanistic or interpretive analysis linking image features to behavioral shifts.

(3) Transfer results to unseen models (e.g., GPT-4V, Claude Sonnet) are very small and mostly directional, so the claim of “universal” transferability feels overstated.

### Questions
(1) How stable are these steering effects over different prompts or seeds? Is the learned image really robust, or do small text changes break the effect?

(2) The paper focuses on three behaviors, could the same idea work for other safety directions (e.g., toxicity, helpfulness), or is it limited to the chosen datasets?

(3) Have the authors looked at what the optimized images actually look like? Any visible structure or pattern that hints at how they interact with the vision encoder?

(4) Since VISOR++ depends on adversarial optimization, how reproducible are the results across runs — do different random inits yield similar steering effects?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Existing behavioral control methods have limited transferability across different VLM architectures. To address this, the authors propose VISOR++, a method that enables behavioral steering using optimized visual inputs, without requiring access to model internals. They show that VISOR++ can reliably influence model behavior along three axes: refusal, sycophancy, and survival instinct. Also, they show that VISOR++ exhibits generalization to unseen models while having minimal impact on unrelated task performance.

### Strengths
- The motivation for this work is clear.
- The authors propose VISOR++, a technique that reframes behavioral steering from a latent-space intervention into an input-space control problem.
- VISOR++ avoids the need for white-box model access, making it applicable even to closed or API-served VLMs.

### Weaknesses
- VISOR++ is evaluated using only two models for image optimization, which makes it difficult to assess how broadly the method generalizes. Moreover, the behavioral shifts reported in Table 2 are often quite small, raising questions about the practical significance of the approach. 
- The paper does not provide a clear explanation for why VISOR++ works. Could you share your intuition or analysis on the method?

### Questions
- Figure 1 is somewhat confusing: both example responses appear identical, making it unclear what behavioral difference the figure is intended to highlight. Clarifying this would improve readability.
- It is also unclear whether VISOR++ operates orthogonally to common image augmentations. For instance, does cropping, resizing, or compression diminish the steering effect? 
- Could you add an experiment where steering vectors from one VLM are transferred directly to another VLM as a baseline comparison to see their generalization?

### Soundness
3

### Presentation
3

### Contribution
3
