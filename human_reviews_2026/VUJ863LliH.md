# Multimodal Information is All You Need for Adversarial Purification via Diffusion Models

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Adversarial defense aims to find true semantic labels of adversarial examples, where diffusion-based adversarial purification as intriguing adversarial defense methods can restore data perturbed by unseen attacks to clean distribution without training classifiers. However, unimodal diffusion-based approaches rely on noise schedules to implicitly preserve labels, whereas recently proposed multimodal variants add textual control but require adversarial training and heavy distillation. Both approaches lack theoretical guarantees.
In this work, we propose MultiDAP that uses multimodal diffusion models for adversarial purification. MultiDAP first learn prompts from clean text-image pair data for clean image generation, where context tokens are numerical instead of text templates such as ``a photo of $\cdot$'' for rich contextual information and hence enhance adversarial robustness. Given learned prompts and adversarial examples, MultiDAP then purify inputs via minimizing regularized DDPM losses iteratively for only a few steps. Theoretical guarantees for two phases are also provided. In experiments, our proposed model achieve improvement of zero-shot adversarial defense performance over unimodal diffusion models and multimodal variants with text templates.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces MultiDAP, a adversarial purification method that leverages multimodal diffusion models to defend against adversarial attacks. Unlike prior unimodal approaches that rely solely on noise schedules or multimodal methods requiring adversarial training and heavy distillation, MultiDAP learns class-agnostic, continuous prompt embeddings from clean image-text pairs. These learned prompts serve as robust semantic anchors to guide the diffusion process. During purification, the method minimizes a regularized DDPM-based loss through a few-step gradient optimization in pixel space, avoiding multi-step reverse diffusion. Theoretically, the authors provide guarantees for both the prompt learning phase and the purification phase.

### Strengths
1. Compared to previous methods that relied solely on unimodal image information for adversarial purification, this paper enhances adversarial purification by incorporating textual information.
2. This paper provides rigorous theoretical analysis, including the training-phase theorem "Prompt learning improves the likelihood lower bound" and the testing-phase theorem "Expected descent of the purification loss."

### Weaknesses
1. During training, the method requires a classifier to generate adversarial examples. The method relies heavily on the specific classifier and the adversarial attack algorithm used, which can further undermine the method's generalization performance.
2. The experimental section is relatively weak, with very limited and outdated baseline comparisons, making it difficult to validate the effectiveness of the method.
3. The method introduces a considerable number of hyperparameters, which makes practical optimization highly challenging. Furthermore, the paper lacks a sensitivity analysis of these hyperparameters.

### Questions
Current unimodal methods typically use unconditional checkpoints for diffusion model weights. Could you elaborate on how you incorporate textual information into an unconditional diffusion model? If you switch models and their corresponding weights, please explain how to ensure a fair comparison.

### Soundness
3

### Presentation
2

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
This paper proposes MultiDAP, a multimodal diffusion-based adversarial purification method that uses learnable prompt embeddings instead of fixed text templates. The approach consists of two phases: (1) learning class-agnostic numerical prompt tokens from clean text-image pairs using DDPM loss, and (2) purifying adversarial examples by minimizing regularized DDPM loss with the learned prompts for only 5 iterations. The authors provide theoretical guarantees showing that prompt learning improves the likelihood lower bound (Theorem 1) and that few-step purification achieves expected descent (Theorem 2). Experiments on CIFAR-10 demonstrate that MultiDAP achieves 72.38% robust accuracy under AutoAttack ($\ell_\infty$, $\epsilon=8/255$), slightly outperforming the Likelihood Maximization baseline (71.68%) while claiming to be more efficient than DiffPure.

### Strengths
The paper addresses a relevant problem in adversarial defense. The key contributions include:

The motivation for using learnable numerical tokens instead of hand-crafted text templates (e.g., "a photo of a") is reasonable and well-articulated. The distinction between class-agnostic prompts for purification versus class-specific prompts for classification is appropriate given that ground-truth labels are unknown for adversarial inputs.

The theoretical analysis provides formal justification for the approach. Theorem 1 connects prompt learning to maximizing the evidence lower bound, and Theorem 2 establishes convergence guarantees for the purification procedure with bounded variance assumptions.

The ablation studies in Table 2 systematically investigate the effect of purification steps and timestep ranges, providing useful insights about the trade-offs between robustness and clean accuracy.

Algorithm 1 and Algorithm 2 are clearly presented and appear to be reproducible given the implementation details provided.

### Weaknesses
The experimental evaluation is severely limited to CIFAR-10 (32x32 images upsampled to 256x256). For a paper with the ambitious title claiming "multimodal information is all you need," validation on larger-scale datasets like ImageNet is essential to demonstrate generalizability. The upsampling from 32x32 to 256x256 also raises questions about whether the method truly works at the intended resolution or benefits from artifacts introduced by upsampling.

The paper cites Lei et al. (2025) on one-step control purification (OSCP) throughout the introduction and related work but completely omits experimental comparisons. This is a critical omission for several reasons. First, in my knowledge, OSCP represents the current SOTA in multimodal diffusion-based purification. Second, OSCP can achieve purification in a single step, whereas MultiDAP requires 5 steps. This directly contradicts the paper's claimed efficiency advantage. The authors state `MultiDAP attains slightly higher robust accuracy but requires only 5 purification steps whereas DiffPure typically involves dozens of Langevin iterations` but fail to mention that OSCP requires only 1 step. The comparison with DiffPure (ICML 2022) feels like cherry-picking an outdated baseline to claim efficiency gains.

The performance improvement over existing baselines is marginal and potentially not significant. MultiDAP achieves 72.38% vs LM's 71.68% on AutoAttack ($\ell_\infty$), a difference of only 0.7%. No confidence intervals or statistical significance tests are provided. Moreover, examining Table 1 reveals that AT-EDM-$\ell_\infty$ achieves 70.90% with comparable performance, and the "a photo of a" template baseline already reaches 70.29%, suggesting the learned prompts add only ~2% improvement.

The claimed efficiency advantage is questionable even compared to DiffPure. While the authors use 5 steps versus `dozens of Langevin iterations,` no wall-clock time comparisons are provided. Given that each step of MultiDAP involves forward passes through both the VAE encoder and the Stable Diffusion U-Net (which is much larger than DiffPure's model), the actual computational cost per step may be substantially higher. Without concrete timing measurements, the efficiency claim is unsubstantiated.

The theoretical contributions are relatively standard. Theorem 1 essentially restates known ELBO optimization results for diffusion models, and Theorem 2 applies standard projected SGD convergence analysis. The connection between these theoretical guarantees and the empirical performance is tenuous. For instance, the theory does not explain why N=5 is optimal or why performance degrades beyond 10 steps.

Important technical details are missing or unclear. How are the M=16 prompt tokens initialized? What do the learned prompts actually capture? Why is M=16 optimal rather than other values? The paper would benefit from deeper analysis of what makes the learned prompts effective for adversarial purification. Visualization or interpretation of the learned embeddings would significantly strengthen the contribution.

The regularization weight $\lambda=0.9$ for $R(x_0, x^{adv}) = \|x_0 - x^{adv}\|_2^2$ appears to be manually tuned without justification. How sensitive is the method to this hyperparameter? The choice seems arbitrary and no ablation is provided.

The evaluation lacks adaptive attacks specifically designed to break the purification mechanism. While AutoAttack is parameter-free, adaptive adversaries aware of the prompt-guided purification could potentially craft stronger attacks that exploit the fixed prompt embeddings.

### Questions
Could the authors provide a direct comparison with Lei et al. (2025) OSCP method in Table 1? Given that OSCP achieves one-step purification while MultiDAP requires 5 steps, this comparison is essential to justify the contribution. If MultiDAP is less efficient than OSCP, what advantages does it offer to compensate?

Can you provide wall-clock time measurements comparing MultiDAP, OSCP, and DiffPure? The number of denoising steps alone does not reflect computational cost, especially since Stable Diffusion models are significantly larger than the models used in DiffPure.

What happens when using the learned prompts from CIFAR-10 on other datasets (e.g., CIFAR-100, ImageNet, or STL-10)? Do the prompts transfer or need to be retrained? This would help assess whether the learned prompts capture general semantic priors or merely overfit to CIFAR-10.

Can you provide visualizations or interpretations of the learned prompt embeddings? For instance, how do they differ from text-initialized prompts quantitatively? What happens if you project them back to the nearest text tokens - do they correspond to meaningful words?

In Table 2, why does performance degrade significantly beyond N=5 steps? The robust accuracy drops from 72.38% at N=5 to 53.75% at N=50. This is counterintuitive if more iterations should better minimize the purification loss. Does this indicate instability in the optimization or overfitting to adversarial noise?

How does the method perform under adaptive attacks where the adversary has knowledge of the purification mechanism? Have you tested against attacks that specifically target the prompt-guided denoising process?

Could you provide results on ImageNet to demonstrate scalability? The current evaluation on upsampled 32x32 images is insufficient for a paper claiming a general multimodal purification framework.

### Soundness
3

### Presentation
2

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
This paper presents MultiDAP, a novel adversarial purification method that leverages a multimodal diffusion model. The core idea is to learn a class-agnostic, continuous prompt from clean image-text data, which then guides a fast (5-step) purification process to remove adversarial perturbations. The method is positioned as a zero-shot defense that does not require adversarial training or knowledge of the true label during inference. The paper provides both theoretical guarantees for the prompt learning and purification steps and demonstrates strong empirical results on CIFAR-10 against strong adversarial attacks like PGD and AutoAttack, outperforming unimodal diffusion-based baselines in both robustness and efficiency. 

The key contributions of this paper are: 1) It uses a learned, class-agnostic prompt in a multimodal diffusion model (Stable Diffusion) for adversarial purification. This effectively injects rich, data-driven semantic priors into the purification process; 2) It proposes a purification mechanism formulated as regularized DDPM-loss minimization in pixel space, which converges in very few steps (e.g., 5), making it significantly more efficient than methods requiring lengthy reverse diffusion chains;  3) It provides rigorous theoretical analysis for the proposed theorems.

### Strengths
1. The idea of moving beyond unimodal purification and hand-crafted text templates by learning a robust, semantic prompt is highly innovative and well-motivated by the limitations of existing work.

2. The experiments are comprehensive within the CIFAR-10 domain. The comparison against strong baselines and the inclusion of powerful, adaptive attacks solidly validate the method's effectiveness.

3. The proofs are sound and directly support the design choices.

### Weaknesses
1. The exclusive use of the CIFAR-10 dataset, which is low-resolution (32x32) and upsampled to 256x256, raises questions about the method's scalability and effectiveness on native high-resolution datasets (e.g., ImageNet). The performance on complex, real-world images remains unverified. 

2. There is a lack of comparison with more recent multimodal adversarial defense methods, especially those built on large Vision-Language Models (VLMs) like CLIP. Comparing with a CLIP-based zero-shot classifier or a VLM-based purification method [1] would better situate the performance.

4. The paper repeatedly emphasizes the high efficiency of MultiDAP ("only 5 steps required"). However, this comparison ignores the cost of each step. One step of DiffPure usually only requires one U-Net forward propagation. However, one step of MultiDAP requires calculating the gradient, which requires one forward propagation and one backward propagation. The computational cost of backpropagation is much greater than that of forward propagation. Please provide the average time for each image in the rebuttal and compare it fairly with the baseline calculation speed.

3. The paper does not evaluate the method against adaptive attacks specifically designed to circumvent it. An attacker aware of the purification defense could, for instance, create attacks that are robust to the purification process itself. [2] The absence of such an evaluation leaves the robustness claim incomplete.

4. The learned prompt embeddings are central to the method but are treated as a black box. The paper doesn't provide any analysis—for instance, showing the closest natural language words to the learned embeddings or visualizing the feature maps they induce in the U-Net.

5. The choice of hyperparameters (eg., $M$, $λ$) is not thoroughly ablated. 

[1] CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification

[2] DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification

### Questions
1. Whether the method maintains its performance and efficiency on larger-scale datasets like ImageNet without prohibitive computational cost or a drop in robust accuracy?

2. The current framework is tailored for the image-text modality. Is it also applicable to other multimodal settings or different tasks beyond classification?

3. While the prompt provides semantic guidance, it is unclear how it helps to remove adversarial noise specifically. Is the prompt making the loss landscape smoother, or is it providing a stronger pull towards the clean data manifold?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper introduces MultiDAP, a multimodal diffusion-based adversarial purification method. 
   - It aims to overcome limitations of prior unimodal diffusion defenses, such as implicit label retention via noise scheduling, and multimodal variants that require adversarial training or heavy distillation without theoretical guarantees. 
- MultiDAP first learns numerical contextual tokens from clean text–image pairs to enhance adversarial robustness, and then combines these learned tokens with adversarial examples to minimize a regularized DDPM loss through only a few purification iterations. 
- The authors also provide two theoretical guarantees. Experiments on CIFAR-10 show that MultiDAP achieves superior zero-shot adversarial defense performance compared with both unimodal diffusion and template-based multimodal baselines.

### Strengths
- MultiDAP breaks away from the reliance on manually designed text templates by learning numerical, context-aware tokens from large-scale clean text–image pairs. This yields class-agnostic semantic priors with richer contextual representations, improving generalization and avoiding the limitations of handcrafted prompts.

- The paper provides solid theoretical support for both key stages:
   - Theorem 1 proves that optimizing the contextual tokens monotonically increases the variational lower bound of data likelihood, ensuring the effectiveness of the learned priors.
   - Lemma 1 establishes unbiasedness and bounded variance for single-sample gradients, while Theorem 2 proves that the few-step iterative purification process ensures monotonically decreasing loss.

### Weaknesses
- Experiments are insufficient, being limited to the small-scale CIFAR-10 dataset.

- The comparison lacks recent baselines, such as CausalDiff (NeurIPS 2024) and CLIPure (ICLR 2025).

> CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense(NeurIPS 2024), 

> CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification(ICLR 2025)

- The paper does not discuss the computational cost of the two-stage framework, prompt learning and purification, nor the overhead introduced when training lightweight classification models for defense.

### Questions
- Please report the robust accuracy and inference latency on larger or more complex datasets such as an ImageNet subset (e.g., 1,000 validation images) or CIFAR-100, to validate the method’s stability under broader label spaces.
- Please include comparisons with CausalDiff, CLIPure, and DiffGuard under the same perturbation budget using AutoAttack, and report both FLOPs and measured inference time in milliseconds.
- How the proximity regularizer and prompt length affect the performance–efficiency trade-off.
-  Need to estimate or empirically evaluate the local Lipschitz constant $L$ and gradient variance upper bound $\sigma$ of the U-Net in pixel space $[0,1]^d$, to verify whether the constants in Theorem 2 are practically controllable.
- Need to report results under adaptive attacks: assuming the attacker is aware of $p^*$ and the full purification process.

### Soundness
2

### Presentation
3

### Contribution
2
