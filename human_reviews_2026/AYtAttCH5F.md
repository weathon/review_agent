# Diffusion-based Cumulative Adversarial Purification for Vision Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Vision Language Models (VLMs) have shown remarkable capabilities in multimodal understanding, yet their susceptibility to adversarial perturbations poses a significant threat to their reliability in real-world applications. Despite often being imperceptible to humans, these perturbations can drastically alter model outputs, leading to erroneous interpretations and decisions. This paper introduces DiffCAP, a novel diffusion-based purification strategy that can effectively neutralize adversarial corruptions in VLMs. We theoretically establish a certified recovery region in the forward diffusion process and meanwhile quantify the convergence rate of semantic variation with respect to VLMs. These findings manifest that adversarial effects monotonically fade as diffusion unfolds. Guided by this principle, DiffCAP leverages noise injection with a similarity threshold of VLM embeddings as an adaptive criterion, before reverse diffusion restores a clean and reliable representation for VLM inference. Through extensive experiments across six datasets with three VLMs under varying attack strengths in three task scenarios, we show that DiffCAP consistently outperforms existing defense techniques by a substantial margin. Notably, DiffCAP significantly reduces both hyperparameter tuning complexity and the required diffusion time, thereby accelerating the denoising process. Equipped with theorems and empirical support, DiffCAP provides a robust and practical solution for securely deploying VLMs in adversarial environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **DiffCAP**, a diffusion-based cumulative adversarial purification framework for vision-language models (VLMs). It adaptively determines the minimal diffusion steps by monitoring the cosine similarity between semantic embeddings, thus improving efficiency while maintaining robustness. The authors provide theoretical guarantees for convergence and demonstrate strong empirical results across multiple datasets and models.

### Strengths
* Theoretical proofs guarantee the convergence of semantic embeddings under forward diffusion and define a region where adversarial perturbations vanish.  
* Extensive experiments validate the effectiveness of the proposed method.  
* Unlike DiffPure’s fixed diffusion length, DiffCAP dynamically determines the minimal diffusion time via cosine similarity of VLM embeddings — an elegant and effective per-input adaptivity.

### Weaknesses
* The method represents an incremental improvement over DiffPure.  
* Although diffusion steps are adaptively determined per image, the use of a fixed similarity threshold may be domain-dependent and needs further validation.  
* The implementation of adaptive attacks is not clearly explained; gradient explosion could occur when performing full differentiable attacks and should be clarified.  
* Using CLIP-B/32 for similarity measurement may be unreliable if attackers exploit the same encoder to craft adversarial examples.

### Questions
1. Clarify how adaptive white-box attacks are implemented and whether gradient instability was observed.  
2. Analyze the sensitivity of the similarity threshold τ across different domains and models.  
3. Discuss potential vulnerabilities when using CLIP-B/32 as the embedding backbone, and evaluate robustness under CLIP-aware adaptive attacks.  
4. Provide additional justification to highlight conceptual differences from DiffPure and demonstrate that DiffCAP is more than an incremental improvement.

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
This paper presents DiffCAP, a diffusion-based strategy that effectively neutralizes adversarial damage in VLMs. By adding minimal noise to adversarially damaged images, DiffCAP alters their latent embeddings within the VLM. The process involves progressively injecting random Gaussian noise until the embeddings of two consecutive noisy images reach a predefined similarity threshold, neutralizing the adversarial effect. A pre-trained diffusion model is then used to denoise the stabilized image, restoring a clear representation for VLM output generation. DiffCAP reduces hyperparameter tuning complexity and diffusion time, accelerating the denoising process. With strong theoretical and empirical support, DiffCAP provides a robust and practical solution for deploying VLMs securely in adversarial environments.

### Strengths
- DiffCAP proposed in the paper significantly outperforms existing defense techniques, providing a robust and practical solution for securely deploying VLMs in adversarial environments.
- The theoretical analysis presented in the paper provides a strong theoretical foundation for the DiffCAP approach.
- The paper is well-written, well-structured, and easy to follow.

### Weaknesses
- The paper proposes a diffusion-based adversarial purification method. Compared to previous diffusion-model-based purification approaches, the main improvement is the introduction of a learnable, adaptive forward-noise schedule. While the paper provides a relatively detailed theoretical analysis for this approach, the analysis is based on the assumption of a classifier. Extending it to vision-language models (VLMs) would require further explanation and justification.
- The paper lacks defense experiments under large adversarial perturbations, such as an attack budget of 16/255.
- Additionally, it is unclear whether DiffCAP is equally effective on other VLMs, such as Qwen-3VL, LLaVA-OneVision, MiniGPT, or InternVL-3.5.

### Questions
- The paper mentions that the threshold $\tau=0.96$ was determined using subsets of 100 randomly selected images from each of the datasets mentioned above. How sensitive is the threshold $\tau$ to the number of images in these subsets? Have experiments been conducted with different subset sizes? Furthermore, would using Algorithm 2 to determine separate thresholds $\tau$ for different datasets or tasks improve performance? Similarly, does the threshold vary across different CLIP architectures, and have experiments been conducted in this regard?
- For Figure 2, it is recommended to include the corresponding legends for the $x$- and $y$-axes directly on the figure for clarity.

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
4

### Summary
This paper proposes DiffCAP, a diffusion-based cumulative adversarial purification method for VLMs. The key idea is to inject forward diffusion noise until the cosine similarity between consecutive VLM embeddings exceeds a threshold τ, then apply a pretrained diffusion model to reverse-denoise the stabilized image. The authors provide theory (a certified recovery region and a bound on semantic change across forward steps), and report improvements over adversarially trained encoders and purification baselines (e.g., DiffPure, CLIPure) on IC/VQA/ZSC across multiple datasets and attack strengths.

### Strengths
- Clear motivation vs. fixed-time purification: The paper pinpoints a limitation of prior works (DiffPure/CLIPure) that rely on a fixed diffusion time and proposes an input-adaptive stopping rule.
- Theoretical grounding: Thm. 1 formalizes a certified recovery region; Thm. 2 quantifies how the embedding change between adjacent forward steps shrinks toward terminal time (under a linear schedule).
- Simple, plug-in pipeline: DiffCAP requires no retraining of the downstream VLM and can wrap existing models.

### Weaknesses
- Adaptive steps vs. timesteps/step size analysis is limited; τ estimation may not generalize: Although the paper contrasts fixed-time baselines and includes an ablation over thresholds and step sizes (Fig. 2), it does not systematically quantify how different timestep schedules, step sizes (Δt), or stopping times impact robustness–fidelity across tasks/models/attack budgets. The τ selection uses mean similarity from a clean–adversarial pair set (Alg. 2); while a brief MNIST check is given, a broader out-of-domain analysis (style, resolution, long-tail categories) is missing. This raises questions about generalization and robustness when protected images lie outside the calibration domain.
- Computational overhead is under-characterized relative to baselines: The paper presents a diffusion-time box plot and notes a practical Δt trade-off, but lacks a unified, end-to-end comparison against all baselines under the same hardware/batch settings, plus a breakdown of time share (noise injection vs. embedding passes vs. reverse denoising), memory footprint, and τ-calibration cost.
- Threat-model scope (gray-box) leaves white-box adaptivity underexplored: Attacks assume gradient access to the task model but not the defense pipeline; while BPDA/EOT are considered, a fully white-box attacker aware of the stopping rule/τ and differentiating through the similarity check is not reported in the main text.
- Calibration details and reproducibility: τ is fixed to 0.96 from 100 pairs; the paper should report variance over random seeds/pair subsets and whether per-dataset or per-task τ materially helps (or hurts) clean fidelity.

### Questions
- The threshold τ is calibrated on a fixed set of clean–adversarial pairs. How well does it generalize to out-of-distribution data? Have you explored task- or dataset-specific τ values?
- Can you provide end-to-end runtime and memory comparisons with all baselines under identical hardware and batching conditions? Specifically, what is the time breakdown across noise injection, embedding computation, and reverse denoising?
- The current evaluation assumes the attacker cannot differentiate through the stopping rule. How does DiffCAP perform against a fully white-box adversary who knows τ and can approximate gradients through the cosine similarity check (e.g., via differentiable surrogates)? - How do different diffusion step sizes (Δt) or non-linear time schedules affect the robustness–fidelity trade-off? Is the current linear schedule and fixed Δt sufficiently justified across diverse models and attack budgets?
- Which pretrained diffusion model and scheduler do you use？

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DiffCAP, a novel diffusion-based adversarial purification method for VLMs. The key innovation is a cumulative noise injection approach with an adaptive stopping criterion based on VLM embedding similarity, which dynamically determines the minimal diffusion steps needed for purification. The method is supported by theoretical analysis establishing a certified recovery region and quantifying semantic convergence rates during forward diffusion. Experiments across three VLMs and six datasets demonstrate superior performance compared to existing defenses.

### Strengths
* The paper provides rigorous theoretical contributions with Theorem 1 (certified recovery region) and Theorem 2 (convergence rate quantification). The theoretical analysis effectively motivates the algorithm design and provides insights into why cumulative diffusion works.
* The per-input adaptive diffusion time is a significant improvement over fixed-time approaches like DiffPure. This addresses a key limitation in prior work and reduces computational overhead while maintaining effectiveness.
* The method is shown to maintain strong robustness even against adaptive white-box attacks like BPDA and BPDA+EOT with a high attack budget ($l_{\infty}^{8/255}$), which is crucial for a credible defense mechanism.

### Weaknesses
* The paper only tested two large VLMs (OpenFlamingo and LLaVA-1.5). Why were only these two models tested? Are they representative?
* There is little discussion of the potential trade-offs between robustness and semantic fidelity during the denoising process. Adversarial purification methods like DiffCAP aim to improve robustness, but they might introduce distortions or loss of information in the process, especially when dealing with complex multimodal data.
* The method's key hyperparameter, the similarity threshold $\tau$, is determined via a separate calibration Algorithm 2 on a subset of data. While the paper justifies the re-usability of $\tau=0.96$ for natural images and shows it's robust to domain shift, this preliminary calibration step is still necessary, introducing a dependency on a clean, representative dataset for optimal performance in a new domain.

### Questions
Please refer to the weakness part.
And some suggestions:
* Test the defense on models beyond the chosen VLMs, such as newer or more complex architectures, to ensure that DiffCAP can generalize to a wider variety of models and attack scenarios.
* Evaluate perceptual quality and semantic preservation of DiffCAP’s purified outputs, particularly in image generation or image captioning tasks, to ensure that robust defenses do not come at the cost of severely distorting the input data.

### Soundness
3

### Presentation
4

### Contribution
3
