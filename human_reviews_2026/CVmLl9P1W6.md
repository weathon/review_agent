# TESSER: Transfer-Enhancing Adversarial Attacks from Vision Transformers via Spectral and Semantic Regularization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Adversarial transferability remains a critical challenge in evaluating the robustness of deep neural networks. In security-critical applications, transferability enables black-box attacks without access to model internals, making it a key concern for real-world adversarial threat assessment. While Vision Transformers (ViTs) have demonstrated strong adversarial performance, existing attacks often fail to transfer effectively across architectures, especially from ViTs to Convolutional Neural Networks (CNNs) or hybrid models. In this paper, we introduce \textbf{TESSER}, a novel adversarial attack framework that enhances transferability via two key strategies: (1) \textit{Feature-Sensitive Gradient Scaling (FSGS)}, which modulates gradients based on token-wise importance derived from intermediate feature activations, and (2) \textit{Spectral Smoothness Regularization (SSR)}, which suppresses high-frequency noise in perturbations using a differentiable Gaussian prior. These components work in tandem to generate perturbations that are both semantically meaningful and spectrally smooth. Extensive experiments on ImageNet across 14 diverse architectures demonstrate that TESSER achieves +10.9\% higher attack succes rate (ASR) on CNNs and +7.2\% on ViTs compared to the state-of-the-art Adaptive Token Tuning (ATT) method. Moreover, TESSER significantly improves robustness against defended models, achieving 53.55\% ASR on adversarially trained CNNs and +15\% higher ASR on robust ViTs. Qualitative analysis shows strong alignment between TESSER's perturbations and salient visual regions identified via Grad-CAM, while frequency-domain analysis reveals a 12\% reduction in high-frequency energy, confirming the effectiveness of spectral regularization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes TESSER, a new adversarial attack framework to improve the transferability of adversarial examples generated from Vision Transformers (ViTs), particularly to CNN and hybrid architectures. The method combines two key strategies: (1) Feature-Sensitive Gradient Scaling (FSGS), which modulates gradients based on token importance derived from intermediate feature activations, and (2) Spectral Smoothness Regularization (SSR), which uses a differentiable Gaussian blur to suppress high-frequency noise in the perturbations. Extensive experiments on ImageNet across 14 diverse architectures demonstrate that TESSER achieves state-of-the-art attack success rates (ASR), outperforming prior methods like ATT.

### Strengths
1.  **Problem Significance:** The paper addresses the challenging and important problem of black-box transferability, specifically for ViT-to-CNN attacks, which is a known bottleneck for many existing methods.
2.  **Novel Combination:** The approach of combining semantic-aware (FSGS) and spectral-aware (SSR) regularization is a novel and valuable research direction for enhancing attack transferability.
3.  **Strong Empirical Results:** The method demonstrates significant performance gains over strong baselines across a wide array of target models, including standard ViTs, CNNs, and, importantly, adversarially defended models.
4.  **Comprehensive Analysis:** The paper provides both qualitative (e.g., Grad-CAM visualizations) and quantitative (e.g., frequency-domain analysis) evaluations to support the claims and validate the effectiveness of the proposed components.

### Weaknesses
1.  **The core assumption linking L2-norm to importance is unconvincing.** As a thought experiment, one could train two models on completely opposite (dual) tasks: one to identify the foreground and one to identify the background. The notion of "importance" for these two tasks would be entirely different. However, since the L2-norm is derived from a fixed layer, it would likely highlight the same regions for both tasks. This seems highly unreasonable and requires rigorous validation.
2.  **The paper does not specify how the Class (CLS) token is handled** within the FSGS framework.
3.  Given that the method relies on the magnitude of tokens, there is a **risk of overfitting to the specific activation patterns of the surrogate model**, which could harm transferability to models with different architectures.

### Questions
1.  Could the authors provide a more rigorous validation for the core assumption that a token's L2-norm correlates with its semantic importance? How would the method address the concern raised by the foreground/background dual-task thought experiment?
2.  Please clarify how the CLS token is processed by the FSGS mechanism. Is it included in the gradient scaling calculations, or is it treated separately?
3.  How do the authors mitigate the risk that FSGS overfits to the "important" tokens of the surrogate model? Is there evidence that this L2-norm-based importance metric is generalizable across different architectures (i.e., from a ViT surrogate to a CNN target)?

### Soundness
3

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
4

### Summary
This paper introduces TESSER, an adversarial attack framework designed to enhance transferability from ViT models. The method employs two main components: Feature-Sensitive Gradient Scaling to focus perturbations on semantically relevant regions identified by activation norms, and Spectral Smoothness Regularization to promote low-frequency, smoother perturbations. The authors claim this combination significantly improves black-box attack success rates against a diverse set of target architectures, including CNNs and robustly trained models.

### Strengths
1.  The work tackles a critical and open problem in adversarial robustness: improving the cross-architecture transferability of adversarial attacks, especially from ViTs to CNNs.
2.  The experimental setup is thorough, evaluating the attack against 14 different models, which provides a strong basis for the empirical claims.
3.  TESSER appears to outperform existing state-of-the-art baselines, including strong methods like ATT and DiffAttack, across nearly all tested scenarios.
4.  The paper includes detailed ablation studies and qualitative analyses.

### Weaknesses
The method's design is overly intuitive, and its conclusions require stronger proof. For instance, the paper claims that TESSER utilizes different frequency bands of noise information compared to previous transfer attacks. This claim should be substantiated, at a minimum, by designing experiments using low-pass or high-pass filters to verify this difference.

### Questions
Regarding the effectiveness of SSR, the paper claims it suppresses high-frequency noise. Can the authors provide a more controlled experiment to prove this? For example, could you apply ideal low-pass/high-pass filters to the perturbations and compare the spectral properties against TGR or ATT? This would help verify that TESSER is genuinely leveraging the frequency domain in a more effective or different manner.

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
The paper presents TESSER, a new method for improving the transferability of adversarial attacks. It uses two main ideas: Feature-Sensitive Gradient Scaling, which scales gradients based on the L2-norm of tokens, and Spectral Smoothness Regularization, which uses Gaussian blurring to smooth the perturbation. The goal is to create perturbations that are both semantically aligned and spectrally smooth, thereby enhancing black-box transferability from ViTs to CNNs and defended models.

### Strengths
1.  The core methodology is intuitive and well-motivated. Focusing the attack on "important" regions (via FSGS) while removing "model-specific" high-frequency noise (via SSR) is a logical approach.
2.  The method shows strong performance against adversarially trained CNNs and robust ViTs. 
3.  The ablation in Table 4 clearly breaks down the contributions of applying FSGS to the Attention, QKV, and MLP modules, justifying the decision to combine all three.

### Weaknesses
1.  The authors should supplement their baseline comparisons by benchmarking against the wider set of attacks available in the `TransferAttack` repository (https://github.com/Trustworthy-AI-Group/TransferAttack).
2.  The setting for the perturbation budget (epsilon) is unusual.
3.  The experimental design involves resizing inputs to different dimensions, but the potential effect of this `resize` operation on transferability is not discussed.

### Questions
1.  Could the authors provide results using a more standard epsilon value, such as $8/255$? The current value of $16/255$ is a very large perturbation budget for ImageNet. Is this high value necessary for the method's success?
2.  Please discuss the potential role of the `resize` operation (to $224 \times 224$) in the attack's success. When attacking a model like Inc-v3, which expects $299 \times 299$ input, could this resolution mismatch be implicitly acting as a form of input diversity, thus aiding transferability?
3.  Have the authors considered comparing TESSER against the broader set of SOTA methods in the `TransferAttack` repository? This would help to more firmly establish the paper's contribution relative to the field.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors propose TESSER, a novel adversarial attack framework designed to improve the transferability of adversarial examples generated from Vision Transformers (ViTs). To address the lack of semantic selectivity in perturbations, TESSER introduces a Feature-Sensitive Gradient Scaling (FSGS) module, which estimates token importance using the L2 norm of intermediate activations and adaptively scales gradients, thus suppressing high-norm tokens in shallow layers (treated as noise) while amplifying those in deeper layers to reinforce semantic features. In addition, the authors incorporate Spectral Smoothness Regularization (SSR), which employs a differentiable Gaussian prior to suppress high-frequency noise in perturbations. Extensive experiments conducted on various ViT and CNN architectures using the ImageNet benchmark dataset demonstrate that TESSER achieves superior attack performance compared with baseline methods.

### Strengths
1. This paper introduces Feature-Sensitive Gradient Scaling (FSGS), which steers perturbations toward semantically meaningful features to improve cross-architecture generalization.

2. This paper further proposes Spectral Smoothness Regularization (SSR) to encourage smoother and low-frequency perturbations that exhibit greater robustness across different architectures.

3. The proposed TESSER methodology demonstrates strong attack transferability across diverse target models.

### Weaknesses
1. The novelty and justification of FSGS and SSR are insufficiently supported. FSGS builds on the assumption that high-norm tokens carry richer semantic information, yet its claim that high-norm tokens in shallow ViT layers represent noisy signals lacks empirical or theoretical validation. Similarly, the SSR component (applying Gaussian smoothing to suppress high-frequency noise) closely resembles TI-FGSM [1], but the paper neither cites nor compares with this prior work. The rationale for selecting Gaussian blur over alternative smoothing techniques is also unexplained.

2. The claim that TESSER’s perturbations achieve better semantic alignment is not convincingly demonstrated. In Section 4.4, the authors rely solely on Grad-CAM for evidence. However, Grad-CAM and its variants were originally developed for CNNs, and their applicability to ViTs is limited compared with ViT-specific interpretability methods (e.g., Attention Rollout [2], Libragrad [3]). Moreover, this section presents only a few qualitative visualizations in Figure 2 and lacks any quantitative evaluation to substantiate the claim.

3. The experimental comparisons are incomplete, raising concerns about fairness and the practicality of TESSER. The performance gains reported in Tables 1–3 omit comparisons with FPR [4], a closely related state-of-the-art method that shares a similar objective. Moreover, several baseline results (e.g., ATT, TGR) appear to be cited directly from the ATT paper, though it remains unclear whether the experimental settings and execution environments of these methods are consistent with those used in this work.

4. The practicality of TESSER framework is questionable. It introduces numerous hyperparameters (e.g., $\lambda$, $\tau$, $l_{cut}$, $\epsilon$, $\omega$) that require extensive tuning, which increases methodological complexity and hinders reproducibility. However, the paper provides no sensitivity analysis to assess the impact of these parameters.

5. The ablation study is insufficient and fails to isolate the contributions of the paper’s core components. The proposed framework comprises three main modules: FSGS, SSR, and Module-wise Gradient Modulation. However, the ablation study (Table 4) only evaluates TESSER on different module combinations without excluding the Module-wise Gradient Modulation component, which closely resembles the ATT baseline. As a result, the individual effects of FSGS and SSR cannot be clearly identified.

[1] Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks. CVPR 2019.

[2] Quantifying Attention Flow in Transformers. ACL 2020.

[3] LibraGrad: Balancing Gradient Flow for Universally Better Vision Transformer Attributions. CVPR 2025.

[4] Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement. CVPR 2025.

### Questions
1. The proposed SSR appears functionally similar to the Gaussian kernel convolution used in TI-FGSM for gradient smoothing. Could the authors clarify the key distinctions and provide a stronger justification for the novelty of SSR?

2. The FSGS method assumes that high-norm tokens in shallow ViT layers represent “noise signals” that should be suppressed. Could the authors provide empirical evidence or theoretical justification to substantiate this assumption?

3. The claim of enhanced semantic alignment is supported only by qualitative Grad-CAM visualizations. Could the authors provide quantitative evidence to validate this claim?

4. Could the authors conduct a more comprehensive ablation study to better evaluate the individual contributions of FSGS and SSR?

5. Could the authors clarify how TESSER’s attack performance compares with FPR (CVPR 2025)?

6. Given that TESSER involves numerous hyperparameters, could the authors include a sensitivity analysis of key hyperparameters to demonstrate the method’s robustness and practicality?

### Soundness
2

### Presentation
2

### Contribution
2
