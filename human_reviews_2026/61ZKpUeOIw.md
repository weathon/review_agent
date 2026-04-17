# MGHF: Multi-Granular High-Frequency Perceptual Loss for Image Super-Resolution

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2

## Abstract
An avalanche of innovations in perceptual loss has advanced the super-resolution (SR) literature, enabling the synthesis of realistic and detailed high-resolution images. However, most of these approaches rely on convolutional neural network (CNN)-based non-homeomorphic transforms, which result in information loss during guidance and often necessitate complex architectures and training procedures. To address these limitations—particularly the information loss and unwanted harmonics introduced by CNNs—we propose a diffeomorphic transform–based variant of a computationally efficient invertible neural network (INN) for a naive Multi-Granular High-Frequency (MGHF-n) perceptual loss, trained on ImageNet. Building on this foundation, we extend the framework into a comprehensive variant (MGHF-c) that integrates multiple constraints to preserve, prioritize, and regularize information across several aspects: texture and style preservation, content fidelity, regional detail preservation, and joint content–style regularization. Information is prioritized through adaptive entropy-based pruning and reweighting of INN features, while a content–style consistency regularizer regulates excessive texture generation and ensures content fidelity. To capture intricate local details, we further introduce modulated PatchNCE on INN features as a local information preservation (LIP) objective. As another thread in the tapestry, we present the theoretical foundation, showing that (1) the LIP objective compels the SR network to maximize the mutual information between super-resolved and ground-truth modalities, and (2) a diffeomorphic transform–based perceptual loss enables more effective learning of the ground-truth distribution manifold compared to non-homeomorphic counterparts. Empirical results demonstrate that the proposed MGHF objective substantially improves both GAN- and diffusion-based SR algorithms across multiple evaluation metrics, and the code will be released publicly after the review process.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper introduces a Multi-Granular High-Frequency (MGHF) perceptual loss framework designed to overcome information loss and artifacts common in CNN-based super-resolution methods. It replaces non-homeomorphic CNN transforms with diffeomorphic, invertible neural networks (INNs) to preserve information flow. Two variants are proposed: a basic MGHF-n, trained on ImageNet, and a comprehensive MGHF-c, which incorporates multiple constraints for texture, style, and content fidelity. The model employs entropy-based feature pruning, a content–style consistency regularizer, and a modulated PatchNCE-based local information preservation (LIP) objective to enhance fine details. Theoretical analysis shows that the LIP term maximizes mutual information between SR and ground-truth images, while diffeomorphic transforms enable better manifold learning. Experiments demonstrate that MGHF significantly improves both GAN- and diffusion-based SR models across multiple benchmarks.

### Strengths
The paper is quite dense and contains a substantial amount of material.

### Weaknesses
I found the way the paper was presented to be very confusing and unnecessarily complex, making it difficult to understand what was going on.

For example, why is the information loss and unwanted harmonics introduced by CNN a problem? How do they affect the results? Invertible neural network (INN) is lossless by definition, so why is it useful to include the very complex theorems? Why is it necessary to introduce diffeomorphism here? 

I'm not an expert on diffeomorphisms, and this paper is very confusing to read.

Therefore, I feel that I do not have the expertise (diffeomorphisms) to assess this paper and suggest the AC to seek opinions from other reviewers.

### Questions
As mentioned above, I feel that I do not have the expertise (diffeomorphisms) to assess this paper and suggest the AC to seek opinions from other reviewers.

### Soundness
2

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
This paper proposes a Multi-Granular High-Frequency Perceptual Loss (MGHF) for image super-resolution (SR), leveraging invertible neural networks (INNs) to mitigate information loss and harmonic distortion inherent in conventional CNN-based perceptual losses. Two variants are introduced: MGHF-n, a naive INN-based perceptual loss, and MGHF-c, a comprehensive framework incorporating adaptive feature weighting, content-style consistency, and a local information preservation objective. The paper provides theoretical proofs supporting the superiority of diffeomorphic transforms over non-homeomorphic CNN transforms and demonstrates empirical improvements across multiple SR models and benchmarks.

### Strengths
1. The design of MGHF-n and MGHF-c is technically sound and creative, combining INN-based feature extraction with entropy-based pruning, adaptive weighting, content-style consistency, and contrastive local information preservation (PatchNCE). The hierarchical integration of these components is well-motivated.
2. The experiments cover a wide range of SR models (GANs, diffusion, transformers) and datasets (RealSR, DrealSR, DIV2K, etc.), using both reference (PSNR, SSIM, LPIPS) and non-reference metrics (CLIPIQA, MUSIQ, MANIQA). Ablation studies and robustness tests under various degradations and scaling factors further validate the method.
3. The INN-based detail feature extractor (DFE) is shown to be significantly more parameter- and memory-efficient than VGG-based extractors (Table 5), making it practical for real-world SR pipelines.
4. The paper includes feature map visualizations (Figure 3), output comparisons (Figures 8–9), and toy examples (Figure 11) that effectively illustrate the information-preserving properties of the proposed method.

### Weaknesses
1. The writing is often dense and notation-heavy, making it difficult to follow, especially in Section 2 and the appendix. Key concepts (e.g., AWDFE, LIP) could be explained more clearly. The structure of the DFE and the exact role of each loss component could be better modularized and summarized.
2. It is unclear whether the enhanced baselines (e.g., OSEDiff+MGHF-c) are trained from scratch or fine-tuned from pre-trained models. Training details (e.g., dataset splits, optimization settings) are also insufficiently described. 
3. More recent diffusion-based SR methods (e.g., SR3, CDM, LDM variants) are not included in the comparison.
4. Despite emphasizing perceptual quality, the paper relies solely on automated metrics. A user study or human evaluation would strengthen the claims of visual improvement.
5. The paper lacks a sensitivity analysis for hyperparameters such as  alpha etc. Their selection process and robustness are not discussed.
6. There is no discussion of scenarios where MGHF may underperform, such as under extreme upscaling, domain shift, or adversarial corruptions.

### Questions
1. Could you provide a direct ablation comparing MGHF-n (INN-based) with a VGG-based perceptual loss under the same settings, to isolate the benefit of the diffeomorphic transform?

2. How sensitive is the method to the choice of hyperparameters (e.g., alpha number of pruned features)? Is there empirical evidence of robustness?

3. Are there failure cases or limitations where MGHF does not perform well, such as with non-ImageNet data, extreme scaling factors, or specific real-world degradations?

4. Could you provide more qualitative results or a user study to support the perceptual improvement claims? Have human evaluators been used?

5. What is the rationale behind the pruning strategy in AWDFE? Is there a risk of losing important high-frequency details?

6. Can you clarify whether the MGHF-enhanced models are trained from scratch or fine-tuned? Please provide more detailed training protocols.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes MGHF (Multi Granduality High-Frequency) perceptual loss, aiming to improve conventional (CNN-based) perceptual losses. MGHF tackles information loss in conventional CNNs theoretically, and proposes to use INNs as a effective tool to achieve perfect information preservation. The naive version (MGHF-n) if further improved into a comprehensive variant (MGHF-c) by introducing constraints to improve preservation of texture, style, fidelity and details. Experimental results show that MGHF-c (and sometimes MGHF-n) often outperforms baselines in standard benchmarks.

### Strengths
- Utilizing INNs to improve conventional perceptual loss is interesting, and to the best of my knowledge, it is also novel.
- MGHF often outperforms baseline methods in terms of quantitative evaluation.

### Weaknesses
My primary concern is about the fundamental of this work, considering the **perception-distortion trade-off** and **information preservating with INNs**. Please counter-argue and provide according experimental results if necessary. 

---

**Weakness1**

I appreciate the effort of the authors'. However, I have doubt about the fundamental of this work. 
The authors propose to use an INN as a tool to preserve all information (theoretically proven); thereby "addressing the perception distortion trade-off (Line144)". 

However, the reviewer would like to claim that "perfect information preservation (the main contribution of this work)" contrarily induces the perception distortion trade-off (contradiction to the aim of this work); not addressing it.

This claim is supported by proofs the authors have provided.
- For instance, "Proof D.6.3 [Perceptual Loss Optimality]" concludes as "minimizing perceptual loss with INN leads to minimum distortion". However, regarding the PD trade-off [1] theory, this indicates that perceptual loss with INN (the main contribution of this work) is theoretically proven to lead to blur.
- Also, "Proof D.1" considers perceptual superiority however discusses based on L2 risk (distortion). This again indicates that minimizing perceptual loss with INNs simply indicates minimizing L2 loss. Again, this must induce blur.
- Also, "Proof D.1" concludes that "perceptual loss is extactly identical to the true L2 loss", again contradicting with authors aim.
- Also, the "Proof D.6.4" indicates perfect frequency preservation. However, it is straightforward that when attempting to regress high frequency components that posses randomness, it must lead to blur [2].

While the authors have claimed significant information loss of VGG as a limitation in perceptual guidance, I would like to argue that info loss is the key factor that enables "perceptual" guidance despite using the L2 regression form. If it did not have any info loss, the L2 regression form would have induced blur due to the PD trade-off. 

In fact, I would like to argue that the authors are also fundamentally using losses that has information loss, despite aiming for perfect info preservation. For instance, Gram matrix removes all spatial relationship and only preserves inter-channel correlationship. Despite using a info preservation perfect INN, the actual loss (after the Gram matrix calculation) has info loss; contradicting with claims of the authors.


[1] **(Please refer to the arxiv version)** Blau, Yochai, and Tomer Michaeli. "The Perception-Distortion Tradeoff." arXiv preprint arXiv:1711.06077 (2017).

[2] Lee, MinKyu, et al. "Auto-Encoded Supervision for Perceptual Image Super-Resolution." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

---

**Weakness2**

I have strong doubts about the authors claim of significant information loss in VGG (in Figure 3). Specifically, considering that spatial resolutions are reduced within the VGG architecture, the reviewer has concerns about the methodology of mere (spatial) feature visualization to verify the information loss/preservation.

Accordingly, while the authors have claimed significant information loss (e.g., chaotic information loss can be observed even in layer 3_3 in Figure 3), the reviewer would like to counter-argue that it is not true. 

Regarding to Fig.16 of DIP (Deep Image Prior) [3], most information can be reconstructed (with feature inversion) even from deep layers layers as layer 5_3 of VGG, and almost perfect reconstruction in layer 3_3 (while the authors show significant info loss). This indicates that most information (including low-level info) are preserved within the deep layer of VGG, which directly contradicts with the authors claim.

Overall, the reviewer agrees that information loss in conventional CNNs (including VGG) do indeed happen (which I believe is actually a positive aspect, see Weakness1); the current work has issues in quantifying it. 


[3] **(Please refer to the arxiv version for Fig.16)** Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Deep Image Prior." arXiv preprint arXiv:1711.10925 (2017).

---

**Weakness3**

This work needs a heavy revision regarding the presentation. 
- For instance, most parts in the Introduction section are simply listings of prior works, which should  belong to the Related Works section. I suggest aiming to provide more intuitions about the overall method and motivations of the authors.
- Also, only the 1) final performance and 2) feature visualization experiments are in the main article. The reviewer strongly suggests to include important analyses that are currently in the appendix to the main article.
- Additionally, the format is significantly altered (e.g., no spacing between paragraphs). While I appreciate the effort and understand the difficulties due to strict page limits, the current state of the formatting significantly limits readability.

### Questions
Please refer to the **Weakness**.

### Soundness
1

### Presentation
2

### Contribution
3
