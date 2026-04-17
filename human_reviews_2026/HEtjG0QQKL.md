# DDAE++: Enhancing Diffusion Models Towards Unified Generative and Discriminative Learning

- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
While diffusion models excel at image synthesis, useful representations have been shown to emerge from generative pre-training, suggesting a path towards unified generative and discriminative learning. However, suboptimal semantic flow within current architectures can hinder this potential: features encoding the richest high-level semantics are underutilized and diluted when propagating through decoding layers, impeding the formation of an explicit semantic bottleneck layer.
To address this, we introduce *self-conditioning*, a lightweight mechanism that reshapes the model's layer-wise semantic hierarchy *without external guidance*. By aggregating and rerouting intermediate features to guide subsequent decoding layers, our method concentrates more high-level semantics, concurrently strengthening global generative guidance and forming more discriminative representations.
This simple approach yields a dual-improvement trend across pixel-space UNet, UViT and latent-space DiT models with minimal overhead. Crucially, it creates an architectural semantic bridge that propagates discriminative improvements into generation and accommodates further techniques such as contrastive *self-distillation*.
Experiments show that our enhanced models, especially self-conditioned DiT, are powerful dual learners that yield strong and transferable representations on image and dense classification tasks, surpassing various generative self-supervised models in linear probing while also improving or maintaining high generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces DDAE++, an enhanced architecture and training framework for Diffusion Models (DMs) aimed at achieving unified generative and discriminative learning. The core problem addressed is the sub-optimal utilization of rich semantic features within the standard U-Net architecture of DMs, which hinders the quality of the learned representations.

DDAE++ proposes two main architectural and training modifications:

1. Self-Conditioning Mechanism: It introduces a lightweight self-conditioning mechanism that explicitly routes rich, high-level semantic features (extracted from intermediate layers of the U-Net) directly into the denoising process.

2. Augmented Self-Supervised Learning (SSL) Objective: It trains the model with an auxiliary SSL objective using contrastive learning, guided by different views (data augmentations) of the same instance.

Through these modifications, DDAE++ claims to establish a stronger representation bottleneck within the U-Net, leading to superior performance in both generative tasks and discriminative tasks.

### Strengths
1. Unified and Principled Approach: The paper successfully integrates generative and discriminative objectives within a single, coherent diffusion framework. This addresses a critical gap by turning the denoising process into a powerful representation learner, maximizing the utility of the pre-training stage.

2. Architectural Efficiency: The core mechanism is lightweight self-conditioning, which does not require significant changes to the base U-Net structure or a substantial increase in computational complexity, making it practical for adoption.

3. Strong Representation Learning Results: The performance of the learned representations, particularly demonstrated through improved linear probing and transfer learning results (e.g., on ImageNet, CIFAR-100), clearly establishes DDAE++ as a state-of-the-art method among diffusion-based representation learners.

### Weaknesses
1. Lack of Component Analysis and Insight: While the effectiveness of the proposed components (e.g., the CLS token and the AUG token) is demonstrated via ablation, the paper lacks a deep analytical insight into why these mechanisms are effective from a feature flow perspective. The claim of establishing a "stronger representation bottleneck" remains largely descriptive rather than mechanistically proven. 

2. Marginal Generative and Discriminative Improvements: The quantitative improvements in generative quality (e.g., FID and Acc.) shown in Table 2 appear to be marginal over baselines, especially considering the added complexity of the SSL objective. This raises questions about the practical utility of DDAE++ specifically for users primarily focused on high-fidelity generation.

3. Incomplete Foundational Comparison (SSL): The paper claims to unify generation and self-supervised learning, yet it omits a crucial empirical comparison with highly relevant semantic-enhanced diffusion methods like REPA or other state-of-the-art masked modeling/denoising approaches like MaskDiT. 

4. Missing Evaluation on Complex Tasks (T2I): The evaluation is primarily conducted on unconditional and class-conditional image synthesis (CIFAR-10, ImageNet). A critical test for a modern diffusion enhancement would be its performance on the more complex, widely-used Text-to-Image (T2I) generation task. The current evaluation leaves uncertainty regarding DDAE++'s utility and scalability in the T2I domain, where high-level semantic flow is paramount.

### Questions
1. In table 1, the authors demonstrate DDAE++ have sample quality++ and feature quality++ compared to DDAE. Despite performance enhancement, what are the fundamental  improvements of DDAE++? 

2. Did the augmented SSL loss introduce any noticeable challenges in training stability or convergence speed compared to the original DDAE objective? Do the authors need special techniques (e.g., warmer learning rates, different schedules) to stabilize the combined generative and discriminative training?

3. Can the authors visualize or quantitatively measure the semantic content or feature diversity of the latent space (e.g., through Singular Value Decomposition or dimensionality) at the self-conditioning point before and after applying DDAE++?

4. What is the sensitivity to the choice of the internal layer used for extracting the semantic token (CLS/AUG)? Is there a theoretical or empirical justification for the chosen layer depth?

5. Why can SSL improve the generation capabilities of diffusion models. Have the authors explored scenarios where the improved discriminative performance (SSL) comes at a noticeable detriment to generative quality (FID)?

6. Please provide a quantitative comparison of DDAE++'s generative performance (FID) and representation learning performance (Linear Probing and Transfer Learning) against REPA.

7. Could the authors comment on the scalability and expected performance of DDAE++ when applied to a modern, large-scale Text-to-Image (T2I) diffusion model (e.g., Stable Diffusion)? Do the authors anticipate needing further modifications to the self-conditioning mechanism to handle the complexity of the cross-attention layers in T2I models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This manuscript explores leveraging the generative representation in the diffusion model for downstream discriminative purposes. Specifically, they propose self-conditioning, routing important semantic features obtained by probing into the decoder layer of the diffusion networks, along with the timestep condition. On top of that, the authors follow EDM and MoCo v3 for data augmentation and contrastive losses. Experimental results demonstrate that the proposed technique enables improved discriminative accuracy while maintaining the FID over the self-reported baselines on standard image benchmarks.

### Strengths
* Analyzing and improving the representation of generative models is an important topic.
* The method is applicable to various network backbones.
* The ablation study part is performed in detail.

### Weaknesses
**Experimental Evaluation**: The reported generative metric, i.e., FID, in this paper, is worse than expected. For example, the official EDM [1] on CIFAR-10-uncond is 1.97, which is much better than the reported 2.23 in the baseline in Table 2. While I understand that re-training the model from scratch could be computationally intensive, the mismatched generative performance makes it hard to judge the efficacy of the proposed method on a well-trained diffusion model. I strongly suggest considering adding fine-tuning experiments on top of a pre-trained diffusion model with the proposed architectural change, and then examining whether there is a performance gain in the discriminative capacity while maintaining the original generative ability.

**Clarity**: There are multiple inappropriate, vague statements in the paper, which should be revised:
   * While determining the intermediate layers is a crucial procedure, the description on lines 203 -- 208 is not informative. For example, it is unclear what "short, parallel training runs" are. There should be at least a reference to the details of this experiment.
   * On lines 247--249, "a vector of transformation parameters is used..." is not informative. The augmentation used should be explicitly specified, either using a reference (if you are following [1] in the exact same way) or in the paper. 
   * On lines 294 -- 295, "after stabilizing some hyperparameters in feature evaluation" is also not informative.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a simple yet effective modification called **self-conditioning** to improve both the generative and discriminative capabilities of diffusion models. The method reroutes semantically rich intermediate features to guide decoding layers, creating a stronger representation bottleneck without external supervision. This architectural enhancement allows diffusion models to better integrate discriminative objectives such as contrastive self-distillation, achieving consistent dual improvements across various backbones (UNet, UViT, DiT) and datasets. Experiments show that DDAE++ boosts both generation quality (lower FID) and feature quality (higher linear probing accuracy) with minimal computational overhead, positioning diffusion models as unified learners for generation and recognition.

### Strengths
The paper presents a clear and intuitive idea, supported by solid experimental analysis. In particular, the layer-wise feature analysis via linear probing in Figure 5 provides valuable insights into why the proposed DDAE method is effective — this diagnostic perspective is worth learning from. Overall, both Table 3 and Figure 3 convincingly demonstrate that DDAE is an effective and practical approach.

### Weaknesses
The claim in Figure 6 that "Self-conditioning facilitates the optimization and narrows the loss gap between un- and class-conditioning" does not hold. From the curves shown, we can only observe that both the conditional and unconditional loss curves converge faster, but there is no evidence indicating that the gap between them actually decreases.

### Questions
The main text contains excessive whitespace, and the formatting in the appendix (around Line 712) is messy and requires careful reorganization.

### Soundness
3

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
5

### Summary
This paper addresses a key limitation in diffusion models—namely, the sub-optimal semantic information flow affecting the quality of learned representations—by introducing a lightweight architectural mechanism called self-conditioning. The method aggregates high-level semantic features from intermediate layers and injects them back into the decoding pathway, forming a more effective bottleneck for discriminative features without relying on external cues. Additional integration of contrastive self-distillation and principled data augmentations is demonstrated to further unify and amplify both generative and discriminative capacities of diffusion models. Empirical evidence on several backbones (UNet, UViT, DiT) and datasets (CIFAR-10/100, ImageNet) shows dual improvements in generation (FID/IS) and representation (linear probing accuracy compared to strong self-supervised baselines).

### Strengths
- The work identifies and addresses a genuine architectural weakness in standard diffusion models: the absence of a discriminative bottleneck due to distributed semantic flow. The “self-conditioning” mechanism is both simple and effective—clearly illustrated in Figure 1 and Figure 2—and does not require external supervision.

- The empirical evaluation is extensive and well-controlled, covering a variety of diffusion model backbones (UNet-based, UViT, DiT) and datasets (CIFAR-10/100, Tiny-ImageNet, ImageNet at various scales). The methodology section and ablations demonstrate care in validating architectural choices, as well as strong generalization of the performance boosts. The results are thoroughly tabulated in Table 2, Table 3, and Table 4, and consistently show that self-conditioning brings gains to both generative (FID/IS) and discriminative (linear accuracy) metrics.

- The method is lightweight (as detailed in the ablations and parameter increase figures/tables in Sections 5.1 and 5.3) and can be retrofitted to diverse diffusion architectures.

### Weaknesses
- Although the experiments are broad, there is a bias towards popular image datasets, especially at lower resolutions. In Section E.1 (Limitations and Future Research Directions), the authors admit that their results stop at ImageNet 256x256 and DiT-base scale due to compute constraints, and do not demonstrate scalability for larger models/datasets relevant to modern AI.

- The ablation studies in Table 5 focus on demonstrating parameter sensitivity and certain hyperparameter effects (e.g., MLP head design, distillation timestep, etc.). However, there is little diagnostic quantification of why self-conditioning enhances the bottleneck: for instance, layer-wise representation transferability, sparsity/concentration metrics, or feature alignment before and after rerouting are not rigorously analyzed.

- The comparison in Table 4 is unconvincing because the FID scores on ImageNet for both the baseline and the self-conditioning variant indicate that the models have not reached an optimal level of convergence. While we acknowledge that computational constraints may limit the number of training epochs, conclusions drawn from comparisons between undertrained models are not reliable.

### Questions
- Mathematical Formalization: Can the authors provide a precise, equation-level definition of the “time-adaptive scaling” used in self-conditioning? How is this parametrized and learned? Please specify if there is a generalizable mathematical principle or derived gradient supporting its use.
- Rigorous Bottleneck Selection: Is there theoretical—or at least empirical—evidence that the training-loss-based search for the bottleneck layer always selects optimal features for discriminative purposes? Could the authors provide a more principled justification for this heuristic?
- Limits of Dual-Improvement: In Table 2 / Table 3, there are cases where FID is marginally degraded, or improvements are not observed concurrently. Can the authors break down these exceptions and clarify the precise conditions under which self-conditioning may not help (or may hurt)?
- Feature Concentration Analysis: Beyond qualitative shifts shown in Figure 5, have the authors performed any quantitative analysis (e.g., feature entropy, linear separability, mutual information with ground truth) of the “condensed bottleneck” feature? This would greatly strengthen the core claim.
- Augmentation Sensitivity: Given the importance and variability of data augmentations, can the authors provide more systematic experiments or concrete recommendations on how to choose or tune these for new datasets? What are the consequences if the geometric pipeline is poorly matched to the data?
- Layer Dynamics: The authors mention that the peak accuracy emerges at a different layer than the one chosen for self-conditioning (Figure 5, Section 5.3). Could the authors elaborate on the implications of this misalignment for both generation and discrimination?
- Scalability Limits: Are there any architectural or optimization bottlenecks (beyond compute cost) that emerge at higher dataset resolutions or with substantially larger backbones/latent spaces?

### Soundness
2

### Presentation
3

### Contribution
2
