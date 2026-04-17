# ProSAR: Prototype-Guided Semantic Augmentation and Refinement for Time Series Contrastive Learning

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Contrastive learning has advanced the representation learning of vision, language, and graphs, yet its success hinges greatly on the data augmentation that helps preserve semantic contents while providing the view diversities. Multivariate time series, however, are noisy, non-stationary in nature, and largely opaque to the human inspection. Therefore, a direct use of the the hand-crafted transforms, such as jitter and scaling, may unfortunately destroy the critical temporal cues or introduce false negatives, weakening the performance of downstream tasks.
To address this, we propose ProSAR, a prototype-guided semantic augmentation and refinement framework for time series contrastive learning. Most critically, ProSAR's approach is founded on an information-theoretic principle for co-designing the semantic data augmentations and learnable prototypes, aiming to generate views that maximize the information about an associated semantic prototype while discarding the prototype-irrelevant content. ProSAR then implements this by introducing a novel prototype-conditioned semantic segment extraction mechanism, where the temporal characteristic segments are identified based on their dynamic time warping (DTW) alignment to these learnable time-domain prototypes, ensuring that the generated views can capture high-level semantic events. Building upon these temporal characteristic segments, the targeted augmentations, operating in both the time and frequency domains and informed by the DTW alignments, can thus preserve the temporal dynamics while constructing views that adhere to the information-theoretic objectives. Furthermore, prototypes are dynamically refined in a feedback loop, where the latent representations of these prototypes are refined via clustering under the prototypical contrastive training, and in turn guide evolution of the time-domain prototypes through a decoding consistency mechanism, thus fostering a progressive learning of robust representations. Experiments on diverse time-series benchmarks demonstrate that ProSAR outperforms recent contrastive and self-supervised representation learning methods in the downstream forecasting and classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ProSAR, a prototype-guided semantic augmentation and refinement framework for self-supervised time-series representation learning. Built upon the information-bottleneck principle, ProSAR co-designs learnable prototypes and data augmentations to preserve task-relevant temporal semantics while discarding noise. Specifically, it introduces (1) prototype-conditioned semantic segment extraction via DTW alignment, (2) targeted augmentation in both time and frequency domains, and (3) a dual-prototype refinement loop linking latent and time-domain prototypes through decoding consistency.

### Strengths
PProSAR is conceptually elegant and highly readable. It connects information-theoretic augmentation design with prototype learning, offering both interpretability and empirical strength. The co-design of prototypes and augmentations is well-motivated, and the dual-loop refinement provides a unified view bridging input and latent spaces.

### Weaknesses
1. While the framework is grounded in the information-bottleneck principle, the derivation stops at intuitive propositions. There is no formal proof that the proposed co-optimization converges or that the learned prototypes indeed approximate the latent semantic variable.

2. The use of DTW for semantic segmentation is computationally intensive (O(T²)), which may limit scalability for long sequences or large datasets. The paper should quantify training cost and discuss potential accelerations.

3. The framework introduces multiple components, yet lacks sensitivity analysis.

### Questions
See the above Weaknesses and the following:

1. What is the actual computational cost of DTW-based segmentation per epoch? Have you tried Soft-DTW or pruning techniques to improve efficiency?

2. How would ProSAR handle irregular or very long time series?

3. Is the prototype refinement stable under streaming or online updates?

4. Could the prototype-guided augmentation concept generalize to other modalities (spatial-temporal graphs, videos) or multi-domain transfer tasks?

5. Can you provide qualitative examples showing what a “semantic prototype” represents—e.g., typical waveform patterns or frequency signatures?

6. Provide a more formal analysis (e.g., gradient coupling or fixed-point stability) to justify the convergence of the prototype–augmentation co-design process?

### Soundness
2

### Presentation
3

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
This paper proposes ProSAR, a prototype-guided semantic augmentation and refinement framework for time series contrastive learning. The core idea is to jointly learn data augmentation strategies and semantic prototypes under an information-theoretic constraint: time-domain prototypes obtained through DTW alignment guide the generation of augmented views, applying different perturbations to semantic and non-semantic segments, while latent-space clustering and decoding consistency iteratively refine the prototypes. The proposed method aims to produce semantically consistent yet diverse views, achieving superior performance to self-supervised baselines such as AutoTCL and FreRA on both forecasting and classification benchmarks.

### Strengths
Incorporating learnable prototypes into the data augmentation process represents a meaningful conceptual innovation, breaking through the limitations of fixed or purely heuristic augmentation strategies in traditional contrastive learning.

The proposed dual-prototype mechanism—comprising time-domain and latent-space prototypes—and its iterative refinement loop demonstrate a coherent and logically consistent system design.

### Weaknesses
The authors did not compare ProSAR against several representative time-series representation learning models such as TSLANet and AimTS. Compared with these baselines, the reported results are not particularly competitive.

Although the idea of prototype-guided augmentation is interesting, the overall contribution appears incremental. The differences between ProSAR and prior works like AutoTCL and AimTS remain relatively small.

The claimed interpretability is unconvincing—especially the visualizations shown in Section D.4, which do not make much sense and fail to clearly demonstrate semantic consistency or prototype meaning.

The framework consists of multiple submodules (DTW segmentation, STFT alignment, dual prototypes, clustering, and decoding consistency), yet the ablation studies only examine the augmentation operation and the semantic segmentation, which is insufficient to validate the contribution of each component.

The reliance on DTW alignment and clustering could introduce significant computational overhead for long or high-frequency sequences. Although the paper briefly acknowledges this issue, it lacks a concrete analysis of time complexity or runtime performance.

### Questions
See weakness

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
The paper presents ProSAR, a self-supervised framework that integrates information-theoretic principles with learnable prototypes to guide semantic augmentation for time-series contrastive learning. It introduces prototype-conditioned segment extraction using DTW and a dual prototype refinement loop between latent and time-domain prototypes. Experiments on forecasting and classification benchmarks show consistent improvements over recent SSL baselines.

### Strengths
S1. The paper clearly identifies the limitation of heuristic or random augmentations in time-series CL and grounds its design in an information-bottleneck formulation.

S2. Results across both forecasting and classification tasks are strong and consistent, with comprehensive ablations demonstrating component contributions.

S3. The paper is generally well-written and the framework diagram effectively illustrates the mechanism.

### Weaknesses
W1. The idea shares conceptual similarities with prior prototype-based methods (e.g., MHCCL, AimTS); the contribution is more an integration than a fundamentally new paradigm.

W2. The DTW-based semantic segmentation and dual refinement introduce substantial computational cost and hyperparameter sensitivity.

W3. Although prototypes are claimed to be “semantic,” the paper provides minimal qualitative analysis of what semantics they actually capture.

W4. Lacks comparison with large-scale pretrained or generative SSL frameworks.

### Questions
Q1. How computationally expensive is the DTW-based segmentation step? Can the framework scale to large datasets such as Traffic or PEMS?

Q2. The method introduces both time-domain and latent-space prototypes. How sensitive is the performance to the number of prototypes or their initialization?

Q3. Can the learned prototypes be visualized or qualitatively analyzed to confirm that they correspond to meaningful temporal semantics rather than cluster artifacts?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
ProSAR (Prototype-Guided Semantic Augmentation and Refinement) is a self-supervised learning framework developed for multivariate time series (TS) contrastive learning (CL), addressing the limitation that standard hand-crafted augmentations risk destroying critical temporal cues and semantic content in noisy, non-stationary TS data. ProSAR’s approach is founded on an information-theoretic principle derived from the Information Bottleneck, aiming to generate augmented views that maximize the information about an associated semantic prototype (P) while discarding content irrelevant to that prototype. This objective is implemented using learnable time-domain prototypes as explicit semantic anchors, which guide the identification of temporal characteristic segments in the input time series (x) via Dynamic Time Warping (DTW) alignment. Experimental evaluations on diverse benchmarks demonstrate that ProSAR achieves superior performance in learning discriminative representations, attaining the highest mean accuracy (0.764) and the best mean rank (1.867) on the UEA multivariate time series archive for classification, and consistently surpassing comparison methods in forecasting tasks.

### Strengths
The submission is written clearly and is well structured, making the main ideas and technical contributions easy to follow. The motivation is articulated convincingly, and the authors provide sufficient context for why the problem is relevant and timely. Additionally, the related work section is thorough and appropriately cited, demonstrating a solid understanding of the existing literature and situating the proposed approach within the broader research landscape. Overall, the presentation is polished and the narrative is coherent and well motivated.

### Weaknesses
W1. 

While the paper is generally well written, the claimed novelty of the proposed approach is not clearly articulated or sufficiently demonstrated. The authors state that their method offers better prototypes with better semantics, but it remains unclear how these prototypes differ from or improve upon existing prototype-based contrastive learning approaches. The manuscript would benefit from a more explicit and detailed discussion of what is fundamentally novel, ideally supported by conceptual distinctions, empirical evidence, or ablations that isolate the proposed contribution. For instance, with the expressions in Lines 72 - 75, as well as Lines 92 - 96, it remains unclear how the proposed prototype-based anchors significantly differ from existing ones, and it remains unclear how the proposed method can improve the interpretability, contrallability of the anchors.


W2. 

For Line 161, "these prototypes are dynamically refined to steer the augmentation policy"; however, existing prototype-based or clustering-based constrastive learning approaches also dynamically update the prototypes. What are the key differences? 


W3. 

It remains unclear how the proposed prototypes substantively differ from those used in existing prototype-based methods. The paper would benefit from a clearer explanation of the conceptual or algorithmic distinctions. In addition, the experimental evaluation could be strengthened by including comparisons with a broader range of prototype-based and clustering-based contrastive learning approaches, which would help more convincingly demonstrate the advantages of the proposed method.

### Questions
Please see Weaknesses above

### Soundness
3

### Presentation
2

### Contribution
2
