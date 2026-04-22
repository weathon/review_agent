# Towards Better Generalization in Lifelong Person Re-Identification with Flatness-Aware Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Lifelong person re-identification (LReID) requires models to continuously learn from sequentially arriving domains while retaining discriminative power for previously seen identities. A key challenge is to prevent catastrophic forgetting without access to old data, especially under exemplar-free constraints. In this paper, we propose a novel LReID method that unifies selective flatness-aware optimization, dual-model training, and model interpolation. Specifically, we maintain two separate models per task: a {stability model} trained with the distillation loss to retain the prior knowledge, and a {plasticity model} optimized solely for the current domain. To improve the performance of generalization and retention, we selectively apply Sharpness-Aware Minimization (SAM) only to the distillation loss, guiding the stability model toward flat and robust solutions. After task-specific training, these two models are fused through weight-space interpolation, producing a single model that balances stability and adaptability. The resulting model is used to initialize both branches for the next task, enabling continual knowledge integration. Our method is lightweight, modular, and readily compatible with existing LReID frameworks. Extensive experimental results consistently demonstrate that the proposed flat-minima-guided model fusion strategy consistently improves the overall performance of LReID.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the challenge of lifelong person re-identification (LReID), where models sequentially adapt to new domains while avoiding catastrophic forgetting. The proposed approach unifies three main components: (1) selective Sharpness-Aware Minimization (SAM) applied to only the knowledge distillation loss, (2) dual-model training where stability and plasticity branches are optimized independently, and (3) interpolation of these two models’ weights for a fused model that balances retention and adaptability. Experiments and ablation studies on LReID benchmarks demonstrate consistent improvements in generalization and knowledge retention over a range of state-of-the-art baselines.

### Strengths
1. The application of SAM to the distillation branch provides a fresh angle within LReID. The connection to generalization and robustness is clearly supported by the visualizations in Figure 2 and Figure 3.
2. The proposed method is easily embedded into various existing LReID architectures, as shown by empirical integration with six state-of-the-art baselines.

### Weaknesses
1. While selective SAM to the distillation loss is novel in this context, the core ideas, dual-branch training and linear weight interpolation, are not entirely new and can be viewed as straightforward extensions of existing regularization and model fusion paradigms (Exponential Moving Average in DKP, DASK).
2. The ablation in Table 2 and Table explores various losses for SAM, but does not consider alternative fusion approaches, such as non-linear, confidence-weighted, or meta-learned combinations of the two branches. Given that linear weight interpolation is a central design choice (Section 4.3 and Figure 4), the lack of comparison with more advanced model merging strategies leaves a gap in validating the optimality of their method.
3. The method’s reliance on dual-model maintenance and per-branch optimization likely increases both memory and computation costs compared to standard single-model baselines, but no discussion or empirical measurement of these costs is provided.
4. The fixed hyperparameter $\lambda$ for model fusion is justified with an ablation in the appendix, but the rationale for choosing a universal value across all settings is limited.

### Questions
The method requires maintaining two full network. Can you provide measurements or analysis of computational and memory overhead (training time, inference speedup/slowdown, GPU memory footprint) relative to single-model baselines?

### Soundness
2

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
5

### Summary
This paper addresses the lifelong person reidentification (LReID) task, which has been extensively investigated recently. This paper aims to unify selective flatness-aware optimization, dual-model training, and model interpolation. Promising performance is achieved compared to existing works.

### Strengths
1. This paper is well-structured and smoothly written.

2. Promising performance is achieved compared to existing works, verifying the effectiveness of the proposed framework.

### Weaknesses
1. Unclear motivation. This paper does not explain the necessity of unifying selective flatness-aware optimization,  dual-model training, and model interpolation.

2. Limited analysis of the relation with the LReID task. This paper introduces an LReID method, while the ReID-relevant loss is not introduced.

3. Limited illustration. This paper does not contain the framework figure containing the main data stream, making it hard for readers to understand some key designs, such as dual-model training and model interpolation.

4. Unfair comparison. The training setting of this paper is different from the previous papers, where an unusual optimizer, ASDM, is adopted. Therefore, it is unclear whether the improvement in this paper is achieved via training setting bias compared to the existing works.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an LReID method that unifies selective flatness-aware optimization, where a stability model trained with distillation loss retains prior knowledge, and a plasticity model optimized solely for the current domain. It further selectively applies Sharpness-Aware Minimization (SAM) only to the distillation loss, guiding the stability model toward flat and robust solutions.

### Strengths
1. The structure is complete.
2. The experimental results verify the effectiveness of the proposed method to some extent.

### Weaknesses
1. The motivation is unclear. I don't understand what the authors mean by "well-behaved regions of the loss landscape." Also, the definition of "sharp or incompatible solutions" is confusing. The authors should use the simplest language possible to explain their ideas.
2. The proposed method is not clear enough. Due to the lack of a diagram to illustrate the method, I am unclear about what the authors did and how the method works.
3. The lack of visualization experiments makes it difficult to intuitively understand why the proposed method leads to the final performance improvement.

### Questions
See the comments below.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a novel framework for lifelong person re-identification 
(LReID) task. The framework, which unifies selective flatness-aware optimization, 
dual-model training, and model interpolation, achieves a promising performance and reduces catastrophic forgetting rates.

### Strengths
This paper originally generates a framework which can be easily interpolated and used.

This paper is written smoothly and does not have any long sentences which may lead to understanding difficulty.

### Weaknesses
Limited illustration. This paper does not have any figures that illustrate the whole framework, which includes the dual-model training. And this may lead to a misunderstanding about how the framework actually works.

Unmatched result. The results in Table 4 do not align with Table 2, as the experiment for both has the same configuration, while the results are very different. Also, there is a typo in Table 4, Line 3.

Limited formula. This paper provides limited formula derivations. Some formulas, such as the derivation of the selective SAM gradient, have not been given, which may lead to difficulties for readers to replicate.

Lack of consistent integration. This paper does not provide enough support for the integration of the main modules, which may make it seem like three modules instead of one framework.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
