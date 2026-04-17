# LRConfNet: Logical Reasoning-Driven Confidence Adjustment and Regularization for Hierarchical Classification of Degraded Images

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Hierarchical classification (HC) is widely applied in remote sensing and natural image analysis. However, real-world degradations—such as noise, blur, occlusion, and low resolution—often compromise fine-grained predictions for HC. Existing methods struggle to balance coarse- and fine-level accuracy, handle sparse hierarchies, and integrate multi-modal features, particularly under low-confidence predictions and complex semantic structures. We propose LRConfNet, a unified framework that addresses these challenges by combining Uncertainty Quantification (UQ) with Logical Reasoning Regularization (LogReg) to dynamically adjust classification paths. A Vision Transformer (ViT) backbone extracts global visual features, while a Semantic-Guided Cross-Attention module enables multi-modal fusion. When fine-grained confidence is low, LRConfNet triggers a logic-driven hierarchical fallback mechanism, guided by LogReg, to back off to coarse-level predictions and avoid over-classification. To further enhance generalization, we introduce a multi-level loss optimization strategy with adaptive weight adjustment. An attention enhancement loss and attention-gradient fusion are incorporated to refine spatial focus, especially confronting degraded conditions and data scarcity. Moreover, a position prompting mechanism reinforces feature selection in sparse hierarchies. Extensive experiments on degraded remote sensing and natural image benchmarks show that LRConfNet significantly outperforms SOTA methods, demonstrating superior robustness and adaptability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LRConfNet, a novel framework designed to improve hierarchical classification performance on degraded images (e.g., noisy, blurred). The core challenge addressed is that degradation increases uncertainty, leading to over-classification errors where a model makes a fine-grained prediction with low confidence. LRConfNet introduces two main components: (1) it implements a logical reasoning-driven confidence adjustment mechanism. This module uses uncertainty quantification, based on feature entropy and variance, to dynamically adjust confidence thresholds for predictions at different levels of the hierarchy. If the confidence for a fine-grained class (e.g., species) falls below its adjusted threshold, the model performs a "hierarchical fallback," reverting to a higher-level, more reliable prediction (e.g., family or order). This ensures logical consistency and prevents unreliable fine-grained labels. (2) The framework includes position-aware attention enhancement. PAAE improves spatial localization by incorporating a position prompting strategy and a multi-level loss that enhances attention on informative regions, especially under uncertain conditions. It also addresses class imbalance through adaptive weighting. The method is evaluated on degraded versions of remote sensing and natural image datasets.

### Strengths
**S1**. The paper effectively tackles the specific and challenging problem of hierarchical classification under severe image degradation, a scenario often overlooked.

**S2**. The concept of dynamic, uncertainty-aware confidence adjustment with hierarchical fallback is logically sound. It directly alleviates the key problem of over-classification.

**S3**. The ablation studies are thorough, clearly demonstrating the individual contribution of each proposed module and their synergistic effect. The use of multiple hierarchical metrics provides a robust assessment. The results show significant and consistent improvements over the baseline and competing methods on the benchmark datasets.

### Weaknesses
**W1**. Totally, my primary impression while reading the paper is that the method feels like a heavy stack of technical tricks, e.g., numerous modules and loss functions, which gives an overall sense of limited novelty. Even if there are novel elements, they appear *“diluted”* by this extensive engineering complexity. Therefore, it is recommended that the paper place greater emphasis on highlighting its core innovative components during the writing process.

**W2**. Furthermore, the paper's core motivation remains somewhat unclear. The introduction lists several important challenges: **(1)** the emergence of low-confidence samples, **(2)** severe error propagation, **(3)** the lack of confidence-aware fallback strategies, and **(4)** the difficulty in balancing performance across hierarchical levels. However, this approach comes across as merely piling up problems rather than establishing a sharp, focused narrative, making it hard to discern the central research question.

While all these problems are indeed significant, could the authors not provide more direct evidence to substantiate their existence and demonstrate that their proposed solution effectively addresses them? For instance, how exactly is the claimed problem of "error propagation" validated within the experiments? A more targeted analysis showing how predictions propagate errors in baseline models versus being mitigated by the proposed fallback mechanism would significantly strengthen the paper's claims.

**W3**. The evaluation is confined to two specific types of datasets (remote sensing ships and birds). Furthermore, the authors state on line 241, *"These complexity metrics are computed for both family and species levels,"* which gives me the impression that the work is overly focused on the specifics of their dataset rather than addressing a more generalizable problem. The generalizability of LRConfNet to other domains (e.g., medical imaging, document classification) with different hierarchical structures remains unproven.

**W4: The paper does not explicitly report a detailed comparison of computational cost** (e.g., FLOPs, latency, or model size). The paper introduces numerous modules and loss functions, such as uncertainty quantification (UQ), dynamic thresholding, and attention enhancement. Furthermore, it adds a fallback strategy for cases of low confidence. All of these components undoubtedly increase the computational complexity and inference time compared to simpler models, which could be a significant concern for real-time applications.

**W5**. The authors mention using *pre-trained GloVe (Global Vectors for Word Representation) embeddings* to extract textual information from labels. In the current era of large language models, is this type of embedding somewhat outdated? Or does the proposed method demonstrate insensitivity to the quality of the embeddings? If the method is indeed insensitive to embedding quality, could this also imply that the modules designed around these semantic priors are not critically important?

**W6: Other minor issues**. What does the superscript (e) in Equation 5 denote? It seems this index was not mentioned in the preceding equations.

### Questions
All the issues can be found in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a model named LRConfNet (Logical Reasoning Confidence Network), which integrates logical reasoning into confidence estimation within neural networks. The main goal is to enhance model reliability by enforcing logical consistency in confidence prediction. The authors introduce a logic consistency loss that regularizes the network according to first-order logical constraints, and jointly optimize it with standard prediction loss. Theoretical derivations are provided to justify the model’s learning dynamics, and experiments are conducted across multiple tasks.

### Strengths
-	The topic is timely and relevant, addressing logical consistency in uncertainty modeling.
-	The idea of embedding first-order logical constraints into confidence learning is conceptually novel.
-	Theoretical grounding is provided through formal definitions and gradient-based derivations.
-	Cross-domain validation (language and vision) demonstrates the model’s potential generality.
-	Overall writing and structure are clear and readable.

### Weaknesses
-	The method introduces too many task-specific hyperparameters, most of which lack clear justification or analysis. This heavy reliance on manual tuning weakens the persuasiveness and reproducibility of the proposed approach.
-	Weak theoretical persuasiveness: In the section “DETAILED THEORETICAL DERIVATIONS AND EXTENDED ANALYSIS,” the convergence analysis relies on strong assumptions (e.g., convexity and differentiability of the logical term) without empirical validation or ablation. This limits the applicability of the theoretical insights.
-	Notable detail issues in the manuscript: There are visible inconsistencies, such as duplicated rows in Table 12, the formatting error in the header of Table 4, and the misplaced dash in Line 1028. These presentation flaws slightly undermine the paper’s professional impression.

### Questions
1.	The sensitivity analysis explores only a few discrete parameter combinations. Could the authors provide more systematic or continuous analyses to better demonstrate the model’s robustness?
2.	The paper mentions several empirically fixed hyperparameters, but does not explain how these values were determined. Could the authors clarify the process or criteria used to choose them?
3.	The sections A.4 DETAILED THEORETICAL DERIVATIONS AND EXTENDED ANALYSIS, A.5 LOGICAL REASONING REGULARIZATION, A.6 DYNAMIC CONFIDENCE ADJUSTMENT, and A.7 HIERARCHICAL FALLBACK AND SEMANTIC CONSISTENCY mainly provide symbolic formulations without rigorous proofs or reasoning. Could the authors clarify whether these are intended as conceptual explanations or if any formal theoretical validation supports them?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes LRConfNet, a Hierarchical Classification (HC) framework tailored for degraded images. The approach combines (i) Logical Reasoning Regularization (LogReg) to enforce parent–child consistency in both probabilities and features, (ii) a feature-complexity–based uncertainty quantification that adapts decision thresholds, enabling a logic-driven hierarchical fallback when fine-grained confidence is low, and (iii) Position-Aware Attention Enhancement (PAAE) with class balancing and attention guidance to improve localization under noise and sparsity. Experiments on two constructed benchmarks, HRSC-Deg and CUB-Deg, show gains over a range of HC baselines, with strong improvements in hierarchical precision/recall and fine-grained accuracy; ablations attribute gains to both LogReg and PAAE.

### Strengths
1.The proposed LRConfNet introduces a feature-complexity-based uncertainty quantification mechanism that integrates confidence estimation with hierarchical path decision-making, clearly distinguishing it from fixed-threshold methods. The motivation and innovation are well-grounded.
2.The manuscript is well-organized and clearly written. The mathematical formulations of the loss functions, threshold updates, and hierarchical fallback are precise. Although the algorithm consists of multiple modules, the inclusion of illustrative figures and pseudocode effectively aids understanding of the overall framework.
3.The paper conducts extensive comparative and ablation experiments using multiple evaluation metrics to comprehensively assess model performance. The complementary effects of LogReg and PAAE are clearly analyzed, providing strong empirical validation.

### Weaknesses
1.The proposed hierarchical fallback mechanism essentially resembles hierarchical selective classification, yet the paper lacks direct comparisons or analytical discussion with related methods.
2.The method introduces multiple weighting and threshold hyperparameters, but no parameter sensitivity or stability analysis is provided, which is critical for deployment and generalization. Moreover, the paper does not include visual comparisons to demonstrate the effectiveness of the approach.
3.The manuscript lacks case studies of successful and failed classifications, as well as a discussion on method limitations and potential future improvements.

### Questions
1.The proposed hierarchical fallback mechanism conceptually resembles hierarchical selective classification or rejection strategies, yet no direct comparison is provided. It is recommended that the authors include relevant baseline methods or analytical experiments to clearly demonstrate LRConfNet’s improvements and advantages in hierarchical confidence modeling and decision strategies.
2.The proposed framework contains several key hyperparameters (e.g., weighting factors and threshold bounds), but lacks a systematic parameter sensitivity study.
3.To more intuitively validate the effectiveness of the proposed modules, it would be helpful to include visualization results such as attention heatmaps, feature embedding distributions, or hierarchical decision paths. These results can better illustrate how the LogReg and PAAE modules function under degraded conditions and contribute to performance improvements.
4.The paper would benefit from qualitative analyses of both successful and failed classification cases to reveal the decision logic across hierarchical levels and identify failure patterns. In addition, a more explicit discussion of the limitations and potential future directions, such as evaluation on real degraded images.

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
4

### Summary
This paper proposes LRConfNet, a unified framework for hierarchical classification (HC) of degraded images. Its core innovation lies in integrating Uncertainty Quantification (UQ) and Logical Reasoning Regularization (LogReg) to enable dynamic confidence adjustment and hierarchical fallback. When the model has low confidence in fine-grained predictions, it automatically falls back to more reliable coarse-grained predictions, thereby improving robustness and accuracy under degraded conditions such as noise and blur. The effectiveness of the method is validated through experiments on two newly constructed degraded datasets (remote sensing and natural images).

### Strengths
1. The studied problem of hierarchical classification is an interesting research topic with rich application scenarios.
2. Experimental results demonstrate that the proposed method outperforms state-of-the-art (SOTA) methods significantly.

### Weaknesses
1. **Paper Writing and Formatting:** The paper appears to be written in a rushed manner, with numerous formatting inconsistencies that hinder readability. For example, in the Introduction section, the text on the left side of Figure 1 is clearly squeezed by the figure; Figure 2 contains overly complex elements with small text, making it difficult to intuitively grasp the key points the authors intend to convey. It is recommended that the authors spend more time polishing the paper and optimizing the layout of figures to present the key points more clearly.
2. **Related Work:** Section 2.2 focuses on the development of Vision Transformers (ViTs), but no strong relevance between this work and ViTs is observed. This paper does not involve direct modifications to ViTs, and ViTs have been proposed for many years. If Section 2.2 is intended to be parallel to Section 2.1 (Hierarchical Classification), the authors should consider including content related to technologies more relevant to the current study.
3. **Logical Coherence:** The paper employs a large number of techniques, resulting in an overly engineering-oriented presentation. It is challenging to understand how each component contributes to the overall framework. Although the authors provide detailed mathematical formulations of these techniques in the appendix, intuitive explanations for the motivations of using these techniques are lacking. The authors should further elaborate on the core motivation of each technique (rather than solely relying on ablation studies to demonstrate their effectiveness).

### Questions
1. **Related Work Timeliness:** The methods compared in this paper are mostly from 2024 or earlier. As we are not direct researchers in this field, we are curious whether there are any newer research works in 2025. If such works exist, should the authors discuss and compare them in the paper?

### Soundness
2

### Presentation
1

### Contribution
1
