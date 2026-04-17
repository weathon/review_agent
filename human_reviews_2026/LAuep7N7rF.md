# Incomplete Multi-View Multi-Label Classification via Shared Codebook and Fused-Teacher Self-Distillation

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Although multi-view multi-label learning has been extensively studied, research on the dual-missing scenario, where both views and labels are incomplete, remains largely unexplored. Existing methods mainly rely on contrastive learning or information bottleneck theory to learn consistent representations under missing-view conditions, but loss-based alignment without explicit structural constraints limits the ability to capture stable and discriminative shared semantics. To address this issue, we introduce a more structured mechanism for consistent representation learning: we learn discrete consistent representations through a multi-view shared codebook and cross-view reconstruction, which naturally align different views within the limited shared codebook embeddings and reduce feature redundancy. At the decision level, we design a weight estimation method that evaluates the ability of each view to preserve label correlation structures, assigning weights accordingly to enhance the quality of the fused prediction. In addition, we introduce a fused-teacher self-distillation framework, where the fused prediction guides the training of view-specific classifiers and feeds the global knowledge back into the single-view branches, thereby enhancing the generalization ability of the model under missing-label conditions. The effectiveness of our proposed method is thoroughly demonstrated through extensive comparative experiments with advanced methods on five benchmark datasets. Code is available at https://github.com/xuy11/SCSD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework, SCSD, designed to address incomplete multi-view data scenarios that involve both missing views and missing labels. The approach incorporates three core components: (1) a shared codebook that facilitates the learning of discrete and consistent representations by aligning heterogeneous views within a compact latent space; (2) an adaptive view-weight estimation mechanism that leverages label correlations to enhance the reliability of fused predictions; and (3) a fusion-guided self-distillation strategy that transfers holistic knowledge from the fused predictions back to individual view branches, thereby improving generalization and strengthening inter-view collaboration.

### Strengths
1.The SCSD framework enhances the learning process from three complementary aspects—representation learning, decision-level fusion, and training strategy—resulting in a well-organized and conceptually coherent design.
2.The model exhibits strong scalability, and its discrete representation learning module can be easily adapted or extended to other tasks and architectures.
3.The ablation studies clearly demonstrate that the fusion-teacher self-distillation mechanism substantially boosts classification accuracy, validating its effectiveness.

### Weaknesses
1.While the framework integrates several functional modules that could affect computational efficiency, the paper does not provide a time complexity or efficiency analysis. Including such an evaluation would offer readers a clearer picture of the model’s practical scalability.
2.The codebook utilization analysis in the appendix lacks details on how the utilization rate is quantified, which reduces reproducibility.
3.In the experimental section, the authors primarily report superior results over baselines but do not offer deeper performance interpretation or insight. A more thorough discussion would enhance the paper’s contribution.
4.Since SCSD jointly addresses both missing views and missing labels, it would be helpful to discuss whether any trade-off exists between these two factors—specifically, whether the model exhibits higher sensitivity to one form of incompleteness than the other.

### Questions
See the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper addresses the challenge that existing incomplete multi-view multi-label classification methods fail to effectively capture stable and discriminative shared semantic representations. To overcome this limitation, the authors propose a novel framework named Incomplete Multi-View Multi-Label Classification via Shared Codebook and Fused-Teacher Self-Distillation (SCSD). The SCSD model learns discrete and consistent representations across multiple views through a shared codebook and a cross-view reconstruction mechanism. Moreover, a view-weight estimation strategy is introduced to adaptively assess the relative importance of different views, thereby improving the quality of the fused prediction. In addition, a fused-teacher self-distillation framework is designed to enhance the generalization capability of the model. Extensive experiments conducted on multiple benchmark datasets demonstrate the superior performance and effectiveness of SCSD compared with existing state-of-the-art methods.

### Strengths
1.	The proposed approach effectively integrates a shared codebook and cross-view reconstruction to learn consistent discrete representations across multiple views. The method is conceptually well-founded and supported by solid experimental evidence.
2.	The weight estimation strategy and a teacher self-distillation framework are introduced to improve the quality of view fusion and the generalization of the proposed model.
3.	Comprehensive experiments on five publicly available benchmark datasets show that SCSD achieves superior performance across various evaluation metrics, confirming its robustness and effectiveness.

### Weaknesses
1.	In Figures 3(a) and 3(b) (page 8), the axis labels are too small, which negatively affects readability. Enlarging the font size would improve the presentation quality.
2.	In the overall optimization objective (Eq. 10), four different loss terms are included, but a weighting coefficient is applied only to the third one. The reasoning behind this particular design choice should be further clarified.
3.	The process for determining the size of the shared codebook is not clearly explained. It would be valuable to discuss whether increasing the codebook size consistently improves performance or whether the gains plateau beyond a certain point.
4.	The manuscript would benefit from a more detailed discussion of the limitations of the proposed approach, which would make the evaluation of SCSD more balanced and comprehensive.

### Questions
Refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the SCSD framework, which aims to learn consistent multi-view representations under missing-view scenarios. The method enforces representation consistency through a shared cross-view codebook, strengthens semantic alignment via cross-view reconstruction, and adaptively refines view-specific weights based on label correlations. By integrating these components into a unified architecture, the proposed approach achieves robust and reliable performance across various benchmark datasets, demonstrating both novelty and promising experimental potential.

### Strengths
1.	The study addresses the dual-missing problem in multi-view multi-label learning, a challenging and practically significant research direction that has received limited prior exploration.
2.	The proposed shared codebook mechanism effectively mitigates instability in semantic consistency learning under missing-view conditions, outperforming existing methods theoretically and empirically.
3.	The experimental design is rigorous and systematic, and the ablation studies clearly validate the contribution of each component within the SCSD framework.

### Weaknesses
1.	Although Table 2 presents extensive quantitative results, the corresponding analysis in Section 3.4 is relatively concise. A more detailed discussion from multiple perspectives (e.g., robustness, scalability, or dataset characteristics) would improve the interpretability of the findings.
2.	The overall loss function in the method section comprises several components. Adding a concise explanatory paragraph summarizing their roles would help readers grasp the optimization objective more intuitively.
3.	While the use of a shared codebook to learn discrete representations is effective, the paper offers insufficient intuitive explanation or conceptual justification for why discretization enhances performance. Providing such insight would strengthen the theoretical grounding of the approach.
4.	Given the potential heterogeneity of feature distributions across different views, it would be valuable to discuss whether the shared codebook design might unduly restrict view-specific representations, potentially leading to semantic compression or information loss.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the dual-missing problem in multi-view multi-label learning by introducing the SCSD model. The proposed framework leverages a shared codebook to learn consistent representations across views, employs an adaptive weighted fusion mechanism to aggregate multi-view predictions based on their reliability, and adopts a self-distillation strategy to enhance model generalization. Experimental results on five benchmark datasets consistently demonstrate that SCSD outperforms existing state-of-the-art methods under various incomplete-view settings.

### Strengths
1.	The proposed weighted fusion strategy, grounded in the label correlation structure, effectively assesses the contribution of each view without introducing extra parameters, thus maintaining model efficiency.
2.	The model exhibits robust and stable performance under different levels of missing data, indicating strong adaptability to scenarios with both missing views and missing labels.
3.	The paper is clearly presented and well-structured, with coherent notation and logical flow, making it accessible and easy to follow.

### Weaknesses
1.	The parameter sensitivity analysis lacks a systematic evaluation of how varying the codebook size and embedding dimension affect model performance—two crucial factors that directly influence representational capacity.
2.	The problem formulation in Section 2.1 is somewhat lengthy, and the accompanying notations and descriptions could be further streamlined to improve clarity and readability.
3.	The hyperparameter tuning ranges for $\alpha$ and $\lambda$ are provided ($[1e^{-2}, 2e^{1}]$ and $[1e^{-3}, 5e^{-1}]$, respectively), but the rationale behind these specific search intervals is not discussed.
4.	It remains unclear whether the shared codebook mechanism introduces constraints related to the number of views or the feature dimensionality, and a discussion on this point would strengthen the paper’s technical completeness.

### Questions
Refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
