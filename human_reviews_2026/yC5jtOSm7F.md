# From Histopathology Images to Cell Clouds: Learning Slide Representations with Hierarchical Cell Transformer

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
It is clinically crucial and potentially beneficial to analyze and directly model the spatial distributions of cells in histopathology whole slide images (WSI). However, existing methods typically analyze WSIs via image representation learning and ignore the importance of cell distributions. Thus, it remains an open question whether deep learning models can directly and effectively analyze WSIs from the semantic aspect of cell distributions. In this work, we argue that each WSI can be regarded as a collection of cells and propose a new scheme consisting of cell detection and cell cloud modeling to tackle these challenges. Firstly, we propose a novel human-in-the-loop label refinement method to finetune the pretrained cell detection and classification model. Then, a novel hierarchical Cell Cloud Transformer (CCFormer) is proposed to model the cell spatial distribution. Specifically, a Neighboring Information Embedding module is proposed to characterize the distribution of cells within the cell neighborhood, and a Hierarchical Spatial Perception module is proposed to learn the spatial relationship among cells in a bottom-up manner. Clinical analysis indicates that clinical evaluation metrics directly based on counting cells can effectively assess patients' survival risk, offering significant potential for analyzing and modeling cell distribution in WSIs. Besides, extensive experiments on survival prediction and cancer staging show that CCFormer achieves state-of-the-art performances and evidently outperforms other competing methods by learning from cell spatial distribution alone.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work presents a new method, named Hierarchical Cell Cloud Transformer, to address the WSI-level tasks, including survival analysis and cancer staging. The authors designed the Human-in-the-Loop Label Refinement strategy to reduce the cost of manual annotation for building the cell segmentation model. Besides, they introduce a novel Neighboring Information Embedding (NIE) to capture neighborhood cell distribution at the cell level, and a Hierarchical Spatial Perception (HSP) method to model cell spatial distribution information in a bottom-up manner. Clinical analysis and extensive experiments on multiple public WSI datasets were performed to demonstrate the efficacy of the cell cloud framework and the CCFormer model.

### Strengths
The manuscript presents a new idea for whole slide image analysis from a single-cell perspective. The proposed method captures the spatial information of cells, such as cell interaction, by the hierarchical Cell Cloud Transformer, which can enhance the WSI-level representation. The proposed method outperforms existing methods significantly across multiple datasets and tasks. This work might provide a new insight into WSI analysis for the research community.

### Weaknesses
The proposed approach might be complex, which involves cell segmentation, patch classification, feature representation and aggregation, and WSI prediction. It is not end-to-end and might be computationally extensive. Experimental analysis might not be very comprehensive, such as missing results and evaluation metrics.

### Questions
1)	My major concern about this work is that the proposed method proceeds in two steps: first cell segmentation, and then WSI-level analysis. As mentioned by the authors, each cancer dataset contains 200 to 700 million cells. There is inevitably a significant computational cost for segmentation, which largely hinders real-world applications. Please elaborate on the test time per WSI in the manuscript.
2)	In the hierarchical spatial perception, the authors applied mean aggregation at the low levels, but used maximum aggregation at the last level. Besides, for the appearance feature, the authors applied global-average pooling, which is also different from the last-level operation. Please elaborate on this.
3)	Some results are missing in Table 1 without any explanation, such as PAMOE. As PAMOE is publicly available, it is possible to run such experiments.
4)	Did the authors borrow results from existing works? The results of MambaMIL in the published paper are not the same as those in Table 1, but the results of PAMOE are the same as those in Table 1. I am quite fused. Moreover, PAMOE used LGG dataset for evaluation and achieved promising results (i.e., 0.793) for survival prediction, however this work skips the comparison.
5)	For evaluation metrics for cancer staging, the authors used the Macro-F1 score only. However, accuracy and AUC were commonly adopted in previous works. Please supplement these as well. Please specify whether the improvement in the percentage format is in absolute or relative terms.

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
3

### Summary
This paper proposes treating WSIs as cell point clouds, a new paradigm distinct from MIL. Specifically, the paper introduces a human-in-the-loop label-refinement method to tackle domain shift between the pretraining dataset (PanNuke) and the target dataset (the TCGA series), and a cell-cloud transformer to model cell spatial distribution. Extensive experiments demonstrate the effectiveness of this method.

### Strengths
1.	The paper is generally well-organized and easy to follow, and would be better if the methodology section could easier to understand.

2.	The main idea of treating WSIs as cell point clouds is innovative and aligns well with clinical practice.

3.	The experimental results demonstrate the effectiveness of this method along with its components.

### Weaknesses
1.	The choice of PanNuke is not entirely convincing. In Appendix C, the authors state that PanNuke is chosen due to the distribution similarities between PanNuke and TCGA. What about merging other segmentation datasets, such as MoNuSAC and Lizard, since more data generally leads better generalizability?

2.	The authors are advised to disclose the foundation model’s precision and recall (or related metrics) used during HLLR, which would help solidify the effectiveness of the approach.

3.	There should be more baselines for comparison in the main experiments: for graphs, HEAT (CVPR) and the Integrative Graph-Transformer framework (MICCAI); for patch-feature MIL, HDMIL (CVPR) and R2T-MIL (CVPR).

4.	Formatting issues: \citep{xxx} should be used in most cases, while the authors use \citet{xxx}. Define CCFormer when it first appears in the main text. In addition, what is the difference between hierarchical CCFormer and CCFormer? If they are the same, please use the terminology consistently.

### Questions
In lines 156 to 158, were these experiments conducted by the authors, or are they results from previous studies?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a novel paradigm for WSI analysis by modeling them as "cell clouds" instead of image patches. The authors propose a two-stage approach: first, they convert WSIs into cell point clouds using a cell detection model fine-tuned with a Human-in-the-Loop strategy; second, they introduce a hierarchical transformer, CCFormer, to learn spatial distributions from these clouds for downstream tasks. Experiments on survival prediction and cancer staging across TCGA datasets show that the method achieves state-of-the-art performance, particularly when fused with appearance features.

### Strengths
The paper's primary strength is its novel formulation of the WSI analysis problem. Shifting the focus from conventional image patches to a holistic "cell cloud" representation is an original and interesting direction. This cell-based perspective holds the potential for better model interpretability by directly linking predictions to cellular spatial patterns, a significant advantage over abstract patch-based features. The technical approach is also reasonably well-designed; the proposed Neighboring Information Embedding (NIE) and Hierarchical Spatial Perception (HSP) modules are logical and well-motivated components for capturing local and global cell distribution characteristics within this new framework.

### Weaknesses
Despite its novel perspective, the paper suffers from several significant weaknesses that undermine the reproducibility, experimental depth, and the broader impact of its claims.

*   **W1: Unsubstantiated Claims of Cost-Efficiency.** The paper repeatedly suggests its approach is efficient and low-cost, but fails to provide the necessary evidence or context.
    *   **Ambiguous Annotation Cost:** The Human-in-the-Loop Label Refinement (HLLR) is presented as a "low-cost" solution. However, this claim is made in a vacuum. The paper neglects to discuss that the primary competitors, weakly-supervised MIL methods, require only inexpensive slide-level labels, whereas HLLR requires a subset of highly granular (and thus expensive) cell-level annotations. To be convincing, the authors must provide a direct discussion of this trade-off, ideally quantifying the expert time required and justifying why this higher-quality annotation effort is more cost-effective than standard weak supervision.
    *   **Inadequate Computational Cost Analysis:** The claim of 2.1 minutes/WSI on a high-end NVIDIA H20 GPU is not a meaningful benchmark without a direct comparison to the main baselines. The paper is missing a critical experiment: an end-to-end timing comparison on the *same hardware* against a representative patch-based method (e.g., UNI feature extraction + MIL aggregator).

*   **W2: Critical Lack of Evaluation for Upstream and Downstream Components.** The paper's evaluation is incomplete, making it difficult to understand the source of performance gains and the model's behavior.
    *   **Missing Cell Classifier Performance:** The methodology's success hinges on the quality of the initial cell cloud, yet the performance of the cell detection and classification model is never reported (e.g., F1-score, Precision). Without this, it is impossible to disentangle the contributions of the cell classifier from the CCFormer architecture, hindering error analysis and reproducibility.
    *   **Insufficient Ablation for FusedCCFormer:** The FusedCCFormer model shows significant performance gains, suggesting that patch-based appearance features are crucial. However, the paper lacks a proper ablation study. The authors should present results for three distinct models under their framework: (1) CCFormer using only cell features, (2) a baseline using only the patch-based foundation model features (e.g., global average pooling of UNI features), and (3) the combined FusedCCFormer. This would clarify the complementary value of each feature type.

*   **W3: Lack of Clarity and In-depth Interpretation.** Key aspects of the methodology and results are not clearly explained.
    *   **Unclear Visualizations:** The interpretation of several figures is opaque. For instance, the conclusion drawn from the similarity map in Figure 6(c) is not well-supported by the provided explanation. It is unclear how the visualization demonstrates that the model "comprehends semantic relationships" beyond simply showing an attention map. A more rigorous explanation is needed.
    *   **Missed Opportunity for Interpretability Analysis:** A primary motivation for cell-based analysis is the potential for high-level interpretability—linking patient outcomes to specific cellular compositions and spatial arrangements (e.g., tumor-immune interactions). Despite claiming this as an advantage, the paper does not provide any in-depth analysis of the learned representations to offer such clinical or biological insights. This is a significant missed opportunity to showcase the most compelling advantage of their paradigm over less interpretable patch-based methods. Given that patch-based approaches likely have an edge in annotation and computational cost, a strong demonstration of interpretability is crucial to justify the proposed framework.

*   **W4: Limited Practicality due to Hardware Specificity.** The experiments were conducted on an NVIDIA H20, a specialized datacenter GPU. The lack of benchmarks on more common hardware (e.g., consumer-grade RTX series) makes it difficult to assess the practical applicability and accessibility of the method.

### Questions
Thank you for this interesting work. I have several questions that I hope you can address during the rebuttal phase. My assessment of the paper could change significantly based on your responses.

1.  **Regarding Cost-Benefit Analysis:** Your work's central premise is a shift towards a new paradigm. To evaluate this shift, a clear understanding of its costs is crucial.
    *   **Q1.1 (Annotation Cost):** Could you please provide an estimate of the expert annotation time (in hours) required to label the 743 samples for the HLLR fine-tuning? More importantly, could you elaborate on why this targeted, cell-level annotation is more cost-effective than using readily available slide-level labels required by standard weakly-supervised MIL methods?
    *   **Q1.2 (Computational Cost):** Could you provide an end-to-end computational time comparison between your full pipeline (cell detection + CCFormer) and a leading patch-based baseline (e.g., UNI feature extraction + MambaMIL) on the same hardware for a representative WSI? This would provide a much-needed direct comparison of efficiency.

2.  **Regarding Component Evaluation & Ablation:** The contribution of each part of your pipeline is currently unclear.
    *   **Q2.1 (Cell Classifier Performance):** What was the performance (e.g., F1-score for each class) of your fine-tuned DPA-P2PNet cell classifier on a held-out test set? Providing these metrics is critical for understanding the quality of the input to CCFormer and for assessing the reproducibility of your results.
    *   **Q2.2 (Feature Ablation):** The performance leap of FusedCCFormer is substantial. Could you provide ablation results that compare three models: (1) CCFormer (cell features only), (2) a baseline using UNI features only (e.g., with global average pooling), and (3) FusedCCFormer? This would precisely quantify the individual and complementary contributions of cell-spatial and image-appearance features.

3.  **Regarding Interpretability:** The potential for interpretability is a key advantage of your cell-based approach, but it remains underexplored.
    *   **Q3.1 (Clarification of Figure 6c):** Could you please provide a more detailed explanation of how the attention map in Figure 6(c) demonstrates the comprehension of "semantic relationships"? Specifically, what are the semantic characteristics of Region I, II, and III that lead to these high/low attention scores?
    *   **Q3.2 (Deeper Interpretability Analysis):** Beyond visualizations, have you performed any analysis to link specific learned cell spatial patterns (e.g., high density of immune cells near tumor clusters) to patient outcomes? Showcasing even one such concrete example would significantly strengthen the claim that your method offers superior interpretability over patch-based models, which is arguably the most important justification for the additional costs of your framework.

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
3

### Summary
This paper aims to address the problem of WSI analysis by introducing the spatial distribution of cells as a key marker, which has not been explored in previous work. Specifically, in the proposed framework, the authors treat the cells in a WSI as a point cloud. There are two major components in the proposed model: (1) an active-learning–based expert annotation strategy to reduce labeling workload, and (2) a hierarchical transformer architecture that extracts cell distribution information at different scales. The model is evaluated on several datasets, and the experimental results demonstrate that the proposed method outperforms prior approaches on two tasks: survival prediction and cancer staging.

### Strengths
* The idea of introducing cell distribution and treating cells as a point cloud is interesting and reasonable, and has not been explored previously.

* The hierarchical approach to modeling the cell cloud is well-motivated and technically sound.

* Given the extremely high cell density in WSIs, the HLLR module offers a sensible solution to reduce annotation effort.

* The paper is well-written and easy to follow.

* The experiments are rigorous and comprehensive.

### Weaknesses
* Although the idea of modeling cell distribution for WSI analysis is novel, it is not very feasible in practical applications. WSIs are extremely high-resolution, and the high cell density results in a substantial annotation burden—even with an active-learning strategy. In addition, the computational complexity grows significantly with cell density. However, the paper does not provide any runtime or memory usage metrics for training or inference, nor does it compare the computational cost with existing models.

* CCFormer relies heavily on the performance of cell detection, yet the paper does not report any quantitative results (e.g., accuracy, precision) for the detection step, nor does it analyze how detection quality affects overall slide-level performance.

* CCFormer utilizes only cell spatial distribution and cell type for WSI analysis, while ignoring other informative features such as texture, morphology, and overall tissue structure. These features are typically crucial in patch-based methods and are important in clinical pathology.

* Some claims are made without adequate explanation or justification. For example, the statement “Heavy reliance on the foundation models results in high computational costs and suboptimal performance” is not supported with evidence or comparison.

### Questions
* How robust of  WSI level analysis w.r.t. cell detection performance? 
* How the computational complexity of the proposed method compared with other patch-level models.
* How sensitive of CCFormer to the choice of number of group anchors at different level and other hyper-parameters of the hierachical grouping ?

### Soundness
3

### Presentation
2

### Contribution
2
