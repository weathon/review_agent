# Spatial Structure and Selective Text Jointly Facilitate Image Clustering

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Image clustering is a fundamental task in visual machine learning. A key research direction in this field is the incorporation of prior knowledge. Recently, such prior knowledge has evolved from internal compactness constraints to external textual guidance. In particular, the introduction of textual modalities through CLIP has demonstrated impressive performance. However, CLIP is designed primarily for image–text alignment and may not be sufficient to capture clustering structures. Moreover, existing approaches often assume that textual features are universally beneficial, overlooking their varying suitability for different datasets. To address these issues, we propose using spatial structure and selective text jointly to facilitate image clustering (SATC). Specifically, we design a graph attention network (GAT)-based encoder to capture relational dependencies among image patches, thereby extracting spatial features to facilitate clustering. In addition, we introduce a textual feature selector that uses the potential clustering compactness of textual features as the selection criterion and adaptively integrates them into the clustering process. Theoretical guidance is provided for this selector. Finally, the cluster assignment is produced through tri-modal mutual distillation. Extensive experiments on 18 benchmark datasets demonstrate the effectiveness of SATC. The experimental results further verify the rationality of the textual feature selector. **Project Page:** 👉 https://zizhjiu.github.io/SATC/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the task of image clustering. It introduces a framework named **Spatial Structure and Selective Text Jointly Facilitate Image Clustering, a novel framework desigend to overcome the limitations of existing CLIP-based deep clustering methods, particularly regarding their representation of spatial structures and indiscriminate use of text features.** 

The framework fuses visual, spatial, and selectively chosen text features using a designed Tri-modal mutual distillation strategy. 

Extensive experiments across 18 benchmarks show SATC’s good performance and effectiveness.

### Strengths
1. Superior Clustering Performance: The proposed method consistently achieved highest clustering results compared to extensive prior works across 18 benchmarks, showing substantial improvements.
2. The idea of incorporating spatial structure is interesting and effective.
3. The idea of textual selection is also insightful
4. Efficiency and Scalability: SATC not only achieves higher clustering accuracy but also maintains competitive or even lower running times compared to the TAC baseline across most datasets

### Weaknesses
**Potential Flag:** The Use of Large Language Models (LLMs) is **not disclosed** in the current manuscript, which is a violation of the new rule imposed by ICLR this year.

**W1:** Experiments on novel, unseen datasets are needed. All evaluated datasets in the current work might be explicitly leveraged during CLIP training. Therefore, it is hard to confirm the effectiveness of the proposed framework without control experiments on complete unseen, novel images. It is practical and important because clustering methods are often used to explore and understand unseen, novel images without labels.

**W2:** The authors discussed the pros and cons of visual and textual modalities in CLIP, and their effects on image clustering. Why not compare with DINOv2, v3 for image clustering under the same model size? A comparison with DINOv2 and v3 that uses a single modality is necessary. KMeans + DINOv2 / v3 features is a good baseline.

**W3:** The textual feature selector and tri-modal objective function are both depent on empirically set threshold $\tau$ and $\alpha$. How do the authors select these hyperparameters? What are the selection criteria or dataset? The authors mentioned it is “based on extensive experiments” at Line#331,  however, If all parameters tuned on the  test set, it is unfair.

If the above primary concerns could be addressed during the discussion stage, the reviewer is open to rise the rating.

### Questions
**Q1:** The Tri-modal mutual distillation framework utilizes three loss types: distillation ($L_{distill}$), consistency ($L_{consist}$ with $\lambda_1=1.0$), and entropy ($L_{entropy}$ with $\lambda_2=5.0$). What specific theoretical or empirical justification underpins the choice to set the **weight of the entropy loss ($\lambda_2$) five times higher** than the consistency loss ($\lambda_1$)?

**Q2:** The final cluster assignments are consistently produced by the **distilled visual cluster head**. How would the clustering performance be affected if the final assignment were instead derived from the distilled spatial cluster head or the distilled textual cluster head, especially given that mutual distillation is shown to be superior overall?

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
Based on the externally guided clustering paradigm, this paper further leverages spatial information to distinguish visually and textually similar instances. The proposed method is extensively evaluated on 18 image clustering datasets, demonstrating superior performance over previous studies.

### Strengths
1. The contribution of this work is clear, i.e., leveraging the spatial feature in addition to visual and textual features to facilitate image clustering. Such a motivation is straightforward.
2. Extensive experiments across 18 datasets demonstrate the effectiveness of the proposed method.
3. The ablation study on incorporating textual semantics with the compactness metric is interesting.

### Weaknesses
1. The writing in section 3.1 is confusing. Where exactly is the graph attention applied? On different images, or on patches within a single image?
2. Besides the pre-trained CLIP model, a pre-trained ResNet-50 model is also utilized in the proposed method. It is questionable why ResNet-50 is needed, since CLIP could already extract both image- and patch-level features. Does the performance improvement of the proposed method come from introducing the ResNet model?
3. How are the textual compactness metrics in Eq. 5 used? It should be explained more clearly in the subsection.
4. Since the proposed SATC is more efficient than TAC, experimental results on the full ImageNet-1K are expected.

### Questions
My major concerns lie in whether the performance gain comes from the proposed method additionally leverages a pre-trained ResNet-50 model. Besides, some details of the proposed method should be explained more clearly. I will raise my score if my concerns are well addressed.

### Soundness
3

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
4

### Summary
The work combines three different modalities (visual, spatial, textual) to enhance clustering performance on a variety of image datasets. One key contribution is the newly introduced spatial modality that encodes relationships between image patches. To effectively leverage the different modalities a new framework is introduced to enforce cross-modal alignment in image clustering. Additionally, the authors establish an adaptive textual feature selector that estimates the benefits of using textual features during clustering. This prevents performance degradation on datasets where textual descriptions are uninformative or misleading. The experiments report SOTA performance across the vast majority of the 18 datasets.

### Strengths
1. The novel textual feature extractor is well motivated and proves to be of great benefit to the clustering performance
2. Comprehensive empirical validations were made on various datasets
3. SOTA results on a vast majority of datasets

### Weaknesses
1. The results are missing standard deviations to estimate the actual statistical significance of the proposed method
2. Are spatial features really a contribution or could visual-textual be sufficient? An ablation studies on the impact of the addition of spatial features for clustering would be great.
3. The compactness metric threshold appears to be found through exhaustive search rather than principled derivation
4. Typo in 324/328.

### Questions
1. Please provide the standard deviations for all datasets to estimate the statistical significance of the reported improvements.
2. What specific relational dependencies do the spatial features capture that CLIP's ViT doesn't already encode?
3. How was the 0.33 threshold determined? Based on the train/val split or post-hoc on the test data? Why does the usage of textual features need to be a binary decision rather than being modeled by weights determining their impact?

### Soundness
4

### Presentation
3

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
This paper proposes SATC (Spatial structure and Selective Text for Clustering), a tri-modal image clustering framework integrating visual, spatial, and textual information. It employs a GAT-based spatial encoder to capture relational dependencies among image patches and a compactness-aware textual feature selector to adaptively incorporate useful textual cues. These modalities are fused through tri-modal mutual distillation to improve clustering quality. Experiments on 18 benchmark datasets show that SATC consistently outperformed state-of-the-art methods such as TAC and SPICE in accuracy, robustness, and efficiency.

### Strengths
1.Originality:
The idea of combining spatial structure modeling with selective textual guidance offers a reasonable and incremental improvement over existing CLIP-based clustering frameworks.
2.Quality:
The methodology is technically sound and well-executed. The design of the spatial encoder, textual selector, and tri-modal distillation is coherent, and the experiments are comprehensive.
3.Clarity:
The paper is generally well-written and logically structured. The framework and algorithms are clearly explained, supported by intuitive figures and detailed appendices.
4.Significance:
The proposed method achieves consistent improvements across 18 datasets, showing robustness and general applicability. The approach offers a practical advancement in multi-modal unsupervised image clustering.

### Weaknesses
1. Limited theoretical grounding for the compactness threshold (τ=0.33) — while empirically validated, a more formal justification or sensitivity analysis would strengthen the argument.
2. Underdeveloped analysis of failure cases: The paper could better analyze cases where text features hurt performance (e.g., CIFAR-10), which would strengthen the argument for “selectivity.”
3. Comparative baselines: Recent multi-modal clustering approaches beyond TAC (e.g., self-supervised multi-modal alignment models from 2024–2025) are not included.

### Questions
1. Apart from TAC, are there any newer multi-modal clustering methods, such as the multi-modal alignment model for 2024-2025? Why aren't these methods taken into account for comparison?
2. The paper mentions the use of Graph Attention networks (GAT) to capture the spatial relationships between image patches, but does not elaborate on why GAT was chosen instead of other types of graph neural networks or transformer. What is the basis for choosing GAT?
3. The paper mentions that text feature selection is based on compactness (τ) and sets a fixed threshold (τ = 0.33). If the threshold is lower than this, the use of text information is abandoned, and only spatial and visual information is used. In this case, the authors believe that the text information may have provided negative benefits. However, the text-guided image clustering methods have been affirmed in some compared papers. How can this be explained?

### Soundness
3

### Presentation
3

### Contribution
2
