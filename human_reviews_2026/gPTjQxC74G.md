# Joint Adaptation of Uni-modal Foundation Models for Multi-modal Alzheimer's Disease Diagnosis

- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder and a leading cause of dementia worldwide. Accurate diagnosis requires integrating diverse patient data modalities. With the rapid advancement of foundation models in neurobiology and medicine, integrating foundation models from various modalities has emerged as a promising yet underexplored direction for multi-modal AD diagnosis. A central challenge is enabling effective interaction among these models without disrupting the robust, modality-specific representations learned from large-scale pretraining. To address this, we propose a novel multi-modal framework for AD diagnosis that enables joint interaction among uni-modal foundation models through modality-anchored interaction. In this framework, one modality and its corresponding foundation model are designated as an anchor, while the remaining modalities serve as auxiliary sources of complementary information. To preserve the pre-trained representation space of the anchor model, we propose modality-aware Q-formers that selectively map auxiliary modality features into the anchor model’s feature space, enabling the anchor model to jointly process its own features together with the seamlessly integrated auxiliary features. We evaluate our method on AD diagnosis and progression prediction across four modalities: sMRI, fMRI, clinical records, and genetic data. Our framework consistently outperforms prior methods in two modality settings, and further demonstrates strong generalization to external datasets and other neurodegenerative diseases such as Parkinson’s disease.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a multi-modal framework for AD diagnosis that enables joint interaction among uni-modal foundation models through modality-anchored interaction. The experimental results show the effectiveness of the proposed framework.

### Strengths
(1) This paper proposes modality-aware Q-formers that selectively map auxiliary modality features into the anchor model’s feature space, enabling the anchor model to jointly process its own features together with the seamlessly integrated auxiliary features.

 (2) The proposed method is evaluated on AD diagnosis and progression prediction tasks involving the four most common data modalities, and experimental results validate the effectiveness of the proposed framework.

### Weaknesses
(1) This paper designates one modality’s foundation model as an anchor and freezes most of its parameters to preserve its feature space, while projecting auxiliary modalities’ features extracted by other foundation models into this space for cross-modal interaction. This technical novelty is very limited.

 (2) There are several existing works about Modality-aware Q-formers (Tong et al., 2024; Zong et al., 2024; Alayrac et al., 2022; Liu et al., 2023a). Thus, it is not clear about the difference between the proposed model with these existing works. 

(3) This paper should compare the proposed model with more state-of-the-art AD diagnosis methods.

 (4) The authors should clearly summarize the contributions of this paper.

### Questions
(1) There are several existing works about Modality-aware Q-formers (Tong et al., 2024; Zong et al., 2024; Alayrac et al., 2022; Liu et al., 2023a). Thus, it is not clear about the difference between the proposed model with these existing works. 

(2) This paper should compare the proposed model with more state-of-the-art AD diagnosis methods.

(3) The authors should clearly summarize the contributions of this paper.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a novel multi-modal framework for diagnosing Alzheimer's Disease (AD) through the joint adaptation of uni-modal foundation models. The central concept is the utilization of a "modality-anchored interaction" strategy. In this strategy, the foundation model of one modality serves as an anchor, and features from other auxiliary modalities are projected into its feature space via a proposed "Modality-aware Q-former". This enables interaction among modalities while maintaining the pre-trained representations of the anchor model. The framework is evaluated in the context of AD diagnosis and progression prediction, leveraging four modalities: sMRI, fMRI, clinical records, and genetic data, and it showcases superior performance compared to previous approaches.

### Strengths
- The proposed modality-anchored interaction framework for integrating uni-modal foundation models is a novel and interesting approach.
- The paper addresses the important problem of multi-modal AD diagnosis, which has significant clinical relevance. The use of foundation models is a promising direction.
- The method shows strong performance on AD diagnosis and progression prediction, outperforming several baselines. The generalization experiments on external datasets are a key strength.

### Weaknesses
- The proposed framework is quite complex, involving multiple foundation models, Q-formers, and a two-stage training process. This might make it difficult to reproduce and apply in practice.
- While the paper provides some analysis, more in-depth ablation studies could further clarify the contribution of each component (e.g., the cross-modal Q-former, the LoRA fine-tuning).
- The paper states that each modality is used as an anchor in turn, and the final prediction is an aggregation. It would be interesting to see an analysis of how the choice of anchor modality affects performance. Is there an optimal anchor?

### Questions
1.  Could you provide more details on the computational cost of the proposed framework, both during training and inference?
2.  Have you experimented with other methods for aggregating the predictions from the different anchor models? For example, a weighted average based on the confidence of each model.
3.  How sensitive is the model to the number of learnable queries in the Q-formers?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to use multiple Q-formers to align various medical modalities for Alzheimer's disease prediction. The authors specifically use train these Q-formers by taking one modality as an anchor, and aligning the other modalities to the latent space of the anchored modality.

### Strengths
The quantitative results are quite convincing, the authors compare the method across three different datasets, and they perform various ablation studies.

### Weaknesses
1) The authors focus a lot on classification performance, but there are many other tasks that are possible within these datasets. Specifically, to further strengthen their results the authors can show that their model also performs better when predicting biomarkers, age, etc. The classification tasks, although important, are not the only way to evaluate these models.
2) The approach is not that technically novel. There are many papers that align modalities using Q-formers, and it is therefore unclear to me what the technical novelty is in this paper beyond a new application area. A paper that mostly focuses on a new application area is fine, but the application area is quite limited (the authors restrict themselves to Alzheimer's disease instead of broadly neurological disorders), and although results are clearly better in the tables, it is unclear how significant these results are.
3) This brings me to the third point, which is that the authors do not seem to use cross-validation when evaluating their model(s), and do not compute confidence intervals or standard deviations for their results. Especially in neuroimaging, where the exact training and test subsets can have a large impact on the results, it is important to perform experiments over multiple training and test sets, and even initialization seeds, to ensure results are repeatable.
4) The authors mention modality-aware Q-formers in the introduction, but do not discuss the referenced papers in depth in the related work section. In general, I think the authors can do a much better job at placing their work into the context of current work, especially given how large the field of multimodal foundation models is. Make sure to highlight exactly how your model and use of Q-formers is different, and why this is leading to performance improvements.

### Questions
1) Figure 2: What is the difference between the authors' method and a Q-former?
2) The unimodal performance for the M4Survive and LateFusion models is sometimes worse than the unimodal performance, which begs the question: How fair are the comparisons with the baselines? Do the authors use the same unimodal foundation models and fine-tuning approaches? For example, in the authors' model they use fine-tuned unimodal foundation models, is this also the case for the multimodal baselines?

### Soundness
3

### Presentation
2

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
The paper proposes a multi-modal framework for Alzheimer’s Disease (AD) diagnosis that integrates several uni-modal foundation models (for sMRI, fMRI, clinical records, and genetic data). The core innovation is the modality-anchored interaction mechanism, where one modality serves as an anchor, and others act as auxiliary inputs. To align heterogeneous feature spaces, the authors introduce modality-aware Q-formers, transformer-based connectors that selectively map auxiliary features into the anchor modality’s feature space. The method is evaluated on ADNI, OASIS-3, and PPMI datasets for AD and Parkinson’s disease, showing state-of-the-art accuracy and generalization under both modality-complete and modality-incomplete scenarios.

### Strengths
1. Innovative framework: The modality-anchored interaction and modality-aware Q-former are novel and conceptually strong for integrating pre-trained models across heterogeneous medical modalities.
2. Strong performance: The proposed method consistently outperforms baselines across tasks, achieving superior accuracy, specificity, and sensitivity in AD diagnosis and progression prediction.
3. Comprehensive evaluation: Results are validated on multiple datasets (ADNI, OASIS-3, PPMI), demonstrating solid generalization to out-of-distribution data and to a different disease.
4. Thorough experimentation: Includes modality-complete/incomplete settings, ablation on foundation model choice, number of queries, and visualization of attention maps for interpretability.

### Weaknesses
1. Limited interpretability for clinicians: While attention visualization is provided, the interpretability of the fused multimodal decision process remains abstract; more clinical linkage to specific biomarkers would be valuable.
2. Complexity and scalability: The two-stage fine-tuning pipeline increases computational demands; practical feasibility in clinical deployment is not analyzed.
3. Dependence on large pre-trained models: The framework assumes availability of powerful foundation models, which may not always be accessible or easy to fine-tune with limited resources.
4. Limited cross-modal interpretive analysis: The paper would benefit from a deeper analysis of why certain modality combinations improve results (e.g., specific complementarity between clinical and imaging data).

### Questions
1. Q-former Configuration: How sensitive are the results to the number of queries and attention heads in the modality-aware Q-former? Could a lighter version achieve comparable performance?
2. Scalability: Can the approach be generalized to more than four modalities or to non-neuroimaging domains?
3. Interpretability: Can the method mechanisms be linked to specific neurobiological or genetic markers for clinical interpretability?

### Soundness
3

### Presentation
4

### Contribution
3
