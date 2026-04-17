# Histopathology-Genomics Multi-modal Structural Representation Learning for Data-Efficient Precision Oncology

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6, 2, 2, 4

## Abstract
Fusing histopathology images and genomics data with deep learning has significantly advanced precision oncology. However, genomics data is often missing due to its high acquisition cost and complexity in real-world clinical scenarios. Existing solutions aim to reconstruct genomics data from histopathology images. Nevertheless, these methods typically relied only on individual case and overlooked the potential relationships among cases. Additionally, they failed to take advantage of the authentic genomics data of diagnostically related cases that are accessible from training for inference. In this work, we propose a novel Multi-modal Structural Representation Learning (MSRL) framework for data-efficient precision oncology. We pre-train a histopathology-genomics multi-modal representation graph adopting Graph Structure Learning (GSL) to construct inter-case relevance based on the data inherently. During the fine-tuning stage, we dynamically capture structural relevance between the training cases and the acquired authentic cases for precise prediction. MSRL leverages prior inter-case associations and authentic genomics data from diagnosed cases based on the graph, which contributes to effective inference based on the single histopathology image modality. We evaluated MSRL on public TCGA datasets with 7,263 cases across various tasks, including survival prediction, cancer grading, and gene mutation prediction. The results demonstrate that MSRL significantly outperforms existing missing-genomics generation approaches with improvements of 1.44% to 3.12% in C-Index on survival prediction tasks and achieves comparable performance to multi-modal fusion methods. The code and data are available at https://github.com/WkEEn/MSRL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a two-stage framework, MSRL for Robust inference in precision oncology when genomics is missing at test time. Specifically, in stage one, self‑supervised multi‑modal Graph Structure Learning (GSL) is used to learn inter‑case structure over WSIs, genomics, and fused features. In stage two, a dual-branch with an online branch (WSI only, learns a genomics “prompt” via an SNN Inductor) and a target branch (uses authentic genomics during training, updated via EMA) was used for finetuning. In experiments, MSRL outperforms WSI‑only methods and prior “missing‑genomics reconstruction” approaches by 2.45–3.12% C‑Index on survival.

### Strengths
1.	Clarity of high-level idea and task framing: The authors make clear the distinction between conventional reconstruction and their structure guided reconstruction augmented with authentic training genomics.
2.	Methodological coherence: The pipeline that (i) pretrains a multi‑modal structural graph; (ii) transfers it to task-specific fine‑tuning; and (iii) retrieves structural support from training cases at inference is well motivated. The dual-branch (online/target with EMA) is a sensible way to stabilize learning and avoid collapse, and the hierarchical alignment losses are thoughtfully designed.
3.	Ablations and analyses: The KNN vs GSL ablations, loss ablations (especially showing $L_{salign}$ is critical), buffer source ablation, and buffer size/inference time analysis are useful and strengthen the paper.

### Weaknesses
1.	Evaluation fairness: The paper includes G‑HANet and LD‑CVAE (g.+h.→h) baselines that also aim to impute genomics. However, it is not clear whether they are allowed to retrieve/use training-case genomics at inference in the same way. If not, MSRL enjoys an extra source of information that makes comparisons less equitable. 
2.	All results are intra TCGA: an external‑cohort generalization test is highly valuable. This is especially important because MSRL leverages structural similarity and retrieval. Generalization across centers is precisely where structural priors may shift.
3.	Report performance when only a fraction (e.g., 10–50\%) of training cases have genomics (partial-paired training) and when the buffer is limited.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

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
This work proposes a Multi-modal Structural Representation Learning (MSRL) framework that pre-trains a histopathology-genomics multi-modal representation graph via graph structure learning to capture inherent inter-case relevance using paired TCGA pan-cancer data. Then, during fine-tuning, with the guidance of multimodal representation, an inductor is learned to reconstruct genomic features, which then serves as a proxy when genomic data is missing during inference.

### Strengths
1. The idea of structural alignment based on graph learning is new for addressing the problem of missing genomic data.
2. The performance of the proposed method is better when using genomic data only during inference, although there is a gap compared to the model with complete multimodal data for inference.

### Weaknesses
1. It is hard to follow the manuscript with so many notations. Moreover, a lack of intuitive insights behind the model design makes it difficult to understand the problem the authors wanted to address.
2. The related works are not comprehensive for both Fusion-based Multi-modal methods and Multi-modal methods with missing modality. For example, DisPro [1], a multimodal model for survival prediction that is robust to missing modalities, is considerably closer to the proposed method. It should be discussed and compared in the experimental section.
3. So many important details are missing in the methodology section. For example, it is confusing how the buffer assists in the modeling. And what did Readout stand for? Did the model use complete data during training? Or both missing data and complete data.
4. Data contamination issue. Given that the pretrained data is from TCGA and the finetuning datasets also come from TCGA, although they cannot see the label, there is still data contamination. External validation is necessary to validate the generalizable ability of the proposed method. For example, CPTAC is recommended to validate the model’s generalization.
5. Given that the proposed method can handle both missing and complete modalities during inference, the performance of complete modalities should be presented for fair comparison to multimodal fusion approaches. This can deepen the understanding of the performance boundary of the proposed model.
6. The ablation studies are incomplete.

    1) The effectiveness of various loss components in the pretraining should be validated.

    2) It is confusing about what MSRLonline_buffer represents.
    3) Each component of alignment loss (fused feature, structure, and graph alignment) should be investigated.
    4) The effectiveness of the buffer should be explored by removing it.
    5) The target branch can be validated by removing it and the performance combination between the online branch and the genomic data.
    6) To showcase the superiority of the proposed pretraining method, various pathology foundation models should be compared by replacing the pretrained weights, e.g. UNI [2], Virchow[3], and especially mSTAR [4], which also leveraged genomic data to enhance pathology.
7. Given that the model can handle missing modality, various missing ratios should be investigated.

[1] Xu Y, Zhou F, Zhao C, et al. Distilled Prompt Learning for Incomplete Multimodal Survival Prediction[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 5102-5111.

[2] Chen R J, Ding T, Lu M Y, et al. Towards a general-purpose foundation model for computational pathology[J]. Nature medicine, 2024, 30(3): 850-862.

[3] Vorontsov E, Bozkurt A, Casson A, et al. A foundation model for clinical-grade computational pathology and rare cancers detection[J]. Nature medicine, 2024, 30(10): 2924-2935.

[4] Xu Y, Wang Y, Zhou F, et al. A multimodal knowledge-enhanced whole-slide pathology foundation model[J]. arXiv preprint arXiv:2407.15362, 2024.

### Questions
See weakness.

### Soundness
2

### Presentation
1

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
This paper introduces Multi-modal Structural Representation Learning (MSRL), a novel framework to address the issue of missing genomics data in precision oncology. The core contribution is a two-stage approach: first, it pre-trains a graph structure learner on paired WSI-genomics data to capture inter-case relationships. Second, during fine-tuning and inference, it leverages this learned structure and a feature buffer of authentic training cases to guide predictions from WSI data alone. The method demonstrates improved performance over existing reconstruction-based approaches on survival prediction and other diagnostic tasks.

### Strengths
The primary strength of this paper lies in its novel reframing of the missing modality problem in computational pathology. Instead of treating each case in isolation, the authors introduce a structural, context-aware paradigm. The originality stems from the creative integration of self-supervised Graph Structure Learning (GSL) with a dual-branch fine-tuning strategy, effectively creating a system that learns how to leverage a knowledge base of related cases for inference. This is a significant departure from standard reconstruction methods. The quality of the work is supported by extensive experiments on multiple TCGA cohorts, demonstrating not only superior performance but also the value of each component through thorough ablation studies.

### Weaknesses
**1. Methodological Concerns on Scalability and Buffer Management:**

The core mechanism of MSRL relies on a feature buffer of historical cases during inference, which raises concerns about its long-term viability. The paper lacks a discussion on the strategy for maintaining and updating this crucial component over time. The current approach seems to imply a static buffer constructed from the training set, which would quickly become outdated in a real clinical setting where new data arrives continuously and data distributions may shift (e.g., due to new equipment or patient demographics). To address this, the authors should elaborate on a long-term maintenance strategy. For instance, would a First-In-First-Out (FIFO) queue be appropriate, or could a more intelligent, uncertainty-based or diversity-based sampling strategy be used to keep the buffer representative and up-to-date?

**2. Experimental Concerns on Fairness of Comparison and Robustness:**

A critical issue lies in the fairness of the experimental comparison. MSRL's inference process is "non-isolated" as it actively queries a knowledge base (the feature buffer) of labeled training samples. In contrast, the baseline methods (e.g., G-HANet, LD-CVAE), while trained on the same data, perform inference in an "isolated" manner for each new case. This paradigm difference may grant MSRL an inherent and potentially unfair advantage. To disentangle the benefits of the proposed GSL-based model from the benefits of this "open-book exam" paradigm, the authors should implement a stronger, paradigm-aligned baseline. A simple yet powerful baseline would be to use a standard image retrieval model to find the *K* most similar cases for a new WSI from the training set, and then simply average their corresponding ground-truth genomic data to serve as the reconstructed features. This would help clarify whether the performance gain truly comes from the sophisticated structural learning or primarily from leveraging nearest neighbors' information.

Additionally, the study's reliance on the relatively standardized TCGA dataset limits its generalizability. Real-world clinical data is notoriously heterogeneous, suffering from significant batch effects. It is unclear how robust MSRL is to such noise, as the GSL module could be misled by technical artifacts rather than true biological signals. The authors should strengthen their evaluation by conducting cross-cohort or cross-institutional validation experiments to provide a more convincing assessment of the method's real-world applicability.

**3. Lack of In-depth Interpretability Analysis:**

While the paper's title emphasizes "Structural Representation Learning," the analysis of the learned structure remains superficial. The core strength of this method should be its ability to uncover non-obvious, clinically meaningful relationships between patients. The authors should conduct several in-depth case studies to enhance interpretability. For instance, can they showcase a pair of patients with histologically distinct tumors (perhaps even from different cancer types) that MSRL connects with a strong edge? If so, can this connection be explained by shared molecular pathways, genetic mutations, or similar treatment responses documented in biomedical literature? Such analysis would greatly elevate the paper's scientific impact.

**4. Room for Improvement in Presentation and Clarity:**

The overall presentation of the paper could be significantly improved. The connection between the stated motivation and the proposed method is not immediately clear, requiring the reader to reread sections to grasp the core idea. The description of the methodology, along with the quality of the framework diagrams, could be refined for better clarity and intuition. A more direct and streamlined narrative would enhance the readability and accessibility of this otherwise promising work.

### Questions
1. **On Buffer Management:** Could you elaborate on the long-term strategy for the feature buffer? Given that a static buffer may become outdated in clinical practice due to data distribution shifts, have you considered dynamic updating mechanisms like a FIFO queue or more advanced sampling strategies (e.g., based on uncertainty or diversity) to maintain its relevance?

2. **On Fairness of Comparison:** The comparison to "isolated" inference baselines may be unfair due to MSRL's "non-isolated" use of a feature buffer. To better isolate the contribution of your proposed structural learning, would you be willing to add a stronger, paradigm-aligned baseline? For example, one that uses a standard image retrieval model to find K-Nearest Neighbors and averages their ground-truth genomics data for reconstruction. How do you expect MSRL would perform against such a baseline?

3. **On Robustness to Heterogeneity:** The experiments are confined to the standardized TCGA dataset. How would MSRL's Graph Structure Learner (GSL) perform in the presence of significant batch effects from heterogeneous real-world data? Have you considered or can you discuss how the model might distinguish true biological signals from technical artifacts during graph construction?

4. **On Interpretability of the Learned Structure:** Could you provide a concrete case study from your results to demonstrate the model's ability to uncover non-obvious, clinically relevant relationships? For example, can you highlight a pair of histologically dissimilar patients that were strongly connected by the GSL and explain the potential underlying molecular basis for this connection?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose MSRL, a novel architecture for multi-modal survival analysis and modality completion. To address the limitations of prior methods, the underutilization of gene data and the neglect of inter-patient relationships, this paper introduces a new approach that trains a gene encoder using self-supervised graph learning. The framework is validated on both modality completion for survival analysis and WSI classification.

### Strengths
The proposed method aligns well with its stated motivation, offering targeted solutions for the challenges in modality completion. The staged training approach is novel, consists of first pre-training a gene encoder with self-supervised graph learning, followed by aligning encoders trained on single modalities. The experiments are comprehensive, covering two downstream tasks and including KM-plot analysis and GSL visualizations in the supplementary materials.

### Weaknesses
The paper directly uses cross-modal kNN to construct cross-modal maps. However, due to the inherent differences between WSI and genes, it is difficult to guarantee the balance of this mapping method.
The ablation studies are missing experiments on the losses used during the representation pre-training. Additionally, I am curious about the function and impact of the "Buffer" component.

### Questions
Explain why it is reasonable to directly connect WSI and genes using kNN while the large differences between the two modalities may lead to unbalanced mapping.
The experimental validation would be significantly strengthened by additional ablation studies. Specifically, an ablation on the individual components of the pre-training loss is required to dissect their respective contributions. Additionally, the role of the buffer component should be explicitly clarified, and its impact empirically validated

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MSRL, a framework that integrates histopathology whole-slide images (WSIs) and genomics data to improve precision oncology, especially when genomics data are missing. Unlike prior methods that reconstruct genomics features from single cases, MSRL leverages inter-case relationships through graph structure learning to model structural relevance among patients. Based on large-scale evaluation from TCGA across survival prediction and precision diagnosis tasks, MSRL outperformed existing missing-modality reconstruction approaches in C-index and achieved performance comparable to full multi-modal fusion models.

### Strengths
1.	Introduces a Multi-modal Structural Representation Learning (MSRL) framework that uniquely combines graph structure learning with histopathology–genomics fusion, enabling inference from WSIs alone by leveraging inter-case relationships.

2.	Conducts extensive experiments on over 7,000 TCGA cases across multiple cancer types and tasks (survival prediction, staging, and mutation status), with comparisons to both unimodal and multi-modal baselines.

### Weaknesses
1.	While the paper provides a detailed overview of existing reconstruction-based solutions for handling missing genomics modalities, the discussion omits another important line of research — knowledge distillation or modality-distillation frameworks that transfer multi-modal information into a single-modality model for inference [1-2]. These approaches (e.g., modality distillation or teacher–student setups trained with full modalities but deployed with WSIs only) address the same practical problem more elegantly, without explicitly synthesizing genomics data. Including and contrasting these methods would strengthen the contextual positioning of MSRL, clarify its unique advantages over distillation-based solutions, and provide a more balanced view of the current landscape of missing-modality learning in computational pathology.

$\quad$ [1] Jin, Cheng, et al. "Genome-Anchored Foundation Model Embeddings Improve Molecular Prediction from Histology Images." arXiv preprint arXiv:2506.19681 (2025). 

$\quad$ [2] Xu, Yingxue, et al. "Distilled Prompt Learning for Incomplete Multimodal Survival Prediction." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

2.	The paper states that MSRL leverages authentic genomics data from diagnostically related cases during inference through graph structure learning. However, it remains unclear how this mechanism avoids potential information leakage between training and inference phases. In a realistic clinical setting, test patients may come from a distribution distinct from the training cohort, and their “related” cases’ genomics data would not be available. The manuscript would benefit from clarifying whether the model restricts graph construction to training data only, how it ensures that no test information is indirectly used, and whether any domain-shift analysis (e.g., across cancer subtypes or institutions) was performed to assess robustness when patient distributions differ. Without this clarification, it is difficult to judge the practical feasibility and generalizability of using authentic genomics profiles as auxiliary signals during inference.

3.	The paper pre-trains the MSRL model on a large-scale TCGA pan-cancer dataset and then evaluates it on TCGA subsets (e.g., BRCA, STAD, HNSC, COADREAD). While the authors mention that “all test data are excluded from the first-stage pre-training,” the training and evaluation still originate from the same overarching dataset and share highly similar data distributions. This setup risks information leakage or at least unfairly optimistic performance, as patient-level overlaps, stain distributions, or cancer-specific morphological priors could be implicitly retained from pre-training. To ensure a fair assessment of generalization, it would be valuable to include evaluations on external cohorts or at least cross-center TCGA splits to verify that the model’s performance is not inflated by intra-dataset correlations. 

4.	The AUC values reported for BRCA and NSCLC staging tasks (around 0.66) are surprisingly low compared with prior WSI-based studies, where standard MIL models such as ABMIL often achieve near-perfect discrimination (>0.95 AUC) [1] for such coarse-grained classification tasks. This discrepancy raises concerns about dataset construction, label definition, or preprocessing consistency. The paper should clarify whether these staging labels were derived from TCGA clinical metadata (which may be incomplete or imbalanced), or whether additional filtering or downsampling was applied. Without this clarification, it is difficult to interpret the reported improvements or compare them meaningfully with prior literature.

$\quad$ [1] Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." Nature medicine 30.3 (2024): 850-862.

5.	The literature review in this work is not thorough enough. First of all, for the distillation-based missing modality modeling:

$\quad$ [1] Jin, Cheng, et al. "Genome-Anchored Foundation Model Embeddings Improve Molecular Prediction from Histology Images." arXiv preprint arXiv:2506.19681 (2025). 

$\quad$ [2] Xu, Yingxue, et al. "Distilled Prompt Learning for Incomplete Multimodal Survival Prediction." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

Meanwhile, for fusion-based multi-modal methods:

$\quad$ [3] Xu, Yingxue, and Hao Chen. "Multimodal optimal transport-based co-attention transformer with global structure consistency for survival prediction." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

$\quad$ [4] Zhang, Yilan, et al. "Prototypical information bottlenecking and disentangling for multimodal cancer survival prediction." arXiv preprint arXiv:2401.01646 (2024).

Moreover, the MIL methods used for comparison did not show in the Related Work section.

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a multi-modal structural representation learning (MSRL) framework to address the issue of missing modality (it is genomics in this paper). It consists of a pre-training stage and a fine-tuning stage. In the first stage, i.e., pre-training, paired histology-gene data is collected to train GNN encoders for different modalities (including gene, histology, gene-histology fused) via a self-supervised graph contrastive learning strategy, where one node represents the corresponding features of one patient. In the second stage, a dual-branch fine-tuning strategy is adopted. It uses an online branch that takes histology WSIs as input to generate gene representation and fuses multi-modal representations (original histology + generated gene features) for prediction. To ensure the quality of gene representation generation, fused-feature, structure, and graph-level alignment are imposed, with an EMA-based model updating technique. Experiments on five datasets are conducted and their results show that the proposed framework obtains the best overall performance in compared baselines.

### Strengths
- The motivation of MSRL is clearly presented. Moreover, the analysis of the limitations of existing methods is rational.
- The proposed framework considers the potential relevance between different cases, which remains understudied in existing works.
- The core idea of the proposed framework seems interesting as it employs the WSI as the condition to generate gene features with several alignment constraints to ensure the generation quality.

### Weaknesses
My major concerns as as follows:
- The presentation of the Methods part: This part introduces many notations, individual steps, loss functions, new concepts, and new terminologies, making it hard to follow. The authors are strongly encouraged to improve it by carefully organizing each step in the algorithm and making sure their representation is self-contained and can convey information more efficiently.
- Quite-complicated solution: The proposed framework, overall, is complicated, with many techniques (self-supervised graph contrastive learning, mixup-based augmentation, dual-branch updating, etc) stacked, making it look like an over-engineering scheme.
- Unclear technical contributions in the first stage: It is unclear which algorithms are proposed by this work.
- Insufficient experimental evidence: the authors mentioned that their framework is proposed for a data-efficiency purpose. However, the experiments are missing. One could say this framework is not data-efficient as it requires pre-training on large-scale paired data. Furthermore, the authors should design new experiments (different modality-missing rates) to verify the data-efficiency advantage of the proposed framework. Besides, a state-of-the-art baseline (Xu et al., Distilled Prompt Learning for Incomplete Multimodal Survival Prediction, CVPR 2025) is not compared. The authors could also add some general, representative methods (those also address modality missing issues) for comparisons to increase the soundness.

In summary, this paper still needs substantial improvements in terms of presentation and key experimental designs to support its claims.

### Questions
- What do the authors mean by 'data-efficient' in the title? If it refers to missing data, designing new experiments (different modality-missing rates) would be better?
- Line 87: Could the authors explain in which scenarios genomic data is needed for cancer diagnosis? In most cases, histopathology WSIs (as the gold standard) are sufficient, right?
- Could Figure 2(b) be further improved to show how the missing modality is addressed more clearly? The current version is not clear enough.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 7

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a graph structure learning-based approach to fuse genomics with histology for survival prediction. The approach aims to capture the relationship between cases by constructing a graph per modality and between modalities. It consists of a pre-training phase and fine-tuning phase. The pre-training phase learns graph structures for each modality, with contrastive loss preserving intra-modal representations that are invariant to slight pertubations to the graph structure, as well as inter-modal representations that seek to align nodal representations across modalities. Meanwhile, the fine-tuning phase trains two branches: an online branch which processes only the histologic image and generates the genomic representation with an SNN, and a target branch which processes both the WSI and true genomic representation. A hierarchical alignment module encourages the histology-only online branch to have the same output as the histology+genomic target branch. During inference, only the online branch (histology only) is used. 

The method, called MSRL, is pre-trained on histology+genomic data from TCGA, and evaluated on a subset of these cases for survival prediction. Comparison against 13 total benchmark approaches (genomics-only, histology-only, histology + genomic,, and inferred genomic) demonstrate that MSRL consistently outperforms existing approaches across five survival tasks and four diagnostic tasks. 

Ablation studies investigate relevant questions, including how MSRL performance changes when exposed to histology only versus both modalities at inference. The model is also benchmarked against a sufficient number of tasks and methods. However, pre-training and evaluation are performed entirely on cases from TCGA. Their method may be particularly susceptible to distribution shift due to its reliance on generated representations and encoding of all patient samples in their generated graph, and should therefore be evaluated on external cohorts such as CPTAC as well. 

Additionally, MSRL utilizes extensive pre-training for the slide encoder, but performs benchmarks entirely against randomly initialized models. Additional comparisons with open-weight pre-trained slide encoders should be performed.

### Strengths
- The method is benchmarked against an appropriate number of tasks and organs
- The method is benhmarked against an appropriate number of approaches across a range of histology-only, histology + genomics, and inferred genomic approaches, demonstrating meaningful improvements compared to these benchmarks.
-  The proposed method is novel and addresses a new method to incorporate genomic information through a graph-based approach. 
- The ablations are well-selected and provide helpful insight into the effect of each component on performance.

### Weaknesses
- Pre-training and validation is performed entirely on cases from TCGA. Performance on out of distribution cases, from cohorts external to TCGA, should be investigated. It is possible that MSRL, which both retains all sample-level embeddings from pretraining in the form of nodes, and generates simulated genomic information, will struggle to generalize to new clinical distributions. 
- While MSRL utilizes pre-training, it is benchmarked against randomly initialized approaches. MSRL should also be benchmarked against existing lightweight slide encoders such as THREADS, CHIEF, and FEATHER. 
- The methods consist of many steps and moving parts. The section is difficult to follow, with many elements introduced without sufficient motivation or context, described below:
- Line 224-226: Please describe the post-processing steps in the appendix
- Line 231-232: Clarify the meaning of this sentence. The motivation behind using contrastive learning with intra-modal InfoNCE is not clear. In particular, please clarify that the goal is invariance between each respective node, and why this invariance is desirable. 
- Line 280: Please define the purpose of the “feature buffer” and what it represents.
- Line 276-277: Please clearly describe the purpose of the online branch and target branch. 
- Line 289-290: Many new terms are introduced here. Please define more clearly either the Inductor module or the meaning of a “genomics prompt”. 
- Line 295: Please define the meaning of the $Readout$ operation
- Line 305: make more clear what the hierarchical nature of this loss is. 
- Line 312: clarify how alignment constraint helps learn “robust representations”. What about this loss makes the online branch more “robust”?
- In addition to describing the design motivations more clearly in the methods, the motivation for using GSL is not clearly stated in the introduction. 
- There is no mention of GSL or graphs in related works

### Questions
While no cases from evaluation were used during pre-training, were the cases also stratified by patient? If a patient has multiple cases divided across pretraining and validation cohorts, then their genomic information may still be encoded in the pretraining stage. The splits used for pretraining and evaluation should be shared.

### Soundness
2

### Presentation
2

### Contribution
3
