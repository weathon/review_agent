# GrapHist: Large-Scale Graph Self-Supervised Learning for Histopathology

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 0, 0

## Abstract
Self-supervised vision models have achieved notable success in digital pathology. However, their domain-agnostic transformer architectures are not designed to inherently account for fundamental biological elements of histopathology images, namely cells and their complex interactions. In this work, we hypothesize that a biologically-informed modeling of tissues as cell graphs offers a more efficient representation learning. Thus, we introduce GrapHist, a novel graph-based self-supervised framework for histopathology, which learns generalizable and structurally-informed embeddings that enable diverse downstream tasks. GrapHist integrates masked autoencoders and heterophilic graph neural networks that are explicitly designed to capture the heterogeneity of tumor microenvironments. We pre-train GrapHist on a large collection of 11 million cell graphs derived from breast tissues and evaluate its transferability across in- and out-of-domain benchmarks, spanning thorax, colorectal, and skin cancers. Our results show that GrapHist achieves competitive performance compared to its vision-based counterparts, while requiring four times fewer parameters. It also drastically outperforms fully-supervised graph models on cancer subtyping tasks. Finally, to foster further research, we release eight digital pathology graph datasets used in our study, establishing the first large-scale benchmark in this field.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This study tackles the existing gap arising from the absence of dedicated graph-based foundation models in histopathology by introducing GraphHist, a graph-based self-supervised learning framework. Built upon the GraphMAE paradigm, GraphHist is pretrained on 11 million cell graphs, demonstrating competitive performance across both in-domain and out-of-domain benchmarks. Furthermore, the authors report the release of a comprehensive collection of eight graph datasets to foster future research and development in this domain.

### Strengths
1. In the field of histopathology, the development of graph-based foundation models constitutes a pivotal and forward-looking research direction. This work is dedicated to advancing the methodological and conceptual foundations of this emerging area.
2. The manuscript is well-articulated, providing a clear and coherent exposition that effectively conveys the core motivation underlying the study.

### Weaknesses
1. The paper claims to have released eight digital pathology graph datasets; however, these datasets are already publicly available, and no anonymized links are provided for verification or access.
2. The references are outdated; please incorporate recent literature, particularly within the Introduction and Related Work sections.
3. The approach presented in this paper primarily builds upon established methods such as GraphMAE, ACM, and ABMIL, and exhibits limited novelty.
4. The paper lacks comparison with recent self-supervised methods, limiting the demonstration of the proposed approach’s advantages.
5. The performance of GraphHist on patch-level subtyping tasks is substantially lower than that of MAE, indicating that, due to its limited generalizability, the proposed framework cannot be regarded as a true foundation model and is only suitable for a limited set of tasks.

### Questions
See the Weaknesses section for details.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces a graph-based self-supervised framework for histopathology region-of-interest (ROI) classification. The topic is relevant and aligns with current efforts to leverage graph representations for digital pathology.

### Strengths
The paper addresses an important and timely problem in computational pathology.

### Weaknesses
1.	Limited related work: The discussion of prior work is incomplete. Several relevant approaches using cellular graphs for whole-slide image (WSI) analysis, such as [1], are not considered.
2.	Limited downstream tasks: The framework is only evaluated on ROI classification. Including additional downstream tasks such as image retrieval or slide-level classification, as explored in works like UNI, would strengthen the paper and better demonstrate generalization.
3.	Limited metrics: Only F1 score is reported. Since F1 can be misleading when class distributions are imbalanced (it ignores true negatives and focuses solely on the positive class), metrics such as AUC-ROC would provide a more complete assessment. 
4.	Insufficient baselines: Let’s just talk about self-supervised learning. Pathology foundation models (PFMs) are also trained self-supervised and are already capable of ROI-level classification, segmentation and image retrieval, and slide-level weakly supervised learning and with experiments and results in their corresponding papers. They should also be included as baselines. Only UNI results on slide-level tasks are reported in table 8 in appendix, and it outperforms all baselines including GraphHist itself by a large margin, which raises questions about the contribution of the proposed method. 
5.	Poor performance: GraphHist’s poor performance in table 4 is not justified. Authors acknowledge that GraphHist performs poorly on non-cancerous tissues. And in real-life usecase, many WSI only has a small percentage of cancer tissues, which could significantly restrict the model’s applicability.
6.	Regarding computational efficiency: model parameters say limited things about models. Please also report peak GPU memory, inference runtime, FLOPs. Also DINOv2 was designed for a broader domain and dataset, which is apparently large. Computational efficiency should be benchmarked against same-domain models such as UNI. Also Cell segmentation overhead is not counted.
7.	Experimental design: The distinction between patch-level and slide-level experiments is unclear. The slide-level setting appears to simply aggregate patch-level predictions, offering limited support.
8.	The to-be-released digital pathology graph dataset is only acquired by applying existing cell segmentation method StarDist to public datasets, offering limited scientific contribution. 
9.	Overall Structure and Claims:The paper’s structure is difficult to follow, and several claims (e.g., large-scale learning, biologically informed modeling) are not well supported. The scale of pretraining (6,407 WSIs) is relatively modest compared to foundation models like UNI (>100k WSIs). Furthermore, the paper does not provide evidence that biological priors improve interpretability or performance. 

[1] Lu, Wenqi, et al. "Capturing cellular topology in multi-gigapixel pathology images." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020.

### Questions
1.	Spatial relationships are crucial for modeling WSIs. What is the motivation for evaluating on patch-level datasets?
2.	Table 8 shows that UNI substantially outperforms the proposed method. Given this, could the authors clarify what novel insights or contributions this work offers relative to existing PFMs?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The author proposed a novel graph-based self-supervised framework for histopathology (GraphHist). The model trained on over 10 millions cell graph derived from breast whole slide image. The approach demonstrate some performance advantages in the downstream evaluation. Also, eight graph dataset would be released to contribute the research community.

### Strengths
- A novel attempt to combine graph-based model and self-supervised pretraining technique to digital pathology
- The open source cell graph data is valuable given the laborious work to collect them.

### Weaknesses
- Over claim of generalization to other domain: Given the result presented by the author (Table 2). I find it is difficult to believe GraphHist is competitive to MAE given the large gap (e.g., 71.6 vs 54.91, 69.48 vs 55.94 and around 10% gap on average). Calling this competitive is unconvincing. Similarly on Figure 5, MAE outperforms GraphHist on all but one classe

- Lack of many baselines: The author claim that graph-based learning create structured-aware embedding but failed to prove it by comparing to other foundation model such as UNI-v2, Virchow, Giga-path, PRISM, H-optimus,...etc just name a few. The author also did not give enough discussion on the recent development of self-supervised learning model in digital pathology. I find it not convincing to believe that the graph-based learning indeed provide some advantages on the downstream tasks without comparing to the models aforementioned.

- The downstream task evaluated in the paper cannot justify structure-aware benefit: The downstream tasks in the paper are all disease subtyping or patch classification. Contrary to the selling points (i.e., structure-aware embedding has benefits) the author want to show, these tasks do not need the context of the tumor. They can be decided just by observing whether some morphology is present or not in a set of patch.  The tasks that requires interaction of patch (i.e., tumor context) to solve is survival analysis, which is neither discussed nor evaluated in the paper. 

- The scalability bottleneck: Constructing cell graph is very expensive and error-prone. First, we need human to label cell data to train nuclei segmentation then we can construct the graph. This step involves a lot of human work and nuclei detection is far from perfect, which hinders the proposed approach to scale to using more data

### Questions
Q1: I find it's too difficult to believe GraphHist is comparable to MAE. Can the author address this?

Q2: Why there are so many baselines and recent developments are lacking?

Q3: What would author think about the survival analysis, I believe this is more suitable playground for graph-based approach used in digital pathology

### Soundness
3

### Presentation
3

### Contribution
2
