# Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Graph anomaly detection (GAD) has garnered increasing attention in recent years, yet remains challenging due to two key factors: (1) label scarcity stemming from the high cost of annotations and (2) homophily disparity at node and class levels. In this paper, we introduce Anomaly-Aware Pre-Training and Fine-Tuning (APF), a targeted and effective framework to mitigate the above challenges in GAD. In the pre-training stage, APF incorporates node-specific subgraphs selected via the Rayleigh Quotient, a label-free anomaly metric, into the learning objective to enhance anomaly awareness. It further introduces two learnable spectral polynomial filters to jointly learn dual representations that capture both general semantics and subtle anomaly cues. During fine-tuning, a gated fusion mechanism adaptively integrates pre-trained representations across nodes and dimensions, while an anomaly-aware regularization loss encourages abnormal nodes to preserve more anomaly-relevant information. Furthermore, we theoretically show that APF tends to achieve linear separability under mild conditions. Comprehensive experiments on 10 benchmark datasets validate the superior performance of APF in comparison to state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce Anomaly-Aware Pre-Training and Fine-Tuning (APF), a two stage framework for supervised GAD. Experiments show the effectiveness of the framework.

### Strengths
1. The authors introduce Anomaly-Aware Pre-Training and Fine-Tuning (APF), a two-stage framework for supervised GAD. 
2. Experiments show the effectiveness of the framework.

### Weaknesses
1. The novelty can be a question, as all the components are from existing work, which means such a framework can be considered an implementation rather than a contribution. 
2. How to choose the hyperparameters for new datasets can be a question since the combinations of hyperparameters require extensive experiments for grid search, as shown in Appendix H.4.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenges of label scarcity and homophily disparity in graph anomaly detection (GAD). It proposes a novel framework called Anomaly-Aware Pre-Training and Fine-Tuning (APF), which incorporates a label-free anomaly metric (Rayleigh Quotient) and dual spectral filters during pre-training to capture both semantic and anomaly-sensitive signals. For fine-tuning, APF employs a gated fusion mechanism and an anomaly-aware regularization loss to adaptively handle node- and class-level homophily disparities.

### Strengths
1. Drawing on recent research findings, this paper provides an in-depth analysis of local data issues in graph anomaly detection.

2.It creatively applies Rayleigh Quotient to node-level subgraph selection and demonstrates its effectiveness through experiments.

3.Experiments span 10 diverse datasets, ensuring robustness and generalizability across domains such as social networks and finance.

### Weaknesses
1. The authors only cite the relationship between spectral energy (i.e., the quantification of the "right shift" phenomenon) and anomaly degree, but do not clearly discuss its connection to graph anomaly detection. For example, is this anomaly due to abnormal node attributes or abnormal edge connections? Will the two types of anomalies differ in spectral density / energy?

2. Considering the use of the MRQSampler algorithm in this paper, the authors should add a discussion of UniGAD to clarify the paper's distinction and contribution.

3. The design of the Dual-filter Encoding lacks detailed explanation.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses graph anomaly detection under label scarcity by introducing APF, a two-stage framework that combines anomaly-aware pre-training with granularity-adaptive fine-tuning. The core methodology consists of: (i) a pre-training stage that uses the Rayleigh Quotient to guide subgraph sampling and employs both low-pass and high-pass learnable filters to capture both semantic and anomaly-specific signals, and (ii) a fine-tuning stage that adaptively fuses these representations via a gated fusion network with anomaly-aware regularization. The authors provide theoretical analysis under an Anomalous Stochastic Block Model to demonstrate potential linear separability and conduct extensive experiments on 10 benchmark datasets to validate the effectiveness of APF.

### Strengths
**S1**. The paper articulates the challenges in GAD, particularly label scarcity and homophily disparity, in a clear and accessible manner. The motivation for using the Rayleigh Quotient as a label-free anomaly indicator is well-explained, and the distinction between local and global homophily effectively highlights limitations of uniform processing schemes. The paper is easy to follow.

**S2**. The incorporation of the Rayleigh Quotient for subgraph selection and the use of separate high-pass and low-pass spectral filters represent sensible design choices for capturing both general semantic patterns and anomaly-specific signals.

**S3**. The authors conduct extensive experiments across 10 diverse real-world datasets.

### Weaknesses
**W1**. While the paper combines several techniques, the individual components are largely incremental adaptations of well-established methods. The Rayleigh Quotient has been extensively used in prior GAD work [1,2,3], and spectral filtering with Chebyshev polynomials is standard practice [4], representing a modest modification rather than a fundamental innovation. The paper would benefit from more clearly articulating what specific technical innovations go beyond combining existing techniques and why this particular combination is uniquely effective.

**W2**. The theoretical analysis in Section 3.3 and Theorem 1 suffers from several critical limitations that undermine its relevance to the actual method: (i) The ASBM assumes Gaussian-distributed node features, which is unrealistic for most real-world datasets that contain heterogeneous, sparse, or even categorical features; (ii) The theoretical setup assumes perfect knowledge of node homophily patterns and proposes applying filters accordingly, whereas the actual method uses a data-driven gated fusion network that does not explicitly model or utilize such discrete pattern assignments; (iii) The proof relies on a linear classifier with frozen filtered features, but APF uses learnable polynomial filters, MLP encoders, and a trainable fusion mechanism, creating multiple layers of learnable transformations that are not captured by the theoretical model. These gaps make it unclear what practical insights the theory provides beyond general intuition that adaptive filtering could be beneficial.

**W3**. Comparing Table 1 with the original GADBench paper [4] reveals substantial discrepancies that are difficult to reconcile. While some variation is expected across different hardware/software configurations, differences of this magnitude are concerning, particularly since the authors claim to use GADBench's implementation and evaluation protocol.

**W4**. The paper exclusively reports AUPRC, AUROC, and Rec@K, but omits F1 score—a standard and important metric for imbalanced classification that provides complementary information about precision-recall tradeoffs.

---

**Reference**

[1] J. Tang et al. *Rethinking Graph Neural Networks for Anomaly Detection*. ICML 2022. 

[2] Y. Gao et al. *Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum*. WWW 2023.

[3] X. Dong et al. *Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection*. ICLR 2024

[4] J. Tang et al. *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection*. NeurIPS 2023.

### Questions
**Q1**.What are the substantial discrepancies between your reported baseline results and those in the original GADBench paper? Please provide a detailed explanation.

**Q2**. F1 scores for all methods in the main experimental results should also be reported.

**Q3**. How does the theoretical analysis in Theorem 1 relate to the actual deployment of APF?

### Soundness
2

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
This paper introduces AFP, a GAD framework designed to address label scarcity and local homophily disparity through a two-stage design. In the pre-training stage, a Rayleigh Quotient and dual spectral filter-based module are trained to learn anomaly-sensitive information. In the fine-tuning stage, a fusion network and anomaly-aware regularization adapt to the test graph. Theoretical analyses are provided for separability. The proposed method shows competitive performance on the selected benchmark datasets. My view is that the proposed method is an improved strategy to take the advantages of both unsupervised and supervised training.

### Strengths
1. The proposed method and problem setting are interesting and well-motivated.
2. The paper provides theoretical analyses to justify the model design.
3. A comprehensive set of benchmark datasets is used, and the method achieves competitive performance across them.

### Weaknesses
1. While many baseline methods are included, they should be categorized by the type of anomaly information used (i.e., supervised, semi-supervised, or unsupervised) to clarify performance differences.
2. The paper explores a unique within-graph pre-training and fine-tuning setting. A discussion comparing it with other GAD settings using label information, such as meta-learning GAD (Meta-GDN, Ding et al., 2021), cross-domain GAD (Commander, Ding et al., 2021; ACT, Wang et al., 2023), and the more recent spectral paper (DSGAD, Zheng et al., 2025) would better position the contribution.
3. Most baselines are purely unsupervised or purely supervised. It would be helpful to discuss how combining supervised and unsupervised strategies (joint learning) would perform compared to the proposed method.
4. Since pre-training generally enables faster adaptation under distribution shift, could the authors comment on whether the proposed approach could extend to cross-dataset GAD or detecting novel types of anomalies as studied in (PreNet, Pang et al., 2022; GDN-AugAN, Zhou et al., 2023; and NSReg, Wang et al., 2025)?

### Questions
Please refer to my weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
