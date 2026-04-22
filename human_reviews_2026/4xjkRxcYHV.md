# A Dual-View Contrastive Learning Framework for Heterogeneous Graph Representation Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Heterogeneous graph representation learning leverage the rich semantics and complex structural relationships within heterogeneous graphs. However, existing methods often fail to capture long-range semantic dependencies and localized structural patterns simultaneously. Therefore, we propose a novel Dual-View Contrastive Learning framework (DVCL) for heterogeneous graph representation learning. Specifically, the Graph Schema View Module (GSVM) is conducted to model the structural dependencies by leveraging a relational graph neural network with type-aware message passing and adaptive residual connections. Then, the Semantic Meta-Path Mamba Module (SMPMM) is designed to capture high-order semantic dependencies through a globally enhanced Mamba backbone, equipped with multi-resolution fusion and directional positional encodings. Moreover, a dynamic bidirectional contrastive learning is constructed to integrate the semantic view and structural view, treating each view as a learnable augmentation of the other to ensure robust and complementary representations. Extensive experiments on four datasets demonstrate that the proposed method consistently outperforms state-of-the-art methods, in terms of classification and clustering tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on heterogeneous graph representation learning, aiming to simultaneously capture long-range semantic dependencies and localized structural patterns, which are often overlooked by existing methods. The authors propose a novel framework named DVCL, which integrates a Graph Schema View Module (GSVM) for structural modeling via relational graph neural networks and a Semantic Meta-Path Mamba Module (SMPMM) for high-order semantic modeling using a Mamba-based sequence encoder. A bidirectional contrastive learning mechanism aligns these views to produce robust representations. Experimental results on four datasets show that DVCL outperforms state-of-the-art methods in node classification and clustering tasks.

### Strengths
1.	The use of Mamba architecture for meta-path sequences in heterogeneous graphs is innovative, as it addresses long-range dependency modeling with linear complexity.
2.	Comprehensive evaluations on four datasets provide strong empirical support, and the inclusion of ablation studies and the hyperparameter analysis validates component contributions.
3.	The paper is well-structured, with clear module descriptions and algorithm outlines. Figure 1 gives a very detailed overview of the proposed model.

### Weaknesses
1.	The hyperparameter analysis only examines embedding dimension $h$, temperature $\tau$, and loss weight $\lambda$, ignoring critical parameters like the number of Mamba layers.
2.	No theoretical or experimental analysis of convergence, complexity bounds, or robustness guarantees is included.
3.	The authors provide complexity analysis in Section 2.5, but the paper does not discuss practical scalability on very large graphs or compare with baselines empirically.
4. Although meta-path mamba module is novel to me, the motivation of using it should be clarified in introduction.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DVCL, a contrastive heterogeneous graph representation learning approach. DVCL leverages two views: one from the local heterogeneous neighborhood captured by a Graph Schema View Module (GSVM), another from high-order metapath-based dependencies captured by a Semantic Meta-Path Mamba Module (SMPMM). In experiments, DVCL outperforms recent supervised, generative, and contrastive methods on some representative heterogeneous graph datasets for node classification and node clustering.

### Strengths
1. Heterogeneous graph representation learning is a fundamental task that could benefit a wide spectrum of downstream applications.
2. This paper is easy to read. It is well-written and clearly organized overall.

### Weaknesses
1. The novelty of the proposed method is quite limited. There are already quite a few studies that have proposed contrastive methods for heterogeneous graph representation learning, particularly the schema view + metapath view strategy.
2. The technical quality of this paper is not strong enough. The key innovations and advantages of the proposed method are not clear enough. Also, some important tasks are missing, such as link prediction.

### Questions
Based on the review above and some other issues, the reviewers have the following questions or comments.
1. DVCL and HeCo are quite similar in that they both adopt a schema view + metapath view contrastive strategy. The main differences seem to be just the choices of the respective view encoders. Simply swapping the encoder modules to other neural architectures (proposed by other researchers) does not contribute much to the paper novelty. More elaboration on this is needed.
2. The literature review is missing some important related studies, including DMGI [1], PT-HGNN [2], CPT-HG [3], and HGCML [4]. The authors may need to include them and discuss how DVCL is different and why their proposed designs are better.
3. The experiments only cover two node-level tasks, which may not comprehensively reflect different methods' capabilities. Including link-level tasks (e.g., link prediction) or even graph-level tasks can make the experimental results more convincing.
4. There are quite a few typos or grammatical errors, including but not limited to
    * Page 1 Line 013: "leverage" => "leverages"
    * Page 1 Line 046: missing space in "… Liao et al. (2022).In order to …"
    * Page 2 Line 065: extra period in "Consequently,.multi-perspective …"
    * Page 2 Line 065: "helerogeneous" => "heterogeneous"
    * Page 2 Line 066: missing space in "relation-specificviews"
    * Page 2 Line 088: extra space in "… message passing , an SMPMM … "
    * Page 2 Line 095: "Deep View-aligned Contrastive Learning (DVCL)" => "Dual-View Contrastive Learning (DVCL)"
    * Page 4 Line 165: "stagel" => "stage"
    * Page 6 Line 306: "Datasets.we …" => "Datasets. We …"
    * Page 9 Line 444 & Page 17 Line 873 & Page 17 Line 914: "Clssifications" => "Classifications"

[1] Chanyoung Park, Donghyun Kim, Jiawei Han, Hwanjo Yu: Unsupervised Attributed Multiplex Network Embedding. AAAI 2020: 5371-5378

[2] Xunqiang Jiang, Tianrui Jia, Yuan Fang, Chuan Shi, Zhe Lin, Hui Wang: Pre-training on Large-Scale Heterogeneous Graph. KDD 2021: 756-766

[3] Xunqiang Jiang, Yuanfu Lu, Yuan Fang, Chuan Shi: Contrastive Pre-Training of GNNs on Heterogeneous Graphs. CIKM 2021: 803-812

[4] Zehong Wang, Qi Li, Donghua Yu, Xiaolong Han, Xiao-Zhi Gao, Shigen Shen: Heterogeneous Graph Contrastive Multi-view Learning. SDM 2023: 136-144

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the limitations of heterogeneous graph representation methods that struggle to capture both long-range semantic dependencies and localized structural patterns, this paper introduces a Dual-View Contrastive Learning (DVCL) framework that jointly models structure-aware and semantics-aware representations for heterogeneous graphs.

### Strengths
1. The dual-view contrastive framework effectively integrates structural and semantic modeling.
  
2. The module design is clear, and each component contributes significantly.

### Weaknesses
- Figure 1 is poorly drawn; the nodes are distorted, labels are inconsistent, and the layout is overcrowded.
  
- The improvement over baselines such as HGMS is minor, generally within two to three points, and some compared methods are outdated.
  
- Figure 3 lacks comparison with the strongest baseline HGMS, reducing the credibility of visualization.
  
- The method relies heavily on predefined meta-paths, limiting generalization to graphs without clear semantic paths and lacking automatic meta-path discovery.
  
- The experimental setup is limited, focusing only on node-level tasks and missing newer baselines and broader downstream evaluations.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
