# PCA Feature Alignment is Sufficient for Building Graph Foundation Models

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Graph foundation models (GFMs) aim to pretrain graph neural networks (GNNs) that can generalize to new graph datasets in a zero-shot manner, requiring little or no additional training. This goal is challenging because graph data collected from diverse domains often exhibit significantly different node features and topological structures, and standard GNNs are sensitive to such variations. Prior efforts either restrict learning to a single domain (e.g., molecular graphs) or rely on heavy auxiliary pipelines that transform raw features into surrogates using large language models (LLMs) or feature-graph constructions. However, these approaches are often computationally expensive or restricted to specific domains.

In this work, we find that principal component analysis (PCA) is a simple, efficient feature alignment for GFMs. We show that PCA-aligned features satisfy two properties central to zero-shot GFM generalization:  (i) equivalence on identical datasets (identical datasets with only feature dimensions permuted or node order permuted always generalize to invariant or equivalent graph representations) and (ii) representation bounded on latently same datasets (non-identical datasets with same latent space always generate graph representations within bounded distance and prediction error). Building on this alignment method, we develop a Mini-GFM framework that is trained once across multiple datasets; at generalizing time, it only requires PCA alignment of the new dataset and optimization of a shallow, task-specific linear head. Across diverse node- and graph-classification benchmarks, this approach delivers competitive zero-shot performance compared with other baselines while using substantially lower preprocessing cost. These theoretical and empirical results validate the sufficiency of PCA alignment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the challenge of zero-shot generalization of Graph Foundation Models (GFMs) across datasets. Addressing issues such as heterogeneous node features, distinct topological structures of graph data from different domains, and the limitations of traditional methods, it proposes to use Principal Component Analysis (PCA) as a feature alignment tool and constructs the Mini-GFM framework. Through experiments on node classification and graph classification tasks across multiple domains, the paper verifies the framework's competitiveness in zero-shot scenarios while reducing preprocessing costs, providing a new approach for efficient generalization of GFMs.

### Strengths
1. Graph Foundation Models represent an important cutting-edge direction in the field of graph learning.

2. The experiments use 16 datasets for node classification and graph classification across different domains, including both small-scale standard datasets and the large-scale Amazon-2M dataset, showing a relatively rich selection of datasets.

### Weaknesses
- The paper proposes that when the original feature dimension D is smaller than the target alignment dimension D’, the feature is repeatedly concatenated until its dimension exceeds D’ for processing. **However, it fails to consider the information redundancy that may be introduced by repeated concatenation of high-dimensional features.**

- Although the paper demonstrates the advantage in training time on the Amazon-2M dataset, it does not analyze the computational complexity of PCA alignment in ultra-large-scale high-dimensional feature graphs (e.g., feature dimension D>10000, number of nodes exceeding 10 million). The core steps of PCA involve the calculation of the feature covariance matrix; when D is extremely large, the storage of the covariance matrix and eigenvalue decomposition will face memory and time bottlenecks. **Such high-dimensional graph data are common in practical industrial scenarios, and the paper does not explain how to solve this efficiency problem.**

- The paper assumes that datasets from different domains share a low-rank latent matrix **H** (Eq. 8), but does not verify the rationality of this assumption through experiments.

- Theorem 1 aims to prove the invariance of PCA to feature permutation. In the proof process, it mentions “**X’ =P_X·X**”, but according to the definition in the previous text, the feature permutation transformation should be “**X’ =X·P_X**” (Eq. 6), and there is **an error in the order of matrix multiplication here**. Additionally, there is a **citation error** in the proof (Line 618 of the paper).

- The paper does not analyze the generalization performance degradation pattern of Mini-GFM when the latent spaces of datasets differ significantly (e.g., molecular graphs and social network graphs), which makes me concerned about its practicality.

- The paper only sets two ablation baselines: "Repeat" and "Message", and does not conduct a sensitivity analysis on the key parameters of PCA alignment (e.g., the selection of target dimension).

- The paper only verifies node classification and graph classification tasks, and **does not involve common tasks in the field of graph learning such as link prediction and graph generation**.

- The experimental results are not marked with **standard deviations or confidence intervals**, making it impossible to judge the stability of the experimental results.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper argues for the use of PCA to align the feature space of different graphs, so that a single backbone model can be trained that will inductively generate embeddings for a new graph. These inductively generated embeddings then can be used directly for downstream tasks, such as node/graph classification/regression, and link prediction. This is overall, a novel idea.
The paper also presents a supporting theoretical argument for PCA being a good candidate for feature alignment. 
Some evaluation has been done on this idea in this paper, but I find the evaluation severely lacking.

### Strengths
- The idea of using simple PCA for inductive feature alignment across datasets is novel.
- The paper presents theoretical arguments to back the use of PCA.

### Weaknesses
- The paper classifies its presented method as "zero-shot" when it clearly needs labels to learn a classification/regression head. The term "inductive" is more standard to be used in this scenario. "Zero-shot" indicates a lack of need for labels to generalize to a new task.
- Handling scale invariance: The proposed method only handles the scale invariance of the type $SX$, but fails to mention the case $XS$. If this case is not realistic, a textual argument should be made as to why not. 
- Very few evaluations were done, and they do not include any standard deviation measure. For these problems, I would expect a paper to have a results table something like Table 5 in GraphAny.
- The paper could additionally be improved with an evaluation on the task of link prediction (though my other concerns are far more important than this one).
- The way of using bold text within table 3 is very different from what is standard. It does not recognize when a simple GNN beats the proposed method. Additionally, the fact that bold is not used for GNN is not mentioned anywhere. This will confuse readers into thinking the proposed methods are always better than a GNN.
- Overall, the paper spends too much space explaining classic concepts of GCNs and PCA, and too little space on doing actual evaluation. 

**Typos**:
- Line 053 - "presentations" -> "representations"
- Line 135 - "multi" should be "multiply"
- Linee 158 - "Node permutation equivalence" -> "Node permutation equivariance" (equivariance is the standard term to use here)

### Questions
- Normalization: Why form of normalization is used? It is not mentioned near line 201.

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
The authors focus on addressing the issue of inconsistent feature dimensions across different graph datasets in Graph Foundation Models (GFMs). They propose that Principal Component Analysis (PCA) is sufficient for aligning graph features. Based on this, they introduce a simple GFM method to achieve cross-dataset generality.

### Strengths
- Graph foundation models is a valuable research area.
- Investigating how to align features across different graph datasets is worthwhile.
- Restating the properties of the PCA holds some value.

### Weaknesses
- The theoretical content appears somewhat dated or previously established. In my view, the two properties of PCA emphasized in the paper ("Dimension Permutation Invariance" and "Node Permutation Equivalence") are already well-recognized in the academic community. As noted in S3, the paper's contribution is primarily a restatement of PCA's properties rather than a new proof.
- In my opinion, the theory presented does not adequately support the claim that "PCA Feature Alignment is Sufficient for Building Graph Foundation Models." The proposed theory relies on Assumption 3, which posits that different graph datasets share latent features. However, this assumption may not hold for graph datasets. For instance, some graph datasets have no feature, which clearly does not satisfy Equation 8.
- The experiments conducted in the paper seem insufficient and could benefit from greater rigor. For a graph foundation model, it would be appropriate to test the proposed method in few-shot scenarios. The current experimental setup is not typical for GFMs (e.g., the division of pre-training and testing graph data, as well as the data split ratios).
- The authors claim that PCA can align graph features. To further validate this view, I suggest supplementing experiments that transfer from citation datasets to molecular datasets (such as BBBP, Tox21, QM9, ZINC) to further substantiate this view. This would be particularly valuable in strengthening the argument.
- The proposed method lacks novelty. Existing works already employ PCA to unify features across different graph data, such as RiemannGFM [1], AnyGraph [2], All-In-One [3], and others.
- Existing work, such as FUG [4], has already analyzed the relationship between PCA and graph feature unification. This paper could benefit from discussing them.

[1] RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry, WWW-25.

[2] AnyGraph: Graph Foundation Model in the Wild, Arxiv-24

[3] All in One: Multi-task Prompting for Graph Neural Networks, KDD-23

[4] FUG: Feature-Universal Graph Contrastive Pre-training for Graphs with Diverse Node Features, NeurIPS-24

### Questions
Please see the weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper directly uses principal component analysis to align feature from different graphs and proposes the Mini-GFM framework. The method is validated across multiple datasets, demonstrating its effectiveness. Compared to existing methods, the proposed method is efficient and scalable for large graph data.

### Strengths
S1. The solved problem is valuable.

﻿
S2. The proposed method is validated on diverse datasets, including both node and graph classification tasks.

﻿
S3. The authors provide a detailed time complexity analysis.

### Weaknesses
W1. The proposed method is too simple and lacks novelty. Many existing works (e.g., MDGFM) have already used PCA to align features across different datasets. 

W2: The experimental evaluation lacks comparison with several state-of-the-art graph foundation models, such as MDGPT and MDGFM.

W3. The paper lacks few-shot learning experiments. Few-shot generalization is an important capability for graph foundation models.

### Questions
Q1. Why does the proposed method outperform the original fully trained GNN on some datasets (e.g., KarateClub)? 

Q2. How does the proposed method perform when pre-trained on only a single dataset? Can the proposed method truly acquire extensive graph knowledge through multi-domain pre-training?

### Soundness
2

### Presentation
2

### Contribution
1
