# Refining Heuristic-Based Bitcoin Address Clustering with Graph Neural Networks

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Bitcoin’s pseudonymous nature makes it challenging to analyze user-level activity, since a single user may control multiple identifiers (addresses). Existing heuristic-based methods attempt to identify addresses belonging to the same user, but they often produce flat cluster assignments with limited modularity and are prone to errors such as merging different users together. In this work, we propose a method for refining heuristic-obtain clusters by grounding our clustering on contrastive embeddings yielded by graph neural networks . Our contribution is threefold: (i) we release a publicly available dataset of Bitcoin transaction graphs containing a substantial number of clusters; (ii) we propose a methodology for learning address embeddings consistent with heuristics, and back it up with solid theoretical foundations and empirical results; (iii) through hierarchical clustering, we allow a finer analysis of heuristic clusters and provide a quantitative criterion for flagging suspicious merges.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles Bitcoin address clustering by starting from standard heuristics that often over merge, then learning contrastive GNN embeddings designed to remain consistent with those heuristics.  Within each heuristic cluster it applies agglomerative hierarchical clustering and selects a "data-driven cut" to flag and split suspicious merges, yielding both a refined flat partition and a multi resolution view.   The authors also release a transaction graph dataset and provide theory and empirical evidence, while noting the absence of large scale ground truth user identities that would allow definitive validation.

### Strengths
1. Clear, practically important problem: refining over-merged Bitcoin heuristic clusters and exposing hierarchy.

2. Coherent method: contrastive GNN embeddings aligned with heuristics followed by hierarchical clustering with a data-driven cut; solid ablations and sensible diagnostics.

3. Interpretability and resources: hierarchical outputs aid analysis, and the released large transaction graph dataset increases reproducibility.

### Weaknesses
1. The empirical evaluation does not include recent SOTA baselines in Bitcoin clustering, so the significance of the reported improvements is unclear.

2. The supervision relies on heuristic clusters that may contain overmerges, which risks reinforcing existing errors rather than correcting them.

3. The agglomerative refinement step may not scale to very large clusters, and there is no runtime or memory analysis to assess its practicality.

4. Validation is based on intrinsic or heuristic-based metrics rather than externally verified labels, and results may be sensitive to the choice of linkage and cut threshold.

### Questions
1. **SOTA Baselines.** Please include head-to-head comparisons with recent SOTA pipelines for Bitcoin clustering and collapse prevention, for example Möser et al. (2022) [1], Schnoering et al. (2024) [2], Wang et al. (2024) [3] or similar recent methods. Use the same refinement pipeline and metrics for fairness.

2. **Ground truth.** Do you have externally verified labels or credible proxy labels to estimate false merges and false splits within heuristic clusters? Even a small audited subset would help calibrate the precision and recall of splits.

3. **Sensitivity.** Please report sensitivity to linkage type, distance metric, and the silhouette cut rule. Stratify by cluster size to show whether small vs large clusters require different settings.

[1] Moser, M., Narayanan, A., 2022. Resurrecting address clustering in Bitcoin. Financial Cryptography.

[2] Schnoering, H., Porthaux, P., Vazirgiannis, M., 2024. Assessing the efficacy of heuristic-based address clustering for Bitcoin. arXiv:2403.00523.

[3] Wang, X. et al., 2024. Exploring unconfirmed transactions for effective Bitcoin address clustering. The Web Conference.

> I find the paper promising, and I am open to raising my score if recent SOTA baselines and credible split-quality validation are added.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a two-stage methodology for refining heuristic-based Bitcoin address clusters, which are known to be flat and prone to erroneous merges. The work proposes first training a GNN using a contrastive loss to learn embeddings that are consistent with the initial heuristic clusters. In the second stage, agglomerative hierarchical clustering is applied to these embeddings within each heuristic cluster. By cutting the resulting dendrogram at a learned threshold, the method aims to identify and split these suspicious merges.

### Strengths
S1. The paper addresses a significant limitation of existing Bitcoin analysis methods.

S2. The release of a new, and publicly available dataset of Bitcoin transaction graphs is a contribution.

### Weaknesses
W1. The stated goal is to correct flawed heuristic clusters, specifically "cluster collapse”. However, the GNN is trained with a contrastive loss where positive pairs are drawn from the same heuristic cluster, and negative pairs from different clusters. This training objective is in direct opposition to the goal of finding and separating distinct user entities that were erroneously merged within that same cluster C_i. The paper fails to provide a convincing justification for why an embedding space trained to compress a cluster should be suitable for de-agglomerating it.

W2. The theoretical analysis in Section 4 appears disconnected from the paper's refinement objective. The theory provides no theoretical justification for the refinement step.

W3. The current baselines are generic clustering or representation learning methods. A proper comparison would be against other methods designed to solve the same problem: refining heuristics.

W4. It is recommended authors open-source the code to reproduce the results.

### Questions
Q1. For the Training Objective, could please elaborate on the core intuition of the methodology? The GNN is trained with a contrastive loss that explicitly pulls all nodes in a heuristic cluster C_i together. How does this training objective produce an embedding space that is suitable for the second stage, which aims to find and separate erroneously merged nodes within that same cluster C_i?

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
4

### Summary
The paper proposes to use graph neural networks with slightly modified loss function to compute embeddings in Bitcoin graphs to address clustering.

### Strengths
Evaluation of different Graph neural network approaches on address clustering in bitcoin.

The code and the data will be open sourced.

### Weaknesses
The proposed approach appears fairly straightforward, with limited novelty in the design of the loss function. Address clustering is performed by generating GNN-based embeddings, followed by the application of a standard clustering algorithm.

Also, the method does not seem scalable to real-world Bitcoin network data, where graph sizes are extremely large. This scalability limitation is likely one of the key reasons why GNN-based methods are rarely applied directly in practice for such blockchain analysis tasks anyway.

The theoretical results seem rehash of existing theorems.

### Questions
None.

### Soundness
2

### Presentation
3

### Contribution
1
