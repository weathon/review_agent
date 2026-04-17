# AEGIS: Authentic Edge Growth In Sparsity for Link Prediction in Edge-Sparse Bipartite Knowledge Graphs

- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Bipartite knowledge graphs in niche domains are typically data-poor and edge-sparse, which hinders link prediction. We introduce AEGIS (Authentic Edge Growth In Sparsity), an edge-only augmentation framework that resamples existing training edges—either uniformly simple or with inverse-degree bias degree\_aware—thereby preserving the original node set and sidestepping fabricated endpoints. To probe authenticity across regimes, we consider naturally sparse graphs (game design pattern’s game–pattern network) and induce sparsity in denser benchmarks (Amazon, MovieLens) via high-rate bond percolation. We evaluate augmentations on two complementary metrics: AUC-ROC (higher is better) and the Brier score (lower is better), using two-tailed paired $t$-tests against sparse baselines. On Amazon and MovieLens, copy-based AEGIS variants match the baseline while the semantic KNN augmentation is the only method that restores AUC and calibration; random and synthetic edges remain detrimental. On the text-rich GDP graph, semantic KNN achieves the largest AUC improvement and Brier score reduction, and simple also lowers the Brier score relative to the sparse control. These findings position authenticity-constrained resampling as a data-efficient strategy for sparse bipartite link prediction, with semantic augmentation providing an additional boost when informative node descriptions are available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper tackles the challenge of link prediction in edge-sparse bipartite knowledge graphs. Traditional graph augmentation methods—especially those involving random or synthetic edge generation—tend to distort structure and yield unreliable gains. The authors propose AEGIS (Authentic Edge Growth in Sparsity), a framework for authenticity-constrained edge-only augmentation. Rather than synthesizing new nodes or endpoints, AEGIS resamples existing edges within the training graph, preserving the node set and two-mode bipartite structure. Two policies are introduced: (i) AEGIS-Simple (uniform resampling of existing edges) and (ii) AEGIS-Degree (inverse-degree biased resampling to favor low-degree “cold-start” nodes). To benchmark performance under sparsity, the authors simulate extreme edge removal (bond percolation at 0.99 drop rate) on MovieLens and Amazon datasets, and evaluate on a naturally sparse Game Design Pattern (GDP) graph. Complementary metrics—AUC-ROC (ranking ability) and Brier score (probabilistic calibration)—are reported using paired t-tests. Five augmentation strategies are compared: AEGIS-Simple, AEGIS-Degree, random Erdos–Rényi additions, perturbation-based synthetic edges (SMOTE-style), and semantic-KNN completion (adding links between semantically similar nodes using cosine similarity). Results show that copy-based AEGIS variants maintain parity with sparse baselines, while semantic-KNN augmentation substantially improves both AUC and Brier. Random and synthetic edges generally degrade performance. The analysis highlights that semantic richness of node attributes governs augmentation success: GDP’s detailed textual descriptions yield the strongest calibration gains, whereas MovieLens’s minimal metadata offers little benefit.

### Strengths
1. The authors propose a well-defined and technically clear framework for the Rule-based augmentation setting over Sparse Single-relation Bipartite KGs. This makes the analysis straightforward to assess. 

2. The use of AUC and Brier are good evaluation criteria; this I commend the authors focusing on these instead of the usual metrics which provide an incomplete picture.

### Weaknesses
1. The experimental analysis is severely lacking. The authors experimented with the single 0.99 percolation pass, however, it is important to demonstrate what happens at different values such as – 0.95, 0.9, 0.8 and such. Furthemore, the choice of 1% seems arbitrary, without any citation or specific design choice. 

2. Similar to the above point, the choice of 100x upscaling/augmetation is also interesting, but there is no supporting experiment for the variation of this augmentation factor.  

3. A major concern is the lack of experimental details around the Hetero GAT (I am assuming the authors mean Graph Attention Network’s Heterogeneous variant here [2]) 

4. Some concerns about the benchmarks used. Especially, the size of the benchmarks in terms of the numbers of nodes in each of the partitions. Practically speaking, one would also want to see the behavior on larger sized bipartite graphs, akin to the link prediction literature for KGs [1]. Can the authors shed some light on the selection of the benchmarks, the GDP benchmark is especially small. 

5. The AEGIS-simple and AEGIS-degree methods are quite common in the literature ([3] as an example). Why did the authors then need to rename those? More specifically, of the first two claims on page 1 in the introduction, only claim 1 seems to be reasonable, albeit with the caveats mentioned above. 

[1] Sardina, J., Kelleher, J. D., & O’Sullivan, D. A Survey on Knowledge Graph Structure and Knowledge Graph Embeddings. arXiv preprint arXiv:2412.10092 (2024). 

[2] Heterogeneous Graph Attention Network Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Peng Cui, P. Yu, Yanfang Ye 

[3] Fitting the Linear Preferential Attachment Model Phyllis Wan1 , Tiandong Wang2 , Richard A. Davis1 , and Sidney I. Resnick2

### Questions
1. The major questions surround the experimental choice for the bond percolation value, the augmentation scale. 

2. Since the Hetero GAT experimental details are lacking in the paper, there are further questions around that. Especially, depending upon the number of GAT layers used, the effective aggregation radius around the nodes changes. That had a huge impact on the downstream performance, involving all the 5 upstream augmentation methods considered here. 

3. Can the authors provide some runtime analysis and degree distributions of the nodes in the benchmarks considered?

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
5

### Summary
This paper explores the impact of edge augmentation methods in the context of link prediction over bipartite graphs. To this end, it considers five different techniques aiming to augmented sparse edge information, each with its own variation (e.g., leveraging semantic node information, fully random, degree-biased, ...). The main empirical setting for this problem is a set of dense graphs (Amazon, MovieLens, GDP) that are sparsified, with the resulting link prediction performance following each augmentation strategy being compared against both the sparse baseline and the original graph. 

Given its experimental results, the paper offers a discussion on the significance and effectiveness of each of its 5 proposed techniques, highlighting that only semantically-driven augmentation was a meaningful improvement over the sparse baseline in this context.

### Strengths
- The setting is well-designed: Starting with a dense graph and working this into a sparse setting offers natural upper and lower bounds for performance, and sets the stage well for an analysis of augmentation techniques. 
- The intuitions behind the paper's methods are easy to follow.

### Weaknesses
- The paper makes no substantial contribution, as its main insights are highly predictable just from the conceptual empirical setting. Indeed, given a 99% sparsification and the introduction of such a large number of ultimately random edges, it is hard to imagine any other outcome than those reported in this paper. The authors mention exploring other sparsification settings as a future work. However, I think this is a necessary step towards obtaining any meaningful insights that may add more color to the different augmentation algorithm dynamics. At the moment, the extreme sparsification and randomization levels render any comparison hard to interpret. 

I suggest to the authors that they explore a more continuous augmentation gradient, showcasing how performance evolves as augmentation levels increase (fixing sparsification level), and then doing the opposite (changing sparsification level, i.e., how much of the original graph is preserved, given a fixed level of augmentation), to actually study the robustness and sensitivity of the 5 techniques. I also highly recommend emphasizing performance in more realistic settings (augmentation will usually be within the same order of magnitude as the number of existing edges to maintain a reasonable amount of signal to noise). 

- The paper lacks meaningful baselines to compare against its own approaches, and doesn't seem to bring any of the works from the related work section into its experimental analysis. More generally, the paper offers limited experimental insights and needs a substantial amount of additional work (including the above suggestions) with other baselines to be ready for publication. 

Overall, this work is insufficiently developed in my opinion, and is based in a sparsification setting that is not optimal for the link prediction and augmentation analyses being targeted. Moreover, it doesn't offer any conceptually novel approaches or techniques. As a result, I don't think this work should be published in its current form.

### Questions
N/A

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces AEGIS, an edge-only augmentation framework targeting extreme edge sparsity in single-relation bipartite graphs. AEGIS duplicates existing training edges (either uniformly — simple — or inverse-degree biased — degree-aware) to densify training supervision without creating new nodes. The authors compare AEGIS to three other augmentation families : **random ER-like, perturbation-synthetic (SMOTE-style), and semantic-KNN**.

Evaluation uses ROC-AUC and Brier score; significance is assessed with paired two-tailed t-tests and Cohen’s d. Main empirical claim: copy-style AEGIS matches sparse baselines, while semantic-KNN recovers AUC and calibration when node descriptions are informative

### Strengths
1. **Clear, reproducible algorithmic description**: The AEGIS variants and competing augmentation algorithms are specified precisely (pseudo-code, sampling weights, semantic-KNN algorithm).

2. **Insightful empirical observation**: Resampling (authentic) augmentation is a stable baseline; semantic similarity–based augmentation is required to meaningfully recover performance when textual metadata are informative.

3. **Paper well organized**

### Weaknesses
1. **GAT model tested**: performance could differ with other GNNs or matrix-factorization baselines.

2. **AEGIS shows weak gains**: 
- Often statistically indistinguishable from baseline on MovieLens & Amazon 
- Semantic KNN provides most improvements, overshadowing AEGIS resampling.

3. **Ambiguous effect of degree-aware resampling**: The inverse-degree (degree-aware) resampling is intended to help cold-start nodes, but results are mixed; e.g., GDP AUC decreases significantly for degree-aware (−0.028). The paper should investigate why degree-aware can hurt ranking while improving calibration.

### Questions
1. What would be results on big datasets like ogbl-wikikg2 or ogbl-biokg?
2. The experiments currently focus on one percolation case (q=0.01). How would they change for other values such as q = 0.05 or 0.1?
3. How will the results improve if you use other models like GraphSAGE, SEAL/GNN subgraph method, or KGE models (e.g., ComplEx or RotatE)?
4. Can you give some explanations for **Weaknesses** 3 and 4?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes AEGIS, a simple  framework for link prediction in edge-sparse bipartite graphs. Instead of generating synthetic edges, AEGIS augments training data by resampling authentic existing edges, with variants like degree-aware and semantic KNN sampling. Experiments show that it has some improvements in some cases.

### Strengths
1. The paper tackles an ealistic problem — link prediction under extreme edge sparsity in bipartite knowledge graphs, which is interesting.

2. The proposed idea of authentic edge augmentation is intuitive and easy to implement, avoiding noisy or unrealistic synthetic edges.

### Weaknesses
1. The paper is not well organized, and the motivation behind the proposed authentic edge–based augmentation is not clearly articulated.

2. The method description is limited to pseudo-code without sufficient explanation of design details or implementation. 

3. The proposed AEGIS variants perform very close to the baseline, raising questions about the method’s effectiveness. What's the definition of "baseline" in the experiment?

4. The contribution is relatively incremental — resampling existing edges is conceptually simple.

### Questions
please refer to the weakness

### Soundness
2

### Presentation
2

### Contribution
2
