# Exchangeability of GNN Representations  with Applications to Graph Retrieval

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 6, 8, 4, 6

## Abstract
In this work, we discover a probabilistic symmetry, called as exchangeability in graph neural networks (GNNs). Specifically, we show that the trained node embedding computed using a large family of graph neural networks, learned under standard optimization tools,  are exchangeable random variables. This implies that the probability density of the node embeddings remains invariant with respect to a permutation applied on their dimension axis. This results in identical distribution across the elements of the graph representations.  Such a property enables approximation of transportation-based graph similarities by Euclidean similarities between order statistics. Leveraging this reduction, we propose a unified locality-sensitive hashing (LSH) framework that supports diverse relevance measures, including subgraph matching and graph edit distance. Experiments show that our method helps to do LSH more effectively than baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper claims to uncover an exchangeability property in GNN embeddings, suggesting that node embedding dimensions are probabilistically symmetric and invariant under permutation. Based on this assumption, the authors approximate transportation-based graph similarities using Euclidean distances between order statistics and build a unified LSH framework supporting multiple graph similarity measures. While the idea is conceptually interesting, the theoretical justification and empirical validation of the claimed exchangeability remain questionable and insufficiently supported.

### Strengths
- The writing is good.
- The quite novel angle is graph matching area.
- The performance is superior.

### Weaknesses
- The assumptions required to achieve exchangeability are overly restrictive and may not hold in realistic training scenarios.
- The experimental validation of exchangeability is weak.
- The choice of baselines is limited.

### Questions
- The introduction states that “the expected embedding matrix $\mathbb{E}[[x(u)]_{u\in V}]$ collapses to a rank-one matrix.” However, no empirical evidence (e.g., covariance analysis or SVD spectrum visualization) is provided to substantiate this claim.
- Real-world graphs typically vary in size. How does the proposed GraphHash framework handle graphs with different numbers of nodes, especially when a query graph contains more nodes than any graph seen during training?
- Do the padding nodes used to equalize graph sizes introduce additional noise or bias that may negatively affect retrieval performance?
- The conditions described in Sec. 3.1 for achieving exchangeability seem too strong. Many standard graph learning losses are not permutation invariant, and the optimizer’s equivariance depends critically on this property of the loss function.
- More SOTA works regarding graph match should be included, such as GNN-PE[1], IsoNet++[2]
- No hyperparameter sensitivity analysis.


- The proposed experiment for validating the exchangeability of embedding dimensions is conceptually flawed and cannot support the stated claim. The authors attempt to verify exchangeability by checking whether the marginal distributions of embedding elements are identical. However, this approach only measures the stability or randomness of parameter initialization and model convergence, not the permutation invariance or exchangeability of embedding dimensions within a single trained model.
In particular:
	- Exchangeability concerns the joint distribution of embedding dimensions within a model, not the marginal consistency of embeddings across independently trained models.
	- The observed similarity (or dissimilarity) of marginal distributions across models provides no evidence for or against permutation invariance in the learned representation space.
	- The experiment conflates stochastic training variation with structural symmetry in the embedding space.
	- No statistical test is performed on the joint dependency structure between embedding dimensions, which is essential for verifying exchangeability.


[1] Ye Y, Lian X, Chen M. Efficient exact subgraph matching via gnn-based path dominance embedding[J]. Proceedings of the VLDB Endowment, 2024, 17(7): 1628-1641.

[2] Ramachandran A, Raj V, Roy I, et al. Iteratively refined early interaction alignment for subgraph matching based graph retrieval[J]. Advances in Neural Information Processing Systems, 2024, 37: 77593-77629.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces and proves a probabilistic symmetry in trained GNN node embeddings: under standard i.i.d. layer-wise initializations and across a broad class of architectures, the embedding dimensions behave as exchangeable random variables. Experiments validate this claim by showing near-identical per-dimension marginals across massive independently trained models. Leveraging this insight, the authors propose new graph retrieval methods based on order-statistic features, achieving competitive performance compared to prior work.

### Strengths
- The paper introduces an interesting perspective for understanding GNN representations and proposes a series of sensible applications that leverage this new finding.
- The reduction from permutation-equivariant training dynamics to exchangeability of embeddings is clear and broadly applicable under the paper’s assumptions.
-  The transition from transport similarity to Fourier features and ultimately to random-hyperplane LSH is elegant and well-justified, making the paper logically coherent throughout.
- Training with many random seeds and demonstrating that per-dimension marginals remain stable provides direct and compelling empirical support for the theoretical claim.

### Weaknesses
- The guarantees rely on permutation-invariant losses. Common heads (e.g., linear classifiers, feature-wise norms) can break this unless parameters are aligned under permutations; the practical scope should be specified more sharply.

- While the embedding dimensions are exchangeable, this does not imply independence—identical marginals do not preclude cross-coordinate dependencies. The proposed 1-D order-statistic surrogate could ignore the joint structure of the embeddings. Without quantifying inter-dimensional dependence (e.g., via covariance or effective rank) or demonstrating robustness to such dependencies, it remains unclear when this 1-D compression faithfully preserves information critical for retrieval.

### Questions
How do retrieval quality and the Proposition-7 approximation error scale with $D$? Can you report MAP vs $D$ and discuss the smallest $D$ where the 1-D surrogate is reliable?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper formalizes *embedding-dimension exchangeability* for GNNs under specific conditions (i.i.d. initialization, permutation-invariant loss, and equivariant optimization) and leverages this property to approximate high-dimensional transport-based graph similarity via per-dimension sorted one-dimensional matching.  

This leads to the **GraphHash** framework, which applies Fourier feature mapping and LSH for efficient graph retrieval.  

Experiments on small TU-style molecular datasets (PTC-FM/FR/MR, COX2) show favorable accuracy–efficiency trade-offs compared with several baselines.

### Strengths
1. **Novel theoretical formulation — Exchangeability of GNN embeddings.**  
   The paper introduces the concept of *embedding-dimension exchangeability* in GNNs, showing that under mild assumptions, different embedding dimensions can be treated as exchangeable random variables.  
   This extends classical symmetry analyses (e.g., permutation invariance) into the probabilistic domain and provides a new theoretical lens for understanding GNN representations.

2. **Principled link between theory and computational efficiency.**  
   Using the exchangeability property, the authors reduce the high-dimensional optimal transport (OT) distance to a sum of one-dimensional sorted distances.  
   This theoretically justified simplification avoids solving full OT, achieving a significant computational reduction.

3. **Unified and efficient retrieval framework.**  
   The proposed **GraphHash** system combines this OT-based distance with locality-sensitive hashing (LSH), providing a unified, efficient, and theoretically grounded approach to graph retrieval.

4. **Strong empirical performance.**  
   On four TU benchmark datasets (PTC-FR/FM/MR, COX2), GraphHash consistently outperforms several strong baselines (FourierHashNet, DiskANN, IVF) in both *Subgraph Matching (SM)* and *Graph Edit Distance (GED)* tasks, achieving higher MAP and NDCG scores.

5. **Good scalability and trade-off behavior.**  
   The method achieves a favorable accuracy–efficiency balance, maintaining more than 50% of exhaustive MAP performance with much lower retrieval cost.

6. **Reproducibility and clarity.**  
   The theoretical derivations are detailed, and the authors provide code, data splits, and reproducibility documentation.

### Weaknesses
## **Weaknesses and Limitations**

1. **Idealized theoretical assumptions.**  
   The exchangeability theorem relies on assumptions rarely satisfied in modern GNNs. Components such as normalization layers (BatchNorm, LayerNorm), multi-head attention, temperature scaling, dropout, and adaptive optimizers can break permutation equivariance. The paper does not analyze which architectural choices preserve or violate exchangeability, limiting the generality of the theorem.

2. **Zero-padding heuristic lacks theoretical justification.**  
   When graphs have different node counts, the smaller graph is zero-padded before per-dimension sorting and matching. This can distort the node-value distribution and create spurious matches, especially for graphs with large node-count disparities. No theoretical bound or comparison with **Unbalanced/Partial OT** is provided, leaving the robustness of this approximation unclear.

3. **Weak statistical validation of exchangeability.**  
   The empirical evidence mainly consists of qualitative plots. There are no formal statistical tests (e.g., KS, MMD, or energy distance) or analyses of inter-dimensional dependence (correlation, mutual information). It remains unclear whether dimensions are merely identically distributed or also weakly independent—an important distinction for the proposed matching approach.

4. **Limited task scope despite a broadly applicable distance.**  
   While related works also focus on retrieval, the proposed transport-based distance could naturally extend to other tasks (e.g., k-NN classification, clustering, or graph alignment). Evaluating at least one additional downstream task would strengthen claims of broader applicability.

5. **Missing comparisons with the strongest recent methods.**  
   The experiments lack comparisons with several contemporary state-of-the-art systems, such as **IsoNet++**, **CORGII**, and recent **sliced-Wasserstein/set-hashing** approaches like **SLoSH** and **SWWL**. Without these baselines, it is difficult to assess how GraphHash performs relative to the current best.

6. **Small embedding dimensionality in the chosen datasets.**  
   All experiments use low-dimensional (≈10D) node embeddings on small molecular graphs. The exchangeability assumption and 1D matching approximation may not hold as well in higher-dimensional spaces where inter-dimensional dependencies are richer. A sensitivity analysis with respect to embedding dimensionality would help assess generality.

### Questions
1. **Zero-padding and simplification of the matching process**  
   The current approach simplifies node alignment by zero-padding smaller graphs before per-dimension sorting and matching. Could the authors provide more discussion or ablation on this choice? It seems that this step simplifies the transport computation considerably, but it is unclear how much accuracy or theoretical rigor is sacrificed as a result. A deeper justification or empirical analysis would make the approach more convincing.

2. **Extension to other downstream tasks**  
   While the paper focuses on graph retrieval, the proposed transport-based similarity measure appears more general. Would it be possible to show at least one additional downstream task (e.g., k-NN classification, clustering, or graph alignment) to demonstrate broader applicability and potential beyond retrieval?

3. **Dataset scale and comparison completeness**  
   The current datasets (PTC-FR/FM/MR, COX2) are relatively small in both node count and embedding dimensionality. Could the authors evaluate the method on larger or more complex datasets to test its scalability? In addition, many strong contemporary baselines (e.g., IsoNet++, CORGII, SLoSH, SWWL) are missing. Including such comparisons would help non-specialist reviewers better assess the actual competitiveness and validity of the proposed approach.

4. **Optional clarifications related to weaknesses**  
   Some of the weaknesses mentioned (e.g., idealized assumptions, limited statistical validation of exchangeability) might be addressed by additional experiments or discussion. The authors may selectively respond to these points if relevant.

---

**General comment:**  
If the authors can convincingly address these concerns—especially providing more insight into the zero-padding mechanism, exploring at least one additional task, and extending experiments to larger or more challenging datasets—I would be happy to raise my score. A simple but effective idea, if rigorously justified, can indeed make for a strong and impactful contribution.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a novel approach, GraphHash, addressing the graph retrieval problem. It demonstrates that the node embedding dimensions for a wide class of GNN methods are invariant under permutation operations applied along the dimension axis. This property is referred to as exchangeability, and the authors design a locality-sensitive hashing (LSH) framework by leveraging this property in order to efficiently approximate transportation-based graph similarities. The proposed method is supported by theoretical analysis and evaluated on four benchmark datasets for the graph retrieval task.

### Strengths
- The paper is well-structured and clearly presented, so the technical content is easy to follow.
- It provides a rigorous theoretical analysis supporting the proposed framework and its underlying assumptions.
- It includes an experimental evaluation demonstrating the effectiveness of the method across multiple datasets.

### Weaknesses
- The experimental evaluation is limited to relatively small benchmark datasets and lacks large-scale or real-world scenarios.
- The paper focuses primarily on retrieval performance but does not examine the memory and computational requirements.

### Questions
- Does the method require a fixed number of nodes per graph? Since the hash function depends on sorting node embeddings, how is hashing handled if graphs undergo updates that change the embedding order?
- What is the computational and space complexity of the proposed method? and how does it scale with graph size and embedding dimensionality compared to existing approaches?
- Based on Proposition 7, how does varying $D$ influence the performance? Similarly, what is the impact of $dim_h$ and $dim_T$ for different values?
- Since the hashing relies on random Fourier features and random hyperplanes, how stable is the performance across different seeds? Could the authors report variance for different runs?
- Could the authors elaborate on how sensitive the exchangeability property is to violations of the assumptions, such as non-i.i.d. initialization?
- It would be helpful to include the exact (non-approximate) transportation-based similarity in the comparison to better quantify the accuracy loss.
- In Figure 2, it is difficult to distinguish the performances clearly due to overlapping points. It would be better to summarize retrieval performance using a single aggregate score, such as area under the curve or a related metric.

**Additional comments:**
- In Line 163, it could be nice to mention that $\Psi$ and $ \Theta$ indicate the weight matrices in order to avoid the confusion that $\Psi$ is an input feature. 
- Line 1034, “invovled” -> involved

### Soundness
4

### Presentation
4

### Contribution
3
