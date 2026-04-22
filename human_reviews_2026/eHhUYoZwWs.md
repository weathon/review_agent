# AdaSpec: Adaptive Spectrum for Enhanced Node Distinguishability

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Spectral Graph Neural Networks (GNNs) achieve strong performance in node classification, yet their node distinguishability remains poorly understood. We analyze how graph matrices and node features jointly influence node distinguishability. Further, we derive a theoretical lower bound on the number of distinguishable nodes, which is governed by two key factors: distinct eigenvalues in the graph matrix and nonzero frequency components of node features in the eigenbasis. Based on these insights, we propose AdaSpec, an adaptive graph matrix generation module that enhances node distinguishability of spectral GNNs without increasing the order of computational complexity. We prove that AdaSpec preserves permutation equivariance, ensuring that reordering the graph nodes results in a corresponding reordering of the node embeddings. Experiments across eighteen benchmark datasets validate AdaSpec's effectiveness in improving node distinguishability of spectral GNNs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes to enhance spectral GNNs by improving their node distinguishability. Namely it starts by observing that node distinguishability is limited by two factors, the repeated eigenvalues in the graph adjacency matrix and the missing frequency components in the node features. Based on this, the paper proposes to modify the graph shift operator used in spectral GNNs with constructions that improve the two factors, and provides theory supporting this. Finally, experiments on many transductive datasets are performed by using the learned graph shift operator in various spectral GNNs. Results show improvements over baselines and show that the number of distinct eigenvalues is indeed increased on several datasets.

### Strengths
1. **Motivating objective.** The focus on improving node distinguishability in spectral GNNs is well-motivated and addresses a relevant limitation of existing approaches.

2. **Theoretical development.** The paper provides substantial theoretical analysis in support of the proposed method, offering formal insights into the factors influencing node distinguishability.

3. **Clear experimental framing.** The experiments are structured around clear research questions, making it easy to understand the intended empirical validation.

4. **Breadth of evaluation.** The method is evaluated on a range of standard transductive benchmark datasets, demonstrating applicability across multiple settings.

### Weaknesses
1. **Practical relevance of the theoretical bound.** The main conceptual contribution is an improved lower bound on the number of nodes that can be distinguished. However, it remains unclear how meaningful this bound is in practice, or whether it provides actionable guidance for model design or empirical performance.

2. **Clarity of assumptions.** Some theoretical results (e.g., Theorem 5.2) rely on assumptions that are either not fully stated or not clearly motivated. It would be helpful to explicitly articulate these assumptions and discuss their necessity and scope.

3. **Novelty claims may be overstated.** The statements “To the best of our knowledge, no existing work has systematically analyzed the interaction between the graph matrix and node features in determining node distinguishability in spectral GNNs” and “In this work, we demonstrate that node distinguishability is influenced by the eigenvalue multiplicity and the missing frequency components of node features in the eigenbasis of the graph matrix” appear stronger than warranted. Similar themes were examined in prior work, particularly [1]. The contribution would be clearer if the relationship to [1] were more explicitly discussed and the novelty claims were calibrated accordingly.

4. **Readability and exposition.** The paper is difficult to follow in several places. A clearer introduction to the problem setting, intuition for the theoretical results, and more accessible mathematical presentation would greatly improve readability.

5. **[Minor]Limited significance of empirical results.** The experimental findings are not particularly strong: in Table 2, 25 out of 40 comparisons are not statistically significant, and in Table 3, 27 out of 35 are not significant. That said, this is standard in the field. 

Reference:
[1] Wang, X., & Zhang, M. (2022). How powerful are spectral graph neural networks? In ICML (pp. 23341–23362). PMLR.

### Questions
1. Fig 1 a) The paper states that a spectral GNN cannot distinguish nodes 1 and 3. Should a GNN be able to distinguish these? It seems like nodes 1 and 3 are isomorphic as the node features is [1, 0, 1, -1, -1].
2. Proof of theorem 5.1: “More unique coefficients in characteristic polynomial implies more unique eigenvalues of the matrix.” Can you clarify what you mean by this? Are you claiming that more different coefficients imply more roots to a polynomial? A counterexample to this claim is: (x-1)^3 =x^3-3x^3+3x-1 has 4 distinct coefficients but a single root and (x-1)(x-2)(x-3)=(x^2-3x+2)(x-3) = x^3 -6x^2 +11x -6 has three distinct roots but three distinct coefficients.
3. Theorem 5.2 relies on Theorem A.1 which require C to have no repeated eigenvalues, hence the statement of Theorem 5.2 is wrong/misleading since it seems like it applies to any matrix C. Particularly, since the motivation is that the adjacency has repeated eigenvalues, and a modification of A will play the role of C makes me question the usefulness of this statement in the context of the paper.
4. Table 4: Please add the base performance of ChebNet.
5. Table 5: please add the statistics for all datasets, not necessarily in the main text.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This is a good paper that mainly investigates how distinct eigenvalues and missing frequency components affect node distinguishability. The paper proposes AdaSpec, which includes three modules: INCREASE DISTINCT EIGENVALUES, SHIFTS EIGENVALUES FROM ZERO, and INCREASE FREQUENCY COMPONENTS. The proposed method is supported by solid theoretical proofs, and extensive experiments demonstrate the effectiveness of AdaSpec.

### Strengths
1.The paper studies how the graph matrix and node features jointly influence node distinguishability, which is an interesting direction.

2.The paper provides solid theoretical proofs to support the proposed method.

3.The time complexity analysis and comprehensive experiments further validate the effectiveness of AdaSpec.

### Weaknesses
1.In line 175, the paper states that “The presence of zero eigenvalues can hinder node distinguishability”, but I did not find any detailed explanation about this point. It should be the zero frequency components, not the zero eigenvalues, that hinder node distinguishability. The authors need to clarify the role of Section 5.2 in detail; otherwise, this section should be removed.

2.In lines 91–94, the paper mentions that eigenvalue correction does not ensure permutation invariance, but no further explanation is provided, leaving readers unclear about why this is the case.

3.In Figure 1, why can’t node 1 and node 3 be distinguished? Based on the current analysis, this conclusion does not seem directly supported.

4.Although the paper mainly studies node distinguishability, only Figure 1 involves this concept; both the method and experiment sections lack relevant discussion. The authors could conduct further analysis, for example, by relating their work to the Weisfeiler–Lehman (WL) test.

5.The two factors affecting node distinguishability mentioned in this paper — distinct eigenvalues and nonzero frequency components — have already been discussed in prior work, such as Wang & Zhang (2022). This weakens the contribution of the present paper.

### Questions
please see Weaknesses

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
4

### Summary
This paper investigates the problem of node distinguishability in spectral Graph Neural Networks (GNNs). The authors state that a spectral GNN's ability to distinguish nodes is theoretically lower-bounded by two main factors: the number of distinct eigenvalues of the graph matrix ($d_M$) and the number of non-zero frequency components of the node features in the matrix's eigenbasis. Based on this analysis, the paper proposes AdaSpec, a plug-in module that generates an adaptive graph matrix $\Omega(A,X)$ designed to maximize this lower bound. AdaSpec consists of three components: $\Omega_D(A)$ to increase distinct eigenvalues using a learnable diagonal matrix, $\Omega_S(A)$ to shift eigenvalues from zero, and $\Omega_F(X)$ to increase the number of non-zero frequency components by incorporating feature information. The authors provide theoretical guarantees that AdaSpec maintains permutation equivariance and empirically demonstrate its effectiveness at improving node classification, particularly on heterophilic datasets.

### Strengths
S1.  The paper tackles an important and specific problem—node distinguishability for spectral GNNs.

S2. The design of the AdaSpec module is well-motivated. Each of its three components ($\Omega_D$, $\Omega_S$, $\Omega_F$) is explicitly designed to address a specific limitation identified in the theoretical analysis.

S3. The experiments are extensive, covering 18 benchmark datasets with diverse characteristics (homophilic, heterophilic, large, and small). The ablation study in Table 4 validates the contribution of each component of AdaSpec.

### Weaknesses
W1. The most critical flaw is the failure to compare AdaSpec against other relevant graph augmentation or rewiring methods. The paper frames its contribution as a graph matrix generation module, which is functionally a form of learnable graph rewiring or augmentation. Although the authors claimed that they are the first to study node distinguishability, many other papers on spectral GNNs studied the expressive power of spectral GNNs, where node distinguishability was explicitly or implicitly studied. As we can see from the methodology and the experiments, AdaSpec is in fact a graph augmentation method for spectral GNN. Adding a comparison with other related augmentation methods in spectral GNN is necessary.

W2. The related work section acknowledges graph rewiring techniques (e.g., DropEdge, DiffWire, FoSR) but dismisses them as "fundamentally different", arguing they operate in the spatial domain. This distinction is not convincing. The goal is the same: modify the graph structure to improve GNN performance.AdaSpec also demonstrates its improved GNN performance, not in terms of node distinguishability. On the other hand, there are many methods that operate in the spectral domain.


W3. The experiments only compare spectral GNNs with AdaSpec to the same GNNs without it (i.e., using a fixed matrix). This demonstrates that some form of adaptation is better than none, but it fails to show that AdaSpec is superior to, or even competitive with, other existing augmentation/rewiring techniques. The observed performance gains might simply stem from adding any adaptive rewiring, rather than from the specific spectral motivations of AdaSpec.


W4.   The paper's central concept, "node distinguishability," is not formally defined until Section 4 (Definition 4.1). This is a major structural flaw. The term is used in the title, abstract, and throughout the entire introduction without a precise technical definition.

### Questions
Q1. To validate the paper's central claim, the authors must demonstrate that their spectrally-motivated adaptive matrix is more effective than other spatially-motivated or general-purpose adaptive matrices.

Q2. It is better to move the formal definition of node distinguishability (Definition 4.1) to the preliminaries (Section 3) or to provide a concise, informal definition in the introduction.


Q3. Could the authors elaborate on the unique necessity of the $\Omega_S(A)$ component given the presence of $\Omega_D(A)$? Does the learnable diagonal matrix $B$ in $\Omega_D(A)$ not already provide sufficient flexibility to shift eigenvalues, including the zero eigenvalues?
The ablation study (Table 4) shows that $\Omega_S(A)$ on its own has inconsistent and often poor performance (e.g., on Citeseer and Cora), suggesting it may be redundant or unnecessary.

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
3

### Summary
AdaSpec is an adaptive spectral module for GNNs that learns to modify a graph’s spectral structure (eigenvalues and frequency components) to make node representations more distinguishable. It generates an adjusted graph matrix that increases the number of distinct eigenvalues and enhances the frequency coverage of node features in the spectral domain. The approach is theoretically grounded, preserving permutation equivariance and providing provable guarantees on node distinguishability.

### Strengths
- AdaSpec provides a clear theoretical analysis linking graph spectra and node features to node distinguishability, and even derives a lower bound on how many nodes can be distinguished.
- Unlike prior spectral GNNs that use fixed graph matrices, AdaSpec learns to adjust the graph’s spectral structure (eigenvalues and frequency components), leading to improved representational power without extra computational cost.
- The method maintains a critical property for graph learning (permutation equivariance) ensuring that node reordering doesn’t change the model’s predictions, which preserves theoretical and practical soundness.
- The model is rigorously tested across 18 benchmark datasets, consistently improving node distinguishability and classification performance, showing that the theoretical ideas transform into real-world gains.

### Weaknesses
- While AdaSpec focuses on adaptive graph matrix generation, prior works like ARMA-GNN [1], SpecFormer [2] already explored adaptive spectral filtering or learnable spectral responses. AdaSpec’s contribution lies more in its theoretical framing (node distinguishability + eigenvalue diversity) than in introducing a fundamentally new mechanism.

[1] Graph Neural Networks with convolutional ARMA filters
[2] Specformer: Spectral Graph Neural Networks Meet Transformers

- Although AdaSpec theoretically explains how distinct eigenvalues improve distinguishability, it doesn’t provide much empirical interpretability (how the learned spectral modifications relate to graph structure or which frequencies become emphasized). This makes the adaptive mechanism somewhat of a black box.

- The paper claims “no increase in computational complexity,” but adaptively generating or modifying a graph matrix could introduce training instability or hidden overhead.
The supplementary doesn’t discuss runtime comparisons or scalability to very large graphs, areas where methods like GPR-GNN [3] are better optimized. What will be the performance of AdaSpec on the cases of very large graphs?

[3] Adaptive Universal Generalized PageRank Graph Neural Network

- Recent works such as SpecFormer achieve similar goals but with stronger end-to-end learnability and better scaling to large, dynamic graphs.

- Can AdaSpec handle temporal or evolving graph spectra, or is it limited to static adjacency matrices?

### Questions
See all the points in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
