# Conformalized Predictions in Hypergraph Neural Networks via Contrastive Learning

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Hypergraph representation learning has gained immense popularity over the last few years due to its applications in real-world domains like social network analysis, recommendation systems, biological network modeling, and knowledge graphs. However, hypergraph neural networks (HGNNs) lack rigorous uncertainty estimates, which limits their deployment in critical applications where the reliability of predictions is crucial. To bridge this gap, we propose Contrastive Conformal HGNN (CCF-HGNN) that jointly accounts for aleatoric and epistemic uncertainties in hypergraph-based models for guaranteed and robust uncertainty estimates. CCF-HGNN accounts for epistemic uncertainty in HGNN predictions by producing a prediction set/interval that leverages the topological structure and provably contains the true label with a pre-defined coverage probability, while accounting for aleatoric uncertainty that originates from augmentations on the structure of the hypergraph. To enhance the power of the predictions, CCF-HGNN performs an additional auxiliary task of hyperedge degree prediction with an end-to-end differentiable sampling-based approach. Extensive experiments on real-world hypergraph datasets demonstrate the superiority of CCF-HGNN by improving the efficiency of prediction sets while maintaining valid coverage.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces two methods for conformal prediction on hypergraph neural networks (HGNNs): CCF-HGNN and CF-HGNN. As noted by the authors, CF-HGNN is a naive extension of CF-GNN (Huang et al., 2024) for hypergraphs, which they include as a baseline for CCF-HGNN. CCF-HGNN employs two additional mechanisms—contrastive augmentations and a loss and degree prediction head—to account for aleatoric uncertainty.

While CF-HGNN is not part of the authors' main contribution, I believe it is still a valuable contribution, as it gives another CP method for HGNNs that achieves coverage requirements and is likely more scalable (see Weaknesses).

### Strengths
- This paper deals with an important and relevant area, which is uncertainty quantification for HGNNs
- The paper accounts for **both** aleatoric and epistemic uncertainty, which is missing in prior (graph) CP literature
- Both methods achieve the desired coverage guarantees and outperform baselines

### Weaknesses
- One limitation of CF-GNN compared to other graph CP methods (e.g., DAPS, NAPS), it is a lot more computationally expensive (as you are training a separate conformal model). Could you discuss how CF-HGNN and CCF-HGNN scale
- Two of the four datasets used were for binary classification. While CP still works, it is less meaningful when the only set options are nothing, either a 0/1 label, or everything. It would be nice to have seen more multi-class datasets. There seem to be several datasets here (under Hypergraphs with labeled nodes): https://www.cs.cornell.edu/~arb/data/

### Questions
- In Assumption 1, do you mean to say permute calibration and test *nodes*, rather than *edges*? Should L165 end with be (..., $\mathcal{V} _ {\pi}$, 
$\mathcal{E} _ {\pi}$) rather than (..., 
$\mathcal{V}$, 
$\mathcal{V} _ {\pi}$)
- For your Bounded Coverage assumption for Theorem 1, do you have any examples of contrastive augmentations that would violate this assumption? It would be nice to include those in the manuscript
----
- Nit: Algorithm 1 has some abuse of notation ($\alpha$ is attention and significance level), which makes it a bit confusing

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper works on uncertainty quantification of hypergraph neural networks. The paper proposes a Contrastive Conformal
HGNN (CCF-HGNN) that jointly accounts for aleatoric and epistemic uncertainties in hypergraph-based models for guaranteed and robust uncertainty estimates. Compared with previous methods, it accounts for aleatoric uncertainty by leveraging contrastive learning on the structure of the hypergraph. The authors provide extensive experiments to show the effectiveness of the method.

### Strengths
1. The challenges of uncertainty quantification on hypergraph neural networks are clearly shown and explained. And the proposed method appropriately addresses all the listed challenges.
2. Theoretical and quantitative analyses are provided to show the rigorousness of the work.
3. The paper is well written and organized.

### Weaknesses
1. This paper argues that aleatoric uncertainty is not covered in the conformalized methods. This confuses me, as based on my understanding, the epistemic and aleatoric uncertainty have already been covered in the conformal prediction. Please explain why the CF-GNN or other conformal prediction methods do not cover aleatoric uncertainty.

2. The baselines in this paper are not enough. The authors argue that structural high-order information is considered to address the challenges in uncertainty quantification of hypergraph neural networks. Then, methods such as [1] should also be included as a competitive baseline that have already considered high-order structural information.

3. What is the backbone model used in the experiments? I understand that this paper focuses on the uncertainty of hypergraph neural networks. But will different hypergraph models have different influences on the results?

[1] Soroush H Zargarbashi, Simone Antonelli, and Aleksandar Bojchevski. 2023. Conformal prediction sets for graph neural networks. In the International Conference on Machine Learning. PMLR, 12292–12318.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles how to give reliable uncertainty for hypergraph neural networks. The authors propose CCF-HGNN, which combines two ideas: (1) conformal prediction (APS/RAPS) to guarantee that the true label is covered, plus a topology-aware, differentiable training trick to make the prediction sets as small as possible; and (2) contrastive learning with simple structural augmentations (dropping hyperedges or edges) so the model learns representations that are robust to noisy structure (handling data uncertainty). They also add a lightweight auxiliary task that predicts hyperedge degrees, using attention and differentiable top‑k selection to focus on the most informative hyperedges, which further sharpens the predictions.

### Strengths
1. The problem is important and timely: uncertainty quantification for HGNNs is underexplored yet crucial for reliable deployment.

2. The proposed method appears reliable, with sound design choices (conformal calibration, contrastive robustness, auxiliary structural task) and both theoretical and empirical support.

3. The paper clearly attributes techniques to prior work, allowing readers to trace components (e.g., conformal prediction, contrastive learning, attention, Gumbel-Softmax) to their sources.

### Weaknesses
The method is largely a composition of existing techniques, as the authors themselves acknowledge (combining aleatoric via contrastive augmentation and epistemic via conformal prediction). The conformal setup closely follows Huang et al. (2024), the contrastive augmentations draw from Wei et al. (2022), and the auxiliary degree prediction leverages standard attention and differentiable top-k sampling. While the integration is well-executed, the incremental novelty—particularly relative to Huang et al. (2024) on the conformal side—feels modest. I am less familiar with hypergraph-specific precedents, but within conformal prediction the step beyond prior art seems mild.

### Questions
In Table 2, APS coverage appears substantially above the target 0.95 across multiple datasets. While APS can be conservative in practice, the magnitude and consistency of the overshoot suggest a potential calibration or implementation issue.

### Soundness
2

### Presentation
3

### Contribution
2
