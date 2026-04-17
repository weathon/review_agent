# Learnable Kernel Density Estimation for Graphs

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
This work proposes a framework LGKDE that learns kernel density estimation for graphs. The key challenge in graph density estimation lies in effectively capturing both structural patterns and semantic variations while maintaining theoretical guarantees. Combining graph kernels and kernel density estimation (KDE) is a standard approach to graph density estimation, but has unsatisfactory performance due to the handcrafted and fixed features of kernels. Our method LGKDE leverages graph neural networks to represent each graph as a discrete distribution and utilizes maximum mean discrepancy to learn the graph metric for multi-scale KDE, where all parameters are learned by maximizing the density of graphs relative to the density of their well-designed perturbed counterparts. The perturbations are conducted on both node features and graph spectra, which helps better characterize the boundary of normal density regions. Theoretically, we establish consistency and convergence guarantees for LGKDE, including bounds on the mean integrated squared error, robustness, and generalization. We validate LGKDE by demonstrating its effectiveness in recovering the underlying density of synthetic graph distributions and applying it to graph anomaly detection across diverse benchmark datasets. Extensive empirical evaluation shows that LGKDE demonstrates superior performance compared to state-of-the-art baselines on most benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LGKDE, a learnable kernel density estimation framework for graphs that integrates GNN-based representations, a deep MMD metric, and a multi-scale KDE with learnable mixture weights. The method contrasts normal graphs with their structure-aware perturbed counterparts to learn a density function capable of modeling complex graph distributions. The authors provide theoretical guarantees on consistency, convergence, and robustness, and demonstrate empirical superiority on twelve benchmark graph anomaly detection datasets. The study aims to unify deep representation learning and nonparametric density estimation within a principled framework.

### Strengths
1. Clear problem setting and motivation. The paper tackles a fundamental yet underexplored problem of graph density estimation, framing it in a principled manner that naturally connects to graph-level anomaly detection.

2. Integration of theoretical and empirical perspectives. The authors not only design a learnable KDE model but also provide non-trivial theoretical analyses (consistency, convergence rate, and robustness bounds), lending rigor to the framework.


3. Conceptually coherent design. The use of a deep MMD metric space and structure-aware perturbations offers a reasonable and interpretable approach to contrastive density learning, aligning well with the paper’s stated objectives.

### Weaknesses
1. Motivation and Empirical Support

The paper claims that existing graph learning methods trained with standard supervised objectives tend to capture spurious signals, making them fragile under distributional shifts. However, this statement lacks systematic evidence or quantitative validation. The only supporting illustration is a qualitative t-SNE visualization, which is insufficient to substantiate the general claim. A dedicated experiment or section demonstrating the degradation of standard methods under controlled distributional shifts or noise perturbations would make the motivation more convincing.

2. Methodological Assumptions and Theoretical Validity

   The theoretical derivation of LGKDE relies on a strong assumption that the intrinsic dimensionality ( d_{\text{int}} = 1 ), simplifying the density estimation formulation and constants. Furthermore, the convergence and robustness proofs implicitly assume that the graph space admits a smooth Riemannian structure that allows Taylor expansion, which may not hold for heterogeneous or discrete graph distributions. These assumptions weaken the generality of the claimed theoretical guarantees.

3. Incomplete Baseline Coverage

   Although the paper compares LGKDE with various graph anomaly detection (GAD) models and a two-stage GAE+KDE pipeline, it omits several directly related density-based approaches. In particular, there is no comparison with Sun & Fan (2024), which also employs an MMD-based metric learning framework, or with graph normalizing flow and energy-based models that perform explicit density estimation. Including these baselines would clarify whether LGKDE provides genuine methodological advancement beyond existing learnable density estimators.

4. Insufficient Ablation on Perturbation Module

   The proposed structure-aware perturbation mechanism is central to the model’s objective, yet its role is not rigorously dissected. The ablation studies only vary the number of KDE bandwidths and mixture weights, but do not test variants using (a) feature-swap only, (b) spectral perturbation only, or (c) random edge perturbation. Since the perturbation strategy directly drives the density contrastive learning, such ablations are crucial to justify its necessity and effectiveness.

### Questions
1. Could authors provide quantitative experiments showing how existing baselines (e.g., OCGIN, SIGNET) fail under controlled distribution shifts, to empirically support your motivation?
2. How realistic is the assumption ( d_{\text{int}} = 1 )? Have authors tested whether relaxing this assumption (e.g., learned or dataset-dependent intrinsic dimension) affects performance or theoretical consistency?
3. Why were Sun & Fan (2024) or other explicit density estimation methods (graph flows, EBMs) excluded from the comparison? Are there implementation or conceptual reasons?
4. Can authors report additional ablations on the perturbation module—particularly feature-only, spectral-only, and random perturbations—to show which component contributes most to the performance gain?

### Soundness
2

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
3

### Summary
The paper proposes LGKDE, a graph-level density estimation framework that learns graph embeddings with a GNN, defines a multi-scale KDE in an MMD-based metric space, and trains by contrasting each graph’s density against carefully designed structure- and spectrum-aware perturbations. The authors claim consistency, convergence, robustness, and generalization guarantees, and evaluate LGKDE mainly on graph anomaly detection benchmarks, reporting improvements over prior methods.

### Strengths
1. The paper formulates graph-level density estimation through a theoretically grounded framework that integrates graph neural network embeddings with learnable multi-scale kernel density estimation in an MMD-based space.
2. The authors provide L1-consistency and convergence rate results (Theorems 4.1 and 4.2), establishing statistical soundness and connecting the method to nonparametric theory under intrinsic dimension assumptions.
3. The framework includes structure- and spectrum-aware perturbations for density contrast, a clear complexity analysis with sub-sampling for scalability, and empirical validation on graph anomaly detection benchmarks showing steady gains over prior methods.

### Weaknesses
1. The distinction between LGKDE and deep density estimation methods is not sharply articulated, leaving unclear where LGKDE provides a fundamental advantage.
2. It is not clear how the learned MMD metric is constrained to prevent overfitting of the density landscape (e.g., through regularization of kernel parameters or Rademacher-style control), and how this constraint is reflected in the stated generalization bound.
3. More remarks or further insights are needed to help the readers better understand the theorems.

### Questions
1. What practical procedure do you recommend for bandwidth selection at multiple scales, and how sensitive are results to bandwidth mis-specification?
2. Can you quantify how spectral perturbations change graphs in spectral energy vs. topological terms, and correlate that with performance gains?

### Soundness
3

### Presentation
3

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
The paper proposes LGKDE, a principled framework for graph-level density estimation, aiming to unify deep graph representation learning with adaptive kernel density estimation. LGKDE represents each graph as a distribution over learned node embeddings and measures pairwise similarity via a deep Maximum Mean Discrepancy (MMD) metric. Density is then estimated with a multi-scale kernel density estimator whose weights are learned. A novel density contrasting loss maximizes densities of normal graphs relative to structure-aware perturbed counterparts, with perturbations applied to both node features and graph spectra.

### Strengths
1. The paper is clearly written, with a clear problem statement and a logically organized presentation.
2. The paper focuses on a relatively unexplored but important subarea of nonparametric graph density estimation, which directly underpins graph-level anomaly detection.
3. Synthetic validations recover known distributions, and broad benchmarks for anomaly detection show consistent AUROC, AUPRC, and FPR95 improvements over competitive baselines.

### Weaknesses
1. While complexity analysis is given, the framework requires pairwise deep MMD computation for KDE and generation of multiple perturbed samples, which might be costly for very large datasets. Empirical runtime/memory profiles for large-scale sparse graphs are lacking.
2. Perturbed samples are not true anomalies. The performance gain depends on the quality of the perturbations, and it is unclear how LGKDE would perform when the perturbations poorly reflect anomalous structures.

### Questions
1. How does LGKDE scale in practice when N is very large or graphs have high average degree? Could you provide empirical runtime and memory usage breakdowns for large-scale settings?
2. Why were $\tau_{1}$ and $\tau_{2}$ fixed at 0.5 and 0.75, respectively? Did you explore alternative threshold values, and how sensitive are the results to changes in $\tau_{1}$​ and $\tau_{2}$​?
3. The deep MMD metric involves taking the supremum over $\Gamma_{\mathrm{emb}}$. For large $S = \|\Gamma_{\mathrm{emb}}\|$, this can be computationally expensive. Do you have an efficient approximation or kernel selection strategy to handle large $\Gamma_{\mathrm{emb}}$ without significantly increasing runtime?

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
This paper proposes a learnable kernel density estimation framework for graphs that integrates GNNs with maximum mean discrepancy based kernel learning. The method jointly learns graph representations and adaptive kernel bandwidths by contrasting densities of normal graphs with perturbed counterparts, where perturbations are applied to both node features and graph spectra. The paper provides theoretical guarantees for consistency, convergence, robustness, and generalization, and reports superior performance in graph-level anomaly detection across multiple benchmark datasets compared with existing methods.

### Strengths
The paper tackles an important and underexplored problem which is relevant to applications such as anomaly detection and molecular graph analysis. The proposed approach is described clearly and supported by theoretical derivations. The experimental section is comprehensive, covering a wide range of datasets and showing that LGKDE consistently improves upon baseline methods. The presentation is coherent, and the proposed framework bridges traditional kernel-based methods and modern deep learning–based graph modeling.

### Weaknesses
The novelty of the contribution is limited, as the method mainly combines known components (GNN embeddings, MMD distances, and KDE) rather than introducing a fundamentally new concept. The proposed perturbation strategy and contrastive density objective are incremental variations on existing ideas in self-supervised learning and graph anomaly detection. The related work discussion is incomplete; several recent graph density and contrastive learning approaches are not adequately compared or discussed. The experimental comparison omits some strong baselines such as flow-based and diffusion-based graph generative models. Moreover, the analysis of the results is largely descriptive and does not provide convincing evidence explaining why the proposed model outperforms others beyond parameter tuning.

### Questions
How sensitive is LGKDE to the choice of bandwidth scales and perturbation strength? Can the authors provide ablation results showing whether the observed gains come from the learnable KDE component or simply from additional parameters in the GNN backbone?

### Soundness
3

### Presentation
3

### Contribution
3
