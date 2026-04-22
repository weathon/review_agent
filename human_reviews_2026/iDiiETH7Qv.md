# Sheaves Reloaded: A Direction Awakening

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Sheaf Neural Networks (SNNs) are a powerful algebraic-topology generalization of Graph Neural Networks (GNNs), and have been shown to significantly improve our ability to model complex relational data. While the GNN literature proved that incorporating directionality can substantially boost performance in many real-world applications, no SNNs approaches are known with such a capability. To address this limitation, we introduce the Directed Cellular Sheaf, a generalized cellular sheaf designed to explicitly account for edge orientations. Building on it, we define a corresponding sheaf Laplacian, the Directed Sheaf Laplacian $L^{\widetilde{\mathcal{F}}}$, which exploits the sheaf's structure to capture both the graph’s topology and its directions. $L^{\widetilde{\mathcal{F}}}$ serves as the backbone of the Directed Sheaf Neural Network (DSNN), the first SNN model to embed a directional bias into its architecture. Extensive experiments on twelve real-world benchmarks show that DSNN consistently outperforms many baseline methods. The source
code can be found at https://github.com/hakanaktas0/DSNN.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper extends Sheaf Neural Networks (SNNs) to directed graphs by introducing the Directed Cellular Sheaf and the corresponding Directed Sheaf Laplacian (DSL), which explicitly encode edge orientation through complex-valued, direction-aware restriction maps. Building on this framework, the authors propose the Directed Sheaf Neural Network (DSNN), enabling principled learning on directed and heterophilic graphs. Experiments on synthetic and real-world datasets demonstrate that DSNN effectively captures directional dependencies and outperforms existing SNN and GNN models where edge directionality is crucial.

### Strengths
1. This paper introduces a mathematically principled extension of SNNs to directed graphs through the Directed Cellular Sheaf and Directed Sheaf Laplacian.

2. It effectively captures asymmetric and directional relationships while maintaining robustness to heterophily.

3. The experiments demonstrate consistent performance gains on both synthetic and real-world directed graph datasets.

### Weaknesses
1.  Most compared models are from 2020–2022, with only one from 2024. The evaluation lacks more recent direction-aware or topology-based GNNs, which weakens the empirical evidence for DSNN’s claimed advantages.

2. The paper does not clearly justify why Sheaf Neural Networks are the right framework for addressing heterophily or extending to directed graphs. Given the recent shift toward graph foundation models and unified architectures, it remains unclear whether adapting SNNs is the most effective or timely direction, rather than developing more generalizable approaches.

### Questions
1. The introduction should better articulate why extending SNNs remains valuable in 2025. Specifically, it should discuss the advantages of combining SNNs with GNNs for handling directed graphs and heterophily, and explain why these structural refinements are still meaningful in the era of more unified and general graph learning paradigms.

2. The comparison set is mostly limited to models from 2020–2022, with only one 2024 method included. More recent GNNs addressing heterophily and oversmoothing (from 2024–2025) should be added to strengthen the empirical evaluation and demonstrate DSNN’s relevance against state-of-the-art models.

3. The paper does not include baselines from graph prompting, graph foundation, or pre-trained graph models, which have recently become dominant in node classification and general graph learning. Including such comparisons would position the work more clearly within the modern landscape of graph representation learning.

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
3

### Summary
The paper introduces the Directed Cellular Sheaf, which incorporates edge orientation into cellular sheafs. Based on the novel introduction, the paper proposes the Directed Sheaf Laplacian that serves as the backbone of the Directed Sheaf Neural Network. Experimental results validate the efficacy of the proposed GNN, with theoretical analysis on the properties of the Directed Cellular Sheaf.

### Strengths
- The topics of Sheaf Neural Networks and directed graphs are significant.
- The proposed Directed Cellular Sheaf admits satisfactory properties.
- The proposed Directed Sheaf Neural Network seems to work well in experiments.
- Complexity analysis is provided together with implementation code.

### Weaknesses
- It is not clear what $\tilde{F}^0$ means from Definition 1 without a clear reference or prior definition.
- The description of DSBM is not clear and without a reference in the main text. Should refer to [1].
- Some citation styles are not correct. For example, \citep should be used for the reference at the end of line 291.
- Some more works can be considered for comparison, e.g., [2] and [3].
-  The concept can benefit from some motivating examples of the Laplacian. Also, there is a figure sheaf.png in the anonymous github link that may help with illustration.

Reference:

[1] He, Y., Reinert, G., & Cucuringu, M. (2022, December). Digrac: Digraph clustering based on flow imbalance. In Learning on Graphs Conference (pp. 21-1). PMLR.
[2] Badea, T. A., & Dumitrescu, B. (2025). Haar-Laplacian for directed graphs. IEEE Transactions on Signal and Information Processing over Networks.
[3] Lin, L., & Gao, J. (2023, June). A magnetic framelet-based convolutional neural network for directed graphs. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.

### Questions
What does $\tilde{F}^0$ mean?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Directed Cellular Sheaves and a corresponding Directed Sheaf Laplacian, enabling Sheaf Neural Networks to incorporate edge directionality, a missing capability in current SNNs. The authors prove key spectral properties and show that this formulation recovers classical sheaf Laplacians and magnetic Laplacians as special cases. They further propose DSNN, demonstrating consistent gains on both real-world graphs and synthetic directional SBM settings.

### Strengths
**1. Principled directional sheaf formulation**

The paper introduces directed cellular sheaves and a corresponding directed sheaf Laplacian, providing the first rigorous sheaf-theoretic framework for directed graphs and addressing a clear limitation of existing SNNs.

**2. Solid theoretical foundation**

The authors prove Hermiticity, PSD spectrum bounds, and show that the proposed operator recovers classical sheaf Laplacians and magnetic Laplacians as special cases, demonstrating a sound and unifying mathematical design.

**3. Comprehensive experimental results**

Across both real-world and synthetic benchmarks, the model outperforms existing SNNs and competitive direction-aware GNNs, with especially strong results in heterophilic and direction-dominated settings, validating the benefits of directional sheaf modeling.

### Weaknesses
**1. Limited intuition for the directional mechanism**

While the mathematical construction is provided, the paper provides limited high-level insight into **how and why** the complex restriction maps enhance directional information flow in practice. The introduction of the complex phase feels algebraically motivated rather than guided by an intuitive model of directional propagation.

**2. Scope of experimental evaluation**

The evaluation focuses primarily on small-to-medium-scale datasets. There is no demonstration on larger real-world directed benchmarks (e.g., OGB-ArXiv, and arxiv-year). 

**3. Ablations could be deeper**

3.1. The effect of stalk dimension $d$

3.2. Sensitivity to direction sparsity or unreliable edge orientation (i.e., direction noise)

3.3. Effect of learning vs. fixing the phase $q$

**4. Writing clarity**

The definition and construction of the directed cellular sheaf are mathematically sound but presented in a dense, notation-heavy manner. Adding intuitive explanations, intermediate steps, and conceptual guidance (e.g., how complex phases encode directional flow at a high level) would make the framework more accessible and easier to follow for a broader audience beyond sheaf specialists.

### Questions
1. Is $q$ learned or tuned per dataset? If tuned, how stable is performance across $q$ values?

2. Does DSNN maintain benefits in settings where directional edges are sparse or only weakly informative?

3. How does performance degrade if a portion of edge directions are flipped or randomized?

4. Why the random split is adopted for the node classification instead of the widely-used public splits? Performance on small homophilic and heterophilic benchmarks can vary noticeably with random seeds, so it would be useful to justify this choice and clarify whether public splits are also tested.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Directed Cellular Sheaf and corresponding Directed Sheaf Laplacian (DSL) and  Directed Sheaf Neural Network (DSNN), extending Sheaf Neural Networks (SNNs) to directed graphs through complex-valued restriction maps. The authors prove that the DSL is Hermitian, positive-semidefinite, and upper-bounded by 2, and that it generalizes both the classical Sheaf Laplacian and the Magnetic/Sign-Magnetic Laplacians. Empirically, DSNN consistently outperforms GNN and SNN baselines on node-classification and direction-prediction tasks across 12 real-world and several synthetic datasets.

### Strengths
* Novelty and rigor: The formulation is original and mathematically principled. The theoretical results are sound and clearly proven.
* Unifying framework: DSNN subsumes NSD, MagNet, and SigMaNet as special cases.
* Empirical performance: DSNN variants achieve top performance on 10/12 node-classification and 8/10 direction-prediction benchmarks, with large margins on heterophilic and directed graphs.
* Writing quality: The paper is nice to read.

### Weaknesses
* Empirical scope: Experiments focus on small-to-medium graphs; scalability to large or temporal directed graphs remains untested.
* Ablations: Only sensitivity to q is analyzed. Additional studies on stalk dimension d or learned restriction-map architectures would strengthen the empirical section.
* Baselines: Comparison omits some recent direction-aware or heterophilic GNNs (e.g., DiGCL, DPGNN).
* Minor clarity issues: Dense notation and a few typos (“SNNs approaches”).
* Limited intuition: The geometric meaning of complex restriction maps and the global parameter q could be discussed more intuitively; why complex numbers versus real skew-symmetric forms?

### Questions
1. Could q be made learnable per edge or per graph?
2. Have you examined the spectral properties or phase distributions of learned complex restriction maps?
3. Could you comment on DSNN’s scalability?

### Soundness
3

### Presentation
3

### Contribution
3
