# Lightweight and Interpretable Transformer via Unrolling of Mixed Graph Algorithms for Traffic Forecast

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Unlike conventional "black-box" transformers with classical self-attention mechanism, we build a lightweight and interpretable transformer-like neural net by unrolling a mixed-graph-based optimization algorithm to forecast traffic with spatial and temporal dimensions.
We construct two graphs: an undirected graph $\mathcal{G}^u$ capturing spatial correlations across geography, and a directed graph $\mathcal{G}^d$ capturing sequential relationships over time. 
We predict future samples of signal $\mathbf{x}$, assuming it is ``smooth'' with respect to both $\mathcal{G}^u$ and $\mathcal{G}^d$, where we design new $\ell_2$ and $\ell_1$-norm variational terms to quantify and promote signal smoothness (low-frequency reconstruction) on a directed graph.
We design an iterative algorithm based on alternating direction method of multipliers (ADMM), and unroll it into a feed-forward network for data-driven parameter learning. 
We insert graph learning modules for $\mathcal{G}^u$ and $\mathcal{G}^d$ that play the role of self-attention. 
Experiments show that our unrolled networks achieve competitive traffic forecast performance as state-of-the-art prediction schemes, while reducing parameter counts drastically.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a lightweight transformer-like architecture for traffic forecasting by unrolling a mixed-graph-based ADMM optimization algorithm.

### Strengths
- Addresses the important problem of reducing transformer parameters while maintaining performance
- Mathematical framework grounded in optimization theory provides some theoretical foundation
- Experiments across multiple traffic datasets (METR-LA, PEMS-BAY, PEMS03/04/07/08)
- Attempts to bridge model-based and data-driven approaches through algorithm unrolling
- Clear motivation for using both spatial and temporal graphs

### Weaknesses
- The novelty claims are exaggerated. Directed graph signal processing exists extensively in the literature. The statement "for the first time" regarding smooth signals on directed graphs is incorrect. References to prior work on directed graph Laplacians and spectral analysis are missing.

- The symmetrization in Theorem 3.1 undermines the main contribution. By converting Ld_r via (Ld_r)^T Ld_r, the method essentially reduces to undirected graph processing, contradicting claims about novel directed graph handling.

- Limited experimental superiority. Table 1 shows the proposed method is often outperformed by simpler baselines like STID and SimpleTM. The "competitive" framing hides that the method rarely achieves best performance.

- The interpretability claim is weak. Knowing layers correspond to ADMM iterations doesn't provide actionable insights. What traffic patterns do learned graphs capture? How do practitioners use this "interpretability"?

### Questions
1. Can you provide citations for prior work on directed graph Laplacians and signal processing? How does your DGLR/DGTV differ from existing directed graph smoothness definitions?

2. Why does symmetrizing via (Ld_r)^T Ld_r not reduce your approach to standard undirected graph methods? What specifically remains "directed" after this operation?

3. Table 1 shows your method is often outperformed by simpler baselines (STID, SimpleTM). Why should practitioners choose your approach over these simpler alternatives?

4. Can you provide comprehensive ablations showing: (a) undirected graph only, (b) directed graph only, (c) different numbers of unrolled layers, (d) DGLR vs DGTV?

5. What is the actual performance-parameter tradeoff? Can you plot accuracy vs. parameters for all methods to show where your approach sits?

6. How does "interpretability" help practitioners? Can you show examples of what the learned graphs reveal about traffic patterns that couldn't be discovered otherwise?

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
This paper introduces a lightweight and interpretable transformer-like architecture for spatio-temporal traffic forecasting by unrolling a mixed-graph optimization algorithm. The model represents data using an undirected graph for spatial correlations and a directed graph for temporal relationships. A key contribution is the design of novel L2-norm (DGLR) and L1-norm (DGTV) regularizers to quantify and promote signal smoothness on directed graphs. By unrolling an Alternating Direction Method of Multipliers (ADMM) solver for this optimization problem, the authors create an interpretable neural network where each layer corresponds to an optimization step and the graph learning modules function as a self-attention mechanism. Experiments show the model achieves competitive performance against state-of-the-art methods while using drastically fewer parameters.

### Strengths
1. The paper is built on a robust theoretical foundation, leveraging well-established concepts from graph signal processing (GSP) and optimization, particularly the unrolling of mixed-graph-based algorithms. This theoretical rigor provides a strong basis for the proposed model's design and effectiveness.

2. A major strength is the model's interpretability. Unlike "black-box" transformers, this architecture is a "white-box" by construction. Each layer of the neural network directly corresponds to an iteration of the ADMM optimization algorithm, making the model's internal operations mathematically transparent.

3. The approach demonstrates a significant reduction in model parameters, achieving performance comparable to state-of-the-art methods while using only a fraction of the parameters (6.4% of transformer-based PDFormer). This makes the model highly efficient, especially in memory-constrained environments.

### Weaknesses
1. While the proposed method demonstrates significant parameter reduction, the paper does not provide an in-depth analysis or comparison of the runtime efficiency. Given the lightweight nature of the model, including performance benchmarks related to computation time and memory usage would further highlight its advantages.

2. Although the model is interpretable and achieves competitive results, it slightly underperforms compared to some existing baseline methods.

3. The paper includes a substantial amount of mathematical derivation, which, while thorough, may make the content dense and difficult to follow for readers not well-versed in the specific theoretical background. A clearer, more concise explanation of the key steps and concepts would improve accessibility.

4. The scalability of the method to larger networks is uncertain, as it has only been tested on graphs of hundreds of nodes; specifically, the cost of conjugate gradient iterations and graph learning modules could become prohibitive.

### Questions
See weakness

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
This paper proposes a novel lightweight and interpretable Transformer-like architecture for traffic forecasting by unrolling a mixed-graph-based optimization algorithm. Specifically, it first learns an undirected graph $G_u$ and a directed graph $G_d$ to capture spatial and sequential information respectively, then designs three smoothness prior GLR for $G_u$, and DGLR, DGTV for $G_d$ to predict the future samples. An ADMM-based optimization algorithm is then unrolled into neural layers for end-to-end parameter and graph learning, where the graph-learning modules play a self-attention-like role with far fewer parameters Extensive experiments demonstrate its effectiveness in traffic forecast performance with fewer parameters.

### Strengths
S1. This paper creatively constructs a mixed graph to model spatial-temporal data. Meanwhile, it proposes a novel regularization terms directed graph Laplacian regularizer (DGLR) and directed graph total variation (DGTV) for directed graphs, which resolves the issue that the asymmetric Laplacian of a directed graph is difficult to analyze spectrally.

S2. This paper has a real small parameter count compared to state-of-the-art models, as shown in Table 1. Meanwhile, its prediction performance remains highly competitive.

### Weaknesses
W1. This paper needs further polishing its statement. For instance, in the introduction, the jump from discussing model-based methods directly to DL-based methods is abrupt.

W2. The non-negative weights limit the model's expressive power, as it cannot directly model inhibitory relationships between nodes, unable to match the performance of the real Transformer.

W3. Although this paper provides the model parameters, it does not specifically analyze the computational complexity of the training or inference stages

### Questions
Q1. This paper separates spatial relationships and temporal relationships entirely in traffic forecast performance. But will it lose complex and coupled spatial-temporal information in the real world?

Q2. Why was Mahalanobis distance chosen to measure the distance between node features? What is the advantage of it?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a lightweight and interpretable transformer-like neural network by unrolling a mixed-graphs optimization algorithm for spatio-temporal traffic forecasting. The key idea is to unroll an ADMM solver for the mixed-graphs optimization problem: an undirected graph $G^{u}$ encodes spatial correlations and a directed graph $G^{d}$ encodes temporal causality. The authors introduce two priors DGLR ($\ell_{2}$) and DGTV ($\ell_{1}$) to promote low-frequency (smooth) reconstruction on directed graphs, while graph learning modules act as a parameter-efficient self-attention surrogate. On META-LA and PEMS03, the proposed model are compatible with baseline algorithms in accuracy with much fewer parameters ($\approx$ 34K, 6.4% of PDFormer). For the priors, the ablation study shows their importance. Overall, this paper addresses an important issue in spatio-temporal traffic forecasting and the proposed algorithm is promising. However, this paper could be improved with discussion and verification of influence of low-pass on non-smooth or long-range effects and more extensive experimental evaluation.

### Strengths
- The proposed algorithm has a theoretical background with unrolling. Eacy layer corresponds to an iteration that minimizes an explicit mixed-graph objective, which increases interpretability. 
- The authors provide a perspective on the low-pass on directed graphs by modeling temporal causality with directed Laplacian/TV regularizers. It is beyond standard undirected graph signal processing and not limited to heuristic smoothing. 
- With the directed graph learning, the learned graph weights act like attention scores and admit a Mahalanobis metric-learning interpretation. 
- In experiments, the proposed algorithm is compatible with baseline algorithms in accuracy with much fewer parameters.

### Weaknesses
- Low-pass on directed graphs may decrease non-smooth or long-range effects. In both theory and experiments, this points should be further explored. 
- Convergence guarantee with ADMM may be questionable because the optimization problem in Eq.(6) is not convex. It should be further discussed for nonconve ADMM such as choice for $\rho$ and step sizes (c.f., Hong et al. 2016). 
- Experimental evaluation is limited because (i) there are only two datasets; (ii) the datasets are sub-sampled to 1/3, and (iii) trained for only 70 epochs. Verification on the full-scale datasets and more dataset would strengthen the proposed method. 
- The readability could be improved. For example, the overall algorithm should be shown in the main text or appendix. 

[Hong et al. 2016] Mingyi Hong, Zhi-Quan Luo, and Meisam Razaviyayn. Convergence analysis of alternating direction method of multipliers for a family of nonconvex problems. SIAM Journal on Optimization, 26(1), 2016.

### Questions
Please answer the points listed in the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
