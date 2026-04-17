# Online Learning of Nonlinear Autoregressive Processes over Cellular Complexes

- Decision: Reject
- Scores: 2, 2

## Abstract
Real-world time-series are often high-dimensional, structured with higher-order dependencies, and exhibit time-varying dynamics, making them challenging to model effectively. Abstract Cellular Complexes (ACCs) provide a principled way to capture such higher-order topological structure, serving as a powerful inductive bias for multivariate time-series modeling. While recent methods based on message-passing neural networks and Hodge Laplacians exploit higher-order relationships, they are primarily designed for offline settings and lack adaptability to streaming and non-stationary environments. In this work, we introduce a framework for nonlinear autoregressive modeling over ACCs, where predictive functions are defined in a reproducing kernel Hilbert space (RKHS) induced by shift-invariant kernels. We further propose an efficient online learning algorithm to estimate these functions. Experimental results on synthetic and real-world datasets demonstrate that our method shows competitive or improved performance in prediction accuracy and adaptability under streaming conditions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the challenge of modeling high-dimensional, nonlinear, and non-stationary time-series data that possess higher-order structural dependencies (e.g., brain networks, fluid dynamics, water networks). The authors propose a novel nonlinear autoregressive framework defined over Abstract Cellular Complexes (ACCs), which is a generalization of graphs that includes nodes, edges, faces, etc. Each element (cell) can exchange information with its faces and cofaces via kernel-based interactions. The model is implemented in a Reproducing Kernel Hilbert Space (RKHS), capturing nonlinear dependencies while maintaining interpretability.

### Strengths
- This work is the first framework to study online nonlinear autoregressive modeling on cellular complexes.

- Proposed methods are mathematically rigor with multiple definitions and supported by theoretical error bound (theorem 1).

- Random Feature variant (RF-HORSO) makes the method tractable in streaming contexts.

- Experiments are performed in both synthetic and real-world datasets.

### Weaknesses
- I view the contribution on modeling the interaction between higher- and lower-order topologies is still novel and interesting. However, in a broader view, relationship between vertex to edge, edge to vertex, vertex-to-vertex, and edge-to-edge have been explored in nonlinear dynamical systems with autoregressive temporal dynamics [1,2], and online adaptation possibilities [3,4].

[1] Neural relational inference for interacting systems, ICML, 2018

[2] Neural Relational Inference with Efficient Message Passing Mechanisms, AAAI, 2021

[3] Online Relational Inference for Evolving Multi-agent Interacting Systems, NeurIPS, 2024

[4] Online Multi-agent Forecasting with Interpretable Collaborative Graph Neural Networks. IEEE TNNLS, 2022

- While conceptually and mathematically interesting, it would be nice if more real-world applications can be discussed where relationships between different-dimension cells emerged in high-order systems. While L33-36 discuss this a bit, I would like to see more mathematical properties or detailed examples.

- Experimental results are overall weak. Only ocean datasets are used (Ocean Cellular and Ocean Edge), and RF-HORSO performs better only in the Ocean Cellular dataset.

- RF-HORSO’s loss seems not converging in both datasets (Figure 3(a) and (b)). The lowest error is achieved at the very early timestep in Figure 3(a) and then the loss is highly fluctuating and increasing, which seems concerning. In contrast, other methods exhibit relatively stable though their loss is a bit higher.

### Questions
- Could the authors elaborate on why online algorithms are required for this benchmark (Ocean datasets)? What are the evolving factors that need to be adapted by models in this dataset? - otherwise, online learning may not be necessary.

- Why RF-HORSO exhibits much large variance in MSE over time? 

- Are the number of learnable parameters comparable in the compared methods (RF-HORSO, S-VAR, SC-VAR)? 

- Are these methods optimized with the similar level of learning rate?

- How fast are these methods in online optimization? - to make a fair comparison.

- Could the authors elaborate the definition of faces and cofaces before referring them for the readers? 

- Probably, better to show the examples in Figure 1 better corresponding to the definitions. For example, L_sigma^k is not shown in Figure, instead showing L_tau^k. Also, it would be nice to elaborate the difference between blue-dot line and pink line in Figure 1 (though L127 discuss it a bit). 

- I feel it's little confusing to follow the notation. In L144, x_k(S) denotes the values attached to k-cells, which I assume k-dimensional cells. That is, S includes the k-dimensional cells. In Figure 1, L_{tau}^{1} in x_{1}^{t-p}(L_{tau}^{1}) should represent 0-dimensional (not 1-dimensional, where k=1) neighbors of tau, based on Definition 2 (i.e., one dimension smaller), which is not aligned with L144.

### Soundness
3

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
2

### Summary
This paper proposes NL-HORSO for online learning of nonlinear autoregressive time series over Abstract Cellular Complexes (ACCs). The method uses kernel functions in RKHS to model dependencies between cells at different topological levels (nodes, edges, polygons), with RF-HORSO providing a scalable random feature approximation. Dynamic regret bounds are provided, and experiments are conducted on synthetic and real datasets.

### Strengths
1. Theoretically grounded: Provides dynamic regret bounds for online learning over topological structures using standard COMID analysis

2. Computationally tractable: RF-HORSO addresses scalability through random feature approximation with fixed dimensionality

3. General framework: Unified treatment of signals across nodes, edges, and higher-dimensional cells

4. Comprehensive appendix: Detailed proofs, derivations, and experimental settings

### Weaknesses
**Soundness Issues:**

1. Shallow theoretical contribution: Dynamic regret analysis (Theorem 1) applies standard COMID+RF results; extension to ACCs is primarily notational

2. Unvalidated assumptions: Assumptions A1-A6 are not verified empirically; practical implications of regret bound unclear given noisy experimental results

3. Missing justification: No explanation for why RKHS formulation is necessary or how it fundamentally connects to topological structure

**Presentation Issues:**

4. Unclear motivation flow: Abrupt jump from ACC motivation (Section 1) to RKHS formulation (Eq. 1) without explaining why kernels are needed

5. Heavy notation without intuition: Definitions 1-6 introduce dense formalism before providing insight; roles of f^p_{σ,c} vs g^p_{σ,c} unclear

6. Key insights buried: Connection to linear convolution (Appendix B.1) should appear in main text to build intuition

7. Late visualization: Figures 4-6 can help but appear after complex equations; early intuitive diagrams needed

**Contribution Issues:**

8. Incremental novelty: Primarily extends existing graph kernel methods (RFNL-TIRSO) to track upper/lower adjacencies; no new algorithmic or theoretical insights

9. Unconvincing experiments:

RF-HORSO shows high variance (Fig 3a) yet claims "competitive performance"

TopoLMS learns identity mapping (Table 4), suggesting topology isn't beneficial

No ablation comparing ACC vs. graph vs. independent time series


**Questionable datasets:** 

Ocean/Navier-Stokes choices not justified - unclear why these require cellular complex structure rather than graphs


**Missing validation:** 

No demonstration of when ACC structure provides advantage; claim of "first work on online learning over cellular complexes" lacks evidence this is actually a useful problem

### Questions
1. Can you provide intuition for when ACC structure provides advantage over graphs? What properties of the time series/system necessitate modeling triangles, not just edges?

2. Why does TopoLMS converge to identity (Table 4)? Does this suggest the cellular complex structure isn't beneficial for this data?
In Figure 3a, RF-HORSO shows much higher variance than baselines. Is this acceptable? How do you interpret "competitive performance" given this instability?

3. Can you compare against simply treating all cells as independent time series (ignoring topology)? This would clarify the value of the topological inductive bias.

4. The regret bound scales with path length W_σ[T]. How does this compare empirically to the observed error? Are the assumptions A1-A6 verified on your datasets?

### Soundness
2

### Presentation
2

### Contribution
2
