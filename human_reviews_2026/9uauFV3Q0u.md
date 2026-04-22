# Spectral Sheaf Filtering: A Topological Approach to Spatio-Temporal Modeling

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 0, 8

## Abstract
Spatio-temporal data pose significant challenges for graph-based learning due to their complex, non-stationary dependencies and the limitations of conventional message passing in capturing high-order, asymmetric interactions. We introduce Spectral Sheaf Filtering (SSF), a novel and theoretically grounded framework that redefines information propagation on graphs using the algebraic topology of cellular sheaves. By assigning vector spaces and restriction maps to nodes and edges, SSF encodes context-dependent, localized dynamics that extend far beyond traditional adjacency structures. To further enhance expressivity and efficiency, we introduce spectral filtering over the sheaf Laplacian, enabling frequency-aware decomposition via the graph Fourier transform while emphasizing latent spectral features. This spectral view allows SSF to adaptively modulate information flow across frequency components, effectively mitigating oversmoothing in deep graph neural networks. Extensive experiments on diverse spatio-temporal traffic forecasting benchmarks show that SSF consistently outperforms state-of-the-art methods, especially in long-horizon forecasting tasks. Our results highlight the value of topological structures in advancing graph learning for spatio-temporal systems. The code is available at: https://github.com/anonymous-submisssion/SSF.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on addressing the challenges of graph-based learning in spatio-temporal data, such as complex non-stationary dependencies and the limitations of conventional message passing. It proposes the Spectral Sheaf Filtering (SSF) framework, a pioneering approach that leverages cellular sheaf algebraic topology to redefine graph information propagation—assigning vector spaces and restriction maps to nodes and edges to capture context-dependent localized dynamics. A key innovation is the introduction of spectral filtering over the sheaf Laplacian, enabling frequency-aware decomposition via graph Fourier transform to mitigate oversmoothing in deep graph neural networks. Theoretically, the paper establishes the eigendecomposition of the sheaf Laplacian, generalizing Fourier modes to the sheaf-theoretic setting. Experimentally, on five spatio-temporal traffic forecasting benchmarks (METR-LA, PEMS-BAY, PEMS04, PEMS08, NAVER-Seoul), SSF outperforms state-of-the-art methods, especially in long-horizon forecasting, with significant error reductions and maintains efficient training speed.

### Strengths
1. The motivation in the introduction, especailly shown in Figure 1 is convincing and easy to catch. The perspective of utilizing sheaf representation for spatial-temporal graph tasks is novel.
2. The formulation of sheaf representation and analysis is very clear.
3. The paper conducts comprehensive experiments to evaluate the proposed methods, the the prediction accuracy improves significantly.

### Weaknesses
1. The most severe concern is the clarity of the proposed model. Though the introduction of sheaf formulation is clear, it is unclear how the spatail-temporal information of **previous t timestamps** of node is utilized to output the prediction of nodes in **the further T' timestamps**.  It is also unclear what is the loss function and how the model is trained. This concern is crucial that affects the clarity of this paper.
2. The paper lacks the complexity analysis for the proposed method, as the Laplacian decomposition is time-consuming. It is also better to list the running time of decomposition in appendix F in addition to the training runtime.

### Questions
See details in weakenss.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces *Spectral Sheaf Filtering (SSF)* for spatio temporal node regression/prediction task. It builds a cellular sheaf with learned restriction maps on vertex–edge incidences, forms the *sheaf Laplacian*, and performs propagation via spectral heat kernel filtering. A rolling window of past graph signals is processed slice by slice through these sheaf spectral layers, followed by a small MLP head for multi horizon node prediction. Experiments on five traffic datasets report strong performance with ablations on filtering, stalk dimension, and number of modes.

### Strengths
1. **Clean spatial operator.** The cellular sheaf with learned restriction maps yields a well-defined sheaf Laplacian and an interpretable quadratic form that directly encodes edge-wise compatibility.

2. **Principled frequency control.** Spectral heat kernel filtering in the sheaf eigenbasis addresses oversmoothing while retaining useful high frequency content; ablations on filter on or off, stalk dimension (d), and top (k) modes support the design.

3. **Strong empirical performance.** Competitive results across standard traffic benchmarks, clear algorithmic presentation, and interpretability hooks via low frequency modes and the nullspace.

### Weaknesses
1. **Temporal head underpowered.** After a strong spatial backbone, the temporal modeling is a small MLP. A comparison to lightweight temporal modules (1D temporal conv, GRU, tiny attention) is missing.

2. **Metric sanity.** Some reported MAPE values look unusually small given the MAE and RMSE, suggesting possible differences in definitions, units, or de normalization. Baseline training and metric computation parity are not fully documented.

3. **Domain breadth.** All experiments are traffic speed forecasting. A second non traffic domain (or transfer tests) would strengthen claims of general utility.

4. **Efficiency specifics.** Top (k) eigensolvers can dominate runtime. The schedule for recomputing eigenpairs and a timing breakdown are not fully reported.

### Questions
1. **Temporal modeling.** Please compare your MLP head to a small temporal module per node: 1D temporal conv, GRU, or a tiny attention block. Report accuracy and wall clock.

2. **Metrics.**  Several tables show MAPE notably lower than what the reported MAE would imply. Please provide exact formulas for MAE, RMSE, and MAPE, confirm that evaluation is on de normalized predictions, state the units, and specify any epsilon or clipping used in the MAPE denominator.

3. **Scalability.** How often are eigenpairs recomputed during training. What are typical (k) and stalk dimension (d). Include a timing breakdown for eigensolve, forward, and backward, and a scaling curve with (N), (d), and (k).

4. **Generality.** Can you add one non traffic dataset of the same task type (e.g., environmental, power, or something similar) or a generalization test: inductive node split or cross city transfer.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes Spectral Sheaf Filtering (SSF) for spatio-temporal forecasting. The authors (i) build a cellular sheaf on a sensor graph with learnable restriction maps, (ii) define a sheaf Laplacian and apply spectral filtering via a heat kernel, performing “message passing in the spectral domain”, and (iii) report results on benchmark datasets. Key claims include improved expressivity, mitigation of over-smoothing, global receptive fields with O(1) operations, and reduced computational burden from using spectral filtering.

### Strengths
- Clear motivation to move beyond uniform edge propagation; sheaves are a natural tool to model asymmetric/high-order relations.

### Weaknesses
- The core technical component, spectral filtering of a Laplacian, is classical in graph signal processing (GSP), including design/learning of filters and universal approximation with FIR/IIR (e.g., Shuman et al., 2013; Sandryhaila & Moura, 2013). The paper does not acknowledge this body of work adequately nor contrast SSF against modern spatio-temporal spectral approaches (e.g., Einizade et al., NeurIPS 2024, which provides stability/over-smoothing analysis and SOTA forecasting). As written, replacing the standard Laplacian by a sheaf Laplacian plus a heat kernel filter is an incremental variant of well-established spectral filtering; the “novelty” claim (Sec. 1, contributions) is overstated.
- The paper claims spectral methods “achieve a global receptive field in O(1) operations”, computing (and back-propagating through) eigendecompositions is not O(1) and is typically O(kN^2) for k eigenvector-eigenvalue pairs. Therefore, the manuscript claims spectral filtering reduces computational burden without a rigorous cost analysis. These are contradictory/incorrect as stated.
- “This decomposition is interpretable” is claimed but not demonstrated.
- Oversmoothing and expressivity improvements are claimed but not theoretically analyzed, and the empirical evidence directly targeting these phenomena is missing.
- The scalability claim in the conclusion is not substantiated. While Table 6 reports per-epoch training time, scalability is dominated by (i) the eigendecomposition of the (sheaf) Laplacian, and (ii) dense transforms like $U^T X$. As the number of nodes increases, both the eigenvalue decomposition and the dense multiplications become bottlenecks in time and memory.
- The paper states a spectral decomposition $L_F=U \Lambda U^T$ and interprets eigenpairs as “frequencies”, but never establishes conditions under which the proposed Laplacian is symmetric positive semidefinite (and thus diagonalizable by an orthonormal basis with nonnegative eigenvalues). This is critical for using a heat kernel and a Fourier-like transform.
- Equations (6-8) implement project-filter-reconstruct (Fourier transform, diagonal filter, inverse transform). Calling this a message passing layer is misleading: it is spectral filtering.
- No comparison against sheaf GNNs trained in the spatial domain. Without this, it is unclear whether the gain comes from the sheaf modeling or simply from spectral filtering.
- No comparison with CITRUS (Einizade et al., NeurIPS 2024), which reports SOTA on the same class of tasks on longer horizons, and includes oversmoothing analysis. Similarly, no comparison with more classical ARMA graph-temporal filters.
- The algorithm input mentions “hyperedge index” although the core model is defined on graphs.
- The claim “novel spectral filtering” is too broad. Spectral filtering for graph signals is a decade-old area (Shuman et al., 2013; Sandryhaila & Moura, 2013; many follow-ups), with universal filter design and time-graph IIR filters; recent spatio-temporal spectral GNNs exist. Please moderate claims and cite appropriately.


**General comment**: The paper reads like it was written heavily relying on LLMs: a repetitive introduction, broad claims stated without proof, and numerous inconsistencies. Even if LLMs were used, which is acceptable, the authors have not verified facts, moderate claims, and aligned the narrative with the actual technical content.

### Questions
- Under which assumptions on $L_F$ is symmetric PSD?
- What is the actual computational complexity?
- Is $\alpha$ in the heat kernel a learned parameter per layer or a fixed hyperparameter? How is it selected?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper applies a new sheaf-based graph network approach for spatiotemporal forecasting. The model achieves very impressive results on the METR-LA dataset.

### Strengths
- Originality: The proposed method SSF is novel in traffic prediction.
- Quality: The model is validated on several traffic datasets. The data preprocessing pipelines seem consistent with SOTA methods.
- Clarity: The paper provides appropriate background for readers to understand the sheaf GNNs.
- Significance: The reported improvement on METR-LA is significant.

### Weaknesses
- Clarity
  - The paper could benefit from a better description of the information flow. While Figure 2 shows the general framework, it is unclear how spatiotemporal data is processed by SSF.
  - Most works in the field report MAE, RMSE, MAPE on all 3 horizons (3, 6, 12). Having Table 5 in the main paper instead of Table 1 would allow for better comparison.
- Motivation
  - It seems like sheaf GNNs were designed and evaluated on hypergraph datasets. Traffic networks are static. The paper would benefit from showing the need for sheaf GNNs in traffic data.

### Questions
- Is the temporal dimension of the data flattened and used as the node embedding? Is that what line 18 does in Algorithm 1? Is there a linear layer to project the initial node embeddings to the hidden space?
- In the provided GitHub codebase, you call `python run.py --model_id metr_12`. Do you train a separate model for each forecasting horizon?
- Can you provide a pretrained model for result reproduction? Are the baseline method results copied from the original papers?

### Soundness
4

### Presentation
2

### Contribution
3
