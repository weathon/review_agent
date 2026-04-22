# Enabling arbitrary inference in spatio-temporal dynamic systems: A physics-inspired perspective

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Modern spatio-temporal learning techniques usually exploit sampled discrete observations to foresee the future. Actually, spatio-temporal dynamics are continuous and evolve continuously across time and space, thus modeling  spatio-temporal dynamics in a continuous space can be a long-standing challenge. Existing deep learning architectures often fail to generalize to unseen regions and new graph topologies, while many physics-driven approaches are confined to Euclidean grids and  poorly scale to complex graph structures. To address this gap, we propose PhySTA, a physics-inspired spatio-temporal learning framework designed for efficient and scalable arbitrary inference over graph-structured data. PhySTA integrates two key modules: (1) Continuous Operator-based Spectrum-Temporal Learning (CoSTL), which leverages a Graph-Time Fourier Neural Operator combined with Time-Gated Spectral Segmentation Perception to model continuous dynamics in operator space, and (2) Adaptive Multi-scale Interaction (AMI) that constructs multi-scale subgraphs and introduces node-edge coupled convolution to capture discrete interaction patterns and refine continuous predictions. By bridging operator learning with node-edge-graph interaction, PhySTA achieves both continuity-aware dynamic modeling and hierarchical interactive refinement. Extensive experiments across large-scale benchmarks demonstrate that PhySTA attains state-of-the-art accuracy while reducing computation cost and lowering parameter overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PhySTA (Physics-inspired Spatio-Temporal Learning for Arbitrary Inference), a framework that models continuous spatio-temporal dynamics on graph-structured data by integrating operator learning with multi-scale graph neural networks. The core idea is to bridge the gap between continuous real-world dynamics and discrete sensor observations, enabling reliable inference in unobserved regions.

The framework comprises two main modules:
Continuous Operator-based Spectrum-Temporal Learning (CoSTL): Uses a Graph-Time Fourier Neural Operator (GT-FNO) and Time-Gated Spectral Segmentation Perception to model continuous dynamics in the joint spectral domain

Adaptive Multi-scale Interaction (AMI): Employs a novel Node-Edge Coupled Convolution and Multi-scale Subgraph Partition (coarse-mid-fine hierarchy) to capture discrete multi-scale interactions and refine the continuous predictions.

### Strengths
Strong Theoretical and Architectural Novelty: The approach successfully extends the Fourier Neural Operator (FNO), a tool for continuous Euclidean domains, to non-Euclidean graph domains (GT-FNO) using the magnetic Laplacian and a joint graph-time spectral decomposition. This is a significant theoretical advance in spatio-temporal GNNs.

Robustness to Data Sparsity: PhySTA consistently achieves top performance across all datasets under varying degrees of node masking (up to $70\%$ sparsity). This directly validates its core claim of enabling robust "arbitrary inference" in unobserved regions

fficiency and Low Overhead: PhySTA achieves state-of-the-art accuracy while using a lower number of parameters ($123,474$) and less GPU memory ($6,042 \text{MB}$) compared to most deep GNN baselines like AGCRN and STG-ODE

### Weaknesses
1. Scalability Bottleneck of Spectral Decomposition: The paper acknowledges that the Graph Fourier Transform (GFT) step, which involves spectral decomposition (magnetic Laplacian), can be computationally demanding on very large graphs9. Since GFT's complexity is $O(N^2)$, this $N^2$ dependency limits the framework's scalability for massive real-world graphs (e.g., city-scale mobility or power grids with $N \gg 1000$). It is crucial to discuss concrete techniques like Nyström approximation or sparse spectral methods to mitigate the $O(N^2)$ bottleneck

2. Ambiguity of Multi-scale Aggregation: The AMI module's multi-scale single-layer modeling (coarse, mid, fine graphs) claims to capture complex interactions efficiently. However, the Louvain community detection used for subgraph creation is non-differentiable, potentially complicating end-to-end training. The authors should clarify how the node2center mapping and the three-level graph structure are integrated into the differentiable training pipeline.

### Questions
Addressing GFT Scalability (The Major Bottleneck): The paper acknowledges that the $O(N^2)$ complexity of the full Graph Fourier Transform (GFT) limits scalability. Since current spatio-temporal benchmarks scale up to 716 nodes (SD), the $N^2$ barrier remains a practical constraint for city-scale graphs ($N>5000$). Could the authors propose and experimentally validate a concrete, scalable approximation technique within the GT-FNO architecture (e.g., using a localized sparse spectral filter, Nyström approximation on the magnetic Laplacian, or spectral compression) to push the scalability beyond the current constraint?


Differentiability and Justification of Multi-scale Graph Construction: The Adaptive Multi-scale Interaction (AMI) relies on Louvain community detection to generate coarse and fine subgraphs. Since community detection is generally non-differentiable, how is the graph hierarchy generation process integrated into the end-to-end training pipeline? Furthermore, given the complexity of AMI, can the authors provide a side-by-side comparison (e.g., in the Appendix) of the performance gains versus a simpler, fully differentiable multi-scale strategy, such as one based on standard differentiable pooling (DiffPool/TopK)?


Detailed Analysis of Spectral Component Contributions (TGSSP): The ablation study suggests that the specialized Time-Gated Spectral Segmentation Perception (TGSSP) module contributes only minor gains compared to other components. Given the module's complexity (bandwise parameterization, time-gating, complex eigenvalues/magnetic Laplacian), can the authors provide a more detailed visualization or analysis of what TGSSP learns? For example, showing how the learned per-mode scaling factor ($\alpha_k$) and the time-gating factor ($g(\omega)$) prioritize or suppress specific frequency bands over time would better justify the module's role in modeling non-stationary dynamics.

More applications: could it be applied to weather and climate forecasting?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on addressing the challenge of arbitrary inference in spatiotemporal dynamic systems, where existing methods either fail to generalize to unseen regions and complex graph structures or are confined to Euclidean grids. The proposed framework PhySTA integrates two innovative core modules: Continuous Operator-based Spectrum-Temporal Learning (CoSTL), which extends neural operators to non-Euclidean domains via a Graph-Time Fourier Neural Operator (GT-FNO) and Time-Gated Spectral Segmentation Perception for continuous dynamics modeling, and Adaptive Multi-scale Interaction (AMI) that constructs multiscale subgraphs and uses node-edge coupled convolution to capture discrete interactions. Theoretically, GT-FNO is proven to have universal approximation capability for continuous spatio-temporal graph operators with controllable \(L^2\) error. Experimentally, on traffic and air quality benchmarks (PEMS-BAY, SD, KnowAir), PhySTA achieves state-of-the-art accuracy across various missing data ratios, reduces computation cost significantly, and has fewer parameters and lower GPU memory consumption compared to baselines, demonstrating robust generalization for arbitrary inference even in sparse sensor scenarios.

### Strengths
1. The paper is well-structured, and its notations are generally clear.
2. The proposed approach demonstrates performance improvements in most experimental settings compared to recent baseline methods.

### Weaknesses
1. In the motivation section, the authors propose that the modeling of continuous changes relies on graph spectral modeling, but the correlation between the two is not clearly established.
2. It is not specified where the truncation operation is performed.
3. In Equation 7, the graph is divided into three layers: coarse, mid, and fine. However, these subgraphs only exist on partial nodes. Therefore, it remains unclear whether the extracted features (X_coarse, X_mid, X_fine) have missing node features respectively.
4. Issues with notations and formatting:
   - In Line 256, there is a missing space before "Inspired"; in Line 264, there is a missing space before "The".
   - In Equation 8, how are the two predicted values (y_costl and y_ami) obtained? Is the result of Equation 5 equivalent to Y_AMI?
5. How do continuous spectrum-temporal modeling and adaptive multi-scale interaction handle dynamic topology changes respectively? At present, the study seems to treat the underlying graph (i.e., adjacency matrix A) as static, without corresponding discussions on dynamic scenarios.

### Questions
see details in weakness

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
3

### Summary
The paper proposes PhySTA, a physics-inspired framework that unifies continuous operator learning with graph-based spatio-temporal modeling. It introduces two main modules:
(1) a Graph–Time Fourier Neural Operator (GT-FNO) equipped with Time-Gated Spectral Segmentation Perception (TGSSP) for modeling continuous spectral dynamics on graphs, and
(2) an Adaptive Multi-Scale Interaction (AMI) mechanism that captures multi-scale node–edge relationships via coupled convolution and hierarchical graph construction.
A Continuity–Discreteness Interaction Module (CDIM) further fuses both continuous and discrete predictions for arbitrary inference in unobserved regions.
Experiments on large-scale traffic and air-quality datasets demonstrate strong accuracy, robustness, and efficiency compared with several state-of-the-art baselines.

### Strengths
1. Novel integration of physics-inspired operator learning and GNNs:
The proposed GT-FNO extends Fourier Neural Operators to non-Euclidean graphs, enabling continuous modeling over directed graphs—a clear conceptual innovation.

2. Multi-scale and coupled graph design:
The AMI module effectively captures long-range, multi-level dependencies within a single layer, addressing over-smoothing and inefficiency issues seen in deep GNNs.

3. Strong empirical performance and efficiency:
PhySTA achieves consistent improvements across datasets with fewer parameters and memory cost (≈74% FLOP reduction), showing excellent trade-offs between accuracy and scalability.

4. Comprehensive experiments and ablation analysis:
The inclusion of multiple datasets, mask ratios, and detailed component ablations provides good evidence of robustness and interpretability.

### Weaknesses
* Limited comparison to recent operator-based or physics-informed baselines:
The paper mainly compares against classical and GNN-based methods (STGCN, DGCRN, etc.), but omits recent neural operator or PDE-based baselines such as Graph Neural Operator (Li et al., 2023) or Geo-FNO (Li et al., 2024). These would strengthen the claim of operator generalization.

* Writing quality and presentation:
The exposition is heavy and sometimes unclear, especially in the methodology section. Some mathematical notations are inconsistent, and figures (e.g., Fig. 2, Fig. 3) are not fully self-explanatory. The authors could simplify and streamline the presentation for readability.

* Ablation and interpretability could be expanded:
Although the ablation table is informative, qualitative insights on how each frequency band or subgraph level contributes to the final prediction are missing. Visualizations of spectral energy distribution or temporal gating behavior would enhance interpretability.

* Scalability limitation not sufficiently addressed:
The reliance on magnetic Laplacian spectral decomposition may hinder scalability for very large graphs. While this is briefly mentioned in the limitations, empirical evaluation on larger graphs would make the claim more convincing.

* Incomplete baseline coverage:
Some recent transformer-based and neural-operator hybrid methods (e.g., Graphormer, SpaceTimeFormer) are missing from comparison, which may weaken the “state-of-the-art” claim.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
