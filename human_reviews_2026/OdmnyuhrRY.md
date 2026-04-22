# Mixture of Spectral Wavelets on Simplicial Complex: Analysis of Brain Connectome with Neurodegeneration

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Understanding how pathological changes disrupt communication patterns in the brain requires models that examine interactions across multiple structural levels of a network. Many existing graph-learning methods emphasize node representations while providing limited treatment of signals defined on edges, and their use of predetermined spectral filters can restrict sensitivity to heterogeneous frequency behavior.
To overcome these issues, we introduce a framework that couples a simplicial wavelet–based representation—capable of handling signals on both vertices and connections—with an adaptive filtering module that selects informative spectral components in a data-dependent manner. This combination enables flexible multi-scale analysis and highlights structural patterns relevant to neurodegenerative conditions. Evaluations on widely used brain graph benchmarks show consistent gains in predictive performance as well as clearer interpretation of disease-related network alterations. The implementation of this work will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel framework, SSWT-SpMoE (Mixture of Spectral Wavelets on Simplicial Complex with Spectral Mixture of Experts), for brain connectome analysis in the diagnosis of neurodegenerative diseases such as Alzheimer’s disease (AD) and Parkinson’s disease (PD). The authors aim to overcome two main limitations of existing Graph Neural Networks (GNNs) in this domain: (1) the neglect of crucial edge-level information due to a predominant focus on node features, and (2) the restricted ability of spectral GNNs to capture diverse frequency components because of fixed-bandwidth filtering. To address these issues, the proposed framework integrates two key components. The Spectral Simplicial Wavelet Transform (SSWT) extends spectral graph wavelet transforms to higher-order simplicial complexes using the Hodge Laplacian, enabling joint multi-scale analysis of both node and edge features. The Spectral Mixture of Experts (SpMoE) employs a gating mechanism to dynamically select the most relevant wavelet scales (“experts”) for each graph, allowing adaptive multi-scale spectral filtering. Together, these innovations enable richer spectral representations and adaptive frequency analysis tailored to each brain graph.

### Strengths
### **Technical innovation: Joint spectral node–edge analysis and adaptive filtering**
The paper presents *strong methodological originality* by introducing the Hodge Laplacian into the spectral domain through the Spectral Simplicial Wavelet Transform (SSWT), enabling joint multi-scale spectral analysis of both node and edge features in brain graphs. In addition, the proposed Spectral Mixture of Experts (SpMoE) employs a graph-level gating mechanism to dynamically select relevant spectral wavelet “experts,” effectively overcoming the limitations of fixed-bandwidth filters in conventional spectral GNNs. The effectiveness of these components is further supported by comprehensive ablation studies.

### **Demonstrated generalization across structural and functional networks**
The framework is evaluated on *both structural and functional brain networks*, demonstrating robust and consistent performance across different graph modalities. This cross-modality validation highlights the model’s strong generalization capability and reinforces its applicability to diverse neuroimaging scenarios, reducing concerns of overfitting to a specific data type.

### **Interpretability and clinical relevance**
The interpretability analysis provides compelling evidence of biological plausibility: regions and connections with high Grad-CAM activations align well with established AD/PD-related brain areas such as the thalamus, hippocampus, and putamen. Moreover, the expert selection patterns correspond to different disease stages, suggesting that the proposed model offers not only predictive power but also clinically meaningful and interpretable insights into disease mechanisms.

### Weaknesses
### **Unfair experimental setup and limited baselines**
The most critical limitation lies in the fairness of the experimental comparison. The paper mainly compares against models that use BOLD signals as node features, rather than methods that leverage connectivity-matrix-based node features—which have become the de facto standard in brain graph learning. Recent studies, such as ***NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics (NeurIPS 2023)*** , have demonstrated that using connectivity matrix rows as node features yields significantly stronger performance than using raw BOLD signals. Without comparisons to such message-passing GNNs using connectivity-based features, the empirical advantage of the proposed spectral approach remains unconvincing.

**Suggested improvement:** The authors should include comparisons with recent message-passing GNNs (e.g., GraphTransformer, BrainGNN, or models evaluated in NeuroGraph) using connectivity-matrix node features to establish a fair and rigorous benchmark.

### **Limited scope of datasets and unclear domain specificity**
Although the paper evaluates the model on neurodegenerative disease datasets, the scope remains narrow. It does not include experiments on widely used brain network benchmarks such as ***ABIDE (autism) or ADHD-200***, which are commonly adopted in brain connectome modeling. This raises concerns about the generalizability of the proposed method beyond neurodegenerative contexts.

**Suggested improvement:** The authors could either (a) extend experiments to additional benchmark datasets to demonstrate broader applicability, or (b) clearly justify why the proposed edge-focused spectral approach is particularly suitable for neurodegenerative diseases and less so for other brain disorders, thereby sharpening the paper’s conceptual focus.

### **Accessibility and clarity of the method section**
The paper’s theoretical development, while rigorous, may be difficult to follow for readers unfamiliar with spectral graph theory and simplicial complexes. The key advantages of the proposed SSWT-SpMoE framework are not sufficiently illustrated in intuitive or visual terms.

**Suggested improvement:** The authors could enhance clarity by adding conceptual diagrams, toy examples, or intuitive explanations that highlight how SSWT enables joint node–edge analysis and why the adaptive spectral filtering improves over fixed-bandwidth methods.

If the authors can provide additional experiments demonstrating that their method outperforms ***SOTA message-passing GNNs using connectivity-based node features on standard benchmarks***, the technical contribution would be much more convincing, and I would be inclined to reconsider my overall assessment of the paper.

### Questions
**1. Scalability and approximation**

Given the model’s reliance on spectral decomposition, how does it scale to larger graphs with tens of thousands of nodes or edges? Have the authors explored approximate spectral methods (e.g., Chebyshev polynomial approximation) to reduce computational cost? If so, how would such approximations affect the SpMoE gating mechanism and its ability to select optimal spectral scales?

**2. Kernel function choice**

The kernel function $\mathcal{K}$ used in Eq. (7) is a critical design choice, and Figure 1 suggests the use of the heat kernel $\mathcal{K}_{s}(\lambda)=e^{-s\lambda}$. Which kernel was actually employed in experiments? How sensitive is the method to the kernel choice, and how was its form or scale parameter determined?

**3. Hodge decomposition and interpretability**

Since the Hodge Laplacian decomposes signals into gradient, curl, and harmonic components, does SSWT-SpMoE explicitly or implicitly leverage these subspaces? Could the authors discuss whether different spectral components correspond to meaningful neurobiological or topological patterns (e.g., gradient-like atrophy flows or rotational dysfunctions) in neurodegenerative diseases?

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
This paper develops the wavelet transform on 0-simplices and 1-simplices based on Hodge Laplacian, to overcome the difficulty of jointly analyzing the node and edge features. Adaptive wavelet scale selection is applied in the node and edge representation learning based on $L_0$ and $L_1$ operators. Application in brain network datasets validates the effectiveness of the proposed model.

### Strengths
1) The Hodge Laplacian adopted in this paper solves the mixture of node feature and edge feature in traditional GNNs. It allows the representation learning of node feature and edge feature within its own domain, which solves the difficulty of processing node and edge feature simultaneously.

2) The network proposed in this paper adopts the wavelet scale selection in multi-scale representation learning, which introduces flexibility compared with the learning with fixed wavelet.

3) The experiment part is adequate with multiple datasets considered and results visualization provided, which is convincing and straightforward.

### Weaknesses
1) The method proposed in this paper requires a complete eigenvalue decomposition of both the $L_0$ and $L_1$ operators, which leads to huge computational and storage costs, since the complexity of EVD is O($N^3$) and the size of eigenvalue matrix is O($N^2$).

2) The representation learning strucure of the proposed method is a shallow network with adaptive graph wavelet transform, and lacks a hierachical structure. Meanwhile, the proposed method is sensitive to the hyper-parameters, such as the range of wavelet scales and the number of experts.

### Questions
1) In this paper, the node feature and edge feature are processed within its own space, and only interacts with each other in the final step of label prediction. Will that lead to a potential mismatching of the node and edge features? 

2) It seems that the scale selection mainly concentrate on a certain scale. Will more experts lead to a better performance?

### Soundness
4

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
This paper presents a novel framework for brain network analysis that jointly models node and edge features to better understand neurodegenerative disease progression. The method introduces two core components: the Spectral Simplicial Wavelet Transform (SSWT), which enables multi-scale spectral analysis using both node and edge information, and the Spectral Mixture of Experts (SpMoE), which performs adaptive, data-dependent spectral filtering. By capturing richer spectral representations and dynamically selecting relevant frequency scales, the framework overcomes the limitations of fixed-bandwidth spectral GNNs. Experiments on benchmark brain network datasets demonstrate improved classification performance and interpretability, suggesting strong potential for advancing neurodegenerative disease analysis.

### Strengths
1. The paper proposes a novel framework extending spectral graph wavelets to simplicial complexes, enabling multi-scale spectral analysis of both node and edge features.

2. The paper demonstrates state-of-the-art performance on ADNI and PPMI datasets, with results supporting the model’s clinical interpretability and applicability to brain network analysis.

### Weaknesses
1. The analysis of model behavior appears to focus primarily on the SpMoE component, while the justification and deeper examination of the SSWT design are relatively limited. Providing more analysis or empirical evidence to support the effectiveness and design choices of SSWT would strengthen the paper and clarify its individual contribution within the overall framework.

2. The paper references Hodge-GNN [1], which also utilizes the Hodge Laplacian in its implementation, but it is not discussed in the related work or included as a baseline for comparison.

3. Additional details, such as the data splits and number of classes in the graph classification tasks, should be provided. Some results in the main table differ from those in the original paper, and these details could help clarify the discrepancies.

4. The color maps in Figure 3 could be improved, as the edges are currently too faint to be clearly identified.
 
[1] Joonhyuk Park, et al. Convolving directed graph edges via Hodge Laplacian for brain network analysis. In MICCAI, 2023.

### Questions
Please refer to the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The work is to propose a neural network framework to analyse data of the brain connectome, with an application to assess neurodegeneration. The core of the work is to develop an adaptive method to use spectral wavelets defined on simplicial complexes (nodes and edges). Signals on the nodes and on the edges are then processed through these wavelets. In this work, a method is designed to be fully end-to-end trainable to learn the optimal scales for each of them; this aspect is obtained by introducing a layer which aggregates the wavelet coefficients obtained at various scales, and estimate a weight (the “gating function”) to be used to mix the wavelet coefficients. 

The work is sound but it’s not novel. The originality is clearly overstated: the 1st claimed contribution is: “We extend the Spectral Graph Wavelet Transform (SGWT) to higher order simplicial complex via Hodge Laplacian”. This does not hold: wavelets deigned from then Hodge Laplacian were already proposed in Roddenberry et al., 2022 ; note also that NN for simplicial complexes are also designed in Ebli et al., 2020 (where, even if they are not using wavelets (and band-pass filter), one finds the same construction as the one in eq. as (7), yet for a low-pass filter). Now, the authors account for the work of Yang et al. (2022) who developed simplicial neural networks also. 

The 2nd claim is the design of the mixture of experts to select the scales. This is not that original as, the authors tell so, it’s coming from Shazeer et al. (2017) (for the gating functions, for the load balancing loss). This article has been quoted nearly 4000 times ; so incorporating it in a framework is not that original. And I have not seen any originality for this instance.

The 3rd claim is that it works. It seems so on the provided examples. However, they are specific to the domain of neuroscience and from Table 1, the results are incremental as compared to what AGT provides. 

References:

Ebli, S., Defferrard, M., Spreemann, G.: Simplicial neural networks. arXiv preprint arXiv:2010.03633 (2020)

Roddenberry, T.M., Frantzen, F., Schaub, M.T., Segarra, S.: Hodgelets: Localized spectral representations of flows on simplicial complexes.  ICASSP 2022, pp. 5922–5926 (2022)

### Strengths
The strength of the article is : 

1. The overall idea is sound, and the resulting method appears to work for the analysis of data on neurodegenerative illnesses.

### Weaknesses
The weakness of the article are numerous, as written in the summary. The claimed contributions are not important, nor actually novelties for the first two. Among weaknesses:

1. **The work is incremental**. Spectral wavelets were already considered (see at least Roddenberry et al., 2022) ; Mixture of experts is not novel ; on the task under question, AGT from 2024 obtains similar results.

2. The work provides no generalisation at all in the domain of ML: it is limited to node and edges (without any insight to higher order simplicial complexes); it fails to capture other higher-order relations (e.g. edge-to-edge relations through triangles); it is tested only on one task on brain dataset for neurogenerative disease. There are many elements of interpretation for people in neuroscience in section 5 which might be too technical for most of the audience (and at least, they are too technical for the reviewer to check their correctness and their relevance; the conference is one in ML, not in neuroscience). 

3. Section 4.1, including Lemma 1, is trivial from the existing literature (on graph signal processing with wavelets and with simplicial complexes). Also, the authors states to things which appear wrong to me:

* Section 4.1 considers “a directed graph” ; this is misleading. The graph has to be directed for the use of Hodge Laplacians, so that a direction is set to know when a positive is positive between two nodes, by the features obtained are in truth agnostic to direction. So it’s better to say that one considers an undirected graph, and then that one puts some arbitrary orientation  to account for the flows.

* “This ensures that L1 can be reliably utilized for robust graph analysis in any undirected setting.” ; I think it holds only of one uses the magnitude of the wavelet coefficients; keeping the sign will cause problems, I think.

4. In 4.2, the authors used a sampled version of the continuous wavelet transforms, for a discrete set of scales: this does not constitute a complete representation if done without any care. Shouldn’t the scale be constrained somehow ?

5. In 4.3: one should explicitly define Softmax, Softplus, what is an expert, and so one. The whole paragraph on “Gating Network” is not well explained and not sufficient, and, if one is not familiar with previous works, difficult to follow.

6. For Eq. (13), $f_{s_p}$ and $\omega_{t_q}$ should be defined with mathematical formulas, not only words underneath.

7. The role of the parameter $h$ (which appear to be crucial) is not really explained in 4.3.

8. The choices of the kernels $\mathcal{K}$ appear to be arbitrary; why these ones ? Why not combine band-pass and low-pass (scaling functions) ones as in usual WT ? Is it sound to have the same hidden dimension h for low-pass and band-pass features ?

9. For experiments, a threshold is applied on brain networks, while it is known that many results about brain connectome can be sensitive to thresholding. Did you check the robustness to thresholding ?
 
10. In 5.2, one expects that adding edge signals give more information ad more leeway (more parameters) to learn specific properties of the data. One would expect to see in the ablation study also: use of Nodes + LP only ; use of Nodes + BP only.

### Questions
I already asked my questions above.

### Soundness
2

### Presentation
2

### Contribution
1
