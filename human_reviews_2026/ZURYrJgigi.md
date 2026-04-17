# Topological Anomaly Quantification for Semi-supervised Graph Anomaly Detection

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Semi-supervised graph anomaly detection identifies nodes deviating from normal patterns using a limited set of labeled nodes. This paper specifically addresses the challenging scenario where only normal node labels are available. To address the challenge of anomaly scarcity in real-world graphs, generative-based methods synthesize anomalies by linear/non-linear interpolation or random noise perturbation. However, these methods lack a quantitative assessment of anomalies, hindering the reliability of the generated ones. To overcome this limitation, we propose a generative graph anomaly detection model based on topological anomaly quantification (TAQ-GAD). First, we design a topological anomaly quantification module (TAQ), which quantifies node abnormality through two topological metrics: The node boundary score (NBS) quantifies the boundaryness of a node by evaluating its connectivity to labeled normal neighbors. The node isolation score (NIS) assesses the structural isolation of a node by evaluating its connection strength to other nodes within the same category. This anomaly measurement module dynamically screens nodes with high anomaly scores as pseudo-anomaly nodes. Subsequently, the topological anomaly enhancement (TAE) module generates virtual anomaly center nodes and constructs their topological relationships with other nodes. Finally, the method integrates normal and pseudo-anomaly nodes on the enhanced graph for model training. Extensive experiments on benchmark datasets demonstrate TAQ-GAD’s superiority over state-of-the-art methods and effectively improve anomaly detection performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes CER-GOD, a novel graph outlier detection framework to address the '' Homophily Trap'', which is a critical issue where graph convolutional operations blur the feature representations of normal and anomalous nodes that are neighbours. The proposed method synergistically integrates two main components: a self-discriminative masking spoiler that learns to reweight graph edges to suppress contaminating information flow from heterogeneous neighbours, and a clustering-based outlier detector that generates unsupervised pseudo-labels to guide this reweighting process. To ensure stable training and prevent clustering collapse, a diversity loss is introduced as a regularization term. The proposed CER-GOD method is jointly optimized as an end-to-end framework. Extensive experiments on multiple benchmark datasets demonstrate that CER-GOD significantly outperforms a wide range of state-of-the-art baselines.

### Strengths
1. The paper addresses a critical and well-articulated problem in GNN-based anomaly detection: the "Homophily Trap". The authors provided clear motivation, supported by insightful empirical analysis (Figure 1), highlighting how neighbourhood aggregation can contaminate node representations and fundamentally hinder outlier identification.
2. The proposed CER-GOD framework is methodologically sound and the idea is novel. The main innovation lies in the synergistic joint optimization of a self-discriminative masking spoiler and a clustering-based detector. This design creates a powerful feedback loop where pseudo-labels from clustering guide the edge reweighting, and the refined graph structure, in turn, yields more discriminative embeddings for improved clustering.
3. The comparison against a comprehensive set of baselines fully validates the effectiveness of CER-GOD. Additionally, the authors provided convincing qualitative evidence, such as t-SNE and mask visualizations, to further enhance the  persuasiveness and interpretability of the approach.
4. The authors provided the implementation code of the proposed method, which increases the reproducibility.

### Weaknesses
1. The choice of the Chebyshev distance for the MMD kernel calculation should be elaborated. While the intuition is provided, the paper would be strengthened by an empirical comparison against a more conventional Euclidean-based RBF kernel to justify this specific design.
2. The diversity loss $l_{diversity}$ introduces a crucial hyperparameter $\epsilon$ to control the minimum proportion of samples per cluster. However, there is no discussion regarding the setting of $\epsilon$ or any corresponding parameter sensitivity analysis.
3. The paper distinguishes the edge reweighting mechanism from graph rewriting methods. Could the authors further elaborate on the fundamental advantages of such learnable masks over heuristic-based hard edge removal or addition? A deeper discussion would be valuable.
4. The framework relies on a feedback loop where pseudo-labels guide the mask optimization. In the early training phase, these pseudo-labels might be noisy, which potentially leads the model to a suboptimal solution. How does the model ensure stability during the training phase? Is there a warm-up phase, or does the joint optimization naturally navigate towards a good solution?

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
4

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
This paper proposes TAQ-GAD, a semi-supervised graph anomaly detection method designed for scenarios where only normal nodes are labeled. The approach introduces a Topology-Aware Quantification (TAQ) module to measure each node’s boundary and isolation scores, identifying potential anomalies based on structural characteristics. It further employs a Topology-Aware Enhancement (TAE) module that creates virtual anomaly centers and applies risk-aware pseudo-labeling to strengthen anomaly representation. Experiments on multiple benchmark datasets show that TAQ-GAD significantly outperforms existing methods, demonstrating strong robustness and generalization ability.

### Strengths
The paper tackles a challenging semi-supervised graph anomaly detection setting where only normal nodes are labeled; this is a realistic and underexplored problem.

The proposed TAQ-GAD framework is conceptually clear, combining structural quantification (NBS/NIS/PIS) with topology-aware data augmentation (TAE).

The TAQ module provides interpretable metrics that capture both boundary proximity and structural isolation, which are intuitive and effective.

### Weaknesses
1. Methodological Limitation — Not Realistic for True Anomaly Detection:

The proposed TAQ-GAD framework heavily relies on neighborhood label statistics, especially through its Node Boundary Score (NBS) and Node Isolation Score (NIS).However, in real anomaly detection scenarios:
* Anomalies are inherently rare, often less than 1–5% of total nodes.
* The assumption that “normal nodes cluster together and anomalies cluster together” is unrealistic — in many graphs (e.g., financial fraud, cybersecurity, IoT networks), anomalies are structurally mixed within normal communities.
* Consequently, computing NBS or NIS using the local density of labeled normal nodes implicitly assumes a strong topological separation between normal and anomalous nodes, which may not exist in practice.

TAQ-GAD is not learning “anomalousness” from data, but rather imposing a strong heuristic constraint that fits specific benchmark graphs.

2. Label Dependency — Not Truly Semi-Supervised

Although the paper claims to address the semi-supervised anomaly detection setting, its implementation contradicts this claim:

* The paper uses 60% / 20% / 20% train/validation/test splits, meaning a large fraction of nodes (including many normals) are labeled for training.

* In contrast, GGAD and other baseline works typically use only 15% labeled normal nodes to simulate a truly low-label semi-supervised scenario.

* Furthermore, both works control labeled ratio via the same variable ρ, but ρ in TAQ-GAD refers to a much larger absolute label count, making the comparison unfair.

TAQ-GAD’s strong performance likely stems from having access to far more label information, not from the novelty of its topology quantification.

3. The definitions of NBS and NIS inherently bias the model toward datasets with: Clear community separation between normal and anomalous nodes, homogeneous degree distributions (since both metrics rely on K-hop neighbor density). This structural bias means the method may overfit to benchmark datasets like Amazon or Reddit, which have cleaner topologies, and fail on real-world heterogeneous graphs (e.g., fraud networks, financial transaction graphs).

4. Because both TAQ and TAE modules are built upon graph topology and labeled-normal density, the framework is not agnostic to graph structure. The “topology-aware” quantification might work as a feature engineering trick in specific graphs, but lacks generalizability across: Dynamic or time-evolving graphs, Multi-relation heterogeneous graphs, or Graphs with partial/noisy labeling.

### Questions
Factually speaking, datasets in the graph domain often have their own particularities depending on the scenario and task. For example, in the datasets used in this paper, the clear separation between normal and anomalous nodes largely comes from the way researchers artificially define and split the data for different tasks. This setting is far from real-world conditions. Therefore, I believe the authors should consider developing a more general and realistic approach, rather than designing a method that only fits the specific characteristics of these benchmark datasets.

### Soundness
3

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
4

### Summary
This paper tackles semi-supervised Graph Anomaly Detection (GAD) with only normal node labels. The proposed framework, TAQ-GAD, introduces two topological metrics—Node Boundary Score (NBS) and Proxy Isolation Score (PIS)—to identify pseudo-anomalies and assign pseudo-labels. A Topological Anomaly Enhancement (TAE) module further generates virtual anomaly centers and their topological relations, improving the quality of pseudo-anomaly generation.

### Strengths
1. The logical flow of the paper effectively guides the reader through the proposed solution and its evaluation.
2. Formal definitions and theoretical analysis of the proposed metrics enhance interpretability and reliability. 
3. The experimental evaluation is fairly comprehensive, covering multiple datasets and various baseline methods.

### Weaknesses
1. The proposed method is topology-centric, and performance may degrade on graphs with noisy or unreliable structure. Feature-level anomalies may not be captured when the detection relies solely on topology, which could limit robustness in cases where structural information is noisy or uninformative.
2. While PIS is claimed to be a key component of TAQ-GAD, its weight in the scoring function is set almost negligibly compared to NBS. Specifically, the implementation fixes $\lambda_{1} = 1$ for NBS and $\lambda_{2} = 0.001$ for PIS across all datasets. This large imbalance (1000:1 ratio) raises doubts about whether PIS truly contributes to the model. Moreover, no sensitivity analysis on λ₁ and λ₂ is provided, making it unclear if PIS has any substantive impact on performance.

### Questions
1. In Table 2, the ablation study reports results for “+NBS”, “+NIS”, and “+NBS+NIS”. However, according to Equation (5), the pseudo-anomaly scoring in TAQ-GAD is actually based on NBS and PIS, rather than NIS. Could you clarify whether the ablation table contains a writing error, or if NIS was intentionally used instead of PIS in the ablation experiments?
2. Although $\alpha$ and $\beta$ show robustness, the selection of $\tau$ (pseudo-anomaly ratio) can heavily influence performance and may require dataset-specific tuning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TAQ-GAD, a semi-supervised graph anomaly detection framework that quantifies node abnormality using two topological metrics: Node Boundary Score (NBS) and Node Isolation Score (NIS). These metrics guide pseudo-anomaly selection, while the Topological Anomaly Enhancement (TAE) module refines them through risk-based label flipping and virtual anomaly centres. The paper achieve significant performance improvement on the selected datasets.

### Strengths
1. The proposed method is fairly easy to follow.
2. The framework plot is informative, making it easier to understand the overall workflow.
3. The proposed methods achieved competitive performance on the selected dataset.

### Weaknesses
1. My understanding is that the proposed method mainly achieves performance gain through pseudo labelling. However, I didn’t find (at least the paper not specifically mentioned) any comparison with training SOTA supervised GAD methods on pseudo labels generated by naïve pseudo labelling strategies. This is essential for evaluating whether the proposed pseudo labelling methods are more effective.

2. The authors mentioned unsupervised methods, “their heavy reliance on intrinsic graph structures to define anomalies introduces fundamental ambiguity, often failing to distinguish genuine semantic anomalies from rare yet normal patterns.” However, the proposed method leverages node degree and graph homophily. Why are these graph properties more robust for GAD?

3. In Eq. 9 the regularisation term is similar to weight decay. Why are two balancing parameters required?

4. The baseline selection, including unsupervised and semi-supervised baselines, adapts some well-known unsupervised methods for semi-supervised evaluation. However, how they are adapted for semi-supervised GAD is not clearly discussed. This is important for fair comparison. In addition, some results appear to be missing in Table 1. Also, most of the baselines are unsupervised methods; it would make the evaluation stronger if supervised and more semi-supervised methods were studied.

5. In the related work section, the semi-supervised methods part mixes methods that use labelled anomalies and those that only use labelled normal training data. They should be discussed separately. Other than BWGNN, there are a lot more recent supervised methods that are not discussed. Also, recent generalist methods have shown promising cross-dataset performance. The relationship of the paper’s setting with those methods' should be discussed.

6. In the references, there are quite a few papers that were listed as arXiv preprints but are actually published papers in well-regarded conferences and journals.

### Questions
Please refer to my weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
