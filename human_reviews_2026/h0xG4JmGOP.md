# GDEGAN: Gaussian Dynamic Equivariant Graph Attention Network for Ligand Binding Site Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Accurate prediction of binding sites of a given protein, to which ligands can bind, is a critical step in structure-based computational drug discovery.  Recently, Equivariant Graph Neural Networks (GNNs) have emerged as a powerful paradigm for binding site identification methods due to the large-scale availability of  3D structures of proteins via protein databases and AlphaFold predictions. 
The state-of-the-art equivariant GNN methods implement dot product attention, disregarding the variation in the chemical and geometric properties of the neighboring residues. To capture the variation in properties, we propose GDEGAN (Gaussian Dynamic Equivariant Graph Attention Network), which replaces simple dot-product attention with adaptive kernels that recognize binding sites. The proposed attention mechanism captures variation in neighboring residues using statistics of their characteristic local feature distributions.  Our mechanism dynamically computes neighborhood statistics at each layer, using local variance as an adaptive bandwidth parameter with learnable per-head temperatures, enabling each protein region to determine its own context-specific importance.
Our model shows better predictive performance, outperforming existing methods with relative improvements of 37-66 \% in DCC and 7-19 \% DCA success rates across COACH420, HOLO4k, and PDBBind2020 datasets. These advances have direct application in accelerating protein-ligand docking by identifying potential binding sites for therapeutic target identification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces GDEGAN, a novel Gaussian Dynamic Equivariant Graph Attention Network designed for predicting protein-ligand binding sites. The core innovation is the replacement of standard dot-product attention with a Gaussian Dynamic Attention mechanism. This new mechanism adapts to the local chemical and geometric heterogeneity of protein surfaces by using the mean and variance of neighboring residue features to compute attention scores, leading to state-of-the-art performance on three established benchmark datasets.

### Strengths
The novelty lies in the successful adaptation and application of a probabilistic, variance-aware attention mechanism to the domain of 3D equivariant graph representations for a critical bioinformatics task. While building upon a strong backbone (GotenNet), the proposed attention module is a distinct and impactful innovation. It provides a more physically grounded inductive bias by assuming that variance in learned features is a meaningful signal, a departure from standard similarity-based dot-product attention.

The model achieves substantial and consistent improvements over strong baselines, including the current state-of-the-art EquiPocket, across three diverse datasets (COACH420, HOLO4k, PDBbind2020). The reported relative gains (e.g., 37-66% in DCC) are compelling.

### Weaknesses
**Key Flaw:** The most critical weakness is the ambiguity in the formulation of the core Gaussian attention mechanism. Specifically, the dimensionality of the neighborhood statistics μ_i and (σ_i)^2 in Equations 5 and 6 is unclear in the context of Equation 7. Since h_j is a high-dimensional feature vector, μ_i and (σ_i)^2 should also be vectors (element-wise mean and variance). However, Equation 7 uses (σ_i)^2 as if it were a scalar value for modulating the attention kernel's bandwidth. This lack of clarity is a major impediment to understanding and reproducing the method and casts doubt on its technical soundness. The authors must explicitly define how the vector variance is converted into the scalar used in the denominator.

**Methodological Issues:** The central hypothesis of the paper, while intuitive, could be better substantiated. The authors assume that high local feature variance is a reliable signal for binding sites and that standard dot-product attention cannot capture this.

1. Justification of Hypothesis: This assumption is presented as a given but lacks direct empirical or theoretical support. Is there prior work suggesting a strong correlation between feature variance and functional sites?

2. Expressive Power of Baselines: The paper argues that multi-layer GNNs with standard attention are insufficient. However, it is plausible that a sufficiently deep model could learn to approximate similar context-aware behavior implicitly. The paper does not provide a compelling argument or experiment to rule out this possibility.

**Experimental Evaluation Issues:** The experimental section is strong but could be improved.

1. Training Dynamics: The introduction of a data-dependent variance term (σ_i)^2 in the denominator of an exponential function could potentially lead to training instability (e.g., vanishing or exploding gradients) if the variance becomes very small or large. The paper does not discuss the training dynamics or present loss curves to demonstrate that the model converges as robustly as the baseline.

2. Lack of Parameter Sensitivity Analysis: The model introduces a learnable temperature parameter ξ for each attention head. An analysis of the model's sensitivity to this hyperparameter would strengthen the results and provide insights into the mechanism's behavior.

### Questions
Clarifying these issues will be crucial for a more thorough assessment of the paper's quality.

1. Clarification of Equations 5-7: This is the most critical point. Please provide a precise mathematical definition for how the neighborhood variance (σ_i)^2 is used in Equation 7. Given that h is a vector, (σ_i)^2 from Equation 6 should also be a vector. How is this vector transformed into the scalar value required in the denominator of Equation 7? Is it the mean of the vector elements, their L2 norm, or something else?

2. Justification of the Core Hypothesis: Could you provide further justification for the core assumption that local feature variance is a superior signal for identifying binding sites compared to what can be learned by standard attention mechanisms? Perhaps you could show a correlation analysis on a validation set between the learned (σ_i)^2 values and ground-truth binding site locations.

3. Training Stability: Did the use of the (σ_i)^2 term in the attention calculation lead to any training instability? Could you please present the training and validation loss curves for GDEGAN and the GotenNet(full) baseline to demonstrate that the proposed mechanism allows for stable convergence?

4. Computational Overhead: Remark 2 discusses the theoretical computational complexity. Could you provide the empirical wall-clock inference and training time overhead of the Gaussian Dynamic Attention layer compared to the standard dot-product attention layer in GotenNet? This would give a clearer picture of the practical trade-offs.

### Soundness
2

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
The authors propose a variation of vanilla dot product attention based on adaptive kernels as a motivation to capture local geometric and chemical features of the residues to predict protein-ligand binding sites. The approach calculates mean and variance of a residue's neighborhood on the fly. They show the approach combined with ESM embeddings as node features  significantly outperform previous SOTA GotenNet that was based on dot product attention. 

The authors claim that current best ligand binding site prediction methods that are based on message passing networks use context-agnostic attention mechanisms and the binding sites in a protein are often clustered based on their local geometric and chemical features. They construct a protein structure graph with residues as nodes and edges between the residues determined by a threshold on C-alpha distances. And the third component of the graph is the C-alpha coordinates. They initialize the nodes using ESM-2 node embeddings and then project to hidden dimensional space using learned transformations. Nodes are labeled 1 or 0 based on closeness to ligand atoms. The task is given a protein graph, predict the binding probability of each residue. The authors adopt Gotennet and modify the attention part and representations for scalars and tensors. They design basis functions based on spherical harmonics to preserve equivariance. They then use these steerable features and the invariant scalars from the projected pLM embeddings to create the message passing networks for both nodes and edges. The node features go through a dot product with the RBF features ensuring differentiability. Steerable features are initialized to 0 and then updated during training. The node features (from pLM) are used to compute mean and variance on the fly. 

Experiments are conducted on three benchmark datasets and an additional ablation study is performed.

### Strengths
The paper is well organized in terms of the limitations of the current binding site prediction models. The adoption of GotenNet and modifying the vanilla attention with the proposed approach to make the attention more dynamic and aware of an atom's local neighborhood is a an interesting approach. The key contribution is the idea of computing a neighboring atom’s features from Gaussian distribution defined by the target atom’s local neighborhood in the model. The results in Table 1 show that GDEGAN beat the baseline models on three benchmark datasets.

### Weaknesses
From the results in Table 1 it shows the proposed method beats GotenNet on all three datasets except on the failure rate. However, from the ablation study shows (in Table 2) that  the main boost comes from the ESM embeddings for both methods. As the authors show that the proposed approach is beneficial as structural heterogeneity increases. Since protein ligand binding site is determined by the chemical fingerprint it would be interesting to see if the method relied no only on the C-alpha atoms but an all atom graph model like GearBind. 

If that is a significant stretch the authors could also try using embeddings from structure aware protein models such as Prostt5 or SaProt or even surface-structure aware protein models such as AtomSurf ? If the main hypothesis is that the Gaussian kernels give the additional boost in performance by capturing the local chemical and geometric features then one could test that by extracting residue embeddings from the advanced protein models that are trained on structure and surface features constructed from local neighborhoods. Without that comparison it is hard to determine if the proposed approach is the optimal method to capture local geometric and chemical characteristics of the binding residues.  

In any case when referring to chemical and geometric features the authors did not cite a few other relevant papers:
1. MaSIF: https://www.biorxiv.org/content/10.1101/606202v1.full.pdf
2. AtomSurf: Which combines structure and surface (https://arxiv.org/pdf/2309.16519)
3. GearBind: https://www.biorxiv.org/content/10.1101/2023.08.10.552845v1

### Questions
Please add other SoTA methods on protein ligand binding tasks such as HoloProt (https://arxiv.org/pdf/2204.02337)

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
3

### Summary
The paper introduces GDEGAN, a Gaussian Dynamic Equivariant Graph Attention Network for protein–ligand binding site prediction. Its central idea is to replace dot-product similarity with a Gaussian kernel whose weights are determined by local neighborhood statistics and a learnable temperature, implemented within an SE3-equivariant graph architecture. The approach targets the strong geometric and chemical heterogeneity of protein surfaces and includes a clear treatment of symmetry, noting that the use of ESM-2 scalar features yields SE3 rather than full E3 equivariance. The training objective addresses class imbalance and directional cues. Experiments on COACH420, HOLO4K, and PDBbind2020 report consistent gains in DCC and DCA, substantial reductions in failure rate, and faster inference versus strong baselines. Attention visualizations align with pocket regions, offering an interpretable account of model behavior.

### Strengths
Strength 1：Uses a local Gaussian kernel with adaptive bandwidth from neighborhood statistics and a learnable temperature, yielding context-aware attention suited to heterogeneous protein surfaces.
Strength 2：Provides formal analysis showing the proposed attention preserves SE(3) equivariance under the chosen feature representation, giving a clear geometric justification.
Strength 3 ：Demonstrates consistent improvements over strong baselines across standard pocket benchmarks, supported by ablations and qualitative visualizations, with competitive or better inference efficiency.

### Weaknesses
Weakness 1： The paper claims to capture geometric structure and handle variation among neighboring residues, but the evidence is mostly indirect. It should demonstrate which previously hard geometric challenges are now addressed, with targeted analyses rather than only aggregate metrics and visuals.
Weakness 2：Comparisons with prior graph-attention variants are incomplete, especially kernelized attention methods. A deeper analysis against these baselines is needed to substantiate the claimed contribution and clarify what is genuinely new.
Weakness 3：Key terms should be standardized for readability, including “Gaussian kernel,” “Gaussian attention,” and “Protein-aware Structural Embeddings.” A thorough pass is recommended. Also correct the misspelling of “temperature” in the figures.
Weakness 4：The proposed variant may introduce extra computational cost. The paper should provide complexity measurements and hyperparameter sensitivity analyses to quantify overhead and practical trade-offs.

### Questions
Q1：Can you report how the learnable temperature evolves and distributes during training? 
Q2: How does using a learnable temperature compare to a fixed bandwidth parameter?
Q3：Which specific geometric structures previously handled poorly by dot-product or standard graph attention are now captured better? 
Additional questions and suggestions please refer to the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
