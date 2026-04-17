# Physics-Inspired All-Pair Interaction Learning for 3D Dynamics Modeling

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 2

## Abstract
Modeling 3D dynamics is a fundamental problem in multi-body systems across scientific and engineering domains and has important practical implications in trajectory prediction and simulation. While recent GNN-based approaches have achieved strong performance by enforcing geometric symmetries, encoding high-order features or incorporating neural-ODE mechanics, they typically depend on explicitly observed structures and inherently fail to capture the unobserved interactions that are crucial to complex physical behaviors and dynamics mechanism. In this paper, we propose PAINET, a principled SE(3)-equivariant neural architecture for learning all-pair interactions in multi-body systems. The model comprises: (1) a novel physics-inspired attention network derived from the minimization trajectory of an energy function, and (2) a parallel decoder that preserves equivariance while enabling efficient inference. 
Empirical results on diverse real-world benchmarks, including human motion capture, molecular dynamics, and large-scale protein simulations, show that PAINET consistently outperforms recently proposed models, yielding 4.7% to 41.5% error reductions in 3D dynamics prediction with comparable computation costs in terms of time and memory. Our codes, baseline models and datasets are available at https://github.com/Icarus1411/PAINET.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses a key limitation of existing GNN-based methods in 3D dynamic modeling of multi-body systems: over-reliance on explicit observed structures, which fails to capture unobserved all-pair interactions critical for complex physical behaviors. It proposes PAINET, an SE(3)-equivariant neural architecture and evaluates on three real-world benchmarks: human motion capture (CMU dataset), molecular dynamics (MD17 dataset), and large-scale protein simulation (Adk dataset), comparing with classic models (Linear, RF, MPNN, EGNN) and state-of-the-art (SOTA) models (EGNO, HEGNN, GF-NODE).

### Strengths
1. model operation deisng with physical priors
2. several benchmark evaluations on different scenarios
3. good efficiency

### Weaknesses
1. Incomplete Theoretical Analysis​
Insufficient explanation of the energy function’s physical meaning: While extended from Zhou et al.’s (2004) quadratic energy, its connection to real physical systems (e.g., van der Waals forces, protein hydrogen bonds) is unclear. The concavity assumption of the ρᵢⱼ function only cites Yang et al. (2021) without analyzing its validity boundaries in 3D dynamic scenarios (e.g., particle type differences, dynamic interaction changes), raising doubts about applicability in extreme cases (e.g., strong nonlinear force-driven systems).​
Though SE(3) equivariance is fully proven, model performance under complex symmetries (e.g., mirror, scaling symmetry) common in real physics is unaddressed. Additionally, equivariance preservation of adaptive pairwise mappings Φ and Ψ in the attention network is not verified individually; while overall equivariance holds, the symmetry transfer mechanism of individual modules remains unclear.​
2. Room for Improvement in Experimental Design​
Molecular dynamics experiments ignore force field impacts (e.g., CHARMM, AMBER), only validating on MD17’s quantum force field, limiting generalization to classical force fields. Testing focuses on 8 common small molecules, lacking complex molecules (e.g., metal-containing molecules, polymers), restricting applicability in specialized chemical scenarios.​
Protein simulation relies solely on Adk protein equilibrium trajectories, excluding other protein types (e.g., membrane proteins, antibodies) or non-equilibrium dynamics (e.g., protein folding, ligand binding), insufficiently evaluating performance in complex conformational changes. Multi-step prediction only tests T=5, without exploring error accumulation at longer steps (e.g., T=10, T=20), inadequately validating long-term prediction ability.
3. More evalutions on large molecules, such as MD22.

### Questions
The authors should seriously address the concerns shown in Weakness point by point to improve the quality of the paper.

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
Current state-of-the-art models for 3D dynamics prediction (e.g., GNNs, EGNNs) rely on explicitly observed structures between particles. This limits their ability to capture crucial unobserved interactions (e.g., long-range forces, dynamically forming structures), leading to errors in long-term trajectory prediction and simulation, especially in complex systems like molecular dynamics and protein folding. A principled, SE(3)-equivariant neural architecture designed to model all-pair interactions, including unobserved ones.

### Strengths
Models Unobserved Interactions: This is its core innovation. Unlike previous GNN-based models that rely on fixed, observed structures (e.g., distance-based graphs), PAINET is designed to capture latent, all-pair interactions (e.g., long-range forces, dynamically forming bonds), which is crucial for accurate long-term dynamics prediction.

Principled, Physics-Inspired Formulation: The model is not a purely black-box architecture; it is derived from the minimization of an energy function, providing a theoretical foundation for its attention mechanism and linking it to physical principles.

### Weaknesses
A more clear description on the computational cost should be described. See Questions.

### Questions
1) Can you explain why the line in Figure 7 shows linear complexity with respect to particle numbers but not O(N^2)?
2) The training and inference time should be compared among the baseline.
3) Could you help to explain how to connect the motivated examples in the introduction with the examples in experiments? Are there prior knowledge on the all-pair interations in experiments?
4) I am not an expert in molecular dynamics and can not justify the significance of the accuracy improvement. Therefore, I will refer other reviewers' comments on it.

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
This paper introduces PAINET (Physics-Inspired All-Pair Interaction Network) for 3D dynamics modeling across molecular, protein, and motion capture datasets. The method formulates an explicit energy function whose gradient defines pairwise attention weights, ensuring each network layer corresponds to an energy descent step. The encoder models implicit all-pair interactions, while the decoder uses an equivariant EGNN for SE(3)-consistent trajectory prediction. The authors claim PAINET learns latent long-range interactions without explicit graph structures, outperforming prior GNN-based models such as EGNN, HEGNN, EGNO, and GF-NODE.

### Strengths
1. The attention update is rigorously derived from an energy descent principle, providing a clear physical interpretation. The energy-derived attention mechanism replaces softmax with gradient-based weights, improving interpretability.
2. Experiments show consistent gains across motion, molecular, and protein dynamics tasks at comparable computational cost.
3. The paper is well-organized and readable, with clear motivation and intuitive explanations.

### Weaknesses
1. The reported results for key baselines, particularly EGNO and GF-NODE, appear inconsistent with their original publications, casting doubt on the validity of the claimed improvements. For example, in Table 1, PAINET reports EGNO achieving F-MSE = 14.2 (×$10^{-2}$) on Walk and F-MSE = 4.15 (×10) on Run—values that deviate substantially from EGNO’s original paper (F-MSE = 8.1 (×10-2) on Walk and F-MSE = 3.39 (×$10^{-1}$) on Run). Similar inconsistencies appear for GF-NODE and other baselines, likely due to preprocessing or implementation differences. These discrepancies need careful clarification.
2. The paper lacks ablations isolating the contributions of the energy function, Φ/Ψ mappings, and all-pair interactions. It remains unclear whether performance gains stem from the proposed energy mechanism or general architectural effects.

Missing related work and baselines:
1. SE(3) Equivariant Graph Neural Networks with Complete Local Frames; ICML 2022;
2. AlphaNet: Scaling Up Local Frame-based Atomistic Foundation Model, Npj Comput. Mater. (2025)

### Questions
1. Please clarify the differences in preprocessing, subject selection, frame rate, or hyperparameters that might explain the inconsistent EGNO and GF-NODE results.
2. How sensitive is the model to the specific choice of potential function ρ? Did you test other learnable monotonic forms beyond the quadratic–quartic setup?
3. Could you include or discuss comparisons removing the energy weighting, Φ/Ψ mappings, or limiting interactions to local neighbors to substantiate the claimed benefits?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes PAINET, a principled SE(3)-equivariant architecture for modeling all-pair interactions in multi-body dynamical systems. The model comprises: (1) a novel physics-inspired attention mechanism derived from the minimization trajectory of an energy function, and (2) a parallel decoder designed to maintain equivariance while enabling efficient inference. Empirical evaluation across diverse real-world benchmarks demonstrates the effectiveness of the proposed approach.

### Strengths
* The integration of all-pair interactions with a parallel, equivariant decoder is technically well-founded and appears to be effectively implemented.
* PAINET demonstrates strong empirical performance, outperforming existing baselines on several challenging multi-body system benchmarks.

### Weaknesses
* Several key aspects of the methodology are insufficiently motivated or explained, making it difficult to follow.
  * The rationale for regularizing the distance between updated and current node embeddings (lines 184-185) is unclear. What specific issue does this address, and how was the regularization strength chosen?
  * The description of the underlying physical principle (lines 192-199) is presented without a clear connection to the model's technical design. It is not evident how this principle directly informs the architecture or contributes to the reported performance.
  * The introduction of the functions $\phi_{ij}$ (line 211 and Eq. 7) and  $\psi_{ij}$ (Eq. 7) appears abrupt. The logical flow and the specific roles these functions play in the overall framework need to be more clearly articulated.

* The paper lacks a well-defined, physically meaningful accuracy threshold for the multi-body dynamics tasks. Without such a benchmark, it is difficult to assess whether the reported performance improvements translate to practical utility in real-world applications.

### Questions
* How was the number of trajectory steps $T$ determined for different S2T tasks (e.g., $T=5$ in Table 1)? Was this hyperparameter tuned on a validation set, and is it consistent across systems with different dynamical properties?
* For predicting trajectories longer than the predefined $T$, what is the inference procedure? Does the model operate autoregressively, and if so, how are potential error accumulations mitigated?

### Soundness
2

### Presentation
1

### Contribution
2
