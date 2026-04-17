# Permutation-Invariant Spectral Learning via Dyson Diffusion

- Decision: Reject
- Scores: 6, 6, 4, 4, 6

## Abstract
Diffusion models are central to generative modeling and have been adapted to graphs by diffusing adjacency matrix representations. The challenge of having up to $n!$ such representations for graphs with $n$ nodes is currently partially mitigated by using permutation-equivariant learning architectures. However, despite their computational efficiency, existing graph diffusion models struggle to distinguish certain graph families, unless graph data are augmented with ad hoc features.
This shortcoming stems from enforcing the inductive bias within the learning architecture.
In this work, we leverage random matrix theory to analytically extract the spectral properties of the diffusion process, allowing us to push the inductive bias from the architecture into the dynamics. Building on this, we introduce the Dyson Diffusion Model, which employs Dyson's Brownian Motion to capture the spectral dynamics of an Ornstein–Uhlenbeck process on the adjacency matrix while retaining all non-spectral information.
We demonstrate that the Dyson Diffusion Model can accurately learn the spectrum, outperforming existing graph diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a new way to do graph diffusion by moving permutation invariance from the neural architecture to the diffusion dynamics. Instead of diffusing adjacency matrices with GNN-based architectures, the authors use random matrix theory and Dyson Brownian Motion to directly model and generate the eigenvalue trajectories of graphs.

### Strengths
**1. Novel problem formulation**

The paper introduces a new perspective on graph diffusion by shifting permutation invariance from neural architecture to stochastic dynamics. Leveraging Dyson Brownian Motion to model spectral evolution is grounded in random matrix theory, offering a principled alternative to the WL-limited message-passing paradigm.

**2. Addresses a fundamental limitation in graph generative models**

The work directly tackles the expressivity constraints of GNN-based diffusion models and clearly demonstrates how current architectures fail on WL-equivalent graphs. The proposed DyDM avoids these blind spots and captures spectral distributions that traditional models struggle with.

### Weaknesses
**1. Limited scope: generates spectra, not full graphs**

Experiments merely demonstrates spectral generation. No adjacency-level reconstruction or topology sampling is shown, so the model currently functions as a spectral generator, not a full graph generator. This may limit perceived practical impact unless downstream graph construction is demonstrated.

**2. Scalability and efficiency unclear**

The Dyson SDE requires adaptive step control and an equilibrium shooting rescue mechanism. The paper lacks of computational complexity analysis, training/inference time comparison and memory usage reports.

**3. Evaluation narrowly focused on spectrum fidelity**

Evaluaition mainly focus on spectral statistics (i.e., mean and marginal Wasserstein distance). More broader evaluation could be better. e.g., dowmstream task performance (i.e., graph generation), visual or qualitative graph samples.

**4. Limited benchmarks**

Experiments emphasize synthetic WL-equivalent cases and small community/brain datasets (i.e., for Brain datasets of size 5 to 10 vertices). No results on widely used graph-gen benchmarks (e.g., ZINC, Planetoid citation graphs, Proteins). This makes generalization claims harder to assess.

### Questions
1. Is full graph reconstruction via the eigenvector SDE currently feasible, or is DyDM’s scope limited to spectral generation at this stage?

2. Dyson dynamics have singular drift near eigenvalue collisions. While adaptive control handles this, how robust is the proposed solver under near-degenerate spectra, such as graphs with high symmetry or repeated eigenvalues?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes the Dyson Diffusion Model (DyDM), which analytically decomposes an Ornstein–Uhlenbeck diffusion on graph adjacency matrices into eigenvalue (spectrum) and eigenvector dynamics via random matrix theory: eigenvalues follow Dyson Brownian Motion (permutation-invariant), so the score can be learned with any architecture while preserving non-spectral information through an eigenvector SDE—yielding more accurate spectrum learning than GNN/transformer graph diffusion and mitigating GI/WL expressivity blind spots.

### Strengths
1. Principled permutation invariance via dynamics: Shifts the inductive bias from architecture to the SDE itself (DBM), enabling architecture-agnostic spectral learning and avoiding GI/WL limitations.

2. Information-preserving and effective: Retains non-spectral content (theorem-backed) and empirically outperforms GNN/transformer baselines on spectrum predict

### Weaknesses
1. Scope limited to spectral generation: The current experiments focus on generating graph spectra, without demonstrating adjacency-level reconstruction or topology sampling. As a result, the method operates primarily as a spectral generator rather than a full graph generative model. Demonstrating downstream graph construction would strengthen the practical impact.

2. Scalability and computational efficiency not established The proposed approach involves adaptive step-size control and a shooting mechanism for stability, yet the paper does not report computational complexity, runtime comparisons, or memory usage. Without such analysis, it remains unclear how well the method scales to larger graphs or real-world workloads.

### Questions
See the weakness

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
The paper proposes a new diffusion model for graph generation, called the Dyson Diffusion Model (DyDM), which aims to achieve a permutation-invariant diffusion process.
The model is based on the Dyson Brownian Motion, a modified version of the Ornstein–Uhlenbeck (OU) process that includes a repulsive force between eigenvalues, allowing the diffusion to evolve in the spectral domain in a physically coherent manner.
The central idea is to shift permutation invariance from the model architecture (as done in GNNs or transformers) to the diffusion dynamics itself, by operating directly on the spectrum of the graph adjacency matrix.

### Strengths
The paper is well written and clearly organized.
The proposed formulation is interesting and promising, as it represents a relevant step toward a better understanding of permutation-invariant graph generation based on spectral properties.
Moreover, the derivation grounded in the Dyson Brownian Motion provides the model with a solid theoretical foundation (although I did not verify every formula in detail).

### Weaknesses
The work introduces a theoretically elegant idea, but unfortunately, it falls short in terms of experimental validation (expected for this venue).
- The model operates only on eigenvalues and does not reconstruct graphs. This limits its practical relevance, as it only generates spectral distributions, not concrete graph structures.
- The experimental evaluation is limited, as it does not include standard benchmarks (e.g., QM9, ZINC, ENZYMES) nor significant competitors in spectral generation (e.g., SPECTRE, GGSD).
- A brief intuitive explanation of the Ornstein–Uhlenbeck process would improve clarity, since it represents the theoretical basis of the model.
- The advantage of using Dyson diffusion over other spectral models (e.g., those based on Transformers) should be better motivated.
In particular, both types of models are permutation-invariant, but differ only in how they treat the physics of the spectrum: the Dyson Diffusion Model does not introduce a new kind of invariance, but rather a more physically consistent formulation of diffusion. However, it does not demonstrate a clear empirical benefit over existing approaches.
- The related work section omits important recent studies on spectral reconstruction, such as:

  -Minello, Giorgia, Alessandro Bicciato, Luca Rossi, Andrea Torsello, and Luca Cosmo. Generating Graphs via Spectral Diffusion. Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025).


  -Martinkus, Karolis, Andreas Loukas, Nathanaël Perraudin, and Roger Wattenhofer. Spectre: Spectral Conditioning Helps to Overcome the Expressivity Limits of One-Shot Graph Generators. ICML 2022.

### Questions
The authors mention that the approach could be extended to other matrices beyond the adjacency one (e.g., Laplacian or normalized Laplacian).
I was wondering whether the authors conducted any preliminary exploration to at least gain some intuition about how the method would behave with these alternative spectral representations.

*Minor* Line 77, page 2: “we show that an OU diffusion on the graph can be dissected into diffusion of the …”
The acronym OU, referring to the Ornstein–Uhlenbeck process, should be explicitly defined when first introduced.

### Soundness
3

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
3

### Summary
This work introduces the Dyson Diffusion Model (DyDM), a novel approach that learns graph spectra through an Ornstein–Uhlenbeck-driven diffusion process without relying on GNNs or graph transformers. DyDM preserves full graph information and enables the formulation of an eigenvector SDE. Experiments show that DyDM outperforms existing spectral learning methods and highlights the limitations of GNN-based graph diffusion models.

### Strengths
1. The paper introduces a novel diffusion model that enables graph spectrum learning without relying on GNNs or Transformers.

2. The model demonstrates strong empirical performance on distinguishing WL-equivalent but non-isomorphic graphs

### Weaknesses
* As mentioned in paper, "work on the set of symmetricreal matrices". The method is limited applicability to general graph types, such as such as directed or attributed graphs.

* Lack of structural recovery evaluation makes it unclear how effectively the model can reconstruct full graph structures.

* The evaluation is limited to statistical fidelity of the spectrum. This limits the understanding of the model’s utility.

* Figure 2 claims that DyDM successfully distinguishes WL-equivalent graphs A and B. However, the paper does not clearly explain how these graphs are constructed. Whether types of A differ in their spectra.


* While the Brain dataset is used , most baseline models are not evaluated on it. The paper does not explain why these comparisons are missing. 

* The paper overlooks foundational spectral methods in graph theory, such as spectral clustering and Laplacian-based analysis. And does not discuss the computational cost or applicability limitations of eigenvalue decomposition.

[1] Gallagher, Ian, Andrew Jones, Anna Bertiger, Carey E. Priebe, and Patrick Rubin-Delanchy. "Spectral embedding of weighted graphs." Journal of the American Statistical Association 119, no. 547 (2024): 1923-1932.
[2] Chung, Fan RK. Spectral graph theory. Vol. 92. American Mathematical Soc., 1997.

### Questions
* How exactly are graphs A and B constructed? Are the adjacency matrices of different versions of graph A identical or permutation variants of the same matrix?

* Why are baseline models missing on Brain?

* Can the model handle attributed or directed graphs?

* Can the model be used for downstream tasks?

* How does the runtime of the DyDM scale with graph size (n) compared to training a GNN-based model? A runtime/memory comparison would be valuable.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces the Dyson Diffusion Model (DyDM), a novel approach for permutation-invariant graph generation via spectral learning. The key idea is to leverage Dyson’s Brownian Motion (DBM) to model the evolution of graph spectra during an Ornstein–Uhlenbeck diffusion process on adjacency matrices. By analytically decoupling the eigenvalue dynamics (which are permutation-invariant) from the eigenvector dynamics, DyDM avoids the limitations of prior graph diffusion models that rely solely on permutation-equivariant architectures like GNNs or graph transformers. The authors demonstrate that DyDM outperforms existing methods in learning graph spectra, especially on challenging graph families where traditional models fail due to Weisfeiler–Leman (WL) equivalence.

### Strengths
1.The theoretical foundation is strong, building on well-established results from random matrix theory and stochastic differential equations.
2.Experimental results are comprehensive and compare against multiple state-of-the-art baselines (EDP-GNN, GDSS, ConGress, DiGress) across synthetic and real-world datasets.

### Weaknesses
1.The method currently focuses on spectral generation and does not fully address the generation of the entire graph (eigenvectors are not modeled in the generative process, though their dynamics are derived).
2.The numerical challenges of simulating Dyson-BM (e.g., singularities, adaptive step sizing) may limit scalability or ease of implementation.

### Questions
1.How does DyDM scale with graph size n, especially given the need for adaptive step sizing and the conditioning on non-crossing events?
2.Have the authors considered applying DyDM to other symmetric matrix data (e.g., covariance matrices) beyond graphs?
3.How sensitive is the model to the choice of hyperparameters α,β and the time schedule?

### Soundness
2

### Presentation
3

### Contribution
3
