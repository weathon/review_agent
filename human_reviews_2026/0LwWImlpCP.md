# Electrostatics from Laplacian Eigenbasis for Neural Network Interatomic Potentials

- Decision: Reject
- Scores: 2, 4, 6, 6, 2

## Abstract
In this work, we introduce $\Phi$-Module, a universal plugin module that enforces Poisson’s equation within the message-passing framework to learn electrostatic interactions in a self-supervised manner. Specifically, each atom-wise representation is encouraged to satisfy a discretized Poisson's equation, making it possible to acquire a potential $\boldsymbol{\phi}$ and a corresponding charges $\boldsymbol{\rho}$ linked to the learnable Laplacian eigenbasis coefficients of a given molecular graph. We then derive an electrostatic energy term, crucial for improved total energy predictions. This approach integrates seamlessly into any existing neural potential with insignificant computational overhead. Our results underscore how embedding a first-principles constraint in neural interatomic potentials can significantly improve performance while remaining hyperparameter-friendly, memory-efficient and lightweight in training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Φ-Module, a plug-and-play Laplacian-eigenbasis module that embeds the Poisson equation as a self-supervised physical constraint within graph neural network (GNN) interatomic potentials.

### Strengths
The proposed module can predict atomic potential ϕ and charge ρ from learned eigenbasis coefficients and adds an electrostatic energy term to improve total-energy and force predictions.

### Weaknesses
The paper repeatedly claims that Φ-Module captures non-local electrostatic interactions, yet no benchmark explicitly demonstrates this. Improvements in energy and force MAE alone are insufficient proof of non-local interaction modeling.

For MD22, the authors selected two of the smallest molecules, while the claim centers on modeling non-local interactions. Larger systems such as Ac-Ala15-NHMe or DHA are more appropriate to evaluate the alleged long-range capability. Without them, the argument remains speculative.

On the OE62 experiments, the baselines (SchNet, DimeNet++, PaiNN, etc.) are 2019–2021 architectures. Recent high-order equivariant models—MACE, eSCN, NequIP, Equiformer-V2, ViSNet—represent the current state of the field. Because the Φ-Module is advertised as a general plug-in, results on these modern architectures are essential for a credible evaluation. 

On the MD22 experiments, the reported performance differences from the original ViSNet paper are unusually large for the same datasets. The authors should clarify training settings (data splits, learning rates, cutoff, etc.) and reproduce the original ViSNet baseline under identical conditions.

### Questions
See Section Weakness.

### Soundness
3

### Presentation
3

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
This work introduces $\Phi$-Module, an extension to atom-based machine learning interatomic potentials that intends to resolve long-range energy contributions. $\Phi$-Module is derived from the discretized Poisson equation for electrostatics and uses the graph Laplacian for efficient long-range propagation. In their experimental evaluation, the authors find the $\Phi$-Module to be an effficient addition to MLIAP, reducing errors at low computational overheads.

### Strengths
* The proposed $\Phi$-module presents a novel addition to the field of MLFFs.
* The experiments suggest a very valuable extensions with favorable runtime-accuracy tradeoff.
* $\Phi$-module appears quite modular and presents an easy integration.

### Weaknesses
1. The authors claim that their method is self-supervised and does not need external labeled data (l.50-51 + abstract). However, the losses introduced in l.174 and l.179 do not ensure that predictions improve, the method additionally **needs** the standard supervised learning loss.
2. The use of 1D convolutions over the nuclei breaks the permutation invariance. I find the paper lacks to discuss this disadvantage. One could in general neglect permutation invariance in MLFFs, does that yield similar improvements?
3. The authors leave a lot of questions open or are not specific in several key areas, see questions.


Minor:
* l.26-33: DFT and MLFFs are quite different things and the section suggest they accomplish the same.
* l.34 which alternatives?
* Table 1 is never referenced.
* l.432 formatting

### Questions
1. What is used as graph connectivity for the graph laplacian? A radial cutoff or molecular bonds? In either case, doesn't bond-breaking or going out of the cutoff radius, introduces jumps in the energy surface?
2. Why is the Laplacian weighted by the pairwise distance, intuitively, I'd expect it to be weighted by the inverse of the distance?
3. l.173 is the same alpha-Net used for phi and rho?
4. Couldn't one enforce the PDE and net zero loss analytically? How does that compare to training?
5. What is the x-axis in Figure 4? (Also the figure is never referenced)
6. Table 1: What about other MLFF+Phi module combinations?
7. Are the hyperparameter optimization from l.684 used for all experiments? This seems quite excessive given that no hyperparameter optimization is done for any of the baseline models.
8. What units are Table 4 and what target, which dataset, etc?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a module for learning the electrostatic interactions for interatomic potentials based on sparse graph neural networks and Poisson equation.

### Strengths
- The proposed method is simple yet performative.
- The paper is very well presented with the advantage of the designed model clearly highlighted.
- The tradeoff between efficiency and performance is discussed in detail.
- The problem addressed in this paper is of importance to the molecular modeling community.

### Weaknesses
- There seems to be limited novelty in the proposed graph representation module.

### Questions
- Could you kindly elaborate more on the choice of using spectral graph neural networks, rather than the spatial ones, which are more popular?

### Soundness
3

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
3

### Summary
The paper introduces a Phi module, a module that can be added to any MLIP architecture to introduce a long-range electro statics inductive bias. They imitate the computational structure of a poission equation to learn a sort of latent charges that can be trained self-consistently.

### Strengths
-The method is well motivated and makes intuitive sense
-The experimental evaluation is thorough
-The improvements are very consistent

### Weaknesses
-The accuracy improvements are rather small
-The spectral decomposition introduces an N^2 scaling operation, which could become problematic for larger-scale simulations. The paper only benchmarks memory, but not runtime with system size; this should be benchmarked and could change my opinion
-There have been works before that incorporate explicit charge equilibrium/coulomb interactions, in particular https://www.nature.com/articles/s41467-020-20427-2 . A comparison would be appropriate
- The phi module only targets electrostatics and not other long range effects, which may make fully learned approaches preferable

### Questions
- Do you use a dense connectivity graph L? And why is it weighted by d_ij, doenst this imply that far away atoms interact stronger than closer ones?
- Why did you use VisNet in favour of a newer architecture?

Remark:
"In quantum chemistry, the task of correct prediction of atomic energies is paramount, but stands
a great challenge…” Atomic energies are not a well defined concept, did you mean molecular energies?
“Some of those require prior data in the form of partial charges or
dipole moments, which is costly to retrieve using DFT”, If the DFT calculation is already converged to get the molecular energies, it is trivial to get dipoles or partial charges at negligible costs
“...and gives the opportunity to process large macromolecules with the Φ-Module. This decision also keeps us away from the ambiguity of invariance and sorting of eigenvalues and eigenvectors during their computations - we strictly get k-selected eigenvalues and their corresponding eigenvectors without the need to sort them anyhow.” This is not convincing, if the Laplacina has a degenerate eigenspace the order will be arbitrary and specifics will be subject to numerical noise, a conjugated gradient solver doesnt change this

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Φ-Module, a physics-informed plugin designed to integrate electrostatic interactions into graph neural networks in a self-supervised way. By enforcing Poisson’s equation within the message-passing framework, Φ-Module enables models to learn atomic potentials and charges represented in the Laplacian eigenbasis of molecular graphs. This approach captures long-range electrostatic effects that standard local message passing often misses. The module includes a lightweight subnetwork, α-Net, that predicts eigenbasis coefficients, allowing derivation of an electrostatic energy term that improves total energy predictions with minimal computational overhead. Experiments on the OE62 and MD22 benchmarks show accuracy gains across several neural potentials

### Strengths
1. The paper proposes an integration of a first-principles physical law (Poisson’s equation) into GNN-based interatomic potentials. By embedding the Poisson constraint, the model learns to produce physically meaningful electrostatic potentials and charges in a self-supervised way, without requiring any ground-truth charges or external fields.
2. Φ-Module is designed as a universal augmentation that can be attached to essentially any GNN architecture for molecules. The authors demonstrate this generality by incorporating Φ-Module into multiple established models 
3. A claimed advantage is that Φ-Module introduces very little overhead. The spectral α-Net and Laplacian eigenbasis computation are lightweight, adding roughly 5–10% to training time per epoch and a modest amount of memory usage.

### Weaknesses
1. The biggest weakness is generalizability and scalability. Experiments are restricted to OE62 and MD22. There’s no end-to-end training on truly realistic, large, diverse, million–sample datasets (e.g., OMol 25) or cross-dataset transfer demonstrating generalization across broader chemistries. As a result, it’s unclear how Φ-Module scales in sample size or generalizes to diverse systems.
2. Although the overall results favor Φ-Module, the improvements are not uniformly overwhelming. In OE62, one baseline GNN saw only ~5% error improvement with Φ-Module, which is relatively modest. On MD22, the Φ-augmented model did not win on every single metric – the original ViSNet still had the best outcome on 2 of the 14 comparisons, and in 3 of 14 cases Φ-Module failed to set a new state-of-the-art. This indicates that the benefits, while present, can be incremental, task-dependent, or just random (no error bar / std is given)
3. As a plug-in, Φ-Module adds four new hyperparameters and extra computations (solving for eigenvectors) to a model. While the authors argue this is lightweight, one might be concerned about implementation complexity.

### Questions
1. Could you clarify how the Laplacian eigenpairs are computed during training? The paper suggests using a fixed number k of eigenvectors and a batched eigendecomposition approach. Is this done via an iterative solver each message-passing step, or are eigenvalues computed once per epoch/structure and reused? 
2. The paper references alternatives like adding Ewald summation to GNNs or using pre-computed partial charges. Did you consider comparing Φ-Module’s performance to such methods? 
3. Beyond the tasks in this paper, how general is Φ-Module’s applicability? For instance, can it handle systems with periodic boundary conditions (common in materials simulations where Ewald is often needed)?

### Soundness
3

### Presentation
3

### Contribution
2
