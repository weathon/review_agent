# Towards a Transferable Acceleration Method for Density Functional Theory

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Recently, sophisticated deep learning-based approaches have been developed for generating efficient initial guesses to accelerate the convergence of density functional theory (DFT) calculations. While the actual initial guesses are often density matrices (DM), quantities that can convert into density matrices also qualify as alternative forms of initial guesses. Hence, existing works mostly rely on the
prediction of the Hamiltonian matrix for obtaining high-quality initial guesses. However, the Hamiltonian matrix is both numerically difficult to predict and intrinsically non-transferable, hindering the application of such models in real scenarios. In light of this, we propose a method that constructs DFT initial guesses by predicting the electron density in a compact auxiliary basis representation using
E(3)-equivariant neural networks. Trained exclusively on small molecules with up to 20 atoms, our model achieves an average 33.3% reduction in SCF iterations for molecules three times larger (up to 60 atoms). This result is particularly significant
given that baseline Hamiltonian-based methods fail to generalize, often increasing the iteration count by over 80% or failing to converge entirely on these larger systems. Furthermore, we demonstrate that this acceleration is robustly scalable: the model successfully accelerates calculations for systems with up to 900 atoms (polymers and polypeptides) without retraining. To the best of our knowledge,
this work represents the first and robust candidate for a universally transferable DFT acceleration method. We also released the SCFbench dataset and its accompanying code to facilitate future research in this promising direction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new paradigm for accelerating DFT self-consistent field (SCF) convergence by predicting electron densities in an auxiliary basis using E(3)-equivariant neural networks. Unlike prior models that predict the Hamiltonian (H) or density matrix (D), this approach predicts density coefficients (ρ), from which the Coulomb (J) and exchange–correlation (Vₓc) matrices are reconstructed to form the initial Fock matrix. The authors release SCFbench, a new dataset (43,862 molecules up to 20 atoms, OOD set up to 60 atoms, elements H–S) with Hamiltonians, density matrices, and density coefficients.

### Strengths
* Figure1 is interesting. The author demonstrate predicting the electron density could improve the OOD performance. The method transfers across XC functionals (BLYP, SCAN, B3LYP, PBE0) and basis sets (def2-TZVP/QZVP), retaining ≈ 15–25 % speed-up.

* SCFbench is timely; the paper includes careful splits, ID and OOD settings, and extensive reporting (Tables 1–2; Appx. E tables). The dataset spans H, C, N, O, F, P, S with multiple auxiliary bases (def2‑universal‑jfit and ETBs). The summary plots in Fig. 3 (p. 6) are helpful.

### Weaknesses
- The predicted electron density $ \\rho $ is not enforced to integrate to the total number of electrons $ N_e $ or remain positive (see App. C).  
-§5.1 defines RIC and mentions convergence within 50 iterations, but practitioners need explicit iteration counts and wall-times. Provide a detailed timing breakdown for key components—model forward pass, 3c-ERI contraction for $ \\mathbf{J} $, XC grid evaluation, and eigensolve—on a reference CPU/GPU.  
-  The table states “energy unit = Hartree” but includes MAE(C) and C-similarity, which are dimensionless. Clarify the units and definitions used.  
-  Please also report the Fock-matrix error $ \\| \\mathbf{F}^{(0)} - \\mathbf{F}^* \\|_F $ and the density error $ \\| \\rho^{(0)} - \\rho^* \\|_2 $, computed on the same grid used for XC integration.
- For meta-GGA, the kinetic energy density $ \tau $ is approximated by the von Weizsäcker form, while for hybrids, HF exchange is built from a superposition-of-atomic-densities density matrix (Appx. C, Eqs. 15–17). Consequently, the reported transferability to SCAN and hybrid functionals assesses a slightly different initial guess than that implied by the predicted $\tilde{\rho}$ alone. This distinction should be explicitly noted when interpreting Table 2 (p. 9).
- Please specify DIIS settings / damping / level shifts used during SCF for all methods; RIC can be sensitive to these solver details (Sec. 5.1)

### Questions
See weaknesses.

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
This paper proposes a new paradigm for accelerating DFT self-consistent field (SCF) iterations by predicting the electron density coefficients in an auxiliary basis, instead of the Hamiltonian or density matrix. The model employs E(3)-equivariant networks (NequIP and QHNet variants) and is trained on the newly released SCFbench dataset. The approach achieves consistent ∼33% SCF step reduction across systems up to three times larger than training molecules, demonstrating strong transferability across exchange–correlation functionals and basis sets.

### Strengths
1. Physically well-motivated formulation; electron density is indeed the fundamental variable of DFT.
2. Public release of SCFbench dataset, beneficial to the community.

### Weaknesses
1. Limited practical usefulness: 
The proposal does not avoid SCF; it only changes the initial guess. If the model really works, why didn't the authors directly learn the final density of the converged SCF calculations? That will require only one extra step to construct the Hamiltonian to solve for energies or other related physical properties.

2. THe computational saving is unclear:  
The paper optimizes “SCF step count,” not wall-clock time including ML inference + integral construction + XC quadrature from the predicted density. Without end-to-end timing on CPU/GPU stacks, it’s hard to judge net throughput gains in realistic workflows.

3. The claim (“universally transferable”) is over-stated: 
Evidence is restricted to molecular systems with light elements (H,C,N,O,F,P,S), no periodic solids, and or medium-size molecules with >20 atoms. Transfer tests across functionals/bases are helpful but moderate (e.g., RIC ~85% on B3LYP5/def2-TZVP OOD), which still implies limited speedups in those regimes. True “universality” would need broader chemical domains and periodic boundary conditions.

### Questions
1. Wall-clock time: What are the net speedups including ML inference, J and $V_{\mathrm{xc}}$ evaluation from auxiliary density, diagonalization, and SCF mixing—on representative CPU and GPU nodes? Please report time-to-energy for realistic job batches. 
2. Direct-to-solution baseline: Have you tried predicting converged density and performing a single update/diagonalization? If not, what prevented it? Empirical evidence would clarify whether your framing is a necessity or a choice.
3. Generalization to slightly larger molecules: WHat is your model performance on larger molecules? For example, MD17/MD22 datasets?

### Soundness
1

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
DFT is one of the most important methods in computational chemistry, offering the possibility of computing energies and forces for (weakly correlated) molecules and solids with high accuracy and, compared to other methods, modest computational effort. DFT calculates these quantities from the electron density using auxiliary (and empirical) potential functions. Finding the electron density requires the solution of a nonlinear eigenvalue problem, which can only be tackled using iterative algorithms (the most famous among them is called SCF). Such iterative algorithms crucially hinge on good choices of initial values, within the basis of attraction of the SCF dynamics. The paper introduces an ML-based method to obtain such initial values. 

Contrary to previous approaches, the proposed method directly operates on expansion coefficients of the electron density with respect to a given basis. From this, the matrices which make up the nonlinear eigenvalue problem can be computed via quadrature. This representation of the electron density has the advantage of being transferable to larger systems. Indeed it is shown that training on systems with up to 20 atoms yields equally good initializations for systems with up to 60 atoms.

### Strengths
The idea of directly using the electron density allows to obtain a transferable initialization method is very nice. In particular the possibility to train on small systems while still getting good initializations on larger systems is extremely promising.

### Weaknesses
The claim of "universal transferability" might be a bit strong given that the method is only tested on systems with up to 60 atoms - keeping in mind that conventional DFT calculations often treat systems with hundreds or thousands of atoms. More evidence on larger scales would be needed to fully validate this claim. 

Unless I am missing something the gain in the number of iterations is around 34% for a certain experiment but varies considerably across different experiments (see Table 2).

The paper seems to exclusively focus on RIC. However, this is not the only indicator of efficiency. In order to make a fair comparison one needs to monitor the wall time (keeping in mind that assembling the relevant matrices from the expansion coefficients requires the numerical solution of complicated quadrature problems) as well as the final accuracy. Without a careful comparison of overall wall time and accuracy the results become meaningless because we do not know the true complexity, or what the algorithm converges to. Therefore, this information has to be included in a revision.

### Questions
A detailed comparison in wall time and accuracy has to be included. 

The authors might reconsider the term "universal transferability", as it might be a slight overstatement.

### Soundness
3

### Presentation
3

### Contribution
3
