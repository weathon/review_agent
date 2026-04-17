# Shoot from the HIP: Hessian Interatomic Potentials without derivatives

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Fundamental tasks in computational chemistry, from transition state search to vibrational analysis, rely on molecular Hessians, which are the second derivatives of the potential energy. Yet, Hessians are computationally expensive to calculate and scale poorly with system size, with both quantum mechanical methods and neural networks. In this work, we demonstrate that Hessians can be predicted directly from a deep learning model, without relying on automatic differentiation or finite differences. 
We observe that one can construct SE(3)-equivariant, symmetric Hessians from irreducible representations (irrep) features up to degree $l$=2 computed during message passing in graph neural networks. 
This makes HIP Hessians one to two orders of magnitude faster, more accurate, more memory efficient, easier to train, and enables more favorable scaling with system size. We validate our predictions across a wide range of downstream tasks, demonstrating consistently superior performance for transition state search, accelerated geometry optimization, zero-point energy corrections, and vibrational analysis benchmarks.
We open-source the HIP codebase and model weights to enable further development of the direct prediction of Hessians.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors investigate the accuracy of directly predicting molecular Hessian, rather than deriving them from numerical derivatives of forces or automatic differentiation, for predicting vibrational properties of molecules. They claim that forward prediction is “one to two orders of magnitude faster, more accurate, more memory efficient, easier to train, and enables more favorable scaling with system size” compared to autodifferentiation. For evaluation, the authors use an EquiformerV2 backbone to predict Hessians of the HORM dataset of organic molecules and compare HIP-EquiformerV2 against AlphaNet, LEFTNet, LEFTNet-df, and EquiformerV2 using autodifferentiation.

This paper addresses an important bottleneck in atomistic simulation. Benchmarks of this kind are valuable for the community, as they clarify the tradeoffs between direct prediction and differentiation-based approaches and help researchers decide which method is most appropriate for a given use case. The experiments are promising, but the claims about speed and generality need expanded benchmarking and clearer implementation transparency to be a solid resource for the community and thus merit acceptance.

### Strengths
* The paper is well written, accessible, and gives a solid overview of relevant literature, including clear derivations of how Hessians can be constructed from edge features using Clebsch–Gordan coefficients.
* The experimental evaluation is comprehensive, comparing multiple aspects of Hessian prediction: convergence steps, wall time, accuracy of geometry optimization, zero-point energy, transition-state search, frequency analysis, and raw elementwise accuracy.
* The authors clearly state limitations, such as reliance on ground-truth DFT data and the current restriction to small molecular systems.

### Weaknesses
* The direct prediction of Hessians is not, on its own, a sufficiently novel contribution to carry the paper. Its impact depends on demonstrating clear and reproducible efficiency–accuracy tradeoffs across architectures and datasets, which currently feels underexplored.
* It is unclear how much of the difference in AD vs. HIP accuracy for EquiformerV2 arises from training setup and task formulation since EquiformerV2 was not originally designed or tuned for Hessian prediction vs. being a “fundamentally” better approach to learn second derivatives directly. The authors should discuss how architectural or training adaptations might influence these results, and whether the comparison reflects representational capacity or differing task suitability.
* While numerical derivatives scale poorly, they remain heavily used due to simplicity and established workflows. A direct wall-time comparison (finite-difference forces vs. HIP vs. AD) would contextualize the claimed speedups.
* The comparisons of memory, batching, and wall time are highly implementation-dependent. The paper would benefit from a reproducible table specifying hardware, batch size, precision, and AD implementation details (e.g., JAX vs. PyTorch) since these factors dominate observed scaling.
* When training Hessians via AD, one would typically not predict full Hessians every iteration but use sampled Hessian-vector products or partial training, which can substantially mitigate costs. This nuance should be acknowledged to make the efficiency discussion fairer.
* As far as I can tell, the HORM benchmark is not peer-reviewed. Including results on an established dataset (e.g., Hessian-QM9) or a small material-system test would strengthen generality claims.
* The paper evaluates HIP vs. AD only for EquiformerV2. It would be informative to test lighter-weight architectures (e.g., NequIP, MACE, Allegro) to see whether the efficiency and accuracy gaps persist across model families.

### Questions
1. Could the authors clarify the exact computational setup for timing and memory benchmarks—hardware type, precision, AD backend (JAX vs. PyTorch), and batch sizes? These details are critical for reproducibility.
2. How sensitive are the reported speed and memory gains to the choice of AD implementation or backend?
3. Given that only the Hessian head was retrained on a frozen backbone, do the authors expect joint end-to-end training of energies, forces, and Hessians to change the relative trends between HIP and AD?
4. Would the HIP approach generalize straightforwardly to periodic or larger material systems, or are there challenges related to locality assumptions and cutoff sparsity?

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
The paper introduces the Hessian Interatomic Potential (HIP) framework, which leverages SE(3)-equivariant neural networks to directly predict the full (3N \times 3N) molecular Hessian matrix, thereby accelerating the computation of second-derivative information in molecular simulations. In terms of architecture, HIP employs EquiformerV2 as the core equivariant backbone, coupled with a Clebsch–Gordan (CG) expansion–based prediction head to construct the final Hessian tensor with proper rotational symmetry. Furthermore, the method incorporates a loss function that includes eigenvector-based constraints, which guide the model to produce Hessian predictions with more accurate eigenvalue spectra, improving the quality of vibrational and stability properties derived from the predicted matrices.

### Strengths
1. The paper aims to provide a comprehensive evaluation of the predicted Hessian matrices through multiple downstream tasks, including geometry optimization, zero-point energy (ZPE) estimation, and transition-state (TS) searches to test the accuracy and the traditional methods.

### Weaknesses
1. The paper lacks sufficient discussion and comparison with existing SE(3)-equivariant networks designed for predicting equivariant matrices. A series of works following PhiSNet have explored predicting the Hamiltonian matrix, which is also an equivariant matrix similar in nature to the Hessian. The architectures developed in those studies could, in principle, be applied to Hessian prediction as well. Therefore, including an introduction and comparison with these related models would help demonstrate the advantages of the proposed approach in this prediction task.

2. Furthermore, regarding the loss function design, paper [1] introduces a method that imposes constraints on the eigen-energies of the Hamiltonian matrix. The loss function presented in Section 3.3 of the current submission appears to be identical to that in [1], and this overlap should be clearly acknowledged and discussed.

3. Overall, the writing and organization of the paper are clear. However, my main concern lies in the novelty of the machine learning contribution. Most of the techniques employed have already been introduced in prior works, and the Hessian matrix prediction task closely resembles Hamiltonian matrix prediction in its formulation—an area that has been previously studied but not sufficiently discussed or differentiated in this submission.

### Questions
N/A.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a derivative-free method to predict Hessian matrices for interatomic potential models, which reduces the computational cost of Hessian evaluation. The method directly constructs SE(3) equivariant symmetric Hessians from higher-order equivariant features of a graph neural network. This lowers both memory and time costs for training and inference that involve Hessians in interatomic potentials. The paper validates the trained models on multiple downstream tasks and the accuracy remains consistently strong.

### Strengths
1.It proposes a novel derivative-free Hessian prediction method and a loss function tailored to the Hessian subspace.
2.The Hessian prediction method guarantees equivariance and symmetry, and the approach is simple, reasonable, and self-consistent.
3.The method accelerates Hessian prediction, reducing complexity from O(N^2) to O(N), and it also lowers memory usage.
4.The method achieves high accuracy and performs well across various downstream tasks.

### Weaknesses
1.Direct Hessian prediction is not applicable when DFT Hessian data are lacking.
2.The reliability of direct Hessian prediction is uncertain.
3. This method will be quite likely to fail when long-time simulation is required.

### Questions
1.Obtaining Hessian data from DFT is difficult, and there are currently few DFT datasets that include Hessians. In this context, AD-based Hessian prediction can produce Hessians without training specifically on Hessians. In contrast, direct Hessian prediction necessarily requires Hessian data for training, which limits its applicability.
2.The generated Hessian matrices satisfy rotational equivariance and symmetry. However, direct Hessian prediction does not arise from differentiating energy or forces. Could this lead to other physical inconsistencies? For example, physically there should be six zero eigenvalues. Although this is included in the loss, it is not guaranteed by the model architecture. The paper provides extensive numerical and downstream evidence for effectiveness, and direct Hessians do not suffer from the non-conservative force issue. Even so, this potential non-physical aspect remains concerning. Could the authors offer theoretical discussion and clarification?
3.In Table 1, the Hessian accuracy of the AD baselines is similar to that reported in the HORM paper, but the eigenvalue accuracy differs considerably. Was a different error metric used?
4.The authors claim improved Hessian prediction accuracy. With Equiformer V2 as the backbone and only training a separate Hessian head, improvements are indeed observed, even in loss ablations. However, the Equiformer V2 backbone comes from HORM and has already been trained with Hessian data. This work then trains on that backbone. The authors state this isolates the effects of energy and forces, but this seems unfair. Could the authors provide results where the backbone is trained only on energy and forces, then frozen, and the Hessian head is trained on top?
5.Among the AD baselines, some models use direct forces. Even with AD, the resulting Hessians are not fully physical. The comparison between LEFTNet and LEFTNet-df shows that conservative force models yield more accurate Hessians. The AD baseline for Equiformer V2 uses direct forces. Would an AD Hessian with conservative forces on Equiformer V2 achieve higher accuracy?
6.Typos: line 102 Hesians. line 465 should be Table 3. line 1001 transtion.
7. Can this method work for long time simulation?

### Soundness
2

### Presentation
3

### Contribution
2
