# Efficient Prediction of SO(3)-Equivariant Hamiltonian Matrices via SO(2) Local Frames

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
We consider the task of predicting Hamiltonian matrices to accelerate electronic structure calculations, which plays an important role in physics, chemistry, and materials science. Motivated by the inherent relationship between the off-diagonal blocks of the Hamiltonian matrix and the SO(2) local frame, we propose a novel and efficient network, called QHNetV2, that achieves global SO(3) equivariance without the costly SO(3) Clebsch–Gordan tensor products. This is achieved by introducing a set of new efficient and powerful SO(2)-equivariant operations and performing all off-diagonal feature updates and message passing within SO(2) local frames, thereby eliminating the need of SO(3) tensor products. Moreover, a continuous SO(2) tensor product is performed within the SO(2) local frame at each node to fuse node features. Extensive experiments on the large QH9 and MD17 datasets demonstrate that our model achieves superior performance across a wide range of molecular structures and trajectories, highlighting its strong generalization capability. The proposed SO(2) operations on SO(2) local frames offer a promising direction for scalable and symmetry-aware learning of electronic structures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces QHNetV2, a neural network designed for predicting Hamiltonian matrices. QHNetV2 achieves global SO(3) equivariance through a set of efficient SO(2)-equivariant operations.

### Strengths
* The paper is well-structured and clearly written, offering an accessible and concise overview of both the problem and the proposed solution.
* The results presented on the two datasets show that the model performs well.

### Weaknesses
* The main claimed innovation of this paper lies in using SO(2) to efficiently predict Hamiltonians while preserving equivariance. However, techniques involving SO(2) tensor-product equivariance have already been extensively studied in both machine learning force field and Hamiltonian prediction research. Therefore, the contribution of this work appears rather trivial and not significant. What is the specific difference between the SO(2)-based module design in this paper and that in [1-3]?
* The experiments were conducted on only two datasets. It would strengthen the paper to include additional evaluations on other datasets, such as $\nabla^2$ DFT [4], to further demonstrate the model’s generality.
* The PhiSNet results reported for MD17 are not the original ones, but rather the reproduced results from the QHNet implementation. Such a comparison is unfair, and I could not find any statement in the paper clarifying that the PhiSNet results were extracted from QHNet paper.

[1] Passaro S, Zitnick C L. Reducing SO (3) convolutions to SO (2) for efficient equivariant GNNs[C]//International conference on machine learning. PMLR, 2023: 27420-27438.

[2] Liao Y L, Wood B, Das A, et al. Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations[J]. arXiv preprint arXiv:2306.12059, 2023.

[3] Wang Y, Li H, Tang Z, et al. Deeph-2: Enhancing deep-learning electronic structure via an equivariant local-coordinate transformer[J]. arXiv preprint arXiv:2401.17015, 2024.

[4] Khrabrov K, Ber A, Tsypin A, et al. $\nabla^ 2$ DFT: A Universal Quantum Chemistry Dataset of Drug-Like Molecules and a Benchmark for Neural Network Potentials[J]. Advances in Neural Information Processing Systems, 2024, 37: 36869-36889.

### Questions
See Weaknesses.

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
The paper proposes QHNetV2, a Hamiltonian prediction network that achieves global SO(3) equivariance via locally constructed SO(2) frames. By replacing the heavy Clebsch–Gordan tensor products with efficient SO(2)-equivariant operations (Linear, Gate, LayerNorm, Tensor Product), the model significantly reduces computational complexity while maintaining physical symmetry. Experiments on QH9 and MD17 show competitive accuracy.

### Strengths
1. Use of local SO(2) frames to achieve global SO(3) equivariance to reduce costs
2. Improved efficiency gains with comparable or better accuracy.
3. Promising direction for scalable learning of quantum Hamiltonians.

### Weaknesses
1. While the idea of achieving global SO(3) equivariance through SO(2) operations is elegant, it is not entirely novel. Similar concepts have been explored in previous works, such as eSCN, which the authors themselves acknowledge. The main contribution here lies in applying this framework to Hamiltonian prediction. As a result, the methodological innovation is somewhat limited, and the paper would benefit from a clearer articulation of what new theoretical or algorithmic insights distinguish it from existing SO(2)-based approaches.
2. In principle, replacing SO(3) tensor products with SO(2) operations should substantially reduce computational cost. However, the manuscript provides only limited benchmarking evidence, and the reported efficiency gains (e.g., 4.34× in Table 3) appear modest given the expected complexity reduction. It would be valuable to see a more systematic analysis of scalability — for instance, how efficiency varies with model size or with increasing molecular size and number of atoms (e.g., on MD17 systems).
3. The experimental setup appears somewhat narrow. The QM9 dataset primarily contains small molecules with similar compositions and sizes, which may make generalization within this dataset relatively easy. The MD17 trajectories also focus on closely related structures, which test temporal consistency rather than true out-of-distribution generalization. It would greatly strengthen the paper to evaluate the trained models on unseen molecular systems — for example, testing a model trained on QM9 to predict Hamiltonians of larger alkanes (CH4, C2H6, …, C20H42, …). Such experiments would help demonstrate whether the model captures transferable physical relationships rather than dataset-specific correlations.

### Questions
My questions are listed in the weakness section, here I only have one more question:
1. How do you think about the difficult to generalize this model to materials that have periodic boundary condition? Is there any technical issue you have to solve before you can do it?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes QHNetV2, an efficient equivariant neural network for predicting SO(3)-equivariant Hamiltonian matrices. The method replaces all expensive SO(3) tensor product operations with SO(2) operations performed in local reference frames. Each node, edge, and node pair defines its own local SO(2) frame through minimal-frame canonicalization and frame averaging. Within each frame, all message passing, linear, gate, and feed-forward (FFN) operations are executed in SO(2)-equivariant form, fully removing the need for SO(3) Clebsch–Gordan tensor products (while still using efficient SO(2) tensor products for feature mixing). Experiments on the QH9 and MD17 molecular datasets demonstrate strong predictive accuracy and a significant speedup over QHNet.

### Strengths
1. Comprehensive use of SO(2) operations. The model systematically generalizes SO(2)-equivariant operations beyond the message passing step, extending them to the FFN, gate, and linear layers, and allowing an entirely SO(2)-based neural architecture. This contributes to higher efficiency and stability.

2. Strong and broad empirical performance. The model achieves state-of-the-art accuracy on QH9 and competitive results on MD17, showing that the proposed SO(2) local frame formulation generalizes well across different molecular prediction tasks. Quantitative efficiency results indicate a significant speedup over QHNet, likely attributable to the elimination of SO(3) tensor products.

3. Solid engineering execution. The design of SO(2) linear, gate, and FFN modules is unified and modular. The method is cleanly implemented and could be easily generalized to other physics-informed learning tasks.

### Weaknesses
1. Limited novelty of the core operator. The SO(2) linear/tensor-product mechanism has already been introduced and optimized in eSCN [1] and subsequently adopted in EquiformerV2 [2] and eSEN [3]; moreover, it has been used in downstream Hamiltonian modeling such as DeepH-2 (which employs EquiformerV2’s SO(2) convolution) [4]. As a result, the present paper’s contribution appears primarily as extending these existing SO(2) operations to additional layers (FFN, gate) and applying them to Hamiltonian prediction. The authors should clarify what is theoretically or representationally new beyond these prior SO(2)-equivariant frameworks and implementations (including DeepH-2).

2. No dynamic weighting (attention) mechanism. Equivariant Transformers such as EquiformerV2 and eSEN show that dynamic weighting (attention) improves adaptivity and expressivity. QHNetV2 instead uses a deterministic (non-attention-based) aggregation scheme, without dynamic weighting. Why not adopt dynamic weights? Is it for computational reasons or due to theoretical constraints of SO(2) operations?

3. Potential loss of inversion equivariance. The Hamiltonian requires strict inversion symmetry (even/odd parity). SO(2) operations handle rotations but not reflections. How does the model preserve inversion equivariance? Moreover, since the features are all even functions, how can the network capture odd (antisymmetric) matrix components?

4. Experiments are only on molecular datasets (QH9, MD17). Can the proposed SO(2) local frame generalize to periodic or crystalline systems, where translational and lattice symmetries play a role?

5. Realistic Hamiltonians often include spin–orbit coupling, which requires complex-valued spinor representations (SU(2) symmetry). Can this SO(2)-based model handle such cases, or would an SU(2)-equivariant extension be necessary?

6. Possible geometric degeneracy in local frame construction.
The paper adopts a “minimal frame averaging” approach to define local SO(2) reference frames from aggregated neighborhood geometry. However, when atomic neighborhoods are nearly isotropic (e.g., central atoms in symmetric environments), the averaged direction may vanish or fluctuate, leading to unstable or ill-defined local frames. Moreover, such averaging can smooth out fine local structural details and erase genuine geometric anisotropy, causing the constructed frame to lose its physical interpretability. This degeneracy risks breaking rotational consistency and numerically violating SO(3) equivariance. The authors should clarify how such issues are mitigated.

[1] Passaro, Saro, and C. Lawrence Zitnick. "Reducing SO(3) convolutions to SO(2) for efficient equivariant GNNs." ICML 2023.
[2] Liao, Y. L., Wood, B., Das, A., & Smidt, T. (2023). Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations. ICLR 2024.
[3] Fu, X., Wood, B. M., Barroso-Luque, L., Levine, D. S., Gao, M., Dzamba, M., & Zitnick, C. L. (2025). Learning smooth and expressive interatomic potentials for physical property prediction. ICML 2025.
[4] Wang, Y., Li, Y., Tang, Z., Li, H., Yuan, Z., Tao, H., ... & Xu, Y. (2024). Universal materials model of deep-learning density functional theory Hamiltonian. Science Bulletin, 69(16), 2514–2521.

### Questions
Could you please clarify where the Hamiltonian labels for the MD17 dataset come from? The original dataset does not include such labels, and the paper does not specify whether they were generated by the authors themselves.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets fast, accurate prediction of Kohn–Sham Hamiltonian matrices. It introduces SO(2) local frames that allow most computations to be carried out with SO(2)-equivariant layers while preserving global SO(3) equivariance via canonicalization/minimal frame averaging. The approach avoids explicit SO(3) CG tensor products in message passing and feature updates and adds several SO(2)-equivariant building blocks—Linear, Gate, LayerNorm, and a (continuous) SO(2) tensor product (TP) for feature fusion. Architecture-wise (Fig. 2, p. 6), node features are updated in the global space with node‑wise interactions + SO(2) TP in a local node frame, whereas off‑diagonal (pair) features—which dominate the Hamiltonian—are updated and kept in their SO(2) pair frames using an SO(2) FFN. Experiments on QH9 and MD17 show  lower Hamiltonian‑MAE vs prior art and speedups over SO(3)-TP methods (Tables 1–3, pp. 9, 14)

### Strengths
1. The theory of connecting the SO(2) to minimal frame averaging is interesting. Appendix C shows how frame averaging collapses to the canonical rotation when the local model is equivariant to the stabilizer.
2. Empirical improvements are clear and practically relevant. Laudable MAE reductions on QH9 (including OOD) and solid speedups vs a strong SO(3)-TP baseline demonstrate usefulness (Table 1 & 3, pp. 9, 14).

### Weaknesses
1. The scalability of the model remains unclear. The authors should evaluate and compare performance with SPHNet and QHNet across systems of increasing Hamiltonian size to assess how well the proposed approach scales.
2. The discussion of related work is underdeveloped. Given that the contribution lies within the design of equivariant architectures, the paper should discuss prior SO(3) and SO(2)-equivariant models and spherical scalarization models (e.g., e3nn, SE(3)-Transformers, TFN, Allegro, SEGNN, GotenNet, ViSNet) to contextualize how the proposed approach differs in representation efficiency, inductive bias, and computational scaling.
3. Ablation is insufficient. The ablation analysis in its current form is too coarse to justify the key architectural claims. It toggles components like the SO(2) TP and SO(2) FFN but never clarifies what replaces them when removed—whether the block is skipped, replaced with an SO(2) Linear, or swapped with an SO(3) TP—which makes the results in Table 4 uninterpretable. The ablation also conflates multiple design factors: it does not isolate the effect of the continuous SO(2) TP’s parameters (v, M), the “keep-in-frame” vs. “re-project” strategy for pair features, or different node-frame constructions (nearest-neighbor vs. averaged). Moreover no runtime or memory profile connects the claimed efficiency gains to actual costs. A more systematic ablation—explicit replacements, controlled cost–benefit sweeps, and frame-robustness tests—would make the paper’s design rationale far more convincing.
Minor:
1. Table 3 (p. 14) uses “maximum available batch size” on a single A6000. Because max batch differs by model/memory footprint, throughput (samples/s) could be confounded. A stronger comparison would fix (i) hardware/precision, (ii) batch size, (iii) sequence length/graph size bins, and report per‑sample latency and FLOPs to isolate algorithmic speedups. Please also report precision (FP32/FP16/BF16) and cudnn/e3nn kernel variants used

### Questions
1. The node frame is defined by the nearest-neighbor direction (Eq. 13, p. 7), which the paper acknowledges may introduce discontinuities when neighbor assignments change. While future remedies such as averaging over O(n) frames are mentioned, the current formulation may still experience gauge flips. Could you quantify how frequently these occur in practice and describe their effect on training stability and convergence?  
2. The authors of eSCN derive the SO(2) convolution complexity as $O(L^3)$. Could you similarly derive and report the computational complexity of your proposed SO(2) tensor product (TP) in terms of $L$? (i.e., for a given $L$, first apply a rotation to align features into the SO(2) subspace, perform the SO(2) tensor product there, then rotate back to the original frame—please include these rotation steps in the overall computational complexity analysis) In practice, what value of $v$ is used in the many-body expansion? How do runtime and memory scale with $M_{\max}$, $v$, and the number of channels? The ablation in Table 4 (p. 14) is unclear—what exactly does “no SO(2) TP” mean? Was it replaced with an SO(3) TP or omitted entirely?
3.Can the expansion module also leverage SO(2) representations to further reduce computational cost?
4. Could the authors provide a detailed runtime and memory profile of the network? Specifically, which modules (e.g., SO(2) TP, FFN, gating, or LayerNorm blocks) dominate the computation or memory usage? Such profiling would be valuable for identifying future optimization and scaling directions.

### Soundness
2

### Presentation
2

### Contribution
2
