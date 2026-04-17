# Geometric Kolmogorov Superposition Representation of group invariant function for computational science

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
The Kolmogorov-Arnold Theorem (KAT), or more generally, the Kolmogorov Superposition Theorem (KST), establishes that any non-linear multivariate function can be exactly represented as a finite superposition of non-linear univariate functions. Unlike the universal approximation theorem, which provides only an approximate representation without guaranteeing a fixed network size, KST offers a theoretically exact decomposition. The Kolmogorov-Arnold Network (KAN) was introduced as a trainable model to implement KAT, and recent advancements have adapted KAN using concepts from modern neural networks. 

However, KAN struggles to effectively model physical systems that require inherent equivariance or invariance geometric symmetries as 
$E(3)$ 
transformations, a key property for many scientific and engineering applications. 
In this work, we propose the Geometric
Kolmogorov Superposition Representation (GKSR), a novel extension of KAT,
and Geometric Kolmogorov Superposition Network (GKSN), its implementation,
which incorporate invariance over various group actions,
including $O(n)$, $O(1,n)$, $S_n$ and general $GL$, 
enabling accurate and efficient modeling of these systems. 

Our approach provides a unified approach that bridges the gap between mathematical theory and practical architectures for physical systems, expanding the applicability of KAN to a broader class of problems. We provide experimental validation on molecular dynamical systems and particle physics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper proposes a variation of Kolmogorov Arnold networks for group invariant tasks such as molecular dynamics simulation.

### Strengths
The paper's presentation looks visually clear. I am not well positioned to judge the originality and significance of the work, as I am not directly working on KANs and not sufficiently familiar with all of the maths here.

### Weaknesses
Experiments on MD17 and MD22 are insufficient to make claims about molecular simulation capabilities, as these datasets are not used anymore in cutting edge ML Interatomic Potentials (MLIP) papers anymore to the best of my knowledge. For instance, a major limitation is that evaluation on MD17/22 requires training separate models for each molecular system to be simulated. Hence, models are never truly evaluated for generalization to new molecules unseen in training. 

Despite the limits of the MD17/22 datasets, the authors have also not compared to any state-of-the-art MLIPs in the tables.

### Questions
Following up on the weakness identified regarding the MD17/22 evaluation, I have a question: Is the proposed GKSR architecture able to handle variable sized input molecules?

The authors end the paper with a sentence that I felt was pretty mysterious: "further investigation will show if this architecture can be extended to implement machine learning interatomic potentials". Can you clarify/elaborate this sentence?

What is currently missing from GKSR that prevents it from implementing practical MLIPs?

An other major consideration for MLIPs is speed/runtime: while in theory KANs may help improve MLIPs, how does this impact practical usage speed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Geometric Kolmogorov Superposition Representation (GKSR) and a practical GKSN layer that extend KST/KAN to explicitly handle group symmetries (O(n), O(1, n), GL(n), and permutations). The core idea is to express invariant functions via univariate functions applied to group invariants (e.g., inner products), with a tightened representation that reduces feature growth from quadratic in nodes to linear by projecting onto a spanning set. The network implementation adds a near-linear “residual-like” path to ease optimization. Empirically, GKSN matches or outperforms strong equivariant/physics baselines on Lennard-Jones, MD17/MD22, and Lorentz-invariant jet-tagging tasks.

### Strengths
- The paper is technically sound and (to my knowledge) appears to be the first KAN-family work that systematically incorporates group symmetries within a clean formulation. 
- The presentation is rigorous, well organized, and self-consistent.
- The proposed GKSN shows competitive empirical performance, standing as a solid proof of concept supported by comprehensive experiments, including ablation studies (e.g., with and without heuristic invariants) and compelling applications such as symmetry discovery.

### Weaknesses
- I find no critical weaknesses. A minor limitation is the inclusion of certain heuristic, handcrafted invariants (e.g., norms or pairwise distances, …), which slightly compromise the purity of the theoretical framework. While this is understandable from an implementation standpoint, it blurs the line between learned and prescribed invariants.

- I tried to review the code, but the anonymous GitHub repository returns ‘The requested file is not found’ for all files.

### Questions
This is not a criticism but a conceptual curiosity:
- Can the invariant construction be made learnable (e.g., by parameterizing the bilinear form) to adapt to approximate or broken symmetries?

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
2

### Summary
In this submission, the authors propose the Geometric Kolmogorov Superposition Representation (GKSN), which extends the Kolmogorov–Arnold theorem (KAT) to settings where functions are constrained by symmetry groups such as $E(3), O(1), O(1, n), S_n$, and $GL$. Motivated by this theoretical formulation, the proposed architectures are implemented and evaluated on the MD17 and MD22 datasets.

### Strengths
1. This work represents a meaningful step toward incorporating symmetry constraints into the KAN family of models, as symmetry is a fundamental property in physical systems.

2. The work considers widely discussed symmetries into the framework including  $E(3), O(1), O(1, n), S_n$, and $GL$.

### Weaknesses
1. Experiments on MD17/MD22. These benchmarks primarily evaluate atomistic energy (and often forces). Please report the absolute energy error (e.g., MAE in meV) and, if applicable, force error (meV/Å). It would be helpful to include efficiency metrics—such as parameter count, FLOPs, and wall-clock time per epoch—and a direct comparison with invariant GNN baselines (e.g., SchNet) to better understand GKSN’s accuracy–efficiency trade-off.

2. Positioning vs. existing invariant GNNs. The comparison and claimed advantages over established invariant architectures remain unclear. What is the computational complexity of GKSN relative to standard invariant GNNs, and how does its performance scale with model size?

3. Relation to frame averaging. Many invariant GNNs handle multiple symmetries through frame averaging. Please elaborate on how GKSN differs conceptually and practically from such approaches. What are the trade-offs in terms of inductive bias, sample efficiency, and computational cost? Under what conditions would GKSN be expected to outperform frame-averaged architectures?

4. Overall, the idea is promising and potentially impactful, but the work would benefit from clearer quantitative comparisons and a deeper discussion of how GKSN advances upon and relates to existing invariant GNN methods and frame-averaging techniques.

### Questions
N/A.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Geometric Kolmogorov Superposition Representation (GKSR), a novel theoretical framework that extends the Kolmogorov-Arnold Theorem (KAT) to inherently incorporate geometric symmetries. The primary goal is to address a key limitation of standard Kolmogorov-Arnold Networks (KANs), which struggle to model physical systems requiring $E(3)$ invariance or other group symmetries.

### Strengths
1. The paper's primary strength is its originality. It provides the first, to my knowledge, principled theoretical framework for unifying the exact representation property of KAT with the geometric consistency required in scientific computing.

2. The experiments, particularly on the MD17 and MD22 datasets, show a clear and consistent performance benefit for GKSN over the equivalent EMLP/GMLP baselines.

### Weaknesses
1. The paper's key theoretical improvement reduces complexity from a naive $O(m^4)$ to $O(m^2n^2)$. While a significant theoretical step, this is still quadratic in the number of particles ($m$). 

2. The authors motivate GKSN using the exact representation property of KAT . However, the practical implementation (GKSN) uses ReLU as the basis for the univariate functions, not the splines or other smooth functions typically required for the theoretical guarantee. This choice is likely for practical efficiency, but it weakens the connection to the core theoretical motivation.

### Questions
1. Could the authors comment on the expected performance of a naive KAN (without the invariant-feature preprocessing) on these tasks? This baseline seems critical to understanding whether the performance gains stem from the KAN architecture itself or from the new geometric representation.

2. The empirical results (e.g., Fig 2b, 2d)  show GKSN is much more stable than GMLP, especially with permutation invariance. This is a strong practical result. Do the authors have a clear intuition for why the KAN-based superposition architecture provides this added stability compared to a standard MLP?

### Soundness
3

### Presentation
3

### Contribution
3
