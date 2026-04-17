# Diffeomorphic Optimization

- Decision: Reject
- Scores: 8, 0, 6

## Abstract
Optimization is a challenging task due to the rugged nature of the optimization landscape and the concentration of data on a low-dimensional manifold. Our approach starts from the observation that flow and diffusion models map the data manifold to a smooth and simple base space. We thus propose to reparameterize the optimization problem in terms of these simple base-space variables. Using concepts from differential geometry, we demonstrate that this reparameterization naturally constrains optimization to the data manifold and results in a smoother optimization surface. We extend diffeomorphic optimization to matrix groups, such as $SO(3)$ and $SE(3)$, which allows us to empirically demonstrate the effectiveness of our approach in the highly relevant task of protein design.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Diffeomorphic Optimization, a novel framework that performs optimization constrained to the data manifold learned by a generative model. The core idea is to reparameterize the optimization problem in the latent base space of a diffeomorphic generative model (e.g., a flow or diffusion model). By leveraging differential geometry, the authors prove that gradient descent in the latent space is first-order equivalent to Riemannian gradient descent on the data manifold. The method is further extended to matrix Lie groups, such as SO(3) and SE(3), enabling applications to 3D geometry problems like protein structure refinement and protein–ligand docking. Empirical results demonstrate faster convergence, smoother optimization trajectories, and improved geometric consistency compared to Euclidean and guidance-based methods.

### Strengths
This work makes a significant contribution at the intersection of geometric machine learning and generative model optimization.
- Introduces a new and conceptually elegant framework linking differential geometry and generative model optimization.
The diffeomorphic parameterization idea is both theoretically deep and practically impactful.
- The proofs (Section 3) are mathematically solid, with correct use of Riemannian and Lie group formalism.
- Writing is precise and pedagogical; the paper explains nontrivial geometry intuitively without oversimplifying.
- Provides a general foundation for manifold-aware optimization, relevant to numerous scientific and AI domains.
-  Demonstrates real impact on SE(3) applications (e.g., protein docking) where geometric consistency is crucial.

### Weaknesses
- While results on protein-related tasks are convincing, evaluations on non-biological manifolds (e.g., moulecules, crystal materials) would help establish broader generality.
- The theoretical guarantees rely on the diffeomorphism being smooth and bijective. In practice, diffusion models may only approximate this property. A discussion or metric quantifying deviation from diffeomorphism would strengthen the work.

### Questions
- How sensitive is the optimization quality to the accuracy of the learned diffeomorphism?
Does the framework degrade gracefully if the generative mapping g slightly violates invertibility or smoothness?
- Have the authors explored second-order extensions (e.g., Riemannian Newton or natural gradient methods) within the same diffeomorphic framework?
- Is the adjoint-state method in Section 4 computationally scalable for large molecule systems (many SE(3) transformations)?
- For reproducibility, will the authors release the code and pretrained diffeomorphic models used in the experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper is based on D-flow and extends gradient-based guidance in SO(3) space to guide protein backbone generation. The demonstrated applications are interesting. However, the contributions are limited in methodology, experiments, and theory (see weakness).

### Strengths
- Clear presentation: The paper provides illustrative toy data experiments, algorithm pseudocode, and detailed implementations of geometric operators. The writing is clear and free of obvious errors.
- Interesting domain application: protein backbone secondary structure modification, pocket-ligand docking, and energy optimization are important applications in the field of protein design.

### Weaknesses
- From a method perspective, this method extends D-flow to SO(3) space, which has already been explored in previous literature, e.g., [1].
- From an application perspective, the paper lacks quantitative evaluation against existing guided generation methods.
- From a theoretical perspective, the main results (Theorems 1 and 2) are restatements of well-established results in Riemannian optimization and matrix Lie groups (e.g., [2,3]), with limited novel theoretical contribution.

[1] Wang, Luran, et al. "Training free guided flow matching with optimal control." arXiv preprint arXiv:2410.18070 (2024).

[2] Absil, P-A., Robert Mahony, and Rodolphe Sepulchre. Optimization algorithms on matrix manifolds. Princeton University Press, 2008.

[3] Do Carmo, Manfredo Perdigao, and J. Flaherty Francis. Riemannian geometry. Vol. 2. Boston: Birkhäuser, 1992.

### Questions
- Complexity and efficiency are not clearly discussed.
- It is unclear how much the proposed method differs from D-flow in practice. If we ignore the rotations (i.e., frame orientations) and only consider Cα atoms in protein backbone generation, which is in SE(3), the method appears to reduce to a standard D-flow formulation. The paper does not explicitly clarify whether any additional benefits arise beyond applying D-flow to Euclidean coordinates embedded in SE(3).

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Diffeomorphic Optimization, a novel method for optimizing arbitrary differentiable cost functions on data manifolds by leveraging flow-based and diffusion generative models. The key insight is that these models learn diffeomorphic (smooth and invertible) maps from simple base distributions to complex data distributions, allowing optimization to be performed in the simpler base space rather than directly on the data manifold.  A significant technical contribution is extending this framework to matrix Lie groups (SO(3) and SE(3)), which are crucial for protein structure representation. The authors develop two methods for backpropagation through ODE solvers on these groups: (1) repurposing existing autograd engines to compute Riemannian gradients, and (2) deriving an adjoint state method for matrix Lie groups. The method is demonstrated on protein design tasks using state-of-the-art generative models (FrameFlow, DiffDock, AlphaFlow), showing successful optimization of secondary structure, protein-ligand docking scores, and Rosetta energy functions while maintaining physically plausible structures throughout the optimization trajectory.

### Strengths
The paper provides a mathematically rigorous foundation connecting differential geometry, generative models, and optimization. Theorem 1 establishes that gradient descent in base space is equivalent to gradient descent on the data manifold up to quadratic corrections, giving the core theoretical motivation. The two proposed methods for handling backpropagation (autograd repurposing and adjoint state method) are both theoretically sound and practically implementable.

The method demonstrates clear improvements across diverse protein tasks.

### Weaknesses
The method requires backpropagation over the entire ODE integration trajectory, which is computationally expensive. While the authors argue this is acceptable in protein design due to wet-lab bottlenecks, it limits broader applicability.

While the authors' optimization scheme allows us to enforce manifold constraints, it is not clear whether it biases the optimization towards certain minima. As the optimization objective is highly non-convex, it is essential to explore and obtain a reasonable estimate of the global minima rather than focusing on gradient descent to find a local minima. Sampling allows us to see a broader range of solutions rather than finding a single optimal solution. Under this light, it is unclear if the proposed methodology provides a practical advantage over guided sampling methods.

The theory and experiments do not provide any details on the accumulation of estimation errors during backpropagation over the entire ODE trajectory. Some analysis of the estimation error would be highly relevant for practical adoption.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3
