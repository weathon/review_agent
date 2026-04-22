# High-dimensional Mean-Field Games by Particle-based Flow Matching

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Mean-field games (MFGs) study the Nash equilibrium of systems with a continuum of interacting agents, which can be formulated as the fixed-point of optimal control problems. 
They provide a unified framework for a variety of problems, including both potential and non-potential games, with applications in areas such as generative modeling.
Despite their broad applicability, solving high-dimensional MFGs remains a significant challenge due to fundamental computational and analytical obstacles. 
In this work, we propose a particle-based deep Flow Matching (FM) method to tackle high-dimensional MFG computation.
In each iteration of our proximal fixed-point scheme, particles are updated using first-order information, and a flow neural network is trained to match the velocity of the sample trajectories. 
Theoretically, in the optimal control setting, we prove that our scheme converges to a stationary point sublinearly, and upgrade to linear (exponential) convergence under additional convexity assumptions.
Our proof uses FM to induce an Eulerian coordinate (density-based) from a Lagrangian one (particle-based), and this also leads to certain equivalence results between the two formulations for MFGs when the Eulerian solution is sufficiently regular.
Our method demonstrates promising experimental performance on MFGs in high dimensions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper is concerned with solving potentially high-dimensional Mean-Field Games (MFGs), which are challenging due to their optimal control fixed-point structure. Existing solvers often rely on mesh-based discretizations, and as a result become computationally infeasible when applied to high-dimensional problems. To overcome this, the authors propose a particle-based deep flow matching method along with a trust-region fixed-point scheme for efficiently solving high-dimensional MFGs.

### Strengths
1. The paper is well written and easy to follow. 
2. The claims are justified with proofs whenever necessary.

### Weaknesses
1. Since the focus is on solving high-dimensional problems, out of the three example cases, it seems only the image-to-image translation results are noteworthy. 
 
2. Line 473: "We can observe that our method produces smooth and coherent translations, particularly in terms of color consistency and reduction of visual artifacts." The authors only present the results obtained using their method, and this claim is rather relative than absolute. Hence, results using alternative methods are perhaps needed to support this claim/observation.

### Questions
Given that there exist deep learning implementations for solving MFGs, would it be possible to compare against such methods (I was able to find some below)? I think this result would greatly strengthen the paper if indeed the proposed method converges orders of magnitude faster as hinted in the paper. 


[1] Chen, X., et al., *Physics-Informed Neural Operator for Coupled Forward-Backward Partial Differential Equations*. 

[2] Chen, X., et al., *A Hybrid Framework of Reinforcement Learning and Physics-Informed Deep Learning for Spatiotemporal Mean Field Games*.

### Soundness
3

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
This paper proposes method for solving deterministic mean field games by combining a particle-based approach with a flow network by alternatively optimizing particle trajectories to minimize cost and the flow network to match the optimized particles. The proposed numerical scheme is shown to converge sublinearly for potential MFGs and linearly for optimal control problems under suitable assumptions. Numerical experiments on a non-potential MFG toy example confirm the convergence of the proposed scheme, while an image-to-image translation task on a learned latent space shows the proposed method produces smooth transitions and yields lower FID compared to baseline methods.

### Strengths
- To the best of my knowledge, the proposed method for solving deterministic mean field games is novel in the literature
- The writing is clear and easy to understand
- The empirical results help to demonstrate the effectiveness of the proposed method

### Weaknesses
- While the authors claim to solve (general) mean field games, it seems like the work only tackles **deterministic** mean field games, i.e., where the dynamics of each agent do not have any stochastic terms (e.g., equation in Line 170 doesn’t any diffusion related terms, and the evolution of the particles in (6) is an ODE and not a SDE). This could be somewhat misleading to readers, especially as this is not stated in either the title, abstract, or in the main body of the paper.

### Questions
- In relation to the above comment on the deterministic dynamics, it would be helpful to the readers for the author to cite and briefly discuss these additional works that do tackle this subclass, (e.g., [A, B]) and potentially compare against some of these methods (though not required), as well as ones that don’t make this assumption (e.g., [C, D, E, F]).
- Minor: Typo with “following” in line 469

[A] Gomes, Diogo, Julian Gutierrez, and Mathieu Lauriere. "Machine learning architectures for price formation models." *Applied Mathematics & Optimization* 88.1 (2023): 23.

[B] Assouli, Mouhcine, et al. "Initialization-driven neural generation and training for high-dimensional optimal control and first-order mean field games." *arXiv preprint arXiv:2507.15126* (2025).

[C] Lin, Alex Tong, et al. "Alternating the population and control neural networks to solve high-dimensional stochastic mean-field games." *Proceedings of the National Academy of Sciences* 118.31 (2021)

[D] Liu, Guan-Horng, et al. "Deep generalized schrödinger bridge." *Advances in Neural Information Processing Systems* 35 (2022)
[E] Chen, Yongxin. "Density control of interacting agent systems." *IEEE Transactions on Automatic Control* 69.1 (2023)

[F] Liu, Guan-Horng, et al. "Generalized Schr\" odinger Bridge Matching." *arXiv preprint arXiv:2310.02233* (2023).

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes solving mean field games (MFGs) using a particle-based flow matching method. For a certain type of MFG, the proposed method replaces the backward best-response calculation and forward population simulation in a typical fixed-point iteration method with first-order gradient updates and particle-based flow matching. Theoretical convergence guarantees are provided for potential MFGs and the optimal control setting.

### Strengths
- The paper tackles the important challenge faced by fixed-point iteration methods for solving MFGs.
- The proposed method is novel and reasonable.
- The paper is mathematically rigorous and provides theoretical convergence analysis for the proposed method for certain classes of MFGs.

### Weaknesses
### Optimal Transport or Mean Field Game?

The model considered in this work is heavily inspired by optimal transport models but does not have much MFG flavor.

First, the cost function has a very special structure: the running cost is separable into a control-related cost and a population-related cost. Furthermore, the control-related cost term is a simple quadratic function. These are two significant simplifications of the cost function that are not satisfied by general MFGs.

Second, a very special dynamics model, an Euclidean spatial state space with deterministic velocity control, is considered. This imposes strong assumptions, including continuity and determinism in the state evolution and that agents have full knowledge of the dynamics. General games do not satisfy these assumptions.

Third, because of the simple dynamics model considered, agents' transitions are independent of the population distribution. This significantly reduces agents' coupling and thus weakens the model's connection to the general dynamic MFG framework, where both agents' rewards and transitions depend on the population distribution.

These model considerations and simplifications make sense from the perspective of optimal transport, but they substantially limit the applicability of the proposed method to general MFGs.

### The Claim of "Simulation-Free" and Its Benefits

It is mentioned multiple times that the proposed method is "simulation-free", meaning that it does not simulate a PDE/ODE system. However, the trajectories of the particles still need to be simulated according to ODE (6). I do not see the fundamental difference between these two types of "simulation".

Furthermore, it is claimed that other deep flow methods that simulate the population struggle in high-dimensional settings. However, it is not clear to me what the benefits of a particle-based flow matching method are in this regard. Specifically, the high-dimensional nature also requires an (exponentially) large number of particle trajectories $\{ X_{i,t_{j}}^{(k)} \}_{i,j}^{n,m}$ to approximate the true trajectory field $X^{(k)}$.

### Related Work

Recognizing the challenges faced by fixed-point iteration methods, several recent works \[1,2,3\] also propose simple methods that eliminate the forward-backward (best-response and population simulation) structure and demonstrate better convergence properties. These works are closely related to the discussions in the paper:

- They are generalized Frank–Wolfe algorithms that update parameters using first-order gradient information. Thus, these methods do not need to calculate the exact best response or the induced population at each iteration. They share the same motivation that "when the objective changes at each step, moving toward the best response of the current step eventually leads to an MFNE" (Lines 78–79) and that "convergence to a fixed point can still be achieved as long as each update yields improvement with respect to the current objective" (Lines 293–294).
- They are truly simulation-free methods, as the population distribution is learned in a model-free manner and can be estimated from a single trajectory of a single representative agent.
- \[3\] addresses continuous state–action spaces by incorporating function approximation.

\[1\]: Angiuli, A., Fouque, J.-P., Laurière, M., 2022. Unified reinforcement Q-learning for mean field game and control problems. Mathematics of Control, Signals, and Systems 34, 217–271.  
\[2\]: Zeng, S., Bhatt, S., Koppel, A., Ganesh, S., 2025. Learning in herding mean field games: Single-loop algorithm with finite-time convergence analysis, in: International Conference on Artificial Intelligence and Statistics. PMLR.  
\[3\]: Zhang, C., Chen, X., Di, X., 2025. Stochastic semi-gradient descent for learning mean field games with population-aware function approximation, in: International Conference on Learning Representations. PMLR.

### Questions
- Why do you call the update rule (8) a "trust-region strategy"? Where is the trust-region radius, its update rule, and the rejection rule? Isn’t it a soft regularization method? Do you mean that you want to constrain the next iteration to be near the current iteration?
- It is stated that the proposed method circumvents the "main challenge in solving MFGs … enforcing the continuity equation constraint." Is it because it learns a velocity field that matches the trajectories, so it automatically satisfies the continuity equation? Am I correct in understanding that other works enforce the continuity equation by simulating a population consistent with the control, while you learn a control consistent with the population trajectories?
- How are $n_{1}$ and $n_{2}$ related to $n$ in Algorithm 1?
- What is the non-potential MFG considered in Section 4.2? Is it purely illustrative or a model for real applications?

Please also see the questions in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2
