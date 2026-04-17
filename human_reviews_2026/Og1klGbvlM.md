# HOTA: Hamiltonian framework for Optimal Transport Advection

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8, 6

## Abstract
Optimal transport (OT) has become a natural framework for guiding the probability flows. Yet, the majority of recent generative models assume trivial geometry (e.g., Euclidean) and rely on strong density-estimation assumptions, yielding trajectories that do not respect the true principles of optimality in the underlying manifold. We present Hamiltonian Optimal Transport Advection (HOTA), a Hamilton–Jacobi–Bellman based method that tackles the dual dynamical OT problem explicitly through Kantorovich potentials, enabling efficient and scalable trajectory optimization. Our approach effectively evades the need for explicit density modeling, performing even when the cost functionals are non-smooth. Empirically, HOTA outperforms all baselines in standard benchmarks, as well as in custom datasets with non-differentiable costs, both in terms of feasibility and optimality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a Hamiltonian Framework for Optimal Transport Advection. The contribution of this work appears to be the loss functions (Eqs. 14-17) and the algorithm, which learns dynamical OT problems with a dual Hamiltonian formulation. The authors claim that the proposed approach surpasses SOTA performance in low-dimensional settings.

### Strengths
* The proposed method is mostly sound, with a clear presentation of the mathematical derivation.
* The outlined experiments validate the effectiveness of the formulation.

### Weaknesses
* The motivation, reasoning, and verification for considering the dual GSB problem are not clear and need enhancement.
* The theoretical arguments in Section 3 are somewhat well known from stochastic mechanics. Therefore, I think contributions of this work seems to be focused in the methodology part.
* Sample inefficiency. I believe Eq. (17) contains the gradient of Hessians, which cannot be easily computed in high dimensionality. The computation of Eqs. (14-15) is seemingly not sample-efficient for large models. The authors are encouraged to discuss this limitation if the method requires more training time than the benchmarks.

### Questions
* Can HOTA be applied to image-to-image translation tasks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for estimating the time-dependent Kantorovich potential $s:[0,1]×R^d \rightarrow R$ using tools from optimal transport and Hamilton–Jacobi–Bellman (HJB) theory. The terminal potential $s(1,⋅)$ is learned using the Kantorovich dual formulation, and intermediate values $s(t, \cdot)$ are obtained by solving the HJB equation backward in time. The approach is demonstrated on 2D synthetic datasets, a 3D sphere-to-sphere transport, and opinion depolarization task.

### Strengths
1. The paper clearly states the goal of learning the full space–time potential $s(t,x)$. This only requires a single neural network parameterizes the value function.

2. The theoretical derivation is well structured, connecting the Kantorovich duality with the HJB equation in a coherent way. Visualizations of the learned transport trajectories and potentials are helpful for intuition.

### Weaknesses
1. **Limited originality relative to prior work.** The overall methodology, estimating a terminal potential via the Kantorovich dual and propagating it using HJB, closely resembles prior work [1]. The main difference is the inclusion of a state-dependent running cost, but the theoretical structure remains largely unchanged. As a result, the contribution appears incremental rather than conceptually novel.

2. **Concerns about algorithmic practicality.** 

- Optimizing the unregularized Kantorovich dual objective is known to be unstable and highly sensitive to initialization and regularization.

- Accurately enforcing the HJB equation corresponds to solving a nonlinear PDE, which is notoriously difficult, especially in high-dimensional spaces.

- - Even small numerical errors in estimating $s(t,x)$ may lead to inaccurate gradient estimation $\nabla s(t, x)$. For instance, if value function has little oscillatory or noisy estimates, that would change the gradient estimation of the value function $s$ drastically, leading to inaccurate estimation of the control. 

- The method follows an Eulerian perspective, requiring accurate approximation of $s(t,x)$ across the entire state space. This becomes computationally infeasible beyond very low-dimensional domains, suggesting limited scalability beyond 2D or 3D, or toy datasets.

3. **Experimental validation is limited and low-dimensional.**

- All experiments are conducted on synthetic datasets. These do not convincingly demonstrate scalability or robustness.

- There are no experiments on standard OT benchmarks, such as LiDAR point cloud transport [2], image-to-image translation [1], or Gaussian-to-GMM transport in higher dimensions.

- The paper does not include analyses of PDE residuals, HJB constraint violation, or dual objective convergence, so it is unclear whether the training process actually solves the intended optimization problem.


References

[1] Scalable Simulation-free Entropic Unbalanced Optimal Transport

[2] Generalized Schrodinger Bridge Matching

### Questions
- Can the authors test their method on higher-dimensional synthetic data (e.g., Gaussian mixtures in 10–50 dimensions)?

- Could they also demonstrate performance on standard datasets such as LiDAR point clouds, CelebA image translation, or similar benchmarks?

- Can the authors report training curves for (1) the HJB residual (PDE violation), and (2)vthe Kantorovich dual loss? This would help verify whether the proposed method truly solves the coupled dual–HJB system.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a learning approach to address the dynamical optimal transport (OT) problem with non-smooth cost functionals, enabling the incorporation of the underlying geometry of the data manifold. The proposed method is developed by leveraging the relationship between the dynamical OT problem and the Hamilton–Jacobi–Bellman (HJB) equation, where the value function is approximated using a neural network to learn the transport mapping. Experimental results on synthetic datasets demonstrate that the proposed method achieves better alignment with the target distribution and incurs lower transport costs compared to existing (Generalized) Schrödinger Bridge models.

### Strengths
The paper makes a original contribution by establishing a novel connection between the Generalized Schrödinger Bridge (GSB) problem and the Hamilton–Jacobi–Bellman (HJB) equation. By approximating the value function with a parametric model, the authors develop a tractable learning framework for modeling the dynamics of optimal transport (OT) between distributions. This provides a fresh perspective on OT-based diffusion models with fewer assumptions on flow densities.

The work is of strong technical quality, with a sound theoretical foundation and convincing experimental validation. Results on synthetic datasets show improved target matching and reduced transport cost compared to existing (Generalized) Schrödinger Bridge methods. The inclusion of code further supports reproducibility.

The paper is clearly written and well structured, making the complex ideas accessible. Its significance lies in broadening the applicability of OT methods and opening new possibilities for learning-based transport and diffusion modeling.

### Weaknesses
A key limitation of the paper is its exclusive reliance on synthetic datasets, which constrains the evaluation of the method’s applicability to real-world problems. While the authors acknowledge that their approach struggles with complex data such as images, they do not provide a clear explanation or empirical justification for this limitation. A more detailed analysis of why the model underperforms in such settings, along with experiments or ablation studies incorporating stronger inductive biases in the architecture, would provide concrete evidence to support their claims and guide future improvements.
From a theoretical perspective, the paper would be strengthened by a formal convergence analysis of the learning algorithm to ensure stability and reliability. Additionally, a discussion of the approximation error introduced by neural network parameterization and the estimation error due to sampling would offer valuable insights into the computational behavior and robustness of the approach. Addressing these aspects would enhance both the theoretical rigor and the practical significance of the work.

### Questions
Could the authors provide a case study or example illustrating a scenario where non-smooth cost functionals pose challenges?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new way to solve Generalized Schrödinger Bridge problems by tying the HJB value function $s(t,x)$ to a Kantorovich terminal condition. The idea is to optimize two losses jointly. First, enforce that the endpoint potential matches the Kantorovich potential via a Lagrangian relaxation. Second, ensure $s$ really is a value function by having it satisfy the corresponding HJB equation as a PINN-style constraint. With that in place, the optimal drift is recovered as $-\nabla s$. They test on standard 2D toy setups and also explore scaling to higher dimensions on the sphere.

### Strengths
The method is original in how it ties the value function to a Kantorovich terminal condition and enforces the HJB with a PINN, giving a clean route to recover the drift as a gradient. The empirical results are strong, with clear improvements over baselines and ablations that isolate the effect of the replay buffer, EMA targets, and gradient balancing. The technical presentation is mostly clear, with a coherent objective that aligns the endpoint constraint and the dynamics constraint, and enough detail to make the optimization reproducible. In terms of significance, the approach looks broadly useful for GSB and related control and transport problems, and the sphere experiments suggest promise beyond small 2D toy examples.

### Weaknesses
The paper reads as if it targets readers already deeply familiar with the topic, which raises the entry cost for newcomers. Several parts of the exposition could be tightened, especially the ablation study. See the Questions section. The main practical limitations are that, first, enforcing the HJB equation in higher dimensions and on more complex targets may be challenging, and second, the learned drift requires evaluating $\nabla s$ at inference.

### Questions
1) Please clarify the inference-time cost of computing the network gradient. How does evaluating $\nabla s$ compare to other methods at inference?

2) Could you provide more detail on how the buffer is managed? In particular, why is only the first trajectory added in the pseudocode? Also, please explain why you use the interpolation sample steps $N_0$ in the algorithm.

3) Why does the buffer improve results? Is the benefit primarily stability, or mainly computational efficiency? In an ideal setting with unlimited compute, would resampling fresh data each step be strictly better?

4) In the ablation, what exactly does "without EMA" remove? Could you discuss alternatives to the PINN formulation around equations 14–15? For example, could you remove EMA and enable stop-grad on specific terms, and what tradeoffs led you to the final choice?

5) What motivated the inclusion of the angular acceleration term, and how does it affect results in practice?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces HOTA, a novel method for solving the Generalized Schrödinger Bridge (GSB) problem. The approach is based on a dual formulation that seeks a Kantorovich potential aligning the source and target distributions while satisfying the associated Hamilton–Jacobi–Bellman (HJB) constraint.

### Strengths
The paper proves that duality holds under mild conditions. It further demonstrates that the method is robust to complex geometries and remains effective even for non-smooth cost functions. By employing the Euler–Maruyama scheme to approximate the solution of the underlying SDE, the proposed approach avoids explicit density modeling and thereby simplifies the learning process. In numerical experiments, HOTA consistently outperforms several state-of-the-art methods across a range of benchmark settings.

### Weaknesses
While the proposed method demonstrates impressive performance and scalability, the experimental setup and model architecture appear relatively simple. Since the paper directly compares with GSBM, it would be greatly strengthened by including evaluations on more complex datasets used in that work, such as AFHQ.

Moreover, as only a lightweight 4-layer MLP was tested, it remains unclear whether the reduced computational overhead reported in Appendix B would persist when employing a heavier model, such as a U-Net.

The HJB constraint is enforced via an $L^2$ loss over sampled trajectories, which does not guarantee that the solution satisfies the HJB equation over the entire space $\mathbb{R}^d$. As noted in [1], the $L^2$ loss may also not be ideal for solving HJB equations. Therefore, the authors should investigate, at least numerically, how well the learned solution satisfies the HJB constraint and examine the behavior of $s_\theta$ in out-of-sample regions.

**Minor issues/typos**:
- Line 183: “c” in “c-transform” was not put in a math environment.
- Line 652: “are” should be “and”.
- Line 786: “Buy” should be “By”.

**Reference**:

[1] Chuwei Wang, Shanda Li, Di He, and Liwei Wang. Is $L^2$ physics informed loss always suitable for training physics? [arXiv: 2206.02016](https://arxiv.org/abs/2206.02016)

### Questions
HOTA currently uses linear interpolation to initialize the trajectories. Could a more informed initialization be designed by leveraging the state cost $U(x)$? In several cases shown in Figure 1, linear trajectories obviously pass through regions that the final solution should avoid, which intuitively seems to hinder learning efficiency.

The paper should define the acronym EMA (Exponential Moving Average) upon its first occurrence and clarify the relationship between the target model $\bar{s}$ and the main model $s_\theta$.

### Soundness
3

### Presentation
4

### Contribution
3
