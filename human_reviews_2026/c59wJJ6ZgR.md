# DRiFT: Differentiable Grid-Based Rigid-Fluid Coupling for Training and Control

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Intelligent agents, interacting with physical environments, require an accurate
understanding of the consequences of their action for efficient learning. Such
agents are often trained inside simulated environments to alleviate over dependence
on data, and gradients from such a simulation can help in training the agent. To this
end, we present an end-to-end differentiable grid-based fluid simulation including
strong two-way coupling with rigid bodies. In the forward pass, the solid-fluid
boundary conditions are converted to a monolithic linear pressure solve using a
variational method. For the backpropagation, we introduce a novel method of
calculating and propagating gradients for the combined fluid-solid state using the
adjoint method, which runs faster than the forward solve. This implementation,
which is customized for coupling rigid bodies with inviscid fluids, is more suitable
over general purpose methods like automatic differentiation, for use cases where
performance is key for analyzing overall flow patterns and learning fluid properties.
We demonstrate the utility of our simulator in training a neural network to learn
optimal control for general target states. Additionally, we show the effectiveness
of our differentiable simulator in isolation, by using the generated gradients for
simple derivative based optimization tasks. Finally, we showcase the accuracy,
robustness and efficiency of our gradient computation method.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces DRiFT, a grid-based differentiable simulator designed to handle strong two-way coupling between rigid bodies and inviscid fluids. The primary contribution is a complete, end-to-end differentiable pipeline. In the forward pass, solid-fluid boundary conditions are unified into a monolithic linear pressure solve using a variational method. For the backward pass, the authors derive a novel adjoint method to efficiently back-propagate gradients for the combined fluid-solid state. They claim this adjoint pass is faster than the forward solve and more suitable than general-purpose automatic differentiation (AD) for performance-critical applications. The utility of the simulator and its gradients is demonstrated in derivative-based optimization tasks, such as initial state estimation, and for training neural network controllers for optimal control.

### Strengths
- Non-trivial technical contribution: the paper presents a complete, end-to-end differentiable simulator for strong two-way rigid-fluid coupling using a Eulerian method. The analytical derivation of the adjoint pass for the entire pipeline, especially for the monolithic variational pressure solve.
- The resulting adjoint-based gradient computation is demonstrated to be exceptionally efficient. The authors report that the backward pass is faster than the forward solve and significantly outperforms general-purpose automatic differentiation frameworks like PhiFlow in runtime tests. This is promising, and it pushes in a direction that will benefit both the fluid dynamics and ICLR communities.
- The practical utility of the simulator is demonstrated on diverse optimization tasks. The computed gradients are successfully used for solving inverse problems and for training neural network policies for optimal control.

### Weaknesses
1) L064: It is not clear whether $\mathcal{L}$ is the objective terminal state, or the loss function that compares it to the predicted terminal stated. It appears to be described as the former, which is inconsistent with the typical machine learning literature.
2) Related work outlining a review of differentiable programming are fairly limited. One could think of [1, 2] among other studies
3) The method is limited to inviscid fluid simulation, limiting its applicability.
4) L304: the website link appears to be broken, or doesn't work. While the still frames are nice, they don't give an overview of the initial velocity which is often optimised. Having videos would significantly complement the results.
5) Figure 5 appears to show a single method, not two (including DiffFR) as claimed in L374.
6) The overview of the pipeline is interesting, but still limited. The Pseudo-code in Algorithm 1 would be much more useful if placed in the main text.

### Minor issues:
- L173: Figure ?

### References
- [ 1 ] Toshev et al., JAX-SPH: A Differentiable Smoothed Particle Hydrodynamics Framework, ICLR 2024 Workshop on AI4Differential Equations in Science
- [ 2 ] Nzoyem et al., A Comparison of Mesh-Free Differentiable Programming and Data-Driven Strategies for Optimal Control under PDE Constraints, SuperComputing Workshop on AI4Science

### Questions
1) L447: What steps were taken to ensure the Runtime Analysis against PhiFlow is fair ? Please provide details of this experiment, which is incredibly vague in the current manuscript ! (This is all the more concerning given that your code is in C++, while PhiFlow is in Python, which should definately be slower than yours) 
2) What specifically makes the adjoint pass faster than the forward pass, as evidenced in Table 2. This is counter-intuitive, and the paper does not emphasize this enough. Please provide computationaal complexity analysis for Algorithms 2 and 3.
3) What are the main challenges in extending the proposed adjoint derivation to include viscous effects?
4) The paper claims the approach is highly scalable, but the experiments use relatively coarse grids (e.g., in L314). What is the computational and memory cost of this appraoch, and how far can you scale on the hardware avaialable?
5) L352: Specifically in the text optimisation, why was rotation not considered ?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a grid-based two-way coupled differentiable fluids solver. The authors implement an adjoint method for computing control variable updates relative to a user-defined reward function. One of the contributions that the authors present is a derivation of analytic gradients for computing updates for the gradients of the pressure solve, that affect both the fluid and the coupled rigid body. The authors also use a variational formulation for the fluid-solid boundary, and a ghost fluid method for the fluid-air interface. Their results demonstrate that the proposed approach works reasonably for a set of predefined low-resolution guidance tasks.

### Strengths
- Working with fluids optimization is a though problem. It usually slow, hard to debug, requires a lot of memory, boundary conditions can be tricky, and the method has to be implemented precisely in all its stages to produce correct results. 
- The proposed variational approach and the ghost fluid discretization for liquid-air interfaces are solid modelling choices. 
- The derivation of the analytical updates seem correct, and the results demonstrate that the method works.

### Weaknesses
- The quality of the results is sub-par. Grid resolutions are really coarse and there are not a lot of different examples. Also hard to really evaluate the method, since I was not able to access the website with the video results. 
- The method does not seem efficient. DiffFR uses 237699 particles and it is 2.3 times slower than the proposed method, but the authors compare against a fairly coarse grid setup of 39x15x24 = 14k cells. Therefore the proposed method uses about 5% of the variables of DiffFR and its x2 faster than it. 
- A first-order (Semi Lagrangian advection with forward Euler) fluid solver is an outdated approach. Hybrid (grid + particles), impulse-based methods are more effective and precise. 
- Missing references: "Honey, I Shrunk the Domain: Frequency-aware Force Field Reduction for Efficient Fluids Optimization", "Efficient Solver for Spacetime Control of Smoke".
- Minor typos
   - On plots, whats the concept of Epochs? Shouldn't it be iterations?
   - L173: Figure ??
   - L107: Langrangian

### Questions
I would like to understand what are the computational limitations of the proposed approach. How the memory and time would scale with grid size and what would be effective ways to deal with longer simulations.

### Soundness
3

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
2

### Summary
This paper presents DRiFT, a differentiable grid-based fluid simulation with two-way coupling between fluid and rigid bodies. The grid-based discretization enables analytically deriving the gradients over each phase of the fluid simulation, covering fluid velocity advection, pressure solve for boundary conditions, rigid body update, and velocity correction for boundary conditions. The authors demonstrate that gradients from DRiFT can be used for simple optimization tasks, including training a neural network to predict a control sequence. DRiFT obtains 5-10x speedup over a comparable Eulerian differentiable fluid simulation that uses auto-differentiation from PyTorch to compute gradients.

### Strengths
- The authors demonstrate how to analytically derive gradients for grid-based fluid simulation. They demonstrate that the gradients computed by adjoint-based gradient computation can be used for optimization of the initial state to achieve some desired final state. given some cost function, and to optimize control forces by training some neural-network based controller to predict a control sequence.

- The proposed differentiable fluid simulation considers strong two-way coupling between fluid and rigid bodies. I believe most prior work on differentiable soft-body simulation is only capable of one-way coupling.

- The authors release their differentiable fluid simulation code, which benefits reproducibility.

### Weaknesses
- Based on the videos in the supplementary, the simulated fluid appears highly viscous. How stable is the simulation over longer simulation times? How accurate is the fluid simulation compared to Aquarium, PhiFlow, or other methods for fluid simulation, given that the proposed method uses a grid-based discretization?

### Questions
It would be helpful to discuss hybrid Eulerian-Lagrangian approaches for fluid simulation such as the Material Point Method (MPM) used in differentiable simulators Fluidlab [1], DaXBench [2], or Rewarped [3]. For instance, I believe MPM easily handles simulating different types of fluids in the same scene. How easy is it to extend DRiFT to multiple fluids?

[1] https://arxiv.org/abs/2303.02346

[2] https://arxiv.org/abs/2210.13066

[3] https://arxiv.org/abs/2412.12089

Line297: Does DRiFT support parallel simulation? Or is OpenMP/CUDA only used to parallelize operations in a single physics scene?

Figure 4: Choice of epochs to visualize seems arbitrary (39, 58, 19). How stable is the optimization procedure? Running simulations with different random seeds or number of iterations and including error bars for Figure 6 would be helpful.

Section 6.2: Are the control forces executed in an open loop, i.e the policy predicts the entire control sequence?

- - -

[minor]

Line173: Missing figure number reference

Figure 4: Should be epoch 19 not 10?

### Soundness
2

### Presentation
3

### Contribution
2
