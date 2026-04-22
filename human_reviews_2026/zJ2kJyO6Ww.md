# Physics-Informed Conditional Diffusion for Multi-Modal PDEs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Many physical systems that are represented by partial differential equations (PDEs) admit multiple valid solutions, such as eigenstates of differential operators, or wave modes, yet most neural PDE surrogates are deterministic and collapse to averages. This multiplicity of solutions is especially predominant in various engineering and scientific domains ranging from acoustics and seismology to quantum systems. With the ability to generate or complete sparse measurements, diffusion-based approaches to solve PDEs by sampling physically valid solutions are gaining traction as an alternative to traditional numerical solvers. In this paper, we present a novel physics-informed conditional diffusion framework for multi-modal PDEs, called PDEDIFF, that learns distributions over solution fields from sparse, irregular samples while enforcing governing equations and boundary conditions through mesh-free residual penalties computed by automatic differentiation. PDEDIFF is capable of effectively solving PDEs with multiple valid solutions by learning $\mathbb{P}[Y|X]$, i.e., it learns a solution field $Y$ for a corresponding input spatial information $X$. Unlike Physics‑Informed Neural Networks (PINNs), which minimize residuals around expected values $\mathbb{E}[Y|X]$ and hence tend to regress toward a conditional mean, PDEDIFF samples diverse physically consistent solutions by integrating PDE residuals directly into the diffusion objective. Our results indicate that generative, physics-informed diffusion is a practical tool for uncertainty-aware and multi-modal PDE modeling in low-to-moderate dimensions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper suggests an approach for physics (i.e. PDE) informed diffusion models that work on PDE systems that are not well-posed. This approach has several advantages: independences of input positions, i.e. being meshless;

### Strengths
The approach is sound and the paper is mostly understandable. There is novelty (despite being a combination of well-known techniques) in the paper that would in general make it a fit for ICLR. Enough additional information is in the appendixes.

### Weaknesses
My main problems is with the experiments:
 - I do not understand how you can reasonably compare to PINNs, which are a regression technique and thereby, kind of by definition, present you just a mean function.
 - Your experiments are all linear or almost linear equations (no systems, only weak non-linearities). It sounds somewhat restrictive to avoid non-linear equations. Furthermore, for linear PDE systems, there is a strong linear of research for generative models using Gaussian Processes, which might very well be superior to the presented approach in the 3 out of 4 test cases.
(This is despite the nice choise of evaluation metrics.)
I would need a clear indication of me misreading this problem to change my rating.

There are some minor points with presentations. While the paper is understandable, it does is not easy to read the paper and it would benefit from stylistic improvements.

### Questions
l. 223 definite article in "the probability distribution"?
in (10), do you need to reconstruct $\hat\Psi_0$ each time in training?
l. 306: Which eigenvalues?
l. 323: Which eigenstates?
l. 351: Which nodes?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a scheme for recovering distributions over solution fields of PDEs using physics informed diffusion models, Specifically they train a DDPM with a PINN style autodiff loss. The method is tested on a few benchmarks looking at eigen decompositions, Schrodinger equations, and helmoltz problems.

### Strengths
- The paper is written clearly.

- The proposed method shows advantages over PINNs, however this is not the gold standard in computation of these problems.

- The appendix takes care is carefully explaining the experiments therein.

### Weaknesses
- There exist standard methodology for computing the solution to the mentioned problems, such as finite elements with eigen decompositions. Why are these not compared to? It is difficult to assess the usefulness of the proposed methodology if there is not shown to be a concrete advantage of this method against classical schemes.

- I suspect finite difference stencils are more computationally efficient and less memory intensive than using automatic differentiation through the model. However, there seems to be no mention of computational efficiency, runtime evaluation, etc. Why was this not reported?

- Overall the methodology is not very well motivated, as existing tools exist to carry out these computations, and limited testing is presented.

- If the advantage of the method is to blend data and physics, this should much more clearly be stated, explained, and explored with numerical experiments.

- Overall the novelty of the proposed method is quite limited, it mainly encompasses combining an autodiff residual loss with a diffusion model.

### Questions
- Can you please explain what the term "modality recall" means? (page 2 contribution 4)

- If one were to use this method, how would you choose the c_res tradeoff parameter in practice? This hyperparameter selection problem is underexplored.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors address a problem faced by PINN approaches: when a partial differential equation (PDE) problem has multiple feasible solutions (multimodality), typical neural network approximators converge toward the mean and "smear" the modes. PDEDIFF is a conditional diffusion model that learns the distribution P(Y∣X) over solution fields and incorporates physical constraints (the equation and boundary conditions) directly into the diffusion target via a residual penalty calculated by automatic differentiation on an arbitrary set of points (mesh-free). Unlike PINN, which minimizes the residual around E[Y∣X] and loses modality, PDEDIFF generates diverse, physically consistent solution samples.

### Strengths
1. Clear experiment details. I fully understand the details 
2. The multi-modality is very hard goal for achievent in PINNs. And that work gives us a nice vector
3. Good concept, nice toy examples

### Weaknesses
1. Not enough baselines, as SPINN, DeepONet, Separable FNO etc. 
2. Not enough different equations, like reaction-diffusion and heat equation.
3. I didn't see the experiments with high frequency equations

### Questions
Thanks for your paper! I have several questions.
1. Did you check experiments with normalization of coords before learning? It can boost your quality 
2. If to be honest, need comparison with DeepONet and SPINN with free parameters. 
3. Can you compare your approach on Klein Gordon 3d in SPINN?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Authors propose PDEDIFF, a mesh-free physics-informed conditional diffusion framework, to address the key limitation of existing PDE solvers: most methods (e.g., PINNs) collapse multi-modal PDE solutions to conditional means, while standard diffusion models lack explicit physical constraints. The core innovation of PDEDIFF lies in two aspects: first, it learns joint distributions over PDE solution fields and key parameters/eigenvalues, enabling sampling of diverse physically consistent modes; second, it uses automatic differentiation (instead of grid-dependent finite differences) to compute PDE residuals and boundary conditions, supporting sparse, irregular input samples without re-designing solvers for different PDEs. Experiments validate PDEDIFF on 1D/2D linear  and nonlinear PDEs, showing it outperforms PINNs in capturing multi-modal solutions via metrics like P-MSE and Wasserstein-1 distance. Overall, PDEDIFF aims to serve as a generative surrogate for uncertainty-aware multi-modal PDE modeling in low-to-moderate dimensions.

### Strengths
- The paper has a clear motivation to solve multi-modal PDE modeling, a long-standing gap of PINNs. Unlike PINNs that regress to conditional means and blur distinct physical solutions (e.g., quantum eigenstates), PDEDIFF explicitly learns the joint distribution of 
u(x) and λ, which allows it to sample diverse, physically valid modes—this fills a critical need in physics-informed learning for systems with multiple solutions.
- PDEDIFF’s mesh-free design with automatic differentiation enhances practical flexibility. By replacing finite differences with automatic differentiation for residual calculation, it supports sparse and irregular input samples.

### Weaknesses
- The background introduction and problem definition of $\lambda$ (eigenvalue/parameter) are insufficient. The paper treats $\lambda$ as a core component of the joint distribution but fails to clarify $\lambda$ s role across different PDE types. For example, as $\lambda$  is an eigenvalue for Schrödinger equations, and a wave number for Helmholtz equations, what's the position and meaning of $\lambda$ for PDEs without the eigenfunction-value form? In such case, will $\lambda$ still explicitly link to multi-modality, and is still reasonable to generate $\lambda$? More clarification and discussion are encouraged.


- The experiment could be enhanced. The current setting lacks comparisons with state-of-the-art diffusion-based PDE methods like CocoGen[1] and DiffusPDE[2]. Current baselines only include vanila PINNs, which is not convincing enough. Besides, as the paper claimed that the method can be expanded to non-zero boundary condition, it better to demostrate it in the experiment.


- As the work set the mesh-free encoder as a selling point, lack of discussion on existing literature [3,4,5] with similar design limit the novelty of proposed work.    

Ref:

[1]: Jacobsen, Christian, Yilin Zhuang, and Karthik Duraisamy. "Cocogen: Physically consistent and conditioned score-based generative models for forward and inverse problems." SIAM Journal on Scientific Computing 47.2 (2025): C399-C425.

[2]: Huang, Jiahe, et al. "DiffusionPDE: Generative PDE-solving under partial observation." Advances in Neural Information Processing Systems 37 (2024): 130291-130323.

[3]: Du, Pan, et al. "Conditional neural field latent diffusion model for generating spatiotemporal turbulence." Nature Communications 15.1 (2024): 10416.

[4]: Chen, Panqi, et al. "Generating Full-field Evolution of Physical Dynamics from Irregular Sparse Observations." arXiv preprint arXiv:2505.09284 (2025).

[5]: Wu, Haixu, et al. "Transolver: A fast transformer solver for pdes on general geometries." arXiv preprint arXiv:2402.02366 (2024).

### Questions
Do you have plans to validate PDEDIFF on higher-dimensional PDEs (e.g., 3D Schrödinger equations) and non-Dirichlet boundaries (e.g., Neumann conditions)? If so, please outline technical challenges (e.g., memory for high-dimensional ) and preliminary results

### Soundness
3

### Presentation
2

### Contribution
2
