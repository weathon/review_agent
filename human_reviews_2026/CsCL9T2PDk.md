# Newton-PINet: A fast physics-informed neural network with Newton linearization for meta-learning nonlinear PDEs

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Scientific machine learning has opened new avenues for solving parameterized partial differential equations (PDEs), enabling models to learn a family of PDEs and generalize to unseen instances. In this context, data-driven operator learning methods typically require large training data, while physics-informed neural networks (PINNs) trained with PDE-based loss functions suffer from challenging optimization landscapes and limited generalization, especially for nonlinear PDEs. To resolve these issues, we develop Newton-PINet, a physics-informed network enhanced by Newton linearization, offering an effective meta-learning framework for nonlinear PDEs. It (i) introduces a physics-informed multilayer network with skip connections from early hidden layers to the output, where the final-layer weights are computed using least-squares method; (ii) adopts a two-stage learning strategy that first leverages gradient-based training to learn robust representations from the available training tasks, and then performs gradient-free fine-tuning on the output layer for fast task-specific generalization; and (iii) incorporates a Newton linearization method to speed up the least-squares iteration for nonlinear PDE problems. Newton-PINet achieves relative errors three orders of magnitude lower than recent neural solver baselines on a challenging nonlinear reaction-diffusion benchmark, even while using 16$\times$ fewer training tasks and an order of magnitude less training time (under 2 minutes against the several hours these baselines required). This work advances the meta-learning of PINNs toward data-efficient, fast, and generalizable physics solvers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The Newton-PINet method is a new algorithm that aim at solving parametric PDEs using meta-learning. The method is composed of several components : a specific design of layers, a newton linearization and a regularization in the objective. The paper demonstrate very efficient results in terms of performance and computationnal time both at training and inference.

### Strengths
- The paper demonstrate very strong results in term of MSE and computational time, both at training and inference. 
- The problem tackled is a well-known issue of PINNs that is currently studied.

### Weaknesses
-	I feel like the paper is a bit rushed : some references are missing/not well writen, legends of figures are not precise, some numeric application are empty etc... 
-	There is a rich content, but it is difficult to follow and understand the method. Maybe pseudo code would help ? 
-	References are old : in the data driven and operator learning section, the most recent reference is from 2021. A very rich literature have emerged on these subject in the 5 past years.

### Questions
-	Could you define specifically what is a task ? I understood it as solving a single instance of a PDE, but I think this should be formally defined at the begining for clarity (maybe in section 2 ?)
-	line 157 Could you justify the use of skip connexions ? (done in appendices, but while reading though the main part, I wondered about this point)
-	Could you calrify notations ? i=points, s=data, k=iterations (eg line 162)? 
-	What means HOT line 196?
-	lines 186-197: Are the loss of the linearization quantified ? When linearizing an expression, one can expect some error, are they quantified? In what extend do they influence the results? 
-	How are created the matrices A ? How big is it ? Isn’t inversion costly ? How hard are they to compute.derive? What happens when the matrix are not tractable? 
-	Do you have any insight about why MSE+LES in Newton PINet works rather well ? The LSE loss term does not incurs a big difference ?
-	Why some cells in tab 2 are missing ? 
-	Could you ablate the time block strategy ? My point here is to isolate in what extend do the time block strategy helps in the performance. 
- Tab 3, I could not fid the results of table 3 in the work cited (Wei et al. 2025b). Moreover, I couldn't find any reference to PPINNs in this work, which could indicate a mismatch in the references. 
-	I am not an expert in Newton method so I could not check the entire theory behind the convergence analysis. 
-	Do you have any insight about why is there such a difference wrt to other baselines ? 3 order of magnitude is a lot of improvement with a very short training time and data requirement (eg on tab 3). Moreover, what justifies the very short training and inference time of the proposed method? Meta-learning methods are usually long to train, due to the inner/outer loop. Additionally, inference can be costly, because of the inner steps required to adapt to new instances. 
- In Fig2 a) and b), why the errors aren't decreasing with respect to the number of iterations? 
- To the best of my knowledge, optimizing a residual loss on the PDE residual, often complicates training. The observe this phenomenon in Fig2)e with PINet. What explains that this effect is much smaller when using Newton-PINet?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a meta learning framework for rapid adaptation to new PDEs, introducing a gradient based trained transferable hidden representation and converting learning new PDE tasks into a fast closed-form Jacobian-free Newton linearization-Tikhonov least-squares solve on a skip-connected output layer, achieving gradient-free fine-tuning.

### Strengths
1. The use of closed-form Tikhonov least-squares speed up adaptation to new PDE solving task is novel to me. Usually, I see people use gradient based fine tuning for learning new PDEs, such as transfer learning. 
2. I am not familiar with meta learning, but the method seems novel and useful and the experimental setup is solid and achieves substantially better results than the baselines.

### Weaknesses
Due to the complex loss landscape, PINNs are known to fail easily when the PDE coefficient is large [1]. For example, when the coefficient (e.g., viscosity) in Burgers’ equation is large, will the proposed method also achieve good results?

[1] Krishnapriyan, Aditi, et al. "Characterizing possible failure modes in physics-informed neural networks." Advances in neural information processing systems 34 (2021): 26548-26560.

### Questions
None.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Newton-PINet, a physics informed meta learning framework for efficiently solving families of nonlinear PDEs. The model combines skip connected PINN architectures with a Tikhonov regularized closed form output layer and a Jacobian-free Newton linearization for nonlinear terms. This design enables gradient free, near instant adaptation to unseen PDEs.

### Strengths
1) Clear motivation and novel combination of PINNs, meta-learning, and Newton linearization.

2) Gradient-free closed-form adaptation drastically reduces computation time.

3) Strong empirical results across diverse PDEs, showing orders-of-magnitude gains in speed and accuracy.

4) Demonstrates high data efficiency and generalization with few training tasks.

### Weaknesses
1) The claimed “Jacobian-free Newton” method lacks formal derivation or convergence proof.

2) Construction of the linear system Aw=b is under-specified, hindering reproducibility.

3) No analysis of the stability or sensitivity of the Tikhonov regularization parameter.

4) Meta-gradient propagation through the closed-form solution is not explained.

### Questions
1) Question in detail of the Newton Linearization
The description of the Jacobian free Newton linearization raises questions about its mathematical rigor.
The update rule (in Fig 1 / near line 208)
(uux) k+1 ≈ u ku k+1 x + u k+1u k x − u ku k
is presented as a Newton type linearization, yet it differs from the classical Newton Raphson formulation that explicitly involves the Jacobian vector product. The paper claims second order convergence but does not specify the conditions under which this convergence holds.
A more formal derivation or convergence proof would be necessary to justify calling this method “Newton like” in the strict numerical analysis sense.

2) Question in construction of linear system
The paper repeatedly states that the matrices A and b are assembled from PDE residuals, initial conditions, and boundary conditions, but it does not provide an explicit mathematical formulation of these components. In nonlinear PDEs such as the Burgers equation, the residual terms (e.g. burger’s equation) make A depend nonlinearly on both u and w.

It remains unclear how these nonlinear dependencies are linearized or evaluated when forming A^(k) in each iteration, and whether differential operators are computed analytically or via automatic differentiation. Clarifying this construction is essential for reproducibility and for understanding how the closed form Tikhonov solution applies to nonlinear PDEs. Did I miss some parts?

### Soundness
2

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
2

### Summary
Newton-PINet is a meta-learning framework for PINNs that pretrains the hidden representation once and adapts to new PDE instances by solving only the output layer with Tikhonov-regularized least squares. To handle nonlinear dynamics, it replaces Picard with a Jacobian-free Newton linearization that achieves quadratic convergence and improves stability. For time-dependent problems it uses temporal domain decomposition, training on short blocks and composing them for long-horizon rollout. Across diverse nonlinear PDE benchmarks, including a 2D Helmholtz case, the method shows strong data efficiency, fast adaptation, and robustness where Picard-based approaches struggle.

### Strengths
1. Clear derivation of the Jacobian-free Newton linearization and a contrast with Picard’s first-order vs Newton’s second-order convergence (including a Burgers example and an Appendix proof). 

2. Only the output layer is updated at test time via closed-form Tikhonov solves; this, together with time-block training, makes long-horizon inference practical.

3. PINN with the output weights solved by Tikhonov regularization improves nonlinear representation while keeping adaptation inexpensive.

### Weaknesses
1. Most studies are 1D; there is one 2D problem (Helmholtz). It remains unclear how the Tikhonov solves and Newton iterations scale in 2D/3D, multiphysics, or turbulence-like regimes. 

2. Because adaptation reduces to solving regularized least-squares systems, guidance on low-rank structure/iterative solvers for very large collocation sets would strengthen the practical scaling story.

### Questions
1. For high-resolution 2D/3D cases, how do you keep the Tikhonov update solves tractable from a linear-algebra standpoint?
 
2. Can you quantify the Newton step’s convergence radius and sensitivity to initialization for strongly nonlinear/chaotic settings (e.g., low-viscosity K-S)? Any empirical ablations?

3. Have you tested Neumann/Robin/mixed or discontinuous BCs beyond periodic/Dirichlet? 

4. Did you try overlapping blocks, residual re-seeding, or simple filters (e.g., Kalman-style smoothing) to mitigate drift at block boundaries during very long rollouts?

### Soundness
2

### Presentation
3

### Contribution
3
