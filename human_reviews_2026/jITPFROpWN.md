# Learning Koopman Representations with Controllability Guarantees

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 2

## Abstract
Learning nonlinear dynamical models from data is central to control. Two fundamental challenges exist: (1) how to learn accurate models from limited data, and (2) how to ensure the learned models are suitable for control design of the nominal system. We address both by enforcing a critical \emph{a priori} property of the nominal system during learning: \emph{controllability}. Controllability guarantees the existence of control policies that can drive the learned model from any initial state to any desired state. From a modeling perspective, it captures key structural features of the nominal system, thereby improving data efficiency. For downstream control, it enables the use of modern techniques such as model predictive control (MPC). Our approach is based on controllability-preserving Koopman representation learning. Rather than learning dynamics directly in the nominal state space, we learn in a latent space where the system admits a linear representation. We prove that controllability of the learned latent model implies controllability in the nominal state space. To enforce this property, we introduce a novel canonical parameterization of the latent dynamics matrices. We further incorporate Gramian-based regularization to shape the degree of controllability, yielding well-conditioned models for control. Implemented as an end-to-end Neural ODE framework, our method learns models that are both predictive and controllable from limited data. Experiments on nonlinear benchmarks demonstrate accurate long-horizon prediction, reliable MPC performance, and substantially improved data efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework for learning Koopman representations of nonlinear control systems that are guaranteed to be controllable. The authors derive conditions linking output-to-output controllability (OOC) and standard state-output controllability (SOC), and introduce a canonical parameterization of the latent linear dynamics matrices A_\theta, B_\theta that ensures controllability by construction. Additionally, a controllability Gramian regularizer is used to improve the conditioning (“well-controllable” property). The learned model can be used directly in model predictive control (MPC). Experiments on benchmark systems demonstrate improved sample efficiency and control performance compared to standard Deep Koopman (DKO) and MLP-based models.

### Strengths
S1. Embedding controllability directly into the Koopman parameterization is novel. Prior deep Koopman and neural ODE models treat controllability as an afterthought or soft constraint. The proposed canonical form and similarity transform provide an elegant, principled way to enforce this property.

S2. The proofs linking OOC↔SOC and the use of the finite-time output controllability Gramian are mathematically well-grounded and consistent with classical control theory. This makes the paper attractive to both ML and control communities.

S3. The ability to plug the learned model directly into linear MPC is a significant engineering advantage. The demonstration that the approach maintains performance under limited data conditions is also persuasive.

### Weaknesses
W1. There is limited novelty relative to the existing Koopman literature. While embedding controllability guarantees is new, the core architecture—learning a linear operator in latent space via encoder + linear dynamics + decoder—is largely inherited from standard deep Koopman frameworks. The contribution thus appears incremental in form but conceptual in framing. The authors should clarify in what sense their controllability parameterization extends beyond “structured Koopman learning” or prior work on “control-aware embeddings.”

W2. Controllability parameterization may constrain expressive power. The use of a canonical controllable form plus similarity transform P_\theta guarantees controllability but may significantly restrict the learned dynamics manifold. It is unclear whether the proposed structure can approximate arbitrary Koopman operators as the dimension of z increases, or whether expressiveness is traded off for structural rigidity. 

W3. There seems to be no stability guarantees. The method ensures controllability but says little about the stability of the learned system -  an equally critical property for control deployment. Without constraints on the eigenvalues of A_\theta, the learned linear system may be unstable, making MPC optimization difficult or ill-posed.

W4. All benchmarks are relatively small and well-known toy systems (Pendulum, CartPole, etc.). While these are standard, they do not convincingly demonstrate the method’s scalability or robustness to real-world noise and unmodelled dynamics. 

W5. Computational cost and training stability are unclear. The paper lacks analysis of training complexity, numerical conditioning, or the computational overhead introduced by the Gramian regularizer (which involves integrating matrix exponentials).

W6. The connection to Koopman theory is not fully justified. Although the paper uses the Koopman terminology, the approach essentially learns a linear latent-space model via a neural encoder. There is limited evidence that this latent space truly captures Koopman-invariant subspaces or observables.

### Questions
Q1. The paper constrains A_\theta, B_\theta through a controllable canonical form plus similarity transform P_\theta. Does this restriction reduce the expressive power of the learned Koopman operator? Please clarify whether your parameterization still provides a universal approximation property for arbitrary nonlinear dynamics in the lifted space, or discuss the trade-off between controllability enforcement and representational flexibility.

Q2. The current method enforces controllability but not stability. Have you observed instability in the learned latent dynamics (e.g., exploding eigenvalues of A_\theta)? It would be useful to regularize eigenvalues of A_\theta or provide evidence that the MPC formulation mitigates instability during rollout.
	
Q3. The Gramian term involves computing W_T^y. How is this computed efficiently during training, and how does it scale with the latent dimension N? It would be helpful to include complexity analysis or ablation to show that the regularizer does not dominate training time.
	
Q4. The framework is called “Koopman,” but it remains unclear whether the learned observables correspond to Koopman-invariant subspaces or simply linear latent embeddings. Can you empirically or theoretically justify that \varphi_\theta(x) approximates Koopman eigenfunctions or invariant coordinates? Spectral or mode analysis (e.g., eigenvalue comparison with known systems) would strengthen the Koopman interpretation.
	
Q5. Can your framework scale to higher-dimensional or partially observed systems (e.g., PDEs, power network, or soft-robot system)? Discuss computational limits and whether encoder-decoder architectures can handle such cases without losing controllability guarantees.
	
Q6. The paper does not compare against recent Koopman or geometric control approaches (e.g., KEEC) that emphasize structural invariance. What are the pros and cons of controllability-based representations and equivariance-based representations?

Q7. Parameters like the Gramian regularization weight and time horizon T may strongly affect results. How sensitive is performance to these settings? Suggest including an ablation or at least a qualitative discussion on how these hyperparameters influence both controllability conditioning and prediction accuracy.

Q8. The paper claims the learned model can be “directly integrated” with MPC. Was this tested in a closed-loop simulation with constraints? Suggest presenting control performance metrics (tracking error, energy use, robustness) under realistic MPC settings to demonstrate deployability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a dynamics learning method that enforces controllability as a structural prior in order to improve the quality and usability of the learned dynamics. In particular, this paper:
* Introduces a Koopman-based framework for learning linear surrogate models using Neural ODEs.
* Enforces controllability of the learned model by design, by proposing a particular parameterization of the to-be-learned Koopman operator. This parameterization is based on the observation that enforcing state-to-output controllability in the lifted space (easy) is equivalent to enforcing output-to-output controllability in the original space (hard).
* Provides experiments on four control settings (mountain car, pendulum, cartpole, gene regulatory network), showing improved quality of the learned dynamics and improved controllability of those dynamics compared to SOTA baselines.

### Strengths
* The idea of the paper is clear, simple, and well-grounded: Encode controllability as a structural prior enforced by design within neural ODE learning. The method and motivation are very clearly explained, and the clear notation facilitates understanding of how the Koopman operator is integrated into the neural ODE framework.
* The proposed parameterization is backed by theoretical proofs, proving controllability.
* The proposed loss regularization is a nice idea to ensure not just the binary notion of "controllable," but also improve the degree of controllability.
* The experimental results on mountain car, pendulum, and cartpole compare against SOTA baselines, and clearly show improved quality of dynamics learning in low-data regimes (comparable quality in higher-data regimes), as well as improve controllability of the learned dynamics when used within MPC.

### Weaknesses
* The experimental settings are all quite small, with the largest setting (GRN = gene regulatory network) including 6 dimensions and 3 control inputs, and with the best results on the smaller single-input settings (GRN is the only multi-input setting, and results are more more marginal). It would be nice to see additional and more convincing validations on larger systems.
* For large systems, the eigenvalue computation needed for the Gramian regularization term might be expensive to compute. In general, it would be nice to see relative training cost between the different methods reported.

### Questions
* What do results look like on larger experimental systems?

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
5

### Summary
The paper consider the problem of modeling and control of nonlinear systems via Koopman operator theory (KOT). Particularly, the paper ensures controllability by using a certain canonical parameterization and also are able to tune the degree of controllability to ensure better control performance. In addition they use Neural ODE to learn a KOT based model Simulation results present the efficacy of their approach

### Strengths
The main strengths are as follows:
1) The efficacy of their approach over other methods is clearly show.
2) The literature review in the introduction is extensive.
3) The use of Neural ODE is a smart choice over MLPs; one main reason being free from what discretization issues while converting the continuous to discrete systems.

### Weaknesses
I have worked on Koopman operators for a while now. I believe these are the following weakness:

1) The KOT for control literature is quite rich today. For instance works that consider controllability for KOT based models, use Neural ODE or similar techniques exist in literature. 
2) The motivation for this work is not clear to me.
3) The approximation error between the KOT and real system is not considered. This is particularly important because Koopman operator is an infinite dimensional linear operator and to make it of practical value, a finite approximation of the Koopman operator via EDMD is usually made.
4) Missing real world examples of more sophisticated nonlinear systems such as quadruped or humanoid is missing. There are already hundreds of paper that consider standard and simple nonlinear systems.

### Questions
I have the following questions:

1) Since the Koopman model is just a surrogate model, there is an approximation error between actual and approximated model. This is not considered in the paper. In addition, there exists works that consider this approximation rigorously (see [1]).
2) There are also papers that consider the controllability of the KOT surrogate model. It is not clear to me on how your work is significantly different from [2].
3) It is not always possible to make a nonlinear system controllable by ensuring controllability in the KOT model.
4) There has been lots of work that convert unknown dynamics into Koopman based models and use the surrogate model for control design purposes (see [3,4,5] ). It is not clear how your work is different from them. Yes, you mention that you also ensure controllability of the KOT model, but [2] does this as well
5) In simulation results, although you have compared your approach with some prior methods, the nonlinear systems considered are too simple. I would like examples of more complex robotic systems such as quadruped and humanoid where efficiency of KOT models is yet to shown. 
6) Showing that the proposed approach works on more complex nonlinear dynamics such as quadruped or humanoid needs to shown. In addition, they must be well motivated to show their efficacy compared to RL based control policies
7) Ensuring controllability as one of your main contributions would not add significant contribution to the paper given the fact that there exists some work that do it



[1] Mamakoukas, G., Di Cairano, S. and Vinod, A.P., 2022, June. Robust model predictive control with data-driven Koopman operators. In 2022 American Control Conference (ACC) (pp. 3885-3892). IEEE.

[2] Choi, Joonwon, Minhyun Cho, Hyunsang Park, Vishnu Vijay, and Inseok Hwang. "On The Controllability Preservation of Koopman Bilinear Surrogate Model." In 2024 IEEE 63rd Conference on Decision and Control (CDC), pp. 3457-3462. IEEE, 2024.

[3] Korda, M. and Mezić, I., 2018. Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control. Automatica, 93, pp.149-160.

[4] Zinage, V. and Bakolas, E., 2023. Neural koopman lyapunov control. Neurocomputing, 527, pp.174-183.

[5] Salzmann, T., Kaufmann, E., Arrizabalaga, J., Pavone, M., Scaramuzza, D. and Ryll, M., 2023. Real-time neural mpc: Deep learning model predictive control for quadrotors and agile robotic platforms. IEEE Robotics and Automation Letters, 8(4), pp.2397-2404.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a Koopman-based representation learning framework, implemented as an end-to-end Neural ODE, that learns nonlinear dynamical models from limited data while ensuring the learned models remain suitable for control.  It enforces controllability by construction via a new canonical parameterization of the latent linear dynamics (A, B), and shows that controllability of the learned latent model implies controllability of the original (nominal) system.  The method also shapes the degree of controllability by adding a finite-horizon output-Gramian regularizer that enlarges the smallest eigenvalue and controls the condition number to promote well-conditioned models for control. For downstream control, the learned linear surrogate enables a convex quadratic-program MPC in the lifted space with receding-horizon execution.  Empirically, on standard nonlinear benchmarks (pendulum, mountain car, cartpole), the approach achieves more accurate long-horizon prediction and better MPC performance with improved data efficiency compared to Deep Koopman Operator and MLP baselines.

### Strengths
1. Learning the Koopman operator is an interesting and important classical control system problem.

### Weaknesses
1.	This paper has marginal algorithmic or theoretical contributions. See the details below.
2.	The main contribution is a method to learn the Koopman operator. However, the proposed methods are quite standard, typical in any Koopman operator learning approach. While the paper prominently states the connection with Neural ODE, it is not really clear why that is relevant or interesting. Lifting the state x to a high-dimensional variable z and modeling the non-linear system as a linear system at this lifted space is indeed the standard approach of Koopman theory. So, the reason for presenting neural ODE as new approach is not clear. 
3.	The theoretical results presented are standard results from linear systems theory. In particular, Theorem 1 about controllability and Theorem 2 about reparameterization to a suitable form are standard results. So, it is not clear why these are presented as novel results, and how they contribute to the Koopman operator theory. 
4.	Once the Koopman operator is learned, LQR or MPC are the standard ways to design a control policy. There is no novelty in that part.
5.	The experiments are done on three very simple tasks: mountain car, pendulum, cartpole. It is not clear if the proposed methods can scale to even a slightly more difficult settings, say, mujoco environments.

### Questions
1. Aren't Theorems 1 and 2 about any linear systems? Any specific connection to Koopman operators?
2. Will the proposed approach work even in slightly highly dimensional non-linear systems, such as simple MuJoCo environments?

### Soundness
2

### Presentation
2

### Contribution
1
