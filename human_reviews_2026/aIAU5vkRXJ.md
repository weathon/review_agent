# SAD-Flower: Flow Matching for Safe, Admissible, and Dynamically Consistent Planning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Flow matching (FM) has shown promising results in data-driven planning. However, it inherently lacks formal guarantees for ensuring state and action constraints, whose satisfaction is a fundamental and crucial requirement for the safety and admissibility of planned trajectories on various systems. Moreover, existing FM planners do not ensure the dynamical consistency, which potentially renders trajectories inexecutable. We address these shortcomings by proposing SAD-Flower, a novel framework for generating Safe, Admissible, and Dynamically consistent trajectories. Our approach relies on an augmentation of the flow with a virtual control input. Thereby, principled guidance can be derived using techniques from nonlinear control theory, providing formal guarantees for state constraints, action constraints, and dynamic consistency. Crucially, SAD-Flower operates without retraining, enabling test-time satisfaction of unseen constraints. Through extensive experiments across several tasks, we demonstrate that SAD-Flower outperforms various generative-model-based baselines in ensuring constraint satisfaction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author introduces SAD-Flower, a framework for generating safe, admissible, and dynamically consistent trajectories using flow matching (FM). The core problem addressed is that standard generative planners, including those based on FM, lack formal guarantees for satisfying state, action, and dynamics constraints. SAD-Flower reframe the trajectory generation process as a controllable dynamical system by augmenting the learned vector field of the FM with a virtual control input. The authors provide theoretical guarantees for constraint satisfaction and demonstrate empirically across several benchmarks that SAD-Flower significantly outperforms baselines in constraint adherence while maintaining competitive task performance.

### Strengths
1. SAD-Flower is the first, to my knowledge, to provide a unified framework for safety, admissibility, and dynamic consistency with formal guarantees within a flow matching context.
2. The paper is well-structured and easy to follow.
3. The proposed method is theoretically well-grounded.

### Weaknesses
1. A potential weakness is that the experiments are conducted in environments with relatively low-dimensional state and action spaces. Since the CLF formulation relies on an accurate dynamics model, it is unclear how well the proposed approach would generalize to more challenging, high-dimensional environments, such as the OGBench Humanoid benchmark, where learning an accurate dynamics model is notoriously difficult.
2. There is limited analysis of how computational cost scales with problem complexity like horizon length, the number of constraint and state/action dimensionality. 
3. Many recent works have explored diffusion sampling that satisfies constraints and ensures dynamic consistency. However, the current paper overlooks these relevant works. I encourage the authors to include a discussion of the following references like:

[1] Refining Diffusion Planner for Reliable Behavior Synthesis by Automatic Detection of Infeasible Plans, 2023

[2] Resisting stochastic risks in diffusion planners with the trajectory aggregation tree, 2024

[3] Inference-Time Policy Steering through Human Interactions, 2025

[4] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning, 2025

[5] Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models, 2025

### Questions
See weaknesses

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
This paper proposes SAD-Flower, a novel control-augmented flow matching framework to address the limitations of existing flow matching (FM) based planners in providing formal guarantees for safety, admissibility, and dynamic consistency of generated trajectories. By introducing a virtual control input and leveraging nonlinear control theory, specifically Control Barrier Functions (CBFs) and Control Lyapunov Functions (CLFs), SAD-Flower ensures constraint satisfaction and dynamic consistency without requiring retraining for unseen constraints. The framework uses a quadratic program (QP) to determine the minimum-norm control input. Extensive experiments on navigation, locomotion, and manipulation tasks demonstrate SAD-Flower's superior performance in adhering to constraints compared to various generative-model-based baselines.

### Strengths
1. The paper provides a rigorous theoretical foundation by integrating nonlinear control theory (CBFs and CLFs) to offer formal guarantees for safety, admissibility, and dynamic consistency. This is a significant improvement over existing FM-based methods that often lack such assurances.
2.  SAD-Flower's ability to enforce new or tighter constraints without requiring model retraining is a major practical advantage, enhancing its real-world applicability and robustness.
3. The comprehensive experimental evaluation across diverse and challenging environments (Maze2d, Hopper, Walker2d, Kuka Block-Stacking) consistently demonstrates SAD-Flower's superior performance in satisfying constraints and achieving higher rewards compared to several baselines.

### Weaknesses
1. The introduction of a virtual control input and the need to solve a QP at each step for the minimum-norm control input may introduce significant computational overhead, potentially limiting real-time application in very high-dimensional or time-critical systems. While the paper mentions "lightweight QP-based formulation," specific runtime comparisons or analysis of the QP's complexity for varying state/action spaces could strengthen this.
2. Defining appropriate CBFs and CLFs, especially for complex, high-dimensional real-world tasks, can be challenging and might require expert knowledge, potentially limiting the ease of adoption for new problem setups.
3. While experiments cover various tasks, the scalability of SAD-Flower to environments with extremely complex, non-smooth, or highly dynamic constraints, or very long horizons, could be further explored.

### Questions
1. Could the authors provide a more detailed analysis or empirical comparison of the computational cost (e.g., inference time per step) of SAD-Flower compared to baselines, specifically highlighting the overhead introduced by the QP solver for different problem complexities?
2. Are there any guidelines or automated methods proposed for the systematic design of CBFs and CLFs for new, complex tasks, or does it heavily rely on manual finetuning?
3. How sensitive is SAD-Flower's performance to the choice of hyperparameters, especially those related to the CBF/CLF formulation and the QP solver settings?
4. For long-horizon tasks or those with sparse rewards, trajectory optimization can be challenging. How does the "flow matching" aspect of SAD-Flower (which often relies on expert demonstrations or a pre-trained policy) perform in such scenarios? Does the control augmentation inherently improve performance in these difficult settings, or are there specific limitations?
5. How does SAD-Flower conceptually and practically compare to other hybrid control architectures that combine learning-based methods with formal verification or safety filters (e.g., using model predictive control (MPC) with safety guarantees)? What are the distinct advantages and disadvantages of SAD-Flower in relation to these alternative approaches?
6. Can the framework be extended to multi-agent planning scenarios where cooperative or adversarial interactions are present, and how would the constraint satisfaction and dynamic consistency guarantees be maintained in such complex settings?
7. Have there been any preliminary tests or considerations for deploying SAD-Flower on real-world robotic platforms, and what challenges might arise regarding sensor noise, actuation limits, and real-time execution?

### Soundness
2

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
3

### Summary
This paper proposes SAD-Flower, a method to enforce constraints on trajectories generated by flow matching (FM) models. The authors aim to guarantee safety (state constraints), admissibility (action constraints), and dynamic consistency. The core idea is to augment the standard FM ODE with a virtual control input $u_t$, which is solved for at test time using a Quadratic Program (QP). This QP is formulated to satisfy conditions derived from Control Barrier Functions (CBFs) for safety and admissibility, and a Control Lyapunov Function (CLF) for dynamic consistency. The method is shown to enforce constraints more effectively than several baselines on a few benchmark tasks.

### Strengths
1. The paper addresses the important and practical problem of constraint enforcement in generative planning.

2. The proposed method outlines three key aspects in trajectory planning, namely safety, admissibility, and dynamic consistency, and combines Control Barrier Constraints, Control Lyapunov Constraints, and Constrained Minimum-Norm Optimal Control to achieve constraint-aware generative planning.


3. The empirical results show that the proposed method does achieve better constraint satisfaction than the chosen baselines, especially under stricter test-time constraints.

### Weaknesses
1. The novelty of the paper seems marginal. The idea of using control-theoretic guidance during the sampling process of a generative model is already established. Specifically, SafeDiffuser [1], cited by the authors, already introduced the core concept of using CBFs to project and guide the sampling steps of a diffusion model to enforce state constraints, also CoBL-Diffusion[2] have introduced control lyapunov functions to diffusion planning. Therefore, the paper appears to be a straightforward combination of Flow Matching + CBF/CLF control.

2. The authors assumes the feasibility of QP. Theorem 5.1 is entirely conditional on the QP (Eq. 5) being feasible at every integration step. In robotics and control, QP infeasibility (e.g., due to conflicting constraints or a poor reference $v_t^\theta$) is a well-known and common failure mode. The paper offers no robust fallback strategy (e.g., constraint relaxation, slack variables) and fails to analyze what happens when this core assumption is violated.

3. The proposed method is heavily dependent on an accurate dynamics model, The entire guarantee for dynamic consistency relies on the CLF $V(\tau_t)$, which requires explicit knowledge of the system dynamics $f(s,a)$ and its derivatives. The authors state this model "can be learned", but this undermines the claim of a "guarantee." The method hence might not guarantee true dynamic consistency; it only guarantees consistency with respect to a learned, approximate model $\hat{f}$. If the learned model is suboptimal, the additional guidance might even decrease the quality of the generated trajectory.

4. Given the method's critical dependence on a learned dynamics model $\hat{f}$, the paper should provide some form of sensitivity analysis or ablation on the model accuracy and how it affects the planning performance.

### Questions
1. Could the authors please clearly differentiate their contribution from a straightforward application of standard CLF-CBF-QP control to a flow-matching vector field, especially in light of prior work like SafeDiffuser that already established CBF-guided generative planning?


2. CoBL-Diffusion[2] seems like a closely related work to the proposed method, which also leverage control barrier and lyapunov functions. Authors do mention this work in the related work and state that their "formal guarantees are missing, and action constraints are not addressed". However, while with formal guarantees, from the experiment results, SAD-Flower will still violate the constraints, especially in higher dimension agent state space like the Walker2D tasks. In addition, this method seems like an important baseline and the authors do not compare with it.


3. For the Dynamics Model, since the dynamic consistency "guarantee" is only with respect to a learned model $\hat{f}$, how can this be considered a formal guarantee of true physical consistency? What is the empirical performance (consistency violation) when evaluated against a ground-true dynamics $f$ (if possible), and how does this degrade with the accuracy of $\hat{f}$? Is it easy to obtain a good $f$ in a more general setting?


4. By definition in the paper (e.g., in line 238), obtaining the left-hand side of Equation (CBF-s), (CBF-a), (CLF) requires differentiability. Can the proposed method work when these equations are not differentiable?


5. What were the exact planning horizons ($H$) used for all experiments in Table 1? How does the QP solve time actually scale with $H$ and the dimension of the agent's state space?




[1] SafeDiffuser: Safe Planning with Diffusion Probabilistic Models   
[2] CoBL-Diffusion: Diffusion-Based Conditional Robot Planning in Dynamic Environments Using Control Barrier and Lyapunov Functions

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper uses Flow Matching to learn a model for planning in maze and robotics domains. They propose to modify regular flow matching to include a control term. This control term is designed so that once the FM process is run for sufficient iterations, the resultant should not violate certain safety, feasibility and dynamics constraints. The algorithm is evaluated on maze and robotics domains, where a model is trained from expert data as well as to stay within constraints. Comparisons with other generative models shows almost perfect adherence to constraints while maintaining high performance.

### Strengths
1. The results demonstrate that including the control terms proposed in the paper leads to solutions that better adhere to constraints across three different tasks. 

2. The paper combines ideas from generative modeling and control theory in an interesting way to demonstrate practical results. 

Overall the paper demonstrates an interesting way to incorporate reasoning about hard constraints into generative modeling for planning.

### Weaknesses
The paper frames the modeling of trajectories together with various contraints as planning. One of the advantages of classical planning approaches to the sorts of tasks presented in the paper (RRT style algorithms for low level problems, PDDL for bi-level planning) is their generalisability to novel settings without the need for re-training. Will using Flow Matching based methods as "planners" provide this flexibility? 

Overall, it is not clear to me whether such an approach benefits downstream planning tasks. Since experts (such as navigation algorithms) are being used to generate training data for the model, how does the final performance of FM compare to running the expert directly on the task?  

The evaluation of this approach is also limited to mazes and a single robotics application. The approach may have potential towards being applied to complex grasping, where the additional constraint control term helps generate valid grasps in complex environments. Such additional settings are not evaluated.

### Questions
1. Is the transition function known for the domains being evaluated? Specifically for the Kuka robotics task? Is this method applicable to domains where such a function would be computationally difficult to evaluate (e.g. stochastic or complex environments with passive dynamics)?

2. Does the block stacking task consider cluttered environments where the system has to reason about difficult grasps / avoiding collision with blocks?

3. How much data is used to learn the model?

4. Does the training data include constraints violations? If not, then why does the learned model exhibit constraint violations?

5. How does this compare to a baseline where you sample from a model and reject violating samples?

### Soundness
3

### Presentation
3

### Contribution
2
