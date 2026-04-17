# PINNverse: Accurate parameter estimation in differential equations from noisy data with constrained physics-informed neural networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 4

## Abstract
Estimating unknown parameters in differential equations from noisy, sparse data is a common inverse problem in science and engineering. While Physics-Informed Neural Networks (PINNs) have shown promise, their standard training paradigm, which relies on a weighted-sum loss, often leads to overfitting and fails to enforce physical laws in the presence of noise. This failure stems from the inability of gradient-based methods to find balanced solutions on the complex, often non-convex, Pareto fronts that arise in such multi-objective settings.
We introduce PINNverse, a new training paradigm that overcomes these limitations by reformulating the learning process as a constrained optimization problem. Instead of balancing competing objectives with ad-hoc weights, PINNverse minimizes the data-fitting error subject to the explicit constraint that the differential equations and boundary conditions are satisfied. To solve this, we employ the Modified Differential Method of Multipliers (MDMM). By simultaneously updating network weights and Lagrange multipliers (via gradient ascent) in a single optimization loop, this method avoids the expensive nested loops required by conventional augmented Lagrangian techniques and seamlessly integrates with standard optimizers like Adam. This enables convergence to any point on the Pareto front---including concave regions inaccessible to standard PINNs---while adding negligible computational overhead.
Experiments on four challenging ODE and PDE benchmarks demonstrate that PINNverse achieves robust and accurate parameter estimation even with significant data noise and poor initial guesses, successfully preventing overfitting and ensuring strict adherence to the governing physics. By solving the forward and inverse problems concurrently, PINNverse enables efficient parameter inference in systems where repeated forward evaluations with classical numerical solvers would be computationally prohibitive.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles parameter estimation in differential equations from noisy observations. Instead of the standard weighted-sum objective used in PINNs, it formulates learning as constrained optimization: minimize data loss subject to physics (PDE/IC/BC) equality constraints. The method uses a Modified Differential Method of Multipliers (MDMM) and updates network weights, physical parameters, and Lagrange multipliers in a single loop. Across several benchmarks, it reports higher robustness and accuracy than vanilla PINNs and Nelder–Mead under noise and initial condition mismatch.

### Strengths
- (Novelty & Technical contribution) PINNverse successfully avoids nested optimization loops of traditional augmented Lagrangian methods, reducing computational costs. This delineates PINNverse from other models that use the augmented Lagrange method for PINNs.
    
- (Empirical contribution) PINNverse robustly infers parameters even with substantial data noise and poor initial guesses, scenarios where even robust classical numerical optimizers fail.
    
- (Broader impact) PINNverse is compatible with SOTA optimizers such as Adam.

### Weaknesses
- (Relevant citation) An important reference appears missing, which uses the augmented Lagrange method: “Enhanced Physics-Informed Neural Networks with Augmented Lagrangian Relaxation Method (AL-PINNs)“ [https://arxiv.org/abs/2205.01059](https://arxiv.org/abs/2205.01059). Could the authors discuss this work?
    
- (Presentation) My main concern is that the main text is not self-contained. The description of the proposed method, *PINNverse* (lines 162–196), relies heavily on Appendix A. Please refer also to the questions below.
    
- (Theoretical contribution) The paper would benefit from convergence analysis.
    
- (Empirical contribution) Hyperparameter ablations are recommended. Sweep penalty magnitude, dual LR, $\lambda$ initialization/schedule, and report sensitivity.
    
- (Reproducibility & Fair comparison) I could not find hyperparameter configurations for all models. Are all baselines properly tuned?
    
- (Reproducibility) Error bars are missing. Statistical analysis is important because PINNs tends to be sensitive to seeds and hyperparameters.
    
- (Reproducibility) It would be recommended to add README to the submitted code for interested readers.

### Minor Comment

- (Line 220) An upside-down exclamation mark should be corrected.

### Review Summary

Although the novelty and technical contributions are sound, the presentation requires improvement. In addition, I am concerned about the poor reproducibility and experimental settings of the results. Therefore, I am inclined to recommend rejection.

### Questions
- Why and how does PINNverse avoids nested loops? Could the authors elaborate on the following statement in the main text?:
    
  > (Lines 185-187) MDMM elegantly avoids this by updating the neural network parameters, differential equation parameters, and Lagrange multipliers all at once within a single backward pass.

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
This paper proposes PINNverse, a novel training method that reframes the learning process of Physics-Informed Neural Networks (PINNs) as a constrained optimization problem, thereby further addressing the issue of loss weight balancing among multiple objectives in PINN training. The core of PINNverse is the Modified Differential Method of Multipliers (MDMM), which updates network weights and Lagrange multipliers (via gradient ascent) within a single optimization loop. Experiments on four differential equation benchmarks demonstrate that PINNverse achieves robust and accurate parameter estimation, even in the presence of significant data noise and poor initial guesses. It successfully prevents overfitting and ensures strict adherence to the governing physics.

### Strengths
1. The paper is well-written and easy to follow.
2. Experiments on multiple benchmarks show that PINNverse achieves better performance compared to vanilla PINNs.

### Weaknesses
1. Line 220: typo
2. Due to the highly non-convex nature of PINN-based differential equation solving, MDMM cannot guarantee strict satisfaction of hard constraints and PINNverse may still converge to local optima. Therefore, the authors' claim of "ensuring strict adherence to the governing physics" is unsubstantiated. The loss curves in Figures 2, 3, and 4 clearly show that the DE/IC/BC constraints are not strictly met during optimization. Experiments only confirm that PINNverse converges to superior solutions compared to the vanilla PINN.
3. Based on point 2, PINNverse more closely resembles a dynamic loss weighting adjustment method. Its weight update rule, based on MDMM, falls into the same category as existing adaptive weighting methods based on gradients or losses, differing primarily in the update formula and theoretical foundation. Intuitively, PINNverse not only introduces new loss terms ($L_i^2$ in Line 179) but also continuously increases the loss weights for PDE/IC/BC constraints during training (via the dual update in MDMM). These operations inevitably bias PINNverse towards reducing these constraint losses, making the model prioritize satisfying physical constraints. The reviewer recommends that it would be better to compare PINNverse with other SOTA dynamic loss weighting methods to clarify the advantages of introducing MDMM.
4. Subplots (c) in Figures 2, 3, 4, and 5 show a similar trend: PINN and PINNverse have comparable loss reduction in the early training stages. In the later stages, the data loss of the PINN continues to decrease, while the DE/IC/BC losses of PINNverse continue to decrease. The reviewer suspects this is related to a significant increase in the $\lambda_i$ during the training of PINNverse. Detailed epoch-$\lambda_i$ curves would help illustrate the influence of MDMM more clearly.
5. Based on point 4, the reviewer suggests to add an ablation study: train a vanilla PINN using the final weights for the DE/IC/BC losses obtained from PINNverse training. This ablation study would help clarify whether the performance improvement of PINNverse over the vanilla PINN is due to the larger physics term weights or the dynamic weight updates based on MDMM.

### Questions
Please refer to the weaknesses. If the authors can adequately address points 3 and 5, and convincingly demonstrate that employing MDMM for dynamic loss weighting is a superior approach, the reviewer would consider raising the score.

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
5

### Summary
The paper introduces PINNverse, a reformulation of the standard Physics-Informed Neural Network (PINN) training process as a constrained optimization problem, solved using the Modified Differential Method of Multipliers (MDMM).  The paper directly targets a persistent weakness in the PINN framework that of unreliable parameter inference from noisy or sparse data, a very relevant problem.
Most PINN research focuses on soft-constraint or weighted-sum losses. Here physics residuals are used as hard equality constraints, allowing PINNverse to avoid the need for ad-hoc weight tuning and reach non-convex regions of the Pareto front, which standard PINNs cannot access.
While related to earlier hPINN/PECANN/ADMM-PINN work, the novelty lies in integrating MDMM with gradient-based deep learning optimizers (e.g., Adam) without nested loops, enabling efficient simultaneous updates of both primal (network/parameter) and dual (Lagrange multiplier) variables. The paper is largely complete but not exhaustive.  The methodology is sound and well-presented, but broader benchmarking and ablation studies would improve completeness.

### Strengths
The paper is well motivated, is clear to read and has a good set of relevant refernces.
Novelty: While related to earlier works, the novelty of this paper lies in integrating MDMM with gradient-based deep learning optimizers without nested loops, enabling efficient simultaneous updates of both primal (network/parameter) and dual (Lagrange multiplier) variables.
Relevance: Inverse problems for differential equations are ubiquitous in science and engineering. and very relevant especially for high dimensional complex systems such as battery chemistry models.
Results: Across 4 benchmarks (Kinetic Reaction, FitzHugh–Nagumo, Fisher–KPP, Burgers) are presented, however, results on real experimental datasets or high-dimensional PDEs woudl be more appreciated.

### Weaknesses
Missing the following:
1.Detailed ablation on penalty parameters, learning-rate effects, or sensitivity to multiplier initialization
2.Comparative results with adaptive weighting or multi-objective PINN variants (e.g., NSGA-PINN, hPINN), which would strengthen empirical completeness
3.Higher-dimensional PDEs or irregular geometries
discussions on the following would help
The convergence proof for MDMM in the stochastic, high-dimensional setting of neural networks is assumed but not formally analyzed.
No quantitative uncertainty or confidence interval analysis is presented for parameter estimates.
No results on real experimental datasets or high-dimensional PDEs.
Baseline PINNs use fixed unit weights.Stronger baselines (adaptive or Pareto-PINN) could better test robustness.
Scalability is not explored, only 1D/low-dimensional PDEs are tested; unclear behavior for 3D or coupled multiphysics systems.

### Questions
the approach depends heavily on simultaneous primal–dual updates. Given that deep-learning optimizers already operate on stochastic mini-batches, what guarantees do you have that the resulting stochastic saddle-point dynamics remain convergent or at least bounded?

PINNverse claims to access concave regions of the Pareto front. Could you visualize or quantify this property—perhaps by mapping Pareto trajectories or comparing multi-objective fronts empirically?  Figure(1) shown are schematics .

The paper assumes perfect PDE form and boundary conditions. How will the algorithm behave if there are modeling gaps or partially known boundaries—does the constraint formulation make the solution brittle?

In the Fisher–KPP and Burgers experiments, PINNverse’s advantage is pronounced under noise. Could this be partly due to the additional regularization implicit in the quadratic penalty term rather than the MDMM ?

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
4

### Summary
This paper proposes converting PINN (a single-loop soft penalty method) into a hard-constrained optimization problem—treating data loss as the objective, with IC, BC, and PDE residuals as equality constraints, and bounded constraints on the PDE coefficient—using the Modified Differential Method of Multipliers (MDMM) to solve it, because PINN cannot reach concave parts of the Pareto front.

### Strengths
This paper clearly explains the problem PINN is facing and how converting to a hard constrained problem can help. Meanwhile, this paper clearly explains MDMM and its advantage.

### Weaknesses
1: This paper does not provide baseline methods for solving inverse problems. A well-known hard constrained paper is [1].

2: PINN is well-known for its inability to solve the forward problem when the PDE coefficient is large [2]. This paper does not demonstrate performance in predicting the inverse problem for high PDE coefficients.

3: This paper restricts PDEs to those with a single scalar coefficient. However, the PDE coefficient can also be a function. This paper needs additional experiments to demonstrate the proposed method's performance in such cases.

[1] Lu, Lu, et al. "Physics-informed neural networks with hard constraints for inverse design." SIAM Journal on Scientific Computing 43.6 (2021): B1105-B1132.

[2] Krishnapriyan, Aditi, et al. "Characterizing possible failure modes in physics-informed neural networks." Advances in neural information processing systems 34 (2021): 26548-26560.

### Questions
The method imposes a box (bound) constraint on the PDE coefficient, requiring it to lie between specified lower and upper limits. If the goal is to infer the coefficient, how are these bounds determined a priori? I’m not convinced that this constraint is justified.

### Soundness
2

### Presentation
3

### Contribution
2
