# Epigraph-Based Multigrid PINNs for Two-Player General-Sum Differential Games

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
In continuous-time, noncooperative, safety-critical settings, players with distinct goals and shared state constraints are naturally modeled by two-player general-sum differential games with state constraints. Solving such games requires numerically computing the Nash equilibrium of coupled Hamilton–Jacobi (HJ) PDEs, but these computations suffer from the curse of dimensionality (CoD). Physics-informed neural networks (PINNs) offer a scalable alternative, yet the resulting control is derived by maximizing the Hamiltonian with respect to gradients of the learned value function. Consequently, inaccurate value approximations can result in unsafe closed-loop control policies. Prior attempts to improve the accuracy of learned values have relied on supervised data, but these data are costly to obtain and might be unavailable in high-dimensional settings. To overcome these challenges, we propose Epigraph-based Multigrid PINNs (EMP), a fully self-supervised framework that eliminates dependence on supervised data. EMP introduces a learned value-driven rollout sampling strategy that leverages informative values and their gradients along closed-loop trajectories for PINN training, and applies multigrid refinement to improve the accuracy of value approximations. Together, these components yield safe and scalable control policies. Evaluations on 5D and 9D vehicle systems and a 13D drone system demonstrate that EMP achieves lower collision rates than all epigraph-based baselines under comparable budgets, highlighting its effectiveness for safety-critical multi-agent interactions without supervised data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper shows a novel PINN method for two-player general-sum differential games. The paper follows the epigraph formulation, where the value function is augmented with an auxiliary state input $z_i$ to indicate the “budget” for the loss, and is characterized as the viscosity solution to the HJ equations, which is usually learned via a BVP solver from PDE and boundary conditions. But this method can be inaccurate, so the authors incorporate the dynamics of the value gradient to rollout trajectories via Hamiltonian maximization over the current learned value function and then refine the value function and their gradients. To alleviate the error accumulation through imperfect value function learning, the authors adopt a multi-grid refinement process, using projection and interpolation to refine the value estimates. The proposed learning method produces a policy that demonstrated leading performance over various two-player dynamical systems, including 5D vehicle driving scenarios, 9D road avoidance, and 13D drone collision avoidance tasks, with consistently higher safety rates.

### Strengths
1. Strong empirical results (the authors have compared to other epigraph-based methods and methods that used BVP and PMP, and methods that don’t have value gradient dynamics or multi-grid refinement).
2. Persuasive ablation studies to show the benefit of multi-grid refinement for the value function.

### Weaknesses
1. Didn't compare to some other baselines (like MPC methods or trajectory-optimization methods).
2. Didn't show how many random seeds were used in the experiment.
3. The paper writing is a bit dense in math (lack of explanation when introducing the Epigraph formulation - my suggestion is to look at how the EF-PPO paper [1] writes about the derivation).

References:
1. So, Oswin, and Chuchu Fan. "Solving stabilize-avoid optimal control via epigraph form and deep reinforcement learning." arXiv preprint arXiv:2305.14154 (2023).

### Questions
1. Can you comment on “L136-137, complete information” -> how strong is this assumption？
2. Why not consider multi-level refinement (instead of 2h->h->2h, try to do 4h->h->4h, or even higher resolution down-scaling), or other interpolation methods?

### Soundness
3

### Presentation
2

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
The authors propose EMP, a method for solving continuous-time two player general-sum differential games with state constraints. EMP solves for the Nash equilibrium of two coupled HJ PDEs by augmenting epigraph based PINN solver for the coupled HJ PDE with an additional loss term using value and gradient information from closed-loop rollouts, as well as a novel multigrid refinement stage after PINN training to further reduce approximation errors. Empirical evaluations on three case studies show the proposed components reduce constraint violations compared to existing methods.

### Strengths
- The proposed multigrid refinement is very interesting and seems very effective at high dimensional problems
- Empirical evaluations on the case studies show strong performance improvements from the proposed method, especially on the high dimensional case study

### Weaknesses
- The idea of using multigrid refinement in PINNs does not seem to be new [A, B, C], yet the authors do not discuss or compare against these existing methods of using multigrid refinement for PINNs. Some type of discussion would be appreciated to see how the proposed multigrid refinement compares to other methods of implementing ideas
- The sensitivity of the method’s performance to various hyperparameters is not clear
    - The loss has four terms and hence has coefficients C1, C2, C3 to balance each loss term against each other. The values of C1, C2, C3 are not disclosed in the paper. How sensitive is the method to choices of these values?
    - The timestep of the multigrid refinement ($h=0.02s$) seems independent from the ∆t used for the rollout during PINN training ($\Delta t=0.02s$). How were these values chosen? How does the choice of h affect the effectiveness of the proposed multigrid refinement?
- The 2nd question in the results ("What impact do different activation functions have on the safety of control policies via learned values?”) seems quite strange. The activation functions considered (comparing relu, tanh, sin) and conclusion (tanh performs the best, relu underperforms, performance of sin is case study dependent) are **exactly the same** as in Zhang et al. (2024).

[A] Riccietti, Elisa, et al. *Multilevel Physics Informed Neural Networks (MPINNs)*. 2022, openreview.net/forum?id=g5odb-gVVZY.

[B] Dong, Daiwei, et al. "PINN-MG: A Multigrid-Inspired Hybrid Framework Combining Iterative Method and Physics-Informed Neural Networks." arXiv preprint arXiv:2410.05744 (2024).

[C] Zhang, Enrui, et al. "Blending neural operators and relaxation methods in PDE numerical solvers." *Nature Machine Intelligence* 6.11 (2024): 1303-1313.

### Questions
- I am confused about how the authors count the dimension for each case study. For example, the joint state space for case 1 is [d1, v1, d2, v2], which is 4d, but the authors call it 5d. The same is true for case studies 2 and 3, where an extra dimension of 1 seems to be added to the problems? I’m guessing that this corresponds to the additional dimension from the epigraph variable, but I would argue this additional state dimension comes from the proposed method and is not *intrinsic* to the problem itself. Hence, the claims of 4d, 9d and 13d seem misleading.
- Line 204: “In our problem settings, we take \lambda_i = 1.” What does \lambda_i mean? Why can you just (seemingly arbitrarily) take \lambda_i = 1?
- Line 269: What is \mathcal{X}_{GT}? This is used without definition.
- Line 459: “Multigrid refinement addresses this challenge by accelerating training” I’m don’t see how this claim is justified in the paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies two-player general-sum differential games with shared state constraints. Solving the coupled HJ PDEs is hard at scale, and small gradient errors can make the resulting controllers unsafe. The method, Epigraph-based Multigrid PINNs (EMP), combines an epigraph formulation with value-driven rollouts that generate self-supervised value/gradient targets and a two-level FAS multigrid refinement to reduce residuals. On 5D, 9D, and a 13D drone case, EMP reports lower collision rates than epigraph-based baselines, and ablations show both rollout and multigrid are necessary (without multigrid, collisions occur; without rollout, trajectories are safe but off ground truth).

### Strengths
- **Self-supervised and targeted.** Value-gradient rollouts supervise both values and gradients—the quantities used for Hamiltonian control—removing dependence on fragile PMP/BVP supervision. 
- **Error control that matters.** The multigrid correction attacks the time-coupled residuals that accumulate during rollouts, improving value accuracy where it affects safety. 
- **Evidence beyond toy systems.** Results scale to 13D and the ablation clearly shows the contribution of each module.

### Weaknesses
1. **No equilibrium diagnostics.** The paper focuses on safety but does not measure Nash consistency (e.g., best-response error/regret).  
2. **Comparative scope.** Baselines are epigraph-centric; I didn't  see comparisons to standard general-sum MARL/Nash-learning methods.
3. **Numerical/robustness details.** The epigraph max is non-smooth and constraint activation induces gradient jumps; smoothing/subgradient handling and robustness to noise/model mismatch are not reported, and inference-time cost is unclear.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Epigraph-based Multigrid PINNs (EMP), a self-supervised framework for two-player general-sum differential games under state constraints. The proposed methodology combines value-driven rollout sampling with multigrid refinement to improve value function approximation accuracy, while alleviating the need for supervised data. The experiments show promising performance as EMP outperforms the presented baselines achieving higher safety performance.

### Strengths
1. The authors address an important limitation of prior literature relying on expensive and fragile boundary value problem (BVP) solvers for supervision. 

2. The value gradient dynamics derivation (Section 3.2) is mathematically sound, building on Bokanowski et al. (2021) and Hermosilla & Zidani (2023). In addition, the multigrid FAS adaptation (Section 3.3) is clearly presented with explicit operators. Finally, the full methodology (Section 3.4) is also presented with sufficient clarity. 

3. EMP demonstrates strong empirical results achieving 0% collision rates across all benchmarks and outperforms the baselines including the supervised EHPINN method. The ablation studies presented are also useful for isolating contributions, showing that multigrid is essential for complex systems.

### Weaknesses
1. **Limited novelty of core components.** It seems as the main merit of this paper stems from the combination of existing ideas into a single framework rather than a substantially novel contribution. In particular, the epigraph technique for state-constrained games is incorporated from Zhang et al. (2024). The value gradient dynamics follows standard results from Bokanowski et al. (2021). Furthermore, multigrid FAS is also a rather common PDE solver technique. While the integration is non-trivial and empirically effective, the paper lacks a novel theoretical insight or algorithmic innovation. 

2. **Weak theoretical guarantees.** The convergence analysis presented in Section C.3.1 provides only a monotonicity result $\hat{V}_i^{k+1} \leq \hat{V}_i^k$, which does not guarantee convergence to the correct Nash equilibrium value. 

3. **Ground truth and limited baseline comparisons.** The evaluation relies on BVP solvers to generate ground truth, but these only provide local optima rather than global Nash equilibria. This is addressed with multiple initial guesses, but validation is limited since only one 5D case (Fig. 20c) is compared against level set methods. In addition, all baselines are epigraph-based methods (EPPO, EHPINN, EPIML), with no comparison to other game-theoretic solvers mentioned in related work (e.g., Fridovich-Keil et al., 2020b for iterative methods).

### Questions
1. Apart from examining collision rates, how can you verify that the learned policies constitute a Nash equilibrium? 

2. Given that BVP solvers only provide local optima, have you validated the ground truth in additional cases beyond the single 5D comparison with level sets (Fig. 20c)? Also, how sensitive are the results to the choice of initial guesses for the BVP solver? 

3. Have you compared against non-epigraph-based game solvers (e.g., iterative LQ methods) that are cited?  

4. How sensitive is the proposed framework to hyperparameter tuning?

### Soundness
3

### Presentation
3

### Contribution
2
