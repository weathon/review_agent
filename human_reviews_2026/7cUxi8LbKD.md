# SHAPO: Sharpness-Aware Policy Optimization for Safe Exploration

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Safe exploration is a prerequisite for deploying reinforcement learning (RL) agents in safety-critical domains. In this paper, we approach safe exploration through the lens of epistemic uncertainty, where the actor’s sensitivity to parameter perturbations serves as a practical proxy for regions of high uncertainty. We propose Sharpness-Aware Policy Optimization (SHAPO), a sharpness-aware policy
update rule that evaluates gradients at perturbed parameters, making policy updates pessimistic with respect to the actor’s epistemic uncertainty. Analytically we show that this adjustment implicitly reweighs policy gradients, amplifying the
influence of rare unsafe actions while tempering contributions from already safe ones, thereby biasing learning toward conservative behavior in under-explored regions. Across several continuous-control tasks, our method consistently improves both safety and task performance over existing baselines, significantly expanding their Pareto frontiers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Sharpness-Aware Policy Optimization, a novel approach to enhance safe exploration in Reinforcement Learning by addressing the actor's epistemic uncertainty. The core idea is to use the "sharpness" (sensitivity of the policy to parameter perturbations) as a practical proxy for uncertainty, particularly in regions with scarce data.
SHAPO leverages principles from Sharpness-Aware Minimization (SAM). The proposed update rule operates in a pessimistic manner: it first finds the worst-case policy parameter θ within a local neighborhood that minimizes the performance objective (maximizes risk), and then computes the policy gradient at this "pessimistic" point. This results in a modification of the policy gradient that analytically promotes conservative behavior by amplifying the penalty for rare, unsafe actions while attenuating the reward for rare, safe actions. The method is implemented as a plugin on top of existing safe RL algorithms (TRPO) and demonstrates improved performance on the safety-efficiency Pareto frontier across various Safety Gym tasks.

### Strengths
1.  Applying Sharpness-Aware Minimization (SAM), typically used for improving generalization in supervised learning, to Safe RL is a  conceptually elegant contribution. This provides a new lens for addressing epistemic uncertainty in the actor network.
2. The analysis (Section 3.3) and empirical results support the claim that the SHAPO gradient effectively introduces a pessimistic bias. The method consistently achieves a lower frequency of catastrophic events and generally improves the safety performance of multiple SOTA safe RL baselines when used as a plugin.
3. The update rule is straightforward and can be readily integrated with different Policy Optimization algorithms without fundamental changes to their core optimization objective, showcasing high modularity.

### Weaknesses
1. The paper suffers from a crucial disconnect between the theory and implementation. Proposition 2 defines the perturbation magnitude \delta_{Down}​	 as a function of the sample size n and confidence level α. However, in the practical implementation and hyperparameter search (Appendix D), \delta_{Down}​ is treated as a fixed hyperparameter, which contradicts the theoretical guidance for an annealing schedule based on n.
2. The modeling of the posterior distribution Q(θ) is solely "motivated" by the BvM theorem. This represents a strong, unproven assumption, as the strict regularity conditions required for BvM are unlikely to hold in high-dimensional, non-convex deep RL settings, thus introducing a significant theoretical gap.
3. There is an explicit inconsistency in the definition of the covariance matrix for the distribution Q(θ) between the main text and the appendix proof of Proposition 5, which uses a distribution corresponding to a precision matrix proportional to $\sqrt n$. This error undermines the rigor of the derived relationship for $\delta _{Down}$.
​4. The paper focuses on comparing SHAPO's performance against standard constrained optimization methods (CPO, CRPO, etc.). To fully validate the claim of solving epistemic uncertainty in the actor, the baselines should ideally include methods that explicitly handle uncertainty or risk sensitivity, such as: Policy Gradient methods utilizing Dropout or Ensembles on the actor. Risk-Sensitive Policy Optimization methods (e.g., those based on CVaR or other risk measures).
5.  While $\delta _{Down}$ is the critical new hyperparameter of the method, the paper provides only a qualitative discussion of its sensitivity (Appendix D) and lacks a comprehensive, quantitative ablation study.

### Questions
1. Given the theoretical prescription that $\delta _{Down}$ should decay with the number of samples n, why was a fixed, tuned hyperparameter used in the implementation? Could the authors provide an ablation study comparing the fixed hyperparameter approach with the theoretically suggested annealing schedule?
2. Please clarify and correct the discrepancy in the definition of the posterior/likelihood distribution Q(θ) between Section 3.2 and Appendix B. A corrected, rigorous proof for Proposition 2 (or 5) is required.
3. To solidify the contribution of using "sharpness" as an uncertainty proxy, could the authors compare SHAPO's performance against policy gradient methods that explicitly model actor uncertainty or risk, such as an Ensemble Actor approach or a CVaR-based policy optimization method?

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
4

### Summary
The paper explores the use of sharpness aware optimization for safe RL. Sharpness aware optimization introduces a max-min problem instead of maximization to flatten the optimization profile. In my understanding this max-min problem would aim to prefer a flatter maximum to a sharp peak maximum in the parameter space. This approach aims to improve robustness of the optimization solution to epistemic uncertainty. The authors conduct a series of experiments that show an improvement of performance of different algorithms with and without SHAPO.

### Strengths
1.The idea is novel to my best knowledge and quite interesting. 
2. The paper contains a deep discussion on my sharpness aware optimization is a good fit for ML problems in general. In section 3.3, the authors dive deeply into why SHAPO can be a good fit for RL.
3. The experiments and ablation study are great

### Weaknesses
I couldn't come up with many weaknesses, but here are a couple 
* The policy update section 3.1 is building the solution bottom-up, but maybe a top-down approach would be slightly easier to read. 
* It would be good to explain how the policy update is actually implemented. The authors present a quadratic optimization problem in  TRPO style update and state that SHAPO update can be applied to any on-policy algorithm. Can the authors elaborate how SHAPO can be applied to a PPO update?

### Questions
* The approach makes total sense and it seems relevant not only for safe RL, but also for RL. What happens if SHAPO is applied without a safety constraint? Did the authors perform this ablation study? 
* Can the authors provide similar Figures to Fig 6 and 7, but in terms of episode return vs episode cost? I think their decision to focus on cost rate is correct, but having a full picture (maybe in appendix) would be good.
* Can the authors elaborate what’s the base method for Saute RL? As the authors are aware, Saute RL can be applied to any RL algorithm.   
* Safety gymnasium has many more environments that could be useful for your future research and evaluations. Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark https://arxiv.org/abs/2310.12567
* What’s the additional computational burden that the approach adds to the algorithm?

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
3

### Summary
The paper studies the application of sharpness awareness minimization (SAM) in policy optimization. SAM aims at optimizing the model such that the worst-case loss in its neighboring parameters is low, encouraging the optimizers to find low-loss solutions that are also flat. In particular, the authors extend Fisher-SAM, a prior SAM method that leverages Fisher information matrix to estimate the local geometry to efficiently estimate the local loss-maximizing parameter, to trust-region policy optimization for safe exploration. The paper provides an interpretation of Fisher-SAM as a pessimistic estimation of the loss subject to the uncertainty quantified by the Fisher matrix. This in turn allows the authors to interpret the worse-case expected return in the neighboring parameters informed by the Fisher matrix as an uncertainty lower-bound on the actual expected return. The paper evaluates the proposed method, SHAPO, on safety gym and shows that the proposed method outperforms prior methods in terms of the return/cost trade-off.

### Strengths
- The paper is easy to read and the proposed method is well-motivated and technically sound. 
- The empirical results show statistically significant improvements of the proposed method over prior safe RL methods. Ablations are thorough and show that major components of the algorithms all contribute to the effectiveness of the proposed method.
- Even though SAM/Fisher-SAM is not new, the application of it in the context of safe RL is new and seems effective.

### Weaknesses
- The proposed method requires solving the natural gradient direction that maybe very expensive. 
   - I tried to look for the implementation details in the paper but could not find it. It would be good if the authors could include more implementation details in the paper for better reproducibility. 
   - I checked the linked anonymous codebase briefly and it seems that the implementation involves running an iterative conjugate gradient procedure (https://anonymous.4open.science/r/Safe-Policy-Optimization-813E/safepo/single_agent/shapo.py). This procedure can incur a non-trivial amount of computation overhead. It would be good if the authors could include more details on the run-time of the algorithm compared to prior methods in the paper. 

- Hyperparameter sensitivity analysis is missing from the empirical evaluations. In Appendix, the authors mentioned that "Most common choice of SHAPO hyperparameters that gave the best performance across tasks and baselines was $\delta_{\mathrm{Down}} = 0.0001$ and $\rho_{\mathrm{critic}} = 0.01$", but it is unclear how different hyperparameters influence the performance of the algorithm. Including some analysis on the sensitivity of the performance with respect to these hyperparameters (especially $\delta_{\mathrm{Down}}$ could help gain a better understanding of the robustness and effectiveness of the algorithm.

### Questions
- In Figure 3, 6 and 7, how are the shapes of the circles around the mean/average performance determined?
- In algorithm 2, how is $U_\theta$ computed?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Sharpness-Aware Policy Optimization (SHAPO) to address the challenge of safe exploration in safety-critical RL. The core idea is to use the actor's sharpness as a practical proxy for epistemic uncertainty. The method implements a pessimistic policy update. In each step, it first solves an inner-loop optimization to find the "worst-case" perturbed parameters ($\theta_0 + \epsilon_{Down}$) that minimize the policy objective within a trust region which is defined by the Fisher Information Matrix (KL divergence). It then computes the policy gradient ($\tilde{g}$) at this worst-case point and uses this pessimistic gradient for the final policy update .
Analytically, the authors show this process implicitly reweighs gradients, amplifying the effect of rare, unsafe actions and tempering the effect of rare, safe actions.

### Strengths
- The paper tackles the critical challenge of safe exploration in a practical and useful manner, which is a prerequisite for real-world RL deployment.

- The analytical insights, including Proposition 3 on gradient adjustments for rare actions and the reinterpretation of perturbations as pessimistic quantiles (Proposition 2), are clear and well-supported.

- Evaluations across multiple baselines and environments demonstrate consistent gains, with Pareto improvements and reduced catastrophic failures

### Weaknesses
-  Major Theory-Implementation Mismatch

The paper's theoretical justification for being "uncertainty-aware" (presented in L249-256)  hinges on the idea that the inner trust region $\delta_{Down}$ should adapt based on the amount of data $n$. However, the actual implementation described in Appendix D uses a fixed, grid-searched $\delta_{Down}$.

- Gap in Intuitive Justification

The paper argues its strength comes from using the Fisher metric . However, the core intuition for why it is safe (Section 3.3)  is derived using a simplified Euclidean metric (Appendix C, L778: "For simplicity, we consider here a Euclidean perturbation...").

### Questions
- Could you clarify whether the adaptive scaling rule was implemented or tested in any form?
If not, how do you reconcile this discrepancy between the theoretical motivation and the actual implementation?
- Could you prove that the core safety intuition from Figure 2  holds for the Fisher-based update, not just the simplified Euclidean case  shown in the appendix?

### Soundness
3

### Presentation
3

### Contribution
3
