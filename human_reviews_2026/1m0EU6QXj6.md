# SafeMPO: Constrained Reinforcement Learning with Probabilistic Incremental Improvement

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 4

## Abstract
Reinforcement Learning (RL) has demonstrated significant success in optimizing complex control and planning problems. However, scaling RL to real-world applications with multiple, potentially conflicting requirements requires an effective handling of constraints. We propose a novel approach to constraint satisfaction in RL algorithms, focusing on incrementally improving policy safety rather than directly projecting the policy onto a feasible region. We accomplish this by first solving a nonparametric surrogate problem which is guaranteed to contract towards the feasible set, and then cloning that solution into a neural network policy. As a result, our approach improves stability, particularly during early training stages, when the policy lacks knowledge of constraint boundaries. We provide general theoretical results guaranteeing convergence to the safe set for this class of incremental systems. Notably, even the simplest algorithm produced by our theory produces comparable or superior performance when compared to highly tuned constrained RL baselines in challenging constrained environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a novel approach to constrained reinforcement learning, referred to as SafeMPO: Instead of relying on projection based approaches, which project the policy onto the constraint-satisfying set at every step, SafeMPO relies on incremental improvements by ensuring that each policy update monotonically increases the likelihood of safety, rather than enforcing hard constraint satisfaction at every step.

The motivation for this is that projection based approaches can be overly restrictive, effectively preventing the discovery of optimal, constraint satisfying solutions by enforcing strict constraint satisfaction at every step.

The paper builds on MPO and the EM algorithm, but extends MPO to handle additional safety constraints. In the E step, the method computes the improved, nonparametric distribution (in terms of return **and** constraint violation) $q^\ast$ over states and actions in closed form, subject to a KL constraint for $q^\ast$ to remain close to the current policy. Then, in the M step, the policy is fitted to the improved distribution $q^\ast$.

The paper provides theoretical guarantees showing that the incremental improvement process forms a contraction toward the feasible set, yielding geometric convergence under mild assumptions. 
SafeMPO is evaluated empirically on three environments from the OmniSafe benchmark and compared with a number of appropriate baselines.

### Strengths
The proposed incremental approach to constrained reinforcement learning is, to the best of my knowledge, novel and well-motivated. Rather than enforcing strict constraint satisfaction at every update via projection of the policy onto the feasible set, the method ensures that each policy is at least as safe as the previous one. In settings where incremental progress toward the feasible set is acceptable, this formulation is intuitively less restrictive and may enable the discovery of higher-return, constraint-satisfying solutions that projection-based methods could fail to discover.

The proposed SafeMPO method is principled, and the paper's theoretical analysis provides guarantees for a practical constrained RL algorithm, including monotonic safety improvement and geometric convergence toward the feasible set under mild assumptions.

Finally, by leveraging a nonparametric surrogate optimization in the E-step and a supervised M-step, the method inherits the stability and low-variance optimization behavior of MPO-style algorithms while extending them naturally to constrained settings.

### Weaknesses
The proposed approach is principled and I very much agree with the motivation and would like to see (an improved version of) this work published. However, the following points strongly impact my score:

**(1) Clarity and exposition**
While the paper uses typical notation from variational inference, the presentation of key background material should be improved substantially. The paper assumes strong familiarity with MPO and the EM algorithm, but neither is introduced sufficiently, making the paper not self-contained.

In particular, the introduction of MPO (lines 120–124) is confusing and somewhat misleading. What exactly is 
$q$ here? Unlike what is stated, and if I am not mistaken, $q$ is not an approximation of the RL objective, but rather an auxiliary, implicit distribution over states and actions used to optimize it.

Nitpick: There are also a few other statements that lack clarity, IMO. For example, the feasible set is never properly defined. The $\epsilon$-ball on line 67 is not defined. Referring to “the neural network” as part of SafeMPO is problematic, since there are multiple neural networks in RETRACE alone (RETRACE is also mentioned only briefly and not explained in a self-contained manner).

**(2) Experimental validation**
The empirical results are limited and somewhat contradictory. In all reported environments, the standard SafeMPO variant fails to find a policy satisfying the cost constraint (B=25), as the asymptotic costs after 10M steps remain above this threshold. The found policies are therefor **not** part of the feasible set. The chosen environments are relatively simple navigation tasks, making this result somewhat concerning. Lowering the cost bound (to B=20) to achieve feasibility (on 2/3 environments) is not an adequate solution, as it effectively alters the optimization objective in a way similar to tuning the relative weights of reward and cost. Thus, the experiments do not convincingly demonstrate that the proposed algorithm converges to constraint-satisfying solutions in practice.

**(3) Disconnect between motivation and findings**
The authors argue that the incremental approach in SafeMPO is less restrictive and can therefore discover solutions that are hard to find using projection-based methods, as nicely illustrated in the didactic example in Figure 1. It would strengthen the paper’s claims considerably if the authors demonstrated an environment where such exploration challenges due to constraints exist, and where SafeMPO successfully overcomes them.

**(4) Lack of environment and implementational details**
The environments are not sufficiently explained. The state and action spaces are not stated, making it hard to know how easy or hard these environments truly are. Similarly, the hyperparameters of SafeMPO aren't stated, limiting the interpretability and reproducibility of the results.

### Questions
1. In equation 8a, the term $\kappa\ \log (\frac{x}{\kappa})$ seems to be constant in (s,a). How does it affect maximization over $Q(s,a)$?

2. Why do all almost all of your baselines fail to find feasible solutions, even on the simplest environment? Might it be that there is insufficient capacity in the estimators for the cost function or policy networks?

3. Why does the performance of SafeMPO in terms of return increase on 2/3 environments when a stricter cost coefficient is use?

### Soundness
2

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
3

### Summary
This paper presents SafeMPO, a constrained reinforcement learning algorithm derived from the maximum a posteriori policy optimization (MPO) framework. The method aims to guarantee asymptotic safety constraint satisfaction with contiguous policy improvement. The authors formulate an expecation maximiation algorithm with a constrained optimization problem in the E-step (Theorem 1), introduces a modified policy update that includes a safety barrier (Eq. 8a), and provides theoretical results relating the algorithms contraction and monotonicity properties to safety performance. Empirically, SafeMPO is evaluated on standard continuous-control safety benchmarks (SafetyGym) and compared to existing constrained policy optimization baselines.

### Strengths
- I find the idea interesting and novel. Combining MPO-style policy updates with constrained optimization is a natural direction that should provide some interesting insights. The attempt to incorporate constraint satisfaction directly into the policy updates of an MPO-style algorithm is conceptually nice.
- The derivation of the constrained optimization E-step are motivated optimization. The authors include a variety of additional theoretical results (contraction and convergence) to their algorithmic contribution. 
- The motivation for combining the probabilistic policy optimization framework with constrained objectives is clear, and the paper structure (E-step / M-step / theory / experiments) is easy to follow.
- The paper is overall well-written.

### Weaknesses
**Conceptual and theoretical clarity**:

- I found the E-step section, particularly around Theorem 1, quite hard to follow. It is not clear to me why an optimal solution to Theorem 1 would not guarantee a safety improvement, as constraint (7c) seems to directly enforce this. Any solution violating safety improvement should be infeasible under that constraint, so the subsequent introduction of the barrier function in Eq. (8a) is hard to comprehend from my position. Going to the appendix reveals that the authors likely mean that the first order KKT conditions permit points that are not feasible, but these conditions in my understanding are not a characterization of global optima. Either way I believe this should be explained more explicitly, as it lays the foundation for the motivation of subsequent algorithmic components. 
- The barrier function added as an additional regularizer in (8a) makes the algorithm less "pleasing" to me. If I am correct, the introduced term behaves similarly to a Lagrange multiplier or penalty term, which raises the question of whether the same problem is being solved twice - once in the primal constraints and again in the added regularizer. In general I find it slighly confusing that the authors seem to pursue a "a nonzero improvement in our safety constraints" but do not formalize their primary constraints as such, instead permitting zero-improvement (7c). 
- The authors’ usage of the term "contraction mapping" (Corollary 2 and 3) confuses me a bit. Contractions are properties of operators, yet the text seems to apply the notion to entire inequalities. It is a bit unclear to me what mapping is claimed to be contractive, and with respect to which norm or metric.

**Conceptual tension with the control-as-inference view**:

This is a discussion on a more philisophical level, but I believe there is a tension between soft control formulations (where policies emerge from defining optimality from a lens of energy-based exponential distributions) and hard safety constraints. The paper incorporates constraints onto a probabilistic inference objective, but what does the introduction of hard constraints imply for the soft nature of the probabilistic treatment of control? In the classical control-as-inference setting, maximum entropy solutions naturally emerge as solutions to the probabilistic problem, which appear to be incompatible with hard constraints if left untreated. This conceptual issue perhaps deserves some discussion.

**Experiments**:

The experiments look executed well but their results look rather inconclusive to me. The differences between SafeMPO and existing constrained methods are minor across most benchmarks, and it is unclear in which scenarios SafeMPO’s particular constraint handling makes a significant difference. The authors should better articulate where SafeMPO is expected to excel (e.g., in nonconvex or highly stochastic safety settings) and provide corresponding experiments.

### Questions
- Could you clarify in why the optimal solution to Theorem 1 does not guarantee safety improvement? Does this follow from a characterization through KKT conditions? If so, do these condition allow a general statement as done about the global optimal solutions? 
- How does the behavior of the barrier function in Eq. (8a) differ from a standard Lagrange or penalty term? Why is the non-zero improvement of constraint satisfaction permitted in the primary constraints in Eq. (8a) if it appears to be the authors wish to ensure non-zero improvement? 
- In what sense is the E-step mapping a "contraction"? What operator or function is being iterated, and in which norm?

### Soundness
3

### Presentation
2

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
This paper proposes SafeMPO, a novel algorithm for Constrained Reinforcement Learning (CRL) that extends Maximum a Posteriori Policy Optimization  to the CMDP setting. Instead of using primal–dual updates or projection-based recovery, SafeMPO enforces *incremental safety improvement* via a *log-barrier constraint on a safety-order-preserving function \(K\)*. The method yields a closed-form E-step distribution \(q^*(a|s)\) and theoretical guarantees that the iterates contract toward the feasible set under mild assumptions. Empirical results on **Safety-Gymnasium** tasks demonstrate competitive or superior reward-cost trade-offs compared to strong baselines like CPO, PCPO, RCPO, and CPPOPID, with notably low variance across runs.

### Strengths
- The paper offers a novel probabilistic viewpoint on safe RL: treating safety improvement as a likelihood maximization problem rather than a constraint projection.  
 - The barrier-regularized E-step and its EM-style update are well derived and integrate naturally with existing MPO framework.
- Theoretical contributions are well-structured. The paper first proves that naive improvement bounds do not necessarily contract toward feasibility, then introduces a log-barrier to guarantee convergence (Theorems 2–4). I did not check all the theoretical results but they look correct and rigorous 
- The algorithm looks simple.
- Experiments across three standard safety environments show SafeMPO achieving near-SOTA returns with moderate constraint violations.

### Weaknesses
- **Experimental scope is limited.**  
  Only three tasks from the Safety-Gym benchmark are evaluated, which is insufficient to demonstrate the generality of the approach. The study would be strengthened by including a broader range of tasks to better assess scalability and robustness.  

- **Ablation studies are underdeveloped.**  
  The authors should conduct sensitivity analyses on key hyperparameters—specifically the step-size parameter \(\kappa\), the KL divergence radius \(\varepsilon\), and the maximal dual variable \(M_\lambda\)—since the theoretical convergence guarantees explicitly depend on these quantities.  

- **No exploration of alternative safety likelihoods.**  
  Although the paper acknowledges that other likelihood formulations beyond the truncated exponential are theoretically possible, no experiments evaluate such alternatives. Empirical validation of different safety likelihoods would be necessary to justify the exclusive use of the truncated exponential form.  

- **Baselines are outdated.**  
  The experimental comparison omits several recent state-of-the-art constrained RL algorithms, such as **Constrained Variational Policy Optimization (CVPO)** [1] and imitation-based constrained methods [2]. Including these baselines is essential for establishing the competitiveness of the proposed approach.  

**Refs**

- [1] Liu, Z., Cen, Z., Isenbaev, V., Liu, W., Wu, S., Li, B., & Zhao, D. (2022). *Constrained Variational Policy Optimization for Safe Reinforcement Learning*. In *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)*, PMLR 162, 13644-13668. [https://proceedings.mlr.press/v162/liu22b.html](https://proceedings.mlr.press/v162/liu22b.html)

 - [2] Hoang, H., Mai, T., & Varakantham, P. (2024). Imitate the Good and Avoid the Bad: An Incremental Approach to Safe Reinforcement Learning. In Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (Vol. 38, No. 11, pp. 12439-12447). AAAI.

### Questions
Please address the points I raised in the Weakness.

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
3

### Summary
This paper introduces a constrained policy optimization algorithm based on Maximum-a-Posteriori Policy Optimization (MPO), providing theoretical guarantees for convergence to the feasible set of the constrained problem. The approach extends MPO’s control-as-inference formulation by adding a “policy is safe” event alongside the “policy achieves goal” event. The authors argue this improves conditioning, as safety can be improved even outside feasibility. Experiments in Safety-Gymnasium (Ji et al., 2023) show partial empirical gains.

### Strengths
- Novel idea of redefining MPO using a safety probability event  
- Engaging and well-motivated introduction  
- Theoretical results appear sound (not fully verified)

### Weaknesses
- Experimental results are not fully convincing  
- Expected stronger sample efficiency gains since MPO is off-policy  
- Theoretical clarity: Corollary 3 seems central but is not well explained  
- Mathematical presentation issues: $ C(a,s) $ is estimated but never defined or used; unclear whether this refers to $ G(s,a) $ or $ K(s,a) $. Clarify how these are estimated via Retrace.

### Questions
- Why can the most likely posterior policy not be non-feasible?

### Soundness
3

### Presentation
3

### Contribution
3
