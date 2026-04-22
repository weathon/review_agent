# Complexity-Driven Policy Optimization

- Avg Score: 1.50
- Decision: Reject
- Scores: 2, 2, 2, 0

## Abstract
Policy gradient methods often balance exploitation and exploration via entropy maximization. However, maximizing entropy pushes the policy towards a uniform random distribution, which represents an unstructured and sometimes inefficient exploration strategy. In this work, we propose replacing the entropy bonus with a more robust complexity bonus. In particular, we adopt a measure of complexity, defined as the product of Shannon entropy and disequilibrium, where the latter quantifies the distance from the uniform distribution. This regularizer encourages policies that balance stochasticity (high entropy) with structure (high disequilibrium), guiding agents toward regimes where useful, non-trivial behaviors can emerge. Such behaviors arise because the regularizer suppresses both extremes, e.g., maximal disorder and complete order, creating pressure for agents to discover structured yet adaptable strategies. Starting from Proximal Policy Optimization (PPO), we introduce Complexity-Driven Policy Optimization (CDPO), a new learning algorithm that replaces entropy with complexity. We show empirically across a range of discrete action space tasks that CDPO is more robust to the choice of the complexity coefficient than PPO is with the entropy coefficient, especially in environments requiring greater exploration.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a novel use of LMC complexity as a regulariser for on-policy RL algorithms that helps with exploration in a more structured way than naive policy entropy regularisation. They argue LMC complexity favours policies that are neither fully deterministic nor fully uniform. They present a set of empirical results comparing against PPO with and without policy regularisation and show some improvements in terms of stability and rewards.

### Strengths
- The problem of how to efficiently and effectively induce exploration in RL is an extremely relevant one.
- The method is well presented and the intuition behind it is clear.
- The paper is well written.

### Weaknesses
- I am not convinced of the depth of the contribution. While the problem addressed is very relevant, I am not sure the modification proposed (using LMC as a regulariser) is enough of a contribution in itself: it is a (arguably small) modification over well studied entropy regularisers. The theoretical analysis is also quite limited, so that does not help understand the advantages of using LMC versus other existing methods for encouraging exploration. While I agree that simply regularising for fixed entropy penalties seems naive and inefficient, it has advantages: It is very interpretable (in terms of the effects it causes in the policies and in the optimisation problem solution) and well studied (also in e.g. off-policy RL) both theoretically [1] and empirically. The authors themselves admit that the behaviour of the gradient descent algorithm is quite complex in terms of the LMC gradients, increasing or decreasing action probabilities depending on the policy.
- For such a paper to have enough of a contribution to the field, (if lacking a theoretical one) I would expect a very thorough empirical study on the impact of LMC regularisers in a wide range of RL benchmarks. The evaluation is very limited, with only a handful of environments and only against PPO. While I understand why the authors would prefer to focus on a narrower empirical setting, it fails to add to the contribution. In general, I cannot consider the improvements observed to be of sufficient relevance. In most of the environments the improvement is only observed against some of the regulariser coefficients, and the CARTerpillar environment (while maybe illustrative) is quite constructed and not enough of an argument in itself.


[1] Levine, Sergey. "Reinforcement learning and control as probabilistic inference: Tutorial and review." arXiv preprint arXiv:1805.00909 (2018).

### Questions
- What is the reason behind comparing both regularisers with the same set of coefficients? Taking a Lagrangian interpretation, the regularisation coefficient fixes the optimisation constraint bound, but given that the two regularisers are fundamentally different there is no reason to believe they should be regularised with the same strength. It could very well be that the range picked is simply too wide, and it impacts the entropy regulariser more than the LMC. 
- This raises a secondary point; most experimental results are reporter by averaging across all results for different coefficients, but if one takes the best performing one, the performance increase is not really apparent.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors observe that maximum entropy methods can sometimes lead to excessive exploration. They propose an exploitation objective that mitigates this problem. The objective consists of two parts, one that motivates a uniform policy and the other a deterministic policy, thus allowing control over its randomness.

### Strengths
The overall observation is accurate and the approach is fair, supported by a method from the literature.

### Weaknesses
Several weaknesses limit the impact of the paper at this stage.
1. Some explanations and justifications are rather vague when it comes to the concept of exploration and, above all, structure. This is more a matter of presentation. 
2. The literature on exploration methods is fairly limited. It only presents the foundation papers. In particular, much of the literature has already addressed some of the limitations of action-entropy exploration by introducing regularization on the visitation measure of states, state-actions, or features. This is discussed very briefly and is not compared to the proposed method empirically.
3. Section 2.2 is supposed to discuss the notion of complexity in RL but is limited to two papers. I find the discussion in this section lacking in substance.
4. The entropy function $S$ is not a function of state, as indicated in Eq. (2) and Eq. (5). It is defined in the pseudo-code, and the definition includes the expectation $E_t$, which marginalizes the effect of states.
5. Based on the pseudocode, I get the impression that only one iteration is performed on the PPO loss. This could be clarified. If this is the case, we end up with a classic policy gradient. Line 121, the motivation for PPO is slightly incorrect to me. In its initial version, PPO does not use a critic and does not really address the problem of policy gradient variance. The main advantage of PPO is that it reuses samples to perform several policy updates. The clip implements the idea of a “trust region”: if the policy has changed too much, the gradient is zero.
6. The regularization only works for discrete action spaces; this should be stated explicitly.
7. Section 4.2 is, in my opinion, incorrect. First, the existence of multiple optima does not imply anything about robustness. Furthermore, the discussion neglects the effect of expectation on states and does not discuss the optima as a function of theta. I do not think this section provides real insight, and certainly no theoretical justification for the superiority of the exploration objective.
8. In my opinion, the experimental setting does not allow for a proper comparison of exploration bonuses. Authors compare the aggregate effect over different weights on the regularization term. The performance for the average weight does not indicate how the two methods perform when they are correctly calibrated individually for the application. Typically, we observe here that PPOwEnt performs similarly to CDPO when we take the two best run for both methods. Smaller weigths are systematically needed for action-entropy regularization, which makes sense intuitively as it explores more agressively.
9. Benchmarks are so simple that the trivial method without exploration achieves already very good performance. Anyway, the difference between performance is very small.

### Questions
How is the gradient of the entropy bonus computed in practice?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes a new regularization method for policy gradient algorithms, named Complexity-Driven Policy Optimization. The core idea is to replace the standard entropy bonus, which is often used to encourage exploration but can lead to inefficient policies, with a LMC complexity bonus.

### Strengths
- The paper identifies a good but overlooked problem with entropy regularization. Its objective is a purely random policy, which is not an optimal exploration strategy.
- The proposed solution of using LMC complexity to find a "structured stochastic" policy is simple and intuitive.

### Weaknesses
- I am not quite convinced of both the problem and the proposed solution. There are many easier ways to reduce the effect of regularization instead of the solution proposed in this work. The major and possibly only advantage of the proposed “complexity” measure is to balance stochasticity with structure. However, this complexity might also cause the policy to be neither structured nor explored enough. Since this new complexity measure is heuristic, the paper focuses on empirical experiments instead of theoretical analysis. However, the experiments conducted are not convincing. The paper only compares CDPO to entropy regularization. While this is the most direct comparison, it misses comparisons to the work relative to the broader field of exploration.
- Applicability limited to discrete action spaces.

### Questions
How does this policy-based regularization approach compare to reward-based intrinsic motivation methods (e.g., ICM, RND)? It's unclear if CDPO is an alternative to these methods or a complementary technique that could be combined with them.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes Complexity-Driven Policy Optimization (CDPO), which replaces the standard entropy bonus in PPO with a “complexity” bonus defined as the product of Shannon entropy and disequilibrium (the deviation from a uniform distribution). This regularizer encourages exploration when the policy is nearly deterministic and imposes structure when it approaches pure randomness, alleviating the tedious tuning of the entropy coefficient and the tendency to collapse into uniform randomness. The authors conduct experiments on discrete-action tasks and show that CDPO is more robust to the regularization coefficient.

### Strengths
Originality: The contribution is incremental; it simply borrows the off-the-shelf LMC complexity measure without a deep analysis of why this particular form is superior in the context of the objective function.
Quality: The learning curves in the figures are presented in a rudimentary way and could be improved.
Clarity: Concepts, equations, pseudocode, figures, and detailed appendices are all provided, so reproducing the work is straightforward.

### Weaknesses
1. Disequilibrium relies on discrete action probabilities; the paper offers no clue on how to extend the method to continuous domains.
2. No convergence guarantees or regret bounds are given; the approach remains heuristic without theoretical evidence that complexity regularization is better than entropy.
3. Baselines are limited to PPO ± entropy; comparisons with other exploration-oriented methods such as SAC are missing, and all tested environments are rather simple.
4. It is unclear what benefits higher complexity brings. Although the paper analyzes the gradients of both entropy and complexity, the ultimate performance still depends on the environment’s objective function, leaving the relationship between complexity and algorithmic performance unexplained.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1
