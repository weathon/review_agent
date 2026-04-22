# Beyond Softmax and Entropy: Convergence Rates of Policy Gradients with $\boldsymbol{f}$-SoftArgmax Parameterization $\&$ Coupled Regularization

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 0

## Abstract
Policy gradient methods are known to be highly sensitive to the choice of policy parameterization. In particular, the widely used softmax parameterization can induce ill-conditioned optimization landscapes and lead to exponentially slow convergence. Although this can be mitigated by preconditioning, this solution is often computationally expensive. Instead, we propose replacing the softmax with an alternative family of policy parameterizations based on the generalized $f$-$\textit{softargmax}$.
We further advocate coupling this parameterization with a regularizer induced by the same $f$-divergence, which improves the optimization landscape and ensures that the resulting regularized objective satisfies a Polyak--Łojasiewicz inequality. Leveraging this structure, we establish the $\textit{first explicit non-asymptotic last-iterate convergence guarantees}$ for stochastic policy gradient methods for finite MDPs $\textit{without any form of preconditioning}$. We also derive sample-complexity bounds for the unregularized problem and show that $f$-PG, with Tsallis divergences achieves $\textit{polynomial sample complexity}$ in contrast to the exponential complexity incurred by the standard softmax parameterization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a general f-regularized policy gradient method with coupled parametrization, which generalizes the popular KL regularization with softmax parametrization. Theoretically, they demonstrate global convergence and show an improved trade-off between sample complexity and regularization bias compared to the classical entropy-softmax combination.

### Strengths
The authors propose a general framework for studying policy gradient methods in f-divergence-regularized RL with coupled parametrization, providing strong theoretical guarantees.

### Weaknesses
- The experiments are conducted on a 5×5 GridWorld, which limits the generalizability of the proposed policy gradient method.
- Apart from the theoretical contributions, which I do not feel qualified to fully assess, it is unclear to me what practical advantage the proposed algorithm offers compared to using KL regularization with softmax parametrization.

### Questions
- What are the main factors limiting the proposed algorithm from being evaluated on a broader set of environments, such as Atari?
- How does this approach compare to prior work [1] showing that Mirror Descent, with different choices of mirror map and optimization space (e.g., logits or policy), can lead to novel regularized policy gradient objectives?

[1] Vaswani, S., Bachem, O., Totaro, S., Müller, R., Garg, S., Geist, M., Machado, M.C., Castro, P.S. and Roux, N.L., 2021. A general class of surrogate functions for stable and efficient reinforcement learning. arXiv preprint arXiv:2108.05828.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes generalization of entropy regularized policy gradient methods using f-divergences, as well as corresponding parameterizations.

The authors shows global convergence results the proposed methods, as well as experimental results to verify the theoretical findings.

### Strengths
1. Generalizing the entropy regularizer to f-divergences is a reasonable idea.
2. The projection operator to avoid deterministic policies is useful.

### Weaknesses
1. The work seems to be generalizing and recovering existing work (entropy, Tsallis). It seems this work is not suggesting some new regularization which is better performing than existing ones.

### Questions
1. I am wondering how the methods perform when the interest is to obtain the unregularized optimal value/policy. I can imagine the operator to avoid small probability could fail (since it is unavoidable to get close to deterministic policies in that case). Does your method provide better iteration/sample complexity?

2. The authors mentioned that entropy and Tsallis, which are two important existing regularization, can be recovered by the f-divergences. Are there any new regularization which can perform better than existing ones (or the other way those two are the best possible)?

3. It is claimed that for the PL inequality, "our proof, based on the properties of Fenchel-Legendre conjugation, is much simpler". After checking it seems to me the key ideas of smoothness and PL inequality proofs are largely similar to existing proofs in entropy regularized proofs, especially the lower bounding of policy gradient using Eq. (52) and $H(w_\theta)$, and upper bounding the suboptimality gap by Eq. (51). Could you elaborate how Fenchel-Legendre conjugation makes your proofs much simpler (my understanding is that this helps proving upper bounding suboptimality gap in general f-divergences)?

### Soundness
3

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
This paper extends the entropy regularized softmax PG in Mei et al. 2020b to the a general regularization based on the coupled parameterization. Convergence guarantees are also established.

### Strengths
Extension of entropy regularized softmax PG.

### Weaknesses
* Presentation can be improved. In particular, more discussion of the theoretical results will be helpful for the reader to get a better understanding since there are quite a lot of parameters involved. 
* More numerical experiments can be conducted, even though the authors argue that the goal is to verify the effectiveness of Tsallis divergence. It is known that (entropy regularized) softmax PG is highly inefficient compared to (entropy regularized) NPG due to that the appearance of the policy in the exponential term may cause a mislead. I am wondering whether the new algorithm based on Tsallis divergence will be efficient than NPG or not. At least, tests for the exact setting can be conducted. 
* References are missing. For example in "Elementary analysis of policy gradient methods" by Liu et al 2024, it is shown that softmax PG (without regularization) can achieve sublinear convergence for ANY constant step size though the problem dependent constant still exists.

### Questions
* In Mei et al. 2020b, there are exists sublinear convergence result for the non-regularized softmax PG. As far as I can see, the authors only extend the regularized counterpart. Is it right? Even though the sample complexity can be obtained for the non-reguarlized case by choosing the parameter carefully, the sublinear convergence result in the exact setting for the non-regularized case cannot be obtained from the linear result for the regularized case. Isn't it?
* A major theoretical contribution claimed by the authors is that there is no problem dependent hidden in the convergence rate. Is it fully due to the projection operator or also relies the the particular divergence? Will it also be helpful for removing the constant in the sublinear convergence of the non-regularized softmax PG?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This submission derives fast rates (i.e., $\tilde{O}(\epsilon^{-1})$) for learning against the $f$-divergence regularized objective in discounted MDPs, which is achieved in a batch-online manner. In the algorithm design, the *log-linear policy with a reference model $\pi^\text{ref}$* under KL-regularization is extended to the so-called "coupled parametrization" in this submission. Finally, the submission claims a separation result between Tsallis-entropy regularization and entropy regularization by comparing *two upper bounds*.

### Strengths
- This is the first work deriving fast rates $\tilde{O}(\epsilon^{-1})$ for learning w.r.t. divergence-regularized objectives in the discounted settting.
- The proofs are correct.

### Weaknesses
> **To AC: the reviewer is willing and ready to further discuss about this submission with AC or even SAC if needed.**

- Every time $[f']^{-1}$ appears in this submission, it is **mathematically wrong**: because, for example, chi-square divergence is already nice enough, but the $a \mapsto \max\{0, a\}$ (also appears as ReLU in the lierature) in the Lemma G.2 of [10] already certificates that it is not possible to simply subsume the solution to this constrained optimization problem using $[f']^{-1}(\cdot)$ and the complementary slackness condition might need to be tackled in a case-by-case way.

- The smoothness property (i.e., the Hessian spectrum upper bound) is trivial and does not even deserve a theorem, i.e., Theorem 4.3, because it is well known that the Fenchel conjugate of a $\alpha$-strongly-convex function is $\alpha^{-1}$-smooth (under very mild qualitative regularity conditions); which is nearly exactly the case for the regularized objective with equation (9) as the parametrization
- The artificial assumptions from Line 201 to Line 210 are far from enlighting and even **known to be** highly unnecessary for important cases like reverse-KL regularization and chi-square divergence regularization: 
  1. to be concrete, the reviewer **strongly disagrees** with Line 215-216 (the sentence around "which prevents") because the analysis of learning w.r.t. the reverse-KL-regularized objectives have become sharp (in terms of the dependency on $\epsilon$, i.e., $\tilde{O}(\epsilon^{-1})$) for contextual bandits and episodic MDPs in the hybrid setting [6, 1], for contextual bandits in the offline setting [9,7], and for both contextual bandits and episodic MDPs in the online setting [8].
  2. Even without Assumption P, and even in the pure offline setting without exploration, the analysis of learning against many divergence regularized objectives is doable, as manifested in the [7]
- The term "coupled parametrization" is confusing because it is just an extension of the "log-linear policy *with a reference policy*" to the general $f$-divergence setting, see, e.g., Definition 1.1 in [1]
- The PL condition, i.e., the "essentially strongly concave" property of the regularized objective **should not appear as a brandly new contribution** in this submission because it has been presented in a more minimalist setting in the offline contextual bandits setting in [7]


- If the authors do plan to claim the separaion between Tsallis regularization and vanlla entropy regularization for learning against the unregularized objective at the end of Section 5, they should not compare two upper bounds.
- The study of learning w.r.t. divergence-regularized value functions and objectives dates back to a long line a previous effors **no later than** [2, 3], at least in the episodic MDP setting. And **divergence-regularized performance difference lemma** appeared **no latter than** Section 5 in [8] and Lemma 3 in [5]
  - If the authors do consider their analysis is totally unrelated to the divergence-regularized performance difference lemma (or the so-called soft peformance difference lemma) in the literature, they should justify it **technically instead of secretly**.
  - Also, the **divergence-regularied Bellman operator** has been illustrated in detail in both [3] and [5], which is certainly not a "newly introduced" concept in this submission.
- The reviewer does not want to claim that the authors are **plagiarizing or rephrasing previous works in a more involved way**, but the authors should respect the previous efforts in the theory community in a decent way.
  - If the authors do consider the discounted setting in this submission is fundamentally different from the finite-horizon episodic MDP settings or the contextual bandit settings considered in [1-10] in the literature and does not plan to discuss the relation between this submission and any of [1-10], they should justify it **technically instead of secretly**.
  - **To AC: the reviewer is willing to discuss about this point if necessary.**


References

[1] Foster, Dylan J., Zakaria Mhammedi, and Dhruv Rohatgi. "Is a Good Foundation Necessary for Efficient Reinforcement Learning? The Computational Role of the Base Model in Exploration." arXiv preprint arXiv:2503.07453 (2025).


[2] Xiong, Wei, et al. "Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint." arXiv preprint arXiv:2312.11456 (2023).

[3] Xie, Tengyang, et al. "Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf." arXiv preprint arXiv:2405.21046 (2024).

[4] Huang, Jiawei, et al. "Can rlhf be more efficient with imperfect reward models? a policy coverage perspective." arXiv preprint arXiv:2502.19255 (2025).

[5] Yuan, Yurun, et al. "Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning." arXiv preprint arXiv:2505.15311 (2025).


[6] Zhao, Heyang, et al. "Sharp analysis for kl-regularized contextual bandits and rlhf." arXiv preprint arXiv:2411.04625 (2024).

[7] Zhao, Qingyue, et al. "Towards a Sharp Analysis of Offline Policy Learning for $ f $-Divergence-Regularized Contextual Bandits." arXiv preprint arXiv:2502.06051 (2025).

[8] Zhao, Heyang, et al. "Logarithmic regret for online kl-regularized reinforcement learning." arXiv preprint arXiv:2502.07460 (2025).


[9] Aminian, Gholamali, et al. "Theoretical Analysis of KL-regularized RLHF with Multiple Reference Models." arXiv preprint arXiv:2502.01203 (2025).


[10] Huang, Audrey, et al. "Is best-of-n the best of them? coverage, scaling, and optimality in inference-time alignment." arXiv preprint arXiv:2503.21878 (2025).

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
1
