# Best-of-Both-Worlds Policy Optimization for CMDPs with Bandit Feedback

- Avg Score: 3.60
- Decision: Reject
- Scores: 3, 3, 3, 6, 3

## Abstract
We study online learning in constrained Markov decision processes (CMDPs) in which rewards and constraints may be either stochastic or adversarial. In such settings, Stradi et al. (2024b) proposed the first best-of-both-worlds algorithm able to seamlessly handle stochastic and adversarial constraints, achieving optimal regret and constraint violation bounds in both cases. This algorithm suffers from two major drawbacks. First, it only works under full feedback, which severely limits its applicability in practice. Moreover, it relies on optimizing over the space of occupancy measures, which requires solving convex optimization problems, an highly inefficient task. In this paper, we provide the first best-of-both-worlds algorithm for CMDPs with bandit feedback. Specifically, when the constraints are stochastic, the algorithm achieves $\widetilde{\mathcal{O}}(\sqrt{T})$ regret and constraint violation, while, when they are adversarial, it attains $\widetilde{\mathcal{O}}(\sqrt{T})$ constraint violation and a tight fraction of the optimal reward. Moreover, our algorithm is based on a policy optimization approach, which is much more efficient than occupancy-measure-based methods.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studied the constrained Markov decision process (CMDP) problem where the transition kernel is stochastic unknown, and the reward and costs could be adversarial and stochastic settings. The paper leverages the techniques from the existing literature to address the trade-off between rewards and constraint violations. The theoretical analysis is provided to justify the proposed algorithm.

### Strengths
The proposed algorithm achieves some interesting results, such as sublinear regret and constraint violation for stochastic settings and a constant competitive ratio for adversarial settings.

### Weaknesses
1. The paper studied the Constrained Markov Decision Process (CMDP) problem in both stochastic and adversarial settings. However, it heavily relies on the framework established by previous works (e.g., policy optimization in Luo et al., 2021; primal-dual design and analysis in Castiglioni et al., 2022b; Stradi et al., 2024b). This combination seems not interesting and the technical contribution of this paper is somewhat limited for a theory paper, especially considering the extensive literature on CMDPs, both in theory and algorithmic sides.

2. The metric of cumulative constraint violation is problematic in safety-critical applications in the paper because it may have limitations since it allows for compensation among different rounds. 

3. The proposed method adopts a model-based approach and is limited to tabular settings. It might not be scalable to the setting with a large state and action space. 

4. The paper primarily focuses on theoretical aspects; however, it would be beneficial to include numerical experiments to validate the proposed algorithm. By comparing it with representative studies in the extensive literature, such as the LP-based method discussed in Efroni et al. (2020) and Stradi et al. (2024b), as well as a model-based RL method in reference [1] and a model-free RL method in reference [2], the practical performance of the algorithm can be more clearly demonstrated. BTW, [1] and [2] are very related work and not mentioned in the paper. 

[1] Tao Liu, Ruida Zhou, Dileep Kalathil, PR Kumar, and Chao Tian. Learning policies with zero or bounded constraint violation for constrained MDPs. NeurIPS 2021.

[2] Arnob Ghosh, Xingyu Zhou, and Ness Shroff. Provably efficient model-free constrained rl with linear function approximation. NeurIPS 2022.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper studies the problem of learning a constrained MDP under bandit feedback. The authors propose a policy-based algorithm to reach $\tilde{O}(T^{3/4})$ regret and violations. By assuming that the Lagrangian multiplier  is bounded by $\Lambda$, the regret and violation bounded could be improved to $\tilde{O}(\Lambda\sqrt{T})$.  In the meantime, the proposed algorithm could also achieve similar regret and violation bounds under the adversarial setting, which is referred as "best-of-both-worlds" property.

### Strengths
1. This paper considers the bandit feedback setting of learning a CMDP, which is the case of many practical problems..

2. The proposed regret\& violation bounds match the best previous result under full-information feedback.

### Weaknesses
1. The technical novelty is limited given the previous work (Stradi et. al., 2024). (Stradi et. all 2024) worked on the same problem with full-information feedback. As far as I can see, there is no substantial difficulty to combine their algorithm with classical estimators for online learning with bandit-feedback.

2. The key problem in (Stradi et. al., 2024) is not resolved (whether it is possible to remove the slater condition number in the violation and regret bound. In other words, whether a $\tilde{O}(\sqrt{T})$ regret & violation bound is achievable without further assumptions for the best-of-both-worlds setting). It is known (Jin et. al., 2020) that for reward-adv MDP without constraints, it is possible to reach an $\tilde(O)(\sqrt{T})$ regret, while for stochastic reward and constraints, there exists algorithm (Agrawal et. al., 2022) to reach $\tilde{O}(\sqrt{T})$ regret\& violation bound. So the question is that: is it possible to get an $\tilde{O}(\sqrt{T})$ regret (violation) bound for CMDP with adv reward (constraints)?

3. As claimed by the authors, one possible benefit by policy optimization is to avoid solving some convex optimization problem. But there is empirical results to demonstrate that the policy-based algorithm is more efficient than the occupancy-based algorithm.

References:

[Jin et. al., 2020] Learning Adversarial MDPs with Bandit Feedback and Unknown Transition

[Agrawal et. al., 2022] Regret Guarantees for Model-Based Reinforcement Learning with Long-Term Average Constraints

### Questions
I do not understand why the authors design  the algorithm based on policy optimization. It seems there is no substantial difficulty in applying the occupancy-based algorithm (Stradi et. al., 2024) with the same reward\&constraint estimators. Although the policy optimization algorithm seems more practical, it might leads to worse dependence on other parameters. Is there any potential advantages by applying policy optimization to solve CMDP?

Typo: it should be $\rho\geq T^{-1/8}H\sqrt{112m}$ in Condition 2?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes an algorithm for online CMDPs with bandit feedback under both stochastic and adversarial constraints settings. Applying an existing policy optimization algorithm to a primal-dual approach to policy optimization, the algorithm achieves $\sqrt{T}$ regret and constraint violation bounds for stochastic constraints under certain conditions, and a $\sqrt{T}$ condition violation under adversarial constraints.

### Strengths
The paper considers the CMDP setting with bandit feedback in both stochastic and adversarial settings, which is more practical. The authors provide theoretical guarantees on regret and constraint violation, with a specific analysis of the dependency on Slater’s condition. By focusing on policy space optimization rather than occupancy measures, the paper presents a more computationally efficient alternative for real-world applications where solving convex programs per episode is infeasible.

### Weaknesses
- The paper’s theoretical results rely heavily on Condition 2. As discussed in Section 2.4, $\rho \in [0, H]$; however, it is unclear how this condition could hold even when $T = 1$. Even setting aside this issue, if Condition 2 does hold, it implies that $\rho$ must be a relatively large value as $T$ increases, which is likely to be the case in online settings. According to Theorem 4, the Lagrangian multiplier is bounded by $c/\rho^2$, resulting in a very small constant close to zero. In such cases, the reward function becomes the dominant term, yielding a regret rate of $\sqrt{T}$.

- The theoretical analysis heavily relies on prior work by Stradi et al. (2024b) and Luo et al. (2021). 
- No simulations are included to support the efficiency claims made by the authors.

A lot of relevant papers on online CMDPs are missing:

[1] Arnob Ghosh, Xingyu Zhou, and Ness Shroff. Provably efficient model-free constrained rl with linear function approximation. Advances in Neural Information Processing Systems, 35:13303–13315, 2022

[2] Tao Liu, Ruida Zhou, Dileep Kalathil, Panganamala Kumar, and Chao Tian. Learning policies with zero or bounded constraint violation for constrained mdps. Advances in Neural Information Processing Systems, 34:17183–17193, 2021.

[3] Adrian Müller, Pragnya Alatur, Giorgia Ramponi, and Niao He. Cancellation-free regret bounds for lagrangian approaches in constrained markov decision processes. arXiv preprint arXiv:2306.07001, 2023.

[4] Adrian Müller, Pragnya Alatur, Volkan Cevher, Giorgia Ramponi, and Niao He. Truly no-regret learning in constrained mdps. arXiv preprint arXiv:2402.15776, 2024.

[5] Honghao Wei, Xin Liu, and Lei Ying. Triple-q: A model-free algorithm for constrained reinforcement learning with sublinear regret and zero constraint violation. In International Conference on Artificial Intelligence and Statistics, pages 3274–3307. PMLR, 2022.

[6] Qinbo Bai, Amrit Singh Bedi, Mridul Agarwal, Alec Koppel, and Vaneet Aggarwal. Achieving zero constraint violation for constrained reinforcement learning via primal-dual approach. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 3682–3689, 2022.

[7] Dongsheng Ding, Xiaohan Wei, Zhuoran Yang, Zhaoran Wang, and Mihailo Jovanovic. Provably efficient safe exploration via primal-dual policy optimization. In International conference on artificial intelligence and statistics, pages 3304–3312. PMLR, 2021.

[8] Dongsheng Ding, Chen-Yu Wei, Kaiqing Zhang, and Alejandro Ribeiro. Last-iterate convergent policy gradient primal-dual methods for constrained mdps. Advances in Neural Information Processing Systems, 36, 2024.

### Questions
If Condition 2 holds, then the regret and violation are $\Lambda \sqrt{T}$. According to Theorem 4, $\Lambda \leq \frac{c}{\rho^2} \leq \frac{c}{T^4}$. Does this mean that the regret and violation can achieve an order of $T^{1/4}$? How can this be true, as it would even better than the lower bound?

### Soundness
2

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
2

### Summary
The authors introduce an algorithm for online learning in CMDP with either stochastic and adversarial constraints while only requiring bandit feedback unlike in prior work. The resulting regret and constraint violation bounds matches prior work with full-feedback and retain the property of not requiring the knowledge of the Slater's parameter. Additionally, the authors show that using a weaker-baseline that only requires a policy to satisfy a constraint at each episode, the propose algorithm achieve sub-linear regret and constraint violations.

### Strengths
- The proposed method extends prior work to a more practical setting while retaining similar guarantees
- The authors do a good job in presenting the problem setting and algorithm

### Weaknesses
- It's unclear why is the useful to consider a weaker baseline since it is unclear what is the befit (from a practical standpoint) of only considering the optimal policy that  respect the constraints at each episode
- Having simple numerical experiments would help showcase the benefit of updating the primal based on a policy optimization approach vs the occupancy measure space.

### Questions
- Besides allowing the algorithm to be more efficient by, is there any theoretical benefit of conducting the analysis via an policy optimization approach. For example, could the analysis have been done using an occupancy-measure-based methods like in prior work?
- Spelling: 176 "ti" -> "it"

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors present an algorithm for adversarial CMDP. They showed that it achieves $\mathcal{O}(T^{3/4})$ regret and constraint violation both in the stochastic and adversarial setting with modified baselines.

### Strengths
1. Clarity on presentation.
2. Sound analysis.

### Weaknesses
**Minor Comments**

1. I think the definition of the occupancy measure is wrong. See the questions below for more details.

2. The bonus update seems ambiguous (lines 312, 313). In this update, both LHS and RHS have the term $B_t(x, a)$, and the only initial condition given is $B_t(x_H, a)=0$, $\forall a$ (this is mentioned in the algorithm, not in the main text). I guess the layered structure of the state space is essential to fill up all entries of $B_t(\cdot, \cdot)$ using the stated update. This should be emphasized in the main text for the reader's benefit. 

**Major Comments**

3. Theorem 3: In the worst case, the loss range upper bound can be an increasing function of $T$. Hence, in general, the regret upper bound may no longer be $\mathcal{O}(\sqrt{T})$, and thus the algorithm FS-PODB cannot be claimed to be no-regret.

4.  Condition 2 holds for small $T$. Therefore, Theorem 5, 7 and 8 also hold for small $T$. I think the authors are misusing the $\mathcal{O}(\cdot)$ notations in these theorems. Note that a $\mathcal{O}(\sqrt{T})$ regret should mean that the algorithm incurs at most $k\sqrt{T}$ regret for all $T\geq T_0$ where $k, T_0$ are some constants. This is clearly not true for Theorem 5, 7 and 8. 

5. In the stochastic setting, the regret obtained by the authors for large $T$ is $\mathcal{O}(T^{3/4})$ (Theorem 6). This should be the only result reported by the authors in the stochastic setting, and Theorem 5 should be discarded. The introduction and abstract should be changed accordingly. Surely, $\mathcal{O}(\sqrt{T})$ should not be highlighted as the primary result since it might mislead the readers. In the above light, the result of the authors seems weaker than Stradi et. al. (2024b). The claims accordingly have to be weakened in the introduction.

6. Similarly, Theorem 7, 8 should be discarded and only Theorem 9 should be kept in the adversarial setting with modified baseline. Since Theorem 7 was one of the main contributions of this paper, I am not sure what contribution of the paper remains.

7. The authors did not discuss the computational complexity of their algorithm in each iteration, although it was once of the main motivation mentioned in the introduction. In order to compute this complexity, the authors should take $S^H$ as the size of the state space since a general state space may not have a layered structure. The authors should compare their computational complexity with that of the earlier papers which they claimed to have more difficult problems to solve in each iteration.

### Questions
1. (Line 144): It seems like the occupancy measure, $q^{P, \pi}$ should be a function of $h$. However, it certainly is not. Am I missing something? Should there be an averaging over $h$ instead?

2. I see that the regret is computed using the optimal policy in hindsight. However, I believe a more sensible approach should have been to compute the optimal policy for each episode separately and then use that to compute regret. In summary, instead of using $T\mathrm{Opt}\_{\bar{r}, \bar{G}}$ as the baseline in regret computation, the term $\sum_{t} \mathrm{Opt}\_{r_t, G_t}$ should have been used. Any comment on why the second approach is not used?

3. Theorem 7: Can the roles of regret and constraint violation be reversed? In other words, is it possible to derive a $\mathcal{O}(\sqrt{T})$ regret bound while the constraints are satisfied up to some fractions?

4. Section 4.2.1: Shouldn't $\mathrm{Opt}^{\mathcal{W}}$ be a function of $t$ (since $G_t$ is used in the optimization instead of its time averaged value)?

### Soundness
3

### Presentation
3

### Contribution
2
