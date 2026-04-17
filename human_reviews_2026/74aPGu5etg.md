# Follow-the-Perturbed-Leader for Decoupled Bandits: Best-of-Both-Worlds and Practicality

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
We study the decoupled multi-armed bandit (MAB) problem, where the learner selects one arm for exploration and one arm for exploitation in each round. The loss of the explored arm is observed but not counted, while the loss of the exploited arm is incurred without being observed. We propose a policy within the Follow-the-Perturbed-Leader (FTPL) framework using Pareto perturbations. Our policy achieves (near-)optimal regret regardless of the environment, i.e., Best-of-Both-Worlds (BOBW): constant regret in the stochastic regime, improving upon the optimal bound of the standard MABs, and minimax optimal regret in the adversarial regime. Moreover, the practicality of our policy stems from avoiding both the convex optimization step required by the previous BOBW policy, Decoupled-Tsallis-INF (Rouyer & Seldin, 2020), and the resampling step that is typically necessary in FTPL. Consequently, it achieves substantial computational improvement, about $20$ times faster than Decoupled-Tsallis-INF, while also demonstrating better empirical performance in both regimes. Finally, we empirically show that our approach outperforms a pure exploration policy, and that naively combining a pure exploration with a standard exploitation policy is suboptimal.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the decoupled MAB problem, where an action is used for exploration while and another one is used to exploit. The authors show that a particular instantiation of the FTPL algorithm may attain best-of-both-worlds guarantees while being computationally more efficient than the state-of-the-art best-of-both-worlds algorithm for the same setting, which instead is based on FTRL.

### Strengths
The techniques employed in the work are interesting and the results seem both rigorous and novel. Overall, I believe that the results are well presented. Specifically, the authors did a good job in providing a precise overview of the state-of-the-arts techniques for the problem studied. Moreover, the authors put a lot of effort into explaining the main idea behind the proofs and the techniques employed. On the technical side, I believe this is a good paper.

### Weaknesses
The main weakness is the significance of the results obtained. Indeed, this paper focuses on a really specific topic, that is, uncoupled MAB, in which the bounds attained by FTPL do not improve the state-of-the-arts (and optimal) ones. Thus, to me, the contribution mainly lies in avoiding the convex optimization step of FTRL which I believe is not enough to meet the acceptance bar.

### Questions
Can the authors elaborate on the contributions of the works? I am glad to increase my score if the answers turn out to be pretty positive.

### Soundness
3

### Presentation
3

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
In this paper, authors study the best-of-both-worlds problem for decoupled multi-armed bandits under the Follow-the-Perturbed-Leader (FTPL) framework. Compared with previous FTRL-based algorithms which require solving a convex optimization problem at each round, the proposed algorithm is computationally efficient and obtains near-optimal results in both stochastic and adversarial regimes. Authors also conduct experiments to show that the proposed algorithm indeed outperforms Decoupled-Tsallis-INF algorithm in terms of both regret and running time.

### Strengths
- This paper is known as the first to study the best-of-both-worlds problem for decoupled multi-armed bandits under the Follow-the-Perturbed-Leader (FTPL) framework. The studied problem is well-motivated, given the computational issue of FTRL-based algorithms.

- The proposed algorithm is simple and intuitive. The algorithm adopts Pareto perturbations, used in previous FTPL work, to the decoupled bandit problem.

- Authors also empirically show the superiority in some experiments in terms of the regret performance and running time.

### Weaknesses
I have two major concerns.

- In this paper, the regret bound in the stochastic setting is $O(\sqrt{\frac{K}{\Delta_{\min} } \sum_{i \neq i^\*} \frac{1}{\Delta_i} } +\frac{K}{\Delta_{\min}})$, which is worse than the best-known result $O( \sqrt{  \sum_{i \neq i^\* } \frac{1}{\Delta_i^2} } +K )$ by [*]. For example, if the dominant term is $1/\Delta_{\min}$ (i.e., $\Delta_{\min}$ is sufficiently small) and all other arm gaps are constant level, say $\Delta_i=0.5$. In this case, the bound in this paper is dominated by $O(\frac{K}{\Delta_{\min}})$, but their bound is $O(\frac{1}{\Delta_{\min}})$. In other words, their bound is $K$ times smaller than the bound in this paper.

- The technical contribution is limited. For example, Pareto perturbation or its generalization has been widely applied in FTPL literature. While this paper is the first to use it in the decoupled multi-armed bandit problem, the techniques mostly follow previous work. The key difference that the proposed algorithm needs not to do geometric resampling is attributed to the decoupled setting, in which the algorithm can sample and weight by a self‑chosen probability distribution.

[*] Tiancheng Jin, Junyan Liu, and Haipeng Luo. Improved Best-of-Both-Worlds guarantees for multiarmed
bandits: FTRL with general regularizers and multiple optimal arms.

### Questions
Do authors believe that arm-dependent learning rates will improve the regret also for FTPL-based algorithms? If so, can authors provide a short and intuitive discussion on which part in your current analysis can be refined via using arm-dependent learning rates.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the decoupled bandit problem, a relatively less-studied topic within the bandit literature that possesses non-trivial application value. Specifically, the authors propose a policy based on the Follow-the-Perturbed-Leader (FTPL) framework using Pareto perturbations. The paper presents theoretical results for two cases: stochastic and adversarial environments. The new algorithm offers practical value, as it does not require solving optimization problems, unlike other methods. It also provides theoretical contributions, such as achieving a worst-case optimal regret bound in the adversarial setting. Experiments are provided to demonstrate the algorithm's effectiveness, though they utilize a limited choice of baselines.

### Strengths
1. The paper provides a comprehensive comparison with prior work and a sufficient background for the problem.
2. The algorithm demonstrates non-trivial improvements over existing methods.

### Weaknesses
1. The paper could benefit from adhering more strictly to writing conventions; notations are frequently used before they are defined (e.g., $w_t$ on line 136). It is strongly recommended to define mathematical notations before or as they are introduced in the same sentence to improve readability.
2. The analysis for the adversarial environment relies on a strong constraint, assuming that the gaps are constant.

### Questions
1. The authors comment that their problem is different from a pure exploration problem. Typically, an anytime pure exploration framework can be viewed as a process at each step involving: (1) selecting an arm, (2) pulling the arm and receiving a reward, and (3) recommending the arm currently inferred to be best. Note that the recommended arm (3) is not necessarily the same as the pulled arm (1), and it is only recommended, not pulled. I wonder what the precise difference is between this pure exploration setup and the problem addressed in this paper if we pull the arm in (3).

2. Could the authors compare the performance of their proposed algorithm with methods from the pure exploration, such as the Sequential Halving algorithm presented in "Revisiting simple regret: Fast rates for returning a good arm" (ICML 2023)? which is anytime. 

3. Regarding the baseline EB-TC, how is the exploitation arm chosen? Is it the same as the arm that is sampled? This needs clarification.

4. It seems the confidence level for the experimental results are missing.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper revisits the decoupled bandit problem and proposes a computational efficient BOBW algorithm via FTPL, where the computational complexity improves by a $K$ factor.

### Strengths
1. The writing of this paper is clear and easy to follow.
2. The proposed algorithm enjoys an efficient computation complexity, while still achieving the best-of-both-world guarantee.

### Weaknesses
1. The primary concern of the reviewer is that the contribution and novelty of this paper are too thin to be accepted. Only one computational improvement is proposed, and the design and analysis of the FTPL technique largely follow prior works. The reviewer suggests that the authors consider extending their FTPL technique to a broader range of problems, especially more challenging setups, e.g., combinatorial bandits, rather than the simplified MAB setup in the current paper. 
2. Several places in this paper are unclear:
    - The “$20$ times faster” claim is not clear.  Does that mean computational complexity, or just from the empirical experiments?
    - The “adaptive adversary” in Line 128 is an unclear term. It has two possibilities: strong (in response to current and past actions) and medium (only in response to past actions).
    - What is $\sigma_{t,i}$ in Eq.(7)?

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
1
