# Combinatorial Rising Bandits

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Combinatorial online learning is a fundamental task for selecting the optimal action (or super arm) as a combination of base arms in sequential interactions with systems providing stochastic rewards.
It is applicable to diverse domains such as robotics, social advertising, network routing, and recommendation systems.
In many real-world scenarios, we often encounter rising rewards, where playing a base arm not only provides an instantaneous reward but also contributes to the enhancement of future rewards, e.g., robots improving through practice and social influence strengthening in the history of successful recommendations.
Crucially, these enhancements may propagate to multiple super arms that share the same base arms, introducing dependencies beyond the scope of existing bandit models.
To address this gap, we introduce the Combinatorial Rising Bandit (CRB) framework and propose a provably efficient and empirically effective algorithm, Combinatorial Rising Upper Confidence Bound (CRUCB).
We empirically demonstrate the effectiveness of CRUCB in realistic deep reinforcement learning environments and synthetic settings, while our theoretical analysis establishes tight regret bounds. Together, they underscore the practical impact and theoretical rigor of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies combinatorial semi-bandits with rested rising rewards, a setting relevant to diverse application domains such as robotics and maintenance scheduling. To address this problem, the authors propose the CRUCB algorithm and provide corresponding regret guarantees. They further demonstrate the effectiveness of the method through empirical evaluations.

### Strengths
This paper is the first to consider combinatorial semi-bandits with rising rewards.

### Weaknesses
1. My main concern is the novelty of the proposed algorithm. The paper does not clearly identify what non-trivial challenges arise specifically from incorporating rising rewards into the combinatorial semi-bandit setting, nor how the approach fundamentally differs from existing methods for rising or combinatorial bandits.

2. It is unclear whether the presented regret bound is tight.

3. The role of window sizes and their effect on performance is not discussed. Providing guidance or theoretical justification for the choice of window size would strengthen the practical relevance of the method.

### Questions
1. Could the authors elaborate on the unique challenges of incorporating rising rewards into the combinatorial semi-bandit setting? It is not fully clear what makes this extension technically difficult compared to standard combinatorial or rising bandits.

2. The novelty of the proposed algorithm relative to prior work on rising bandits is not fully articulated. A more explicit comparison against existing methods (e.g., in the algorithm section) would better highlight the contribution.

3. The discussion on effective window size remains insufficient. In particular, the choices of window parameters for equations (10), (11), and (12) are not explained, and it is unclear how these influence theoretical guarantees or empirical performance.

4. Regarding the experimental setup:
   - How were the window sizes selected for the sliding-window type baselines?
   - How was  R-ed-UCB (with bandit feedback) adapted to the combinatorial semi-bandit feedback model?

### Soundness
2

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
3

### Summary
This work makes the first step to study the combinatorial rising bandit problem, where the mean of the chosen base arm in round $t$ will increase in round $t+1$. The authors propose the Combinatorial Rising UCB (CRUCB) algorithm, the core of which is to use a UCB constructed using the empirical mean from the latest sliding window, the exploration bonus, and the predicted improvement. Both regret upper bound and the lower bound are established for the proposed algorithm. Experiments on synthetic environments and deep reinforcement learning settings are conducted.

### Strengths
1. This seems to be the first work in the literature to model and resolve the combinatorial rising bandit problem.
2. The paper is generally well-written.

### Weaknesses
1. **Novelty**: Though the problem formulation is relatively new, the technical novelty seems somewhat limited. To me, the core part of the proposed algorithm is to use a UCB constructed using exploration bonus, the empirical mean, and the predicted improvement in the past sliding window, which is exactly the way [1] deals with rising multi-armed bandits (MABs). Of course, their method cannot deal with the combinatorial rising bandit problem. However, it is not clear to me whether there are additional technical difficulties when combining the method to deal with rising MABs and non-rising combinatorial bandits (say, [2]). As such, this prevents me from giving a positive rating for this work. And due to my relatively limited expertise in combinatorial bandits, I’d like to maintain a low level of confidence in my rating.

[1] Metelli et al. Stochastic Rising Bandits. ICML, 22.

[2] Chen et al. Combinatorial multi-armed bandit with general reward functions. NeurIPS, 16.

### Questions
1. As the reward of some chosen base arm $i$ in round $t$ will increase in round $t+1$, this problem seems a special case of adversarial bandits with an adaptive adversary. In this view and assuming $ r(S, \boldsymbol{\mu}) =\sum_{i\in S}\mu_i$, I think that the algorithms in previous works for adversarial combinatorial bandits with semi-bandit feedback (say, [3,4]) are applicable to this problem and that $O(\sqrt{T})$ regret upper bound is achievable. However, when $ f(n)=(n+1)^{-c}$ and $c=1.1$, this seems to contradict the lower bound of $\Omega(T^{2-c})=\Omega(T^{0.9})$. Do I miss something?

[3] Audibert et al. Regret in online combinatorial optimization. Mathematics of Operations Research, 13.

[4] Zimmert et al. Beating Stochastic and Adversarial Semi-bandits Optimally and Simultaneously. ICML, 19.

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
4

### Summary
This work studies a MAB setting where it combines two existing setting Combinatorial bandits (gets selected a set of arms) in each round and the arms also gets better in each round. The setting poses an interesting challenge where in which the optimal policy has to balance out with good early pieces and late bloomers. They propose a new UCB style algorithm, CRUCB that takes into account of future reward and constructing a upper bound for each arm and according selecting the best arm. They provide regret theoretical guarantees for this algorithm complemented by experiments on different scenarios.

### Strengths
The work introduces a new bandits setting that combines combinatorial and rising bandits. The setting is interesting and seems to have a wide range of application.

The authors proposed a new algorithm CRUCB for this setting showcase its performance with a regret upper bound in $K$ and $L$. 

The work also includes a lower bound to highlight the difficulty of this setting and to further complement the results of CRUCB's regret upper bound. 

The authors provide a good experimental setup to validate CRUCB in shortest planning task and hierarchical RL tasks comparing against baselines in rising and combinatorial baselines.

### Weaknesses
The Solver is treated as an oracle that exist to solve this combinatorial problem and cost associated is not discussed. 

Many combinatorial tasks in real world practice exhibits non-monotonic rewards but the work assumes $r$ as a monotonic function. The assumption of reward formulation being monotonic in $S$ \& $\mu$ limits its use case and generality.  

*also refer questions section.

### Questions
1. Does CRUCB suffer any extra cost from the Solver as compared to the greedy selection of arm  ? So, by treating Solver as a oracle that exist, is there any cost that is not accounted in the regret ?

2. In case, if the $\mu_i(n+1) - \mu_i(n) = 0$, Does the setting reduce to a Standard Combinatorial algorithm ? and how does the regret compare against other existing Combinatorial algorithms ? Can you provide which term in Theorem 3 is unavoidable in comparison to those algorithm ? 

3. Also, if each super arm is exclusively just the base arm and the $|S|$ = $K$, so does the setting reduces to a rising bandits then in that case as there is an assumption of concavity of $\mu_i$, how does CRUCB's regret compare against the rising bandits algorithm ? Does CRUCB recovers the regret for rising bandits ?
 
4. Many practical scenario involve piecewise rising (i.e. the reward rise with pulls, gets stagnant for a period and then rise), does violating Assumption 1 breaks the regret guarantee ?

5. Does Regret bound in Theorem 3 becomes sublinear if the rising effect did not become saturated ?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a combinatorial rising bandit algorithm that handles cases where the expected reward of each base arm increases over time in a combinatorial setting. The paper is theoretically solid, providing both regret lower and upper bounds based on the proposed UCB-based algorithm. Simulations and the shortest-path problem both demonstrate the performance of the proposed approach. Overall, I think this is a well-written paper with enough detail to support its claims.

### Strengths
1. The paper is well structured, with a clear flow and proper introduction of concepts, making it easy to follow even for readers who may not work directly in this area.
2. The theoretical results are sound. I think this problem is genuinely challenging given both the combinatorial nature and the non-stationary reward that changes as arms are pulled over time. The authors set a good pace to introduce the theoretical results.
3. The simulation part is clear and effectively supports the theoretical findings.

### Weaknesses
See Questions below.

### Questions
1. I’m curious whether the authors have considered extending the framework to a Thompson Sampling (TS)-based algorithm under this setting. I saw that the simulations include TS baselines. While I understand that CRUCB should perform better since it is specifically designed to handle both the combinatorial and rising-reward aspects, I’d be interested to hear how the authors view the challenges of extending this to a TS version, since TS often slightly outperforms UCB in many settings.
2. In Theorem 3, I don’t quite understand the meaning of q. Does the theorem hold for any q \in [0,1], or is it a pre-specified value determined from upstream? How is q decided in practice?
3. What is the general computational complexity of the algorithm? I’m curious which part dominates the computation time: is it the combinatorial optimization part (the solver in CRUCB) or the rising reward evaluation part (Future-UCB calculation)? Either a theoretical complexity order or an empirical runtime illustration would be helpful.
4. It would also be nice if the authors could provide more examples or motivating scenarios showing where combinatorial rising bandit problems are commonly encountered.

### Soundness
4

### Presentation
3

### Contribution
3
