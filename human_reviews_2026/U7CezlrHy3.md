# COP-Q: Safety-First Reinforcement Learning with Cholesky Ordered Projection

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Using uncertainty in Q-values to mitigate overestimation, enhance exploration, and ensure safety has proven effective in single-objective deep Q-learning. However, when learning vector-valued Q-functions for correlated goals, uncertainties become intertwined across objectives. Conventional approaches either treat uncertainty in each objective independently or collapse them into one dimension, often resulting in unstable learning, low sample efficiency, limited exploration, and particularly unsafe behaviours. To address these challenges, this study proposes Cholesky Ordered Projection Q-learning (COP-Q), a novel method that guides safety-first exploitation and exploration using full multi-objective uncertainty. We first propose generalized multi-objective confidence bounds via covariance matrix factorization. For priority-ordered objectives, such as in safety-critical or cost-constrained reinforcement learning, Cholesky factorization is employed to incorporate inter-objective covariance into confidence bounds in a conditionally sequential manner. The lower bound yields conservative temporal difference targets to reduce overestimation, while the upper bound assigns optimistic Q-values to promote exploration. COP-Q is evaluated on standard MuJoCo and velocity-constrained SafetyVelocity-v1 benchmarks, demonstrating robust safety performance and competitive total returns. The proposed method is compatible with various deep Q-learning frameworks with minimal computational overhead, making it practical for a wide range of multi-objective and constrained reinforcement learning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In multi-objective RL, uncertainties of Q functions for different objectives may become intertwined, introducing extra challenges. Existing methods treat uncertainty in each objective independently or collapse them into a scalarized dimension. In this work, the authors introduce Cholesky Ordered Projection Q-learning (COP-Q), using full multi-objective uncertainty by covariance matrix factorization. Extensive experiments on standard MuJoCo and velocity-constrained SafetyVelocity-v1 benchmarks, demonstrating the effectiveness of COP-Q.

### Strengths
- Figures 1 and 2 clearly show the insights and contributions of this work.

- It is interesting and novel to introduce Cholesky Ordered Projection into RL.

- Extensive experiments in both standard settings and safe settings show that COP-Q can handle different objectives in the same time.

### Weaknesses
- I'm curious that if the Scalarization weight is fixed, what is the major difference between multi-objective RL and single-objective RL (R = u^T r)?

- As mentioned in lines 146-149, the Scalarization weight u might be fixed or changed. However, I find that the main algorithm seems designed for fixed u, what about handling the changing u (if I have any misunderstanding, please point it out)?

- In Fig. 1, the authors mention that the uncertainty of Q total may be low, but the uncertainty of each Q value may be high. Are there any theoretical or experimental observations supporting this insight?

Overall, I think the idea of this work is novel, but there are still some concerns. I'd like to adjust my score if the authors can address my concerns.

### Questions
See weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper aims to quantify uncertainty in multi-objective reinforcement learning.
This is realized using Cholesky factorization in the multi-objective Q-space.
By having a richer representation of the uncertainty, the overall performance is slightly improved.

### Strengths
- The paper explicitly considers the uncertainty in the multi-objective Q-space.
- The approach follows a hierarchical schema, where certain objectives (e.g., safety) are prioritized over others.
- Experiments indicate a slight improvement over compared methods.

### Weaknesses
- The paper is extremely difficult to follow, in particular, Sec. 4.1:
The applied steps (Eq. 6-8) require more explanations, clearly motivating what the goal of the subsequent transformations is, and properly explaining all variables. e.g., $R$. Similar holds for Eq. (10), where $C_{clip} = CR$ is introduced to drop $R$, but never used again. One can assume that $C_{clip}$ became $L_{clip}$ through Sec. 4.2, but this connection can be made clearer, or avoid dropping R altogether.
- The paper also does not state clearly its assumptions / properly define the variables. For example, the paper says that (A.1) always holds if $C$ is symmetric. However, for $C=-I$, this clearly does not hold for any $u$, indicating that there are certain restrictions on which values $C$ can take.
- It is unclear what is meant by the priorization of the objectives in Sec. 4.2. Are these given by the scalarization vector $u$? 
- The term "uncertainty" is not well defined.

### Questions
- Can you explain the term "the level of optimism when facing uncertainty" more intuitively than simply stating (13)?
- Does it matter how $u$ is set to determine the priorization of the objectives and vice-versa? 
- Can you quantify the "minimal computational overhead" of your approach?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
COP-Q introduces a novel multi-objective Q-learning method leveraging Cholesky factorization to incorporate inter-objective covariance into uncertainty estimation. It prioritizes safety-critical objectives via ordered projection, yielding conservative TD targets for overestimation reduction and optimistic bounds for exploration. Evaluated on MuJoCo and SafetyVelocity-v1, COP-Q shows robust safety, competitive returns, and improved sample efficiency.

### Strengths
1. Innovative Uncertainty Modeling: First work to integrate Cholesky factorization into multi-objective Q-learning, capturing objective correlations and priorities (e.g., safety-first).

2. Strong Empirical Results: Outperforms baselines in safety-critical tasks while maintaining high returns.

3. Theoretical Soundness: Confidence bounds generalize clipped double-Q learning, with rigorous projection derivations.

### Weaknesses
1. Assumes Q-values follow multivariate Gaussian (Eq 5), but no ablation on its validity.

2. Limited Task Diversity: Experiments focus on locomotion; lacks MORL Pareto-frontier or high-dim task validation.

3. Variance in Exploration: REDQ+COP-OAC shows high variance in Humanoid (Fig 5), attributed to UTD ratio but unverified.

4. Sec 4.3 uses biased covariance estimator (denominator = N+1). Justify this choice.

5. Why there is no result about COP-Q-svc on halfcheetah experiment in Figure 4 top.

### Questions
Please refer to Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Cholesky Ordered Projection Q-learning (COP-Q) to enhance safety-first exploitation and exploration for vector-valued Q-functions. In particular, COP-Q first introduce the generalized multi-objective confidence bounds for Q-values and then employ Cholesky factorization to encodes full multi-objective uncertainty following the priority structure of objectives. COP-Q achieves good performance and training efficiency compared with existing baselines on MuJoCo benchmarks.

### Strengths
1. This paper introduces Cholesky Ordered Projection into Multi-Objective RL so that different objectives can be considered with different priority.
2. COP-Q shows good performance and training efficiency compared with existing baselines on MuJoCo benchmarks.

### Weaknesses
### Concerns on Mujoco  benchmarks
1. In Figure 4, the training curve in halfcheetah seems to be wrong.
   1. There are only two curves in the figure.
   2. The blue curve (COP-Q-vsc) on top is exactly same as Cholesky on bottom
      1. Since Cholesky is same as COP-Q-svc in other environments, there is something wrong in the figure.

2. For baseline in MuJoCo, why doesn't COP-Q compare with MORL methods, such as RMORL [1], PGMORL [2], MORL-Adaptation [3], MO-MPO [4].

[1] He, X., Hao, J., Chen, X., Wang, J., Ji, X., & Lv, C. (2024). Robust multiobjective reinforcement learning considering environmental uncertainties. *IEEE Transactions on Neural Networks and Learning Systems*, *36*(4), 6368-6382.

[2] Xu, J., Tian, Y., Ma, P., Rus, D., Sueda, S., & Matusik, W. (2020, November). Prediction-guided multi-objective reinforcement learning for continuous robot control. In *International conference on machine learning* (pp. 10607-10616). PMLR.

[3] Yang, R., Sun, X., & Narasimhan, K. (2019). A generalized algorithm for multi-objective reinforcement learning and policy adaptation. *Advances in neural information processing systems*, *32*.

[4] Abdolmaleki, A., Huang, S., Hasenclever, L., Neunert, M., Song, F., Zambelli, M., ... & Riedmiller, M. (2020, November). A distributional view on multi-objective policy optimization. In *International conference on machine learning* (pp. 11-22). PMLR.

### Concerns on Constrained benchmarks

1. While this paper compares on SafetyVelocity-v1, it may have following concerns
   1. The feasible set of SafetyVelocity-v1 is not tight. The relationship between reward and cost are more likely to be independent to each other.
      1. For example, the agent can achieve the maximum rewards on a wide range of costs. 
   2. This issue will simplify the testing cases.
2. Thus it is suggested to test on more benchmarks, whose feasible set is tighter.
   1. BulletSafetyGym:
      1. BallRun, BallCircle, DroneRun, DroneCircle, AntRun, AntCircle
   2. SafetyGymnasium:
      1. PointCircle1-v0, PointCircle2-v0, CarCircle1-v0, CarCircle2-v0, 
3. It is suggested to compare with more recent methods, such as RLSF [1]
4. What are the criteria for selecting the cost threshold?
   1. It is better to test on multiple thresholds across a wide range of costs.
5. The cost in Figure 6 and Table E.3 seems to be unmatched.
6. It is better to list the results of PPOSimmerPID and CUP in Table E.3.

[1] Reddy Chirra, S., Varakantham, P., & Paruchuri, P. (2024). Safety through feedback in Constrained RL. *Advances in Neural Information Processing Systems*, *37*, 139938-139967.

### Other Concerns

1. In most figures, some methods seems that it hasn't converged within setting episodes.
   1. For example, walker2d and ant in Figure 5.
   2. For example, most on-policy method in Figure 6.
   3. What is the performance after convergence?
   4. Does COP-Q performs better than these methods after convergence?

### Questions
Please refer to the Weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
3
