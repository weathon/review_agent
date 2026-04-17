# AEGIS: Almost Surely Safe Offline Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Ensuring safety guarantees in offline reinforcement learning remains challenging, especially when safety constraints must hold almost surely, i.e., along every possible trajectory. Moreover, as pre-specifying a single safety budget (constraint threshold) is often challenging, it is desirable to learn a foundation policy that can be deployed across a broad range of budgets. We introduce AEGIS (Almost-Sure Epigraph-Guided Implicit Safety), an almost surely safe offline RL framework that can guide diffusion policy training via critics that respect constraints across all feasible budgets. AEGIS characterizes the feasible set of initial state-budget pairs as the epigraph of a feasibility critic updated via the worst-case backup. Building on the proposed characterization, we extend Implicit Q-Learning (IQL) to train both feasibility and reward critics. We use these critics to bias a diffusion policy toward high-value feasible actions. Consequently, AEGIS turns diffusion from a generative prior into a safety-aware controller, enabling a single general policy to respect various budgets without further tuning. Empirical results on the DSRL benchmark and humanoid locomotion tasks show that AEGIS achieves high feasibility with competitive returns, generalizing across feasible constraint thresholds.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the problem of learning an almost surely safe policy that can be deployed to tasks with varying safety budgets from offline data. The authors propose an framework called AEGIS that augments the state with the remaining safety budget to deal with varying budgets, and learns a worst-case feasibility critic to ensure almost safety. The algorithm learns a diffusion policy guided by both feasibility and reward critics. Experimental results on DSRL benchmark and humanoid locomotion tasks show that AEGIS achieves high feasibility with competitive returns.

### Strengths
1. The proposed methods are theoretically sound, supported by theorems with proofs.
2. The writing of the paper is clear.

### Weaknesses
1. The contribution of this paper is unclear. This claimed contribution is learning a policy from offline data that (1) is almost surely safe, and (2) can handle varying budgets. However, learning an almost surely safe policy is already achieved by Sootla et al. [1], and methods for handling varying budgets are already proposed by Lee et al. [2] and Guo et al. [3].
2. The novelty of this paper is unclear. Many of the proposed concepts and methods are similar to existing works. The feasible set defined in Section 3.1 is similar to the concept of maximum feasible region in safe control and safe RL (see [4]). The feasibility value function defined in Eq. (4) is similar to the optimal cost value function in standard safe RL. The feasibility Bellman operator defined in Section 3.3 is essentially a robust Bellman operator (see [5]). The expectile regression for value function learning is a widely used technique in offline RL [6].
3. The necessity of the reward penalty is a flaw of the proposed algorithm. This penalty should be sufficiently large to ensure safety, which seems difficult to choose in practice.

[1] Sootla, A., Cowen-Rivers, A. I., Jafferjee, T., Wang, Z., Mguni, D. H., Wang, J., & Ammar, H. (2022, June). Sauté rl: Almost surely safe reinforcement learning using state augmentation. In *International Conference on Machine Learning* (pp. 20423-20443). PMLR.

[2] Lee, J., Heo, J., Kim, D., Lee, G., & Oh, S. (2023, October). Dual variable actor-critic for adaptive safe reinforcement learning. In *2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 7568-7573). IEEE.

[3] Guo, Z., Zhou, W., Wang, S., & Li, W. (2025, March). Constraint-conditioned actor-critic for offline safe reinforcement learning. In *The Thirteenth International Conference on Learning Representations*.

[4] Yang, Y., Zheng, Z., Li, S. E., Duan, J., Liu, J., Zhan, X., & Zhang, Y. Q. (2023). Feasible policy iteration. *arXiv preprint arXiv:2304.08845*.

[5] Li, Z., Hu, C., Wang, Y., Yang, Y., & Li, S. E. (2024). Safe reinforcement learning with dual robustness. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

[6] Kostrikov, I., Nair, A., & Levine, S. (2021). Offline reinforcement learning with implicit q-learning. *arXiv preprint arXiv:2110.06169*.

### Questions
1. What is the contribution of this paper with regard to existing works, e.g., [1-3]?
2. What is the novelty of the proposed methods (as detailed in Weakness 2) with regard to existing works?
3. How to choose the reward penalty in practice, and what is its impact on the performance?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Almost-Sure Epigraph-Guided Implicit Safety (AEGIS), an almost surely safe offline RL framework based on Sauté RL. AEGIS considers both state and cost budget as the epigraph and designs a feasibility critic based on feasibility bellman operator. To approximate estimate the worst-case backup, AEGIS applies expectile regression inspired by IQL. Finally, a diffusion model guided by the feasibility critic is trained as a generative policy to generate high-value feasible actions for various budgets. AEGIS achieves better feasibility rate and constrained reward return compared to some baselines on some tasks.

### Strengths
This paper guarantees almost surely safety guarantee on discounted constraints.

This paper enables this guarantee across all feasible budgets through the framework of Sauté RL

The use of IQL to approximate feasibility Bellman operator is well-motivated.

This paper tests their method on a high-dimensional humanoid locomotion task, which is closer to real world cases.

### Weaknesses
### Weakness on problem setup

1. Why do you consider the discounted value as the original constraint, considering that the discount factor is just used as an approximation of non-discounted value to ensure the convergence of RL?
   1. Sauté -RL [1] formulates in this way while Paper [2] on Sauté -RL regards the non-discounted value as the original constraint in Sauté RL.
   2. However, real-world constraints are most non-discounted. For example,
      1. Constrain the cumulative cost of a company in the next year to be smaller than a specific value.
      2. Constrain the carbon emissions from the power grid in the next year to be smaller than a specific value.
   3. The discount factor is a way to approximate the original constraint to make RL method converge. Thus existing methods will transform the original constraint to discounted constraint in the code.
      1. This issue is also explained detailed in C2IQL [3].
   4. However, in this paper, the constraint is originally defined as discounted and all experiments are evaluated on discounted value. 
      1. Why does this paper formulated in this way?
      2. Why not estimate on discounted value and test on non-discounted value since real-world constraints tend to be non-discounted like all previous works? 
      3. Is there any gap or any influence?

[1] Sootla, A., Cowen-Rivers, A. I., Jafferjee, T., Wang, Z., Mguni, D. H., Wang, J., & Ammar, H. (2022, June). Sauté rl: Almost surely safe reinforcement learning using state augmentation. In *International Conference on Machine Learning* (pp. 20423-20443). PMLR.

[2] Castellano, A., Min, H., Mallada, E., & Bazerque, J. A. (2022, May). Reinforcement learning with almost sure constraints. In *Learning for Dynamics and Control Conference* (pp. 559-570). PMLR.

[3] Zifan, L. I. U., Li, X., & Zhang, J. C2IQL: Constraint-Conditioned Implicit Q-learning for Safe Offline Reinforcement Learning. In *Forty-second International Conference on Machine Learning*.

### Weakness on Experiments

1. Benchmarks are incomplete:
   1. For SafetyGymnasium, it is acceptable to select 4 tasks due to the large number of tasks.
   2. However, the following benchmarks are incomplete:
      1. Bullet-Safety-Gym contains 8 tasks while only 4 tasks are tested.
      2. Velocity dataset contains 5 tasks while only 2 are tested.
   3. The reason why all tasks in these benchmarks should be tested are:
      1. In one benchmark, different tasks reflects different feature or faces different difficulties. It is necessary to test them all to show the generality of proposed method.
      2. Thus, most existing papers (except for some early articles) test all tasks if corresponding benchmarks are selected.
      3. Test all tasks in one benchmark is more fair because it can eliminate the suspicion of selecting suitable environments.

2. Baselines are a little old and need to be updated. 
   1. The baselines are mainly from 2019-2023, which are too old since there are lots of method proposed within 2 years.

   2. Here are some related methods
      1. Methods related to generative model and feasibility:
         1. FISOR [4], which focus on feasibility and hard constraint on zero-cost.
         2. OASIS [5], which focus on generating feasible trajectories.

      2. Methods related to broad range budgets:
         1. CAPS [6] 
         2. CCAC [7] 

   3. Why CDT is modified to $\gamma$-CDT? 
      1. CDT is designed to non-discounted cost via bypassing the Bellman backup with discount factor. This makes CDT closer to real-world non-discounted constraint.
      2. How about inputting the discounted value as non-discounted value to CDT directly?

3. It is suggested to test the performance on non-discounted value as all previous works did.
   1. It is unrealistic to regard discounted values as constraints considering real-world constraints are most non-discounted.

   2. No matter in RL or Safe RL, the objective is always maximizing non-discounted cumulative rewards or constraining non-discounted cumulative costs.
      1. The discount factor is merely a compromise made by RL for convergence and stability.

      2. In RL, even if the value function estimates discounted cumulative rewards, the final performance is still evaluated on non-discounted cumulative rewards.

4. While this paper explained why new metrics are proposed, it is not clear on comparison and evaluation. Here are some suggestions.
   1. The primary evaluation criteria on normalized costs and rewards should also be included:
      1. It can more intuitively reflect the performance of the algorithm.
      2. It can be cross-validated with the results from other papers.
   2. More explanation on FR and CRR should be included:
      1. What is the meaning of "rate of generated trajectories"
         1. Methods like BCQ-Lag, CDT do not generate any trajectory
      2. Why you utilize FR and CRR? What is the advantage? 
         1. For example: 
            1. why this paper calculates discounted value for CRR?
               1. why not non-discounted value?
            2. why this paper only focus on the performance of feasible trajectories?

5. An ablation study on applying AEGIS to different policy structures is needed to demonstrate
   1. Why is diffusion policy the most suitable policy?
   2. The problem of other policy and the reason for the problem.
   3. Is AEGIS generalizable to different policy structures? 

[4] Zheng, Y., Li, J., Yu, D., Yang, Y., Li, S. E., Zhan, X., & Liu, J. (2024). Safe offline reinforcement learning with feasibility-guided diffusion model. *arXiv preprint arXiv:2401.10700*.

[5] Yao, Y., Cen, Z., Ding, W., Lin, H., Liu, S., Zhang, T., ... & Zhao, D. (2024). Oasis: Conditional distribution shaping for offline safe reinforcement learning. *Advances in Neural Information Processing Systems*, *37*, 78451-78478.

[6] Chemingui, Y., Deshwal, A., Wei, H., Fern, A., & Doppa, J. (2025, April). Constraint-adaptive policy switching for offline safe reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 39, No. 15, pp. 15722-15730).

[7] Guo, Z., Zhou, W., Wang, S., & Li, W. (2025, March). Constraint-conditioned actor-critic for offline safe reinforcement learning. In *The Thirteenth International Conference on Learning Representations*.

### Questions
Please refer to Weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies almost sure safety in offline RL. The authors consider the cost budget-augmented MDP, which augments the state $s$ to $\tilde{s}\doteq(s,\delta)$, and derive the feasibility V and Q value function along with the Bellman operator. In short, the feasibility value function is used to measure whether an augmented state (or state-action pair) is feasible to satisfy the constraint. The authors propose to use expectile regression to estimate the feasibility value and further employ a diffusion model as final policy. The experiment on offline safe RL benchmark shows that the proposed AEGIS method exceeds baselines in terms of almost sure safety.

### Strengths
- The paper is clearly written. The preliminaries and related background are sufficiently introduced.
- Overall the method is novel. The use of feasibility function converts the trajectory-wise safety constraint satisfaction to transition-level value estimation. The use of expectile regression makes it a good estimation according to the infimum in definition.
- The experiment shows that AEGIS has better almost sure safety.

### Weaknesses
- While most contents of methodology (sec.4.1~4.3) focus on feasibility and reward value function, the final policy uses a diffusion model. I believe the key contribution of this paper is on the value function part instead of a diffusion policy. Therefore, it is necessary to run an ablation of removing diffusion policy. Additionally, it will make the performance comparison in table 1 more fair as most baselines are not using diffusion policy.
- In table 1, though AEGIS achieves higher feasibility rate, the constrained reward return is much lower than many baselines. In other words, the policy of AEGIS is safer but also more conservative. In this case, it's hard to say AEGIS is better especially many baselines are not designed for almost sure safety setting.

Minor issues:
- line 69, 18%p -> 18%

### Questions
- Do you use the same or different hyperparameters (e.g., reward penalty n, $\tau$, and others in diffusion model) for each task?
- The reward value function takes $\delta$ as input. However, when training the reward value functions, AEGIS uniformly samples one $\delta$. Why do you use uniform sampling? Does this mean for each state-action pair $(s,a)$ the Q or V value will only be trained on one $\delta$?

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
2

### Summary
The paper proposes AEGIS, an offline safe RL method aiming at almost-sure safety across safety budgets. The key idea is to view the feasible state-budget pairs as augmented states to show that there exists an optimal policy for the feasible set, and feasible augmented states are the epigraph of a feasibility critic that can be learned offline via an expectile-based Bellman update. Then a diffusion policy is guided by both this feasibility critic an a reward critic, so that sampled actions are within budget and high-value. Empirical studies on DSRL and a humanoid task report higher feasibility rates than several baselines.

### Strengths
1. The paper addresses almost-sure constraint, a harder notion of safety, in offline RL, which is under-explored.
2. The epigraph construction of state-budget pairs offers a clean perspective of tackling the problem with Bellman operators.

### Weaknesses
1. The authors repeatedly claims "almost-surely safe offline RL", yet the method approximates the worst-case with a finite-$\alpha$ expectile and a finite penalty. The experiments show many tasks with feasibility rates well below 1.0 (c.f. table 1). 
2. The feasibility operator is defined with an essential sup, but the practical algorithm instead adopts an expectile and then trains a diffusion policy with guidance. The text notes that the two methods collide as $\alpha\rightarrow 1$. However, there is no finite-sample guarantee that the implemented $\alpha$ and guidance deliver almost-sure feasibility.
3. The empirical results are thin and not compelling. Gains in feasibility rates often come with reduced constrained return.
4. The method relies on learning a worst-case feasibility critic from a fixed dataset, but offline data can rarely cover the unsafe tails needed for an ess sup backup. I am curious how would the author improve the robustness against narrow missspecified or narrow datasets?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
