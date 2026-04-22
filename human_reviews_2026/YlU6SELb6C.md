# Feasible Policy Optimization for Safe Reinforcement Learning

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 8, 4

## Abstract
Policy gradient methods serve as a cornerstone of reinforcement learning (RL), yet their extension to safe RL, where policies must strictly satisfy safety constraints, remains challenging. While existing methods enforce constraints in every policy update, we demonstrate that this is unnecessarily conservative. Instead, each update only needs to progressively expand the feasible region while improving the value function. Our proposed algorithm, namely feasible policy optimization (FPO), simultaneously achieves both objectives by solving a region-wise policy optimization problem. Specifically, FPO maximizes the value function inside the feasible region and minimizes the feasibility function outside it. We prove that these two sub-problems share a common optimal solution, which is obtained based on a tight bound we derive on the constraint decay function. Extensive experiments on the Safety-Gymnasium benchmark show that FPO achieves excellent constraint satisfaction while maintaining competitive task performance, striking a favorable balance between safety and return compared to state-of-the-art safe RL algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper talks about an efficient approach where, instead of enforcing the constraint satisfaction explicitly after every policy update, it talks about maximizing the Value function in the present feasible domain and then progressively expanding the feasible domain.
In particular, the paper solves the problem in two ways. At first, they solve 
$\max_{\pi} \mathbb{E}_{x \sim d_{init}}\left[ \mathbb{I}\left[{F^{\pi_{k}} \leq 0}\right]V^{\pi}(x)\right]$
s.t. $ \max_{\pi} \mathbb{E}_{x \sim d_{init}}\left[ \mathbb{I}\left[{F^{\pi_{k}} \leq 0}\right]F^{\pi}(x)\right] \leq 0$

Simultaneously, they aim to expand the feasible range by solving the following optimization problem.
$\max_{\pi} \mathbb{E}_{x \sim d_{init}}\left[ \mathbb{I}\left[{F^{\pi_{k}} > 0}\right]F^{\pi}(x)\right]$
s.t. $ \max_{\pi} \mathbb{E}_{x \sim d_{init}}\left[ \mathbb{I}\left[{F^{\pi_{k}} \leq 0}\right]F^{\pi}(x)\right] \leq 0$

They show that the two problems have a common optimization solution and further finds the feasibility bounds of their solution.

### Strengths
1. The Constraint enforcement at every step is shown to be overly conservative and not necessary
2. Talks about a Feasible Policy Optimization algorithm which aims to get the safe policy at every instant
3. At each step it can guarantee policy improvement almost surely.
4. Evaluated on some standard Safe RL benchmarks.

### Weaknesses
1. The writing was little hard to follow with a lot of minor gaps in equations for example, In equation 11 $k$,  should come in the subscript for the equations. $d^{\pi_{k}}$ and not $d^\pi k$ and  if $A^{\pi_{k}}(x,u)$ was introduced

2. The reachability set is little too stringent. I get that it is required for the proofs but it created confusions too. For example. In Theorem 1 the paper claims that if $\pi_{k+1}$ gets a value in $$ \mathcal{R}^{\pi_{in}(\mathcal{X}_{init} \cap X^{\pi_{k}}) $$, $\pi_{k+1}$ would be replaced by $\pi_{in}$. This need not be true always. This is because $\mathcal{R}$ is a reability state. And according to definition of $\mathcal{R}$ introduced in the paper, it might be possible that $\pi_{k+1}$ and $\pi_{k}$ can have some overlapping reachability sets but it does not necessarily means that it will switch to some other policy. 

3. Over usage of $c$, used as cost as well as $\mathbb{I}\left[ h(x) > 0\right]$ for indicator constraint violation. Is the constraint violation same as cost or not. These small details made it little hard to follow,

### Questions
1. The reachability set is little too stringent. I get that it is required for the proofs but it created confusions too. For example. In Theorem 1 the paper claims that if $\pi_{k+1}$ gets a value in $$ \mathcal{R}^{\pi_{in}(\mathcal{X}_{init} \cap X^{\pi_{k}}) $$, $\pi_{k+1}$ would be replaced by $\pi_{in}$. This need not be true always. This is because $\mathcal{R}$ is a reability state. And according to definition of $\mathcal{R}$ introduced in the paper, it might be possible that $\pi_{k+1}$ and $\pi_{k}$ can have some overlapping reachability sets but it does not necessarily means that it will switch to some other policy. It would be helpful if this is explicitly clarified.

2. In Figure 1, we observe that the return and cost of FPO are the best, which is great. I am just curious to know that only FPO has the lowest uncertainty range in terms of cost. But its uncertainty range in terms of return is a little higher. So, if we run for long enough as you have for your experiments, why is the uncertainty range for other algorithms, such as PPO-Lag and FOCOPS, so vast? Because for PPO-Lag, there is a possibility that it can go less than 0.08 in cost while giving better value return. For example, if you check the Point Button plot, PPO-lag has supremely outperformed FPO.

3. Also curious how the given FPO algorithm would perform compared to CRPO

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
3

### Summary
The paper proposes a new policy optimization scheme for safe reinforcement learning based on limiting trajectory-wise constraint violations rather than expected constraint violations. The authors develop a dynamic programming method to estimate policy feasibility and combine it with PPO, achieving a scalable RL algorithm that maintains high rewards while avoiding unsafe states.

### Strengths
- Novel safety framework introducing the constraint decay function $ F $  
- Focus on trajectory-wise feasibility rather than expected feasibility (even though $ F $ is estimated in expectation, it measures violations more directly)  
- Clear writing and presentation  
- Excellent Figure 1 (should be standard for safe RL)

### Weaknesses
- More recent baselines would strengthen the empirical comparison (beyond Ji et al., 2023)  
- Curiosity: isn’t $ F $ typically underestimated due to truncated trajectories? Could this affect feasibility estimation?

### Questions
- See weak points.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose feasible policy optimization to avoid over-conservatism in safe RL problem. Specifically, it progressively increases feasible region when updating the value function by maximizing the value function in feasible region and minimizing the feasibility function outside. The experiments on safety gymnasium benchmark show that it achieves better balance between reward and cost constraint.

### Strengths
- The idea of this paper, i.e., update feasibility function and value function, is clearly presented.
- The topic of this paper is to the interest of safe RL community.

### Weaknesses
From the experiment (e.g, in fig.2), it is hard to say FPO is superior to baselines.
- First, the baselines in some tasks are not well set up. For example, PointCircle task is obviously much easier than PointButton and CarGoal. However, the experiments show that PPO-Lag, RCPO, FOCOPS cannot learn a relatively safe policy but they can perform relatively well on PointButton or CarGoal. Actually, I believe the authors did not even try to set a reasonable lagrangian coefficient for them on PointCircle because they perform just like unconstrained RL. As a reference, PPO-Lag can learn well on PointCircle in other library [1].
- Second, the authors claim they follow the original hyperparameter in Omnisafe. However, Omnisafe tuned the hyperparameter for safety constraint $=25$ (and the results show that the baselines can indeed make cost smaller than 25) while this paper uses cost limit $=0$. 
- Consider the performances of three hardest tasks (PointButton, CarGoal, CarPush), FPO performs similarly to PPO-Lag and RCPO: FPO has lower reward and a little lower cost.
- The velocity tasks are MUCH easier than navigation tasks in terms of reward-cost trade-off [2]. Meanwhile, the baselines on swimmervelocity is absolutely not well tuned. The ppo-lag and FOCOPS perform well in [1].

### Questions
- The authors use $F^\pi(x) \leq 0$ to denote the feasibility region but also use $F=E_\tau[\gamma^{N(\tau)}]$ which is $\geq 0$. So what's the physical meaning of $F^\pi(x) < 0$?
- The compared baselines (e.g., Lagrangian-based) can learn different policies w.r.t different constraint limits. How does FPO adjust to different safety preference?

[1] https://fsrl.readthedocs.io/en/latest/tutorials/benchmark.html

[2] https://safety-gymnasium.readthedocs.io/en/latest/environments/safe_velocity.html#costs

### Soundness
2

### Presentation
2

### Contribution
2
