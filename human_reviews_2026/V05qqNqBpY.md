# Finite-Time Analysis of Actor-Critic Methods with Deep Neural Network Approximation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Actor–critic (AC) algorithms underpin many of today’s most successful reinforcement learning (RL) applications, yet their finite-time convergence in realistic settings remains largely underexplored. Existing analyses often rely on oversimplified formulations and are largely confined to linear function approximation. In practice, however, nonlinear approximations with deep neural networks dominate AC implementations, leaving a substantial gap between theory and practice. In this work, we provide the first finite-time analysis of single-timescale AC with deep neural network approximation in continuous state-action spaces. In particular, we consider the challenging time-average reward setting, where one needs to simultaneously control three highly-coupled error terms including the reward error, the critic error, and the actor error. Our novel analysis is able to establish convergence to a stationary point at a rate $\widetilde{\mathcal{O}}(T^{-1/2})$, where $T$ denotes the total number of iterations, thereby providing theoretical grounding for widely used deep AC methods. We substantiate these theoretical guarantees with experiments that confirm the proven convergence rate and further demonstrate strong performance on MuJoCo benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper established a finite-time convergence result for a proposed DNN-based actor-critic reinforcement learning algorithm.

Specifically, the authors considered a challenging RL setting with 1) continuous state and action spaces, 2) Markovian samplings, and 3) average reward model. To address this problem, they developed a *single-timescale DNN-based* actor-critic algorithm, and, under some assumptions, proved a convergence rate of $\tilde{\mathcal{O}}(T^{-0.5})$. Finally, this paper presented some numerical results to corroborate their theoretical findings.

### Strengths
**The general RL setting is impressive.** As most works on RL theory focus on tackling the discount reward model and discrete (even if can be infinite) state and action spaces, this paper extends the results to a more general setup.

**The theoretical analysis is concrete.** Even though some of their techniques and analytical methods are similar to those found in existing research, the authors demonstrate commendable effort in handling a more practical and challenging scenario. I appreciate this theory-driven work, particularly in an era where empirical performance usually overshadows theoretical significance.

In addition, the paper is well-organized and the writing is pretty good.

### Weaknesses
My main concern is: **The contributions and insights are not highlighted.** For example, the authors claim that the "single-timescale" design is a primary contribution, yet their analysis lacks a discussion of the associated challenges and the techniques employed to resolve them. Likewise, for continuous spaces, the difficulties introduced by this setup and the authors' solutions remain unexplained.

While these points could represent the main contributions, the authors merely enumerate their findings without offering in-depth discussion or critical analysis.

Besides, I find some claims could be incorrect or inaccurate. (See Questions 2,4.)

### Questions
**Please try to answer the following questions:**

1. While the single-timescale method offers practical advantages, its convergence rate is inferior to some two-timescale algorithms [1,2]. This raises a question of potential tradeoff: could the noisier observations inherent in strongly coupled actor-critic methods contribute to this performance disparity?

2. I disagree with the claim regarding $m$-dependence in Lines 70~78. Numerous studies, including [3-4], present $m$-dependent (and depth-dependent) convergence results to emphasize the influence of DNNs. Actually, this paper's findings also connect to the width $m$, though it is implicitly contained within $\epsilon_{app}$. (To some extent, the authors are encouraged to better characterize the $\epsilon_{app}$ with the parameters of the DNNs.) Thus, I think the $m$-dependence is supposed to be a strength rather than a drawback.

3. Can the authors provide the key differences in the analysis among MLP, CNN and ResNet? It could have been interesting, but it is regrettable that the authors do not elaborate on it in sufficient detail.

4. The authors state that a "stationary policy" is optimal due to non-convexity. However, given that numerous studies demonstrate global convergence results [1,2,5,6], should this also be acknowledged by the authors?

5. The numerical experiments are kind of limited in two ways: 1) No other baselines are included; 2) The impact of depth and width remains unclear.

[1] Closing the gap: Achieving global convergence (last iterate) of actor-critic under markovian sampling with neural network parametrization. ICML’24.

[2] Finite-Time Global Optimality Convergence in Deep Neural Actor-Critic Methods for Decentralized Multi-Agent Reinforcement Learning. ICML’25.

[3] Neural Temporal-Difference Learning Converges to Global Optima. NeurIPS’19.

[4] Convergence of Actor-Critic Methods with Multi-Layer Neural Networks. NeurIPS’23.

[5] Sample and Communication-Efficient Decentralized Actor-Critic Algorithms. ICML’22.

[6] Improving sample complexity bounds for (natural) actor-critic algorithms. NeurIPS’20.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes an actor–critic algorithm with finite-time analysis under a neural network function approximation setting. Compared with previous studies, the paper establishes a sample convergence rate for environments with continuous action spaces. In addition, the authors present simulation results to demonstrate the effectiveness of the proposed approach.

### Strengths
The study derives theoretical guarantees by establishing the sample convergence rate of the Actor-Critic algorithm for MDP with continuous action spaces. 

In addition, the authors include simulation experiments that demonstrate the empirical validity of the theoretical results.

### Weaknesses
1. For Table 1, I am confused whether the comparison is fair, especially for the (Tian et.al. 2024). 
(a) Firstly, the sampling process is Markovian, both in Actor and Critic part. The restart setting is just to achieve different distribution for policy gradient under the discount finite horizon setting.  Please carefully check this part and correct the table.
(b) Besides, the width of the Neural Network is for the $\epsilon$-order approximation error of the value function. Your work choose to avoid this width but will lead to approximation error.   Finally, previous works with neural approximation will converge to $ \mathcal{O}(\epsilon)$ accurate set but your work will converge to $ \mathcal{O}(\epsilon+\epsilon_{approx})$. 

2. Compared with (Tian et.al. 2024) and other previous works, could the author detailed  explain where are the techique novelty or improvement. I went through the proof sketch but analysis looks like standard.

### Questions
See weaknesses, please.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides the first finite-time convergence analysis for single-timescale actor-critic (AC) algorithms utilizing deep neural network approximation in continuous state-action spaces under the time-average reward setting. The authors prove convergence to a stationary point at a rate of  $\tilde{O}({T}^{-1/2})$ for the coupled reward, critic, and actor errors. The theoretical claims are substantiated with experiments on the Pendulum task and MuJoCo benchmarks, demonstrating the superior approximation capability of neural critics over linear ones and empirically validating the predicted convergence rate.

### Strengths
1. This paper  provides a finite-time analysis for the challenging single-timescale neural AC setting, with continuous spaces and Markovian sampling, is a substantial theoretical advance.
2. The paper goes beyond pure theory by including comprehensive experiments. The empirical confirmation of the $\tilde{O}({T}^{-1/2})$ convergence rate on Pendulum and the demonstration of strong performance on MuJoCo benchmarks provide crucial support for the theoretical results and highlight the practical relevance of the analyzed algorithm.
3. The paper is well-structured and easy to follow.

### Weaknesses
The analysis operates in the neural tangent kernel (NTK) or overparameterized regime, where the network is wide enough to be well-approximated by its linearization around initialization. This regime, while theoretically fruitful, does not fully capture the feature learning dynamics that are believed to be crucial for the success of deep learning in practice.

### Questions
1. How is $m$ avoided in the convergence result? Does it depend on the assumption that the network is wide enough?
2. What is the convergence rate guarantee for the discounted reward setting, as it is more common in RL formulation?

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
The paper studies single-timescale actor--critic (AC) with deep neural network function approximation in continuous state--action spaces under the average-reward objective. It analyzes a practical AC loop that jointly updates a reward estimator, a TD(0)-style critic, and a policy-gradient actor with Markovian on-policy samples. The main theoretical result is a finite-time convergence guarantee to a stationary point at a $\tilde{O}(T^{-1/2})$ rate (up to logarithmic factors), simultaneously controlling reward-estimation, critic, and actor errors. Experiments (Pendulum, MuJoCo-style tasks) illustrate empirical trends, including a measured slope near $-1/2$ on log--log plots and improvements from neural critics over linear baselines.

### Strengths
1. Realistic setting: single-timescale updates with Markovian sampling in continuous spaces---closer to practice than idealized double-loop or two-timescale analyses.

2.  Finite-time $\tilde{O}(T^{-1/2})$ rate: matches the best known dependence on $T$ (up to logs) for this setting; jointly tracks three coupled sources of error.

3. Empirical checks: (i) Pendulum results where the neural critic better aligns with an RVI baseline than a linear/RBF critic; (ii) empirical slope $\approx -0.51$ consistent with theory; (iii) MuJoCo ablations show depth/width benefits over linear critics.

4. Assumptions documented and motivated: geometric mixing/ergodicity, an exploration inequality, and smoothness/Lipschitz properties for policy and dynamics, with discussion of when exploration can fail.

### Weaknesses
1.  The theory assumes sufficiently wide networks and projects critic updates to remain near initialization. It is unclear how necessary/tight this is or how it maps to common unconstrained training with Adam/weight decay.

2. Guarantees include an $O(\varepsilon_{\text{app}})$ term from critic approximation, but there is limited guidance for architectures/regularization that make $\varepsilon_{\text{app}}$ small in practice; experiments do not quantify this floor or test misspecification.

### Questions
Is projection onto a radius constraint essential for the analysis, or could similar guarantees hold for unconstrained (Adam/SGD) updates with weight decay?

### Soundness
3

### Presentation
3

### Contribution
3
