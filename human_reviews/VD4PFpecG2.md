# Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Risk-sensitive linear quadratic regulator is one of the most fundamental problems in risk-sensitive optimal control. 
In this paper, we study online adaptive control of risk-sensitive linear quadratic regulator in the finite horizon episodic setting. 
We propose a simple least-squares greedy algorithm and show that it achieves $\widetilde{\mathcal{O}}(\log N)$ regret under a specific identifiability assumption, where $N$ is the total number of episodes. If the identifiability assumption is not satisfied, we propose incorporating exploration noise into the least-squares-based algorithm, resulting in an algorithm with $\widetilde{\mathcal{O}}(\sqrt{N})$ regret. 
To our best knowledge, this is the first set of regret bounds for episodic risk-sensitive linear quadratic regulator. 
Our proof relies on perturbation analysis of less-standard Riccati equations for risk-sensitive linear quadratic control, and a delicate analysis of the loss in the risk-sensitive performance criterion due to applying the suboptimal controller in the online learning process.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a least-squares greedy algorithm for the online adaptive control of the risk-sensitive linear quadratic regulator (LQR) in a finite-horizon, episodic setting, which achieves a regret bound of $O(\log n)$ under identifiability assumption. It also proposes adding exploration noise to the least-squares algorithm when the identifiability assumption does not hold, resulting in a modified approach with a regret bound of $O(\sqrt{n})$.

### Strengths
- The authors introduce two algorithms based on least-squares methods tailored for LEQR. The first least-squares greedy algorithm without exploration noise achieves a logarithmic regret bound $O(\log n)$ under an identifiability condition, and the second algorithm, designed for scenarios where the identifiability assumption does not hold, achieves a $O(\sqrt{n})$ regret bound.
- This work provides the first regret bounds for finite-horizon episodic LEQR, showing the existence of logarithmic regret under identifiability assumption.
- The theoretical analysis relies on perturbation analysis of special Riccati equations for the risk-sensitive LEQR setting, where additional matrices ($\widetilde P$) are present in the Riccati equations due to the risk sensitivity. The authors show nontrivial analysis techniques addressing the challenges in the analysis.
- The authors also introduce novel analysis on risk-sensitive loss.

### Weaknesses
- It would be beneficial to establish matching lower bounds for both algorithms to provide a more comprehensive understanding of their optimality.
- Typo/Inconsistency of the plot descriptions: Regret performance of Algorithm 1 (System 1) vs Effect of the risk parameter (Algorithm 1 System 1)

### Questions
- As the authors noted, studying regret bounds for LEQR in the infinite-horizon average-reward (non-episodic) setting would be interesting. What challenges arise in extending the current proof methods to this setting?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies an online linear quadratic regulator (LQR) problem with risk-sensitive cost functions. In particular, when the environment can provide enough exploration noise, a least-square model fitting algorithm is provided and it achieves log N regret. When the environment is not "noisy" enough, the authors present a gaussian exploration method that achieves square root N regret. Numerical experiments are provided to validate the proposed algorithms.

### Strengths
The paper is well-written and easy to follow. All the assumptions are well-motivated and standard in the literature. The proposed algorithms are simple and can be easily implemented by practitioners. 

----

### Weaknesses
*1.* I think the authors should provide an explicit dependence of gamma in the two theorems, i.e., the big O should contain gamma orders, as gamma is the key in the studied risk-sensitive LQR problems. With gamma dependence in the results, the authors could compare the regret order with risk-neural MDP/LQR problems. Discussions/guidance on gamma selections could be included.

*2.* Literature/discussions on risk-averse RL/LQR problems are limited, in particular, the choice of risk-averse metrics and the reason for studying the LEQR problem.

*3.* The authors could provide a more interesting real-world control problem in the simulation. 

----

### Questions
*1.* Do the two theorems hold when gamma goes to 0, i.e., the limit is right-continuous in gamma? 

*2.* Typo: line 218-219, "We" play.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the online adaptive control of risk-sensitive Linear Quadratic Regulator (LQR) problem in a finite-horizon episodic setting, which is referred to as the Linear Exponential-of-Quadratic Regulator (LEQR) problem. The goal is to design the online algorithm that selects control that minimizes the total regret, i.e., the difference between the sum of minimum exponential risk-sensitive cost (defined in Eq. 2) and the algorithm's cost in each episode.

The authors introduce two algorithms: a least-squares greedy algorithm (Algorithm 1) that achieves $\tilde{O}(\log N)$ regret under identifiability condition (Assumption 1) and a second algorithm (Algorithm 2: Least-Squares-Based Algorithm with Exploration Noise) that incorporates exploration noise to achieve $\tilde{O}(\sqrt{N​})$ regret without the identifiability assumption. Finally, the authors have shown the different performance aspects of the proposed algorithms on two synthetic LEQR systems.

### Strengths
**The following are the strengths of the paper:**
1. This paper is the first to provide regret bounds for the episodic finite-horizon LEQR problem, which has applications in risk-sensitive control problems in areas like finance and healthcare.

2. The authors proposed two algorithms with sub-linear regret bounds guarantees. The regret bounds are derived using perturbation analysis of modified Riccati equations, which incorporate exponential risk-sensitive cost (defined in Eq. 2).

3. Finally, the authors have demonstrated the different performance aspects (sub-linear regret and effect of risk parameter and horizon on regret) of the proposed algorithms on two synthetic LEQR systems.

### Weaknesses
**The following are the weaknesses of the paper:**
1. Since verifying the identifiability assumption (Assumption 1) for a given problem may not be possible, the first algorithm may not be useful in practice. 

2. Both proposed algorithms are restricted to fixed finite-horizon settings and linear dynamics, which limits their real-world application, where horizon length can vary across episodes and problems with non-linear dynamics.

3. The following parts of the paper are not clear enough:
    - Why are the $N$ episodes in Algorithm 1 divided into epochs of increasing lengths to estimate the system matrices?
    - Why does the paper only consider the exponential risk-sensitive cost? Are there any real-world motivating examples for this choice?

### Questions
Please address the weaknesses of the paper. I have a few more questions/comments:
1. Instead of using exponential noise, can a UCB (upper confidence bound)-based exploration strategy in Algorithm 2?

2. How to extend these results for other risk measures, e.g., coherent risk measures (VaR, CVaR, EVaR, etc.) as studies in the following paper:
	1. Lam et al., [Risk-Aware Reinforcement Learning with Coherent Risk Measures and Non-linear Function Approximation](https://openreview.net/pdf?id=-RwZOVybbj)

3. What is the overall dependence of regret bounds given in Theorem 1 and 2 on horizon length ($T$) and risk-sensitivity parameter ($\gamma$)?


**Minor comment:**

Adding the layman's explanation and consequences of Assumption 1 will help readers identify when this assumption can hold in practice. Also, details (in the appendix and refer to the main paper) can added on how the standard Riccati equations are modified to incorporate exponential risk-sensitive costs.

I am open to changing my score based on the authors' responses.

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
4

### Summary
This paper considers the risk-sensitive LQR problem with unknown system matrices. The authors consider the standard LEQR formulation to capture the risk-aware scenario in the control problem and study the episodic problem setting where the system needs to be reset to the initial condition at the end of each episode. The authors propose online learning algorithms to solve the problem and show that the regret of the algorithm is bounded as \sqrt{N}, where N is the number of episodes in the problem. Under some additional assumption, the authors show that the regret bound becomes \logN.

### Strengths
1.	This paper initiate the consideration of risk-sensitive formulation of LQR in an online episodic setting.
2.	The paper rigorously characterize the regret bounds of the proposed algorithms.

### Weaknesses
1. Although the authors mentioned a previous work Basei et al. 2022 that considers an episodic control problem, it would be good to further explain why it is important/of practical interest to consider the episodic setting in the control problem. In addition, it would be good to discuss what the major challenge is if we move to the non-episodic setting which is the most standard setting in control problems (i.e., the system involves continuously and does not reset).
2. It would be good to explain more how the LEQR formulation captures the risk-aware scenario in Section 2.1. The authors also mention that their analysis extends to \gamma<0. What does \gamma<0 stand for in the risk-aware scenario? In general, it would be good to provide an intuitive explanation of how different values of γ (positive, negative, or zero) correspond to different risk attitudes. This would help readers better understand the practical implications of the LEQR formulation.
3. The authors should justify their choice to use only data from epoch l-1 for estimation in Algorithm 1, and discuss whether using data from multiple previous epochs could improve accuracy. Additionally, they should clarify whether the i.i.d. assumption on initial states is necessary for their results to hold.
4. The optimal controller based on the initial estimate \theta^1 needs to satisfy Assumption~1 as well in Theorem 1. Could you justify this assumption on \theta^1, i.e., how do you choose \theta^1 so that the assumption is satisfied? 
5. Regarding the regret bounds provided in Theorems~1 and 2, they depend exponentially on the horizon length T for each episode. However, in episodic MDP problems, the regret bound depends on T only polynomially (e.g., https://proceedings.mlr.press/v70/azar17a/azar17a.pdf). Could the authors comment on this issue:  whether this is fundamental to the risk-sensitive setting or if it's an artifact of the regret analysis? Moreover, would it be possible to improve this dependence to polynomial in T?
6. Could you explain why the least squares approach in Algoirthm~2 needs a regularization term?
7. Please explicitly specify how the initial estimate \theta^1 is chosen in Algorithm~2.

### Questions
Please refer to Weakness.

### Soundness
3

### Presentation
2

### Contribution
2
