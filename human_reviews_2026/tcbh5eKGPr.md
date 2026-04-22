# Mind Your Entropy: From Maximum Entropy to Trajectory Entropy-Constrained RL

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Maximum entropy has become a mainstream off-policy reinforcement learning (RL) framework for balancing exploitation and exploration. However, two bottlenecks still limit further performance improvement: \textit{(1) non-stationary Q-value estimation} caused by jointly injecting entropy and updating its weighting parameter, i.e., temperature; and \textit{(2) short-sighted local entropy tuning} that adjusts temperature only according to the current single-step entropy, without considering the effect of cumulative entropy over time. In this paper, we extends maximum entropy framework by proposing a trajectory entropy-constrained reinforcement learning (TECRL) framework to address these two challenges. Within this framework, we first separately learn two Q-functions, one associated with reward and the other with entropy, ensuring clean and stable value targets unaffected by temperature updates. Then, the dedicated entropy Q-function,  explicitly quantifying the expected cumulative entropy, enables us to enforce a trajectory entropy constraint and consequently control the policy’s long-term stochasticity. Building on this TECRL framework, we develop a practical off-policy algorithm, DSAC-E, by extending the state-of-the-art distributional soft actor-critic with three refinements (DSAC-T). Empirical results on the OpenAI Gym benchmark demonstrate that our DSAC-E can achieve higher returns and better stability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Trajectory Entropy-Constrained Reinforcement Learning (TECRL), which augments maximum-entropy RL by introducing a separate entropy Q-function to estimate cumulative (trajectory-level) entropy and enforce an entropy constraint, thereby decoupling reward and entropy learning to mitigate non-stationarity from temperature updates. Built upon the DSAC framework, the method (DSAC-E) is evaluated on continuous-control benchmarks where it shows moderate improvements in stability and return.

### Strengths
(1) The method is motivated by a clear limitation in standard maximum-entropy RL: non-stationary targets when temperature is updated.

(2) The trajectory-entropy concept is intuitively appealing and could inspire alternative formulations of long-term exploration control.

(3) The proposal is practical and seems to fit within existing off‐policy actor‐critic frameworks, which aids reproducibility and relevance to applied RL.

(4) Empirical results appear to show meaningful improvement (both higher returns + improved stability) on commonly used benchmarks, which strengthens the claim of benefit.

### Weaknesses
(1) Omission of relevant baseline:
S²AC[1], a max-entropy RL method results in maximizing the expected future entropy without the need for tuning the temperature parameter (Figure 2). The policy is modeled as SVGD sampler. A sufficient number of steps leads to good convergence in bothe smooth and non-smooth landscapes.

(2) Evaluating the proposed method on the multi-goal environment used in S²AC [1] would strengthen the paper’s claims and help illustrate the intuition behind trajectory-level entropy control.

(3)  The contribution appears incremental relative to existing entropy-regularized RL frameworks.

(4) The write-up lacks some refinement: The text repeatedly refers to “non-stationary Q-values,” whereas the non-stationarity arises in the distribution induced by joint reward–entropy learning, not in the Q-values themselves. Also, the effect of the tempreture parameter on the actor stationarity and the short-sighted local entropy tuning only becomes clear in section 2, the introduction need to be adjusted to foster early understanding of these key concepts.


[1] Messaoud S, Mokeddem B, Xue Z, Pang L, An B, Chen H, Chawla S. S $^ 2$ AC: Energy-Based Reinforcement Learning with Stein Soft Actor Critic. ICLR, 2024.

### Questions
(1) How does your approach compare S²AC[1]?
(2) Could you evaluate your method on the multi-goal environment used in S²AC [1]?

### Soundness
2

### Presentation
2

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
The paper "Mind Your Entropy: From Maximum Entropy to Trajectory Entropy-constrained RL" identifies two bottlenecks in standard maximum-entropy RL and proposes a trajectory entropy-constrained RL framework to address them. The approach decouples the critic into a reward-centric and an entropy-centric critic to avoid non-stationary Q-value estimation caused by automatic temperature adjustment. The entropy critic further enables a trajectory-level definition of entropy. Experiments on several MuJoCo tasks show performance gains over baseline algorithms.

### Strengths
1. The idea of separating the critic into reward-centric and entropy-centric components is intuitive and conceptually clear.
2. The trajectory-level entropy constraint provides a novel perspective that could inspire future work on entropy-based control.
3. The empirical results show noticeable improvements on several MuJoCo environments compared to strong baselines.

### Weaknesses
- Unclear theoretical analysis:

The analysis in Subsection 3.3 is not sufficiently clear. It is not convincing that the trajectory entropy budget necessarily leads to a higher performance bound. The metric used to define the "performance bound" should be explicitly stated. Equation (18) alone does not imply that enforcing an entropy budget guarantees performance improvement (the inequality logic C <= A+B does not lead to C >= A).

- Inconsistent experimental results:

Some results differ from those reported in the original DSAC-T paper (e.g., Reacher-v2, where the original reports -3+/-0). The paper should clarify the source of these discrepancies and justify why only eight environments are selected for evaluation.

- Ambiguity in Equation (9):

The definition of H_budget in Eq. (9) is unclear. Please provide a precise explanation or derivation.

- Unexplained negative entropy budget:

Section 4.1 (“Our method”) mentions that $H_\theta$ is a negative value. The reasoning behind this choice should be discussed.

### Questions
Questions are already discussed in the Weaknesses section, but are listed below for clarity:

- In Subsection 3.3, how exactly does the trajectory entropy budget lead to a higher performance bound? Please clarify the theoretical connection and specify the metric defining "performance bound".

- In Eq. (9), how is 𝐻_budget formally defined or derived?

- Why is ​$H_\theta$ stated to be negative in Sec. 4.1?  

- Some experimental results (e.g., Reacher-v2) differ from the DSAC-T paper. Could you explain the source of these discrepancies and the reason for evaluating only eight environments?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Trajectory Entropy-Constrained RL (TECRL), which decouples reward and entropy learning by training two critics: a reward critic $Q_r$ and an entropy critic $Q_e$. The entropy critic estimates cumulative future entropy so the policy can be optimized under a trajectory-level entropy constraint. Building on DSAC-T, the authors instantiate DSAC-E and claim higher stability and returns on 8 MuJoCo tasks. They also present a simple performance-bound argument showing how choosing the entropy budget affects the attainable return, plus ablations for the two components (reward-entropy separation and the trajectory entropy constraint) and sensitivity to the entropy-budget scale $\rho$.

### Strengths
- The proposed technique is simple and clearly presented, with writing that is easy to follow.
- TECRL offers significant performance gains compared to the DSAC-T baseline, especially for certain control tasks such as Humanoid, Ant, and Walker2d.

### Weaknesses
- The method introduces an additional hyperparameter, $\rho$, which appears to require environment-specific tuning (e.g., $\rho=20$ for Humanoid/Walker2d vs. $\rho=1$ elsewhere), increasing tuning complexity. 

- The theoretical investigation is limited. The “performance bound” (Sec. 3.3) fixes $\alpha_\text{soft}^{\*}$ from the MaxEnt optimum and algebraically relates return to $\mathcal{H}^{\*}\_{\text{soft}} - \mathcal{H}\_{\text{budget}}$. However, it doesn’t yield a constructive guarantee for selecting $\rho$ or quantify optimality.

- Results are limited to classic MuJoCo tasks; there is no validation on harder/modern continuous-control suites or high-dimensional, contact-rich domains.

**Minor Error:**
- “we extends maximum entropy framework” → “we extend the maximum-entropy framework.” in Line 17.

### Questions
- In MaxEnt RL, the fixed-point iteration yields $\pi(a|s) \propto \exp(\frac{1}{\alpha} Q(s,a))$. Could the authors provide a theoretical justification for convergence under TECRL? Specifically: (i) under what conditions does $Q_e$ converge? (ii) what is the resulting closed-form (or fixed-point) characterization of the optimal $\pi$ in TECRL?
- $H_0$ in Line 260 appears undefined. Is it $\mathcal{H}_0$ introduced in Eq. (6)? Please standardize the notation.
- The choice of $\rho$ seems nontrivial and varies from 1 to 20 across tasks. When should a larger vs. smaller $\rho$ be preferred? Please provide a practical tuning guideline (e.g., ranges, scaling with horizon/entropy targets) and include analyses on additional environments to demonstrate robustness.
- To inform $\rho$ selection, could the authors report the policy’s action variance (or entropy) over training and at evaluation time, and relate these trends to returns and constraint satisfaction?
- Could the authors analyze compute? What’s the end-to-end throughput and wall-clock cost of adding $Q_e$ (forward/backward breakdown)?
- TECRL shows large gains on Humanoid, Ant, and Walker2d but only parity elsewhere. Which task properties (e.g., horizon, contact complexity, multimodality, reward shaping, exploration difficulty) drive the improvements? A brief failure-mode analysis would be helpful.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new framework, Trajectory Entropy-Constrained Reinforcement Learning (TECRL), to address two perceived "bottlenecks" in standard maximum entropy RL algorithms like Soft Actor-Critic (SAC). The identified bottlenecks are: (1) "non-stationary Q-value estimation," which the authors claim is caused by the temperature parameter alpha being updated simultaneously with the Q-function, thus destabilizing the Bellman target; and (2) "short-sighted local entropy tuning," which only constrains the current single-step entropy rather than the cumulative entropy over a whole trajectory.

To solve this, TECRL proposes two main changes, Reward-Entropy Separation (RES) and Trajectory Entropy Constraint (TEC). The authors instantiate this framework in a practical algorithm called DSAC-E, which builds on the state-of-the-art DSAC-T. Experiments on 8 MuJoCo tasks show that DSAC-E outperforms DSAC-T and other strong baselines.

### Strengths
- The idea of completely decoupling the reward and entropy value streams into two separate critics ( $Q_r$ and $Q_e$ ) is a novel and clean architectural approach.
- The proposed algorithm, DSAC-E, demonstrates state-of-the-art performance on a suite of standard MuJoCo benchmarks, consistently outperforming its direct predecessor, DSAC-T, as well as SAC and other baselines.
- The ablation study in Table 2 effectively isolates the performance contributions of the two main components (RES and TEC), showing that each one adds value to the final algorithm.

### Weaknesses
1. The paper's entire motivation rests on solving two "bottlenecks," but the justification for their existence and severity is weak and not supported by evidence.
- Non-stationary Q-value: The paper asserts that updating alpha makes the Q-target non-stationary. While true, this is a minor effect compared to the policy pi and Q-function Q themselves being updated, which is the primary source of non-stationarity in all bootstrapped RL. The paper provides no empirical evidence (e.g., plots of target value variance) to prove that the changing alpha is a significant source of instability that actually hinders performance in SOTA methods.
- Short-sighted local tuning: The paper claims local entropy tuning is a fundamental flaw, yet it provides no evidence that this mechanism is a primary bottleneck. The very success of SAC and DSAC-T, which use this "flawed" mechanism, suggests it is a highly effective and robust heuristic. The motivation feels more like a post-hoc justification for a new method rather than a response to a well-documented problem.

2. Limited Experimental Scope: The empirical validation is not general enough. All 8 environments are standard continuous control tasks from the MuJoCo/OpenAI Gym benchmark. While DSAC-E shows strong performance here, these tasks are very similar in nature (locomotion/simple manipulation). To make a convincing case for the general superiority of the TECRL framework, it would need to be validated on a more diverse set of domains, such as pixel-based control (e.g., DMControl), tasks with sparse rewards, or more complex robotics manipulation challenges.

### Questions
- Can you provide direct evidence (e.g., plots of Bellman error or target value variance over time) to support the claim that Q-value estimation is significantly non-stationary in SAC/DSAC-T (due to alpha) and that your RES method actually produces more stable Q-value estimates?
- The paper claims TEC allows for the "strategic distribution" of entropy. Can you provide an analysis (e.g., visualizing policy entropy in different parts of the state space) to show this is happening, rather than the mechanism simply finding a better average entropy level?
- Why was the experimental evaluation limited to MuJoCo? Have you considered testing the generality of TECRL on other domains, such as pixel-based tasks, where the exploration challenge is different?

### Soundness
3

### Presentation
3

### Contribution
1
