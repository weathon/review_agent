# AutoSafe: A Safe Neural Policy Network with Safety Common Sense

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Recognizing and avoiding danger is a fundamental capability of biological intelligence, yet this principle is less explored in the design of neural policies in today's artificial intelligence. We present AutoSafe, a novel architecture that embeds safety common sense directly into neural policies for safety sensitive applications. In particular, AutoSafe integrates a lightweight model-based Safety Evaluation Module that continuously evaluates the risk of safety violations and leverages a model-based safe policy as Safety Correction Module to correct potentially unsafe actions at runtime. By incorporating these two designs as part of the policy itself, AutoSafe can be seamlessly integrated into actor-critic based reinforcement learning algorithms to maximize performance while maintaining safety. We evaluate AutoSafe on a suite of continuous-control benchmarks, demonstrating that AutoSafe consistently outperforms other safe reinforcement learning baselines. Finally, we showcase the applicability of the proposed architecture in a real-world continual learning scenario on a cartpole system.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes AutoSafe, a safe reinforcement learning framework that blends a learned policy with a model-based safety controller. It constructs a safe action using a Lyapunov function and a safety envelope. The approach seems to yield a differentiable, end-to-end trainable architecture that automatically regulates the trade-off between performance and safety. The paper provides theoretical analysis alongside its experimental evaluation.

### Strengths
The strengths of this paper are:
- AutoSafe extends existing Lyapunov-based Safe RL methods in a somewhat novel way. The proposed ERA is novel and neat. While smooth blending between controllers is not new, the ERA’s specific exponential design and learnable temperature seem fresh in this contextual literature, and I appreciate its addition.
- Real-world experiment is a great addition!
- Multiple test-suite envs show good generalisability.

### Weaknesses
The weaknesses of this paper are:
- Correct me if I have misunderstood. The safety guarantee of AutoSafe holds if the system dynamics are known/partially known, the model mismatch ||w|| is uniformly bounded (under assumption), and the states are fully observable. For the experiments, how much of the systems dynamics was encoded as a priori? Does the uniform bound of ||w|| hold under real-world dynamics outside of controlled experiments?
- The paper never states how the safety violation is defined? Is it a priori safety constraint? Is it a terminal state?
- I feel like some baselines are missing or could be replaced. See references below, as some examples:
- - Approximate model-based shielding for safe reinforcement learning
- - End-to-End Safe Reinforcement Learning through Barrier Functions
- - Safe Model-based Reinforcement Learning with Stability Guarantees
- - etc
- Novelty seems incremental, and without comparison to stronger baselines, it is hard to gauge genuine strengths which is a shame because I enjoyed the proposed ERA and paper as a whole

### Questions
- Could the authors clarify how much prior model knowledge was used for each experimental domain?
- In the real-world robot setting, does the uniform bound on ||w|| realistically hold?
- Could the authors explain why the stronger baselines were excluded, or comment on how AutoSafe would perform relative to these methods (e.g., computational overhead, robustness, and safety tightness)?
- Could the authors elaborate on what aspects they consider fundamentally new relative to prior Lyapunov or barrier-function safety integration approaches outside of ERA?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents AutoSafe, a SafeRL method designed to train a policy to optimize an objective given constraints. 
AutoSafe builds on control theory for systems with linear dynamics and linear constraints to derive a model-based safety test and a safety policy. These are directly integrated in the policy learning through a smooth merging approach. It is shown for SAC in the paper. The method is tested on several environments and against several safe RL baselines, showing better performance. There is also a study about the temperature additional parameter $T$ which is learned.

### Strengths
The paper takes part in an important line of work. Building safe RL agents is an important challenge towards real world applications and works in this direction should be encouraged.
The article’s argument is well built. The method is sound under the stated hypotheses, as it allows agents to learn optimal policies while ensuring the respect of constraints. The paper showcases experiments on different environments, including a real world device, showcasing the performance of the method. There are also relevant additional results which allow to better understand the influence of temperature $T$. Finally, the paper is generally easy to read, with an apparent effort to convey the information clearly, although there is some missing information or confusing notation here and there.

### Weaknesses
There are three main weaknesses to the paper.

First, the approach is limited to systems for which (1) the dynamics are (approximately) known and (2) the dynamics and the constraints can be expressed in a linear manner so as to compute the exact solution for matrices K and P, needed for the safety-check and safe policy. In itself, this is not a problem because these assumptions can be considered true for several relevant applications. Yet, they should be explicitly stated in the introduction and abstract, as the proposed method does not solve the problem of Safe RL, but it proposes a solution for specific types of problems (known and linear dynamics, and linear constraints). It does not do it through “Safety Common Sense” but by benefitting from the specific assumptions of this type of problem. The paper would benefit from a less shiny, more rigorous packaging.

Second, the contribution itself is not clear to me. Overall, the proposed method is very, very similar to the general framework of (white-box) shielding [1,2]/provably safe RL [3] (specifically, the concept of Control Barrier Functions [4]), where prior knowledge of the world is used to determine if the action is safe and otherwise replace it with a safe action. AutoSafe is integrated into the policy and the policy parameters $\theta$ are learned accordingly, but this seems actually equivalent to the agent learning on an environment where the shield is deployed. The only different is the smoothening between the safe and the parameterized policy, but given a safe policy and a value of lambda (which is conditioned on s and stable once $T$ has converged), the parameterized can learn to compensate for the safe policy bias. The safe policy therefore only matters when $\lambda = 1$; and the approach is once again equivalent to white-box shielding. The fact that $T$ can be learned and change during the learning introduces weird learning dynamics but it is not clear how and why this would be beneficial.

The results show the approach to be better than white-box shielding (called “simplex”). I am not sure why this is the case. The authors refer to “non-stationarities” in the simplex learning, which I don’t understand (white-box shielding is stationary if the shield is conditioned on (s,a), as it does not change during learning). This is the last important weakness of the paper: there are not enough information about the simplex baseline (see questions to the authors) to correctly interpret the results.

[1] Könighofer et al. (2017) Shield synthesis

[2] Alshiekh et al. (2018) Safe reinforcement learning via shielding  (cited by the authors)

[3] Krasowski et al. (2023) Provably Safe Reinforcement Learning: Conceptual Analysis, Survey, and Benchmarking

[4] Hsu et al. (2023) The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems

### Questions
**Introduction:** 	
In the second paragraph, it is said that the issue with external safety layers is that they often over constrain the agent’s performance. This can be the case for learned safety layers. Well built shields can be less conservative while having full safety guarantees (a bit like what is done in this paper…) but they need prior knowledge of the environment. 

**Section 2.2**
Equation (3): is $\dot{s}$ the time derivative of $s$? While this is conventional in control and dynamics fields, it could be worth defining for the RL/ML community.

**Section 3.1**
Doesn’t the weighted summation approach assume that the action space is “ordered” (for a lack of better word)? In the sense that if $a_1 > a_2$, the effect on $s’ \sim P(s,a)$ is always in the same direction? More concretely: if in $a \in [-1,  0]$ you go to left (-1 max move, 0 no move) and $a \in [0, 1]$ you go right (0 no, 1 max), it works. If inversely between -1 and 0 you go to left, this time with (-1 no, 0 max) and between 0 to 1 you go right (0 more, 1 less), you have a discontinuity in the effect of the action on the environment, and this would completely fail (although it would be a valid MDP).

**Section 3.1.3**
Does $T$ depend on $s$? It should not as I understand it (should be an automatically adjusting hyperparameter of the learning algorithm), but here it is said that $T$ is an additional prediction head of the policy network, which is conditioned on $s$. 

**Section 4**
How does the integration of the safety policy into the hybrid policy compare with training with a shield which ensures the action is safe after it is predicted by the agent, and replaces it otherwise? Because $\pi_\theta$ can be learned and can compensate $\pi_{safe}$ in $\pi_\lambda$ for any $z(s) < 1$, I believe this is actually equivalent to a shield which would let $\pi_theta$ untouched until $z(s) = 1$ and then take over with $\pi_{safe}$.

**Section 5.2**
This is a positive comment: I really enjoyed this section!

**Section 6**
How is the runtime safety shielding (“Simplex”) generated? How is the safety checked, and which safety action is enforced

**Section 6.1**
I do not understand how the safe policy (for Simplex) intervenes in a “non-stationary” way. I also do not understand how this is different from Autosafe, except for the smoothening.

**Section 6.2**
Here, it seems $T$ is not state-based (see question on 3.1.3). Still, I don’t understand the analysis. As $\pi_\theta$ could completely absorb $\pi_safe$ for any $z(s) < 1$, I believe the value of $T$ at convergence should have no impact on the performance. Overall, this complexifies the learning process without a solid explanation for the better performance at convergence observed in the results. Could it be related to a smoother learning process, such as $\pi_{safe}$ guiding the learning at the beginning because it has a relatively good performance compared to random policies? 

**Section 6.3**
Given the assumptions over the systems for which Autosafe can be applied to, I think “holding the promise of bridging domain gaps such as sim-to-real and offline to online RL” is a bold claim. 


## Typo/clarity comments: 

Section 3 - there is section 3.1, 3.1.1, 3.1.2, 3.1.3, but not 3.2. Could it be restructured without the .1. ?

Section 3.1.3 - it could be worth reiterating that $\lambda$ is capped at 1 here, even if z can be higher than 1.

Section 4 - typo at **T**he stochastic action (capital T is not needed)

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
2

### Summary
This paper proposes a novel safe reinforcement learning (DRL) architecture named AutoSafe, which aims to embed "safety common sense" directly into the neural policy to address DRL guarantee issues in safety-sensitive applications. The core idea of AutoSafe is to fuse a traditional, high-performance DRL policy ($a_{\theta}$) with a model-based, verifiable safe policy ($a_{safe}$) through a smooth and adaptive mechanism.

### Strengths
- Novel Internal Fusion Architecture: Unlike traditional methods relying on external "safety shields" for hard switching (e.g., Simplex) , AutoSafe integrates the safety evaluation and correction modules as internal components of the policy network itself. This design makes the entire policy end-to-end differentiable and easy to integrate into existing Actor-Critic algorithms (like SAC).
- Real-World Applicability Validated: The paper does not stop at simulation. It successfully deploys AutoSafe on an embedded device (Raspberry Pi) for a physical Cartpole system. This experiment demonstrates the architecture's ability to conduct safe, continual learning  under real-world physical noise and delays, greatly enhancing the method's credibility and practical value.

### Weaknesses
- Strong Dependence on Linear Models: The entire safety foundation of AutoSafe is built upon the pre-calculated, model-based SCM ($K$ matrix) and SEM ($P$ matrix. As shown in Appendix A, these matrices are obtained by solving an LMI, which depends on a known, linear system dynamics model ($\dot{s} = As + Ba$). This prerequisite may not be met for many of the highly non-linear, difficult-to-model complex problems that DRL aims to solve.

### Questions
- Dependence on State Vector $s_t$: AutoSafe's safety core (SEM and SCM) relies heavily on a precise, complete, and low-dimensional state vector $s_t$ as input. If the agent can only learn from high-dimensional observations (e.g., pixels), how would this architecture obtain the $s_t$ needed to calculate the risk $z_t$? Would it require an additional (and possibly imperfect) state estimator? How would errors from this estimator affect AutoSafe's safety guarantees?
- Performance under $a_{safe}$ Goal Conflict: The paper analyzes $T$'s adaptiveness when $a_{safe}$ is "suboptimal" in performance. But what happens if the goal of $a_{safe}$ (e.g., stabilize at the equilibrium $s^*=\{0\}$) is in fundamental conflict with the DRL reward goal (e.g., reach a target $\hat{s}=\{0.1\}$)? Could this internal conflict lead to high-frequency oscillations between the two objectives, or cause the DRL policy to fail to learn?

### Soundness
3

### Presentation
3

### Contribution
3
