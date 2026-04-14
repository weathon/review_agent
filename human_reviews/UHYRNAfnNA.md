# Modification-Considering Value Learning for Reward Hacking Mitigation in RL

- Decision: Reject
- Scores: 6, 5, 1, 5

## Abstract
Reinforcement learning (RL) agents can exploit unintended strategies to achieve high rewards without fulfilling the desired objectives, a phenomenon known as reward hacking. In this work, we examine reward hacking through the lens of General Utility RL, which generalizes RL by considering utility functions over entire trajectories rather than state-based rewards. From this perspective, many instances of reward hacking can be seen as inconsistencies between current and updated utility functions, where the behavior optimized for an updated utility function is poorly evaluated by the original one. Our main contribution is Modification-Considering Value Learning (MC-VL), a novel algorithm designed to address this inconsistency during learning. Starting with a coarse yet value-aligned initial utility function, the MC-VL agent iteratively refines this function based on past observations while considering the potential consequences of updates. This approach enables the agent to anticipate and reject modifications that may lead to undesired behavior. To validate our approach, we implement MC-VL agents based on the Double Deep Q-Network (DDQN) and Twin Delayed Deep Deterministic Policy Gradients (TD3), demonstrating their effectiveness in preventing reward hacking in diverse environments, including those from AI Safety Gridworlds and the MuJoCo gym.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Modification-Considering Value Learning (MC-VL) to address reward hacking in reinforcement learning, where agents exploit unintended strategies to maximize rewards without achieving the desired outcomes.
Using the General Utility RL (GU-RL) framework, MC-VL refines the agent's utility function by evaluating policy updates across entire trajectories, helping to avoid short-term reward maximization that harms long-term objectives.
Implemented with Double Deep Q-Networks (DDQN), the method prevents reward hacking in safety-critical environments, offering a novel solution for achieving long-term alignment in RL.

### Strengths
- The paper is well written.
- Reward hacking is one of the main reasons why reinforcement learning is not widely deployed in real-world applications; any contribution that tackles this issue is important for the community.
- Using the utility function to avoid short-term reward maximization that harms long-term objectives is an elegant way to avoid value hacking.

### Weaknesses
- The evaluations are solely on toy environments; it would be interesting to test this approach on more complex dynamics and reward functions, such as in the continuous control suite MuJoCo, where reward hacking and undesired behaviors could be more prevalent.

### Questions
- Could MC-VL be considered not only as an approach for avoiding reward hacking, but also as a potential method for improving exploration?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper designs a novel algorithm Modification-Considering Value Learning (MC-VL) to address a phenomenon known as reward hacking in RL. MC-VL allows an agent to optimize its current utility function while considering the future consequences of utility updates, where the concept of current utility optimization is formalized using the General Utility RL framework. To the best of the authors’ knowledge, MC-VL is the first implementation of an agent that optimizes its utility function while considering the potential consequences of modifying it. The authors demonstrate the effectiveness of MC-VL in preventing reward hacking across various grid-world tasks, including benchmarks from the AI Safety Gridworlds suite.

### Strengths
1. The motivation is well justified, the method is reasonable
2. The algorithm addressed reward hacking through the lens of General Utility RL is novel and interesting
3. The empirical results show the effectiveness of the proposed algorithm, including benchmarks from the AI Safety Gridworlds suite.

### Weaknesses
1. MC-VL is based on the General Utility RL framework and the basic RL method for implementing MC-VL is only DDQN. Why did authors not try to implement MC-VL based on other RL methods, such as PPO or DDPG?
2. As stated in L65-L72, the robot grasping task is a typical case, why did authors not choose this environment to evaluate the performance? In addition, how does the proposed algorithm scale with a larger number of agents and more complex environments?
3. The comparison baseline method is only the basic RL method DDQN, it is better to compare more methods addressing reward hacking, such as the methods authors introduce in Related Work

### Questions
I have asked my questions in the Weaknesses section. I have no further questions.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
3

### Summary
This paper focuses on preventing reward hacking, in which optimizing a misspecified reward function causes undesired behavior. In particular, the authors focus on reward tampering, in which an agent can modify its reward function to receive higher reward with undesirable side effects. The authors propose a modification to DQN-like algorithms which decides whether to add newly collected transitions to the replay buffer depending on whether it seems like they will improve the current policy. They test the method, MC-DDQN, on some toy environments and find that it seems to prevent agents from reward hacking.

### Strengths
The paper focuses on an important problem and takes what appears to be a novel approach.

### Weaknesses
* **Mathematical correctness:** while the MC-DDQN algorithm is claimed to be grounded in General Utility RL, there are a number of issues with the presented algorithm that make it difficult to understand what it is actually doing. Consider Algorithm 2, for example. The ⇝ notation is already non-standard (e.g., it does not seem to be any of the uses [here](https://math.stackexchange.com/questions/680786/the-meaning-of-rightsquigarrow-in-math)). On line 2, the algorithm seems to suggest setting $U_{t+1}$ to $U^{\pi\_t}\_{VL\_t}$; however, on the first iteration of the for loop, $\pi\_t$ is not yet defined. On line 3, the algorithm refers to $\tau^\pi\_\rho$, which is not defined anywhere. In line 4, $\pi\_t$ takes as input the previous *transition* $T\_{t-1}$, despite policies being defined as taking states as input on line 133. The first part of line 5 should be $D \gets D \cup \\{ T\_{t - 1} \\}$. On the second part of line 5, $U\_{RL}$ has not been defined, only $F_{RL}$. Thus, *the core algorithm is completely unclear from the given description, given that most of the notation is undefined*.

 * **Limited experiments and evidence of usefulness:** given that the mathematical basis for the MC-DDQN algorithm is unclear, the remaining contribution of the paper lies in the experimental results. However, the experiments only focus on four gridworlds, and furthermore, in most of the experiments the proposed algorithm simply prevents the policy performance from dropping, which could just as easily be accomplished by freezing the policy when switching to the full environment. 

Out of these two weaknesses, the first—that the main algorithm is ill defined—is the primary reason I rate the paper with a low score.

### Questions
How can Section 4 be reworked such that the concepts and algorithms presented are well-defined?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explores how modifying the utility function’s design could mitigate the issue of reward hacking in RL. It introduces a new algorithm called MC-VL, aimed at addressing cases where RL agents maximize reward signals by exploiting incomplete or ambiguous reward functions, potentially misaligning with the designer's actual goals.

### Strengths
1. This paper proposes a novel approach to address inconsistencies during the learning process.  
2. It discusses reward hacking within the framework of GU-RL, providing in-depth insights and theoretical support for tackling this issue.

### Weaknesses
- The paper notes that MC-VL requires predictions about future behaviors, which could entail higher computational costs and may limit its applicability in more complex environments.  
- The proposed method assumes that the agent can learn the correct utility function from observed rewards. In more complex or uncertain environments, the performance of the algorithm could suffer if the reward model fails to generalize accurately to new trajectories.  
- The experiments are primarily conducted in relatively simple grid-world environments, leaving the algorithm’s effectiveness in more complex, high-dimensional environments still unverified.

### Questions
1. Were all experiments conducted by training on the "Safe version" before transitioning to the "Full version"? If so, how would the proposed algorithm generalize to typical environments?  
2. The authors mention that hyperparameters significantly influence algorithm performance, listing many in Table 1 of Appendix E, such as "Inconsistency check training steps l," "Inconsistency check rollout steps h," "Number of inconsistency check rollouts k," and "Predicted reward difference threshold." However, it’s unclear how these parameters specifically impact the proposed algorithm or how they should be selected.  
3. Reward hacking is common across various environments, as mentioned in the robotic context at the paper's beginning. Could this algorithm potentially be applied to continuous action versions, like TD3 for DDQN? If extended to continuous environments, would the accuracy of the utility function become more critical to the algorithm's performance?  
4. Could the authors validate the algorithm's effectiveness in more complex environments, such as those with high-dimensional inputs (e.g., continuous or image-based inputs)?
5. check the definnation of $\lambda(\tau)$

### Soundness
3

### Presentation
2

### Contribution
2
