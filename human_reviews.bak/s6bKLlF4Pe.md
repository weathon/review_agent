# Provable Knowledge Transfer using Successor Feature for Deep Reinforcement Learning

- Decision: Reject
- Scores: 3, 3, 6, 8

## Abstract
This paper explores knowledge transfer using successor features (SFs) in reinforcement learning (RL) scenarios where the reward function changes across tasks while the environment's dynamics remain the same. Under this framework, the Q-function of a task can be decomposed into a successor feature and a reward mapping: the former characterizes the transition dynamics, and the latter characterizes the task-specific reward function.
This Q-value function decomposition, coupled with a policy improvement operator known as "generalized policy improvement" (GPI), simplifies the search space for finding the optimal Q-function when transferring knowledge from one task to another that shares the same transition dynamics. As the optimal policy can be directly derived from the optimal Q-function, the SF \& GPI framework exhibits promise in enhancing efficiency and effectiveness in decision-making compared to traditional RL methods like Q-learning.
However, despite the observed superior performance of SF \& GPI in numerical experiments, their theoretical foundations remain largely unestablished, especially when learning successor features using deep neural networks in conjunction with deep Q-network (SF-DQN). To the best of our knowledge, this paper provides the first convergence analysis with provable generalization guarantees for SF-DQN with GPI. Moreover, our theoretical results reveal that SF-DQN \& GPI significantly accelerate the policy transfer across tasks and indicate that SF decomposition outperforms non-representation learning approaches, such as deep Q-network (DQN), with simultaneously faster convergence rate and improved generalization. Numerical experiments on real RL tasks support the superior performance of SF-DQN \& GPI, quantitatively aligning with our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper investigates theoretical properties of the combination of SF-DQN + GPI algorithm for multi-task reinforcement learning. New theoretical results show that the above combination leads to better performance by reusing prior tasks using GPI.

### Strengths
1. The paper proposes new theoretical insights into the Successor feature + GPI algorithm for the deep multitask reinforcement learning setting. The authors prove the convergence of this algorithm at a rate 1/T to the optimal Q function with multi-layer neural network and also quantify the generalization error of the learned Q function. 
2. The paper also shows what is the benefit afforded by using the successor feature formulation for transfer learning — the perfomance gap in Q function learned via DQN is worse than a factor of (1+\gamma)/2.
3. Finally authors verify that the trends suggested by their bounds like dependence on initialization of weight vectors and task relevance holds by experiments on a toy domain.

### Weaknesses
1. I am doubtful about the significance of convergence results. The convergence result with GPI follows the same rate as the convergence rate without GPI. It is hard to tell directly what is the difference in the constants. Having a thorough discussion with some examples would serve to give readers a better understanding of the upper bound.
2. I have the same question about transfer results, how important is the factor of  (1+\gamma)/2. Why does it make the bound substantially looser? Even then we are comparing upper bounds which are not tight, so what is the correct way to make sense of this result?
3. The experiments seem to present results that seem obvious. If weights are initialized closer to optimal weights convergence is fast and if the weight of transfer task is similar to prior task the transfer is fast. This is also similar to bounds we have seen for successor features in previous works.

### Questions
1. On page 5, Equation 15, should the gradient for theta be wrt. the best action across all (tasks) successor features based Q function or just the gradient with best action across the successor feature based Q function of current task.
2. In equation 17, I dont believe the notations are correct or have been explained previously in the paper. What is P(s_tau\in .)? What does v_\tau stand for?
3. How is \phi obtained in all the experiments? I dont think \phi is trained in this work based on the exposition?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the multi-task RL evironment where the environments have different reward functions but share the same underlying transition dynamics. The paper proposes a Q-function decomposition and a generalized policy improvement (GPI) algorithm to to find the optimal Q-function. This paper provides theoretical results for the speed of convergence rate and generalization. Numerical experiments are also provided.

### Strengths
- transfer learning and successor features are interesting avenues of research
- the claims are ambitious

### Weaknesses
- The Theorems do not seem to be correct. For instance, it does not seem possible that the size of a replay memory impacts the bound as it does given that it could all be filled with bad tuples.
- The experiments do not follow all good practice (see questions below)
- There are some typos, e.g. in Assumption 1 "such that minimizes (12) for"

### Questions
THEORY
- Q-learning does not have any valid proof of convergence when used in conjunction with deep learning. Can you clarify how divergence is prevented in this case? In the results from the paper, a large enough replay memory ensures good convergence. How can the replay memory size by itself be a good measure of the quality of the tuples that are used? (It could all be filled with bad tuples).

EXPERIMENTS
- Why is there no variance in the results reported for the experiements?
- Why does SFDQN (GPI) has a normalized average reward that decreases with the number of episodes? Given the theory that the paper describes, this should slowly converge towards the optimum.

### Soundness
1 poor

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies knowledge transfer using a SF-DQN architecture in multi-task RL problems (each task is defined by a separate reward function, while they share the same environment). A number of theoretical results are presented for various cases of convergence and knowledge transfer.

### Strengths
Theoretical analysis of SF when combined with function approximation is quite interesting and can be insightful.

### Weaknesses
- The first half of the paper is confusing as the authors seem to conflate optimal value/policy at the task level with some generic $Q^*$ and $\pi^*$. It was hard to follow and figure out what they mean by the notations and/or possible notational mistake. 

- As an important point, note that the GPI theorem induces improvement for the case of several policies but **the same** reward (i.e., only one task). If there are multiple rewards, then maximum of their corresponding value functions is simply meaningless and does not represent any physical concept regarding a new reward function (see the comments bellow). [This is OK when using SF as the $w_i$ corresponds to the current task and only $\psi_j$ comes from some previous task.]

- There were a number of other typos (a proof read aside from the points mentioned above is strongly suggested).

### Questions
- If you have multiple tasks, then what do you mean by $Q^*$ and $\pi^*$? In fact, both $\pi^*$ and $Q^*$ are not well-defined unless the reference for optimality is specified. For instance, you may define a total reward as the summation of all the task-level rewards, whose $\pi^*$ and $Q^*$ would be different from the case where a total reward is a *weighted* sum of task-based rewards or some non-linear function of them. 

- Eq (13) is not a policy improvement operator (PI is to take an argmax of the current value to induce a new policy). 

- Last three lines of page 4 --> First, your reference to $Q^*$ is senseless. More importantly, this setup has nothing to do with your settings. Maximum of several $Q$ functions corresponding to various rewards may not even be a valid $Q$ function for a different reward signal. One can easily construct an MDP with say three tasks and two actions, where in a certain state, $a_1$ is the optimal action for task 3, while $\max_{i\in \{ 1,2 \} } Q_{i}$ give you action $a_2$. Indeed it is theoretically possible that $\max_{i\in \{1,2\}} Q_{i}$ gives you suboptimal action in *all* states. 

- Table 1 --> what is $\phi_{i}(\Theta^{*}_{i})$ ? $\phi$ is supposed to be the feature vector. 

- Assumption 3 is unclear. Note that $\pi(a|s) \in [0,1]$, Hence, for sufficiently large $C$ this assumption is always true! Am I missing something?

- Is (20) a s fair assumption? For most cases, $K$ would be a large number, so it looks like that the initial approximation of $\psi_1$ must be already very good! Perhaps a better presentation could be that there has to be some pre-training for $\psi_1$ before using this algorithm?

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a theoretical study over successor feature (SF) based deep Q learning and multi-task transfer learning using generalized policy improvement (GPI). The manuscript provides several theories showing the convergence properties of SF training using DQN, SF training with GPI and transfer gap with GPI.

### Strengths
The targeted problem of theoretical analysis for transfer learning using SF is important as SF has achieved many empirical successes in deep RL.

Experimental results verified the gap in SF training and transfer learning as indicated by the proposed theory.

### Weaknesses
The proposed framework requires the knowledge of a ground truth feature for state action tuple such that they can linearly represent the true reward function.

Other questiones:

How is the convergence result presented in this paper connected with the GPI theory presented in the original transfer learning with successor feature paper? In the original paper it is assumed that approximated Q functions are given with a deviation $\epsilon$. How would the theories in this work explain the individual terms in the original GPI theory?

Typo:

Sec. 2, paragraph 2: minimize -> maximize, discount factor not shown in the cumulative formula.

### Questions
See weakness part.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
