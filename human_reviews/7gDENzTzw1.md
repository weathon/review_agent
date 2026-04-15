# Belief-Enriched Pessimistic Q-Learning against Adversarial State Perturbations

- Decision: Accept (poster)
- Scores: 6, 6, 5, 8

## Abstract
Reinforcement learning (RL) has achieved phenomenal success in various domains. However, its data-driven nature also introduces new vulnerabilities that can be exploited by malicious opponents. Recent work shows that a well-trained RL agent can be easily manipulated by strategically perturbing its state observations at the test stage. Existing solutions either introduce a regularization term to improve the smoothness of the trained policy against perturbations or alternatively train the agent's policy and the attacker's policy. However, the former does not provide sufficient protection against strong attacks, while the latter is computationally prohibitive for large environments. In this work, we propose a new robust RL algorithm for deriving a pessimistic policy to safeguard against an agent's uncertainty about true states. This approach is further enhanced with belief state inference and diffusion-based state purification to reduce uncertainty. Empirical results show that our approach obtains superb performance under strong attacks and has a comparable training overhead with regularization-based methods. Our code is available at https://github.com/SliencerX/Belief-enriched-robust-Q-learning.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the vulnerability of reinforcement learning agents to adversarial attacks by manipulating state observations. The authors propose a algorithm focusing on developing a pessimistic policy that accounts for uncertainties in state information. This is supplemented by belief state inference and diffusion-based state purification techniques.

### Strengths
Addressed an important problem in Reinforcement Learning, providing fresh perspective and insights.
The experimental design and methodology are well-constructed.

### Weaknesses
The innovative aspects presented over WocaR-DQN appear to be incremental in nature.

### Questions
How extensive or generalizable are the empirical results presented? Given that the study focuses on the Continuous Gridworld, which is relatively simple, and only includes two Atari games. 

Considering that Atari game screens are depicted using 8-bit (256 levels) RGB, yet typically select colors from a narrower palette, how effective are the different attack budgets (15/255, 3/255, 1/255) tested in this study? Specifically, can these perturbations alter the colors enough to cause confusion with other colors in the game's limited palette? If not, could the observed robustness of the learning method simply be a consequence of learning to filter out specific colors?

Is the proposed method also applicable to environments like Mujoco or Mountain Car, where the input variables are more continuous and less discrete than those in Atari games?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a novel RL algorithm that aims to protect RL agents from adversarial state perturbations. The authors propose a pessimistic DQN algorithm that takes into consideration both the worst-case scenarios and belief about the true states. The algorithm also features a diffusion-based state purification method for applications like Atari games. The paper shows empirical results demonstrating that their approach significantly outperforms existing solutions in robustness against strong adversarial attacks, while maintaining comparable computational complexity.

### Strengths
1. The empirical results show performance improvement under strong attacks compared to baseline methods. The algorithm works well for both simplistic environments and more complex scenarios like Atari games with raw pixel input.
2. The algorithm's computational overhead is comparable to existing methods that use regularization terms.

### Weaknesses
1. The algorithm assumes access to a clean environment during training, which may not always be the case in real-world applications.
2. While the diffusion model adds robustness, it also adds computational overhead, potentially making it slower at test time.

### Questions
1. How sensitive is your algorithm to the choice of hyperparameters?
2. Given the requirement for a clean training environment, how would your method perform in a scenario where such an environment is not readily available?
3. The diffusion model increases computational complexity during the test stage. Are there ways to optimize this without compromising the robustness?
4. Why PA-AD is not evaluated on continuous gridworld?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors study the problem of defense in presence of perceived state attack in Reinforcement learning and propose a method to approximately solve the Stackelberg equilibrium between the agents and the adversary. Their method involves solving pessimistic Q-learning and estimating belief state of the agent and using them for state purification. They propose two algorithms called BP-DQN and DP-DQN to defend against adversarial attacks.

### Strengths
1. Proposes a novel method to defend against adversarial attacks in RL by combining pessimistic Q-learning with belief state estimation and state purification objective.
    
2. Authors provided theoretical results to compare the policy found by their algorithm to the optimal policy.
    
3. Presented interesting empirical experiments to demonstrate effectiveness of their algorithm on several examples.

### Weaknesses
1. It has been shown that Stackelberg Equilibrium as defined in Definition 2 need not always exist, refer to theorem 4.3 [https://arxiv.org/pdf/2212.02705.pdf](https://arxiv.org/pdf/2212.02705.pdf). So, finding an approximate solution for them is meaningless. However, non-existence of Stackelberg equilibrium is a worst case phenomenon. Authors should incorporate this in their paper.
    
2. It will be great if authors can include a short paragraph before section 3.2 discussing a big picture of their strategies before diving into each one of them. It would also help to include some mathematical details in each section.
    
3. Abstract wrongly mentions that past methods either regularize or retrain the policy. However, methods like Bharti et.al. just purify the states directly.

### Questions
1. It is well known that defense against perceived state attack requires solving a partially observable MDP which is a hard problem to solve in general. Could you clarify how your method is able to avoid these hardness issues?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes two new algorithms to robustify agent policies against adversarial attacks. It formulates the problem of finding a robust policy as a Stacleberg game (where agents choose policies), and then further incorporates beliefs into the derived algorithm. For pixel-based observation spaces, the game uses a diffusion-based method to derive valid possible states. The paper is very well structured and easy to follow.

### Strengths
- the paper addresses many shortcomings of current works
- the theoretical algorithms and derivations are insightful
- the practical implementation of the derived algorithms are well motivated

### Weaknesses
- the paper assumes that both the clean MDP and the perturbation budget are known to both the victim and the attacker
- it would be interesting to run an ablation on these assumptions. How well does the method work if the budget is not known exactly, or if the MDP transition function is not known exactly?

### Questions
see weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent
