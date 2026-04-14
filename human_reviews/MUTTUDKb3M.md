# DRIMA: Differential Reward Interaction for Cooperative Multi-Agent Reinforcement Learning

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Multi-agent reinforcement learning (MARL) owning to its potent capabilities in complex systems has gained remarkable research attention nowadays, in which collaborative decision-making and control for multi-agent systems is one of the key research focuses.
The prevalent learning framework is centralized training with decentralized execution (CTDE), in which the decentralized execution realizes strategy flexibility, and the use of centralized training ensures stationarity and goal consistency while becoming incapable when facing scalability and complexity situations.
To address this issue, we follow the concept of distributed training with decentralized execution (DTDE).
Decentralization is naturally accompanied by the game during the learning process, which has not been entirely studied in related work, resulting in the constrained strategy combination of MARL.
In this paper, we devise a novel approach of differential reward interaction (DRI) with conflict-triggered for the distributed evaluation that enables overall goal consistency through highly efficient local information exchange.
With this collaborative learning method, the DRI-based MARL can eliminate the notorious issue of converging to saddle equilibriums of stochastic games.
Meanwhile, it possesses provable convergence and is well compatible for general value-based and policy-based algorithms.
Experiments in several benchmark scenarios demonstrate that DRIMA realizes collaborative strategy learning with enhanced global goal-achieving.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper follows the distributed training with decentralized execution (DTDE) training paradigm and focuses on the cooperation between multiple distributed agents. The authors argue that the conflicting payoffs of different agents would emphasize the game between agents and hinder cooperative behaviors. To mitigate this, they proposed a conflict-triggered differential reward interaction (DRI) method to reconstruct the individual rewards. The proposed method, DRIMA, is tested on matrix games, MPE, and SMAC to verify its effect.

### Strengths
1. This article provides a detailed description of the use of DRIMA in the context of specific tasks from easy (Prisoner's dilemma) to hard (general Markov games). The pseudo-code in the appendix also makes the proposed method easier to follow.
2. The authors have proved the convergence of learning with the differential reward, which makes DRIMA technically solid.
3. Experiment and theory corroborate each other very well in DRIMA. The use of DRIMA effectively helps the raw algorithms jump out NE and reach the global optimum in matrix games.

### Weaknesses
1. The performance of DRIMA in SMAC is not significant. Even with the inclusion of communication between agents, DRIMA is still inferior to CTDE in some scenarios. It makes me concerned about the value of the proposed method in real-world tasks.
2. It seems DRIMA often exhibits a slower start in experiments. This is detrimental to sample efficiency, and I suggest the authors give further explanation or improvements.

### Questions
1. When tested on SMAC, how do the authors define the individual rewards of the agents?
2. See Weaknesses 2.

### Soundness
3

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
3

### Summary
The paper proposes a conflict-triggered differential reward interaction (DRI) method to identify and reshape conflicting payoffs between agents. The experiments conducted on matrix games and continuous space tasks MPE demonstrate the effectiveness of the algorithm.

### Strengths
1. The paper is well-motivated overall.
2. The paper proposes deals with saddle equilibriums based on conflict-triggered differential rewards.

### Weaknesses
1. The proposed algorithm lacks experimental comparisons with recent works on MARL cooperation.
2. The current Figure 2 is not very comprehensible for understanding the conflict-triggered mechanism of DRI. The boundaries between the various colored surfaces are not easily distinguishable.
3. Including experimental results from more complex SMAC scenarios in the main paper would better demonstrate the applicability of the proposed method.

### Questions
1. As the method exploit the MFT concept, could it be applied to large-scale multi-agent environments?
2. I am wondering whether conflict-triggered differential rewards might hinder early exploration.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Although the Centralized training with decentralized execution (CTDE) framework is currently popular, it faces challenges in terms of scalability and handling complex tasks. This paper introduces a new approach called differential reward interaction (DRI) with conflict-triggered, which enables agents to achieve a unified overall goal by exchanging local information with each other. This method also overcomes the issue of saddle equilibrium in policy combination, a common challenge in game theory and optimization. The effectiveness and superiority of the proposed method, DRIMA, are demonstrated through its application in matrix games, Multi-Agent Particle Environment (MPE) scenarios, and Star-CraftⅡ.

### Strengths
1.	The overall structure of the paper is complete, the content is sufficient, and the experimental evidence is persuasive.
2.	The paper proposes that if personal rewards conflict with neighbor rewards, personal rewards should be reshaped as the neighbor-averaged reward to maintain the group consistency. This idea is similar to Reward Centering or Advantage Functions.

### Weaknesses
1.	The paper does not clearly define "differential rewards" or "conflicts".
2.	The purpose of Equation 2 is not intuitively explained. Providing a straightforward example would greatly help in understanding it.
3.	The relationship between Equation 2 and the differential action-value function is not explained.
4.	In Section 2.2, $\mu_\pi^i$ in the differential action-value function represents the average reward per step for agent i. However, in Section 3.2, $\bar\mu_\pi$ represents the average of the average rewards per step for the agent itself. Is there a contradiction between the two? So, I think the formula in section 2.2 may have been written incorrectly.
5.	Around lines 244-247, should not $\beta_t$ in Equation 4 be the step size of the Actor?
6.	The example in Figure 2 is somewhat abstract. Could you explain in more detail the issue of gradient directions under different views?

### Questions
1.	Did you draw on the idea of the advantage function? If so, where does the specificity of your method lie? What are the advantages compared to methods using the advantage function?
2.	Did you draw on the concept of Reward Centering? If so, what is the specificity of your method?
3.	Some questions about the Weakness section.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
An important issue in DETD is the interactions among agents, which manifest in simultaneous decision-making and reward conflicts. To address these challenges, this paper introduces DRIMA, a differential reward interaction-based MARL method. Specifically, the authors model the information interactions among agents as an undirected graph network, where each node represents an agent and each edge represents the connections between them. Based on this, they propose an exchange reward rule based on the overall performance of the policy. They evaluate the quality of the team policy based on the differential reward of the current agent's action. The authors conduct a theoretical analysis of their method from the perspectives of two-player matrix games and multi-player Markov games. Their experiments focus on general-sum games, specifically validating the method's effectiveness in matrix games, MPE, and SMAC.

### Strengths
1.	The authors conducted a theoretical analysis of the convergence of their method.
2.	The authors used "Two-player matrix game" and "Multi-Player Markov Game" to analyze the effectiveness of their method.

### Weaknesses
1.	The interaction rule in line 164 lacks a formal definition, and $r^i-\bar{\mu}^{N^i}$  and $\bar{r}^{N^i}-\bar{\mu}^{N^i}$ lack more direct definitions, making it difficult to understand.
2.	The paper does not provide sufficient experiments in SMAC to demonstrate its effectiveness.

### Questions
1.	What’s the formal definition of “reward interaction”?
2.	In line 193, what do $r^i_t-\bar{\mu}_t^{N^i}$ and $\bar{r}_t^{N^{-i}} - \bar{\mu}_t^{N^i}$ represent? And how does the opposite sign between them “indicates the conflict contribution of action decision relating to the individual and other neighbors’to the current neighbor-joint policy”? As a key contribution of the paper, the authors need to carefully explain the meaning of each part in this section to help readers understand their ideas.

3.	In the authors' paper, each agent in the MAS is modeled with an individual reward. However, in the SMAC environment, each agent shares the same reward. How does the method in the paper address this situation?

4.	Based on Question 3, in many cooperative multi-agent systems, each agent shares a team reward, meaning that $r^i$ in Formula 2 is the same and equal to $\bar{r}^{N^i}$. Does the conflict-triggered differential reward interaction still apply in this case?

5.	In the MPE experiments, the method was not compared with QMIX. Please explain the choice of baselines.

### Soundness
2

### Presentation
2

### Contribution
2
