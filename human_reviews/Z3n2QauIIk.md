# A Finite-Time Analysis of Distributed Q-Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
Multi-agent reinforcement learning (MARL) has witnessed a remarkable surge in interest, fueled by the empirical success achieved in applications of single-agent reinforcement learning (RL). In this study, we consider a distributed Q-learning scenario, wherein a number of agents cooperatively solve a sequential decision making problem without access to the central reward function which is an average of the local rewards. In particular, we study finite-time analysis of a distributed Q-learning algorithm, and provide a new sample complexity result of $\tilde{\mathcal{O}}\left( \max\left\\\{  \frac{1}{\epsilon^2}\frac{t_{\text{mix}}}{(1-\gamma)^6 d_{\min}^4 }  ,\frac{1}{\epsilon}\frac{\sqrt{|\mathcal{S}||\mathcal{A}|}}{(1-\sigma_2(\boldsymbol{W}))(1-\gamma)^4 d_{\min}^3}  \right\\\} \right)$ under tabular lookup setting for Markovian observation model.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies distributed Q learning, where the agent communicate through a graph. The finite-time convergence under both iid and Markovian samples are studied.

### Strengths
The problem of distributed Q learning is important and interesting. The convergence results also are SOTA.

### Weaknesses
1. Although the notations are defined mathematically, it can be more clear with additional explaniation. For example, in lines 225-227, it would be godd to explan what $\delta, \Delta$ are. 
2. The introduction of the switched system looks odd to me. If it is only for proof technique, might should introduced in the proof part. 
3. The proof technique is standard by following single agent Q-learning, and analysis for multi agent system is also studied. For example, a quick search results in a study [1], where robust multi-agent Q learning is studing. The non-robust is a special case with $R=0$. Hence it make me feel the novelty here is incremental. 
4. Analysis of Markovian vs iid is also not new. 



[1] Wang, Yudan, et al. "Data-driven robust multi-agent reinforcement learning." 2022 IEEE 32nd International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2022.

### Questions
What is the novelty compared to existing works?

Can you address the weakness part?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper provides the first sample complexity bounds for distributed Q-learning scenario in tabular setting. It is a topic of increasing interest and importance given the empirical success achieved in applications of single-agent reinforcement learning.

### Strengths
It is the first paper that provides the sample complexity bounds for tabular distributed Q-learning without relying on any strong assumptions.

### Weaknesses
1.	In Corollary 3.8 and Corollary 4.2, the restrictions regarding the step size alpha are missing. These restrictions are used in the proof but are not mentioned in these two corollaries. Additionally, the restriction on $\alpha$ in line 1194 of the proof for Corollary 3.8 should be proportional to $\epsilon$ rather than its reciprocal.
2.	Instead of comparing the dependencies on graph structure, I recommend a comparison of the entire sample complexity bounds with the other two baselines.
3.	The experiment design is not very clear. For example, how are the transition functions and rewards generated? What is the number of agents? Why do you choose to compare your method with QD-learning? Moreover, $S = 2$ seems small and it is better to try some larger-scale experiments.
4.	There are some undefined notations and typos in the paper:
- The agent collection $V$ and the discount factor $\gamma$ are not defined at the beginning of the preliminary section.
- In the Bellman equation, $s$ in the maximum term should be $s’$.
- In line 220, the dimensions of the two matrices are incorrect.
- In line 270, $R_{max}$ is not defined.
- In line 436, it should be noted that $\rho < 1$.
- In line 1006, the content exceeds the page limitation.

### Questions
Your multi-agent MDP setting differs from the conventional MAMDP model, as you assume the agents share a common trajectory. Is this setting applicable in real-world scenarios, and are there other papers employing a similar setting?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
In this paper, the authors present a finite-time analysis of distributed Q-Learning, where the agents need to cooperatively maximize an average of local rewards. Specifically, the agents do not have access to any global information and can only communicate with neighboring agents during the learning process. They show that a distributed Q-learning algorithm has a $\tilde{\mathcal{O}}\left(\max \left\\{\frac{1}{\epsilon^2} \frac{t_{\text{mix}}}{(1-\gamma)^6 d_{\min }^4}, \frac{1}{\epsilon} \frac{\sqrt{|\mathcal{S}||\mathcal{A}|}}{(1-\sigma_2(\mathbf{W}))(1-\gamma)^4 d_{\text {min }}^3}\right\\}\right)$ sample complexity bound.

### Strengths
* The paper is the first to give sample complexity bounds for the distributed Q-learning algorithm without restrictive assumptions.

* The results are technically sound.

### Weaknesses
* The technical contribution of this paper is unclear. It appears that the results are obtained by accommodating the analysis technique in [1] to the distributed setting. I suggest the authors elaborate on their technical contribution in the intro.

* The sample complexity bounds are of the order $\tilde{\mathcal{O}}\left(\max \left\\{\frac{t_{\text {mix }}}{\epsilon^2} \frac{|\mathcal{S}|^2 |\mathcal{A}|^{2}}{(1-\gamma)^6}, \frac{1}{\epsilon} \frac{|\mathcal{S}|^{\frac{5}{2}}  |\mathcal{A}|^{\frac{5}{2}}}{(1-\gamma)^4\left(1-\sigma_2(\mathbf{W})\right)}\right\\}\right)$. This bound is significantly larger compared to existing results in single-agent and federated RL. The authors should discuss whether this large bound arises from a loose analysis or the inherent nature of the problem setting. It may also help to compare the results with results on federated RL.

* The main body of the paper is written in an immense technical style. I suggest the authors try to make the proof outlines more accessible and discuss the implications of their theoretical results.


Since I am not very familiar with the field of MARL, I choose to assign a low confidence score to my evaluation.

[1] D. Lee and N. He. A unified switching system perspective and convergence analysis of q-learning algorithms. Advances in Neural Information Processing Systems, 33:15556–15567, 2020.

### Questions
* What would happen if we choose matrix $W$ to let all agents be able to communicate with each other? Would the problem setting degenerate into a centralized case? Is a better theoretical guarantee possible?

* Except for the case in Appendix B, are there any other choices of $W$ in common examples of practical applications? What is the scale of $\sigma_2(W)$, respectively?

### Soundness
3

### Presentation
2

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
This work explores multi-agent reinforcement learning, where multiple agents collaboratively learn their individual policies in order to maximize the total cumulative rewards of all agents. They introduce a distributed Q-learning algorithm in which each agent updates its Q-estimates by aggregating the Q-values of its neighbors, weighted by a specified weight matrix. Furthermore, they provide a finite-time convergence analysis of the proposed algorithm in a tabular setting under mild assumptions.

### Strengths
1. The paper presents a finite-time convergence analysis under relaxed conditions.

### Weaknesses
1. The sample complexity presented is significantly distant from the tightest bound for the single-agent case in \cite{1}. In particular, considering that \( d_{\text{min}} \le \frac{1}{SA} \), the dependence on \( \frac{1}{(d_{\text{min}})^3} \) and \( \frac{1}{(d_{\text{min}})^4} \) seems critical. It remains uncertain whether the reported sample complexity adequately reflects the impact of key parameters on the learning process.
2. While the analysis is presented under milder conditions compared to the baselines, it is conducted in a tabular setting, which is simpler than those of the baselines. Therefore, it remains unclear whether the relaxation of assumptions in the simpler setting shows sufficient technical novelty.
3. It lacks a comparison with other decentralized RL algorithms, such as [2], which could provide better insights into the unique dependencies observed in decentralized scenarios and the unique challenges present in multi-agent settings. While this paper addresses a more general scenario where agents' actions can influence the rewards of others, it appears that this setting can encompass decentralized reinforcement learning with environmental heterogeneity, where agents aim to maximize average rewards, which seems more relevant to this work than the single-agent case.

[1] Li, C. Cai, Y. Chen, Y. Wei, and Y. Chi. Is q-learning minimax optimal? a tight sample complexity analysis. Operations Research, 72(1):222–236, 2024. \
[2] Tong Yang, Shicong Cen, Yuting Wei, Yuxin Chen, and Yuejie Chi. Federated natural policy gradient methods for multi-task reinforcement learning. arXiv preprint arXiv:2311.00201, 2023. \

### Questions
1. I wonder if  \frac{1}{(d_{\text{min}})^4} also appeared in the sample complexity bound in previous works [3,4].
2. Could you provide intuitions or simple examples that explain what enables the relaxation of the assumptions? Additionally, I am curious whether this relaxation is specific to the tabular setting.

[3] P. Heredia, H. Ghadialy, and S. Mou. Finite-sample analysis of distributed q-learning for multi-agent networks. In 2020 American Control Conference (ACC), pages 3511–3516. IEEE, 2020. \
[4] Finite-time convergence rates of decentralized stochastic approximation with applications in multi-agent and multi-task learning.

### Soundness
3

### Presentation
2

### Contribution
2
