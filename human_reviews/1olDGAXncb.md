# $f$-Divergence Policy Optimization in Fully Decentralized Cooperative MARL

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Independent learning is a straightforward solution for fully decentralized learning in cooperative multi-agent reinforcement learning (MARL). The study of independent learning has a history of decades, and the representatives, such as independent Q-learning and independent PPO, can obtain good performance in some benchmarks. However, most independent learning algorithms lack convergence guarantees or theoretical support. In this paper, we propose a general formulation of independent policy optimization, $f$-divergence policy optimization. We show the generality of such a formulation and analyze its limitation. Based on this formulation, we further propose a novel independent learning algorithm, TVPO, that theoretically guarantees convergence. Empirically, we show that TVPO outperforms state-of-the-art fully decentralized learning methods in three popular cooperative MARL benchmarks, which verifies the efficacy of TVPO.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper utilizes f-divergence, specifically the total variation, to generalize the KL divergence in independent policy optimization.

### Strengths
- The presentation is clear and easy to follow.

### Weaknesses
- The application of f-divergence in policy optimization is not new; a comprehensive analysis of various distance constraints in policy gradients has been provided in [1].

- Extending existing single-agent analysis to the multi-agent setting is reasonable, but some assumptions are questionable. Specifically, the approach assumes full observability in MARL making the setting difficult to distinguish from single-agent reinforcement learning. Under full observability, what meaningful difference remains between centralized and decentralized control?

- The performance improvement appears marginal. With full observability, IPPO has already demonstrated near-optimal performance on SMAC and Multi-Agent MuJoCo. Were the baseline hyperparameters tuned to achieve their optimal reported performance?

- Why is win rate not used as the evaluation metric for SMAC-v2 tasks?


[1] Zhang, Junyu, et al. "Variational policy gradient method for reinforcement learning with general utilities." Advances in Neural Information Processing Systems 33 (2020): 4572-4583.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper explores independent learning in the multi-agent reinforcement learning (MARL) setting and introduces f-divergence policy optimization. The authors analyze the limitations of the method with an illustrative example and propose defining the f-divergence as the total variation distance. Theoretical and experimental results confirm the effectiveness of the proposed approach.

### Strengths
1. Detailed related work in the Fully Decentralized Learning field.
2. The paper introduces a well-grounded technique for achieving monotonic improvement in multi-agent optimization through decentralized learning.
3. The paper is well-structured and easy to follow.

### Weaknesses
1. The relevant work of CTDE is incomplete and lacks recent work, such as HASAC[a] and MAT[b].
2. Assuming global information might influence the impact of this work.
3. While the experiment results appear promising, the contribution is slightly insufficient compared with existing work[c,d].


a. Liu, Jiarong, et al. "Maximum Entropy Heterogeneous-Agent Reinforcement Learning." The Twelfth International Conference on Learning Representations.

b. Wen, Muning, et al. "Multi-agent reinforcement learning is a sequence modeling problem." Advances in Neural Information Processing Systems 35 (2022): 16509-16521.

c. Grudzien, Jakub, Christian A. Schroeder De Witt, and Jakob Foerster. "Mirror learning: A unifying framework of policy optimisation." International Conference on Machine Learning. PMLR, 2022.

d. Su, Kefan, and Zongqing Lu. "Decentralized policy optimization." arXiv preprint arXiv:2211.03032 (2022).

### Questions
1. Why use different metrics for SMAC (win rate) and SMACv2 (return)?
2. Due to the assumption of the global state, I suggest using Markov games [a] as the multi-agent framework.

a. Littman, Michael L. "Markov games as a framework for multi-agent reinforcement learning." Machine learning proceedings 1994. Morgan Kaufmann, 1994. 157-163.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes TVPO for cooperative Markov games, with the update rule of each agent as $\pi^i_{t+1}=\arg\max_{\pi^i} \sum_{a_i} \pi^i(a_i | s)Q_i^{\pi_t}(s,a_i)-\omega D_{TV}(\pi^i(\cdot|s)|| \pi_t^i(\cdot|s) )$ and shows that the algorithm can converge monotonically to the NE of the game. Moreover, TVPO with the adaptive $\beta$ in PPO shows superior empirical performance over previous algorithms.

### Strengths
- The empirical performance of TVPO is superior to previous SOTA
- The writing is clear except for several typos (see weaknesses)
- The proofs are easy to follow
- Compared to previous algorithms, TVPO is easy to implement

### Weaknesses
## Comparison to Related Work
My major concern is that this paper seems to miss several relevant literature. For instance, [1], [2] both proposed algorithms for independent learning in potential Markov games, which include the cooperative Markov games investigated in this paper. Further, [1] proposed a policy gradient algorithm and [2] proposed a policy iteration algorithm, which is highly relevant to this paper.

Moreover, the algorithm in [2] can also use the adaptive $\beta$ in PPO. Therefore, I'm wondering if TVPO will be superior to [2] when both using an adaptive $\beta$.

## Writings
- $i$ is superscript for $\pi$ but subscript for $V,Q$
- The $M$ in Proposition 4.2 and Section 4.2 differs
- Line 152: such as...

I would be happy to raise the score if the author can resolve the issues above.

[1] Leonardos, Stefanos, et al. "Global convergence of multi-agent policy gradient in markov potential games." arXiv preprint arXiv:2106.01969 (2021).

[2] Fox, Roy, et al. "Independent natural policy gradient always converges in markov potential games." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.

### Questions
- Is the $V^*$ in Theorem 4.6 the stationary point instead of the value function corresponding to the optimal policy?
- In the second line of Eq (23), it seems to be $\Rightarrow$ instead of $\Leftrightarrow$. Because $f$ is convex instead of strongly convex

### Soundness
2

### Presentation
3

### Contribution
2
