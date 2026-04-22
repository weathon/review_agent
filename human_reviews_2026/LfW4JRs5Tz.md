# Resource-Efficient Model-Free Reinforcement Learning for Board Games

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 4, 6, 4, 4

## Abstract
Board games have long served as complex decision-making benchmarks in artificial intelligence. In this field, search-based reinforcement learning methods such as AlphaZero have achieved remarkable success. However, their significant computational demands have been pointed out as barriers to their reproducibility. In this study, we propose a model-free reinforcement learning algorithm designed for board games to achieve more efficient learning. To validate the efficiency of the proposed method, we conducted comprehensive experiments on five board games: Animal Shogi, Gardner Chess, Go, Hex, and Othello. The results demonstrate that the proposed method achieves more efficient learning than existing methods across these environments. In addition, our extensive ablation study shows the importance of core techniques used in the proposed method. We believe that our efficient algorithm shows the potential of model-free reinforcement learning in domains traditionally dominated by search-based methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the KLENT algorithm for board games to achieve more efficient learning. Both KL regularization and entropy regularization are used to assist the training of the policy network, while λ-returns are employed to train the value function network. Experiments are conducted on five board games: Animal Shogi, Gardner Chess, Go, Hex, and Othello, and the ablation study demonstrates the contribution of each component.

### Strengths
1. The paper is well-organized and easy to follow.

2. The paper includes extensive experiments conducted on five game environments, along with numerous ablation studies.

### Weaknesses
1. The comparison with baseline algorithms in the experiments is not very clear, so I cannot be sure whether the proposed KLENT algorithm actually outperforms AlphaZero. For more details, please refer to the Question section.

2. The paper’s originality is somewhat limited: using $\lambda$-returns to train the value function network is a common practice, and employing the exact solution of the optimization objective to train the policy is also used before. Therefore, it seems that the core innovation of KLENT lies mainly in the simultaneous use of KL and entropy regularization. As illustrated in Line 131, "we aim to empirically show that this combination is effective for achieving efficient learning in the board game domain."

### Questions
1. Grill et al. (2020) used KL regularization to constrain the learning of the policy and replaced MCTS with an exact solution, which is very similar to KLENT. Comparing Equation (2) in this paper with Equation (7) in Grill et al. (2020), it appears that the main difference is the additional entropy term. Could you carefully analyze the differences between the two and include Grill et al. (2020) as a comparison model in the experiments?

2. In the evaluation, is the policy learned by KLENT used to guide MCTS for comparison with AlphaZero, or is the comparison based solely on the policy learned by the neural network without MCTS?

3. Using the number of simulator evaluations as a resource constraint makes sense, and I understand that approach. However, I am also curious: in AlphaZero, collecting a single training data point via MCTS requires multiple neural network calls, whereas KLENT does not. I would like to know the respective amounts of data collected by KLENT and AlphaZero, and how much difference there is in their actual runtime.

4. The performance of DQN in Hex is comparable to KLENT, but it performs poorly in the other games. Can you explain this phenomenon?

5. How many MCTS simulations does AlphaZero perform during training and evaluation?

Grill, Jean-Bastien, et al. "Monte-Carlo tree search as regularized policy optimization." International Conference on Machine Learning. PMLR, 2020.

### Soundness
2

### Presentation
3

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
The paper proposes KLENT, a model-free reinforcement learning algorithm for board games. The method is presented as a "simple" alternative to complex search-based methods like AlphaZero. KLENT combines three existing RL techniques: Kullback-Leibler (KL) regularization for gradual policy updates, entropy regularization for exploration, and $\lambda$-returns for value function learning. The authors conduct experiments on five board games (e.g., Animal Shogi, 9x9 Go, Hex) and argue that KLENT is significantly more "resource-efficient" in terms of training-time simulator evaluations compared to AlphaZero, Gumbel AlphaZero, DQN, and PPO. An ablation study is also presented to show that all three components are necessary for this efficiency.

### Strengths
The paper is well-written and easy to follow. The experiments are relatively extensive, covering five different games and including a detailed ablation study (Section 5.2) and hyperparameter analysis (Appendix D), which validate the components of the proposed method. The main result on training efficiency is interesting; for example, in 9x9 Go, KLENT achieves a high win rate (~89%) using only 800M simulator evaluations, while the paper's AlphaZero implementation requires ~4000M evaluations to reach a similar level (Appendix G.3, Table 6). This demonstrates good sample efficiency for learning a good reactive policy.

### Weaknesses
My main concerns with this paper are its limited novelty and the questionable fairness of its central experimental comparison.

1.  **Limited Novelty:** The paper's primary contribution is applying a combination of existing techniques to a new domain. The "KLENT" algorithm is a straightforward combination of KL regularization, entropy regularization, and $\lambda$-returns. As the authors note in Section 3.1, the combination of KL and entropy regularization has already been theoretically analyzed in prior work (e.g., Vieillard et al., 2020 and [1]). Furthermore, $\lambda$-returns are a foundational technique in RL (Sutton, 1988). Thus, the paper is an application study, not a proposal of a new algorithm.

2.  **Lack of Domain-Specific Insight:** The proposed method is a general model-free RL algorithm. It doesn't seem to incorporate any specific inductive biases or improvements for two-player, zero-sum, perfect-information games, which is the entire domain of study. This makes it less of a direct competitor to AlphaZero, which is fundamentally built on MCTS as a policy improvement operator that leverages the game structure.

3.  **Flawed Core Comparison:** The paper's central claim of superior "efficiency" (e.g., in Figure 1 and 5) is based on a flawed premise. The main experiments evaluate all agents, including AlphaZero, *without* search at test time. AlphaZero is an algorithm *designed* to use search (MCTS) to plan; evaluating its learned policy network in a purely reactive way is not a meaningful measure of its performance. This setup inherently favors the model-free KLENT, which is trained specifically to produce a strong reactive policy. The paper's metric of "simulator evaluations" also penalizes search-based methods during training by counting MCTS rollouts, which is what they are *supposed* to do.

4.  **Misleading Scalability Argument:** The paper frames KLENT as a "resource-efficient" alternative. However, the strength of AlphaZero is its ability to scale performance with *computational complexity* (i.e., more search). While Appendix H attempts to show KLENT scaling with test-time search, it relies on a heuristic (using the policy-Q inner product as a value estimate) and the performance gains are minimal. This suggests the learned model is not well-suited for deep planning. The paper's own data in Table 6 shows that AlphaZero *does* eventually match KLENT's performance, it just takes a larger training budget. This implies KLENT is *sample-efficient* at learning a good-enough policy, but it is not a more scalable or asymptotically superior approach, which is the key to superhuman performance in this domain.

[1] Lee D. Entropy-Augmented Entropy-Regularized Reinforcement Learning and a Continuous Path from Policy Gradient to Q-Learning

### Questions
1.  The novelty of KLENT seems to be the *empirical* combination of KL-reg, entropy-reg, and $\lambda$-returns. Given that the paper cites prior work like Vieillard et al. (2020) that already analyzes the KL/entropy combination, could you please clarify what the *algorithmic* contribution is?

2.  Your main results in Figure 5 are based on evaluating AlphaZero "without search". How can this be considered a fair comparison when AlphaZero's entire design is centered on using search to improve its policy at decision time? Would not a comparison with equal *wall-clock* search time at evaluation be more appropriate?

3.  In Appendix H, you plug the KLENT-trained network into an MCTS. This relies on a heuristic $V(s) = \sum_a \pi(a|s)Q(s,a)$. How much does this heuristic limit the search? Does the small performance gain (Fig 11) not simply confirm that KLENT, as a model-free method, is not designed to scale with computational search in the same way AlphaZero is?

4. Table 6 shows AlphaZero eventually matching KLENT's peak performance, albeit with a 5x larger training budget. Does this not frame the paper's contribution as one of *sample efficiency* to a certain performance level, rather than a truly superior or more scalable algorithm?

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
The paper explores policy regularization problem for deep reinforcement learning algorithms applied to board games. Authors propose a KLENT algorithm with an actor-critic architecture that uses KLD-regularization and entropy regularization simultaniously. Proposed algorithm demonstrates state-of-the-art performance on board games including Go, Chess, Shogi and other games, outperforming AlphaZero and classic DRL algorithms.

### Strengths
- KLENT shows state-of-the-art performance on board games benchmark, outperforming AlphaZero and classic DRL algorithms.
- The proposed KLENT algorithm is relatively simple which allows for further improvements and application of key ideas in the future research
- A solid research on hyperparameter sensitivity for proposed algorithm.

### Weaknesses
- Ablation study is not conducted well enough. While it is present in the paper, there are no direct comparision of KLENT with ablated features and KLENT without ablation.

### Questions
**Questions:**
- Why did you choose board games as only benchmark and limit your study to it? It seems that your proposed algorithm can be applied to any environment that is suitable for actor-critic algorithms, as it do not rely on anything specific to board games.
- The algorithm seems to be similar to Soft Actor-Critic algorithm with additional KL-divergence regularization. Is there anything else aside from KLD regularization thet differs KLENT from Soft Actor-Critic? If not, it would be better to describe KLENT as SAC modification rather then an entirely novel algorithm.

**Suggestions:**
- It would be good to briefly describe in introduction or background section how algorithms designed for board games can generalize to some real-world applications. Such description will pinpoint an importance of this class of algorithms.
- In section 3.1 you describe various regularizations that were used in previous works, in which you claim that PPO uses only KL regularization, which is not correct. KL-regularized PPO is only one of two variants (another one is [PPO with clipping](https://arxiv.org/pdf/1707.06347) surrogate). Also, PPO uses entropy regularization for both its variants, which is also described in original paper. Please, reffer to this, to carefully check algorithm description provided in section 3.1.
- In figure 2 captions "state value" and "action value" is a bit confusing. It's better to change them to "value function" and "Q-function" respectively as it is aligns more with commonly used terminology.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a model-free reinforcement learning algorithm KLENT, which uses actor-critic archiecture. For training of the actor it uses both KL divergence and entropy regularization and for the training of the action-value critic it uses the $\lambda$-return. The authors admit that very similar algorithms have been proposed before and aim to empirically show that they work well in the domain of board games. The authors then compare their algorithm in 5 different games with several strong baselines including AlphaZeros trained actor. Moreover, authors provide ablation study to show the effect of each KLENT component.

### Strengths
* The policy update rule introduced in 4.1 is novel to the best of my knowledge and could be applied in more algorithms for decision-making that use KL divergence regularization (possibly outside perfect information games).
* The performance of KLENT is improvement over strong baselines like PPO and Gumbel AlphaZero (without rollouts in test-time).
* The detailed experimental section including ablation study and test on a large game (19x19 Go).
* When combined with MCTS test-time search, KLENT could be used to replace AlphaZeros training part.

### Weaknesses
The main algorithm is almost identical to [1] and [2], which are applicable to broader class of games. The main difference to those algorithms seem to be the new policy-update rule and that KLENT does not use the regularization policy update from [2]. However, the paper does not explain the relationship in detail nor does it provide direct empirical comparison. 

Compared to [2], which is off-policy, KLENT works only on-policy. However, the extension to off-policy KLENT seems plausible.

The strongest part of AlphaZeros performance is the additional test-time search (as evidenced by [3]). Even though KLENT could use MCTS in the test-time like the AlphaZero, which authors do in appendix H, it does break one of the strong points raised by the authors, namely "using simpler model-free algorithm to achieve similar performance as AlphaZero". I believe that KLENT would not be able to outpeform AlphaZero if AlphaZero used test-time search (and KLENT did not), given sufficient number of rollouts. Adding MCTS to KLENT could yield better performance than AlphaZero, but at the cost of being as complex as AlphaZero (possibly more complex, since the training and testing would be vastly different). I believe the paper would be stronger if it was formulated as an alternative to full AlphaZero, including search, with much better performance (due to cheaper training).

The KL regularization changes the equilibrium in the new regularized game. So it is not clear how far from the optimal strategy KLENT converges. I suppose KLENT would converge to some Quantal Response Equilibrium as in [1].


[1] Samuel Sokota, et. al. A unified approach to reinforcement learning, quantal response equilibria, and two-player zero-sum games. In Deep Reinforcement Learning Workshop NeurIPS 2022, 2022.

[2] Julien Perolat, et. al. From Poincaré Recurrence to Convergence in Imperfect Information Games: Finding Equilibrium via Regularization. Proceedings of the 38th International Conference on Machine Learning, in Proceedings of Machine Learning Research 2021, 2021.

[3] Ti-Rong Wu, Ting-Han Wei, and I-Chen Wu. Accelerating and improving alphazero using population based training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34,pp. 1046–1053, 2020.

### Questions
How difficult is it to make KLENT off-policy? Is it as simple as replacing n-step $\lambda$-return with Retrace?

Based on the similarities with [1] and [2], which are designed for imperfect information games, how difficult would it be to adapt the KLENT to imperfect information games?

Have you run KLENT with any smaller examples to verify KLENT converges to the optimal strategy?

In all of the games the training was ran for more than 350M simulator evaluations. How many networks were trained with each algorithm and what was the time spent on the training?

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper gives a simple model-free RL method for board games that eliminates look-ahead tree search during training. The motivation is to reduce the heavy computational requirements of search-based approaches like AlphaZero. The method uses a policy optimization approach with lambda returns.

### Strengths
The paper is clearly written, and reports convincing performance gains in learning efficiency. KLENT achieves higher win rates or faster training progress than heavy search-based algorithms (AlphaZero, Gumbel-AlphaZero) under the same computational budget. Environment and baseline is comprehensive, and hyper-parameter and implementation details are given. A key strength is the algorithm’s simplicity relative to AlphaZero-style methods. By avoiding MCTS, the method is much easier to implement and requires less specialized data structure.

### Weaknesses
It is important to note that the paper’s value may lie in the empirical finding that this straightforward combination works remarkably well on complex board games. Demonstrating that “model-free RL (with proper regularization) can rival search-based methods in these games” is a useful result for the community, especially for those who cannot afford massive search-based training. However, the lack of algorithmic novelty means the paper’s contributions are primarily empirical and engineering-oriented. To strengthen the paper as a research contribution, the authors would need to either provide new insights arising from this combination or propose an innovative modification. Currently, the success of the method, while notable, can be attributed to known techniques applied diligently rather than to a creative new idea introduced by the authors. That is, beyond the combination of existing very common (and nowadays standard) tricks, the approach does not introduce a fundamentally new mechanism or theory.

The related work discussion could be stronger in distinguishing this work from prior efforts. There have been other attempts at applying deep RL to board games (the authors cite AlphaZero alternatives and some resource-reduced versions). However, the paper does not cite any older approach that tried model-free learning in these games – possibly because prior to AlphaZero, non-search RL (like TD-Gammon for backgammon, or Atari methods applied to simpler board games) existed.

### Questions
A little speculation: the authors might consider whether a small amount of search at test-time could further boost the performance?

### Soundness
4

### Presentation
4

### Contribution
2
