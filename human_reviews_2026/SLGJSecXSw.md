# FPDou: Mastering DouDizhu with Fictitious Play

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
DouDizhu is a challenging three-player imperfect-information game involving competition and cooperation. Despite strong performance, existing methods are primarily developed with reinforcement learning (RL) without closely examining the stationary assumption. Specifically, DouDizhu's three-player nature entails algorithms to approximate Nash equilibria, but existing methods typically update/learn all players' strategies simultaneously. This creates a non-stationary environment that impedes RL-based best-response learning and hinders convergence to Nash equilibria. Inspired by Generalized Weakened Fictitious Play (GWFP), we propose FPDou. More specifically, to ease the use of GWFP, we adopt a perfect-training-imperfect-execution paradigm: we treat the two Peasants as one player by sharing information during training, which converts DouDizhu into a two-player zero-sum game amenable to GWFP’s analysis. To mitigate the training-execution gap, we introduce a regularization term to penalize the policy discrepancy between perfect and imperfect information. To make learning efficient, we design a practical implementation that consolidates RL and supervised learning into a single step, eliminating the need to train two separate networks. To address non-stationarity, we alternate on-policy/off-policy updates. This not only preserves stationarity for $\epsilon$-best-response learning but also enhances sample efficiency by using data for both sides. FPDou achieves a new state of the art: it uses a 3$\times$ smaller model without handcrafted features, outperforms DouZero and PerfectDou in both win rate and score, and ranks first among 452 bots on the Botzone platform. The anonymous demo and code are provided for reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors present FPDou, which achieves SOTA performance in DouDizhu using an extension of generalized weakened fictitious self play. In training, DouDizhu is treated as a two-player zero-sum game where the two peasants are treated as a team sharing perfect information. To extend the game to 3 players at deployment with full imperfect information, partial-observation feature extractors are trained for the peasants to have similar features as the perfect information encoders. These imperfect information features are then used in deployment. FPDou outperforms agents from prior work in skill.

### Strengths
The paper presents an impressive and difficult to achieve empirical result, attaining SOTA performance in DouDizhu.

### Weaknesses
While the main empirical result is very impressive, there are several issues in my opinion with the communication of the method. Revising the paper with clearer language would help address many of my concerns.

1. The presentation of the proposed approach is unclear. In line 203, it's claimed that a mixture of deep RL and supervised learning is used to learn a $\epsilon$-best response as well as the average policy, yet this supervised learning component of the optimization is never mentioned again. Where in Algorithm 1 or the loss functions in section 4 does the method ensure that an average policy is learned? How is the average-policy supervised learning facilitated, and what loss is used? The theoretical explanation in section 3.2 follows, but it isn't clear how the process in the last paragraph of section 3.2 is actually implemented, which seems to be a key detail.

2. The proposed approach to make partially observed features similar to fully observed features is a heuristic. The paper relies on an L2 feature regularize to “recover” the imperfect-info policy. This is a concise solution but somewhat ad-hoc. Unless there is a properly specific to DouDizhu that allows it, there’s no guarantee that an NE with imperfect-information is similar to an NE with teammate-shared perfect information. It would improve the paper if the authors could please clarify any convergence implications or limitations of using perfect-info training and imperfect-info execution.
   

I believe a missing limitation is that because the peasant policies are trained to only work with each other, they may not effectively cooperate well in ad-hoc peasant teams with other players/policies. (This limitation itself is not a weakness)

Also see Questions, which concern unclear aspects of the method.

### Questions
a) Unless I am confused, the distinction between on-policy and off-policy seems to misuse vocabulary. According to Algorithm 1, the off-policy Q-learning method is used in all stages, always drawing data from an (off-policy) replay buffer $\mathcal{D}$. To my understanding, at no point is "on-policy" RL ever actually used. Would a more appropriate terminology be something like off-policy learning with staggered opponent freezing?

b) The replay buffer size of 100,000 seems very small for keeping historical data from all past best responses. How are you ejecting data from this replay buffer? Are you using reservoir sampling like in NFSP [1]? How does this replay buffer ensure that you can produce an average policy over a long training history across many days?

c) Section E.4: What is "Fraction of off-policy data in each batch: 0.5"? I don't see explicit mention of this elsewhere in the paper. Are there two sources of data from which batches are constructed rather than the single replay buffer used in line 14 of Algorithm 1?

[1] Heinrich, J., & Silver, D. (2016). Deep Reinforcement Learning from Self-Play in Imperfect-Information Games.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces an algorithm for learning to play DouDizhu. It is an amalgamation of ideas for building a new SOTA.

### Strengths
The paper does an extensive evaluation of the proposed algorithm. There many tasteful practical choices made, from observing that the reward is discrete and using distributional RL to balancing training with win percentage.

### Weaknesses
To start with text itself, there are lots of choice of words that do not make reading the paper easier. 
- 038 surely advancements have not been introduced to the game itself
- 050 "this reduces DouDizhu to two-plahyer zero-sum" I think that a noun is missing
- The appendix conflates the literature on extensive form games and game theory. In particular (@ 899) game theory does not assume perfect recall. 
- The paper refers to sequence form strategies as strategies. 
- At this level, introducing Kuhn's theorem is odd and it is not clear why realization-equivalent (I think it should be equivalence) is introduced.
- The derivation of the GWFP (both in appendix and the main text) takes a lot of space just to make the point that sampling from the average sequence form policy is equivalent to first sampling a sequence form and then sampling from that sequence form policy.
- 1278 the sum diverges and does not converge to zero. 
- The paper if fixated on being theoretically correct and grounded in GWFP but conveniently ignores that the perfect training imperfect execution (PTIE) paradigm is clearly not safe. The paper that introduced PTIE claims that it is an extension of centralized training, distributed execution but this is clearly not the case, the value net (the central element of training) is not used in execution but the Q functions training with PTIE are distilled into the average policy.
- Going to the contributions of the paper, the only substantial contribution seems to be the off-policy, on-policy flags introduced in the loop
   -  GWFP has been used for this game, so has PTIE, and learning sequence form averages from replay bufer (cf PSRO)
- Twice, the paper claims that policy churns helps the exploration but policy churn is omnipresent everywhere so what makes DouDizhu special?

### Questions
Figure 12 a is the sampling strategy used to retrain a network from scratch? If so how long did the training take, how does the choice of temperature affects the WP?
Can you explain ADP again?
Table 2: Can the error be estimated with bootstrap?

### Soundness
2

### Presentation
2

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
This paper presents FPDou, a new RL framework designed to master the three-player imperfect-information card game DouDizhu. The paper adapts Generalized Weakened Fictitious Play to a deep RL setting, addressing the non-stationarity issue of multi-agent training. The authors convert the three-player game into a two-player zero-sum formulation by treating the two Peasants as a unified agent, and also apply alternative on-policy/off-policy updates and distributional Q-networks. Empirically, FPDou achieves SOTA performace with smaller models.

### Strengths
- The idea of the paper is overall well-motivated. The analysis of instability in simultaneous self-play is interesting, and the proposed solution effectively addresses this issue under the given setting, providing valuable insights for the research community.
- The empirical performance is strong — the paper achieves SOTA results using a smaller model, while maintaining a reasonable training cost.
- The writing is clear and easy to follow.

### Weaknesses
- The main obstacle to applying GWFP to the DouDizhu problem is its two-player zero-sum game setting, which the paper addresses by merging the peasants and adding a regularization term. However, simply aligning the latent representations with or without perfect information does not eliminate the need for perfect information and lacks a sound rationale. Therefore, FPDou is unlikely to satisfy the PTIE framework, since prior related work used only perfect information during policy evaluation.
- The explanation of the off-policy component of the framework is unclear. If a fixed opponent is required, why is off-policy learning necessary? Which algorithm is used for the updates? How does off-policy learning ensure the stability of the model?
- The paper sets a 0.5 win-rate threshold to ensure the ε-best response, which still introduces a degree of heuristics. There is also a potential risk that, in some iterations, the model may never reach a 0.5 win rate, thereby blocking training. This affects the method's generalizability. Although the authors mention an automated threshold adjustment process in the appendix, there is no noticeable difference in performance, which is somewhat counterintuitive and warrants further explanation.
- I still have concerns regarding the generalizability of the paper, considering that it solely focuses on DouDizhu and many design choices and findings are problem-specific. For instance, are there existing works combining GWFP with RL in other games? How can other multi-agent games be generally transformed into two-player zero-sum settings? Which components of FPDou’s design could provide insights or inspiration for researchers working on different tasks?

### Questions
- More information on addressing the training-execution gap should be provided, such as visualizations of the regularization term during training or more effective methods to prevent agent cheating.
- The motivation and methodology for off-policy learning need to be further explained.
- Related work on applying GWFP to other games, as well as more generalizable takeaways, needs to be added.

### Soundness
3

### Presentation
3

### Contribution
2
