# Offline Preference-Based Value Optimization

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
We study the problem of offline preference-based reinforcement learning (PbRL), where the agent learns from pre-collected preference data by comparing trajectory pairs. 
  While prior work has established theoretical foundations for offline PbRL, existing algorithms face significant practical limitations: some rely on computationally intractable optimization procedures, while others suffer from unstable training and high performance variance.
  To address these challenges, we propose Preference-based Value Optimization (PVO), a simple and practical algorithm that achieves both strong empirical performance and theoretical guarantees.
  PVO directly optimizes the value function consistent with preference feedback by minimizing a novel \emph{value alignment loss}.
  We prove that PVO attains a rate-optimal sample complexity of $\mathcal{O}(\varepsilon^{-2})$, and further show that the value alignment loss is applicable not only to value-based methods but also to actor–critic algorithms.
  Empirically, PVO achieves robust and stable performance across diverse continuous control benchmarks. 
  It consistently outperforms strong baselines, including methods without theoretical guarantees, while requiring no additional hyperparameters for preference learning.
  Moreover, our ablation study demonstrates that substituting the standard TD loss with the value alignment loss substantially improves learning from preference data, confirming its effectiveness for PbRL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To address the issues of high computational complexity and instability in PbRL, this paper proposes an optimization objective named the value alignment loss and validates the effectiveness of the proposed method both theoretically and experimentally.

### Strengths
1) Through experiments, the paper verifies that PVO outperforms other methods for most tasks.

2) The paper provides sufficient theoretical derivation and analysis, including a derivation of the algorithm's computational complexity.

### Weaknesses
1) Algorithmic Aspect: The algorithm is essentially consistent with IQL [1], showing limited innovation. IQL minimizes $(R+\gamma V-Q)^2$, while PVQ is equivalent to minimizing $(\sum_l (Q_l-\gamma V_l-R_l))^2$.

2) Theoretical Aspect: The theoretical analysis is similar to that in paper [2], offering limited contribution.

[1] Offline Reinforcement Learning with Implicit Q-Learning, ICLR, 2021.

[2] Provable offline preference-based reinforcement learning, ICLR, 2024.

### Questions
1) Motivation: Before Definition 1, the authors discuss learning a value function consistent with preference feedback. How is this reflected in the proposed value alignment loss?

2) IQL Parameters: The framework is similar to IQL. For a fair comparison, what is the performance of IQL when its advantage weight parameter $\beta$ is set to be consistent with PVO?

3) Transition Model: In the practical deployment described in Section 3.4, the environment model does not seem to be used. What, then, is the purpose of training the transition model?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work addresses a critical gap in offline preference-based reinforcement learning: the tradeoff between theoretical guarantees and practical usability in existing methods. By proposing Preference-based Value Optimization, which uses a value alignment loss unifying value-based and actor-critic PbRL with rate-optimal guarantees, it delivers a unified solution that excels both in theory and experiments.

### Strengths
1. The work targets the well-documented tradeoff in existing offline preference-based RL methods, where theoretically rigorous approaches are often computationally intractable, and practical implementations suffer from suboptimal sample complexity or training instability.
2. One advantage of the work's insight is its shift from the standard offline PbRL paradigm (first infer a reward model, then train the value function using that reward directly) to a different viewpoint: it anchors value function learning to preference alignment via an "induced reward" derived from the value function itself, rather than treating the reward model as the sole driver of value training.

### Weaknesses
1. The first weakness is the paper’s inconsistent formatting of mathematical formulas, specifically, the absence of terminal punctuation in Line 318.
2. The paper’s baseline set is limited in scope. Several existing works in PbRL use generative models, like trajectory generative adversarial networks and diffusion models for preference modeling, to infer preference-aligned behavior without relying on intermediate reward models. These should also be included.

### Questions
See weaknesses.

### Soundness
3

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
This paper studies offline preference-based reinforcement learning and introduces Preference-based Value Optimization (PVO). The method directly aligns the learned value function with the reward inferred from human preferences through a novel value alignment loss, ensuring consistency between value estimation and preference supervision. The authors provide theoretical guarantees and strong empirical results, showing that PVO achieves stable and competitive performance across continuous-control benchmarks.

### Strengths
- The proposed value alignment loss is conceptually sound, novel, and powerful enough to achieve strong results.

- The empirical evaluation is comprehensive and consistent, demonstrating that PVO outperforms existing preference-based approaches across multiple continuous-control benchmarks.

- The paper is clearly written and well-organized, offering valuable intuition on how human preference signals can be effectively integrated into value-based offline reinforcement learning.

### Weaknesses
- The reward learning module closely follows standard preference-based MLE approaches and thus contributes limited novelty in this part.

### Questions
- Line 3 of Algorithm 1 differs from Eq. (4); there appears to be a sign inconsistency (the “+” and “−” symbols might be reversed).
- The proposed value alignment loss appears conceptually related to the value function inconsistency introduced in VIPO: Value-Inconsistency Penalized Offline Reinforcement Learning. A discussion clarifying the connection or distinction between these two ideas would be appreciated.

### Soundness
4

### Presentation
3

### Contribution
3
