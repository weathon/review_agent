# DAL: A Practical Prior-Free Black-Box Framework for Non-Stationary Bandits

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We introduce a practical, black-box framework termed Detection Augmented Learning (DAL) for the problem of non-stationary bandits without prior knowledge of the underlying non-stationarity. DAL accepts any stationary bandit algorithm as input and augments it with a change detector, enabling applicability to all common bandit variants. Extensive experimentation demonstrates that DAL consistently surpasses current state-of-the-art methods across diverse non-stationary scenarios, including synthetic benchmarks and real-world datasets, underscoring its versatility and scalability. We provide theoretical insights into DAL's strong empirical performance, complemented by thorough experimental validation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Detection Augmented Learning (DAL), a parameter-free, black-box framework for non-stationary bandits. DAL takes a stationary bandit algorithm and a change-point detection subroutine as inputs, and, through a forced-exploration mechanism, it adapts to non-stationary environments without requiring prior information. Extensive experiments on multiple benchmarks are conducted to validate the effectiveness of the proposed framework.

### Strengths
- This paper proposes a general framework for non-stationary bandits and establishes order-optimal regret guarantees in the piecewise-stationary setting. For the drifting case, the paper provides partial insights.
- The experimental evaluation is _thorough and diverse_, covering multiple bandit setups and including realistic datasets, which enhances the practical significance of the work.

### Weaknesses
- From a theoretical perspective, the main idea of augmenting a stationary bandit algorithm with a change-point detection module has been explored in prior work, limiting the conceptual novelty.
    
- Although the framework is claimed to extend naturally to contextual bandits, this case is not rigorously analyzed.
    
- The analysis for the drifting case remain limited, which constrains the overall contribution of the framework.

- Some assumptions, such as those in Proposition 4.2, require clearer justification or guidance on how they can be verified in practice.

### Questions
- How sensitive is DAL to the choice of covering set $A_e$ in large continuous action spaces?
- DAL depends critically on a GLR-type change detector, but the implementation specifics are not fully described, e.g., what is the exact testing statistics and threshold used for triggering restarts? How are the false alarms controlled?


I find the paper’s practical relevance to be stronger than its theoretical depth, and I would appreciate it if the authors could clarify the points raised above.

### Soundness
3

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
3

### Summary
For non-stationary bandits, most existing methods — such as restart, weighted/discounted, and sliding-window methods — can get good empirical performance and near-optimal regret guarantees. However, they rely on strong prior knowledge about the non-stationarity of environment. In contrast, MASTER achieves optimal regret without requiring such prior knowledge, but it is very complex: it runs many learners in parallel, which makes it hard to use in practice and often weak in experiments.

This paper focuses on the piecewise-stationary setting and proposes a black-box method that achieves (near-)optimal regret and strong empirical performance. The method keeps a small covering set of arms, occasionally pulls arms from this set to detect changes, and restarts the base learner when a change is detected. This removes the need to know the degree of non-stationarity and avoids maintaining many parallel learners.

### Strengths
1. The method provides an algorithm with theoretical guarantees that does not rely on prior knowledge of the environment, and it also shows strong empirical performance.
2. The method is general: it acts as a black-box change detector that can be wrapped around different types of bandit algorithms, and it works across multiple bandit settings.

### Weaknesses
1. The method does not provide theoretical guarantee for the drifting case. This is expected, because the change-detection mechanism is designed for abrupt changes, not for drifting changes. The paper only shows empirical performance on drifting, but bandits are primarily a theoretical setting, so having a matching optimal regret guarantee there is important and is currently missing.

2. Compared to MASTER, this paper’s analysis in the piecewise-stationary setting relies on an extra assumption:  changes in the environment must be separated by a sufficiently long stable period. This assumption appears inside Theorem 4.4, but it is not stated clearly as its own assumption. I suggest the authors make this assumption explicit and discuss it up front. Otherwise, the comparison to prior work (MASTER) is not fair, and the assumption feels too hidden.

### Questions
1. The paper repeatedly uses the broad term “non-stationary bandits,” but after reading the paper, the theory really only covers the piecewise-stationary case. For drifting, there is no matching theoretical analysis, but only experiments. By this standard, any prior piecewise-stationary bandit method could also run on a drifting simulation and then claim to solve “non-stationary bandits,” which would be an overclaim. Since the proposed method is not specifically designed for drifting, I believe the paper (including the title) should make it explicit that the setting is piecewise-stationary, not general non-stationary.

2. Prior work on piecewise-stationary bandits already has prior-free detection-and-restart methods. It is not yet clear to me what the real difficulty is in turning those approaches into a black-box wrapper, and how this paper goes beyond that in a substantive way.

I would be happy to raise my score if the authors can make the requested revisions and clarify these points.

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
3

### Summary
This paper focuses on a classical problem, that of learning in non-stationary bandits. The idea, essentially, is to augment a standard bandit algorithm with a "change detector". Classical bandit algorithms have their theory (and presumed applications) made under the stationarity assumption, which is not necessarily true in practice. Such changes can take the form of both abrupt and gradual changes. The authors propose a framework based on (1) detecting change of distribution by considering shifts in mean action rewards (2) forced exploration according to a schedule, forcing the bandit algorithm to essentially "drift" in state space. The mean-action shift is done by choosing an "appreciable" mean shift, exploiting some structure of the problem in deciding on which one. Some theoretical results on regret are provided.

### Strengths
Strengths: this is a nice problem, and one that has been considered by many authors over the years. The approach, while fairly simple, is effective. The experiments seem to be justifiable and demonstrate the performance of the method.

### Weaknesses
Weaknesses: the paper is not so easy to digest and understand at times. The tuning of the methods seems challenging, and the authors do not convince the reader otherwise. No details on the construction of the covering set are provided, as an instance.

Questions: what if the process contains a mix of abrupt and gradual changes? Can this method be augmented with memory, allowing to go back to previous regimes, instead of effectively starting from scratch every time?

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work focus on the regret minimization problem in non-stationary bandits. It proposed the DAL technique to detect unknown changes in the environment. Both numerical experiments and theoretical analysis are presented in this work.

### Strengths
1. Many related works are discussed.
2. Numerical experiments are done in various datasets.

### Weaknesses
This work presents a set of numerical results and a set of analytical results while neither of them fully convince me the superiority of the algorithm. I wonder what is the key contribution/focus of the work. Some key concerns are as below:
1. Abstract: It is claimed that 'DAL accepts any stationary bandit algorithm as input' while Propositions/theorems (e.g. Theorem 4.4) come with some assumptions/conditions. It is somehow confusing.
1. Line 28: It is claimed that 'MABs fall into ... PB, NPB, CB'. I feel maybe it is not that proper to say so. For example, contextual bandits can also be viewed as a parametric setting from some perspective.
1. Algorithm 1: I think the algorithm is a key contribution of this work, while the pseudocode is not that easy to understand.
  1. What is $N_e$?
  1. When will $D( \ldots )=\text{detection}$ (in line 6)?
1. Many subplots in Figures 1 and 2 present the regret/reward of only a portion of discussed algorithms? Do those missing algorithms perform better than DAL? An explanation is appreciated.
1. Proposition 4.2: It is a bit unusal that the Lipschitz constant $BL_u$ does not affect the bound on $|V_T|$. Some explanations are appreciated.
1. Theorem 4.4 comes many conditions/assumptions without discussions. Besides, how the regrets stated in the paragraph beginning from Line 414 is not that clear. Some explanations here are also appreciated.


Besides, here is one minor suggestion:
1. The algorithms should be arranged in the same order in the legend box for Figures 1 and 2.

### Questions
See *Weaknesses* above.

### Soundness
2

### Presentation
2

### Contribution
2
