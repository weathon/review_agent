# Achieving Sample and Computational Efficient Reinforcement Learning by Action Space Reduction via Grouping

- Decision: Accept (poster)
- Scores: 6, 6, 5, 6

## Abstract
Reinforcement learning often needs to deal with the exponential growth of states and actions when exploring optimal control in high-dimensional spaces (often known as the curse of dimensionality). In this work, we address this issue by learning the inherent structure of action-wise similar MDP to appropriately balance the performance degradation versus sample/computational complexity.  In particular, we partition the action spaces into multiple groups based on the similarity in transition distribution and reward function, and build a linear decomposition model to capture the difference between the intra-group transition kernel and the intra-group rewards. Both our theoretical analysis and experiments reveal a *surprising and counter-intuitive result*: while a more refined grouping strategy can reduce the approximation error caused by treating actions in the same group as identical, it also leads to increased estimation error when the size of samples or the computation resources is limited. This finding highlights the grouping strategy as a new degree of freedom that can be optimized to minimize the overall performance loss. To address this issue, we formulate a general optimization problem for determining the optimal grouping strategy, which strikes a balance between performance loss and sample/computational complexity. We further propose a computationally efficient method for selecting a nearly-optimal grouping strategy, which maintains its computational complexity independent of the size of the action space.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addressed the issue of explosive state-action spaces in high-dimensional RL problems. More specifically, it proposed to partition the action spaces into multiple groups based on the similarity in the transition distribution and reward function to reduce the size of the action space. Theoretical analysis shows that a finer representation reduces the approximation error resulting from grouping and also introduces higher estimate error when estimating the model. In addition, this work formulated a general optimization problem to select the grouping function.

### Strengths
- The proposed grouping strategy for high-dimensional RL problems is novel.

- This paper provides theoretical analysis of the proposed method and the conclusions are interesting.

- The writing is clear and easy to follow.

### Weaknesses
- Since this work focuses on MDPs with large action spaces that exhibit group-wise similarity, so it's application could be narrow.

- The current experiments are only conducted on toy tasks. So it's not unclear how would the proposed method perform in more complex tasks.

- Typos: "of each sate-group pair" ==>  state,

### Questions
- How would the proposed perform when compared with other kinds of action abstraction methods?

- I am not sure when do we need the proposed method. In which RL tasks do we specific need the proposed method to perform well?

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In large action space tasks, action grouping can make learning tractable. This paper formulates and objective to combine the performance loss due to grouping along with the computational and sample-efficiency benefits of grouping. They propose a practical approximation to this objective, that can provide a strategy for selecting near-optimal yet sample-efficient grouping strategies, altogether.

### Strengths
- The paper has a well-written flow and provides all the necessary prerequisities to understand the key contribution and distinction from the prior work.
- The novel insight of cumulatively considering the performance loss due to grouping with limited samples and compute is an important relationship to understand and subsequently help with selecting a strategy for action grouping.
- The practical algorithm makes the complexity of optimal grouping search to be proportional to the number of groups, which makes the problem tractable to solve.

### Weaknesses
- There is no empirical demonstration of the practical viability of the practical method to estimate the performance-complexity trade-off. Since this is proposed as a "practical" approximation to the original joint optimization objective, it should be shown how this enables one to select a better action grouping strategy in an RL environment.
- Extension to practical domains: How do the insights and theorems built on tabular value and policy iteration apply to deep RL? This is an important concern because, at the scale of large action spaces, such as those involved in recommender systems, deep RL is the only viable solution.

### Questions
- If doing the action abstraction online with RL, how important is the concern about the sample and computational complexity of estimating the abstractions? Wouldn't the sample efficiency be subsumed within the sample complexity of RL itself?
- If the groupings are learned, how can the lower-level policy be feasibly obtained using domain knowledge? With the group compositions potentially changing non-stationarily, the lower level policy might itself be very challenging to learn or provide with domain knowledge.

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies abstraction in reinforcement learning at the level of the action space. The authors propose to group similar action under an action grouping method, using this grouping method they can significantly reduce the size of the action space and lead to faster convergence.
The authors also analyze the loss induced by this grouping mechanism and show that it depends on the approximation error of the grouping method and the estimation error of the MDP induce by the grouping method.
It is then shown that it is possible to strike a balance between these two terms and that using a abstraction even if not perfect can lead to better performance.
Finally the author show how to optimize the grouping function to minimize this error.

### Strengths
This paper studies action abstraction, historically it has been less studies than state abstraction or abstractions over multiple timesteps.
Overall the paper is well written and the main ideas are explained in detail.
The main result of the Theorem 1 is also interesting and new as far as I known.

### Weaknesses
The notation used in the paper were often not precise and made it more difficult to understand the work presented:
- In section 3.1 it is not explained if the state space and action are discrete, finite or bounded.
Because of that is also unclear if |g| defined in section 3.2 can be infinite. In the same section for the definition of $\pi^i$ should it be $\{h\}$ instead of $h$?
- It would also be helpful to properly define the input of $\mathbb{P}_1$ and $\mathbb{P}_2$ to highlight that one applies on the original action space and the other applies on the lifted action space.
- How is K defined in section 3.3? I don't understand the notation used.

My big issue with this paper is that I do not understand how Algorithm 1 can lead to an almost optimal policy. It seems like an exploration term is missing to ensure a good coverage of the action / state space. Right now the algorithm could be stuck in a local minima while being able to accurately model the lifted MDP. Please let me know if I missed something

"However, the relationship between the performance loss and the goruping function is surprising. A grouping function with a larger number of groups can sometimes achieve better performance."
I don't think this is particularly surprising when this is the reason why we use abstractions! Similar results exisit in the context of state abstraction, when looking at finite time guarantees while abstraction introduces an approximation error it can also lead to much better performance. [1, 2]

Section 5.2, given that $\mathbb{P}$ and R are bounded isn't assumption 4 always verified?

[1] A unifying framework for computational reinforcement learning theory, Lihong Li
[2] Approximate Exploration through State Abstraction, Taiga et al. arxiv 2018

### Questions
Can you explain to me how Algorithm 1 is working without doing exploration?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes an approach to achieve better sample efficiency in high-dimensional RL. To that end, the authors find a low-rank representation of the high-dimensional MDP by partitioning the action space in groups according to the predictive capabilities of the transition and reward models. In particular, the presented approach reduces complexity (induced by the abstraction) while addressing the decline in performance due to abstraction errors.  For that, the authors build a linear decomposition model to capture these relationships and optimize over the grouping strategy to strike this trade-off.  The authors also establish (tight) bounds for the performance drop due to action grouping. Finally, the paper presents a general optimization problem that considers the trade-off between performance loss and complexity.

### Strengths
The paper proposes an interesting idea (group the action space according to the environment model) and addresses a challenging problem (high dimensional RL). The paper is also generally well written and states the details of the derivations clearly. The authors review the existing work in detail. The inclusion of theoretical results solidifies the claims made in the paper.

### Weaknesses
The paper can be improved (and the reviewer is happy to raise the score if these points are met) if the authors would provide some empirical results. It is yet not apparent if the ideas of the presented approach ramifies in practical advances. As applications, the authors already mention control systems with potentially millions of actions available at each step or recommender systems with large number of potential items. Another possibility would be to test on muscular systems [1], since action space grouping would be quite effective for tasks involving such systems.

In general, it would also help to understand the ramifications of the assumptions of this paper for potential applications. Can this method solve all possible RL tasks, or is a set of tasks excluded? In the paper, the authors write: “We specifically focus on MDPs with large action spaces that exhibit group-wise similarity, where only approximate grouping strategies of the action space are available”. Which particular tasks are meant in this sentence?

[1] [https://github.com/MyoHub/myosuite](https://github.com/MyoHub/myosuite)

### Questions
See weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
