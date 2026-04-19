# Sample Efficient Myopic Exploration Through Multitask Reinforcement Learning with Diverse Tasks

- Decision: Accept (poster)
- Scores: 6, 6, 8, 5

## Abstract
Multitask Reinforcement Learning (MTRL) approaches have gained increasing attention for its wide applications in many important Reinforcement Learning (RL) tasks. However, while recent advancements in MTRL theory have focused on the improved statistical efficiency by assuming a shared structure across tasks, exploration--a crucial aspect of RL--has been largely overlooked. This paper addresses this gap by showing that when an agent is trained on a sufficiently diverse set of tasks, a generic  policy-sharing algorithm with myopic exploration design like $\epsilon$-greedy that are inefficient in general can be sample-efficient for MTRL. To the best of our knowledge, this is the first theoretical demonstration of the "exploration benefits" of MTRL. It may also shed light on the enigmatic success of the wide applications of myopic exploration in practice. To validate the role of diversity, we conduct experiments on synthetic robotic control environments, where the diverse task set aligns with the task selection by automatic curriculum learning, which is empirically shown to improve sample-efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the use of multitask training to enable myopic (overly simplistic) exploration methods to discover solutions to difficult MDPs which would otherwise not be tractable to learn directly.

### Strengths
The core argument of this paper is salient and interesting, and seems to be well supported by the theoretical arguments. The experimental validation in a deep RL context is also appreciated in what is otherwise a theory paper.

### Weaknesses
I'm not well versed in recent theoretical/tabular RL exploration literature, so I can't speak very well to the novelty and significance there, but in a deep RL context this work seems relevant, but also very closely related to existing approaches such as goal-conditioned RL (Hindsight Experience Replay in particular) and to some extent automatic curriculum generation methods. 

I think these connections are very interesting, and this theoretical analysis isn't redundant with that work, but it does leave me feeling like this paper would me more interesting/have a stronger contribution if that connection were explored more. As it is I'm left feeling that while this specific argument is to my knowledge novel, it overlaps a lot with prior work. 

In addition, I also felt like the presentation of the paper needs some significant polish. There's some issues with grammar and odd phrasing throughout the paper, and I while I appreciate the intuition provided for various definitions/theorems I felt like I frequently lost the thread on those.

Overall, I think this is solid work that could be high impact, but it needs a little more polish to really shine. As such I'm inclined to recommend rejection, but I also admit that I don't have a good sense of the impact on the tabular RL exploration literature, so I will caveat that I can't properly evaluate that aspect and I will defer to other reviewers there.

### Questions
-While I generally follow the argument, this paper has some rough grammar and odd word choice in places. I'd recommend a thorough editing pass to improve the language.

-The explanation of the multitask setup in Section 2.1 confused me as to how the tasks are getting selected. Is there a structure or order in which tasks are chosen among M?

-Likewise Definition 1 is a little confusing. C is a function of beta and delta? I'm confused as to why sample complexity doesn't depend on either the algorithm itself or the task(s) being learned. The following paragraph seems to think C is a function of the MDP, but this isn't part of the definition.

-How does algorithm 1 differ from the cited Zhumabekov 2023 policy ensemble method? It seems like algorithm 1 is essentially an ensemble of policies, of which one is sampled for each episode?

-How does this idea of multitask myopic exploration differ from normal goal-directed RL methods like hindsight experience replay? The motivating example in Figure 1 seems roughly in line with such methods, and seems like it should share their limitations (e.g. large state spaces and low-dimensional manifolds of interesting/human-relevant tasks).

 -I find the term "myopic exploration gap" a little confusing. I understand the intuition (how much could a myopic exploration method improve upon the current best policy at any given time), but I wouldn't call that a gap. Maybe something like "myopic exploration potential?" A gap would imply it is comparing myopic exploration to another (optimal?) exploration algorithm. I know this term is coming from previous literature, but it seems confusing unless I'm misunderstanding the definition here.

-Doesn't PPO (like all on-policy policy gradient methods) have issues with epsilon greedy optimization due to it's off-policyness? How did you resolve this issue for the experiments?

-In the tabular case, multitask myopic exploration relies on coverage assumptions in the space of possible tasks (if I understand correctly), but it's not tractable to assume this in the deep RL case. Did this factor come up in the BipedalWalker experiments at all? Do the assumptions (mostly, at least) hold?

-Some more details/analysis on the deep RL experiments in the main paper would be appreciated, such as performance/training curves. I realize the focus of the paper is tabular/theoretical, but this topic has a lot of connections to methods used in deep RL (such as goal-directed RL and automatic curricula, as noted), and in my opinion exploring that connection further would be very interesting.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the statistical efficiency of exploration in Multitask Reinforcement Learning (MTRL). The authors show that when an agent is trained on a sufficiently diverse set of tasks, a generic policy-sharing algorithm with myopic exploration design like ϵ-greedy that are inefficient in general can be sample-efficient for MTRL. To validate the role of diversity, the authors conduct experiments on synthetic robotic control environments, where the diverse task set aligns with the task selection by automatic curriculum learning, which is empirically shown to improve sample-efficiency.

### Strengths
1.	The paper shows that when an agent is trained on a sufficiently diverse set of tasks, a generic policy-sharing algorithm with myopic exploration design like ϵ-greedy that are inefficient in general can be sample-efficient for MTRL.
2.	The paper is well-written and easy to follow.
3.	To the best of my knowledge, this is the first theoretical demonstration of the "exploration benefits" of MTRL, which is insightful for future research on efficient exploration in deep RL.

### Weaknesses
1.	The assumption that the task set is adequately diverse may be too strong in deep RL. Although the authors discuss implications of diversity in deep RL, it remains unclear to me. The authors may want to provide more insight into how to define and design a diverse task set for efficient exploration in deep RL.

### Questions
Please refer to Weaknesses for my questions.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the potential exploration benefits of multitask reinforcement learning from a theoretical perspective. This paper shows that when the set of tasks is diverse enough (measured by multitask MEG), a generic policy-sharing algorithm with myopic exploration is sample-efficient. Importantly, such myopic exploration is common in practice, and computationally efficient (unlike GOLF which requires solving nested optimization oracles). The paper also gives concrete examples of tabular/linear MDPs such that the diversity condition is satisfied. In the end, the paper validates the proposed theory with experiments and builds connections with curriculum learning.

### Strengths
- The general idea of this paper is novel and natural. Sample efficiency of myopic exploration is an important topic.

- The paper is very well-written.

- The theoretical results are sound.

- Discussion on limitations and comparison with prior works are adequate.

### Weaknesses
- The only weakness of this paper in my opinion is that examples where multitask MEG is bounded are too restrictive. Definition 7 is a very strong requirement, and intuitively, diverse tasks can be defined more general. Moreover, the feature coverage assumption is additional since it is not needed for learning linear MDPs with strategic exploration.

### Questions
- Is it possible to relax Definition 7?

- The offline learning oracle solves  $f_1,...,f_h$ simultaneously. Can you do them sequentially and have similar guarantees?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper claims that in the scenario of multitask-RL, a naive exploration strategy is enough. It formalizes their intuition in Def 3 and provide the theoretical guarantee in Theorem 1.

### Strengths
The proof seems to be sound.

### Weaknesses
1. It provides a possible explanation to explain the success of naive exploration in the case of multitask RL. However, it is hard to validate such explanation.

2. The intuition of the proof is that, if we have a base policy class with good coverage, we are able to find the optimal policy by combining the base policies with naive exploration. However, it have been well known that exploration is simple when we have good coverage. Therefore, their contribution seems not novel enough. 

[1]. Xie, Tengyang, Dylan J. Foster, Yu Bai, Nan Jiang, and Sham M. Kakade. "The role of coverage in online reinforcement learning." arXiv preprint arXiv:2210.04157 (2022).

### Questions
1. What does Def 3 mean in linear MDP?

2. Can you provide an example where Def 3 holds and the coverage of the base policies is poor?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
