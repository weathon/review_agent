# Prioritized Soft Q-Decomposition for Lexicographic Reinforcement Learning

- Avg Score: 5.67
- Decision: Accept (poster)
- Scores: 6, 5, 6

## Abstract
Reinforcement learning (RL) for complex tasks remains a challenge, primarily due to the difficulties of engineering scalar reward functions and the inherent inefficiency of training models from scratch. Instead, it would be better to specify complex tasks in terms of elementary subtasks and to reuse subtask solutions whenever possible. In this work, we address continuous space lexicographic multi-objective RL problems, consisting of prioritized subtasks, which are notoriously difficult to solve. We show that these can be scalarized with a subtask transformation and then solved incrementally using value decomposition. Exploiting this insight, we propose prioritized soft Q-decomposition (PSQD), a novel algorithm for learning and adapting subtask solutions under lexicographic priorities in continuous state-action spaces. PSQD offers the ability to reuse previously learned subtask solutions in a zero-shot composition, followed by an adaptation step. Its ability to use retained subtask training data for offline learning eliminates the need for new environment interaction during adaptation. We demonstrate the efficacy of our approach by presenting successful learning, reuse, and adaptation results for both low- and high-dimensional simulated robot control tasks, as well as offline learning results. In contrast to baseline approaches, PSQD does not trade off between conflicting subtasks or priority constraints and satisfies subtask priorities during learning. PSQD provides an intuitive framework for tackling complex RL problems, offering insights into the inner workings of the subtask composition.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the challenges of lexicographic MORL problems by introducing a novel approach called prioritized soft Q-decomposition. This technique leverages the value function of previously learned subtasks to constrain the action space of subsequent subtasks. The experimental results conducted on both simple and complex scenarios substantiate the efficacy of this method.

### Strengths
The paper is well organized and easy to read.
The proposed method, in its simplicity, manages to be effective in tackling the problem.

### Weaknesses
The proposed method appears to be sensitive to the parameter ε, and the manual selection of this parameter is non-trivial. 
Additionally, the paper falls short in providing a comparative analysis with existing lexicographic MORL algorithms.

### Questions
1. The paper employs equation 7 to approximate ε-optimality based on equation 1, but the relationship between the two equations is not very clear. Including more theoretical insights could enhance the paper's rigor.
2. It's worth pondering whether εi should be state-dependent. Is it possible to find a constant εi that precisely represents the task requirements? If not, it might indicate that the action space is overly restricted in some states, hindering exploration in subsequent tasks, or, conversely, that undesired actions cannot be excluded in certain states.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an algorithm called prioritized soft Q-decomposition (PSQD) to solve complex lexicographic multi-objective reinforcement learning (MORL) problems. In the setting, n subtasks are prioritized, and the available policy set is reduced as each subtask is optimized in the predefined order. Instead of explicitly representing the available policy sets, the authors consider restricting action space to satisfy the lexicographic order. For implementation, Soft Q-learning (SQL) is adopted to deal with continuous action space. Numerical results show that PSQD performs better than the other baselines in the considered environments, where there are two subtasks.

### Strengths
- The authors provide mathematical formulations on the main paper to support the soundness of the algorithm.
- This paper contains a dense appendix for detailed explanations.
- The authors provide extensive study on previous works.

### Weaknesses
1. Presentation needs to be improved. While the considered setting - lexicographic MORL - is clear, the flow of the algorithm is hard to understand. High-level pictograms can help the readers understand the content.

2. There is no pseudocode, so I am confused about the implementation of the algorithm. As far as I understand, the proposed algorithm is one of the following:

Candidate 1) For subtask 1 to n-1, run parallel SQL in equation (12). Then restrict action space satisfying epsilon-optimalty of subtasks 1 to n-1. In the restricted action set, run SQL for subtask n in equation (12) and recover optimal policy using (13).

Candidate 2) For subtask 1, run SQL in equation (12). Acquire restricted action space A_1.  Run (12) on A_1 and acquire A_2. ... After acquiring A_{n-1}, run (12) and (13) for subtask n.

Which one is right? Please provide a pseudocode.

3. For clarity of the setting, I want to raise several fundamental discussions regarding the lexicographic setting.
- When lexicographic MORL setting is considered in practice?
- What is the clear difference between constrained (MO)RL?
- Who decides the priority order? Is it valid to assume that we always know the priority order?
- How do orders of subtasks affect the final performance in experiments? If the order is crucial, discussion on setting order is important.
- What if there are tasks that are not prioritized (e.g., "equally" important, or "we do not know")?

4. Experiments deal with only two subtasks. Can the authors show another environment containing more than two subtasks?

5. For reproducibility, it would be better for the authors to provide anonymous source code.

6. There is confusion in eq. (7). Does Q_i in eq. (7) mean the optimally trained one (i.e., Q_i^*)?

### Questions
Please check the weakness part. Additional questions are as follows.

7. In number 2 in Weakness, if one of the candidates is PSQD, do the authors think that the other one is also a valid algorithm? (It looks like candidate 1 does not use order information of 1 > ... > n-1). 

8. If SQL is used, PSQD can be extended to discrete action space since the integral is changed to summation in equation 5. Then we may compare PSQD with the previous work of Skalse et al (2022). 
Also in that discrete action case, do authors think that action set refinement is still valid?

9. Confusing notation in eq (1). J_0, epsilon_0, Pi_0 is not explicitly defined.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a new methods to learn Reinforcement Learning policies that obey lexicographic subtask constraints. To make this efficient, the presented method creates zero shot Q-functions and strategies for the priority constrained task by composing transformed versions of the individual subtasks through their limitation to the indifference space of actions for the higher priority tasks and the transformation of the reward and Q value function to infinitely penalize such action choices. The resulting one shot version of each task value function and policy can then be adapted offline or online to achieve potentially near-optimal performing policies.
The main contributions in the paper are in the novel methodology and corresponding learning algorithm to form RL strategies by composing subtask functions such as to obey priority constraints expressed in the value function of the higher level tasks.

### Strengths
The paper tries to address two very important problems in Reinforcement Learning for control tasks in: i) providing an effective means of composing overall behavior from learned subtasks without significant need for new data collection and re-learning, and ii) to enforce strict priority constraints during composition as well as subsequent learning to optimize from the initial one-shot policy. These abilities are very important in the area of robot control in order to be able to enforce learned safety and performance constraints when new task compositions have to be learned.
The approach seems overall sound (although the description is lacking in a few places) and the results demonstrate that the method can form both single-shot strategies and more optimized policies using additional off-line (or on-line training) training that maintains the highest priority constraint.

### Weaknesses
The main weaknesses of the paper are in a lack of discussion of the full range of situations in the presentation of the underpinnings of the framework and in the incomplete description of the experiments presented in the paper.
The former here seems the biggest weakness and mainly relates to the complete absence of a discussion and consideration of cases where subtasks might not be compatible and thus the indifference space of actions might become empty. It would be very important for the authors to discuss this situation and how the algorithm would react under those circumstances. This is even more important as in the description of the decomposed learning task only the highest and the lowest priority subtasks are discussed  (where in the case of contradictory tasks, the lowest priority task would no longer have a usable actions space and could therefore not have a policy). 

In terms of the experiment presentation, it would be very useful if the main paper could contain at least a basic description of the environment, the action space, and in particular the subtasks and corresponding reward functions. For the latter (the reward functions), even the Appendices do not seem to contain more than an rough description of the principles of the reward function of the obstacle subtask. It would be very important for reproducability but also for a better understanding of the reader if the authors were to include the exact reward function for each subtask as well as the \epsilon thresholds that were used for the experiments presented in the paper (and these should be in the main paper).

Another slight weakness is that while the paper indicates that the pick of thresholds \epsilon is difficult, it does not provide any analysis of this. A brief ablation in term of \epsilon for the obstacle task in the 2D navigation experiment would have been very useful, as would be a brief discussion how such thresholds might be picked and what the tradeoffs of different picks are.

### Questions
The main questions arise out of the weaknesses stated above:
How does the proposed approach deal with incompatible subtasks ? Does it simply eliminate all tasks with empty indifference action sets for higher priority tasks and then operate in the same way as presented in the paper ?

How sensitive is the approach to the specific choice of \epsilon thresholds ? Is there a way that an ablation could be performed that would investigate the sensitivity of the top priority task's threshold in the navigation experiments ?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
