# Generalize and Guide: Decomposing Rewards for Few-Shot Inverse Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Inverse reinforcement learning (IRL) provides a powerful framework for learning from demonstrations. However, many realistic tasks include natural variations (i.e. a cleaning robot in a house with different furniture configurations), making it impractical to provide enough demonstrations to fully specify the task in every scenario. We tackle the problem of few-shot IRL with multi-task demonstrations, where an agent must learn a new task from limited demonstrations by leveraging data from other related tasks. Unlike prior methods that rely on expensive meta-training or are restricted to offline imitation, our approach learns a reward function that can be directly optimized through online interaction. We introduce Multi-task discriminator Proximity-guided IRL (MPIRL), a novel method that learns a generalizable and informative reward function for effective few-shot IRL. Our key insight is to decompose the reward into two components: (1) a multi-task discriminator that recognizes and rewards expert behavior in different task variations, and (2) a dense, proximity-to-expert reward that guides the agent in non-expert states.  This composite reward structure enables effective policy optimization even when demonstration data is limited. We demonstrate the effectiveness of our method on multiple challenging navigation and manipulation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Multitask Discriminator Proximity-Guided IRL (MPIRL), an approach designed for effective few-shot IRL. Specifically, the authors design an approach for a setting in which an agent is provided with a large, multi-task demonstration dataset, a limited number of demonstrations for a new target task, and access to the target environment for online learning. MP-IRL produces a reward function that 1) recognizes expert behavior across intra-task variations and 2) provides a learning signal that guides the agent toward expert states when it deviates out of distribution. The authors explain how they build on previous work (GAIL) to adapt to the multi-task, few-shot setting and produce the aforementioned dual-component reward function. The authors find that their approach outperforms several other techniques, including BC, GAIL, SQIL, DVD, and PEMIRL, achieving higher performance on target tasks and faster learning than baselines.

### Strengths
+ The paper studies an interesting and realistic problem.
+ The illustrative example is intuitive and beneficial in understanding the paper.
+ The results section provides a multifaceted analysis of MPIRL and sufficient evidence that MPIRL outperforms prior baselines.

### Weaknesses
- It isn't clear why this framework should outperform meta-learning frameworks. Could the authors provide further justification regarding this and the poor performance of PEMIRL during test time?
- It is unclear how MPIRL works during test time. Could the authors provide some information about how MPIRL works during deployment?

### Questions
1. Could you describe how the task demonstration $\tau$ is encoded into the discriminator and how the target task is encoded during online learning?
2. Could the authors provide further information on why the triangle inequality is important?
3. Please address the weaknesses noted above.

### Soundness
3

### Presentation
4

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
This paper proposes MPIRL to leverage multi-task demonstrations to enhance imitation learning when the number of same-task demonstrations is limited. The proposed approach combines a multi-task discriminator with a constraint-based proximity function to produce a dense reward signal.

### Strengths
The motivation and main idea of this paper is good.

### Weaknesses
- Something in the tables and figures is confusing. "X" in Table 1 seems to mean "no" but in fact indicates "yes". $\max p(s_t)$  in Figure 2 sounds like trying to push the agent trajectory away from the experts, while I can understand it means $\max [p(s_t)-p(s_{t+1})]$.
- In the FactorWorld experiments, the authors use significantly more interaction steps than prior work. Notably, in the early stages, their method does not noticeably outperform the baselines. As the number of steps increases, the baseline method generally saturates, but the proposed method continues to increase its success rate. This suggests it would be worth investigating the performance of the proposed and baseline methods when combined with stronger exploration strategies. Moreover, if the method only works with such a high interaction cost, its practical potential may be limited.

### Questions
- In Section 3, all tasks "share the same state and action spaces". How do the authors understand the state space? For example, if the state is RGB image, much more tasks besides FactorWorld could potentially share the same state space once their actions are aligned. In this case, the authors’ method could indeed leverage these additional tasks. However, if the state is defined in terms of physical object properties, different tasks in FactorWorld involve different objects, making it unclear how they could truly share the same state space. This raises the question of whether the assumption of a shared state space is realistic.
- In Figure 6, it seems that the number of additional tasks providing expert demonstrations does not significantly affect the results. Does the method really leverage information from other tasks? It appears that the experts and rollout trajectories from other tasks are mainly used for adversarial training. Would using non-expert data from the main task as “expert” videos for some auxiliary tasks yield similar performance?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies few-shot inverse reinforcement learning (IRL): an agent learns a new task, utilizing only a handful of target demonstrations alongside a large multi-task demonstration dataset and online environment access. The paper proposes to decompose the reward function into a multi-task discriminator, which can leverage the data from prior tasks, and a proximity reward, which provides a dense signal estimating proximity to expert states. Experiments show that the proposed method outperforms the selected baselines.

### Strengths
- Learning from multi-task data is a relevant topic in the RL community. The paper helps fill an unexplored setting (IRL with multi-task data) in the literature.
- The overall presentation is clear. The proposed method is intuitively explained and easy to understand.

### Weaknesses
Key concerns regarding the experimental validation:
- Most online baselines fail to outperform simple BC. Given that GAIL and SQIL are relatively older works (at least 5 years old), they may not represent the current state-of-the-art in online imitation learning. I personally suggest evaluating MPIRL against more recent single-task IRL approaches, such as [1] + BC loss (which also learns a proximity function) or ROT/RDAC [2].
- All experiments are conducted in state-based environments. Additional experiments in visual-based settings would better demonstrate the method's robustness in more complex environments.

[1] Lee, Youngwoon, et al. "Generalizable imitation learning from observation via inferring goal proximity." Advances in neural information processing systems 34 (2021): 16118-16130.

[2] Haldar, Siddhant, et al. "Watch and match: Supercharging imitation with regularized optimal transport." Conference on Robot Learning. PMLR, 2023.

### Questions
- Is the implementation of "GAIL + Proximity" in Figure 15 exactly equivalent to MPIRL when the prior task data is excluded? (E.g., does its discriminator also take $\tau$ as input?) If not, could you provide the ablation of MPIRL vs MPIRL w/o prior task data? This would be very helpful for understanding the impact of multi-task data.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes MPIRL that combines the idea of meta learning and IRL, featuring learning to learn in an inverse RL setup. The core idea is to learn a two-part reward function that consists of 1) a discriminator predicting whether a state-action pair is considered expert conditioned on a particular demonstration and 2) a proximity based reward from the expert.

### Strengths
+ The paper is well written and motivated 
+ The presented approach is simple and clear, evaluated on a wide range of domains.
+ Strong empirical performance over the baseline, and comprehensive ablation.

### Weaknesses
- Why is the proximity function independent of the target task? Given two subsequent states, aren't there tasks such that under one the policy gets closer to the expert but under the other it's the opposite?
- While the presented approach seems sounded, I highly recommend the authors to add a algo-box to help the readers understand the presented approach quicker.
- Can the authors also compare to multi-task max-entropy IRL?

### Questions
My main question is on the input of the proximity function. I.e. why is the proximity function independent of the target task? Given two subsequent states, aren't there tasks such that under one the policy gets closer to the expert but under the other it's the opposite? 

Also, can the authors provide a learning curve where RL gets the oracle reward, to help understand how close the presented approach is with the performance upperbound?

### Soundness
2

### Presentation
3

### Contribution
3
