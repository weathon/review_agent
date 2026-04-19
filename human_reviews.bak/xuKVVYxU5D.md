# Single-Trajectory Distributionally Robust Reinforcement Learning

- Decision: Reject
- Scores: 3, 3, 6, 8, 6

## Abstract
To mitigate the limitation that the classical reinforcement learning (RL) framework heavily relies on identical training and test environments, Distributionally Robust RL (DRRL) has been proposed to enhance performance across a range of environments, possibly including unknown test environments. As a price for robustness gain, DRRL involves optimizing over a set of distributions, which is inherently more challenging than optimizing over a fixed distribution in the non-robust case. Existing DRRL algorithms are either model-based or fail to learn from a single sample trajectory. In this paper, we design the first fully model-free DRRL algorithm, called distributionally robust Q-learning with single trajectory (DRQ). We delicately design a multi-timescale framework to fully utilize each incrementally arriving sample and directly learn the optimal distributionally robust policy without modeling the environment, thus the algorithm can be trained along a single trajectory in a model-free fashion. Despite the algorithm's complexity, we provide asymptotic convergence guarantees by generalizing classical stochastic approximation tools. Comprehensive experimental results demonstrate the superior robustness and sample complexity of our proposed algorithm, compared to non-robust methods and other robust RL algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work develops a three time-scale algorithm, aiming to solve the robust RL problem with a single trajectory with an online fashion. The work is important considering the current works in the area.

### Strengths
1. The idea is interesting. To solve the unbiased estimation issue, another time scale is introduced. I believe this idea is interesting and novel.
2. The experiment results are promising.

### Weaknesses
1. The presentation of the results is infusing and can be improved. E.g., there are too many assumptions made to imply the results. If this work is focusing on the three time-scale stochastic approximation framework itself, it is OK to make these assumptions; But this works is for a concrete problem, i.e., DR-RL, I believe it should not be reasonable to make all these assumptions. 
2. As mentioned, with so many assumptions made, I doubt the soundness of the convergence results. E.g., how can I know that under the DR-RL problem, the associated ODE has a unique global asymptotically stable equilibrium?

### Questions
1. The work use gradient descent to solve the support function. Why does this DRO problem can be solved by GD? Why is this problem smooth? Why does the gradient exist? I didn't see this gradient descent approach much used in previous DRO works.    
2. How do you justify the assumptions you made?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors proposed a distributionally robust variant of Q-learning aiming to solve distributionally robust MDP in an online fashion. Their algorithm utilizes a three-timescale stochastic approximation framework and possesses an almost surely convergence guarantee. They conduct thorough experiment illustrating the performance of their algorithm.

### Strengths
They extend the classical two-timescale stochastic approximation framework into the DRO problem, and design an online algorithm for DRRL. Comprehensive experiments are conducted to illustrate the performance.

### Weaknesses
The writing is problematic. I find the paper hard to follow. There are many notations/terms lack of definitions and several critical statements lack of discussion and explanation. Moreover, there are typos and factual errors in this paper, which make the results not credible. All in all, it's hard to verify the validity of this paper.

### Questions
1. What does the misspecified MDP in the paper mean?
2. Badrinath & Kalathil (2021) and Roy et al. (2017) didn't use R-contamination model. Their uncertainty set is a general one and covers yours. Both of them study the online setting and present asymptotic results. Why don't you compare with their works in experiment and compare their theoretical results with yours.
3. Why the first equation on page 5 is true? 
4. On page 6, why keeping the learning speeds of $\eta$ and $Q$ different can stabilize the training process?
5. The three-timescale regime used in algorithm 1 is complicated and provided with limited explanations. Why and how it works? 
6. In the proof, there are more than 10 assumptions and most of them don't have any explanation. Why do they hold in your case? Are they necessary? Are they reasonable in practical problems?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper considers a distributionally robust RL problem. A model-free online TD Q-learning type method is developed to find the robust Q values without relying on a simulator. Specifically, a three-timescale framework is introduced to approximate the robust Bellman equation and asymptotic analysis is provided. The main algorithm is validated through a tabular RL example; a more practical DQN type algorithm is introduced to handle more complicated examples.

### Strengths
- This paper considers distributionally robust MDP using f-divergence as the uncertainty set, which is novel.
- The motivations of not using SAA to approximate robust bellman equation are clear.
- The reasons of not using multilevel Monte-Carlo method are clear.

### Weaknesses
- I guess the paper was written in parallel with the Panaganti 2022 (Robust offline) paper. However, as the Panaganti 2022 paper is published for over 6 months, it is better to compare and discuss the difference choices of uncertainty sets and problem settings. The authors claim that this paper is the first model-free DR RL paper in the literature, which is not true as the Panaganti 2022 is also model-free. To some extent, their paper considers a harder offline problem while this paper considers an online version. 
- I don't think this is an issue when evaluating the contribution of this paper. However, it is better to state the contribution according to the latest literature (indeed, RFQI was mentioned in the experiment) and explicitly comment on the differences. 
- The proposed algorithm 1 only works in the tabular setting. This is fine. The authors introduce a more practical algorithm using DQN type of learning scheme, which is provided in algorithm 4 in the Appendix. However, algorithm 4 looks almost the same as algorithm 1. Is this paper an older version? If yes, please provide the full algorithm 4 description in the future. More importantly, I believe algorithm 4 could potentially have a greater impact on DRRL problems; its connection and modifications with algorithm 1 could be further explained in details in the main paper.

### Questions
- In my opinion, the combination of online RL with DRO is a bit weird. It makes more sense to study offline RL with DRO. The question is that why would you prefer to learn a robust policy in the testing environment, i.e., this robust policy is not optimal in the testing environment. Is it because there are some technical challenges when applying f-diverngence to offline dataset, e.g., when no e-greedy policy is allowed? 
- Same to the weakness, please compare with Panaganti 2022 in details and explicitly.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work designs model-free algorithms for distributionally robust RL problems in a sample-efficient manner. It proposed a three-time scale algorithm that solves a class of robust RL problems using the uncertainty set constructed by the Cressie-Read family of f-divergence. A theoretical asymptotic guarantee has been provided. Moreover, this work conducted experiments to evaluate the performance and sample efficiency of this work.

### Strengths
1. It targets an interesting problem: design a model-free algorithm for distributionally robust RL problems.
2. A three-timescale algorithm has been proposed that enjoys an asymptotic guarantee and practical sample efficiency.
3. The introduction of the algorithm is clear and easy to follow.

### Weaknesses
1. For the experiments in Figure 5. It seems the proposed algorithm DDQR has a very similar performance compared to the existing robust algorithm SR-DQN. It will be helpful to add more discussion about this.
2. As mentioned in the algorithm, the three-timescale serves as a key role in the algorithm to ensure convergence. So it will be better to introduce what is the three learning rates that the practical algorithm uses. And ablation study using different learning rates will give a message about whether the algorithm is sensitive to the learning rate.

### Questions
1. As the update of $\eta$ is independent for each $(s,a)$ pair, will the proposed algorithm works for $s$-rectangular cases?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a novel distributionally robust reinforcement learning (DRRL) algorithm which is model-free and uses single trajectories to update Q function estimates and compute the robust policy. The approach consists of a multi-scale approximation scheme that utilizes existing stochastic approximation results. Asymptotic convergence is proven. Moreover, the method is evaluated experimentally on two tabular environments and a DQN-based implementation is tested on larger (and widely used) control tasks.

### Strengths
- Distributionally robust reinforcement learning is a relevant and active area of research. While a few methods have been already proposed, the paper is novel in that it considers a model-free setting, without assuming access to a simulator but only to single trajectory data. I believe this is significantly more practical and makes a good contribution.

- A reasonable amount of experiments illustrate the features of the proposed approach, compared to existing DRRL baselines.

- The paper is nicely written and the results look sound, although i could not verify their proofs.

### Weaknesses
- Only asymptotic convergence is proven, and no sample-complexity guarantees. Do the authors have a guess on how these may compare with previous work, e.g. (Panaganti et al. 2022)?

- In Section 4.2, a practical implementation of DQR is utilized for the experiments. How is this implemented? I would be nice to discuss such implementation in the main text.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
