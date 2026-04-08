## Human Reviewer 1

### Summary
The paper proposes a new quasi-metric for reinforcement learning state representations. This quasi-metric captures the minimum number of time-steps taken to get from one state to another. They introduce two sampling based mechanisms using a combination of three losses to learn a state representation that captures the geometric structure under this quasi-metric. They show results from 3 different environments where their method performs better.

### Strengths
The work is largely well written and does a good job explaining their choice of quasi-metric. They also show that the manner in which they define the minimum action distance, $d_{MAD}$ leads to a unique function. They benchmark with respect to QRL.

### Weaknesses
I believe the main weakness is that the empirical evaluation is limited and for what empirical evaluation they have presented I dont see a significant gain fro this added complexity in the algorithm.

In the QRL [1] paper they benchmark their results on continuous control environments and also massive 2d mazes. Here the authors have benchmarked the results on PointMaze, OGBench PointMaze, CliffWalking, KeyDoorGridWorld, and NoisyGridWorld. While I see the value in making these choices and thank the authors for explaining these choices I believe they should also validate these results on control environments like FetchReach and FetchPush from control and image observations.  A broader evaluation set would be more convincing.

I also believe that the benefits from implementing a complex algorithm, with three balanced losses and multiple hyperparameters, is not very significant on environments other than the KeyDoorGridWorld env, compared to QRL. While they do provide a quantitative measure RatioCV I would appreciate a qualitative plot such as the one described in Figure 1 but for the learned representation in one of the test bed environments.

I also believe you are learning an abstractions over the state space. The cartoon in figure 1 of your paper matches the abstraction based representation that Allen et al [2] learn in their Figure 1. It might be helpful to compare with related work in the space of state abstractions. Your ReLu based loss also has some similarity to their ReLu based loss (see Section 5 of their work), although their formulation is more transition kernel based.

Minor issue:

Line 295: There is no $s_j$ in this equation but you are including it in the sampling subscript.

------------------------

Refrences:

[1] Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning, Tongzhou Wang, Antonio Torralba, Phillip Isola, Amy Zhang

[2] Learning markov state abstractions for deep reinforcement learning, Allen, Cameron and Parikh, Neev and Gottesman, Omer and Konidaris, George

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
The focus of the submission is to learn the minimum action distance (MAD), defined as the minimum number of actions required to transition between states. The submission argues that the prior work on learning the minimum action distance relies on symmetric distance metrics to approximate the MAD, while submission’s proposal supports the use of asymmetric distance metrics (or, quasimetrics).

### Strengths
See below.

### Weaknesses
The paper states that it tries to learn a distance function between pairs of states that can later be used by an RL agent to learn more efficiently. I believe then the paper tries to learn some sort of model of the environment which can be used by the RL agent. However, there is no mention of model based reinforcement learning or even comparison against methods in model based RL. If the goal of the paper was to learn a model of the environment, why not compare against these methods and provide insights that match the description made in the background.

There are also several claims in the introduction regarding the definition of the metric and how this metric will accelerate learning or make the policy generalize better. These claims are not substantiated in the submission. There are no experiments either showing acceleration in learning or generalization. There are only correlation results, where the prior method QRL seems to perform either better or comparable to the proposal of the submission.

I think one problem with the paper is that it doesn’t necessarily communicate what is exactly about and which category the paper belongs to. This makes the paper less accessible. 

I also think that it is critically problematic to propose a different environment than prior work and test newly proposed algorithms solely in the different environment. For more robust and transparent comparison I would have liked to see comparison in the environments that prior work used.

Another problem is that QRL [3] uses a different network in the original paper. The submission reports that they changed the original QRL to neural network state embedding to 512 - 512 - 128 and neural network IQE projector 128-512-2048. Why has the original implementation been changed? 

Again Hilbert representation models [2] also use a completely different network. Couldn’t it be the difference we see in the comparison between prior work and the proposal of the submission is simply due to the network changes? This kind of comparison is not reasonable to verify the claims made in the submission.

Another concerning thing is that the size of the network used for TDMadDist and MadDist is simply larger than both of the algorithms that are being compared to. This could highly affect the results and would be biased towards demonstrating the proposed methods of the submission performing better. I would like to see how it performs with the exact same networks.

Results in Figure 4 does not seem to have std for MadDist. The std of the TDMadDis seems to be quite high.


[1] METRA: Scalable Unsupervised RL with Metric-Aware Abstraction, ICLR 2024. 

[2] Foundation Policies with Hilbert Representations, ICML 2024.

[3] Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning, ICML 2023.

### Questions
See above.

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
2

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes an effective similarity metric, Minimum Action Distance (MAD), to capture the underlying structure of an environment.
Based on MAD, a new framework to measure the similarities between the states is as follows. An agent (with an initialized policy) collects state trajectories from an unknown environment, which are used to learn a state embedding that implicitly defines a distance function between states.

### Strengths
1. The definition of MAD in equation (1) conforms to the fundamental property of a type of distance.
2. It is an interesting idea to define a new distance between the states by the collected trajectories.
3. When it is hard to get the reward signals, the new framework can be seen as unsupervised or semi-supervised reinforcement learning to optimize the policy.

### Weaknesses
1. The source code of the new algorithm is not open-sourced.
2. The algorithm for MAD and the time complexity analysis should be given.
3. The experimental environments are simple and monotonous. All of them are maze problems.

### Questions
1. Is the new framework finding the optimal action path by minimizing the distance between the starting and ending points in the environment？
2. How to initialize a policy to collect trajectories？

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper focuses on state representation learning in RL. The paper presents a self-supervised approach to learn an embedding space in which the quasimetric distance between embedding pairs reflects the Minimum Action Distance. Owing to this desirable property, the learned embeddings effectively capture the underlying structure of the environment, making them useful for goal-reaching tasks. Experimental results on a diverse suite of designed environments demonstrate the effectiveness of the proposed method.

### Strengths
1. The proposed method is theoretically well grounded, and the authors provide a thorough theoretical foundation to support their approach.

2. The paper is clearly written and well organized. The notation is consistent throughout, and the figures effectively facilitate understanding of the proposed method.

3. The experimental results are strong, showing that the proposed approach outperforms the baseline methods by a substantial margin.

### Weaknesses
1. The paper lacks a discussion of related work on other state representation learning methods (e.g., [a, b, c, d]). Including a comparison and discussion of these works would help contextualize the proposed approach.

2. The method assumes that the agent can initiate a random walk from any arbitrary state in the environment to sample trajectories for training. This assumption may be impractical in real-world scenarios.

3. The overall contribution appears limited. If not overlooked, the proposed method primarily extends the framework of Steccanella & Jonsson (2022) by introducing asymmetric Minimum Action Distance estimates, without offering additional conceptual or methodological innovations.

**Reference**

[a] Agarwal, Rishabh, et al. "Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning." International Conference on Learning Representations.

[b] Wang, Kaixin, et al. "Towards better laplacian representation in reinforcement learning with generalized graph drawing." International Conference on Machine Learning. PMLR, 2021.

[c] Wang, Kaixin, et al. "Reachability-aware Laplacian representation in reinforcement learning." Proceedings of the 40th International Conference on Machine Learning. 2023.

[d] Wu, Yifan, George Tucker, and Ofir Nachum. "The Laplacian in RL: Learning Representations with Efficient Approximations." International Conference on Learning Representations.

### Questions
1. Eqn.1: The interpretation of $d_\mathrm{MAD}$ could be made more intuitive and clear. Presenting it without further explanation may make it difficult to understand, particularly because the $\mathrm{argmax}$ operation appears counterintuitive when the goal is to represent the “minimum” action distance. Providing an intuitive explanation or example would help clarify its meaning.

2. Line 121: Shouldn't ReLU be a mapping $\mathbb{R}^d \to [0, +\infty)^d$?

3. Line 239: Isn't vector a special case of a matrix? Please clarify the distinction being made here.

4. Eqn.6 & 7: Is the squaring operation applied before or after the ReLU? If it is applied before ReLU, what is the rationale for applying the ReLU?

5. Sec.6.2: What is the motivation of introducing TDMadDist? From Tab.1 it seems that TDMadDist performs worse than MadDist.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4