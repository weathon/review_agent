# Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Effective exploration in reinforcement learning requires not only tracking where an agent has been, but also understanding how the agent perceives and represents the world. To learn powerful representations, an agent should actively explore states that contribute to its knowledge of the environment. Temporal representations can capture the information necessary to solve a wide range of potential tasks while avoiding the computational cost associated with full state reconstruction. In this paper, we propose an exploration method that leverages temporal contrastive representations to guide exploration, prioritizing states with unpredictable future outcomes. We demonstrate that such representations can enable the learning of complex exploratory behaviors in locomotion, manipulation, and embodied-AI tasks, revealing capabilities and behaviors that traditionally require extrinsic rewards. Unlike approaches that rely on explicit distance learning or episodic memory mechanisms (e.g., quasimetric-based methods), our method builds directly on temporal similarities, yielding a simpler yet effective strategy for exploration.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Based on the work of Jiang et al. (2025), this paper proposes a new reward objective that avoids quasi-metric learning and episodic memory. The experiments are conducted in navigation, manipulation, and open-world environments.

### Strengths
1. This paper introduces a reward objective that is easier to compute.
2. The experiments demonstrate that the proposed method leads to the agent visiting more states.

### Weaknesses
1. Temporal Similarity as Intrinsic Reward:

The use of temporal similarity as an intrinsic reward is explored in [1]. The novelty of this approach needs further clarification.

2. Advantages Over [1]: 

Compared to [1], the proposed method (1) avoids quasi-metric learning and (2) eliminates the need for episodic memory. What are the specific benefits of these two design choices? A more detailed explanation would be helpful.

3.  Experimental Results and Task Rewards: 

The experimental results show that the visited states are more diverse. However, how does this diversity benefit RL training with task rewards?
I believe this is not a significant burden on the authors, as they could simply incorporate the extrinsic reward into the current intrinsic reward framework. The absence of this experiment makes it difficult for readers to fully believe in the method's adaptability.

4. Title Clarification: 

The paper is titled "DIVERSE BEHAVIORS," which typically implies that the agent can generate diverse trajectories, not just more diverse visited states. This could be somewhat confusing.


[1] Yuhua Jiang, Qihan Liu, Yiqin Yang, Xiaoteng Ma, Dianyu Zhong, Hao Hu, Jun Yang, Bin Liang, Bo XU, Chongjie Zhang, and Qianchuan Zhao. Episodic novelty through temporal distance. In The Thirteenth International Conference on Learning Representations, 2025.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
C-TeC uses a temporal contrastive critic to give intrinsic reward as the negative similarity between (s_t,a_t) and discounted future states—favoring states with diverse, hard-to-predict futures, without episodic memory and friendly to off-policy RL. It plugs into PPO/SAC, can target specific future-state subspaces, improves coverage in Ant/Humanoid mazes, and beats ETD on Crafter. Contributions: a simple contrastive exploration objective, its info-theoretic view, and consistent empirical gains across domains.

### Strengths
- Provides an information-theoretic interpretation: in the idealized limit, the intrinsic reward equals $−D_{KL}​(p_T​(s_f​∣s_t​,a_t​)∥p_T​(s_f​))$, linking the objective to mode-seeking over future-state distributions.

- Clear algorithmic specification (sampling of future offsets, InfoNCE update, PPO/SAC integration) that is implementable and off-policy friendly.

- Demonstrates flexible targeting of exploration to task-relevant subspaces, a practical advantage for complex embodiments and object-centric tasks

### Weaknesses
- The “future-diversity” view overlaps conceptually with recent contrastive or mutual-information–based exploration (e.g., APT/CIC/Plan2Explore/APS). The paper doesn’t clearly isolate what is new beyond replacing episodic memory with a temporal contrastive critic

- The intrinsic reward hinges on design knobs that may materially affect behavior: future horizon/window sampling, temperature in InfoNCE, negative-sampling strategy, and choice of future subspace (e.g., positions vs. full features)

- Using negative similarity as reward risks mode chasing or collapse if the representation is not sufficiently regularized.

### Questions
- You connect the objective to $−D_{KL}​(p_T​(s_f​∣s_t​,a_t​)∥p_T​(s_f​))$ or $I((s_t,a_t);s_f)$, under what assumptions (temperature τ, negative sampling, batch size) does the empirical contrastive loss yield a consistent estimator of this quantity? Can you give a finite-sample bound relating representation error 𝜖 to the bias in $r_{int}$?

- How do you choose the distribution over offsets k∼p(k) for the “discounted future”? Did you try adaptive or curriculum schedules for k (e.g., increase max k over training)? Please add sensitivity/ablation for p(k) and the discount used inside the similarity.

- Contrastive objectives can be biased by replay imbalance (e.g., many near-duplicate futures from recent policies). Do you stratify negatives by time/episode, or reweight by importance? Ablate (i) in-episode vs. cross-episode negatives, (ii) balanced vs. raw replay sampling, and report stability/sample-efficiency impacts.

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
5

### Summary
This paper focuses on the problem of using intrinsic motivation to promote exploration in reinforcement learning, primarily by designing a new intrinsic reward. This new intrinsic reward directly follows the previous work "Episodic Novelty Through Temporal Distance (ETD)." The main motivation is to simplify the ETD method, with the following improvements:

- ETD learns the temporal distance between states.  Distance meaning it should at least satisfy the properties of a quasi-metric. C-TEC demonstrates through experimental results that learning a quasi-metric is unnecessary; it is sufficient as long as the representation can reflect the temporal relationship between states.
- ETD calculates the intrinsic reward by computing the distance between the current state and every state in the episodic memory, taking the minimum value as the intrinsic reward (backward-looking). In contrast, C-TEC samples a future state according to a geometric distribution and calculates the distance between the current state and that future state as the intrinsic reward (forward-looking).

### Strengths
The algorithm uses the idea of temporal distance as an intrinsic reward. It has some contributions in simplifying ETD's algorithmic modules, and it was found on continuous control tasks and the Crafter task that simplifying these modules does not lead to performance degradation. I believe this contribution is sufficient.

Changing ETD's intrinsic reward from backward-looking to forward-looking is interesting.

### Weaknesses
- The comparison results between ETD and C-TEC on Crafter are very strange. In the original ETD paper, the results for the method and its baselines were all in the (7.0 ~ 9.0) range. However, here the result for ETD is less than 0.5, and even C-TEC is only 2.0.

- The paper's title, "Discovering Diverse Behaviors via Temporal Contrastive Learning," is somewhat strange. I believe "Discovering Diverse Behaviors" corresponds to learning a set of policies, each with a different behavior. However, the paper is clearly not doing this. This title is ambiguous.

- In line 256, the paper divides the intrinsic reward into a "surprise" term and a "familiarity" term. The "surprise" term can be understood as promoting exploration, but I do not understand the role of the "familiarity" term. I also do not quite understand the role of "mode-seeking" in exploration. Does the author mean that a high reward should be given to states that have been visited but are temporally distant, rather than giving a high reward to unvisited states?

- C-TEC is (forward-looking) while ETD is (backward-looking). Besides the fact that the forward-looking approach does not require an episodic memory, are there any other advantages compared to the backward-looking approach? I find the current Crafter results unconvincing, and I believe that results from some toy examples would be more illustrative.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This article proposes C-TeC (Curiosity via Temporal Contrastive Learning), a novel exploration method that relies on intrinsic rewards and contrastive learning to reach unexpected but meaningful states. This technique differentiates itself from prior methods by its simplicity, avoiding the use of quasimetric learning. The method is also compatible with classic algorithms such as Soft Actor-Critic and Proximal Policy Optimization. Finally, empirical results show strong exploration performance on locomotion, manipulation, and open-world tasks.

### Strengths
The presented empirical results show generally strong performance across various tasks, providing solid evidence of state-of-the-art performance.

The article presents sufficient theoretical background to justify the proposed method, going beyond simple empirical results.

The proposed approach is conceptually simple yet broadly applicable, removing complex components such as episodic memory or explicit distance metrics while maintaining competitive results. This simplicity makes C-TeC easier to integrate with standard off-policy algorithms like SAC and potentially more stable and efficient in training compared to previous exploration methods.

### Weaknesses
While the authors present extensive ablation studies, the paper does not include ablations related to sample efficiency or the method's sensitivity to hyperparameters.

Additional benchmark comparisons with recent state-of-the-art world-model-based algorithms, such as DreamerV3, would provide valuable insight into the trade-offs between their higher computational cost and their performance compared to the proposed method.

While the authors recognize that this will be tackled in future work, understanding the performance of the proposed method in partially observable environments is crucial to demonstrate the method’s broader applicability.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3
