# Learning to Drive with Two Minds: A Competitive Dual-Policy Approach in Latent World Models

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
End-to-end autonomous driving models trained solely with imitation learning (IL) often suffer from poor generalization. In contrast, reinforcement learning (RL) promotes exploration through reward maximization but faces challenges such as sample inefficiency and unstable convergence. A natural solution is to combine IL and RL. Moving beyond the conventional two-stage paradigm (IL pretraining followed by RL fine-tuning), we propose CoDrive, a competitive dual-policy framework that enables IL and RL agents to interact during training. CoDrive introduces a competition-based mechanism that facilitates knowledge exchange while preventing gradient conflicts. Experiments on the nuScenes dataset show an 18\% reduction in collision rate compared to baselines, along with stronger generalization and improved performance on long-tail scenarios. Code is available at an anonymous repository.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper "Learning to Drive with Two Minds: A Competitive Dual-Policy Approach in Latent World Models" proposes a dual-policy learning framework that trains an Imitation Learning (IL) actor and a Reinforcement Learning (RL) actor in parallel. The two actors periodically merge their model parameters based on predefined merging strategies and performance comparison.

### Strengths
1. The idea of maintaining both IL and RL actors throughout training, rather than discarding IL after warm-up, is interesting and could inspire further exploration.
2. The "backward planning" concept of providing more contextual information to early actions is intuitively appealing.
3. The paper is clearly motivated and connects well with the general problem of combining imitation and reinforcement learning.

### Weaknesses
1. The claimed contribution, "We integrate RL into an end-to-end driving framework by leveraging a latent world model for imagination-based simulation, avoiding reliance on external simulators," is not new. Previous work has explored similar ideas (see [1] for example).
2. The notations in the paper are sometimes confusing. For instance, what is the dimension of the waypoint features s_w? The symbols 
s and s_t are not used consistently. tau_a is an action sequence but is sometimes referred to as a single action (Equation (4)). L_{wm} appears before being defined. The notations for action sampling in Figure 2 are incorrect. The format of the L_{bc} is also incorrect.
3. While the dual-actor training pipeline introduces additional training overhead, the corresponding performance improvement appears minor. 

Reference: 
[1] Scheel, Oliver, et al. "Urban driver: Learning to drive from real-world demonstrations using policy gradients." Conference on Robot Learning. PMLR, 2022.

### Questions
1. The backward planning approach actually breaks the MDP assumption in RL. How can use this backward planning in RL? Also, what is the horizon length for the future states?
2. The latent world model is used during the RL actor learning phase, but when is this world model trained?
3. Is there any particular reason for using the L1 loss for imitation learning in Equation (3)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the limitations of end-to-end autonomous driving models, which suffer from poor generalization when trained with Imitation Learning (IL) and instability when trained with Reinforcement Learning (RL). The authors propose CoDrive, a novel dual-policy framework that synergistically combines IL and RL within a learned latent world model. Instead of relying on external simulators, CoDrive enables imagination-based training where an IL actor and an RL actor are trained in parallel. The core idea is a competitive learning mechanism that facilitates structured knowledge exchange: the IL actor provides expert knowledge, while the RL actor explores novel states and actions. This approach aims to leverage the stability of IL and the exploratory power of RL without their objectives directly conflicting, leading to improved generalization, reduction in collisions on the nuScenes dataset, and better performance in long-tail scenarios.

### Strengths
1. Novel and Well-Motivated Architecture for Integrating IL and RL. The proposed competitive learning mechanism provides a structured way to facilitate knowledge transfer.

2. This approach smartly bypasses the need for high-fidelity, hand-crafted external simulators, thereby mitigating the notorious sim-to-real gap and the dependency on expert demonstrations within the simulator.

3. By enabling imagination-based training, the framework allows the RL agent to perform sample-efficient and safe exploration of countless possible future scenarios.

### Weaknesses
1. In Table 1, the improvement gain on L2 metric is marginal. e.g. SSR+CoDrive is worse than SSR; LAW+CoDrive (PGGS) only achieves 0.01 gain. This raises questions about the practical significance of the proposed method, especially when weighed against its added architectural complexity.

2. In Table 2, the improvement on Navsim test set is also very minor. Taken together, this raises a critical question about the overall effectiveness of the proposed CoDrive framework,

3. How is the efficiency and resource-usage comparison? (Latency / training time / computation overheads)

4. In line 125: "achieve more stable training.". Any number comparison for proving the more stable training than single RL?

### Questions
See Weaknesses.

### Soundness
2

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
4

### Summary
The paper presents a policy framework that integrates an Imitation Learning and Reinforcement Learning approach to jointly learn a driving policy. For trajectory generation for autonomous driving, both learners predict actions but depending on the resulting score, only the best solution is implemented and weights of the worse learner are merged or replaced by the weights of the better learner. The approach is compared on the nuScenes and Navsim dataset against other baselines and improvements to existing baselines are shown.

### Strengths
Creation of a competition-based policy learning approach instead of combined losses is an interesting approach that can potentially avoid limiting the maximum performance due to conflicting action proposals.

The designed interaction between the IL and RL learner seems to reduce the collision rate significantly.

### Weaknesses
While the inverse causality is an interesting idea, the ablation study also seems to show that it has very little or no effect, compared to using no mask.

The results compared to SOTA are not very impressive. The approach seems to have significantly less collisions than plain SSR but has worse L2 scores. Compared with plain LAW it does not seem to improve performance beyond what could be some noise in the evaluation. Tasks like detection and tracking, which seem to help other approaches, are not necessarily hard to implement, given today's datasets. Therefore, the advantage of using this approach compared to other methods is not clear. This could be saved by potentially also adding a detection and tracking task and then having significantly better results across all metrics.

Results on Navsim show improved results compared to SOTA but it seems performance on this test set is very close to human performance for all methods and there is no large significant advantage over existing methods. 

The paper makes an interesting observation that RL achieves higher long-term results after initially the IL learner leads but than flattens out. However, this is compared to the two-stage paradigm which is described as inferior. This would be easy and important to show. It would have been interesting to see as comparison in the same setting to have first an IL and then an RL learner. It is not clear what, apart from potentially some time, is gained by training jointly.

### Questions
Because of the very specific meaning of the Bitcoin symbol and because the RL reward has nothing to do with it the symbol should be replaced in Figure 1.

I in formula 11 could be non-obvious. Please define it somewhere.

For equation 17 it is claimed that RL with the sparse reward is hard to stabilize and therefore a small behavioral cloning loss is added. This somehow goes against the idea of using the competitive system. Also, RL should be able to work with sparse rewards. This part could use more explanations: Why is it hard to stabilize training? Why does the L_bc help? How is \beta found and what effect does it have as beta goes from small to large?

There are minor syntax and grammar issues, e.g. in line 216: "Given the offline imitation dataset ..., using Gaussian Log Likelihood loss can easily fitting the behavior of experts". For a camera ready version I suggest to try to find a way to improve this as good as possible, potentially with outside help if the authors are not native English speakers as it is the case for the majority of our domain.

### Soundness
3

### Presentation
3

### Contribution
2
