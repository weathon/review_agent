# Graph-Theoretic Intrinsic Reward: Guiding RL with Effective Resistance

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 4, 8

## Abstract
Exploration of dynamic environments with sparse rewards is a significant challenge in Reinforcement Learning, often leading to inefficient exploration and brittle policies. To address this, we introduce a novel graph-based intrinsic reward using Effective Resistance, a metric from spectral graph theory. This reward formulation guides the agent to seek configurations that are directly correlated to successful goal reaching states. We provide theoretical guarantees, proving that our method not only learns a robust policy but also achieves faster convergence by serving as a variance reduction baseline to the standard discounted reward formulation. We perform extensive empirical analysis across several challenging environments to demonstrate that our approach significantly outperforms state-of-the-art baselines, demonstrating improvements of up to 59% in success rate, 56% in timesteps taken to reach the goal, and 4 times more accumulated reward. We augment all of the supporting lemmas and theoretically motivated hyperparameter choices with corresponding experiments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new metric termed effective resistance that can be used as an intrinsic motivation reward for goal-conditioned RL tasks. The paper hypothesizes that this reward is better than using one proportional to Euclidean distance to the goal. The paper also performs some analysis to show that the proposed intrinsic reward is almost unbiased with respect to the extrinsic reward, and that using the proposed intrinsic reward leads to improved sample complexity.

The empirical evaluation on a suite of environments called Safety-Gymnasium seek to show that the above proposed theoretical guarantees hold, and that the proposed technique outperforms some reasonable baselines.

### Strengths
Overall the idea is somewhat novel and might be a useful addition to the literature.
* The proposed technique is interesting. Using ideas that consider graph flows to estimate how easy it is to navigate from one point to another has been seen to be useful in the past [1], and this modern revival of the technique could present some benefits.
* Dynamic graph updates are very useful, allowing an environment that changes over time.
* Theoretical analysis seems to be sound and gives some confidence that the proposed approach will not converge to a suboptimal solution and will help with some local exploration.
* The evaluation methodology seems robust and statistically sound, and I especially appreciate the 1000 episode evaluations (5 training seeds and 200 episode evals per seed), which should capture variance in the policy and variance in the training.
* The baselines that are used in the evaluation seem mostly reasonable. There are caveats here that I expand on in weaknesses.



## References
[1] Şimşek, Ö., Wolfe, A.P. and Barto, A.G., 2005, August. Identifying useful subgoals in reinforcement learning by local graph partitioning. In Proceedings of the 22nd international conference on Machine learning (pp. 816-823).

### Weaknesses
There are some issues that keep this paper from being of a quality that I can confidently recommend for acceptance:
* This paper is trying to suggest a new intrinsic motivation based on effective resistance in a graph. But seems to be specific to robotics type problems with objects present in the environment causing navigational or manipulational difficulties. If specific to robotics problems, the setup and writing should clarify and try to position itself accordingly so that it will attract the same community of researchers. It also does not compare to other GCRL methods for exploration in literature [1, 2, 3], or more up to date GCRL benchmarks like OGBench [4].
* Part of the issue here is that the problem setup is specific to continuous state spaces and a 2 dimensional action space (Section 3.1).
* My understanding is that the intrinsic reward is only calculated when the reward enters the agent's field of view. This seems like it will help mostly with local exploration instead of more generally.
* The baselines proposed do not seem to be exploiting the structure of GCRL based problems from what I can tell. One of [1] or [3] would be great additions to show how effective the proposed intrinsic reward is in GCRL problems specifically.


## References
[1] Grace Liu, Michael Tang, and Benjamin Eysenbach. A single goal is all you need: Skills and exploration emerge from contrastive RL without rewards, demonstrations, or subgoals. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=xCkgX4Xfu0

[2] Ma, Y.J., Yan, J., Jayaraman, D. and Bastani, O., 2022. How Far I'll Go: Offline Goal-Conditioned Reinforcement Learning via $ f $-Advantage Regression. arXiv preprint arXiv:2206.03023.

[3] Durugkar, I., Tec, M., Niekum, S. and Stone, P., 2021. Adversarial intrinsic motivation for reinforcement learning. Advances in Neural Information Processing Systems, 34, pp.8622-8636.

[4] Park, S., Frans, K., Eysenbach, B. and Levine, S., 2024. Ogbench: Benchmarking offline goal-conditioned rl. arXiv preprint arXiv:2410.20092.

### Questions
* Could the authors clarify how more general exploration will be handled under the given scheme? Perhaps contrast with [3] from the weakness section, since that is an approach that handles more general exploration?
* Could approaches like Quasimetric learning [1] also learn some metric like effective resistance?

## References
[1] Wang, T., Torralba, A., Isola, P. and Zhang, A., 2023, July. Optimal goal-reaching reinforcement learning via quasimetric learning. In International Conference on Machine Learning (pp. 36411-36430). PMLR.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a novel intrinsic reward mechanism for reinforcement learning in sparse reward environments based on effective resistance from spectral graph theory. The key idea is to construct a time-evolving graph from the agent's observations (specifically LiDAR data) where nodes represent the agent, goal, and environmental objects, and edges encode proximity relationships. The intrinsic reward is defined as the negative change in effective resistance between the agent and goal nodes, encouraging the agent to seek configurations that improve structural accessibility to the goal.

### Strengths
1. The application of effective resistance from spectral graph theory to RL is creative and theoretically grounded. 
2. The paper provides a comprehensive theoretical analysis with multiple lemmas and a main theorem.

### Weaknesses
1. The method is specifically designed for environments where meaningful graph construction from observations is possible. The reliance on LiDAR data limits applicability to certain domains, and it's unclear how this would extend to other observation modalities or higher-dimensional state spaces.
2. Algorithm 1 involves many design choices (clustering threshold τ, connectivity patterns, central node selection) that appear to require careful tuning. The sensitivity analysis (Section A.9) shows some robustness to τ, but the overall complexity raises concerns about generalizability.
3. While the paper compares against several baselines, most are relatively older methods. More recent state-of-the-art intrinsic motivation methods could strengthen the comparison.
4. While Section A.10 provides some runtime analysis, the computational cost of repeated graph construction and effective resistance computation could be prohibitive in real-time applications or larger graphs.

### Questions
1. How does the method scale to environments with many more objects or higher-dimensional observation spaces? What is the computational complexity as a function of graph size?
2. How does the method scale to environments with many more objects or higher-dimensional observation spaces? What is the computational complexity as a function of graph size?
3. How does the method perform in environments with dense rewards? Does the intrinsic reward provide benefits or potentially interfere with learning in such settings?
4. How does the method perform in environments with dense rewards? Does the intrinsic reward provide benefits or potentially interfere with learning in such settings?
5. Beyond τ, how sensitive is the method to other hyperparameters like α and β? The theoretical guidelines (Corollary 1) provide bounds, but practical selection seems to require empirical validation.
6. How does this approach compare to more recent intrinsic motivation methods like NGU, RND, or ICM on the same environments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes a reward shaping methods orignated from spectral graph theory to tackle reward sparsity in reinforcement learning setting. By modifying its instrinsic reward, the agent needs to maintain a graph of its surrounding environment first through its sensors (e.g. LIDARs) every timestep, then calculates its effective resistance between itself and the goal, which will be used as part of its reward construction.  They provide theoretical guarantees to prove they can learn a robust policy and also achieves faster convergence. Through experiments, they show their methods can beat state of the art baselines.

### Strengths
- Good originality: abstract objects into nodes in graph, and design rewards based on the constructed graph. An novel reward shaping methods that encourage the agent to get closer to goal.
- Quality: theoretically sound. Assumptions setup with proper citation and completely. Proved decreasing the effective resistance can also maintain connectivity on the graph. This paper also show the advantage of its algorithms emprically with extensive experiments.

### Weaknesses
- This methods can only be applied to specific domains, for example, robotics navigation tasks in the paper, in which the robots have a suite of sensors that are assumed not having noise and the robots having a good localization capability and can categorizes or recognize objects as nodes in the map. 
- I would appreciate more explanation on how the graph is constructed and what reducing effective resistance would bring on the main text, but overall the paper is easy to follow.

### Questions
- To increase the impact of this paper, can this method apply to a more general MDP setting, e.g. continuous MDP or tabular MDP, how would you construct the graph in such MDPs? specifically, what are nodes/edges/weights in these settings?

### Soundness
3

### Presentation
3

### Contribution
3
