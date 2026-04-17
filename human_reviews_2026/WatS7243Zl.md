# ReDiG: Reinforced Diffusion on Graphs for Decentralized Coordinated Multi-Robot Navigation with Smooth Formation Adaptation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 2

## Abstract
Coordinated navigation is a fundamental capability for multi-robot teams to traverse complex unstructured environments.
During navigation, robots are often required to maintain mission-specific formations, such as wedge formations for enhanced visibility and area coverage.
However, rigid formations can hinder navigation in challenging scenarios like narrow corridors, which demand formation adaptation.
Reinforcement learning (RL) is commonly used for coordinated multi-robot navigation due to its ability to learn through interaction with the environment.
However, its step-wise decision-making process often results in jerky motion.
In contrast, diffusion models generate smoother trajectories through probabilistic denoising, but rely heavily on high-quality demonstrations.
Collecting such demonstrations is challenging in multi-robot systems due to the coordination and synchronization required among individual robots.
To address these issues, we introduce a novel method named Reinforced Diffusion on Graphs (ReDiG) to enable
decentralized coordinated multi-robot navigation with smooth formation adaptation. 
Under a unified learning paradigm, ReDiG integrates:
(1) graph learning for decentralized coordination to enable formation adaptation,
(2) diffusion models for generating smooth individual robot trajectories, and
(3) online RL to refine noisy demonstrations by leveraging feedback from environment interaction, which enables robot synchronization and guides effective diffusion training.
We evaluate ReDiG through extensive experiments in both indoor and outdoor environments using physical robot teams and robotics simulations.
Experimental results show that ReDiG enables smooth formation adaptation and achieves state-of-the-art performance in coordinated multi-robot navigation within complex environments.
More details are available on the project website: https://anonymous23885.github.io/ReDiG

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper develops Reinforced Diffusion on Graphs (ReDiG), a unified learning paradigm for multi-robot navigation. ReDiG contains three components: (1) a graph neural network for decentralized coordination, (2) a diffusion model for individual trajectory generation, and (3) an online reinforcement learning method for team synchronization. Experiments conducted in both simulated and real-world scenarios with various settings illustrate the remarkable performance of the proposed approach.

### Strengths
1. The paper is well-structured and presented in a clear academic style.
2. The authors provide strong motivation for their approach.
3. Experiments with various robot types and formation shapes validate the effectiveness of ReDiG.

### Weaknesses
1. ReDiG is a combination of several different algorithms rather than an elegant one.
2. The graph neural network (GNN) is used as an encoder to encode the robot's observation and state for diffusion models. Similar techniques have been widely employed in prior works [1, 2].
3. A standard diffusion model is used to generate trajectories without any guidance or projection, and prior work indicates its performance degrades rapidly as the number of robots increases [3, 4].
4. The number of robots used in experiments is very limited (even in simulated environments). There is no challenging scenario considered in experiments.
5. The Contextual Formation Integrity (CFI) of ReDiG is lower than that of AFOR in multiple settings.
6. No significance testing is reported across methods





**References**:

[1] Wang, Yutong, et al. "Scrimp: Scalable communication for reinforcement-and imitation-learning-based multi-agent pathfinding." 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023.

[2] Ma, Yixiao, et al. "Privileged Reinforcement and Communication Learning for Distributed, Bandwidth-limited Multi-robot Exploration." arXiv preprint arXiv:2407.20203 (2024).

[3] Shaoul, Yorai, et al. "Multi-Robot Motion Planning with Diffusion Models." The Thirteenth International Conference on Learning Representations.

[4] Liang, Jinhao, et al. "Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models." Forty-second International Conference on Machine Learning.

### Questions
1. The authors claim that ReDiG can generate smooth trajectories. Did they employ any guidance or additional methods during the training or sampling process to ensure the smoothness of the generated trajectories, or did they simply rely on a standard diffusion model?
2. Does the proposed approach provide any theoretical guarantees regarding feasibility, smoothness, or formation?
3. What are the smoothness results of the baseline methods?

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
4

### Summary
ReDiG presents a decentralized online learning framework that synergizes graph neural networks, conditional diffusion policies, and actor-critic reinforcement learning to achieve smooth, adaptive formation navigation for multi-robot teams. Theoretically grounded by convergence bounds for both diffusion and value approximation, the method is validated in simulation and on physical robots, demonstrating great task success rate and superior trajectory smoothness. Limitations include shape-specific retraining, reliance on single-robot demonstrations.

### Strengths
1.Propose a new multi-robot collaboration framework for formation controlling. This framework unifies decentralized graph neural networks, diffusion-based trajectory generation, and reinforcement-learning-driven formation synchronization. 

2.Rigorous convergence guarantee: Proves the explicit upper bounds on the KL divergence between learned and true action distributions for the conditional diffusion model, isolating prior mismatch, denoising error, and discretization error; provides a parallel finite-sample bound for the critic that separates statistical and algorithmic errors, ensuring monotonic improvement of both trajectory smoothness and value estimates during online training.

3.Eliminates expert-demonstration bottleneck : This article bootstraps a diffusion policy via combining it with online RL, converting environmental reward into synchronized, formation-aware demonstrations without costly multi-robot experts.

### Weaknesses
1.Fixed communication radius: The topology of multi-robot system is set manually, the ability of adaptation is absent.

2.Formation-specific training: A separate model must be retrained for each desired shape (wedge, line, circle), precluding on-the-fly formation switching.

3.Demonstration dependency: Initial supervision relies on single-robot planners (A*, RRT); no curriculum or self-supervised pre-training is explored.

### Questions
1.Although 60-step early stopping achieves 223Hz, the latency and CPU/GPU utilisation for the full 100-step model are not given; such data are critical for deployment on resource-constrained situations.

2.Recent multi-agent diffusion or offline RL baselines (MADiff, DoF) are omitted; a comparative discussion would verify ReDiG’s contribution.

3.Only aggregate success rates are presented. Illustrative failure cases—e.g., localization drift, packet loss, or corridor congestion—and their recovery statistics would help ascertain robustness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents ReDiG, a framework which integrates GNNs, diffusion models, and RL to enable multi-robot motion planning. The method relies on a graph neural network to coordinate between trajectories generated by independent diffusion models corresponding to each robot. This ensemble is well motivated by prior success utilizing GNNs, and the reported results demonstrate greater efficiency than the compared baselines.

### Strengths
- **Technical Presentation:** The paper is written in a technical format that is appropriate for the target audience. The exposition is fairly intuitive.

- **Real-World Evaluation:** The inclusion of results from real-world deployment are compelling, providing support for the authors' claims of practical utility.

- **Time Travel Efficiency:** The faster paths provided by ReDiG provide are interesting, indicative of more optimal paths.

### Weaknesses
- **Baseline Selection:** It is surprising that the paper does not compare to any recent diffusion motion planning baselines, e.g., [1-3]. This seems appropriate, especially when claiming SOTA performance. It is difficult to assess the performance on the method without this analysis. Comparison is limited to older RL baselines and GNN-based approaches; because of this, the evaluation seems incomplete. Could the authors please comment on why these methods have not bee compared to?

- **Scope of Contribution:** Presently, I'm not convinced of the overall novelty of the framework. While this ensemble of methods is unique, the methodology centers on ensembling existing tools (e.g., GNNs for coordination, Diffusion Models for motion planning, and RL). While this is indeed a contribution, it seems to fall more on the engineering side of things.


---

[1] Carvalho, Joao, et al. "Motion planning diffusion: Learning and planning of robot motions with diffusion models." 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023.

[2] Shaoul, Yorai, et al. "Multi-robot motion planning with diffusion models." arXiv preprint arXiv:2410.03072 (2024).

[3] Liang, Jinhao, et al. "Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models." arXiv preprint arXiv:2502.03607 (2025).

### Questions
- It seems that better TT comes at the cost of tolerance to $\delta$? Is ReDiG capable of performing this trade-off (e.g., tighter tolerance is required for a particular task)? Has any analysis of this been conducted?

- Could the authors clarify which components of the framework are algorithmically novel, as opposed to a composition of existing techniques?

- What limitations currently prevent ReDiG from scaling to a higher number of robots?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a learning-based method to achieve decentralized multi-robot navigation with smooth formation adaptation. Experimental results in both indoor and outdoor environments demonstrate the effectiveness of the proposed method.

### Strengths
Extensive experiments are conducted in both indoor and outdoor environments using physical robot teams and robotics simulations.

### Weaknesses
The related work section is insufficiently discussed. The use of the diffusion model and its training process are unclear. The proposed algorithm is not clearly described and seems questionable. See Questions part for more details.

### Questions
(1) Since this paper focuses on the distributed formation navigation problem, the authors are strongly suggested to compare their method with existing control-based methods, such as bearing-based and angle-based formation control methods, which enable multi-robot systems to reduce their formation size while maintaining the formation shape when passing through narrow passages.

(2) It is stated in the appendix that classic path planning algorithms (e.g., RRT) can be used to generate expert demonstrations. Could you clarify how the expert trajectories are aligned or matched with the state–action pairs in your method?

(3) The loss function in (1) is hard to understand. How do you determine $\epsilon_k$? Why do you use $a_i^0$ instead of $a_i^k$? How is the notation $\psi$ related to this loss function?

(4) The gradient form of the loss function (1) in line 6 in Algorithm 1 seems wrong.

(5) The actor update step in line 11 in Algorithm 1 seems wrong.

(6) Please give the specific gradient formula for the GNN parameters.

(7) The authors put only the expert trajectory and the transition generated by the diffusion model in the replay buffer. How do you update your actor?

### Soundness
2

### Presentation
2

### Contribution
2
