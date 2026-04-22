# Empowering Multi-Robot Cooperation via Sequential World Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 2, 6, 8, 8

## Abstract
Model-based reinforcement learning (MBRL) has achieved remarkable success in robotics due to its high sample efficiency and planning capability.
However, extending MBRL to physical multi-robot cooperation remains challenging due to the complexity of joint dynamics.
To address this challenge, we propose the Sequential World Model (**SeqWM**), a novel framework that integrates the sequential paradigm into multi-robot MBRL.
SeqWM employs independent, autoregressive agent-wise world models to represent joint dynamics, where each agent generates its future trajectory and plans its actions based on the predictions of its predecessors.
This design lowers modeling complexity and enables the emergence of advanced cooperative behaviors through explicit intention sharing.
Experiments on Bi-DexHands and Multi-Quadruped demonstrate that SeqWM outperforms existing state-of-the-art model-based and model-free baselines in both overall performance and sample efficiency, while exhibiting advanced cooperative behaviors such as predictive adaptation, temporal alignment, and role division.
Furthermore, SeqWM has been successfully deployed on physical quadruped robots, validating its effectiveness in real-world multi-robot systems.
Demos and code are available at:  https://github.com/zhaozijie2022/seqwm

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes *SeqWM*, a framework for multi-robot cooperative learning. It first trains a Dreamer-style world model that predicts future rewards and observations, and then uses MPPI to derive actions based on the learned model. To facilitate multi-robot communication and cooperation, during the world model training stage, the $i+1$-th model receives the predictions from the first $i$ models. The proposed framework demonstrates its effectiveness across various tasks, both in simulation and in real-world experiments.

### Strengths
- Various experiments demonstrate the effectiveness of the proposed framework in both simulation and real-world settings. The experimental results show that the proposed method outperforms several baseline approaches.
- The authors provide extensive visualizations to illustrate the performance of their method, along with accompanying code and videos for more intuitive understanding.

### Weaknesses
- The presentation still has significant room for improvement. For example, the authors state that the world model predicts the “environment observation,” defined as the agent’s perception of the environment. However, it remains unclear what these observations actually are — joint angles, absolute positions, or visual inputs? Such important details are not explicitly mentioned in the paper.
- The discussion of related work is limited. While the authors review prior studies in model-based reinforcement learning and the sequential paradigm, they overlook a substantial body of work in multi-agent learning and multi-robot cooperation.
- The ablation studies are insufficient. The authors conduct ablation experiments on various communication function designs within the world model (Figure 9) and evaluate performance on the BottleCap task. However, other critical design choices, such as those in MPPI or world-model learning, remain unexplored. A key weakness is the lack of justification for why a world model is necessary for this task—see the detailed question below for further discussion.

### Questions
- A critical question for the proposed framework is: why are a world model and planning necessary in this context? From the demonstration videos, it appears that one major issue with baseline methods is that the robots lack awareness of each other’s positions(xy coordinates in world), leading to coordination failures. However, since motion capture data is available, one could simply provide each robot with the positions of the others and train them to complete the tasks. I would encourage the authors to explore these simpler baseline approaches and compare their performance with the proposed framework.
- It is also unclear why MAPPO performs poorly on certain tasks, such as PushBox. According to the Multi-Quad paper, MAPPO indeed shows weak performance on similar tasks, consistent with the authors’ findings here. Could the authors provide insights or analysis explaining why MAPPO struggles in these scenarios?

### Soundness
3

### Presentation
2

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
This paper proposes a model-based RL algorithm for multi-agent cooperative problems. The algorithm formulates the world model and the decision making as sequential. This design reduces the complexity of modeling the joint dynamics. The world model is used in a planner to get actions. Experiments conducted in simulated bidexhand and multi-quadruped tasks show that the sequential paradigm leads to more coordinated behaviors. The algorithm also achieves better sample efficiency compared to model-free and non-sequential model-based baselines.

### Strengths
1. The sequential paradigm is a reasonable design choice to reduce the complexity of a joint world model over multiple agents. Although modeling multi-agent problems as sequential is not new (also studied in one of the baseline methods), applying it to the learning of a dynamic model still has some value to the community.

2. The method achieves strong empirical results in simulation benchmark, achieving higher sample efficiency than all the baselines with a clear margin. The deployment to the real world further strengthened the result.

### Weaknesses
1. There might be some overclaim in contribution (3). Previous work has already demonstrated multi-quadruped pushing in the real world, such as [1].

[1] Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing. ICRA 2025. https://arxiv.org/abs/2411.07104

2. Some figures are not self-contained. The captions in Figure 6 are too small to read and the message to convey here is not clear. The meaning of the y-axis in Figure 8 is not explained.

3. In Figure 3, all baselines achieve zero success in the Pushbox environment. Is there any explanation? 

4. The ablation study on sequential sample generation seems a little incomplete. As we can see in Figure 8, the centralized architecture achieves even lower prediction errors than the sequential architecture. Does it mean that we should prefer centralized rather than the proposed sequential model?

### Questions
1. The sequential paradigm may consume longer inference time since the ${i+1}$-th agent depends on the inference result of the previous $i$ agents. How to overcome this issue when the number of agents is large?

2. Is the action execution of each agent asynchronized? If yes, do you take into account the asynchronous execution during training in simulation?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper successfully integrates the sequential modeling paradigm into model-based multi-agent reinforcement learning and demonstrates both cooperative efficiency and real-world applicability. Addressing the challenging problem of multi-robot cooperation, it introduces a novel Sequential World Model framework. The proposed method not only outperforms existing state-of-the-art approaches in performance and sample efficiency but also achieves real-world deployment on physical robots, demonstrating its ability to handle complex cooperative tasks.

### Strengths
1. The central idea of reformulating joint dynamics modeling as an autoregressive sequential modeling problem is both novel and effective. The sequential design significantly reduces communication and modeling overhead, while enabling complex cooperative behaviors with multi-robots.
2. The paper is well-organized, clearly written, and easy to follow. The motivation and contributions are articulated very clearly.
3. The method achieves good performance and sample efficiency compared to all baselines. One of the strong contributions is the real-world demonstration. It successfully executed complex cooperative behaviors (e.g., pushing, yielding, shepherding) on physical robots. This reviewer believes that this strongly validates the practicality and robustness of the proposed approach.

### Weaknesses
1. The related work is well written, and it covers well model-based RL and the current sequential paradigm in multi-agent systems. It identifies the gap between centralized and decentralized approaches. However, this reviewer feels that deeper comparative analysis with closely related frameworks such as MARIE, CoDreamer, and SeqComm would clarify the unique technical contribution of SeqWM beyond general sequential coordination. Including reflections on potential limitations, such as dependence on agent ordering and communication latency, would also add balance and critical depth.
2. Although extensive and well-structured, this reviewer feels that the ablation experiments are not fully sufficient to validate the claimed contributions. First, there is no direct ablation isolating the effect of the sequential structure on emergent cooperative behaviors such as predictive adaptation, temporal alignment, and role division. It is unclear whether these behaviors arise from the sequential design, the MPPI planner, or improved world model accuracy. Second, it appears that the ablations are performed primarily on BottleCap, not on multi-robot coordination tasks such as Multi-Quad or PushBox, limiting generalization to core cooperative scenarios. Third, other design elements, including the planner, early stopping, and communication cache, are under-analyzed. While briefly mentioned in the appendix, the main text lacks quantitative justification of their contributions. For example, there is no comparison showing whether cooperation emerges even without the sequential structure, such as a “Planner-only” baseline.
3. In sequential planning, with N agents, the Nth agent must wait for all preceding agents to finish planning, which may introduce latency. While the paper demonstrates scalability, it does not analyze how this latency affects responsiveness in larger systems. A discussion on the theoretical convergence and stability of the sequential updates would strengthen the paper.
4. The proposed method’s reliance on a fixed agent order raises questions about sensitivity. There is no in-depth analysis of how agent ordering affects performance or how accumulated prediction errors propagate over long horizons. (1) How is the order determined, manually or learned? (2) What happens if the optimal order changes during execution (for example, in the Shepherd task when the leading robot should switch)? The paper mentions random masking during training to improve robustness, which is a good idea, but it remains unclear whether a fixed or randomized order is used during testing and how this choice affects robustness.
5. While the sequential scheme reduces communication frequency, transmitting entire predicted trajectories as messages may increase communication bandwidth. As N grows, the message size received by later agents could scale linearly, potentially becoming a limitation or constraint.

### Questions
1. If this reviewer understands correctly, SeqWM assumes a leader–follower (predecessor–successor) structure where agents act in a predefined order (e.g., Agent 1 -> Agent 2 -> Agent 3). Would this design limit its applicability to tasks that require true simultaneity or are inherently symmetric without a natural ordering? For instance, how would SeqWM handle scenarios where two robots must react simultaneously to catch a fast-moving, unpredictable object, or where multiple robots must lift a shared payload at the exact same time to maintain stability?
2. How computationally heavy is the proposed planner? how does the runtime scale as the number of agents increases, and what are the practical limits for real-time deployment in physical multi-robot systems?
3. Based on Table 2, it appears that each agent performs approximately K=6 iterations. If that is correct, does the last agent need to wait for all preceding agents to complete their iterations? For example, with five agents, what is the observed planning delay (in ms) for the final agent? At what point (in terms of the number of agents) does sequential latency become a bottleneck for real-time control?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents sequential world models (SeqWM), a novel multi-agent reinforcement learning (MARL) method that incorporates the sequential multi-agent paradigm into model-based MARL. In SeqWM agents have sequential, independent autoregressive world models, each conditioned on the predecessor, avoiding reliance on centralized stable communication channels. SeqWM is compared with both model-free and model-based baselines on various tasks from two simulation benchmarks, in which SeqWM improves both over final performances and sample efficiency. Additional experiments (ablations, scalability, real-world deployment) further solidify the relevance of SeqWM.

### Strengths
Quality:
- reproducibility is supported by publicly available code and hyperparameters
- the presented method is compared to relevant model-free and model-based baselines
- the method is investigated on multiple simulated domains and in the real-world

Significance:
- results on the simulated domains suggest that SeqWM not only considerably improves sample efficiency, but can also produce solutions to tasks that baselines cannot solve
- the role of the core idea (sequential paradigm in model-based MARL) is empirically demonstrated with an ablation
- the method is empirically validated w.r.t. scalability and real-world deployment

Originality:
- the method is an original combination of well-established ideas, namely the sequential multi-agent paradigm and world models

Clarity:
- overall, the manuscript is clear and the layout of the content is predictable
- core ideas are easily understandable
- the contribution is properly contextualized both in the introductory and related work sections

### Weaknesses
Quality:
- although the ablations presented are meaningful, more fine-grained ablations would better support the various contributions to the improved performance of SeqWM w.r.t. baselines; however, this would require substantial additional work which might be out of the scope of this work

Significance:
- while computational costs for the planner early-stopping is properly investigated, computational costs of the other compontents is not reported. Discussing this either in the main text or appendix would help framing the applicability of SeqWM
- while SeqWM is validated on challenging and varied robotics-based domains, it is unclear whether its benefits also apply to different kind of domains (e.g. virtual multi-agent environments like StarCraft); focus on robotics environments is nonetheless properly motivated and sufficiently significant as is

Clarity:
- while the presented method is clear, pseudocode is not present in the main text (but it is present in appendix B.1); moving it or just briefly referencing it in the main methodology section would make it easier to follow
- the y axis of figure 8 is confusing: it is reported as "accuracy", but the main text mentions that the decentralized setting is worse, even though its bars are higher (possibly a typo)
- appendix B.1 contains no text besides the pseudocode, which is confusing. Adding simple sentence in the main text of appendix B.1 referencing algorithms 1 and 2 would improve its readability

### Questions
- is the y axis of figure 8 reporting accuracy (more is better) or loss (less is better)?
- how general is SeqWM beyond robotics environments?
- is SeqWM applicable to environments with very high number of agents (e.g. swarms)? What are its limitations in these scenarios?
- how are compute costs distributed among the various components of SeqWM? What are the computational bottlenecks in the reported experiments?

### Soundness
4

### Presentation
3

### Contribution
3
