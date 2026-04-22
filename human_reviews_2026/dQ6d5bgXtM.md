# Translating Flow to Policy via Hindsight Online Imitation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Recent advances in hierarchical robot systems leverage a high-level planner to propose task plans and a low-level policy to generate robot actions.
This design allows training the planner on action-free or even non-robot data sources (e.g., videos), providing transferable high-level guidance.
Nevertheless, grounding these high-level plans into executable actions remains challenging, especially with the limited availability of high-quality robot data.
To this end, we propose to improve the low-level policy through online interactions.
Specifically, our approach collects online rollouts, retrospectively annotates the corresponding high-level goals from achieved outcomes, and aggregates these hindsight-relabeled experiences to update a goal-conditioned imitation policy.
Our method, Hindsight Flow-conditioned Online Imitation (HinFlow), instantiates this idea with 2D point flows as the high-level planner.
Across diverse manipulation tasks in both simulation and physical world, our method achieves 
more than $2\times$ performance improvement over the base policy, significantly outperforming the existing methods.
Moreover, our framework enables policy acquisition from planners trained on cross-embodiment video data, demonstrating its potential for scalable and transferable robot learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents HinFlow. First, it utilizes unlabeled robotic videos and off-the-shelf point trackers to train a high-level planner. Then, for the online interaction, by leveraging the thought of hindsight and regarding the generated flow trajectory as a replaceable goal, the low-level policy can improve itself without expert demonstrations. The author validates the effectiveness across 7 tasks and 2 benchmarks.

### Strengths
- Taking the high-level generated flow trajectory as a subgoal for low-level control is not novel, but regarding this as an replaceable goal in hindsight is interesting, and the experiments prove its effectiveness.
- This method is promising for scalable training because the high-level flow planner needs only action-free video, and the author validates that cross-embodiment data can help improve low-level policy learning.
- In section 5, HinFlow shows a very fast speed in convergence, which is promising to be applied in the real world for a fast policy improvement.

### Weaknesses
- The experiment is inadequate, with only 2 benchmarks and 7 tasks. Hope to see more results in the real world.
- Such kind of hierarchical methods cannot be used for large-data pretrained VLA models, such as $\pi_0$, Gr00t, RDT.
- The flow planner is task-specific and unable to deal with multi-task settings.
- The contribution is incremental.

### Questions
- Could HinFlow work as well in the real world as in simulators?
- Could HinFlow solve multiple tasks with only one flow planner model?
- All the experiments are at most 2 stages. What about using HinFlow in tasks with a longer horizon? Since a flow trajectory is always generated as an immediate subgoal, it may be suitable for complex tasks.
- Since there is no visualization of flow trajectories in wrist view, what's the role of the wrist view?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes to use online interaction to improve the low-level policy that outputs action by taking in plans produced from high-level video-based planners. This is an important question since data with actions are much more expensive to obtain compared to video-only data. The experimental results shows promising performance of the proposed method in simulation environments (LIBERO, ManiSkill).

### Strengths
- The paper tackles a timely and important challenge in hierarchical robot learning: grounding high-level, video-derived plans into reliable low-level control.

- The proposed hindsight flow-conditioning idea is simple, intuitive, and well-motivated, yet leads to strong performance gains with good sample efficiency, achieving large improvements within 80K online interaction steps.

- The experiments span a diverse set of seven manipulation tasks across two widely used benchmarks (LIBERO and ManiSkill), highlighting generality.

- Cross-embodiment results are shows successful cross-robot skill transfer.

- The writing is clear and the paper is well-structured, making contributions easy to follow.

- Reproducibility is strong with detailed implementation descriptions and consistent reporting over multiple seeds.

### Weaknesses
While the reviewer is positive about the contributions and overall quality of this work, some clarifications and additions would further improve its clarity and impact:

- **Baseline comparison with non-hierarchical online learning.**
The current baselines emphasize either offline imitation (BC) or hierarchical approaches. Including a pure online RL baseline, such as PPO — even if it is expected to underperform within the 80K interaction budget — would help isolate the benefit of the flow-based hierarchical decomposition and better contextualize the results in Figure 6.

- **Clarification on environment reset assumptions.**
In real-world deployment, environment resets often incur human effort. The paper does not explicitly state the reset policy, though the figures suggest resets likely occur at the maximum task horizon. Given that noticeable performance gains emerge after ~20K environment steps (roughly 40–200 episodes, depending on the task), briefly discussing how this assumption affects practicality would be helpful.

- **Lack of real-robot experiments.**
Although the method shows strong robustness in simulation, real hardware introduces challenges — such as sensing latency, imperfect tracking, or actuator delay — that could impact the stability of the hindsight-labeling pipeline. A short discussion of this gap and potential mitigation strategies would strengthen the claims around deployability.

- **Runtime and compute cost reporting.**
The reproducibility section is detailed, but there is limited visibility into the wall-clock time or compute required for pretraining and online refinement. Even approximate training-time statistics (e.g., hours per stage on a given hardware setup) would make it easier for practitioners to gauge feasibility.

### Questions
- Could the authors comment on failure modes observed during online adaptation? For example, what happens when the high-level planner proposes inaccurate flow guidance?

- How sensitive is the approach to flow tracking noise during hindsight relabeling? Do the authors apply any filtering strategies to mitigate drift?

- One conceptual thought I had while reading: the hindsight relabeling mechanism seems naturally suited to situations where the robot remains in a continuous rollout without explicit resets. Even if the final state is not the intended goal, it could still be relabeled as a useful subgoal achieved along the way. That said, in these more open-ended scenarios, the robot may occasionally end up in unfamiliar states where the high-level planner struggles to provide sensible flow predictions. I’m curious how the authors view this possibility — do they see opportunities for the system to recognize and adapt to such out-of-distribution situations (e.g., self-recovery behaviors or gradual planner refinement)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents HinFlow, a framework that grounds the point flow predicted from a high-level planner to executable actions using hindsight relabeling. Specifically, as data for flow-conditioned low-level policy training can be expensive to curate, the method deploys the low-level policy in the environment to collect exploratory rollouts, which can be annotated with flow labels using a video tracker. The relabeled data can then be used to update the policy. Evaluations conducted on two robotic manipulation benchmarks showcase the effectiveness of the method, even with environmental perturbations.  Furthermore, the framework enables cross-embodiment generalization, underscoring the robustness of point flow representations.

### Strengths
- The proposed method effectively mitigates the data scarcity problem of grounding point flow to environmental actions via hindsight relabeling.
- The extensive evaluations show that the framework can not only iteratively refine with relabeled data compared to the baselines, but also release the full potential of flow representation in task and embodiment generalization.
- Single-task policy can converge to high performance with a small number of environment interactions, highlighting the sample efficiency of the method.

### Weaknesses
- Both point flow representations and hindsight relabeling have been widely investigated by prior works for generalizability and resolving data scarcity issues, respectively. For video/flow-based methods, more baselines should be considered, like UniPi [1] and FlowDiffusion [2]. Also, it would be helpful to explain why in the Place Sphere task, the performance has a decreasing-and-increasing trend.
- The success of the proposed method heavily depends on the flow quality predicted by the high-level planner. When evaluating cross-embodiment and task generalization, no task is truly novel for high-level planners. It is also important to evaluate the robustness of the framework as a whole when handling tasks unseen to both the high-level planner and the low-level policy.
- High-level planners can be trained on video or action-free demonstrations, which ideally should benefit more from large-scale web videos from the real world for better generalizability. On the other hand, deploying an initially unstable policy for exploration and data collection can be very dangerous or costly on real robots, limiting the real-world applications of hindsight relabeling. 

[1] Du et al. Learning Universal Policies via Text-Guided Video Generation. NeurIPS 2023.

[2] Ko et al. Learning to Act from Actionless Videos through Dense Correspondences. 2023.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a hierarchical robot-learning framework, HinFlow that uses a flow-based high-level planner. The key idea is hindsight relabeling on flows during rollouts and aggregating them back to the data buffer for policy training. The paper shows 84% SR in seven manipulation tasks from LIBERO and ManiSkill using 80K online interaction steps.

### Strengths
The key idea is hindsight relabeling of achieved flows for dense supervision during online practice is well used data augmentation technique that enable sample-efficient policy training. 
The experiments highlight robustness and versatility for zero-shot generalization to novel objects/distractors when the planner covers those visuals.
The paper motivates flows as a compact, appearance-robust high-level representation and positions HinFlow as a simple bridge that grounds those plans into executable policies via online imitation.

### Weaknesses
The proposed approach of hindsight relabeling depends on the flow quality and point tracking/segmentation. Since the results are in sim where this feasible, it is unclear if the approach is robust to sim2real gaps and reliably work in real world.
The point tracking is challenging also due to 2D ambiguity for 3D motions, occlusions and high-speed motions. 

Policy trains on achieved flows but infers under predicted flows and a potential distribution shift has not been analyzed.

### Questions
How sensitive is HinFlow to planner accuracy? Do you have thresholds where online imitation starts hurting? What are the dominant failure modes?
What’s the most straightforward path to 3D motion fields (e.g., point clouds, SE(3) flows)? How would hindsight relabeling extend to 3D?
The evaluation focuses on tabletop pick-n-place tasks, with language conditioning. How much does language help multi-tasking or transfer? Does the language input need to be changed with hindsight relabelled trajectories for policy training?

### Soundness
2

### Presentation
3

### Contribution
2
