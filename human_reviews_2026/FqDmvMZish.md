# Autonomous Functional Play with Correspondence-Driven Trajectory Warping

- Decision: Accept (Poster)
- Scores: 8, 2, 4, 4, 8

## Abstract
The ability to conduct and learn from interaction and experience is a central challenge in robotics, offering a scalable alternative to labor-intensive human demonstrations. However, realizing such "play" requires (1) a policy robust to diverse, potentially out-of-distribution environment states, and (2) a procedure that continuously produces useful robot experience. To address these challenges, we introduce Tether, a method for autonomous functional play involving structured, task-directed interactions.
First, we design a novel open-loop policy that warps actions from a small set of source demonstrations (≤10) by anchoring them to semantic keypoint correspondences in the target scene. We show that this design is extremely data-efficient and robust even under significant spatial and semantic variations. Second, we deploy this policy for autonomous functional play in the real world via a continuous cycle of task selection, execution, evaluation, and improvement, guided by the visual understanding capabilities of vision-language models. This procedure generates diverse, high-quality datasets with minimal human intervention. In a household-like multi-object setup, our method is the first to perform many hours of autonomous multi-task play in the real world starting from only a handful of demonstrations. This produces a stream of data that consistently improves the performance of closed-loop imitation policies over time, ultimately yielding over 1000 expert-level trajectories and training policies competitive with those learned from human-collected demonstrations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents Tether, a system for autonomous robotic play that learns from its own interaction data without human supervision. The method combines a non-parametric correspondence-driven trajectory warping policy—which warps demonstration trajectories to new scenes using semantic keypoint correspondences—with a VLM-guided self-improvement loop that autonomously selects, executes, and evaluates tasks. Starting from only a few human demonstrations, Tether repeatedly performs tasks, identifies successes using a combination of geometric and language-based evaluation, and accumulates new high-quality trajectories for imitation learning. Experiments across 12 real-world manipulation tasks on a Franka Panda arm show strong few-shot performance, robust generalization to unseen objects, and continuous improvement through autonomous play. The system autonomously collects over 1,000 expert-quality trajectories in 26 hours, enabling downstream neural policies to match or surpass those trained on human-collected data.

### Strengths
Tether introduces an elegant and practical approach to autonomous play that avoids the dependence on large-scale pretrained policies or foundation models. The key technical idea, trajectory warping from visual correspondences, is conceptually simple but powerful, enabling robust generalization from as few as 10 demonstrations. The geometric formulation preserves spatial precision while providing semantic flexibility, allowing the method to generalize to unseen objects and scene layouts.
The system-level design is well thought out: integrating VLM-based task selection and success evaluation into a continuous play loop demonstrates genuine autonomy rather than mere offline augmentation. The experiments are impressively thorough, spanning spatial, semantic, and dynamic challenges, and even covering deformable and articulated manipulation. The real-world deployment shows convincing scale and stability, with near-zero human intervention over long periods. Finally, the downstream imitation learning results clearly validate that the autonomously generated data is useful and progressively improves learned policies, providing a strong empirical story of self-improving robotics.

### Weaknesses
1. The trajectory warping mechanism makes a strong simplifying assumption of linear spatial interpolation between matched waypoints, ignoring the coupled effects of gripper orientation, contact geometry, and dynamic feasibility. This can distort end-effector poses for curved or articulated motions such as “Bowl to Shelf” or “Wiping with Cloth,” yet the paper provides no analysis or constraint mechanism to prevent physically invalid interpolations.

2. The correspondence-based matching assumes precise stereo geometry and reliable keypoint alignment, but the paper does not quantify how correspondence errors propagate to 3D waypoint accuracy. Since failed backprojections are simply discarded as “infeasible matches,” it is unclear how often this occurs or how sensitive performance is to visual conditions like lighting or occlusion.

3. The autonomous play loop relies heavily on VLM reasoning for both task sequencing and success evaluation, but the paper treats these modules as black boxes. There is no quantitative assessment of VLM reliability, error rates, or false success detections, which makes it difficult to judge how robust the overall autonomy pipeline truly is.

4. Lastly, while the downstream policy improvements are impressive, the analysis does not disentangle quantity from diversity of collected data. Without measuring coverage or novelty of the self-generated trajectories, it remains unclear whether improvement arises from meaningful exploration or simply repeated collection of similar trajectories.

### Questions
please address the concerns above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces **Tether**, a system that allows robots to learn from "autonomous play" instead of human-collected datasets. The Tether method consists of two primary components:

1. A non-parametric manipulation policy that operates using "correspondence-driven trajectory warping". Given only a few demonstrations, this policy finds semantic keypoint correspondences between a demo pre-image and the current scene. It then selects the best-matched demonstration and "warps" its 3D trajectory to fit the new scene.

2. An autonomous, multi-task "play" procedure that uses this policy to continuously generate new data by querying Vision-Language Models (VLMs) for high-level task selection, planning, and evaluation for success.

The authors demonstrate in real-world experiments that their policy, trained on only 10 demos, outperforms baselines like Diffusion Policy (DP) and VLA models ($\pi_0$). The policy shows generalization to OOD objects and spatial configurations and succeeds on deformable and high-precision tasks.
The autonomous play dataset can further train a Diffusion Policy, which progressively improves and achieves competitive performance.

### Strengths
1. **Effective Warping Policy**: Using semantic correspondences to directly warp a full trajectory is a data-efficient approach, outperforming the baselines.

2. **Strong Empirical Result**: The policy demonstrates generalization to OOD layouts and objects. The evaluation tasks include challenging ones involving high-precision and deformable objects.

3. **Comprehensive Evaluation**: The paper successfully closes the loop. It shows not only that the policy can generate data, but also that the generated data is useful by training a downstream Diffusion Policy, which achieves good performance.

4. **Clarity and Presentation**: The paper is well-written and easy to follow.

### Weaknesses
1. **Open-Loop Execution**: The policy appears to be entirely open-loop at the trajectory level. It computes correspondences, generates a full "action plan. Without a straightforward adjustment to closed-loop execution, this may create limitations on task complexity and horizon.

2. **Task Structure**: The authors claim the play method has "approximately indefinitely composable" structure. However, it is achieved on a deliberately designed, table-top task set. For tasks that are not easily invertible (e.g, washing dishes), the system may not be able to traverse a known, cyclic task graph. 

3. **Baseline Comparisons**: The policy takes keypoints as inputs, which is mostly compared with baselines taking raw pixel images as inputs. Such comparisons could not effectively evaluate the performance of warping vs. imitation learning. Although the author includes KAT as one of the baselines, it is not effectively reproduced. Some simpler keypoint-based policies [1,2] could be included as baselines.

4. **Analysis of Failure Modes**: Given that the core idea of the paper is to make the robots play "autonomously", the analysis of the failure modes is more important, which is not covered in the paper.

### Questions
1. In line 182, how are the keypoints "identified"? Where are the keypoints in Figure 1? Moreover, the waypoints are determined by locations where the gripper toggles. This heuristic seems specific to pick-and-place. How does this definition apply to tasks like "Wiping with Cloth," where the critical part of the trajectory may not involve gripper state changes?

2. How robust is trajectory warping for articulated objects? If geometry ratios differ (e.g., cabinet openings that are higher/wider/deeper), does the warping generalize? Can it achieve cross-object generalization like [2]?

3. Human demonstrations often contain recovery from mistakes, creating data diversity in the strategy level, which trajectory warping may not capture. Is Tether primarily augmenting background/object pose? How does it compare with open-loop motion planning methods (e.g., RRT)?

4. What are the requirements for the task set? Specifically, what are the criteria to ensure cyclic task graph? How to determine if a task need to be segmented into mutiple subtask?

5. The policy finds the nearest-neighbor demo by computing correspondences against all demos in the set $\mathcal{D}$. How does the inference time of this policy scale as the number of demonstrations grows?

6. The core contribution of this paper seems to be the "autonomous data generation" instead of the policy. However, the paper mainly focus on the performace of the policy. How does its data generation compare to methods like [3,4]? 

[1] Haldar, Siddhant, and Lerrel Pinto. "Point policy: Unifying observations and actions with key points for robot manipulation." CoRL, 2025.

[2] Fang, Xiaolin, et al. "KALM: Keypoint Abstraction Using Large Models for Object-Relative Imitation Learning." ICRA, 2025.

[3] Mandlekar, Ajay, et al. "Mimicgen: A data generation system for scalable robot learning using human demonstrations." CoRL, 2023.

[4] Pan, Chuer, et al. "One Demo is Worth a Thousand Trajectories: Action-View Augmentation for Visuomotor Policies." CoRL. 2025.

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
This paper introduces Tether, a system for autonomous robotic manipulation that addresses the data scalability bottleneck in imitation learning through two main contributions: (1) a novel non-parametric policy that uses semantic visual correspondences to warp demonstrated trajectories into new scenes, achieving robust generalization across spatial and semantic variations with few demonstrations per task, and (2) a VLM-guided autonomous multi-task play procedure that enables the robot to continuously generate training data over extended periods (26 hours, producing 1000+ expert-level trajectories) with minimal human intervention. The key insight is that by leveraging strong visual priors from correspondence models rather than requiring massive demonstration datasets, the system can bootstrap from a few human demos to autonomously produce diverse, high-quality data that trains downstream neural policies to performance competitive with human-collected demonstrations, thereby shifting the scaling bottleneck from human time to robot time.

### Strengths
- The paper tackles an important problem in robot learning by demonstrating that autonomous play can generate over 1000 expert-level trajectories across 26 hours with minimal human intervention, offering a compelling alternative to labor-intensive human demonstration collection.
- The correspondence-driven trajectory warping policy demonstrates impressive robustness, outperforming strong baselines including vision-language-action models ($\pi_0$) and LLM-based methods (KAT).
- As evidenced in Fig.7, these tasks are reasonably complex that standard Diffusion Policy fails with limited data.

### Weaknesses
- While Fig.7 demonstrates that scaling to 2000 generated demonstrations improves Diffusion Policy performance, the paper lacks crucial comparisons with strong vision-language-action models like $\pi_0$, $\pi_{0.5}$ when provided with equivalent amounts of data. This comparison is critical because if VLAs can achieve high success rates with significantly fewer demonstrations (e.g., <500), it would diminish the practical value of scaling to 1000+ demonstrations per task.

- The current system generates demonstrations for tasks predefined by human experts, rather than discovering novel tasks beyond the initial demonstration scope. Works like BBSEA (arXiv:2402.08212) demonstrate that foundation models can autonomously propose new learnable tasks in unknown environments. While the present work successfully scales data collection for given tasks, extending the framework to autonomously expand the task repertoire (e.g., discovering new object interactions or compositional skills not present in initial demonstrations) would strengthen the contribution and better justify the "autonomous play" framing.

- The entire evaluation is conducted in one fixed environment configuration (table with two shelves and specific camera placements). While the method demonstrates robustness to spatial variations (object positions) and semantic variations (OOD objects) within this setup, testing across structurally different environments (e.g., different furniture arrangements, room layouts, or workspace configurations) would better validate the generalizability of the approach and strengthen claims about the method's broad applicability.

- The paper employs VLMs for task planning and success evaluation but provides no ablation studies on these critical components.

### Questions
- In the **Warping the Source Demo Trajectory** section, you describe interpolating 6-DOF gripper poses between waypoints, but it is unclear how the binary gripper open/close commands are handled during this warping process.

- For the **Out-of-Distribution Fruit and Containers** evaluation, is this transfer zero-shot (e.g., no strawberry demonstrations provided at all), or were OOD demos included in the given demonstrations?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Tether, a framework for autonomous robotic play that aims to reduce dependence on large-scale human demonstrations for imitation learning. The method combines a non-parametric, correspondence-driven policy for robust generalization with a vision-language model (VLM)-guided autonomous play loop that continuously generates new training data.

The core insight is to leverage semantic image correspondences between demonstrations and novel scenes to warp existing robot trajectories. This approach enables a robot to adapt to a small set of demonstrations (as few as 10) in new environments and tasks without requiring retraining. The authors then deploy this policy in a self-improving cycle of multi-task play: the robot autonomously selects, executes, and evaluates tasks using VLM reasoning for both planning and success detection.

Over 26 hours of autonomous operation, Tether produces over 1,000 successful real-world trajectories with negligible human intervention (~0.26% of runs requiring resets). The collected data are subsequently used to train downstream neural policies, which achieve success rates comparable to or better than policies trained on human demonstrations.

### Strengths
- The paper introduces a clever and practical way to reuse a small number of demonstrations via keypoint correspondence-driven trajectory warping. This method leverages advances in visual correspondence (e.g., DINOv2 + Stable Diffusion) to enable generalization across large spatial and semantic variations without retraining.

- Experiments on 12 diverse manipulation tasks—including deformable, articulated, and precision tasks—show consistent improvement over baselines such as Diffusion Policy, ε₀ (OpenVLA), and KAT. The approach exhibits impressive semantic generalization, successfully adapting demonstrations from, e.g., a pineapple to an apple or a bowl to a cup

### Weaknesses
- One major contribution, at the same time, can be a limitation as well, is the usage of correspondence. While correspondence-based warping enables generalization, its performance is tied to the quality and stability of keypoint matching. In cases with heavy occlusion, textureless surfaces, or large deformations, this component may degrade. It would be nice for the authors to include such discussions in the paper on which external disturbances the policy is robust to, and which disturbances it cannot handle.

- Non-parametric vs. generalization claim: The paper argues that a non-parametric policy generalizes to unseen objects and layouts. However, this non-parametric methods always require access to demonstrations at inference time, which limits scalability and contradicts the claim of “extreme generalization”. It would be helpful for the authors to explain this better in the text.

- Clarification question: How does the relatively coarse 10 cm correspondence threshold and the potential variability of VLM-based verification affect the reliability of results, especially for precision-sensitive tasks like coffee pod insertion with an 8 mm margin?

### Questions
See the points above. I am eager to see the paper get improved.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposed 1) a non-parametric table-top manipulation policy by warping source demonstrations, which has strong generalization performance, and 2) a framework for autonomously generating robot demonstrations with minimal human interventions. The proposed method is evaluated on 12 table-top manipulation tasks. The first part of the experiments shows the proposed non-parametric policy outperforms three baselines. The second part of the experiments shows the proposed framework can continuously gather demonstrations with minimal human interventions, and the downstream policy performance scales well with the autonomously-collected demonstrations.

### Strengths
- The paper addresses an important problem - scaling data collection. Through the experiments, the paper shows that it enables collection of a large number of robot demonstrations with minimal human interventions, and the collected demonstrations are of comparable quality to human demonstrations in terms of downstream policy learning performance.
- The paper is clear.

### Weaknesses
- The baselines for Section 4.2 are not fair comparison, and thus weakens the claim that the proposed Tether policy has superior generalization. Concretely, Tether is given 10 demonstrations, while $\pi_0$ with FAST tokenizer is evaluated zero-shot. Diffusion policy, not meant for few-shot setting, is also disadvantaged by being trained from scratch with 10 demonstrations.

### Questions
- The autonomous play framework proposed in Section 3.2 appears to be policy-agnostic. Can we use the same framework with different policies?

### Soundness
3

### Presentation
3

### Contribution
3
