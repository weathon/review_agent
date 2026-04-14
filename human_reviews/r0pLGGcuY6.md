# What's the Move? Hybrid Imitation Learning via Salient Points

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 5, 6

## Abstract
While imitation learning (IL) offers a promising framework for teaching robots various behaviors, learning complex tasks remains challenging. Existing IL policies struggle to generalize effectively across visual and spatial variations even for simple tasks. In this work, we introduce **SPHINX**: **S**alient **P**oint-based **H**ybrid **I**mitatio**N** and e**X**ecution, a flexible IL policy that leverages multimodal observations (point clouds and wrist images), along with a hybrid action space of low-frequency, sparse waypoints and high-frequency, dense end effector movements. Given 3D point cloud observations, SPHINX learns to infer task-relevant points within a point cloud, or *salient points*, which support spatial generalization by focusing on semantically meaningful features. These salient points serve as anchor points to predict waypoints for long-range movement, such as reaching target poses in free-space. Once near a salient point, SPHINX learns to switch to predicting dense end-effector movements given close-up wrist images for precise phases of a task. By exploiting the strengths of different input modalities and action representations for different manipulation phases, SPHINX tackles complex tasks in a sample-efficient, generalizable manner. Our method achieves **86.7%**  success across 4 real-world and 2 simulated tasks, outperforming the next best state-of-the-art IL baseline by **41.1%** on average across **440** real world trials. SPHINX additionally generalizes to novel viewpoints, visual distractors, spatial arrangements, and execution speeds with a **1.7x** speedup over the most competitive baseline. Our website (http://sphinx-manip.github.io) provides open-sourced code for data collection, training, and evaluation, along with supplementary videos.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents SPHINX, a novel framework for imitation learning in robotics. SPHINX integrates a hybrid approach by leveraging multimodal inputs (point clouds and wrist camera images) and using both waypoint-based and dense action predictions. It introduces a method for switching between these modalities and action types depending on the task phase, which enables better spatial and visual generalization. The paper demonstrates how SPHINX can handle long-horizon tasks with high precision, achieving superior results compared to state-of-the-art baselines, with an average performance improvement of 41.1%.

### Strengths
1. Effective Multimodal Input Usage: The use of point clouds for spatial awareness in long-range movements and wrist-camera images for precision is well-conceived. By switching between input modalities at the appropriate phases of a task, the system exploits the strengths of each modality to improve both efficiency and accuracy.

2. Empirical Performance: The paper presents strong experimental evidence, showing significant performance improvements across multiple real-world and simulated tasks. Specifically, it demonstrates SPHINX’s ability to generalize across novel configurations, camera viewpoints, and visual distractors, which is critical for practical deployments in environments.

### Weaknesses
1. Similarity to SGRv2: I noticed that the waypoint policy in this paper is quite similar to the one proposed in SGRv2 [1]. However, the authors do not discuss this in the related work section, nor do they compare their method against SGRv2 in the experiments. Including a detailed comparison or discussion would help clarify how SPHINX improves upon or differs from existing approaches in waypoint-based policies.

[1] Zhang, Tong, et al. "Leveraging Locality to Boost Sample Efficiency in Robotic Manipulation." arXiv preprint arXiv:2406.10615 (2024).

---

2. Insufficient Visual Generalization Experiments: The experiments on visual generalization are somewhat limited. It remains unclear whether SPHINX’s salient point-based design allows the policy to generalize to different objects within the same category (e.g., cups of varying colors or shapes). Additionally, it would be valuable to test SPHINX’s robustness to changes in environmental factors such as background, lighting, or table textures, which are common in real-world settings. Expanding the generalization tests to cover these variations would provide a more comprehensive evaluation of SPHINX’s capabilities.

3. Limited Exploration of Dynamic Tasks: While SPHINX excels at quasistatic tasks, its performance on dynamic tasks (e.g., tasks requiring fast, reactive control) is not explored. The reliance on a linear controller for waypoints may limit the system’s applicability in environments requiring fast, non-linear movements.

### Questions
For questions and concerns, please refer to the weaknesses section above. Overall, I believe this is a strong paper that effectively leverages the advantages of both waypoint and dense policies. The experimental results significantly outperform the next best state-of-the-art IL baseline, and the writing is clear, making the paper easy to follow and understand.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SPHINX which interleaves 3D saliency-based waypoint prediction with fine-grained policy that uses wrist camera image. The proposed method, SPHINX, is compared against SOTA imitation learning methods for robotic manipulation like Diffusion Policy, 3D Diffusion Policy, HYDRA, and also a fine-tuned openVLA. The authors also compare SPHINX with the ablated waypoint models without saliency points. The experiments were done on 4 real world tasks (cup stack, train track, coffee, open drawer) and two simulated tasks in Robomimic environemnt.

### Strengths
The proposed method seamlessly integrates multiple concepts in robotic manipulation such as waypoints, affordances, and coarse-to-fine, 2D+3D hybrid approach. These techniques were often explored separately in prior works. Leveraging these techniques, SPHINX effectively solves long-horizon tasks (due to the waypoint approach) and is robust to OoD settings (due to the 3D + saliency structure). Empirical results are strong enough to support these claims. SPHINX show substantial improvements over previous SOTA methods. Overall, the paper is of high quality.

### Weaknesses
The primary weakness of this paper lies in its lack of novelty. While the paper successfully integrates many concepts into one, each concept is not novel. For instance, guiding fine-grained manipulation policies with coarse waypoints is not a new approach. Similarly, using saliency prediction to assist manipulation has been widely addressed in prior works, often categorized as affordance-based approach. It is well known that 3D-based policies are inherently more robust than 2D models in OoD scenarios such as existence of distracting objects.

Additionally, there are several keypose-based methods (which are similar to waypoints) that use 3D point clouds, including PerAct, Act3D, 3D Diffuser-Actor, RVT-1/2. It is unclear why the waypoints of SPHINX is better than that of these models.

Lastly, the saliency-based waypoint approach has clear limitations. While it is indeed a very useful inductive bias for many tasks, it may harm the performance in certain cases, particularly when the grippers are far away from any point cloud in the scene. This isn't true for more general methods like Diffusion Policy. Thus, it’s unsurprising that SPHINX outperforms Diffusion Policy, for its added structure and inductive bias. Furthermore, SPHINX relies on additional human-annotated keypoints. While collecting such data is not particularly difficult, this limits SPHINX being applicable to tasks in which the concept of keypoint doesn't really make sense or is unidentifiable for human.

Overall, while this paper presents a well-executed combination of existing ideas, it appears rather incremental. The experimental results are promising, making it a valuable contribution to a robotics-focused conference. However, the contribution may not be sufficient to recommend acceptance at an ML conference, where significant methodological novelty is typically expected.

### Questions
1. Is there any positive transfer between the dense policy part and the waypoint prediction part? Or are these just separate models with no synergy in terms of learning?

2. What specific advantage does SPHINX’s waypoint approach offer? I would expect SOTA keypose models like PerAct, Act3D, 3D Diffuser-Actor, and RVT-1/2 to outperform the proposed waypoint transformer. These models are more structurally sophisticated and battle-tested. They also incorporate locality bias, which I believe would provide benefits similar to those of the paper's saliency-based approach. Adding displacement to saliency essentially assigns affordance to specific, localized areas. Furthermore, these models do not require human-annotated saliency, and outputs SE(3) keypose which would be much stronger than R^3 waypoints.

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
4

### Summary
This paper builds on the HYDRA approach by splitting motion into two modes: waypoint-reaching and dense manipulation. The key improvements are: (1) using point clouds instead of RGB images for more robust waypoint localization and using separate PointNet, (2) upgrading the dense manipulation module from a simple MLP to a diffusion policy for smoother control, and (3) introducing "salient points" that explicitly signal when to switch between modes.

The method is evaluated on three challenging, long-horizon real-world tasks, with results showing clear performance gains over baseline approaches. The paper is clearly written, and the experiments are thoughtful, including interesting tests on out-of-distribution scenarios.

### Strengths
1. **Enhanced Localization with Point Clouds**: The use of point clouds bolsters waypoint localization, offering a more resilient approach than RGB data alone.
2. **Improved Precision in Dense Manipulation**: Replacing the MLP with a diffusion policy for dense manipulation leads to greater accuracy during complex manipulation phases.
3. **Streamlined Mode Switching**: By introducing salient points, the authors provide a clear and novel mechanism for mode switching, simplifying task segmentation and improving execution flow.
4. **Comprehensive Real-World Tests**: The real-world experiments are particularly strong, presenting complex tasks that highlight the robustness and versatility of the approach.

### Weaknesses
1. **Reliance on Salient Points**: The approach depends on expert-determined salient points, which may limit scalability, especially if multiple annotators are involved, as this could introduce inconsistencies in training data.
2. **Lacks Details and Analysis in Experiemnts**: The baseline are too low to tell whether the baselines are correctly run and under fair comparison. It will be better to have some simple tasks which baselines can perform not that bad or provide a failure analysis about how they fail.

### Questions
1. **Real-World Experiment Details**: Could the authors share more on how they measure success rates? Do you count broken down points as OpenVLA, or does it only reflect the final task outcome? A failure analysis would also be insightful.
2. **Point Cloud Format**: What specific format of point cloud data does the waypoint transformer use? Are both RGB and XYZ coordinates involved, or only XYZ? As I am curious how the salient point prediction can distinguish the pink and yellow cup.
3. **Handling Distant Salient Points**: When a salient point isn’t near a physical point—like opening an oven or a door—how does the method handle this? If there’s a substantial distance to any nearby object, does the model has to use the mode of dense manipulation?
4. **OpenVLA Training Clarifications**: The performance of OpenVLA seems unexpectedly low in some tasks, like cup stacking. Could the authors clarify their training setup for OpenVLA? For example, was image augmentation applied, and was it trained on a single or multiple tasks?
5. **HYDRA Performance Insights**: It would be interesting to understand why HYDRA, despite using a similar waypoint and dense manipulation modes, performs way less effectively. Does it struggle more in the waypoint phase or the dense manipulation phase?
6. **Camera Viewpoint Calibration In Novel Viewpoint**: Were novel camera viewpoints recalibrated during inference? Additionally, are the training and inference data in the same coordinate system, like the robot’s base?
7. **Waypoint Transformer**: It’s interesting that the waypoint transformer model can effectively handle unordered points with relatively small data. Could you provide more details on the network structure and training? What's more, were any data augmentations applied?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In this work, the authors consider the task of imitation learning in the context of tabletop robot manipulation tasks. Specifically, they investigate the well-known tension between the need for both long-horizon and short-horizon control: e.g. non-dexterous freespace motion for a large fraction of a trajectory, followed by a short dexterous stretch. They articulate the hierarchical nature of many manipulation tasks, and propose to address the hierarchy with a hybrid policy which can switch between two modes: a high-level waypoint prediction which is grounded in a global observation point cloud, and a low-level fine-grained control which operates on wrist camera inputs. They describe a policy architecture and training procedure to fit this paradigm, as well as a data collection/annotation method for generating datasets with required annotations suitable for training. They then demonstrate the efficacy of their policy on 3 custom real-world manipulation tasks.

### Strengths
# Originality

* While hierarchical approach has been proposed, for instance in HACMan, NDF, TAX-Pose, etc, and explicit mode-switching between policies has been explored, I haven’t really seen this sort of mode switching in imitation learning for long-ish horizon manipulation.
* Giving the policy a mechanism to change its inputs is a neat design consideration
* First time seeing diffusion policy in a hierarchical context.

# Quality

* The real-world experiments are well-designed, and demonstrate good performance on a relatively small number of demos
* The idea is straightforward and a reasonable solution to this class of quasistatic manipulation problems.
* Decent ablation coverage.
* The design of the visual generalization experiments was interesting.

# Clarity

* Very clear and well-written.

# Significance

* Reasonable factorization of the problem, so might be useful for for practitioners looking to solve this class of problems expediently and don’t have needs for generalizing outside of quasi-static pick-and-place tasks.

### Weaknesses
* The model’s hypothesis class (e.g. mode switching between long-distance free space motion, and short horizon fine-grained manipulation) imposes a major constraint on the kinds of problems it can represent effectively. Of course, in the extreme case, either one or the other mode can always be predicted, but in these settings no benefit / insight is offered, and hard attention switching across the inputs limits flexibility.
* The assumptions about dataset preparation are a major weakness/limitation here, for two reasons:
    1. Adding a different label for mode is a major annotation requirement, and can be quite noisy/arbitrary.
    2. The notion of labeled “salient points” may make some sense because humans do attend to salient points reasonably consistently, but generally speaking they are not a consistent/principled thing to expect a human to label - who is to say which specific points should be annotated? Seems quite noisy, and a major source of error.
* Comparisons were somewhat weak.The authors assert that long-horizon goal prediction is not precise enough for this kind of imitation learning, but evidence from e.g. 3DDA, RPDiff, NDF, TAX-Pose, etc. demonstrate that the quasistatic problems considered in the real-world tasks are probably solvable with goal prediction.
* Benchmark tasks are also somewhat weak. While the real-world experiments are good, the authors should compare against other methods on simulated benchmarks (including annotating using their system or a synthetic annotation scheme) such as RLBench where some medium-horizon tasks have been examined with a litany of different approaches.
* There’s not much analysis of sample complexity.
* Several related works are missing / not discussed:
    * Spatial grounded policies: Spatial Action Maps, Transporter Nets, HACMan, 3D Diffuser Actor
    * Relative placement / reasoning: NDF, TAX-Pose, etc.
    * Keyframing: Many of Stephen James’ works on top of RLBench look at long-horizon gripper positional reasoning and keyframing.
    * The “soft salience” mechanism has been used before (e.g. TossingBot, many others), so while it’s pretty standard it would be good to compare / reference
* The variation for initial conditions in the real-world experiments appear somewhat limited, and having the high-level policy run on a global world frame is a bit unfair as a selling point because it doesn’t have to generalize as much by construction (probably could modify the other methods to have similar properties).

### Questions
* Can the authors speak to sample complexity analysis for the different methods considered?
* How could this be used / modified to not require human annotation of mode/salient points directly?
* How does this method perform on simulated benchmarks w/ published baselines/metrics?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces an imitation learning framework, SPHINX (Salient Point-Based Hybrid ImitatioN and eXecution), designed to improve the generalization and efficiency of robotic control in complex manipulation tasks. Traditional imitation learning approaches often struggle with diverse visual and spatial variations, especially in tasks requiring long-term planning and precision. SPHINX addresses these challenges by utilizing a hybrid approach that combines two modes of execution: waypoints for large, coarse movements and dense actions for precise, close-up interactions. They show extensive empirical results that demonstrate SPHINX's effectiveness across both real-world and simulated tasks, showing a substantial performance improvement over existing IL baselines.

### Strengths
1. This paper proposes a two-stage framework. The first stage leverages salient points for generating long-range movements, specifically as waypoints, followed by interpolation for motion planning and control. The second stage employs an image-based action similar to a diffusion policy, where actions are based on EE poses. This framework is reasonable for multi-stage manipulation tasks that require contact.
2. The paper is relatively complete, covering the proposed method, dataset collection, simulations, and real-world experiments. The baselines are adequately diverse, and the real-world experimental demonstrations are effectively presented.
3. The writing in this paper is clear, with accurate illustrations that aid understanding of the proposed framework. Explanations of the data collection module and experiments are well-supported with corresponding videos.

### Weaknesses
1. The novelty of the paper could be further clarified. Overall, applying a hierarchical approach to improve generalization and performance in IL is reasonable. This is novel within IL (only my point, if other reviewers can provide references, I will defer it). However, using a two-stage approach (i.e., long-range navigation + fine-grained manipulation) to handle long-term manipulation tasks is a common strategy, and I suggest the authors add a discussion on this.
2. There are some concerns regarding the visual module. The first stage’s visual module appears to use a transformer for salient point extraction, which resembles affordance learning. It would be beneficial to include a discussion on affordance learning. Additionally, the paper seems to use a limited training set for salient point training, such as 30 samples for cup stacking. Without pre-training, I am skeptical that these datasets can achieve accurate and generalizable visual perception.
3. Concerns related to the robotics aspect. The paper does not seem to mention object segmentation or part-level pose estimation, which also appear to be integrated within the waypoint transformer. It is unclear if this approach is effective or if additional processing was performed in the real-world experiments.

### Questions
1. There seems to be a very basic error in the paper: I believe the EE pose should be 6 DoF, but the paper frequently mentions 7 DoF. Is this due to the use of quaternions for rotation, or does it include the EE’s open/close state as an extra DoF?
2. Generalization problem. Does the generalization reported in the paper include generalization across objects of different appearances and shapes, or is it limited to generalization over the initial pose?
3. Mode switching and subtask decomposition. The paper does not seem to address subtask decomposition for long-term tasks, appearing to rely on mode switching. For the two proposed modes, how does the model ensure continuity across temporal sub-steps, and how does it detect the salient point corresponding to the next subtask (distinct from the previous step)?

Overall, this paper is in good shape, with a reasonable method, sufficient experiments, and clear implementation details. If the above concerns and questions can be addressed, I would raise the score.

### Soundness
3

### Presentation
3

### Contribution
3
