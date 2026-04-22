# AnyPos: Automated Task-Agnostic Actions for Bimanual Manipulation

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Learning generalizable manipulation policies hinges on data, yet robot manipulation data is scarce and often entangled with specific embodiments, making both cross-task and cross-platform transfer difficult. 
We tackle this challenge with**task-agnostic embodiment modeling**, which learns embodiment dynamics directly from ***task-agnostic action*** data and decouples them from high-level policy learning. By focusing on exploring all feasible actions of the embodiment to capture what is physically feasible and consistent, task-agnostic data takes the form of independent image-action pairs with the potential to cover the entire embodiment workspace, unlike task-specific data, which is sequential and tied to concrete tasks. This data-driven perspective bypasses the limitations of traditional dynamics-based modeling and enables scalable reuse of action data across different tasks. 
Building on this principle, we introduce **AnyPos**, a unified pipeline that integrates large-scale automated task-agnostic exploration with robust embodiment modeling through inverse dynamics learning. AnyPos generates diverse yet safe trajectories at scale, then learns embodiment representations by *decoupling arm and end-effector motions* and employing a *direction-aware decoder* to stabilize predictions under distribution shift, which can be seamlessly coupled with diverse high-level policy models. 
In comparison to the standard baseline, AnyPos achieves a 51\% improvement in test accuracy. On manipulation tasks such as operating a microwave, toasting bread, folding clothes, watering plants, and scrubbing plates, AnyPos raises success rates by 30-40\% over strong baselines. These results highlight data-driven embodiment modeling as a practical route to overcoming data scarcity and achieving generalization across tasks and platforms in visuomotor control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes AnyPos, a framework for learning robot actions in a task-agnostic way. Its core contribution lies in two connected parts: first, an automated exploration pipeline that uses a reinforcement-trained mapper and safety constraints to sample the robot’s workspace and generate large quantities of collision-free image–action pairs without teleoperation or task labels; second, an inverse-dynamics embodiment model that predicts feasible joint positions from visual inputs using an arm-decoupled design and a direction-aware decoder for sub-degree precision. Together, these components aim to provide a reusable “feasibility prior” that can later be coupled with high-level policy or video-generation models to execute diverse manipulation tasks more efficiently than task-specific datasets.

### Strengths
The paper’s strengths lie in its clear problem framing and modular design—it separates physical feasibility learning from task semantics through a well-structured pipeline that’s easy to follow. The authors present a fully automated, high-throughput data collection system that efficiently explores the workspace without teleoperation, coupled with a safety-aware exploration scheme that enforces collision avoidance, joint limits, and inter-arm constraints. The precision-oriented inverse dynamics model introduces practical architectural tweaks, such as arm-decoupled estimation and a direction-aware decoder, which demonstrably improve stability and accuracy. Finally, the real-robot replay results show that the learned embodiment prior can be executed reliably, suggesting the overall framework has strong potential as a practical foundation for scalable, modular robot learning.

### Weaknesses
The paper’s main weaknesses lie in its limited empirical validation and unclear justification of core design choices. While the motivation is compelling, the work never actually demonstrates that separating semantics from feasibility improves transfer — all experiments are confined to a single robot, with no cross-task, cross-embodiment, or cross-view evaluations. The decision to use reinforcement learning for deterministic action generation is conceptually weak and lacks comparison against straightforward baselines like inverse kinematics or behavior cloning with physical constraints. The proposed Direction-Aware Decoder adds engineering complexity without clear theoretical grounding or fine-grained ablations to prove necessity. Moreover, the probabilistic decomposition introduced early in the method is never operationalized in training, leaving a gap between the stated formulation and the implemented model. Overall, the paper presents a strong motivation but delivers mostly incremental engineering under a broader conceptual narrative that remains unverified.

### Questions
1. Why is reinforcement learning needed for deterministic action mapping, and what concrete advantages does it offer over a straightforward inverse-kinematics or behaviour-cloning approach with safety and collision constraints?
2. Deterministic vs. probabilistic formulation: The paper presents a probabilistic factorization suggesting a distributional treatment of actions, but the actual system predicts a single deterministic joint configuration. How is this formulation relevant if the model never models uncertainty or multiple feasible actions? Since the “world model” is deterministic, how does this help separate “what to do” from “what is physically feasible,” and what additional insight or capability does this probabilistic claim really provide?
3. How does the model handle camera-view and embodiment differences, given that the state representation is trained purely in the 2D image domain—does it rely on multi-view training, 3D understanding, or re-collection for new viewpoints?
4. What evidence supports that separating feasibility from task semantics improves transfer, and can the embodiment model generalize across unseen tasks or robot morphologies without retraining?
5. What is the real contribution of the Direction-Aware Decoder (DAD)? The paper shows marginal experiments but lacks fine-grained ablation; if it is mainly an engineering tweak, why treat it as a core architectural innovation?
6. What concrete metrics and comparisons demonstrate that the RL-generated dataset is superior (in workspace coverage, safety rate, or diversity) to simpler IK-based or rule-based data generation pipelines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposed a novel method to collect random exploration data to learn good representation for the bi-manual manipulation tasks. THe method use a biased sampling stragtegy for exploration and learns a good representation via inverse dynamics model. Then the model is used to learn downstream imitation learning policy. Experiments show better performance than learning from scratch with pure teleoperation data and previous approach.

### Strengths
1. The method allows random exploration without direct human supervision, which relax the requirement on human demonstration.

### Weaknesses
1. The tasks are not safety critical. The method is hard to generalize to general-purpose tasks, like ones involving safety concerns. Random exploration is not applicable.
2. The writing is poor. The beginning of the paper involves too much distraction of mathematical formulation. It is more straightforward to describe the method in an intuitive way.
3. No ablation on the model design. Why all components are needed?

### Questions
1. How's each component contribute to the final performance? How does baselines implemented? Do they similar architecture?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an embodiment modeling framework that learns a vision-to-action mapping directly from large-scale, automatically collected trajectories. It employs an RL-based explorer to sample EEF targets across the 3D workspace and convert them into feasible joint configurations, creating a large dataset. The inverse dynamics model trained on this data maps visual observations to joint positions with good precision, aided by an arm-decoupled architecture and a Direction-Aware Decoder. The system can then pair this embodiment model with high-level policies such as video-generation or VLA models, achieving real-world bimanual manipulation success while requiring no human teleoperation.

### Strengths
The paper provides a large scale dataset by RL-based exploration, and then recording 610k safe, diverse image-action pairs in 10 hours, which is claimed to be 30× faster than human teleoperation.

The quantitative results for real-world validation looks strong, with high replay success, the demo video also shows performance, though with some failure cases.

The paper presents detailed implementation and hyperparameter reporting.

### Weaknesses
The paper does not provide comparison when replacing learned inverse dynamics model (IDM) with traditional collision-free inverse kinematics (IK). most baselines are vision encoders (ResNet or DINOv2) rather than control pipelines, so the benchmark scope is narrow. Without such an ablation, it is unclear whether the learned model actually outperforms or simply replicates IK performance under noise and multi-arm constraints, since IK is guaranteed to be generalizable but a learned model is not

Variables are introduced abruptly in introduction without clear linkage to the system. The overall abstraction and introduction could better define these terms and explicitly describe the model’s input and output, integrating this explanation with Figure 1 to clarify the pipeline flow.

The cross embodiment experiment is limited given the paper claims embodiment-agnostic modeling

Arms overlap with objects or arms or move out of frame -- conditions that are common in cluttered manipulation scenarios, which the model seems not be able to address.

### Questions
It seems AnyPos depends only on URDF/kinematics and can be “replayed” for new viewpoints. How robust is the IDM to camera shifts?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AnyPos, a framework for task-agnostic embodiment modeling in robotic manipulation. It addresses the critical challenges of data scarcity, task-specificity, and poor cross-platform generalization in robot learning. The core innovation lies in decoupling the learning of physically feasible actions from high-level task semantics. The paper achieve this through a two-stage pipeline: First, an automated, safety-aware exploration process collects a dataset of diverse, feasible robot trajectories without human teleoperation. Second, an inverse dynamics model is trained on this data, employing arm-decoupled estimation and a direction-aware decoder to achieve high-precision action prediction robust to distribution shifts. The resulting embodiment model serves as a reusable "motion prior" that can be seamlessly coupled with various high-level policy models. Experiments show a good improvement in action prediction accuracy and real-world task success rates over strong baselines.

### Strengths
S1) The paper is well written, the proposed pipeline is simple and easy to follow.

S2) The automated data collection framework demonstrates a highly effective strategy for generating large-scale robotic datasets.

S3) The design of arm-decoupled estimation and the direction-aware decoder directly addresses the challenges of high-dimensional action spaces and visual ambiguity in task-agnostic data. These components are empirically shown to be critical for achieving the high-precision action prediction required for real-world deployment.

### Weaknesses
W1) I concerns the practical deployment and generalization of the automated data collection framework. My primary question revolves around the necessity and process of training the RL-based projection policy (`f_RL`). To clarify its scope and limitations: when encountering a new physical scene with different objects and layouts, must a new RL policy be trained from scratch in a simulation that explicitly reconstructs that specific bounded workspace volume? Furthermore, was a single, universal RL policy used to collect all the task-agnostic data for the diverse tasks in the paper, or were multiple scene-specific policies required? Ultimately, I seek to understand the inherent limitations of this approach, specifically, how a policy trained on one scene is expected to perform when deployed in a novel, unseen environment without retraining, particularly regarding its ability to avoid collisions and maintain feasibility.

W2) While the paper's core contribution lies in using improved data to train a more robust inverse dynamics model, the proposed "task-gnostic embodiment modeling" framework ultimately relies on a pipeline combining a video generation model (VGM) with this learned model. This architecture appears to have limited practical utility due to its inherent susceptibility to error propagation. The system's performance is heavily contingent on the quality of the generated videos, where any inaccuracies in the predicted future frames are directly translated into action errors by the inverse dynamics model. A shortcoming of the work is the lack of a detailed analysis quantifying this compounding error and visualizing the pipeline's robustness.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
