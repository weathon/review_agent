# R2RGEN: Real-to-Real 3D Data Generation for Spatially Generalized Manipulation

- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Towards the aim of generalized robotic manipulation, spatial generalization is the most fundamental capability that requires the policy to work robustly under different spatial distribution of objects, environment and agent itself. To achieve this, substantial human demonstrations need to be collected to cover different spatial configurations for training a generalized visuomotor policy via imitation learning. Prior works explore a promising direction that leverages data generation to acquire abundant spatially diverse data from minimal source demonstrations. However, most approaches face significant sim-to-real gap and are often limited to constrained settings, such as fixed-base scenarios and predefined camera viewpoints. In this paper, we propose a real-to-real 3D data generation framework (R2RGen) that directly augments the pointcloud observation-action pairs to generate real-world data. R2RGen is simulator- and rendering-free, thus being efficient and plug-and-play. Specifically, given a single source demonstration, we introduce an annotation mechanism for fine-grained parsing of scene and trajectory. A group-wise augmentation strategy is proposed to handle complex multi-object compositions and diverse task constraints. We further present camera-aware processing to align the distribution of generated data with real-world 3D sensor. Empirically, R2RGen substantially enhances data efficiency on extensive experiments and demonstrates strong potential for scaling and application on mobile manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces R2RGen, a simulator‑ and rendering‑free framework that generates large amounts of real‑to‑real 3D training data for robotic manipulation from as little as a single human demonstration. Instead of collecting many demonstrations to cover spatial variation (object layouts, robot base/viewpoint), R2RGen edits both the point‑cloud observations and the action trajectories directly in 3D, then trains a 3D visuomotor policy (iDP3) purely on the generated data.

Key components:

Pre‑processing: Parse a single demo into complete object, environment, and arm point clouds using segmentation + template‑based 3D tracking; lightly annotate skill/motion segments and the in‑hand/target objects

Group‑wise augmentation with backtracking: Treat each skill as acting on a group (targets + in‑hand), apply identical SE(3) transforms per group, and backtrack from the final skill while maintaining a set of “fixed” objects to preserve causal and multi‑object spatial constraints; plan motions to connect skills.



Overall, I think this paper is technically sound. However, the main concern lies in its novelty and contribution. Most parts of the system’s implementation and pipeline design appear quite straightforward, and I’m somewhat confused about the claimed level of generalization in this work.

### Strengths
Real‑to‑real pipeline that edits both observations and actions. Directly augments point‑cloud observations and end‑effector trajectories in a shared 3D frame; no on‑robot rollouts needed to regenerate visuals.

Handles multi‑object & bimanual skills. The group‑wise augmentation + backtracking idea preserves inter‑object structure (targets + in‑hand) and causality across skills—something object‑centric methods struggle with.

Camera‑aware post‑processing. Project → crop → patch‑wise Z‑buffer → fill → unproject aligns synthetic clouds with RGB‑D sensor statistics, mitigating the “visual mismatch” that plagues large transforms/viewpoint changes.

### Weaknesses
Rigid‑object & known‑template bias. The completion/tracking relies on template‑based 3D tracking and assumes rigidity; non‑rigid/unknown objects or significant shape variation are out of scope.

Environment prerequisites. Needs an empty‑scene capture to build a complete environment cloud and assumes scene static within a trial—less realistic for cluttered/dynamic settings.

Planar/task assumptions. Augmentations use XY translations + Z‑yaw with a tabletop plane fit; generalization to shelves, drawers, vertical workspaces, or tasks needing pitch/roll changes is unclear.

Feasibility checks are under‑specified. Random group transforms plus motion planning can create unreachable poses, collisions, or implausible contacts; the paper doesn’t detail strong kinematic/visibility rejection tests, so label noise risk remains.

Baseline coverage is narrow. Head‑to‑head comparisons focus on DemoGen and only on tasks it can handle; there’s no thorough comparison to simulator‑based generators (e.g., MimicGen variants with minimal rollouts) or strong 2D/3D data‑augmentation baselines.

It is apparent that the simulation generated data is cheaper than the real world operation. Thus, the rendering-free and simulation free setting is somewhat not make sense.

### Questions
The method appears straightforward to implement, and I’m not yet convinced about its core insight or novelty. The contribution currently reads as a relatively small idea. Could you provide a more precise, mathematically grounded formulation to clarify what is fundamentally new here?

My main concern is the “simulation- and rendering-free” claim. It seems that the same data generation could be accomplished in simulation with concise code—for example, using GSworld or other Gaussian-splatting–based densification pipelines. As a result, the argument against simulation feels weak. Please provide concrete evidence and justification for avoiding simulation as a data-generation source (e.g., quantitative comparisons, failure cases where simulation underperforms, or constraints that make simulation impractical for your setting).

GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation

### Soundness
3

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
R2RGen offers a real enhancement Pipeline based solely on visible light video, which can reliably scale a single person's demonstration to hundreds of different point cloud trajectories. It achieves a success rate comparable to that of using 25 times the human demonstration trajectory on 8 real robot tasks with just one human demonstration trajectory +R2RGen. This work has been meticulously designed and extensively evaluated, demonstrating excellent deployment value. However, some design choices lack quantitative sensitivity analysis, and the experiments did not further explore the statistical significance of the results and failure cases. Moreover, under basis uncertainty, the movement operation expansion cannot maintain external calibration - although the movement operation is only used as a test case for limit generalization testing and is mainly left for further consideration in the future. But overall, this is a very valuable piece of work.

### Strengths
1.In response to the problem that previous works could not handle the spatial changes of multi-object structures well, an original "group-level retroactive augmentation" method was proposed to maintain the constraints of multi-object structures, breaking through the single-object limitation of the Baseline method DemoGen.
2. The camera perception post-processing proposed by R2RGen addresses occlusion/missing caused by large rotations, significantly reducing visual mismatch.
3. The experimental scale is sufficient and the experimental process is relatively complete: approximately 1,500 real-machine rollout operations, 8 different types of grasping operation tasks, and it simultaneously includes 4 different levels of ablation experiments, with high confidence in the results.

4.The entire process is based on real-world RGB video data, without the need for simulation or rendering. The mobile base is plug-and-play, and the engineering value is clear.

### Weaknesses
1.Zero analysis of failure cases: Although the experimental volume of the paper is very sufficient and the improvement in the Success Rate is significant compared to the baseline method DemoGen, the analysis of the experimental results only provides the success rate and does not classify failure results such as collision, non-capture, and out-of-bounds, nor does it conduct further analysis of failure cases.
2. Insufficient statistical significance of the experimental results: The main results of the paper, Table 1, lack confidence intervals or hypothesis tests. Relying solely on the current data to illustrate that "achieving the effect of using 25 times the manual demonstration trajectory" may not be the most rigorous.
3. Some hyperparameter sensitive data in the experiment are missing. For instance, the patch-Z-buffer radius r and the depth threshold δ are only given as single values. These two parameters obviously affect the occlusion judgment and the final point cloud quality, but there is no sensitivity curve for the experimental hyperparameters.
4. Although the movement operation is only used for extreme generalization tests, the assumption of using a fixed camera leads to loopholes in the experimental process of the movement operation: assuming that "the RGB-D camera is fixed to the base", the navigation stop error in the movement operation experiment (§ 4.5&D) is greater than 5 cm, which causes the actual external parameters of the camera to change and may introduce geometric inconsistencies.

### Questions
1.In Table 1, all success rates only provide point estimates for a single round of 32 to 64 rolluts, with neither standard deviations nor 95% confidence intervals. For success rates at the 3% to 50% level, when n≤100, the half-width of the binomial interval can reach ± 10%, making it impossible to determine whether "R2RGen is superior to 25× human" is significant. Please provide the 95% interval of the success rate of the main tasks and the significance test results compared with 25 source-human-video.
2. In the main text, only "success rate" is clearly stated, but the failure cases (collision, non-contact, target placement beyond the limit, unfeasible trajectory, etc.) are not classified. The generated data may systematically magnify certain types of errors (such as placement deviations after large rotations), but readers cannot determine whether such problems exist merely based on the main text content. Please provide the confusion matrix of failure cases and compare the failure distribution of DemoGen and R2RGen to explain whether the generation strategy reduces specific errors.
3. In the mobile operation experiment, the camera post-processing still used the same set of internal and external parameters, and no methods such as online reprojection to reduce the impact were performed, which might introduce geometric inconsistencies. If possible, could the distribution of navigation stop errors (mean ±std) be provided in the future?
4. The generated data uses a high-density completed point cloud, while the input during deployment is a 640×480 original depth downsample. Can this supplement the explanation for "inconsistent training-test resolutions" and verify the robustness of the strategy to density differences?
5. The experiments mentioned in the section "Relationship between Performance and Annotation" of the paper were conducted on four types of sources: 1, 3, 5, and 10. However, the figures shown in Figure 5 are for the four types of sources: 1, 2, 3, and 5. Please check, proofread, and make corrections.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
R2RGen offers a real enhancement Pipeline based solely on visible light video, which can reliably scale a single person's demonstration to hundreds of different point cloud trajectories. It achieves a success rate comparable to that of using 25 times the human demonstration trajectory on 8 real robot tasks with just one human demonstration trajectory +R2RGen. This work has been meticulously designed and extensively evaluated, demonstrating excellent deployment value. However, some design choices lack quantitative sensitivity analysis, and the experiments did not further explore the statistical significance of the results and failure cases. Moreover, under basis uncertainty, the movement operation expansion cannot maintain external calibration - although the movement operation is only used as a test case for limit generalization testing and is mainly left for further consideration in the future. But overall, this is a very valuable piece of work.

### Strengths
- In response to the problem that previous works could not handle the spatial changes of multi-object structures well, an original "group-level retroactive augmentation" method was proposed to maintain the constraints of multi-object structures, breaking through the single-object limitation of the Baseline method DemoGen.

- The camera perception post-processing proposed by R2RGen addresses occlusion/missing caused by large rotations, significantly reducing visual mismatch.

- The experimental scale is sufficient and the experimental process is relatively complete: approximately 1,500 real-machine rollout operations, 8 different types of grasping operation tasks, and it simultaneously includes 4 different levels of ablation experiments, with high confidence in the results.

- The entire process is based on real-world RGB video data, without the need for simulation or rendering. The mobile base is plug-and-play, and the engineering value is clear.

### Weaknesses
- Zero analysis of failure cases: Although the experimental volume of the paper is very sufficient and the improvement in the Success Rate is significant compared to the baseline method DemoGen, the analysis of the experimental results only provides the success rate and does not classify failure results such as collision, non-capture, and out-of-bounds, nor does it conduct further analysis of failure cases.

- Insufficient statistical significance of the experimental results: The main results of the paper, Table 1, lack confidence intervals or hypothesis tests. Relying solely on the current data to illustrate that "achieving the effect of using 25 times the manual demonstration trajectory" may not be the most rigorous.

- Some hyperparameter sensitive data in the experiment are missing. For instance, the patch-Z-buffer radius r and the depth threshold δ are only given as single values. These two parameters obviously affect the occlusion judgment and the final point cloud quality, but there is no sensitivity curve for the experimental hyperparameters.

- Although the movement operation is only used for extreme generalization tests, the assumption of using a fixed camera leads to loopholes in the experimental process of the movement operation: assuming that "the RGB-D camera is fixed to the base", the navigation stop error in the movement operation experiment (§ 4.5&D) is greater than 5 cm, which causes the actual external parameters of the camera to change and may introduce geometric inconsistencies.

### Questions
1. In Table 1, all success rates only provide point estimates for a single round of 32 to 64 rolluts, with neither standard deviations nor 95% confidence intervals. For success rates at the 3% to 50% level, when n≤100, the half-width of the binomial interval can reach ± 10%, making it impossible to determine whether "R2RGen is superior to 25× human" is significant. Please provide the 95% interval of the success rate of the main tasks and the significance test results compared with 25 source-human-video.

2. In the main text, only "success rate" is clearly stated, but the failure cases (collision, non-contact, target placement beyond the limit, unfeasible trajectory, etc.) are not classified. The generated data may systematically magnify certain types of errors (such as placement deviations after large rotations), but readers cannot determine whether such problems exist merely based on the main text content. Please provide the confusion matrix of failure cases and compare the failure distribution of DemoGen and R2RGen to explain whether the generation strategy reduces specific errors.

3. In the mobile operation experiment, the camera post-processing still used the same set of internal and external parameters, and no methods such as online reprojection to reduce the impact were performed, which might introduce geometric inconsistencies. If possible, could the distribution of navigation stop errors (mean ±std) be provided in the future?

4. The generated data uses a high-density completed point cloud, while the input during deployment is a 640×480 original depth downsample. Can this supplement the explanation for "inconsistent training-test resolutions" and verify the robustness of the strategy to density differences?

5. The experiments mentioned in the section "Relationship between Performance and Annotation" of the paper were conducted on four types of sources: 1, 3, 5, and 10. However, the figures shown in Figure 5 are for the four types of sources: 1, 2, 3, and 5. Please check, proofread, and make corrections.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a method for training policies from a single real-world demonstration by generating additional training data directly in 3D point cloud space. To achieve this, the RGB-D observation of the demonstration is first processed into object and background point clouds. Then, the demonstration trajectory is segmented into individual skills. For each skill, the method transforms the relevant objects and robot end-effector poses together to create new variations of the same skill. Based on the augmented trajectories, new RGB-D observations from novel viewpoints can be rendered using the processed point clouds. A visuomotor policy is then trained using these synthetic trajectories and observations so that the robot can perform the task in new positions and environments. The approach is tested on real robots for multi-step manipulation tasks, showing that it improves generalization compared to using only the original demonstration and a baseline method.

### Strengths
- The paper is well-written and easy to read.
- The authors provide extensive experiments to demonstrate the effectiveness of the proposed method.

### Weaknesses
- A major weakness of this paper is its reliance on complete object geometry, which is a very strong assumption. In Supplementary A.1, the authors explicitly state that*each object is pre-scanned using an RGB-D camera to obtain the 3D mesh, which is then used for point cloud completion during data augmentation and policy learning. However, this introduces multiple concerns: First, this limits the method to rigid objects, because complete geometry of non-rigid objects cannot be easily achieved during task execution. Although the authors mention that they can fall back to incomplete point clouds for non-rigid objects, Table 2 shows that removing point cloud completion drops the success rate significantly, worse than the baseline method. Second, If objects are scanned, why not the environment?  With a scanned environment, one could render consistent background point clouds from arbitrary viewpoints as in [1].
- The assumption of a static robot base and a static camera during execution also limits the applicability of the proposed method.
- The overall technical contribution is limited. Although generalizing trajectory augmentation to interactions with more than two objects is interesting, scene parsing, trajectory parsing and camera-aware processing are all leveraging established techniques.



[1] Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation. Sizhe Yang, et al. RSS 2025.

### Questions
I believe the paper can be improved by lifting the known object geometry constraint.

### Soundness
2

### Presentation
2

### Contribution
2
