# DexMan: Learning Bimanual Dexterous Manipulation from Human and Generated Videos

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We present DexMan, an automated framework that converts human visual demonstrations into bimanual dexterous manipulation skills for humanoid robots in simulation. Operating directly on third-person videos of humans manipulating rigid objects, DexMan eliminates the need for camera calibration, depth sensors, scanned 3D object assets, or ground-truth hand and object motion annotations. Unlike prior approaches that consider only simplified floating hands, it directly controls a humanoid robot and leverages novel contact-based rewards to improve policy learning from noisy hand–object poses estimated from in-the-wild videos.

DexMan achieves state-of-the-art performance in object pose estimation on the TACO benchmark, with absolute gains of 0.08 and 0.12 in ADD-S and VSD. Meanwhile, its reinforcement learning policy surpasses previous methods by 19% in success rate on OakInk-v2. Furthermore, DexMan can generate skills from both real and synthetic videos, without the need for manual data collection and costly motion capture, and enabling the creation of large-scale, diverse datasets for training generalist dexterous manipulation. 

Video results are available at: https://dexman2026.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
DexMan is an automated “video-to-robot” pipeline that turns third-person monocular RGB videos of humans manipulating rigid objects into bimanual dexterous skills for a full humanoid robot—no camera calibration, depth sensors, scanned 3D assets, or ground-truth hand/object motion needed.

Contributions:

RGB-only → bimanual dexterity: First end-to-end pipeline that converts uncalibrated, third-person monocular videos (real or synthetic) into bimanual dexterous skills on a full humanoid in simulation.

Contact-prior attraction reward: Object-centric correspondence between hand keypoints and mesh vertices + normal alignment to encourage meaningful, stable grasps despite noisy video poses.

Pose estimation with motion cues: Fuses a 6D pose estimator (FoundationPose-style) with 3D point trajectory tracking to stabilize object pose across occlusion/fast motion.

Stable object placement: Sampling-and-simulate step that picks the closest stable configuration from imperfect reconstructions, improving RL training stability.



Conclusion:
The authors have done commendable work on this paper. However, the proposed method is relatively simple, and the reported success rate is below what I would expect—37% in simulation is not very compelling. In addition, the sim-to-real gap remains unaddressed due to the lack of real-world experiments. I strongly recommend including real-robot evaluations and reporting the corresponding results.

### Strengths
Robustness to noisy supervision: The contact-centric reward avoids trivial touches and contact-avoidance minima, markedly boosting success in ablations.

Bimanual + dexterous + humanoid: Tackles a much tougher setting than floating hands or single-arm grippers—evidence of strong system design.

Well-engineered perception stack: Depth normalization, scale estimation, pose refinement with tracked 3D trajectories—reduces temporal jitter and failure rate.

Trainability in sim: Stable-pose sampling and residual-on-retargeting control make PPO training feasible for high-DoF, contact-rich tasks.

### Weaknesses
Sim-only: No real-robot results; sim-to-real gap (contacts, friction, latency) remains unaddressed.

Scene scope: Single-human, rigid tabletop objects; no deformable/articulated objects or mobile manipulation.

Perception–contact decoupling: Hands/objects are estimated first, contacts inferred later; inconsistencies can propagate and misguide rewards.

Action parameterization: End-effector + finger residuals underuse full-arm posture optimization; potential self-collision/occlusion issues in clutter.

### Questions
Sim-to-real: What adaptations (domain randomization, contact model calibration, residual force control, tactile feedback) do you expect are most critical to transfer these policies to a physical humanoid?

Contact-prior extraction: How sensitive is performance to the distance thresholds (τᵢ) and vertex selection heuristics? Could a learned correspondence (e.g., contrastive point-cloud features) replace nearest-vertex?

Failure modes: For the ~60–70% unsuccessful video→skill cases, what dominates—pose drift, unstable reconstructions, grasp selection, or IK infeasibility? Any quantitative breakdown?

Articulated/deformable objects: What changes would be needed—object models, contact reward definition, or control parameterization—to handle doors, drawers, cloth?

### Soundness
2

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
4

### Summary
The paper proposes a pipeline that converts a third-person human manipulation video to a robotic manipulation policy in simulation. First, many off-the-shelf vision models are used to estimate object meshes, object poses, and hand poses from the video. Then, the estimated hand-object interaction data serves as the task goal in simulation and a policy is trained via RL to control a humanoid upper-body to complete the same task. Experiments show that the proposed RL pipeline outperforms ManipTrans, and the whole pipeline can achieve successful policy learning from AI generated videos.

### Strengths
- While many prior works study learning robotic policies from mocap data, learning from pure videos is a novel challenge. This paper is a good initial exploration in this direction.
- The paper writing is easy to follow. Implementation of the pipeline is clear.

### Weaknesses
- My major concern is that, the problem studied in this paper (video-to-policy) is actually divided into two distinct research problems (1. video-to-mocap-data and 2. mocap-data-to-policy). While the paper studies both, for the first problem, the paper has not analyze the accuracy of estimated poses and object meshes and compare with any prior methods; for the second problem, the paper only compares with ManipTrans but ignores many related approaches [1,2,3,4] that learn RL policies from mocap manipulation data.
- For the first video-to-mocap task stage, the method introduces a lot of off-the-shelf models (e.g. depth estimation, semantic segmentation, hand pose estimation, and 3D reconstruction). Such a combined approach will introduce compounding errors to the system. The paper has not analyze errors introduced by each module and present accuracy and failure cases of this task stage.
- According to figures and the website, most tasks are simple pick-and-place tasks, and all tasks from AI generated videos are grasping tasks. It is questionable whether the system works for complex contact-rich manipulation tasks and the video generation model can correctly generate these videos.
- The experimental settings have not consider the sim-to-real potential. The Shadow Hands are simplified, removing the large cylinder arms. The learned behavior has many robot-table collisions according to the videos.

[1] Chen et al., Object-Centric Dexterous Manipulation from Human Motion Data, 2024
[2] Zhou et al., Learning Diverse Bimanual Dexterous Manipulation Skills from Human Demonstrations, 2024.
[3] Gao et al., CooHOI: Learning Cooperative Human-Object Interaction with Manipulated Object Dynamics, 2024
[4] Xu et al., InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions, 2025

### Questions
- How many tasks from the TACO dataset, human videos, and AI generated videos are selected for experiments? What is the selection criterion? Which module in the whole pipeline causes failure cases?
- Are object initial states and robot initial poses randomized for RL training?
- Please refer to the Weaknesses above.

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
3

### Summary
This paper presents DexMan, an automated framework that learns bimanual dexterous manipulation skills for humanoid robots directly from third-person human demonstration videos. Unlike previous approaches requiring calibrated cameras, depth data, or motion capture, DexMan operates entirely on unannotated videos and introduces contact-based reward functions to improve policy learning from noisy hand–object pose estimates. It achieves state-of-the-art results in object pose estimation on the TACO dataset and outperforms prior reinforcement learning methods on OakInk-v2 by 19% in success rate. Furthermore, DexMan can learn from both real and synthetic videos, eliminating the need for manual data collection and enabling the creation of large-scale, diverse datasets to train generalist dexterous manipulation policies.

### Strengths
1. The paper proposes a pipeline that can learn bimanual dexterous manipulation skills in simulation from a third-person monocular human video.
2. The paper is well-written, presenting a complex technical system with conceptual clarity and a logical narrative that is easy to follow.

### Weaknesses
1. The experimental results are limited to simulation and lack sim-to-real experiments.
2. The novelty of the proposed method is limited. The object and hand pose estimation parts leverage some off-the-shelf methods for hand pose estimation and object reconstruction. The policy learning part also uses some common architecture and reward designs similar to [1], [2], [3], [4], [5].

[1] Li K, Li P, Liu T, et al. Maniptrans: Efficient dexterous bimanual manipulation transfer via residual learning[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 6991-7003.

[2] Mandi Z, Hou Y, Fox D, et al. Dexmachina: Functional retargeting for bimanual dexterous manipulation[J]. arXiv preprint arXiv:2505.24853, 2025.

[3] Chen Y, Wang C, Yang Y, et al. Object-centric dexterous manipulation from human motion data[J]. arXiv preprint arXiv:2411.04005, 2024.

[4] Yuan Z, Wei T, Gu L, et al. Hermes: Human-to-robot embodied learning from multi-source motion data for mobile dexterous manipulation[J]. arXiv preprint arXiv:2508.20085, 2025.

[5] Lin, Toru, et al. "Sim-to-real reinforcement learning for vision-based dexterous manipulation on humanoids." arXiv preprint arXiv:2502.20396 (2025).

### Questions
1. In the ablation on reward components, you mention that contact rewards can be more crucial than task rewards. I'm curious whether “task reward” here refers to only object-following reward, only imitation reward, or both.
2. I am wondering whether high-quality object assets can be acquired directly from Trellis, since the extracted mesh should have similar shapes to real objects, as well as have plausible collision for contact-rich dexterous tasks. 
3. How can you align the scales of object and hand meshes with real-world settings? It seems that the pipeline does not have access to indicators of real scale, such as camera extrinsic or real point clouds observation.

### Soundness
2

### Presentation
3

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
This paper presents an approach that can help convert human videos into trajectories for robotic learning of bimanual dexterous manipulation skills. The approach reconstructs the 3D hand and object motion from video and trains an RL policy that reproduces this sequence. This functionality is demonstrated using both real and generated videos.

### Strengths
- I appreciate the effort of the paper to offer an end to end pipeline with all the pieces to learn a robotic skill from videos of human demonstrations.
- When components of the approach are evaluated independently, they achieve better performance than recent baselines (Tables 1 and 2).
- There is clear care in getting a sensible solution for 3D hand and object motion recovery and a robust approach for retargeting and RL policy training.
- The main paper and the supplementary do a good job presenting enough details for the approach and the implementation details.

### Weaknesses
- Table 1 performs an evaluation assuming that the 3D model of the object is provided. The rest of the proposed pipeline (e.g., 3D object reconstruction) is not considered in the evaluation.
- The evaluation only considers independent components of the approach. It would be helpful to factor in all steps. The pipeline is quite elaborate, so it could lead to a fragile method. This is not discussed properly.

### Questions
- I would be interested in seeing a more detailed evaluation of the pipeline and success/failure analysis of all the steps for general videos of human demonstration.
- How does the method deal with occlusions for the object when applying the object reconstruction network (Trellis)?
- Were other networks/approaches considered for the hand/object reconstruction part?

### Soundness
3

### Presentation
3

### Contribution
3
