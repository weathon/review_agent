# WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation

- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
Wrist-view observations are crucial for VLA models as they capture fine-grained hand–object interactions that directly enhance manipulation performance.
Yet large-scale datasets rarely include such recordings, resulting in a substantial gap between abundant anchor views and scarce wrist views. Existing world models cannot bridge this gap, as they require a wrist-view first frame and thus fail to generate wrist-view videos from anchor views alone.
Amid this gap, recent visual geometry models such as VGGT emerge with precisely the geometric and cross-view priors that make it possible to address such extreme viewpoint shifts.
Inspired by these insights, we propose WristWorld, the first 4D world model that generates wrist-view videos solely from anchor views.
WristWorld operates in two stages: (i) Reconstruction, which extends VGGT and incorporates our proposed Spatial Projection Consistency (SPC) Loss to estimate geometrically consistent wrist-view poses and 4D point clouds; (ii) Generation, which employs our designed video generation model to synthesize temporally coherent wrist-view videos from the reconstructed perspective.
Experiments on Droid, Calvin, and Franka Panda demonstrate state-of-the-art video generation with superior spatial consistency, while also improving VLA performance, raising the average task completion length on Calvin by 3.81% and closing 42.4% of the anchor-wrist view gap. See video results at anonymous page: https://wrist-world.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces WristWorld, a novel 4D world model that generates wrist-view videos from anchor (third-person) views alone—addressing a critical data scarcity problem in robotic manipulation where wrist-mounted camera data is rare but highly informative for vision–language–action (VLA) models. The method operates in two stages: (1) a reconstruction stage that extends the VGGT visual geometry model with a new wrist head and a Spatial Projection Consistency (SPC) loss to estimate geometrically consistent 4D point clouds and wrist-view camera poses without requiring ground-truth wrist supervision; and (2) a generation stage that uses a diffusion-based video model conditioned on the reconstructed projections and CLIP-encoded anchor-view semantics to synthesize temporally coherent and spatially accurate wrist-view videos.

### Strengths
WristWorld introduces a new solution to a real-world data bottleneck in robotics: generating high-quality wrist-view videos from only third-person (anchor) views, which are far more abundant in existing datasets. Its two-stage design—geometric reconstruction followed by diffusion-based generation—elegantly decouples pose estimation from synthesis, enabling strong spatial and temporal consistency without requiring any real wrist-view input during inference.

### Weaknesses
The core idea—leveraging geometric priors and diffusion models for cross-view video synthesis—builds heavily on existing frameworks (VGGT for geometry, DiT for generation) without introducing a fundamentally new algorithmic paradigm; the wrist head and SPC loss are incremental adaptations rather than conceptual breakthroughs. 

Lack of direct, head-to-head comparison with the most relevant prior works in the task of anchor-to-wrist view generation, such as the "Exocentric-to-egocentric video generation" method by Liu et al. (2024), which is cited but not quantitatively benchmarked against, leaving the reader uncertain about the actual margin of improvement. The paper's central claim of being the "first" 4D world model for this task is potentially overstated, as other world models like "Tesseract" (Zhen et al., 2025) also learn 4D representations for embodiment, and the specific novelty of the two-stage pipeline is not sufficiently differentiated from this broader context.

The evaluation lacks ablation on real-world generalization: all downstream VLA gains are shown only when the synthetic wrist data is used to augment training on the same domain (e.g., Franka-trained model tested on Franka), leaving open whether the method truly generalizes across robot morphologies or environments.

### Questions
1.Novelty & Baselines: How does your method compare quantitatively against an adapted baseline that uses your reconstructed first wrist frame to initialize a sequence-based model like Liu et al. (2024)? This would more directly validate the need for your full geometry-based pipeline.


2.Single Anchor View: How does performance change if only one anchor view is available (common in real datasets)? Table 5 suggests minimal drop—why?


3.Significance of VLA Gains: Are the reported VLA performance improvements (e.g., +3.81% task length) statistically significant? What is the absolute performance of the true anchor+wrist upper bound? Please clarify the calculation for "closing 42.4% of the gap" to strengthen this key claim. 

4.Real Policy Execution: Were VLA policies using synthetic wrist views deployed on real hardware, or only evaluated on logged data?

5.SPC Loss Supervision: The SPC loss requires 2D correspondences to the wrist view. How are these ground-truth wrist pixels obtained during training, considering the problem assumes a lack of wrist-view data? Is the SPC loss essential, or could simpler reprojection losses (e.g., using VGGT’s own depth) suffice? An ablation is missing.

6.Cross-Morphology Generalization: Does WristWorld generalize to robots with different arms/cameras, or only Franka-like setups?

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
3

### Summary
This work proposes a new wrist video generation model to alleviate the scarcity of wrist video data. It achieves this by proposing a reconstruction-generation pipeline. The reconstruction part aims to reconstruct a 4D point cloud from anchor videos and wrist view camera extrinsic parameters. The imperfect wrist view video can be rendered from these extrinsic parameters. The generation part focuses on generating a perfect wrist view video from these cues.

### Strengths
The targeted problem is meaningful, and the solution is sound. This work proposes to upgrade VGGT with an extra head to predict wrist view camera extrinsics. To do so, this work introduces a correspondence supervision loss, referred to as spatial projection consistency loss. Afterward, this work injects the rendered imperfect wrist view video and anchor video into a video diffusion model to generate a perfect wrist video.

### Weaknesses
The presentation can be improved. I notice that the generated wrist view videos are repeated multiple times in your provided website (intro video and visualization part) as well as in the main manuscript. It is beneficial to showcase qualitative results as extensively as possible. Moreover, it is better to display the anchor video alongside the generated video. Furthermore,  the explanation is unclear: 
1. The used training data is unclear. What did you use for the first stage of reconstruction and generation training? And for the second stage? Why was only 10k Droid data used? What about more Droid data?
2. L150: S is missing an explanation.
3. The subscript of y in lines 213 and 214 is different.
4. The depth term of SPC is not very clear. Why is only the back term supervised? What is the rationale behind this loss design?
5. How do you obtain correspondence points? What algorithm do you use?
6. What does "the i-th external view" mean in L261?
7. There is no explanation about "cross-view fine-tuning." What does it mean? The training strategy seems novel; why don't you provide more details?
8. What video generation model do you use? Is it trained from scratch in a newly designed network?
9. L426:  "persis"?

### Questions
1. The used training data is unclear. What did you use for the first stage of reconstruction and generation training? And for the second stage? Why was only 10k Droid data used? What about more Droid data?
2. L150: S is missing an explanation.
3. The subscript of y in lines 213 and 214 is different.
4. The depth term of SPC is not very clear. Why is only the back term supervised? What is the rationale behind this loss design?
5. How do you obtain correspondence points? What algorithm do you use?
6. What does "the i-th external view" mean in L261?
7. There is no explanation about "cross-view fine-tuning." What does it mean? The training strategy seems novel; why don't you provide more details?
8. What video generation model do you use? Is it trained from scratch in a newly designed network?

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
The paper introduces WristWorld, a two-stage framework designed to synthesize wrist-view videos using only third-person (anchor) views, without requiring a wrist first-frame. Stage 1 (Reconstruction) extends VGGT with a wrist head that predicts wrist camera extrinsics and introduces a Spatial Projection Consistency (SPC) loss to enforce alignment between RGB correspondences and 3D/4D geometry. Stage 2 (Generation) employs a diffusion transformer conditioned on (i) wrist-view projections and (ii) CLIP features derived from third-person views and text instructions, enabling the generation of temporally coherent wrist-view videos. Evaluations on the DROID and Franka datasets demonstrate that WristWorld outperforms baselines across four video quality metrics and improves downstream Visual Language Action (VLA) performance on CALVIN.

### Strengths
- The authors propose a simple yet effective framework for synthesizing wrist-camera views from third-person perspectives.
- The generated videos can directly benefit downstream models such as Visual Language Action (VLA) models.
- The paper presents extensive experimental benchmarks conducted in both real-world and simulated environments.

### Weaknesses
- When examining the 3D baseline without the SPC loss, there is a noticeable difference between the ground truth and the estimated point cloud. Even after applying the SPC loss, a small discrepancy remains. Is this error primarily due to inaccuracies in the estimated point cloud or in the predicted wrist pose? It would be helpful to report the wrist pose error and provide a qualitative comparison using the ground-truth wrist pose, showing both the estimated and ground-truth point clouds.
- How sensitive are the results to the number of cameras used? Can the proposed method achieve comparable performance when the number of cameras is reduced to one or two?

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a framework to generate wrist-view robot manipulation videos, given the input of anchor-view robot videos. To realize this, the anchor-view videos are reconstructed by VGGT and projected to wrist view at the first stage. Next, a video generation model is utilized to fully synthesize wrist-view videos. The evaluation is conducted on several datasets.

### Strengths
1. The paper is well-organized. 
2. The focus on wrist-view manipulation is meaningful, and the evaluation shows that wrist-view data is important for robot tasks.

### Weaknesses
1. The novelty and the contribution are **very limited**, which is not enough for the ICLR bar:  
(1) The proposed method is substantially a *novel-view video* synthesized pipeline; such a "reconstruction-render/project-videogen" idea has been fully investigated in many existing 3D works, such as [1, 2, 3]. This work just utilizes a similar idea to the robot domain, where no specific challenges are discussed or solved.          
(2) The statement of "world model" is *significantly over-claimed*. World-model denotes the ability to predict the state/action of the future, given only the *previous* observation, instead of the complete observation of finishing a task. Further, "4D" World Model is *more over-claimed*, as *NO* 4D prediction is introduced, but only the 2D novel-view video is synthesized.     
(3) The pipeline needs the complete manipulation videos of anchor-view. However, such input conditions are not easy to gain for a robot manipulation task. A natural question is: When the complete manipulation video, although from a different view, is given, **the whole information about the "task completion" (which is more important for the robot) *has been provided***, why still need the wrist-view to complete the task? 
2. No limitation or failure case is provided. 

*Refs*:    
[1] GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control. CVPR 2025.    
[2]  You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale. CVPR 2025.  
[3] SpatialCrafter: Unleashing the Imagination of Video Diffusion Models for Scene Reconstruction from Limited Observations. CVPR 2025.

### Questions
The work may be a good project, but definitely not a very insightful research paper, at least for ICLR or top-tier venues.

### Soundness
3

### Presentation
2

### Contribution
1
