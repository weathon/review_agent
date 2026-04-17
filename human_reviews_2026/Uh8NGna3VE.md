# Background Matters: Robust 3D Human Pose Estimation via Controllable Video Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
n/a

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
While the paper is well written and experimentally detailed, it lacks sufficient novelty and fails to demonstrate competitive performance compared to existing methods. The absence of comparisons with strong baselines further weakens the technical contribution.

### Strengths
1) The paper is clearly written and the proposed idea is clearly presented. Overall the paper is easy to follow.
2) It includes a comprehensive set of experiments.

### Weaknesses
1) The idea of using pose guidance for video generation to improve generalizability has been explored in prior work (e.g., PoseSyn [1]). The novelty of this paper is therefore limited.

2) The reported results fall significantly behind SOTA performance. For instance, on the 3DHP dataset, PersPose [2] (ICCV 2025) achieves less than 75 MPJPE, while this paper reports 124 MPJPE.

3) No quantitative comparison with relevant prior work, both in:

   - Dataset generation methods: PoseExaminer [3], PoseGen [4], PoseSyn, IDOL [5], AdaptPose [6];

   - SOTA 3D pose estimation models: PersPose, PostoMETRO [7]


[1] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data
[2] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation
[3] PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation
[4] PoseGen: Learning to Generate 3D Human Pose Dataset with NeRF
[5] IDOL: Instant Photorealistic 3D Human Creation from a Single Image
[6] AdaptPose: Cross-Dataset Adaptation for 3D Human Pose Estimation by Learnable Motion Generation
[7] PostoMETRO: Pose Token Enhanced Mesh Transformer for Robust 3D Human Mesh Recovery

### Questions
The authors might want to clarify the contributions; compare with related SOTAs; and add more discussions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper utilizes controllable video generation for learning to estimate 3D human poses robustly. RGB Video generation is supposed to help in generating diverse 2D human motion sequence through varying poses, scenes, and viewpoints. Besides using real 2D inputs, the idea behind using real-world detections is to design a robust and generalizable pose estimation model. The authors conduct experimental evaluation on various 3DHPE datasets to show the effectiveness of the proposed approach on real world and corrupt 2D inputs.

### Strengths
1. The paper reframes the usual pose-only augmentation approach into a multi-modal data generation pipeline that explicitly models scene diversity in terms of pose, viewpoint, lighting, etc. which helps in building a generalizable 3D human pose estimation model.
2. The idea of leveraging pose-guided video diffusion models is intuitive and a straightforward implementation should be easily achievable.
3. Experiments include multiple datasets (H36M, PMR, 3DHP, 3DPW) and diverse metrics (MPJPE, P-MPJPE, velocity error). Moreover, the paper evaluates robustness under real-world corruptions (blur, compression, spatter), which strengthens the practical motivation.
4. The effects of filtering ratios, pretraining strategies, and GT vs. detected 2D inputs are well-studied. Also, the results seem to demonstrate consistent improvements across nearly all configurations which suggests robustness of the method.

### Weaknesses
1. The technical contribution mainly lies in data generation and composition rather than in a novel model or algorithm which I feel is a major discussion point. The method builds on existing diffusion video generation models with minimal architectural innovation.
2. The paper relies heavily on pretrained models like Animate Anyone (Hu et al.) and Latent Diffusion model (Rombach et al.) without domain-specific adaptation. It is unclear how much of the improvement comes from the inherent realism of these models rather than the proposed pipeline design.
3. The method is validated primarily on H36M, PMR, and 3DHP which are all captured in a controlled environment. A demonstration on truly unseen in-the-wild datasets (e.g., MPII, COCO-Video, AMASS-based scenes) is missing and would support generalization claims.
4. One of the concerns is that generating and filtering large-scale video data using diffusion models is resource-intensive. The paper lacks any discussion regarding this, e.g., training time, compute requirements, or efficiency trade-offs compared to simpler augmenters like PoseAug (Zhang et al.). No user or perceptual evaluation is provided for the realism or physical plausibility of generated videos.

### Questions
1. What is the precise novelty beyond combining existing diffusion video generation models with pose augmentation?
2. What is the core mechanism by which background variation improves generalization? The hypothesis is intuitive (“background matters”), but the paper lacks an analysis or visualization showing how added background diversity affects learned representations.
3. What is the computational overhead of generating and filtering data? Since video diffusion generation is extremely costly, the scalability of this method to larger datasets or real-time adaptation is unclear. Additionally, for a generalizable model, training a video generation model on extensively diverse datasets seems to be crucial but at the same time unscalable.

### Soundness
3

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
The paper proposes using diffusion models to generate and augment the in-studio dataset to improve the generalization of human pose estimators. Specifically, they argue that 2D-to-3D pose lifting techniques are often trained on in-studio, near-perfect data, whereas in the real world such data are scarce, and artifacts from noise and occlusion can undermine the robustness of pose estimation systems. To address this challenge, the paper introduces a two-stage augmentation pipeline that first trains a controllable video generator (Animate Anyone) on a dataset and then feeds it poses from diverse domains to generate new postures with different backgrounds. The paper supports its claims through extensive experiments, showing that by augmenting and training the models on corrupted/synthesized datasets, results in improved performance.

### Strengths
- The paper is well-written and easy to follow. It explains the technical details and provides adequate clarification.
- The cross-dataset analysis clearly shows that by augmenting RGB videos rather than 2D poses, the performance of cross-dataset generalization can increase

### Weaknesses
- Alternatives to video generation are not compared against. For instance, a simple background-pasting algorithm can serve as a straightforward baseline.
- The computational cost and practicality of the approach are questionable. Training a large video generation model can be much more computationally expensive than rendering a synthetic dataset. Additionally, the paper mentions that 90% of the data is discarded, meaning that the process is highly inefficient and uncontrollable. This limitation and the lack of controllability are not fully addressed in the paper.
- No other baselines are compared against. For instance, while PoseAug tries to augment the 2D poses, it can be a point of comparison. The paper cites these methods, but does not include them in the comparisons.

### Questions
1. Could you please provide a detailed breakdown of the computational cost for 1) training/fine-tuning the video generator, 2) generating the dataset, and 3) training the HPE model? Please provide a comparison with other available approaches cited in the paper on line 58.
2. Please address my points in the above section, and the point about comparing with PoseAug specifically.

### Soundness
3

### Presentation
3

### Contribution
1
