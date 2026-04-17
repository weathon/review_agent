# To View Transform or Not to View Transform: NeRF-based Pre-training Perspective

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Neural radiance fields (NeRFs) have emerged as a prominent pre-training paradigm for vision-centric autonomous driving, which enhances 3D geometry and appearance understanding in a fully self-supervised manner. To apply NeRF-based pre-training to 3D perception models, recent approaches have simply applied NeRFs to volumetric features obtained from view transformation. However, coupling NeRFs with view transformation inherits conflicting priors; view transformation imposes discrete and rigid representations, whereas radiance fields assume continuous and adaptive functions. When these opposing assumptions are forced into a single pipeline, the misalignment surfaces as blurry and ambiguous 3D representations that ultimately limit 3D scene understanding. Moreover, the NeRF network for pre-training is discarded during downstream tasks, resulting in inefficient utilization of enhanced 3D representations through NeRF. In this paper, we propose a novel NeRF-Resembled Point-based 3D detector that can learn continuous 3D representation and thus avoid the misaligned priors from view transformation. NeRP3D preserves the pre-trained NeRF network regardless of the tasks, inheriting the principle of continuous 3D representation learning and leading to greater potentials for both scene reconstruction and detection tasks. Experiments on nuScenes dataset demonstrate that our proposed approach significantly improves previous state-of-the-art methods, outperforming not only pretext scene reconstruction tasks but also downstream detection tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes NeRP3D, a new framework that rethinks how NeRF-based pre-training is used for 3D perception. It argues that existing methods with view transformations (e.g., BEV or voxel space), which introduces a mismatch between NeRF’s continuous radiance fields and discrete voxel priors. NeRP3D removes this step entirely and directly models continuous 3D points, fusing multi-view image features through deformable attention. The same network is used for both pre-training (via RGB and signed distance function) and downstream tasks like 3D detection and occupancy prediction. Experiments on nuScenes show clear improvements across all benchmarks and much sharper scene reconstructions.

### Strengths
1. The paper identifies a conceptual flaw in current NeRF-based pre-training methods and proposes a simple but elegant fix. Removing view transformation aligns the training process with the continuous nature of NeRFs.
2. The unified architecture is clean and consistent, avoiding the typical “throwaway pre-training” stage found in earlier works.
3. Experimental results are strong and consistent. The performance gains are not marginal, especially in image reconstruction and occupancy prediction.

### Weaknesses
1. The paper is conceptually motivated by the mismatch between discrete view-transformed features and continuous NeRF representations. However, it does not provide direct quantitative evidence demonstrating how this conflict affects model performance or how NeRP3D resolves it. Including controlled experiments that isolate this factor, such as measuring representational smoothness or reconstruction error across discretization levels—would strengthen the argument and make the theoretical claim more concrete.
2. The paper lacks an evaluation of runtime, computational cost, or scalability. Because NeRP3D relies on dense point sampling and deformable cross-attention, it may have higher memory and compute overhead compared to voxel-based methods. Reporting training and inference times, FLOPs, and GPU memory usage, along with a discussion of potential optimization strategies (e.g., sampling reduction or pruning), would help clarify whether the approach is practical for real-time or large-scale autonomous driving applications.
3. All experiments are conducted solely on the nuScenes dataset. While the results there are strong, it remains unclear how well NeRP3D generalizes to different environments, sensor setups, or data distributions. Evaluating the model on additional benchmarks such as Waymo Open Dataset or at least providing cross-dataset transfer results, would greatly enhance the paper’s credibility and demonstrate the robustness of the proposed representation across domains.

### Questions
1. How does NeRP3D compare in terms of training time and inference speed to existing voxel-based models?
2. How sensitive is the model to the number and distribution of sampled points during training?
3. Is there any evidence that the pre-trained NeRF-like encoder retains its radiance modeling ability after fine-tuning for detection or mapping?
4. Could this approach be extended to incorporate LiDAR or radar inputs directly, given that it operates in continuous 3D space?

### Soundness
2

### Presentation
2

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
This paper proposes NeRP3D, a NeRF-Resembled Point-based 3D detector for autonomous driving that avoids view transformation by directly modeling continuous 3D representations. The authors argue that existing NeRF-based pre-training methods (UniPAD, SelfOcc) suffer from conflicting priors: view transformation imposes discrete/rigid representations while NeRFs assume continuous functions. NeRP3D preserves the pre-trained NeRF network during downstream tasks and supports adaptive sampling strategies (ray-wise for rendering, spatial for detection). Experiments on nuScenes show modest gains ($1-2\%$) over UniPAD in 3D detection, occupancy, and HD map tasks.

### Strengths
1. Strong Problem Diagnosis. The paper clearly identifies the fundamental tension between discrete View Transformation and continuous NeRF priors. The critique of this "conflicting prior" (Fig 1) is insightful and provides a solid motivation.

2. Architectural Coherence. The unified framework for pre-training (ray-wise sampling) and downstream tasks (spatial sampling) is elegant. It effectively preserves the pre-trained NeRF representation without an intermediate discrete stage.

3. Comprehensive experimental validation. The paper evaluates across multiple pretext tasks (RGB/depth reconstruction) and three downstream tasks

### Weaknesses
1. Limited Methodological Novelty. The core idea of querying 2D features at 3D points to maintain a continuous representation is not new (e.g., pixelNeRF [1], MVSNeRF [2]). The contribution is an architectural choice (avoiding an intermediate voxel grid) rather than a fundamental new modeling approach.

2. Insufficient Ablations. Key design choices are unjustified. (a) Why deformable cross-attention (Eq 2) over standard attention? (b) The point sampling density trade-off (accuracy vs. cost) is not analyzed.

3. Incomplete Cost Analysis. The paper acknowledges high costs (line 484) but fails to quantify them. Key metrics are missing: (a) Inference latency (ms/frame) and (b) Scalability analysis (i.e., the accuracy/cost trade-off vs. the number of sampled points).

4. Missing Comparisons to Key Baselines. The paper only compares against NeRF-pretrain (UniPAD) and VT-based (BEVFormer) methods. It critically omits comparisons to point-based detectors (e.g., DETR3D[3], PETR[4]). These methods share NeRP3D's core philosophy of querying continuous 3D locations. Without this comparison, it is impossible to know if the gains come from NeRF pre-training or simply from the point-based representation itself.

5. Single-Dataset Evaluation. All experiments are confined to nuScenes. Without cross-dataset validation (e.g., Waymo, KITTI, Argoverse), the method's robustness to different sensors, environments, and camera layouts is entirely unverified.

[1] Yu, Alex, et al. "pixelnerf: Neural radiance fields from one or few images." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[2] Chen, Anpei, et al. "Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[3] Wang, Yue, et al. "Detr3d: 3d object detection from multi-view images via 3d-to-2d queries." Conference on robot learning. PMLR, 2022.

[4] Liu, Yingfei, et al. "Petr: Position embedding transformation for multi-view 3d object detection." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.

### Questions
Please see the critical issues and required clarifications detailed in the Weaknesses section.

### Soundness
3

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
This paper critiques existing NeRF-based pre-training methods in autonomous driving due to their reliance on rigid view transformations, creating conflicts with NeRF's continuous representation capabilities. The authors then introduce NeRP3D, a point-based 3D detector that retains continuous NeRF representation from pre-training through downstream tasks. Experimental results on the nuScenes dataset demonstrate considerable improvements in both reconstruction quality and detection performance compared to prior methods.

### Strengths
1. The paper's primary strength is its core insight. Identifying the "misaligned prior" between discrete view-transformation and continuous radiance fields is a valuable critique of existing methods. This moves beyond simply "adding NeRF" and asks a more fundamental question about how it should be integrated.

2. The proposed NeRP3D architecture is an elegant solution to the problem it identifies. The unification of the pre-training and fine-tuning architectures into a single, continuous, point-queryable function is conceptually clean. This design, which preserves the pre-trained network rather than discarding it, is a good contribution to the field of 3D pre-training.

3. The experiments are solid and provide comprehensive evidence for the paper's claims.

### Weaknesses
Overall I think this is a good work. Only some minor weaknesses remain.

1. Computational Cost: The paper admits in the conclusion that the point-based architecture "incurs high computational costs". However, the paper provides no quantitative analysis (e.g., FLOPS, latency, memory) to compare NeRP3D against the view-transform-based baselines. This is a critical omission for a paper aimed at practical autonomous driving applications. 

2. Generalization: While results on the nuScenes dataset are promising, evaluation on additional datasets could help establish broader applicability and robustness of the method.

3. Gains vs. the closest continuous baseline are small. Against GaussianPretrain, the margin on detection is +0.1 NDS / +1.1 mAP, which is within typical variance unless confidence intervals are shown.

### Questions
1. Compute trade-offs. Please provide FPS, peak memory, and point count vs. accuracy curves for NeRP3D and UniPAD/TPV across identical image resolutions and heads. This will show whether the continuous design is cost-effective at scale. 

2. Influence of SDF Prior: The pre-training uses NeuS (SDF), while competitors use standard NeRF (density). How much of the performance gain, especially in geometric fidelity, is attributable to the SDF prior versus the continuous architecture? What happens if NeRP3D is pre-trained with a standard density-based NeRF loss?

### Soundness
3

### Presentation
3

### Contribution
3
