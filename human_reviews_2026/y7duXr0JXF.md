# Human3R: Everyone Everywhere All at Once

- Decision: Accept (Poster)
- Scores: 4, 8, 8

## Abstract
We present Human3R, a unified, feed-forward framework for online 4D human-scene reconstruction, in the world frame, from casually captured monocular videos. Unlike previous approaches that rely on multi-stage pipelines, iterative contact-aware refinement between humans and scenes, and heavy dependencies, e.g., human detection, depth estimation, and SLAM pre-processing, Human3R jointly recovers global multi-person SMPL-X bodies (“everyone”), dense 3D scene (“everywhere”), and camera trajectories in a single forward pass (“all-at-once”). Our method builds upon the 4D online reconstruction model CUT3R, and uses parameter-efficient visual prompt tuning, to strive to preserve CUT3R’s rich spatiotemporal priors, while enabling direct readout of multiple SMPL-X bodies. Human3R is a unified model that eliminates heavy dependencies and iterative refinement. After being trained on the relatively small-scale synthetic dataset BEDLAM for just one day on one GPU, it achieves superior performance with remarkable efficiency: it reconstructs multiple humans in a one-shot manner, along with 3D scenes, in one stage, in real-time (15 FPS) with a low memory footprint (8 GB). Extensive experiments demonstrate that Human3R delivers state-of-the-art or competitive performance across tasks, including global human motion estimation, local human mesh recovery, video depth estimation, and camera pose estimation, with a single unified model. We hope that Human3R will serve as a simple yet strong baseline, which can be easily adapted for downstream applications. Code, models and 4D interactive demos are available at https://fanegg.github.io/Human3R/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the task of reconstructing 3D human motion, camera motion, and scene from monocular videos. Unlike previous multi-stage approaches, it proposes a unified single-shot framework that jointly reasons about humans, cameras, and the surrounding scene. The framework is built upon the 4D reconstruction model CUT3R, which recurrently estimates metric-scale depth and camera poses in an online fashion. To enhance 3D human motion understanding, the method integrates human pose tokens extracted by Multi-HMR with image features from CUT3R’s backbone. During training, the CUT3R weights are frozen, while only the layers responsible for fusing the Multi-HMR tokens and regressing SMPL-X parameters are fine-tuned on the BEDLAM dataset. The proposed model is evaluated on 3DPW and EMDB (subset1) in camera coordinates, and on EMDB (subset2) and RICH in global coordinates. Experimental results show that the model achieves performance comparable to existing HMR methods in camera coordinates and, leveraging CUT3R’s robust global reasoning, delivers promising results in global coordinates.

### Strengths
1. The paper proposes to addresses the challenge of fusing CUT3R features with Multi-HMR features during both training and inference by introducing pixel-aligned feature extraction and Test-Time Sequence Length Adaptation. This enables a unified multi-task learning framework under limited training data, advancing the integration from feature-level fusion beyond the previous result-level fusion used in methods like GVHMR.

2. The experiments reveal several valuable insights. As shown in Table 1, directly integrating CUT3R’s 4D reconstruction features with the human motion features extracted by Multi-HMR via pixel-aligned extraction improves HMR performance in the camera coordinate system. The ablation studies in Section 4.4 further analyze the contribution of each feature type, which is helpful for understanding their respective effects.

3. The visualization results presented are promising and engaging, likely to attract readers’ interest and confidence in the method’s potential.

### Weaknesses
1. The proposed framework estimates global camera poses using CUT3R and fuses them with human features extracted by Multi-HMR to recover human motion. However, the overall computational complexity of the open-source implementation is quite similar to that of GVHMR, which also integrates visual odometry and 2D pose estimation (using different backbones corresponding to the variants described in this paper). Both methods share similar design philosophies and architectural concepts. As shown in Table 2, their performance metrics are also close, while Table 1 reveals a noticeable performance gap. One possible reason for this discrepancy could be the limited number of trainable layers — the proposed approach does not jointly train the Decoder component illustrated in Figure 4, which may restrict performance. Therefore, GVHMR serves as a strong and conceptually comparable competitor, and such a comparison could better reveal the advantages and limitations of the paper’s “low-training” design philosophy.

2. The presentation of the Human Prior module in Figure 4 may cause confusion among readers. The human-prior token used in this work is derived from the pixel-aligned feature vectors between CUT3R’s image features and the image features extracted by Multi-HMR’s backbone. However, Figure 4 omits a clear description of how these tokens are obtained, which may lead readers to mistakenly assume that a single backbone is used for feature extraction. However, both CUT3R and Multi-HMR backbones independently extract their own image features before being fused at corresponding pixel positions.

3. Some of the claimed contributions lack sufficient empirical evidence, particularly the statement that the proposed method “operates on streaming video at real-time speed (15 FPS) without compromising accuracy.” I tested the released implementation on an RTX 4090 GPU and found that the inference speed — when extracting CUT3R and Multi-HMR features online — falls far short of real-time, even significantly below the 5 FPS reported in Table 4 for the largest backbone. It remains unclear how the ViT-B version used in the real-time mode performs and whether accuracy degradation occurs. In addition, the latency details should be explicitly discussed.

### Questions
1. It would be better to provide a more in-depth analysis of Table 1, clarifying whether the observed improvement in the camera coordinate system (compared to Multi-HMR alone) primarily comes from depth accuracy or from other specific spatial dimensions.
2. It would be more appropriate to revise the contribution statements, especially regarding real-time performance claims, to report and justify the corresponding runtime metrics and latency with clear condition.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Human3R, a unified and feed-forward framework for online 4D human–scene reconstruction from monocular videos. In contrast to traditional multi-stage approaches, Human3R performs joint reconstruction of multiple SMPL-X bodies, dense 3D scenes, and camera trajectories within a single forward pass. Built upon pretrained CUT3R and enhanced through fine-tuning, the model achieves real-time performance while maintaining strong accuracy and efficiency across diverse reconstruction tasks.

### Strengths
1. Soundness: The paper presents a simple yet powerful idea, effectively leveraging the rich priors of CUT3R to achieve comprehensive human–scene reconstruction.

2. Ablation studies: The ablation studies are insightful and thorough, particularly those in Table 3, which clearly demonstrate how refining the pretrained priors leads to consistently improved results. The experiments are well-designed and highlight the robustness of the proposed framework.

### Weaknesses
1. Computation Comparison: While the model claims real-time, all-in-one performance, a computation and runtime comparison with other methods would strengthen the argument for its efficiency.

2. Cross Attention: Although the method jointly models human and scene components, it remains unclear how much each modality benefits the other. An ablation removing the cross-attention between humans and scenes could verify this interdependence. Furthermore, visualizing cross-attention activation maps could help confirm whether the network indeed captures correlated human–scene features during reconstruction.

### Questions
1. The ViT encoder from CUT3R may downscale spatial resolution, which might influence human detection and segmentation accuracy. Have the authors analyzed how input or feature resolution affects performance? It would be valuable to evaluate this by resizing the encoded features with various (i, j) scaling factors.

2. In scenarios where multiple humans are heavily occluded and occupy the same head token, how does the model differentiate and handle such cases? A clarification or visualization would be helpful to understand its robustness under severe occlusion.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
- Humans3R predicts humans, scene, and camera jointly with a single model in a single step.
- It freezes CUT3R (a state-based recurrent 3D scene reconstruction model) and augments it with visual prompt tuning (learned tokens) to predict human mesh parameters. Mesh parameters are predicted only from tokens classified as human; this classifier is learned end-to-end.
- The main benefit is simplicity: the work unifies previously separate techniques for scene and human reconstruction.
- To handle longer time horizons, Humans3R adopts TTT3R for test-time sequence length adaptation.
- The main evaluations are done on 3DPW, EMDB1/2 and RICH datasets.
- Main baselines: WHAM, JOSH3R, Multi-HMR.
- Results: Humans3R outperforms corresponding one-stage online methods on both local and global human mesh reconstruction.

### Strengths
- The paper is well written, well organized, and easy to follow; key ideas are presented clearly with intuitive explanations, and the technical details are clear and consistent.

- Humans3R is novel: the first to unify human and scene reconstruction in a single online model with promising results. I am pleasantly surprised to find out that CUT3R-like models can be repurposed (using human priors from the Multi-HMR image encoder) to perform joint scene and human mesh reconstruction. No depth estimation or SLAM needed. This is a strong technical contribution and a promising direction for the community, the resulting simplification is notable and welcome.

- Empirical results are strong and thorough: quantitative evaluations span multiple datasets (3DPW, EMDB, RICH) and method classes (multi-stage, online, offline). Locally (3DPW/EMDB-1), Human3R beats one-stage baselines (e.g., MPJPE 71.2 on 3DPW vs. BEV 78.5); globally (EMDB-2/RICH), it notably improves over online WHAM (e.g., EMDB-2 WA-MPJPE 112.2 vs. 135.6; RTE 2.2% vs. 6.0%).

- Intrinsic robustness via metric-scale scene context. Leveraging CUT3R’s frozen metric-scale scene prior, the system remains stable without calibrated intrinsics and across aspect ratios. It would be valuable to test generalization beyond the reported datasets and qualitative examples; if confirmed, this would address a common failure mode in HMR pipelines.

### Weaknesses
- Dependence on the head-token: The authors already note this, but this is a substantial assumption and, in my view, the primary weakness. Humans3R’s forward pass relies on reliable head-token detection; occlusion or truncation can cause misses. The quantitative evaluations do not reflect this limitation because the datasets lack such scenarios, whereas multi-stage baselines perform reasonably under truncation due to strong train-time cropping augmentations.

- Inference on non-full body + multi-person sequences: The reported results focus primarily on full-body sequences. It would be insightful to understand the performance degradation on challenging quarter/upper-body or face-focused images, where the head is still visible but the sequence is out-of-distribution relative to BEDLAM’s training data. How accurate would the scene reconstruction be in these cases? Also specify crowding limits: what is the maximum number of concurrent identities Humans3R handles in practice (for eg, SLAHMR, CVPR 2023 manages about 10 identities offline)?

- Pixel aligned predictions: The qualitative results (and supplementary videos) are primarily 3D visualizations. They are metrically consistent and impressive. Over longer horizons, predicted camera pose may drift, leading to misalignment. It would be helpful to demonstrate mesh–camera consistency by projecting the mesh into the image and overlaying it, and compare against baselines.

### Questions
Overall, I like this work and rating the work as 8.

My questions are primarily on the mentioned weaknesses.
- How would you remove head-token dependence in the architecture? Please expand on inclusion of body point localizers.
- How well does Humans3R generalize to non-fully body + multi-person scenarios?
- Mesh projection visualization.

### Soundness
3

### Presentation
3

### Contribution
3
