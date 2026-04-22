# S2GO: Streaming Sparse Gaussian Occupancy

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Despite the efficiency and performance of sparse query-based representations for detection, state-of-the-art 3D occupancy estimation methods still rely on voxel-based or dense Gaussian-based 3D representations. However, dense representations are slow, and they lack flexibility in capturing the temporal dynamics of driving scenes. Distinct from prior work, we instead summarize the scene into a compact set of 3D queries which are propagated through time in an online, streaming fashion. These queries are then decoded into semantic Gaussians at each timestep. We couple our framework with a denoising rendering objective to guide the queries and their constituent Gaussians in effectively capturing scene geometry. Due to its efficient, query-based representation, S2GO achieves state-of-the-art performance on the nuScenes and KITTI occupancy benchmarks, outperforming prior art (e.g., GaussianWorld) by 2.7 IoU with 4.5x faster inference.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose S2GO, a streaming sparse query framework for 3D semantic occupancy estimation. This method efficiently summarizes and propagates scene information through sparse 3D queries and decodes it into a semantic Gaussian distribution. Core contributions include a geometric denoising pre-training stage and an efficient Gaussian-voxel transformation implementation. The paper achieves state-of-the-art performance and significant inference speed advantages on nuScenes and KITTI benchmarks. The method is novel, the experimental design is comprehensive, and the results are convincing.

### Strengths
1. A streaming occupancy prediction with Gaussians is an insightful, useful and unexplored problem.

2. To address the above problem, the authors propose a simple but in-depth architecture S2GO. Their solution is elegant. 

3. Extensive experiments on the multiple datasets show the effectiveness of the proposed method.

### Weaknesses
1. It is recommended to enhance the display of module details in Figure 1.

2. The paper focuses on engineering and experimentation in its methodological description, and lacks theoretical analysis or discussion on the representation capabilities of sparse queries or the convergence of the model.

3. The efficient Gaussian-to-voxel splatting proposed in the paper significantly improves the forward/backward pass speed, but lacks comparative experiments on memory usage. It is suggested that the authors supplement this with a comparison of memory consumption.

4. In Figure 3, the small subfigures partially obscure the main visualization. I suggest adjusting the layout so that the smaller insets do not overlap with the main figure, ensuring all visual details are clearly visible.

5. The paper would benefit from a clearer description of the experimental parameter settings (e.g., learning rate, batch size, number of epochs, optimizer, etc.). Providing these details would enhance reproducibility and allow for a more precise evaluation of the experimental results.

### Questions
It is essential to address the above comments in weakness.

### Soundness
3

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
5

### Summary
The paper proposes S2GO, a streaming, query-based 3D semantic occupancy method. Instead of dense voxels or many Gaussians, S2GO maintains a small set of persistent 3D queries that are propagated through time and decoded into semantic Gaussians each step. A two-stage training is used: (1) geometry denoising & rendering pretraining with noised LiDAR anchors and RGB/depth supervision; (2) camera-only semantic occupancy with improved Gaussian-to-voxel splatting (opacity-weighted occupancy and custom CUDA kernels). On nuScenes SurroundOcc and KITTI-360 SSCBench, S2GO reports SOTA IoU/mIoU with much higher FPS vs prior art (e.g., GaussianWorld).

### Strengths
Pros:
1. Clear motivation for sparsity + streaming. The paper articulates the inefficiency of dense grids and dense Gaussian fields for long-horizon temporal fusion and proposes queries (~K) that each spawn a handful (J) of Gaussians to cover local structure—bridging object-centric queries and dense occupancy.
2. Well-designed pretraining. Initializing queries at noised LiDAR FPS points and supervising with denoising + RGB/depth rendering directly teaches queries to “move onto” geometry and Gaussians to model fine shape. Ablations show large gains vs training occupancy from scratch.
3. Principled fix to splatting. Making binary occupancy opacity-weighted (α(x;G) := a · exp(…)) aligns semantics with density in rendering and improves stability/performance; optimized CUDA “blocked” splatting cuts backward time ~20× on A100 (116ms→5.7ms).

### Weaknesses
Cons:
1. Pretraining dependence & potential dataset bias. Stage-1 uses LiDAR-based supervision (depth and LiDAR-anchored queries). Although authors argue LiDAR already exists for label generation, the camera-only promise hinges on quality of this pretraining. The Metric3D zero-shot depth alternative is encouraging but still trails LiDAR-based pretrain. Quantify generalization gap more broadly. Futhermore, the model achieves SOTA performance only with LiDAR-based pre-training ("LiDAR+ϵ" in Table 3). Without this step, performance degrades significantly (as evidenced by the absence of a "no pre-training" ablation in Table 3). This contradicts the paper's claim of a "monocular" method, as LiDAR is required for pre-training a mama critical barrier for pure-vision systems.
2. Ambiguity around semantic consistency. During occupancy training, Gaussians derived from one query share a class—good for local consistency, but might limit fine-grained multi-class boundaries inside a query’s spatial span (e.g., curb vs asphalt vs pole tightly adjacent). Some discussion or per-Gaussian residual semantics would help. 
3. Query identity & drift. The paper relies on velocity-aided propagation and δ-separation but doesn’t quantify identity preservation (e.g., re-association failure under occlusion) beyond qualitative visuals. A per-query lifetime/coverage analysis would strengthen claims.
4. Opacity-weighted α trade-offs. Authors note opacity in α improved accuracy but increased training cost (more affected voxels) before CUDA optimizations; inference-time trade-offs (e.g., memory traffic, batch size limits) aren’t reported. A wall-clock throughput table (train/infer) vs baselines would be useful.

### Questions
1. I am curious how sensitive performance is to δ and history length in the propagation queue; can you plot mIoU vs δ and vs queue length to find a sweet spot?
2. How about failure cases under heavy rain/night where monocular depth priors are less reliable: does zero-shot pretraining still hold up, and how does it affect query placement stability?
3. Whether the shared-class assumption within a query ever blends adjacent classes (curb/road/sidewalk). Could per-Gaussian class residuals recover edges without exploding compute? and whether velocity is actor-consistent: do queries attached to a moving car maintain identity over 5–10s, or do they hop? Any quantitative ID-switch metric?
4. If opacity-in-α encourages degenerate small-scale/high-opacity vs large-scale/low-opacity solutions. Did you inspect learned (s, a) distributions across classes and empty space?
5. The paper fails to evaluate model performance when pre-training and fine-tuning datasets differ (e.g., pre-trained on KITTI, fine-tuned on nuScenes). No experiments address whether the pre-training strategy generalizes across domain shifts, undermining the method’s practical applicability

### Soundness
3

### Presentation
3

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
This paper introduces S2GO, a novel streaming framework for 3D occupancy estimation that utilizes 'sparse 3D queries' instead of conventional dense representations. The core contributions are threefold: (1) A LiDAR-based 'geometry denoising' pretraining stage resolves the spatial learning ambiguity of sparse queries. (2) 'Opacity-weighted occupancy estimation' is introduced to improve accuracy. (3) 'Efficient splatting' via CUDA optimization (20x faster backpropagation) halves the training time. As a result, S2GO demonstrates a +2.7 IoU improvement over the SOTA (Gaussian World) on the nuScenes benchmark and achieves 4.5x faster inference, enabling real-time performance (26 FPS).

### Strengths
- Novel and Well-Motivated Approach: The paper introduces S2GO, a novel streaming framework for 3D semantic occupancy prediction that leverages sparse 3D queries instead of voxels or Gaussian-based 3D representations. This represents a clear conceptual advancement, successfully extending Gaussian splatting to the occupancy domain while addressing the long-standing trade-off between accuracy and computational efficiency.
- Core Contributions Address Key Challenges: The paper's main contributions effectively solve the primary challenges of a sparse-query approach. (1) The 'geometry denoising pretraining' directly tackles the sparse-to-dense mapping ambiguity by teaching queries 3D structural priors, as validated by strong ablation results (Table 3). (2) The 'Opacity-weighted occupancy estimation' improves prediction fidelity and stability. (3) The 'Efficient Gaussian-to-Voxel Splating' CUDA optimization (20x faster backpropagation) overcomes the training bottleneck, halving training time and making the high-accuracy Gaussian representation computationally feasible.
- Strong Empirical Results and Efficiency: S2GO achieves new state-of-the-art performance on both nuScenes and KITTI, outperforming Gaussian World by +2.7 IoU while being 4.5–5.9× faster in inference. Notably, the lightweight S2GO-Small runs at 26 FPS on a single RTX 4090, demonstrating strong potential for real-time deployment in autonomous driving systems. The strong performance is well-supported by a comprehensive analysis, including thorough ablations on all key components. Furthermore, qualitative results (Fig. 3,4) confirm strong temporal consistency and object identity preservation, outperforming prior methods.

### Weaknesses
- Missing Comparisons to Voxel-based Efficient Methods: The paper lacks comparisons to recent efficiency-focused voxel methods (e.g., GSD-Occ (AAAI’25)[1], ProtoOcc (AAAI’25 / CVPR’25)[2,3], and StreamOcc (arXiv’25)[4]). Adding direct comparisons (IoU, FPS, memory) and discussing the design trade-offs (sparse query based GS vs. structured voxel) would strengthen the paper.
- Robustness of Geometric Prior: The pretraining's reliance on high-quality geometric data (LiDAR or strong depth models like Metric3D) is a concern. The analysis would be more robust with sensitivity studies on (a) varying LiDAR sparsity (e.g., 16 vs. 64-beam) and (b) the accuracy of the depth estimation input.

[1] He, Y., Chen, W., Wang, S., Xun, T., & Tan, Y. (2025). Achieving Speed-Accuracy Balance in Vision-based 3D Occupancy Prediction via Geometric-Semantic Disentanglement. Proceedings of the AAAI Conference on Artificial Intelligence, 39(3), 3455-3463.
[2] KIM, Jungho, et al. Protoocc: Accurate, efficient 3d occupancy prediction using dual branch encoder-prototype query decoder. In: Proceedings of the AAAI Conference on Artificial Intelligence. 2025. p. 4284-4292.
[3] OH, Gyeongrok, et al. 3d occupancy prediction with low-resolution queries via prototype-aware view transformation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. 2025. p. 17134-17144.
[4] MOON, Seokha, et al. Mitigating trade-off: Stream and query-guided aggregation for efficient and effective 3d occupancy prediction. arXiv preprint arXiv:2503.22087, 2025.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a new method for sparse occupancy prediction. It proposes several methodological and engineering improvements over prior Gaussian-based approaches, including a refined pretraining strategy and an innovative Gaussian-to-Voxel Splatting operator. These contributions demonstrate both conceptual insight and practical significance. The proposed method achieves superior performance compared to previous Gaussian-based methods.

### Strengths
1. The paper is well-organized and written with clear logical flow.

2. It conducts an in-depth exploration of sparse occupancy prediction, proposing improvements such as pretraining strategies and the Gaussian-to-Voxel Splatting operator that are both insightful and practically valuable.

3. The method achieves better performance than prior Gaussian-based approaches, demonstrating tangible benefits of the proposed design.

### Weaknesses
1. Inference speed comparison (Tab. 1) may be unfair: the proposed method uses 256×704 input images, while other methods use 900×1600. Since the image backbone usually dominates inference time, comparison should be made at the same resolution or explicitly clarified in the paper/table.

2. Methods such as [2] achieve 25.5 mIoU on the SurroundOcc dataset but are not included in the comparison table.

3. While reporting results on Occ3D is commendable, the paper omits several recent competitive baselines with higher RayIoU (e.g., [1], [3], [4]).

4. Minor: L242 — “where M is the # of LiDAR points” appears to contain a small typo.

[1] OPUS: Occupancy Prediction Using a Sparse Set, NeurIPS 2024

[2] Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction, CVPR 2025

[3] STCOcc: Sparse Spatial-Temporal Cascade Renovation for 3D Occupancy and Scene Flow Prediction, CVPR 2025

[4] ALOcc: Adaptive Lifting-based 3D Semantic Occupancy and Cost Volume-based Flow Prediction, ICCV 2025

### Questions
1. The method achieves excellent results using lower-resolution inputs. Could the authors report performance using 900×1600 image size to compare under identical conditions with previous Gaussian-based methods?

2. How is the velocity of Gaussians predicted? Is there explicit velocity supervision during training?

3. In the ablation tables (Tab. 3–5), the best-performing configurations achieve different absolute scores. Could the authors clarify the cause of these variations?

4. Could sparse-based methods potentially outperform BEV-encoding methods? For example, ALOcc [4] achieves 39.3 RayIoU and 30.5 FPS on Occ3D-nuScenes, appearing more efficient.

### Soundness
3

### Presentation
4

### Contribution
3
