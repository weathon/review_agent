# WorldMirror: Universal  3D World Reconstruction with Any-Prior Prompting

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
We present WorldMirror, an all-in-one, feed-forward model for versatile 3D geometric prediction tasks. 
Unlike existing methods constrained to image-only inputs or customized for a specific task, our framework flexibly integrates diverse geometric priors, including camera poses, intrinsics, and depth maps, while simultaneously generating multiple 3D representations: dense point clouds, multi-view depth maps, camera parameters, surface normals, and 3D Gaussians. This elegant and unified architecture leverages available prior information to resolve structural ambiguities and delivers geometrically consistent 3D outputs in a single forward pass. WorldMirror achieves state-of-the-art performance across diverse benchmarks from camera, point map, depth, and surface normal estimation to novel view synthesis, while maintaining the efficiency of feed-forward inference. Code and models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents WorldMirror, a multi-view backbone capable of handling multiple input and output modalities. The potential inputs include images, camera intrinsics and extrinsics, and depth maps, while the outputs encompass pointmaps, surface normals, multi-view depths, camera parameters, and 3D Gaussians. The proposed model outperforms prior baselines that are trained for specific tasks, and experimental results demonstrate that incorporating additional input modalities consistently improves prediction accuracy.

### Strengths
1. The paper writing is clear, and the figures are visually appealing.
2. The proposed method supports unified inputs and outputs, which can be applied to a broader range of applications than previous methods taking only images as input.
3. The experiments across various domains comprehensively validate the effectiveness of the proposed method.

### Weaknesses
1. Using addition for image-like inputs (e.g., depth) might be suboptimal. A straightforward alternative is concatenation along the token dimension, though it may incur higher computational cost. Could the authors evaluate this trade-off?
2. The use of bolding and underlining in the tables is confusing. Conventionally, the best results are bolded and the second-best are underlined. However, in Table 1, WorldMirror (w/ intrinsics) with 0.042 is underlined for the mean accuracy on 7-Scenes, even though it is not the second-best result. Similarly, in Table 6, Ours with 20.29 is bolded for PSNR on RealEstate10K, despite not being the best value in that column.

### Questions
1. For depth map conditioning, the depth maps are normalized to the range [0, 1], which removes absolute scale information. Could this lead to issues or degrade performance?
2. In the appendix, the number of training epochs is listed. How long does the training take in terms of wall-clock time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents WorldMirror, a unified feed-forward model for 3D reconstruction that addresses two key limitations of existing methods. First, it introduces a multi-modal prior prompting mechanism that flexibly incorporates camera poses, intrinsics, and depth maps alongside images. Second, it unifies multiple tasks (point cloud reconstruction, camera estimation, depth prediction, normal estimation, and novel view synthesis) within a single architecture. The authors employ curriculum learning across task sequencing, data scheduling, and resolution to train this multi-task model effectively. Experiments demonstrate state-of-the-art performance across diverse benchmarks, with significant improvements when priors are available.

### Strengths
- The paper is clearly written and easy to follow.

- The multi-modal prior prompting approach is well-designed, with specialized encoding strategies for different modalities (single tokens for compact representations vs. dense tokens for spatial information). 

- The ability to leverage any available priors while maintaining feed-forward efficiency addresses real-world scenarios where auxiliary information is often accessible, with demonstrated improvements.

### Weaknesses
- The training procedure is complex (15 datasets, multi-stage curriculum learning), which raises reproducibility concerns.

- The paper lacks a comparison with other methods that can accept auxiliary 3D information (e.g., Pow3R [1] and MapAnything [2]).

[1] Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors

[2] MapAnything: Universal Feed-Forward Metric 3D Reconstruction

### Questions
While the method accepts various priors, the paper assumes they are accurate and provides insufficient analysis of robustness to noisy or low-quality priors. In real-world scenarios, obtaining high-quality camera poses or depth from LiDAR/RGB-D sensors is often challenging, yet the paper doesn't evaluate degradation under realistic noise conditions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a unified neural network architecture designed for a variety of 3D reconstruction tasks. The inputs can take multiple optional forms, such as RGB images, depth maps, and camera intrinsic or extrinsic parameters. The outputs include point maps, depth maps, surface normals, camera parameters, and 3D Gaussians. Overall, it presents a versatile and general framework. The reported experiments demonstrate impressive quantitative and qualitative results across multiple tasks. This appears to be a large-scale and ambitious project, and I appreciate the significant engineering effort invested by the authors.

In summary, however, the main idea somewhat reiterates a well-known observation. That jointly training on multiple related tasks can improve performance across them. This insight, while valid, is not particularly novel as it has been established in many prior works. See details below.

Nonetheless, the experimental results are strong, and the implementation and training complexity reflect a commendable level of technical contribution. Overall, I feel this paper sits at the borderline of acceptance.

### Strengths
- This work requires a substantial amount of experimentation and engineering effort, which I truly appreciate.

- The experimental results are strong, demonstrating state-of-the-art performance across several tasks such as depth estimation, normal estimation, and novel view synthesis.

- The visualization results are also clear and compelling.

### Weaknesses
The main idea of unifying multiple related tasks into a single framework to achieve mutual performance improvement is not novel — it has been explored and confirmed in numerous prior studies. For example:

- [CVPR 2018] “Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics” jointly learns depth, semantic segmentation, and instance segmentation, achieving better performance than single-task models.

- [CVPR 2018] “Taskonomy: Disentangling task transfer learning” provides a principled framework and empirical evidence showing that related tasks can effectively transfer and boost performance for each other.

- [CVPR 2020]: “Pattern-structure diffusion for multi-task learning” jointly trains depth, segmentation, and surface normals on NYUD-v2 and SUN RGB-D, showing consistent performance gains through multi-task learning.

Notably, the above papers are not cited in this work. There are likely additional relevant studies as well.

I suggest that the authors expand the Related Work section to better discuss how this paper differs from these prior efforts, and to clarify whether it brings any new conclusions or insights that might be of particular interest to the community.

### Questions
What is the GPU memory requirement for processing N images in each prediction task?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents WorldMirror, a feedforward geometric model that supports various priors (intrinsic, pose, and depth) and multiple tasks, including camera pose estimation, normal estimation, point cloud reconstruction, and novel view synthesis. The model is built upon VGGT, extending it with additional task-specific heads and priors to support a broader range of inputs and objectives. Experimental results demonstrate that the proposed model achieves state-of-the-art performance across multiple tasks.

### Strengths
- The proposed model achieves SOTA performance on several tasks, including pose estimation, point prediction, normal prediction, and novel view synthesis.

- The model supports multiple inputs and tasks within a single unified framework.

### Weaknesses
- Some parts of the paper closely resemble previous works but lack proper attribution. For example, the proposed Multi-Modal Prior Prompting is similar to Pow3r, except adapted to a multi-view setting. In Line 75, the paper states “we propose”, but the same training strategy has already been explored in Pow3r. Overall, the proposed model appears to be a combination of VGGT, Pow3r, and AnySplat.

- For the novel view synthesis task, the experiments are restricted to 64 input views, whereas AnySplat reports results on over 100 views and compares performance with optimization-based methods (e.g., original 3DGS).

- The paper lacks performance comparisons with recent SOTA novel view synthesis models such as DepthSplat.

### Questions
- What is the input resolution used during evaluation, and how does it compare with previous methods across different tasks, given that different models may use different input resolutions?

- For the two-view novel view synthesis setting, how does the proposed method compare with NoPoSplat?

- What is the maximum number of input views that the model can handle?

### Soundness
3

### Presentation
3

### Contribution
2
