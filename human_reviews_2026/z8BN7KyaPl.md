# RAVEN: End-to-end Equivariant Robot Learning with RGB Cameras

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Recent work has shown that equivariant policy networks can achieve strong performance on robot manipulation tasks with limited human demonstrations.  However, existing equivariant methods typically require structured inputs, such as 3D point clouds or top-down camera views, which prevents their use in low-cost setups or dynamic environments.  In this work, we propose the first $\mathrm{SE}(3)$-equivariant policy learning framework that operates with only RGB image observations.  The key insight is to treat image-based data as collections of rays that, unlike 2D pixels, transform under 3D roto-translations. Extensive experiments in both simulation with diverse robot configurations and real-world settings demonstrate that our method consistently surpasses strong baselines in both performance and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes an end-to-end approach for learning SE(3)-equivariant visuomotor policies directly from RGB input. The key idea is to convert image patches into geometric tokens. Each token carries canonicalized features in a local frame and an explicit SE(3) pose derived from the pixel ray geometry, using camera intrinsics/extrinsics. They leverage geometric transform attention (GTA) layer then computes attention using relative token poses, enabling SE(3)-equivariant attention. Finally, the decoder predicts end-effector (EEF)-relative actions via action tokens anchored at the current EEF pose. Results are reported on standard simulation benchmarks and real-robot tasks.
The approach is logical and practically appealing. While the experimental results over baselines are sometimes limited, the paper presents a clean pathway to SE(3)-aware RGB policies. The claims would be strengthened by broader EquiDiffPo comparisons, parity with 3D-input methods.

### Strengths
-A clear and principled route to achieving SE(3)-equivariance without relying on 3D point-cloud or voxel inputs—enabled by ray-based tokenization and pose-aware attention for end-to-end equivariant policy learning from image-based observations.
 -A coherent system architecture that serves as a practical guideline for leveraging SE(3) symmetry in pixel-based visuomotor control tasks.

### Weaknesses
- Novelty granularity: The core contribution is the "integration" of known components (ResNet features, ray back-projection, pose-aware attention, and an EEF-relative action head). The encoder is mostly standard aside from attaching per-patch poses. The paper's strongest novelty is at the system level, rather than in the encoder/decoder design individually.
- Experimental strength: Improvements over an equivariant baseline (e.g., EquiDiffPo) are often modest, though the baseline leverages SO(2)-symmetry. Given the emphasis on "SE(3) equivariance from 2D", it would be more compelling to match the performance from the 3D-input methods, (using e.g., point-cloud/voxel) under identical protocols [1,2]. 

<Typos>
p.6, 4.2., 3rd paragraph: convert he action geometric tokens" -> "convert the action geometric tokens"

[1] Wang et al., Equivariant Diffusion Policy (2024).
[2] Zhu et al., SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space (2025).

### Questions
- Include EquiDiffPo comprehensively (at least all simulation tasks) for a fair comparison.
- Add comparisons to 3D input policies (point-cloud/voxel) to substantiate the benefits of RGB-only SE(3)-equivariance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents RAVEN, the first SE(3)-equivariant policy learning framework that operates directly on RGB image observations. The key innovation is treating image patches as collections of 3D rays that transform under SE(3) roto-translations, enabling equivariant processing through geometric tokens.

### Strengths
- The ray-based formulation for achieving SE(3) equivariance from RGB images is elegant and principled. Converting image patches to geometric tokens with SE(3) poses is a creative solution to a real problem in robotics.
- The experiments cover a range of test settings, and the ablation studies help understand each component of the method.
- Overall the experiment results are good, which demonstrates the effectiveness of the proposed method.

### Weaknesses
- **Limited Theoretical Analysis of Ray-Patch Representation**: While the paper states that "each grid cell in the feature map is associated with an element of SE(3)", the justification for how a patch of pixels/rays defines a unique orientation is somewhat hand-waved (Figure 2). A more rigorous analysis of when this representation is well-defined and how discretization affects equivariance would strengthen the work.

- **Incomplete Equivariance Analysis**: The viewpoint generalization task (Section 5.1) perturbs only a single camera, which "breaks global equivariance" as the authors note. This seems to contradict the claimed advantage of camera-agnostic operation ("remaining agnostic to the number and placement of cameras," line 48).

- **Real-World Experiments Limited in Scope**:
  - Only 4 tasks with relatively simple manipulation primitives are considered.
  - No analysis of failure modes or edge cases is provided.

### Questions
- **Ray Discretization**: How does the grid resolution (5×5 in your implementation) affect the quality of the ray-based representation and overall performance?

- **Viewpoint Generalization Paradox**: If the method is truly camera-agnostic, why does performance drop when only the agent-view camera is perturbed (Table 3)? Doesn't this suggest the equivariance guarantee is weaker than claimed?

- **Computational Cost**: What is the actual inference time compared to baselines? Training time is reported, but deployment efficiency matters for robotics.

- **Flow Matching vs Diffusion**: Is the performance gain primarily from the ray-based encoder or from using flow matching instead of diffusion? An ablation would clarify this.

- **Comparison with [1]**: This work also claims RGB-based SO(3) equivariance. What are the key differences, and why wasn't this included as a baseline?


[1] Boce Hu, Xupeng Zhu, Dian Wang, Zihao Dong, Haojie Huang, Chenghao Wang, Robin Walters, and Robert Platt. Orbitgrasp: Se(3)-equivariant grasp learning. arXiv preprint arXiv:2407.03531

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
5

### Summary
The paper proposes a ray-based SE(3) equivariant visual manipulation policy that can be used end-to-end from RGB images without involving traditonal 3D modalities like point clouds. Speficially, a feature map from pretrained 2D image model is lifted to featured rays using known camera parameters. These featured rays are grouped to form a SE(3)-posed geometric token. A series of transformer blocks with Geometric transform attention (GTA) is then applied to equivariantly process visuo-proprioceptive information. Finally, these equivariant representations are decoded into flow-matching prediction. Experimental results on standard benchmarks confirm clear advantage of the proposed method.

### Strengths
### [Strength 1] Novelty
As far as I know, this is the first ray-based equivariant robot policy. While [1] has previously explored equivariant models in Plucker space by leveraging its homogeneous space property as being SE(3)/(SO(2)xR), it was limited to computer-vision oriented applications. As ray-based representation is becoming increasingly popular in 3D vision fields [2,3,4], it is timely to explore its application to robotics field. 

### [Strength 2] Scalability
Most existing SE(3) equivariant manipulation methods are not scalable as they rely on complicated and model-specific operations that are not well-optimized. On the other hand. the proposed method is more compatible with more commonly used operations that are heavily optimized. For instance, GTA is compatible with flash attention, which can greatly improve memory efficiency and training speed. 

[1] Xu et al. ""Equivariant Light Field Convolution and Transformer." 

[2] Jin et al. "Lvsm: A large view synthesis model with minimal 3d inductive bias."

[3] Jiang et al. "RayZer: A Self-supervised Large View Synthesis Model."

[4] Li et al. "Cameras as relative positional encoding."

### Weaknesses
### [Weakness 1] Definition of Equivariance is Questionable
The claimed SE(3) equivariance property of the proposed method is defined with respective to the gripper and camera poses. This is a valid and helpful inductive bias especially when denoising multiple gripper pose trajectory or incorporating multiview information. However, it should be noted that this kind of definition is different from more accepted defintion of equivariant policy where the equivariance of the action is defined with respect to the content of the scene, not to the camera. Assuming correct point cloud registration, scene-to-gripper equivariance is a strict superset of this camera-to-gripper equivariance. As such, it is not quite true that this is "first end-to-end equivariant policy learning method that works with image-based observations." In fact, this is a shared limitation of other equivariant methods based on ray [1] or spherical projection [2]. 

[1] Xu et al., "Equivariant Light Field Convolution and Transformer." 

[2] Hu et al., "3D Equivariant Visuomotor Policy Learning via Spherical Projection"

### [Weakness 2] Effectiveness of Equivariance is Questionable
Experimental results are solid enough to support the effectiveness of the proposed method. However, it is unclear equivariance is indeed the main factor of this observed advantage. The use of GTA-like camera RoPE approach has been reported to be effective in several non-equivariant methods [3,4,5]. Hence, unless an ablation without equivariance is provided (i.e., using nonequivariant MLP), I would be inclined to believe that the performance advantage is mainly from this relative attention property, not from equivariance. In terms of benchmark, I think it is more fair to compare the baseline diffusion policy with the same GTA architecture (but with non-equivariant counterpart) than using the original Unet-based architecture. 

Also, it is unclear why ray-based representation is necessary to achieve the equivariance. Unlike [1] which used homogeneous space convolution, the proposed method lifts rays onto SE(3) space without ever being grounded to ray space again. If so, why not simply tokenize a featured point cloud projected to a unit sphere in the local camera reference frame, and then lifting it onto SE(3)?

[3] Miyato et al. "Gta: A geometry-aware attention mechanism for multi-view transformers."

[4] Kong et al. "Eschernet: A generative model for scalable view synthesis."

[5] Li et al. "Cameras as relative positional encoding."

### Questions
Please find my questions in the weaknesses section.

### Soundness
3

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
3

### Summary
This paper presents RAVEN, an end-to-end SE(3)-equivariant policy learning framework that operates using only RGB camera inputs. The key idea is to interpret 2D image patches as 3D rays defined by known camera intrinsics and extrinsics, which allows the use of SE(3)-equivariant geometric transformers to process visual observations. The encoder outputs geometric tokens that can be fused across arbitrary camera views, and the policy is trained with flow matching to predict continuous-time action trajectories. Empirically, RAVEN achieves strong results on MimicGen and DexMimicGen benchmarks and shows faster training and better viewpoint generalization compared to prior equivariant or diffusion-based policies.

### Strengths
The work introduces a creative and elegant approach to extend SE(3)-equivariance to RGB-only observations, bridging a long-standing gap between equivariant robotic policies (which typically rely on structured 3D inputs) and practical camera-based systems. The ray-based tokenization formulation is novel and theoretically sound.

RAVEN emperically outperforms strong baselines across multiple tasks and even surpasses pretrained diffusion policies while training faster. The evaluation covers both simulation and real-world tasks.

### Weaknesses
Limited applicability in realistic deployments: The method’s core assumption — that camera extrinsics and intrinsics are precisely known and remain static — limits its applicability in unstructured real-world settings. The real-world experiments use carefully calibrated, fixed setups, so the paper does not convincingly demonstrate robustness to realistic calibration noise.

Missing ablations / comparisons on input modality and architecture choices: The evaluation focuses exclusively on RGB input. Despite claiming compatibility with multimodal data, the paper lacks any ablation or comparison using non-equivariant model + depth, RGB-D, or depth-estimation-based representations, which are standard in 3D-aware robot learning literature. It is therefore unclear whether the gains stem from true equivariance or from better geometric bias in the encoder.

### Questions
1. Camera calibration sensitivity: How robust is RAVEN to small errors in extrinsic or intrinsic calibration? Could the authors quantify the degradation in performance?

2. Depth and RGB-D ablation: Have the authors tried integrating depth maps or estimated depth (e.g., from pretrained monocular models) into the ray-based representation? This would make a strong case for the claimed modality-agnostic design.

3. Failure analysis: It would be useful to include qualitative failure cases — for instance, where equivariance assumptions break or where viewpoint perturbations cause prediction drift — to better understand the model’s limits.

### Soundness
3

### Presentation
2

### Contribution
2
