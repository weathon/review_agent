# UniSplat: Unified Spatio-Temporal Fusion via 3D Latent Scaffolds for Dynamic Driving Scene Reconstruction

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Feed-forward 3D reconstruction for autonomous driving has advanced rapidly, yet existing methods struggle with the joint challenges of sparse, non-overlapping camera views and complex scene dynamics. 
We present UniSplat, a general feed-forward framework that learns robust dynamic scene reconstruction through unified latent spatio-temporal fusion. UniSplat constructs a 3D latent scaffold, a structured representation that captures geometric and semantic scene context by leveraging pretrained foundation models. To effectively integrate information across spatial views and temporal frames, we introduce an efficient fusion mechanism that operates directly within the 3D scaffold, enabling consistent spatio-temporal alignment. To ensure complete and detailed reconstructions, we design a dual-branch decoder that generates dynamic-aware Gaussians from the fused scaffold by combining point-anchored refinement with voxel-based generation, and maintain a persistent memory of static Gaussians to enable streaming scene completion beyond current camera coverage. Extensive experiments on real-world datasets demonstrate that UniSplat achieves state-of-the-art performance in novel view synthesis, while providing robust and high-quality renderings even for viewpoints outside the original camera coverage.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes UniSplat, a feed-forward framework for dynamic driving scene reconstruction based on 3D Gaussian Splatting, introducing a 3D latent scaffold that enables unified spatio-temporal feature fusion across multi-view camera inputs and sequential frames. A dual-branch Gaussian decoder jointly handles fine-grained reconstruction and spatial completion, while a streaming Gaussian memory with dynamic-aware filtering progressively accumulates static scene content to address coverage gaps beyond the current field of view.

### Strengths
1. The 3D latent scaffold and spatio-temporal fusion mechanism introduce a more structured way to aggregate multi-view and temporal cues, beyond 2D feature aggregation used in previous feed-forward approaches.
2. Dynamic-aware Gaussian memory is conceptually appealing, enabling persistent scene completion and reducing ghosting artifacts from moving objects.

### Weaknesses
1. Readers may find it difficult to understand why the scaffold representation is fundamentally more suitable for handling sparse multi-view driving scenarios.
2. The streaming Gaussian memory mechanism is effective in mitigating ghosting and compensating camera blind spots. However, it is introduced as a component-level implementation rather than a more formalized recurrent world-state update.
3. The dual-branch decoder (point-based detail refinement and voxel-based completion) shows quantitative gains, but the ablations do not analyze where each branch contributes most (e.g., near-field vs.\ far-field geometry, static vs.\ dynamic regions).

### Questions
1. Regarding the streaming memory, can the authors provide more insight into how the memory size evolves over long sequences? Does the model include any mechanism for memory pruning or confidence-based discarding, and could memory saturation impact performance?
2. For the dual-branch decoder, have the authors observed complementary behaviors between point-anchored and voxel-generated Gaussians across different scene structures? A visual or statistic showing their relative contributions would help clarify the motivation.
3. How sensitive is the overall system to the accuracy of the geometry foundation model used for initializing the scaffold? For example, if the model produces slight drift or inconsistent scale, could this propagate and affect memory fusion stability?

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
This paper introduces UniSplat, a unified framework for spatio-temporal fusion and dynamic scene reconstruction.
The key idea is to construct a latent 3D scaffold where both spatial fusion (within a frame) and temporal fusion (across frames) are performed in the same structured voxel domain.
The previous fused scaffold is warped to the current frame using ego-pose and merged directly, avoiding redundant computation and maintaining geometric consistency.
A dual-branch Gaussian decoder combines point-based details with voxel-based coverage, while a dynamic filtering mechanism accumulates only static components in a memory buffer to enable out-of-FOV reconstruction.
Experiments on the Waymo and nuScenes datasets show that UniSplat significantly improves reconstruction accuracy and efficiency over prior works such as MVSplat, Omni-Scene, and DriveDreamer4D.

### Strengths
Proposes a unified spatio-temporal fusion paradigm in a single latent 3D scaffold, which is a conceptually cleaner and more efficient design than previous separate spatial and temporal modules.
The dual-branch Gaussian decoder effectively balances fine-grained details (point) and global completeness (voxel).
The dynamic filtering + memory streaming mechanism elegantly addresses out-of-FOV reconstruction and ghosting issues caused by dynamic objects.
Strong experimental results and ablations demonstrate the necessity and contribution of each module.
The framework is feed-forward and efficient, making it more practical than diffusion- or transformer-based alternatives.

### Weaknesses
While the framework is conceptually unified, many of its components (voxel scaffold, temporal warping, Gaussian splatting) are adapted from prior work.
The dynamic filtering relies on threshold-based heuristics; it is not clear how robust this is under complex motion or sensor noise.
The paper lacks a deeper comparison with recent diffusion-based or token-based reconstruction frameworks, which could strengthen the positioning of this method.
Some design choices (e.g., the specific form of the dual-branch decoder, scaffold resolution) are under-explained, and might be perceived as empirical rather than principled.

### Questions
Why did you choose to perform spatio-temporal fusion entirely in the latent 3D scaffold domain rather than using separate spatial and temporal modules (as in Omni-Scene)? What specific benefits (efficiency, stability, or accuracy) did you observe?

Could you elaborate on why the dual-branch decoder was designed in its current form? Were other fusion strategies (e.g., adaptive weighting between point and voxel) explored?

How robust is the dynamic filtering threshold under varying dynamic motion or sensor noise? Could adaptive strategies improve generalization?

Why not adopt token-based or diffusion-based representations for the scaffold? Is the explicit voxel representation crucial for efficiency or accuracy?

How does the method behave when the camera view coverage becomes extremely sparse (e.g., two front-facing cameras only)?

Are there scenarios where the streaming memory might accumulate false positives from misclassified dynamic objects, and how would the method handle this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes UniSplat, a feed-forward framework for dynamic driving-scene reconstruction that (i) builds an ego-centric 3D latent scaffold by fusing features from a geometry foundation model and a visual foundation model, (ii) performs spatio-temporal fusion directly in the scaffold via sparse 3D UNets with pose-warped accumulation, and (iii) decodes dynamic-aware Gaussians using a dual branch (point-anchored + voxel-anchored) with a streaming static-memory to complete unseen regions. 

The dynamic prediction is supervised by pseudo-labels. These labels for moving objects are identified via 3D bounding-box tracking; the boxes are projected to the image to prompt SAM2, which returns dynamic segmentation masks. 

On Waymo and nuScenes, UniSplat reports leading image metrics for reconstruction and novel view synthesis, and claims robustness for viewpoints outside camera coverage.

### Strengths
- An engineering-first pipeline. This paper proposes three-stage pipeline with a good motivation for 3D scaffold–space fusion, using sparse 3D UNets for spatial aggregation and pose-conditioned temporal accumulation. Overall it's easy to follow.
- Dynamic-handling. The proposed method can handle dynamic scenes.
- Experimental coverage. Evaluations on Waymo and nuScenes with both quantitative tables and qualitative figures; ablations cover feature composition, spatial vs temporal fusion, and decoder branches.

### Weaknesses
- **A System with Limited Conceptual Novelty**: The primary weakness is the paper's contribution, which appears to be more of a strong engineering effort than a conceptual breakthrough.
  - The “3D latent scaffold” (spatial-fusion) is essentially a sparse voxel grid with fused geometry+semantic features, a very standard way (fused in 3D grids) for fusing 3D features in many domains (like SLAM, or 3D understanding) and recent generalizable 3DGS/voxel/triplane works. 
  - Warping and fusing features/primitives from previous frames by a pose-warped addition is another straightforward and known method used by the community for years (e.g., BEVFormer, BEVFusion and earlier works). 
  - Using powerful, frozen geometry and visual foundation models to provide strong priors is a common practice. 
  - The dual point/voxel decoder branches are a sensible design but I feel this is a compromise to sparse-voxel representations.
  - While the final system is effective, the core ideas themselves are largely incremental, making the work feel less insightful and more engineering-centric.

- Missing Related Work & Supervision Dependency: The paper misses discussion of several highly relevant works on feed-forward dynamic scene reconstruction, such as Flux4D (Want et al., 2025) and STORM (Yang et al., 2025). These methods also tackle large-scale, dynamic outdoor scenes, but they explore unsupervised 4D reconstruction, i.e., they do not require dynamic segmentation masks from external sources like object bounding boxes and SAM2.


- (Not a weakness) Setting novelty aside, this is a well-executed project with sufficient engineering efforts and clear implementation details.


[1] Want et al. "Flux4D: Flow-based Unsupervised 4D Reconstruction." In NeurIPS 2025

[2] Yang et al. "STORM: Spatio-Temporal Reconstruction Model for Large-scale Outdoor Scenes." In ICLR 2025

### Questions
1. Positioning / novelty. Is there any unique capability or perspective/insight offered by the proposed method?

2. Baseline Strength vs. Main Results. I understand that numbers in ablation studies are not directly comparable to those in main results, but it seems like a minimal UniSplat variant (without several add-ons) already performs strongly, possibly already exceeding the competitor results in the main table?

3. End-to-end latency. Please report full inference latency that includes the geometry/vision foundation stages.

4. External-Free and Voxel-Only Variants. Are these experiments feasible in the current setting? 
  - Table 3: Add a no-Geo / no-Sem point (no external models) to show the fully end-to-end trained performance.
  - Table 5: Add a voxel-only prediction variant (no point-anchored branch) to isolate the value of the dual-branch decoder.

### Soundness
3

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
4

### Summary
This paper presents UniSplat, a feedforward framework for dynamic driving scene reconstruction and novel view synthesis. The core idea is to construct a 3D latent scaffold that unifies multi-view spatial and multi-frame temporal fusion. The scaffold is built by leveraging geometry and visual foundation models to encode geometry and semantics in 3D space. UniSplat then introduces a Gaussian decoder to generate the 3DGS reconstruction from the scaffold, and a memory mechanism to maintain the accumulated static Gaussians over time to improve completion. The proposed UniSplat achieves state-of-the-art performance on Waymo and nuScenes datasets and demonstrates SoTA performance for novel view synthesis compared to existing feed-forward reconstruction methods.

### Strengths
* The paper is well written and clearly structured, making the technical ideas easy to follow despite the method’s complexity.
* The proposed scaffold-based fusion mechanism is both intuitive and practical, effectively addressing the challenge of modelling dynamic scenes while supporting progressive memory updates.
* Experimental evaluation is thorough, including comparisons to multiple strong baselines on two large-scale driving datasets, with consistent quantitative and qualitative improvements. Ablation studies also provide convincing evidence of the contribution of each component.

### Weaknesses
* It is not entirely clear how the scaffold-based fusion mechanism handles dynamic content. For example, when a vehicle moves across frames, how are its features aligned or updated consistently during fusion?
* The paper shows novel view rendering by rotating the camera, but it remains unclear whether dynamic actors can be moved or manipulated. Supporting such editing would enable full camera simulation.
* Maintaining and updating a dense 3D scaffold could be computationally expensive for long sequences or large-scale scenes. The paper does not analyze memory usage or runtime trade-offs beyond short sequences.
* The evaluation focuses mainly on reconstruction metrics. Additional experiments on temporal consistency, and robustness under occlusions would make the claims more convincing.
* The predicted dynamic masks appear misaligned with RGB images in the supplementary video (e.g., at 1:03). Clarifying the source of this misalignment would strengthen the work.

### Questions
* How does UniSplat perform when extended beyond driving scenes to other dynamic 4D environments (e.g., human motion, indoor scenes)?
* Could the authors clarify the scalability of the scaffold fusion to longer sequences (e.g., 1–2 minutes) and higher-resolution scenes?
* Can the streaming memory lead to error accumulation or stale Gaussians over time? How is this mitigated?

### Soundness
3

### Presentation
3

### Contribution
3
