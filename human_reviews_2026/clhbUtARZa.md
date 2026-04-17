# VideoArtGS: Building Digital Twins of Articulated Objects from Monocular Video

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Building digital twins of articulated objects from monocular video presents an essential challenge in computer vision, which requires simultaneous reconstruction of object geometry, part segmentation, and articulation parameters from limited viewpoint inputs. Monocular video offers an attractive input format due to its simplicity and scalability; however, it's challenging to disentangle the object geometry and part dynamics with visual supervision alone, as the joint movement of the camera and parts leads to ill-posed estimation. While motion priors from pre-trained tracking models can alleviate the issue, how to effectively integrate them for articulation learning remains largely unexplored. To address this problem, we introduce VideoArtGS, a novel approach that reconstructs high-fidelity digital twins of articulated objects from monocular video. We propose a motion prior guidance pipeline that analyzes 3D tracks, filters noise, and provides reliable initialization of articulation parameters. We also design a hybrid center-grid part assignment module for articulation-based deformation fields that captures accurate part motion. VideoArtGS demonstrates state-of-the-art performance in articulation and mesh reconstruction, reducing the reconstruction error by about two orders of magnitude compared to existing methods. VideoArtGS enables practical digital twin creation from monocular video, establishing a new benchmark for video-based articulated object reconstruction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the problem of reconstructing articulated objects from monocular video, which requires simultaneously estimating geometry, part segmentation, and articulation parameters. The challenge lies in disentangling object geometry from part dynamics when both camera and object parts move together. The paper introduces VideoArtGS, which reconstructs digital twins through: (1) a motion prior guidance pipeline that analyzes 3D tracks and initializes articulation parameters, and (2) a hybrid center-grid part assignment module for accurate part motion capture.

### Strengths
1. The paper effectively addresses articulated object modeling from monocular video by incorporating motion priors from pre-trained tracking models to resolve reconstruction ambiguities.

2. The hybrid center-grid part assignment module combined with the motion prior guidance pipeline offers an end-to-end framework that handles both initialization and accurate part motion modeling.

### Weaknesses
1. The VideoArtGS-20 testing set should not be listed as a contribution, as it simply consists of 20 rendered objects from the existing PartNet-Mobility dataset without significant added value.

2. The reconstructions lack smoothness and exhibit really coarse surfaces, particularly evident in real-world examples in Figure 4 (e.g., the laptop and chair). The geometry appears very noisy.

3. Using GPT-4o to predict joint number and type seems inconsistent with your overall approach. Since you leverage motion priors for other components of the pipeline, why not also predict joint number and type using motion priors?

4. Lack of video demonstrations: Since this work focuses on modeling articulated objects from videos, video results showing temporal consistency and motion would strengthen the demonstration. Static images alone do not fully showcase the method's capabilities.

5. Missing texture visualization: Given that you use Gaussian Splatting as your representation with rendering loss for supervision, I would expect to see textured results in your visualizations.

### Questions
Please see the weaknesses. I am open to raising my score if my concerns are addressed.

### Soundness
2

### Presentation
2

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
The paper presents VideoArtGS, a method for reconstructing articulated 3D objects in a scene from videos. It starts from depth and motion estimation from VGGT and Tapip3d, and cluster points with similar motion pattern into parts. Then optimizes a gaussian deformation field with rendering and tracking losses. It achieves better results than prior works on the Video2Articulation dataset, and the VideoArtGS-20 dataset introduced by this paper.

### Strengths
- The method is sound. The classification of articulated motion into 4 predefined motion types makes monocular reconstruction more well-posed. 
- It represent the motion of points as a mix of rigid transformation given the distance to each part, where the static parts are handled separately with "staticness" logit. This hybrid assignment is simple and seems effective for separating movable vs. base parts
- It introduces a new VideoArtGS-20 dataset and reports of large improvements over prior art
- Strong ablations showing the improvement from motion-prior init and other components.

### Weaknesses
- Contributions are slightly incremental. The core modeling designs are similar to ArtGS. Gaussians + deformation deformation fields is established in prior works; most novelty is in initialization of motion type and assignment of points to parts.
- Heavy reliance on upstream systems (VGGT, TAPIP3D, GPT-4o). This could make the pipeline complex and less scalable.

### Questions
- Can authors clarify the use of GPT-4o vs 3D tracking, in terms of identifying the motion type and clustering?
- The deformation in Eq(1) assigns a soft weighting to each point, while this makes sense from an optimization perspective, the motion most of the objects/furnitures shown in the paper are controlled by a single rigid transformation. 
- How is the occluded part of the scene represented/reconstructed and what happens if some part of the geometry is missing in the input video? e.g., the back of the table
- Are there motion types that are excluded by the 4 motion types used in the paper, and if so, how to combine those?
- The reconstruction results look not as high quality as recent 3d object/part generation models. Potentially combining those would improve the results.

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
This paper presents a method for reconstructing articulated objects from a single monocular video. The authors address the  problem by using two main components: first, a motion prior guidance pipeline that analyzes and filters 3D tracks from an upstream model (TAPIP3D) to initialize articulation parameters, and second, a hybrid center-grid part assignment module to segment static bases from movable parts. The method utilizes an articulation-based deformation field over 3D Gaussian Splatting.

### Strengths
1. This work addresses the challenging problem of articulated object reconstruction from  monocular video.
2. The method achieves a significant reduction in error compared to baselines and demos are looking good.

### Weaknesses
1.  The pipeline's success is critically dependent on the quality of two separate pre-trained models: VGGT for depth/pose and TAPIP3D for 3D tracks. This is a "garbage in, garbage out" system, and failures in these upstream models will cascade.

2. Part segmentation is derived entirely from motion clustering. This will inherently fail for parts that are not moved in the video, parts with very subtle motion, or objects with many parts moving simultaneously. Or if the parts are not moved to its maximum extend. There could be more information like semantic priors to be utilized.

3. The method relies on GPT-4O to predict the number of joints and their types (revolute/prismatic). This is a significant external dependency and a potential point of failure that is not analyzed.

4. The motion pattern analysis (Sec 3.2) involves several components (RANSAC, SVD fitting) and thresholds. This pipeline may be brittle and highly tuned. Thus the generalization ability could be a concern. 

5. On the new VideoArtGS-20 dataset, the authors compare against methods (like Video2Articulation) that were not designed for multi-part objects and had to be manually adapted, which may inflate the performance gap. Similar to the question1 I raised, I doubt the good performance is highly tuned with its own carefully contructed dataset and might not generalize well.

6. The method requires the video to begin with a static sequence to initialize the canonical 3D Gaussians. This is a major limitation that prevents its application to most in-the-wild videos.

### Questions
1. In real world, many articulated objects don't move themselves. The would be humans/robots operate them to open and close the prismatic and revolute joints. This would definately result in occlusion. In the demos, human operators carefully move the objects to prevent occlusion. But if we use web videos where people operates them normally, how would you consider the occlusion problems? As I mentioned in weaknesses 6, the method relies on static start, if the inner part of a prismatic part like a drawer was occluded in the dynamic part, the method would lilely to fail?

2. Since the demo results looks good, I will give my initial score as marginally above the acceptance threshold. But those weaknesses I listed I still got many concerns on the work. I will consider the reviews from other reviewers and might adjust my score if my concerns are not resolved.

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
3

### Summary
This paper introduces VideoArtGS, a system for reconstructing articulated object digital twins—including geometry, segmentation, and joint parameters—from monocular videos. The method builds upon Gaussian Splatting frameworks and integrates 3D motion priors and hybrid part assignments to recover both geometry and kinematics.

Motion Prior Guidance: It uses precomputed 3D trajectories (via TAPIP3D) and adaptive fitting (line/circle) to distinguish prismatic vs. revolute joints. The fitted parameters (axis, anchor, and joint state) are optimized using bidirectional trajectory consistency (c2o and o2o) to ensure temporal coherence.

Hybrid Center–Grid Assignment: The paper proposes a mixed scheme that assigns Gaussians to movable parts based on a center–Mahalanobis distance metric, and to static regions using a voxel-hash grid.

Optimization: The framework assumes the first N frames are static for canonical initialization, then jointly optimizes geometry and motion over the entire sequence using a photometric and consistency loss.

Experiments on Video2Articulation-S and a newly introduced VideoArtGS-20 dataset show strong improvements over Video2Articulation and ArticulateAnything, both quantitatively (two orders of magnitude reduction in axis and position errors) and qualitatively (cleaner reconstructions and more realistic part motion).

### Strengths
Addresses an important real-world problem: reconstructing interactable articulated objects from monocular video.

Integrative framework: elegantly combines motion priors, kinematic reasoning, and differentiable Gaussian rendering.

Solid empirical improvements: consistent quantitative and qualitative gains over existing baselines.

Ablation studies (motion prior and hybrid assignment removal) are provided and clearly demonstrate their necessity.

The method seems reproducible in principle, suggesting strong implementation work.

### Weaknesses
Over-reliance on upstream priors: The system’s success depends on TAPIP3D and VGGT, yet no robustness or substitution tests are conducted (e.g., replacing with weaker or noisier trackers).

Unrealistic static-frame assumption: The model presumes the first N frames are motionless; this limits generalization to natural videos. No adaptive mechanism or failure handling is proposed.

Limited generalization evidence: The “two orders of magnitude” improvement is mainly on synthetic or semi-synthetic datasets; real-world, unconstrained sequences are underrepresented.

Insufficient dataset transparency: Details of VideoArtGS-20’s annotation, coordinate conventions, and release availability are not discussed.

Lack of failure analysis: No qualitative examples of difficult cases (occlusion, partial visibility, transparent parts, trajectory drift) are shown.

Unreported runtime and efficiency: The training time, memory footprint, and inference speed are not compared with prior works.

Terminology and claims: The term “digital twin” suggests full physical consistency, but the framework models only geometric and kinematic properties, not physical constraints.

### Questions
How does performance change if TAPIP3D is replaced or degraded? Please provide robustness tests under noisy trajectories or alternative trackers.

Can the model relax the static N-frame assumption? For example, by learning canonical frames adaptively or detecting stationary segments automatically?

How are prismatic vs. revolute joints differentiated numerically—what thresholds or heuristics are used for line/circle fitting, and how sensitive are results to them?

How are mixed center–grid weights tuned? Are they learned jointly or fixed heuristically?

Please provide runtime and GPU memory comparisons with Video2Articulation and ArticulateAnything.

Are the reported “two orders of magnitude” improvements averaged across all categories, or do some categories dominate?

Will VideoArtGS-20 be publicly released with ground-truth joint annotations and evaluation code? If not, how can others reproduce the reported metrics?

How does the method handle complex real-world artifacts (e.g., flexible cables, soft hinges) that break rigid joint assumptions?

### Soundness
2

### Presentation
2

### Contribution
2
