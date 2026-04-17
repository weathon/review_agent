# Feature Consistent 4D Gaussian Splatting for Realistic Dynamic View Synthesis

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Dynamic novel view synthesis remains challenging due to the complexity of diverse motion patterns. In 4D Gaussians, the temporal dimension further complicates constraint formulation, making temporally consistent rendering difficult. To address this, we introduce 4D Feature Gaussian Splatting (F4DGS), a dynamic rendering algorithm that introduces feature consistency regularization to enable realistic rendering. This regularization jointly synchronizes hierarchical semantic features, velocity, and depth, ensuring coherent motion and appearance. We further extend the regularization beyond static alignment to capture temporal associations over continuous unit time intervals. F4DGS is the first rendering algorithm to explicitly couple velocity and depth for learning motion-consistent 4D representations, enabling high-fidelity, physically plausible rendering of dynamic content. Through comprehensive evaluations on monocular and multi-view dynamic datasets, F4DGS achieves real-time, high-resolution rendering and consistently outperforms existing methods across both quantitative and qualitative benchmarks. Notably, F4DGS achieves a 3.51 PSNR improvement on the Plenoptic dataset with comparable rendering speed and training time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript proposes a dynamic novel view synthesis framework, and claims that its integration of feature consistency regularization can facilitate more realistic rendering. The first part of this regularization aims to align the input and rendered multi-level semantic features by optimizing their optimal transport distance. And they try to improve the temporal consistency by constraining the OT distance to remain stable across frames. The second part augment each Gaussian with velocity and depth attributes and aims to encourage physically realistic motion by optimizing depth-flow consistency on them.

### Strengths
The idea of incorporating semantic consistency to regularize Gaussian Splatting’s optimization is interesting. Ideally, such semantic cues have the potential to help Gaussians preserve consistent object identities in regions where texture boundaries are ambiguous, by leveraging larger context and priors.

### Weaknesses
1.	The poor presentation makes this manuscript seems less serious. Many crucial details are missing in both the main text and appendix，which jeopardizes the reproducibility of some key components and justification of their design choice:
   -	The core contribution of this work seems to be the hierarchical OT–semantic regularization, yet the construction of hierarchical semantic features has not been clearly described. For instance, the ${z_{i}^{l}}$ in Line 202 is introduced without a definition or explanation of how it is exactly obtained.
   -	Please provide a clear definition of the unit time interval and specify the summation ranges in Equations (6) and (9).
   -	What does the depth and velocity attributes in L266 exactly refer to? Are they just some appended parameters like SH coefficients? If so, what guarantees that “velocity encodes both inter-object dynamics and local motion”? It seems that there is nothing to guarantee these attributes correspond to the literal velocity and depth of each Gaussian, and thus Equation (9) is not a reasonable objective, e.g., a trivial solution where $d$ is constant and $v=0$ could minimize it but is meaningless. Even if they indeed represent velocity and depth, the second term in Equation (9) appears to merely to force the motion along the depth dimension.
One the other hand, many statements sound empty or overclaimed:
   -	Why is the motion feature “transferable and invariant” (L078)?
   -	How does “temporal modeling” help “simulate complex light interactions” (L080)?
   -	The paper claims to “enable accurate scene understanding” (L104) without providing any experiments supporting this.
2.	As for the methodology, penalizing the deviation of OT distance loss across frames is an indirect and unstable optimization objective, relying on it to achieve smooth temporal transitions sounds weird.
3.	The evaluation appears not very rigorous. It lacks comparison with recent works--even the latest methods used for comparison was proposed over two years ago, and the performance of GS-based baseline is substantially lower than those in their original paper. In Table 2, the performance of proposed method significantly lags behind Deformable3DGS. Why is it not highlighted instead?
4.	No video results are provided which is important to assess the dynamic novel view synthesis method, and the images shown in the paper appear overly compressed.
5.	Some visualizations are questionable. E.g., the zoom-in regions in Figure 4 are identical to the ground truth, which is implausible. In Figure 5, neither component produces background motion individually, why their combination suddenly yields noticeable flow in this region?

### Questions
Would the proposed OT–semantic regularization also be effective for novel view synthesis of static scenes?

### Soundness
1

### Presentation
1

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
This paper presents F4DGS, featuring two new regularization terms: 4D hierarchical OT-semantics regularization and 4D motion-depth regularization. The core claim is that these regularizations jointly enforce spatiotemporal, semantic, and physical consistency, leading to better rendering quality and realism in dynamic scenes. While the paper demonstrates strong quantitative results and the ideas are intuitively appealing, the presentation and empirical validation make it difficult to fully assess the generalizability of the contributions.

### Strengths
Originality: The explicit coupling of velocity and depth for regularizing 4D Gaussian motion is a novel and interesting contribution. The idea of using Optimal Transport (OT) to align hierarchical semantic features across time is also a fair extension of prior work that focuses primarily on photometric or geometric losses.
Quality: The quantitative results shows a noticable PSNR improvement (e.g., +3.51 on Plenoptic) over baselines like Deformable4DGS while maintaining real-time rendering speeds. This suggests the method is effective.
Clarity: The high-level problem formulation and the breakdown of the method into two main components (OT-semantics and motion-depth) are well organized.

### Weaknesses
1. Insufficient Ablation Studies and Analysis:
Component Interdependence: The ablation study in Table 3 is a good start but remains superficial. It shows that each component helps, but it does not isolate their individual contributions to specific claimed benefits.The study lacks analysis on how each loss term contributes to the final result.
Baseline Comparison: The choice of baselines is inappropriate, most of baselines are NERF based methods, with Deformable4DGS being the only 4DGS method. A more convincing results for would require more comparison with other 4DGS methods. Other attributes, such as temporal consistency metrics (e.g., tLPIPS), optical flow accuracy, or user studies, which are currently absent.
Computational Cost: The paper claims the method does not increase point cloud size, but it is silent on the computational cost of computing the hierarchical semantic features and the motion-depth regularization during training. Given that OT can be expensive, a analysis of the training time overhead introduced by these novel components is crucial to assess the practical efficiency of F4DGS.
2. Vague Implementation Details:
The description of the "hierarchical semantic features" is high-level. Precisely how are the coarse, mid, and fine features extracted from CLIP and SAM? Are different layers of these models used? Are the features fused or used independently? This lack of detail makes replication difficult.
3. Vague Implementation Details:
The description of the "hierarchical semantic features" is high-level. The Pipeline (Fig 2) that explaining this process also lacks of detail,  makeing the concept difficult to understand.

### Questions
1. Does the motion-depth regularization primarily improve temporal smoothness (as measured by a metric like warping error) or static-frame quality (PSNR)? Does the OT regularization mainly resolve semantic ambiguities or also improve sharpness? 
2. Precisely how are the coarse, mid, and fine features extracted from CLIP and SAM? Are different layers of these models used? Are the features fused or used independently?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
1. This paper tackles dynamic novel view synthesis, aiming to achieve temporally consistent and physically realistic dynamic rendering within a 4D Gaussian representation.
2. Existing 3D Gaussian Splatting methods often struggle in dynamic scenes, showing temporal inconsistency, motion artifacts, and lighting or texture discontinuities.
3. The authors propose F4DGS, which introduces feature consistency regularization to jointly constrain semantic features, velocity, and depth over time. The semantic constraint ensures consistent appearance of the same object across frames； The velocity–depth constraint provides physical priors that suppress unnatural motion transitions.
4. The model achieves state-of-the-art results on the Plenoptic Video and D-NeRF datasets, with extensive experiments—especially on D-NeRF—supporting its effectiveness.

### Strengths
1. The method is technically sound and clearly motivated.
2. Experimental validation is thorough and convincing. Tab 3 is really clear.

### Weaknesses
1. The proposed feature consistency regularization mainly acts as a loss constraint. The contribution is somewhat limited.
2. No demo videos are provided, making it difficult to fully assess the claimed high-fidelity, physically plausible rendering.

### Questions
1. Please discuss how this work differs from FreeTimeGS [Wang et al., CVPR 2025], especially regarding temporal modeling and motion consistency.
2. Overall, the paper is clear, concise, and well-validated. If other reviewers agree on its contribution, I believe it deserves publication for discussion.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a series of regularization strategies designed to improve the realism and temporal consistency of 4D Deformable Gaussian Splatting-based scene reconstruction. Specifically, the authors propose three complementary components: a hierarchical semantic feature regularization, a temporal consistency regularization, and a motion–depth regularization. Together, these introduced training objectives aim to enforce both spatial and temporal coherence in dynamic scene reconstruction. Experiments on the Plenoptic Video and D-NeRF datasets demonstrate notable improvements in reconstruction quality, and ablation studies further validate the contribution of each proposed component.

### Strengths
- The proposed method incorporates semantic features into a reconstruction framework which is interesting.
- The experimental results are strong, demonstrating consistent improvements both qualitatively and quantitatively.
- The paper is well-structured, making it relatively easy to follow the main ideas.

### Weaknesses
- **Lack of baselines.** The paper only compares against works published up to 2023, despite several relevant methods appearing more recently (e.g., A1–A4).
- **Limited novelty.** The novelty of the work appears somewhat limited. The method primarily builds upon deformable 3D Gaussian Splatting and introduces a series of additional loss terms. However, the motivation behind the hierarchical OT-semantic regularization is not well justified, and the inclusion of a temporal consistency loss is fairly standard in sequential reconstruction. Overall, the motivation and methodological design feel somewhat fragmented and incremental.
- **Missing explanation.** The use of semantic features is not clearly explained. While the paper mentions CLIP features, it does not clarify how the coarse, mid, and fine features are constructed. A more detailed description of the feature hierarchy and its motivation would improve the paper’s clarity.
- **Poor writing quality.** The writing requires substantial revision. Several sentences are grammatically incorrect or difficult to follow (e.g., line 071). Improving the clarity and precision of the language would significantly enhance readability.

[A1] Yang, Zeyu, et al. "Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting." ICLR 2024.

[A2] Li, Zhan, et al. "Spacetime gaussian feature splatting for real-time dynamic view
synthesis." CVPR 2024.

[A3] Lee, Junoh, et al. " Fully explicit dynamic gaussian splatting." NeurIPS 2024.

[A4] Wang, Yifan, et al. "FreeTimeGS: Free Gaussian Primitives at Anytime and Anywhere
for Dynamic Scene Reconstruction." CVPR 2025.

### Questions
- Include additional baselines to better demonstrate the effectiveness of the proposed method.
- Provide a clearer discussion of the motivation behind each proposed regularization term and explain why they complement each other.

### Soundness
2

### Presentation
1

### Contribution
2
