# 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 10, 6, 6, 6

## Abstract
3D Gaussian Splatting (3DGS) achieves an appealing balance between rendering quality and efficiency, but relies on approximating 3D Gaussians as 2D projections—an assumption that degrades accuracy, especially under generic large field-of-view (FoV) cameras. 
Despite recent extensions, no prior work has simultaneously achieved both projective exactness and real-time efficiency for general cameras. We introduce 3DGEER, a geometrically exact and efficient Gaussian rendering framework. From first principles, we derive a closed-form expression for integrating Gaussian density along a ray, enabling precise forward rendering and differentiable optimization under arbitrary camera models. To retain efficiency, we propose the Particle Bounding Frustum (PBF), which provides tight ray–Gaussian association without BVH traversal, and the Bipolar Equiangular Projection (BEAP), which unifies FoV representations, accelerates association, and improves reconstruction quality. Experiments on both pinhole and fisheye datasets show that 3DGEER outperforms prior methods across all metrics, runs 5x faster than existing projective exact ray-based baselines, and generalizes to wider FoVs unseen during training—establishing a new state of the art in real-time radiance field rendering.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper investigates how Gaussians are projected within the 3D Gaussian Splatting framework and addresses the limitations of existing projection methods for general cameras, particularly those with large fields of view (FoVs) where standard approaches often fail. Specifically, it analyzes an approximation commonly used in the projection process and demonstrates that this approximation can negatively affect the accuracy of Gaussian projection under large FoVs.

To achieve general and accurate Gaussian projection, the paper makes three main contributions. First, it shows that the transmittance of a Gaussian depends solely on the distance between a ray and the Gaussian’s center. Second, it introduces a more precise yet computationally efficient formulation for determining Gaussian bounds. Third, it proposes Bipolar Equiangular Projection (BEAP), a novel technique for enhanced ray sampling that provides more uniform coverage and improves projection efficiency.

Through experiments on diverse datasets and camera configurations, the proposed method demonstrates notable improvements in representation quality and significantly reduces the number of Gaussians required. In contrast, prior methods relied on a much larger number of Gaussians to compensate for distortions in large FoVs.

### Strengths
**Clear and well-structured presentation**: The paper is well-written and easy to follow. It clearly articulates the problems addressed, the proposed solutions, and their significance, supported by clear and informative figures.

**Solid theoretical foundation**: The paper provides comprehensive mathematical formulations necessary to understand the proposed method, with additional derivations and supporting details presented in the appendix. It is also noteworthy that the paper explores alternative angular representations to overcome the limitations of conventional tan-based approaches (Appendix D.2).

**Strong quantitative and qualitative results**: The proposed method demonstrates consistent performance improvements across diverse datasets and camera types, including both fisheye and pinhole cameras.

### Weaknesses
The contribution of the transmittance equation (Eq. 5) to the overall method could be articulated more clearly. In particular, the benefits of this formulation for general camera models are not fully evident, and the paper could better explain how this component integrates with and enhances the proposed framework.

### Questions
1. Very minor, but in Fig. K.2, the “laboratory” and “office” scenes appear to be mixed.

### Soundness
4

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
4

### Summary
This paper presents a new 3D Gaussian renderer designed for large field-of-view cameras, such as fish-eye cameras. It derives a closed-form expression for integrating the Gaussian density along a ray. To enhance efficiency, the paper introduces the Particle Bounding Frustum (PBF), which provides a tight bounding box to cull unnecessary ray-Gaussian intersections. 

Unlike pinhole cameras, where rays are typically sampled uniformly across a 2D plane, this paper proposes sampling rays uniformly from a sphere using Equiangular Projection. This approach allows for interpolation of the rays to produce the output image, further improving quality. 

Experiments conducted on both pinhole and fish-eye datasets demonstrate that the 3D Gaussian Efficient Ray Renderer (3DGEER) outperforms previous methods across all metrics. Additionally, it runs five times faster than existing projective exact ray-based baselines and generalizes well to wider fields of view that were not seen during training.

### Strengths
1. **Strong results.** The experimental results are impressive, achieving state-of-the-art performance across many large field-of-view (FOV) datasets. Additionally, there are significant improvements in memory efficiency and rendering speed compared to 3DGRT and EVER.

2. **Good presentation.** The paper is well-structured and easy to read. The derivation of the formula is generally clear. However, some sections, such as the gradient flow in Figure 2, could be streamlined or moved to the appendix, as they do not seem essential for all readers.

### Weaknesses
**Limited discussion to existing work**.  The paper provides limited  discussion regarding existing work. While there are several derivations of the proposed techniques, including the closed-form expression for integrating Gaussian functions and the tight bounding box, these derivations have been adopted in prior research without clear reference in this paper. 

For the exact projective geometry, Equation 5 in this paper is identical to Equation 15 in HTGS [1].  Similarly, the equations for calculating the tight bounding box (Equations 7 through 10) appear to be equivalent to Equations 8 through 10 in both 2DGS [2] (with the substitutions x = tan(θ) and y = tan(φ)). Both of these derivations originate from source [3]. However, these papers are not explicitly cited, and a more thorough discussion regarding their connections should be included. (The only reference seems to be in Section 2.2 (lines 139-142), which is not entirely accurate since it does not use BVH. )

Furthermore, due to the similarities between these works, it is unclear how this new formula can eliminate the screen-space constraints present in HTGS and 2DGS that limit their field of view, as claimed in lines 285-286. This point requires additional justification.

[1] Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency. Eurographics 2025.
[2] 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. SIGGRAPH'24.
[3] A hardware architecture for surface splatting. ACM TOG 2007.

### Questions
See W1 in weakness.

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
The paper proposes, 3DGEER, a ray-based Gaussian rendering framework that achieves projective exactness with a closed-form integral of anisotropic Gaussians along rays via a canonical transform, eliminating projection linearization error. It introduces Particle Bounding Frustum (PBF) for exact, BVH-free ray–particle association and Bipolar Equiangular Projection (BEAP) for unified arbitrary-FoV sampling that accelerates association and improves quality. Experiments indicate that 3DGEER high FPS while preserving projective exactness across camera models, achieving consistent quality gains over prior exact-ray or splatting baselines.

### Strengths
- The paper proposes a projectively exact closed-form ray integral and an exact, BVH-free Particle Bounding Frustum (PBF) for ray–particle association, improving both accuracy and efficiency.
- The paper proposes BEAP, which uniformly samples rays in angular space, to improve generalization to large FoVs and improve efficiency and rendering quality.
- The method achieves SOTA performance compared to the baselines in both photometric metrics and FPS across camera models and generalizes to wider FoVs than those seen in training data

### Weaknesses
- Missing ablation for projective exactness: The paper does not include a controlled ablation that replaces the projective closed-form integral with 2D projective-space approximation (that is used in baselines) within the same pipeline. Such a study is crucial to learn how much improvement is brought specifically from the projective exact formulation (separate from PBF/BEAP)
- Missing ablation for BEAP: The paper omits a controlled BEAP on/off study within the same pipeline and comparisons to alternative projections (e.g., perspective, equidistant, equal-area). A clear ablation should report quality and efficiency with/without BEAP, stratified by image region (full FoV, center vs. edge/periphery), to identify how much quantitative improvement is brought by BEAP (separate from the other 2 contributions).
- In table 4, the runtime for some baselines are evaluated with different GPU models. It would be more clear if everything could be evaluated with a same GPU model.

### Questions
See weaknesses. I suggest to add the missing ablations to better understand how much improvement each contribution brings separately, and re-run some of the baselines on a consistent GPU for a coherent runtime comparison.

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
4

### Summary
This paper presents a theoretically grounded and practically efficient framework for 3D Gaussian rendering under arbitrary camera models. It derives an analytical ray–Gaussian integral by transforming anisotropic Gaussians into a canonical isotropic space, yielding an exact closed-form transmittance that unifies differentiable rendering and projective geometry without approximation. A key innovation is the Particle Bounding Frustum, which computes exact angular bounds from each Gaussian’s covariance to perform direct frustum–ray association, replacing BVH traversal while preserving geometric correctness. The method further introduces a Bipolar Equiangular Projection that provides uniform ray parameterization and improves stability across wide-FoV distortions. Experimental results demonstrate the significant performance improvement upon existing works.

### Strengths
1. Introduces an efficient PBF-based association strategy that achieves both geometric exactness and near real-time rendering, effectively bridging the gap between ray-tracing accuracy and splatting efficiency.
2. The proposed framework generalizes naturally across pinhole, fisheye, and omnidirectional cameras through its unified angular-space formulation and the Bipolar Equiangular Projection.
3. Demonstrates strong experimental performance and consistent generalization across diverse camera models, datasets, and fields of view, with results that are comprehensive and sufficiently support the paper’s claims.

### Weaknesses
1. An ablation study on the key components is missing.

2. A more detailed discussion of the limitations is needed, for example explaining how sensitive the framework is to ordering effects as discussed in related works.

### Questions
1. The paper reports achieving higher quality with far fewer Gaussians, but it is unclear whether this reduction stems from better ray–particle association, regularization effects, or implementation details.

2. How does the method scale to multi-million Gaussians in large scenes? Without BVH traversal, are there memory or parallelization bottlenecks at high particle counts?

3. On standard pinhole datasets, does rendering quality keep improving with more Gaussians,

4. Beyond numerical stability, what are the key algorithmic differences between 3DGEER and GOF in ray–Gaussian intersection and gradient computation?

### Soundness
3

### Presentation
3

### Contribution
3
