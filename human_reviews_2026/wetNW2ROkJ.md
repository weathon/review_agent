# GeoSplat: A Deep Dive into Geometry-Constrained Gaussian Splatting

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
A few recent works explored incorporating geometric priors to regularize the optimization of Gaussian splatting, further improving its performance. However, those early studies mainly focused on the use of low-order geometric priors (e.g., normal vector), and they might also be unreliably estimated by noise-sensitive methods, like local principal component analysis. To address their limitations, we first present GeoSplat, a general geometry-constrained optimization framework that exploits both first-order and second-order geometric quantities to improve the entire training pipeline of Gaussian splatting, including Gaussian initialization, gradient update, and densification. As an example, we initialize the scales of 3D Gaussian primitives in terms of principal curvatures, leading to a better coverage of the object surface than random initialization. Secondly, based on certain geometric structures (e.g., local manifold), we introduce efficient and noise-robust estimation methods that provide dynamic geometric priors for our framework. We conduct extensive experiments on multiple datasets for novel view synthesis, showing that our framework: GeoSplat, significantly improves the performance of Gaussian splatting and outperforms previous baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper extends the 3D Gaussian splatting to constraints Gaussian primitives to reside on smooth manifolds (or varifold), leveraging tangent and curvature priors estimated directly from the Gaussian field. Specifically they introduce curvature-guided initialization, truncated gradient updates on tangent planes, and shape-consistent regularization to maintain geometric stability during training. Experiments shows that GeoSplat achieved imrpoved rendering quality.

### Strengths
By formulating Gaussian primitives as samples on continuous surface manifolds, the proposed method introduces a principled geometric framework that goes beyond purely relying on photometric optimization.

The motivation and integration of curvature-guided initialization, and shape regularization, truncated gradient updates are more than acceptable.

The extension to a varifold-based formulation is also a strong point, as it generalizes the approach to handle non-smooth or noisy regions where local manifold assumptions may not hold.

### Weaknesses
While the idea of introducing curvature-guided priors is quite interesting, I’m not fully (but certainly partially) convinced it actually brings substantial benefits beyond enforcing smoothness after reading the paper. The major concerns includes followings: 

**W1.** The method seems to have a possibility that bias the reconstruction toward overly smooth surfaces, which could suppress fine geometric details or high-curvature regions.  
This concern is raised from the fact that the paper doesn’t include explicit geometry reconstruction metrics and only photometric ones like PSNR or SSIM are reported. It would be really helpful to convince benefits by showing geometry reconstruction performance on more diverse benchmarks such as **DTU** or **Mip-NeRF 360**, with quantitative metrics like Chamfer Distance or at least some visual ablations that show how the curvature priors affect geometry quality. Also, the experiments are limited to relatively clean indoor scenes (e.g., Replica, ICL-Office), which don’t stress the method’s generality.

**W2**. I'm concerned about the methods vulnerability to SfM initialization. If the initial SfM point cloud is sparse or noisy, the estimated geometric priors (especially the second-order curvatures which are notoriously sensitive to noise) might be unreliable. This could negatively impact all subsequent geometry-constrained steps, from initialization to densification, potentially leading to artifacts or incorrect surface regularization. The paper claims the estimation methods are noise-robust, but their effectiveness on poorly initialized SfM data is not demonstrated.

### Questions
I believe this paper raises important questions about geometric regularization in 3DGS and has significant potential. To better understand the method's contributions and limitations, I would like the authors to clearly address the following points:

**Q1.** Have you observed any cases where the curvature-guided regularization oversmooths fine geometric details or sharp edges? What is your perspective on this potential trade-off between smoothness and detail preservation, especially since the quantitative evaluation (W1) focuses on photometric metrics rather than geometric accuracy?

**Q2.** Following W1, have you evaluated GeoSplat on benchmarks with available ground-truth geometry (like DTU, ScanNet) or more complex, unbounded 360-degree scenes (like Mip-NeRF 360)? Providing quantitative geometric metrics (e.g., Chamfer Distance) on such datasets would be highly valuable to substantiate the claims of improved geometry beyond the indoor scenes presented.

**Q3.** Could you provide a more detailed breakdown of the computational overhead? Specifically, what is the additional training time or memory cost incurred by the dynamic geometric estimations and the geometry-constrained optimization steps compared to the vanilla 3DGS baseline?

**Q4.** How sensitive is GeoSplat to the quality of the initial SfM point cloud? Since the framework relies heavily on geometric priors (normals, curvatures) estimated from these points, how does performance degrade if the initial SfM data is sparse or noisy? This seems particularly critical for higher-order curvature estimates, which are often sensitive to noise.

### Soundness
2

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
This work improves 3DGS optimization from the prior of geometric theory. A set of initialization, regularization, and densification strategies are proposed. Specifically, the Gaussian scales and rotations are better initialized using the neighbor initial points. The training gradients are clip to move mainly on the estimated surface. The Gaussian shapes are regularized to be like 2D circles. The densified Gaussians is sampled more on the tangent plane instead of pure random. All proposed components are verified by ablation study and the overall rendering quality is better than previous work.

### Strengths
The proposed components can benefit all optimization-based 3DGS methods. They all grounded by geometric theory and are intuitively helpful. Another good thing is that they only take little extra training time. The improvement with fewer training images is very significant, which makes 3DGS useful even with few observation.

### Weaknesses
1. The proposed initialization and regularizations seems more fit with the 2DGS instead of 3DGS. The 2DGS itself can also cover the proposed shape regularization. However, comparison and discussion with 2DGS is missing.
2. The ablation for the proposed Gaussian normal estimation in Sedc.3.2 is missing. As it needs tens of seconds and is the most time consuming from the proposed methods, I expect the gain is good.
3. The evaluation is on 12 synthetic datasets. I wonder the effectiveness of the proposed methods in the real-world datasets. There are many real-world datasets, which are standard for NeRF/3DGS series of works to evaluate like mipnerf360.

### Questions
It would be nice if the ablation can also show how each of the proposed component improve the baseline 3DGS. This information is helpful for future work to see the individual improvement from 3DGS so they may can adopt a subset of them based on their need. The current ablation is conducted by removing one component out while keep the other new components activated.

### Soundness
2

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
4

### Summary
This paper proposes GeoSplat, a geometry-constrained optimization framework for 3D Gaussian Splatting (3DGS). Unlike prior works that use low-order geometric priors like normals, this paper's core idea is to leverage second-order geometric quantities, specifically principal curvatures. The authors state this high-order prior is systematically applied to the entire 3DGS training pipeline: (1) curvature-guided covariance initialization , (2) curvature-guided primitive upsampling , (3) shape-constrained optimization losses, and (4) curvature-regularized densification. To estimate these geometric quantities, the paper proposes two complex, noise-robust methods: one based on a local manifold assumption and another based on Varifold theory. Experiments on the Replica and ICL datasets show performance gains over baselines like 3DGS and GeoGaussian

### Strengths
1.The core idea of using second-order geometry (curvature) to guide the anisotropy of Gaussian ellipsoids is novel. Linking Gaussian scale s inversely to curvature is a strong and intuitive prior .
2.The contribution is systematic. The geometric prior is integrated into all three core stages of 3DGS—initialization, optimization, and densification—rather than being just an isolated loss term.
3.The low-resource experiments in Figure 1 are convincing. The results show that GeoSplat's performance degrades much slower than baselines as training views are reduced , validating that the explicit geometric prior provides robustness.
4.Table 3 provides a complete ablation study, demonstrating that each proposed component, such as covariance warm-up and shape regularization, contributes to the final performance.

### Weaknesses
1.Severely Limited Experimental Scope: This is the paper's primary weakness. All experiments are limited to only two indoor datasets, Replica and ICL . For a method claiming to be a general 3DGS framework, this is insufficient.
a.Missing Object-Level Datasets: The method is not evaluated on standard NeRF Blender datasets, such as lego and mic, which are standard for testing fine geometric details.
b.Missing Outdoor 360° Datasets: Critically, the paper completely omits evaluations on unbounded outdoor scenes like Mip-NeRF 360 or Tanks & Temples. A core point of the paper is reducing "floaters" and artifacts . Outdoor 360° scenes are the standard and most difficult testbed for this exact problem. Only showing improvements on indoor ceilings in Figure 2 is not convincing. This omission raises doubts about the method's robustness. Does the heavy reliance on "surface" manifold assumptions  cause the method to fail in unbounded scenes with non-surface elements like the sky or distant backgrounds?
c.Ignoring Non-Surface Objects: The method's assumptions are biased towards solid surfaces. It is unclear how it would perform on volumetric or transparent objects (e.g., hair, smoke), where it might introduce incorrect geometric priors.
2. Section 3.2 on Geometric Estimations is the technical core of the paper. However, the main text provides no intuition or high-level explanation. It only states theorems like Thm. 3.2 and complex final equations like Eq. 12, deferring all derivations to the appendix. This is poor presentation and makes it impossible for a reviewer to assess the method's technical contribution from the main paper.

### Questions
1.Why did the authors not experiment on object-level (Blender) and outdoor 360° (Mip-NeRF 360 / Tanks & Temples) datasets? Given the paper's claims about fixing floaters , omitting the 360° benchmark is a major oversight.
2.What is the computational overhead of the dynamic geometry estimation? The paper claims it is efficient, citing tens of seconds for million-level Gaussians. How many times must this be run during training? What is the total training time increase compared to 3DGS and GeoGaussian?
3.In Tables 1 & 2, the Manifold-based and Varifold-based methods perform very similarly . Given the added complexity of the Varifold method in Eq. 12 , is it necessary? Are there cases where it significantly outperforms the manifold approach?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents GeoSplat, a geometry-constrained optimization framework for Gaussian Splatting that explicitly incorporates higher-order geometric priors—normals and curvatures—throughout the full training pipeline based on the key ideas below.

- Curvature-guided initialization: Curvatures determine Gaussian covariance scales and orientations, producing better surface coverage for initialization.
- Shape-constrained optimization: Truncated gradient updates and curvature-aware regularization reduce floating or needle-like artifacts.
- Curvature-regularized densification: Curvature-weighted split/clone rules prevent outlier Gaussians in high-curvature areas.
- Noise-robust estimation: Proposed two complementary estimators (manifold-based Laplacian-geometry approach and a varifold-based geometric-measure-theory method) that supply dynamic and noise-robust priors during training.

Experiments on Replica and ICL datasets how consistent PSNR/SSIM/LPIPS gains and clear visual improvements, particularly under sparse-view (low-resource) conditions.

### Strengths
- Introduces a comprehensive curvature (higher-order geometry) use in Gaussian Splatting
- Formal derivations (Propositions 3.1 & Appendix D, E) connect manifold differential geometry to practical optimization steps.
- Consistent PSNR/SSIM/LPIPS gains and qualitative improvements, especially fewer floating Gaussians on sparse-view settings.

### Weaknesses
1. Additional comparison required: Recent works[1][2] curve-oriented Gaussian components are second-order primitives for Gaussian Splatting. A direct quantitative or qualitative comparison with these approaches is required to strengthen the novelty claim.
2. Accessibility of implementation: Heavy geometric derivations may limit reproducibility; a clearer pseudocode summary would help.
3. Runtime overhead: estimation frequency and cost are only qualitatively discussed (“tens of seconds per million Gaussians”)—quantitative runtime table would clarify trade-off.
4. Ablation coverage: while each module is tested, inter-dependencies (e.g., curvature estimator w/o truncated gradient) are not fully explored.

[1] Zhang, Ziyu, et al. "Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2025.

[2] Gao, Zhirui, et al. "Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2025.

### Questions
1. Recent works[1][2] also exploit higher-order geometric cues. How does GeoSplat differ from these in terms of formulation or estimation stability? A discussion or comparison table would help clarify the true extent of novelty.
2. Would the proposed method be generalized to outdoor scenes or textureless surfaces?
3. Additional ablation studies on hyperparameters (e.g., nearest neighbor $k$, thresholds $\xi_{\min}, \xi_{\max}$) would support the design choice of the proposed work.
4. Could curvature-guided initialization introduce bias when curvature estimates are noisy early in training?

[1] Zhang, Ziyu, et al. "Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2025.

[2] Gao, Zhirui, et al. "Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2025.

### Soundness
3

### Presentation
3

### Contribution
2
