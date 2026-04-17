# SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Reconstructing 3D scenes from sparse images remains a challenging task due to the difficulty of recovering accurate geometry and texture without optimization.
Recent approaches leverage generalizable models to generate 3D scenes using 3D Gaussian Splatting (3DGS) primitive. 
However, they often fail to produce continuous surfaces and instead yield discrete, color-biased point clouds that appear plausible at normal resolution but reveal severe artifacts under close-up views. 
To address this issue, we present SurfSplat, a feedforward framework based on 2D Gaussian Splatting (2DGS) primitive, which provides stronger anisotropy and higher geometric precision. By incorporating a surface continuity prior and a forced alpha blending strategy, SurfSplat reconstructs coherent geometry together with faithful textures.
Furthermore, we introduce High-Resolution Rendering Consistency (HRRC), a new evaluation metric designed to evaluate high-resolution reconstruction quality. Extensive experiments on RealEstate10K, DL3DV, and ScanNet demonstrate that SurfSplat consistently outperforms prior methods on both standard metrics and HRRC, establishing a robust solution for high-fidelity 3D reconstruction from sparse inputs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
SurfSplat is a novel method for feed-forward scene reconstruction from sparse posed-input images. Unlike previous works that employ 3D Gaussian Splats (GS) reconstructions, SurfSplat proposes a 2DGS approach. This method generates a 2DGS per pixel and incorporates two targeted additions to achieve this. The first one aligns the normals based on the local neighborhood in screen space, ensuring Surface Continuity. The second employs an enforced alpha blending technique that enhances the contribution of occluded Gaussians in the final result. This combination leads to a cleaner reconstruction where the occluded Gaussians remain optimized and contribute to the overall reconstruction quality.
For the reconstruction process, the authors extract features using two ViTs: one for single views and the other for aggregated multi-view information. These concatenated features are then transformed into 2DGS parameters using a 2D U-Net. Additionally, the authors propose a novel metric called HRRC, which renders the scene in higher resolution and compares it with upscaled ground truth images. This comparison highlights any inconsistencies or holes in the reconstructed surfaces.

### Strengths
- The additions to extend feed-forward-based reconstruction methods to 2DGS are well-motivated and explained.
- Overall, the paper is easy to read and follow.
- Utilizing screen-space information to construct a local tangent frame for continuous alignment of the 2DGS based on its neighborhood is a sound strategy. The implementation is also well explained.
- The forced alpha blending is particularly interesting in the feed-forward reconstruction scenario, where each pixel generates a GS that overparametrizes the scene.
- The proposed HRRC metric is intuitively comprehensible and effectively identifies critical issues like holes in the reconstruction.

### Weaknesses
- Although the HRRC metric is intuitively comprehensible, it would be beneficial to also demonstrate the surface normals from the 2DGS. This would not only highlight the smoothness but also serve as an indicator of the Surface Continuity Prior’s effectiveness.
- Furthermore, it would be beneficial to run the method on synthetic scenes and report F1 scores or normal alignment metrics. These metrics would provide insights into the accuracy of the geometry prediction.

### Questions
- Given that 2DGS are frequently easier to mesh, how do the scenes appear after meshing?
- Does the surface continuity prior use a technique to disable or weaken it for discontinuities? How are the 2DGS behaving in this areas?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets the common failure mode of feedforward 3D reconstruction from sparse-view inputs, where existing approaches (e.g., 3DGS-based pipelines) tend to produce discontinuous, fragmented surfaces that exhibit conspicuous artifacts under close-up inspection.

Core contributions:
1、SurfSplat framework. The authors advocate using 2D Gaussian primitives (2DGS) as surface-oriented rendering elements better suited to representing continuous manifolds.
2、Surface Continuity Prior (SCP). The key innovation. Instead of letting the network blindly regress each Gaussian’s rotation and scale, SCP explicitly derives orientation and anisotropic extent from local surface normals computed on neighborhood 3D points. This imposes a geometric prior that enforces local surface continuity and smoothness of the reconstructed geometry.
3、Forced Alpha Blending (FAB). A training strategy that caps per-primitive opacity, preventing foreground splats from saturating alpha and completely occluding background layers. By ensuring gradient flow to deeper/occluded structures, FAB mitigates training instabilities and geometric misalignment.
4、HRRC metric. A new evaluation metric—High-Resolution Rendering Consistency (HRRC)—that reveals holes, cracks, and other geometric defects which standard image-space metrics often miss, by evaluating renderings at higher magnification.

Summary. By combining a strong geometric prior (SCP) with a stabilization strategy for compositing (FAB), the paper substantially improves the geometric fidelity and surface continuity of feedforward reconstruction models, and demonstrates the resulting advantages using the proposed HRRC metric.

### Strengths
1、High Coherence between Representation Choice and Prior (2DGS + SCP): The paper appropriately pivots to 2DGS as fundamental "surfel" elements to address the degradation issues of 3DGS under sparse views. Its core contribution lies in using the Surface Continuity Prior (SCP) to explicitly derive the rotation and scale of each primitive from local geometry, shifting the paradigm from "blind learning" to "geometrically-constrained learning." This effectively enhances the continuity and realism of the generated surfaces.

2、Proposal of a Critical and Practical Evaluation Metric (HRRC): The paper contributes the HRRC metric, which simulates a "close-up inspection" by rendering at higher resolutions. This method effectively exposes hidden geometric artifacts (e.g., holes and discontinuities) that standard metrics fail to capture. This provides strong support for the paper's claims and offers a valuable new tool for the community to assess the quality of sparse-view reconstruction.

3、Pragmatic Integration of a Dual-Branch Encoder (Monocular Prior + Multi-View Geometry): The architecture is pragmatically designed, leveraging the strong monocular prior from Depth Anything V2 and the multi-view consistency from plane-sweep cost volumes. This clear division of responsibility—monocular for completing missing textures and multi-view for ensuring consistency—places the model on a level playing field with current SOTA methods (e.g., MVSplat/DepthSplat series), ensuring a fair and effective comparison.

4、Clarity in Writing and Structure: The paper is well-structured and clearly written, with a logical flow that is easy to follow. The authors articulate the problem, the proposed solution, and its motivation, allowing readers to quickly grasp the core contributions and technical details, which is a significant merit for a high-impact publication.

### Weaknesses
1、Innovation Granularity Focused on Regularization and Priors; Lacks Theoretical Depth: The architecture (2DGS, dual-branch encoder) largely follows recent work. The core novelty is concentrated in the SCP+FAB+HRRC components. While the engineering results are impressive, the theoretical analysis of the SCP prior remains empirical (e.g., regarding its boundaries for curvature, texture frequency, or noise). Furthermore, a systematic comparison against alternative designs, such as "direct regression + regularization" (e.g., curvature smoothing, normal consistency loss), is notably absent.

### Questions
1、Potential Bias in the Proposed HRRC Metric: The HRRC metric compares the high-resolution rendered output against a ground truth image upsampled via bicubic interpolation. Since bicubic interpolation is inherently a smoothing filter that blurs fine textures and sharp edges, could this introduce a bias? Specifically, might this metric favor models that produce smoother, slightly blurred results (closer to the interpolated GT) over models that generate sharp but slightly imperfect details (which might be closer to the true high-frequency information)?

2、Weak Theoretical Support for the Forced Alpha Blending (FAB) Strategy: While FAB is shown to be effective, it essentially functions as a direct "engineering trick." The strategy of hard-clipping opacity and normalizing the final color lacks sufficient theoretical justification. This approach may introduce new rendering biases, particularly when handling semi-transparent objects or complex lighting, raising concerns about its robustness.

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
This paper introduces SurfSplat, a feedforward framework that replaces 3D Gaussian splatting (3DGS) with 2D Gaussian splatting (2DGS) to improve geometry reconstruction and texture fidelity in sparse-view 3D scene reconstruction. The key ideas include a surface-continuity prior linking splat shape and orientation to surface normals, and a forced-alpha blending strategy to prevent opacity collapse. The paper also proposes High-Resolution Rendering Consistency (HRRC) as a new metric to evaluate fine-grained artifacts. Experiments on RealEstate10K, DL3DV, and ScanNet show consistent improvements over baseline methods.

### Strengths
1. Solid methodological design:
Using 2D splats (surfels) instead of 3D Gaussians is an interesting design choice that simplifies the rendering pipeline while retaining expressive power. The authors carefully justify why this helps preserve anisotropy and continuity.
2. Comprehensive experiments:
Results on multiple datasets (RealEstate10K, DL3DV, ScanNet) demonstrate robustness and consistent improvements across settings. Qualitative visualizations effectively support the claims.

### Weaknesses
1. Limited theoretical grounding for the continuity prior:
While the prior is intuitive, its mathematical justification remains somewhat heuristic. The authors could discuss more explicitly how the coupling of rotation and scale influences convergence and stability.
2. Ablation analysis could be deeper:
The ablations are primarily qualitative. Quantitative ablations showing how much each component (surface prior, alpha blending, HRRC) contributes to the final performance would strengthen the empirical evidence.

### Questions
1. Theoretical Motivation of the Surface-Continuity Prior

Could the authors clarify the theoretical reasoning for coupling the splat’s rotation and scale with local surface orientation? Is this prior based on any assumption of local planarity or normal consistency, and how sensitive is the approach to errors in predicted normals?

2. Quantitative Validation and Ablation
    
Beyond qualitative comparisons, could the authors provide quantitative ablations that isolate the contributions of the surface-continuity prior and forced-alpha blending? For example, what are the PSNR / SSIM / HRRC changes when each component is removed?

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
This paper presents a feedforward 3D reconstruction framework from sparse views based on 2D Gaussian Splatting (2DGS), as an alternative to the more common 3D Gaussian Splatting (3DGS) frameworks. Existing feedforward approaches, such as PixelSplat and MVSplat, which use 3D Gaussians as primitives, often suffer from discontinuities (holes) in the reconstructed surfaces and color-biased artifacts when given sparse-view inputs. To address these issues, this paper introduces a surface continuity regularization that enforces smoothness among Gaussians within a local 3D neighborhood. Additionally, a forced alpha blending strategy is proposed to mitigate opacity saturation during rendering. To further highlight the shortcomings of current evaluation protocols, the paper also introduces a new metric, High-Resolution Rendering Consistency (HRRC), designed to better assess geometry fidelity. Experiments on RealEstate10K, DL3DV, and ScanNet demonstrate that the proposed method achieves superior geometric and visual fidelity compared to recent feedforward baselines, especially under the HRRC evaluation.

### Strengths
* Although the overall architecture of predicting per-pixel gaussian is not new, the proposed strategy of enforcing surface smoothness and forced alpha blending is relatively new and interesting. 

* The motivation of the paper is clear, and the paper is well-written and the proposed method is easy to follow and the overall presentation is clear. 

* The paper is solving a relevant problem. Reconstructing a high fidelity surface from sparse input is a relevant problem.

* Comprehensive evaluation is done on various datasets to show the effectiveness of the method.

### Weaknesses
* Why use a single Gaussian per pixel? While the method performs well against baselines that also use a single Gaussian per pixel on the novel view synthesis task (both within and cross-dataset), this design seems to limit performance compared to methods like PixelSplat and HiSplat, which use multiple Gaussians per pixel. Did the authors experiment with predicting multiple Gaussians per pixel? From Tables 1 and 2, it appears that doing so might address the observed issues more effectively than relying solely on the proposed surface continuity and forced alpha blending techniques.


* HRRC may favor the proposed regularization strategies. While HRRC appears to be an effective metric for evaluating feedforward GS-based reconstruction frameworks, it might inherently bias the evaluation toward smoother reconstructions due to the use of upsampled ground-truth images (via bicubic interpolation). Have the authors considered training the model on low-resolution input images and then evaluating on the original ground-truth resolution? Would such an evaluation be valid or provide a fairer measure of reconstruction fidelity? A small experiment with this setting will be helpful.

### Questions
Please refer to weakness.

Minor issues: 
* Line 193: Insufficien -> Insufficient 

* Line 162: the notation of camera intrinsics and poses is not clear.

### Soundness
3

### Presentation
3

### Contribution
3
