# GRASP-GS: Geometric Registration and Dual-Stag Saliency Pruning for Efficient 3D Gaussian Splatting

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
3D Gaussian Splatting (3DGS) has emerged as a powerful paradigm for scene representation, enabling real-time rendering with high visual fidelity by modeling scenes as anisotropic 3D Gaussians. However, existing methods suffer from blurred reconstructions, redundant Gaussians, and high training costs due to sparse Structure-from-Motion (SfM) initialization and heuristic densification. In this paper, we propose GRASP-GS (Geometric Registration and DuAl-Stage Saliency Pruning for 3D Gaussian Splatting), a framework that integrates geometric prior-guided initialization with adaptive saliency pruning. Our method first enhances the initial point cloud by extracting and fusing dense multi-view features with the SfM points through multi-stage refinement. Then, a dual-stage pruning strategy sequentially applies KL-based Rendering Survival Pruning (KL-RSP) to reduce spatial redundancy and Opacity-based Density-Constrained Pruning (ODCP) to eliminate low-contribution Gaussians. Experiments demonstrate that GRASP-GS achieves compact and high-quality scene representations, enabling efficient real-time rendering with enhanced structural integrity and visual quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an improvement over the classic 3DGS method, achieving better rendering results with faster training and fewer primitives. The main enhancements focus on two aspects: initialization and pruning strategy. The initialization leverages geometric estimation methods like MoGe, while the pruning incorporates two complementary approaches, clustering and opacity-aware pruning.

### Strengths
The paper demonstrates a complete structure, clear exposition, and rigorous experiments, with the proposed method proving to be effective. In the ablation studies, it further shows that the improvements brought by the two introduced modules are not independent but function jointly as an integrated whole, which underscores the value of the approach.

### Weaknesses
- Improving 3DGS is extensively studied, with better initialization or pruning strategies being widely explored. Although the paper demonstrates the synergy of the two proposed modules, showing some novelty, the overall contribution still appears to be incremental. If the underlying principles explaining why these two methods synergize and achieve better results were shown, the paper's innovativeness could be further enhanced.
- The paper claims the proposed method is more efficient, but only compares training time and the final number of primitives. Reporting the additional cost introduced by the geometric prior-guided initialization module would make the argument more convincing.

### Questions
Perhaps I misunderstood, but based on the description in the method section, the two pruning strategies are executed sequentially, whereas they are depicted as a loop in the main diagram. This is somewhat confusing.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper “GRASP-GS: Geometric Registration and Dual-Stage Saliency Pruning for Efficient 3D Gaussian Splatting” proposes a new framework to improve the efficiency and quality of 3D Gaussian Splatting (3DGS) for real-time 3D scene rendering. GRASP-GS combines geometric prior-guided initialization, using dense multi-view features aligned with Structure-from-Motion data, with dual-stage saliency pruning that removes redundant Gaussians via KL-based rendering survival and opacity-based density pruning. This integrated approach enhances geometric completeness, reduces redundancy, and accelerates training. Experiments demonstrate that GRASP-GS achieves compact and high-quality scene representations, enabling efficient real-time rendering with enhanced structural integrity and visual quality.

### Strengths
1. The paper clearly identifies three major issues in existing 3D Gaussian Splatting pipelines — blurred reconstruction, redundant Gaussians, and high training cost — and proposes a coherent framework (GRASP-GS) explicitly designed to address them.

2. GRASP-GS is designed as a modular enhancement that can be easily integrated into existing 3DGS frameworks (e.g., 3DGS, AbsGS).

### Weaknesses
1. Could you please follow the submission format? The citation format seems incorrect, which makes the paper a bit difficult to read.

2. The method constructs its dense geometric prior by generating independent single-view point clouds (via MoGe) and later aligning and merging them with sparse SfM points. However, each monocular point cloud is often affine-invariant and lacks global metric consistency—their absolute scales and geometric structures will differ across views. Although the authors mitigate this through geometric registration and filtering. But recent works (e.g., InstantSplat, Intern-GS, MASt3R-GS) have demonstrated that DUSt3R- or MASt3R-based multi-view dense priors can provide more coherent and metrically consistent initialization, directly improving Gaussian distribution. Could the authors clarify the motivation for relying on single-view monocular priors rather than leveraging SOTA multi-view reconstruction methods, or provide some comparative experiments to prove?

3. The experiments are weak, as the paper highlights two main contributions — geometry-based initialization and efficiency through dual-stage pruning — but the experimental comparisons do not adequately cover representative methods from either category. In other word, it does not include initialization-oriented approaches that enhance 3DGS using geometric or learned priors, also omits recent density-control and compression methods, only compare with 2023–2024 baselines is not convincing.

### Questions
Please refer to the Weaknesses above.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes GRASP-GS, an integrated framework for 3D Gaussian Splatting (3DGS) that (i) improves initialization using dense, affine-invariant monocular geometry (MoGe) aligned with SfM via geometric–chromatic consistency checks and multi-stage refinement, and (ii) enforces dual-stage saliency pruning to keep only perceptually important Gaussians. The pruning combines KL-based Rendering Survival Pruning (KL-RSP) using union-find clustering on KL-similar splats and Opacity-based Density-Constrained Pruning (ODCP) that removes low-opacity splats in dense neighborhoods. Across Mip-NeRF360, Deep Blending, and Tanks&Temples, the method reports higher PSNR/SSIM, lower LPIPS, and notably fewer Gaussians and shorter training time than 3DGS and AbsGS baselines (e.g., ~35.6% fewer Gaussians and ~23% faster vs 3DGS).

### Strengths
1. Well-motivated integration. The paper clearly articulates the synergy between better geometric initialization and adaptive pruning, rather than treating them as separate knobs. 
2. Solid empirical gains & analysis. Consistent improvements across datasets and both 3DGS/AbsGS backbones; thorough ablations show the contribution of registration and pruning separately and together. The paper also studies an opacity-reset strategy and argues against it.

### Weaknesses
1. Dependence on MoGe and added overhead. While robust, MoGe inference and the subsequent registration (RANSAC + Kabsch–Umeyama + ICP) add non-trivial preprocessing cost not reflected in the reported training time; the wall-clock overhead should be quantified to fully support “efficiency” claims. 

2. Missing memory/runtime profiling during rendering. Results emphasize PSNR/SSIM/LPIPS, Gaussian counts, and training time, but do not provide GPU memory usage, real-time FPS, or pruning overhead at inference to substantiate “compact & real-time” claims.

### Questions
1. Training schedule fairness. Were baselines like Pixel-GS or AbsGS also run with your modified densification/split thresholds? If not, how do their metrics change when given the same schedule?  

2. Wall-clock efficiency. What is the end-to-end time including MoGe inference and registration per scene? How does that compare to baseline pipelines when measured from images to trained model?

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
5

### Summary
The paper proposes GRASP-GS to improve 3D Gaussian Splatting (3DGS) through two components: (1) geometric prior-guided registration that refines the initialization from sparse Structure-from-Motion (SfM), and (2) dual-stage saliency pruning, including KL-based Rendering Survival Pruning and Opacity-based Density-Constrained Pruning, to remove redundant Gaussians. Integrated into standard 3DGS or AbsGS frameworks, GRASP-GS achieves higher PSNR and SSIM, lower LPIPS, about 23% faster training, and 35% fewer Gaussians on Mip-NeRF360, Tanks & Temples, and Deep Blending.

### Strengths
(1) The framework is intuitive. The geometric registration module, based on geometric foundation models, effectively improves the initialization of sparse Structure-from-Motion (SfM). In addition, to prevent the introduction of redundant Gaussian points, the paper adopts a two-stage pruning strategy that reduces redundancy, speeds up training, and enhances rendering quality.

(2) Extensive experiments are conducted on multiple datasets. The proposed method is compatible with existing 3DGS approaches and demonstrates strong practical effectiveness.

### Weaknesses
(1) The geometric prior-guided registration leverages geometric foundation models to mitigate the limitations of sparse Structure-from-Motion (SfM), which is particularly effective for sparse-view inputs. It would be interesting to evaluate whether the proposed method can further improve performance under extremely sparse-view settings.

(2) More qualitative comparisons are needed to clearly demonstrate the advantages of using fewer Gaussians. In several visual results, GRASP-GS does not show a large difference from the baseline. Since the paper employs geometric prior-guided registration, it would be beneficial to include depth or geometry visualizations to verify that the geometric foundation model leads to improved geometry.

(3) The training is reported on an A800 GPU. Providing peak GPU memory usage would help quantify computational overhead more precisely, especially since vanilla 3DGS can be trained on consumer GPUs such as the RTX 3090.

Minor Weaknesses:

(1) The paper consistently uses \cite for references; it is better to use \citep for proper citation formatting.

(2) A related work reference is missing: Improving Gaussian Splatting with Localized Points Management (CVPR 2025).

### Questions
(1) In Table 3, GRASP-GS demonstrates clear effectiveness. However, it would be more informative to show the results of using only the registration module (e.g., MoGe-based initialization) while keeping the second pruning stage, to better isolate each component’s contribution.

(2) Could more advanced geometric foundation models, such as VGGT, further improve the performance of GRASP-GS?

### Soundness
3

### Presentation
3

### Contribution
3
