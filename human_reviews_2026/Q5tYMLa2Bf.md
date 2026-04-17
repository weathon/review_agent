# CAMO: Category-Agnostic 3D Motion Transfer from Monocular 2D Videos

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Motion transfer from 2D videos to 3D assets is a challenging problem, due to inherent pose ambiguities and diverse object shapes, often requiring category-specific parametric templates.  We propose CAMO, a category-agnostic framework that transfers motion to diverse target meshes directly from monocular 2D videos without relying on predefined templates or explicit 3D supervision. The core of CAMO is a morphology-parameterized articulated 3D Gaussian splatting model combined with dense semantic correspondences to jointly adapt shape and pose through optimization. This approach effectively alleviates shape-pose ambiguities, enabling visually faithful motion transfer for diverse categories. Experimental results demonstrate superior motion accuracy, efficiency, and visual coherence compared to existing methods, significantly advancing motion transfer in varied object categories and casual video scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CAMO, a category-agnostic 3D motion transfer framework that maps motions from a monocular 2D video onto arbitrary 3D targets—without using category templates or reconstructing source 3D meshes. CAMO represents the target as an articulated 3D Gaussian Splatting model driven by an LBS-based kinematic chain. It learns morphology-adaptive parameters (bone lengths, scale, offsets) to handle shape differences and uses dense 2D–3D semantic correspondences to reduce pose ambiguity. The total loss combines differentiable rendering, keypoint, and regularization terms. Experiments on DT4D and Mixamo show state-of-the-art performance in motion accuracy and visual realism, outperforming template-based and reconstruction-to-retarget baselines, while maintaining fast optimization (<10 min per sequence).

### Strengths
1. Category-agnostic design: Avoids SMPL/SMAL or per-class priors; directly optimizes on the target with only monocular source supervision. This reduces error cascade from reconstruct→retarget pipelines.
2. Clarity: The paper is well-presented, easy to follow.
3. Clear empirical gains: SOTA PMD/FID across datasets; ablations isolate the contributions of rendering loss, shape param., and keypoints; efficient optimization on commodity hardware.

### Weaknesses
1. While image-space supervision is attractive, robustness under heavy occlusion or fast motion isn’t systematically quantified; correspondence quality in such regimes is unclear.
2. Real-world setup relies on a render-and-compare camera initialization. The method’s sensitivity to poor initial camera guesses or to calibration drift is not analyzed.
3. For experiments, the authors should provide more visualization results. The demo's examples can not convince me for robust results.
4. About comparison, I'm curious about the comparison with a simple pipeline: using video generation/editing model to get edited object, and then apply dynamic 3D generation model for animation. Can CAMO outperform it in all evaluation aspects?

### Questions
Please see weakness. If all my concerns are well conducted, I'll consider raise my score.

### Soundness
2

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
3

### Summary
The paper proposes CAMO, a template-free, category-agnostic 2D→3D motion transfer method. It avoids reconstruct-then-retarget pipelines by optimizing the target asset directly in image space using (i) an articulated 3D Gaussian representation with morphology-adaptive parameters (bone lengths, global scale, local Gaussian offsets) and (ii) dense 2D to 3D semantic correspondences for disambiguation. Results show lower PMD/FID vs composite baselines (SPT+, NPR+) and Transfer4D across Mixamo & DT4D, plus qualitative real-world demos.

### Strengths
Sound and principled design.
The optimization objective (photometric + SSIM + semantic + temporal regularization) is coherent and mathematically well-founded.
The choice of directly optimizing in image space with an articulated 3D Gaussian structure is both elegant and technically solid.

Clear handling of 3D lifting.
The method performs implicit 3D lifting by analysis-by-synthesis, guided by differentiable rendering and dense 2D to 3D correspondences.
This eliminates explicit 3D supervision and is novel for the community.

Thorough ablations and visualizations.
The paper includes extensive quantitative and qualitative analyses (Tables 1–6, Figs. 5–14) demonstrating the necessity of morphology parameters and semantic losses. Failure cases and challenging cases are all presented and analyzed well.

### Weaknesses
1. Dependence on rigging quality and pre-processing.
The approach assumes access to well-rigged target meshes. Auto-rigging tools (e.g., UniRig, MagicArticulate) introduce noticeable artifacts when bone topology mismatches occur. Is there any way to address this?

2. Temporal scalability and long-sequence degradation.
The time-conditioned MLP cannot effectively model long (>600 frame) motion sequences; it gradually drifts or repeats poses. It's better to give some visualised results.

3. Absence of physical realism metrics.
Evaluation focuses on FID and PMD, but omits motion stability or contact-based metrics (e.g., foot-skating, interpenetration). Without such analysis, claims of realistic motion transfer remain visually but not physically validated.

4. Computation and convergence analysis are limited.
Although the authors report that optimization takes <10 minutes per sequence on an RTX 4090, there is no detailed runtime or convergence study across different mesh complexities or sequence lengths.

### Questions
CAMO delivers a robust, well-implemented, and innovative approach to category-agnostic 2D to 3D motion transfer.
The technical contribution and experimental validation are solid and can be a good contribution to the community.
The paper meets ICLR’s bar for novelty and soundness and is likely to stimulate follow-up work in differentiable 3D motion learning.
I strongly suggest that the authors open-source the source for reproducibility.

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
4

### Summary
This paper presents a category agnostic method for transferring motion from a 2d monocular video to a 3d target mesh. the method optimizes the pose of a target 3D model, represented by an articulated 3dgs framework, directly in the 2d observation space. The core components are a morphology-adaptive shape parameterization，which includes learnable bone lengths, global scale, and local offsets，and a dense semantic correspondence loss. This correspondence, derived from a pretrained feature extractor, aligns the 3d target with the 2d source.

### Strengths
1. The paper is well-written, structured, and easy to comprehend.
2. The quality of the results look good.
3. The methods handle category agnostic motion transfer without parametric models like SMPL and SMAL.
4. The use of articulated 3DGS for differentiable rendering , the leveraging of foundation models for robust semantic understanding , and the carefully designed morphology parameterization come together to form a reasonable and coherent framework.

### Weaknesses
1. Motion is parameterized by an MLP conditioned on a sinusoidal time embedding (Appendix A.2). This architecture is known to struggle with representing very long, complex, or non-cyclic motions, which is confirmed by the paper's own analysis (Appendix A.4) showing degradation on longer sequences.
2. The method is still vulnerable to fundamental 2D-to-3D ambiguities. The paper's failure cases (Fig. 14) show it can confuse left/right limbs or fail during severe self-occlusion, problems that persist despite the semantic correspondence loss.
3. Missing references, there are some works with similar settings:
[1] PhysRig: Differentiable Physics-Based Skinning and Rigging Framework for Realistic Articulated Object Modeling
[2] Puppeteer: Rig and Animate Your 3D Models

### Questions
1. Have you analyzed when these ambiguities are most likely to occur? For instance, does the left-right confusion happen most often from specific camera angles where the 2D projection is maximally ambiguous?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CAMO, a category-agnostic framework for transferring articulated 3D motion directly from monocular 2D videos to arbitrary 3D target meshes, without relying on category-specific templates or explicit 3D supervision.
The method builds upon articulated 3D Gaussian Splatting (3DGS) and introduces:

Morphology-adaptive parameterization (learnable bone lengths, local Gaussian offsets, and global scaling).

Dense 2D–3D semantic correspondence based on foundation features.

Joint optimization of pose and shape with photometric and semantic losses.

Experiments on Mixamo, DeformingThings4D, and real-world videos show clear improvements over baselines like SPT+, NPR+, and Transfer4D.

### Strengths
Category-agnostic capability.
The paper convincingly demonstrates that CAMO generalizes to both humanoids and non-humanoid animals, addressing the typical limitation of category-specific template models.

Clear ablations and metrics.
The quantitative improvements on PMD and FID are substantial (up to 85% improvement on non-quadruped categories). The ablation study effectively supports the importance of morphology parameterization and semantic keypoint supervision.

### Weaknesses
**Evaluation is largely self-contained and lacks cross-domain tests.**
While the results on Mixamo and DT4D are consistent, these datasets are synthetic and often aligned in topology.
Real-world results (Fig. 6) are qualitative only, without any perceptual or human study evaluation.
There’s no analysis of failure cases (e.g., severe occlusions, topology mismatch, or multi-actor scenes).

**Limited discussion on theoretical implications and generalization.**
The “morphology-parameterized” model is empirically motivated, but the paper does not analyze why this representation improves optimization stability or disentangles shape/pose effectively.

**No discussion on identifiability or optimization convergence issues under only 2D supervision.**

**Writing and clarity issues.**
The text contains numerous redundancies and long sentences; several sections (e.g., Sec. 3.1–3.3) are nearly copied from AnyMo.
Figures could be more informative — e.g., Fig. 2 is conceptually overloaded but lacks a clear depiction of data flow or loss supervision points.

### Questions
How robust is CAMO under non-articulated or highly deformable motions (e.g., cloth, jellyfish, smoke-like motions)?

How is the orientation-sensitive feature extractor trained or selected? Is it frozen or fine-tuned during optimization?

Does the method handle camera motion explicitly, or does it assume static background and known intrinsic/extrinsic parameters?

How does the method scale to longer sequences (e.g., 1K+ frames)? Is optimization stable or prone to drift?

### Soundness
3

### Presentation
3

### Contribution
2
