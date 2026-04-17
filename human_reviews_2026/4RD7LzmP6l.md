# AnimalGS: 4D Animal Reconstruction from Monocular video with 3D Gaussian Splatting

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Reconstructing 4D animals from monocular videos is challenging due to large inter-species variation, complex articulations, and the lack of reliable templates. 
We introduce AnimalGS, a test-time optimization framework built on a 3D Gaussian Splatting representation for high-fidelity 4D reconstructions from single videos. 
Grounded in the insight that robust reconstruction emerges from pose-guided optimization rather than strict shape priors, AnimalGS treats priors as coarse initializations and integrates joint-aware and symmetry-aware designs to progressively disentangle motion and appearance. This leads to empirically strong generalization across diverse species and robustness to mismatching with shape priors. 
Extensive experiments demonstrate the superior performance of our approach in geometry, motion, and temporal consistency across a wide variety of animal species.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents AnimalGS, a test-time optimization method for dynamic 3d animal reconstruction from a video, using 3DGS as the geometry representation.

It starts with a coarse shape from the 1st image by Fauna (Li et at 2024) and performs two-stage optimization: a pose refinement step using learnable joint anchors and symmetry augmentation, followed by pose-guided non-rigid deformation represented as Gaussian offsets.

It produces good 4D reconstructions across a variety of animal species, outperforming state-of-the-art methods such as GART, D-3DGS, and GVF-Diffusion.

### Strengths
- The method is technically solid and works better than the referenced existing works.
  - The use of symmetry prior by augmentation is interesting, although related ideas has been used in prior methods.
- Thorough evaluation that includes quantitative results and user studies.
  - The authors assembled a dataset of 87 monocular videos and reported both input and novel view metrics.
- The dynamic 3DGS representation allows fast rendering and interactive visualization.

### Weaknesses
- The paper has limited conceptual difference compared to existing literature. Test-time optimization, LBS-based deformation, and symmetry prior, and skinned 3DGS, are largely extensions of existing works. 
- The improvement from introduced components (e.g., Tab 3) are not that significant.
- The visual results are not great. The reconstruction suffer from blurriness and appearance artifacts.
- Missing baseline and related work: it seems DreamMesh4d has more appealing result and should be compared against.

[A] DreamMesh4D. https://arxiv.org/abs/2410.06756

### Questions
- It would help if the authors could highlight the major difference compared to existing works in writing, such as LASR and DreamMesh4d.
- Compared to symmetry prior, image-to-3d models provides a stronger 3d prior and could help disentangle shape and motion.
- For evaluation, I found the current NVS metrics not convincing as those are distribution-level. Multiview dataset such as "Artemis: Articulated Neural Pets with Appearance and Motion Synthesis" can be an alternative to measure novel view view synthesis scores.

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
This paper introduces a test-time optimization framework for generating time-variant 3D Gaussian Splatting (3DGS) representations of animals from a single monocular video. The method first leverages Fauna, an off-the-shelf model pretrained on large-scale animal data, to acquire a coarse 3D mesh reconstruction. This mesh then initializes a 3DGS field, which is subsequently refined through a joint-aware and symmetry-aware optimization process. The proposed method demonstrates state-of-the-art (SOTA) performance on established benchmarks.

### Strengths
1. The paper's methodology is logically structured, effectively leveraging animal-specific geometric priors to achieve high-quality reconstruction.
2. The proposed method proves effective, with experiments demonstrating strong performance in both seen-view and novel-view settings.
3. The manuscript is clearly written and easy to understand, supported by high-quality figures.

### Weaknesses
1. The method's reliance on a strong animal segmentation prior to remove background influence needs clarification. This dependency seems at odds with Figure 1, which depicts a complex scene with multiple similar animals (e.g., camels). The paper does not appear to detail how the method would distinguish or handle such challenging multi-entity scenarios, making the figure's inclusion potentially misleading.
2. The methodology requires further clarification on several key points:
•	The baseline Fauna model provides per-frame deformed meshes, which this method reportedly discards. Please clarify the rationale for omitting this information and explain its relationship (or lack thereof) to the modules proposed in this paper.
•	Line 198: The initialization of attributes for the 3D Gaussian representation is unclear. Please specify the method and any statistical distributions used (e.g., for random initialization).
•	Symmetry-Aware Encoding: Please clarify if this is implemented simply as 2D data augmentation. Furthermore, the axis of reflection (e.g., horizontal, vertical) should be explicitly stated.
•	Line 215: The notations $J$ and $K$ are used without prior definition.
•	Line 256: The notation $t$ appears to be overloaded. In Line 187, it seems to represent time, whereas here it likely refers to the optimization iteration. Please clarify and consider using distinct notation to avoid ambiguity.
3.  The experimental analysis is missing a comparison of computational and time costs against previous methods. Given that this is a test-time optimization approach, quantifying the efficiency-performance trade-off is essential for a complete evaluation of the method's practical utility.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AnimalGS, an optimization-based framework for 4D animal reconstruction from monocular video using 3DGS as the explicit representation. Unlike previous methods based on category-specific templates or generative priors, the animal shape prior predicted by Fauna is used only as initialization, rather than as constraints.

The innovation of this paper mainly focuses on model design: 1. Joint-Aware achors for robut pose refinement; 2. Symmetry-aware temporal encoding for bilateral cues; Pose-guided deformation based on cross-attention between joint and gaussian features.

Experimental comparisons with Fauna, GART, D-3DGS, and GVF-Diffusion are provided to demonstrate the effectiveness of the proposed pipeline.

### Strengths
1. This paper proposes a template-free pipeline for 4D animal reconstruction with requiring dataset-level supervision.
2. The joint-anchor and symmetry encoding are interpretable designs, which stablizes optimization process.
3. The pose-conditioned deformation provides a clever constrain on 3DGS pipeline.
4. The paper is well-written and easy to understand.

### Weaknesses
1. Optimization-based methods require per-sample training, which is time-consuming.
2. Quantitative metrics are proxy-based without 3D ground truth, which cannot assess the geometric accuracy.
3. The Deformable-3DGS is not fairly compared in an identical setting: Initialize the Gaussians with Fauna and use RGB and Silhouette together for training. This critical result is missing.
4. The method still depends on Fauna for initialization; if Fauna fails catastrophically, it’s unclear how robust AnimalGS remains.

### Questions
Please address my concerns in the weakness part, and I will increase the score accordingly.

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
The paper proposes AnimalGS, a test-time optimization (TTO) framework that reconstructs 4D animal geometry and motion from a single monocular video using a canonical 3D Gaussian Splatting (3DGS) representation. The pipeline (i) initializes from a coarse FAUNA prior (single-image category-agnostic animal model), (ii) performs pose refinement via learnable joint anchors with a symmetry-aware temporal encoding, and (iii) applies pose-guided non-rigid deformation to capture residual motion/appearance. The method optimizes photometric/silhouette losses plus a normal-smoothness regularizer. Experiments on DAVIS/Online/APTv2 report improved input-view PSNR/LPIPS and novel-view KID/FVD vs. Fauna, GART, D-3DGS, and a diffusion baseline “GVF-Diffusion (Trellis-based).” The authors claim strong perceptual quality and present a small user study.

### Strengths
Addresses a high-impact problem: single-video 4D animal reconstruction.

The two-stage TTO with symmetry cueing is practical and fits 3DGS well; ablations indicate each piece helps.

Shows consistent improvements over baseline TTO/3DGS choices reported in the paper.

Implementation appears competent; real-time 3DGS rendering is attractive for downstream use.

### Weaknesses
Visual quality not consistently convincing: Even in curated figures, limbs and fine appendages exhibit shape drift/temporal wobble, and small non-rigid details are often smoothed out; some sequences look over-regularized (likely from the normal-smoothness phase). The paper does not show challenging real-world clips with fast motion/occlusion and admits remaining failure modes (head/tail subtleties). 


Missing strong baselines for direct 3D/4D generation: No side-by-side against Hunyuan3D (image→3D) or Trellis-based pipelines (structured 3D latents). The paper only evaluates GVF-Diffusion as a proxy; that is insufficient—you should compare reconstruction fidelity and geometric consistency vs. (i) Hunyuan3D (image→3D + pose retarget), (ii) Trellis+render-supervised 4D generation, and (iii) a hybrid pipeline (Hunyuan3D init → your TTO).

Under-analyzed comparisons to the articulated-reconstruction literature the community expects:

DreaMo (single casual video → articulated model and motion)

LIMR (Learning Implicit Representation for Reconstructing Articulated Objects)

S3O (dual-phase dynamic shape + skeleton from single video)

PAD3R (pose-aware dynamic reconstruction from casual videos)

MagicArticulate (prepare static models for articulation)
Even if these target broader articulated objects (not only animals), their architectural choices, priors, and metrics are highly relevant. A clear, protocol-matched comparison (or at least cross-dataset transfer) is needed to justify the claimed advantages.

Over-reliance on priors without robustness study: Replace or degrade FAUNA/camera priors; inject mask/trajectory noise; quantify sensitivity.

Assumptions not stress-tested: What if first frames are not static? Provide an automatic canonical-frame detection ablation.

Evaluation scope: Mostly short clips, restrained motions, and curated species. Add unconstrained YouTube sequences, long clips, severe occlusions, and small articulators.

### Questions
Hunyuan3D / Trellis: Can you provide direct, protocol-matched comparisons? For Hunyuan3D, try (a) image→3D per key frame + tracking; (b) image→3D initialization followed by your TTO; (c) evaluate fidelity vs. plausibility trade-offs. For Trellis, evaluate a structured latent 4D pipeline with your same datasets/metrics.

Articulated-reconstruction baselines: Please add DreaMo, LIMR, S3O, PAD3R, MagicArticulate to discussion.

Robustness to priors: How does performance change if FAUNA predictions are noisy/misaligned (pose jitter, wrong joints), masks are imperfect, or cameras are biased? Provide degradation curves.

Canonical-frame assumption: Show results when the first frames contain motion; can you auto-select canonical segments?

Runtime/memory: Report training steps, seconds/iteration, Gaussians count vs. resolution, and Ablate λsmooth schedule; compare to GART / D-3DGS under identical hardware.

Failure analysis: Provide videos where tails/ears/paws fail; diagnose whether pose refinement or deformation is the bottleneck.

Physical plausibility: Do you enforce joint limits or bone-length constraints? If not, quantify bone-length variance over time.

### Soundness
2

### Presentation
2

### Contribution
2
