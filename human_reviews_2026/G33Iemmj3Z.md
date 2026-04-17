# Distractor-free Generalizable 3D Gaussian Splatting

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
We present DGGS, a novel framework that addresses the previously unexplored challenge: \textbf{Distractor-free Generalizable 3D Gaussian Splatting} (3DGS). Previous generalizable 3DGS works are often limited to static scenes, struggling to mitigate distractor impacts in training and inference phases, which leads to training instability and inference artifacts. To address this new challenge, we propose a distractor-free generalizable training paradigm and corresponding inference framework, which can be directly integrated into existing Generalizable 3DGS frameworks. Specifically, in our training paradigm, DGGS proposes a feed-forward mask prediction and refinement module based on the 3D consistency of references and semantic prior, effectively eliminating the impact of distractor on training loss. Based on these masks, we combat distractor-induced artifacts and holes at inference time through a novel two-stage inference framework for reference scoring and re-selection, complemented by a distractor pruning mechanism that further removes residual distractor 3DGS-primitive influences. Extensive feed-forward experiments on the real and our synthetic data show DGGS's reconstruction capability when dealing with novel distractor scenes. Moreover, our feed-forward mask prediction even achieves an accuracy superior to scene-specific Distractor-free methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Distractor-free Generalizable 3D Gaussian Splatting, a method designed for removing transient objects in feedforward GS methods. The main goal is to find proper transient masks and remove transients from optimizing the feedforward network. Firstly, DGGS implemented a robust mask based on robustnerf. Such mask is improved by incorporating a mask_ref (from re-rendered photometric loss) from reference images and project it on the target pose. Such mask is further improved based on multi-view visibility and combines with robustnerf. At test time, DGGS introduces schemes to reduce the usage of images with strong transient objects, and perform additional optimization to remove artifacts in the scene.

### Strengths
1. The authors demonstrated superior performances in feedforward 3DGS compared to prior work based on PSNR/SSIM etc.
3. The masks seem reasonable, and the resultant visual quality is also improved.

### Weaknesses
In general, I think the writing requires a lot of polishing, in terms of readability, background knowledge, e.g., segmentation models which play a significant role, and overall importance/novelty of the method. 

After reading this work, the sense that I get is that, while performance is certainly good, the paper heavily leans on engineering and parameter tuning. This includes multiple important hyperparameters (threshold for masks), requirement for estimated depth to be reasonable for reprojection, integration of an external segmentation model that is minimally mentioned, test time hyperparameters (N views - BTW, N is defined multiple times with different meanings across the text). The overall insight is not very different from prior work, i.e., finding good transient masks such that rendering results can be improved by ignoring transients. I also have concerns about how this can be applied to larger scale scenes, as feedforward GS currently is relatively limited. Large scale scenes will lead to more noise in depth and multi-view inconsistencies, which this method seems to be very sensitive on. 

I lean borderline on this work based on novelty; since there is no borderline, I lean towards borderline reject as I believe the writing can be better.

### Questions
1. The image resolution in the submitted PDF is really low. To the point where this is difficult to read. 
2. Given that NeRF-HUGS produces similar masks as DGGS, can the authors expand on why mvsplat + NeRF-HUGS is significantly worse than DGGS?

### Soundness
2

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
4

### Summary
The paper proposes DGGS, a framework designed to enable generalizable 3D Gaussian Splatting under distractor-rich real-world scenarios. Unlike existing generalizable 3DGS methods that assume clean static scenes, this work introduces: 1) A Reference-based Mask Prediction mechanism leveraging multi-view consistency. 2) A Mask Refinement module using segmentation priors and occlusion-aware auxiliary supervision. 3) A Two-stage inference strategy, including reference scoring and 3D gaussian primitive pruning, to suppress inference-time artifacts.

### Strengths
1. The use of multi-view geometric consistency to correct masks, reflects good insight rather than brute force.

2. Comprehensive experiments, including synthetic distractor construction.

3. Inference-time pruning is practical and effective.

### Weaknesses
1. Heavy reliance on segmentation priors. The pipeline is not truly “feed-forward generalizable” if high-quality segmentation is required and pre-computed.

2. Reference stability assumption unproven. The paper does not quantify how often reference re-rendering is accurate enough to serve as a stable supervisory source.

3. Efficiency cost. Two-stage inference + segmentation noticeably sacrifices speed, which is a key appeal of 3DGS.

4. Mask failure modes not fully analyzed. The limitations section mentions occlusions, but no systematic characterization is provided.

### Questions
1. How robust is the reference mask filtering when references also contain distractors?

2. What is the computational overhead of the full pipeline?

3. Does the segmentation model need retraining or domain adaptation in unseen categories?

4. Could the mask refinement be done without segmentation, e.g., self-supervised feature aggregation?

5. How does performance degrade with increasing viewpoint disparity among references?

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
3

### Summary
This paper proposes a framework called DGGS aimed at making generalizable 3D Gaussian Splatting (3DGS) robust against distractors (transient objects) in real-world scenes. Existing generalizable 3DGS models assume static environments and suffer from training instability and artifacts when transient objects appear. DGGS mitigates this by predicting distractor masks through multi-view geometric consistency and refining them with segmentation priors, then using these masks to exclude distractors during training. At inference, it selects cleaner reference views and prunes distractor-related Gaussian primitives. Experiments demostrate this method outperform the generalizable 3DGS baselines, even better than some scene-specific distractor removal techniques.

### Strengths
- This is the first work addressing distractors in generalizable 3DGS, filling an important gap in real-world usage.
- It significantly boosts robustness and reconstruction quality compared to both baseline 3DGS models and naively transferred scene-specific distractor-free methods.
- The approach generalizes well to unseen scenes and improves inference quality via smart reference selection and pruning.

### Weaknesses
- The method relies on several additional modules, but the sensitivity of the overall performance to the choice or quality of these modules is not discussed.
- The quality of the generated masks depends on the accuracy of segmentation and depth estimation, which may lead to failure cases in scenes with heavy occlusions or imprecise geometry.
- Since the approach depends on mask generation, it is unclear how well it would handle naturally dynamic environments, such as moving trees or water, where mask accuracy could be compromised.

### Questions
Out of curiosity, can the two-stage inference be made more efficient for real-time use? Have you tested foundation segmentation models like SAM-2, and do they meaningfully improve results?

### Soundness
4

### Presentation
4

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
This submission introduces DGGS, a new framework for Distractor-free Generalizable 3D Gaussian Splatting. It tackles two overlooked issues in generalizable 3DGS: (1) training instability due to transient distractors in real-world data, and (2) feed-forward inference artifacts caused by distractors in references. The method proposes a reference-based mask prediction that leverages 3D multi-view consistency to filter robust residual-based masks, a mask refinement stage that decouples disparity-induced errors and uses entity segmentation plus an auxiliary loss, and a two-stage inference procedure with reference scoring and distractor pruning. Extensive experiments on real (On-the-go, RobustNeRF) and synthetic data show consistent improvements over retrained generalizable baselines and scene-specific distractor-free approaches adapted to the generalizable setting, with additional gains from the inference stage.

### Strengths
- Clearly identifies and formulates a new, practically relevant problem: distractor-free generalizable 3DGS.

- Elegant reference-based mask filtering that reduces over-suppression typical of residual-only masks.

- Thoughtful mask refinement: decoupling disparity vs. distractor; auxiliary loss exploiting cross-view occlusion cues.

- Practical two-stage inference: reference scoring and 3D primitive pruning demonstrably reduce artifacts/holes.

- Strong empirical results with comprehensive comparisons, ablations, and both real and synthetic setups.

- Method is modular and can plug into existing generalizable 3DGS pipelines.

### Weaknesses
(1) Dependence on pre-trained entity segmentation during training and inference undermines full “feed-forward” purity and adds latency; domain robustness of the segmenter is not analyzed.

(2) The mask fusion strategy uses intersection across references (conservative), which may lead to under-coverage in low-overlap or high-parallax settings; the trade-off is not deeply quantified.

(3) Distractor pruning can introduce speckle/holes in commonly occluded areas; mitigation is heuristic and the failure modes are only briefly discussed.

(4) Some reliance on depth/warping quality from inferred 3DGS; failure cases when depth is noisy or textures are repeated are not thoroughly dissected.

(5) Fairness concerns: scene-specific methods are adapted into a generalizable training loop but may not reflect their best practices (e.g., stronger per-scene optimization), making cross-paradigm comparisons tricky.

(6) Efficiency overhead from two-stage inference and segmentation is non-trivial; the paper reports times but not detailed profiling or memory usage under varied K, N, and resolution.

### Questions
(1) How sensitive is performance to the quality of the pre-trained segmentation model and its domain shift (e.g., indoor vs. outdoor, low light)? Can lighter/zero-shot segmenters maintain most gains?

(2) Why choose strict intersection for multi-view mask fusion? Have you tried soft/weighted fusion (e.g., confidence weighting by photometric residuals or view angle) to recover more static pixels without raising distractor leakage?

(3) Can the auxiliary loss be extended with photometric/feature consistency terms to lessen dependence on segmentation?

(4) How robust is the approach when the majority of references contain similar distractors (e.g., many frames with the same moving car)? Does reference scoring still find sufficiently clean views?

(5) For pruning, did you evaluate per-primitive confidence aggregation across references (e.g., voting) instead of binary masking per view to reduce speckle?

(6) Could you report memory/time breakdown across stages (feature projection, mask prediction/refinement, scoring, pruning) and how they scale with K and N?

(7) Are there benefits or risks in training with the scoring mechanism online (curriculum-style selection of “cleaner” references) rather than only at inference?

(8) How does DGGS perform when camera intrinsics vary or are noisy? Is U assumed known and consistent?

### Soundness
3

### Presentation
3

### Contribution
3
