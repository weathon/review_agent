## Summary

I2VControl-Camera proposes using dense 2D point trajectories (projected from RGBD point clouds) as control signals for camera-controllable image-to-video generation, replacing the sparse extrinsic matrices or Plücker embeddings used in prior work. A secondary contribution is a per-frame scalar "motion strength" parameter derived from the residuals of rigid fitting, intended to provide adjustable control over the degree of subject (non-camera) motion in generated videos. An iterative algorithm partitions scene points into static/dynamic regions via rigid fitting residuals, and a lightweight adapter injects the control signals into a frozen base video generation model.

## Strengths

- **Point trajectory control signal achieves substantial precision gains.** On RealEstate10K (Table 1), the method achieves RotErr of 0.53 and TransErr of 9.72, compared to CameraCtrl's 1.26/21.60 and MotionCtrl's 2.66/12.70 — roughly 2–2.4× improvements in camera pose accuracy. This is the paper's core empirical contribution and is convincingly demonstrated.
- **Practical data pipeline from ordinary RGB videos.** The three-stage pipeline (Unidepth for depth estimation, SpatialTracker for correspondence, iterative Algorithm 1 for static/dynamic partition) enables training on abundant RGB video without specialized 3D-annotated data. Algorithm 1's iterative residual-based partition is a concrete and reasonable design for separating static/dynamic regions under camera motion.
- **Fair experimental methodology with retrained baselines.** The authors retrained MotionCtrl and CameraCtrl on the same base model (Magicvideo-V2) and training data (Section 4.3), eliminating confounds from different base models — a commendable practice that is uncommon in this area.
- **Model-agnostic adapter design.** The adapter injects control features into temporal self-attention blocks while freezing the base model (Fig. 4), making it portable across different I2V architectures in principle.

## Weaknesses

### Fatal
None.

### Major

- **The theoretical framework (Sec 3.1) is disconnected from the practical implementation and presented as more principled than it is.** The Maclaurin expansion defines R_λ = I + J_F(0, λ) − J_F(0, 0) (Eq. 6), which is a general 3×3 matrix — not guaranteed to satisfy R^T R = I or det(R) = 1. The notation R_λ universally denotes a rotation matrix in the computer vision/geometry literature, and the paper uses it throughout without acknowledging this gap. In practice, (R_λ, t_λ) is obtained via L-BFGS optimization over the static region (Eq. 12), which is a fundamentally different procedure — fitting a rigid transformation to points that are identified as static, not deriving R_λ from a Taylor expansion. The "higher-order infinitesimal" G(p, λ) = o(p) is a local asymptotic statement near p = 0, yet it is applied globally across all scene points in Eq. 9 and Eq. 13. The entire theoretical framework is essentially motivational rather than derivational, but the paper presents it as a principled decomposition (e.g., "we explicitly model the higher-order components of the video trajectory expansion" in the abstract). This disconnect does not invalidate the method — which works well empirically — but it undermines the framing of the theoretical contribution.

- **Missing ablation isolating the effect of the control signal representation.** The paper's central claim is that point trajectories are superior to extrinsic matrices and Plücker embeddings. However, the comparison in Tables 1–2 confounds the control signal representation with the adapter architecture and training data (since MotionCtrl, CameraCtrl, and the proposed method all use different adapter designs). A direct ablation — feeding extrinsic matrices, Plücker embeddings, and point trajectories into the *same* adapter architecture on the *same* base model — would isolate whether the improvement comes from the representation or from other design choices. Without this, the claim that point trajectories are the key factor is not established.

### Minor

- **Motion strength control is a coarse amplitude knob, but the paper's language occasionally overstates its precision.** The abstract claims the method provides "adjustability over the strength of subject motion," and Sec 3.2 says it "accurately gauge[s] and adjust[s] the amplitude of subject motion dynamics." In reality, m_λ is a single scalar per frame averaging all dynamic-point speeds across the image domain (Eq. 9/13). It can modulate *how much* total motion occurs, but not *which* objects move, *how* they move, or *where* motion appears. The qualitative results (Fig. 6) do show semantically plausible motion (a wolf runs, a bear walks), but this is the base model's inference rather than the motion strength signal providing precise control. The paper's conclusion acknowledges this limitation by listing "motion brush" as future work, but the body could be more upfront about the granularity limitations of a single scalar.

- **MSC shows diminishing returns at high motion strength values, suggesting limited effective range.** In Table 2, MSC increases from 18.96 (strength=0) to 38.23 (200) to 47.13 (400) to 47.70 (600). The increment from 400→600 is only 0.57, compared to 8.90 from 200→400, indicating near-saturation. The paper claims this "proves our adjustable motion strength control ability" but the effective range appears limited to lower values.

- **The claim that Plücker embeddings "do not offer any additional information compared to the camera matrix" (Sec 1) is misleading.** While Plücker embeddings are fully determined by camera parameters (so the information-theoretic claim is technically defensible), they provide a denser per-pixel representation that is arguably easier for neural networks to learn from than a single extrinsic matrix per frame. The key differentiator of point trajectories is that they encode scene-specific depth information from the input image, which neither extrinsic matrices nor Plücker embeddings provide. The paper should have emphasized this image-dependent depth advantage rather than making a blanket equivalence claim.

### Trivial
None.

## Nice-to-Haves

- A finer-grained sweep of motion strength values (beyond 0/200/400/600) with a plotted MSC vs. strength curve to characterize the control range and saturation point.
- Evaluation of whether generated motion is semantically appropriate (e.g., running vs. jittering) beyond just amplitude, via a small user study or a motion-plausibility metric.
- Failure analysis of the data pipeline: how often Unidepth + SpatialTracker + Algorithm 1 produces incorrect control signals, especially on challenging scenes (thin structures, reflective surfaces).

## Removed Points

*These points were flagged for removal — treat them with caution.*

- **FID reference distribution concern (Harsh Critic):** The critic argues that FID computed against 2000 WebVid frames measures general image quality rather than task-specific quality. While true, this is a standard practice in the video generation field and CameraCtrl used the same protocol. The critic also notes MotionCtrl achieves lower FID (98.54) than Ours-0 (100.36) on the movable object dataset, but the paper itself provides a reasonable explanation: "movable objects are forcibly held static, resulting in unnatural and insufficiently diverse frames." Removed: FID reference choice is standard, and the paper acknowledges the Ours-0 FID gap.

- **No variance or significance reporting (Harsh Critic):** Reporting single numbers without standard deviations is the norm in this field for large-scale video generation experiments. This is a nice-to-have, not a substantive weakness.

- **Motion strength scale uncalibrated (Harsh Critic):** The values 0/200/400/600 are arbitrary hyperparameter choices. This is typical for conditioning parameters in generative models. Not a substantive weakness.

- **Missing dataset details / reproducibility concerns (Harsh Critic):** The 30K training dataset is described as collected video clips with camera movements and natural motion. Demanding full dataset documentation is impractical for a submission. Removed as a nitpick.

- **Baseline training details (Harsh Critic):** The paper states "the same settings were used" for retrained baselines. Requesting exact learning rates and step counts is a minor reproducibility nitpick.

- **Depth estimation failure modes at inference (Harsh Critic):** While Unidepth is an extra dependency, the paper does mention using it at inference (Sec 3.4), and depth estimation is a mature capability. The concern is valid but speculative without evidence of frequent failures. Moved to nice-to-have.

- **"Principled theoretical decomposition" (Strength Finder):** This strength conflicts with the verified Major weakness about the theoretical disconnect. The decomposition *idea* is sound but the execution is not principled — it is motivational. Removed from strengths.

- **"Effective and adjustable motion strength control for subject dynamics" (Strength Finder):** Partially conflicts with the verified Minor weakness about overstatement. The adjustability is real but the "effective" and "accurate" framing overstates what a single scalar provides. Kept as a weaker version in strengths.

## Novel Insights

The paper's most interesting insight — which it does not fully articulate — is that point trajectories are effective as camera control signals precisely because they encode scene-specific depth information, making them image-aware in a way that extrinsic matrices and Plücker embeddings (which depend only on camera parameters) cannot be. This is the genuine reason for the precision improvement, rather than the "denser representation" argument the paper leans on. The theoretical framework, while appealing, obscures this practical advantage behind a Maclaurin expansion that does not actually drive the implementation.

## Suggestions

- Reframe Section 3.1 as motivational intuition rather than a principled derivation. Acknowledge that R_λ from Eq. 6 is not constrained to be a rotation matrix, and that the practical (R_λ, t_λ) is obtained by fitting a rigid transformation to identified-static points. This honest reframing preserves the useful intuition without overclaiming.
- Add an ablation table where the same adapter architecture is trained with (a) extrinsic matrix input, (b) Plücker embedding input, and (c) point trajectory input, on the same base model and training data. This would cleanly isolate the contribution of the representation.
- Characterize the motion strength control more honestly: it provides amplitude control over total subject motion, not precise dynamics control. Acknowledge the limited effective range (diminishing returns above strength=400) and discuss the per-region/per-object extension as a natural next step.

## Score and Decision

**Calibration anchors used:**

| Paper | Score | Comparison |
|-------|-------|------------|
| UniSim | 7.5 (Accept Oral) | Much stronger: broader scope, well-grounded, significant real-world transfer. This paper is well below. |
| 3DTrajMaster | 6.75 (Accept Poster) | Stronger: novel 3D trajectory task, impressive multi-entity control, extensive results. This paper has weaker novelty and overclaimed theory. |
| EgoSim | 6.0 (Accept Poster) | Similar profile: camera conditioning in video generation, empirical strengths with theoretical gaps. This paper is roughly comparable — better empirical camera precision but worse theoretical grounding. |
| CamTrol | 5.8 (Accept Poster) | Weaker: training-free approach with scene inconsistency and inaccurate camera. This paper is clearly stronger empirically. |
| Camera Motion Guidance | 5.5 (Reject) | Weaker: reads as technical report, limited novelty. This paper has more substantial contribution. |
| TCIG | 1.5 (Reject) | Much weaker: overclaimed results, no novelty. This paper is far above. |

This paper sits in the 5.5–6.5 band. It is clearly above the rejected camera-control papers (CamTrol at 5.8 is an outlier Accept with weaker results) and below the strong poster/oral papers. The strongest comparable anchor is EgoSim at 6.0. This paper has better empirical camera precision results than EgoSim but a more serious theoretical disconnect and missing key ablation. The practical contribution (point trajectories for camera control) is genuine and significant, but the paper overclaims the theoretical contribution and lacks the ablation needed to isolate the representation's effect. I place it at the low end of the Accept range.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>