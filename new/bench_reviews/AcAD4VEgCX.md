Now let me synthesize my review. I have a thorough understanding of the paper, the reviewer critiques, and calibration papers. Let me carefully assess what's valid and what's not.

Key points from the harsh critic:

1. **Mathematical formulation mismatch**: The Maclaurin expansion yields R_λ as a general linear map, not a rotation matrix. The implementation uses least-squares fitting (Alg 1, Eq 12) rather than the theoretical derivation. The G(p,λ) is never explicitly defined in the discrete case. This is a valid and important observation.

2. **Point trajectory vs extrinsic matrix**: The comparison doesn't isolate the representation change - different supervision signals (dense trajectory vs sparse poses) and different training data. No ablation. Valid concern.

3. **Motion strength ad hoc**: The continuous formulation doesn't match the implementation, and at inference it's a user-tuned scalar with no principled connection. The MSC metric is coarse. Valid concern.

4. **Depth/tracking robustness**: No analysis of how upstream errors propagate. Valid but somewhat standard limitation.

Now let me check whether the neutral reviewer's points about missing baselines (CamCo) and the spark's ablation concerns overlap with the harsh critic.

Let me now write my final review.</think>Now I have sufficient calibration context. Let me construct my final review by carefully verifying each critique against the paper.

## Summary

The paper presents **I2VControl-Camera**, a method for precise camera control in image-to-video generation with an additional adjustable subject motion strength parameter. Instead of using camera extrinsic matrices (as in MotionCtrl) or Plücker embeddings (as in CameraCtrl), the method uses 2D point trajectories from projected RGBD point clouds as a denser control signal, and explicitly decouples subject dynamics from camera motion via a trajectory expansion framework, where higher-order terms define a "motion strength" scalar that users can adjust at inference time. An adapter architecture enables training on top of a frozen base model (MagicVideo-V2), and a data pipeline constructs training signals from RGB videos using depth estimation and tracking.

## Strengths

- **Addresses an important and timely problem.** Fine-grained camera control with independent subject motion adjustment is exactly the kind of controllability creators want. The motion strength slider, even if imperfectly principled, is a genuinely useful capability absent from prior methods like MotionCtrl and CameraCtrl.

- **Strong quantitative improvements on camera control precision.** On the RealEstate10K test set, the method achieves RotErr of 0.53 vs. 1.26 (CameraCtrl) and TransErr of 9.72 vs. 12.70 (MotionCtrl) — substantial margins. These gains are consistent with the qualitative evidence of pixel-level alignment shown in Figure 5 and 7.

- **The data pipeline idea is practically valuable.** The approach of constructing training control signals from arbitrary RGB videos (via depth estimation, dense tracking, and iterative rigid fitting in Algorithm 1) enables training on much larger and more diverse data than methods requiring pre-computed camera annotations. This is a useful contribution independent of the theoretical framing.

- **Clear demonstration of the motion strength control concept.** Figure 6 provides convincing qualitative evidence that the scalar parameter meaningfully interpolates between static and dynamic scenes while preserving camera control, and Table 2 shows MSC monotonically increasing from Ours-0 to Ours-600, confirming the control is functional.

- **The adapter architecture is simple and model-agnostic in principle.** The design of conditioning via conv-processed control tokens concatenated with temporal attention tokens is straightforward and plausibly portable to other base models.

## Weaknesses

### Fatal

None.

### Major

- **The theoretical formulation is disconnected from the actual implementation, and the paper overclaims principled decoupling.** The Section 3.1 Maclaurin expansion derives R_λ = I + J_F(0,λ) − J_F(0,0), which is a general linear map, not constrained to be a rotation matrix — it is never enforced to live in SO(3). The actual pipeline (Algorithm 1, Eq. 12) instead solves for R_λ, t_λ via nonlinear least squares fitting on 2D projections of static points, which has no connection to the Jacobian-based definition. Similarly, G(p,λ) is defined theoretically in Eq. 7 as the residual after a linear decomposition, but the discrete implementation (Eq. 13) computes motion strength from frame-to-frame differences of an undefined G. The paper never provides an explicit discrete formula for G(p,λ) in terms of the tracked points and the fitted transform. The core claim — *"we explicitly model decoupled motion representations: 3D rigid point trajectories and motion strength"* — is therefore not supported by a coherent, implemented formulation. What the method actually does is: (1) compute rigid 2D trajectory previews from depth, and (2) add a scalar "motion strength" knob. These are empirically useful but not the principled decoupling the paper claims. This significantly undermines the paper's stated conceptual contribution.

- **The advantage of point trajectories over extrinsic matrices is not isolated by ablation.** The paper claims that point trajectories provide a "denser representation" and superior controllability compared to camera matrices/Plücker embeddings. However, the comparisons retrain MotionCtrl and CameraCtrl on the same base model but with their original sparse control interfaces, while I2VControl-Camera uses dense per-pixel trajectory maps plus a motion strength channel. The performance difference could stem from the density of the conditioning signal, the different training data pipeline (custom 30K videos with tracking), or other implementation details — not necessarily from the representation itself. No ablation tests: (a) the same adapter fed only extrinsic matrices, or (b) point trajectories without the motion strength channel. Without these, the claim that the trajectory representation itself drives the improvement is unsupported.

- **The motion strength control lacks rigorous characterization.** The paper claims "precisely perceive and adjust the amplitude of subject motion dynamics," but provides only four discrete values (0, 200, 400, 600) tested on simple single-object scenes. There is no analysis of: (a) whether the mapping from scalar to actual motion is monotonic and proportional across diverse scenes, (b) whether camera-following accuracy degrades at higher motion strengths (the RotErr/TransErr degrade from 0.76/6.97 at strength 0 to 1.18/8.16 at strength 600 in Table 2, suggesting a trade-off), or (c) whether "motion strength" controls only subject motion vs. also inducing background dynamics or artifacts. The MSC metric (2D rigid alignment of optical flow) conflates all non-rigid motion, including parallax and generation artifacts, with intentional subject dynamics.

### Minor

- **Depth/tracking pipeline robustness is not analyzed.** The method depends heavily on Unidepth for metric depth estimation and SpatialTracker for dense correspondence, yet no analysis is provided of how errors in these upstream modules affect the quality of control signals or final video generation. Showing a few failure cases from inaccurate depth or tracking would increase confidence in the method's practical applicability.

- **CamCo is explicitly mentioned in Section 2.3 and critiqued for causing "small motion dynamic" but excluded from experiments.** Including it as a baseline — or at minimum explaining why it was excluded beyond "limited fixed movement patterns" for AnimateDiff (which is a different method) — would strengthen comparisons.

- **The iterative Algorithm 1 for static/dynamic region extraction is underspecified.** The thresholds ε, α, and N_max are not provided, nor is any analysis of their sensitivity or convergence behavior. Additionally, R in Eq. 12 is solved via L-BFGS without enforcing orthogonality (i.e., R ∈ SO(3)), which means the "rigid transform" fit can produce non-physical transformations. This was noted in the paper's notation but not discussed as a potential issue.

- **The custom 30K-video training dataset is only described at a high level.** No quantitative characterization (distribution of motion types, motion magnitudes, scene categories) is provided, making it hard to assess generalization.

### Trivial

- The MSC column header in Table 2 has both ↓ and ↑ arrows, which is confusing.
- In some figures the "GT Preview" label may mislead readers into thinking these are ground-truth video frames rather than rendered depth projections.

## Nice-to-Haves

- Ablation studies testing point trajectories vs. extrinsic matrices as input (keeping the adapter and training pipeline fixed), and testing with/without the motion strength channel.
- Evaluation on a second base model to support the claimed model-agnostic architecture.
- Temporal consistency metrics (e.g., FVD or warp error) to ensure motion strength adjustment doesn't introduce flickering.
- Intermediate motion strength values (not just 0 and 600) shown in figures, and a quantitative plot of MSC vs. strength value across diverse scene types.
- FID evaluation on video-level features rather than randomly sampled frames from an unrelated dataset (WebVid).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that the method abandons the Maclaurin expansion entirely.** While the implementation uses least-squares fitting rather than the Jacobian-based formula, the theoretical decomposition (linear term ≈ camera motion, residual ≈ subject motion) is consistent with what the method does in practice. The claim that the derivation is "entirely decorative" overstates the issue — the formulation motivates separating the signal even though the estimation differs from the theoretical formula. However, the mismatch between the derivation (R_λ not constrained to rotations) and the implementation (which fits in 2D projection space) remains a valid concern.

- **Harsh critic's claim that FID is "misleading" because it's computed per-frame.** While per-frame FID doesn't capture temporal quality, this is a standard evaluation practice in the video generation community, and the paper is transparent about how FID is computed.

- **Spark's suggestion to evaluate "whether the motion strength actually matches user intent via a user study."** User studies are not standard for purely algorithmic system papers in this venue and would be a nice-to-have but not a weakness.

- **Spark's claim that "FID computed on frames from an unrelated dataset (WebVid) is unusual."** The paper states "we randomly select 2000 video frames from WebVid," which is a standard reference distribution for FID in video generation papers. This is a conventional evaluation protocol.

- **Neutral reviewer's suggestion to include FVD.** Standard but not required; this is a nice-to-have.

- **Human finder's concern about "scale ambiguity of point trajectories."** The entire point of using point trajectories (projecting 3D points) rather than raw extrinsic matrices is to provide signals that are scale-consistent with the input image — the depth estimation anchors the scale. The concern about scale ambiguity is less applicable here than for methods using raw camera translation vectors.

- **Human finder's concern about "limited novelty in the architectural design."** The adapter is straightforward, but the paper's novelty is explicitly in the control signal design, not the architecture. This is not a genuine weakness.

## Novel Insights

The key insight that emerges from combining the reviewers' observations is that the paper's main conceptual contribution — the trajectory expansion decomposition into linear and non-linear terms — is better understood as a *motivation* for a sensible engineering choice (separating rigid camera motion from subject dynamics and computing them via different mechanisms) rather than as a rigorous mathematical framework. The actual system works well in practice, but the "Maclaurin expansion" language creates an expectation of mathematical precision that the implementation does not fulfill, most notably because R_λ is not constrained to SO(3) in the derivation, and G(p,λ) is never concretely defined in discrete form. Reframing the contribution as an engineering approach informed by the decomposition insight, rather than a theoretical advance, would substantially improve the paper's clarity and honesty.

## Suggestions

- Reframe Section 3.1–3.2 to present the trajectory decomposition as a *conceptual motivation* rather than rigorous mathematics. Remove or qualify the Maclaurin expansion language. Provide an explicit discrete definition of G(p,λ) in terms of tracked points and fitted transforms.
- Add at minimum one ablation: the same adapter with only extrinsic matrix conditioning (no point trajectories) on the same training data. This would directly test whether the density of the control signal is the key factor.
- Report Rλ orthogonality during L-BFGS fitting, or add an SO(3) projection step, and discuss whether the unconstrained fit causes artifacts.
- Test 5–7 motion strength values rather than just 4, and plot MSC and RotErr/TransErr as functions of strength. This would validate the claim of "adjustable" (i.e., continuous) motion control.

## Score and Decision

For calibration: CameraCtrl received scores of 6/8/6/6 (Accept Poster), VD3D received 6/8/6/3/8 (Accept Poster), CamCo received 3/5/5/6/5 (Reject), and the "Boosting Camera Motion Control" paper received 6/5/5/6 (Reject). CameraCtrl and VD3D had solid empirical results with clear (if incremental) technical contributions. CamCo was rejected for marginal improvements and lack of novelty over CameraCtrl. This paper (I2VControl-Camera) has noticeably stronger empirical results on camera control precision than CameraCtrl (RotErr 0.53 vs 1.26) and adds the genuinely useful motion strength feature. However, its core theoretical claim of "explicitly modeling higher-order components" is overstated relative to the actual implementation, and the lack of ablations for the representation choice is a significant evidential gap. Compared to CameraCtrl (scored ~6.5 avg, accepted), this paper's empirical contribution is arguably stronger (better precision + new capability), but its theoretical overclaiming is worse, and its evaluation breadth is thinner (single base model, limited motion strength analysis). Compared to CamCo (rejected, avg ~4.8), this paper clearly has stronger results and a more useful feature set.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>