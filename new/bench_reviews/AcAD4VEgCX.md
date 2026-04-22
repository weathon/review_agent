Now I have thoroughly read the paper and all calibration anchors. Let me synthesize my findings.

## Summary

I2VControl-Camera proposes using 3D point trajectories in camera coordinates (instead of extrinsic matrices or Plücker embeddings) as a dense control signal for camera control in image-to-video generation, and introduces a scalar motion strength parameter derived from higher-order trajectory terms to adjust subject dynamics. The method includes an iterative data pipeline (Algorithm 1) for extracting static/dynamic region partitions and camera parameters from raw RGB videos using off-the-shelf depth estimation and tracking, and a model-agnostic adapter architecture that injects the control signal into a frozen base video diffusion model.

## Strengths

- **Dense point trajectory control signal yields substantially better camera precision.** On RealEstate10K (Table 1), the method achieves RotErr of 0.53 and TransErr of 9.72, compared to CameraCtrl's 1.26/21.60 and MotionCtrl's 2.66/12.70 — a 2–5× reduction in error. The gains are large enough to be convincing at face value.

- **Motion strength provides a novel and demonstrated control knob for subject dynamics.** The qualitative results in Fig. 6 clearly show the knob works as intended: at strength 0, subjects remain static; at 600, subjects move naturally. Quantitatively (Table 2), MSC ranges from 18.96 (Ours-0) to 47.70 (Ours-600), bracketing both CameraCtrl (32.69) and MotionCtrl (42.28), demonstrating effective and adjustable control.

- **Practical data pipeline that constructs training signals from ordinary RGB videos without specialized 3D annotations.** Algorithm 1 iteratively separates static and dynamic regions by fitting rigid motion to point trajectories, using only off-the-shelf tools (UniDepth for depth, SpatialTracker for tracking). This is a useful contribution for making the method trainable at scale.

- **Fair experimental comparison via retraining baselines on the same base model and settings** (Sec. 4.3), rather than comparing against published numbers with different base models, training sets, and resolutions. This makes the improvements in Tables 1–2 directly attributable to the control signal design.

- **Preview-based user interaction** (Fig. 5) providing immediate pixel-level visual feedback of intended camera motion before generation — a practical feature not available in prior methods.

## Weaknesses

### Fatal

None.

### Major

- **Complete absence of ablation studies isolating individual contributions.** The paper claims two distinct contributions: (a) point trajectories improve control precision over extrinsic matrices/Plücker embeddings, and (b) motion strength modeling enables adjustable subject dynamics. However, no ablation disentangles these from each other or from the data pipeline. A reader cannot determine whether the RotErr/TransErr improvements in Tables 1–2 come from using point trajectories specifically, from the iterative static/dynamic region separation in Algorithm 1, from the custom 30K training dataset, or from some combination. At minimum, the paper needs: (i) a version using extrinsic matrices + motion strength (to isolate trajectory vs. matrix representation), and (ii) a version using point trajectories without motion strength input (to isolate the motion strength contribution). Without this, the individual claims about *why* each component helps are unsupported. This is a significant gap: compared to CamTrol (5.8, accepted poster with some qualitative ablations) and Ctrl-V (4.0, rejected for no ablation), the complete absence puts the contribution attribution on speculative ground.

### Minor

- **The MSC metric has a structural bias that partially confounds its interpretation across methods.** MSC removes camera-induced optical flow via 2D rigid alignment between adjacent frames and measures the residual. Methods with worse camera control may have larger camera-related residuals that survive the imperfect rigid alignment, inflating their MSC even if subject motion is identical. This means the paper's claim that Ours-0 achieves lower MSC than baselines (Table 2) partially conflates "less subject motion" with "better camera control" — both of which the method achieves. However, the qualitative evidence in Fig. 6 is strong enough to support the motion strength claim independently, so this is a secondary concern about metric design rather than a fundamental flaw.

- **Notation $R_\lambda$ suggests a rotation matrix, but the Maclaurin expansion defines it as a general $\mathbb{R}^{3 \times 3}$ matrix.** The paper writes "$R_\lambda \in \mathbb{R}^{3 \times 3}$" (not SO(3)), so it is technically correct, but the use of "R" notation universally associated with rotation matrices is misleading. More importantly, Algorithm 1 fits $(R_\lambda, t_\lambda)$ via unconstrained L-BFGS optimization (Eq. 12) without enforcing orthogonality, while at inference the user provides actual rotation matrices. Since the network consumes $T_\lambda = \Pi(R_\lambda \cdot \Omega + t_\lambda)$ rather than $R_\lambda$ directly, the mismatch is indirect, but if the unconstrained $R_\lambda$ from Algorithm 1 deviates significantly from a proper rotation (e.g., due to noise or imperfect depth estimation), the computed $T_\lambda$ may differ in distribution from inference trajectories. This deserves acknowledgment.

- **Motion strength $m_\lambda$ conflates spatial extent and speed of motion.** Since Eq. 13 averages over *all* $H \times W$ pixels (including static ones where $\mathcal{G}=0$), a small fast-moving object could have similar $m_\lambda$ to a large slow-moving object. This ambiguity limits the interpretability of specific motion strength values (e.g., the choice of 0, 200, 400, 600 in Sec. 4.2.2) and the training distribution of $m_\lambda$ is not reported, making it unclear whether 600 is in-distribution.

- **The statement that Plücker embeddings "do not offer any additional information compared to the camera matrix used in MotionCtrl"** (Sec. 1) is overly strong. While parameterization equivalence holds (Plücker embeddings are a deterministic function of camera parameters), a denser per-pixel representation can be easier for a network to learn from, even with the same Shannon information. This matters because the paper's own argument for point trajectories rests on their being a *denser* signal than extrinsic matrices.

### Trivial

- The paper does not discuss how many layers receive control signal injection or how these architectural choices were made (Sec. 3.4).

## Nice-to-Haves

- Testing on at least one additional base model (e.g., SVD or a DiT-based model) to validate the claimed adapter independence.
- A sensitivity analysis measuring how depth estimation quality (UniDepth) affects control precision.
- Analysis of consistency: running the same input with the same motion strength multiple times and reporting variance in camera metrics and motion patterns.
- Showing failure cases, particularly for scenes where depth estimation or Algorithm 1's static/dynamic separation fails.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Custom 30K dataset not public / reproducibility concerns about dataset** — Removed per the hard rule against questioning availability of cited entities. The paper cites it; it exists.

- **FID computed on WebVid frames creates reference distribution mismatch** — All compared methods use the same FID reference distribution, making the comparison fair. The reference is not the intended output distribution, but for relative comparison this is standard practice (matching CameraCtrl's protocol). This is generic and not harmful to claims.

- **Baseline retraining may not have had equal optimization effort** — The paper explicitly retrains baselines with the same settings, which is stronger than most comparisons in this field. Demanding proof of equal optimization is a generic one-size-fits-all concern. The asymmetry also slightly favors baselines (since the proposed method is new), which is acceptable per the hard rule.

- **Algorithm 1 convergence not guaranteed** — Valid observation but standard for iterative vision algorithms. The practical effectiveness is demonstrated through results. This is a generic concern applicable to most iterative algorithms.

- **Missing computational cost/efficiency analysis** — Generic concern not standard for papers in this subfield. The adapter architecture is explicitly lightweight.

- **Missing confidence intervals / variance** — Single-run evaluation is the norm for large-scale video generation benchmarks. Nice-to-have, not a weakness.

- **Missing proof / appendix content** — Removed per hard rule; the parser strips appendix sections.

## Novel Insights

The paper makes a subtle but important observation about the relationship between control signal density and controllability: sparser signals (extrinsic matrices) force the network to learn a mapping from low-dimensional parameterization to pixel-level motion, which is inherently underdetermined and dataset-dependent; denser signals (point trajectories) provide the network with the pixel-level answer directly, reducing the learning burden and improving generalization. This insight — that the control signal should match the desired output granularity — is generalizable beyond camera control to other controllable generation settings.

## Suggestions

- Add two key ablations: (1) a version using extrinsic matrices as control signal (same data, same training) to directly test whether point trajectories vs. matrices explain the precision gains, and (2) a version using point trajectories without motion strength input to test whether the motion strength signal contributes adjustment capability. These would substantially strengthen the contribution claims.
- Report the training distribution of $m_\lambda$ values to justify the choice of 0, 200, 400, 600 test values and clarify whether 600 is in-distribution.
- Consider enforcing proper rotation constraints in Algorithm 1 (e.g., via Riemannian optimization on SO(3)) to eliminate the train-inference distribution mismatch in the R signal.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| AnimateDiff | Fx2SbBgcte | 7.0 | Higher: I2VControl has stronger camera-specific results, AnimateDiff has broader impact and ablations |
| Motion Guidance | WIAO4vbnNV | 7.0 | Higher: Motion Guidance has clean ablations, similar dense motion control idea; I2VControl has weaker evidence per component |
| CamTrol | KI1zldOFz9 | 5.8 | Comparable: training-free vs. training-based, both address camera control; I2VControl has much better quantitative accuracy but worse methodological rigor (no ablation) |
| CMG | rDRCIvTppL | 5.5 | Comparable: both target camera control improvements; I2VControl has stronger empirical results but CMG was rejected for insufficient evidence |
| Ctrl-V | n6To2wAOKL | 4.0 | Clearly above: I2VControl has much stronger results and more complete evaluation, though both lack ablation |
| TCIG | RFJGFrMvYj | 1.5 | Far above: I2VControl has genuine technical content and convincing results unlike TCIG's trivial combination |
| FVDM | XYuWS3nrw3 | 3.0 | Well above: FVDM has no qualitative results and missing baselines; I2VControl provides both |

I2VControl-Camera lands between CamTrol (5.8, accepted) and CMG (5.5, rejected). It has stronger empirical results than both, but the complete absence of ablation is a real methodological gap that prevents confident attribution of the improvements to specific design choices. The paper's qualitative demonstrations and quantitative camera precision are genuinely impressive, and the motion strength control addresses a real gap. The missing ablation keeps it below the 6.0 threshold that separated accepted from borderline in this topic area.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>