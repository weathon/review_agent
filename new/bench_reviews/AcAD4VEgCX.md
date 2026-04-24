Now let me search for calibration anchors.Now I have enough information to write the final review. Let me compile it.

---

## Summary
I2VControl-Camera proposes a camera control method for image-to-video generation using dense 2D point trajectories in the camera coordinate system as control signals (instead of sparse extrinsic matrices or Plücker embeddings), combined with a scalar "motion strength" signal derived from higher-order trajectory components. The paper addresses two simultaneous challenges: camera control precision and adjustable subject motion dynamics. A practical data pipeline (Algorithm 1 + L-BFGS fitting with Unidepth + SpatialTracker) extracts these signals from raw RGB video. Experiments on RealEstate10K and a custom movable-object dataset show substantial improvements in RotErr/TransErr over retrained MotionCtrl and CameraCtrl baselines.

---

## Strengths

- **Dense point trajectory control signal achieves large quantitative gains**: Table 1 shows RotErr drops from 2.66 (MotionCtrl) / 1.26 (CameraCtrl) to 0.53 for the proposed method on RealEstate10K, with TransErr improving from 12.70/21.60 to 9.72. These are not marginal gains; they represent a meaningful jump in camera control precision that holds across both static (Table 1) and dynamic (Table 2) test sets.

- **Motion strength provides a genuine and measurable new capability**: Table 2 demonstrates that MSC can be smoothly varied from 18.96 (Ours-0) to 47.70 (Ours-600), spanning and exceeding the range of all baselines (32.69–42.28). This is the first method to explicitly parameterize subject motion strength as a user-tunable scalar alongside camera control.

- **Practical, annotation-free data pipeline**: Algorithm 1 iteratively partitions static/dynamic regions via trajectory linear fitting and uses off-the-shelf tools (Unidepth, SpatialTracker), enabling training on arbitrary in-the-wild RGB video without manual annotation. This is a concrete engineering contribution that makes the approach scalable.

- **Fair experimental comparison**: Section 4.3 explicitly states that MotionCtrl and CameraCtrl were retrained using the same experimental settings, base model (Magicvideo-V2), and resolution, reducing confounds from architecture and training environment differences.

---

## Weaknesses

### Fatal
None.

### Major

- **No ablation isolating the control signal representation from other pipeline differences.** The central claim is that dense point trajectories are superior to extrinsic matrices and Plücker embeddings as control signals. However, the full system simultaneously differs from baselines in: (a) control signal type, (b) training data pipeline quality (depth-lifted 3D fitting vs. SLAM/SfM on dynamic video), and (c) the added motion strength mechanism. Without an ablation holding (b) and (c) fixed while varying only the control signal type, the large RotErr/TransErr improvements (Table 1) cannot be attributed specifically to point trajectories. An ablation training the proposed adapter with extrinsic matrices as control signals (on the same 30K dataset) is necessary to support the core representational claim.

### Minor

- **Theoretical framework gap between Section 3.1 and Algorithm 1.** Section 3.1 defines R_λ via the Jacobian difference (Eq. 6) and claims "unique" R_λ, t_λ from the Maclaurin expansion. However, this yields only a local first-order approximation at p=0 — uniqueness is only established for this specific definition. The practical algorithm (Eq. 12 and Algorithm 1) instead estimates (R_λ, t_λ) via nonlinear reprojection error minimization over the static subset Ω_S, which is a different procedure. The gap is never acknowledged. The mathematical scaffolding motivates the concept but does not derive the practical algorithm; this should be stated more clearly.

- **Motion strength inference-time calibration is undocumented.** The paper experiments with values {0, 200, 400, 600} but never reports the training distribution of computed m_λ values (their range, mean, percentiles). Without this, it is unclear what these scalar values mean relative to the training data, making the method difficult to reproduce or apply to new domains.

- **CamCo is excluded from quantitative comparison without justification.** Section 2.3 cites CamCo (Xu et al., 2024) and notes it has "small motion dynamic," which is directly relevant to the paper's claimed contribution. No reason is given for its exclusion from Tables 1–2. Including it would strengthen the comparative picture, especially on the movable-object dataset.

- **FID metric is uninformative for the paper's core claim.** FID is computed against 2000 randomly sampled WebVid frames, a reference distribution unrelated to camera-controlled video. The differences between methods are tiny (e.g., 155.01 vs. 156.69 in Table 1), and their significance is not established. This metric adds noise rather than signal to the evaluation.

### Trivial

- **Figure 6 demonstrates only extreme motion strength values (0 and 600).** While Table 2 reports intermediate values, the visualization section would benefit from showing smooth interpolation (e.g., 0, 200, 400, 600) for the same scene to demonstrate monotone and continuous control.

---

## Nice-to-Haves

- Ablation: with vs. without motion strength signal to assess its impact on camera control precision at non-zero settings.
- Generalization demonstration: apply the adapter to a second I2V base model (e.g., SVD) to validate the claimed model-agnostic design.
- Analysis of depth estimation error impact on control quality in challenging scenes (reflective surfaces, outdoor distances).
- Human evaluation study comparing motion naturalness across motion strength settings to complement the automatic MSC metric.

---

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **"Plücker embeddings offer no additional information compared to camera matrix"**: The harsh critic flags this as misleading because Plücker embeddings do provide a dense per-pixel encoding. However, the paper's point (that Plücker embeddings lack input-image scene content information, in contrast to point trajectories) is defensible and is the actual argument being made. The precision is slightly off but the underlying claim is not wrong. Removed as a weakness because it is a legitimate, defensible simplification in context.

- **Claim about bias in RotErr/TransErr pose estimation favoring cleaner backgrounds**: Speculative and not grounded in evidence from the paper. Removed.

- **Concern about training data not being identical for baselines**: The paper states "retrain MotionCtrl and CameraCtrl using the same experimental settings," which reasonably encompasses the same training data. This is not an explicit confirmation, but the concern is speculative rather than substantiated. Weakened to a note under minor and not promoted to major.

---

## Novel Insights

The most genuinely novel insight is the explicit decomposition of video point trajectories into a linear rigid component (camera motion) and a higher-order nonlinear component (subject motion), and the derivation of a scalar motion strength metric from the integrated speed of the nonlinear component (Eq. 9). This framing unifies camera control and subject dynamics under a single mathematical framework and enables independent user control over both via a single adapter. The data pipeline that extracts this decomposition from raw video without manual annotation is an underappreciated contribution that makes the approach practical at scale. The conceptual insight that dense, content-aware point trajectories are inherently a better control signal than scene-agnostic camera parameter encodings (whether matrices or Plücker embeddings) is clear and well-motivated, even if not yet fully ablated.

---

## Suggestions

1. **Add the critical missing ablation**: Train a version of the proposed adapter with standard extrinsic matrices as control signals (same 30K dataset, same architecture) and compare RotErr/TransErr with the point trajectory version. This directly tests the core claim.
2. **Document training distribution of m_λ**: Report mean/std/range of computed motion strength values in the training set, and explain how user-provided values (0–600) map to this distribution.
3. **Clarify the theoretical-practical gap**: Add a sentence in Section 3.3 explicitly stating that Eq. 12 is a principled practical estimator of (R_λ, t_λ), distinct from the Jacobian-based theoretical definition.
4. **Include or explain exclusion of CamCo**: Either add CamCo as a baseline or provide explicit justification for its omission.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Score | Comparison |
|------|-----------|------------|
| `/home/wg25r/review_agent/human_reviews/KI1zldOFz9.md` | 5.80 (Accept, Poster) | Training-free camera control for video; similar problem scope but no dedicated data pipeline; accepted despite scene consistency issues |
| `/home/wg25r/review_agent/human_reviews/rDRCIvTppL.md` | 5.50 (Reject) | Camera motion control for DiT video models; similar topic; rejected partly due to incomplete analysis and overclaiming; the paper under review has stronger empirical gains but a similar ablation gap |
| `/home/wg25r/review_agent/human_reviews/Gx04TnVjee.md` | 6.75 (Accept, Poster) | 3D trajectory control for multi-entity motion in video generation; comparably strong results and cleaner methodology than the paper under review |
| `/home/wg25r/review_agent/human_reviews/m8Rk3HLGFx.md` | 5.86 (Accept, Poster) | Multi-camera video generation with similar plug-and-play adapter scope |
| `/home/wg25r/review_agent/human_reviews/n6To2wAOKL.md` | 4.00 (Reject) | Bounding-box controlled object motion for video; weaker contribution and missing baselines; clearly below the paper under review |
| `/home/wg25r/review_agent/human_reviews/TTWxMAwS6n.md` | 4.33 (Reject) | Video generation controllability adapter; weaker results and unclear contributions |

**Reasoning**: The paper sits between rDRCIvTppL (5.50, rejected for incomplete analysis) and Gx04TnVjee (6.75, accepted for strong 3D trajectory approach). The contribution here is genuine and the quantitative improvements are larger than typical, but the missing ablation isolating the key representational claim is a significant methodological gap. The comparison baselines are retrained fairly. The overall contribution (joint camera + motion control, data pipeline, strong results) is comparable to accepted poster papers in the 5.5–6.0 range. The missing ablation prevents a higher score. I assign **5.5**.

**Axis evaluations:**
- *Originality*: Good — decomposing trajectories into rigid+nonlinear components and adding a motion strength dial is a novel framing.
- *Importance of research question*: High — simultaneous camera and subject motion control is practically valuable and underexplored.
- *Claims well supported*: Partially — the system-level performance is well supported; the specific representational superiority of point trajectories is not isolated.
- *Soundness of experiments*: Moderate — fair retraining of baselines, but missing the key representation ablation; FID metric weakly chosen.
- *Clarity of writing*: Good — paper is generally clear; theory-practice gap is unacknowledged.
- *Value to research community*: Moderate-to-high — the data pipeline and motion strength concept are immediately usable.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>