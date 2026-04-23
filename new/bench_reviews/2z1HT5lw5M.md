Now let me search for calibration anchors.Now I have sufficient information to write the final review. Let me synthesize everything.

---

## Summary

This paper introduces "trajectory attention," an auxiliary attention branch that operates along pixel trajectories across frames in video diffusion models, enabling fine-grained camera motion control. Rather than directly adapting temporal attention (which focuses on adjacent frames), the authors propose a two-branch architecture: trajectory attention runs in parallel with temporal attention, inherits its QKV weights, and adds its output as a residual. This design is trained efficiently (~24 GPU hours, 10k clips) on optical flow supervision and applied to camera motion control for images and videos, as well as first-frame-guided video editing.

---

## Strengths

- **Ablation strongly validates the core design (Table 3)**: Each design decision—separate add-on branch vs. tuning, weight inheritance from temporal attention, zero-initialized output projector—yields substantial and consistent improvements. ATE drops from 1.7812 (vanilla) → 0.3147 (+Tuning) → 0.0724 (+Add-on Branch) → 0.0396 (+Weight Inheriting), a ~45× reduction. This directly establishes the contribution.

- **Attention map analysis motivates the two-branch design (Fig. 2)**: Temporal attention concentrates on adjacent frames (scale 0–0.8, diagonal dominant), while trajectory attention distributes attention broadly across the full sequence (scale 0–0.18). This visualization provides genuine, concrete motivation for not merging the two mechanisms, rather than post-hoc rationalization.

- **Training efficiency is well-demonstrated (Section 5.1)**: Only 24 GPU hours on a single A100 with 10k clips. Training only the added trajectory modules rather than the full model is a practical advantage. Cross-frame-count generalization (trained on 12 frames, works on 25 frames) is a non-trivial result.

- **Orthogonal to existing methods, enabling complementary gains (Table 2)**: Combining with NVS_Solver's frame-wise injection improves all metrics (ATE: 0.3572 → 0.3371, FID: 129.3 → 112.2), confirming that trajectory attention addresses a distinct bottleneck from frame-wise optimization methods.

- **Extensibility to full 3D attention (Fig. 9)**: The approach transfers from decomposed spatial-temporal architectures to Open-Sora-Plan's unified 3D attention, demonstrating architectural generality beyond a single model class.

---

## Weaknesses

### Fatal
None.

### Major

- **Table 1 comparison protocol mixes frame counts and base models (Table 1)**: MotionCtrl is evaluated at 14 frames, Motion2V at 16 frames with AnimateDiff backbone, and CameraCtrl/NVS_Solver at 25 frames—while the proposed method appears in all three groups. The paper's footnote acknowledges the base-model mismatch for Motion2V but does not address the frame-count confound. Longer sequences are harder to control precisely; ATE/RPE comparisons across 14 vs. 25-frame settings are not directly interpretable as precision comparisons on equal footing. The paper's defense ("frame limitations of certain models") is reasonable in part, but the absence of any iso-model iso-frame comparison leaves the headline claim partially unsubstantiated. The improvements over MotionCtrl at 14 frames are huge and credible; the 25-frame gap over CameraCtrl (ATE 0.0396 vs. 0.0411) is marginal and more dependent on this protocol holding up.

- **ATE/RPE evaluation pipeline is insufficiently specified**: The paper reports ATE and RPE in "meters" but does not specify: (a) the depth estimator and how metric scale is resolved in Algorithm 3 Step 1; (b) the pose estimator applied to generated videos to recover camera trajectories for comparison against ground truth. Without specifying these, the numbers cannot be reproduced and cross-method comparisons may be dominated by implementation choices rather than actual precision. The phrasing "Estimate the depth map from I given camera pose parameters" in Algorithm 3 is ambiguous about whether monocular (scale-ambiguous) or metric depth is used. This directly affects whether the "meters" unit is meaningful.

### Minor

- **Video editing application lacks quantitative evaluation (Section 5.4)**: The abstract and introduction present first-frame-guided video editing as a primary generalizability demonstration, but Section 5.4 consists of exactly two qualitative examples (Fig. 8). No temporal consistency metric, warp error, CLIP similarity, or user study is provided. Two cherry-picked comparisons against AnyV2V and I2Vedit do not establish the claim that the method "excels in maintaining content consistency over large spatial and temporal ranges." This weakens the generalizability argument, though it is a secondary application.

- **Training–inference trajectory domain gap not discussed**: The model is trained on optical flow-derived trajectories (Section 5.1: Yang et al. 2023a) but at inference for camera control, trajectories are computed from camera poses + depth maps—a different pipeline. Whether the model genuinely generalizes across this domain gap or whether inference-time privileged camera information (precise known poses) contributes to the strong ATE results is not analyzed.

- **FID computed on 230 samples is statistically unstable**: Reliable FID estimates typically require thousands of samples. Single-digit FID differences (e.g., 103.5 vs. 108.7 in Table 1) at N=230 carry high variance and limited statistical weight.

### Trivial

- The description in Algorithm 3 Step 1 ("Estimate the depth map from I given camera pose parameters") is ambiguous. Clarifying whether this is a monocular estimator followed by scale alignment, or a metric depth model, would aid reproducibility.

- The ablation excludes "complete noise" outputs from the Vanilla variant's ATE/RPE statistics. This is noted in the text but a cleaner presentation would report the failure rate as a separate metric rather than silently excluding invalid outputs.

---

## Nice-to-Haves

- A trajectory sparsity ablation (ATE/RPE/FID as a function of trajectory coverage: 100%, 50%, 10%) would validate the claimed "sparse trajectories" support and clarify when the method degrades.

- An iso-model, iso-frame baseline comparison—e.g., running CameraCtrl or MotionCtrl at 25 frames with SVD—would substantially strengthen the paper's precision claim in Table 1.

- At least a user study or frame-consistency metric for the video editing application (Section 5.4) would convert a qualitative demonstration into a quantitative contribution.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic Issue 1 (classified as "Structural/Invalidating")**: The critic calls this evaluation "invalid by design." This is overstated. The field routinely compares methods under their published configurations (see CamTrol paper, where reviewer KI1zldOFz9 also noted non-uniform experimental settings and yet it was accepted). The paper does acknowledge the mismatch. The issue is real and merits a Major weakness, but it does not invalidate the paper's core architectural contribution, which is independently supported by the ablation (Table 3). Moved from Fatal to Major.

- **Harsh Critic Issue 2 (classified as "Structural/Broken metric")**: The ATE pipeline concern is valid but the "evaluation pipeline is completely broken" characterization is too strong. The evaluation protocol (however underspecified) is applied consistently across all methods, limiting the absolute bias from scale ambiguity. Kept as Major but without the "broken instrument" framing.

- **Circular evaluation risk (generated-video pose re-estimation)**: While noting that different methods may produce outputs that are easier/harder for a pose estimator to process is a valid theoretical concern, no evidence is provided that this systematically biases the comparison. Removed as speculative without evidence.

- **"Introduction paragraph 4 misleading claim"**: The critic notes that "does not require specially annotated datasets, such as camera pose annotations" could be misread. The paper's statement is correct in context (training uses optical flow, not camera poses). This is a minor presentation clarity issue at most, removed as a substantive weakness.

---

## Novel Insights

The most genuinely novel observation in this paper is the empirical demonstration that temporal attention and trajectory attention have fundamentally conflicting objectives: temporal attention must balance content consistency with natural dynamics, causing it to implicitly prioritize adjacent-frame coherence; trajectory attention, having access to known dynamics, can optimize purely for long-range alignment. The attention map comparison (Fig. 2) and the ablation table (Table 3) together quantify this conflict concretely—the performance collapse of the "vanilla" approach (ATE = 1.78, FID = 329.6) directly reflects this conflict, while the two-branch architecture resolves it at minimal training cost. This insight may have broader applicability: other forms of explicit motion control (object trajectories, human pose) could benefit from similar auxiliary-branch rather than direct-adaptation designs.

---

## Suggestions

1. Specify depth estimator and pose extraction pipeline completely (model name, scale alignment method) to make ATE/RPE reproducible.
2. Add one iso-model, iso-frame comparison to Table 1 (even for a single baseline at 25 frames with SVD).
3. Add at minimum a frame-consistency metric for Section 5.4 (video editing) computed over a larger test set.
4. Report trajectory sparsity ablation as it is central to the "sparse trajectories" design claim.
5. Report training budget per ablation variant in Table 3 to clarify whether Vanilla had equal optimization budget.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison to paper under review |
|---|---|---|---|
| 3DTrajMaster (3D trajectory control, video gen) | Gx04TnVjee.md | **6.75** (8,8,5,6), Accept-Poster | More thorough quantitative evaluation, novel dataset; slightly stronger evidence base than this paper |
| CamTrol (training-free camera control, SVD) | KI1zldOFz9.md | **5.80** (5,6,6,6,6), Accept-Poster | Also camera motion control for video; similar comparison protocol issues raised by reviewers; paper under review has stronger ablations |
| Camera Motion Guidance (transformer video) | rDRCIvTppL.md | **5.50** (5,5,5,7), Reject | Camera control in transformers; weaker ablations than this paper |
| FreeTraj (tuning-free trajectory) | CU7QfWJ6nC.md | **5.50** (5,6,5,6), Reject | Trajectory-based video motion, no training; comparable scope but different approach |
| VideoDiT | lvgsPjRtLM.md | **2.5**, Reject | Low-quality video generation; clearly weaker than this paper |
| SlowFast-VGen | UL8b54P96G.md | **7.5**, Accept-Spotlight | Stronger temporal video generation; more thorough evaluation; sets high bar |

**Positioning**: This paper is clearly above the low anchors (2.5 range). Among the camera-motion-control papers specifically, it sits above CamTrol (5.80) and rDRCIvTppL (5.50) due to its cleaner ablation, novel architectural insight, and measurably better results. It sits below 3DTrajMaster (6.75) because that paper provides a new dataset and more thorough evaluation. The major evaluation protocol concerns (Table 1 frame-count mixing, unspecified ATE pipeline, video editing qualitative-only) prevent a higher score.

**Evaluation summary:**
- *Originality*: Good. The auxiliary trajectory attention branch with weight inheritance is a technically clean and novel architectural design.
- *Importance of research question*: High. Camera motion control is a fundamental challenge in controllable video generation.
- *Claims supported*: Mixed. The core architectural claims are well-supported by Table 3. The comparison claims in Table 1 are partially undermined by the mixed frame-count/base-model protocol.
- *Soundness of experiments*: Moderate. The ablation is sound; the comparison protocol has real gaps.
- *Clarity*: Good overall, but key evaluation details are missing.
- *Value to research community*: Good. The architectural idea is reusable, training efficient, and demonstrated across multiple applications.

**Final score: 6.0** — Above the camera-control papers that were rejected or borderline accepted (5.5), in line with CamTrol (5.8) but accounting for the stronger ablation evidence. Below 3DTrajMaster (6.75) due to evaluation gaps. Accept as poster with revisions requested on evaluation protocol.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>