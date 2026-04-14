=== CALIBRATION EXAMPLE 18 ===

# Final Consolidated Review
## Summary
SoloPose introduces a one-stage many-to-many spatio-temporal transformer for 3D human pose estimation from video, pairing a CLIP-based spatial encoder with a Swin-transformer temporal module and a novel kinematic GMM heatmap (HeatPose). The paper also contributes the 3D AugMotion Toolkit, which aligns four public mocap datasets (Human3.6M, MADS, AIST Dance++, MPI INF 3DHP) into a unified coordinate system to produce the Human7.1M dataset. Experiments are reported on both Human3.6M and the in-house Human7.1M benchmark.

---

## Strengths

- **Kinematic-aware 3D heatmap with demonstrated ablation gains.** HeatPose encodes the direction and distance to adjacent keypoints as side Gaussian distributions, and the ablation in Table 2 shows a 15.3% / 27.2% reduction in MPJPE / P-MPJPE relative to training the same model with a plain MSE loss. This is a specific and non-trivial gain attributable to the heatmap design.

- **Dataset alignment methodology that targets a real problem.** Figure 1 illustrates a genuine multi-camera coordinate misalignment in Human3.6M that has been largely overlooked in prior work. The use of anatomically grounded reference keypoints (shoulders, pubis) with Kabsch alignment provides a principled, reproducible (in principle) pipeline, and the resulting Human7.1M dataset provides a richer diversity of motions (martial arts, dance, sports) absent from any single source dataset.

- **Demonstrating one-stage superiority over two-stage under its training regime.** Table 2 shows SoloPose (N=30, trained on Human7.1M) achieves 26.0/20.5 MPJPE/P-MPJPE on Human3.6M testing, clearly outperforming all CPN-based two-stage methods (best: FinePOSE w/ CPN: 31.9/25.0). While confounds exist (see Weaknesses), this result at least demonstrates that a direct image-to-3D approach can be competitive despite using shorter temporal windows.

---

## Weaknesses

### Fatal
*(None that fully invalidate the entire paper, but the major issues below collectively undermine the core empirical claims.)*

### Major

- **The Human7.1M comparison is methodologically unsound and the primary claim of "SOTA performance" rests on it.** All baseline models (P-STMO, STCFormer, KTPFormer, FinePOSE) are trained *only* on Human3.6M and then tested on Human7.1M — a dataset drawn from distributions (AIST Dance++, MADS, MPI INF 3DHP) they were never exposed to. SoloPose, by contrast, is trained on the full Human7.1M training split. The performance gap is thus a straightforward out-of-domain generalization failure for the baselines, not evidence that SoloPose's architecture is superior. The paper explicitly states (Section 5.3) that baselines are "pre-trained on Human3.6M training dataset," yet interprets the Table 2 Human7.1M numbers as architectural evidence. The correct experiment would train all baselines on the same Human7.1M training set. Without this, the Human7.1M results provide virtually no architectural insight.

- **On the only fair benchmark (Human3.6M), the SoloPose architecture underperforms SOTA when trained equivalently.** Table 2 row "SoloPose only trained on Human3.6M" gives 38.9/29.9 MPJPE/P-MPJPE — worse than KTPFormer w/ CPN (33.0/26.2) and FinePOSE w/ CPN (31.9/25.0), both also trained only on Human3.6M. Section 5.4.2 claims the model "is more effective than current SOTA methods" but compares only against P-STMO (42.1) and STCFormer (40.5) while silently omitting KTPFormer and FinePOSE. This selective comparison is misleading. The data strongly suggest that the performance gains in the full SoloPose system come almost entirely from Human7.1M's extra training data, not from the proposed architecture.

- **The "cost-efficient" claim is entirely unsupported.** SoloPose processes N=30 raw video frames through a CLIP visual encoder for each spatial feature extraction step, while the competing two-stage methods operate on lightweight 2D skeleton coordinates. Table 1 presents only a binary feature checklist with no FLOPs, parameter counts, inference latency, or memory measurements. Given that CLIP processing on raw images is substantially more expensive than keypoint lifting, the claim of efficiency advantage cannot stand without empirical support. This is a core stated contribution that is not validated.

- **Arithmetic errors in the results section damage credibility.** Section 5.3 states "MPJPE and P-MPJPE are 22.7% and 21.9% lower than FinePOSE with CPN" on H3.6M, but Table 2 gives (31.9−26.0)/31.9 ≈ 18.5% and (25.0−20.5)/25.0 = 18.0%. Similarly, "14.9% and 21.8% lower than FinePOSE" on Human7.1M computes as (26.1−22.7)/26.1 ≈ 13.0% and (20.6−16.9)/20.6 ≈ 18.0%. None of the four stated percentages match Table 2. These are not rounding differences; they are errors of several percentage points.

- **Constant *c* in HeatPose (Eq. 6) is never specified, making the method not reproducible.** The number of side Gaussian distributions is $N_s = D(P_t, P_a)/c$, where $c$ is described only as "a constant" — its value is not given in the paper, the appendix, or the promised code repository. This central hyperparameter determines the entire kinematic structure of HeatPose and must be disclosed.

### Minor

- **CLIP as spatial encoder is an unmotivated design choice.** Section 4.1 applies CLIP (pretrained on image-text pairs) to extract spatial pose features, but gives no justification for why CLIP is appropriate over a standard ViT or ResNet, does not state whether CLIP is frozen or fine-tuned during training, does not specify the input image resolution or how person crops are obtained, and does not analyze whether text-image pretraining representations transfer usefully to pose estimation. This also raises the question of whether obtaining a person crop requires a separate detection step, which would partially negate the "one-stage" framing.

- **Feature map size 1×200×192 is unexplained.** Standard CLIP ViT-B/16 or ViT-L/14 outputs do not produce this shape, yet no reshaping or projection layer is described. The architecture description is incomplete.

- **The Kabsch transformation is computed only for a key frame; propagation to all other frames in a sequence is not described.** Section 3.3 explains how the rotation R and translation T are computed from a key frame, but Section 3 is silent on whether the same rigid transform is applied to all non-key frames, how temporal discontinuities are handled, and whether the transform is shared across clips from the same video.

- **Cross-entropy loss claim contains a technical error.** Section 4.2 states "using a cross-entropy loss function avoids non-convex problems." Cross-entropy loss on neural network parameters is equally non-convex. The correct argument is that cross-entropy produces better-conditioned gradients for probabilistic targets and better handles noisy ground truth — which may be true but is not what the paper asserts.

- **Decoding from the shared volumetric representation is inadequately described.** HeatPose places all keypoints into a single shared 3D volume (Eq. 8). Section 4.2 only mentions "find the maximum of voxels' probability to convert HeatPose back to 3D coordinates," but does not explain how individual joint coordinates are disambiguated from a single volume. For poses where joints are spatially close (e.g., wrist near torso in crouching poses), overlapping Gaussians from different joints will create ambiguous modes.

- **Notational typo in Eq. 1.** The variable $p_{sr}$ is defined twice as "the left shoulder key point" (it should be the right shoulder). This contradicts the coordinate system definition and undermines confidence in the formalization.

- **N=30 vs N=243 temporal window confound is not ablated.** SoloPose uses a 30-frame window while all SOTA baselines use 243 frames. The effect of temporal context length on performance is not isolated — an ablation at N=30 for the baselines or N=243 for SoloPose would disentangle temporal window from architecture.

- **Human7.1M test split uses random frame-level selection with step-16 sliding windows.** Adjacent clips share most frames, so random clip-level splitting may create near-duplicate train/test clips. A subject- or video-level split would be more rigorous to ensure the test set probes genuine generalization.

### Tiny

- **k-means with 3 clusters for key frame selection is unjustified.** Why 3 clusters, and how are action sequences lacking upright frames (e.g., martial arts tumbling, floor exercises) handled? The MADS dataset explicitly includes such actions.

- **Section 5.4.2's ablation conclusion is overstated.** The claim that SoloPose only trained on H3.6M "demonstrates that our SoloPose model is more effective than current SOTA methods" is based on comparisons only against P-STMO and STCFormer, not KTPFormer or FinePOSE. The text should be corrected.

- **The σ_side quadratic scaling** ($\sigma_{side}^i = i^2 \cdot \sigma_{main}$) causes side distributions to grow very rapidly with i; for long limbs with many transitional points, these may overlap with main Gaussians of non-adjacent joints, introducing spurious probability mass into the shared volume.

---

## Nice-to-Haves

- **Train SOTA baselines on Human7.1M** to provide a fair architectural comparison. Without this, the paper's central experimental table is uninterpretable as an architectural evaluation.
- **Ablate the many-to-many design** with a many-to-one variant of SoloPose using the same backbone, to directly test the temporal modeling claim from the Introduction.
- **Compare against one-stage image/video-to-3D baselines** (e.g., MeTRAbs, Geometry-Aware methods listed in Table 1) in the results table. Table 1 acknowledges these methods exist but they are absent from Table 2.
- **Report per-action errors on Human3.6M** to allow standard comparison with existing work.
- **Visualize AugMotion alignment before and after** across datasets (H3.6M, AIST, 3DHP) to demonstrate that the universal coordinate system succeeds across diverse skeleton topologies.
- **Provide a keypoint schema mapping table** showing how the 17/18 joints of H3.6M, MADS, AIST, and 3DHP are harmonized into a unified skeleton before applying Kabsch alignment.
- **Evaluate on an in-the-wild benchmark** (e.g., 3DPW, MPI INF 3DHP test set) to test whether the augmented dataset improves real-world generalization.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Critic: Circular dependency in Eq. 1.** The ratio $M_s / M_{sp}$ is computed over raw data in any original coordinate system (Euclidean distances are coordinate-system invariant), and then used to fix the target coordinates of the universal system. There is no logical circularity — the ratio uses distance measurements that are the same regardless of origin and orientation.

- **Critic: Abstract claim of "superior results" is wholly misleading.** The abstract should be read as a composite claim: SoloPose is superior on Human7.1M and superior to CPN-input baselines on Human3.6M. Both are literally true from Table 2 (with the caveats raised in Major Weaknesses). While the comparison is unfair in ways the abstract does not disclose, the claim is not fabricated outright.

- **Critic: "Two-stage error propagation argument contradicts the experimental design."** Testing with both CPN and GT input is precisely how the paper *demonstrates* the error propagation gap (the GT results show what 2D-lifting models could achieve in principle). This is standard practice in the lifting literature, not a contradiction.

- **Critic: Absence of missing related works.** Per review policy, claims about missing citations are excluded because external sources cannot be confirmed.

- **Critic: No limitations section exists as a formal section.** This is a formatting/style point rather than a substantive flaw.

---

## Novel Insights

The most genuinely interesting observation that emerges from the paper — and that the reviewers do not adequately foreground — is the coordinate system misalignment within Human3.6M itself (Fig. 1): the same pose from different cameras, when converted to the dataset's own global coordinates, produces non-overlapping skeletal representations. If confirmed rigorously, this is a data-quality finding about arguably the most widely used 3D pose benchmark, with implications for all models trained on it. The AugMotion methodology's response (using Kabsch alignment on upright key frames) is reasonable but does not include a quantitative analysis of residual RMSD after alignment; demonstrating that the alignment actually works — rather than just claiming it — would significantly strengthen the paper. The kinematic heatmap idea (encoding limb topology into the distributional structure of the heatmap, not just the peak) is a principled extension of volumetric heatmaps, and the 15% ablation gain is encouraging, though the method needs cleaner exposition (fixed vs. dynamic N_s, shared vs. per-joint volumes) to be fully compelling.

---

## Suggestions

1. **Retrain at least one strong baseline (e.g., FinePOSE) on Human7.1M training data** and compare against SoloPose on the Human7.1M test set. This is the minimum required to make any architectural claim from those results.

2. **Disclose the value of constant *c* and all other HeatPose hyperparameters** (volume resolution w×h×d, σ_main, N_s per joint pair) in a reproducibility section or appendix.

3. **Clarify the CLIP encoder usage**: is it frozen, partially fine-tuned, or fully fine-tuned? How are person crops obtained (bounding box detector, fixed crop, full frame)? What resolution is used? If crops require a person detector, discuss the "one-stage" framing carefully.

4. **Correct the arithmetic in Section 5.3 and Section 5.4.2**, and compare the H3.6M-only ablation against *all* listed SOTA models, not just P-STMO and STCFormer.

5. **Add computational efficiency numbers** (parameters, FLOPs per clip, or inference time on the same hardware) against at least FinePOSE and STCFormer to substantiate or retract the "cost-efficient" claim.

6. **Describe how the Kabsch transformation computed for a key frame is applied to all remaining frames** in the same video sequence, including any handling of temporal gaps or inconsistency.

7. **Clarify the HeatPose decoding step**: specify whether soft-argmax or hard argmax is used per-joint, and explain how individual joints are localized from a single shared volume when their Gaussians overlap.

---

**Summary evaluation:**

- **Novelty:** Moderate. The combination of one-stage video-to-3D, kinematic GMM heatmap, and multi-dataset alignment is a reasonable package, though each individual component builds on established techniques. The kinematic heatmap formulation is the most distinctive piece.
- **Technical soundness:** Weak. Multiple undisclosed hyperparameters, an unjustified and under-specified spatial encoder, a technically incorrect loss characterization, and an incompletely described decoding procedure limit reproducibility and theoretical clarity.
- **Empirical support:** Weak. The primary result table is methodologically unsound as an architectural comparison. The only fully controlled comparison (H3.6M-only training) shows the SoloPose architecture is inferior to the strongest baselines. Arithmetic errors in the results section further reduce confidence.
- **Significance:** Moderate conditional. Human7.1M, if released with proper documentation and the alignment code, would be a useful community resource. HeatPose shows an intriguing ablation gain. Neither contribution is convincingly demonstrated to the standard expected at ICLR.
- **Clarity:** Fair. The overall narrative is readable but the HeatPose section lacks critical implementation details, the efficiency argument is asserted without evidence, and the results section contains arithmetic inconsistencies that confuse interpretation.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 6.0]
Average score: 4.2
Binary outcome: Reject
