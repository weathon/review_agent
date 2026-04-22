Confirmed: The paper uses "Estimate the depth map" in Algorithms 3 and 4 but does not specify which depth estimation method is used or discuss sensitivity to depth errors. This is a valid concern.

Now let me also verify the question about whether the paper discusses trajectory density/sampling or the "copy weights" more carefully.

The paper does not discuss how performance scales with the number of trajectories L. This is a valid concern.

Now I have a comprehensive understanding of the paper. Let me compose the final review.

Key verified weaknesses:
1. **Ablation excludes failure cases** - Confirmed at line 292-293. The vanilla baseline has "complete noise" outputs that are omitted from evaluation. This is a real reporting gap.
2. **Video editing lacks quantitative evaluation** - Confirmed. Section 5.4 only has Figure 8 (qualitative).
3. **Full 3D attention results are only qualitative** - Confirmed at Section 5.6.
4. **Missing specification of depth estimation method** - Confirmed. Algorithms 3 and 4 say "Estimate the depth map" without specifying the method.
5. **Fragmented baseline comparisons** - Confirmed. Different baselines appear in different settings; no baseline appears across all configurations.

Key strengths:
1. Core idea is clean, well-motivated, and principled.
2. Strong ablation showing auxiliary branch design effectiveness.
3. Efficient training (24 GPU hours).
4. Good performance on the primary task (camera motion control on images).
5. Generalizes across frame ranges and to 3D attention architectures.

Let me calibrate the score now. 

Comparing against calibration anchors:
- **High-scoring papers (>7)**: AnimateDiff (avg 7.0, spotlight), Loopy (avg 8.0, oral), SlowFast-VGen (avg 7.5, spotlight) - These have strong, comprehensive evaluations and clear novelty.
- **Medium-scored papers (4-6)**: FreeTraj (avg 5.5, reject), PuppetMaster (avg 5.33, reject), Boosting Camera Motion Control (avg 5.5, reject), CamTrol (avg 5.8, poster) - These have some experimental gaps but reasonable methods.
- **Low-scored papers (<3)**: VideoDiT (avg 2.5, reject) - fundamentally flawed or missing key evaluations.

This paper has a solid core contribution (trajectory attention as an auxiliary branch) with quantitative support for the primary application, but some overclaiming on secondary applications (video editing, 3D attention) that lack quantitative evidence, and a significant ablation reporting gap (excluding failure cases). It's comparable to CamTrol (avg 5.8) or slightly above PuppetMaster (avg 5.33). It's clearly better than the truly weak papers (avg <3).

I'd place it around 5.5-6. It has a real and well-motivated contribution, strong ablation results, but the scope overclaiming and ablation reporting issue are notable weaknesses.</think>Now I have thoroughly read the paper and verified all claims from the reviewers. Let me synthesize the final review.

## Summary

The paper introduces *trajectory attention*, an auxiliary branch alongside temporal attention that performs attention along known pixel trajectories for fine-grained camera motion control in video generation. By sampling features along trajectories and applying multi-head attention in this trajectory-aligned space, then projecting back to the hidden state, the method injects a strong inductive bias for long-range consistency while preserving the original temporal attention's dynamics synthesis. Experiments on camera motion control for images and videos demonstrate improved precision and consistency, and the approach is shown to extend to first-frame-guided video editing and full 3D attention models.

## Strengths

- **Clean, well-motivated core idea**: The auxiliary-branch design resolves a genuine conflict between temporal attention (focused on local dynamics, Fig. 2a) and trajectory attention (needing long-range alignment, Fig. 2b). The inductive bias—explicitly connecting features along known camera trajectories—is intuitive and principled.

- **Strong ablation evidence for the primary design**: Table 3 shows a clear progression from vanilla adaptation (ATE 1.7812, FID 329.6) → tuning (0.3147, 139.2) → add-on branch (0.0724, 112.4) → with weight inheritance (0.0396, 103.5), decisively validating both the auxiliary-branch architecture and the weight-inheriting strategy.

- **Efficient training**: Only ~24 GPU hours on a single A100 with 10k video clips (Section 5.1), making the approach highly practical.

- **Quantitative improvements on the primary task**: On 25-frame image camera control (Table 1), the method achieves the best ATE (0.0396), RPE-rotation (0.1939), and FID (103.5), simultaneously outperforming CameraCtrl and NVS_Solver. On video camera control (Table 2), combining with NVS_Solver yields ATE 0.3371 vs. 0.5112 and FID 112.2 vs. 134.5.

- **Generalization across frame ranges and architectures**: Trajectory attention trained on 12-frame video transfers to 25-frame models (Section 5.1), and extends to full 3D attention (Open-Sora-Plan, Fig. 9), demonstrating the design is not architecture-specific.

## Weaknesses

### Fatal
None.

### Major

- **Ablation excludes failure cases, making vanilla-baseline numbers misleading**: Section 5.5 states that for the vanilla baseline, "some outputs exhibiting complete noise (we omit such invalid results during evaluation, otherwise calculating the statistic results is not feasible)." The ATE of 1.7812 for vanilla is computed on a cherry-picked subset that excludes catastrophic failures. This means the true gap between vanilla and the proposed method is larger than Table 3 reports—potentially much larger—since failed samples (arguably the most informative) are silently dropped. The improvement direction (auxiliary branch > no auxiliary branch) is still valid, but the specific magnitudes in Table 3 should be interpreted with caution. The paper should report the failure rate for each variant alongside the metrics.

- **Video editing claim lacks quantitative evaluation**: The abstract, introduction (Fig. 1), and Section 4.3 prominently present first-frame-guided video editing as a co-equal application, yet Section 5.4 provides only qualitative examples (Fig. 8) with no metrics—no consistency measures, no user study, no quantitative comparison. The claim that the method "excels in maintaining content consistency over large spatial and temporal ranges" (Section 5.4) cannot be established by visual inspection of two examples. This overclaims the scope of what the experiments convincingly support.

- **Full 3D attention extension is purely qualitative**: Section 5.6 demonstrates trajectory attention on Open-Sora-Plan with only Figure 9 (visual examples) and no quantitative metrics. While this is presented as supporting evidence for the method's generality, a claim of architectural generality deserves at least minimal quantitative backing.

### Minor

- **Missing specification of depth estimation method**: Algorithms 3 and 4 both begin with "Estimate the depth map" but do not specify which depth estimator is used, how accurate it needs to be, or how depth errors propagate to trajectory errors. Since the entire pipeline depends on estimated depth for trajectory extraction, this is an important omitted implementation detail.

- **Fragmented baseline comparisons**: Table 1 compares at most three baselines across three different frame counts (14, 16, 25), with no single baseline appearing in all conditions. The 16-frame Motion2V comparison uses a different base model (AnimateDiff vs. SVD), which the paper marks with an asterisk but does not fully address. This makes it harder to draw apples-to-apples conclusions, though the overall trend of improvement is still clear within each setting.

- **No analysis of trajectory density sensitivity**: The method's performance as a function of the number of sampled trajectories L (which controls the compute-precision trade-off) is not studied. Given the claim that "sparse trajectories" are supported (Section 1), understanding the performance-compute frontier would strengthen the paper.

### Trivial
None.

## Nice-to-Haves

- Quantitative evaluation for the video editing application (e.g., frame-wise CLIP similarity, LPIPS, or a standardized editing benchmark) would substantiate this headline claim.
- Reporting the failure rate alongside ablation metrics in Table 3, or defining a protocol (e.g., maximum penalty for failed samples), would make the ablation fully transparent.
- At least one quantitative experiment on a full 3D attention model would solidify the generality claim.
- Sensitivity analysis to depth estimation quality or trajectory density L would clarify practical operating conditions.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Motion2V comparison is confounded by different base models"**: While true (noted with asterisk), this actually makes the comparison *harder* for the authors' method since they use SVD rather than AnimateDiff specifically designed for motion—so this is not an asymmetry favoring the authors. Removed as a weakness per the rule about unfair comparisons that favor the baseline.

- **"Copy Weights from output projector to Q projector is under-specified"**: This appears in the parsed figure description but may be a parser artifact. The text (Section 3.4) clearly states "we initialize the weights of the QKV projectors with those from temporal attention layers," which is the correct description. The figure description's mention of connecting "Out Projector" to "Q Projector" is likely a parser misinterpretation of the diagram. Removed.

- **"The ATE metric citation (Goel et al., 1999) is unusual"**: This is a nitpick about citation style/convention. The citation exists and is presumably appropriate for the ATE metric. Removed per ruling on nitpicks.

- **"Four distinct metrics claim lists only three"**: The paper states "four distinct metrics" and lists ATE, RPE (translation), RPE (rotation), and FID—four values that correspond to the four columns in their tables. This is not an error. Removed as factually incorrect.

- **"No variance, standard deviation, or confidence intervals reported"**: Single-run evaluation without variance is standard practice in large-scale video generation benchmarks. Removed per soft rule (nice-to-have, not a core flaw).

- **"Training on only 10k video clips raises generalization concerns"**: The paper demonstrates generalization to 25-frame models from 12-frame training, and to a different architecture (Open-Sora-Plan). The empirical results speak for themselves. This is a generic concern that doesn't conflict with the evidence presented. Removed as a one-size-fits-all weakness.

- **"The novelty lies in the trajectory sampling/projection, not the auxiliary branch concept itself"**: The paper openly acknowledges inspiration from ControlNet (Zhang et al., 2023a) in Section 3.3. The contribution is the design of the trajectory attention mechanism (sampling, projection, masking), not claiming the two-branch concept itself is novel. Removed as a strawman weakness.

- **"Averaging in back-projection could dilute signal in dense trajectory regions"**: The paper includes validity masks (Algorithm 1, step 6; Algorithm 2, step 4) to handle occlusions and out-of-bounds trajectories. The averaging in back-projection (Algorithm 2, line 6) is a standard approach and no evidence is presented that this causes issues in practice. Removed as unsubstantiated.

- **"Attention map visualization in Fig. 2 shows only one attention head/layer"**: This is a presentation nitpick. The visualization serves its illustrative purpose for the core argument. Removed as trivial.

## Novel Insights

The key insight validated by the ablation is that trajectory attention *must* be decoupled from temporal attention to be effective (Table 3: 46× ATE reduction from vanilla to add-on branch). This is not just about adding more capacity—it is about resolving a functional conflict: temporal attention prioritizes short-range dynamics (concentrating on adjacent frames), while trajectory attention needs long-range alignment. Simply fine-tuning temporal attention for trajectory purposes fails catastrophically (complete noise outputs), confirming that the inductive bias of explicit trajectory-aligned attention is doing more than just providing an extra learning channel.

## Suggestions

- Report the percentage of completely failed outputs for each ablation variant in Table 3 alongside the metrics, so the true performance gap is visible.
- Add at least one quantitative metric for the video editing application (e.g., frame-wise CLIP similarity between generated frames and the edited first frame) even if on a small benchmark.
- Specify the depth estimation method used (e.g., MiDaS, DPT, etc.) and briefly discuss sensitivity to depth errors.

## Calibration Anchors

| Paper | Score | Comparison |
|-------|-------|------------|
| Loopy (avg 8.0, Oral) | 8.0 | Much stronger: end-to-end trained, comprehensive evaluation, clear system-level contribution |
| AnimateDiff (avg 7.0, Spotlight) | 7.0 | Stronger: plug-and-play contribution with broad empirical validation and clear practical impact |
| SlowFast-VGen (avg 7.5, Spotlight) | 7.5 | Stronger: comprehensive experimental design, multiple quantitative evaluations |
| CamTrol (avg 5.8, Poster) | 5.8 | Similar: both propose camera control methods, have some experimental gaps. This paper has a cleaner methodological contribution but overclaims scope |
| PuppetMaster (avg 5.33, Reject) | 5.33 | Weaker: similar auxiliary branch concept but less grounded evaluation and more novelty concerns |
| FreeTraj (avg 5.5, Reject) | 5.5 | Comparable: both are trajectory-control methods for video generation, similar experimental gaps |
| Boosting Camera Motion Control (avg 5.5, Reject) | 5.5 | Comparable: both address camera control in video DiTs; this paper has a more novel mechanism |
| CameraCtrl (N/A, baseline in this paper) | N/A | This paper clearly outperforms CameraCtrl and NVS_Solver on the primary task |
| UniEdit (avg 4.75, Reject) | 4.75 | Weaker: auxiliary branch but overclaimed and limited evaluation, novelty concerns |
| VideoDiT (avg 2.5, Reject) | 2.5 | Much weaker: fundamentally unclear presentation, missing key details |

This paper is clearly above the reject-tier anchors (UniEdit, VideoDiT) but below the strong accept anchors (AnimateDiff, Loopy, SlowFast-VGen). Compared to medium-scoring camera/motion control papers (CamTrol 5.8, PuppetMaster 5.33, FreeTraj 5.5), it has a cleaner and more novel methodological contribution (the auxiliary-branch trajectory attention is a principled design validated by strong ablation), but shares the flaw of overclaiming scope (video editing and 3D attention without quantitative support). The ablation reporting issue (excluding failures) is significant but doesn't invalidate the core result. Placing it slightly above the medium anchors.

## Score Rationale

The core contribution—trajectory attention as an auxiliary branch—is well-motivated, validated by a convincing ablation, efficient to train, and produces clear improvements on the primary task (camera motion control on images). The main weaknesses—overclaimed scope (video editing, 3D attention without quantitative support) and the ablation failure-exclusion—are significant but do not undermine the core contribution on the primary task. Relative to anchors, this sits in the 5.5–6.0 range: above borderline reject papers (PuppetMaster 5.33, FreeTraj 5.5) due to the stronger methodological contribution and cleaner ablation, but below clear accept papers (CamTrol 5.8+) whose experiments more fully support their claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>