## Summary

This paper proposes Union-over-Intersections (UoI), a reformulation of object detection in which box proposals regress only their intersection with the ground truth rather than the full box, and final predictions are formed by taking the coordinate-wise union of these intersections followed by a second refinement stage. The method is evaluated on COCO across Faster R-CNN, Mask R-CNN, Cascade R-CNN, YOLOv3, and Deformable DETR, consistently yielding modest mAP improvements and clear localization gains.

## Strengths

- **Conceptually clean and well-motivated idea.** The paper identifies a genuine localization bottleneck—proposals must extrapolate beyond their visual scope when regressed to full ground-truth boxes—and proposes a simple, intuitive decomposition into intersection regression + union merging.
- **Broad consistent empirical gains.** Table 1 shows mAP improvements of +0.5 to +1.1 across five diverse architectures and multiple backbones, suggesting the reformulation is not an artifact of a single detector family.
- **Improvements are isolated to localization without harming classification.** Table 5 shows classification accuracy is virtually unchanged (76.4% → 76.5%) while localization mIoU jumps sharply (53.7% → 64.4%), and Table 6 corroborates this via the LRP metric, where localization error drops from 17.2 to 12.7 with stable false positive/negative rates.
- **Orthogonality to losses and grouping strategies.** Table 2 demonstrates additive improvements when combining UoI with GIoU, DIoU, and Alpha-IoU losses, and Table 4 shows gains persist across NMS, Cluster-NMS, and Soft-NMS.
- **Evidence that intersection targets ease optimization.** Figure 3(b) shows lower training loss when regressing to intersections rather than full ground truth, supporting the claim that the task is genuinely easier to optimize.

## Weaknesses

### Fatal
None.

### Major

- **Missing ablation isolating intersection regression from multi-box merging.** The paper’s core mechanistic claim—that intersection targets are superior because avoiding extrapolation is easier—is conflated with a second change: retaining and merging up to five boxes per object at inference instead of winner-takes-all selection. Table 8 shows that adding only a second regression head to the baseline (with winner-takes-all post-processing) yields no benefit (37.4 mAP), but it does **not** test the critical control of standard full-GT regression targets combined with the same proposal grouping, coordinate-wise min/max merging, and second-stage refinement. Without this ablation, it remains possible that much of the gain comes from retaining multiple spatial hypotheses at test time rather than from the intersection targets themselves. Table 7’s comparison to box voting (37.5 vs. 38.1) is informative but does not fully close this gap, because box voting averages regressed outputs with confidence weighting rather than merging raw proposals via min/max.
- **Query-based applicability claim is structurally flawed and underspecified.** The abstract and introduction claim the approach integrates “seamlessly” into query-based detectors with “minimal modifications.” However, Deformable DETR has no input proposals to intersect with ground truth; the authors instead “divide the ground truth into quadrants, assign queries to specific parts for part-based regression” (Section 4.2). This is a qualitatively different workaround that abandons the literal intersection mechanism. The adaptation is described in a single vague sentence with no explanation of how the Hungarian matching cost is modified, how many queries are assigned per quadrant, or whether quadrants are axis-aligned and fixed. Because the DETR result is presented as evidence for broad architectural applicability, this claim is misleading.

### Minor

- **No training variance or statistical significance reported.** All quantitative results (Tables 1–8) are reported as single scalars. Gains are modest (typically +0.5 to +0.9 mAP on COCO), and without standard deviations across multiple seeds it is difficult to assess whether the improvements are robust or within training noise. The consistency across architectures partially mitigates this concern, but variance estimates would materially strengthen the empirical argument.
- **Large mIoU gain with small AP$_{75}$ shift is unexplained.** Table 5 reports a +10.7 percentage point improvement in localization mIoU, yet Table 1 shows AP$_{75}$ improves by only +0.5 for the same detector (Faster R-CNN R-50). If the mIoU improvement were concentrated on examples already well above or below the 0.75 threshold, this discrepancy is explainable, but the paper offers no such analysis.

### Trivial

- **Figure 2 pseudocode is ambiguous.** The phrase `M ← top S(C) combined` is undefined, and the loop iterates over `range(len(P))` while elements are simultaneously removed from `P`. Because the surrounding text (“take the minimum … and maximum”) and released code clarify the procedure, this does not block understanding.

## Nice-to-Haves

- **Quantitative analysis of the crowding failure mode.** Figure 4 qualitatively shows that nearby same-class instances can merge into a single box. Reporting the merge rate as a function of object density on crowded COCO scenes would clarify whether the stable LRP$_{\text{FP}}$ in Table 6 masks a harmful trade-off.
- **Ablate the refinement head in context.** A UoI variant without the second regression stage (keeping everything else fixed) would clarify how much of the final gain comes from the initial union-of-intersections versus the learned refinement.

## Removed Points

These points are flagged to be removed, treat them with caution.

- *“A 10-point mIoU gain should normally translate into a much larger AP$_{75}$ shift.”* This expectation is not well-founded. mIoU averages across all matched detections, whereas AP$_{75}$ only counts detections above the 0.75 IoU threshold. A large mIoU improvement concentrated on low-to-mid-quality detections (consistent with the paper’s claim of robustness to poor proposals) would inflate mIoU without strongly affecting AP$_{75}$. This is therefore not a valid discrepancy.
- *Strength Finder claim that Table 8 “isolates the gain to the problem reformulation rather than to deeper networks.”* This over-interprets Table 8. The table only shows that adding a second regression head with winner-takes-all selection does not help; it does **not** isolate intersection regression from the multi-box merging effect at inference, which is the verified major weakness above.

## Novel Insights

None beyond the paper’s own contributions.

## Suggestions

1. **Provide the missing ablation.** Train a Faster R-CNN baseline with standard full-GT regression targets, but at inference apply the same proposal grouping, min/max merging, and second-stage refinement used by UoI. If this matches UoI, the intersection-target hypothesis requires revision; if it lags, the hypothesis is strongly supported.
2. **Clarify or retract the query-based claim.** Either fully specify the Deformable DETR quadrant mechanism (matching cost, query count per quadrant, quadrant definition) or soften the abstract/introduction to state that the method applies to proposal-based and grid-based detectors, with a conceptually related but distinct extension for query-based detectors.
3. **Report standard deviations across 3–5 seeds** for at least the primary Faster R-CNN and Mask R-CNN results to establish the statistical reliability of the gains.

## Score and Decision

I calibrated this paper against several anchor reviews in the human corpus. High-scoring anchors such as **D-FINE** (7.50, Spotlight) and **PointOBB-v2** (7.00, Accept Poster) share the localization-improvement theme but are far more thorough and report larger gains. The plug-and-play detection paper **LGPL** (6.00, Accept Poster) achieved consistent cross-architecture improvements but was criticized for limited scope; our paper covers the standard COCO benchmark and includes direct localization analysis, yet it suffers from a critical ablation gap that LGPL did not have. Medium/low-scoring anchors such as **wFPfYccHJ1** (4.50, Reject) and **A²-DP** (4.75, Reject) were penalized for missing key ablations and ambiguous experimental support—weaknesses that partially match our paper. Relative to these anchors, the present submission sits between the rejected ablation-gap papers and the accepted plug-and-play papers: it has a cleaner idea and stronger standard-benchmark coverage than the low anchors, but the missing control experiment and the oversold query-based claim are material enough to place it below the 6.00 accept threshold. I therefore score it below the borderline.

**Score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>