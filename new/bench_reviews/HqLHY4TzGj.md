Now I have a comprehensive understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

The paper proposes Union-over-Intersections (UoI), a plug-and-play modification to standard object detection pipelines that makes two changes: (1) regressing proposals only to their intersection with the ground truth rather than the full ground truth box, and (2) replacing winner-takes-all NMS with a union-over-intersections operation that merges the regressed intersections from multiple proposals before a refinement step. The method is evaluated on five detection architectures (Faster/Mask/Cascade R-CNN, YOLOv3, Deformable DETR) showing consistent but modest mAP improvements of 0.5–1.1 on COCO.

## Strengths

- **Simple, well-motivated core intuition**: The idea that regressing only to the visible intersection of a proposal with ground truth is an easier learning task than extrapolating beyond the proposal's receptive field is clean and directly supported by Figure 3(b), which shows lower regression loss for intersection targets.

- **Consistent improvements across five architectures**: Table 1 shows positive mAP gains on all five tested architectures with all sub-metrics improving, including two-stage, single-stage, and transformer-based detectors, supporting the claim of broad applicability.

- **Improvements are localization-driven**: Table 5 (classification accuracy 76.4→76.5% vs. localization mIoU 53.7→64.4%) and Table 6 (LRP_Loc drops from 17.2 to 12.7) convincingly show that the gains come from improved localization, not classification, which directly supports the paper's mechanism.

- **Complementary to IoU-based losses and NMS variants**: Table 2 shows UoI stacks on top of GIoU/DIoU/Alpha-IoU losses, and Table 4 shows consistent gains across NMS/Cluster-NMS/Soft-NMS, demonstrating the method captures improvements orthogonal to existing design choices.

- **Minimal inference overhead**: Section 4.1 reports 14.1 fps vs. 14.4 fps for baseline Faster R-CNN, confirming negligible computational cost.

- **Oracle analysis (Figure 3d)** showing the gap widens with better classification is an insightful forward-looking argument.

## Weaknesses

### Fatal
None.

### Major

- **No ablation isolating the two proposed components**: The paper proposes two distinct innovations — intersection-based regression and union-over-intersections grouping — but never provides a 2×2 factorial ablation (standard regression + standard NMS vs. intersection regression + standard NMS vs. standard regression + union grouping vs. intersection regression + union grouping). All reported experiments use both components together (or vary unrelated factors like the NMS implementation or loss function). Without this, the paper cannot substantiate that both components are necessary, or even that either one alone is effective. Given that the title and narrative center on both ideas, this is a significant gap in the evidence chain. The closest experiment is Table 8, which ablates the second regression stage, but this tests the refinement step rather than the two core innovations.

- **Architecture-specific assignment changes confound the claimed "plug-and-play" generality for YOLOv3 and Deformable DETR**: Section 4.2 explicitly states that for YOLOv3, the object-to-grid assignment was changed from center-based to IoU-based (assigning objects to multiple grid cells), and for Deformable DETR, ground truths were divided into quadrants with queries assigned to specific parts. Both changes independently alter which proposals learn which targets and could themselves improve performance. The paper claims "minimal modifications" but these are substantive assignment-strategy changes, not simple regression-target swaps. This makes the YOLOv3 and Deformable DETR improvements partially uninterpretable with respect to the UoI mechanism. The Faster/Mask/Cascade R-CNN results are cleaner in this regard.

- **Instance merging is a structural limitation, not just a corner case**: The union of overlapping same-class detections will merge distinct instances into one box. The paper acknowledges this (Section 4.3, Figure 4) but dismisses it as addressable by "advanced grouping strategies" without evidence. On COCO, images average 7 objects with many crowded same-class scenarios. This issue is inherent to the union operation and directly undermines the claim that UoI is a general "plug-and-play" NMS replacement, since NMS specifically prevents such merging. The paper provides no quantitative analysis of how frequently this occurs.

### Minor

- **Unfair comparison in the second-regression ablation (Table 8)**: The baseline with "2nd regression" regresses to the same target twice, which is trivially redundant. A more informative control would cascade from the NMS-selected box (which, like the union box, is already a better starting point than the raw proposal). The paper's argument that the union box is a better starting point may be correct, but this table does not cleanly isolate it.

- **No per-category analysis or high-IoU threshold breakdown**: UoI should particularly benefit localization at high IoU thresholds, yet AP₈₀ and AP₉₀ are not reported. Similarly, no per-category breakdown is given to show where UoI helps vs. hurts (especially for categories prone to instance merging).

- **No standard deviations or multi-run statistics reported**: The improvements are modest (0.5–1.1 mAP); without variance estimates, it is difficult to assess statistical significance at these margins.

### Trivial
None.

## Nice-to-Haves

- A 2×2 ablation (intersection-only regression with standard NMS vs. standard regression with union grouping vs. full UoI) would definitively establish the contribution of each component.
- Per-category analysis showing which object types benefit and which are hurt by UoI, particularly in crowded same-class scenarios.
- Testing on a more recent one-stage detector (e.g., YOLOv5/v8 or FCOS) to establish currency.
- Quantify the instance-merging failure rate on COCO (e.g., what fraction of same-class instance pairs have IoU above the grouping threshold).

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Regressing-to-GT is not ill-posed" (harsh critic, intro)**: The paper's claim that regressing proposals to the full GT is "ill-posed" is a reasonable framing of a genuine optimization challenge — the intersection target is empirically easier to optimize (Figure 3b). Calling it "ill-posed" may be slightly overstated but does not invalidate the contribution.

- **"λ=0.5 not ablated"**: This is a minor hyperparameter choice; the method is robust enough that the exact weighting is unlikely to change conclusions. Moved from weakness to trivial/nice-to-have, then removed as truly trivial.

- **"YOLOv3 is outdated"**: YOLOv3 is a well-known benchmark architecture. While newer versions exist, this is a scope/suggestion concern, not a substantive weakness. Moved to Nice-to-Have.

- **"Encyclopedic related work"**: Formatting/style concern removed per instructions.

- **"Pseudocode 'top S(C) combined' is underspecified"**: The surrounding text (Section 3) and Figure 2 description clarify this is the union-over-intersections of grouped proposals weighted by score. This is a minor presentation issue, not a reproducibility blocker.

- **"Demand YOLOv5/v8 testing"**: Scope creep; the paper's claim is about general applicability across detection paradigms, which it demonstrates across two-stage, one-stage, and transformer-based architectures.

- **"Missing appendix/proofs"**: Parser artifact; original submission includes supplementary material.

## Novel Insights

The most insightful observation from combining the reviews is that UoI's second regression stage (Eq. 5) actually reintroduces the extrapolation problem the paper argues against — it regresses from the union box to the full ground truth. The paper's defense is that the union box provides a better starting point than individual proposals, which is plausible but only partially supported: Table 8 shows a trivially redundant baseline rather than a principled one (cascading from an NMS-selected box). This means the paper's mechanism story has a tension: intersection regression is advocated because extrapolation is hard, but the refinement stage explicitly performs extrapolation. The quantitative evidence (Tables 5 and 6) suggests the localization gains are real, but whether they come from both components working together or primarily from one remains unestablished.

## Suggestions

- Add a 2×2 ablation (intersection-regression × union-grouping) to establish each component's individual and joint contribution.
- For YOLOv3 and Deformable DETR, run a control where only the assignment-strategy change is applied (without UoI) to isolate the UoI mechanism from the confound.
- Add a fairer second-regression control: apply a second regression head starting from the NMS-selected box to the full GT.
- Report AP at IoU thresholds 0.80 and 0.90, where the localization improvement should be most pronounced.
- Quantify the instance-merging failure frequency on COCO (e.g., by measuring how often same-class instance pairs with IoU > grouping threshold appear).

## Score Calibration

**Comparison anchors:**

- **D-FINE** (avg 7.5, Spotlight): Redefines bounding box regression in DETRs with distribution refinement, achieving up to 5.3% AP gains with thorough ablations. UoI's gains are much more modest (0.5–1.1 mAP), and its ablations are far less complete (no component-level 2×2). Clearly below this anchor.

- **Implicit RL in DINO** (avg 4.33, Withdrawn): Simple ε-greedy modification yielding small 0.3–1.8 AP gains, with concerns about incremental novelty and small gains. UoI has similar-magnitude gains but a better-motivated and more generalizable idea, plus broader evaluation across 5 architectures. Slightly above this anchor.

- **QFree-Det** (avg 5.25, Reject): Eliminates queries/NMS in DETR, with concerns about complexity and whether the deduplication module is necessary. Similar level of contribution complexity; UoI is simpler and has more consistent gains but with the component ablation gap.

- **DAP-loss** (avg 3.0, Reject): Flawed derivations and limited experiments. Clearly above this anchor — UoI has no such fundamental flaws.

- **QO-DETR** (avg 2.2, Reject): Questionable novelty, insufficient ablations. Clearly above this anchor — UoI has a genuine and novel idea.

The paper sits in the 5–6 range: above the clearly weak papers (2–3) and the marginal/incremental ones (~4), but well below the strong detection papers (7+). The core idea is sound and empirically validated, but the missing component ablation and confounded experiments for 2 of 5 architectures prevent strong confidence in the causal claims.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>