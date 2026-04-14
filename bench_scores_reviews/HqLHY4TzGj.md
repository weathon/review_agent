Now I have enough information to write the consolidated review.

---

## Summary

Union-over-Intersections (UoI) proposes two modifications to the standard object detection pipeline: (1) changing the regression target from the full ground truth box to the proposal–ground truth intersection, and (2) replacing winner-takes-all NMS with a union of grouped intersection predictions. The method requires only minor changes to existing regression heads and post-processing, and is demonstrated on five architectures (Faster R-CNN, Mask R-CNN, Cascade R-CNN, YOLOv3, Deformable DETR) on MS-COCO and Pascal VOC, consistently yielding improvements in localization quality and modest mAP gains.

---

## Strengths

- **Specific, non-trivial conceptual contribution**: Reframing box regression as an intersection-learning problem rather than full-GT extrapolation is a genuine reframing of the regression target — not merely a new loss function — and the "wisdom-of-the-crowd" merging provides a principled alternative to winner-takes-all NMS. Most proposals in the space adjust *how* to regress to the full GT; this paper changes *what* is regressed to.

- **Precise isolation of the benefit source**: Table 5 provides a clean dissection: classification accuracy is essentially unchanged (76.4% → 76.5%) while localization mIoU leaps from 53.7% to 64.4%. Table 6 with LRP metrics independently confirms localization error drop (17.2 → 12.7) without increasing false positives or false negatives. Together these pinpoint the improvement mechanism with unusual specificity for an empirical detection paper.

- **Broad architectural coverage with consistent signal**: Gains appear uniformly across proposal-based (Faster/Mask/Cascade R-CNN), grid-based (YOLOv3), and query-based (Deformable DETR) detectors across two backbones. Table 2 further shows UoI is complementary to GIoU, DIoU, and Alpha-IoU losses. This breadth is substantive evidence for the plug-and-play claim within the tested family of architectures.

- **Oracle experiment with forward-looking implication**: Figure 3(d) shows that as classification quality improves (simulated oracle), the UoI advantage over standard NMS widens monotonically. This is a concrete prediction about future scalability — as detectors improve, UoI becomes more effective — that goes beyond a typical ablation.

- **Tight second-stage ablation**: Table 8 demonstrates that adding a second regression stage *without* UoI yields no gain (37.4 → 37.4 mAP), while UoI reaches 38.1. This rules out the competing hypothesis that the improvement comes from added model capacity rather than the intersection-union formulation.

---

## Weaknesses

### Fatal
None.

### Major

- **Evaluation confined to architectures from 2015–2021**: For ICLR 2025, the strongest baseline is Deformable DETR at 44.3 mAP (2021). Detectors such as DINO, Co-DINO, and RT-DETR routinely exceed 50 mAP and employ significantly different matching strategies (e.g., one-to-many with denoising, more refined query initialization). These modern architectures may partially alleviate the extrapolation problem through improved query initialization, making UoI less complementary. The "broad applicability" claim cannot be fully evaluated without at least one modern high-performing baseline.

- **Deformable DETR adaptation contradicts its NMS-free design philosophy without adequate justification**: Deformable DETR uses bipartite matching (Hungarian algorithm) specifically designed to produce non-redundant, one-to-one assignments — no NMS is needed. Applying UoI requires dividing ground truths into quadrants and allowing multiple queries to cover the same object, then re-introducing grouping/merging. This is a fundamental reworking of the set-prediction paradigm, not a "minimal modification." The paper does not explain how this interacts with the bipartite matching constraints, whether the set-prediction loss still applies, or whether the quadrant division produces valid training targets when queries attend to overlapping spatial regions. This adaptation receives only one paragraph and no dedicated analysis.

- **Instance segmentation mask aggregation is never described**: Table 3 reports instance segmentation improvements on Mask R-CNN, attributing them to "effective merging of multiple proposals which leads to more precise segmentation masks." However, the method section describes only *bounding box* unions. How are instance masks combined when multiple proposals' boxes are merged into a union box? Majority vote, union of mask pixels, the highest-scoring mask, or something else? This is a genuine methodological gap — the experimental result in Table 3 rests on an undescribed procedure.

- **Crowded-scene failure mode is acknowledged but not quantified**: Figure 4 and Section 4.3 explicitly identify that UoI can merge adjacent same-class instances into a single box. No evaluation on benchmarks designed for dense scenes (e.g., CrowdHuman) is provided, leaving the practical severity of this failure mode entirely uncharacterized. For applications such as pedestrian detection or autonomous driving, this is not a marginal edge case.

### Minor

- **The refinement stage training procedure is underspecified**: Equation (5) shows a refinement loss mapping the union-of-intersections box B_j to the ground truth. During training, B_j is constructed from the current model's predicted intersection boxes. The paper implies end-to-end joint optimization (L_total = L_intersection + λ·L_refinement) but never states this explicitly, nor discusses how the noisy early-stage intersection predictions in B_j affect stability. Table 8 provides indirect evidence that the two stages work together, but a clear statement of the training procedure (e.g., end-to-end vs. staged, gradient flow through the union operation) is absent.

- **Unfair comparison with one-to-few (Section 4.3)**: The paper compares one-to-few applied to Faster R-CNN/ResNet-101 (40.9 mAP) with UoI applied to Cascade R-CNN/ResNet-101 (43.1 mAP). Since Cascade R-CNN is a significantly stronger base architecture, this comparison inflates the apparent advantage of UoI. The stated argument — that UoI is plug-and-play while one-to-few requires architectural changes — is legitimate, but the numbers as presented are not a valid head-to-head comparison. UoI on Faster R-CNN/ResNet-101 achieves 40.3 mAP, which is actually *below* one-to-few on the same base.

- **Large gap between localization mIoU gain and mAP gain is not explained**: A ~19% relative improvement in localization mIoU (53.7% → 64.4%) translates to only ~1.9% relative mAP improvement (37.4 → 38.1). The likely mechanisms — class-conditional mAP thresholding, interaction with NMS IoU threshold, IoU threshold sweep from 0.5–0.95 — are not discussed. This gap is not a flaw in the method, but the disconnect needs explanation to build reader confidence that the localization metric and task metric tell a coherent story.

- **Grouping hyperparameter (top-5 proposals) applied universally without cross-architecture sensitivity analysis**: Figure 3(c) establishes top-5 using Faster R-CNN on COCO. The same value is applied to YOLOv3, Deformable DETR, and Pascal VOC without any analysis of sensitivity across architectures, object density regimes, or datasets.

### Tiny

- The pseudo-code in Figure 2 uses two separate inner-loop conditions that operate on different box representations (predicted intersection boxes B vs. original proposals P), making it difficult to parse which suppression step affects which pool. A brief prose clarification would help.

- The abstract's claim that the method integrates "seamlessly" into all three paradigms is somewhat at odds with the DETR adaptation described in Section 4.2, which requires dividing GT into quadrants and restructuring the matching. The language should be more precise.

---

## Nice-to-Haves

- An experiment on at least one modern detector (e.g., DINO or RT-DETR) would substantially strengthen the generalizability claim for ICLR 2025, even if limited in scope.

- An adaptive grouping threshold that prevents merging of distinct instances in crowded scenes (e.g., incorporating center-distance constraints or density-based IoU adaptation) would address the acknowledged limitation and expand the method's practical applicability.

- A compute-matched comparison (e.g., wider backbone or extra regression head with standard targets, same parameter count as UoI) would rule out the possibility that the slight gain from the second regression stage comes from added parameters rather than the UoI formulation — though Table 8 already provides strong evidence against this.

- A brief optimization landscape analysis or gradient variance comparison between intersection regression and full-GT regression would provide more rigorous grounding for the "easier task" claim than the loss curve in Figure 3(b) alone.

- Explicit description of how instance segmentation masks are handled in the union merging step.

---

## Removed Points

*These points are flagged for removal — treat them with caution as they may misread the paper or impose non-standard requirements.*

- **Training-inference domain shift (harsh critic)**: The paper's joint total loss (L_intersection + λ·L_refinement) implies end-to-end training in which B_j is computed from the model's own intersection predictions during the forward pass. This is analogous to how Cascade R-CNN uses each stage's outputs as the next stage's input. The concern about oracle vs. predicted B_j during training vs. inference does not appear to apply; removal is warranted, though the paper should state the training procedure explicitly (kept as a minor clarity point above).

- **Statistical significance testing on COCO (harsh critic)**: Requesting confidence intervals or multi-run statistics for COCO benchmark evaluation is not standard in the detection community, where single-run evaluation is the norm. Removed per community standards.

- **Lack of theoretical proofs / optimization landscape as a weakness (balanced reviewer)**: The paper is an empirical systems contribution and provides loss curve evidence (Figure 3b) for the "easier task" hypothesis. Demanding formal convergence proofs is beyond what is expected for this paper type. Moved to nice-to-have.

- **"Well-written paper" / "important topic" / "extensive experiments" (generic strengths from all reviewers)**: Removed as they apply to any paper in the area.

- **Box voting "relies" on TTA framing overstated (harsh critic)**: The harsh critic correctly notes that box voting w/o TTA already competes well (37.5 vs. 38.1 mAP). The paper's framing is slightly overstated but not materially misleading. This is a style nitpick, not a substantive weakness.

---

## Novel Insights

The most genuinely novel observation across the three reviews — beyond what the paper itself highlights — is the potential conflict between the UoI post-processing paradigm and the NMS-free, bipartite-matching design of transformer-based detectors. The paper applies UoI to Deformable DETR via a quadrant-splitting strategy that effectively re-introduces one-to-many assignment into a detector designed to eliminate it, and shows that this still yields gains. This raises an unexplored question: is the benefit of UoI in the DETR setting coming from the intersection regression, the union merging, or from the implicit switch back to one-to-many assignment that the quadrant strategy entails? Separating these effects could illuminate whether UoI's gains on transformer-based detectors are mechanistically the same as on proposal-based ones — a distinction with real implications for the unified narrative of the paper.

---

## Suggestions

1. **Add one modern DETR-family baseline** (e.g., DINO with ResNet-50) to establish relevance to the current detection frontier; even modest UoI gains there would substantially bolster the paper's impact claim.

2. **Describe the mask aggregation procedure for instance segmentation explicitly** — even one sentence or a diagram showing how masks from multiple proposals are combined after the box union step.

3. **Provide a same-base comparison with one-to-few** (both on Faster R-CNN/ResNet-101) alongside the current comparison, so readers can clearly see both the plug-and-play advantage and the head-to-head performance trade-off.

4. **Explain the mIoU vs. mAP discrepancy in Table 5** — e.g., how does a 19% relative localization improvement produce only a 1.9% relative mAP improvement? A two-paragraph analysis would significantly build reader confidence in the method.

5. **Quantify the crowded-scene failure mode**: even a simple analysis of the fraction of detections affected by merging as a function of instance density on the COCO validation set would help practitioners understand when *not* to use UoI.

6. **Clarify the Deformable DETR training procedure** after quadrant assignment: does bipartite matching still apply? Are the quadrant targets compatible with the set-prediction loss? A supplemental section or diagram for this adaptation is warranted given how much it departs from the standard DETR formulation.

---

**Overall assessment across axes:**

- **Novelty**: Moderate-to-good. The specific reformulation of box regression as intersection learning is a distinct and clean idea, not merely an incremental loss variation. The union merging as a principled NMS replacement is less novel (cf. box voting) but is well-motivated by the intersection targets.
- **Technical soundness**: Moderate. Core mechanics are sound; the refinement stage and especially the DETR adaptation have underspecified training details and unresolved architectural tensions.
- **Empirical support**: Moderate-weak for ICLR 2025. Consistent gains on five architectures, but all from 2015–2021; no crowded-scene evaluation despite a known failure mode; mask aggregation for segmentation results is undescribed.
- **Significance**: Moderate. The plug-and-play nature and the oracle scalability argument are practically significant, but the gain magnitude is modest (~0.5–0.9 mAP) and the method's applicability to state-of-the-art architectures is unverified.
- **Clarity**: Good for the proposal-based adaptations; insufficient for the DETR adaptation, the refinement stage training loop, and the segmentation mask handling.