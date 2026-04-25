Now let me search for calibration anchors.Now I have enough information to write the final review. Let me synthesize my findings.---

## Summary

This paper proposes Union-over-Intersections (UoI), a plug-and-play modification to the standard object detection pipeline with two coupled changes: (1) training proposals to regress to their intersection with the ground truth rather than the full GT box, and (2) replacing winner-takes-all NMS with a union-over-intersections aggregation of all grouped proposals. The method is applied to five diverse detector architectures (Faster R-CNN, Mask R-CNN, Cascade R-CNN, YOLOv3, Deformable DETR) and consistently improves localization mAP by 0.5–1.1 points on COCO.

---

## Strengths

- **Consistent localization gains across five architectures (Table 1):** mAP improvements range from +0.5 (Deformable DETR) to +1.1 (Mask R-CNN R101) across proposal-based, grid-based, and query-based detectors with both R-50 and R-101 backbones, supporting the breadth claim.

- **Clean separation of localization vs. classification effects (Table 5):** Classification accuracy stays flat (76.4→76.5%) while localization mIoU jumps from 53.7% to 64.4%, exactly what the method claims to target. The LRP analysis in Table 6 corroborates this by showing large reductions in LRP_Loc (17.2→12.7) with minimal FP/FN change.

- **Second-regression-head ablation confirms UoI framework is load-bearing (Table 8):** Adding a second regression head to standard Faster R-CNN yields no gain (37.4→37.4 mAP), while within UoI it contributes meaningfully (→38.1), ruling out the simple "more parameters" confound.

- **Intersection regression is empirically an easier task (Figure 3b):** Lower training loss convergence under intersection targets versus full-GT targets provides direct evidence for the paper's central mechanistic hypothesis.

- **Oracle experiment (Figure 3d):** As classification accuracy rises from 20% to 100%, the gap between UoI and the baseline widens, making a forward-looking argument for the method's scaling potential.

- **Compatibility with diverse losses and NMS variants (Tables 2, 4):** UoI improves over L1, GIoU, DIoU, and Alpha-IoU baselines, and under NMS, Cluster-NMS, and Soft-NMS, demonstrating genuine orthogonality to other design choices.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing component isolation ablation — the intersection regression target is never isolated from the union grouping step.** The paper has two coupled contributions: (a) intersection-based regression targets, and (b) union-based proposal grouping. No experiment tests the combination of *standard full-GT regression targets with union grouping at test time*. Without this condition, it is impossible to determine whether the intersection regression target is independently necessary, or whether simply replacing NMS winner-takes-all with union aggregation (applied to conventionally trained boxes) is sufficient. Table 8 ablates the second regression head, not this critical component isolation. This is the most significant technical gap: the more theoretically novel contribution — the intersection target design — lacks an independent validation.

- **Misleading comparison against one-to-few (Section 4.3).** The paper claims: "our approach... achieves superior results, which one-to-few cannot." This is directly contradicted by Table 1. One-to-few achieves 40.9 mAP on Faster R-CNN + ResNet101, and UoI on the *same* Faster R-CNN + ResNet101 achieves only **40.3 mAP** — lower. The 43.1 mAP figure cited for "UoI" is from Cascade R-CNN, a substantially stronger base architecture that already achieves 42.5 mAP without UoI. The paper uses a different, stronger base to manufacture a favorable comparison, then attributes the advantage to UoI. This specific comparative claim is unsupported and should be corrected.

### Minor

- **"Plug-and-play with minimal modifications" framing overstates the uniformity of integration.** The R-CNN family requires only target-coordinate changes (genuinely minimal). For YOLOv3, Section 4.2 describes changing the object-to-grid assignment from center-based to IoU-based, assigning objects to multiple grid cells — a redesign of the label assignment strategy. For Deformable DETR, GT boxes are divided into quadrants and queries routed to specific parts — a nontrivial spatial partitioning change. These are qualitatively different modifications from each other and from the R-CNN adaptation. The paper acknowledges different mechanisms are needed but does not reconcile this variation with the "seamlessly integrates" framing.

- **Post-processing algorithm: dynamic M update may alter suppression behavior.** In Figure 2's pseudo-code, M (initialized as the top-scoring regressed box) is updated to the union of all boxes in the growing group C. The first loop condition `iou(M, b_i) >= k` then checks IoU against this ever-growing union box. Since union boxes are geometrically larger, this condition may suppress more regressed boxes than standard NMS would suppress, potentially changing suppression behavior as a side effect of the union expansion. This interaction is not analyzed and could partially explain some gains independently of intersection regression.

- **Instance segmentation: intersection targets interact with mask head (Table 3).** In Mask R-CNN, the mask branch operates on the RoI defined by the box prediction. If training uses intersection-based box targets, the RoI crops differ from those in standard training, yet the paper provides no discussion of how this affects mask quality or whether the box-target change versus the union-grouping change drives the segmentation gains.

### Trivial
None beyond the comparison issue noted in Major.

---

## Nice-to-Haves

- Apply UoI to a modern DETR variant (e.g., DINO) operating at ≥50% AP to confirm the relative gain holds at stronger baselines, which would strengthen the contribution's relevance to current practice.
- Report the localization-mIoU vs. classification ablation (Table 5 equivalent) for at least one other architecture to confirm this is architecture-agnostic.
- Visualize intermediate intersection predictions before union merging, particularly for the YOLO and Deformable DETR adaptations, to build intuition for readers.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Figure 3a shows the benefit narrows as initial IoU increases — supports 'already mitigated' concern."** The paper explicitly acknowledges that at high initial IoU both methods converge (Section 4.3: "When proposals have high overlap with the ground truth, both methods perform similarly"). This is not a weakness the paper ignores; it is honestly presented as supporting the method's use case (low-to-medium quality proposals). Removed as a strawman.

- **Harsh Critic: "Gains of 0.5–1.1 mAP are marginal and training variance cannot be ruled out."** Single-run evaluation is standard practice for large-scale COCO benchmarks at this scale; demanding confidence intervals across seeds is not a norm in this community. Removed as a non-standard reproducibility request.

- **Harsh Critic: "Apply to Faster R-CNN + ResNet101 as direct comparison to one-to-few."** Table 1 already provides this number (40.3 mAP). The criticism is valid (addressed under Major), but the specific request to "add the experiment" is moot since the data already exists in the paper — the problem is the paper *misrepresents* it in Section 4.3.

- **Strength Finder: "Large gains at AP@75 vs AP@50 support tighter box quality."** The Mask R-CNN R101 comparison actually shows AP@50 +0.8 and AP@75 +0.6 — AP@75 is *not* consistently larger than AP@50 across the board (the claim is architecture-dependent). Removed as inaccurate as stated.

---

## Novel Insights

The pseudo-code in Figure 2 reveals an under-analyzed interaction: the growing union box M is used as the reference for the first suppression condition (removing competing regressed boxes), meaning UoI's grouping and NMS suppression are not independent operations. As the union M expands by adding intersections from new proposals, M becomes a geometrically larger box with higher IoU against all nearby regressed boxes, potentially suppressing more boxes than conventional NMS would. This creates a positive feedback between union expansion and suppression — a mechanism distinct from both intersection regression and the union merging idea, which the paper does not analyze. Disentangling this interaction could reveal whether some observed gains stem from altered suppression dynamics rather than the intersection training target itself.

---

## Suggestions

1. **Add the critical missing condition to the ablation:** Run Faster R-CNN with standard full-GT regression targets but with union-based grouping and aggregation at test time (i.e., replace only the post-processing, not the training target). Compare this to full UoI and to the baseline. This single experiment would definitively establish whether intersection regression contributes independently.

2. **Correct or remove the one-to-few comparison in Section 4.3.** The honest comparison is: Faster R-CNN + R101 + UoI = 40.3 mAP vs. one-to-few = 40.9 mAP. The paper should either acknowledge this and claim UoI is complementary (not superior) to one-to-few, or test UoI within one-to-few's framework.

3. **Qualify the "plug-and-play" claim** to acknowledge that YOLO and DETR require more substantial adaptations than the R-CNN family, and discuss what the core "idea" is for each paradigm.

---

## Score and Decision

**Calibration anchors consulted:**

| Paper | Avg Score | Comparison |
|---|---|---|
| D-FINE (MFZjrTFE7h) | 7.5 (Accept Spotlight) | Also redefines regression task in detection; achieves SOTA performance (~55%+ AP), much stronger contribution and cleaner ablations |
| Bounding Box Stability (lmM4Ecm4HJ) | 7.5 (Accept Spotlight) | Novel detection insight with clean experiments; stronger conceptual framing |
| Label Transfer for Detection (ChHx5ORqF0) | 6.0 (Accept Poster) | Similar scope — detection method validated across architectures, incrementally useful, moderate novelty |
| QFree-Det (vyF5aim4US) | 5.25 (Reject) | Detection method with modest contribution, some structural concerns; similar profile to UoI |
| DAP-Loss for DETR (AVAlVPdQp7) | 3.0 (Reject) | Low-quality paper; weak experiments, unclear contribution |

The UoI paper sits below ChHx5ORqF0 (6.0, Accept Poster) on methodological rigor due to the one-to-few comparison flaw and the missing component ablation. The one-to-few issue is particularly damaging: on the paper's own reported numbers, UoI on the same architecture (Faster R-CNN + R101) underperforms one-to-few (40.3 vs 40.9 mAP), yet the paper explicitly claims the opposite. This is a factual misrepresentation of the paper's own results, not a gap in experiments. The missing component ablation is a separate but serious issue: the more novel part of the method (intersection regression targets) is never tested in isolation from the union grouping. Together, these issues reduce confidence in the paper's strongest claims.

On the positive side, the core idea is genuinely novel, the empirical breadth is good, and the localization-vs-classification ablation is clean. The paper is above borderline papers like QFree-Det (5.25) in terms of experimental thoroughness and novelty.

**Originality:** Moderate — the intersection-target idea is genuinely novel, but the gains are modest and the union-aggregation is related to existing box voting approaches.  
**Importance:** Moderate — addresses a real limitation of winner-takes-all NMS.  
**Claims vs. support:** Weak in one respect — the one-to-few comparison is unsupported; the component attribution is incomplete.  
**Soundness:** Adequate but with the ablation gap noted.  
**Clarity:** The pseudo-code and motivation are clear; the YOLO/DETR adaptation descriptions lack uniformity.  
**Value to community:** Moderate — the method is easy to apply and the code is public.

**Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>