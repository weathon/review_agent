## Summary
This paper proposes Union-over-Intersections (UoI), a plug-and-play modification to standard object detection pipelines. Rather than regressing each proposal to the full ground truth box, UoI trains proposals to predict only their intersection with the ground truth; at inference, the detections for a group of proposals are fused via a union operation instead of winner-takes-all NMS. The method is evaluated across five diverse detector architectures (Faster R-CNN, Mask R-CNN, Cascade R-CNN, YOLOv3, Deformable DETR) on MS-COCO and PASCAL VOC, showing consistent but modest mAP improvements and transferring to instance segmentation.

---

## Strengths

- **Consistent multi-architecture empirical validation.** Table 1 reports uniform improvements (0.5–1.1 mAP) across every architecture and backbone tested — two-stage, single-stage, and transformer-based — without cherry-picking a favourable setting. This breadth of positive results is stronger evidence than most plug-and-play papers provide.

- **Principled decomposition of localization vs. classification gains.** Table 5 shows that classification accuracy is essentially unchanged (76.4→76.5%), while localization mIoU jumps from 53.7% to 64.4%, and Table 6 confirms a 26% relative drop in LRP_Loc with negligible change in false-positive and false-negative rates. This is a specific, quantitative attribution of where the gains originate, which is unusual in detection papers.

- **Table 8 specifically refutes the "cascade is the real driver" alternative explanation.** Adding only a second regression stage (without the UoI formulation) yields no benefit (37.4/40.4 → 37.4/40.5), confirming that the gain is tied to the intersection-based target plus union grouping, not merely model capacity.

- **The oracle classification experiment (Figure 3d) reveals a forward-looking insight.** As classification accuracy is artificially improved, the gap between UoI and the traditional pipeline widens monotonically. This establishes that UoI is not a saturating trick but a formulation that will benefit from future classification improvements — a qualitatively useful finding beyond the immediate results.

- **Complementarity with existing loss functions and NMS variants** (Tables 2 and 4). The method consistently adds value on top of GIoU, DIoU, Alpha-IoU, Cluster-NMS, and Soft-NMS, indicating it addresses an orthogonal axis of improvement rather than subsuming existing techniques.

---

## Weaknesses

- **Missing component-level ablation between intersection regression and union grouping.** Table 8 rules out that a second regression stage alone suffices, but no experiment isolates (i) intersection-based regression with standard winner-takes-all NMS or (ii) standard regression targets with union-based post-processing on the regressed boxes. Without this decomposition, it is impossible to know which of the two proposed changes is the primary driver of the +0.7 mAP gain. This is the single most important missing ablation for a method whose two components are presented as independently motivated.

- **Unexplained disconnect between localization mIoU and mAP gains.** Table 5 reports a +10.7 point absolute improvement in localization mIoU (53.7→64.4%). A change of this magnitude at the proposal level should propagate to much larger mAP improvements than +0.7 points. The paper does not explain whether mIoU is computed on all assigned proposals (before NMS/grouping) or on final detections, nor why such a dramatic localization improvement at one level does not materially move the needle at the final mAP level. The same tension exists in Table 6: LRP_Loc drops from 17.2 to 12.7 (−26% relative), yet overall LRP drops from 67.6 to 65.3 (−3.4%). Authors should clarify the computation domains of these metrics and explain the apparent gap.

- **"Plug-and-play" claim is overstated.** Section 4.2 reveals that adapting to YOLOv3 required replacing its center-based object-to-grid assignment with an IoU-based multi-cell assignment strategy (a non-trivial label-assignment change), and adapting to Deformable DETR required partitioning ground truth into quadrants and restructuring the Hungarian matching. These are meaningful architectural decisions, not mere target substitutions. The abstract's claim of "minimal modifications" and "seamless integration" should be qualified; a forthright description of per-architecture engineering requirements would improve reproducibility and set more accurate expectations.

- **The one-to-few comparison is architecturally asymmetric in UoI's favour.** Section 4.3 compares one-to-few on ResNet101 + Faster R-CNN (40.9 mAP from a 39.4 baseline) with UoI on ResNet101 + Cascade R-CNN (43.1 mAP from a 42.5 baseline). Since UoI is never applied to the same Faster R-CNN base as one-to-few, this does not constitute a fair methodological comparison. The stated intent — showing UoI can leverage stronger architectures — is legitimate, but the framing as superiority over one-to-few is misleading. Applying UoI to the same Faster R-CNN setup as one-to-few would be the honest comparison.

- **Crowded-scene failure mode is unquantified.** Figure 4 illustrates that closely overlapping same-class instances can be merged into a single detection. The paper acknowledges this but provides no metric quantifying how often this occurs or its contribution to false negatives on COCO crowd annotations. Given that COCO has crowd regions with annotation `iscrowd=1`, a targeted analysis would clarify the practical severity of this limitation.

---

## Nice-to-Haves

- **Sensitivity analysis for the IoU grouping threshold *k*.** Group size is ablated (Figure 3c), but the IoU threshold used to form proposal groups is not swept. Since this threshold governs which proposals are merged, it may be equally influential, particularly across architectures.

- **Sensitivity curve for λ (refinement loss weight).** λ=0.5 is stated without justification; a brief sweep would confirm robustness.

- **Application to more recent large-scale detectors (e.g., DINO, Co-DETR, YOLOv8).** The tested architectures are canonical but pre-date 2022 SOTA. Demonstrating gains on a 55+ mAP system would substantially strengthen the applicability claim for the community at the time of ICLR 2025.

- **Theoretical or geometric argument for why intersection targets ease optimization.** The empirical evidence (Figure 3b, Figure 3a) is suggestive but not mechanistically explanatory. Even an informal gradient-magnitude argument (e.g., the regression target always lies within the proposal's coordinate range, bounding prediction error) would add depth appropriate for a learning venue.

- **FLOPs/memory breakdown across all architectures.** Only Faster R-CNN FPS is reported (14.1 vs. 14.4), and only Deformable DETR training time is mentioned. A table covering inference latency and additional cost across all five models would complete the efficiency picture.

- **Proposal coverage visualisation.** Showing which spatial subregions of the ground truth each proposal covers before and after union would directly validate the "complementary views" intuition and serve as strong intuitive support.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"No variance bars / statistical significance"** — For large-scale COCO benchmarks with 5k validation images and 80-class evaluation, single-run evaluation is the standard norm. Requesting p-values or multi-run statistics is not expected practice in this setting. Removed.

- **"Tautological easier task"** — The harsh reviewer argued that lower training loss on a modified target is trivially expected. The paper goes beyond this: Table 5 (unchanged classification, improved localization) and Figure 3a (robustness to low-quality proposals) provide non-tautological evidence. Removed as a stand-alone criticism.

- **"Missing related works"** — Per instructions, not assessed.

- **"Comparison with NMS-free detectors is missing" / "No comparison with DINO/Stable-DETR"** — These are nice-to-haves for scope extension, not failures of the core contribution. The paper does include Deformable DETR. Moved to Nice-to-Haves.

- **"Circular reasoning in method motivation"** — The harsh reviewer claimed the intersection-then-union decomposition doesn't obviously beat direct regression and is circular. The paper answers this empirically (Table 1, Table 8). Whether the method is theoretically "obviously better" is not required if the empirical case is made. Removed.

- **"Deformable DETR grouping is non-trivial"** — Partially valid (the quadrant assignment is an architectural choice), already captured under the "plug-and-play overstatement" weakness above. The specific claim about Hungarian matching interactions is speculative without evidence. Partially removed / subsumed.

---

## Novel Insights

The most non-obvious observation is in Figure 3(d): as classifier quality improves (oracle experiment), the benefit of UoI over winner-takes-all *grows*, not shrinks. This is counter-intuitive — one might expect that a better classifier makes winner-takes-all more reliable. Instead, better classification means more correctly-assigned proposals covering different ground-truth subregions, and UoI can harvest all of them, creating an increasing return. This suggests that UoI's value proposition is not a patch on poor classifiers but is architecturally synergistic with classifier improvements — making it more, not less, relevant as detection systems improve.

---

## Suggestions

1. **Add the missing component ablation.** Run (a) intersection regression alone with standard NMS, and (b) standard regression with union grouping of original proposals, alongside the full UoI. This is essential for understanding the individual contributions.

2. **Clarify the mIoU computation domain** (Table 5). State explicitly whether 53.7% and 64.4% are measured on all matched proposals, on final detections after grouping, or on some other subset, and reconcile why such a large change does not produce a proportionally large mAP gain.

3. **Revise the "plug-and-play" framing.** Provide a short summary table or appendix section listing, per architecture, the exact modifications required. This improves reproducibility and avoids overpromising.

4. **Provide a fair apples-to-apples comparison with one-to-few** by applying UoI to the same Faster R-CNN + ResNet101 baseline, or explicitly frame the comparison as "UoI can leverage stronger architectures" rather than a direct performance comparison.

5. **Quantify the crowded-scene failure.** Report per-category or crowd-annotated AP changes, or at minimum show a histogram of cases where UoI merges vs. correctly separates nearby instances, to help readers calibrate where the method should not be applied.

---

**Novelty:** The intersection-based regression target and union-based post-processing constitute a genuine conceptual reframing of the detection objective that, to this reviewer's knowledge, has not been proposed before. The idea is elegant and non-obvious.

**Technical soundness:** Moderate. The method is clean and well-specified for the two-stage case. The missing component-level ablation and the unresolved mIoU/mAP disconnect are real gaps that limit full confidence in the claimed mechanistic explanation.

**Empirical support:** Adequate but not strong. Consistent positive results across five architectures build credibility, and the LRP and localization mIoU analyses add depth. The gains are uniformly modest (0.5–1.1 mAP), and a key ablation is absent.

**Significance:** Moderate. A plug-and-play improvement to the fundamental detection target formulation, if properly validated, has broad applicability. The failure mode in dense scenes and the need for architecture-specific engineering temper the significance.

**Clarity:** Good overall, with the pseudocode (Figure 2) sufficiently conveying the method. The distinction between proposal-level grouping and regressed-box grouping could be more explicitly explained in prose.

MY FINAL SCORE: <pineapple>5.2</pineapple>