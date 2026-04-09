##Summary

The paper proposes an end-to-end framework for multi-view diabetic retinopathy (DR) grading that eliminates the need for external lesion/vessel annotations by self-generating lesion proposals during training and inference. A Grade-Activated Lesion Proposal (GALP) module derives grade-conditioned evidence maps from stage-wise auxiliary classifiers and selects Top-K high-evidence regions as proposals, while a Cross-View Lesion Expert-Guided Regional Fusion (LGRF) module routes these proposals through a gated mixture-of-experts mechanism for selective cross-view fusion. Experiments on two multi-view DR benchmarks (MFIDDR and DRTiD) show that the method matches or surpasses externally-informed baselines without requiring auxiliary annotations.

## Strengths

- **Effective reduction of annotation dependency while preserving accuracy:** The lesion-free variant (83.9% Acc on MFIDDR) surpasses all end-to-end baselines and several externally-informed methods (e.g., LFMVDR with lesion at 82.2%, CVSA with vessel at 82.6%), directly validating the core claim that self-derived proposals can substitute for costly external cues. This is the paper's strongest result and is clearly demonstrated in Table 1.

- **Well-motivated architectural design for the multi-view medical setting:** The GALP→LGRF pipeline is logically coherent: GALP generates spatially-focused proposals that reduce background noise, and LGRF uses cross-view context to gate which experts process those proposals. The ablation (Table 4) confirms that each component contributes meaningfully (removing GALP drops accuracy by 1.2%, removing LGRF by 1.6%).

- **Consistent improvements across two datasets with different view counts:** The method generalizes from 4-view (MFIDDR) to 2-view (DRTiD) settings, achieving 76.0% accuracy on DRTiD versus the prior SOTA CrossFiT (75.6%) that requires OD/macular coordinates. This consistency strengthens the claim of architectural generality within the multi-view DR domain.

## Weaknesses

### Major:

- **The 50% token retention ratio undermines the "lesion proposal" framing and suggests the mechanism acts as a general saliency filter rather than a lesion localizer.** DR lesions typically occupy a very small fraction of the fundus image (often <5%). Retaining 50% of spatial tokens (Section 4.1, confirmed optimal in Fig. 3) means the "proposals" include substantial non-lesion content. The paper's narrative—that GALP identifies "lesion proposals" that "concentrate evidence on grade-relevant areas" and "reduce distraction from non-lesion background" (Section 3.2)—is weakened by this finding. The mechanism appears to function as a coarse saliency pruning step rather than a lesion-specific proposal generator. This matters because it affects interpretability claims and the precision of the cross-view fusion: if proposals include large amounts of background, the LGRF module may be attending to salient but non-lesion structures. The authors should either (a) explain why 50% is appropriate despite the small lesion footprint, (b) provide evidence that the selected regions correlate with actual lesion locations, or (c) recalibrate the framing from "lesion proposals" to "grade-salient region proposals."

- **The ablation study conflates the effects of auxiliary supervision and proposal selection, leaving the source of improvement ambiguous.** In Table 4, "w/o GALP" removes both the auxiliary classification loss and the Top-K proposal selection, replacing them with direct use of all tokens. This makes it impossible to determine whether the performance gain comes from (a) the stronger feature supervision via the auxiliary loss, or (b) the spatial focusing of the Top-K selection, or (c) both equally. A cleaner ablation would include "GALP without proposal selection" (auxiliary loss retained, all tokens passed to LGRF) and "proposal selection without auxiliary loss" (Top-K on raw features without the auxiliary classifier). Without this disentanglement, it is unclear whether the complex proposal mechanism is necessary or whether auxiliary supervision alone would suffice—a significant gap for a paper whose central contribution is the proposal mechanism.

### Minor:

- **No computational efficiency analysis is provided despite significant architectural overhead.** The method adds 4 stage-wise auxiliary classifiers, CAM computation, Top-K selection, an MoE routing network with 6 experts, and cross-view attention at each stage. The conclusion claims the method provides a "scalable solution for large-scale DR screening" without reporting FLOPs, parameter counts, or inference latency versus baselines (e.g., MVCINN or Swin-B). Given that clinical screening requires fast turnaround, this omission limits practical assessment.

- **No qualitative validation that self-derived proposals correspond to actual lesion locations.** The paper claims GALP "derives grade-conditioned evidence maps" and that selected regions are "more likely to contain lesion evidence" (Section 3.2), but provides no visualization comparing proposals against ground-truth lesion masks (which are available in MFIDDR). Without this, it is difficult to verify whether the proposals capture clinically meaningful structures or are statistical artifacts of the auxiliary classifier. This is especially important given the 50% retention ratio concern above.

- **The bootstrap dynamics of the proposal mechanism during early training are unaddressed.** GALP generates proposals from the auxiliary classifier's predictions (Eq. 3), but in early training these predictions are essentially random, meaning GEMs will highlight incorrect regions. The paper does not discuss how the model stabilizes—whether the main loss gradient corrects proposals quickly enough, or whether noisy early proposals significantly slow convergence. A brief discussion or convergence analysis would strengthen confidence in the training procedure.

### Trivial:

- **Gradient flow through the hard Top-K selection step is not explicitly discussed.** The Top-K region selection (Eq. 4-7) involves hard indexing, but the masked average pooling (Eq. 5) and linear projection (Eq. 6) are differentiable with respect to the features. Gradients flow primarily through the auxiliary loss to the features, not through the selection itself. While this is a common pattern (similar to DETR's hard assignment), a brief clarification would be helpful.

## Nice-to-Haves

- **Cross-dataset generalization test** (train on MFIDDR, test on DRTiD or vice versa) to assess robustness to domain shift across imaging devices and populations.

- **Expert specialization analysis:** Whether the M=6 experts in LGRF learn meaningfully distinct lesion morphologies or degenerate into redundant feature extractors. If experts are redundant, the MoE complexity may be unjustified.

- **View dropout robustness test:** Evaluate performance when one or more views are artificially masked during inference, which is clinically relevant when certain views are unavailable.

- **Clinical screening metrics:** Sensitivity at fixed specificity thresholds (e.g., 95% specificity) would better align with deployment requirements than accuracy alone.

- **Lower retention ratios explored more granularly:** Testing α ∈ {0.05, 0.10, 0.20, 0.30, 0.50} would clarify the lesion-localization vs. saliency-filtering behavior.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Missing statistical significance / error bars:** All three reviewers requested standard deviations over multiple seeds. However, single-run reporting is standard practice in the medical imaging and multi-view DR grading community (all baselines in Tables 1-3 also report single values). Demanding confidence intervals for large-scale benchmarks where single-run evaluation is the norm falls under methodological practices not standard in this field. Moved to nice-to-have consideration but not a core flaw.

- **Baseline fairness / different backbone pretraining:** The spark finder questioned different pretraining strategies across datasets. The paper explicitly follows each baseline's own protocol (ImageNet for MFIDDR matching CVSA, EyePACS for DRTiD matching CrossFiT), which is the correct approach for fair comparison. This is not a weakness.

- **"Ours (with lesion)" contradicting the core contribution:** The spark finder suggested the with-lesion variant contradicts the paper's message. However, the paper clearly demarcates "Ours (w/o lesion)" as the primary result and presents the with-lesion variant as demonstrating that the architecture can additionally benefit from external information when available. This is a feature, not a contradiction.

- **Garbled equation in load balancing loss (Eq. 11):** This is a PDF parser artifact, not an author error. Removed as a formatting nitpick per hard rules.

- **Equation 6 appearing abruptly / structural confusion:** Formatting/ordering issue attributable to parser artifacts. Removed per hard rules on formatting nitpicks.

- **Method not generalizable beyond DR:** Criticizing a DR grading paper for not demonstrating transfer to OCT or other modalities is scope creep. The paper's stated scope is multi-view DR grading; evaluating it on that scope is appropriate.

- **MFIDDR segmentation masks are model-generated, not human-annotated:** This concern applies to the "with lesion" variant only, which is a secondary result. The core contribution (w/o lesion variant) is unaffected.

- **Per-grade performance analysis missing:** Factually incorrect—the paper provides detailed per-grade F1, Precision, and Specificity in Table 2 for all 5 DR grades.

## Novel Insights

The most revealing finding is the tension between the paper's "lesion proposal" narrative and the 50% optimal retention ratio. This suggests that what GALP actually provides is not lesion localization per se, but rather a form of learned spatial prior that coarsely separates diagnostically relevant retinal regions from irrelevant periphery. The fact that this coarse prior, when coupled with cross-view expert gating, still matches externally-informed methods is itself an interesting result: it implies that much of the benefit of external lesion/vessel annotations in prior work may come from simply directing attention away from background rather than from precise lesion delineation. If validated (e.g., by showing that even random spatial priors within the retinal area perform similarly), this could reshape how the community thinks about the role of auxiliary annotations in medical image analysis.

## Suggestions

- **Rename "lesion proposals" to "grade-salient proposals" or "evidence proposals"** throughout the paper, or provide quantitative IoU validation against MFIDDR's lesion masks to justify the current terminology. This would align the framing with the empirical finding (50% retention) and strengthen credibility.

- **Add a two-row ablation** isolating auxiliary supervision from proposal selection: (1) auxiliary loss only (all tokens to LGRF), and (2) Top-K selection only (no auxiliary loss, proposals from raw feature norms). This directly addresses the major methodological gap in the current ablation.

- **Provide 2-3 visual examples** of GEM overlays alongside lesion segmentation masks on MFIDDR to qualitatively demonstrate what the proposals capture. This is low-cost and would significantly strengthen the interpretability claims.

- **Include a computational cost table** (FLOPs, parameters, inference time) comparing against MVCINN and Swin-B baselines, given the "scalable solution" claim in the conclusion.