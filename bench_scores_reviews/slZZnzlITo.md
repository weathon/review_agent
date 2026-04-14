## Summary
This paper introduces Multimodal Open Set Recognition (MMOSR), a new task extending OSR to multimodal data settings. The authors empirically identify a "fusion degradation" phenomenon where naively combining multimodal fusion with OSR regularization degrades both closed-set accuracy and unknown rejection ability. To address this, they propose the Multimodal Representation Reactivation Network (MRN), combining bidirectional cross-attention (mutually enhanced fusion) with a Mixture-of-Experts module (adaptive fusion). Experiments on four datasets spanning image-text, audio-visual, and RGB-depth modalities demonstrate competitive performance over multimodal fusion and single-modal OSR baselines, with gains up to 5.23% OSCR.

---

## Strengths

- **Concrete empirical diagnosis of a real failure mode.** Table 1 shows a clear degradation pattern where Fusion-OSR underperforms both Fusion alone and single-modal OSR (AUROC drops of 0.01–5.60 points depending on setting), providing direct motivation for a specialized approach rather than simply asserting that naïve combination fails. This is a useful and specific insight for the community.

- **Multi-modal, multi-dataset scope.** Evaluating across three fundamentally different modality pairs (image-text on Food-101/Flower-102, audio-visual on CREMA-D, RGB-depth on SUN RGB-D) with matched backbones is more thorough than most OSR works, which tend to be single-modality and single-dataset in their primary evaluation.

- **MRN as a plug-in backbone.** The paper shows that MRN improves performance not only standalone but also when combined with existing OSR methods (ARPL-MRN, CSRR-MRN outperform ARPL-ADD, ARPL-CAT, ARPL-GQA consistently), suggesting the fusion design is genuinely beneficial rather than a standalone artifact.

- **Robustness across openness levels.** Table 3 systematically varies known class count from 5 to 80, showing MRN maintains consistent gains over multimodal baselines at all openness levels — more thorough than a single fixed-split result.

---

## Weaknesses

### Fatal
None. The core direction is sound and the empirical results, while imperfect, are not fabricated or contradictory at a fatal level.

### Major

- **The motivating "fusion degradation" experiment (Table 1) is built on a single, weak fusion baseline.** The diagnostic uses only addition-based fusion + max-softmax (OpenAUC) on Food-101. The paper never tests whether more sophisticated fusion methods — including GQA, TMC, or MLA — also suffer degradation when combined with OSR. If cross-attention-based fusion (like GQA) combined with ARPL does not degrade, the entire premise collapses and MRN is solving a self-imposed problem. This is the most critical gap in the paper: the problem must be demonstrated on the same class of methods the solution is targeting.

- **No statistical significance — many gains are within noise range.** Results are reported as single-run point estimates with no standard deviations across seeds. Critically: SUN RGB-D gains are 0.37 AUROC / 0.01 OSCR; CREMA-D shows MRN *underperforming* MLA by 1.05 AUROC and 0.18 OSCR; Food-101 gains over MLA are 0.72 / 1.38. Without error bars over at minimum 3 seeds, none of these small margins can be claimed as reliable improvements. The claim "(1) MRN consistently demonstrates exceptional MMOSR performance across various datatypes" is contradicted by MRN's own underperformance on CREMA-D.

- **The MoE (adaptive fusion) module is never ablated.** Table 4 ablates only the cross-attention directions C₁ and C₂ within the mutually enhanced fusion module, while starting from a baseline that already includes the adaptive fusion. The contribution of the MoE module — which is framed as addressing "limited representation capability" from fusion degradation — is therefore unvalidated. Without an ablation row removing the MoE and replacing it with a simple MLP, it is impossible to know whether the gains come from the MoE structure, raw parameter capacity, or the cross-attention modules alone. This is a significant omission given that MoE with 15 experts is the architecturally heavier component.

- **CREMA-D underperformance is unexplained.** On the audio-visual dataset, MRN scores 66.78 AUROC / 57.32 OSCR versus MLA's 67.83 / 57.50 (MLA is best). This directly contradicts the claim of "consistent" superiority. The paper provides no analysis of why MRN fails on this modality pair, which raises questions about whether the fusion degradation framing generalizes to audio-visual settings or whether the cross-attention design has specific failure modes.

### Minor

- **No quantitative measurement of fusion degradation.** The core diagnostic evidence relies on t-SNE visualization (Figure 2), which is inherently qualitative and can be misleading in high-dimensional settings. Quantitative measurements — intra-class variance, inter-class margin, per-modality feature norm/activation statistics before and after OSR regularization — are needed to rigorously establish that degradation occurs and that MRN alleviates it.

- **Threshold protocol underspecified.** Section 4.3 states "The threshold τ is set to ensure 95% of the known samples are correctly classified" but does not clarify on which split this is calibrated, whether any validation set is used, or whether the threshold is fixed across all random test folds. Given that OSCR (unlike AUROC) is threshold-sensitive, this needs explicit protocol description.

- **Ablation table metric mismatch.** Table 4 reports AUROC and ACC only, while the paper's primary claim and Table 2 emphasize OSCR as the main metric. The ablation should report OSCR to be internally consistent with the evaluation framework.

- **Grad-CAM comparison targets the wrong baseline.** Figure 7 compares MRN against ARPL (a single-modal OSR method), rather than against Fusion-OSR. To demonstrate that MRN specifically recovers representations suppressed by fusion degradation, the comparison must be MRN vs. Fusion-OSR, showing what the reactivation mechanism concretely recovers over the problem configuration it is designed to fix.

- **Cross-attention equation (Eq. 1) notation is nonstandard.** The formula `Softmax(W₁^Q z₁ z₂ W₁^K / √d)(W₁^V z₂)` appears to conflate query-key projection and score computation in a way that does not match standard cross-attention formulations. The missing transpose and projection ordering ambiguity hinder reproducibility. This may be a typographical artifact of PDF extraction, but should be verified.

### Tiny

- Table 3 uses "CSSR" while Table 2 uses "CSRR" — one is a typo.
- The text in Section 3.2 says Fusion-OSR causes unknown samples to "closely resemble known clusters," but the extracted figure caption for Figure 2(d) describes unknowns as "more dispersed again." This internal inconsistency should be clarified (likely a figure-labeling confusion between Figures 1 and 2).

---

## Nice-to-Haves

- **Capacity-matched MLP control.** Include an ablation replacing the MoE with a single MLP of equivalent total parameter count. This isolates whether the MoE *structure* (specialized experts with routing) matters versus pure capacity increase.

- **Computational cost analysis.** A brief table of parameter counts and inference latency for MRN vs. strongest baselines (MLA, TMC) would address practical viability concerns for the embedded/robotics use case in the introduction.

- **Ensemble of single-modal OSR models as baseline.** A simple prediction-ensemble of per-modality OSR classifiers is a natural competitive baseline that the paper does not include. Demonstrating that MRN beats this would more directly justify the fusion-based approach.

- **Theoretical or geometric intuition for fusion degradation.** Even a loss-landscape sketch or mutual information analysis showing why cross-entropy + OSR regularization conflicts with multimodal fusion alignment would strengthen the motivation beyond the single empirical observation.

- **Failure case analysis.** Showing examples where MRN still fails (misclassified unknowns, incorrect rejections of knowns, audio-visual cases) would delineate the method's scope and guide future work.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Criticism that multimodal OOD detection literature is absent as a baseline.** The paper cannot be expected to compare against every adjacent setting; the OSR vs. OOD detection distinction is well-established in the field. Absence of MCM-style baselines is at most a nice-to-have, not a flaw undermining the contribution.

- **Criticism that OSR baselines (ARPL 2022) are "dated."** The paper includes CSRR (TPAMI'23), ASH (ICLR'23), MLA (CVPR'24), and OpenAUC (NeurIPS'22), which is a reasonable suite spanning the last three years. The criticism that ARPL is outdated ignores that it remains a relevant strong discriminative baseline and is used in combination experiments specifically.

- **Criticism about missing theoretical proof for fusion degradation.** This is an empirical systems paper proposing a new task and method. Requiring formal theory is outside the standards of the OSR/multimodal community for this type of contribution.

- **Criticism of Figure 1 as a "conceptual cartoon."** All motivation figures are conceptual illustrations; this is standard practice. The figure is appropriate for its purpose.

- **Demanding multi-modality-count scalability analysis (3+ modalities).** The paper explicitly scopes to dual-modality for the main contribution and notes the pairwise extension. Criticizing the absence of a 3-modality experiment is scope creep.

- **Criticism that CLIP/CoOp/MaPLe comparison is "unfair."** The direction of any potential unfairness benefits the baselines: CLIP uses far larger pretraining data than MRN. That MRN outperforms these pretrained models despite less supervision is, if anything, a stronger result.

---

## Novel Insights

The most genuinely valuable insight in this paper — which neither sub-reviewer fully articulated — is that the failure of Fusion-OSR is *directional*: fusion improves closed-set ACC (by pooling modality information) while simultaneously *hurting* unknown detection (AUROC decreases vs. best single-modal baseline in most settings of Table 1). This asymmetric failure pattern is a specific and actionable diagnostic: it suggests that OSR regularization and multimodal alignment objectives are geometrically at odds in the representation space. The cross-attention + MoE combination addresses this by maintaining modality-specific discriminative signals through the cross-attention read-out, while the MoE provides diverse representation pathways that resist the homogenization imposed by OSR compactness constraints. Whether this mechanism is genuinely the cause of the fix or whether the improvement comes simply from capacity and better optimization remains unresolved by the ablations, but the directional failure observation itself is a meaningful and testable hypothesis for the community.

---

## Suggestions

1. **Replicate Table 1's diagnostic with stronger fusion methods (GQA, MLA combined with OSR).** If GQA+ARPL and MLA+CSRR also degrade relative to their fusion-only counterparts, the fusion degradation claim becomes substantially more credible and general. If they don't, revise the problem framing accordingly.

2. **Add a full ablation row removing the MoE (replace with single MLP, matched parameters).** This is the single most important missing experiment. Results in Table 4 should include: encoder-only, encoder + single MLP, encoder + MoE only (no cross-attention), and the full MRN.

3. **Report mean ± std over at least 3 random class splits/seeds for all main results.** Given margins of 0.01–1.38 OSCR on some datasets, this is necessary to make any quantitative claim credible.

4. **Add a quantitative degradation measure.** Report intra-class feature variance and mean pairwise inter-class distance for (a) Fusion-only, (b) Fusion-OSR, (c) MRN — this directly validates the core diagnostic claim with a single table.

5. **Explain and analyze CREMA-D underperformance.** Analyze why audio-visual data breaks the pattern, whether the cross-attention design is ill-suited to asynchronous audio-visual sequences, and what modification could address it.

6. **Fix Eq. 1 notation.** Verify that the cross-attention formula matches the implementation; provide explicit tensor shapes in an appendix to support reproducibility.

---

**Evaluation summary:**
- *Novelty:* Moderate — formalizing MMOSR as a task is useful; the architectural components (cross-attention, MoE) are standard, but their combination is purposefully motivated.
- *Technical soundness:* Weak-to-moderate — the core diagnostic experiment is insufficiently thorough, the ablation is missing a key component, and the cross-attention equation has notation issues.
- *Empirical support:* Moderate — multi-dataset evaluation is a strength, but lack of statistical testing and the CREMA-D underperformance weaken the empirical case materially.
- *Significance:* Moderate — MMOSR is a real and important problem; the paper opens a research direction more than it closes one.
- *Clarity:* Acceptable — the paper is generally readable, but internal inconsistencies between Figure 2 labels and prose, and between the ablation metrics and main evaluation metrics, detract from overall coherence.