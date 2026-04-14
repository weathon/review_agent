## Summary

This paper proposes Multimodal Open Set Recognition (MMOSR), extending OSR to multimodal input settings (image-text, audio-visual, RGB-depth). The authors empirically demonstrate that naively combining multimodal fusion with standard OSR regularization causes "fusion degradation," where OSR's compaction pressure suppresses modality-specific representations and reduces unknown detection ability. To address this, they introduce the Multimodal Representation Reactivation Network (MRN), combining a cross-attention-based mutually enhanced fusion module with an MoE-based adaptive fusion module, and benchmark it across four datasets against a broad set of baselines.

---

## Strengths

- **Concrete evidence of a previously unreported failure mode.** Table 1 presents a specific, reproducible result: applying OpenAUC on top of addition-fused multimodal features (Fusion-OSR) degrades AUROC by up to 5.60 points versus naive Fusion alone, and underperforms the single text modality (Text-OSR achieves 91.57 vs. Fusion-OSR's 86.43 at the 20-class split). This is a genuine empirical finding that motivates the paper's direction regardless of the theoretical explanation.

- **Broad multi-dataset, multi-modality benchmark construction.** The paper evaluates on image-text (Food-101, Flower-102), audio-visual (CREMA-D), and RGB-depth (SUN RGB-D) data, covering three fundamentally different modality types. The baseline comparison spans single-modal OSR, multimodal fusion, fusion+OSR combinations, and pretrained vision-language models (CLIP, CoOp, MaPLe). This breadth is a practical contribution to the community beyond what most OSR papers offer.

- **MRN as a drop-in backbone consistently improves paired OSR methods.** When MRN replaces other fusion strategies underneath ARPL and CSRR (Table 2, "Multimodal fusion with OSR methods" rows), it improves both AUROC and OSCR consistently across all four datasets and both OSR methods — including CREMA-D (ARPL-MRN: 64.37/56.79; CSRR-MRN: 66.86/56.90, both best in their group). This is a more robust demonstration of MRN's value than the standalone comparison.

- **MRN, trained from scratch, outperforms large pretrained vision-language models on fine-grained OSR.** Table 3 shows MRN consistently exceeds CLIP (zero-shot), CoOp, and MaPLe (16-shot fine-tuned) across all known/unknown splits on Food-101, despite having substantially fewer pretraining resources. This is a non-obvious and practically meaningful finding about the limits of foundational model transfer to OSR-specific tasks.

---

## Weaknesses

### Fatal
None.

### Major

- **CREMA-D failure is unreported and actively misrepresented.** In Table 2, standalone MRN scores 66.78 AUROC and 57.32 OSCR on CREMA-D, while MLA scores 67.83 AUROC and 57.50 OSCR — MRN is worse on both metrics. The Gain row explicitly confirms this with (1.05↓) and (0.18↓). Yet the paper bolds the MRN row for CREMA-D and in the text claims MRN "consistently demonstrates exceptional MMOSR performance across various datatypes." This is a factual misrepresentation: the caption says "best results are marked in bold," but MRN's CREMA-D values are bolded despite not being best. The paper provides no discussion of this failure or why the reactivation mechanism does not help for audio-visual data.

- **No statistical significance reporting for marginal gains.** Gains on SUN RGB-D (+0.37 AUROC, +0.01 OSCR over MLA) and Food-101 (+0.72 AUROC, +1.38 OSCR) are very small; no standard deviations, confidence intervals, or multiple-seed results are reported anywhere. Without variance information, it is impossible to determine whether these differences are meaningful or within random variation. This is particularly damaging alongside the CREMA-D failure.

- **Ablation study does not cover the adaptive fusion (MoE) component.** Table 4 ablates only cross-attention branches C1 and C2; the very first row (neither C1 nor C2) already includes the adaptive fusion module. There is no experiment removing MoE entirely or substituting it with an equivalent-capacity single MLP. The MoE adaptive fusion is half the proposed method, and its independent contribution to MMOSR performance is entirely unverified.

- **The core motivating experiment (Section 3.2) is too narrow.** The "fusion degradation" analysis uses only one dataset (Food-101) and one OSR method (OpenAUC). The paper then generalizes this to a universal phenomenon and uses it to justify a new task. CREMA-D in Table 2 does not exhibit the same pattern (MLA, a pure fusion method, outperforms MRN there). A robust motivating claim requires evidence across multiple modality pairs and multiple OSR methods in the motivating section.

- **The most natural alternative baseline is absent: score-level fusion of per-modality OSR.** Running OSR independently on each modality and combining rejection scores (e.g., taking the max/product/mean of per-modality MSP) is the most obvious alternative to early-fusion MRN. Without this baseline, there is no evidence that the proposed architectural complexity is necessary, rather than simply ensembling single-modal decisions.

### Minor

- **Unknown rejection relies entirely on standard MSP thresholding, with no OSR-specific mechanism.** Section 4.3 uses maximum softmax probability with a percentile threshold. The training loss (Eq. 3) is classification plus load balancing, with no open-set objective, margin shaping, or uncertainty term. The paper frames MRN as an MMOSR method, but its open-set detection is post-hoc and no different from applying MSP to any classifier. The gains may stem purely from better representations; this should be acknowledged and the scoring function examined.

- **Threshold calibration is underspecified.** Section 4.3 states τ is "set to ensure 95% of the known samples are correctly classified," but does not specify which split (training, validation, or held-out known set). If the training set is used, behavior at test time may differ; if a validation set is used, the protocol must be stated for reproducibility. This also raises the question of sensitivity: how much do AUROC/OSCR change under different percentile thresholds?

- **"Fusion degradation" is not quantitatively operationalized.** The concept is central to the paper's motivation but defined only via t-SNE plots and one performance table on one dataset. No measurable quantity — such as effective feature rank, inter-class margin, modality contribution entropy, or feature norm statistics — is computed to verify the phenomenon or confirm it is resolved by MRN.

- **Grad-CAM comparison (Figure 7) is against a single-modal baseline (ARPL), not a multimodal one.** Comparing MRN's visual attention maps to ARPL — which operates on one modality — cannot demonstrate that MRN's cross-modal reactivation mechanism is responsible for better attention. A comparison against MLA or GQA would be far more informative.

- **Ablation study reports ACC but not OSCR.** Table 4 evaluates the fusion modules using AUROC and ACC. For a paper framed around MMOSR, ablations should report OSCR, which jointly evaluates open-set detection and closed-set classification, rather than ACC alone.

### Tiny

- Sensitivity analysis figures (Figures 4 and 5) show metric curves without error bars, making stability claims qualitative.
- No computational overhead analysis (parameter count, FLOPs, inference time) is provided for the MoE component, relevant given the motivating deployment scenario of robotic systems.
- The text in Section 3.2 refers to both Figure 2c (Fusion) and 2d (Fusion-OSR) while writing "Fusion-OSR methods over-compress," slightly obscuring which phenomenon applies to which model.

---

## Nice-to-Haves

- **Quantitative measurement of fusion degradation**: Compute feature rank, intra-class compactness, or modality contribution entropy before/after OSR regularization and after MRN reactivation across multiple datasets. This would transform a qualitative narrative into a verifiable mechanistic claim.
- **Alternative OSR scoring functions on top of MRN representations**: Compare energy-based scores, logit margins, or prototype distances on MRN features to determine whether MSP is a bottleneck and whether a dedicated scoring function could recover CREMA-D performance.
- **Deeper characterization of CREMA-D failure**: Analyze whether audio-visual modality structure differs in ways that cause the cross-attention "class-relevant = cross-modally correlated" assumption to break down — e.g., modality-private prosodic cues that are discriminative but not aligned with visual frames.
- **Extension of Section 3.2 to at least two modality pairs**: Include CREMA-D or SUN RGB-D in the motivating fusion degradation analysis, even if the result there is weaker, to establish the generality and boundary conditions of the phenomenon.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Training from scratch disadvantages pretrained baselines** (Critic): The paper explicitly states this choice is "to avoid introducing unknown information," and the design is applied uniformly to all methods in Table 2. Pretrained models appear only in Table 3 as a separate comparison group, where the asymmetry favors those baselines (they get pretraining; MRN does not). This is an intentionally conservative comparison that strengthens, not weakens, the paper's claims. Removed per the rule on unfair comparisons that favor baselines.
- **Missing related works on multimodal OOD detection / multi-view uncertainty** (Critic, Spark): Removed per instructions — cannot confirm existence of specific references without external sources.
- **"MLLMs era" discussion is aspirational** (Critic): The forward-looking paragraph in Section 5.3.3 is appropriately framed as future motivation and does not make empirical claims. Not a methodological flaw.
- **GQA not being an appropriate fusion baseline** (Critic): GQA is a published method used here as a multimodal fusion architecture with grouped-query attention; its use as a fusion baseline is reasonable and its existence is confirmed by the citation.
- **Equation 1 tensor shape ambiguity** (Critic): The cross-attention formulation follows standard conventions; the intent is unambiguous and this is a minor notation concern, not a reproducibility failure at the level warranting inclusion.
- **Broader impact section missing**: Not relevant to technical merit evaluation at this venue.
- **"The new task is just a reframing"** (Critic): The benchmark construction across four heterogeneous datasets, with reproducible experimental protocols and the new MMOSR framing, constitutes a genuine contribution independent of how novel the formal definition is.

---

## Novel Insights

The paper's most transferable insight is a *negative* finding: OSR regularization, whose purpose is to compact known-class representations to leave decision space for unknowns, is structurally at odds with what multimodal fusion needs — diverse, modality-specific representations. The concrete evidence in Table 1 (Fusion-OSR underperforming both Fusion and the single best modality at AUROC) is a novel empirical finding with architectural implications beyond this paper. A second observation from Table 3 is equally noteworthy: well-trained task-specific multimodal fusion models, even those trained entirely from scratch, substantially outperform large pretrained vision-language models (CLIP, CoOp, MaPLe) across all tested known/unknown ratios on fine-grained OSR tasks, challenging the assumption that foundation models transfer effectively to rejection-aware recognition. Together, these observations — that OSR and early fusion interact adversarially, and that task-specific training dominates large-scale pretraining in this setting — provide a useful framing for follow-on work.

---

## Suggestions

1. **Correct and discuss the CREMA-D reporting.** Remove erroneous bolding of MRN in Table 2 for CREMA-D, acknowledge the underperformance explicitly in the text, and analyze why audio-visual data may not benefit from cross-attention reactivation (e.g., modality-private discriminative cues).

2. **Add a score-level fusion baseline.** Run OSR independently per modality and combine rejection scores via max/product/average pooling. Report this on all four datasets. This is the first experiment a reviewer will ask for and its absence is the paper's most conspicuous gap.

3. **Report mean ± std over multiple random known/unknown splits.** Even three to five seeds would allow readers to assess whether marginal gains (e.g., +0.37 AUROC on SUN RGB-D) are reliable. This is especially necessary given the CREMA-D failure.

4. **Add a proper MoE ablation row to Table 4.** Replace the MoE module with a single MLP of matched parameter count, and report AUROC and OSCR (not just ACC). Without this, the contribution of the adaptive fusion component cannot be assessed.

5. **Expand Section 3.2 to include at least one additional modality pair.** Rerun the Fusion vs. Fusion-OSR motivating experiment on CREMA-D or SUN RGB-D to establish whether fusion degradation is a general phenomenon or specific to image-text data with a dominant modality.

6. **Quantify fusion degradation with a concrete metric.** Compute effective feature rank or inter-class margin statistics for Fusion, Fusion-OSR, and MRN across datasets. This would convert the central motivating claim from a qualitative observation into a verifiable and replicable finding.