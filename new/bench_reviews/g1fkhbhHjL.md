Now let me run calibration searches to anchor my score.Now I have sufficient information to write the final consolidated review.

---

## Summary

This paper identifies a phenomenon termed "Black Sheep in the Herd": a small fraction (<7%) of spurious attributes in VLM attribute pools disproportionately dominate model predictions, harming out-of-distribution generalization. To address this, the authors propose (1) **Spurious Attribute Probing (SAP)**, which uses a MLLM (GPT-4V) and a Concept Bottleneck Model (CBM) to automatically identify spurious attributes and remove them from existing attribute-based methods; and (2) **Spurious Attribute Shielding (SAS)**, a plug-and-play auxiliary training module that introduces spurious-attribute pseudo-categories to discourage models from relying on spurious features. Both modules are evaluated across 11 datasets, 3 generalization tasks, and 11 PEFT baselines, with consistent gains in OOD accuracy.

---

## Strengths

- **Well-illustrated and novel problem identification (Table 1, Figure 1):** The finding that spurious attributes occupy only <7% of the attribute pool yet rank in the top-3 CBM-weighted features is specific and compelling. Manually removing them yields +2.36% and +1.62% on new-category accuracy (CPL and ArGue, respectively), a clear motivating oracle result.

- **Broad empirical evaluation:** Results are reported over 11 datasets, 3 generalization task settings, and 11 diverse PEFT baselines (prompt tuning, adapters, LoRA, attribute-based), demonstrating consistent OOD accuracy gains of over 2% on average. The plug-and-play integration without modifying base architectures is technically non-trivial.

- **Adaptive threshold strategy validated (Table 4):** The adaptive γ strategy — setting γ_c as the lowest weight among core attributes — outperforms all fixed thresholds (HM 80.38 vs. best fixed 79.81), validating the design choice. The monotonic degradation at both extremes also provides partial evidence that spurious attribute identity matters beyond just data quantity.

- **Efficiency analysis (Table 5):** The selective trick (optimizing only 10% of categories) reduces training overhead substantially (CoCoOp: 6h18m → 4h51m) while preserving most accuracy gain (+0.92 vs. +1.11), a practically useful contribution.

- **Counter-group evaluation (Table 2):** SAS consistently improves more on the counter group than on the standard test set (up to ~6% counter-group gain vs. <1.5% on the full test), providing directional evidence that the model is learning to operate without spurious features present.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing control for SAS's core mechanistic claim (Sections 3.4, Table 4):** SAS introduces both (a) synthetic data (16 images per pseudo-category via Stable Diffusion) and (b) an auxiliary classification loss. The paper argues in Section 4.2 that performance degrades when γ is too high or too low, implying spurious-attribute identity matters. However, when γ=0.0, the pseudo-categories are constructed from *all non-core attributes* identified by SAP, not from random attributes. Consequently, there is no condition where pseudo-categories are formed from truly random or non-spurious attributes at matched data volume. Without this control, the reader cannot rule out that SAS's gains come from domain randomization / auxiliary-task regularization in general rather than from the specific mechanism of spurious-attribute pseudo-categories. This is the most important missing experiment, and it is needed to credibly support the central mechanistic claim.

- **SAP's pipeline is not quantitatively validated against the human-annotation oracle (Section 3.3):** Table 1 demonstrates that manual removal of spurious attributes significantly improves generalization, motivating the paper. However, the paper never shows how closely SAP-automated removal approximates this oracle. There is no precision/recall analysis of GPT-4V's core/non-core classification, and no row in Table 1 (or equivalent table) showing SAP-removal results alongside manual-removal results. The paper establishes that the oracle works and that SAP produces spurious attributes, but leaves the critical gap — how much of the oracle's benefit is recovered by SAP — unaddressed. This is essential for establishing that SAP is a useful substitute for human annotation.

### Minor

- **Circularity in the counter-group construction (Section 4.1, Table 2):** The counter group is constructed by filtering test images with high semantic similarity to SAP-identified spurious attributes, and SAS is trained on the same SAP-identified attributes. While the counter-group results are still informative (the gap between counter-group and test-set gains is meaningful), the evaluation is not fully independent of the training signal. The authors should acknowledge this explicitly and discuss its implications.

- **Number of images per SD prompt not fixed in Table 3 ablation (Section 4.2):** Table 3 varies the number of SD prompts from 1 to 7 and shows HM improves from 78.87 to 80.38. If each prompt generates a fixed number of images, varying prompt count also varies total synthetic data volume. The paper does not clarify whether total image count is controlled, making it ambiguous whether the gain reflects prompt diversity or data quantity.

- **Manual annotation procedure not reproducible without authors (Section 3.2):** The motivating study in Table 1 relies on the authors manually reviewing 5 sampled images with heatmap activations per attribute. The procedure introduces uncontrolled subjectivity and is not reproducible without author involvement, limiting the interpretive value of Table 1 as a rigorous oracle. This does not undermine the overall paper, but it should be disclosed.

### Trivial

- **Qualitative evidence only for saliency shifts (Figure 5):** The saliency map analysis is cherry-picked and lacks quantitative measurement of attention shift across a held-out set. Reporting a metric (e.g., proportion of top-20% saliency mass on target object across a validation split) would strengthen this analysis.

---

## Nice-to-Haves

- **Random pseudo-category control:** Adding a condition where SAS uses randomly sampled *non-spurious* attributes (matched on data volume and number of pseudo-categories) would definitively test whether the spurious-attribute specificity matters.

- **Table showing SAP-removal vs. manual-removal on the same datasets:** Even approximate precision/recall of SAP would substantially strengthen the claim that it meaningfully approximates the human oracle. Adding a "SAP-SA" row to Table 1 is a natural experiment.

- **Open-source MLLM ablation:** The method depends on GPT-4V; an experiment with an open-source MLLM (e.g., LLaVA) would clarify whether SAP generalizes beyond GPT-4V and expand practical applicability.

- **Per-dataset breakdown of SAS gains:** Figure 3 averages over 11 datasets. Reporting per-dataset results in the main text (even selectively) would let readers identify cases where SAS may not help, making the contribution more informative.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: Top-k selection not detailed in main text.** The paper explicitly defers this to Supp. Mat. A. Per policy, criticisms about missing appendix details are removed.

- **Harsh Critic: Per-dataset results deferred to Supp. Mat.** Detailed per-dataset breakdowns are in the supplementary, which exists in the original submission. Removed.

- **Harsh Critic: Implementation details (hyperparameters) not reproducible.** Removed per policy on nitpicks about implementation details deferred to supplementary.

- **Harsh Critic: GPT-4V / ChatGPT / SD pipeline cost and variability.** While a practical concern, the paper does provide implementation details and uses a fixed setup. This does not constitute a methodological flaw.

- **Strength Finder: "The paper addresses a real practical limitation" (generic strength).** Removed as generic without a specific table/figure citation.

---

## Novel Insights

The paper surfaces a genuinely underappreciated failure mode in attribute-based VLM adaptation: the attribute pool itself is contaminated by a small fraction of spuriously correlated concepts that CBMs assign outsized weight to, creating a "black sheep" dynamic where few attributes dominate and hurt OOD generalization. The dual-module framework — one operating at the language representation level (SAP) and one at the visual feature learning level (SAS) — is a structurally coherent response to the two distinct pathways through which spurious attributes harm generalization. The insight that CBM weights can serve as a quantitative proxy for spurious correlation strength, enabling an adaptive per-category threshold, is technically non-obvious and well-motivated by the observed weight distribution variability across categories.

---

## Suggestions

1. Add a "random pseudo-category" control to Table 4 (or a new ablation table) to isolate the value of spurious-attribute specificity in SAS.
2. Add a "SAP-removal" row to Table 1 to directly quantify how much of the manual-oracle gain SAP recovers.
3. Report precision/recall of SAP against the manual annotations used in Table 1 (even approximately), either in the main text or a clearly referenced appendix.
4. Clarify whether total image count is controlled when varying #p in Table 3, and if not, add a matched-data-volume condition.
5. Acknowledge and briefly discuss the circularity between counter-group construction and SAS training in Section 4.1.

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Human Score | Relation to this paper |
|---|---|---|
| MetaCoCo (spurious correlation FSC benchmark) | 7.0 (Accept poster) | Topically related; identifies spurious correlations in FSC but proposes only a benchmark, not a method; comparable empirical breadth |
| AttrVR (attribute-based VR for CLIP) | 6.0 (Accept poster) | Same scope — attribute-based VLM adaptation; similar breadth of evaluation; comparable methodological rigor and similar types of unaddressed weaknesses |
| kIP0duasBb (test-time adaptation for CLIP) | 6.67 (Accept) | Overlapping topic; stronger theoretical grounding but narrower evaluation |
| rCaA79Obsj (circular evaluation issue) | 5.20 (Reject) | Paper rejected primarily for circular evaluation — a concern that is present here but less central and less severe |
| EKfcngSxwD (VLM parameter adaptation, missing ablations) | 4.67 (Reject) | Rejected for lack of ablation isolating contributions; this paper has more ablation coverage |
| Low-scoring anchors (avg <3) | 1.5–2.33 | Fundamentally flawed or trivial contributions; this paper is clearly above these |

**Positioning:** The paper is most similar to AttrVR (avg 6.0) in scope, novelty level, and strength of evaluation. It has a slightly more novel problem framing (the "black sheep" phenomenon is better-motivated than AttrVR's descriptive/distinctive attribute split) and broader empirical validation (more baselines and tasks). However, it has real methodological gaps — most notably the missing random-pseudo-category control and the unvalidated SAP pipeline — that are comparable in severity to the weaknesses flagged in medium-scoring papers. The circular evaluation concern, while valid, is less severe than in the paper scored 5.20 (where it fundamentally undermined the benchmark). This paper's evidence remains directionally informative. The paper does not rise to the level of MetaCoCo (7.0) or the 7.5+ papers, which had both more rigorous evaluation and fewer structural gaps.

**Final score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>