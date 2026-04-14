=== CALIBRATION EXAMPLE 53 ===

# Final Consolidated Review
## Summary
This paper investigates how pretraining data frequency drives the formation of Linear Relational Embeddings (LREs) in transformer language models. The authors demonstrate a strong correlation (Pearson r = 0.82) between subject-object co-occurrence counts in pretraining corpora and the quality of linear representations for factual relations, identify per-model co-occurrence thresholds above which high-quality LREs consistently emerge (regardless of when during training the threshold is crossed), and show that LRE metrics can be used to predict pretraining term frequencies in closed-data models. The authors also release a Batch Search tool for efficiently counting token co-occurrences in tokenized training sequences.

---

## Strengths

- **Quantitatively specific frequency-linearity relationship.** The paper establishes that subject-object co-occurrence—not subject or object frequency alone—is the right unit of analysis (r = 0.82 vs. 0.66 subject-only, 0.59 object-only for OLMo). This is a precise, model-grounded finding that most prior work gestures toward but does not pin down.

- **Training-checkpoint-invariant threshold discovery.** Figure 2 is a genuine result: regardless of which pretraining checkpoint a model is at, once average co-occurrence crosses ~2k (OLMo-7B), ~4k (OLMo-1B), or ~1k (GPT-J), causality is consistently high. The use of 8 training checkpoints spanning 41B–2T tokens strengthens this claim considerably.

- **Novel connection: causality can lead few-shot accuracy.** Section 4.3 identifies that for some relations (e.g., *food-from-country*, 65% causality vs. 42% 5-shot accuracy), the linear representation forms before the model can reliably apply it in prompting. This checkpoint-level dynamic is a specific and underexplored mechanistic observation with implications for understanding in-context learning.

- **Practical tool contribution.** The Batch Search tool—counting token co-occurrences within tokenized training sequences rather than raw documents—fills a genuine gap in the interpretability research infrastructure. The comparison to WIMBD (slope = 0.94) validates it and the tool is released.

- **Data auditing application.** Repurposing LRE metrics to predict pretraining term frequencies in closed-data models is conceptually novel. The demonstration that a regression trained on OLMo features transfers meaningfully to GPT-J (a different dataset, tokenizer, and architecture) without retraining is a principled and practical contribution to the emerging data auditing literature.

---

## Weaknesses

### Fatal
None.

### Major

- **Asymmetric and statistically marginal cross-model generalization (Section 5.3, Table 1).** The central data-auditing claim—that LRE features generalize across models—holds in one direction but is marginal in the other. When the regression is trained on GPT-J and evaluated on OLMo for predicting object occurrences, LRE features score 0.49 ± 0.12 vs. a mean baseline of 0.41 ± 0.17. The confidence intervals overlap substantially, and the paper itself concedes "barely better than baseline." This is a real asymmetry: the claim of cross-model generalizability is convincingly supported only in the OLMo→GPT-J direction. The abstract phrase "low error even on inputs from a different model" overstates the evidence.

- **Optimistic framing of regression accuracy (Section 5, abstract).** "Within one order of magnitude" (10×) is a coarse success criterion over corpora spanning 5–6 orders of magnitude in frequency. The paper reports MAE in natural-log space of 2.1 for LRE features on object occurrences—corresponding to a geometric mean error of roughly e^2.1 ≈ 8×, near the boundary of the stated criterion. Table 2 shows errors reaching 346× (Arcturus/Boötes) and 207× (Prince Harry/Princess Diana), and the mean baseline actually outperforms LRE features on subject-object co-occurrence when evaluated on OLMo (0.67 vs. 0.68). The abstract's "low error" language will mislead readers about practical precision.

### Minor

- **Subject-choice sensitivity in regression predictions.** Table 2 reveals that the same object (Princess Diana) is predicted as 5,826 occurrences when the subject is Prince William but only 131 when the subject is Prince Harry, against a ground truth of 27,094. This factor-of-44 within-relation spread for the same object reveals that the regression is highly sensitive to which subject is chosen to represent a relation—a limitation that is flagged in the table caption but not discussed substantively. For a practical data-auditing tool, this sensitivity to query subject is a meaningful constraint that should be analyzed and bounded.

- **Frequency thresholds presented without uncertainty.** The thresholds are reported as integers (1,998; 4,447; 1,097) that imply false precision. These are defined as the co-occurrence count above which *mean* causality exceeds 0.9, but Figure 2 shows non-trivial scatter around these lines. Bootstrapped confidence intervals or a sensitivity analysis to the threshold definition (e.g., median rather than mean) would be needed to justify precise threshold claims.

- **Scale comparison is confounded and has only three data points.** The claim that scale reduces required exposure (OLMo-7B requires ~2k, OLMo-1B requires ~4k, tentatively attributed to model capacity) is interesting but relies on a comparison across only three models that differ in dataset (Dolma vs. Pile), tokenizer, counting methodology (Batch Search vs. WIMBD), and architecture. The paper appropriately notes "we cannot draw conclusions from only three models," but even this caveat may be insufficient—the confounders are large enough that the scale hypothesis cannot be meaningfully evaluated from existing data.

- **Direction-of-causation confound not discussed.** The paper establishes a strong correlation between frequency and linearity but does not discuss a plausible alternative: that structurally simple or world-regularized relations (e.g., country-capital) are both naturally frequent in internet text *and* intrinsically easier to encode linearly, making frequency a proxy for learnability rather than a cause of linearity. This confound is mentioned nowhere in the paper and should be explicitly flagged in the discussion.

- **No dedicated limitations section.** The paper's scope is narrow (3 models, 1B–7B parameters, 24 factual relations, English only), but none of this is collected in a limitations section. Overgeneralization in the abstract ("language models") is not balanced by explicit scope acknowledgment.

### Tiny

- **Layer choice for LRE fitting not stated in the main text.** Since LRE quality is known to vary by layer, and layer selection is deferred to Appendix C, readers cannot assess the primary results without cross-referencing the appendix. A single sentence in Section 3.1 noting the selected layer would improve self-containedness.

- **PCA treatment for feature importance is underspecified.** Section 5.2 notes that faithfulness and faith probability are replaced by one PCA component explaining 89% of variance, but does not state how many components were retained, whether PCA was fit on train data only, or whether this choice was validated. Reproducibility is impaired.

---

## Nice-to-Haves

- **Control for relation difficulty.** The core correlation might partly reflect that structurally simple relations (e.g., country-capital) are both inherently easy to encode linearly *and* frequently discussed. Matching relations by estimated learnability (e.g., synonymy with relational databases) or testing residual frequency effects after controlling for relation type would strengthen the frequency-causes-linearity interpretation.

- **Causal intervention.** Fine-tuning models on corpora with manipulated fact frequencies (e.g., via targeted data augmentation) would move the paper from correlational to causal, a natural and significant extension.

- **Layer-wise threshold analysis.** The threshold analysis aggregates across layers; a layer-specific version would provide mechanistic detail about *where* in the network the frequency effect manifests.

- **Regression error distribution.** Reporting not just "within-magnitude" accuracy and MAE but also the full distribution (e.g., percentiles, fraction of predictions within 2×, 5×, 100×) would give a more honest picture of prediction quality.

- **Expanded discussion of architecture generalization.** The paper would benefit from a brief analysis of *why* LRE features transfer across OLMo and GPT-J despite different tokenizers and datasets—e.g., whether the transferable signal is concentrated in causality scores rather than raw numerical feature values.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"The paper's 'surprising' framing is unwarranted given Jiang et al. (2024)."** The harsh critic argues that readers familiar with Jiang et al. (2024) will not find the frequency-linearity result surprising. However, Jiang et al. provide theoretical conditions in simplified settings; this paper's contribution is the *empirical characterization at realistic scale across training trajectories*, which is distinct. Removed as a style nitpick.

- **"WIMBD overestimation biases GPT-J thresholds."** The harsh critic flags that a WIMBD slope of 0.94 (systematic 6% overestimation) means GPT-J thresholds may be biased. The paper explicitly acknowledges this in the main text and Appendix D. A 6% overestimation is unlikely to change qualitative conclusions, and the concern is already addressed reasonably. Removed.

- **"Section 2.2 does not sufficiently distinguish from membership inference."** The paper explicitly states it focuses on corpus-level statistics rather than specific example reconstruction. Further elaboration would be a style improvement, not a scientific flaw. Removed as style nitpick.

- **"Unfair LM-only baseline in Figure 3."** The LM-only baseline uses only log probabilities and accuracy, which is intentionally weaker to demonstrate the *additional* value of LRE features. This asymmetry is transparent and beneficial for establishing the contribution. Removed.

- **"Only 24 relations used for regression — likely overfitting."** The random forest is trained with leave-one-out CV, meaning each of 24 held-out sets contains a genuinely unseen relation. Training-set performance is not reported, but given LOO-CV and the model's moderate performance on held-out sets, gross overfitting is unlikely. Removed as speculative without direct evidence.

---

## Novel Insights

The most underexplored finding in the paper—raised somewhat by the harsh critic and worth emphasizing to the authors—is in Section 4.3: for some relations, LRE causality forms *before* few-shot accuracy catches up. This suggests a possible two-stage learning process where the model first encodes a relation's structure linearly (implicitly, through frequency exposure) before it can deploy that structure reliably in prompted inference. This would imply that representational geometry and behavioral competence are at least partially dissociable, and that LRE measurement could serve as an early diagnostic of latent relational knowledge that has not yet surfaced in prompting evaluations. The paper notes this observation but does not develop it; it is arguably as interesting as the core threshold finding and warrants dedicated analysis in a revision.

---

## Suggestions

1. **Revise the abstract and conclusion to accurately characterize the regression results.** Replace "low error" with language like "approximate frequency ranges" and clarify that the claim of cross-model generalization is directionally asymmetric (stronger OLMo→GPT-J than the reverse for object occurrences).

2. **Add confidence intervals to the frequency thresholds** (e.g., via bootstrapping the threshold-definition procedure) and discuss what fraction of relations above/below the threshold fail/succeed, to give a more accurate picture of the threshold's sharpness.

3. **Analyze and discuss the subject-choice sensitivity in regression predictions.** The Prince William/Harry case (same object, 44× prediction spread) should be the basis of a principled discussion of how to select representative subjects for a relation when applying the method to closed-data models.

4. **Add a dedicated limitations section** enumerating the scope constraints (model size range, relation type, language, LRE class) and explicitly state that results may not generalize to instruction-tuned models, non-factual relations, or models outside the 1–7B parameter range.

5. **Add a paragraph in the discussion explicitly naming and addressing the correlation-vs.-causation confound** (i.e., that intrinsic relation learnability may co-determine both representational geometry and corpus frequency), even if the paper cannot resolve it empirically.

6. **Expand the Section 4.3 finding into a more structured analysis.** Provide a table or figure showing, for each relation, the training step at which causality first exceeds a threshold vs. the step at which 5-shot accuracy exceeds a comparable threshold—making the "causality leads accuracy" finding quantitative and falsifiable.

---

**Evaluation summary:**

- *Novelty:* Moderate-to-high. The core finding—connecting pretraining co-occurrence to LRE quality in modern transformers across training dynamics—is specific and fills a genuine gap. The regression/auditing application is novel in concept, though preliminary in execution.
- *Technical soundness:* Moderate. The empirical methodology is careful (Batch Search validation, LOO-CV, multiple checkpoints), but the regression evaluation overstates generalizability and the core threshold claims lack uncertainty quantification.
- *Empirical support:* Solid for the primary finding (r = 0.82, threshold emergence); mixed for the secondary contribution (regression prediction), where results in one direction are marginal.
- *Significance:* Meaningful for the interpretability and data auditing communities. The threshold finding is practically actionable; the regression/auditing contribution needs further development to reach its stated potential.
- *Clarity:* Generally clear, with the exception that the regression results are framed more optimistically than the numbers justify.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
