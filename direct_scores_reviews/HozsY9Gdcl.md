## Summary
This paper introduces **Set-MI**, a method that improves membership inference (MI) for language models by exploiting a *set assumption*: documents sharing metadata (e.g., publication date, license, language) are either collectively present in or collectively absent from a model's training data. Set-MI aggregates individual document-level MI scores within each set and broadcasts the aggregated score back to each member. The authors construct five diverse benchmarks (Wikipedia, Arxiv, Languages, License, Instructions) and demonstrate an average AUROC gain of 0.14 over four Individual-MI baselines. Additional analyses study the effect of model size, deduplication, document length, set size, and noise robustness.

---

## Strengths

- **Operationalizing a principled insight about pretraining data curation.** The observation that LM pretraining corpora are curated by *inclusion criteria* (date cutoffs, license categories, dataset sources) rather than per-document decisions is well-motivated and directly reflected in the set assumption. This is not a trivially recycled idea—it is a specific and accurate structural observation about how large pretraining datasets are actually assembled (e.g., DOLMA's Reddit cutoff), and it is used to design a concrete method.

- **Diverse benchmark suite filling a real gap.** The five benchmarks span temporal, linguistic, licensing, and instruction-tuning dimensions. Constructing MI benchmarks for LMs with known ground truth is non-trivial, and a multi-domain suite with varied notions of "set" is a concrete contribution the community can build on, provided the statistics inconsistencies noted below are corrected.

- **Meaningful robustness analysis.** Section 6's controlled noise injection study—comparing FULL/MAX/MIN aggregation under member-set noise, non-member-set noise, and both—provides genuine practical guidance. The finding that all three aggregators substantially outperform Individual-MI even under high noise ratios (up to 50% flipped labels) is informative, and the qualitative recommendation (MAX when member sets are noisy, MIN when non-member sets are noisy) is actionable.

- **Scaling and deduplication findings.** The systematic study linking larger model size to larger Set-MI gains (and deduplication to reduced gains) connects to prior memorization literature and adds new set-level evidence that directly informs practical deployment choices.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **Benchmark statistics inconsistency between Table 1 and the main text.** Table 1 reports Wikipedia: 1,000 sets / 100,000 docs and Arxiv: 1,000 sets / 100,000 docs. However, Section 4 explicitly states "we subsample 100 sets with 100 documents per set," yielding 10,000 docs—a 10× discrepancy. For License, Table 1 reports 190 sets / 19,000 docs, but Section 4 text says "resulting in 130 sets with 100 documents per set" (13,000 docs). These are not typographic issues: the actual experimental scale determines statistical reliability and reproducibility. If experiments truly used 100 sets, the main claims rest on much smaller samples than implied by Table 1. This must be clarified and corrected.

- **Document-level AUROC is not the right primary metric for a set-based method.** Set-MI assigns an identical score to every document in a set, then reports AUROC at the document level. Because all documents within a set share the same predicted score and the same ground-truth label, they are statistically dependent—effectively contributing the same "vote" |s| times rather than once. This inflates the effective sample size and can make document-level AUROC appear stronger than the method's actual discriminative power over distinct membership decisions. A set-level AUROC (one observation per set) should be the primary reported metric, with document-level AUROC as a secondary figure for compatibility with prior work.

- **Language and Instructions benchmarks may reflect domain separability rather than membership inference.** LiRA + Set-MI achieves **1.000 AUROC** on Languages, and Min-K% Prob achieves **1.000** on Instructions even at the Individual-MI level. These near-perfect scores raise the concern that the model is exploiting distributional differences between languages BLOOM was or was not trained on (e.g., perplexity differences due to absent language coverage) rather than subtle membership signals in the MI sense. Similarly, instruction-tuning datasets may have stylistic markers that separate them from non-member corpora without true memorization. The paper should analyze these benchmarks' difficulty more carefully and clarify whether the 0.14 average improvement is driven by genuinely hard domains or partly by inflated performance on trivially separable ones.

- **Missing pseudo-set control.** The core claim is that the *set assumption*—shared membership—is what drives improvement, not merely variance reduction from averaging more noisy signals. Without a control experiment where documents are randomly assigned to pseudo-sets (destroying the set assumption while keeping set size and averaging constant), it is impossible to separate these two explanations. This is a fundamental validation gap for the paper's stated contribution.

### Minor

- **The negative result (zlib on Instructions: 0.458 → 0.429) is noted but not analyzed.** This is the only case where Set-MI hurts performance, and understanding it is important for practitioners. The paper attributes it qualitatively to poor base signal, but does not investigate whether it is the set assumption being violated, averaging of systematic biases, or something specific to zlib's behavior on instruction-formatted text.

- **Robustness study uses only Loss Attack and one domain (Wikipedia).** Section 6's noise experiment is informative but uses a single base method. Since the main paper emphasizes LiRA and Min-K% Prob as the strongest base methods, showing robustness properties for at least one of those would substantially strengthen the conclusions.

- **Ground-truth labeling method is inconsistent across the paper.** Section 5 uses date cutoffs relative to Pile collection dates as ground truth for Wikipedia and Arxiv. Section 6 explicitly uses 13-gram overlap with the Pile as a "correct" ground truth. It is unclear whether main experiments (Table 2) use the date heuristic or n-gram verification, and to what degree date-based labels introduce false positives/negatives (e.g., documents that existed before the cutoff but were not actually ingested due to filtering).

- **The failure mode where Set-MI underperforms Individual-MI when base methods are below chance** is identified (Section 5.2) but no mitigation is proposed. A practical diagnostic or guard (e.g., check Individual-MI AUROC on a small calibration set before applying Set-MI) would make the method more deployable.

### Tiny

- The zlib formulation uses the ratio of *LM loss* to *zlib entropy*, which is standard, but the notation is slightly inconsistent between the averaging notation for Loss Attack (token probabilities) and the zlib formula. A clarifying note distinguishing log-probabilities from raw probabilities would aid reproducibility.

- Figure 4 (Left) shows performance saturating around 256 tokens yet 1,024 is used in main experiments. A brief justification for this choice would be useful.

---

## Nice-to-Haves

- **Evaluation on a model with a verifiable but unpublished training cutoff** (e.g., using Llama 2's known cutoff date for Arxiv) would demonstrate Set-MI's utility in a setting closer to real auditing use cases, where the training set is not fully public.
- **Set-level calibration or precision-recall analysis** in addition to AUROC would help characterize false accusation rates in applications like contamination detection.
- **Testing non-trivial aggregators** (weighted average by document confidence, robust estimators like trimmed mean) as an optional extension to complement the MAX/MIN/FULL comparison in Section 6.
- **Pseudo-metadata clustering** (e.g., by topic or style) as a strategy for scenarios where explicit metadata is unavailable—could substantially broaden applicability.
- **Expanded deduplication analysis**: the current study uses only Loss Attack on Wikipedia. Extending to LiRA or Min-K% Prob on Arxiv would better characterize when deduplication breaks Set-MI's advantage.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Critic: "Loss Attack averages raw probabilities, which is unusual."** LM(t_i) in this literature is standardly interpreted as log-probability (negative loss), and averaging log-probabilities to form a document score is the conventional Loss Attack formulation. The notation is informal shorthand, not an error.

- **Critic: "LiRA formula is not the standard formulation."** The paper presents a simplified notation for the ratio-based score. The actual LiRA implementation is deferred to Appendix A, which is a reasonable choice for a methods overview; the formula presented is a correct conceptual summary.

- **Critic: "Disjoint-set assumption is restrictive; should allow multiple overlapping grouping axes."** Requiring the method to also handle non-disjoint multi-axis groupings is outside the paper's stated scope. The method is presented as applicable to any partition satisfying the assumption; extending to joint multiple-axis inference is future work, not a flaw.

- **Critic: "Comparison with Set-MI is unfair because it uses extra side information."** The paper is explicit that Set-MI exploits available metadata as an additional input. This is not hidden; comparing against Individual-MI (which does not use metadata) is intentional and makes the contribution clear: metadata enables this gain. The comparison favors the baseline (which uses less information) and thus proves a stronger point for the proposed method.

- **Critic: "The abstract overstates practical reliability."** The conclusion language ("brings up the limit of MI to a practically robust level") is mildly optimistic but not egregiously misleading given the results. The robustness analysis in Section 6 does provide noise-tolerance evidence, and the scope is appropriately constrained to settings where metadata is available.

- **Critic: Requests hypothesis tests and confidence intervals for all results.** Single-run AUROC evaluation is the prevailing norm in LM MI literature. Requiring confidence intervals across all 20+ table cells would be non-standard for this subfield, though the point about variance across random token segment choices is worth a brief mention.

- **Critic: "Ethical discussion is too brief."** The dual-use discussion is concise but covers the key points appropriate for a research paper. Expanding it to a full treatment is a style preference.

---

## Novel Insights

The most genuinely novel observation—surfaced primarily by the spark finder and partially by the harsh critic—is that the paper does not clearly separate two distinct sources of improvement: **(1) the semantic coherence of set-assumption-satisfying groups** (shared metadata implies shared membership, which implies correlated loss signals) versus **(2) generic variance reduction from averaging any set of documents**. The pseudo-set control experiment (randomly assigned groups of equivalent size) is conspicuously absent. If simple averaging over random groups produces similar AUROC gains, the contribution would reduce to "more data per decision point helps," which is less interesting than the set assumption. If the gains are substantially larger with metadata-defined sets than with random pseudo-sets, this would be strong evidence for the set assumption's specific value. The paper as written cannot distinguish between these two explanations, and this gap is the most important open question the work raises.

---

## Evaluation by Axis

**Originality:** Moderate-to-good. The set assumption idea is intuitive and has precedent in clinical MI work (Jagannatha et al.), but its application to web-scale LM pretraining data with natural metadata structures is a meaningful and specific contribution. The benchmark construction adds additional originality.

**Importance of research question:** High. Data transparency, copyright auditing, and contamination detection are pressing concerns for the ML community, and improving MI performance from near-random to substantially above chance is practically relevant.

**Claim support:** Mixed. The main empirical gains are real and well-replicated across models and domains. However, the inflated AUROC on trivially easy benchmarks (Languages, Instructions), the missing pseudo-set control, and the document-vs-set metric issue mean the evidence is less clean than it appears.

**Soundness of experiments:** Moderate. The experimental design is broad and includes good ablations. The benchmark statistics inconsistency and absent set-level metric are real methodological issues that need resolution before the empirical claims can be fully trusted.

**Clarity of writing:** Good. The paper is readable and the central idea is easy to follow. The main clarity issues are in precise notation and benchmark construction reproducibility details.

**Value to the research community:** Moderate-to-good. The benchmark suite, if properly documented and with statistics clarified, would be a useful resource. The main method, though simple, provides actionable guidance.

**Contextualization relative to prior work:** Adequate. The paper covers the main Individual-MI baselines and correctly situates its contribution as orthogonal and additive.

---

## Suggestions

1. **Fix benchmark statistics inconsistency**: Reconcile Table 1 with the text descriptions for Wikipedia, Arxiv, and License. If the experiments used 100 sets × 100 docs (10,000 total for Wikipedia/Arxiv), update Table 1 accordingly. If they used 1,000 sets × 100 docs, update the text.

2. **Report set-level AUROC as primary metric**: Compute AUROC where each set contributes one observation (its aggregated score vs. its set-level ground truth), and compare this to document-level AUROC to demonstrate the difference. Reframe Table 2 around set-level AUROC.

3. **Add pseudo-set control**: Run Set-MI with randomly constituted sets of the same sizes as the metadata-defined sets. Report AUROC relative to the metadata-defined Sets to show the set assumption's marginal contribution.

4. **Analyze Language and Instructions benchmarks for confounding**: Report per-language AUROC for BLOOM to check whether gains track with language coverage in BLOOM's training, and whether held-out languages have systematically higher per-token loss for reasons unrelated to MI. Acknowledge if these benchmarks primarily test "did the model see this language at all" rather than nuanced membership inference.

5. **Clarify ground truth construction**: Explicitly state in Section 5 for each benchmark whether ground truth uses date/metadata heuristics alone or is verified via n-gram overlap, and estimate label noise where applicable.

6. **Analyze the zlib/Instructions failure case**: Check whether the set assumption is violated for those instruction datasets (e.g., partial inclusion from Tulu mixes), or whether zlib scores are anti-correlated with membership for instruction-formatted text, and briefly discuss in the paper.

MY FINAL SCORE: <pineapple>5.8</pineapple>