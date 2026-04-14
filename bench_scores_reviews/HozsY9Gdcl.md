## Summary

This paper introduces **Set-MI**, a method that improves language model membership inference (MI) by aggregating individual document MI scores over metadata-defined sets of documents under the assumption that all documents in a set are either collectively present in or absent from training data. The authors construct five diverse benchmarks (Wikipedia, Arxiv, Languages, License, Instructions) and show that Set-MI improves four Individual-MI baselines by 0.14 AUROC on average. Additional analyses examine the effects of model size, data deduplication, document length, set size, and robustness to violations of the set assumption.

---

## Strengths

- **Consistent, large empirical gains on a realistic problem:** The AUROC improvements in Table 2 are often substantial (e.g., Loss Attack on Arxiv: 0.576 → 0.938; LiRA on Wikipedia: 0.581 → 0.859) and span multiple domains and base methods, demonstrating that signal aggregation across semantically coherent sets is a practically effective lever that prior work missed entirely.

- **Five new diverse MI benchmarks covering distinct auditing use cases:** The paper constructs the first set-level MI benchmarks, covering temporal grouping (Wikipedia, Arxiv), linguistic inclusion (Languages), license-based inclusion (License), and instruction-tuning corpora (Instructions). This fills a clear gap, as prior MI benchmarks for LMs are narrow in domain.

- **Insightful ablations on model size and deduplication:** The finding that Set-MI gains *disproportionately* more from larger models and *loses disproportionately more* from deduplication compared to Individual-MI (Figure 3) is mechanistically interesting and informs when practitioners should expect set-level aggregation to help most. This is not an obvious result.

- **Controlled robustness analysis with realistic provenance verification:** Section 6 goes beyond metadata-inferred labels by verifying membership via 13-gram overlap with the Pile, providing a cleaner ground truth, and then systematically simulating noise. The finding that even FULL averaging significantly outperforms Individual-MI under up to 50% noise in both sets is an actionable result.

- **Method is fully orthogonal and modular:** Because Set-MI is a post-hoc aggregation layer, it is compatible with any future Individual-MI improvement, and the observed 0.824 Pearson correlation (p=0.0002) between Individual-MI and Set-MI performance implies a reliable multiplicative benefit as base methods improve.

---

## Weaknesses

### Fatal
None.

### Major

- **Language and Instructions benchmarks may largely detect distributional shift rather than document-level membership.** Bloom's training languages are publicly documented, and Tulu's instruction datasets are listed in its paper. Near-perfect AUROC (Languages: LiRA 0.908→1.000; Instructions: Min-K% PROB at 1.000 Individual-MI) does not necessarily validate MI capability — it may simply confirm that a model trained on English Wikipedia performs differently on languages it was never trained on, which is a far easier signal than true membership inference. This distinction is important: if some benchmarks are measuring source/domain inclusion rather than document membership, the 0.14 average AUROC gain is inflated and the framing of the paper needs adjustment. The paper does not analyze or even acknowledge this confound.

- **Table 1 statistics appear inconsistent with Section 4 text.** Table 1 reports Wikipedia: 1,000 sets / 100,000 documents and Arxiv: 1,000 sets / 100,000 documents. The benchmark construction text says "We subsample 100 sets with 100 documents per set," which gives 10,000 documents. The Language entry in Table 1 says 200 sets / 20,000 documents, yet the text says 20 languages × 10 sub-splits = 200 sets × 100 documents = 20,000 documents — this checks out. The Wikipedia/Arxiv discrepancy (100 sets described vs. 1,000 sets in Table 1) is unresolved and directly affects reproducibility. If the experiments were run on 100 sets, the table is wrong; if on 1,000, the text is wrong.

- **No evaluation on models with unknown training data, despite the core motivation.** The paper's introduction and Figure 1 motivate Set-MI as a tool for inspecting black-box LMs whose training data is unknown. Yet every target LM used (Pythia, GPT-Neo, BLOOM, SILO, Tulu) has fully documented training data used only to construct labels. The method is never demonstrated on its intended use case — recovering an unknown data cutoff or detecting undisclosed composition in a model. At minimum, one end-to-end case study (e.g., hiding the cutoff and asking whether Set-MI can recover it) would substantially validate the practical framing.

### Minor

- **zlib entropy degradation on Instructions (0.458 → 0.429) is unexplained.** Table 2 footnote says "outperforms in most settings" but does not analyze this failure mode. Given that zlib starts below random (0.458), the aggregation of a below-chance scorer is predictably harmful — which is consistent with the authors' own discussion of Individual-MI AUROC < 0.5 leading to worse Set-MI. The paper should make explicit that Set-MI requires a base scorer better than random to be beneficial, and give practitioners guidance on how to detect this condition without ground truth.

- **Single-domain ablations limit generalizability of findings.** Model size effects (Section 5.2), document length effects (Section 5.4), and set size effects (Section 5.5) are all measured only on Wikipedia. The deduplication analysis (Section 5.3) uses only Loss Attack. Since the benchmarks span five quite different domains, findings on one may not transfer. These sections should be clearly scoped as "Wikipedia case study" rather than general claims.

- **Document-level AUROC slightly misrepresents set-level performance when set sizes vary.** By assigning the set score to every document, AUROC computed at the document level implicitly weights large sets more than small ones. For the Wikipedia/Arxiv benchmarks (100 documents per set), this is fine. For the Language/License/Instructions benchmarks (equal 100-document sub-splits), it is also fine. But in aggregate comparisons or ablations with varying set sizes (Figure 4 right), document-level AUROC and set-level AUROC diverge. Reporting both, or being explicit that equal-sized sets make them equivalent, would avoid ambiguity.

- **"Significantly outperforms" in Table 2 caption has no statistical support.** No confidence intervals, bootstrap resampling, or hypothesis tests are reported. Given that per-cell numbers represent averages over 2–3 models with no per-run variance shown, the word "significantly" is informal. Given ICLR standards, this should be toned down or backed with error bars.

### Tiny

- Section 3's claim that downstream applications are "not affected" by deviations from the set assumption is slightly too strong. "Still directionally informative under moderate noise" is more accurate.
- The robustness section's MAX/MIN threshold of 30% is not validated. A brief sensitivity check would be simple to include.

---

## Nice-to-Haves

- **End-to-end case study:** Train a model, hide its data cutoff, and use Set-MI to recover it. This is the flagship application in Figure 1 but is never demonstrated empirically. Even a simple version would be persuasive.
- **Weighted aggregation schemes:** Simple averaging is reasonable, but confidence-weighted or variance-weighted aggregation might reduce sensitivity to outliers within sets without requiring knowledge of noise structure.
- **Evaluation on OLMo or a model with documented filtering pipelines:** OLMo's training data (DOLMA) has well-documented quality filtering and deduplication, offering a controlled setting to test real (not simulated) violations of the set assumption — a more realistic noise model than uniform random swaps.
- **Score distribution visualizations:** Kernel density plots of Individual-MI scores vs. Set-MI scores for members and non-members would intuitively show whether improvements come from mean shifts, variance reduction, or both.
- **Theoretical variance-reduction sketch:** A brief argument showing that averaging N independent, identically distributed noisy scorers reduces score variance by 1/N and improves expected AUROC under mild distributional assumptions would elevate the method section from heuristic to principled.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Black-box access model inconsistency (token logprobs vs. document loss)"** (Harsh Critic §2.1): The reviewer argues that methods requiring token-level probabilities contradict a "document loss only" assumption. In practice, document loss *is* computed from token-level cross-entropy, and token-level log-probabilities are the standard output of next-token prediction APIs. This is not an inconsistency.
- **"Method is just a heuristic mean with no theoretical treatment"** (Harsh Critic §3 major concern 1): While a theoretical treatment would be a nice-to-have, demanding formal analysis for an empirical aggregation paper imposes a standard not typical in this community for systems/empirical work.
- **"Set-level AUROC should be the primary metric"** (Harsh Critic §3 major concern 3): When all sets have equal size (100 documents per set, as in this paper's design), document-level AUROC computed on set-assigned scores is equivalent to set-level AUROC. The concern has merit in principle but is largely a non-issue given the benchmark design.
- **"Fairness of LiRA baseline without reference model details in main text"** (Harsh Critic §2.2): The paper appropriately refers to Appendix A for implementation details. Delegating baseline specifics to the appendix is standard practice.
- **Demand for confidence intervals / multiple-run statistics** (Harsh Critic §5.1, Spark Finder §1): Single-run evaluation with AUROC is the norm for large-scale MI benchmarking. Requesting per-run variance for 5 domains × 4 methods × multiple models goes beyond community standards. The absence of formal tests is worth noting (and noted above as a minor point), but is not a major weakness.
- **Correlation between Individual-MI and Set-MI on a small sample** (Harsh Critic §5.1): The correlation p=0.0002 is reported in the paper itself and is statistically sound for the claimed claim.
- **"Overreaching in Ethical Considerations / missing misuse discussion"** (Harsh Critic §Ethics): A fuller ethics discussion would be appreciated but is not a scientific weakness and is not unusual to be brief at ICLR.

---

## Novel Insights

The spark finder report raises one observation worth amplifying: the near-perfect Individual-MI AUROC on the Languages and Instructions benchmarks (0.908, 1.000) may indicate these tasks are not testing membership inference in the classical sense but rather *source/domain detection* — a much easier signal than document-level memorization. If this is correct, it would mean the paper's benchmark suite is heterogeneous in what it measures, and the headline 0.14 AUROC gain combines genuine MI improvements (e.g., Wikipedia, Arxiv) with what are effectively source-detection tasks (Languages, Instructions). Disentangling these two regimes — and identifying where set-level aggregation adds value beyond simply detecting whether a language/domain is in scope — would be a genuinely novel contribution to understanding the limits of LM MI evaluation.

---

## Suggestions

1. **Reconcile Table 1 and Section 4.** Verify whether Wikipedia/Arxiv benchmarks use 100 sets or 1,000 sets, and ensure the text, table, and appendix are consistent. This is a prerequisite for reproducibility.

2. **Analyze the Language and Instructions benchmarks separately**, explicitly asking whether results reflect document-level memorization or coarser source inclusion. One diagnostic: if Individual-MI with a model trained on *all* languages achieves near-random AUROC on the Language benchmark, then high AUROC with BLOOM is domain detection. This would change the framing of those results.

3. **Provide a single real-world demonstration of the core use case**: use Set-MI on a model with a withheld data cutoff (e.g., a Pythia model evaluated on a date range that is blinded to the evaluator) to show the method can actually recover the cutoff.

4. **Add explicit practitioner guidance** for when Set-MI will hurt rather than help: the current data shows it degrades when base Individual-MI AUROC is below 0.5. Provide a simple heuristic (e.g., check that Individual-MI average score on held-out random docs is above the threshold) so practitioners can apply this before committing to aggregation.

5. **Validate the 30% trimming threshold** for MAX/MIN aggregation with a brief sensitivity analysis, or justify it theoretically.

---

**Overall assessment:**

Set-MI is a practically motivated, clearly presented, and empirically well-supported contribution. The core idea — leveraging metadata-based set coherence to aggregate noisy individual membership signals — is conceptually simple but demonstrably effective, and the benchmark construction covers genuinely important auditing use cases. The paper is held back primarily by (a) a genuine inconsistency in benchmark statistics, (b) the lack of clarity about whether Language/Instructions benchmarks measure MI or domain detection, and (c) the gap between the stated use case (black-box, unknown training data) and the experimental setup (fully documented training data). These are fixable issues; addressing them would make this a clear contribution to the ICLR community.

- **Novelty:** Moderate — the idea of set-level aggregation is simple, but its application to LM pretraining MI and the benchmark construction are new.
- **Technical soundness:** Adequate — the method is correct and reproducible, though the theoretical underpinning is absent.
- **Empirical support:** Good on the core claim, with acknowledged gaps in the ablation breadth and the Language/Instructions benchmark interpretation.
- **Significance:** Moderately high — membership inference for LLMs is an active, important problem, and a consistently effective and modular improvement is practically valuable.
- **Clarity:** Good, with the exception of the Table 1 inconsistency and the blurred document-vs-set inference framing.