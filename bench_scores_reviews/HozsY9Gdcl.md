## Summary

This paper introduces **Set-MI**, a method that aggregates individual Membership Inference (MI) scores across sets of documents that, by virtue of shared metadata (e.g., creation date, license, source dataset), are expected to be either entirely present or absent from a language model's training data. The authors construct five diverse benchmarks (Wikipedia, Arxiv, Languages, License, Instructions) and demonstrate that applying Set-MI on top of four existing Individual-MI methods yields a mean AUROC gain of 0.14, with additional ablations characterizing the effects of model size, deduplication, document length, set size, and aggregation strategy under simulated membership noise.

---

## Strengths

- **Principled and natural reframing of the MI problem.** Rather than inventing a new scoring function, the paper identifies a real structural property of training data curation (all-or-none inclusion by metadata category) and shows that exploiting it can markedly improve any loss-based MI method. The insight that data pipelines select by inclusion criteria is well-grounded with concrete examples (e.g., DOLMA containing Reddit data only through March 2023).

- **First set-based benchmark suite for LM-MI.** The five benchmarks span genuinely different structural types of set assumption—temporal (Wikipedia, Arxiv), categorical (Languages, License), and dataset-of-origin (Instructions). Constructing these is non-trivial and the resulting suite is a real contribution to the community beyond the method itself.

- **The deduplication finding is a novel and practically important result.** Figure 3 (right) shows that Set-MI's advantage is substantially larger on models trained on *duplicated* data than deduplicated data, and that this gap is much bigger for Set-MI than Individual-MI. This asymmetric sensitivity is a new and informative observation: deduplication disproportionately destroys the signal that set aggregation exploits, which has direct implications for auditing modern models.

- **Robustness section is proactive about the method's own vulnerability.** Section 6 explicitly attacks the set assumption by simulating noise. The comparison of MAX/MIN/FULL aggregation under different noise scenarios is practically useful. The fact that all three aggregation variants substantially outperform Individual-MI even at 50% noise ratio is a meaningful empirical guarantee.

- **Strong document-length ablation.** Figure 4 (left) cleanly shows that Set-MI's advantage over Individual-MI scales with the length of sampled tokens, with Set-MI exhibiting faster saturation—this characterizes a concrete design choice practitioners can make.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Benchmark statistics are inconsistent between Table 1 and the text.** This is a serious reproducibility concern. Table 1 reports Wikipedia and Arxiv as 1,000 sets / 100,000 documents, but the corresponding text sections each say "we subsample 100 sets with 100 documents per set" (= 10,000 docs). For Languages, the text says "resulting in 130 sets" while Table 1 reports 200 sets; for License, the text again says "resulting in 130 sets" while Table 1 reports 190. (Note: internally, Table 1 is self-consistent at 100 docs/set, so the table numbers are likely correct and the text contains copy-paste errors. The "130" in both Language and License sections appears to be copied from the Instructions section.) These discrepancies must be corrected, because reported AUROCs are averages over these datasets and the evaluation scale needs to be unambiguous.

- **Missing random-grouping control.** The paper never tests whether randomly composed groups of the same size also improve AUROC over Individual-MI. Without this baseline, it is impossible to determine whether gains come from the set assumption specifically (the paper's core claim) or simply from statistical variance reduction via averaging MI scores over any N documents. If random groups yield similar improvements, the contribution reduces to "averaging noise away," not "leveraging shared membership." This experiment is essential to validate the central hypothesis. Authors should run this on at least Wikipedia and Arxiv.

- **Perfect and near-perfect scores on Languages and Instructions are unexplained and potentially confounded.** Languages/LiRA achieves 1.000 AUROC with Set-MI, and Instructions/Min-K% Prob achieves 1.000 for *both* Individual-MI and Set-MI. These results may primarily reflect domain/style distribution shift between included and excluded language groups or instruction datasets—not genuine membership inference from model loss signals. If a model was simply never trained on, say, Swahili Wikipedia, any reasonable metric will detect that. The paper should diagnose whether these "easy" cases are MI successes or distribution-shift detections, since including them in the average AUROC gain inflates the headline number. Domain-restricted averages (e.g., over just Wikipedia and Arxiv) would give a more honest picture of the method's value in hard cases.

- **Ground-truth membership is a proxy, not a verified label.** For Wikipedia and Arxiv, the paper labels membership based on whether a document's creation date is before the Pile's data-collection cutoff. However, creation date ≠ inclusion in the Pile due to crawl incompleteness, filtering, formatting failures, and source-level truncations. The paper partially addresses this in Section 6 by using 13-gram overlap against the actual Pile as a "clean" version—but this validation is only used in the robustness section on Pythia-2.8B-dedup, not to validate the main Table 2 labels. The paper should report what fraction of "member" documents (by date proxy) actually have 13-gram overlap with the Pile, and discuss how large this gap is. If it is small, the concern is minor; if it is large, the Table 2 AUROCs may be measuring something other than MI.

### Minor

- **No variance estimates in Table 2.** Table 2 reports single-point AUROC values (some averaged over multiple models, but without standard errors). For a paper whose central claim is a +0.14 average improvement, some measure of variability—at minimum a standard deviation over the multiple models used for Wikipedia, Arxiv, and License—is needed. The zlib/Instructions result (0.458→0.429, a *drop*) is given one sentence without analysis. This deserves at least a diagnostic: does Set-MI hurt when the base Individual-MI score is below chance?

- **Deduplication and model-size ablations are narrow in scope.** The deduplication analysis uses only Loss Attack on Wikipedia; the model-size analysis uses only Wikipedia. Both are presented as general findings about Set-MI, but demonstrating these effects on a second domain (e.g., Arxiv) would substantially strengthen the generalizability claims.

- **The 30% threshold for MAX/MIN aggregation is not motivated.** Section 6 uses "top/bottom 30%" for MAX and MIN without explanation or sensitivity analysis. Since the paper recommends these variants for practical use, users need to know whether 10%, 20%, or 50% would work equally well. A brief sensitivity sweep is warranted.

### Tiny

- **Correlation of 0.824 between Individual-MI and Set-MI performance (Section 5.1) adds limited insight.** It confirms the intuitive observation that Set-MI inherits the quality of its base method. More useful would be identifying conditions under which Set-MI *fails to improve* Individual-MI, or a scatterplot of gain vs. base AUROC.

- **Recommendation to "select the best aggregation based on prior knowledge about the noise" (Section 6)** is circular: in practice, a user auditing a closed model will not know which class of noise dominates. Even a simple heuristic (e.g., "default to FULL unless you have specific evidence of one-sided noise") would be more actionable.

---

## Nice-to-Haves

- **Score distribution visualizations.** Overlaid histograms of member vs. non-member scores before and after aggregation would clarify whether Set-MI primarily increases mean separation, reduces variance, or reshapes the tails—all of which have different implications for threshold-based use.

- **Threshold-level analysis (precision/recall at fixed FPR).** AUROC improvements of 0.14 are encouraging, but applications like contamination detection or copyright auditing require high precision at low false-positive rates. A calibration plot or precision-recall curve at practically relevant operating points would help practitioners assess whether the gains are meaningful at deployment thresholds.

- **Discussion of soft/inferred sets.** The conclusion acknowledges that metadata availability is a limiting assumption but defers relaxation to future work. Even a brief sketch of whether clustering or semantic similarity could construct "soft sets" would help readers assess the method's scope.

- **Set-level AUROC alongside document-level AUROC.** Because Set-MI assigns identical scores to all documents within a set, the effective number of independent predictions is the number of sets, not documents. Reporting set-level AUROC would be conceptually cleaner and would complement the document-level figures.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Document-level vs. set-level inference confusion" (Harsh Critic, Section 1 and 2).** The paper is transparent that the method assigns a set-level score to all documents in the set for comparability with Individual-MI. This is clearly stated in the method section ("we assign the aggregated score from the set to every element within the set, so that the score is directly comparable with previous methods"). This is a design choice, not a conceptual error.

- **"Black-box assumption is inconsistent because token-level probabilities may not be available" (Harsh Critic, Section 2).** The paper explicitly focuses on the setting where "only the loss score of the target model is available," and all four Individual-MI methods it builds on use token-level log-probabilities, which are standardly exposed by the APIs and open models the paper targets. This is not a contradiction.

- **"Comparisons may be unfair because base methods were not tuned comparably" (Harsh Critic, Section 5.1).** The Set-MI aggregation is a wrapper applied identically to all base methods; there is no differential tuning.

- **"Why only four MI baselines?" (Harsh Critic, Section 2).** Loss, LiRA, Min-K% Prob, and zlib are the standard baselines in LM-MI literature. Demanding more without identifying specific methods that should have been included is scope creep.

- **Demanding theoretical variance-reduction analysis or formal proofs** (Harsh Critic). This is an empirical systems paper and formal analysis is not standard for this type of contribution in the field.

- **"Privacy / ethics section is too thin" (Harsh Critic, Ethical Considerations).** The paper's ethics section is brief but appropriate for a paper whose primary contribution is an auditing tool, not a privacy attack. The concern about group privacy is noted as potentially valid, but not a scientific flaw.

- **"Impracticality for truly closed-source models" (Spark Finder).** All experiments require AUROC evaluation, which requires ground-truth membership labels. Evaluating on a model with completely unknown training data is not possible in a rigorous setting. The paper's choice of models with known (but publicly undisclosed during evaluation) training data is the correct experimental design.

- **"Larger dataset sizes needed"** — The benchmarks cover up to 100,000 documents with multiple models. Dataset size is not a meaningful weakness here.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the **asymmetric interaction between deduplication and aggregation-based MI**: deduplication disproportionately suppresses the signal that set-level averaging exploits, more so than it suppresses individual-document signals. This implies that MI via averaging is sensitive to a specific form of data preprocessing that is increasingly common in modern pipelines, and suggests that the gap between theoretical recoverability and practical MI may widen as deduplication becomes the norm. A complementary insight from the reviews is that *trivially easy* benchmarks (Languages at 1.000, Instructions/Min-K% at 1.000) may be measuring near-domain detection rather than memorization—and the field would benefit from a sharper distinction between these two phenomena in benchmark design.

---

## Suggestions

1. **Run a random-grouping control immediately.** Form groups of the same sizes as the real sets but with randomly assigned documents, and compute AUROC under the same averaging scheme. Report this as a baseline in Table 2. This single experiment either validates the paper's central claim or substantially changes what is being claimed.

2. **Reconcile all numbers between Table 1 and the body text.** The "100 sets vs. 1,000 sets" discrepancy for Wikipedia/Arxiv and the "130 sets" copy-paste error for Languages and License must be corrected with explicit footnotes explaining what was actually run.

3. **Add a diagnostic for the 1.000-AUROC results.** For Languages/LiRA and Instructions/Min-K%, provide perplexity-gap analysis or cross-entropy gap between included vs. excluded categories at the *zero-shot* level to determine whether the separation is membership-driven or purely distributional.

4. **Report 13-gram overlap validation fraction for the main Wikipedia/Arxiv benchmarks,** not just for Section 6's "clean" subset. Even one sentence quantifying label quality (e.g., "X% of date-proxied member documents have ≥1 13-gram overlap with the Pile") would substantially increase confidence in the main results.

5. **Add a sensitivity table for MAX/MIN threshold** (10%, 20%, 30%, 50%) in Section 6 or the Appendix, to allow users to calibrate their choice without guessing.

---

**Overall assessment:** The paper has a simple but well-motivated idea, a valuable benchmark contribution, and empirically solid main results. The headline finding (average +0.14 AUROC) is encouraging, but the absence of a random-grouping control, combined with the presence of trivially-solved benchmarks in the average, means the true effect size in genuinely hard cases is unclear. The benchmark statistic inconsistencies are a fixable but significant reproducibility issue. The paper is in solid shape for ICLR but requires these specific corrections to make the empirical claims fully defensible. On the axes of evaluation: **novelty** is moderate (the insight is clear but the method is simple); **technical soundness** is adequate with the noted gaps; **empirical support** is broad but has the identified holes; **significance** is good for the auditing/governance community; **clarity** is mostly strong, with the exception of the stat inconsistencies and the 1.000-AUROC cases.