## Summary
This paper introduces **Set-MI**, a wrapper method that improves membership inference (MI) for language models by aggregating per-document MI scores over metadata-defined groups ("sets") under the assumption that all documents in a set are either entirely in or entirely out of the training corpus. The authors construct five new MI benchmarks spanning Wikipedia, Arxiv, language identity, data license, and instruction-tuning datasets, and demonstrate an average AUROC gain of 0.14 over four Individual-MI baselines. Additional ablations analyze the effects of model size, training-data deduplication, document length, and set size, plus a controlled robustness analysis under simulated membership noise.

---

## Strengths

- **Broadly applicable wrapper design.** Set-MI is formulated as a zero-overhead wrapper over any existing Individual-MI scoring function, verified empirically across four qualitatively different base methods (Loss Attack, LiRA, Min-K%, zlib entropy). The positive correlation between Individual-MI and Set-MI performance (r = 0.824, p = 0.0002) also provides an actionable prescription: improving base MI methods will compound into larger Set-MI gains.

- **Novel benchmark suite.** To the authors' knowledge, these five benchmarks are the first set-structured MI benchmarks for LMs, and collectively the most domain-diverse. Covering temporal cutoffs (Wikipedia/Arxiv), language identity, license category, and fine-tuning datasets addresses a real gap since prior work typically evaluates on a single domain.

- **Concrete motivating examples grounded in real data-pipeline practice.** The paper provides specific, verifiable examples (DOLMA's March 2023 Reddit cutoff, SILO license categories, Tulu instruction-dataset composition) rather than toy constructions, making the set assumption credible for practitioners.

- **Insightful deduplication finding.** The result that deduplication widens the gap between Duplicated and Deduped models *more* for Set-MI than for Individual-MI is a genuinely novel empirical observation about memorization dynamics that is not obvious a priori and has implications for data-curation research.

- **Scaling analysis.** The finding that Set-MI benefits disproportionately more from larger model sizes, while Individual-MI improvement is modest, adds concrete empirical content to the general understanding that memorization scales with model capacity.

---

## Weaknesses

### Fatal
None. The core contribution is sound and the empirical gains are real.

### Major

- **Benchmark statistics are internally inconsistent, harming reproducibility.** Table 1 states Wikipedia has 1,000 sets / 100,000 documents and Arxiv has 1,000 sets / 100,000 documents, but the construction text says "We subsample 100 sets with 100 documents per set" for each — giving 10,000 documents, not 100,000, and 100 sets, not 1,000. For the Language benchmark, the text says "resulting in 130 sets" but 20 languages × 10 subsets = 200 sets, matching Table 1 (200 sets / 20,000 docs). For License, the text again says "resulting in 130 sets" but 19 source datasets × 10 subsets = 190 sets, matching Table 1 (190 / 19,000). These appear to be systematic copy-paste errors in the text, but until resolved a reader cannot reproduce the benchmarks or interpret the scale of the evaluation. This must be corrected for the paper to be replicable.

- **No uncertainty quantification.** The entire empirical contribution rests on AUROC point estimates, but no confidence intervals, bootstrap intervals, standard deviations, or significance tests are reported anywhere. This matters because many improvements are modest in absolute terms (e.g., Loss Attack on Wikipedia: 0.524 → 0.575; zlib on License: 0.647 → 0.674), and each estimate is computed from a single random 1,024-token span per document. Without error bars it is impossible to assess whether these gains represent reliable signal or sampling noise. This is a core requirement for an empirical claim paper, not a methodological nicety.

- **Several benchmarks plausibly measure domain/distribution shift rather than true membership.** (a) *Language*: Bloom's per-language loss differences could reflect tokenizer coverage and overall language competence rather than document-level membership, because the model was trained with language selection as a first-order design choice. (b) *License*: different license categories often correspond to qualitatively different dataset topics and writing styles, so a model may separate them without memorizing any specific documents. (c) *Instructions*: the target model (Tulu-v1) is fine-tuned rather than pretrained on the instruction datasets, and the set label is the dataset identity itself — the model may recognize the format of an unseen ShareGPT conversation without that specific conversation being in training. These confounds do not invalidate the benchmarks outright, but the paper provides no control (e.g., removing the distributional cue while keeping the membership cue) to disentangle the two effects. The scientific claim — "Set-MI leverages membership signals" — requires at least acknowledging and, ideally, partially ruling out this alternative explanation.

- **Robustness analysis (Section 6) is too narrow to support general claims.** The robustness experiments use a single base method (Loss Attack), a single domain (Wikipedia), and a single model (Pythia 2.8B-dedup) under synthetic noise generated by random replacement of members/non-members. This does not substantiate the claim that Set-MI is robust "under practical settings" across the paper's five benchmarks and four base methods. Different base methods have different tail shapes and calibration, and real-world violations of the set assumption (version updates, partial crawls, deduplication artifacts) do not follow a uniform random replacement model.

### Minor

- **The zlib + Set-MI failure on Instructions (0.458 → 0.429, below random) is not adequately analyzed.** The paper lists this number in Table 2 and notes the general caveat that poor Individual-MI can hurt Set-MI, but provides no domain-specific explanation for why zlib specifically fails here while the other three methods improve. This is a direct counterexample to the claim that "Set-MI significantly improves Individual-MI on most settings," and understanding it would strengthen the paper.

- **Date-based sets create near-perfect correlation between the grouping variable and the membership label.** For Wikipedia and Arxiv, documents are labeled as members iff their creation date precedes the collection cutoff, and sets are defined by creation date. Set-MI on these benchmarks is therefore largely testing whether the model encodes the temporal training boundary, not individual document memorization. The 13-gram overlap validation in Section 6 partially addresses this for the robustness experiment, but not for the main results in Table 2. The paper should discuss whether temporal-cutoff monotonicity is the dominant driver of gains on these two benchmarks.

- **The 30% threshold for MAX/MIN aggregation in Section 6 is unjustified.** No sensitivity analysis is provided and it is not clear whether this value was selected on the same data used for evaluation. A brief justification or sensitivity curve is needed.

- **No random-set control.** Aggregating over any batch of documents reduces estimator variance. The paper does not show that aggregating over randomly assembled (non-membership-correlated) groups fails to achieve similar gains. Such a control would confirm that the set assumption — and not mere variance reduction — is what drives performance.

### Tiny

- A single random 1,024-token span is drawn per document for all experiments. The sensitivity of AUROC estimates to this random draw is not reported. Even a brief note on the variance across multiple draws (or evidence that results are stable) would strengthen confidence in the reported numbers.

- The set-size ablation (Figure 4 right) keeps set count fixed while varying set size, but does not control for total tokens observed. A comparison at equal total token budget (e.g., 1 doc × 1,024 tokens vs. 4 docs × 256 tokens) would better isolate the benefit of the set assumption from the benefit of observing more tokens.

---

## Nice-to-Haves

- **Random-set baseline.** Run Set-MI by grouping documents into sets with no membership correlation (e.g., random date assignments) and compare AUROC. This would directly quantify how much of the gain is due to meaningful shared membership versus pure averaging noise reduction.

- **Evaluation on a model with genuinely unknown training data.** All target LMs have published training-corpus details used to define ground truth. Applying Set-MI to a semi-unknown model (e.g., using later-revealed information about GPT-2 or an early LLaMA checkpoint) would validate the practical use case more convincingly.

- **Score distribution visualization.** Plotting per-document score distributions for member vs. non-member sets before and after aggregation would make it visually clear whether aggregation genuinely separates the classes or uniformly shifts all scores.

- **More sophisticated aggregation alternatives.** Median, trimmed mean, or a simple confidence-weighted average are natural competitors to the mean that might be more robust to outliers. The paper explores MAX/MIN only in the noise robustness section; briefly comparing these alternatives in the main experiment would strengthen the design choice.

- **Automatic set discovery.** The current method requires a practitioner to know which metadata attribute defines membership-correlated sets. A brief discussion or preliminary experiment on clustering-based set discovery would broaden applicability.

- **LiRA reference-model sensitivity.** The paper notes that finding a good reference model is difficult in practice. A brief ablation varying the reference model quality would clarify whether LiRA-based Set-MI is stable or highly sensitive to this choice.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **"The method is too simple for ICLR"** (Harsh Critic): Simplicity is not a disqualifying weakness when the empirical contribution and benchmark construction are substantive. Many impactful systems papers succeed through principled evaluation of simple methods. Removed.

- **"Missing stronger or more recent LM-specific baselines"** (Harsh Critic): No specific missing method is named; this is a generic criticism that applies to any paper. The four methods chosen span the main families of black-box MI scoring (loss, LiRA, n-gram, compression). Removed.

- **"Loss Attack notation uses probability rather than log-probability"** (Harsh Critic): The paper consistently uses probability notation throughout Section 2.2 for all methods and Figure 2 illustrates the same. While log-probability is more common in practice, this is a presentation choice, not a technical error. Removed.

- **"Demanding theoretical proofs for when averaging improves AUROC"** (Harsh Critic): This paper is an empirical systems contribution. Requesting formal proofs of AUROC improvement under averaging is not a standard expectation for this type of work. Removed.

- **"Calibration/threshold discussion required"** (Harsh Critic): The paper evaluates with AUROC throughout, which is threshold-free. Requesting threshold calibration analysis is a nice-to-have at best and not a substantive weakness. Removed.

- **Strength: "the paper is well-written / the topic is important"** (Generic): These are not retained as named strengths as they apply to virtually any paper in the area.

---

## Novel Insights

The most insightful observation that emerges from synthesizing all three reviews is that Set-MI's gains on date-based benchmarks (Wikipedia, Arxiv) may reflect the model having internalized the **temporal training boundary** — a smooth monotonic signal — rather than per-document memorization. This interpretation, if correct, changes the scientific claim: the paper would partly be showing that LMs encode coarse temporal data-selection policies, which can be recovered by aggregating loss signals over time-cohort groups. This is itself a meaningful finding, but it is distinct from (and arguably stronger and more tractable than) the document-level membership inference framing. A clean disentanglement — for example, using 13-gram overlap labels for Wikipedia/Arxiv in the main experiments rather than only in Section 6 — would clarify whether gains come from temporal policy recovery, genuine per-document memorization, or both, and would substantially sharpen the paper's narrative.

---

## Suggestions

1. **Fix benchmark statistics:** Reconcile Table 1 with the construction text. For each domain, state clearly whether the reported numbers are the full collected benchmark or the experimental subsample, and ensure these are consistent throughout.

2. **Add uncertainty quantification:** Report AUROC estimates with 95% bootstrap confidence intervals (or at minimum standard deviations across random token draws). For large-scale benchmarks where this is expensive, report at least a small-scale variance study.

3. **Add a random-set control:** Include one experiment where Set-MI is applied over randomly composed groups (size-matched to real sets). This is a cheap but decisive experiment that verifies the mechanism.

4. **Use 13-gram overlap labels in the main Wikipedia/Arxiv experiments:** Since this cleaner labeling is already computed for Section 6, applying it to the main results table would partially address concerns about proxy ground truth and temporal confounding.

5. **Analyze the zlib + Instructions failure case explicitly:** Identify whether this failure is due to zlib Individual-MI being below 0.5 (and thus aggregating in the wrong direction), or whether something about the Instructions domain specifically causes Set-MI to degrade. A sentence or two in the main paper would suffice.

6. **Widen the robustness analysis to at least two base methods and two domains:** Even adding Min-K% on Arxiv alongside the existing Loss Attack on Wikipedia setup would substantially strengthen the generalizability of Section 6's claims.

7. **Explicitly separate the temporal-boundary effect from per-document memorization:** Use 13-gram overlap labels in the main Wikipedia/Arxiv experiments, and add a discussion of whether removing the exact temporal ordering of sets changes the AUROC, to verify whether the set assumption is doing the work rather than temporal monotonicity.