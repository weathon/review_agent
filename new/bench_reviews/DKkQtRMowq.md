Now I have a thorough understanding of the paper and can write the final consolidated review.

---

## Summary

DS² introduces a data selection pipeline for instruction tuning that corrects LLM-generated quality scores via a learned score transition matrix (estimated from k-NN consensus statistics without ground-truth labels) and promotes diversity through a long-tail scoring mechanism. The paper shows that selecting 3.3% of data (10k from 300k) using DS² outperforms full-data training on the OpenLLM leaderboard, and that at 1k samples, DS² can match or surpass the human-curated LIMA dataset under certain rating models.

## Strengths

- **Principled modeling of LLM rating errors via a score transition matrix.** Unlike prior LLM-based data selection methods (AlpaGasus, DEITA) that take raw LLM scores at face value, DS² explicitly models the error structure through a transition matrix estimated from k-NN consensus statistics (Definition 3.1, Eq. 1). The estimated matrices in Figure 3 provide concrete, model-specific evidence of differing error patterns — GPT-4o-mini shows a strong diagonal (stable ratings) while LLaMA and Mistral exhibit significant off-diagonal mass — validating the premise that score correction is needed.

- **3.3% of data outperforms full-dataset training across multiple settings.** Table 3 demonstrates that DS² with only 10k samples consistently outperforms the full 300k data pool across all three rating models on LLaMA-3.1-8B (e.g., 61.4 vs. 57.7 with GPT-4o-mini ratings; 60.2 vs. 57.7 with LLaMA ratings). This is a meaningful and well-supported claim.

- **Score curation improves performance across different algorithms, not just DS².** Table 5 shows that curation improves AlpaGasus (58.1→59.5) and DEITA (59.7→60.6) on LLaMA-3.1-8B with GPT ratings, indicating the transition matrix captures genuine correctable error structure rather than being an artifact of the selection algorithm.

- **Comprehensive baseline comparison and data scaling analysis.** The paper compares against 9 baselines spanning statistical metrics, LLM-based methods, gradient-based methods, and full-data training across 3 base models and 3 rating models. Figure 5 effectively demonstrates DS² maintains strong performance at very small sample sizes (2.5k) where most baselines degrade.

- **Curation reduces sensitivity to the choice of rating model.** Figure 7 (Right) shows the maximum performance gap across rating models drops from 1.60 to 0.70 with curation, a practical benefit for users choosing between open-source and commercial raters.

## Weaknesses

### Fatal
None.

### Major

- **TruthfulQA performance degrades severely under multiple conditions, and the paper does not acknowledge or discuss this.** In Table 4, DS² with LLaMA-generated scores at 1k samples achieves 4.4 on TruthfulQA vs. LIMA's 32.1 on LLaMA-3.1-8B — a near-complete collapse on a truthfulness benchmark. Even at 10k samples (Table 3), curation hurts TruthfulQA with LLaMA ratings (50.2→45.4) and GPT ratings (51.5→50.3). Table 5 shows curation hurts DEITA's TruthfulQA (50.1→45.5) and AlpaGasus's GSM (66.0→61.5). The paper's Section 5.2 claims "score curation can consistently improve the average performance," which is true on average but masks systematic regressions on safety-critical benchmarks. The absence of any discussion about when and why curation degrades specific capabilities is a significant gap, especially for a method targeting alignment.

- **The abstract's claim of "matching or surpassing LIMA" is selectively true and overclaimed.** The abstract states DS² "matches or surpasses human-aligned datasets such as LIMA with the same sample size (1k samples)," but Table 4 shows Ours(L) on LLaMA-3.1-8B averages 49.3 vs. LIMA's 50.2 — it does *not* match LIMA with LLaMA ratings. The claim holds for GPT and Mistral raters but fails for the LLaMA rater. Given that one of the paper's key selling points is making weaker open-source raters viable, the failure case under the weakest rater is precisely the scenario that matters most.

- **The core mechanism — estimated transition matrix T — lacks direct empirical validation.** The entire DS² pipeline rests on estimating T from k-NN statistics without ground-truth labels, extending Zhu et al. (2021) from categorical labels to ordinal scores. The paper provides no empirical validation that the estimated T is correct. A straightforward test would be: on a subset with human annotations, compare the estimated T against the observed error transition matrix. Without this, the heatmaps in Figure 3 are uninterpretable — we cannot tell whether the estimated T captures real error patterns or is an artifact of the LP solution under violated assumptions. The downstream performance improvements provide only indirect, circular evidence.

### Minor

- **No variance or significance tests are reported for any result.** Fine-tuning 7-8B parameter models has non-trivial run-to-run variance. The claimed improvements from curation are modest (0.4–1.2 average points across rating models in Table 3), and without error bars, it is impossible to assess whether these are meaningful or within noise. This is particularly concerning given that Table 5 shows curation hurting individual benchmarks by 4-5 points while improving averages by ~1 point.

- **The extension from categorical label noise to ordinal scores (0–5) lacks theoretical justification.** The framework from Zhu et al. (2021) was designed for categorical classification labels where the transition matrix represents class-conditional label flipping. Ordinal scores have inherent structure (being wrong by 1 is different from being wrong by 4), yet the method treats them as unordered categories. This does not invalidate the approach, but the paper should acknowledge the simplification and discuss its implications.

- **The confidence probability parameter (default 0.5) controls how aggressively scores are corrected but has no sensitivity analysis.** This is a critical hyperparameter — too aggressive correction could introduce more errors than it fixes. A brief sensitivity study would strengthen confidence in the method's robustness.

- **The k-NN clusterability assumption is defended verbally but not empirically.** The paper argues that scoring considers "broader quality metrics" and "consensus vectors average over violations" (Section 3.2), but does not test what fraction of 2-NN pairs actually share the same ground-truth score. A single empirical measurement would be far more informative than the verbal argument.

### Trivial
None.

## Nice-to-Haves

- Validation of the estimated transition matrix T against human annotations on a data subset, which would directly confirm the core mechanism works as theorized.
- Analysis of what score curation actually does to the data distribution — how many samples are re-scored, from which scores to which scores — to reveal whether T-based curation is doing something principled or just redistributing scores.
- Reporting results with standard deviations across at least 3 fine-tuning seeds for the main results.
- Qualitative examples of score corrections with human verification, making the mechanism tangible.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "The diversity score is simply k-NN density estimation and adds little beyond k-NN-10 baseline."** The paper does not overclaim novelty for this component — Section 4.2 presents it as a complementary mechanism. The fact that k-NN-10 is a baseline does not invalidate DS²'s combined quality+diversity approach, and k-NN-10 does not include score curation. This is not a substantive weakness.

- **Harsh critic: "Unfair comparison with LIMA because DS² selects from a 300k pool."** DS² is selecting from the same data that any automated method would use, while LIMA is a manually curated dataset — this is the standard comparison setup in the "less is more" literature. The different candidate pool sizes reflect the fundamental advantage being claimed (automated selection from large pools). The paper could acknowledge this structural difference more explicitly, but it is not an unfair comparison.

- **Harsh critic: "Apples-to-apples comparison with AlpaGasus is vague, details relegated to appendix."** The appendix was stripped by the parser; this information exists in the original submission. Flagging absent appendix content is not a valid criticism under our rules.

- **Harsh critic: "Missing related works."** We cannot verify the existence of unspecified related works.

- **Strength finder: "Curation makes weaker open-source models viable alternatives to commercial LLMs for rating."** While Table 3 shows LLaMA with curation (60.2) matches GPT without curation (60.2), this strength is undermined by the TruthfulQA degradation under LLaMA ratings noted above. The "viability" claim is partially contradicted by the safety regression. Moved here per rules on filtering strengths that conflict with verified weaknesses.

## Novel Insights

The most insightful observation across the reviews is the systematic pattern where DS²'s score curation tends to improve average benchmarks but specifically degrades TruthfulQA under multiple rating models (LLaMA and GPT at 10k, LLaMA and Mistral at 1k). This suggests that the curation mechanism may be trading truthfulness for performance on other axes — perhaps because score curation's tendency to correct toward majority scores inadvertently removes samples that are crucial for safety training, or because the transition matrix framework treats all errors as equally bad regardless of their safety implications. This pattern deserves explicit investigation.

## Suggestions

- Report TruthfulQA results separately and discuss when/why curation degrades truthfulness. Consider whether a safety-aware threshold could be incorporated into the curation mechanism.
- Validate the estimated transition matrix T on a subset with human annotations — even a small-scale validation (e.g., 500 human-scored samples) would dramatically increase confidence in the core mechanism.
- Add error bars from at least 3 fine-tuning seeds, especially for the curation improvement claims which are in the 0.4–1.2 point range.
- Qualify the abstract's "matches or surpasses LIMA" claim to specify the conditions under which it holds, or report results across all rating models.

## Evaluation

**Originality:** The application of transition matrix estimation from label noise literature to LLM rating errors is a reasonable and somewhat novel framing. However, the extension from categorical to ordinal scores is assumed rather than justified, and the diversity component is standard k-NN density estimation.

**Importance of research question:** Understanding and correcting LLM rating errors for data selection is a practically important problem, especially for making open-source models viable as raters.

**Claims support:** The main claim (3.3% outperforms full data) is well-supported. The LIMA comparison claim is selectively true. The score curation claim is true on average but masks important per-benchmark regressions.

**Soundness of experiments:** Comprehensive baselines and multiple base/rating models, but lack of variance reporting and no direct validation of the core mechanism weaken the evidentiary basis.

**Clarity:** The paper is generally well-structured with clear pipeline description (Figure 1), though some mathematical notation (Eq. 1) is dense.

**Value to community:** Practical value for data-efficient instruction tuning, especially the finding that curation reduces sensitivity to rating model choice. The TruthfulQA issue, however, raises safety concerns that need addressing.

## Calibration

**Anchors compared:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Data Selection via Optimal Control | /home/wg25r/review_agent/human_reviews/dhAL5fy8wS.md | 8.0 | Far stronger theoretical contribution with PMP-based formulation; DS² is weaker theoretically and has validation gaps |
| Self-Alignment with Instruction Backtranslation | /home/wg25r/review_agent/human_reviews/1oijHJBRsT.md | 8.0 | More elegant iterative method with cleaner results; DS² has more baselines but also more unacknowledged failures |
| Understanding and Mitigating Label Noise | /home/wg25r/review_agent/human_reviews/TjhUtloBZU.md | 8.5 | More novel topic with comprehensive study; DS² is less novel and has overclaimed results |
| DELIFT | /home/wg25r/review_agent/human_reviews/Fty0wTcemV.md | 6.0 | Similar profile: data-efficient instruction tuning with modest gains; DS² has better baseline coverage but similar magnitude of improvements |
| Rethinking Data Selection at Scale | /home/wg25r/review_agent/human_reviews/qUJsX3XMBH.md | 4.4 | Negative results paper; DS² has positive contributions and more thorough evaluation |
| GLAN | /home/wg25r/review_agent/human_reviews/MpCxUF8x61.md | 5.25 | Good idea with execution issues; similar quality tier to DS² |
| Noisy Data Pruning | /home/wg25r/review_agent/human_reviews/6PGT9OJX5N.md | 3.0 | Unoriginal; DS² is clearly better with genuine contributions |
| Data-Evolution Learning | /home/wg25r/review_agent/human_reviews/cADdVJYiIG.md | 2.5 | Overclaimed with insufficient validation; DS² is better but shares some validation gaps |

DS² sits above the low anchors (has genuine contributions and positive results), below the high anchors (weaker theory, overclaimed, unacknowledged failures), and is comparable to medium anchors like DELIFT and GLAN. The unacknowledged TruthfulQA degradation and selective LIMA comparison claim push it below DELIFT (which had cleaner results), placing it in the 5.0-5.5 range.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>