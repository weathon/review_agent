=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
## Summary

TiEBe introduces a benchmark of 23,446 question–answer pairs for evaluating LLM factual recall of notable events across 23 geographic regions, 13 languages, and a 10-year time span (2015–2025). QA pairs are generated from news sources cited in Wikipedia retrospective pages using DeepSeek-V3, and nine LLMs are evaluated via an LLM-as-judge approach. The paper documents large geographic disparities in recall, strong correlations between model performance and country-level socioeconomic indicators (GDP, HDI), and sharp degradation for low-resource languages such as Tok Pisin and Amharic.

## Strengths

- **Unique multi-axis evaluation design**: TiEBe simultaneously evaluates factual recall along temporal, geographic, and linguistic dimensions at scale. No existing benchmark combines all three axes with this coverage (23 regions, 13 languages, 10 years), making it a genuinely new evaluation tool for the community.

- **The native-language "double penalty" finding**: The paper discovers that correlations with socioeconomic indicators are consistently stronger in the native-language setting than in English (Table 3: Spearman 0.767 vs. 0.728 for GDP, 0.747 vs. 0.549 for HDI). This is a novel and actionable insight—it suggests low-resource countries face compounding disadvantages (underrepresentation + poor multilingual training), not just a single axis of bias. This goes beyond what prior work like WorldBench or BLEND reports.

- **Grounding QA pairs in external source documents**: By scraping cited news articles rather than testing directly on Wikipedia text, TiEBe reduces (though does not eliminate) the risk that models answer from memorized Wikipedia passages. This is a meaningful methodological choice that differentiates TiEBe from TemporalWiki and similar Wikipedia-based benchmarks.

- **Transparent evaluation pipeline with human validation**: The LLM-as-judge is validated against human annotators on 200 samples (88.5% agreement for DeepSeek-V3, 91% for GPT-4o), and full prompts, model versions, and execution dates are provided in Appendix B, supporting reproducibility.

## Weaknesses

### Major:

- **Translation quality for low-resource languages is unvalidated, yet drives a core claim.** The paper reports severe performance drops for Tok Pisin and Amharic (Section 4.3), but these questions were translated by DeepSeek-V3 without any human verification of translation fidelity. If the translations into Tok Pisin or Amharic are semantically inaccurate or ambiguous, the measured performance gap conflates translation failure with knowledge failure. Given that these two languages anchor one of the paper's headline findings, this is a serious validity concern. A small human validation of translated QA pairs—even 20–30 samples per low-resource language—would substantially strengthen the claim.

- **Ground truth QA pair quality is not independently validated.** The paper validates the *judge* against humans (200 samples), but never validates whether the *generated QA pairs themselves* are factually correct. If DeepSeek-V3 hallucinated a detail when generating a question-answer pair from a news source, that hallucinated "ground truth" becomes the evaluation standard. The 200-sample human evaluation only checks whether the *judge's correctness decision* matches a human, not whether the gold answers are correct. For a benchmark paper, this gap in the validation chain is significant.

- **Statistical robustness of country-level correlations is not addressed.** The correlations in Table 3 treat each of 22 countries as a single data point, but the accuracy estimates underlying those points have vastly different variances: Papua New Guinea contributes an accuracy estimate from only 118 questions while the UK contributes from 4,242. The paper does not report confidence intervals, use weighted regression, or bootstrap to verify that the GDP/HDI correlations are robust to the high variance in small-sample countries. If the correlation is driven largely by a few well-measured countries, the "systemic imbalance" narrative may be overstated.

### Minor:

- **The abstract's correlation claim is imprecise.** The abstract states "a Pearson correlation of more than 0.7 between models' performance in TiEBe and various countries' socioeconomic indicators," but Table 3 shows Pearson correlations above 0.7 only for the native-language setting (English Pearson with GDP is 0.562, HDI is 0.518). Since the abstract does not specify this distinction, it overstates the English-setting result.

- **DeepSeek-V3 serves dual roles as QA generator and judge.** The same model that created the ground truth QA pairs also evaluates model answers. While the judge evaluates *other models'* answers (not its own), there is a potential for systematic bias: the judge may be more lenient toward answer phrasings that match DeepSeek-V3's own generation patterns. The 88.5% human agreement partially mitigates this, but the agreement rate may not be uniform across languages or question types, and an 11.5% divergence rate on 23,446 questions means ~2,700 potentially misjudged answers.

- **Data contamination risk is acknowledged but not quantified.** Section 6 acknowledges that Wikipedia-derived content may have been seen during pretraining, but the paper does not attempt to measure contamination (e.g., checking for n-gram overlap with CommonCrawl) or to analyze whether higher-accuracy regions show signs of memorization. This limits the interpretability of accuracy differences: some gaps may reflect differential contamination rather than differential knowledge.

- **Temporal analysis mostly confirms known cutoff effects.** The finding that accuracy drops and refusal rates rise after models' training cutoffs (Figure 4) is expected. The more informative question—whether models exhibit temporal decay *within* their training window (e.g., 2015 vs. 2022 events for a 2023-cutoff model)—is not analyzed, even though the data could support it.

### Trivial:

- The imbalance in question counts per region (Table 1) is a natural consequence of Wikipedia retrospective coverage rather than a design flaw, but it should be acknowledged when interpreting regional comparisons.

## Nice-to-Haves

- A balanced subset with capped question counts per region (e.g., 200 per country) would enable fairer cross-regional comparison and could be released alongside the full dataset.
- Correlating TiEBe scores with scores from WorldBench or BLEND on shared regions would clarify whether TiEBe captures distinct signal or replicates known disparities through a different lens.
- A within-training-window temporal decay analysis (e.g., 2015 vs. 2020 vs. 2022 accuracy for models with 2023 cutoffs) would add novelty to the temporal dimension beyond confirming cutoff effects.
- Reporting LLM-as-judge agreement rates separately by language (especially Tok Pisin and Amharic) would verify that the evaluation methodology is reliable for the very languages where the most dramatic findings occur.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Notorious" vs. "notable" terminology**: This is a style nitpick; the meaning is clear from context.
- **Missing related works**: Per review rules, I cannot confirm the existence of uncited works.
- **Reproducibility concerns about undisclosed hyperparameters or implementation details**: Trivial reproducibility nitpicks are excluded per rules.
- **Demanding continual learning experiments**: The paper scopes itself as a benchmark contribution; evaluating continual pretraining methods is outside its stated scope.
- **Questioning the existence or availability of cited models (Sabiá-3, Sabiazinho-3, Llama 4 Maverick, etc.)**: Per rules, if the paper cites them, they exist.
- **Requesting user studies or theoretical proofs**: Not standard for an empirical benchmark paper.
- **Evaluation cost discussion**: Outside the paper's stated contribution scope.
- **Prompt sensitivity analysis across languages**: While potentially informative, this is generic and not a core flaw of the benchmark as presented.

## Novel Insights

The most novel finding is the "double penalty" structure revealed by comparing English vs. native-language correlations with socioeconomic indicators: the Spearman correlation between accuracy and GDP jumps from 0.728 (English) to 0.767 (native), and with HDI from 0.549 to 0.747. This suggests that the global inequity in LLM performance is not just about *what* countries are represented, but about *how* that representation is linguistically mediated—low-HDI countries suffer not only from less content about them, but from dramatically worse multilingual capabilities for their languages. The weak correlation with population (Spearman ~0.15) despite strong GDP/HDI correlations is also notable: it implies that sheer demographic size does not translate to better model performance, countering a common assumption that large populations naturally generate more training data.

## Suggestions

- Add human validation of translated QA pairs for at least the two most critical low-resource languages (Tok Pisin, Amharic)—even 20 samples each would help establish whether the language-gap finding is robust to translation quality.
- Report confidence intervals or bootstrapped standard errors for country-level accuracy estimates, especially for small-sample countries, and verify that the GDP/HDI correlations hold under weighted or bootstrap analyses.
- Add a brief within-training-window temporal analysis (e.g., accuracy for 2015–2017 vs. 2018–2020 vs. 2021–2022 for models with 2023 cutoffs) to move beyond confirming expected cutoff effects.
- Clarify the abstract's correlation claim by specifying "in the native-language setting" or reporting both English and native-language correlations.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
