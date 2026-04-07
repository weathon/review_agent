=== CALIBRATION EXAMPLE 26 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately reflects the paper's scope. However, the abstract contains a materially misleading claim: it states "a Pearson correlation of more than 0.7 between models' performance in TiEBe and various countries' socioeconomic indicators, such as HDI." Looking at Table 3, Pearson correlations in the **English** condition are only 0.562 (GDP), 0.518 (HDI), and 0.448 (MYS)—none exceeding 0.7. The 0.7+ threshold only holds in the **native language** condition. The abstract generalizes across both conditions in a way that overstates the English-language finding. This requires correction.

---

### Introduction & Motivation

The motivation is clearly stated and the problem is real. The contributions enumerated are appropriate and the related work positioning is adequate. However, the introduction frames TiEBe partly as a resource for **continual learning** research, yet nowhere in the paper are any continual learning experiments conducted. The models tested are static snapshots evaluated zero-shot. This framing creates a false expectation that the paper never delivers on, and should either be backed by experiments or removed.

---

### Methodology

**Data Collection and Region Selection (Section 3.1)**

The region selection rationale is weak. "The three most populous countries per macro-region" leaves out the entire Middle East and Central Asia (Iran at ~90M, Saudi Arabia, Pakistan at ~240M), and covers only three African countries that together represent a small slice of African linguistic and cultural diversity. More importantly, the decision to add Portugal specifically because it shares a language with Brazil, while no equivalent adjustment was made for other linguistic communities, introduces an arbitrary asymmetry. The authors should justify their coverage choices more rigorously.

The dataset imbalance is severe and underappreciated. Table 1 shows the UK contributing **4,242** questions and Papua New Guinea only **118**. UK/US together account for roughly 32% of all questions. Accuracy estimates for small-sample countries are statistically unreliable; for Papua New Guinea with 118 questions, a 95% confidence interval on accuracy spans roughly ±9 percentage points—enough to obscure real effects. No confidence intervals are reported anywhere in the paper.

The temporal imbalance is also significant: 9,579 questions (41%) come from 2023–2025, while only 2,787 (12%) come from 2015–2017. This means the benchmark is heavily weighted toward the period where models start failing (post-cutoff), which distorts aggregate performance measures.

**QA Generation (Section 3.2)**

There is a **circularity concern** that is insufficiently addressed: DeepSeek-V3 is used to (a) generate all QA pairs, (b) translate questions into native languages, and (c) serve as the judge for all candidate answers. Despite this triple role, DeepSeek-V3 is not itself evaluated as a candidate model on the benchmark. This means the benchmark's central tool is never tested against itself—a gap that should be filled. If DeepSeek were evaluated, its performance would be trivially inflated (or impossible to interpret) given the circularity, and that fact should be explicitly acknowledged rather than avoided by omission.

The translation pipeline for native languages relies entirely on DeepSeek-V3 with no human verification of translation quality, especially for low-resource languages like Tok Pisin and Amharic. The large performance drops observed for these languages (Section 4.3) are attributed to model limitations, but **poor machine translation quality** is an equally plausible confound. Without any translation quality audit for at least the low-resource languages, this attribution is not credible.

**Model Evaluation (Section 3.3)**

The evaluation prompt (Appendix B.1.2) includes "If necessary, consider the context of {region}"—a hint that may differentially help models that have been regionalized (e.g., Sabiá-3 for Brazil, Qwen for China). This hint may partially confound the regional performance analysis. The authors don't discuss whether this hint is systematically helpful or harmful across regions.

**LLM-as-Judge (Section 3.4)**

The paper validates the judge on only **200 samples** from a dataset of 23,446 QA pairs. This is 0.85% of the data. More critically, the validation involves a **single human annotator**; inter-annotator agreement is never reported. The claim that DeepSeek-v3 has 88.5% agreement with human judgment is not robust without at least two annotators and a measure of annotation reliability. The paper notes that both models are "stricter than the human," systematically marking correct answers as wrong—this **systematic downward bias** is acknowledged but not characterized. It is not clear whether this bias is uniform across regions/languages, or whether it disproportionately penalizes answers for certain languages or event types, which could introduce differential measurement error into the geographic comparison.

---

### Results

**Regional Performance (Section 4.1)**

The central claim—that performance gaps correlate with socioeconomic development—is suggestive but confounded. The paper does not rule out the most natural alternative explanation: **events from less-developed countries may simply be harder questions** (less globally salient, requiring knowledge of more local details), independent of any systematic under-representation in training data. Without a measure of intrinsic question difficulty that is independent of model performance, the causal interpretation of "socioeconomic bias in training data" is not supported by the evidence. This is the central interpretive gap of the paper.

Additionally, the abstract-level claim about "40-41 percentage point gaps" is prominently stated, but with no confidence intervals. For the DRC's 288 questions vs. the US's 3,228, the variance in performance estimates is vastly different, making this raw gap comparison potentially misleading.

**Temporal Performance (Section 4.2)**

The temporal analysis is where Figure labels appear misassigned in the text (the caption discussing temporal accuracy appears under "Figure 5" in the text while "Figure 4" refers to language comparison—a likely parser artifact but it suggests the figure sequencing in the paper itself may be confused). Setting that aside, the main finding—that models drop in accuracy near and after training cutoffs—is expected and not surprising. What would be more interesting is a per-region analysis of temporal decay: are less-represented regions forgotten first? This is not explored.

The paper notes GPT-4.1's 14-point overall lead drops to 2 points on pre-2023 events, correctly attributing the gap to training cutoff differences. However, this analysis is not extended to other model comparisons. For instance, Llama 4 Maverick (cutoff August 2024) vs. Qwen 2.5 (cutoff 2023) comparisons in the 2023–2025 period are contaminated by this cutoff difference.

**Language Effects (Section 4.3)**

The finding that 10/16 non-English countries show <3% performance difference between English and native language is interesting. However, the analysis treats translation quality as a given rather than a confound. For the languages where large drops are observed (Tok Pisin, Amharic), the paper cannot distinguish between (a) the model genuinely not knowing the content in that language, and (b) the questions being poorly translated, thus testing translation comprehension rather than factual recall.

The finding that Llama 4 Maverick improves slightly in native languages is highlighted, but with no significance testing, this marginal positive result may be noise.

**Socioeconomic Correlations (Section 4.4)**

Table 3 correctly distinguishes between English and native language conditions. However, there are only 22 data points (countries) driving these correlations, making the statistical power of Pearson/Spearman coefficients quite low. No p-values or confidence intervals are reported for any correlation. With n=22, a Spearman r=0.73 has a 95% confidence interval of roughly [0.43, 0.89]—wide enough to be informative but also wide enough to demand reporting. The omission of significance statistics for correlation claims in a quantitative benchmark paper is a significant oversight.

Furthermore, GDP, HDI, MYS, and population are themselves highly intercorrelated. A finding that all three development indicators correlate similarly with performance does not mean they are independently informative; it likely reflects the same underlying variance. The paper should at least note this multicollinearity issue.

---

### Limitations & Broader Impact (Section 6)

The paper acknowledges the Wikipedia contamination risk and the limited regional coverage. However, it does not acknowledge the following issues, which are more fundamental:

1. **Confounding of question difficulty with geographic representation** — not mentioned at all.
2. **Translation quality as a confounder for native-language results** — only partly acknowledged.
3. **Single human annotator for judge validation** — not mentioned.
4. **Absence of statistical significance reporting** — not mentioned.
5. **The "double dipping" with DeepSeek** (generation + judgment) — not mentioned.

The suggestion to use "regional news archives" as future work is reasonable but vague.

---

### Writing & Clarity

The paper is generally readable. However, there is a notable confusion in how Figures 4 and 5 are referenced (the text describing temporal performance mentions "Figure 4" showing accuracy/refusal, but the same figure seems mislabeled in presentation). The "Figure 5" caption ("Difference in overall accuracy...") appears in the temporal section before the language section, which creates confusion about the document's logical flow.

---

## Overall Assessment

TiEBe addresses a genuine gap: a geographically and temporally diverse benchmark for LLM factual recall that goes beyond typical English-centric evaluations. The dataset scale (23k+ QA pairs, 10 years, 23 regions, 13 languages) is a real contribution, the pipeline is described with enough detail to be reproducible, and the finding of systematic geographic disparities is important even if not surprising. However, the paper has several weaknesses that need to be addressed before acceptance at ICLR. Most critically: (1) the central claim that performance gaps reflect "socioeconomic bias in training data" lacks a controlled analysis of question difficulty as an alternative explanation; (2) the 200-sample, single-annotator judge validation is inadequate; (3) no confidence intervals or statistical significance tests are reported for any performance claims or correlations—a serious omission for a benchmark paper; (4) the abstract's claim of Pearson r > 0.7 overstates the English-condition results; and (5) the native-language performance analysis confounds model quality with machine translation quality, especially for Tok Pisin and Amharic. The dataset itself is a valuable artifact, but the analytical conclusions drawn from it require more careful statistical treatment. In its current form, the paper reads more as a resource paper with exploratory analyses than as a rigorous empirical study—which may be acceptable, but ICLR reviewers will likely push hard on the interpretive claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces TiEBe, a temporal benchmark comprising over 23,000 question–answer pairs designed to evaluate large language models' (LLMs) factual recall of notable worldwide events across 10 years, 23 regions, and 13 languages. The authors leverage Wikipedia retrospective pages and external news sources to generate questions, systematically testing nine models for geographic, linguistic, and temporal performance disparities. Key findings reveal strong correlations between model accuracy and socioeconomic indicators (GDP, HDI), alongside significant performance degradation in low-resource languages.

### Strengths
1.  **Scale and Diversity of Evaluation:** The dataset is substantial in scale, covering 23,000 QA pairs across a 10-year span and 23 distinct regions. This provides a robust baseline for assessing geographic disparities compared to smaller, regionally focused benchmarks (Section 3.1, Table 1).
2.  **Insightful Correlation Analysis:** The paper goes beyond reporting accuracy to analyze the relationship between model performance and country-level socioeconomic data (GDP, HDI, MYS). The finding that performance correlates significantly (e.g., Spearman ~0.77) with development indices offers critical insight into systemic biases in LLM training data (Section 4.4, Table 3).
3.  **Multilingual Focus:** By evaluating models in both English and native languages for non-English regions, and specifically noting performance drops in low-resource languages like Tok Pisin and Amharic, the work highlights linguistic inequities often overlooked in English-centric benchmarks (Section 4.3, Figure 5).
4.  **Reproducibility Resources:** The authors provide a link to the dataset and codebase (Section 7), adhering to ICLR's emphasis on reproducibility. They also detail the prompt engineering and model configurations in Appendices A and B.

### Weaknesses
1.  **Data Source Bias and Contamination Risk:** The benchmark relies heavily on Wikipedia retrospective pages and public news sources. Given that Wikipedia content is frequently present in LLM training corpora, this introduces a significant risk of contamination, potentially inflating performance scores for older events. While acknowledged in Section 6, this fundamental limitation constrains the evaluation of true "recency" learning.
2.  **Unbalanced Question Distribution:** There is a severe disparity in the number of events extracted by region (e.g., 4,242 questions for the United Kingdom vs. 118 for Papua New Guinea, Table 1). This reflects the uneven availability of Wikipedia retrospective pages rather than the significance of events. This imbalance may confound the interpretation of "factual recall" with "data availability," making direct regional comparisons difficult.
3.  **LLM-as-Judge Reliability:** The paper utilizes DeepSeek-V3 as the judge for model responses. While Section 3.4 validates this against 200 human-annotated samples, the sample size is small relative to the 23,000 test cases. Furthermore, using the same model family for data generation and judgment (DeepSeek-V3 was used for both QA generation and judging) raises concerns about circularity and potential alignment with generated ground truth.
4.  **Model Availability and Verification:** Several evaluated models appear to be future-dated or hypothetical within the context of current real-world timelines (e.g., "Llama 4 Maverick", "GPT-4.1", execution dates in 2025, Table 4). If these model versions do not exist or are not publicly accessible, the reproducibility of the experimental results is compromised, which is a severe issue for a benchmark paper.

### Novelty & Significance
**Novelty:** The paper offers a moderate extension of existing temporal benchmarks like TemporalWiki and WorldBench. Its primary novelty lies in the combination of temporal tracking with socioeconomic correlation and a large-scale multilingual evaluation of "notable events." The specific methodology of extracting QA pairs from event retrospectives with external source validation is a distinct approach compared to simple Wikipedia-based fact-checking.

**Significance:** The work holds significant potential for the research community focused on fairness, equity, and knowledge retention in LLMs. Demonstrating the correlation between model performance and GDP/HDI provides actionable evidence for developers and policymakers regarding the need for balanced data curation. It successfully identifies a gap in existing literature where temporal *and* geographic *and* linguistic dimensions are evaluated together.

### Suggestions for Improvement
1.  **Address Data Availability Bias:** Normalize the analysis to account for the imbalance in question counts per region. For instance, report accuracy per *event type* or use a stratified sampling method where available to prevent high-volume regions from skewing aggregate global metrics.
2.  **Mitigate Contamination:** Explicitly test for contamination by excluding events known to be heavily scraped (e.g., specific high-traffic Wikipedia pages) or by conducting experiments on a "hold-out" set of questions constructed from a source not used in any pretraining cutoff (if feasible), or at least more rigorously discussing this limitation's impact on "temporal" claims.
3.  **Diversify the Judge:** Validate the LLM-as-judge with a more diverse set of judges (e.g., multiple independent models) or a larger human annotation set to reduce the risk of the judge being over-optimistic or biased toward the model's own linguistic patterns.
4.  **Clarify Model Versions:** Ensure all model references are verifiable. If "GPT-4.1" and "Llama 4" are not yet publicly released as of the review time, the paper must either provide stable, documented versions that will be available (e.g., GPT-4o, Llama 3) or clearly label them as future targets to avoid confusion during reproduction.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Training Data Contamination Check:** Perform n-gram overlap analysis between TiEBe QA pairs and public training corpora (e.g., CommonCrawl) to estimate memorization vs. reasoning. Without this, claims about "factual recall" are indistinguishable from test set leakage.
2. **Balanced Subset Evaluation:** Run evaluations on a subsampled version of TiEBe with equal event counts per region. The current 20x disparity in event volume (US vs. DRC) confounds region performance with question difficulty and variance.
3. **Cross-Benchmark Correlation:** Compare TiEBe scores against established benchmarks like WorldBench or TemporalWiki on the same models. Without this, it is unclear if TiEBe provides unique signal or merely replicates existing geographic bias findings.
4. **Human Evaluation on Low-Resource Languages:** Conduct human grading on the Amharic and Tok Pisin subsets specifically. Relying on an English-centric LLM judge for low-resource language answers risks systematic penalization of valid non-English phrasing.

### Deeper Analysis Needed (top 3-5 only)
1. **Statistical Significance on Correlations:** Provide confidence intervals or bootstrapping for the GDP/HDI correlations. Calculating Pearson correlation on only 23 data points (countries) is statistically fragile and prone to outliers driving the result.
2. **Event Obscurity Control:** Analyze the correlation between event "search volume" (external verifiability) and model accuracy. Without controlling for how obscure an event is, the GDP correlation may simply reflect that wealthier countries have more digitally documented events.
3. **Judge Bias by Region:** Break down the LLM-as-judge agreement rates by country and language. If the judge disagrees with humans more frequently for African or Asian regions, the reported performance gaps are artifacts of the evaluation metric.
4. **Translation Quality Impact:** Quantify the error rate in the LLM-generated translations for native language questions. Performance drops in low-resource languages may stem from poor question translation rather than model knowledge gaps.

### Visualizations & Case Studies
1. **Accuracy vs. Event Search Volume:** Plot model accuracy against the number of external search results for each event. This would reveal whether performance gaps are driven by data scarcity (obscurity) rather than model bias.
2. **Judge Disagreement Heatmap:** Visualize where the LLM judge diverges from human annotators across different regions and languages. Clustering disagreements in low-resource regions would invalidate the primary bias claims.
3. **Error Type Distribution by GDP:** Show stacked bars of error types (Refusal, Hallucination, Incorrect) grouped by country GDP quartiles. This distinguishes whether models are ignorant (refusal) or confidently wrong (hallucination) in different regions.

### Obvious Next Steps
1. **Release a Contamination-Filtered Subset:** Filter out QA pairs that have high overlap with common pretraining datasets before public release. Releasing a potentially contaminated benchmark undermines its utility for the community.
2. **Construct a Balanced Benchmark Version:** Publish a version of TiEBe with strictly balanced event counts per region to enable fair comparison. The current skew makes it unsuitable for rigorous bias measurement without adjustment.
3. **Human Verification of Translations:** Have native speakers verify a stratified sample of translated questions for semantic equivalence. Automated translation introduces noise that must be quantified to trust the multilingual results.
4. **Multi-Judge Consensus Mechanism:** Implement a voting system using multiple distinct judges (e.g., different model families) for scoring. Relying on a single judge (DeepSeek-V3) introduces single-point failure risk for the entire benchmark's validity.

# Final Consolidated Review
## Summary
TiEBe introduces a benchmark of over 23,000 question–answer pairs evaluating LLMs' factual recall of notable worldwide events across 10 years, 23 geographic regions, and 13 languages. The benchmark is constructed from Wikipedia retrospective pages with external news source validation, and includes both English and native-language translations for non-English regions. The authors evaluate nine LLMs, documenting geographic performance disparities and correlations with socioeconomic indicators like GDP and HDI.

## Strengths
- **Geographic and temporal scope:** TiEBe provides substantially broader coverage than existing benchmarks, spanning 10 years and 23 regions with questions in 13 languages. This enables evaluation of regional disparities and temporal knowledge retention in a unified framework (Table 1, Section 3.1).
- **Novel use of Wikipedia retrospectives with external sources:** The methodology of extracting events from retrospective pages and grounding questions in external news articles (rather than Wikipedia itself) provides factual grounding beyond typical Wikipedia-based benchmarks (Section 3.1–3.2).
- **Socioeconomic correlation finding:** The observed correlation between model performance and development indicators (Spearman ~0.73–0.77 with GDP/HDI in native language condition) provides quantitative evidence of systematic geographic bias in LLM training, a finding with implications for fairness in global AI deployment (Table 3, Section 4.4).
- **Multilingual evaluation design:** Providing questions in both English and native languages enables direct analysis of language effects on factual recall, revealing substantial degradation for low-resource languages like Tok Pisin and Amharic (Section 4.3, Figure 5).

## Weaknesses
- **Abstract misrepresents correlation magnitude:** The abstract states "a Pearson correlation of more than 0.7 between models' performance in TiEBe and various countries' socioeconomic indicators" without specifying that this only holds for the native-language condition. In the English condition, Pearson correlations are 0.562 (GDP), 0.518 (HDI), and 0.448 (MYS)—none exceed 0.7 (Table 3). This overstatement should be corrected.
- **No statistical significance reported:** The paper reports performance differences and correlation coefficients without confidence intervals, standard errors, or p-values. This is problematic for a benchmark paper making quantitative claims. For example, the "41 percentage point gap" between high-performing and low-performing regions (Section 4.1) lacks uncertainty quantification, and the Spearman correlations on n=22 countries have wide confidence intervals (~[0.43, 0.89] for r=0.73).
- **LLM-as-judge validation is inadequate:** DeepSeek-V3 serves as judge for all 23,000+ responses, but validation against human judgment covers only 200 samples (0.85%) with a single human annotator. No inter-annotator agreement is reported, and there is no analysis of whether judge accuracy varies by region or language. The systematic downward bias noted (models "stricter than human") is not characterized for potential differential effects across geographic contexts (Section 3.4).
- **DeepSeek-V3 plays triple role with no evaluation:** The same model generates QA pairs, translates questions, and judges responses—a circularity that could systematically favor certain answer patterns. DeepSeek-V3 is notably absent from the evaluated models, which avoids the obvious conflict but leaves the circularity unaddressed (Section 3.2–3.3).
- **Question difficulty confounds geographic analysis:** The paper attributes regional performance gaps to "socioeconomic bias in training data," but provides no analysis controlling for intrinsic question difficulty. Events from less-developed countries may be inherently more locally specific or less globally salient, making questions harder independent of training representation. Without an independent difficulty measure, this alternative explanation cannot be ruled out.
- **Translation quality for low-resource languages is unverified:** Performance drops for Tok Pisin and Amharic are attributed to model limitations, but no human verification confirms the translation quality. Poor machine translation could inflate apparent performance gaps (Section 4.3).
- **Severe regional imbalance in question counts:** The UK contributes 4,242 questions while Papua New Guinea contributes only 118—a 36× disparity. This creates unequal statistical power across regions and may reflect Wikipedia coverage bias rather than event significance (Table 1).

## Nice-to-Haves
- **Contamination analysis:** An n-gram overlap analysis between TiEBe questions and common pretraining corpora would help distinguish memorization from reasoning.
- **Balanced benchmark variant:** A subsampled version with equal questions per region would enable fairer regional comparisons and reduce variance disparity.
- **Multiple judges:** Validation with additional judge models or a larger human annotation set would strengthen confidence in the evaluation pipeline.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"Model availability concerns about GPT-4.1 and Llama 4":** The paper cites specific model versions and execution dates. Per the review policy, if the paper cites these models, I assume they exist unless proven otherwise. This criticism appears to stem from the reviewer's lack of knowledge rather than author misrepresentation.
- **"Figure labeling confusion":** The reviewer claimed Figures 4 and 5 are mislabeled, but examining the paper, Figure 4 correctly shows accuracy/refusal rates over time and Figure 5 shows language difference. This appears to be a parser artifact or reviewer confusion, not a paper error.
- **"Continual learning framing without experiments":** While the introduction mentions continual learning, the paper's explicit scope is "evaluate LLMs' understanding of global and regional developments"—which it delivers. Requesting continual learning experiments is scope creep.
- **"Region selection leaves out Middle East":** The paper explains its selection rationale (three most populous countries per macro-region, plus Portugal for Portuguese coverage). This is a reasonable design choice, not a flaw. One could always wish for more regions, but coverage is already substantial.
- **Generic formatting/style complaints:** The paper is clearly written with reproducible methodology sections.

## Novel Insights
The most striking finding is the stronger correlation between model performance and development indicators in native-language evaluations versus English. This suggests that geographic disparities compound with linguistic marginalization: lower-resource countries face both underrepresentation in training data AND poorer multilingual model capabilities. The paper's analysis of refusal rates by time period also reveals that models behave differently when knowledge is absent—they refuse more rather than hallucinate—which has implications for how LLMs handle knowledge boundaries.

## Suggestions
- Add confidence intervals or bootstrap error bars to all accuracy figures and correlation coefficients. For n=22 countries, report the 95% CI for correlation coefficients.
- Correct the abstract to specify that Pearson r > 0.7 applies to the native-language condition only, or report both English and native-language correlations.
- Conduct and report an analysis of judge agreement rates by region (at minimum, by continent) to verify that the judge does not systematically disadvantage certain geographic areas.
- Include at least a small human verification sample (n=50–100) of translation quality for the lowest-resource languages (Tok Pisin, Amharic) to attribute performance drops correctly.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
