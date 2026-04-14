=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary

This paper critically re-examines *Machine Translation from One Book* (Tanzer et al., 2024), which claimed that prompting long-context LLMs with a grammar book enables XLR translation. Through careful ablation—manually separating parallel examples from grammatical explanations in the grammar book—the authors demonstrate that virtually all translation gains stem from parallel data, not grammatical explanations. They further show that fine-tuning a small encoder-decoder model (NLLB) on the same parallel data matches Gemini's grammar-book-prompted performance, and introduce a typological feature prompt that, while unhelpful for translation, achieves leading results on grammaticality judgment and IGT prediction. The central recommendation is that data collection for XLR MT should prioritize parallel data over linguistic description.

---

## Strengths

- **Precise disentanglement of a confounded signal in prior work.** MTOB lumped parallel examples with grammatical explanations under a single "grammar book" condition. The manual split into `BOOK_para` and `BOOK_non-para` is the crux of the replication: `BOOK_para` consistently matches or beats `BOOK_all` while `BOOK_non-para` collapses (up to 8 CHRf++ below `BOOK_para` into kgv, up to 7 points below 0-SHOT for npi/gug). This ablation directly falsifies the narrative that grammatical explanations drive XLR translation performance—something no prior work had done for this benchmark.

- **Statistically grounded main claim.** Rather than resting solely on point estimates, Section 5.1 fits a regression of CHRf++ against test-set type coverage across *all* prompt settings (0-shot through BOOK_all). All settings, including BOOK_all, fall within the 95% confidence interval of the coverage-based regression line (p < 0.005, F-test). This means the modest edge of BOOK_all over BOOK_para can be entirely accounted for by its marginally greater vocabulary coverage, not by any independently useful grammatical content—making the null result on grammatical explanations statistically defensible rather than merely anecdotal.

- **Constructive contrast via typological prompting.** Rather than simply delivering a negative result, the paper introduces a novel method—extracting categorical typological features from Grambank and constructing a structured prompt—that outperforms BOOK_all on grammaticality judgment (83% vs. 76% for SHUFFLE) and achieves leading morpheme accuracy on IGT prediction (46.1% vs. best supervised baseline at 45.4%). This bifurcation—grammar helps *linguistic* tasks but not MT—is a nuanced and practically informative finding specific to this work.

- **Practical efficiency argument is well-evidenced.** Table 4 shows that NLLB fine-tuned on PARA_book (34.2 CHRf++) is within 0.2 CHRf++ of Gemini+BOOK_all (34.4) into kgv, using ~5× fewer tokens (18k vs. 99.6k) and local GPU compute. This is not a generic "small model is cheaper" claim; it is precisely calibrated to the same data source and the same directional comparison.

- **Generalisation beyond Kalamang.** Extending results to npi (Nepali) and gug (Guarani) with the large-scale FLORES devtest set (1,012 examples) strengthens the core finding beyond a single language. BOOK_non-para harms both seen low-resource languages by up to 7 CHRf++ relative to 0-SHOT, while BOOK_para is neutral or slightly positive.

---

## Weaknesses

### Fatal
None.

### Major

- **Absence of significance testing for secondary task results.** The claim that "LLMs *can* exploit grammatical information for relevant linguistic tasks" rests primarily on two comparisons: (1) TYP+BOOK_para at 83% vs. BOOK_para at 80% on SHUFFLE grammaticality judgment (3 more correct answers on 100 binary questions; estimated SE ≈ 3.8%), and (2) TYP+BOOK_para at 46.1% vs. BOOK_para at 45.4% morpheme accuracy on a 447-word IGT test (~3 additional correct morphemes). Neither comparison is accompanied by a significance test or confidence interval. The regression analysis in §5.1 is explicitly limited to translation settings. Without paired tests (e.g., McNemar's for accuracy, or bootstrap for morpheme accuracy), it is unclear whether the typological prompt's advantage on these tasks is replicable signal or noise—which directly undermines the paper's secondary narrative that grammar helps when the task is right.

- **Grammar book filtering methodology not validated.** The paper's core empirical contribution hinges on the manual split of the grammar book into parallel and non-parallel content, but no filtering criteria, guidelines, or inter-annotator reliability metrics are reported. For a binary split that determines every experimental condition in the paper, readers cannot assess how consistently the filtering was applied, whether implicit bilingual signal (e.g., morpheme glosses embedded in prose) was inadvertently included in BOOK_non-para, or whether the authors' judgements would generalise to other books. This is a gap in methodological transparency that weakens confidence in the ablation.

### Minor

- **Llama-I output failure for gug is unquantified.** The paper notes that "the model often fails to output translations on the first line for BOOK_p settings" for gug–eng with Llama-I (Table 3, BOOK_para shows 11.8 vs. 0-SHOT at 23.6). The word "often" is not quantified: how many of the 1,012 FLORES examples are affected? CHRf++ is sensitive to empty or malformed outputs, so this failure mode could be responsible for most or all of the apparent drop rather than any content-related effect. At minimum, the percentage of failed outputs should be reported, and ideally the score should be computed over valid outputs only.

- **SWAP_adj task validity is uncertain.** The paper acknowledges "we cannot guarantee all corruptions are ungrammatical," and results show all settings cluster near 60–65% on SWAP_adj—consistent with many adjacent swaps being grammatically acceptable in a language with flexible word order (Kalamang has a degree of argument-order flexibility). This means the SWAP_adj condition may partly measure sensitivity to canonical ordering rather than grammaticality per se, limiting what can be concluded from it. The authors should explicitly discuss this caveat in interpreting SWAP_adj scores.

- **Narrow empirical base for the policy recommendation.** The conclusion that "data collection for XLR MT is better focused on parallel data over linguistic description" is stated broadly but rests on three languages, only one of which (kgv) is truly unseen by LLMs. For npi and gug, the LLM's prior competence suppresses the expected gains from grammar books, making it impossible to distinguish "grammar books never help" from "grammar books only help when the model has no prior knowledge." The recommendation should be more explicitly hedged to the conditions tested—specifically, languages at the kgv level of documentation, using current-generation LLMs.

### Tiny

- **Abstract phrasing "within 0.2 CHRf++ of Gemini with a grammar book"** refers to the BOOK_all condition specifically (34.4 vs. 34.2 CHRf++ into kgv). While numerically accurate, this comparison disappears when PARA_train is added; a brief clarification of which Gemini setting is the reference would prevent misreading.

- **5\*-SHOT retrieval is the strongest book-parallel-equivalent setting** (38.9 into kgv with Gemini) yet receives limited discussion. Since it uses far fewer tokens (~0.8k) while outperforming BOOK_all (34.4), it is arguably the most token-efficient finding in the paper and merits explicit discussion alongside the NLLB fine-tuning comparison.

---

## Nice-to-Haves

- **Ablation of typological feature categories.** The typological prompt is currently tested as an undivided unit. Systematically removing feature categories (e.g., word order vs. case marking vs. tense) would identify which features drive the improvements on grammaticality judgment and IGT, strengthening the claim that *typological knowledge* specifically is useful rather than the prompt format.

- **RAG-style targeted retrieval of grammar sections.** The paper attributes part of the grammar book's failure to a "needle-in-a-haystack" effect. Testing whether selecting and injecting *relevant* grammar sections (e.g., verb morphology chapters when translating verb-heavy sentences) could unlock grammatical utility would directly address this hypothesis and clarify whether the failure is fundamental or a prompt engineering artifact.

- **Inference cost estimates.** A brief monetised estimate (USD per 1,000 translations) for Gemini+BOOK_all vs. NLLB+PARA_book would sharpen the practical efficiency argument, given that Gemini's 99.6k-token context carries API costs not borne by NLLB.

- **Broader model coverage.** Testing a second frontier model (e.g., GPT-4o or Claude) would help confirm that the grammar book null-result is not Gemini-Flash-specific, given that Flash was chosen over Pro for cost reasons.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[REMOVED] Regression analysis is circular.** The harsh critic argued the regression is "nearly tautological" because more parallel data raises both coverage and CHRf++. This misreads the analysis. The regression in Figure 2 plots *all* prompt settings—including BOOK_all (grammar + parallel), BOOK_non-para (grammar only), and zero/few-shot baselines—on the same line. The key claim is that BOOK_all does *not* fall above the coverage-predicted line; if grammar were adding independent signal, it should. The circularity objection would hold if only parallel-data settings were included, but they are not.

- **[REMOVED] BOOK_all's 3.6-point advantage over BOOK_para (into kgv with Gemini) is non-trivial.** The critic flagged 34.4 vs. 30.8 as a meaningful gap that the coverage explanation fails to address. But the paper shows this gap is statistically accounted for by the greater type coverage of BOOK_all (which contains the non-parallel text), demonstrated by the regression in §5.1. Additionally, PARA_book_IGT (33.7, comparable tokens to BOOK_para) partially closes this gap by restructuring the same parallel data. The concern is addressed.

- **[REMOVED] "No significant Pearson correlation between token count and CHRf++ (p=0.997) is striking and suggests multicollinearity."** This misunderstands the test. The point of this test is to show that more grammatical explanation tokens (BOOK_non-para contributing 81k tokens to BOOK_all vs. BOOK_para's 18k) do not improve translation. p=0.997 is the correct and expected result if token volume from non-parallel content is irrelevant.

- **[REMOVED] Grammar books not designed for machine consumption should be a more prominent limitation.** The paper explicitly addresses this in §2: "the kgv grammar book is not designed for language learning, but for describing theoretical linguistic phenomena—which MTOB's authors note limits LLMs to a basic competence." This is already a prominent framing element, not an underacknowledged gap.

- **[REMOVED] Abstract's "comparable" claim is misleading given the 10-point gap to Gemini+BOOK_all+PARA_train.** Gemini+BOOK_all+PARA_train uses extra parallel training data unavailable to NLLB in the comparison. The abstract compares to the Gemini+BOOK_all condition only (34.4 vs 34.2), which is the correct like-for-like comparison. Invoking the much stronger setting that adds PARA_train is not a fair objection.

- **[REMOVED STRENGTH] "Well-structured with logical flow."** Generic; applies to any competent paper.

- **[REMOVED STRENGTH] "Reproducibility — hyperparameters detailed."** Generic.

---

## Novel Insights

The paper surfaces a practically consequential asymmetry: grammatical information—even when it exists and is freely available—fails to provide MT gains beyond what is predicted by the vocabulary coverage it incidentally contains, yet the same LLM can exploit that grammatical information when the task intrinsically requires morphosyntactic reasoning (glossing, grammaticality judgment). This suggests that the failure mode is not the LLM's inability to read or process linguistic descriptions, but rather a mismatch between the form in which grammar books encode linguistic knowledge and the form in which in-context learning for sequence-to-sequence generation can exploit it. The additional finding that typological features—discrete, cross-linguistically standardized, one-liners—outperform full grammar text on linguistic tasks points toward a principle: structured, task-aligned linguistic knowledge is more useful than descriptive prose, even when both encode the same underlying facts. This has implications for prompt engineering for any low-resource task where linguistic metadata databases exist.

---

## Suggestions

1. **Report significance tests or confidence intervals for every primary comparison in the grammaticality judgment and IGT results.** Given test sets of 100 and 447 items respectively, even 3–7% apparent gains may not be statistically reliable. A McNemar's test for judgment accuracy and a bootstrap confidence interval for morpheme accuracy are computationally cheap and would either confirm or properly qualify the typological prompting claims.

2. **Document and justify the parallel/non-parallel filtering criteria in the paper body** (not just in an appendix). Describe the decision rules used to assign each line to BOOK_para or BOOK_non-para, and quantify inter-annotator agreement for a random sample. This is necessary to make the ablation reproducible and to allow others to apply the same split to other grammar books.

3. **Quantify the Llama-I output failure rate for gug-eng BOOK_para** and report scores on the subset of valid outputs separately. If the failure rate is substantial, Table 3's Llama-I gug numbers for BOOK_para should not be compared directly to BOOK_all numbers.

4. **Conduct an ablation of the typological prompt by feature category** (word order, morphological complexity, case systems, etc.). Even a coarse grouping—morphosyntactic vs. lexical features—would substantially strengthen the claim that typological structure is the operative ingredient, and would guide practitioners who want to build similar prompts for other languages.

5. **Hedge the policy recommendation explicitly to the documented conditions.** The recommendation to prioritize parallel data collection should clearly state it applies to languages and LLMs matching the conditions tested: languages with documented typological features, LLMs trained primarily on high-resource data, and tasks where parallel supervision is the primary learning signal. Morphologically complex languages with very sparse or no parallel data may present a different trade-off.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0]
Average score: 7.3
Binary outcome: Accept
