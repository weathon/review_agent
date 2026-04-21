Now I have enough context to write the review. Let me synthesize everything.

Key findings:

1. **The +0.7 ChrF++ claim is inconsistent with Table 2.** Into kgv, BOOK_all (34.4) vs BOOK_para (30.8) = +3.6, not +0.7. The -0.3 for kgv→eng checks out (34.4 vs 34.7). This is the paper's most prominent quantitative claim and it appears wrong.

2. **The regression analysis has collinearity concerns.** As the harsh critic notes, with only one condition (BOOK_all) that combines both high coverage and grammatical explanations, the regression can't disentangle the effects. However, the paper does have multiple prompt settings that vary in coverage and content, and the overall pattern—BOOK_all ≈ BOOK_para when coverage is similar—is consistent across settings. The regression is suggestive but not definitive.

3. **No variance/statistical testing on main comparisons.** The test set has only 100 examples for kgv, and differences like 0.3 ChrF++ (which the paper relies on) are well within noise without variance estimates.

4. **The "within 0.2 ChrF++" claim for fine-tuning.** NLLB fine-tuned achieves 34.2 vs Gemini BOOK_all's 34.4 into kgv (0.2 gap), but in kgv→eng, the gap is 5.8 (28.6 vs 34.4). The claim is direction-selective.

5. **The grammaticality judgment task is narrow.** All corruptions are word-order perturbations; the paper acknowledges this.

6. **Single LLM (Gemini-1.5-Flash) used for central claims.** Llama can't fit BOOK_all for kgv, so the key comparison rests on one model.

However, these are all in the context of a paper that makes a genuinely important contribution: the decomposition of grammar books into parallel vs. explanatory content, showing that parallel examples drive translation gains. This is a clean and actionable finding with practical implications.

Now let me compare this to calibration anchors:

- Papers that challenge prior claims and show simpler methods work: "Never Train from Scratch" (avg 8), "Redefining Bioactivity Prediction" (avg 7), "Hard-to-Beat Baseline" (avg 6), "Intriguing Properties of Data Attribution" (avg 6)
- The ASROB/MTOB extension paper scored 5.5 (reject)
- Low-quality papers with fundamental methodology issues scored <3

This paper has real contributions (the ablation decomposition, the fine-tuning comparison, the typological prompting) that are novel and practically important. But it has the numerical inconsistency in its central claim and limited statistical rigor. The +0.7 error undermines the abstract's framing significantly.

I'd place this paper around 6.0-6.5. It's a solid contribution with important findings, but the numerical inconsistency and statistical gaps hold it back from being clearly above threshold.</think>## Summary

This paper re-evaluates the claim from MTOB (Tanzer et al., 2024) that long-context LLMs can exploit grammar books for extremely low-resource (XLR) translation. By decomposing grammar books into parallel examples (BOOK_para) and grammatical explanations (BOOK_non-para), the authors show that virtually all translation gains come from the parallel examples, not the explanations. They demonstrate this for Kalamang, Nepali, and Guarani, show that fine-tuning a small NLLB model on parallel data is competitive with long-context LLM prompting, and introduce a typological feature prompt that achieves leading results on grammaticality judgment and gloss prediction—tasks where grammatical knowledge is more relevant.

## Strengths

- **Novel and important ablation disentangling parallel examples from grammatical explanations.** This decomposition directly targets the core confound in MTOB's claim. Table 2 shows BOOK_para outperforms BOOK_non-para by 7–8+ ChrF++ into kgv (30.8 vs 22.6 for Gemini), and Table 3 shows BOOK_all and BOOK_non-para actually decrease performance below 0-SHOT for npi and gug (e.g., gug→eng: BOOK_non-para at 10.1 vs 0-SHOT at 23.6 for Llama-I). This is a clean and actionable finding.

- **Fine-tuning comparison is practically important.** Table 4 shows NLLB-1.3B fine-tuned on PARA_book achieves 34.2 ChrF++ eng→kgv, outperforming Gemini with the same data in-context (26.6) by 7.6 points. This demonstrates the useful signal from grammar books can be leveraged far more efficiently than prompting million-token-context LLMs.

- **Typological prompting achieves strong results on linguistically relevant tasks.** TYP + BOOK_para achieves 83% accuracy on the hardest grammaticality judgment setting (Shuffle, beating BOOK_para's 80% and BOOK_all's 76% in Figure 1/Table) and 46.1% morpheme accuracy on IGT prediction, beating all supervised baselines (Table 5). This supports the nuanced conclusion that LLMs can exploit grammatical knowledge when the task is appropriately aligned.

- **Generalization beyond Kalamang.** Results extend to Nepali and Guarani (Table 3), showing that the failure of LLMs to exploit grammatical explanations is not specific to one unseen language.

- **Methodological improvements over MTOB.** Combining directional test sets to 100 examples, using ChrF++ instead of ChrF, and introducing the critical parallel/non-parallel ablation address specific weaknesses in prior work identified in Section 2.

## Weaknesses

### Major

- **The central quantitative claim in the abstract/introduction appears inconsistent with Table 2.** The paper states that adding grammatical explanations to parallel sentences adds "+0.7 ChrF++ into kgv" (Section 1, line 26). However, from Table 2, BOOK_all scores 34.4 vs BOOK_para at 30.8 into kgv for Gemini—a difference of +3.6 ChrF++. The -0.3 for kgv→eng (34.4 vs 34.7) is correct, but +3.6 is not negligible and directly undermines the framing that explanations add "no significant advantage." If the +0.7 refers to a different comparison (e.g., coverage-matched conditions), this must be explicitly stated, as the natural reading is BOOK_all vs BOOK_para. This numerical inconsistency in the paper's most visible claim requires clarification or correction.

- **The regression-based argument that type coverage fully explains translation gains cannot support the strong causal claim that explanations add nothing (Section 5.1, Figure 2).** With ~15 data points, the 95% CI bands are wide enough that "falling within them" is a weak test. More fundamentally, there is near-complete collinearity: BOOK_all is the only condition combining both high vocabulary coverage and grammatical explanations, making it impossible for a univariate regression to disentangle whether explanations provide independent value. Interestingly, a direct coverage-matched comparison already exists in Table 2—PARA_book+W (34.7) vs BOOK_all (34.4) into kgv—where adding explanations to coverage-equivalent parallel data actually slightly decreases performance. This comparison supports the paper's thesis far more cleanly than the regression but is not analyzed in this framing.

### Minor

- **No variance estimates or significance tests on main translation comparisons.** With a 100-example test set and ChrF++ scores differing by as little as 0.3 points (the kgv→eng direction: BOOK_all 34.4 vs BOOK_para 34.7), these differences are within plausible noise. While single-run evaluation may be standard for expensive LLM experiments, the paper's strong claims about "no significant advantage" would be better supported by confidence intervals or multiple runs.

- **The "fine-tuning within 0.2 ChrF++" claim is direction-selective (Section 1).** NLLB fine-tuned achieves 34.2 vs Gemini BOOK_all's 34.4 eng→kgv (0.2 gap—favorable), but in kgv→eng the gap is 5.8 (28.6 vs 34.4—unfavorable). The abstract highlights only the favorable direction.

- **The core translation comparison for kgv rests on a single model (Gemini-1.5-Flash).** Llama-3.1-8B's context window is too small for kgv BOOK_all, meaning the central claim about grammar books vs. parallel data cannot be verified with a second LLM for the primary language.

- **The grammaticality judgment task (Section 3.4) tests only word-order corruptions.** All three settings (SWAP_adj, SWAP_ran, SHUFFLE) involve word-order perturbations, missing morphology, agreement, and other grammatical phenomena that grammar books extensively cover. The authors acknowledge that not all corruptions may be ungrammatical, which limits the task's validity as a comprehensive test of grammatical knowledge.

### Trivial

- None.

## Nice-to-Haves

- **Coverage-matched ablation isolating explanations from vocabulary.** Comparing PARA_book+W (34.7 into kgv) vs BOOK_all (34.4 into kgv) directly—where vocabulary coverage is controlled but explanations differ—would substantially strengthen the paper's argument. This comparison exists in Table 2 but is not analyzed in service of the thesis.

- **Error analysis of what grammatical explanations should theoretically help with.** If the test set only rewards vocabulary matching and basic word order, ChrF++ may not detect explanation-driven improvements in morphology or agreement. An analysis mapping grammar book coverage to specific grammatical phenomena would clarify whether the limitation is in the LLM, the metric, or both.

- **Multiple runs with different temperatures/seeds** to estimate variance on the 100-example test set.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "The paper uses Gemini-1.5-Flash rather than Pro, which limits claims."** This is a cost-benefit tradeoff that the authors explicitly acknowledge. Using Flash is standard practice for long-context experiments and Pro's marginal improvements wouldn't change the fundamental finding that parallel data dominates explanations. Downgraded to minor.

- **Harsh critic: "The IGT prediction results (0.7% absolute improvement) are not meaningful without variance estimates."** While true, the paper's core claim is about translation, not IGT prediction. The IGT/TYP results are presented as supplementary evidence that grammar helps on linguistically relevant tasks when properly formatted. The 0.7% is not central to the thesis.

- **Harsh critic: "The educational analogy to worked examples vs. discovery learning is speculative."** The authors clearly label this as tentative ("Our results thus tentatively support a divergence"), and it's presented as a discussion point, not a core claim. Removing it is unnecessary nitpicking.

- **Strength finder: "Addresses methodological limitations of MTOB" and "Well-motivated choice of auxiliary tasks."** These are valid but generic. The specific methodological improvements (combined test set, ChrF++, ablation) are already captured in other strengths.

- **Strength finder: "Statistical evidence that vocabulary coverage, not grammatical content, explains gains (p < 0.005)."** As analyzed above, this regression-based argument has collinearity issues that limit its causal strength. The p-value indicates a correlation, not that coverage is the sole driver. The raw comparison (BOOK_para ≈ BOOK_all when coverage-matched) makes the point more directly.

- **Harsh critic: "BOOK_para+W vs BOOK_all comparison not made."** This is partially addressed—the data is in Table 2—but it's fair that it's not analyzed narratively. Moved to Nice-to-Haves.

## Novel Insights

The most important insight from the reviews is that the paper's own data supports its thesis more cleanly than the paper's own analysis presents it. The coverage-matched comparison (PARA_book+W at 34.7 vs BOOK_all at 34.4 into kgv, and PARA_book+W at 34.7 vs BOOK_all at 34.4 kgv→eng) already shows that adding explanations to coverage-matched parallel data yields zero or negative gains—but the paper relies on a regression argument with collinearity problems instead. Correcting the +0.7 figure and centering the argument on these direct comparisons would strengthen the paper considerably.

## Suggestions

- Correct the +0.7 ChrF++ figure in the abstract and introduction. Based on Table 2, the actual difference between BOOK_all and BOOK_para into kgv is +3.6 for Gemini. If a different comparison is intended, specify it explicitly. This correction is essential as the current figure is the paper's most prominent quantitative claim.

- Replace or supplement the univariate regression argument with direct coverage-matched comparisons already in Table 2. The PARA_book+W vs BOOK_all comparison (vocabulary-controlled ablation) makes the paper's case far more convincingly.

- Add variance estimates (even from 2–3 runs at different temperatures) to support claims about small ChrF++ differences, especially in kgv→eng where differences are 0.3 points.

- Present the fine-tuning vs. LLM comparison in both directions honestly, acknowledging the 5.8 ChrF++ gap in kgv→eng alongside the 0.2 gap in eng→kgv.

## Score and Decision

**Calibration anchors:**
- "Never Train from Scratch" (avg 8.0): Challenges prior claims that SSMs dramatically outperform Transformers, shows simpler pretraining closes the gap. More thorough and statistically sound than this paper, but similar spirit of "simpler methods work." This paper is below it due to the numerical inconsistency and limited statistical analysis.
- "Redefining Bioactivity Prediction" (avg 7.0): Identifies spurious features in prior work, proposes better evaluation. Similar contribution profile (rebutting prior claims with better methodology) but with more thorough dataset and analysis.
- "Hard-to-Beat Baseline" (avg 6.0): Shows simple GDA matches complex adaptation methods for CLIP. Similar paper profile—simple method beats complex one, challenges prior claims. This paper is comparable but has a numerical error in its central claim and weaker statistical analysis.
- ASROB/MTOB extension (avg 5.5): Directly related domain (Kalamang, same grammar book resources), extending MTOB to speech. This paper is stronger than ASROB because it provides a substantive conceptual contribution (the parallel vs. explanations decomposition) rather than just a new benchmark.

This paper makes a genuine contribution by decomposing grammar books into parallel and explanatory content, showing that LLMs exploit parallel examples but not grammatical explanations for translation. The fine-tuning comparison and typological prompting for linguistic tasks further strengthen the practical impact. However, the +0.7 numerical inconsistency in the abstract, collinearity in the regression argument, and lack of variance estimates are real weaknesses that affect confidence in the precise strength of the claims. The overall finding direction is almost certainly correct, but the presentation and statistical rigor need improvement.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>