Now let me run calibration searches.## Summary
This paper investigates whether LLMs can effectively learn to translate an extremely low-resource language (Kalamang, as well as Nepali and Guarani) from a grammar book, specifically challenging the MTOB benchmark's claim. Through careful ablation, the authors decompose grammar books into parallel examples (BOOK_para) and explanatory text (BOOK_non-para), finding that almost all translation benefit comes from the parallel examples, not grammatical explanations. They also introduce a novel typological feature prompt that achieves leading results on grammaticality judgment and IGT prediction — showing that grammar CAN help LLMs, but only for linguistically appropriate tasks and in the right form.

## Strengths
- **Novel and directly motivated ablation** (Tables 2 & 3): Splitting grammar books into BOOK_para vs. BOOK_non-para is the right experiment to run to settle MTOB's claim. BOOK_non-para underperforms BOOK_para by 7–8 CHRf++ for kgv and causes up to 7-point degradation below 0-shot for npi/gug, while BOOK_para equals or beats BOOK_all. This is a clean, reproducible finding across three languages.

- **Practical fine-tuning result** (Table 4): NLLB-1.3B fine-tuned on PARA_book matches Gemini with the full grammar book (34.2 vs 34.4 CHRf++ eng→kgv), demonstrating that the MTOB challenge is adequately solved by standard MT methods applied to the book's parallel data.

- **Typological prompting** (Figure 1, Table 5): TYP+BOOK_para achieves 83% accuracy on the hardest grammaticality judgment setting (vs. 80% for BOOK_para alone) and 46.1% morpheme accuracy on IGT prediction (beating all supervised baselines). This is a novel, actionable contribution showing LLMs can exploit grammar in the right form for linguistically relevant tasks.

- **Multi-language generalization** (Table 3): The parallel-over-grammar finding generalizes to npi and gug with 1012-sentence FLORES test sets, making the result substantially more robust than a single-language study.

- **Token efficiency analysis**: BOOK_all uses ~5× more tokens than BOOK_para for equal or worse performance, providing concrete practical guidance for resource-constrained XLR NLP. The needle-in-a-haystack hypothesis (harder to retrieve relevant parallel examples in longer context) is a well-motivated, if untested, explanation.

- **Methodological improvements over MTOB**: The paper clearly identifies and corrects several MTOB shortcomings — combining directional test sets to 100 examples, switching to CHRf++ (which accounts for word order), and performing controlled ablations MTOB lacked.

## Weaknesses

### Fatal
None.

### Major
None.

### Minor

- **The vocabulary-coverage regression overstates its statistical conclusion** (Section 5.1, Figure 2): The regression plots CHRf++ against test-set type coverage across ~15 data points and interprets "all settings fall within the 95% CI" as positive evidence that grammar adds nothing beyond vocabulary. This conflates low statistical power with evidence for the null hypothesis — falling within a wide CI does not confirm the null. More critically, the paper never asks whether BOOK_non-para and BOOK_all leave *systematically different residuals* from the regression line compared to BOOK_para; that would be the decisive test. The central finding — grammar adds no advantage over parallel data — is more convincingly supported by the direct pairwise comparisons in Tables 2 and 3 than by this regression, and the paper's rhetorical emphasis on the regression is somewhat misleading about what it actually proves.

- **No confidence intervals on primary CHRf++ comparisons** (Tables 2, 3): The kgv test set has 100 sentences, yet no bootstrap confidence intervals or significance tests accompany the key BOOK_all vs. BOOK_para differences (e.g., 34.4 vs. 34.7 CHRf++ kgv→eng for Gemini). Several highlighted differences are small (0.3–3.6 points) and could plausibly fall within sampling noise. The npi/gug results with 1012 FLORES examples are on firmer ground, but the kgv-specific claims need uncertainty quantification.

- **Grammaticality judgment corruption validity** (Section 3.4): The authors correctly acknowledge "we cannot guarantee all corruptions are ungrammatical." SWAP_adj performance hovers at 56–65% — barely above chance — which is consistent with some swapped-adjacent-word sentences being grammatically acceptable in Kalamang's morphologically rich system. Since the grammaticality judgment results carry weight for the claim that typological prompting helps LLMs for grammar-relevant tasks, the near-chance SWAP_adj scores are a real limitation that is acknowledged but unresolved.

- **Gemini Flash vs. Pro** (Section 4.2): MTOB used Gemini Pro/Ultra, while this paper uses the less capable Flash for cost reasons. Direct numerical comparisons to MTOB figures are therefore not strictly like-for-like. The paper implicitly acknowledges this but it warrants an explicit caveat: if a stronger model better exploits grammar explanations, the conclusions could change.

### Trivial
- The abstract claim that typological prompting "achieves leading results on these more relevant tasks" is technically accurate for morpheme accuracy on IGT (46.1% vs. 45.2%) but a margin of 0.9% over the prior supervised best is a razor-thin lead; more hedged phrasing would be appropriate.

## Nice-to-Haves
- Bootstrap confidence intervals on all CHRf++ pairwise differences in Tables 2–3 (especially kgv) would resolve the significance concern without re-running experiments.
- An oracle experiment for the needle-in-a-haystack hypothesis: prompt with BOOK_all but highlight relevant parallel examples to test whether the BOOK_all ≈ BOOK_para gap is noise vs. signal-to-noise ratio.
- Error analysis for IGT prediction: characterizing *where* TYP+BOOK_p helps (new grammatical morphemes?) vs. hurts (word segmentation?) would illuminate *why* typology helps here.
- Qualitative examples contrasting BOOK_all and BOOK_para translations to make the failure mode of grammar explanations vivid.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Strength Finder's regression strength**: Retained in main review but weakened from "strong quantitative evidence" to a supporting point, because the regression interpretation has the validity concern described above.

- **Harsh Critic's "NLLB comparison framing"**: The critic notes NLLB has massively multilingual pretraining providing cross-lingual transfer. While true, this does not undermine the paper's practical point that the same parallel data + specialist model matches expensive long-context LLMs. This is not a weakness of the paper's argument.

- **Harsh Critic's abstract overclaim about "leading results"**: Borderline but the abstract claims are qualified ("on these more relevant tasks"), so this is more a precision concern than a real overclaim. Moved to Trivial.

- **Harsh Critic's demand for a stronger/more recent Gemini**: Valid as a limitation (model-specific null result), but already partly addressed in discussion. Testing larger models is a legitimate next step, not a paper flaw.

- **Harsh Critic's critique of the MTOB human baseline discussion**: The observation that the human baseline "may learn from examples at test time" is actually a legitimate methodological critique of prior work. Removed from weaknesses since the current paper doesn't repeat this flaw.

## Novel Insights
The paper's most genuinely novel contribution is the architectural insight that grammar books are functionally heterogeneous: their parallel examples constitute only ~18% of total tokens but carry nearly all translation-relevant signal, while the descriptive text — despite being the primary purpose of a grammar book — is either ignored by LLMs or actively harmful in translation contexts. The complementary finding that typological features (a highly compressed, cross-linguistically standardized summary of grammar) *do* help LLMs for linguistically-oriented tasks suggests that the failure is not about grammar per se, but about the mismatch between unstructured descriptive text and the format of information LLMs can exploit in-context. This points toward a useful design principle: for XLR NLP, the value of linguistic resources is primarily their parallel examples for translation and their structured, typology-level representations for grammatical tasks — not the prose explanations that constitute most of a grammar book's pages.

## Suggestions
- Add bootstrap confidence intervals (even in a footnote or appendix) for the kgv CHRf++ pairwise differences.
- Reframe the vocabulary regression conclusion more carefully: "grammar conditions do not systematically deviate above the regression line established by vocabulary coverage" is defensible; "grammar adds no significant advantage" requires the residual analysis.
- Clarify that the SWAP_adj near-chance results are likely partly a validity artifact (some corruptions may be grammatical in Kalamang), not purely a failure of LLMs to acquire grammar.
- Add one sentence explicitly noting that comparisons to MTOB figures use a weaker model (Flash vs. Pro) and that this limits direct numerical comparison.

---

## Score and Decision

**Calibration anchors used:**
- *ASROB* (MTOB speech extension, avg 5.5, Rejected): A weaker paper that merely extends the MTOB benchmark to speech without deep ablation or novel methodology. The current paper is substantially stronger on every dimension.
- *3KDbIWT26J* ("Reasonableness of LLM Translation," avg 5.75, Accepted poster): Similar in type — empirical investigation of LLM translation mechanisms — but narrower in scope and contribution. The current paper has richer experiments and a clearer, more actionable finding.
- *farT6XXntP* ("ALMA paradigm shift in MT," avg 6.75, Accepted poster): Stronger on practical impact and novel training methodology for high-resource directions; the current paper is more focused on a diagnostic research question with XLR languages but has comparable experimental rigor.
- Low-scoring anchors (*UnstiBOfnv*, *aJ581rzuE7*, avg ~3.67): Substantially weaker — narrow scope, limited experiments, less clear contributions. The current paper clearly exceeds this tier.

**Reasoning:** The paper sits between the 3KDbIWT26J anchor (5.75, accepted) and the farT6XXntP anchor (6.75, accepted). It has genuine, multi-language empirical contributions, a focused research question directly challenging an established benchmark, and introduces a novel typological prompting method. The weaknesses are real (regression overstatement, missing CIs on small kgv test set, grammaticality validity) but do not invalidate the main findings — the Tables 2 and 3 comparisons are convincing even without formal tests, particularly for npi/gug with 1012 FLORES examples. I place this at **6.0**, reflecting a solid accepted-poster contribution with identifiable but addressable statistical gaps.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>