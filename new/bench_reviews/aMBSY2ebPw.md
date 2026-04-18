## Summary

This paper critically examines whether LLMs can exploit grammatical explanations in grammar books for extremely low-resource (XLR) machine translation, challenging claims from prior work (Tanzer et al., 2024/MTOB). Through careful ablations separating parallel examples from explanatory text in grammar books, the authors find that nearly all translation improvements stem from parallel examples rather than grammatical explanations. They show that fine-tuning a small NLLB model on parallel data matches or beats expensive long-context LLM prompting with full grammar books. They introduce a typological feature prompt that achieves leading results on grammaticality judgment and IGT prediction—tasks more aligned with grammar books—arguing for task-appropriate data: parallel data for translation, grammatical data for linguistic tasks.

## Strengths

- **Clean ablation methodology disentangling parallel data from grammatical explanations.** The manual splitting of grammar books into BOOK_para and BOOK_non-para is a real methodological contribution. Finding that BOOK_para matches or outperforms BOOK_all while BOOK_non-para substantially underperforms provides concrete evidence against MTOB's implicit claim that the grammar explanations themselves drive translation improvements.

- **Practical demonstration that fine-tuning matches long-context LLMs.** Showing that NLLB-1.3B fine-tuned for one hour on parallel data achieves 34.2 vs. 26.6 chrF++ (eng→kgv) compared to Gemini with the grammar book in-context is practically significant for XLR practitioners with limited compute budgets.

- **Thoughtful task selection beyond translation.** Testing grammaticality judgment and IGT prediction is well-motivated and provides a more nuanced picture. The finding that typological features help these grammar-aligned tasks but not translation supports the "task-appropriate data" thesis coherently.

- **Generalization beyond a single language.** Testing on Kalamang (truly unseen), Nepali, and Guarani (seen but low-resource) broadens the validity of findings, though with important caveats (see weaknesses).

## Weaknesses

### Fatal
None.

### Major

- **Overclaiming from limited experimental scope to strong prescriptive conclusions.** The paper's central claim—"we find no evidence that long-context LLMs can make effective use of grammatical explanations for XLR translation"—and its recommendation that "data collection for XLR MT is best focused on parallel data over linguistic description" extend well beyond what the experiments demonstrate. The experiments show that dumping raw, uncurated grammar text (81k+ tokens for kgv) into a prompt alongside parallel examples provides negligible benefit over the parallel examples alone. They do not show that grammatical knowledge is useless for MT in general—only that one particular, naive way of presenting it (the entire book in context) doesn't help beyond vocabulary coverage. The paper itself acknowledges the needle-in-a-haystack problem (§5.1) but doesn't test any retrieval-augmented or curated approach to presenting grammar, such as short rule summaries or retrieval of relevant passages. A more honest conclusion would be: "in the current prompting paradigm, the marginal benefit of raw explanatory text over parallel examples for MT is negligible." The prescriptive recommendation about field data collection priorities is not adequately supported.

- **Small and noisy evaluation for the primary language.** The combined kgv test set contains only 100 examples. No confidence intervals, bootstrap tests, or statistical significance tests are reported for the chrF++ scores, even though key contrasts involve small differences (e.g., +0.7 chrF++ for adding explanations to parallel data into kgv, or 0.2–0.5 point differences for typological prompting). The original MTOB reviewers flagged this same concern. With 100 test sentences, even moderate score differences may not be statistically reliable, which undermines confidence in the regression analysis (§5.1) claiming that type coverage "directly models" translation performance. The regression itself has only ~10 prompt conditions per direction—far too few data points to support causal inference about whether grammatical explanations have no independent effect.

- **The type coverage regression is too weak to support causal claims about grammar being unnecessary.** The authors fit a univariate linear regression of chrF++ against test-set type coverage with ~10 data points per direction, finding all conditions fall within the 95% CI. This establishes correlation, not causation. Type coverage is confounded with quantity of parallel data—all high-coverage conditions also contain many parallel examples. There is no control for alternative explanations (e.g., phrase-level or construction-level overlap in parallel examples, which differs from having the same vocabulary in explanatory text). Additionally, no R², residual analyses, or robustness checks are reported. The correct reading is "vocabulary coverage is a strong predictor in our settings," not "grammar explanations provide no additional value."

### Minor

- **Limited model diversity for long-context experiments.** The primary kgv results rely on Gemini-1.5-Flash as the only model that can fit the full grammar book in context. Llama-3.1-8B cannot handle kgv or gug BOOK_all due to context limits, so the core claim about grammatical explanations helping vs. not helping rests primarily on one proprietary model.

- **BOOK_all sometimes underperforms BOOK_para without explanation.** For npi and gug, BOOK_all actively hurts compared to 0-shot (Table 3). This is mentioned but not analyzed. Whether this reflects retrieval failure, active interference from the grammar text, or context-length degradation matters for interpreting the results, particularly since the paper itself raises the needle-in-a-haystack concern.

- **No error analysis by linguistic phenomenon.** The paper establishes that grammatical explanations don't help aggregate chrF++ scores, but doesn't investigate whether they help for specific linguistic phenomena (morphological agreement, word order changes, case marking) that chrF++ may average out. Such analysis could reveal whether grammatical knowledge helps selectively even when the aggregate effect is negligible.

### Trivial

- The grammaticality judgment corruptions (word swaps, shuffles) are relatively crude and may not test comprehension of specific grammatical rules described in the book, though the task is well-motivated as a first step.

## Nice-to-Haves

- Testing additional long-context models (e.g., Claude, GPT-4o) for the BOOK_all condition to reduce reliance on a single model.
- A noise/context-length control experiment: adding random or non-grammatical text of similar length to BOOK_para, to isolate whether the small advantage of BOOK_all over BOOK_para is from grammar explanations or simply longer context.
- Reporting bootstrap confidence intervals for key score differences, especially on the 100-sentence kgv test set.
- Curated grammar summaries (short rule cheat-sheets of 10–20 key typological facts) as an intermediate condition between raw book text and typological prompts, to better test whether structured grammar helps MT.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"BOOK_non-para is used adversarially" (Harsh Critic #1)**: While the experimental design uses BOOK_non-para as an unstructured mass of text, this is precisely the use case MTOB proposed—feeding an entire grammar book to an LLM. The paper is testing the actual claim from prior work, not constructing a strawman. However, the *generalization* from "raw grammar book text doesn't help" to "grammatical explanations cannot help" is overbroad, which is captured in the Major weakness above.

- **"Typological prompting undercuts the anti-grammar conclusion" (Harsh Critic #4)**: The paper does NOT say grammar is useless in general—it specifically says grammatical/typological data helps for linguistic tasks (grammaticality judgment, IGT). The recommendation to prioritize parallel data applies specifically to MT. This is internally consistent; the apparent tension is a framing nuance, not a contradiction.

- **"NLLB fine-tuning comparison is not apples-to-apples" (Neutral Reviewer #3)**: The paper's comparison is explicit about this and is making a practical point: given the same data, a cheap fine-tuned model matches or beats an expensive LLM. This is a valid practical comparison, not a controlled experiment isolating data format from learning paradigm.

- **"Languages and grammars not representative of typical XLR conditions" (Harsh Critic #2, subpoint 3)**: The paper tests three languages spanning the spectrum from truly unseen (kgv) to seen-but-low-resource (npi, gug), which is more than most XLR work. All XLR work has limited language coverage; criticizing this is generic.

- **"Demand for human evaluation" (Harsh Critic #2, subpoint 4)**: The paper explains that engaging proficient Kalamang speakers is infeasible, and includes a small qualitative analysis in the appendix. Human evaluation for XLR languages is often impractical; chrF++ is used appropriately.

- **"Fine-tuning on grammar book text" (Spark #2)**: Fine-tuning an MT model on explanatory grammar text is not a standard or well-motivated baseline; this is outside the paper's scope.

- **"No variance/significance reported" (several reviewers)**: While this would strengthen the paper, for XLR settings with limited evaluation data, single-run reporting with chrF++ is the community norm (as used in MTOB and most comparable work). This is noted as a nice-to-have but not a substantive weakness.

## Novel Insights

The paper's most important insight is the empirical demonstration that the translation performance attributed to grammar books in MTOB is almost entirely driven by the parallel examples they contain, not the grammatical explanations. The type coverage regression provides a mechanistic explanation: what matters most is having target-language vocabulary in context, not grammatical rules. This reframes the practical question from "how do we get LLMs to understand grammar books?" to "what is the most token-efficient way to present relevant vocabulary and examples?"—a question with a clear answer (parallel data). The secondary insight that typological features help grammar-aligned tasks but not translation itself is a genuine contribution to understanding how linguistic knowledge interacts with task type in LLMs.

## Suggestions

- Temper the main conclusion to reflect what the experiments actually show: raw grammar book text provides negligible marginal benefit over parallel examples for MT in the tested prompting paradigm. Avoid claiming that grammatical explanations are useless for MT in general.
- Report bootstrap confidence intervals for the 100-sentence kgv test set, which would clarify which score differences are statistically meaningful.
- Add a short control experiment (even informal) with curated grammar summaries or retrieval-based grammar selection, to address the most obvious counterargument that structured grammar presentation might help MT.
- Analyze at least a few translation examples to show what kinds of errors are made with parallel data vs. grammar explanations, moving beyond aggregate metrics.

## Score and Decision

Calibration comparisons:
- **MTOB** (tbVWug9f2h.md): scored 6, 8, 8 (accept spotlight). The current paper provides an important corrective to MTOB with cleaner ablations but has weaker evaluation scope (still only 100 kgv test sentences) and overclaims.
- **ASROB** (sjvz40tazX.md): scored 8, 3, 5, 6 (reject). Similar scope limitations (one language, one LLM family) but the current paper has a much cleaner experimental design and clearer contribution.
- **IntGrad MT** (SmxM4POTBk.md): scored 6, 3, 8, 6, 3 (reject). The current paper has stronger methodology and a clearer message.
- **Democratizing LLMs** (Nfu3bUkmdH.md): scored 5, 6, 6 (reject). Similar XLR focus but weaker experimental design.

The paper makes a genuine and important contribution by showing that MTOB's claimed translation gains from grammar books are largely attributable to parallel examples. The fine-tuning comparison and typological prompt for IGT are valuable additions. However, the overclaims (from "raw grammar text doesn't help" to "grammatical explanations are ineffective for XLR MT" to "prioritize parallel data over linguistic description") and the small/underpowered evaluation for kgv are substantive weaknesses that limit confidence in the strongest conclusions. The paper is borderline but leans toward accept because the core finding is robust and important for the field.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>