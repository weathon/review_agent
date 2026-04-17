Now I have a thorough understanding of the paper, reviewer inputs, and calibration data. Let me now write the final consolidated review.

## Summary

This paper re-examines the claim from MTOB (Tanzer et al., 2024) that long-context LLMs can learn to translate an extremely low-resource language (Kalamang) from a grammar book in-context. Through careful ablations that disentangle grammar books into parallel examples (BOOK_para) and explanatory text (BOOK_non-para), the authors find that nearly all translation improvements come from the parallel examples rather than grammatical explanations—a finding that generalises to Nepali and Guarani. They show that fine-tuning a small NMT model (NLLB-1.3B) on the same parallel data matches or exceeds long-context LLMs, and introduce a novel typological feature prompt that substantially improves performance on grammaticality judgment and interlinear gloss prediction, demonstrating that grammatical knowledge can help LLMs when provided in an appropriate form for relevant tasks.

## Strengths

- **Rigorous and novel ablation of grammar book components.** The paper performs a crucial disentanglement that MTOB left open—separating grammar books into parallel examples and explanatory text—and demonstrates consistently across three languages that BOOK_para matches or outperforms BOOK_all, while BOOK_non-para lags significantly (Tables 2, 3). This directly addresses a gap flagged by MTOB's own reviewers.

- **Statistical analysis linking performance to vocabulary coverage.** The regression analysis in Section 5.1 (Figure 2) showing translation performance is statistically modelled by test set type coverage, with all settings within the 95% CI, provides a quantitative foundation for the claim that grammatical explanations add no significant advantage beyond increased vocabulary (p < 0.005, F-test). This is a clean, well-motivated piece of analysis.

- **Practical and important finding that fine-tuned NMT is competitive.** Table 4 shows NLLB-1.3B fine-tuned on PARA_book achieves 34.2 ChrF++ into kgv compared to Gemini's 26.6 with the same data, directly answering a question raised by MTOB reviewers about how this approach stacks up against standard MT. This highlights computational and token efficiency advantages of smaller specialist models.

- **Constructive pivot to linguistic tasks where grammar helps.** The typological feature prompt achieves leading results on grammaticality judgment (83% on SHUFFLE, Figure 1) and IGT prediction (46.1% morpheme accuracy, Table 5), beating supervised baselines. This adds important nuance—grammar is not useless, but needs task-appropriate formulation.

- **Generalisation beyond Kalamang to Nepali and Guarani.** Testing on seen low-resource languages with established FLORES test sets (Table 3) demonstrates the pattern is not specific to one language, strengthening the findings. The consistent finding that BOOK_non-para actively hurts performance for npi/gug (below 0-shot) is a notable result.

## Weaknesses

### Major:

- **Over-interpretation of narrow empirical evidence as a general recommendation for XLR data collection.** The paper's headline conclusion—"data collection for multilingual XLR tasks such as translation is best focused on parallel data over linguistic description"—is substantially stronger than the evidence warrants. All core experiments assume the availability of 1k+ parallel examples (BOOK_para for kgv = 1,239 examples; FLORES dev sets for npi/gug). The regime that would genuinely test the recommendation—where *only* descriptive material exists and parallel data is scarce or non-existent—is never evaluated. The paper does not test with, e.g., 10–50 parallel examples or a truly description-only setting. This matters because the recommendation to "prioritise parallel data over linguistic description" is most consequential exactly where the paper has no evidence: for languages where description already exists but parallel data does not. The conclusion should be scoped to the tested regime (~1k+ parallel examples available), or the paper should test the very-low-parallel-data regime where grammar-only might still help.

- **The BOOK_para vs BOOK_non-para split is an imperfect proxy for "grammar explanations vs parallel examples."** BOOK_para is defined as "parallel glossed examples and bilingual word/phrase pairs" extracted via formatting, while BOOK_non-para is "the remainder" including all prose—syntactic descriptions, morphological paradigms, lexical notes, metalinguistic discussion, and potentially unaligned example fragments. This is not a clean separation into "rules vs examples"; it is "aligned bilingual signal vs everything else." The key negative result (BOOK_non-para underperforms) could therefore be a "noise vs aligned signal" finding rather than definitively "grammar explanations don't help." The paper would be stronger with at least some targeted probes on explicitly rule-like content (e.g., paradigm tables, overt correspondence rules) rather than the formatting-based residual. The paper partially acknowledges this in discussion but still phrases conclusions as about "grammatical explanations" specifically.

- **Small Kalamang test set (100 sentences) limits confidence in fine-grained comparisons.** The combined test set of only 100 examples limits statistical power, particularly for the small ChrF++ differences the paper discusses (e.g., +0.7 for BOOK_all over BOOK_para into kgv, differences of a few points in some settings). The authors themselves criticised MTOB's 50-example sets as "potentially too small for making wider generalisations," but combining to 100 is still a modest improvement. No bootstrap confidence intervals or variance estimates are reported, making claims about "no significant advantage" difficult to verify purely from the reported numbers. The regression in §5.1 is run on fewer than 15 data points.

### Minor:

- **Inconsistent benefits of typological prompting for translation.** The typological prompt shows mixed results for translation: it helps kgv→eng but not eng→kgv (Table 2), and generally underperforms BOOK_para for npi/gug (Table 3). The paper does not deeply investigate why the direction matters, which limits understanding of when and how grammatical information is beneficial.

- **Grammaticality judgment task may primarily test surface word-order sensitivity.** The corruption schemes (swap adjacent words, swap random words, shuffle) primarily test word-order awareness. For a language with rich morphology like Kalamang, many crucial grammatical distinctions (agreement, case marking, verb morphology) may not be captured by such perturbations. The paper acknowledges that corrupted sentences cannot be guaranteed ungrammatical, but does not verify even partially that the corruptions violate known Kalamang word-order features from the grammar.

- **npi/gug results should be framed more carefully.** For these seen low-resource languages, 0-shot Gemini scores are already high (e.g., 65.2 ChrF++ for npi→eng), so effect sizes are inherently small. The paper acknowledges they are "not unseen" but treats them as substantial evidence backing the main conclusion. The fact that BOOK_non-para actively *hurts* npi/gug performance (below 0-shot) but not kgv suggests different dynamics for seen vs unseen languages, which is not deeply analysed.

- **Limited analysis of why grammatical explanations fail for translation.** The paper shows *that* they fail but does not deeply investigate *why*—whether this is due to the needle-in-a-haystack retrieval problem, the descriptive (non-didactic) format of grammar books, LLMs' inability to reason from rules, or some combination. The discussion touches on this but remains speculative.

### Trivial:

- The claim about "matching" between NLLB and Gemini could benefit from variance reporting; without confidence intervals, small ChrF++ differences may be within noise.

## Nice-to-Haves

- Testing the regime with very few parallel examples (10–50) vs grammar-only, which would directly test the paper's most consequential recommendation.
- Ablating the typological prompt by feature category (word-order vs morphological features) to understand what kind of grammatical knowledge helps.
- RAG-style experiments that retrieve relevant grammar sections per input sentence, to distinguish retrieval failure from reasoning failure.
- Bootstrap confidence intervals for Kalamang results given the small test set.
- Error analysis by linguistic phenomenon (e.g., word order, agreement, morphology) to test whether grammar helps with specific phenomena even if not in aggregate.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Reproducibility concerns with Gemini as API-only model.** The neutral reviewer flagged that Gemini is API-only with potential version changes. However, the paper does test with the open-weight Llama models as well, and API-based models are standard in the field. The paper's core ablation patterns (BOOK_para vs BOOK_non-para) are consistent across both Gemini and Llama where testable. This is not a meaningful reproducibility concern beyond what is standard.

- **Unfair computational cost comparison between API LLMs and fine-tuned models.** The human finder raised that the paper does not quantify costs. However, the paper's point is precisely that small fine-tuned models are more practical—this is a favourable asymmetric comparison that strengthens the paper's argument about simpler alternatives being sufficient. Requesting detailed cost analysis goes beyond the paper's scope.

- **NLLB-1.3B matching Gemini is "expected" because it is much smaller.** The spark reviewer suggested this comparison is not surprising given the parameter difference. But this misses the point: the surprise is that a small specialist model fine-tuned on *the same data* matches a vastly larger general-purpose LLM with a 1M-token context. This was an open question from MTOB.

- **Missing comparison to transfer learning from related languages, data augmentation, etc.** The human finder flagged that MTOB reviewers asked about "standard neural MT, trained on the little parallel data that's available + parallel data from related languages; traditional rule-based MT." This paper's contribution is a focused ablation study, not a comprehensive MT system comparison. Adding all possible XLR MT techniques is scope creep beyond the paper's stated research question about the source of translation ability in grammar books.

- **Missing confidence intervals/variance across runs.** While desirable, single-run evaluation without variance reporting is common in the field for API-based models. The paper already improves on MTOB by combining test sets and using ChrF++. This is a nice-to-have, not a core flaw.

## Novel Insights

The paper reveals an important asymmetry in how LLMs exploit linguistic information: the same grammatical knowledge that is essentially useless for translation (as measured by ChrF++) becomes highly effective when provided as typological features for tasks aligned with structural analysis (grammaticality judgment, gloss prediction). This suggests the limitation is not that LLMs *cannot* use grammar, but that translation quality as measured by surface-level metrics is primarily bottlenecked by lexical coverage rather than grammatical competence. This reframes the problem from "LLMs cannot exploit grammar" to "translation evaluation metrics and the translation task itself may not be the right probe for grammatical knowledge acquisition," which is a more nuanced and productive framing.

## Suggestions

- **Scope the headline conclusion more carefully.** Change the abstract/conclusion to say "given the availability of ~1k parallel examples" or "in the tested regimes" rather than making an unqualified recommendation about XLR data collection priorities. If possible, add a small experiment with very few parallel examples to test whether grammar helps in that regime.

- **Add bootstrap confidence intervals to Kalamang results** to strengthen claims about "no significant advantage" from grammar explanations, especially given the 100-sentence test set.

- **Better characterise the content of BOOK_non-para** (e.g., proportion of paradigm tables, rule-like statements, metalinguistic prose) to justify interpreting it as "grammatical explanations."

## Score and Decision

**Calibration comparisons:**

- **MTOB (tbVWug9f2h)** — scores 6, 8, 8, accepted as spotlight. This is the paper being directly critiqued. It introduced a novel benchmark with interesting framing but left the grammar-vs-examples question unaddressed. This paper provides the key missing ablation, directly answers reviewer questions, and adds the typological prompt contribution.

- **SCALE (yisfNWUEsD)** — scores 5, 6, 6, rejected. A low-resource MT paper with promising empirical results but questioned for limited evaluation scope and weak baselines. This paper has more rigorous ablations and a clearer research question, but similarly overclaims.

- **IntGrad MT (SmxM4POTBk)** — scores 6, 3, 8, 6, 3, rejected. A low-resource LLM-MT paper with methodological issues in baselines and evaluation. This paper is more methodologically sound with better baselines and more rigorous analysis.

- **ALMA (farT6XXntP)** — scores 8, 5, 6, 8, accepted as poster. A fine-tuning approach for LLM translation. This paper is somewhat more novel in its research question (challenging an existing claim) but less novel in methodology (the typological prompt is a moderate contribution).

- **Inductive Linguistic Reasoning (8XQ1hLbwmU)** — scores 1, 5, 5, 6, rejected. A linguistic reasoning paper with deep issues in interpreting what models learn. This paper avoids those errors by being more careful in interpretation, though it still overclaims.

This paper makes a genuine, important contribution by conducting the crucial ablation that MTOB left open and by showing NMT baselines are competitive—directly resolving questions raised by MTOB reviewers. The typological prompt for linguistic tasks is a nice secondary contribution. The main weakness is the overreach in conclusions, but the core empirical findings are solid and clearly presented. The paper is stronger than the rejected papers above (SCALE, IntGrad MT) in terms of experimental rigor and research question clarity, but not as strong as the top-accepted papers due to the overclaiming and small test set. It sits in the solid-but-not-exceptional range, comparable to papers scoring around 5.5-6.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>