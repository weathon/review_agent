## Summary

This paper challenges prior claims that long-context LLMs can learn to translate extremely low-resource (XLR) languages from grammar books, showing through careful ablations that nearly all translation gains come from parallel examples embedded in grammar books rather than from grammatical explanations. The authors disentangle grammar books into parallel and non-parallel components across three languages (Kalamang, Nepali, Guarani), demonstrate that fine-tuning small encoder-decoder models on parallel data is competitive with expensive long-context LLMs, and introduce a novel typological feature prompt that achieves leading results on more linguistically-relevant tasks (grammaticality judgment and interlinear gloss prediction).

## Strengths

- **Rigorous and important ablation of grammar book content**: The manual separation of grammar books into BOOK_para and BOOK_non-para directly addresses a critical confound in MTOB. The finding that BOOK_para matches or outperforms BOOK_all while BOOK_non-para lags by 7–8 CHRf++ is clear and compelling.

- **Practical fine-tuning comparison**: Table 4 showing NLLB-1.3B fine-tuned on ~1.2k–1.6k parallel sentences matching or exceeding Gemini+grammar book is an important practical result. It reframes "translation from one book" as a standard XLR MT problem with modest parallel data, which has immediate practical implications.

- **Task-appropriate evaluation extending beyond translation**: Testing grammaticality judgment and IGT prediction reveals a nuanced picture—grammar helps for linguistically-relevant tasks when provided in a useful form (typological features), even if not for translation. This prevents the paper from being purely negative.

- **Novel typological feature prompt**: The TYP prompt achieves 83% on grammaticality judgment (Shuffle) and 46.1% morpheme accuracy on IGT prediction, outperforming supervised baselines. This is a genuine methodological contribution.

- **Multi-language generalization**: Testing across Kalamang (truly unseen) and Nepali/Guarani (seen but low-resource) demonstrates that the core finding is not an artifact of a single language.

## Weaknesses

### Major

- **Statistical analysis overinterprets univariate regression**: Section 5.1 regresses CHRf++ against test-set type coverage and concludes that "the book's grammar explanations provide no significant advantage over its parallel sentences." However, a significant univariate relationship with type coverage does not demonstrate that other factors have *no* effect—it only shows that type coverage explains substantial variance. Grammar explanations could contribute benefits orthogonal to lexical coverage (e.g., word order, agreement) that are undetectable in a univariate model. The proper test would be a multivariate regression including both type coverage and an indicator for grammar explanations. The paper's strongest claims about grammar being useless for translation rest on this inadequate statistical foundation.

- **Scope of conclusions exceeds experimental coverage**: The conclusion states "we find no evidence that long-context LLMs can make effective use of grammatical explanations for XLR translation" and recommends "data collection… is best focused on parallel data over linguistic description." These are broad claims supported by: (a) 2 model families (Gemini-Flash and Llama-3.1-8B, with Llama unable to fit the full kgv book); (b) a single prompting strategy—dumping entire book sections into context; (c) only 1 truly unseen language. The paper itself identifies the needle-in-a-haystack problem as a potential explanation, but does not test retrieval-based or structured approaches that could mitigate it. The "find no evidence" framing is careful, but the prescriptive data-collection recommendation goes beyond what the experiments establish. More defensible is: with current LLMs under naive full-book prompting, parallel data dominates grammar explanations for translation.

- **Small test set for primary translation claims**: The kgv test set contains only 100 examples. The paper combines MTOB's two 50-example directional test sets and uses CHRf++, but even so, differences of 0.5–3 CHRf++ between conditions (e.g., BOOK_all vs BOOK_para into kgv: +0.7; into eng: −0.3) are within what could be noise on 100 examples. No bootstrap confidence intervals or permutation tests are reported for these pairwise differences. The paper criticizes MTOB's tiny test sets but then draws strong conclusions from a set that is only 2× larger.

### Minor

- **Seen vs. unseen language generalization is conflated**: For Nepali and Guarani, which are "seen" by the LLMs, 0-shot baselines are already strong (e.g., 65.2 CHRf++ for npi→eng), leaving little headroom for grammar to help. Conversely, BOOK_non-para sometimes hurts below 0-shot on these languages, which may reflect interference with pretrained knowledge rather than grammar being inherently unhelpful. The paper acknowledges these are "seen low-resource languages" but the conclusion generalizes findings across both types without sufficient caveats.

- **Grammaticality judgment evaluation has limited scope**: The corruption-based design (swap/shuffle) produces relatively simple perturbations, and the paper acknowledges that corrupted sentences may not always be ungrammatical. The 0-shot baseline at 54–57% accuracy is well above chance (50%), suggesting that some prior knowledge or superficial cues are available even without the grammar book. Testing only on kgv (and not on npi/gug with higher expected baselines) also limits generalizability of this positive finding.

- **Inconsistent typological prompt results for translation**: The TYP+BOOK_para prompt does not consistently improve over BOOK_para alone for translation (e.g., it underperforms on gug→eng and npi→eng), while helping more on linguistic tasks. The paper honestly reports this inconsistency, but it means the typological prompting contribution is strongest only for non-translation tasks, which somewhat limits its practical significance given that MT is the primary focus.

## Nice-to-Haves

- Error analysis by linguistic phenomenon (morphology, word order, agreement) comparing BOOK_all vs BOOK_para outputs to determine *why* grammar explanations don't help, rather than just that they don't in aggregate.
- Testing retrieval-based approaches that extract relevant grammar rules per input, rather than dumping entire book sections into context, to address the needle-in-a-haystack issue the paper identifies.
- Bootstrap confidence intervals on CHRf++ scores, especially for the 100-example kgv test set, to clarify which small differences are meaningful.
- Ablation of the typological feature prompt (feature subsets, template variations) to understand what drives its effectiveness.
- Testing with additional long-context models (e.g., GPT-4o, Claude) to ensure findings are not model-specific.

## Removed Points

- **"Only two model families tested"** — Softened and integrated above as a scope limitation rather than treated as a fatal flaw. Testing two model families is reasonable for a focused study; the issue is the breadth of claims, not the number of models per se.
- **"No inter-annotator agreement on grammar book splits"** — This is a nitpick about annotation methodology for a manual filtering task. The splits are based on textual formatting (parallel examples vs. explanatory prose), which is relatively unambiguous in these grammar books.
- **"Grammar book quality varies across languages"** — While true, the paper uses the available grammar books as-is, which is ecologically valid. Controlling for book quality would require synthetic grammar books, going well beyond the paper's scope.
- **"Supervised IGT baselines have limited training data"** — This is an inherent property of the XLR setting, not an unfair comparison. The supervised models are trained on the same data available for the in-context approaches.
- **"Needle-in-a-haystack placement experiments"** — While interesting, this would constitute a separate line of investigation. The paper's scope is evaluating MTOB-style prompting, not optimizing document retrieval.
- **"Overclaim about 'LLMs cannot effectively exploit grammatical explanations'"** — Partially removed; the "find no evidence" phrasing in the paper is actually quite careful. The prescriptive conclusion about data collection priorities is the part that overreaches. Integrated the overclaim concern as a scope issue above.

## Novel Insights

The paper's most insightful finding is the divergence between what helps LLMs vs. what helps humans for language tasks: parallel examples (discovery learning) outperform grammar explanations (worked-example-based instruction) for LLM translation, which inverts the known human educational preference. The typological prompting results then reveal that this is not about LLMs being unable to use grammar at all, but about *form* and *task* alignment—structured typological features help on grammaticality judgment and gloss prediction, tasks where grammatical knowledge is directly relevant, but not on translation where vocabulary coverage dominates. This task-appropriateness principle is the paper's genuine conceptual contribution, and it would be strengthened by centering the narrative around it rather than the more negative "grammar books don't help translation" claim.

## Suggestions

- Reframe the conclusion around the task-appropriateness principle rather than the broad negative claim about grammar explanations. The paper's strongest contribution is not "grammar is useless for MT" but "different tasks need different data formats—parallel data for translation, structured grammatical knowledge for linguistic analysis."
- Add bootstrap confidence intervals for pairwise CHRf++ comparisons on the 100-example kgv test set, or acknowledge the uncertainty around small effect sizes more explicitly.
- If possible, run a multivariate regression (type coverage + grammar explanation indicator) to properly test whether grammar explanations add anything beyond vocabulary coverage, rather than relying on the univariate regression.

## Score and Decision

Calibration references:
- MTOB (tbVWug9f2h.md): scores 6, 8, 8 → spotlight. This paper provides a substantial, systematic critique with novel ablations, extensions, and methods.
- Inductive linguistic reasoning (8XQ1hLbwmU.md): scores 1, 5, 5, 6 → rejected. Had disconnect between claims and evidence; the current paper's evidence is much stronger.
- IntGrad MT (SmxM4POTBk.md): scores 6, 3, 8, 6, 3 → rejected. Had weak baselines; this paper has stronger baselines including fine-tuned NLLB.
- ASROB (sjvz40tazX.md): scores 8, 3, 5, 6 → rejected. Also extended MTOB with limited scope.

This paper makes a genuine, timely contribution by dissecting what actually drives MTOB-style translation performance and introducing practical alternatives (fine-tuning, typological prompting). The core experimental finding is robust. Its main weaknesses are overclaiming relative to the experimental scope and an inadequately supported statistical argument. Relative to MTOB (which was accepted as spotlight with similar or less methodological rigor), this paper's contribution is a complementary meta-scientific critique that the community needs. It is not at the same level of novelty as MTOB (which introduced the benchmark), but its empirical work is more thorough.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>