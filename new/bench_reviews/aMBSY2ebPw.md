## Summary

The paper revisits the “Machine Translation from One Book” (MTOB) setting for Kalamang and related scenarios. It splits grammar books into parallel examples vs. non-parallel explanatory text, shows that almost all MT gains come from the parallel parts rather than explanations, and that fine-tuning a small MT model on the same parallel data matches or beats long-context LLM prompting. It then introduces a typological feature prompt and demonstrates that grammar-related signals (especially typology plus glossed examples) help more linguistically oriented tasks—grammaticality judgment and interlinear gloss prediction—than they help translation.

## Strengths

- **Well-defined and timely research question.** The paper directly interrogates a prominent claim that LLMs can “learn to translate from one grammar book,” an important question for XLR NLP. It focuses explicitly on what signal grammar books actually provide and where that signal matters.

- **Careful ablation of grammar book content.** The manual split into BOOK\_{para} (parallel/glossed examples and word/phrase pairs) vs. BOOK\_{non-para} (explanatory text with no parallel examples) is central and well integrated throughout the experiments (Sec. 3.1, 4.1). Across multiple models, languages, and tasks, BOOK\_{para} consistently outperforms BOOK\_{non-para}, and often matches or exceeds BOOK\_{all}. For example, in Table 2 with Gemini, for eng→kgv, BOOK\_{para} scores 30.8 vs. 22.6 for BOOK\_{non-para}; for kgv→eng, 34.7 vs. 27.5.

- **Solid, practical MT baseline comparison.** Fine-tuning NLLB-1.3B on the same Kalamang parallel data (PARA\_{book}, with/without PARA\_{train} and backtranslation) is a strong baseline that many prior “grammar-book LLM” works omit. Table 4 shows NLLB fine-tuned on PARA\_{book} achieves 34.2 CHRF++ for eng→kgv, substantially higher than Gemini’s 26.6 with PARA\_{book} in-context, and comparable or better when adding PARA\_{train}. This is a valuable, concrete result: simple MT fine-tuning can match or beat grammar-book-prompted LLMs for this benchmark.

- **Useful multi-language extension.** The study covers three languages: Kalamang (unseen by pretraining), plus Nepali and Guaraní (seen low-resource) evaluated on FLORES devtest (1012 examples). While these do not cover the full space of XLR languages, they move beyond a one-language case study and show the same basic trend: BOOK\_{para} is neutral or mildly positive, while BOOK\_{non-para} often hurts performance relative to 0-shot (Table 3).

- **Insightful token/type-coverage analysis.** Section 5.1 fits regressions of CHRF++ vs. test-set type coverage for several prompt settings and shows strong, significant positive correlations (p < 0.005). All observed MT conditions lie within the 95% confidence intervals. Combined with a negative result for token-count vs. CHRF++, this strongly supports the claim that vocabulary coverage, not sheer prompt length, drives most of the MT gains.

- **Reframing to task-appropriate uses of grammar.** Rather than declaring grammar “useless,” the paper pivots to tasks more naturally aligned with grammatical description—grammaticality judgment (Sec. 3.4; Fig. 1) and IGT prediction (Sec. 3.5; Table 5). It shows that typological prompting plus BOOK\_{para} improves Kalamang grammaticality judgment over BOOK\_{para} alone by up to 3% and yields the highest morpheme accuracy for IGT (46.1%), surpassing several supervised baselines.

- **Novel typological feature prompting.** The construction of a language-invariant typological prompt from Grambank (Sec. 3.3) is an interesting idea, giving a principled, compact way to encode cross-lingual structural information. Its gains on grammaticality judgment and IGT indicate that properly packaged grammatical knowledge can be exploited by LLMs for some tasks.

- **Clear writing and positioning.** The narrative is easy to follow, related work is fairly and specifically discussed, and the authors explicitly connect their methodological choices to perceived shortcomings in MTOB. The conclusions about token efficiency, “needle in a haystack” effects, and the distinction between MT vs. linguistic tasks are well argued qualitatively.

## Weaknesses

### Fatal

None. The experimental work is sound overall; the main issues concern overgeneralisation and some over-strong causal language, not a fundamental flaw that invalidates all results.

### Major

- **(1) Kalamang MT evaluation set is very small for the strength of the claims.**

  The central unseen-language claims hinge on a 100-sentence Kalamang test set created by combining the two 50-example directional sets (Sec. 3.1, 4.1). This is still very small, especially given:

  - The training/prompt universe is tiny and tightly interrelated: PARA\_{book}, PARA\_{train}, WORDLIST and Dictionaria all derive from a few documentation resources (Sec. 4.1). The paper does not report any check for exact or near-duplicate overlap between these and the 100 test examples.
  - Many key Kalamang comparisons differ by only a few CHRF++ points. For instance, Table 2 (Gemini, eng→kgv): BOOK\_{all} = 34.4 vs. BOOK\_{para} = 30.8 (3.6 points), and for kgv→eng: BOOK\_{all} = 34.4 vs. BOOK\_{para} = 34.7 (−0.3). The abstract itself highlights +0.7 and −0.3 CHRF++ deltas. On 100 sentences, such small gaps are exactly in the range we would expect from modest sampling noise or a few easier/harder sentences.
  - There are no reported confidence intervals or bootstrap significance tests on CHRF++ for Kalamang, nor any sensitivity analysis to test-set variation.

  The authors do perform a regression over multiple prompt settings (Fig. 2), which is helpful, but the *pointwise* claim that grammar explanations add “no significant advantage” for Kalamang MT is not justified at the level of individual comparisons like BOOK\_{all} vs. BOOK\_{para} on 100 test items. This does not erase the strong trend “BOOK\_{para} ≫ BOOK\_{non-para},” but it does mean that fine distinctions and strong negative claims based on ~1–3 point gaps should be more cautiously phrased.

- **(2) Over-interpretation of the vocabulary-coverage regression as causal evidence against grammar explanations.**

  Section 5.1 fits univariate least-squares models of CHRF++ vs. test-set type coverage and finds strong correlations. It then states:

  > “These linear regressions show that translation performance can be directly modelled by test set vocabulary coverage, and that the book’s grammar explanations provide no significant advantage over its parallel sentences.”

  On the evidence presented, we can indeed say that type coverage is a strong predictor across these prompts. But the analysis is purely correlational and conflates several factors:

  - Conditions differ simultaneously in content type, token count, distribution of examples, and model behaviour (Gemini vs. Llama). Type coverage and “presence of explanations” are not independent.
  - A univariate regression cannot test whether, *controlling for coverage*, the presence of grammar explanations has any independent effect. There is no regression including an indicator for BOOK\_{all}/BOOK\_{para}/BOOK\_{non-para}, no interaction terms, and no multivariate analysis.
  - The conclusion that grammar explanations provide “no significant advantage” is specifically tied to this regression, but that step is not statistically supported: what we genuinely learn is that type coverage explains a lot of the variance, not that explanations are provably irrelevant.

  This is a case of analytic overreach. It does not negate the observed pattern that BOOK\_{non-para} performs clearly worse than BOOK\_{para} and that high-coverage parallel data dominates, but it weakens the strength of the anti-explanation claim.

- **(3) Generalisation and recommendations go beyond what three languages and this setup can support.**

  The paper repeatedly generalises its findings and recommendations to “XLR languages” and to data-collection strategy, for example:

  - Abstract: “…we find no evidence that long-context LLMs can make effective use of grammatical explanations for XLR translation, we conclude data collection for multilingual XLR tasks such as translation is best focused on parallel data over linguistic description.”
  - Conclusion: “We find no evidence that LLMs can effectively exploit grammatical explanations for low and extremely low-resource MT in Kalamang, Nepali, and Guarani… We therefore emphasise … data collection efforts … are better focused on parallel data over linguistic description…”

  The underlying evidence:

  - One truly unseen language (Kalamang), with a 100-example test set and the caveats above.
  - Two seen low-resource languages (Nepali, Guaraní) for which Gemini already has strong 0-shot performance (e.g., 42.5 and 65.2 CHRF++ for eng↔npi; Table 3). Here BOOK\_{all}/BOOK\_{non-para} often slightly hurt performance; BOOK\_{para} is neutral or mildly beneficial, but most differences are on the order of 1–2 CHRF++.
  - Only two LLMs are tested (Gemini-Flash and Llama-3.1-8B), with Gemini as the only model that can accommodate the largest grammar books fully (Sec. 4.2), meaning many key comparisons (BOOK\_{all} vs. BOOK\_{para}) are only evaluated on a single proprietary model.

  These are meaningful case studies, and the *qualitative* takeaways—naive full-book prompting is inefficient, parallel examples dominate—seem robust. But the much stronger normatively loaded conclusion (“data collection should be focused on parallel data over linguistic description” in general) is not fully warranted. It does not account for:

  - Languages with different morphological typologies or orthographies.
  - Different model architectures or pretraining regimes where grammar-like text might play a different role.
  - More sophisticated uses of grammar (structured extraction, retrieval over selected sections, LLM-written didactic summaries) as opposed to raw-book dumping.

  At present, the data support a narrower statement: *for the tested models and naive “entire-book” prompting schemes, sentence-aligned parallel/glossed examples are far more useful for MT than descriptive text sections*.

- **(4) Grammaticality judgment and IGT tasks, though promising, rest on small or noisy evaluation setups.**

  For grammaticality judgment (Sec. 3.4, Fig. 1):

  - The authors explicitly acknowledge: “we cannot guarantee all corruptions are ungrammatical (since no author speaks Kalamang).” In a language where word order flexibility is plausible, SWAP\_{adj} and SWAP\_{ran} corruptions could result in acceptable variants; SHUFFLE is more clearly extreme, but even there some outputs might be marginally acceptable.
  - Reported improvements from typological prompting over BOOK\_{para} are modest (e.g., 80% vs. 83% in SHUFFLE); without language-expert validation, it is unclear to what extent these differences reflect true grammatical knowledge vs. sensitivity to distributional familiarity or memorisation of seen strings.

  For IGT prediction (Sec. 3.5, Table 5):

  - The test set consists of 97 examples (447 words) from Dictionaria. That is commendably from a different source than training, but very small.
  - The paper compares Gemini+typology against various supervised baselines with very different capacities and training data. The headline that TYP+BOOK\_{para} achieves the highest morpheme accuracy (46.1%) is correct on this setup, but given the small test and architectural differences it is hard to interpret as a decisive win for this method.

  These experiments are still useful—and they do support the main *directional* point that grammar is more helpful for linguistic tasks than for MT—but the strength of the claims about “leading results” and “all-round competence” on these tasks is somewhat overstated.

### Minor

- **Ambiguity in what counts as “parallel” vs. “explanatory” content.**

  BOOK\_{para} is defined as “parallel glossed examples and bilingual word/phrase pairs” (Sec. 4.1), harvested by formatting heuristics following Nordhoff & Krämer (2022). BOOK\_{non-para} is “the remainder of the book” (explanations without aligned examples). In practice:

  - BOOK\_{non-para} will still include some latent translation information: paradigms, example fragments, in-text translations of individual words, metalinguistic comments that paraphrase meanings.
  - BOOK\_{para} contains not only full-sentence translation pairs but also phrase-level pairs and glossed morphology; WORDLIST is lexicon-like.

  The paper interprets BOOK\_{non-para} as “grammar explanations” and BOOK\_{para}/WORDLIST as “parallel data,” but this dichotomy is considerably cleaner at the level of aligned sentences than at the level of all linguistic information. The main empirical trend (aligned examples ≫ explanations for MT) still holds, but the conceptual framing “parallel vs. grammar” occasionally over-simplifies the heterogeneity of both subsets.

- **Limited analysis of npi/gug beyond aggregate scores.**

  While results on FLORES are solid and trends are described (Sec. 5), there is little further exploration of why, for example, BOOK\_{non-para} hurts Gemini on eng↔gug by up to ~7 CHRF++ vs. 0-shot (Table 3), or why 5*-SHOT helps Llama-I substantially. Some error analysis or at least a breakdown by direction/language family could sharpen the interpretive claims.

- **Typological prompt behaviour for MT is under-analysed.**

  For Kalamang, TYP+BOOK\_{para} helps into English relative to BOOK\_{para}, but not clearly into kgv; for Nepali and Guaraní, TYP+BOOK\_{para} often underperforms BOOK\_{para} or even 0-shot (Table 3). The paper notes this inconsistency, but the explanation is brief (“supports above finding that LLMs fail to effectively exploit grammatical information for MT”). A deeper look—e.g., whether typology helps especially on certain syntactic phenomena—would enrich the story.

### Trivial

- Some notation and labelling could be clearer. In Fig. 1, there are duplicate column headings “Book\_p”; from the text it seems one of these might be BOOK\_{non-para}. This is minor but slightly confusing.

- A few acronyms (e.g., BT for backtranslation) are used in results tables before being fully spelled out in the main text, though they are eventually explained.

## Nice-to-Haves

- **Bootstrap or randomisation-based significance tests for CHRF++ differences on kgv.** With only 100 test sentences, reporting confidence intervals or p-values for key contrasts (BOOK\_{all} vs. BOOK\_{para}, BOOK\_{para} vs. PARA\_{book}^{IGT}, etc.) would greatly strengthen the quantitative basis of the claims.

- **Some linguist-validated grammaticality annotations.** Even a small subset of Kalamang sentences with expert-labelled “grammatical/ungrammatical” pairs would make the grammaticality judgment results more interpretable and would calibrate how noisy the current corruption-based proxy is.

- **More thorough error analysis.** For at least a sample of kgv and npi/gug test items, showing side-by-side outputs under BOOK\_{all}, BOOK\_{para}, BOOK\_{non-para}, and TYP+BOOK\_{para} would help clarify *how* explanations and typology do or do not help.

- **Exploring more structured grammar use.** Experiments with automatically extracted rule lists, LLM-summarised grammar sections, or retrieval over a pre-indexed grammar book might show whether it is the unstructured, long-text format that hinders MT, rather than the grammatical information per se.

## Removed Points

These points are flagged to be removed from consideration as weaknesses; treat them with caution.

- **Questioning existence or availability of cited resources.** Any concern that the Kalamang grammar, dictionaries, or Grambank features might not exist, or that particular models/benchmarks are “unreleased,” would be invalid given the paper’s citations and must be disregarded.

- **Overly strong reproducibility complaints about API-based Gemini.** The use of Gemini-1.5-Flash via API is a legitimate design choice here; while it does limit exact reproducibility, this is currently common practice. Demanding release of proprietary model weights or exhaustive implementation minutiae goes beyond standard expectations and is not a substantive flaw.

- **Claims that the paper fails to use CHRF++ or standard metrics.** The authors explicitly adopt CHRF++ to address a criticism of MTOB (Sec. 3.1, 4.3) and are transparent about evaluation. Any critique that they “should have used CHRF++” or “ignored word order” would be factually incorrect.

## Novel Insights

The most valuable conceptual insight is the reorientation from “Can LLMs learn translation from a grammar book?” to “What parts of a grammar book actually matter, and for which tasks?” The data show that, in this setup, aligned examples and glossed text dominate MT performance, while dense descriptive explanations are token-inefficient for MT but become beneficial when converted into typological features and applied to clearly grammatical tasks (grammaticality judgments, IGT). This suggests a division of labour: for XLR MT, parallel signal—whether harvested from grammars, dictionaries, or other sources—is the primary driver, whereas descriptive grammar is more naturally leveraged for documentation-centric tasks when suitably structured. This distinction goes beyond MTOB-style demonstrations and offers a more nuanced view of how linguistic resources should be targeted and repurposed.

## Suggestions

- **Tone down categorical conclusions and sharpen their scope.** Rephrase statements like “we find no evidence that LLMs can effectively exploit grammatical explanations for XLR translation” to clearly situate them in the tested setting: current Gemini/Llama models, full-book prompting, these three languages, and these evaluation sets. Similarly, reframe the recommendation on data collection from a universal prescription to a strong empirical tendency in this paradigm.

- **Augment Kalamang evaluation robustness.**
  - Perform bootstrap resampling of the 100-sentence test set to estimate confidence intervals for key CHRF++ differences.
  - Explicitly verify that no sentence in the test set is duplicated in PARA\_{book}, PARA\_{train}, WORDLIST, or the Dictionaria dev set; if some near-duplicates exist, quantify their impact.

- **Clarify and (if possible) refine the “parallel vs. explanation” operationalisation.** Provide more detailed criteria and (ideally) some statistics about what “leaks” into BOOK\_{non-para} (e.g., counts of in-text translated words or paradigms). If feasible, add a sub-ablation that removes clearly example-like fragments from BOOK\_{non-para}, to better approximate “rules-only” vs. “examples-only” conditions.

- **Strengthen the grammaticality and IGT narratives.**
  - For grammaticality, report at least some manual checks of corruption outcomes and consider designing additional corruption patterns that match known Kalamang properties from the grammar.
  - For IGT, discuss more explicitly how much of the test morpheme inventory is unseen in training; if feasible, present results split by seen vs. unseen morphemes.

- **Analyze typological prompt failures in MT more deeply.** Try to characterise for which categories or constructions typology helps or hurts. Even a small qualitative study could help explain why typology is beneficial for linguistic tasks yet inconsistent for translation.

- **Bridge to more structured grammar usage.** Given the negative results for raw BOOK\_{non-para}, consider adding at least a pilot experiment with a small, LLM-generated summary of crucial rules (e.g., 1–2 pages) derived from the grammar, to test whether more compact, didactic formulations of explanations fare better.

## Score and Decision

### Calibration

I compared this paper conceptually to:

- **MTOB (Machine Translation from One Book)**: a strong paper (human scores around 7–8) that introduced the Kalamang benchmark but had limited ablations and a tiny test set. The current paper improves some aspects (better metric, grammar vs. parallel ablations, MT baselines) but does not reach the same level of broad impact or clarity in its central claim, partly due to overgeneralisation.

- **Other low-resource MT/LLM prompting papers** (e.g., ones critiqued for small test sets and overclaiming): these often received mid-range scores (4–6) when claims outran evidence. Compared to the weaker ones (scores ~3), this paper is more careful experimentally and offers more substantial analysis. Compared to stronger ones with broader validation (scores ~7+), this work’s main limitation is that its decisive-sounding conclusions are not as thoroughly supported across languages and settings.

Relative to these anchors, this paper presents meaningful, well-executed experiments and a useful typological prompting idea, but its conclusions need tempering and some analysis is over-interpreted. I see it as solid but not yet at the bar of the strongest empirical papers in this niche.

**Overall score: 6.0**

The paper has real contributions (practical MT baselines for MTOB, parallel-vs-explanation ablation, typology for IGT), and the empirical work is largely sound. However, I would require more cautious framing and some additional robustness checks before recommending acceptance at a top venue.

MY FINAL SCORE: <pineapple>6.0</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>