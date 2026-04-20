## Summary
This paper presents a careful empirical analysis of what drives translation performance when LLMs are prompted with grammar books. By manually disentangling grammar books into parallel examples and non-parallel explanatory text, the authors demonstrate that almost all translation gains come from parallel data, not grammatical explanations. The paper introduces a novel typological feature prompt that shows promise for linguistic tasks like IGT prediction. The core finding is well-supported and actionable: parallel data, not grammatical explanations, is what matters for machine translation.

## Strengths

- **Rigorous disentanglement of parallel vs. non-parallel content**: The manual filtering of grammar books into `BOOK_para` and `BOOK_non-para` (Tables 1-3) is the paper's central methodological contribution, directly addressing a critical confounder in MTOB. Results show `BOOK_para` (18k tokens) nearly matches `BOOK_all` (100k tokens) for Gemini in both directions, while `BOOK_non-para` lags by 7-8 CHRF++.

- **Fine-tuned NLLB-1.3B matches Gemini on same data**: Table 4 shows NLLB-1.3B trained on PARA_book achieves 34.2 CHRF++ into kgv, outperforming Gemini's 26.6 with identical parallel data in-context. This strongly supports the practical conclusion that computationally cheaper specialist models suffice.

- **Effective typological prompting for IGT prediction**: The novel TYP + BOOK_para prompt achieves leading morpheme accuracy (46.1%, Table 5), beating all supervised baselines by 1-5% and BOOK_all by 6%. This demonstrates LLMs CAN exploit grammatical information when the task and data format are appropriate.

- **Generalization across three typologically distinct languages**: Testing Kalamang (unseen), Nepali, and Guarani (both seen) strengthens the claim that results are not language-specific artifacts.

- **Addresses methodological limitations of prior work**: Combining directional test sets (50→100), using CHRF++ (properly accounting for word order), and noting evaluation issues in baseline work shows careful attention to evaluation rigor.

## Weaknesses

### Major

- **Limited statistical rigor on core translation claims**: The paper reports precise CHRF++ differences (e.g., +0.7 points for BOOK_all over BOOK_para, "drops up to 8 points") on a 100-sentence test set without confidence intervals, standard deviations, or significance tests. While large effects (7-8 point gaps) are unlikely to be noise, the small differences cited in the abstract and introduction as "significant" cannot be distinguished from variance. Given the sensitivity of character n-gram metrics to length mismatches and rare morphemes in XLR languages, sub-2 point fluctuations are expected variance. This undermines the precise claims about what is "significant" versus "no significant advantage" made throughout the paper.

### Minor

- **Regression analysis limitations**: The univariate regression of CHRF++ on type coverage (Figure 2) shows correlation but doesn't cleanly separate the effect of grammatical explanations from the additional vocabulary coverage they provide. Since BOOK_all inherently carries type coverage from its non-parallel portion alongside any explanatory content, the regression demonstrates type coverage matters but doesn't rule out that explanations provide conditional value at matched coverage levels. The paper's conclusion that explanations add "no significant advantage" is reasonable but slightly overclaimed given this confound.

- **Grammaticality judgment task design limitations**: As noted in Section 3.4, the task asks models to choose original sentences against corrupted versions using SWAP_adj, SWAP_ran, and SHUFFLE strategies. The marginal gains observed (+3% for TYP + BOOK_para over BOOK_para in the hardest SHUFFLE setting) are small, and the task may not fully isolate grammatical knowledge acquisition from basic sequence likelihood modeling. However, the paper appropriately uses this as supplementary evidence for the broader claim that LLMs can use grammatical information for linguistically-relevant tasks, not as proof of deep grammar acquisition.

- **Inconsistent transfer of typological prompting to translation**: Table 3 shows that for translation, TYP + BOOK_para performs inconsistently—it helps into English for Kalamang but shows mixed results for Nepali and Guarani, sometimes underperforming BOOK_para alone. The paper's introduction presents typological prompting as "more effective than explanations into eng, but not into XLR languages" which slightly oversells the consistency of these results.

### Trivial

- None

## Nice-to-Haves
- Bootstrap confidence intervals or resampling analysis for the 100-sentence test set would strengthen claims about small performance differences
- Multivariate regression controlling for both type coverage AND explanation content would more directly address the core question
- Error analysis showing specific cases where typological prompting improves grammaticality judgment would strengthen the IGT findings

## Removed Points

- **Criticism about overclaiming mechanistic findings from the statistical analysis**: The harsh critic argues the regression conflates vocabulary coverage with explanatory utility. While the regression has limitations, the paper's primary conclusion (parallel data drives performance, BOOK_para ≈ BOOK_all) is directly supported by the empirical results showing similar performance with much less data. The regression is actually the strongest piece of evidence for the main claim.

- **Criticism about grammaticality judgment task being "invalid"**: The critic claims the task only tests sequence likelihood, not grammar acquisition. While the task has limitations (acknowledged above as a minor weakness), the paper uses it as supportive evidence for the secondary claim that LLMs can use grammatical information for linguistic tasks. The +3% improvements from typological prompting are meaningful in context.

- **Criticism about underpowered evaluation protocol**: The paper uses 100 sentences and reports precise differences without confidence intervals. This is valid as a minor concern about statistical rigor, but not a "structural" flaw. The large effect sizes (7-8 CHRF++ differences) are robust, and the concern primarily applies to small differences.

- **Criticism about FLORES pre-training contamination for Nepali/Guarani**: The paper acknowledges these are "seen" languages and uses them to generalize beyond Kalamang. This is a known limitation but doesn't invalidate the core contribution for unseen languages.

- **Criticism about "needle-in-a-haystack" being "speculative"**: The paper explicitly labels this as a potential explanation ("could partially explain"), not a finding. The paper appropriately acknowledges this limitation.

- **Criticism about typological prompting "overselling"**: The paper is actually quite measured about typological prompting for translation, noting inconsistent results. The introduction slightly overstates it, but the results section is appropriately cautious.

## Novel Insights
The paper makes a genuinely valuable contribution by systematically disentangling parallel data from grammatical explanations in grammar books—a confound that prior work (MTOB) did not address. The finding that BOOK_para (a 5x smaller prompt) nearly matches BOOK_all performance is both surprising and practically important for XLR data collection priorities. The typological feature prompt's success on IGT prediction (beating supervised baselines) suggests a promising direction for leveraging structured linguistic information in machine-readable formats. The paper appropriately emphasizes task-appropriate data: parallel examples for translation, typological features for linguistic tasks.

## Suggestions
1. Report bootstrap confidence intervals or variance estimates for all CHRF++ scores on the 100-sentence test set, especially for claims about "significant" vs "no significant" differences
2. Consider multivariate analysis controlling for both type coverage and content type to more directly address the confound between explanations and vocabulary
3. Provide specific error analysis examples showing where typological prompting makes correct predictions that parallel-only prompts fail on
4. Temper the framing of "no significant advantage" to acknowledge that the regression shows correlation rather than cleanly establishing no conditional benefit from explanations at matched coverage

## Calibrated Scoring Reasoning

Compared to calibration anchors:
- **sjvz40tazX.md** (ASROB, MTOB extension): Scores 8,3,5,6, avg ~5.5, Reject. Similar topic (MTOB, Kalamang, grammar books, ablation). This paper's more thorough disentanglement and stronger empirical support for its claims puts it above this anchor.
- **3KDbIWT26J.md** (LLM translation without parallel data): Scores 6,6,5,6, avg ~5.75, Accept. Similar ablation-focused methodology on translation. Comparable quality.
- **farT6XXntP.md** (ALMA, LLM translation): Scores 8,5,6,8, avg ~6.75, Accept. Stronger methodological contribution (novel fine-tuning approach). This paper is slightly below due to narrower scope and some analytical limitations.
- **YUefWMfPoc.md**: Scores 6,5,6,6, avg ~5.75. Similar concerns about statistical rigor and confidence intervals. This paper is comparable.
- **eznTVIM3bs.md** (Babel Tower): Scores 6,3,6,6, avg ~5.25. Similar presentation/analysis clarity issues but solid experiments.

This paper's strongest contribution—the clean disentanglement showing parallel data drives translation—is well-supported and answers an important question. The statistical rigor concerns are real but minor relative to the strong empirical findings. The paper sits comfortably in the 5.5-6.5 range, comparable to other solid ablation papers that were accepted. I position it at 6: solid empirical contribution with some analytical limitations, worth accepting but not outstanding.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>