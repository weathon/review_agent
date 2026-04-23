Now I have a thorough understanding of the paper and calibration anchors. Let me synthesize the final review.

## Summary

This paper investigates whether long-context LLMs can effectively exploit grammatical explanations in grammar books for extremely low-resource (XLR) machine translation, directly challenging Tanzer et al. (2024)'s MTOB claim. Through a clean ablation splitting grammar books into parallel examples (BOOK_para) and explanatory text (BOOK_non-para), the authors find that nearly all translation improvements stem from parallel examples, not grammatical explanations. They support this with a regression analysis showing performance is explained by test-set type coverage, demonstrate that fine-tuning small NLLB models matches long-context LLMs on the same parallel data, and show that grammatical knowledge (via a novel typological feature prompt) does help on linguistically-relevant tasks like grammaticality judgment and IGT prediction.

## Strengths

- **The core ablation is well-designed and addresses a real gap in MTOB.** Splitting grammar books into BOOK_para and BOOK_non-para is the right experiment. Table 2 shows BOOK_para (30.8 ChrF++ eng→kgv) massively outperforms BOOK_non-para (22.6) while using ~4.5× fewer tokens, and BOOK_para marginally outperforms BOOK_all into eng (34.7 vs 34.4). This directly challenges MTOB's framing.

- **The type-coverage regression analysis (Figure 2) provides a compelling explanatory mechanism.** The univariate regression of ChrF++ against test-set type coverage yields significant models (p < 0.005, F-test) in both directions, with all prompt settings falling within 95% CIs. This gives a parsimonious explanation: vocabulary coverage, not grammatical understanding, drives the residual BOOK_all advantage over BOOK_para.

- **Fine-tuning NLLB on parallel data challenges the need for long-context LLMs.** Table 4 shows NLLB-1.3B fine-tuned on PARA_book achieves 34.2 ChrF++ into kgv vs. Gemini's 26.6 with the same data — a 7.6-point advantage at a fraction of the cost. This is practically significant for the XLR MT community.

- **Testing grammaticality judgment and IGT prediction as alternative tasks is well-motivated.** The finding that typological prompting (TYP + BOOK_para) achieves leading results on IGT prediction (46.1% morpheme accuracy, Table 5) and grammaticality judgment (83% on SHUFFLE, Figure 1) while failing to help translation coherently supports the "task-appropriate data" thesis.

- **Multi-language evaluation strengthens generalizability.** Extending to Nepali and Guarani (Table 3) shows consistent patterns: BOOK_para matches or outperforms BOOK_all, while BOOK_non-para often decreases performance below 0-shot.

- **Methodological improvements over MTOB** are concrete: ChrF++ over ChrF, combined 100-example test set over 50, and the ablation MTOB lacked.

## Weaknesses

### Fatal
None.

### Major

- **The abstract reports +0.7 ChrF++ as the effect of "adding explanations to parallel sentences" into kgv, but the raw difference in Table 2 is 3.6 points (BOOK_all=34.4 vs BOOK_para=30.8).** The +0.7 appears to be the regression residual after accounting for type coverage, but this is never explicitly labeled as such in the abstract or introduction. A reader would reasonably interpret "+0.7" as the measured effect of adding grammatical explanations, which it is not — the measured effect is 3.6. The paper's argument is that this 3.6 is explained by vocabulary coverage rather than grammar, which may be correct, but presenting the regression-adjusted number without context in the abstract is misleading and understates the raw gap by ~5×. This matters because the paper's central claim hinges on the magnitude of the grammar explanation effect. The introduction repeats the same +0.7 figure without clarification.

- **The "fine-tuning matches long-context LLMs" claim is direction-selective.** The abstract states "competitive results within 0.2 ChrF++ of the performance of Gemini with a grammar book into kgv" (NLLB: 34.2 vs Gemini BOOK_all: 34.4). But in the reverse direction (kgv→eng), Table 4 shows NLLB scores 28.6 vs Gemini's 34.4 — a 5.8-point deficit. The abstract specifies "into kgv" which is technically accurate, but the framing of "competitive" and the conclusion's claim that "fine-tuning small MT models matches the performance of costly long-context LLMs" implies broad parity that only exists in one direction. The asymmetric result needs honest discussion.

### Minor

- **No confidence intervals or significance tests on the direct ablation comparisons in Table 2.** The core comparison (BOOK_all vs BOOK_para) on a 100-example test set lacks bootstrap CIs or paired tests. While the regression in Section 5.1 provides some statistical grounding, it tests whether type coverage predicts scores overall, not whether BOOK_all significantly outperforms BOOK_para directly. With N=100, ChrF++ sampling variance can be substantial, and the 3.6-point gap into kgv could plausibly reflect noise.

- **The regression analysis has partial circularity.** The regression in Section 5.1 fits on all data points including BOOK_all, then notes BOOK_all falls within the 95% CI. A cleaner test would fit the regression on non-BOOK_all settings only, then predict BOOK_all's score. If the prediction matches, the type-coverage explanation is strongly supported; if BOOK_all significantly exceeds the prediction, grammatical explanations contribute beyond vocabulary. The current setup makes it easier for BOOK_all to fall within the CI since it helped define the regression line.

- **The generalization to seen languages (npi, gug) involves a qualitatively different mechanism than for unseen kgv.** For Nepali and Guarani, BOOK_all and BOOK_non-para often *decrease* performance below 0-shot (Table 3: gug→eng, BOOK_all=38.7 vs 0-shot=41.3). The paper notes this but frames it as "generalising our findings," when the mechanism is different: for seen languages, grammar books may actively interfere with prior knowledge rather than simply failing to add value. This distinction affects the practical recommendations.

- **Typological prompting results for translation are inconsistent and modest.** TYP + BOOK_para into kgv (31.4) doesn't beat BOOK_all (34.4), and for npi/gug (Table 3), TYP + BOOK_para often underperforms BOOK_para alone. The paper acknowledges this, but the "novel typological feature prompt" contribution is primarily validated on linguistic tasks, not translation — the framing could be clearer.

### Trivial
None.

## Nice-to-Haves

- A controlled experiment where BOOK_para is augmented with additional vocabulary from BOOK_non-para *without* grammatical explanations, to isolate whether the 3.6-point gap is truly vocabulary alone.
- Qualitative examples showing what BOOK_non-para fails to provide that BOOK_para succeeds at, strengthening the argument beyond aggregate scores.
- Error analysis on the kgv→eng direction where NLLB substantially underperforms Gemini, to inform the practical recommendation.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Typological prompt construction details "relegated to Appendix D"**: This is a complaint about missing appendix content. Per rules, the parser strips appendices; the original submission includes this content.

- **SHUFFLE condition validity concern (n-gram frequency vs grammatical knowledge)**: The observation that BOOK_para outperforms BOOK_non-para on SHUFFLE is actually *consistent* with the paper's thesis — parallel examples provide useful signal even for grammar-related tasks. Interpreting this as a weakness of the task design misunderstands the paper's argument, which is precisely that parallel examples carry more signal than explanations.

- **Request for testing on additional unseen XLR languages**: This is a generic "one more language" request. The paper already tests three languages (one unseen, two seen) and explicitly scopes its contribution. Testing more unseen languages would strengthen but is not a core flaw.

- **Nepali/Guarani seen-language issue as overclaiming generalization**: The paper does discuss the different dynamics for seen languages (Section 5, para 2: "perhaps the model's prior competence...mean there is less to be gained"). The generalization claim is qualified.

- **Few data points in regression scatter plots (~10-12)**: While true, this is inherent to the experimental design (limited number of prompt configurations). The regression is significant (p < 0.005) despite few points, which actually strengthens rather than weakens the finding.

- **Overselling typological prompting as "leading results"**: The 0.5% morpheme accuracy improvement over BOOK_para on IGT (46.1 vs 45.4) is indeed modest. This is noted under Minor weaknesses above as "inconsistent and modest" for translation.

## Novel Insights

The paper makes an important observation about the divergence between human and LLM learning strategies for translation: humans learn translation more effectively from worked examples with explicit explanations (example-based learning), while LLMs in-context learn more effectively from unannotated parallel examples (discovery learning). The paper suggests this may stem from prompts with parallel data aligning more closely with LLMs' instruction-tuning data than grammar book explanations — an insight that could inform how we design prompts for XLR tasks more broadly.

## Suggestions

- Correct the abstract and introduction to either report the raw +3.6 ChrF++ difference into kgv and explain that the regression attributes ~2.9 points to type coverage, or explicitly label the +0.7 as a regression-adjusted residual. This is the single most important revision.
- Report the kgv→eng fine-tuning comparison honestly alongside the favorable eng→kgv direction, and discuss why the asymmetry exists (vocabulary issues? fluency?).
- Add bootstrap confidence intervals to Table 2's key comparisons, at minimum for the BOOK_all vs BOOK_para comparison with Gemini.

## Score and Decision

**Calibration anchors compared:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| "The False Promise of Imitating Proprietary LMs" (debunking with strong empirical evidence, clean presentation) | 7.0 | This paper has similar debunking spirit but weaker presentation (misleading abstract statistic, selective framing) |
| "Reasonableness Behind Unreasonable Translation" (ablation on LLM translation, methodological limitations) | 5.75 | This paper has a cleaner core experiment and more practical implications |
| ASROB/MTOB speech extension (same benchmark, limited novelty) | 5.5 | This paper has substantially more novelty and analytical depth |
| ALMA fine-tuning for MT (practical MT improvement, some overclaim) | 6.75 | This paper has comparable practical significance but more framing issues |
| Misleading abstract papers (cherry-picked statistics) | 3.0-4.5 | This paper is clearly stronger with genuine empirical contributions |

The paper makes a genuine and important empirical contribution — the BOOK_para/BOOK_non-para ablation directly addresses a gap in MTOB, and the type-coverage analysis provides a compelling explanation. However, the misleading +0.7 statistic in the abstract (vs. the raw 3.6) and the direction-selective "competitive" fine-tuning claim are significant presentation problems that undermine trust in the paper's framing. These are not fatal — the core finding is directionally correct — but they are Major weaknesses that a reviewer would weigh against acceptance. The paper sits above the medium-scoring ablation papers (5.5-5.75) due to its cleaner experimental design and more compelling analysis, but below the high-scoring debunking papers (7.0) due to the framing issues.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>