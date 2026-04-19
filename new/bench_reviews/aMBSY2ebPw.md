I can see the paper DOES report statistical significance tests for their regression analysis (p < 0.005 for the type coverage regression). The harsh reviewer's claim that "no statistical significance tests on CHRf++ are reported for differences between prompt conditions for Kalamang specifically" is partially inaccurate - the paper does report significance for the regression analysis, though not bootstrap CIs for individual CHRf++ differences.

Now let me check what the paper claims about the "no significant advantage" claim:

Looking at line 230: "These linear regressions show that translation performance can be directly modelled by test set vocabulary coverage, and that the book's grammar explanations provide no *significant* advantage over its parallel sentences."

The paper uses "significant" in a statistical sense (referring to their regression analysis), not just as a qualitative claim. This is important context.

Now let me formulate my final review based on all this analysis and calibration.

Key findings from my verification:
1. The harsh reviewer's claim about BOOK_all showing "synergy" is weakened by the actual data - for kvg-eng, BOOK_all (34.4) is actually slightly worse than BOOK_p (34.7)
2. The paper DOES report statistical significance for their regression analysis
3. The paper's claim about "no significant advantage" is backed by regression analysis showing type coverage explains performance
4. The 100-sentence test set for Kalamang is a legitimate limitation
5. Missing bootstrap CIs for individual CHRf++ comparisons is a valid minor weakness
6. The confounds between explanations and other properties (length, domain mismatch) are real but the paper does address token efficiency

Based on calibration:
- Papers with strong ablations and clear contributions (like 5Ca9sSzuDp) got 8s
- Papers with weak evaluation and small test sets got 3s
- Borderline papers with solid empirical work but limited novelty (like MCjVArCAZ1) got 5s
- The related MTOB extension paper (sjvz40tazX) got 8/3/5/6 and was rejected

This paper has:
- Clear, well executed ablation study (major strength)
- Statistical validation of claims (strength)
- Multiple languages tested (strength)
- Novel typological prompting contribution (strength)
- Practical findings about fine-tuning vs prompting (strength)

Weaknesses:
- 100-sentence test set for Kalamang is small (minor)
- Missing bootstrap CIs for individual comparisons (minor)
- Claims could be slightly tempered (minor presentation issue)
- Confounds between length/domain and explanation content not fully disentangled (minor)

This is a solid empirical paper with clear contributions. Based on calibration against similar empirical ablation papers that got 8s, and considering the paper's strengths outweigh its weaknesses, I would score this in the 6-7 range (borderline accept to weak accept).

Let me write my final review:

## Summary
This paper investigates whether LLMs can effectively use grammar books for extremely low-resource (XLR) machine translation, specifically examining the contribution of grammatical explanations versus parallel examples. Through careful ablation of grammar books into parallel and non-parallel subsets across Kalamang, Nepali, and Guarani, the authors demonstrate that parallel examples drive translation performance while grammatical explanations provide no statistically significant advantage beyond vocabulary coverage. The paper further shows that fine-tuning small encoder-decoder models matches long-context LLM performance, and introduces a novel typological prompting approach that achieves leading results on linguistically-focused tasks like IGT prediction.

## Strengths
1. **Rigorous ablation design isolating parallel data from explanations**: The manual separation of grammar books into BOOK_para and BOOK_non-para subsets (Table 1) and systematic comparison across BOOK_all, BOOK_p, and BOOK_¬p conditions (Table 2) provides clear causal evidence. Results show BOOK_non-para achieving only 22.6 CHRf++ versus 30.8-34.7 for BOOK_para in Kalamang, demonstrating parallel examples are the primary signal.

2. **Statistical validation of core claim**: Section 5.1 presents regression analysis (Figure 2) showing CHRf++ scores are significantly predicted by test set type coverage (p < 0.005), with all prompt settings falling within 95% confidence intervals. This provides empirical grounding for the claim that explanations add no significant advantage beyond vocabulary exposure.

3. **Multi-task evaluation broadening contribution**: Beyond MT, the paper tests grammaticality judgment (Figure 1) and IGT prediction (Table 5), demonstrating LLMs can exploit grammatical knowledge when provided in appropriate form—typological prompting achieves 46.1% morpheme accuracy on IGT, beating supervised baselines.

4. **Practical demonstration of compute-efficient alternatives**: Table 4 shows NLLB-1.3B fine-tuned on ~1.2k parallel sentences achieves 34.2 CHRf++, competitive with Gemini prompted with the same data (26.6), validating that standard MT approaches remain viable for XLR settings.

5. **Generalization across multiple languages**: Findings extend beyond the unseen Kalamang to Nepali and Guarani (Table 3), where BOOK_non-para often degrades performance versus 0-shot (e.g., Guarani eng-gug drops from 26.6 to 19.3), confirming results are not language-specific artifacts.

## Weaknesses

### Fatal
None

### Major
None

### Minor
- **Small test set for Kalamang limits generalizability**: The 100-sentence combined Kalamang test set (§3.1) is relatively small for drawing strong normative conclusions about data collection priorities across XLR languages. While appropriate for demonstrating qualitative trends, the absence of bootstrap confidence intervals or variance estimates for CHRf++ differences means some 1-3 point variations may be noise rather than signal.

- **Confounds between explanation content and prompt properties**: BOOK_non-para differs from BOOK_p not only in containing explanations versus parallel examples, but also in token count (81k vs 18k for Kalamang), domain alignment, and structural format (Table 1). The paper's token-efficiency analysis shows no significant relationship between prompt length and performance within the BOOK family (§5.1), but does not fully disentangle whether poor BOOK_non-para performance stems from grammatical content being unusable versus needle-in-a-haystack retrieval difficulties or domain mismatch with FLORES test data.

- **Overly categorical framing of negative findings**: The abstract's claim "we find no evidence that long-context LLMs can make effective use of grammatical explanations for XLR translation" and conclusion that "data collection...is best focused on parallel data" are stronger than what the experimental design supports. The experiments test only one specific usage pattern (raw grammar text stuffed into prompts) on three languages; this justifies conclusions about current prompting methods rather than fundamental limitations of grammatical knowledge for MT.

### Trivial
- **Incomplete error analysis for MT results**: The paper focuses on aggregate CHRf++ scores without linguistically-informed error analysis examining whether explanations affect specific error types (agreement, case marking, word order) that CHRf++ may underweight.

## Nice-to-Haves
- **Structured access to grammar books**: Future work could explore converting grammar books into indexed, retrievable knowledge bases rather than full-text prompts, potentially mitigating needle-in-a-haystack effects while preserving explanatory content.

- **Additional typologically diverse unseen languages**: Testing on one more genuinely unseen XLR language with markedly different typological profile (e.g., polysynthetic or ergative) would strengthen generalizability of the "parallel over grammar" conclusion for MT.

## Removed Points
These points are flagged to be removed, treat them with caution:

1. **"Overstated central claim" from Harsh Critic**: While the paper's framing is somewhat categorical, the regression analysis with p < 0.005 does statistically support the "no significant advantage" claim. The reviewer's point about BOOK_all potentially showing synergy is contradicted by the actual data—for kvg-eng, BOOK_all (34.4) performs slightly worse than BOOK_p (34.7), not better.

2. **Criticism about unfair comparison between fine-tuning and prompting**: The paper explicitly compares NLLB and Gemini on equivalent PARA_book data (Table 4), and the claim is about matching performance given the same parallel data, not about computational efficiency. This is not an unfair comparison—the asymmetry favors demonstrating fine-tuning works well.

3. **Claim that no statistical significance tests are reported**: The paper reports F-test p-values (p < 0.005) for the regression analysis and explicitly states all settings fall within 95% confidence intervals (line 230). While bootstrap CIs for individual CHRf++ comparisons would strengthen the paper, the absence does not invalidate the core analysis.

4. **Weakness about missing related work on alternative grammar exploitation**: Per hard rules, I cannot critique missing related works without external verification. The paper adequately cites prior work on grammar-in-context methods.

5. **Reproducibility concerns about undisclosed hyperparameters**: The paper reports LoRA hyperparameters (α=16, r=16, lr=1e-4, 5 epochs, batch size 4) for Llama fine-tuning (§4.2). Requests for training logs or additional implementation details are minor nitpicks per the hard rules.

## Novel Insights
The most novel observation emerging from this work is the divergence between human and LLM learning strategies for translation: while humans benefit from worked examples with explicit grammatical rules (§5 Discussion), LLMs learn more effectively from unannotated parallel examples alone, suggesting prompts with parallel data align more closely with LLMs' instruction-tuning distribution than grammar-book explanations. Additionally, the finding that typological abstractions—rather than raw explanatory text—enable LLMs to exploit grammatical knowledge for linguistic tasks (IGT, grammaticality judgment) reveals a promising avenue for encoding linguistic expertise in a form accessible to LLMs. The regression analysis demonstrating that vocabulary type coverage fully mediates translation performance provides a quantitative framework for evaluating future prompt designs.

## Suggestions
1. Conduct a matched-coverage experiment where BOOK_non-para is truncated to match the type coverage of BOOK_p, isolating whether explanations confer any benefit conditional on equal vocabulary exposure.

2. Add bootstrap confidence intervals or permutation tests for key CHRf++ comparisons, particularly for Kalamang where the 100-sentence test set raises uncertainty about whether 1-3 point differences are reliable.

3. Include a small-scale qualitative error analysis (e.g., 30-50 sentences) examining whether explanation-containing prompts affect specific grammatical phenomena like agreement, case, or word order, even if aggregate CHRf++ shows no improvement.

4. Temper the abstract and conclusion language to clarify that findings apply to "grammar books used as raw in-context prompts" rather than making broader claims about grammatical knowledge being unusable for MT.

## Score and Decision

**Calibration reasoning:** I compared this paper against several anchors:
- High-scoring empirical papers with strong ablations (5Ca9sSzuDp: 8,8,8,8) featured carefully designed ablation studies and excellent execution—this paper matches that pattern with its BOOK_para/BOOK_non-para disentanglement.
- The related MTOB extension paper (sjvz40tazX: 8/3/5/6, Reject) had similar scope but received criticism for limited novelty; this paper is more novel with its grammar-book ablation and typological prompting contributions.
- Borderline empirical papers (MCjVArCAZ1: 5,5,3,5) were rejected due to small effect sizes (<0.2 Cohen's d); this paper shows larger, more consistent effects (8+ CHRf++ differences between BOOK_p and BOOK_¬p).
- Accepted papers with solid empirical contributions but limited novelty (farT6XXntP: 8,5,6,8) are analogous—this paper has clearer novelty in the grammar-book disentanglement.

The paper's core ablation is well-executed and answers an important question in XLR NLP. The statistical validation, multi-language generalization, and demonstration of fine-tuning as a viable alternative are all substantive contributions. Weaknesses are primarily presentational (overly strong claims) and methodological (small test set, missing CIs) rather than fundamental flaws. Compared to calibration anchors, this paper is stronger than the 5-scored borderline papers and comparable to weak-accept (6-7) empirical contributions.

Relative to the field's standards for empirical XLR MT papers, the methodology is sound, the claims are largely supported (with minor tempering needed), and the contribution is clear. The paper does not reach the 8-level of exceptional papers with multiple strong applications or theoretical breakthroughs, but it exceeds the 4-5 level of papers with fundamental flaws or trivial contributions.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>