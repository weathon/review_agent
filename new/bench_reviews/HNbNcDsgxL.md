## Summary

This paper proposes Delta, a training-free inference-time method that mitigates text hallucinations by applying contrastive decoding between original and randomly masked input prompts. The method is evaluated on extractive and multiple-choice QA benchmarks, with the most notable result being a large improvement in SQuAD v2 no-answer exact match.

## Strengths

- **Simplicity and deployability.** Delta requires no retraining or external data and is trivial to implement on top of existing inference pipelines (Section 3, Equation 3).
- **Honest reporting of boundary conditions.** The authors report slight declines on CommonsenseQA and MMLU (Table 2; Section 5.3), which usefully delineate that the method is context-dependent.
- **Notable SQuAD v2 abstention signal.** The large no-answer exact match swing on SQuAD v2 (Table 1) is an interesting empirical effect that, if properly analyzed, could inform future calibration or abstention research.

## Weaknesses

### Fatal
None.

### Major
- **Evaluation protocol conflates extractive QA accuracy with hallucination reduction.** The central claim is that Delta mitigates “text hallucinations,” yet the experiments rely exclusively on exact-match and F1 metrics on extractive QA datasets. On SQuAD v2, the overall EM improvement is driven almost entirely by a +14.53 pp jump in NoAns_EM, while HasAns_EM actually drops by 1.6 pp (Table 1). Better abstention on unanswerable questions is not synonymous with targeted hallucination reduction, and the drop in HasAns_EM suggests a conservative-bias trade-off that the paper does not analyze. Without hallucination-specific metrics (e.g., on open-ended generation benchmarks such as TruthfulQA or HaluEval) or an analysis of error types, the construct validity of the evaluation is insufficient to support the broad claim.
- **Missing direct comparison with the most relevant inference-time baselines.** The paper positions Delta as “novel” and “more generalizable than CAD” (Section 2) and adapts ideas from Visual Contrastive Decoding, yet it runs no experiments against CAD, DoLA, or standard contrastive decoding under identical conditions. Because random masking is functionally related to context ablation, the incremental value of the masking strategy over existing contrastive decoding methods is unverified. This omission makes it impossible to assess whether Delta improves upon, matches, or underperforms prior training-free techniques.

### Minor
- **Selective framing of TriviaQA and Natural Questions results.** The abstract highlights gains of 7 pp and 2 pp on TriviaQA and NQ, but these occur only with sampling; without sampling, Delta degrades performance on both datasets (Table 1: TriviaQA 48.27→48.13; NQ 14.88→14.57). The paper describes these non-sampling results as “marginal” (Section 5.2) when they are in fact negative, which obscures the method’s sensitivity to the decoding strategy.
- **No statistical validation.** Main results are reported to five decimal places without standard deviations, confidence intervals, or significance tests (Table 1), leaving the stability of small gains unclear.
- **Ablation scope is narrow.** The ablation study (Section 6; Figure 2) is limited to SQuAD v1.1 with sampling. It does not examine the mask token choice, the effect of masking strategy (random versus targeted), or the parameters that drive the SQuAD v2 NoAns behavior.
- **Theoretical justification is thin.** The argument that masking amplifies hallucinations (Section 3.2) rests on a single contrived example (the “moldy banana”) and offers no broader evidence that the effect generalizes beyond that illustration.

### Trivial
None.

## Nice-to-Haves
- Evaluation on an open-ended generation benchmark with hallucination-specific metrics (e.g., FactScore, HaluEval, or TruthfulQA) to substantiate the broad claim of text hallucination mitigation.
- Head-to-head comparison with CAD and DoLA on identical models and datasets.
- Analysis of why greedy decoding hurts on TriviaQA and NQ while sampling helps, including qualitative examples of model outputs.
- Ablation on the choice of MASK token and masking ratio.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **“Distinctive technical formulation” strength (Strength Finder).** While Delta uses random masking rather than full context removal, without empirical comparison against CAD it is premature to claim this is a distinctive or advantageous formulation.
- **“Large, measurable reduction in hallucination for unanswerable questions” strength (Strength Finder).** This conflicts with the verified major weakness about evaluation validity: the SQuAD v2 NoAns_EM improvement reflects increased abstention, which has not been shown to be equivalent to hallucination reduction, and it coincides with a drop in HasAns_EM.
- **Criticism about EOS token “severely disrupting autoregressive attention.”** The reviewer offers no empirical evidence that this choice caused problems; the paper’s results show some improvements despite the unusual choice, so this concern is speculative.
- **Complaint that the paper is “not even a paper.”** The paper presents a coherent method, experiments, and analysis. It is weak in places but meets the threshold of a research submission.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
1. Compare Delta directly against CAD and DoLA under identical conditions to establish incremental value.
2. Evaluate on at least one open-ended generation benchmark with hallucination-specific metrics rather than relying solely on extractive QA exact match.
3. Report precision/recall for answerable versus unanswerable classification on SQuAD v2, and analyze whether Delta improves calibration or simply biases the model toward abstention.
4. Include standard deviations or run statistical significance tests for the main results.

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/Th6NyL07na.md` (DoLa, avg 7.25, Accept): Strong empirical results across multiple datasets, direct comparison with contrastive decoding baseline, clear analysis. Delta is clearly below this.
- `/home/wg25r/review_agent/human_reviews/aNYabH9Th4.md` (RITUAL, avg 5.00, Withdrawn): Very similar spirit—training-free adaptation of VCD ideas with random perturbations. RITUAL directly compared with existing methods and evaluated on proper hallucination benchmarks (POPE, CHAIR, MME). Delta lacks both, so it falls below this anchor.
- `/home/wg25r/review_agent/human_reviews/dlUjNdybnq.md` (Prior-aware decoding, avg 5.50, Reject): Simple contrastive inference method with a theoretical framework and experiments across 11 models. Delta has no theory and only one model, placing it below this anchor.
- `/home/wg25r/review_agent/human_reviews/zgXGNXkC0F.md` (GACD, avg 4.75, Withdrawn): Gradient-based contrastive decoding with missing baseline comparisons and weak experimental validation. Delta is comparable—simpler method but similarly missing key comparisons and with a weaker evaluation framing.
- `/home/wg25r/review_agent/human_reviews/t15cWqydys.md` (Decoding-free candidate selection, avg 3.00, Reject): Weak protocol, no new methods, missing baselines. Delta is better than this due to its coherent method and some empirical results.

**Reasoning:** Delta is a straightforward adaptation of existing contrastive decoding ideas to text via random masking. Its core weaknesses are (1) an evaluation protocol that does not adequately measure hallucination, conflating extractive QA accuracy and abstention with factuality, and (2) a complete absence of empirical comparison with the most relevant baselines (CAD, DoLA, contrastive decoding). These are significant gaps that place the paper below medium anchors like RITUAL (5.0) and prior-aware decoding (5.5). However, the paper is clearly written, reports negative results honestly, and produces an interesting SQuAD v2 signal, keeping it above the very low anchors (~3.0). Relative to the calibration cluster, a score of **4.5** is appropriate.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>