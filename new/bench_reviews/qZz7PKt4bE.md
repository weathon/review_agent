Now let me search for calibration papers to properly score this work.Now I have enough to write the full review. Let me compile my findings.---

## Summary

The paper proposes AutoTune, a framework for adapting pre-trained time series transformers (specifically Chronos T5 Mini) to target domains by combining Low Rank Adaptation (LoRA) with Limited Discrepancy Search (LDS) for hyperparameter optimization, distributed via Ray Tune. The key claim is that this combination efficiently finds optimal LoRA configurations within a budget of 10 trials, yielding competitive or superior performance to full fine-tuning, with an average MASE improvement of 5.21% over zero-shot inference across 10 out-of-domain benchmark datasets.

---

## Strengths

- **Evaluation breadth and variance reporting**: The 10 benchmark datasets span 6 distinct domains (transport, energy, finance, economics, retail, nature) from the Monash Repository, all excluded from Chronos pre-training (Table 1). Results are averaged over 5 runs with standard deviations reported consistently in Table 3 — this is good empirical practice.

- **Clear algorithmic specification**: Algorithm 1 provides complete pseudocode for the LDS-based search, and Figure 2 illustrates the search tree traversal with a concrete example (3 binary variables, discrepancy = 1), making the method understandable.

- **Autotuned Mini model beats larger zero-shot models on select datasets**: Table 4 shows the 20M-parameter autotuned Mini outperforms the 710M-parameter zero-shot Large on Australian Electricity (0.831 vs 1.411) and Exchange Rate (1.631 vs 2.214), and beats the zero-shot Small model on 6/10 datasets. While this result is expected from the literature, it is concrete and correctly demonstrated.

---

## Weaknesses

### Fatal
None that invalidate results outright, but the following Major weaknesses collectively undermine the paper's core claims.

### Major

- **The central algorithmic contribution (LDS) is completely unevaluated**: The paper's novel claim is that LDS is an *efficient* search strategy for LoRA hyperparameters. Yet the experiments include no comparison against (a) random search with the same 10-trial budget, (b) grid search, or (c) default LoRA hyperparameters with no search at all. Every result in Table 3 compares "autotune" against zero-shot inference and full fine-tuning with *fixed* hyperparameters. We cannot determine whether LDS outperforms selecting 10 random LoRA configurations, or whether any LoRA configuration—even the default—would achieve nearly the same result. This is the single most critical missing experiment in the paper.

- **Comparison against full fine-tuning is asymmetric in favor of the proposed method**: The autotune method receives an HPO budget of 10 trials (searching over learning rate, batch size, gradient accumulation steps, and LoRA-specific hyperparameters), while full fine-tuning is evaluated with *fixed* hyperparameters (no HPO budget). This is not a controlled comparison. The headline result—autotune outperforms full fine-tuning on some datasets—cannot be attributed to LoRA or LDS specifically; it may simply reflect the benefit of any hyperparameter search versus no search.

- **Efficiency claim is entirely unsupported**: The abstract promises "strong performance-cost trade-offs" and the introduction emphasizes computational efficiency as a central motivation. The paper never reports wall-clock time, trainable parameter count, or memory usage comparisons between autotune, full fine-tuning, or zero-shot inference. All experiments are run on a MacBook Pro M3 Max. Without this evidence, the efficiency claim—which is part of the paper's identity—is unsubstantiated.

- **Single-model, single-size evaluation**: The paper states LoRA autotuning is "highly transferable across different target domains" and "can be easily extended to other time series foundation models," but evaluates only one model (Chronos T5) at one size (Mini, 20M parameters). No other architecture is tested. The generalizability claim is asserted rather than demonstrated.

### Minor

- **Factual error in metric name**: Section 4 states "we use mean absolute *squared* error (MASE) as the evaluation metric." MASE is Mean Absolute *Scaled* Error, a distinct metric. This error recurs throughout the paper and introduces uncertainty about what was actually computed, even if the Ansari et al. (2024) implementation was followed.

- **Algorithm 1 pseudocode bug**: The SCORE procedure (line 24) uses `y*` (the global best) inside `TrainModel` instead of the passed argument `y`. This is inconsistent with the procedure signature `SCORE(y, X_train, X_val, M)` and may indicate a logic error in the presented pseudocode (though the actual implementation may be correct).

- **Contradiction about in-domain vs. out-of-domain status**: Section 4 states "We use these datasets as they have not been used in the pre-training phase of the Chronos T5 models." But Section 5 explains that autotune underperforms on traffic, weather, and electricity because "the pre-trained Chronos T5 model has seen datasets from the aforementioned domains during the pre-training phase." The paper does not resolve which characterization is correct, and the claimed "out-of-domain" MASE improvement of 4.76% depends on this categorization.

- **Average improvement is dominated by a single outlier**: The reported 5.21% average MASE improvement (Figure 3) is heavily skewed by the Exchange Rate dataset (20.59%). Three datasets show *negative* improvement: FRED-MD (−7.82%), ETT (Hourly) (−0.13%), ETT (15 min.) (−0.56%). This skew is not disclosed and makes the headline average somewhat misleading.

- **Conclusion overstates performance**: The conclusion states autotune "outperforms full fine-tuning specifically for out-of-domain datasets." From Table 3: autotune beats full fine-tuning on 5 datasets, full fine-tuning beats autotune on 4 datasets, with 1 tie. Selectively characterizing this as "outperforms" without acknowledging the near-parity is an overclaim.

### Trivial

- The comparison of a fine-tuned small model to larger zero-shot models (Table 4) is framed as a contribution, but fine-tuned smaller models routinely outperform larger zero-shot models — this is well-established. It does not require a new algorithm to demonstrate and should not be elevated as a headline result.

---

## Nice-to-Haves

- **Performance vs. trial-count curve**: Showing how MASE evolves across the 10 trials would validate whether 10 is a sufficient budget and help understand LDS's convergence behavior.
- **LDS traversal path visualization**: Showing which configurations LDS visits vs. random search would help justify the structured-search design choice.
- **Sensitivity to max discrepancy**: The paper tests discrepancy values of 4 and 8 but does not report results broken down by discrepancy value.
- **Multivariate extension**: The paper mentions this as future work; even a preliminary result would strengthen the generalizability claim.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Missing related works on HPO (BOHB, Optuna, PBT)**: Per the review guidelines, missing related work is not cited as a weakness, since external sources cannot be confirmed.
- **Comparison against larger full-fine-tuned models is invalid** (Harsh Critic framing): The fact that autotune uses LoRA (fewer parameters) while full fine-tuning trains all weights is actually an *advantage being demonstrated* for the proposed method. Criticizing this asymmetry as "invalid" misreads the purpose — the paper is precisely trying to show LoRA + LDS is competitive despite fewer trainable parameters.
- **Strength: "LDS achieves strong results within 10 trials"** (Strength Finder): This is removed as a strength because without a random search comparison, we have no basis for concluding LDS per se achieves strong results — any 10 random LoRA configurations might perform equally well. The claim is circular.
- **Strength: "Distributed implementation for practical scalability"**: This is a system choice (using Ray Tune) rather than a research contribution; removed as generic.

---

## Novel Insights

The reviews surface one genuinely useful observation beyond the paper's own claims: the in-domain/out-of-domain contradiction reveals that the paper's framing of where LoRA + LDS helps most is inconsistent and not cleanly supported by the experimental design. A cleaner definition of domain overlap with the pre-training corpus, paired with a systematic analysis of when PEFT outperforms full fine-tuning as a function of domain shift magnitude, would be a meaningful contribution that the current paper gestures at but fails to deliver.

---

## Suggestions

1. **Add a random search baseline with 10 trials**: This is the single most important experiment the paper is missing. Compare LDS-10 against Random-10 and Default (no search) on all 10 datasets.
2. **Report efficiency metrics**: Wall-clock time per trial, total search time, and trainable parameter counts must accompany the efficiency claim.
3. **Fix the MASE acronym**: Consistently use "Mean Absolute Scaled Error" throughout.
4. **Fix Algorithm 1 line 24**: Replace `y*` with `y` in the `TrainModel` call within the SCORE procedure to match the procedure's signature.
5. **Clarify in-domain vs. out-of-domain categorization**: Either cite the Chronos pre-training data distribution precisely, or remove the domain-based explanation of results.
6. **Test at least one additional model architecture or size**: Even Chronos T5 Small would extend the scope claim.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Comparison |
|------|-----------|------------|
| `/home/wg25r/review_agent/human_reviews/xTrAA3UKPa.md` | 2.00 | SWGA: Distributed HPO for time series; rejected for narrow comparison (only vs. GA baseline), minor contributions, no convergence evidence. Very similar weakness profile to this paper. |
| `/home/wg25r/review_agent/human_reviews/igGeaxOiFM.md` | 3.00 | HoLoRA: LoRA variant with insufficient differentiation; all scores 3. Lacks ablations and strong baseline comparisons. |
| `/home/wg25r/review_agent/human_reviews/bYsieh8LE2.md` | 3.67 | ULoRA: LoRA generalization paper; rejected for limited novelty and weak baselines. Slightly stronger than this paper in experimental design. |
| `/home/wg25r/review_agent/human_reviews/l80AgHoRaN.md` | 4.33 | LoRA-XS: Has clearer theoretical motivation and broader comparison; represents the borderline of borderline-accept territory. |
| `/home/wg25r/review_agent/human_reviews/dCcY2pyNIO.md` | 6.25 | In-context TSP: Strong positive result, clearly novel architecture change, competitive comparisons — well above this paper. |
| `/home/wg25r/review_agent/human_reviews/tqh1zdXIra.md` | 8.00 | Quick-Tune: Joint HPO + model selection with 20k+ configurations, meta-dataset, thorough ablations — the high ceiling for this type of work. |

**Assessment**: This paper's weakness profile is most similar to SWGA (2.00) — a distributed HPO method for time series that was rejected for combining existing techniques without isolating the search contribution, narrow comparison, and missing convergence/efficiency evidence. The paper under review is marginally stronger (more datasets, variance reporting, cleaner presentation) but has the same central flaw: the HPO algorithm (LDS) is never compared to simpler alternatives, making the core claim unverifiable. It also lacks efficiency metrics and tests only one model. The LoRA-XS paper (4.33) represents a better-justified paper in the same general space — this paper does not reach that bar.

**Originality**: Low — combines LoRA + LDS + Ray Tune without modification to any component.
**Importance of research question**: Moderate — domain adaptation for time series transformers is relevant.
**Support for claims**: Poor — the key claim (LDS efficiency) has no experimental support.
**Soundness of experiments**: Weak — missing critical baseline, asymmetric comparison, no efficiency measurement.
**Clarity**: Adequate but contains verified factual errors and pseudocode inconsistency.
**Value to community**: Low in current form — the combination is straightforward and results are mixed.

**Score: 3.0** — Below the borderline-reject cluster (3.5–4.5) because the single most critical experiment (random search comparison) is missing, the efficiency claim is entirely unsupported, and the paper tests only one model. The paper is not fraudulent or without effort, but it fails to substantiate its central claims at the level expected for ICLR.

MY FINAL SCORE: <pineapple>3</pineapple>
MY FINAL DECISION: <orange>Reject</orange>