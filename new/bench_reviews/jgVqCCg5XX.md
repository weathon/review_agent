Now let me look at the calibration papers to score this work appropriately.Now I have enough information to write the final consolidated review.

---

## Summary

This paper investigates how training corpus size versus model parameter size affects LLM performance on medical reasoning tasks of varying complexity. The authors introduce MedResEval, a benchmark derived from MedQA by reformulating questions along three dimensions (reducing available clues, expanding decision space, increasing reasoning steps) plus a diagnosis-simulation task. Across 18 LLMs from Llama and Qwen families, they find that while data-scaled small models can match older large models on simple tasks, they consistently underperform on complex reasoning tasks, and they propose a "difficulty-dependent scaling law" to quantify this gap.

---

## Strengths

- **Well-motivated, timely research question.** The paper targets a practically important issue: whether aggressively data-scaling small models suffices for complex, real-world reasoning in high-stakes domains. The finding that Llama3-8B (15T tokens) approaches Llama1-65B (1.4T tokens) on MedQA but still lags substantially on harder task variants is both clear and informative.

- **Principled benchmark construction.** Using MedQA as a single source and generating diverse task reformulations (clue reduction, decision space expansion, multi-step consistency, diagnosis simulation) prevents the knowledge-domain shift confound that plagues multi-source benchmarks. The three-factor decomposition of reasoning complexity is a sensible framework.

- **Comprehensive empirical coverage.** Evaluating 18 LLMs across multiple generations of two prominent families (Llama and Qwen), at two size tiers, with both base and instruction-tuned variants, plus GPT-4o/GPT-4o-mini reference points, provides reasonably broad coverage for a single-domain study. The qualitative patterns — large models consistently outperforming small ones on complex tasks regardless of data scaling — are robust across both families.

- **Task-wise normalization.** Equation 4's normalization against random-guess and assumed-maximum performance is thoughtful and addresses the comparability problem arising from different task formats and class structures.

- **Quantitative attempt at a scaling law.** Rather than stopping at qualitative observations, the authors fit a power-law formula and derive specific extrapolations (e.g., that a 7B model would need ~157T tokens to match a 70B model on complex tasks). Even if the fit is fragile (see below), the attempt to quantify the gap is a meaningful contribution.

---

## Weaknesses

### Fatal
None. The qualitative core finding — small data-scaled models still lag large models on harder reasoning task variants — is robustly supported by the raw experimental comparisons, and this observation is genuinely interesting.

### Major

- **Scaling-law fit is based on severely sparse data and undisclosed outlier exclusion.** The power-law curves in Table 1 are each fit by pooling Llama and Qwen across the same size class (e.g., all "~7B" models regardless of family). But each family contributes only 3 data points (Llama: 1.4T, 2T, 15T; Qwen: 3T, 7T, 18T), giving at most 6 total points per size-difficulty combination for a 3-parameter fit. Fitting a 3-parameter model (P_MAX, D_0, α) to 3–6 noisy points on log–log axes will nearly always yield a high R². More critically, footnote 5 states only "some outliers were not considered in the fitting" without specifying which models were excluded or why. Inclusion or exclusion of a single point can materially shift the exponent. Yet the paper builds its central quantitative claims ("error reduction rates are 1.3× greater… 2× greater") entirely on these exponents, and extrapolates the fit to 157T tokens — two orders of magnitude beyond the data range (1–18T). These precise numerical claims are not reliably supported by the evidence as presented. The paper over-claims predictive structure from what is essentially a handful of benchmark points with selective curve-fitting.

- **Model generation is deeply confounded with model size, undermining causal claims.** The paper's headline framing — "greater emphasis must be placed on model parameter scales" — treats parameter count as the causal variable. But the comparisons conflate parameter count with generation improvements: Llama1/2/3 and Qwen/Qwen1.5/Qwen2/Qwen2.5 differ in tokenizer, architecture, training recipe, data curation quality, and post-training pipeline, not just data volume. The paper acknowledges this partially ("Llama3… possibly due to other factors such as training data quality"), but then proceeds to make generation-agnostic universal claims in its abstract and conclusion. Without within-generation, within-family controlled comparisons (e.g., Llama3-8B vs. Llama3-70B, both trained on 15T tokens), the causal attribution of the performance gap to parameter count — rather than data quality or architectural improvements — is not supported. The supported claim is more modest: within these particular families and data regimes, data-scaling 7B models does not close the gap to ~70B models on complex task variants.

### Minor

- **Only two model size tiers studied.** The analysis is restricted to ~7B and ~70B models. Without intermediate checkpoints (13B, 30B, 40B), it is impossible to determine whether the observed effects follow a smooth trend or a threshold, which directly bears on the scaling-law framing. Footnote 3 acknowledges this but understates its impact on the conclusions.

- **No uncertainty quantification on performance differences.** No confidence intervals, standard errors, or statistical tests are reported for any performance number. This matters especially on complex tasks where normalized scores are extremely low (e.g., Llama 7B on complex tasks: 3.6%), making it unclear whether small observed differences are statistically meaningful.

- **Anomalous per-task behavior is unexplained.** Figure 5 shows the expanding-decision-space task performance for Llama ~7B *decreasing* with more data (4.9 → 1.9) while 70B jumps dramatically. Likewise, the increasing-reasoning-steps task shows negative normalized performance (below random) for 7B models at both data scales (-0.7, -2.0). These anomalies suggest possible prompt design or label balance issues but are not analyzed; they cast doubt on whether the "complexity" grouping cleanly captures reasoning difficulty versus task-construction artifacts.

- **Instruction tuning analysis is limited.** Table 2 evaluates only four Qwen instruction-tuned variants and no Llama-instruct models, yet draws the broad conclusion that "instruction tuning does not reduce the performance gap between models with different parameter scales." Adding at least Llama3-instruct (8B vs. 70B) would substantiate this claim.

- **MedQA data contamination risk is unaddressed.** MedQA is one of the most widely benchmarked datasets for LLMs and may appear in pre-training corpora of some evaluated models. This could systematically inflate simple task performance in ways that differ by model generation, introducing an additional uncontrolled confound.

### Trivial

- The conclusion states "scaling the model size *always* leads to a more pronounced performance improvement" — "always" is too strong given only two families and two size tiers. The claim should be scoped appropriately.

---

## Nice-to-Haves

- Adding intermediate model sizes (13B, 34B) from at least one family would allow verification of whether the scaling effect is smooth or threshold-based, substantially strengthening the scaling-law narrative.
- Including a human clinician baseline on MedResEval tasks, even at small scale, would contextualize the normalized performance scores and provide ecological validity anchoring.
- A per-task error breakdown (failure modes: reasoning errors vs. format misunderstandings vs. contradiction detection failures) would help distinguish whether the parameter-scale gap reflects genuinely deeper reasoning capacity or robustness to task reformulation.
- Per-task scaling curves (rather than only the aggregated simple/complex split) would let readers verify the claimed pattern holds at the individual-task level.
- Testing the scaling-law formula on one non-medical domain (e.g., logical reasoning or MATH) would strengthen the claim that the difficulty-dependent scaling is not domain-specific.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Missing related works (all reviewers):** Removed per hard rules — no external sources to confirm existence of specific prior works the reviewers might have had in mind.
- **Lack of evaluation on non-medical domains (Spark):** This is scope creep. The paper is explicitly framed as a medical case study; evaluating on MATH or GSM8K is a reasonable next step but not a flaw in a paper that explicitly scopes itself to medicine.
- **Ecological validity of MCQ-based reformulations (Human Finder, Neutral):** The paper does not claim MedResEval fully replicates clinical practice — it explicitly says it "more accurately reflects real-world scenarios" while acknowledging the limitations of the MCQ base. Weakening to a note that more naturalistic tasks would further validate the findings.
- **Reproducibility concerns (hyperparameters, CoT details):** Removed per hard rules — self-consistency details, temperature settings, etc. are standard implementation details.
- **Request for human baseline as a mandatory requirement (Human Finder):** Reasonable as a nice-to-have, but not standard for scaling-law papers. Moved to Nice-to-Haves.

---

## Novel Insights

The most genuinely novel observation in the paper — and the one that survives after filtering methodological concerns — is the stark asymmetry in how model size interacts with reasoning complexity: data scaling can close the gap between small and large models on standard and semantically varied benchmarks, but leaves a large, persistent gap on task variants that require reducing available clues, expanding decision space, or chaining reasoning steps. This challenges the narrative that recent data-heavy small models (e.g., Llama3-8B on 15T tokens) are approaching full capability parity with 70B models. The diagnosis-simulation task, which combines contradiction detection, information revision, and multi-class open-set diagnosis, provides an especially clear case where parameter scale matters even after instruction tuning. The normalization framework (Eq. 4) enabling cross-task comparison is a useful methodological contribution for the medical AI community.

---

## Suggestions

1. **Reframe the scaling-law claims.** Replace the strong quantitative ratios ("1.3×," "2×") with interval estimates derived from bootstrap resampling of the available data points. Clearly disclose which specific model checkpoints were treated as outliers. Consider demoting the scaling-law formula from a central claimed contribution to an illustrative approximation.
2. **Strengthen causal isolation.** Add a within-generation, within-family controlled comparison as primary evidence (e.g., Llama3-8B vs. Llama3-70B, both at 15T tokens). Present cross-generation comparisons as supplementary context rather than primary evidence for the parameter-scale claim.
3. **Report per-task scaling curves separately** before aggregating into simple/complex groups. This would let readers assess whether the claimed pattern holds uniformly or is driven by specific task variants.
4. **Explain anomalous per-task results** (especially the decreasing performance of 7B Llama on expanding-decision-space with more data and the negative normalized performance on two-step reasoning). These require either empirical investigation or at minimum a methodological explanation.
5. **Tone down "always" and "essential" in conclusion language** to reflect the actual scope: "in the Llama and Qwen families, at 7B and 70B scales, within our task framework."

---

## Score and Decision

**Calibration comparisons:**

- *jjfve2gIXe* (U-shaped emergent abilities, Accepted Poster, scores 6/6/6/8): This paper also studies difficulty-stratified scaling but uses 56 LLMs, provides a clearer mechanistic explanation, and delivers a working predictive pipeline. Our paper uses ~6 data points per fitted curve, has no clear mechanism, and has strong confounds. It is substantially weaker than this accepted paper.

- *D5v491uCzm* (Sloth, Rejected, scores 5/6/3/3): This paper also fits scaling laws across model families and is rejected partly for the same reason — differences in training configurations and data processing confound cross-family scaling analysis. Our paper shares this weakness but adds a concrete (if imperfect) benchmark as a contribution, and its qualitative findings are cleaner.

- *BDisxnHzRL* (Scaling Law Prediction, Withdrawn, scores 5/6/3/3): Shares the sparse-data-points-for-scaling-law concern, and was rejected/withdrawn at a similar weakness level.

- *E3LDsbUSRZ* (CliBench, Rejected, scores 3/5/6/3/5): Medical benchmark paper with limited methodology. Our paper has more coherent experimental framing and a more interesting finding, placing it above this.

**Assessment:** The paper has a genuine and interesting core observation, a reasonable benchmark design intuition, and moderately broad empirical coverage. However, the two major weaknesses — the near-unfounded precise quantitative scaling claims and the confounded causal attribution — undermine its primary framing as a scaling-law discovery paper. The qualitative findings are solid but not novel enough to carry the paper on their own. Relative to the calibration papers, this paper sits between D5v491uCzm (rejected, ~4 avg) and jjfve2gIXe (accepted, ~6.5 avg). Given its stronger qualitative findings than D5v491uCzm but much weaker quantitative rigor than jjfve2gIXe, and considering that the major weaknesses cannot be fully resolved in a rebuttal (they are structural), I place this paper at **4.5**.

**Originality:** Moderate. The benchmark construction idea is novel; the scaling-law framing is borrowed and under-supported.  
**Importance of research question:** High. Whether data scaling can substitute for parameter scaling in complex reasoning is practically critical.  
**Support for claims:** Weak for the quantitative claims; moderate for the qualitative ones.  
**Soundness of experiments:** Moderate — good scope, but thin data for curve fitting and significant confounds.  
**Clarity of writing:** Good — well-organized and readable.  
**Value to research community:** Moderate — the benchmark and qualitative observation are useful; the scaling-law component needs substantial revision.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>