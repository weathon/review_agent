Now I have all the information I need. Let me write the final review.

## Summary

The paper studies how to pre-train language models under fixed data and unlimited compute, proposing the **asymptote** of a scaling law (lim_{N→∞} or lim_{K→∞}) as the metric for comparing training recipes. They show that standard approaches (epoching, parameter scaling) overfit under data constraints; that tuning regularization (particularly weight decay 30× larger than standard practice) yields monotone power-law scaling; that ensemble scaling achieves a lower asymptote than parameter scaling; and that a joint recipe combining both reaches an asymptote of 3.17 at 200M tokens—5.17× more data-efficient than the baseline by their metric. Distillation preserves 83% of the ensembling benefit in a smaller model, and improvements transfer to downstream benchmarks.

## Strengths

- **Novel evaluation framework via loss asymptotes**: Proposing the asymptote of a scaling law as the metric for infinite-compute comparisons is a genuine conceptual contribution (Section 3, Figure 1). This is the right metric for the stated problem—when compute is unconstrained, comparing at fixed compute budgets is the wrong question.

- **Concrete and surprising finding on weight decay**: The discovery that optimal weight decay under severe over-parameterization is 0.8–3.2 (30× the standard 0.1 from Brown et al. 2020), enabling monotone power-law scaling at 140× the Chinchilla ratio (Figure 3, Section 3), is practically useful and well-supported by the hyperparameter search.

- **Ensemble scaling outperforms parameter scaling with quantitative scaling laws**: Figure 4 shows ensemble member count K scales with a ~1/K power law, achieving a strictly lower asymptote (3.34 vs. 3.43 for parameter scaling at 300M). The qualitative finding that "it is better to train multiple small models than one large model" is directly observable in the data and does not depend on extrapolation.

- **Non-asymptote evidence for data efficiency**: The paper provides direct (non-extrapolated) evidence: "the best 1.4B model at 200M tokens is 2.09× more data efficient than our baseline" (Section 5.1) and "our best ensemble of five 1.4B models is itself 3.75× more data efficient" (Section 5.2). These are substantial gains independent of asymptote estimation.

- **Correction of prior scaling law**: The paper correctly identifies that Muennighoff et al. (2023)'s functional form fails because it predicts monotone loss decrease with epoching, while actual runs overfit—acknowledged but sidestepped in that work's Appendix D (Section 2.1, Figure 2).

- **Pre-registered downstream evaluation**: Benchmarks (PIQA, SciQ, ARC Easy) were selected before seeing results (Section 7), providing a strong test of generalization from validation loss improvements.

## Weaknesses

### Fatal
None.

### Major

- **Asymptote estimates are under-constrained, making quantitative claims fragile**: The core scaling law fits use 4 parameter counts (150M, 300M, 600M, 1.4B) for a 3-parameter model (A, α, E), leaving 1 degree of freedom. Two of the four points (150M, 300M) are close in parameter count and contribute little to constraining the long-range behavior. The asymptote E is extremely sensitive to the fit parameters—small changes in α shift the extrapolated limit substantially. Every headline number (3.43, 3.34, 3.17, 5.17×) depends on these extrapolations. While the qualitative ranking (ensemble < regularized < standard) is supported by direct observations, the precise quantitative claims are not reliable at this level of constraint. The paper does not report confidence intervals or sensitivity analysis on the asymptote estimates, which would be essential for trusting these numbers. **Why it matters**: The paper's abstract leads with "5.17× less data" and "3.17 asymptote"—these are the claimed contributions. If the asymptote estimates have large uncertainty, the paper overclaims.

- **Experiments are at a scale far below where the stated problem applies**: All primary results use 200M–1.6B tokens with models up to 1.4B parameters. The paper's framing concerns a "compute-rich future" where compute vastly exceeds data, which would manifest at scales orders of magnitude larger. The data scaling law analysis (Section 5) uses only 4 token counts (200M–1.6B), again with 3-parameter fits. The claim that "data efficiency improvements persist at higher token counts" (Section 5.3) rests on exponents being similar (0.23–0.24) across recipes, but with 4 noisy points per recipe, this is weak evidence. The finding that optimal weight decay is 30× standard practice may be scale-specific. **Why it matters**: The paper frames itself as studying the future of pre-training, but provides no evidence that its findings generalize beyond the small-scale regime it studies. The 9% downstream improvement is at 200M tokens against an unregularized baseline.

- **Self-distillation comparison does not isolate the mechanism (Section 6.2)**: The self-distilled 300M student is trained on D + D′ tokens (real + synthetic) while the teacher was trained on D tokens. The paper attributes the improvement to "self-distillation" and "implicit ensembling" (citing Allen-Zhu and Li 2023), but the comparison conflates two effects: (a) the benefit of synthetic data specifically, and (b) the benefit of seeing more total training data. A proper control—training a 300M model on D tokens with equivalent extra compute spent on additional epochs with tuned regularization—is absent. **Why it matters**: Without this control, it is unclear whether self-distillation provides any benefit beyond simply training longer on more (synthetic) data, which weakens the claim that "self-distillation vastly outperforms the teacher" as a distinct phenomenon.

### Minor

- **Misleading exponent comparison with Chinchilla (Section 3)**: The paper compares its parameter scaling exponent (1.02) with Chinchilla's (0.34), stating "this suggests that when we better leverage the data, there is faster improvement from larger models." These exponents measure fundamentally different regimes: the paper's exponent is from fixed D with varying N, while Chinchilla's is from jointly scaled D and N. The two are not directly comparable, and the implied claim of "faster improvement" from the exponent difference is misleading.

- **Joint scaling recipe uses heuristic hyperparameters (Section 4.3)**: The inner limit (K → ∞) uses "the heuristic of taking the optimal regularized hyperparameters with 2× epochs and 0.5× weight decay" rather than properly tuned hyperparameters, acknowledged as due to "experimental constraints." The 3.17 asymptote estimate may be sensitive to this choice. The paper is transparent about this, but the sensitivity is not analyzed.

- **Near-identical data scaling exponents may be coincidental (Section 5.3)**: The data scaling exponents across recipes are 0.23–0.24, which the paper treats as meaningful evidence that improvements persist. With only 4 noisy data points per 3-parameter fit, near-identical exponents could easily be coincidence rather than a robust pattern.

### Trivial
None.

## Nice-to-Haves

- Report confidence intervals or bootstrap error bars on the fitted asymptote E to quantify the reliability of quantitative claims.
- Test alternative scaling law functional forms (e.g., logarithmic) to assess whether the ranking of recipes depends on the assumed power-law form.
- Provide at least one experiment at a meaningfully larger scale (e.g., 10B+ tokens) to support generalization claims.
- Add a control for self-distillation: train a 300M model on D tokens with additional epochs (matching the compute budget of the self-distilled student) to isolate the effect of synthetic data.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Experiments at non-trivial scale required (10B+ tokens, 1B+ parameter models)"**: The harsh critic demands experiments at 10B+ tokens. While the scale gap is real (Major weakness above), demanding a specific scale target is a nice-to-have, not a requirement for acceptance. The paper does provide evidence at 4 token counts up to 1.6B, and the data scaling laws are a reasonable (if noisy) attempt at extrapolation.

- **"6–8 parameter counts spanning up to 5B–10B"**: Same as above—requesting a specific number of data points or scale is a nice-to-have. The paper's 4-point fits are a legitimate concern, but specifying exact targets is overly prescriptive.

- **"9% improvement is relative to an unregularized model—a weak baseline"**: The comparison is to the standard recipe (which is the paper's baseline), and the paper explicitly shows that the standard recipe overfits without regularization. The baseline is not "weak" in the sense of being artificially bad—it is the natural starting point that practitioners currently use.

- **"Only three benchmarks at 200M scale with no external baseline at similar scales"**: The benchmarks are standard for models at this scale (citing Thrush et al. 2025), and were pre-registered. Requesting more benchmarks or external baselines is generic and would not change the conclusions.

- **"Discussion does not acknowledge limitations about extrapolation or scale"**: While the Discussion section (Section 9) is brief, the paper does acknowledge limitations throughout—e.g., "data scaling laws are expected to be noisy" (Section 5.3), the heuristic for the inner limit (Section 4.3), and the small default token count (Section 2).

- **"Functional form validation needed—test logarithmic scaling"**: This is a reasonable suggestion but is a nice-to-have. The power-law form with asymptote is the standard in scaling law literature (Kaplan et al., Hoffmann et al.), and the paper's qualitative findings don't depend on it.

- **"Predicted vs. observed loss at intermediate scales"**: This is a presentation suggestion that wouldn't change the evaluation.

- **Strength removed: "Open-source reproducibility"** — while commendable, this is a generic strength that doesn't specifically support the paper's core claims.

- **Strength removed: "Data scaling laws predict persistent data efficiency gains"** — this is partially undercut by the Major weakness about under-constrained fits; the "prediction" relies on 4-point, 3-parameter fits that are too noisy to strongly support this claim as a standalone strength.

## Novel Insights

The paper's most insightful observation—that ensemble scaling achieves a lower loss asymptote than parameter scaling under data constraints—has a clean theoretical intuition (different ensemble members learn different "views" of the data per Allen-Zhu and Li 2023) and direct empirical support. This inverts the common assumption that parameter scaling is always the most effective use of compute: when data is the bottleneck rather than compute, spending compute on diversity (multiple independent models) outperforms spending it on capacity (a single larger model). The practical implication—that even K=3 ensembles beat the regularized recipe's asymptote—is a striking and actionable finding.

## Suggestions

- Add uncertainty quantification (bootstrap CIs) on all asymptote estimates and data efficiency ratios; this would directly address the biggest concern and is straightforward to compute from existing data.
- Add a compute-matched control for self-distillation: train a 300M model on D tokens for additional epochs with tuned regularization, spending the same compute as the self-distilled student. This would isolate whether self-distillation provides value beyond "more training."
- Tone down the abstract's quantitative claims: lead with the qualitative finding (ensembles beat parameter scaling under data constraints) and the directly observed 3.75× data efficiency, rather than the extrapolated 5.17× figure.

## Calibration

**Anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Scaling with Collapse | 3YKeB9R1g9 | 8.0 | More convincing at scale; loss curve collapse at practical LLM scales. This paper is weaker due to small-scale experiments and under-constrained fits. |
| Theory of Data Curation | 8KcjEygedc | 7.5 | Stronger theoretical grounding with exact scaling law curves validated on ImageNet. This paper has less theory but more practical recipe findings. |
| Scaling Laws with Weight Decay | Q3yLIIkt7z | 7.0 | Theoretical phase diagrams for weight decay scaling. More rigorous but narrower in scope. This paper is broader but less rigorous. |
| Distilled Pretraining | PNm2dl7HcY | 5.5 | Similar topic (distillation + pretraining) at larger scale but with methodology gaps. Comparable quality. |
| Reasoning Scaling Laws | v3mJ4f4Mnc | 4.4 | Novel synthetic testbed with limited generalizability. This paper has a more realistic setting and more practical findings; clearly above this. |
| Scale-time Equivalence | WB2ejxmIFt | 2.0 | Unrealistic assumptions invalidating theory entirely. This paper is far better—qualitative findings are directly supported by data. |
| Unified Neural Scaling Laws | dnuIoVjeGR | 3.0 | Overly expressive functional form lacking grounding. This paper's power-law assumption is standard and its qualitative findings are scale-independent. |

This paper sits above the low-scoring papers (which had fundamental theoretical flaws or completely unsupported claims) and below the high-scoring ones (which had stronger theory or larger-scale validation). It is comparable to medium-scoring papers like Distilled Pretraining (5.5) and the reasoning scaling law paper (4.4), but with a more novel conceptual contribution (the asymptote framework). The quantitative overclaiming and scale limitations pull it down from a higher score.

**Originality**: The asymptote-based evaluation framework for infinite-compute comparisons is genuinely novel. The combination of regularization + ensemble scaling + distillation under data constraints is a new and well-structured investigation.

**Importance**: The research question is timely and important—compute growing 4×/year vs. data at 1.03×/year means the data-constrained regime is approaching.

**Claims support**: Qualitative claims are well-supported; quantitative claims (5.17×, 3.17 asymptote) depend on under-constrained extrapolations.

**Soundness of experiments**: Careful hyperparameter search, pre-registered benchmarks, and multiple scaling law analyses. Limited by scale and number of data points for fits.

**Clarity**: Well-written with clear problem formulation and logical progression through recipes.

**Value to community**: High—practitioners will find the weight decay finding and ensemble scaling result immediately useful, and the asymptote framework is a valuable conceptual tool.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>