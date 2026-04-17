Now I have enough context to write the final review. Let me synthesize everything carefully.

## Summary

This paper studies language model pre-training under the regime where compute is abundant but data is fixed and limited. The authors show that standard data-constrained approaches (more epochs, larger models) overfit, and that aggressive L2 regularization (weight decay ~30× standard practice) restores monotone parameter scaling. They propose comparing scaling recipes by the asymptote of their scaling laws under infinite compute, finding that ensembling independently trained models achieves a lower loss asymptote than single-model parameter scaling. Combining regularization, parameter scaling, and ensemble scaling yields estimated 5.17× data efficiency at 200M tokens, and distillation can transfer 83% of this benefit to a smaller single model.

## Strengths

- **Novel and timely problem framing.** The observation that compute grows ~4×/year while web data grows ~1.03×/year clearly motivates studying the data-constrained, compute-rich regime. The formalization of this setting and the "asymptote" comparison metric is a clean conceptual contribution that directly addresses an under-explored regime.

- **Important practical finding on weight decay.** The discovery that optimal weight decay under data constraints is ~30× larger than the Brown et al. (2020) default of 0.1 is a concrete, actionable, and surprising result, supported by systematic tuning across multiple parameter counts (Figure 3, right table). This alone is likely to influence practice.

- **Well-designed experimental pipeline.** The paper systematically builds from standard overfitting (Section 2) → regularization (Section 3) → ensembling (Section 4) → data scaling (Section 5) → distillation (Section 6) → downstream evaluation (Section 7), providing a complete story. Using validation loss as the primary design metric and holding out all benchmark evaluations until the end is a strong methodological choice.

- **Compelling distillation and self-distillation results.** The finding that 83% of ensemble gains transfer after distillation into an 8× smaller model (Section 6.1), and that self-distillation of a 300M model into a 300M student outperforms the teacher (Section 6.2), are practically important and based on direct experimental comparisons rather than extrapolation.

- **Clear demonstration that standard recipes overfit.** The systematic demonstration that both increasing epochs and increasing parameter count eventually increase loss under data constraints (Section 2, Figures 2) provides a firm empirical foundation for motivating the search for alternative strategies.

## Weaknesses

### Major

- **Core asymptotic claims rely on extrapolation from extremely sparse scaling law fits with no uncertainty quantification.** The headline numbers—regularized asymptote 3.43, ensemble asymptote 3.34, joint asymptote 3.17, 5.17× data efficiency—are derived from power-law fits to just 4 parameter counts (150M–1.4B) and 4 token counts (200M–1.6B), then taking limits as N,K→∞. No error bars, confidence intervals, or goodness-of-fit diagnostics are reported for these fits. Small perturbations to any single data point could meaningfully shift the estimated asymptote, and the paper acknowledges the data scaling laws are "expected to be noisy" (Section 5.3) without quantifying that noise. The qualitative findings—that regularization helps, that ensembling helps, that they compose—are well-supported; the precise quantitative asymptotic claims are not. This is the paper's most significant evidential gap because the framing and abstract center on the asymptote-based metric.

- **The asymptote metric (limit as N,K→∞) is not convincingly established as practically meaningful.** The paper proposes evaluating recipes by lim_{N,K→∞} L, but real systems always face finite compute and inference-cost constraints. The ordering of limits is chosen for "convenience" rather than any principled justification (Section 4.3). For ensembles, K→∞ implies unbounded inference cost, which the paper partially addresses via distillation (83% retention), but the gap between the asymptote-based claims and any finite-budget reality is not systematically analyzed. The paper would be stronger if it showed, for example, compute-vs-loss Pareto frontiers at realistic budgets rather than relying solely on asymptotic comparisons.

- **All experiments are at very small scale (200M tokens, models up to 1.4B), and the claim that gains "persist at higher token counts" relies on extrapolation.** The data scaling laws in Section 5 are fit on just 4 token counts and then extrapolated. The paper does not include even a single experiment at, e.g., 5B or 10B tokens to validate the extrapolation. While the qualitative conclusion that regularization and ensembling help is likely to hold at scale, the quantitative data efficiency factors (2.29×, 5.17×) could change substantially. This contrasts with papers like Gadre et al. (2024), which validated scaling predictions with actual large-scale runs.

- **Downstream evaluation is narrow.** Only three benchmarks (PIQA, SciQ, ARC Easy)—all multiple-choice, all relatively easy for models at this scale—are used to validate that validation loss improvements transfer to downstream capabilities. The 9% average improvement is reported without variance across seeds or runs and without per-benchmark breakdowns in the main text. This is adequate for establishing correlation at this scale but insufficient for the broad claims about "data-efficient pre-training in a compute-rich future."

### Minor

- **The hyperparameter search for ensemble scaling is heuristic.** The joint scaling recipe uses a fixed heuristic of 2× epochs and 0.5× weight decay (Appendix D.4) rather than fully optimized hyperparameters for ensemble members, yet the 3.17 asymptote estimate depends on this choice. No sensitivity analysis is provided.

- **Only weight decay is explored as a regularization mechanism.** Dropout, label smoothing, data augmentation, and stochastic depth are not considered. While L2 weight decay is the most standard choice, the claim that regularization "fixes" data-constrained scaling is specifically a claim about weight decay regularization, and the generality of this conclusion is unknown.

- **The inference cost of ensembles is mentioned but not systematically analyzed.** The paper notes that ensembles cost NK total parameters but does not provide a Pareto analysis of when ensembling versus parameter scaling is preferable at matched inference FLOPs.

### Trivial

- The claim that epoching "contradicts" Muennighoff et al. (2023) slightly overstates the disagreement; that work explicitly acknowledged and filtered overfit runs, which the authors themselves note.

## Nice-to-Haves

- Add confidence intervals or bootstrap-based error bars on the estimated asymptotes and data efficiency ratios to establish whether the differences between recipes are statistically significant.
- Run at least one larger-scale experiment (e.g., 5B tokens) to directly test the data scaling law extrapolation rather than relying purely on fitted power laws.
- Include a compute-vs-loss Pareto analysis at finite budgets showing when ensembling wins over parameter scaling in practice, complementing the asymptotic analysis.
- Expand downstream evaluation to include generative tasks (e.g., lambada, hellaswag) to test whether the validation loss improvements transfer more broadly.

## Removed Points

- **Criticisms about the existence or availability of cited models/data/benchmarks.** All cited entities (DCLM, PIQA, SciQ, ARC Easy, etc.) are assumed to exist per review rules.

- **Criticism that Muennighoff et al. (2023) is "contradicted."** The paper correctly identifies a gap (their functional form doesn't handle overfitting) and the original work acknowledged this; the phrasing could be more precise but is not factually wrong.

- **Demand for comparison with synthetic data generation.** The paper explicitly scopes itself to studying how to leverage compute via epoching, parameter scaling, regularization, and ensembling under fixed data. Synthetic data generation is a different approach (changing the data, not the training recipe) and falls outside the stated scope. This is a reasonable scope choice.

- **Reproducibility concerns about hyperparameter search procedures.** Comprehensive hyperparameter search at every scale is standard in the field; requesting exhaustive search documentation or claiming the results are not reproducible is a nitpick beyond what's standard.

- **Claims that the paper does not consider inference cost at all.** Sections 4.1 and 6 directly address inference cost, and the title and framing explicitly focus on pre-training under infinite compute. The paper acknowledges this and proposes distillation as a remedy.

## Novel Insights

The concept of comparing pre-training recipes by the asymptote of their scaling laws (rather than compute-matched performance) is genuinely novel and directly operationalizes the "infinite compute, finite data" regime in a way that prior scaling law work has not. This reframing makes it possible to ask "what's the best we could ever do with this much data?" as a well-defined question. The finding that this asymptote itself follows a power law in data, and that data efficiency gains are approximately constant across scales (if the power-law exponents match), is an interesting preliminary observation—if validated at larger scale, it could meaningfully inform how to allocate compute in data-limited settings. The self-distillation result (300M→300M improving over the teacher) is also notable and counterintuitive given the model-collapse literature, and aligns with the Allen-Zhu & Li interpretation of self-distillation as implicit ensembling.

## Suggestions

- **Add uncertainty quantification on all asymptote estimates.** Even a simple bootstrap over the 4 data points per fit would reveal whether the claimed differences (e.g., 3.43 vs. 3.34) are within noise. This is the single most important revision.
- **De-emphasize the precise quantitative asymptote claims in the abstract and introduction** and instead foreground the qualitative findings (regularization is critical; ensembles beat parameter scaling; distillation transfers gains), which are well-supported. The current framing creates an expectations gap that the experiments cannot fully meet.
- **Add a finite-budget Pareto analysis** (loss vs. total FLOPs or inference cost) alongside the asymptotic comparison. This would make the findings more actionable for practitioners who are compute-rich but not compute-infinite.

## Score and Decision

**Calibration anchors:**

- *iZeQBqJamf* (Scale reliably with over-training): Scores 6–8, Accept Poster. Extensive experiments (104 models, up to 6.9B params), validated predictions at large scale. This paper is less rigorous in scale but more novel in framing.
- *bmrYu2Ekdz* (PolyPythias): Scores 6–8, Accept Poster. Small-scale but systematic empirical contribution (up to 410M params). This paper's scope and novelty are comparable.
- *xGM5shdGJD* (Hitchhiker's Guide to Scaling Laws): Scores 3–8, Reject. Had methodological issues with scaling law practices. This paper has cleaner methodology but similarly aggressive extrapolation from sparse data.
- *BDisxnHzRL* (Predicting Downstream Performance): Scores 3–6, Withdrawn/Reject. Small models (7B max), limited downstream evaluation. This paper has a more novel framing.

This paper occupies a middle ground: the problem framing, the regularization finding, and the distillation results are genuine contributions. However, the headline quantitative claims (5.17×, specific asymptotes) rely on extrapolation from 4-point fits without uncertainty quantification—a weakness similar to what hurt the rejected scaling law papers. The narrow downstream evaluation and small scale are additional concerns but not fatal to the qualitative story. Compared to the "Language models scale reliably" paper (which validated its laws at scale), this paper makes stronger claims with weaker evidence. Compared to the rejected papers, it has a more novel framing and clearer empirical findings. I place it between the rejected scaling-law papers (3–5 range) and the accepted empirical scaling papers (6–8 range).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>