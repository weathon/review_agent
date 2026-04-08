=== CALIBRATION EXAMPLE 40 ===

# Final Consolidated Review
## Summary

DISCO proposes selecting evaluation samples that maximize inter-model disagreement (measured by JSD or Predictive Diversity Score) rather than sample representativeness, combined with a "model signature" representation fed to simple regressors (kNN/Random Forest) for performance prediction. The approach achieves strong results on MMLU, HellaSwag, Winogrande, ARC, and ImageNet, reducing evaluation cost by >99% with minimal accuracy prediction error, outperforming prior methods like TinyBenchmarks and Metabench at fixed sample budgets.

## Strengths

- **Conceptual reframing from representativeness to disagreement**: The core insight—that samples eliciting diverse model responses are more informative for performance prediction than representative samples—is both simple and powerful. This directly challenges the clustering-based paradigm of prior work (Anchor Points, TinyBenchmarks) with a much simpler greedy selection criterion.
- **Strong empirical results at fixed sample budget**: At K=100 samples, DISCO (PDS + RF) achieves 1.07%p MAE / 0.987 rank correlation on MMLU, substantially outperforming TinyBenchmarks' best (2.08%p MAE / 0.927 rank) and remaining competitive with Metabench which requires 1.5–4.5× more samples (Table 1).
- **Chronological split for realistic evaluation**: Training on older models and testing on newer ones (Section 5.2) is a principled choice that better reflects real-world deployment than random splits, addressing concerns raised in recent literature (Zhang et al., 2025). DISCO remains robust under this split (0.987 rank, Table 2a).
- **Model signatures bypass complex psychometric modeling**: Replacing IRT-based latent parameter estimation with direct regression on output concatenations is practically appealing and empirically superior, achieving better results with simpler machinery.

## Weaknesses

### Major:

- **Proposition 1's injectivity assumption is violated for accuracy, weakening the theoretical optimality claim**: The proof that MI(S(m); ŷ_i) = JSD relies on S(m) being injective with respect to m (i.e., every model has a unique accuracy value). In practice, many distinct models achieve identical accuracy on a benchmark. When S is not injective, the Markov chain ŷ_i → m → S(m) only yields MI(S(m); ŷ_i) ≤ MI(m; ŷ_i) = JSD, so JSD is an *upper bound* on the relevant mutual information, not an equality. The paper does not bound the gap between MI(m; ŷ_i) and MI(S(m); ŷ_i), so the claim that JSD/PDS selection is "information-theoretically optimal" for accuracy prediction is not fully established—it is optimal for *model identification*, which is a different (and stronger) task. This matters because it undermines the theoretical justification for the specific selection criterion, even though the empirical results are strong.

- **Performance degrades sharply under distribution shift between source and target models, and the dismissal may be premature**: Appendix F shows rank correlation drops from ~98.7 (chronological split) to 89.2 when target models substantially outperform source models. The paper dismisses this as "unrealistic" (Section F), but in an era of rapid capability gains, scenarios where newer models significantly outperform older source models are plausible. The paper provides no mechanism to detect or adapt to such shifts. This is a meaningful limitation for a method whose purpose is to evaluate *future* models.

### Minor:

- **Terminology "Condensation" risks confusion with dataset condensation/synthesis literature**: "Dataset Condensation" in current literature (e.g., Zhao et al., 2023; Wang et al., 2018) refers to synthesizing new training data points. DISCO performs subset selection (choosing existing samples), which is traditionally "coreset selection" or "anchor point selection." Using "Condensation" may mislead readers about the technical mechanism.
- **Small target model set limits confidence in generalization claims**: The chronological split uses only 40 target models (Section 5.2). While standard deviations are reported (Appendix D), 40 test points provides limited statistical power for claims about robustness to "future models," especially given the non-i.i.d. nature of the chronological split.
- **Source model subset selection for PDS computation introduces an unexplained hyperparameter**: Appendix I reveals that using all 385 source models for PDS computation yields worse results than a subset (M=100 for MMLU), explained as "redundant models diluting heterogeneity." This means the method requires tuning M as a hyperparameter, and the relationship between ensemble composition and PDS quality is not well-characterized.

### Trivial:

- Inconsistent notation between S_D[f] (Section 3) and S(m) (Proposition 1) for model performance.

## Nice-to-Haves

- Characterize the actual samples DISCO selects (e.g., show 10–20 examples alongside random/Anchor-corr selections). This would make "disagreement-based selection" concrete and help practitioners understand what the method finds informative.
- Cross-dataset generalization experiment: train on MMLU source models, test on HellaSwag target models. This would test whether model signatures capture general capability vs. dataset-specific patterns.
- Per-subtask analysis on MMLU (57 tasks): report prediction error by category (STEM, humanities, etc.) to reveal whether DISCO works uniformly or has blind spots.
- Provide pre-computed anchor points for common benchmarks to lower the adoption barrier for users who cannot afford the offline stage.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Metabench comparison fairness (harsh critic)**: The critic argued the comparison is unfair because Metabench uses more samples (150–450 vs. 100). However, this asymmetry *favors* the baseline (Metabench), not the author's method. Per the rules, this is removed. The paper also transparently marks these results with † and explains the difference in the footnote.
- **Vision baselines not tuned aggressively (harsh critic)**: This is speculative—there is no evidence the baselines were unfairly implemented. Without concrete evidence of mistuning, this is not a legitimate criticism.
- **Missing wall-clock time comparisons (transferable weakness)**: The paper provides GPU-hour comparisons (Appendix B), which is the standard for this type of work. Wall-clock time would be hardware-dependent and less reproducible.
- **Insufficient ablation studies (transferable weakness)**: The paper actually provides extensive factor analysis (Table 2: model split, stratification, source model count, dimensionality reduction, prediction model) and compression rate analysis (Figure 5). This criticism is factually overstated.
- **Limited to multiple-choice tasks (all three reviewers)**: The paper explicitly acknowledges this in Section 6 ("Limitations") and explains the constraint clearly. Per the rules, weaknesses already addressed by the authors should be weakened. This is a known scope limitation, not an unaddressed flaw.
- **Demand for experiments on open-ended generation tasks (spark finder)**: This is outside the paper's stated scope. The paper clearly defines its setting (classification/multiple-choice) and leaves open-ended tasks for future work.
- **Reproducibility concerns about hyperparameters (transferable weakness)**: Implementation details including all hyperparameters are provided in Appendix I. This concern is not applicable.

## Novel Insights

The disconnect between Proposition 1's theoretical claim (optimality for model *identification* via injectivity) and the actual task (predicting a *scalar* accuracy) reveals an interesting open question: samples that best distinguish model identities may be overkill for accuracy prediction. A sample where Models A and B disagree but have the same accuracy provides no information for accuracy prediction despite contributing to JSD. This suggests an even more efficient selection criterion might exist—one that directly targets MI(accuracy; ŷ_i) rather than MI(model_identity; ŷ_i)—potentially allowing further compression or improved prediction with the same budget.

## Suggestions

- Qualify the theoretical claim: Replace "information-theoretically optimal" with "information-theoretically motivated" or add a discussion of the injectivity gap and its practical implications, since the empirical results stand on their own.
- Provide a failure-detection mechanism: A simple confidence metric (e.g., distance of a target model's signature to the nearest source model in PCA space) could flag when DISCO's prediction is unreliable, making the method safer for deployment.
- Release pre-computed anchor sets for MMLU, HellaSwag, ARC, and Winogrande alongside the code. This would make the online stage immediately usable without the 3284 GPU-hour offline investment, dramatically lowering the adoption barrier.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
