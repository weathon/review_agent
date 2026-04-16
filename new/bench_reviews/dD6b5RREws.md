## Summary

This paper re-examines the bootstrap rate (BR) hyperparameter in Random Forests, exploring values greater than 1.0 (sampling more observations than the training set size). Across 36 datasets and 18 RF configurations with BR values from 0.2 to 5.0, the authors find that BR > 1 often yields better accuracy than standard BR ≤ 1, contradicting prior work by Martínez-Muñoz & Suárez (2010). They further investigate what determines the optimal BR, conclude it is more a dataset property than a configuration one, and build a binary classifier (81.88%–88.81% accuracy) predicting whether the optimal BR exceeds 1.

## Strengths

- **Addresses a genuinely under-explored and practically relevant hyperparameter.** The finding that BR > 1 can improve accuracy contradicts prior dismissal of this regime and has direct implications for RF implementations in major libraries (scikit-learn, Weka, H2O.ai currently disable BR > 1). The observation that BR = 1 (the default) is rarely optimal (only 2/36 datasets) is striking and practically useful.

- **Broad experimental sweep providing useful empirical characterization.** 36 datasets, 18 RF configurations, 10 BR values, and 400 repetitions per configuration is thorough enough to reveal real patterns. The qualitative analysis of BR curve shapes and the finding that RF(nf_all) behaves differently from other configurations offers genuine insight into bias-variance-diversity tradeoffs in RFs.

- **Thoughtful attempt to connect optimal BR to local class structure.** The k_l statistics and their correlation with optimal BR is a meaningful mechanistic hypothesis — that datasets with more homogeneous neighborhoods benefit from higher BR — even though the correlations are modest.

## Weaknesses

### Major:

- **Overclaiming that BR > 1 "often yields superior results" contradicts the paper's own analysis.** The authors report that the number of datasets favoring BR > 1 versus BR ≤ 1 varies from +5 to −4 to 0 depending on the significance level chosen (lines 143–146), concluding the split is "roughly comparable." Yet the abstract claims "statistically significant improvements" and Section 6 states "BR > 1 often yields superior results." The strongest defensible statement is: "on some datasets, BR > 1 can be optimal; on others, BR ≤ 1 is optimal; the split is roughly even." This is still a useful finding (it overturns the prior blanket dismissal), but the "often superior" framing is not supported by the authors' own statistical analysis.

- **The paired t-test methodology is structurally biased.** Per dataset, the paper selects the single best (configuration, BR) pair and then compares it against all configurations from the opposite BR group. This confuses model selection with hypothesis testing: the "best" is chosen post-hoc, creating selection bias, and comparing one winner against a pool including many suboptimal setups inflates apparent significance. Standard multi-comparison frameworks (e.g., Friedman–Nemenyi or pairwise Wilcoxon with Holm correction across all datasets simultaneously) would be more appropriate. The reported p-values in Table 1 are therefore not trustworthy evidence for the comparative claim.

- **The claim that "optimal BR is more a property of the dataset" is under-supported.** The evidence is qualitative visual inspection of BR curves across only 7 winning RF configurations (all one-at-a-time perturbations of sklearn defaults) on a coarse grid of 10 BR values. The paper never quantifies agreement across configurations — e.g., does the optimal BR rank consistently across configurations for a given dataset? Variance decomposition or a quantitative agreement metric would be needed. Moreover, the winning configuration varies widely across datasets (from nt_500 to ml_5 to nf_all), suggesting hyperparameters actually matter considerably for performance, even if the BR curve "shape" is somewhat consistent.

- **The binary BR predictor is evaluated on too few datapoints with too many features to be convincing.** With 36 (or 24) training points and 12,685 engineered features, even the leave-two-out CV with per-fold feature selection risks severe overfitting. Feature selection on a training fold of 34 observations that nearly overlaps with the full dataset can exploit noise. The reported 81.88%–88.81% accuracies should be viewed as optimistic upper bounds rather than established generalization performance. No baseline comparison (e.g., majority class, simple logistic regression on raw features) is reported, and no confidence intervals are provided.

### Minor:

- **The extreme BR values (0.2 and 5.0) are most frequently optimal, suggesting the search range should be wider.** The paper itself acknowledges this (lines 197–199). If the optimal BR is often below 0.2 or above 5.0, the reported "winners" may be artifacts of the grid boundaries rather than true optima, weakening claims about specific BR regimes.

- **No comparison with training without bootstrapping (using all data).** When BR > 1, each bootstrap sample contains more unique observations. A natural question is whether simply using all N training instances per tree (RF with bootstrap=False) achieves similar gains, which would suggest the benefit comes from more data per tree rather than from the specific BR > 1 mechanism. The absence of this baseline leaves the mechanism unclear.

- **Accuracy on imbalanced datasets is misleading without discussion.** Several datasets have very low accuracy (Abalone: 26.8%, LED Display: 66.6%) or significant class imbalance, yet only accuracy is reported. Whether class imbalance interacts with optimal BR is unexplored.

- **Only classification accuracy is reported; effect sizes are not quantified.** The paper reports which BR wins but not by how much. A 0.1% average improvement, even if statistically significant, may not be practically meaningful. Without reporting the magnitude of accuracy differences, practical significance cannot be assessed.

### Trivial:

- The synthetic example in Figure 3 (class_sep = 1.95 vs 2.0 flipping optimal BR from 5.0 to 0.2) is a nice illustration of sensitivity but uses an extremely narrow perturbation that may not generalize.

## Nice-to-Haves

- **Computational cost analysis.** BR = 5.0 means each tree processes 5× the data. Reporting accuracy-vs-training-time Pareto fronts would help practitioners decide whether BR > 1 is worth exploring. The paper acknowledges this gap but does not address it.

- **Expand the BR search range** (below 0.2 and above 5.0, plus finer intermediate values like 1.5, 2.5) to determine whether the frequent success of extreme grid values is a boundary artifact.

- **Provide a formal or even informal bias-variance analysis** of how BR > 1 affects the ensemble. The intuitive explanation (more unique observations vs. less diversity) could be formalized to predict when BR > 1 should help, rather than relying solely on post-hoc empirical observation.

- **Report effect sizes** (mean accuracy difference between optimal BR > 1 and BR ≤ 1 per dataset) alongside statistical significance, so readers can assess practical importance.

## Removed Points

- **"The paper doesn't compare against BR=1 no-bootstrapping."** This was suggested by Spark but the paper does test BR=1.0 (standard bootstrap). The alternative of no bootstrapping at all is a natural baseline but is outside the paper's stated scope of analyzing the BR parameter. Moved to Nice-to-Have rather than a core weakness.

- **"Dataset scope is limited to small, mostly UCI tabular benchmarks."** This is a generic criticism applicable to many empirical ML papers. The 36 datasets include the 30 from the baseline paper plus 6 additional ones, which is reasonable for an exploratory hyperparameter study. This doesn't undermine the paper's specific claims.

- **"Include XGBoost and neural networks."** This is scope creep — the paper is specifically about Random Forest's bootstrap rate, not a comparison across algorithms.

- **"The BR > 1 benefit might just come from not comparing against alternative uses of the same compute budget."** While valid, this is asking the paper to answer a different question (cost-benefit of BR tuning vs. other tuning axes). The paper's question is specifically about the BR parameter, and this critique is better suited as a future direction.

- **"Reproducibility concerns about code availability."** The paper provides a code link; reproducibility nitpicks are removed per instructions.

## Novel Insights

The observation that BR > 1 can help — and specifically that the benefit appears tied to class-neighborhood homogeneity (datasets with more uniform local structure prefer higher BR) — is genuinely novel. The insight that RF(nf_all) consistently prefers lower BR, consistent with the diversity-information tradeoff (when feature subsampling is removed, more diversity from bootstrapping is needed), provides an elegant confirmation of ensemble theory within the BR framework. This suggests that the optimal BR is indeed linked to an interplay between the two diversity sources in RF, which is a useful theoretical hook even if the paper's own statistical framework cannot fully substantiate it.

## Suggestions

1. **Reframe the primary claim** from "BR > 1 often yields superior results" to "BR > 1 can be optimal on a substantial subset of datasets, contradicting the prior conclusion that it is generally ineffective." This is equally novel but far more defensible.

2. **Replace the per-dataset t-test with a proper multi-comparison framework** (e.g., Friedman test with Nemenyi post-hoc, or Wilcoxon signed-rank with Holm correction) comparing the best BR ≤ 1 vs. best BR > 1 performance across all 36 datasets and configurations simultaneously.

3. **Quantify the consistency of optimal BR across configurations** per dataset (e.g., report the dispersion of winning BR values across the 18 configurations for each dataset, or compute rank correlations), rather than relying on qualitative curve-shape comparisons.

4. **Add the "no bootstrapping" (use all data) baseline** — this is the most natural comparison point for practitioners considering whether to enable BR > 1.

5. **Report effect sizes** — for each dataset where BR > 1 wins, report the mean accuracy difference between the optimal BR > 1 configuration and the best BR ≤ 1 configuration, so readers can judge practical significance.

## Score and Decision

**Calibration anchors:**
- *x8mr9zGkpr* (Attributing Model Behavior: dataset vs. hyperparameter): Empirical study of RF/SVM hyperparameters, limited novelty, overclaims → scores 1–5, rejected. Our paper has similar weaknesses (overclaiming, limited theory) but more focused novelty (BR > 1 is genuinely underexplored).
- *PlZIXgfWPH* (HP Loss Landscapes): More comprehensive empirical study with better methodology → scores 5–8, still rejected. Our paper is less comprehensive.
- *MCjVArCAZ1* (Pre-training vs. Meta-learning): Fair comparison study with marginal effect sizes → scores 3–5, rejected. Our paper's effect sizes are unreported but the comparisons are methodologically flawed.
- *NSCO5QgbTSq* (DynFrs, RF unlearning): Novel method for RF, accepted poster → scores 5–8. Our paper is purely empirical with no method contribution.

This paper makes a worthwhile observation (BR > 1 can help; BR = 1 is rarely optimal) but severely overclaims based on flawed statistical methodology, under-supports the "dataset property" claim, and presents a predictor that cannot be trusted given the evaluation setup. The core finding is useful but the framing needs substantial correction. This places it below typical acceptance thresholds for top venues.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>