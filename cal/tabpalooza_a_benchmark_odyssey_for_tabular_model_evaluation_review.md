=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary

TabPalooza proposes a new benchmark for tabular data classification and regression, constructed to maximize two competing objectives: **diversity** (measured via rank reconstruction error against existing benchmarks) and **efficiency** (limiting dataset count). The authors aggregate datasets from existing benchmarks and Kaggle into a Universal Dataset Pool (UDP), extract 112 meta-features, and use hierarchical clustering to select representative subsets of 100 (classification) and 140 (regression) datasets. The core claim is that TabPalooza achieves superior diversity—operationalized as lower reconstruction error—compared to existing benchmarks like TabZilla, TabArena, and OpenML-CC18.

## Strengths

- **Functional diversity metric via rank reconstruction error ($d_r$):** Rather than relying solely on static meta-feature counts, the paper defines benchmark quality by how well a subset preserves model ranking behavior across the broader landscape (Section 3.1.1). This is a meaningful and novel operationalization—it moves beyond "how many meta-feature buckets are covered" to "does this benchmark predict what other benchmarks would conclude," which directly addresses the stated problem of inconsistent rankings.

- **Systematic pipeline with explicit diversity-efficiency trade-off:** The construction methodology—meta-feature extraction, correlation-based feature selection, hierarchical clustering with size control, and alignment validation—provides a principled and reproducible framework for benchmark construction (Sections 3.1–3.2). The inflection point analysis for size selection (Figure 1) makes the efficiency argument concrete rather than arbitrary.

- **Comprehensive dataset aggregation:** The Universal Dataset Pool integrates datasets from five classification benchmarks, three regression benchmarks, and 3,654 additional Kaggle datasets, yielding 501 classification and 335 regression datasets after deduplication and validation (Section 4.1). This is a substantial curation effort that provides a richer foundation for subset selection than any single prior benchmark.

## Weaknesses

- **Conceptual slippage between "diversity" and "representativeness":** The paper equates low reconstruction error with high diversity, but these are distinct concepts. A benchmark that perfectly mirrors another benchmark's rankings would have zero reconstruction error yet could be entirely non-diverse (e.g., containing only easy datasets if the target only contains easy datasets). True diversity implies coverage of the data manifold—including hard, noisy, or atypical datasets—while low $d_r$ merely indicates representativeness of ranking behavior. The paper does not establish that low $d_r$ implies broad meta-feature coverage, which is the more intuitive notion of diversity.

- **Discretization of regression targets undermines regression meta-features:** For regression tasks, the paper discretizes continuous targets into 10 equal-frequency bins to apply PyMFE (Section 3.1.3). This destroys information about regression-specific properties such as noise level, target range, and residual distribution—precisely the features that would differentiate regression difficulty. Since these meta-features drive both the correlation-based feature selection and the Euclidean-distance-based alignment (Section 3.1.5), the regression benchmark's diversity claims rest on potentially uninformative features.

- **Negative meta-feature prediction deltas on TabArena weaken alignment claims:** In Table 2, the meta-feature-based rank predictor performs *worse* than the trivial baseline on TabArena for ACC (delta = −0.040) and F1 (delta = −0.164). The paper acknowledges these as "exceptions" without explanation. If meta-features fail to predict rankings on TabArena datasets, the alignment pipeline—which relies on meta-feature distances to select representative datasets—may not reliably capture TabArena's characteristics in TabPalooza. This inconsistency is not analyzed.

- **No size-controlled ablation to isolate selection quality from pool size:** TabPalooza is selected from a pool of 501/335 datasets, while comparison benchmarks like TabZilla (27) and TabArena (33) are much smaller. Even with the alignment pipeline selecting subsets matching target size, TabPalooza draws its 27 representatives from a pool of 100 candidates versus TabZilla's pool of only 27. The lower $d_r$ could partially result from having more candidates rather than from better selection. A baseline selecting 100 random datasets from UDP (or applying the same pipeline to a same-sized random pool) would isolate the contribution of the meta-feature-driven selection.

- **Solvability filter biases the benchmark toward tractable problems:** Excluding datasets with AUC < 0.55 or R² < 0.2 (Section 4.1) removes inherently difficult or noisy tasks. While these may be uninformative for model selection, they test robustness and calibration—important practical capabilities. A benchmark claiming superior diversity should arguably include some "hard" tasks, or at minimum discuss how this filter shapes the benchmark's scope.

- **Exclusion of datasets with >10 classes limits real-world relevance:** To accommodate ICL model constraints, datasets with more than 10 categories are excluded (Section 4.1). High-cardinality categorical problems (e.g., product codes, user IDs, diagnosis codes) are among the most common industrial tabular tasks. This constraint means TabPalooza cannot evaluate a significant and practically important regime of tabular learning.

- **Threshold and configuration choices lack sensitivity analysis:** The final benchmark configurations (Kendall+0.07 for classification, dCor+0.12 for regression) are selected from a grid but without reported sensitivity analysis (Section 4.4). It is unclear whether small perturbations to these thresholds substantially change benchmark composition or alignment scores.

- **LLM-based Kaggle parsing lacks validation:** The use of DeepSeek-r1 and Qwen3 to identify target columns and train/test splits (Section 4.1) is creative but unvalidated. No manual spot-check or accuracy estimate is provided. Systematic misidentification of targets (e.g., confusing features with labels) would introduce label noise into a significant portion of the benchmark—precisely the kind of systematic error that cross-verification between two LLMs may not catch if both make similar errors.

## Nice-to-Haves

- A bootstrapping or subsampling analysis showing that model rankings on TabPalooza are stable across different random selections of the 100/140 datasets from clusters, which would strengthen the "reliable evaluation" claim.
- Compute cost benchmarking (wall-clock time, not just dataset count) comparing TabPalooza against TALENT and TabArena to substantiate the efficiency claim in practical terms.
- An ablation studying which meta-feature categories (General, Statistical, Info-theory, Landmarking, Model-based) drive the clustering, to verify that diversity is based on meaningful semantic differences rather than trivial properties like row/column counts.
- A dedicated Limitations section discussing the regression discretization issue, the solvability filter bias, and the class cardinality constraint.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Title critique ("Odyssey" is metaphoric):** Pure style nitpick; removed per hard rules.
- **Grammar issue ("Efficiency limit the total number of datasets"):** Formatting/style nitpick; removed per hard rules.
- **Missing related works on core-set/clustering-based selection:** Removed per hard rules against flagging missing related works without external verification.
- **Many-to-one mapping concern in alignment:** The reviewer speculated that when source $S$ is smaller than target $T$, multiple datasets in $T$ map to the same dataset in $S$, affecting independence. However, TabPalooza (100/140 datasets) is *larger* than most target benchmarks (TabZilla: 27, TabArena: 33), making this concern largely inapplicable in practice.
- **Reproducibility of random cluster selection:** The reviewer worried that random selection from clusters introduces non-determinism. However, the benchmark is released as a fixed artifact on HuggingFace, so the final benchmark composition is deterministic and reproducible regardless of pipeline randomness.
- **Commercial LLM API reproducibility concern:** The positive reviewer claimed commercial LLM APIs hinder reproducibility. However, the paper uses Ollama (local deployment) with specified open-weight models (DeepSeek-r1:32b, Qwen3:32b), not commercial APIs. This concern is factually incorrect.
- **Demand for numerical values in heatmaps (Figures 2–3):** Visual design preference, not a substantive weakness.
- **Demand for standardized compute budgets across all models:** The benchmark's contribution is the dataset collection and diversity methodology, not a fair model comparison. The model rankings are used instrumentally to compute $d_r$, not as standalone claims.

## Novel Insights

The rank reconstruction error metric ($d_r$) reveals an important asymmetry: TabPalooza serves well as a *source* for reconstructing other benchmarks but is poorly reconstructed *by* other benchmarks (Section 4.5). This asymmetry is actually meaningful—it suggests TabPalooza's meta-feature space is a superset of what other benchmarks cover. However, this raises a subtler question the paper does not explore: if TabPalooza is a "superset" benchmark, then using it as a standard may over-represent certain dataset regimes that are rare in practice. The optimal benchmark for the community may not be the one that best reconstructs all other benchmarks, but the one that best reflects the distribution of real-world tabular problems—a distinction the current methodology does not make.

## Suggestions

- **Add a size-controlled ablation:** Select 100 random datasets from UDP (without meta-feature-guided clustering) and compute $d_r$ against the same target benchmarks. This directly isolates the value of the proposed selection pipeline from the advantage of a larger pool.
- **Validate regression meta-features:** Either compute regression-specific meta-features that do not require discretization (e.g., target skewness, noise estimates from residual analysis) or empirically test whether the discretized meta-features actually correlate with regression model rankings on a held-out set.
- **Provide manual validation of LLM-parsed datasets:** Spot-check 50–100 Kaggle datasets to estimate the error rate of the LLM target identification, and report this explicitly.
- **Report sensitivity of benchmark composition to threshold choices:** Show how the top-5 most important meta-features and the final dataset list change when Kendall threshold varies from 0.05 to 0.09, to demonstrate robustness.
- **Discuss or quantify the impact of the >10-class exclusion:** Report how many UDP datasets are excluded by this constraint and analyze whether the excluded datasets occupy a distinct region of meta-feature space.

---

**Assessment by axis:**

- **Novelty:** Moderate-High. The rank reconstruction metric as a benchmark evaluation tool is genuinely novel. The construction pipeline is competent but relies on standard techniques (clustering, correlation filtering).

- **Technical soundness:** Moderate. The core framework is reasonable, but several gaps—regression meta-feature validity, unexplained negative prediction deltas, lack of size-controlled ablation—reduce confidence in the "superior diversity" claim.

- **Empirical support:** Moderate. Experiments are extensive in coverage (multiple benchmarks, metrics, model families) but incomplete in isolating the contribution of the selection methodology from confounds like pool size.

- **Significance:** Moderate-High. A principled, community-adopted tabular benchmark would be highly valuable given the current fragmentation. TabPalooza is a credible step toward this, though the methodological gaps must be addressed for it to serve as a definitive standard.

- **Clarity:** Moderate. The paper is generally readable with clear structure. Some definitions are imprecise (diversity vs. representativeness), and the meta-feature evaluation section (3.1.4) is dense. Parser artifacts in equations and tables are noted as formatting issues.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 2.0, 2.0, 2.0]
Average score: 1.2
Binary outcome: Reject
