=== CALIBRATION EXAMPLE 44 ===

# Final Consolidated Review
## Summary
ComputAgeBench introduces a standardized benchmark framework for evaluating epigenetic aging clocks using blood DNA methylation data. The authors formalize four properties of biological age, curate 66 harmonized public datasets covering 19 aging-accelerating conditions (AACs) and over 10,000 samples, define four evaluation tasks, and propose a cumulative BenchScore metric. Applying this framework to 13 published clock models, they demonstrate that second-generation clocks (trained on survival outcomes) consistently outperform first-generation clocks (trained on chronological age), and that high chronological age prediction accuracy anti-correlates with biological age acceleration detection—directly empirically validating the long-discussed "biomarker paradox."

---

## Strengths

- **Unprecedented scale of harmonized data.** Aggregating 66 GEO datasets from more than 50 studies into a single unified metadata structure with consistent sample annotation (tissue, platform, condition class) is a non-trivial engineering and curation contribution that no prior comparative study has matched at this scale.

- **Explicit formalization of the biomarker paradox.** The paper is rare in explicitly operationalizing the four formal properties of biological age and using the biomarker paradox—that minimizing chronological age prediction error forces Δ → 0—as the mathematical motivation for the benchmark design. This is a genuinely clarifying contribution to a field that has largely left this tension implicit.

- **Evidence-based and pre-specified AAC criteria.** The three-criterion definition for AACs (decreased life expectancy, chronicity, systemic manifestation) is stated *a priori* and applied consistently, addressing a key criticism of prior comparative studies where validation data were chosen post-hoc or varied across labs (as highlighted in the Moqri et al. 2024 review cited by the authors).

- **Empirical validation of generation-level performance difference.** The result that second-generation clocks outperform first-generation clocks on the more rigorous AA2 task—confirmed across 42 datasets and 13 models—is the most comprehensive empirical test of this conjecture to date and directly informs future clock design decisions.

- **Open reproducibility.** The Google Colab notebook enabling full benchmark reproduction on public data is a concrete and valuable contribution for community adoption, especially given the typical fragmentation of this field.

---

## Weaknesses

### Fatal
None. The paper has real methodological limitations but they do not invalidate the core contribution.

### Major

- **Cell-type composition confounding is the elephant in the room.** The authors acknowledge in Discussion that "blood DNA methylation generally comes from the immune cells, which would be directly affected by the HIV," yet this is framed as unsurprising rather than as a potential invalidation of the ISD results. For bulk blood DNAm, known to reflect a mixture of cell types (granulocytes, monocytes, T/B/NK cells), conditions that alter cellular composition—particularly HIV, autoimmune diseases, and inflammatory conditions—will shift bulk methylation profiles independently of cellular aging pace. If a clock detects "accelerated aging" in HIV patients because HIV-driven lymphopenia and immune activation alter CpG sites that happen to overlap with aging-associated CpGs, this tells us about the clock's sensitivity to immune perturbation, not necessarily about biological aging per se. Since ISD (predominantly HIV) accounts for 7–10 points in AA2 scores and 9/9 in AA1 for several top models, the benchmark's primary positive signal is dominated by a condition class where the causal mechanism is ambiguous. This should be analyzed via cell-type deconvolution (e.g., Houseman or EpiDISH methods on the benchmark data), or at minimum, a sensitivity analysis showing results with ISD excluded from BenchScore rankings.

- **Universal CVD failure is unresolved.** Every single one of the 13 models scores 0/3 on the CVD class in AA2 (Fig. 3E, Table in text). The Discussion offers three possible explanations but does not disambiguate between them: (a) all existing clocks genuinely fail to detect CVD-related aging in blood, (b) CVD does not accelerate blood epigenetic aging in the way the benchmark assumes, or (c) statistical power is insufficient for the available CVD datasets. If explanation (b) is correct, CVD should not be in the benchmark—its inclusion inflates the denominator of all scores and gives the false impression that clocks are being tested against a valid signal. A power analysis for the CVD datasets and/or citation of direct evidence that CVD accelerates blood DNAm aging would substantially clarify this.

- **BenchScore (Eq. 2) is ad hoc and lacks robustness validation.** Several specific issues compound here: (1) only *positive* bias is penalized (`max(0, Med(Δ))`), so a clock with large negative bias (e.g., YingDamAge at −14.5 years) receives no AA1 penalty despite this being equally problematic for the task; (2) AA2 scores (max 42) and AA1 scores (max 24) are summed directly, giving AA2 nearly double the implicit weight simply because more datasets fall into that task—this weighting is not justified; (3) no ablation or sensitivity analysis shows that the ranking of models is robust to reasonable alternative formulations (e.g., different penalty functions, per-condition normalization, or treating tasks separately). The authors acknowledge "there could be a more optimal solution," but given that BenchScore determines the headline conclusion, this caveat requires a concrete demonstration of ranking stability.

- **Statistical power not established for smallest datasets.** The minimum thresholds of ≥5 AAC samples per dataset and ≥10 AAC samples across all datasets for a condition class are very low. No power analysis is provided. For small-to-moderate aging acceleration effect sizes (Cohen's d ~ 0.3–0.5), n=5 AAC samples has essentially no power at α=0.05. The FDR correction across all 42 heterogeneous datasets simultaneously further complicates interpretation: this pools tests from conditions with radically different effect sizes and sample sizes, and may over-penalize genuine effects in small datasets while under-penalizing conditions with many datasets. Negative results in small-n datasets cannot be interpreted as evidence that the clock fails—they may simply reflect underpowered tests.

### Minor

- **Normality justification for parametric tests is insufficient.** The paper states the Welch's t-test was chosen "due to the assumption of normal distribution of Δ, a fundamental trait of the multivariate linear regression models commonly used in aging clock construction." This justification applies only to linear regression-based clocks (roughly half of the 13 tested). For non-linear or non-regression clocks, the normality assumption may not hold. No robustness check (e.g., comparison with Wilcoxon rank-sum test) is provided.

- **Effect size completely absent from reporting.** The binary pass/fail criterion (adjusted p < 0.05) is coarse. Two datasets can both "pass" with effect sizes of 0.3 and 3.0, respectively, but are counted identically in the score. Reporting Cohen's d or standardized mean difference alongside p-values would allow distinguishing borderline from robust detections and would add substantially to the interpretability of results.

- **ISD/HIV dominance inflates and potentially distorts model rankings.** Given that nearly all clocks perform near-perfectly on ISD in AA1 (9/9 for multiple models) and that second-generation clocks are particularly attuned to immune biomarkers, the BenchScore leaderboard likely reflects which clocks are best calibrated to immune-associated methylation changes rather than general biological aging. A stripped-down analysis excluding ISD would reveal whether the generation-level performance difference holds for non-immune disease classes.

- **GEO selection bias.** Studies are more likely to be published and deposited when the investigators found DNAm differences between cases and controls. Selecting benchmark datasets from GEO therefore systematically biases toward finding effects—inflating all clocks' apparent sensitivity. This limitation should be explicitly acknowledged, as it affects the interpretation of all positive results.

- **Platform mixing impact not quantified.** The benchmark pools 27K, 450K, and 850K array data with missing CpGs imputed using SeSAMe reference values. While imputation comparison results are in Appendix A.3, the effect of platform heterogeneity on AA1 and AA2 task outcomes (e.g., does performance differ systematically between 27K-trained datasets and 850K datasets?) is not reported.

### Tiny

- The uncertainty values (e.g., 7.6 ± 0.1 years) in Table 1 are never described as bootstrap confidence intervals in the main text; the reader cannot verify the source of these intervals.

- The paper states in the abstract that "there is no standardized methodology to validate and compare epigenetic clock models as yet." Section 2.2 appropriately qualifies this—the distinction is scale, standardization, and open access, not complete absence of prior work. The abstract framing should be sharpened.

---

## Nice-to-Haves

- **Cell composition sensitivity analysis.** Even a basic Houseman deconvolution and correlation between estimated immune cell fractions and Δ would empirically test whether clocks are detecting aging or immune composition. This would substantially strengthen claims about what the benchmark is actually measuring.

- **ISD-excluded BenchScore.** Reporting BenchScore with and without ISD datasets would clarify whether the generation-level ranking is robust or primarily driven by HIV/immune conditions.

- **Bias recalibration experiment.** Testing whether subtracting Med(Δ_HC) per clock from all predictions improves AA1 scores would distinguish fundamental model failure from correctable calibration shift, helping authors better characterize the types of clock deficiency.

- **Longitudinal validation subset.** Even a small longitudinal dataset where within-subject Δ changes over time could be tracked would provide evidence that the cross-sectional benchmark captures meaningful variation.

- **Live leaderboard.** A static Colab notebook, while commendable, does not prevent future clock authors from tuning their models to these specific 66 datasets. A hidden-test submission system would maintain the benchmark's integrity as a community resource over time.

- **Training-benchmark condition overlap analysis.** While the paper confirms no *data* overlap (Section 3.4), it does not discuss whether clocks trained on datasets that explicitly included disease status in training (some second-generation clocks are trained on samples with various conditions) could have indirectly learned the benchmark AAC signals. A brief discussion would be useful.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Critic: "Training data overlap invalidates unbiased evaluation"** (Spark Finder) — The paper explicitly states in Section 3.4: "we also ensured that no data in the benchmark was used to train any of the selected clocks." This concern is directly addressed and should not be raised as a weakness.

- **Critic: "AA2 counts are cited as circular" because AAC criterion 3 (systemic) overlaps with test design** — The paper's test is genuinely informative because the clock training objective (chronological or survival age prediction) is independent of the benchmark's AAC signal. The circularity is not of the logically invalidating type the critic implies.

- **Critic: "ICLR venue mismatch"** — Venue suitability is not a technical weakness. The benchmark and methodology have genuine ML relevance (surrogate endpoint validation, evaluation under latent ground truth), and evaluating whether a paper belongs at a venue is the PC's judgment, not a weakness of the scientific content.

- **Critic: "Abstract overstated"** — The abstract is adequately qualified by Section 2.2, which explicitly lists prior comparative efforts and explains why they fall short. The distinction the paper draws (scale + standardization + open access) is substantive.

- **Critic: "Requesting confidence intervals for all large-scale benchmarks"** (FDR hierarchical correction demand) — While a hierarchical correction would be more principled, this is not a standard requirement in this community and the current approach is reasonable given the exploratory nature of the benchmark.

- **Critic: "No missing related works to cite"** — Per instructions, not raised.

---

## Novel Insights

The juxtaposition of AA1 and AA2 task results reveals a subtlety not well-emphasized in the paper's framing: models like Zhang19_EN (19/24 AA1, 2/42 AA2) and Hannum (17/24 AA1, 1/42 AA2) rank high on the AA1 leaderboard but near the bottom on the more rigorous AA2. The BenchScore correctly demotes them only *partially*, through the bias penalty—but the paper's own tables show these models have large positive biases (Med(Δ) = 9.6 and 6.3 years respectively), meaning they detect "accelerated aging" in AAC patients largely because they predict *all* samples as older than they are. This suggests a previously unformalized failure mode: a clock can achieve high AA1 scores through global overestimation rather than through genuine sensitivity to biological aging. The BenchScore penalization partially captures this, but the paper could more forcefully characterize this as a distinct and dangerous failure mode—a clock that systematically adds years to every patient's biological age estimate will appear to "detect aging acceleration" in almost any disease cohort regardless of whether true aging acceleration exists.

---

## Suggestions

1. **Run cell-type deconvolution (e.g., EpiDISH or Houseman) on all HC and AAC samples** and report the correlation between immune cell fraction estimates and Δ. If a large fraction of the ISD-class signal disappears after adjusting for cell composition, this should be prominently disclosed and ISD weights in BenchScore revisited.

2. **Add a BenchScore sensitivity table** in the appendix showing model rankings under at least 3 alternative formulations (e.g., treating AA1 and AA2 with equal weight per dataset, geometric combination, per-condition-class normalization). If rankings are stable, the ad-hoc concern is mitigated; if not, a revised formula is needed.

3. **Provide power estimates for the CVD and MBD condition classes.** If even a 5-year mean acceleration difference would be undetectable at n=5 per dataset under the current test, this should be reported and CVD/MBD datasets should either be flagged as underpowered or excluded from the denominator of model scores.

4. **Report Cohen's d** (or a non-parametric equivalent) alongside p-values for all AA2 datasets, at minimum in the supplementary. This would allow distinguishing statistical significance (driven by sample size) from biological meaningfulness.

5. **Report BenchScore with ISD excluded** as a sensitivity check; this addresses both the cell-type confounding concern and the dominance concern without requiring new experiments.

6. **Clarify in main text** what the ±values in Table 1 represent (bootstrap CI, standard error, etc.), and explicitly state the number of bootstrap iterations.

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 6.0, 8.0]
Average score: 6.2
Binary outcome: Reject
