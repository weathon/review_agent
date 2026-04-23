Now I have enough information to synthesize the final review. Let me compile my findings.

## Summary

This paper introduces a CATE evaluation framework based on observational sampling from RCT data, centered around the statistical parameter Q (= MSE minus a constant) and its estimator Q̂. The framework enables CATE model evaluation without counterfactual ground truth when propensity is known, and the authors prove Q̂ is unbiased with O(1/√N) convergence. The paper's primary contribution is a large-scale benchmark (16 models, 12 real-world RCT datasets, 43,200 variants) producing striking findings: 62% of CATE estimates are "degenerate" (worse than a zero-effect predictor), 80% fail to beat a constant-effect model, and orthogonality-based methods win only 30% of the time.

## Strengths

- **The Q̂ evaluation framework is a genuine methodological contribution.** Lemma 3.1 establishes unbiasedness of Q̂ when propensity is known, and Theorem 3.12 proves O(1/√N) convergence—all without relying on the orthogonality assumptions that underpin existing proxy losses. This directly addresses the self-serving bias problem the paper identifies (Section 1), where evaluating an R-learner with R-loss creates a conflict of interest.

- **The control variates unification (Propositions 3.9–3.11) is a clean theoretical contribution.** Showing that DR-loss and R-loss are special cases of the Q̂ variance-reduction framework clarifies the relationship between estimation losses and evaluation metrics, and extends beyond the results of Chernozhukov et al. (2023), which required orthogonality assumptions.

- **The observational sampling benchmark design addresses a real gap.** Using real-world RCT outcomes (not simulated) for evaluation while training on observational subsamples with selection bias avoids both simulation bias and self-serving bias—a meaningful advance over semi-synthetic benchmarks.

- **Empirical validation of Q̂ as an oracle ranking surrogate (Figure 1).** Q̂ variants achieve MRR above 0.8 against oracle rankings across all evaluation dataset sizes on semi-synthetic Hillstrom data, substantially outperforming all alternative evaluation metrics (which only reach 0.45–0.75). This directly validates the theoretical claim that Q̂ preserves MSE ranking.

- **The paper raises an important and underappreciated question** about whether CATE methods' theoretical guarantees translate to real-world performance. The finding that 96% of datasets have at least one non-degenerate model (Section 4.2) provides evidence that some heterogeneity is captureable, making the high degenerate rates for most models genuinely concerning.

## Weaknesses

### Fatal
None.

### Major

- **The headline "62% degenerate" finding is uninterpretable without quantifying how much heterogeneity exists in the datasets.** Since Q(τ̂) = MSE(τ̂) − E[τ²(X)], a degenerate estimator (Q ≥ 0) means MSE(τ̂) ≥ E[τ²(X)]. If true heterogeneity E[τ²(X)] is small relative to estimation noise, even a good CATE estimator would appear degenerate. The paper never reports E[τ²(X)] or Var(τ(X)) for any dataset, making it impossible to distinguish "methods fail to capture existing heterogeneity" from "there is insufficient heterogeneity to capture." While the fact that 96% of datasets have at least one non-degenerate model provides partial evidence that heterogeneity exists, this does not quantify its magnitude. Without an oracle or near-oracle baseline, the 62% statistic is a headline without proper context—this fundamentally undermines the paper's central narrative as stated.

- **The model comparison confounds CATE strategy with base learner choice and implementation quality, yet results are presented as findings about CATE strategies.** Models like s.xgb.cv (S-learner + XGBoost + CV) and dml.linear (DML + linear regression) differ in base learner flexibility, hyperparameter tuning, and implementation library—not just CATE estimation logic. Most critically, dml.xgb has a 99.0% degenerate rate (Table 1), far worse than dml.lasso at 48.4%. A 99% degenerate rate for a flexible method like XGBoost-based DML is a red flag for implementation or tuning failure, not a valid finding about the DML framework. The paper does not investigate this anomaly and instead aggregates it into the claim that "orthogonality-based learners underperform" (Finding 3). When the confound is this extreme, the aggregate finding is unreliable.

- **Insufficient disaggregation of results across experimental conditions.** The 43,200 dataset variants span 12 datasets × 4 sizes × 3 treatment percentages × 3 confounding levels × 100 repetitions, yet the paper reports only aggregate statistics. There is no analysis of which conditions drive failure—whether it is small sample sizes, high confounding, low heterogeneity, or specific datasets. Finding 3's explanation ("a combination of factors, including the data-generating process and modeling choices") is vacuous, and the paper explicitly defers causal explanation to "ongoing research." This leaves the core empirical finding as an observation without explanation.

### Minor

- **The use of dml.lasso as a "constant-effect estimator" (Finding 2) is unclear.** The paper states "We use Double ML with a Lasso base learner (dml.lasso) to construct τ̂_B, a constant-effect estimator," but does not explain how DML+Lasso produces a constant (rather than linear-in-X) CATE estimate. If dml.lasso actually produces a linear (not constant) effect model, the 80% figure would have a different interpretation. Clarifying this configuration is important for interpreting Finding 2.

- **The theoretical contributions, while correct, are incremental relative to existing IPW literature.** Lemma 3.1 follows directly from Horvitz-Thompson unbiasedness. The generalization theorems (3.7–3.8) are standard importance-weighting and stability results. The claim of being "the first to provide asymptotic guarantees for oracle ranking under such general conditions" (Remark 3.2) overstates novelty, as similar guarantees are implicit in prior IPW-based evaluation work. The contribution is more in the unification and application than in fundamentally new theory.

### Trivial
None.

## Nice-to-Haves

- An oracle CATE baseline (e.g., using the semi-synthetic setting where ground truth is available) to establish the achievable performance floor and contextualize the degenerate rates.
- A controlled experiment holding the base learner fixed (e.g., all strategies using XGBoost with the same tuning protocol) to separate CATE strategy effects from base learner confounds.
- Per-dataset or per-condition breakdowns of degenerate rates alongside estimated heterogeneity magnitude to reveal whether the problem is methods or data.
- Investigation of the dml.xgb 99% degenerate rate anomaly—is it a bug, a tuning issue, or a genuine finding about DML with flexible learners?

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic's claim that observational sampling novelty is overstated because "observational sampling for causal evaluation dates to LaLonde (1986)."** The paper itself explicitly acknowledges this history (Section 1: "The history of observational sampling dates back as long as the field of causal inference itself. For instance, LaLonde (1986) used it..."). The novelty claim is specifically about applying observational sampling to CATE evaluation with the Q̂ framework, not observational sampling per se.

- **Harsh Critic's claim that Q̂ is "essentially an IPW-based evaluation metric, which has its own well-known variance issues."** The paper directly addresses this through the control variates framework (Section 3.2), which is a core contribution. Dismissing it because the base estimator is IPW-based ignores the variance reduction contribution.

- **Strength Finder's claim about "first result providing asymptotic guarantees for oracle ranking under such general conditions" as a core strength.** This overstates the theoretical novelty; similar guarantees are implicit in prior IPW work. The real contribution is the unification and application, not the asymptotic result itself.

- **Harsh Critic's complaint about critical details being "relegated to appendices."** The appendices are stripped by the parser; they exist in the original submission. This is not a valid criticism.

- **Harsh Critic's claim that Finding 4 ("all learners are weak learners") is "the inverse of a tautology."** The finding that no method achieves a dominant win share is not tautological—it is a genuine empirical observation. The paper is correct that this contrasts with prior studies that show certain methods dominating on semi-synthetic data.

- **Harsh Critic's complaint about the observational sampling nonlinearity levels possibly including "unrealistically extreme confounding."** Without evidence from the paper that the confounding levels are unrealistic, this is speculative. The paper describes three levels of "assignment mechanism nonlinearity" as a design choice, and without seeing the specific levels, criticizing them as unrealistic is unfounded.

## Novel Insights

The most insightful observation across the reviews is the tension between the paper's two contributions: the Q̂ methodology (which is sound and well-validated) and the empirical findings (which are striking but under-analyzed). The paper's strongest evidence that CATE methods genuinely struggle is the 96% figure—most datasets have at least one non-degenerate model, suggesting heterogeneity exists—combined with the 62% aggregate degenerate rate. However, without quantifying heterogeneity magnitude, the paper cannot calibrate expectations: it's unclear whether 62% is surprisingly high or exactly what we'd expect given the signal-to-noise ratio in these datasets. This is not a fatal flaw (the methodology and the question are both valuable), but it means the paper is more successful as a methodological contribution than as an empirical verdict on CATE methods.

## Suggestions

- Add a table reporting estimated E[τ²(X)] for each dataset (computable from RCT data using η) to contextualize the degenerate rates.
- Investigate and discuss the dml.xgb anomaly explicitly—either confirm it's a genuine finding or acknowledge it as a potential implementation issue and report results with and without it.
- Provide at least one disaggregated view (e.g., degenerate rates by dataset or by sample size) to move beyond aggregate statistics and reveal which conditions drive failure.
- Consider a controlled comparison (e.g., S-learner vs T-learner vs R-learner all with XGBoost base learner) to isolate CATE strategy effects from base learner confounds.

## Evaluation on Key Axes

- **Originality:** The Q̂ framework and its control variates unification are novel and useful, though built on well-known IPW foundations. The observational sampling application to CATE evaluation is a meaningful extension. The empirical findings are striking but partially confounded.
- **Importance of research question:** High. Understanding whether CATE methods work on real data is critically important for the field, and the self-serving bias observation is valuable.
- **Claims well supported:** Partially. The theoretical claims are well-supported; the empirical headline claims (62% degenerate, orthogonality underperformance) are undermined by unquantified heterogeneity, confounded comparisons, and lack of disaggregation.
- **Soundness of experiments:** The benchmark design (observational sampling, real-world datasets, 43,200 variants) is large-scale and principled, but the analysis is shallow and the model comparison is confounded.
- **Clarity of writing:** Generally clear, though some key details (constant-effect estimator configuration, heterogeneity magnitudes) are missing from the main text.
- **Value to community:** Moderate-to-high. The Q̂ framework and the benchmark methodology are valuable even if the empirical findings need further analysis.

## Calibration

Compared against the following anchors:

**High band (>7):**
- aXuWowhIYt (7.0, Accept poster): Standardizing SCMs for benchmarking—clean theoretical contribution + empirical validation, better-controlled experiments. Our paper has a similar "rethinking benchmarks" theme but less rigorous empirical analysis.
- wmV4cIbgl6 (7.33, Accept spotlight): CausalRivers benchmark—large-scale real-world dataset with clear methodology. Our paper has a smaller data contribution but adds theoretical framework.

**Medium band (4–6):**
- om5z1n0mXA (6.0, Reject): Rethinking GNN benchmarks—found simple methods match GNNs, proposed new metric. Similar profile to our paper; scored 6 but rejected. Our paper has stronger theory but more severe empirical interpretation issues.
- iaP7yHRq1l (5.5, Accept poster): Causal discovery under assumption violations—found methods underperform with some theoretical explanations. Very similar profile; accepted as poster despite confounded experiments.
- qUJsX3XMBH (4.4, Reject): Data selection methods fail to beat random—striking finding but limited depth. Our paper has more theoretical depth and better methodology.
- hom2oeHCnz (5.33, Reject): Real-world debiasing analysis—found methods fail on real biases. Similar "methods fail" paper, rejected.

**Low band (<3):**
- aoW5Sm8Op8 (2.33, Reject): Survival models benchmark—unclear takeaways, poor presentation. Our paper is significantly better with clear methodology and theoretical backing.
- 2wwPG1wpsu (2.5, Reject): LST-Bench with degenerate behavior—more like a technical report. Our paper has much stronger theoretical foundations.

Our paper sits between the medium and low bands for empirical rigor but has theoretical contributions that lift it above the low-scoring papers. It's comparable to papers in the 4.5–5.5 range that find methods underperform but have interpretability or depth issues. The theoretical framework (Q̂) is a genuine contribution that these comparison papers lack, but the empirical findings are the paper's stated primary contribution and they have significant weaknesses.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>