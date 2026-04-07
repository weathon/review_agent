## Summary
This paper proposes four metrics for assessing individual fairness in ML models: Proxy Dependency Score (PDS), Counterfactual Stability Rate (CSR), Attribution Independence Score (AIS), and Intra-Cohort Decision Consistency (IDC). The metrics aim to capture proxy reliance, counterfactual robustness, attributional independence, and within-cohort consistency respectively. The authors evaluate these metrics on Adult and COMPAS datasets, showing cases where group and individual fairness metrics diverge.

## Strengths
- The paper provides clear pseudocode (Algorithms 1-4) and formal definitions for each metric, making the framework actionable for practitioners seeking to audit models.
- By proposing four distinct metrics targeting different fairness aspects, the work offers a multi-dimensional diagnostic toolkit that goes beyond single-score fairness evaluations.
- The empirical results in Table 1 successfully illustrate concrete cases where group metrics (e.g., Disparate Impact of 1.456 for COMPAS sex) flag unfairness while individual metrics (CSR 0.773) reveal problematic instability—demonstrating the tension between group and individual fairness in practice.

## Weaknesses
- **PDS is not an individual-level fairness metric.** Equation 1 defines PDS as `1 − Accuracy(M')/Accuracy(M)`, a ratio comparing two models' accuracies. This is a model-level ablation test, not an individual fairness assessment. The paper frames PDS as measuring "influence of protected attributes transmitted through proxy variables," but the formula only measures whether removing protected attributes hurts overall accuracy—no individual-level discrimination is quantified.
- **Negative PDS values are unexplained.** Table 1 reports PDS values of -0.0014, -0.009, and -0.0123, meaning the shadow model (without protected attributes) outperforms the original. The paper never discusses the semantics of negative scores, nor why the "fairness range" of [-0.2, 0.2] would accommodate models where protected attributes *hurt* performance.
- **All "fairness ranges" are unjustified.** The paper presents PDS ∈ [-0.2, 0.2], CSR ∈ [0.8, 1], IDC ∈ [0.8, 1], and AIS ∈ [0.8, 1] as thresholds without theoretical derivation, calibration study, or citation. These appear arbitrary and create a false sense of objectivity.
- **IDC implementation contradicts its stated purpose.** Section 3.4 claims IDC quantifies consistency "across individuals who are nearly identical in terms of their non-protected features." However, Algorithm 4 applies KMeans to `X_test` (the full feature matrix), which includes protected attributes. This directly contradicts the metric's definition—clustering on protected attributes means similar cohorts are defined using the very features the metric should ignore.
- **AIS uses Pearson correlation inappropriately.** Equation 3 computes `1 − |corr(Attr_f(x), Protected(x))|`. For categorical protected attributes (race, gender), computing a scalar Pearson correlation is methodologically problematic. The metric would require correlation ratio or mutual information for categorical variables. Additionally, Table 1 reports AIS as a range [Min, Max], implying per-feature computation, but Equation 3 yields a scalar—this inconsistency is not explained.
- **CSR flip mapping is underspecified.** Algorithm 2 states "apply flip mapping" for counterfactual generation but never defines what flip mapping is used for Adult or COMPAS. For multi-valued categorical attributes, the counterfactual generation mechanism is critical for reproducibility.
- **No model specification or training details.** The experimental section never states what ML model(s) were trained (logistic regression? random forest? neural network?), nor provides hyperparameters, train/test split ratios, or random seeds. All results in Table 1 are uninterpretable without this information.
- **No comparison to existing individual fairness metrics.** The paper cites John & Saha (2020), Galhotra et al. (2017), and Li et al. (2023) as existing individual fairness approaches but never benchmarks against them. Readers cannot assess whether PDS, CSR, AIS, and IDC provide information beyond existing tools.
- **No theoretical connection to individual fairness foundations.** The paper repeatedly invokes Dwork et al. (2012)'s Lipschitz condition on similarity metrics but none of the four metrics operationalize this formalization. IDC uses KMeans on raw features rather than a domain-specific similarity metric, which is precisely what Dwork et al. argue against.

## Nice-to-Haves
- Evaluation on deep neural networks or modern architectures beyond what appears to be standard tabular classifiers.
- Demonstration of practical utility by training models with these metrics as regularization terms.
- Analysis of correlation between the four metrics to establish they provide complementary rather than redundant information.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **Claims about future-dated references being invalid**: The reviews flagged 2025 references (Plecko & Bareinboim, Gennaro et al., Molnar) as potentially invalid, but the paper cites them and we cannot verify their non-existence. If the paper cites them, assume they are valid arXiv preprints or forthcoming works.

- **Generic criticisms about societal impact sections**: The criticism that Sections 5-6 are "survey-style" and "add no scientific value" is overly harsh. While these sections are indeed broad and could be condensed, they contextualize the work within real-world AI fairness challenges—a legitimate aspect of applied ML research.

- **"Only two datasets" as a fatal flaw**: While expanding beyond Adult and COMPAS would strengthen the paper, these are standard fairness benchmarks. The critique is valid but not devastating for an initial framework proposal.

## Novel Insights
The most striking observation from combining the reviews with the paper is the **fundamental category error in PDS**: it claims to measure individual fairness but computes a population-level accuracy ratio. This is not a minor implementation detail—it undermines the framing of PDS as part of an "individual fairness framework." A genuine individual fairness metric must assign scores to individuals, not aggregate model-level comparisons. The empirical finding that group and individual metrics diverge is less novel than the paper claims (Kleinberg et al. 2016 proves this mathematically), but demonstrating specific cases where Disparate Impact and CSR give contradictory signals on real datasets has practical diagnostic value—if only the underlying metrics were technically sound.

## Suggestions
- **Reformulate PDS as an individual-level metric** or remove it entirely. If the goal is to measure proxy reliance per individual, consider computing attribution-based dependence scores for each prediction, not model-level accuracy ratios.
- **Fix IDC to cluster only on non-protected features.** The current Algorithm 4 clusters on all features; either modify the implementation or revise the metric's definition to acknowledge this limitation.
- **Replace Pearson correlation in AIS** with appropriate measures for categorical variables (e.g., mutual information, correlation ratio) or explicitly one-hot encode protected attributes before computing correlations.
- **Specify the exact model architecture, hyperparameters, and training procedure** to enable reproducibility.
- **Derive or empirically calibrate fairness thresholds** rather than presenting arbitrary ranges as objective cutoffs.
- **Add baseline comparisons to existing individual fairness metrics** (e.g., fairness through awareness verification, counterfactual fairness implementations) to demonstrate what the proposed metrics add beyond prior work.