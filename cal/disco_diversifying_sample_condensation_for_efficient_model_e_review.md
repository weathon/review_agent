=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
## Summary
This paper introduces DISCO (Diversifying Sample Condensation), a method for efficient model evaluation that selects a small subset of test samples by maximizing *diversity in model responses* (using metrics like Predictive Diversity Score) rather than sample diversity. It provides an information-theoretic justification (Proposition 1) and demonstrates state-of-the-art performance prediction across multiple language (MMLU, HellaSwag, Winogrande, ARC) and vision (ImageNet) benchmarks, achieving up to 99.3% inference cost reduction with minimal error.

## Strengths
- **Novel and well-motivated core idea**: The shift from selecting for data representativeness to selecting for model disagreement is a clear, intuitive advance backed by a theoretical link to mutual information and Jensen-Shannon divergence (Proposition 1). This provides a principled foundation distinct from prior clustering-based or IRT-based approaches.
- **Strong and extensive empirical validation**: The method is rigorously evaluated across four major language benchmarks and ImageNet, using a realistic chronological split (training on older models, testing on newer ones). Results consistently show superior accuracy and rank correlation compared to strong baselines like Anchor-corr/gp-IRT and tinyBenchmarks.
- **Practical utility and thorough analysis**: The paper includes a detailed cost analysis (offline/online costs, break-even point ~389 evaluations) and a comprehensive factor analysis (Table 2) exploring sensitivity to design choices like model split, source model count, and dimensionality reduction, demonstrating the method's robustness and deployability.

## Weaknesses
### Major
- **Uncontrolled comparison with Metabench undermines state-of-the-art claims**: Table 1 shows DISCO (100 samples) outperforming Metabench, but the footnote admits Metabench uses more samples (150-450). Without a controlled comparison at equal sample budgets, the claim of unconditional superiority is overstated. A fair comparison is necessary to substantiate the efficiency-precision trade-off.
- **Insufficient analysis of the core disagreement metrics**: The paper proposes both PDS and JSD as selection criteria but provides limited insight into when and why one outperforms the other (e.g., PDS+RF beats JSD+RF on MMLU and HellaSwag but not on Winogrande and ARC). A deeper ablation or theoretical analysis linking these metrics to dataset properties (e.g., number of classes, entropy) is missing.

### Minor
- **Theoretical optimality rests on strong assumptions**: Proposition 1 assumes a uniform prior over source models (A1) and deterministic model predictions (A2). While the authors discuss these in Appendix G and the method works well empirically, the gap between assumptions and real-world conditions (e.g., stochastic sampling, non-uniform model pools) should be more explicitly acknowledged in the main text as a limitation of the theory.
- **Chronological split validation could be stronger**: Only one cutoff date (Jan 13, 2024) is used. Testing with multiple temporal splits (e.g., quarterly cutoffs) would better assess generalization to future model distributions and provide a clearer picture of robustness to temporal shift.
- **Limited exploration of prediction model alternatives**: The paper shows Random Forest and kNN work well, but does not compare against other potentially stronger parametric models (e.g., gradient boosting, neural networks) on the high-dimensional model signatures. This leaves open whether the simplicity of RF is optimal or merely sufficient.

### Trivial
- **Visualization of high-PDS samples**: While not critical, including examples of questions with high vs. low PDS scores would help intuitively validate what "model disagreement" captures (e.g., ambiguous vs. clear-cut questions).

## Nice-to-Haves
- **Adaptive sample selection**: Exploring a dynamic variant that updates the selected subset based on the target model's own predictions could further improve efficiency, though static selection is already a contribution.
- **Extension to open-ended tasks**: The paper correctly notes DISCO's limitation to multiple-choice/classification tasks. Suggesting potential adaptations (e.g., using candidate outputs as "classes") would broaden scope, but is outside the current contribution.
- **Statistical significance tests**: While means and standard deviations are reported, formal statistical tests (e.g., paired t-tests) between DISCO and key baselines would strengthen the claims, though the effect sizes are large and consistent.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **"The method is not conceptually simpler"**: The harsh critic argued that DISCO's pipeline (source model evaluation, metamodel training) is not simpler than clustering. This is a subjective interpretation; the paper clearly demonstrates that the *selection rule* (top-k by disagreement) and *prediction* (signature + RF) are simpler than IRT parameter estimation or iterative clustering, which is a valid claim.
- **"Incomplete cost analysis missing token costs"**: The paper provides detailed GPU-hour and storage costs (Appendix B, Tables 4-6). Demanding token-level accounting is beyond standard practice and not required to demonstrate efficiency gains.
- **"Need to test on post-submission models (2025-2026)"**: This demands validation on models released after the paper's writing, which is unrealistic for a submission. The chronological split already tests generalization to unseen models in a realistic way.
- **"Requires testing on more vision benchmarks (CIFAR-100, Places365)"**: While broader evaluation is always beneficial, the paper's claim of being "domain-agnostic" is sufficiently supported by strong results on both language (4 benchmarks) and vision (ImageNet, a standard and diverse benchmark). This is scope creep.
- **"Missing comparison to active evaluation baseline"**: The paper focuses on static subset selection for efficient evaluation; comparing to dynamic, target-model-aware active evaluation is a different research direction and not required for the stated contribution.

## Suggestions
- **Conduct a controlled comparison with Metabench** at equal sample budgets (e.g., 100 samples) to fairly establish DISCO's superiority. If this is infeasible, temper the "state-of-the-art" claims and explicitly discuss Metabench's higher sample requirement as a drawback in efficiency.
- **Add a brief analysis or discussion** in Section 5.4 clarifying when PDS or JSD might be preferable, potentially linking to dataset characteristics (e.g., number of answer choices, entropy of source model predictions).
- **Mention the theoretical assumptions (A1, A2) explicitly in the main text** when presenting Proposition 1, and briefly discuss their practical relaxations, as is done in Appendix G, to provide full transparency.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
