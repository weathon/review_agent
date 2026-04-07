## Summary
This paper proposes a method for constructing Lower Prediction Bounds (LPBs) for counterfactual survival times under different treatments in the presence of general right-censored data. By combining conformal prediction with a potential outcomes framework, it introduces a reweighting scheme that transforms the problem into a weighted conformal inference task. The method provides a marginal coverage guarantee (up to weight estimation error) and possesses a doubly robust property. Empirical results on synthetic and real-world clinical data demonstrate valid coverage and informative prediction bounds.

## Strengths
- **Novel and important problem formulation**: The paper tackles the crucial challenge of uncertainty quantification for counterfactual survival outcomes with general right-censoring, a significant gap in high-stakes domains like personalized medicine.
- **Strong theoretical foundation**: The core methodological contribution—a reweighting scheme to handle the covariate shift induced by censoring—is theoretically justified. Theorem 4.1 provides a non-asymptotic, distribution-free coverage lower bound, and Theorem 4.2 establishes doubly robust asymptotic coverage, improving upon prior PAC-type guarantees.
- **Comprehensive and convincing empirical evaluation on synthetic data**: Experiments across six diverse synthetic settings show the method consistently achieves coverage close to the nominal level and produces less conservative (more informative) LPBs than other coverage-guaranteeing conformal baselines (e.g., "Naive", "Focus"). Robustness to outliers is also demonstrated.

## Weaknesses
- **Theoretical guarantee for optimized τ is not proven**: The method optimizes the quantile level τ per test point to maximize the LPB. While Figure 11 suggests this does not harm coverage empirically, the theoretical guarantees (Theorems 4.1 & 4.2) are stated for a fixed τ. The lack of a formal proof for the data-dependent τ* leaves a gap between theory and practice.
- **Empirical validation of the doubly robust property is absent**: Theorem 4.2 is a key theoretical advantage, but the paper does not include experiments that intentionally misspecify either the quantile model or the weight model to demonstrate that coverage is maintained when the other is correct. This limits the empirical support for a major claim.
- **Real-world coverage claim is not directly verifiable**: On the real clinical dataset, true survival times for censored patients are unknown, making it impossible to directly evaluate the empirical coverage guarantee—the method's central promise. While the results are clinically plausible, the claim of "validity" in real data is indirect and relies on synthetic experiments.
- **Sensitivity to violations of core assumptions is unexplored**: The method relies on strong ignorability (including independence between potential outcomes and censoring time) and SUTVA. The paper does not investigate, even via simulation, how violations of these untestable assumptions affect the coverage guarantee, which is important for practical reliability.

## Nice-to-Haves
- Including a sensitivity analysis for the core causal assumptions (e.g., introducing simulated unmeasured confounding) would help users understand the method's robustness in real-world settings.
- Providing confidence intervals or statistical tests for the reported empirical coverage rates (e.g., via binomial tests) would strengthen the evaluation, as is standard in conformal prediction work.
- Discussing computational cost and scalability more explicitly, given the need to train both a quantile regressor and a weight model, and to optimize τ for each test point, would be helpful for practitioners.

## Removed Points
*These points are flagged to be removed, treat them with caution*

**Strength or Weaknesses that are removed:**
- **(Weakness - Overstated "exact" guarantee)**: The paper's use of "exact" coverage is slightly overstated, as Theorem 4.1 includes an error term dependent on weight estimation. However, this is a standard presentation in conformal prediction literature (e.g., weighted conformal prediction provides "exact" coverage conditional on the weight estimate), and the theorem clearly quantifies the error. This is more a matter of terminology than a substantive flaw.
- **(Weakness - Requires data splitting)**: The need for a separate calibration set is a inherent limitation of split-conformal methods, not a specific weakness of this paper's contribution. The paper follows standard practice in the field.
- **(Weakness - Limited comparison to non-conformal baselines)**: While broadening comparisons could be interesting, the paper's primary contribution is relative to other conformal methods for survival analysis. The chosen baselines ("Uncab", "Naive", "Focus", "Fused") are the most directly relevant state-of-the-art.
- **(Weakness from Spark Finder - Missing validation on realistic synthetic data derived from real covariates)**: This is a specific experimental suggestion that, while valuable, is not a core flaw. The paper's synthetic data is already designed to mimic realistic clinical trial scenarios (see Appendix C.1 and Table 3).

## Novel Insights
The paper's key novel insight is recognizing that, under standard causal assumptions, the problem of providing a marginal coverage guarantee for counterfactual survival times with right-censoring can be transformed into a covariate shift problem between the distribution of all covariates (P_X) and the distribution of covariates for uncensored, treated individuals (P_{X|W=w,e=1}). This shift can be corrected via reweighting, allowing the application of weighted conformal prediction to achieve a strong, non-asymptotic coverage bound. This insight elegably bridges causal survival analysis with the conformal prediction toolkit.

## Suggestions
- Provide a theoretical justification or proof sketch for why the post-hoc optimization of τ (to maximize the LPB for each test point) does not violate the marginal coverage guarantee, or at least discuss this point explicitly in the theory section.
- Add an experiment demonstrating the doubly robust property: for example, on a synthetic dataset, show that coverage remains valid when the quantile regression model is severely misspecified but the weight model is correct, and vice-versa.
- In the real-data experiment section, more clearly state that the coverage rate cannot be directly computed and that the "validity" claim is extrapolated from synthetic results and the plausibility of the derived LPBs. Consider adding a semi-synthetic experiment using the real covariates to bolster this claim.