## Summary
This paper studies robust decision making when given forecasts that satisfy only partial calibration guarantees, formalized as H-calibration. The authors adopt a minimax perspective: choose a decision policy that maximizes expected utility under the worst-case distribution consistent with the promised H-calibration constraints. They characterize the optimal robust policy via a duality argument, showing it is always a best response to an adversarially adjusted belief. A key theoretical result is that when H includes the tests for decision calibration, the robust policy collapses to the simple plug-in best response—meaning that this tractable calibration notion recovers the same strong decision-theoretic guarantee as full calibration. The paper also instantiates the framework for common calibration notions (e.g., self-orthogonality from squared-loss training) and provides experiments on regression datasets.

## Strengths
- **Novel theoretical framework:** The paper introduces a minimax robust decision-making framework for partially calibrated forecasts, bridging calibration theory and robust optimization in a fresh and principled way.
- **Key theoretical insight:** It proves that under decision calibration (a tractable condition), the robust policy collapses to the plug-in best response, recovering the strong “trustworthiness” semantics of full calibration. This surprising result identifies a practical calibration target that suffices for optimal decision making.
- **Practical algorithmic implications:** The paper derives efficiently computable optimal policies for common calibration notions (e.g., self-orthogonality from squared loss, bin-wise calibration), making the framework applicable to standard training pipelines without requiring new calibration procedures.
- **Clear and rigorous exposition:** The paper is well-structured, with a precise problem formulation, thorough theoretical analysis (including duality and special cases), and complete proofs in the appendix.

## Weaknesses
### Major:
- **Missing empirical validation of the central theoretical claim:** The paper’s key result is that decision calibration leads to plug-in optimality, but the experiments only test a much weaker calibration condition (self-orthogonality). There is no demonstration with a decision-calibrated forecaster, leaving a gap between theory and empirical support.
- **Narrow experimental scope:** Experiments are limited to two regression datasets with one-dimensional outcomes and a small, discrete action set (three actions). The paper motivates the problem for high-dimensional multiclass prediction, but does not demonstrate the framework on such tasks, limiting evidence of its practical applicability.
- **Restrictive utility assumption:** The entire analysis requires utilities that are linear in the outcome probabilities (Assumption 2.1). This excludes many real-world decision problems with risk-averse or nonlinear utilities, and the paper does not explore the consequences of violating this assumption.
- **Lack of comparison to alternative robust baselines:** The paper only compares the proposed robust rule to the plug-in rule. There is no comparison to other distributionally robust optimization methods or conformal prediction-based decision rules, making it hard to assess the relative merits of the calibration-based approach.
- **Scalability concerns for large H or action sets:** The paper does not discuss the computational complexity of solving the dual and pointwise minimizations when H is large (e.g., decision calibration with many actions) or when the action set is large. This is important for practical deployment.

### Minor:
- **Adversarial shifts are synthetic:** The worst-case distributions used in experiments are constructed to be worst-case for the plug-in rule while respecting calibration constraints. While theoretically valid, this does not demonstrate robustness under more realistic distribution shifts (e.g., covariate shift, label shift) that might occur in practice.

### Trivial:
*(none)*

## Nice-to-Haves
- Experiments with a post-hoc bin-wise calibration procedure (Proposition 4.5) to show how the framework works with a common, tractable guarantee.
- A discussion or simple experiment quantifying how the robust policy behaves as H approaches the decision calibration set, to illustrate the “sharp transition” more concretely.
- Visualization of the worst-case adjustment q*(v) versus the raw forecast v to help interpret how the robust rule modifies predictions.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **Reproducibility concerns about undisclosed hyperparameters or missing code:** The paper provides sufficient experimental details (datasets, model architecture, splits, utility functions) for a theory paper; requiring full code release or exhaustive hyperparameters is not standard for ICLR.
- **Criticism that the finite-action assumption is a major limitation:** The paper explicitly assumes finite action sets and does not claim to handle continuous actions; this is a scope limitation, not a flaw in the contribution.
- **Request for detailed discussion of how to achieve or verify calibration guarantees in practice:** The paper’s focus is on decision-making given the guarantees, not on obtaining them; such discussion would be scope creep.

## Suggestions
- **Add experiments with decision calibration:** Implement a forecaster that is (approximately) decision-calibrated for a given utility function, and show that the robust rule indeed collapses to the plug-in best response, validating Theorem 4.1.
- **Expand experiments to a multiclass setting:** Test the framework on at least one multiclass classification dataset with a non-trivial utility function to demonstrate applicability to high-dimensional outcomes.
- **Include comparisons to other robust decision-making baselines,** such as distributionally robust optimization with Wasserstein balls or moment constraints, to better position the calibration-based approach.
- **Discuss computational scalability:** Provide an analysis of how the solution complexity scales with |H| and |A|, and suggest approximations for large-scale problems.