I can't grep the PDF directly, but the paper text is provided in the user message. Let me verify the key claim about approximate H-calibration from the paper text:

The paper states in Section 2: "In the main body of the paper we focus on the H-calibration defined above, but in Appendix B we also discuss scenarios in which only approximate H-calibration is available."

This confirms the harsh critic's claim that approximate calibration is deferred to Appendix B and not treated in the main body.

Let me now also verify the claim about experiments being only 1D regression datasets:

From Section 5.1: "Bike Sharing (UCI). The UCI Bike Sharing (daily) dataset... The outcome Y ∈ [0, 1] is the rescaled total rider count" and "California Housing... median house values (rescaled to [0, 1])". Both are indeed 1-dimensional regression targets.

The action sets are small: |A| = 3 in both cases.

Now let me check whether there's any comparison with alternative robust baselines in the experiments:

The experiments compare "Plug-in" vs "Robust" under "i.i.d.", "Worst-case for robust", and "Worst-case for plug-in" settings. There's no comparison with other robust decision-making approaches.

Now I'll write the final review.

## Summary

This paper develops a minimax-optimal framework for decision making under partially calibrated forecasts. Given a forecaster satisfying H-calibration (where H specifies a class of test functions constraining forecast-residual correlations), the authors characterize the decision rule maximizing worst-case expected utility over all distributions consistent with the calibration guarantee. The main results are: (1) a duality-based characterization of the minimax-optimal robust policy (Theorem 3.1), (2) a sharp transition result showing that decision calibration—the weakest calibration notion that makes plug-in best response minimax-optimal—collapses the robust rule to best response (Theorems 4.1–4.2), and (3) concrete instantiations exploiting self-orthogonality from squared-loss training (Proposition 4.4) and bin-wise calibration (Proposition 4.5). Experiments on two 1D regression datasets validate the self-orthogonality instantiation.

## Strengths

- **Elegant and surprising theoretical result**: The identification of decision calibration as the sharp threshold where minimax-optimal robust decision making collapses to plug-in best response (Theorems 4.1–4.2) is a genuinely non-obvious finding. One might expect a gradual interpolation from pure minimax to best response as H enriches, but instead there is a phase transition—this is both theoretically interesting and practically significant, as it identifies a precisely calibrated target for forecaster design.

- **Unified duality characterization (Theorem 3.1)**: The result provides a principled and computationally tractable characterization of minimax-optimal policies for any finite-dimensional H. The reduction to pointwise optimizations (computing adversarial beliefs q* and best-responding) makes the framework implementable rather than purely conceptual, and the dual-variable interpretation is clean.

- **Practical H-classes from standard training procedures**: Proposition 4.4 (self-orthogonality from MSE training with linear heads) identifies a "free" calibration guarantee arising structurally from the most common training paradigm, requiring no special intervention. This gives the framework immediate applicability. Proposition 4.5 (bin-wise calibration → piecewise-constant robust rules) similarly leverages standard post-hoc methods.

- **Simultaneous optimality across decisions (Corollary 4.3)**: The extension showing that a single decision-calibrated forecaster simultaneously supports plug-in optimality for multiple downstream decision problems is a practical and conceptually clean insight.

- **Well-organized exposition**: The paper clearly motivates the problem, develops the framework incrementally, and provides both abstract and concrete instantiations. Figures 1 and 2 effectively communicate the interpolation property and sharp transition.

## Weaknesses

### Major:

1. **Exact H-calibration assumption in main theoretical results with no quantitative approximate analysis**: All main theorems (3.1, 4.1, 4.2, Corollary 4.3) assume perfect H-calibration. Approximate H-calibration is deferred to Appendix B with no formal quantitative results in the body. This is a significant concern because: (a) the "sharp transition" at decision calibration is proved only for *perfect* calibration—small violations could permit meaningful adversarial tilts, potentially undermining the crisp dichotomy the paper presents; (b) any practical procedure (including the self-orthogonality condition from Proposition 4.4, which assumes first-order stationarity of the population loss) yields only approximate calibration; (c) the paper's core motivation—full calibration is intractable in high dimensions, so we need weaker guarantees—implicitly acknowledges that perfect calibration is unrealistic, yet the main results all assume it. A formal approximate analogue bounding worst-case suboptimality of plug-in BR in terms of calibration error ε and properties of u would substantially strengthen the contribution. This is not a logical flaw—the exact results are correct and interesting—but it means the central "trustworthiness" semantics are established only in an idealized regime that is never met in practice.

2. **Narrow experimental scope that does not test the paper's central claim**: The experiments evaluate only the self-orthogonality instantiation (H = {h(v) = v}) on two 1D regression datasets with |A| = 3 actions and specific hand-crafted utility functions. Crucially, they do not test the paper's signature theoretical result—that decision calibration collapses the robust rule to best response—nor do they test any intermediate H-class between the two extremes. No multiclass setting is considered, despite the paper's motivation being precisely the regime where full calibration is intractable due to high-dimensional outcomes. The "adversarial" distributions are constructed from the same dual as the robust policy, making the evaluation somewhat circular (validating internal consistency rather than robustness to realistic shifts). The experiments confirm that the self-orthogonality dual behaves as predicted, but provide no evidence for the broader framework's value under decision calibration or richer H-classes.

3. **Linearity and finite action assumptions limit practical scope**: The paper assumes u(a, v) linear in v and A finite (Assumption 2.1). This is standard in the calibration literature and reasonable for multi-class expected-utility settings, but it excludes many high-stakes decision problems (healthcare, finance) involving risk-averse utilities, nonlinear costs, or continuous action spaces. The paper briefly acknowledges this in the conclusion and speculates about linearization via basis expansions (citing Gopalan et al., 2024b; Lu et al., 2025), but notes these bases "are not always low dimensional enough to be practical." This limitation is honestly disclosed but is consequential—it means the "trustworthiness" semantics apply to a narrower class of problems than the paper's high-stakes motivation suggests.

### Minor:

- **No measurement of calibration error in experiments**: The experiments rely on approximate self-orthogonality from training, but never report the empirical calibration error E[f(X)(Y − f(X))] on held-out data. Quantifying this gap would clarify how close the practical regime is to the theory's exact calibration assumption.

- **No comparison with alternative robust baselines**: The paper compares robust vs. plug-in policies but not against other approaches for robust decision-making under prediction uncertainty (e.g., distributionally robust optimization methods, conformal prediction-based rules). Including even one relevant baseline would clarify the value added by the H-calibration framework specifically.

- **Scalability of decision calibration across tasks (Corollary 4.3)**: The combined test class H^all_dec grows with the total number of actions across all decision problems, which could be large in practice. The paper does not discuss the sample/computational cost of achieving calibration for this union class.

### Trivial:

- No confidence intervals or standard deviations are reported for the experimental utility numbers in Table 1.

## Nice-to-Haves

- Experiments on multiclass data where the framework's advantages are most needed, ideally including a decision-calibrated or bin-calibrated model to test whether the collapse to best response is empirically observed.
- An approximate version of Theorem 4.1/4.2 bounding worst-case suboptimality in terms of calibration error.
- Visualization of the adversarial tilt q*(v) vs. raw forecast v to illustrate how the robust policy modifies beliefs.
- Evaluation under natural distribution shifts (e.g., temporal splits) rather than only adversarially constructed worst-case distributions.

## Removed Points

- **Formatting/style nitpicks**: The harsh critic mentions dense theoretical presentation with heavy notation. While notation is unavoidable in this type of work, this borders on style nitpick and is removed. — *Removed because this is a formatting/style nitpick per the rules.*

- **Claim that experiments are "circular" and therefore invalid**: The harsh critic argues that constructing adversaries from the same dual as the robust policy makes the evaluation circular. However, this experimental design directly tests the theory's predictions (that the robust policy dominates worst-case adversaries consistent with H-calibration), which is a valid and standard way to validate minimax results. The criticism about lacking real distribution shifts is legitimate (and kept above), but the "circularity" framing overstates the issue. — *Removed because it mischaracterizes a valid experimental design choice; the real concern (lack of natural shifts) is captured in a less extreme form.*

- **Claim that no comparison with decision calibration in experiments invalidates the paper**: The spark reviewer and harsh critic emphasize that the central claim (decision calibration → best response) is untested empirically. This is a valid concern (kept as a major weakness), but the claim that this *invalidates* the paper overstates the case—theoretical papers need not always experimentally validate every theorem, and the decision-calibration result is cleanly proved. The concern is about the scope of validation, not the validity of the theory. — *Partially removed: kept as a concern about experimental scope, but not as a claim that the theory is unsound.*

- **Demand for user studies or risk-averse utility experiments**: The neutral reviewer requests evaluation on "genuine decision-making scenarios" like medical treatment decisions. Since the paper is theoretical and explicitly scopes itself to linear utility, this is scope creep. — *Removed as a demand outside stated scope.*

- **Demand for continuous action space results**: The spark reviewer and others ask for extension to continuous action spaces. The paper clearly defines its scope as finite A and acknowledges this limitation. — *Removed as scope creep; the paper already honestly discloses this limitation.*

## Novel Insights

The key novel observation across the reviews is that the paper reveals a previously unknown *structural* property of the calibration hierarchy: it is not a continuum of increasingly conservative policies as H enriches, but rather exhibits a sharp phase transition at the decision-calibration boundary. This means that the practical guidance for downstream decision-makers is binary and crisp—either your forecaster is decision-calibrated for your problem (in which case simply best-respond), or it isn't (in which case you should use the duality-based robust policy). This eliminates the need for a practitioner to carefully calibrate their level of conservatism to the degree of partial calibration, at least above the decision-calibration threshold.

## Suggestions

1. Add a formal approximate version of Theorems 4.1/4.2, even as a proposition with a bound on worst-case suboptimality as a function of ε-H-calibration error. This would bridge the gap between the idealized exact theory and practical application.
2. Include at least one experiment on a multiclass classification task (even if small-scale like a tabular dataset with ~5-10 classes) using bin-wise calibration or a richer H-class, to demonstrate the framework where it is most needed.
3. Report empirical calibration error (E[h(f(X))(Y − f(X))]) on held-out data in the experiments, and discuss how closely the practical regime approximates exact H-calibration.

## Score and Decision

**Calibration anchors:**

- *Provable Uncertainty Decomposition via Higher-Order Calibration* (TId1SHe8JG.md): scores 6/8/8/8, Accept (Spotlight). Strong calibration theory paper with formal guarantees and experimental validation of theoretical claims on real data (image classification). More complete empirical evaluation than the current paper.

- *Reconciling Model Multiplicity for Downstream Decision Making* (uy4EavBEwl.md): scores 6/6/6/6, Accept (Poster). Related paper combining multi-calibration with downstream decision-making. Has novel theory + algorithm + experiments. The current paper has stronger standalone theoretical contributions (the sharp transition result) but weaker experiments.

- *Does Calibration Affect Human Actions?* (XM7INBbvwT.md): scores 3/5/6, Reject. Calibration + decision making paper with limited methodology.

The current paper has a genuinely novel and surprising theoretical contribution (the sharp transition at decision calibration), a clean duality characterization, and practical instantiations. This is strong theoretical work. The main weaknesses are the exact-calibration assumption without approximate analysis, and experiments that test only a very narrow slice of the framework. The theoretical core is stronger than *Reconciling Model Multiplicity* (which got 6s across the board), but the experimental gap is significant. Compared to *Higher-Order Calibration* (Spotlight with 6/8/8/8), the current paper has comparable theoretical novelty but substantially thinner experiments.

Overall this paper makes a real and interesting theoretical contribution—it's definitely above the acceptance threshold but has meaningful gaps that prevent it from scoring higher. The theory is sound and intellectually compelling; the experiments support a narrow claim but not the broader framework.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>