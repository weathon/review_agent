Now I have a comprehensive understanding. Let me synthesize my final review.

Key assessment points:
1. The paper offers a comparative study of three PI measures with theoretical analysis + empirical evaluation.
2. The "unfair comparison" argument from the harsh critic is partially valid but overstated. PVI inherently requires model estimation (it's part of the V-information definition), so it's not that the authors chose a more expensive estimator for fairness - it's intrinsic to the method. However, the computational cost asymmetry is real and underdiscussed. More importantly, the paper should compare against deep ensembles (same computational budget) to show PVI's advantage isn't just from having twice the model capacity.
3. The calibration comparison (Table 3) only compares PVI against MSP and SM, justified by the "requires confidence between 0 and 1" constraint. But temperature scaling, isotonic regression etc. also produce values in [0,1], so this justification is weak. This is a genuine gap.
4. The theoretical framework doesn't perfectly predict PVI's empirical superiority (PSI has higher margin correlation), but the paper acknowledges and discusses this in Section 5, offering a plausible explanation.
5. The paper has genuine theoretical contributions: invariance analysis (Prop 1-3), margin sensitivity (Prop 4, Theorem 1, Prop 5), convergence rates.
6. The Limitations section briefly acknowledges training additional models.

Now for scoring: The paper has real contributions (novel theoretical analysis of PI measures for confidence, systematic empirical comparison) but has substantive weaknesses (computational cost asymmetry, limited calibration baselines, claimed superiority is partly from model capacity). This is similar in profile to YUefWMfPoc (avg 5.75, Reject - similar topic, incomplete baselines) but has additional theoretical contributions. 

I'd place this around 4.5-5.5 range. The unfair comparison concern is real but not fatal since PVI's advantage is inherent to its definition, not an arbitrary choice. The limited calibration baselines are a moderate concern. The theoretical disconnect where PSI has better margin correlation but PVI performs better on tasks is acknowledged and partially explained.

Score: approximately 5</think>## Summary

This paper proposes using three pointwise information (PI) measures—PMI, PVI, and PSI—as post-hoc confidence estimators for deep neural networks, comparing their theoretical properties (invariance, margin sensitivity, convergence rates) and empirical performance on failure prediction and confidence calibration tasks. The main finding is that PVI consistently outperforms PMI, PSI, and standard baselines (MSP, SM, etc.) for confidence estimation, which the authors argue is consistent with PVI having the best balance of invariance and sensitivity properties.

## Strengths

- **Rigorous theoretical comparison of PI measures (Section 3):** The invariance analysis (Propositions 1–3), margin-sensitivity results (Proposition 4, Theorem 1, Proposition 5), and convergence discussion provide useful, non-obvious formal characterizations. Proposition 4's finding that PMI is constant (log 2) for non-overlapping balanced distributions is a genuine insight that motivates PMI's limitations.

- **Systematic experimental evaluation (Section 4, Tables 1–3):** The paper evaluates on four model-dataset combinations with multiple metrics (AUROC_f, AUPR_success, AUPR_error, AURC, ECE), reports standard deviations across 5 runs, and compares 9 methods. The UMAP visualizations (Figure 1) complement the quantitative results.

- **Honest reporting of PSI's margin superiority (Table 1):** Despite favoring PVI, the paper honestly reports that PSI has the highest correlation with margin (0.657–0.846 vs. PVI's 0.327–0.563), and provides a thoughtful discussion of this disconnect in Section 5 (lines 316), distinguishing "sensitivity to decision boundaries" from "predictive reliability."

## Weaknesses

### Fatal
None.

### Major

- **PVI's computational cost asymmetry undermines the strength of claimed superiority over baselines.** PVI requires training an additional full neural network with the same architecture as the original model (line 78: "training two neural networks"), while standard post-hoc baselines (MSP, SM, ML, LM, NE, NG) require no additional training, PMI uses a "shallow 2-layer neural network" (line 56), and PSI uses Gaussian/binning estimators with 500 projections (line 64). The paper's headline claim that PVI "outperforms all existing baselines for post-hoc confidence estimation" does not account for this drastic resource asymmetry. While the paper acknowledges in Limitations (line 320) that "PI measures require training additional models," it does not discuss this confound in the experimental analysis, nor does it compare PVI against a computationally comparable baseline such as a 2-model deep ensemble. Without this comparison, it is impossible to determine whether PVI's advantage arises from its information-theoretic formulation or simply from having access to a second trained model—a crucial distinction for the paper's claims.

- **The calibration experiment (Table 3) omits standard calibration baselines.** The paper justifies comparing only MSP and SM for calibration by stating "confidence calibration requires the confidence estimates to be between 0 and 1" (line 294). However, temperature scaling, isotonic regression, and Platt scaling all produce well-calibrated probabilities in [0, 1], and are the most natural and widely-used baselines for ECE evaluation. Citing Zhu et al. (2022) that calibration methods can be "harmful for failure prediction" does not justify their omission from the calibration experiment specifically. This makes the calibration claim—"PVI significantly outperforms...other benchmark post-hoc methods" (line 296)—unsubstantiated for the calibration setting.

- **The theory in Section 3 does not cleanly predict PVI's empirical advantage.** The paper claims T5 (line 227) that "our theoretical studies predict that PVI may perform the best overall," but the component arguments are partly in tension: T2 argues PSI has *too little* invariance (not invariant to general linear transforms), while T3's own experiments show PSI has the *highest* margin correlation. The explanation in Section 5 that "better sensitivity to margin doesn't necessarily imply better performance" (line 316) is plausible but post-hoc—it was not predicted by the theory. The convergence argument in T4 is also speculative ("we hypothesize"), with no convergence experiments presented. The "prediction" in T5 is more of a qualitative synthesis than a testable theorem.

### Minor

- **PVI's performance improvement over baselines is modest on some metrics and datasets.** On VGG16/STL-10, PVI's AUROC_f (89.35 ± 0.63) is within standard deviations of MSP (88.48 ± 0.97), SM (88.47 ± 0.88), and NE (88.54 ± 0.63). For AUPR_success, multiple baselines are actually bolded alongside PVI. Similarly, in Table 3, PVI's ECE on VGG16/STL-10 (4.91 ± 2.63) overlaps substantially with MSP (7.42 ± 3.09) given the large variances. The sweeping claim of consistent superiority is only strongly supported on AUPR_error and AURC metrics.

- **The PVI estimator's dependence on the original model's architecture is noted but underexplored.** Line 278 states "the way PVI is defined is architecture-dependent, the definitions of PMI and PSI are not," but no ablation studies test how sensitive PVI's advantage is to architectural choices or to the choice of PVI estimator variant (the appendix compares three approaches, but the main experiments use only one).

### Trivial
None.

## Nice-to-Haves

- **Deep ensemble comparison:** Since PVI requires training a second full model, comparing it against a 2-model deep ensemble would clarify whether PVI's gains come from its information-theoretic formulation or simply from having double the model capacity.

- **Broader calibration baselines:** Adding temperature scaling, Platt scaling, or isotonic regression as baselines in Table 3 would substantially strengthen the calibration claim.

- **Computational cost analysis:** Reporting training time, parameter counts, and inference cost per method would help practitioners assess the practical trade-offs.

- **Ablation on PVI estimator variants:** Testing with shallower or otherwise constrained PVI estimators would help isolate the contribution of the information-theoretic measure versus the model capacity.

## Removed Points

- **"PVI is not 'post-hoc'" (from Harsh Critic):** This overstates the issue. PVI does not modify the original model's architecture or training procedure, which is the standard meaning of "post-hoc" in this context (line 19). PVI estimates a quantity *about* the original model's behavior from a separate auxiliary model. Whether it's "post-hoc" is a terminological question, not a methodological flaw. The real concern is the computational cost asymmetry, which is retained above as a Major weakness.

- **"The connection to probabilistic causation is superficial" (from Harsh Critic Section Notes):** The paper presents this as one of four motivating factors, not as a core theoretical contribution. The connection to P(Y|X) > P(Y) is a brief motivation (lines 33–34), and the paper does not overclaim its depth. Removing this weakens the narrative only slightly.

- **"Missing Deep Ensemble comparison" demanded as a critical flaw:** This is a valid concern but falls under Nice-to-Have rather than Fatal. The paper's scope is comparing PI measures against existing post-hoc confidence estimators; deep ensembles are a different paradigm (requiring multiple models trained end-to-end). However, given PVI's cost profile, the comparison would indeed be informative.

- **"Missing appendix/proofs" (potential concern):** The parser strips appendices. The paper references Appendices B, C, D, etc., which are standard and should be assumed to exist.

- **"Demanding convergence experiments" (from Harsh Critic):** The paper presents convergence rate theory in Appendix B.3 and argues T4 based on it. Whether to run explicit convergence experiments is a nice-to-have, not a methodological gap, for a paper that combines theory and experiments.

## Novel Insights

The most interesting observation from the reviews is the tension between PSI's superior margin correlation (Table 1) and PVI's superior downstream performance (Tables 2–3). The paper's explanation—that margin correlation captures "sensitivity to decision boundaries" rather than "predictive reliability"—is plausible but highlights an important gap: if the geometric properties that theory predicts should matter (margin) don't actually determine the best measure, then the theoretical framework needs either refinement (identifying which geometric properties *do* predict downstream success) or a clearer admission that the theory provides only qualitative guidance, not quantitative predictions.

## Suggestions

- Add a 2-model deep ensemble baseline (same computational budget as PVI) to Table 2 to isolate PVI's information-theoretic contribution from its model capacity advantage.
- Add temperature scaling and/or isotonic regression baselines to Table 3, as these are the standard methods for ECE evaluation.
- Tone down the claim of "outperforming all existing baselines" (Abstract, line 15) to acknowledge the computational cost asymmetry—e.g., "outperforms all other post-hoc baselines at comparable or lower computational cost" for the baselines that are actually lower-cost, and qualify PVI's claim separately for its cost tier.
- Report quantitative computational cost (training time, parameter counts) for each PI measure's estimator in the appendix.

## Score and Decision

**Calibration anchors:**

- **High (avg > 7):** ValUES framework (avg 7.5, Accept oral) — comprehensive systematic evaluation of uncertainty methods with thorough ablations and baselines. Our paper has similar scope but less comprehensive baselines and a computational cost confound. uSz2K30RRd (avg 7.33, Accept Spotlight) — strong theoretical contribution on PMI in contrastive learning.

- **Medium (avg 4–6):** YUefWMfPoc (avg 5.75, Reject) — directly analogous: post-hoc confidence estimation for selective classification. Reviewers flagged incomplete baseline comparison for calibration methods. Our paper shares this weakness and adds a computational cost asymmetry issue. jTnHyyGYy2/LoRA-Ensemble (avg 4.5, Reject) — efficient ensemble for uncertainty; reviewers flagged missing compute-matched baselines.

- **Low (avg < 3):** qgLyKwXVDs/FreeLM (avg 2.0, Reject) — unfair comparison claims. AL4tS0HhJT (avg 2.5, Withdrawn/Reject) — confidence estimation with no baselines.

Our paper is stronger than the low anchors (it has genuine theoretical contributions, systematic experiments, and honest reporting). It is weaker than the high anchors (incomplete calibration baselines, computational cost confound). It is comparable to the medium anchors like YUefWMfPoc (avg 5.75, Reject), but has additional theoretical depth that YUefWMfPoc lacked. The computational cost asymmetry is a real concern but is inherent to PVI's definition rather than a deceptive experimental choice—the paper just should have been more transparent about it and controlled for it.

**Assessment by axis:**
- Originality: Moderate — comparative study of existing measures applied to a new task, with novel theoretical analysis
- Importance: Moderate — confidence estimation is important; PI measures are an interesting direction
- Claims support: Weakened by the computational cost confound and incomplete calibration baselines
- Experiments: Systematic but with key gaps
- Clarity: Good — well-structured paper
- Community value: Useful as a reference for PI measures in this context

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>