Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

The paper proposes D-PDDM, a two-stage disagreement-based algorithm for monitoring post-deployment model deterioration (PDD) without access to labels or raw training data during deployment. D-PDDM reformulates PDD monitoring as detecting whether an auxiliary model's disagreement with the deployed model increases under the deployment distribution (D-PDD), and provides finite-sample FPR and TPR guarantees for this detection. The algorithm decouples pre-training (computing a hypothesis subset H_p and disagreement threshold distribution Φ) from deployment monitoring (comparing maximum disagreement on new data against Φ), enabling training-data-free and scalable operation.

## Strengths

- **Clear desiderata-driven framing with a novel combination of properties.** The paper identifies three practical desiderata (unsupervised, training data-free during deployment, robust to non-deteriorating shifts) and Table 1 systematically shows that no prior method satisfies all three. D-PDDM's two-stage decoupling is a genuine architectural contribution that enables training-data-free monitoring — a real practical benefit for privacy-constrained or large-scale ML pipelines (Section 3, Algorithms 1–2).

- **Formal FPR and TPR guarantees.** Theorem 4.2 bounds FPR at α + exponentially decaying term under D-PDD non-deteriorating shifts, and Theorem 4.4 provides sample complexity O(((1+√δ)/(ξ−2ε_f))²(d_p+ln(1/β))) for high TPR under deteriorating shifts. The deployment sample complexity depends on d_p (which can be ≪ d when f is well-trained), a concrete advantage over naive approaches (Section 4.2–4.3).

- **Honest characterization of failure modes.** Theorem 4.5 explicitly identifies Regime 2 (deteriorating shift but ε_q ≤ ε_p) where TPR is only O(α), and Section 4.3.1 with Figure 3 provides a geometric illustration showing how reducing ε_f can move into solvable regimes. This self-critical analysis strengthens trust in the framework.

- **Proposition 4.1 provides interpretable structure.** Equation 8 (ξ = TV − 2η) shows D-PDD depends on both the magnitude of distribution shift and the approximation capacity of the hypothesis class, offering practical guidance on model selection (Section 4.1).

- **Real-world healthcare validation.** Experiments on the GEMINI dataset with temporal and age-based shifts demonstrate applicability beyond synthetic settings (Figures 5–6), showing D-PDDM maintains low FPR on temporal non-deteriorating shifts and competitive TPR on age-based deteriorating shifts.

## Weaknesses

### Fatal
None.

### Major

- **The FPR guarantees apply to D-PDD non-deterioration, not PDD non-deterioration, and this distinction is underemphasized.** Theorem 4.2 guarantees low FPR when ∀h ∈ H_p: err(h; Q_f) ≤ err(h; P_f) (Eq. 10), which is "no D-PDD." This is not equivalent to no actual performance deterioration (no PDD). A shift where model performance is stable but disagreement increases would be flagged as "deteriorating" by D-PDDM despite no real performance loss. The abstract claims "low false positive rates under non-deteriorating shifts" without qualifying that this is relative to the D-PDD definition. Lemma 2.1 establishes equivalence only under g=g' and bounded TV, with probability 1−2ε_f−κ — conditions that exclude concept drift and label shift, the very shifts most likely to cause real deterioration. This gap between D-PDD and PDD is the paper's central conceptual issue and is not sufficiently confronted.

- **The theory-practice gap for H_p computation is unanalyzed.** Algorithm 1 requires computing H_p = {h ∈ H: err(h; P_g) ≤ ε} and solving max_{h ∈ H_p} err(h; D^m) per round. For neural network hypothesis classes, this is a constrained non-convex optimization problem with no guaranteed solution. The paper defers to a Bayesian posterior sampling approximation (Section 3, "Practical considerations") but the theoretical guarantees depend on exact computation. The gap between the theory (exact optimization over H_p) and the implementation (posterior sampling) is never analyzed, so the "provable" guarantees do not formally apply to the algorithm actually used in experiments.

- **Experiments do not test the critical failure mode — disagreement increase without actual performance degradation.** The non-deteriorating shift scenarios (synthetic data with Δ parameter stretching features away from the decision boundary; GEMINI temporal shift) are constructed such that model performance (AUROC) does not drop. These scenarios are also ones where disagreement doesn't increase much — they are D-PDD non-deteriorating by construction. The experiments never test the scenario where model performance is stable (no PDD) but disagreement increases (D-PDD fires), which is the false-positive scenario that practitioners would most care about. Validating low FPR on D-PDD-non-deteriorating shifts is somewhat circular when the test statistic and the condition being verified are aligned by construction.

### Minor

- **Lemma 2.1's equivalence can be weak in practice.** With ε_f = 0.15 and κ = 0.3, the equivalence probability is only 0.4. The paper acknowledges the assumptions but doesn't characterize how often the equivalence holds for realistic parameter values. This limits confidence in the D-PDD proxy for actual PDD.

- **Regime 2's "solution" (train f better) has practical limitations.** Section 4.3.1 suggests reducing ε_f to move out of Regime 2, but the user of a monitoring system may not control f's training, and even well-trained models have non-trivial ε_f on complex tasks. The paper is transparent about this but the remedy is limited.

- **The sample complexity in Theorem 4.4 depends on (ξ − 2ε_f), which can be near-zero.** For typical ML models with ε_f ≈ 0.05–0.15, if ξ ≈ 2ε_f (a realistic scenario for moderate shifts), the sample complexity bound becomes vacuous. The paper doesn't discuss when this denominator is meaningfully positive in practice.

### Trivial
None.

## Nice-to-Haves

- An experiment explicitly testing the PDD-stable / D-PDD-positive scenario (disagreement increase without performance degradation) would substantially strengthen the empirical validation and help practitioners understand the method's practical false positive behavior.

- Analysis of the approximation gap between exact H_p optimization and Bayesian posterior sampling would connect the theory to the implementation.

- Comparison with methods using pre-computed training statistics (e.g., BBSD with cached softmax outputs) would more fairly test the "training data-free" claim.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic: "Proposition 4.1 contradicts the goal of distinguishing deteriorating from non-deteriorating shifts."** The paper itself discusses this tradeoff at length after Eq. 8, noting that for simple function classes η can be high and ξ may not be positive. This is an informative characterization, not a contradiction. The paper uses this to explain when D-PDD is or isn't triggered, which is the point.

- **Harsh Critic: "Table 1 'Training data-free' is misleading because H_p and Φ are compressed forms of training data."** The paper's claim is specifically about not requiring training data *during deployment*. H_p and Φ are pre-computed and stored, so the monitoring stage genuinely does not access raw training data. This is a meaningful operational distinction — the baselines require runtime access to training data or sufficient statistics computed from it.

- **Harsh Critic: "Synthetic experiment gives baselines oracle access to generating distributions, which advantages them."** This actually makes the comparison *harder* for D-PDDM, not easier, since it gives baselines an advantage they wouldn't have in practice. The paper explicitly notes this is to "empower the baselines" (Section 5.1). Per the rules, this is an asymmetry that favors baselines, not the authors' method.

- **Harsh Critic: "AUROC can mask calibration degradation."** This is a generic criticism that could apply to any paper using AUROC as a performance metric. It's outside the paper's stated scope and not a specific weakness of this work.

- **Strength Finder: "D-PDDM is the first method that simultaneously satisfies all four desiderata in Table 1."** This strength conflicts with the verified Major weakness that the FPR guarantees are for D-PDD non-deterioration, not PDD non-deterioration. The "Non-deteriorating" checkmark in Table 1 is qualified in a way that weakens this claim.

- **Strength Finder: "Empirical validation of low FPR under non-deteriorating shifts — Figure 4."** This is circular given the Major weakness about not testing the D-PDD/PDD gap. The FPR is low on D-PDD-non-deteriorating shifts because the test statistic tracks disagreement, which is what D-PDD measures. This doesn't validate robustness to non-deteriorating shifts in the PDD sense.

## Novel Insights

The paper reveals a fundamental tension in unsupervised monitoring that deserves broader recognition: any monitoring method that replaces the unobservable quantity of interest (actual performance deterioration) with an observable proxy (disagreement) must accept either restrictive assumptions for equivalence (as in Lemma 2.1) or a definitional gap that the experimenter must confront. D-PDDM's Proposition 4.1 (ξ = TV − 2η) crystallizes this tradeoff elegantly: more expressive function classes detect more shifts as "deteriorating" (low η → ξ > 0 for most shifts) but also produce more false alarms, while simpler classes are more selective but may miss genuine deterioration. This expressivity-selectivity tradeoff in disagreement-based monitoring is a useful conceptual contribution that extends beyond this specific algorithm.

## Suggestions

- Add an experiment that constructs a shift where err(f; Q_g') ≤ err(f; P_g) but max_{h∈H_p} err(h; Q_f) > max_{h∈H_p} err(h; P_f) — i.e., performance is stable but disagreement increases. This is the most important experiment missing from the paper and would directly address the D-PDD vs. PDD gap concern.

- Qualify the abstract and introduction claims more carefully: replace "low false positive rates under non-deteriorating shifts" with "low false positive rates under shifts that do not increase model disagreement" or explicitly note the D-PDD vs. PDD distinction upfront.

- Add even a brief discussion of the Bayesian approximation gap — either a theoretical bound on how posterior sampling approximates exact H_p optimization, or an empirical ablation varying the number of posterior samples.

## Score and Decision

**Calibration anchors used:**

- **High (7+):** "Querying Easily Flip-flopped Samples" (avg 7.5, poster) — shares model disagreement framework with provable estimators; this paper is weaker because the D-PDD/PDD gap and theory-practice gap are more severe than the mild empirical improvements in the LDM paper. "Entropy is not Enough for TTA" (avg 7.0, spotlight) — shares distribution shift monitoring focus with model disagreement; this paper has cleaner empirical validation but the D-PDDM paper has stronger formal guarantees.

- **Medium (4–6):** "Model Equality Testing" (avg 6.5, poster) — shares post-deployment monitoring with theoretical guarantees; this paper is comparable but has a cleaner theory-practice alignment. "Online change-point detection for LLMs" (avg 5.25, reject) — shares online monitoring with FPR guarantees; D-PDDM is stronger due to more formal theory and healthcare validation. "Explanation Shift Detector" (avg 5.25, reject) — shares shift detection with model behavior focus; D-PDDM has stronger theoretical grounding.

- **Low (<3):** "Weight Decay induces low-rank bias" (avg 2.2, reject) — overclaimed theoretical guarantees with gap between theory and practice; D-PDDM is substantially better because it has genuine contributions and is transparent about limitations, unlike the low-scoring papers which have vacuous or misleading claims. "Exact linear-rate gradient descent" (avg 2.5, reject) — theoretical formula depends on unknowable quantity; D-PDDM's theory is more applicable.

The paper has genuine contributions — a novel monitoring framework, formal guarantees, and real-world validation — that place it above the rejected medium anchors. However, the D-PDD vs. PDD definitional gap, the unanalyzed theory-practice gap for H_p computation, and the untested critical failure mode are significant weaknesses that place it below the accepted medium anchors (like Model Equality Testing at 6.5). The paper is more substantial than the typical 5-score rejection but the overclaiming of "provable robustness under non-deteriorating shifts" (when this is only for D-PDD non-deterioration) and the circular experimental validation keep it from a clear accept. Score: 5.5 — borderline, leaning reject due to the gap between the strength of the claims and the strength of the evidence.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>