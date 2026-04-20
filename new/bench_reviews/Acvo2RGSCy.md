## Summary

This paper introduces DeLLMa, a framework that structures LLM inference-time reasoning through classical decision theory (expected utility maximization) to improve decision-making under uncertainty. The paper demonstrates consistent accuracy gains over baselines across agriculture and finance domains, provides systematic analysis of inference-time compute scaling, and offers human-auditable decision outputs. The core contribution is practical and well-motivated, though the evaluation design has some limitations in scope.

## Strengths

- **Clear problem framing with principled approach**: The paper cleanly separates the decision-making pipeline into state enumeration, forecasting, utility elicitation, and optimization (Section 3), providing an interpretable alternative to end-to-end prompting. The mathematical formalization (Equations 1-2) grounds the approach in established decision theory.

- **Consistent empirical improvements**: Table 1 shows well-calibrated state forecasts (ECE 0.062-0.142), and Figure 2/4 demonstrate that DeLLMa consistently outperforms zero-shot, CoT, and self-consistency baselines. The improvement is more pronounced on larger action sets, supporting the paper's claim about scalability.

- **Practical value of human auditability**: The decision tree visualizations (Figure 3 right, Figure 4 right) concretely demonstrate how the modular structure allows users to inspect intermediate components (state probabilities, utility assignments). This addresses a real need in high-stakes decision support.

- **Systematic ablation of compute scaling**: Figure 3 left provides actionable guidance on the trade-off between inference-time compute (sample size, overlap percentage) and accuracy, with approximately linear scaling observed for both DeLLMa-Pairs and DeLLMa-Top1. This empirical characterization is useful for practitioners.

## Weaknesses

### Major

- **Limited scope of o1 comparison**: Table 3 compares DeLLMa against o1-preview running in zero-shot mode only. While this shows value for the specific claim that structured decision scaffolding improves over zero-shot prompting, it does not address whether DeLLMa outperforms o1 when o1 uses its native chain-of-thought reasoning capabilities. The claim in Section 4 that DeLLMa benefits from "specialized inference-time reasoning for decision making under uncertainty" would be stronger with a comparison against o1 with its internal reasoning steps activated. This constrains the scope of the inference-time scaling claim—it demonstrates value for structured prompting vs. zero-shot, but not necessarily superiority over the full reasoning capabilities of modern models.

### Minor

- **Independence assumption in state forecasting may generate unrealistic joint states**: Algorithm 1 constructs the joint distribution as a product of marginals over latent factors (line 144: π(f₁, ..., fₖ|C) := ∏ᵢ πᵢ(·|C)). The paper acknowledges this is "for computational simplicity" (Section 3.2), but in domains like agriculture and finance, latent factors are naturally correlated (e.g., climate conditions affect yield and price simultaneously). Table 2 shows robustness to misspecified states, suggesting the utility elicitation step can partially compensate, but it remains unclear how often the sampled joint states represent realistic combinations.

- **Small dataset size**: Both domains use 120 instances (Section 4.1, 4.2). While this is reasonable for a proof-of-concept, it limits confidence in the generality of findings, particularly for the o1 comparison and scaling laws which are evaluated on the same small set.

- **Heuristic verbalized probability mapping**: Section 3.2 uses a dictionary V mapping verbalized probabilities ("very likely", "likely", etc.) to numerical values without justification or calibration analysis. This introduces unquantified noise into the posterior distribution π^{LLM}, and the mapping may vary across models and tasks. The paper does not ablate this choice against direct numerical probability elicitation or log-probabilities.

### Trivial

- **Normalized utility results deferred to appendix**: The paper mentions normalized utility in Section 4 ("Evaluation Metrics") but only reports accuracy in the main text, deferring utility comparisons to Appendix B. This makes the main results harder to fully interpret without consulting the appendix.

## Nice-to-Haves

- Evaluate against o1 with its native reasoning mode (chain-of-thought, self-verification) to provide a more complete picture of the inference-time scaling trade-off.
- Investigate risk-sensitive decision-making by modeling utility variance or concavity, which would align more closely with classical decision theory's treatment of risk attitudes.
- Analyze failure cases where the independence assumption produces unrealistic joint states and demonstrate how the utility elicitation step handles these.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Ex-post evaluation critique**: The harsh critic argues that evaluating against "ground-truth optimal action" (the action with highest realized future price×yield) conflates decision quality with forecasting luck. However, this is a standard evaluation paradigm in ML decision-making under uncertainty—evaluating whether the model can predict which action will perform best given available information. The paper's framing is valid: it demonstrates improved outcome quality (choosing the action that actually performs best), not necessarily improved process quality (whether the reasoning was rational given information at decision time). The metric is appropriate for the claimed contribution.

- **Strawman o1 comparison**: While the comparison is limited (zero-shot only), it is not methodologically unsound. The comparison is valid for demonstrating the value of structured decision scaffolding vs. unstructured prompting. The claim is specific and supported—the paper does not claim to beat o1 overall, just that DeLLMa+GPT-4 beats o1-zero-shot on these tasks.

- **Independence assumption "invalidates the core claim"**: The paper explicitly acknowledges the independence assumption and provides empirical evidence (Table 2) that performance is robust to state misspecification. This is a design trade-off, not a fundamental flaw.

- **Missing prediction market literature**: The reviewer suggests related work on prediction markets, but the paper's contribution is about structuring LLM reasoning, not about forecasting market mechanisms. The related work coverage is adequate for the paper's scope.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Clarify the scope of the o1 comparison**: Add a note in Section 4.3 explicitly stating that the o1 comparison is against zero-shot mode, and acknowledge that a comparison against o1 with native reasoning would be useful future work.

2. **Characterize the verbalized probability mapping**: Add a brief analysis or reference supporting the dictionary V used for mapping verbalized probabilities to numerical values, and discuss whether this introduces systematic bias.

3. **Discuss limitations of independence assumption more explicitly**: Add a sentence in Section 3.2 or Section 5 acknowledging that correlated latent factors may produce unrealistic joint states, and reference Table 2 as evidence of robustness.

4. **Report normalized utility in the main text**: Include at least one key normalized utility result alongside accuracy to provide a fuller picture of decision quality.

## Score and Decision

Compared to calibration anchors:
- Papers scoring ~8 (e.g., BIRD 8,8,8; inference-time scaling 8,8,6,8) had more comprehensive experiments, clearer problem formulation, and claims that were fully supported by the evidence. They addressed edge cases, had stronger baselines, or made more fundamental contributions.
- Papers scoring ~5-6 had good ideas with reasonable validation but some limitations in scope, dataset size, or comparison strength.
- Papers scoring ≤3 had fundamental methodological issues or unsupported claims.

This paper has a practical, well-motivated framework with systematic evaluation showing clear improvements. The main limitations are constrained scope of the o1 comparison, independence assumption in state forecasting, and small dataset size. These are real but do not invalidate the core contribution. The paper is comparable to the 6-range anchors (good idea, reasonable validation, some limitations) but slightly below the 8-range anchors due to the constrained baseline evaluation and smaller experimental scale.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>