=== CALIBRATION EXAMPLE 16 ===

# Final Consolidated Review
## Summary
This paper introduces Preference-Based Reward Repair (PBRR), a framework for correcting a misspecified proxy reward function by learning an additive, transition-dependent correction term from human preferences. PBRR uses a targeted exploration strategy that compares trajectories from the proxy-optimized policy to a reference policy, and a novel preference-learning objective that regularizes updates to preserve correctly ranked transitions. The authors prove regret bounds for a tabular variant and demonstrate empirical improvements over baselines on several reward-hacking benchmarks.

## Strengths
- **Empirical effectiveness and data efficiency**: PBRR consistently outperforms strong baselines (e.g., Online-RLHF, RRM) across four challenging, high-dimensional reward-hacking benchmarks (Pandemic Mitigation, Glucose Monitoring, Traffic Control, AI Safety Gridworld), achieving higher performance with fewer preferences (Figure 2). The gains are shown in both jump-start and final performance.
- **Ablations validate design choices**: Through controlled experiments, the paper shows that both components of PBRR—the exploration strategy using a reference policy and the novel loss function (Eq. 3)—are necessary for its success. Removing either leads to worse performance or instability (Figure 3, Appendix G.4).
- **Robustness to reference policy quality**: PBRR works effectively even with a randomly initialized reference policy, broadening its applicability in settings where a high-performing reference is unavailable (Appendix G.8).

## Weaknesses
### Major:
- **Substantial theory–empirical gap**: The theoretical analysis (Section 5) proves regret bounds for a variant of PBRR with optimistic exploration (C₁ > 0) in tabular, linear-reward MDPs with known/unknown dynamics. However, all experiments are conducted in high-dimensional, non-linear settings with C₁ = 0, and the paper acknowledges the theoretical setup is intractable there (Section 6). An alternative analysis (Appendix K) relies on strong, unrealistic assumptions (noiseless, regret-based preferences, infinite data). This disconnect means the theoretical results do not substantiate the empirical method or explain its success, weakening the claimed theoretical grounding.
- **Lack of validation with real human preferences**: All preferences are simulated via a Boltzmann model using the ground-truth reward function. While this is standard for benchmarking, it does not capture the noise, biases, or cognitive limitations of real human feedback. Since the method is explicitly designed for human feedback, this omission limits confidence in its practical deployment (Appendix A acknowledges this but does not address it empirically).

### Minor:
- **Reliance on proxy reward optimism assumption**: The loss function (Eq. 3) is motivated by assuming the proxy reward is optimistic (overestimates true reward). Although the paper shows robustness when this assumption is violated (e.g., in Glucose Monitoring and Appendix G.6) and decays regularization weights heuristically, the method’s performance may degrade for systematically pessimistic proxies without careful tuning.
- **Insufficient analysis of statistical significance and sensitivity**: Most results are averaged over only 3 seeds, which is minimal for high-variance RL experiments. While Appendix G.9 reports 10 seeds for one environment, this should be extended to all key comparisons. Additionally, no sensitivity analysis is provided for hyperparameters like λ₁, λ₂ and their decay schedules, leaving robustness unclear.
- **Limited exploration of reference policy dependence**: Although PBRR works with a random reference policy, the theoretical guarantee (with C₁ = 0) only ensures performance no worse than the reference policy (Appendix K). The paper does not thoroughly analyze failure modes or sensitivity when the reference policy is adversarial or has poor coverage, which could limit applicability in some domains.

### Trivial:
- **Preference model discrepancy**: The theoretical analysis uses regret-based preferences, while experiments use sum-of-rewards labeling for tractability. This inconsistency is discussed and justified in Appendix A, so it does not harm the core empirical claims.

## Nice-to-Haves
- **Visualizations of learned corrections**: For interpretable environments like the AI Safety Gridworld, heatmaps of the correction term \(g(s,a,s')\) could illustrate which transitions are repaired, enhancing understanding.
- **Extended comparison to concurrent work**: While the RRM baseline is adapted from Cao et al. (2025), direct comparison on their robotic manipulation tasks would strengthen claims of broader applicability.
- **Integration with segment-level preferences**: Adapting PBRR to use trajectory segments (common in RLHF) could mitigate credit assignment issues noted in Appendix A and potentially improve data efficiency.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism about theoretical assumptions being too restrictive**: The paper explicitly states the limitations of its theoretical analysis (e.g., linear rewards, tabular settings) and does not claim they directly apply to the empirical settings. Removing as this is acknowledged and does not constitute a factual error.
- **Request for missing related works or comparisons**: Suggestions to compare with unspecified methods are omitted per the rule not to mention missing related works without external sources.
- **Nitpicks on reproducibility**: Details like hyperparameters are provided in Appendix E, and large artifacts (e.g., full training logs) are impractical to include. Such points are removed as they do not reflect substantive flaws.
- **Generic strengths**: Phrases like “the paper is well-written” or “the topic is important” are removed; strengths must be specific to this paper’s contributions.

## Suggestions
- **Conduct a small-scale user study**: Collect real human preferences on one benchmark environment to validate PBRR under realistic feedback noise and biases, addressing the major weakness.
- **Increase statistical rigor**: Run key experiments with at least 5-10 seeds and report confidence intervals to strengthen empirical claims.
- **Perform hyperparameter sensitivity analysis**: Systematically vary λ₁, λ₂ and decay schedules to provide guidelines for tuning and demonstrate robustness.

## Evaluation (using language instead of scores)
- **Novelty**: The core idea of repairing proxy rewards via an additive correction learned from preferences is novel and distinct from learning from scratch or constrained optimization. The integration of a tailored loss and exploration strategy advances the field.
- **Technical soundness**: The method is empirically sound with comprehensive experiments, but the theoretical analysis is disconnected from the empirical settings, reducing its technical cohesion.
- **Empirical support**: Strong and consistent results across multiple benchmarks with ablations, though limited by simulated preferences and modest seed counts.
- **Significance**: Addresses a practical problem in RL alignment (reward hacking) with potential impact in real-world domains like healthcare and autonomous systems.
- **Clarity**: The paper is well-structured, with clear explanations of the method, experiments, and limitations. Figures and appendices support understanding.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 8.0, 2.0]
Average score: 5.0
Binary outcome: Reject
