=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary

The paper proposes Goal-Conditioned Reinforced Supervised Learning (GCReinSL), a method to endow Outcome-Conditioned Behavioral Cloning (OCBC) methods with trajectory stitching capability. The key idea is to incorporate Q-function maximization into supervised learning via expectile regression, using a VAE to estimate Q-values from state occupancy probabilities. The authors prove that Q-conditioned maximization is equivalent to goal data augmentation and demonstrate empirical improvements on stitching-focused benchmarks and D4RL Antmaze tasks.

## Strengths

- **Clear problem motivation with concrete illustration**: The maze example in Section 4.1 effectively illustrates why OCBC methods fail at stitching—they cannot switch from a failing trajectory to a successful one at intersection states because the Q-function remains at zero. The paper correctly identifies this as a fundamental limitation of SL-based offline RL methods.
- **Strong empirical results on D4RL Antmaze-v2**: Table 1 shows substantial improvements over prior sequence modeling approaches (DT, EDT, Reinformer) on medium-play (49.0 vs. 13.2 for Reinformer), medium-diverse (51.7 vs. 10.6), large-play (28.2 vs. 0.4), and large-diverse (30.2 vs. 0.4). These gains are meaningful and demonstrate the method's practical utility.
- **Theoretical connection to goal data augmentation**: Corollary 1 provides a formal grounding for why Q-conditioned maximization enables stitching, connecting the proposed method to a well-understood augmentation paradigm.

## Weaknesses

- **Theorem 4.1's definition of Q_max appears problematic for the claimed mechanism**: The theorem states that as m→1, the predicted Q converges to Q_max = max_{s,a,g} Q(s,a,g), defined as a global scalar maximum. If Q_max is truly a single constant across all state-action-goal tuples, then the Q-prediction carries no state-dependent information, which would undermine the stitching mechanism that relies on predicting different Q-values at different states. The theorem needs clarification—whether Q_max is per-context (state, goal) or truly global—and how state-dependent Q-values emerge from the formulation. This is central to the paper's contribution.
- **Factual inaccuracies in figure captions and text**: Figure 4's caption states "In all cases, GCReinSL achieves the highest success rate," but Pointmaze-Medium RvS shows TGDA (0.60) outperforming GCReinSL (0.50). The text in Section 5.3 states "GCReinSL is inferior to advanced TGDA method" on Antmaze-Medium, but Figure 5 shows GCReinSL (0.28) slightly ahead of TGDA (0.25). These inconsistencies undermine confidence in the reported results.
- **Unclear distinction from prior work**: The paper states it is "inspired by max-return sequence modeling (Zhuang et al., 2024)" but does not clearly articulate what GCReinSL contributes beyond adapting Reinformer to goal-conditioned settings. Both use expectile regression on value-like quantities. The key architectural differences (VAE-based Q-estimation vs. learned value function, goal-conditioned formulation) should be explicitly compared.
- **Training procedure for VAE is underspecified**: Section 4.2 describes the VAE architecture but does not specify whether it is trained jointly with the policy or separately. If jointly trained, the Q-targets are moving targets during optimization, potentially causing instability. If separately trained, the training order and procedure should be explicitly stated. This affects reproducibility.
- **Inconsistent hyperparameter requirements across datasets**: The ablation shows L=500 for Ghugare et al. datasets but L=5 for D4RL Antmaze-v2—a 100× difference in importance samples for probability estimation. No explanation is provided for why such different settings are needed, raising concerns about hyperparameter fragility and computational overhead on certain datasets.

## Nice-to-Haves

- **Visualization of actual stitching behavior**: A figure showing agent trajectories with identified switch points would provide direct evidence that the method is performing stitching rather than memorization.
- **Computational overhead analysis**: The two-step inference pipeline (predict Q, then action) and VAE likelihood estimation (with L importance samples) introduce overhead. Wall-clock inference times would help assess practical viability.
- **VAE Q-estimate validation**: Correlation plots between VAE-estimated Q-values and empirical returns would verify that the probability estimation is actually capturing meaningful Q-information.

## Removed Points

These points are flagged to be removed, treat them with caution:
- **Missing comparisons to HIQL, Contrastive RL, GCIVL**: These methods should be included as baselines. However, the instruction explicitly states not to mention missing related works since external sources cannot be verified.
- **Notation inconsistency (GCReinSL vs GCREinSL)**: This is a minor formatting nitpick that does not affect the paper's technical contribution.
- **Insufficient standard deviations in Figure 4**: While error bars would be preferable, 5 seeds is standard practice. Table 1 includes standard deviations for GCReinSL. The absence in Figure 4 is suboptimal but not a critical flaw.
- **VAE approximation errors not bounded**: While true that ELBO is a lower bound, this is inherent to VAE methodology. The criticism is generic to all VAE-based approaches.

## Novel Insights

The paper's core insight—that Q-conditioned maximization via expectile regression can endow SL methods with stitching capability equivalent to implicit goal data augmentation—is valuable. The observation that OCBC methods fail because they cannot "see" the better Q-value at trajectory intersections (illustrated in Figure 1) provides a crisp conceptual framing. However, the mechanism by which the VAE learns that Q=1 is achievable from state s_t via trajectory τ_2—when this stitched trajectory never appears in the training data—remains insufficiently explained. The theoretical framework claims this emerges from maximizing expected Q under the goal distribution, but the bootstrapping process deserves more rigorous treatment.

## Suggestions

- **Clarify the Q_max definition in Theorem 4.1**: Either re-define Q_max as a per-context quantity Q_max(s,g) = max_a Q(s,a,g), or explain how state-dependent Q-predictions emerge if Q_max is truly global.
- **Fix the factual errors in captions and text**: Correct the claim that GCReinSL achieves the highest success rate in all cases, and reconcile the contradictory statements about performance vs. TGDA.
- **Specify the VAE training procedure explicitly**: State whether the VAE is trained jointly or separately, and provide the training algorithm.
- **Explain the L hyperparameter discrepancy**: Why does Ghugare et al. datasets require L=500 while D4RL only needs L=5? Is this related to dataset structure, dimensionality, or something else?
- **Add analysis of failure cases**: On large-play and large-diverse tasks, GCReinSL achieves ~28-30% while IQL achieves ~53%. Investigating why the gap persists on harder tasks would strengthen the paper.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 1.0, 6.0]
Average score: 3.8
Binary outcome: Reject
