=== CALIBRATION EXAMPLE 24 ===

# Final Consolidated Review
## Summary

The paper introduces Preference-Based Reward Repair (PBRR), an iterative framework that repairs a human-specified proxy reward function by learning an additive, transition-dependent correction term from trajectory-level preferences. PBRR combines (i) an exploration strategy that elicits preferences between trajectories from the proxy-optimized policy and a reference policy, and (ii) a novel preference-learning objective (Eq. 3) that regularizes corrections based on an assumed optimism of the proxy. The authors prove regret bounds in tabular/linear settings matching prior work (Pacchiano et al., 2023) up to constants, and demonstrate strong empirical performance across four reward-hacking benchmark environments, consistently outperforming baselines in data efficiency and stability.

## Strengths

- **Principled and practical problem formulation.** The idea of repairing a proxy reward rather than learning one from scratch is well-motivated and addresses a real gap between manual reward engineering and data-hungry RLHF. The additive correction formulation (Eq. 2) is simple yet effective, and the paper convincingly argues why it can be more data-efficient than learning ab initio—particularly when the correction term lies in a lower-dimensional space than the full reward.

- **Novel preference-learning objective (Eq. 3) with strong ablation support.** The three-term loss—preference cross-entropy plus L₊ (regularize corrections on correctly-ranked pairs) and L₋ (prioritize negative corrections on misranked pairs)—is the paper's most distinctive technical contribution. The ablation in Figure 3 clearly demonstrates that removing L₊ and L₋ leads to both lower performance and instability, validating the design. The qualitative analysis in Appendix H.3 provides a concrete mechanistic explanation for why the standard loss fails (it pushes the reward of reference-policy transitions to infinity, preventing exploration of other regions).

- **Robustness to reference policy quality.** Appendix G.8 (Figure 14) demonstrates that a randomly initialized reference policy suffices for PBRR to match or outperform baselines, significantly lowering the practical barrier compared to methods requiring expert demonstrations. This is a non-obvious and practically important finding.

- **Comprehensive experimental evaluation with strong baselines.** The paper evaluates on four diverse, high-dimensional domains (pandemic mitigation, glucose monitoring, traffic control, AI safety gridworld) and compares against well-chosen baselines including Online-RLHF, RRM (Cao et al., 2025), and multiple state-constrained variants. Additional experiments in appendices (G.3, G.4, G.5, G.6, F) test robustness across multiple axes.

- **Qualitative analysis explaining baseline failure modes.** Appendices H.1 and H.2 provide insightful explanations for why Online-RLHF conflates instrumental and terminal goals and why RRM fails to explore beyond the proxy-induced policy's region—these go beyond standard empirical comparisons and deepen understanding of when and why reward repair is needed.

## Weaknesses

### Major:

- **Significant disconnect between theory and experiments.** The regret bounds in Theorems 5.1 and 5.2 rely on Assumption 5.1 (linearity of trajectory returns in feature embeddings), but Section 6 explicitly states: "Our settings largely involve high-dimensional state spaces where the ground-truth reward function is not linear." More critically, the theoretical analysis depends on the undominated policy set construction (Π_t) and the selection of uncertainty-maximizing policy pairs (Lines 6–11 of Algorithm 1), yet the experiments set C₁ = 0, which entirely disables this mechanism. The paper states this is because "defining undominated policy sets for complex, non-linear reward functions learned from preferences is intractable." This means the algorithm that is theoretically analyzed is not the algorithm that is empirically evaluated. The current wording ("Theorems 5.1 and 5.2 suggest that it may often be possible…") is too suggestive given that the key assumptions are explicitly violated. The paper should clearly acknowledge that the theory does not provide guarantees for the practical variant of PBRR, and ideally discuss what aspects of the theory might transfer to the non-linear setting.

- **Preference model inconsistency between theory and experiments.** There are two separate but related inconsistencies: (a) The theoretical analysis in Appendix K assumes noiseless regret-based preferences (Assumption K.2), while experiments use Boltzmann-sampled preferences based on the sum of rewards (Appendix E.5). The paper acknowledges this in Appendix K.1, noting that "regret is intractable to compute in our empirical environments," but the gap between what is assumed and what is implemented is significant. (b) Section 2 states that preferences are elicited over "full trajectories," and Appendix A argues against segment-level preferences, yet Appendix E.5 reveals that for the Glucose environment (H=5760), trajectories are split into three segments and preferences are elicited over those segments. This directly contradicts the stated design principle and raises questions about whether the theoretical framework covers any of the actual experimental conditions.

### Minor:

- **The optimism assumption's role is incompletely understood.** The loss function (Eq. 3) is explicitly designed around the assumption that proxy rewards are aligned or overly optimistic. Yet Section 6.3 shows PBRR outperforms baselines in Glucose Monitoring where the proxy is pessimistic (penalizes optimal health outcomes due to financial cost), and Appendix G.6 shows a pessimistic proxy in the gridworld still works but requires more preferences. The λ decay mechanism (Appendix E.6) provides a practical escape hatch, but the paper lacks a clear characterization of when and why the method degrades under assumption violations. If L₋ functions primarily as a general regularizer against large updates (rather than specifically enforcing pessimism-motivated corrections), the theoretical motivation for its specific form is weakened.

- **The guarantee with a weak reference policy is loose.** Theorem K.1 guarantees J_r(π*_r̂) ≥ J_r(π_ref) as t → ∞. When π_ref is random (Appendix G.8), this guarantee is trivially satisfied by nearly any reasonable policy. The fact that PBRR substantially exceeds this bound suggests the preference signal does most of the work, which somewhat undermines the framing that the proxy repair structure is essential (vs. simply providing a good prior for RLHF). The paper would benefit from explicitly quantifying how much of PBRR's advantage comes from the proxy repair structure vs. the proxy-as-prior effect.

### Trivial:

- The introduction's claim that manual reward correction is "slow, ad hoc, and depends on RL expertise" (line 39) is plausible but uncited. This is a minor motivational point and does not affect the technical contribution.

## Nice-to-Haves

- **More seeds and statistical rigor.** The main results use 3 seeds. Appendix G.9 shows 10-seed results for one environment with 95% CIs, demonstrating statistically significant differences. Extending this to all environments would strengthen the stability claims.

- **Analysis of the learned correction term g.** The paper claims the correction term lies in a "lower dimensional space" but provides no evidence. Visualizing which transitions receive corrections, their magnitude, and sparsity would validate the core mechanism and provide insight into what PBRR is actually learning.

- **Human preference evaluation.** All preferences are simulated from ground-truth rewards. Testing with real human labelers on at least one environment would validate the practical utility of the framework for reducing human labeling effort.

- **Compute cost comparison.** PBRR requires retraining a policy at every iteration. Reporting wall-clock time or total environment steps (not just preference counts) would clarify whether preference efficiency comes at a computational cost.

- **Systematic variation of proxy misspecification severity.** Each environment has one level of misspecification. Varying how suboptimal the initial proxy is would clarify the method's operating regime.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Baseline tuning uses privileged access to ground-truth rewards.** (Appendix E.3) This asymmetry *favors the baselines*, not the authors' method—PBRR does not use privileged information. Per the review rules, this is not a valid weakness since it makes the baselines stronger, and PBRR still outperforms them.

- **Weakness: Missing comparisons with other relevant baselines.** The paper compares against 5+ baselines including concurrent work (Cao et al., 2025) and multiple ablations. Generic claims about "missing baselines" without specifying concrete, existing methods that should have been included are not actionable. Per the rules, I cannot confirm the existence of unspecified alternative methods.

- **Weakness: Reproducibility concerns about hyperparameters.** The paper provides detailed hyperparameter tables (Table 2, Table 3, Table 4) and describes the tuning procedure. Per the rules, nitpicks about reproducibility of trivial implementation details are removed.

- **Weakness: Formatting and notation issues.** Per the rules, pure formatting/style nitpicks are removed.

- **Weakness: 3 seeds is insufficient.** While more seeds would be better, 3 seeds is the standard in RL research at ICLR. This is moved to nice-to-have rather than a core weakness.

## Novel Insights

The most interesting observation emerging from the reviews is that PBRR's success may rely less on its theoretical motivation (optimism-corrected reward repair) and more on the *implicit regularization* provided by anchoring learning to an existing reward function. The fact that PBRR works with random reference policies and pessimistic proxies—both violating core assumptions—suggests the primary benefit is the inductive bias of "repair, don't rebuild," which constrains the reward function search space in a way that standard RLHF does not. The ablation showing that the standard loss (Eq. 1) applied to the repair framework causes the reward to assign unboundedly high values to reference-policy transitions (Appendix H.3) is particularly revealing: it suggests that without L₊ and L₋, the repair framework actually *amplifies* the credit assignment problem rather than solving it. This implies that the objective terms are doing something closer to preventing reward unboundedness than specifically enforcing optimism-based corrections—a distinction that matters for understanding when and why the method generalizes.

## Suggestions

- **Explicitly scope the theoretical contribution.** Add a clear statement in Section 5 or Section 7 acknowledging that the regret bounds apply to the linear/tabular regime with the full undominated-set exploration mechanism (C₁ > 0), and that the practical algorithm (C₁ = 0, neural network function approximation) operates outside these guarantees. Discuss which aspects of the theory might plausibly transfer and which do not.

- **Reconcile the preference labeling inconsistency.** Either modify the Glucose experiment to use full trajectories (if memory permits) or explicitly acknowledge in the main text that the Glucose environment uses segment-level preferences and discuss the implications for the claims in Appendix A.

- **Add a simple analysis of the correction term.** Even a histogram of correction magnitudes or a heat map of which state-action pairs receive the largest corrections in the gridworld environment would provide evidence for or against the "lower-dimensional correction" claim and help readers understand what the method is learning.

- **Clarify the role of the optimism assumption.** If the method works robustly when the assumption is violated (Glucose, Appendix G.6), provide a clearer explanation of the mechanism. Is L₋ acting as a general regularizer that prevents the correction term from growing unboundedly, rather than specifically encoding an optimism prior? If so, state this explicitly and adjust the framing accordingly.

- **Report total environment steps or wall-clock time alongside preference counts.** This would clarify whether PBRR's preference efficiency comes with additional computational overhead from iterative policy retraining, which matters for practical adoption.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 8.0, 2.0]
Average score: 5.0
Binary outcome: Reject
