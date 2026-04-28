## Summary
This paper proposes FairDual, a dual optimization algorithm for Max-Min Fairness (MMF) in recommendation systems. The authors identify a "Jensen gap" caused by mini-batch sampling in MMF-constrained objectives and provide theoretical bounds on this gap. The method reformulates the fairness constraint using dual variables (shadow prices) and updates them via mirror gradient descent. Experiments across three datasets and six backbone models (including LLM-based recommenders) show FairDual outperforms fairness baselines on both accuracy and fairness metrics.

## Strengths
- **Comprehensive empirical evaluation**: The method is evaluated across three datasets (MIND, Amazon-Book, Amazon-Electronic) and six backbone models (NRMS, RecFormer, BigRec, plus traditional models in appendix), with consistent improvements over six fairness baselines including DRO, S-DRO, and Maxmin Sample (Table 1-2).
- **Theoretical convergence bound**: Theorem 4 provides a bound on the Jensen gap showing sub-linear convergence rate O(B^(-1/2)) with respect to batch size, distinguishing from heuristic re-weighting methods lacking guarantees.
- **Interpretable dynamic weighting**: The dual variable formulation yields interpretable group weights (shadow prices) that dynamically adjust during training, visualized in Figure 3(c) showing increased weights for tail groups (Sports) and decreased weights for head groups (News).

## Weaknesses

### Fatal
None

### Major
- **Theory-practice gap in Top-K handling**: The optimization objective (Equation 1) defines the MMF constraint over the top-K ranking list L_K(u), which is non-differentiable and discontinuous with respect to model parameters. However, the implementation approximates ranking scores by sampling Q items (Section 5.2.2, Line 11-12: "we randomly sample Q items to approximate the ranking scores across all items"). The theoretical analysis (Theorems 1-4) treats the loss as smooth and additive, not accounting for the bias introduced by this sampling approximation or the non-differentiability of the Top-K selection. This creates a disconnect between the theoretical guarantees and the actual algorithm implemented.

- **Unclear novelty over standard Lagrangian DRO**: FairDual reformulates the constrained MMF problem using dual variables and gradient descent updates (Algorithm 1, Equations 7-9), which is mathematically equivalent to standard Lagrangian relaxation for Distributionally Robust Optimization—a well-established technique for worst-group fairness (e.g., Hashimoto et al., 2018; Cotter et al., 2019). The paper claims FairDual uniquely "bridges the Jensen gap," but standard Lagrangian methods already handle non-linear constraints via dual updates. Without a direct comparison to a standard Lagrangian DRO baseline (without FairDual's specific momentum/freezing tricks), it remains unclear whether gains come from the dual formulation itself or from engineering choices.

### Minor
- **Misalignment between optimization objective and evaluation metric**: The optimization objective (Equation 1) targets Max-Min Fairness, which strictly maximizes the utility of the single worst-off group. However, the evaluation metric MMF@K (Section 6.1) "quantifies the aggregated ranking score of the 20% worst-off groups." A method could improve the average of the bottom 20% without improving the single worst group, potentially failing the actual Max-Min objective while appearing successful on the evaluation metric.

- **Jensen gap validation relies on simulation, not real training**: The empirical measurement of the Jensen gap (Figure 3a) uses a simulation setting with "assumption of knowing every user-item true preference score" (Section 4.2). The paper does not provide direct measurement of the gap during actual RS training on real datasets, nor correlation between simulated gap reduction and observed performance gains. It remains unclear if improvements stem from bridging the theoretical gap or from better hyperparameter tuning.

### Trivial
- **Small training sample for BigRec backbone**: Section 6.1 notes "BigRec only utilizes 1024 samples to train due to large computational cost," which is extremely small for an LLM-based recommender and limits the validity of the "large-scale" claim.

## Nice-to-Haves
- Add convergence trajectory plots showing training loss and fairness constraint violation over epochs for FairDual vs. DRO to visually demonstrate convergence to better feasible points.
- Include ablation on the frozen embedding mechanism (β parameter) to isolate whether performance depends on this heuristic or the theoretical dual update.
- Report the utility of the single worst-off group (true Max-Min metric) in addition to the bottom-20% average to verify the Max-Min claim.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point 1 (Top-K Non-Differentiability)**: KEPT as Major weakness—verified against paper. Equation 1 does include L_K(u) and the implementation does use sampling approximation not covered by theory.

- **Harsh Critic Point 2 (Lagrangian DRO Novelty)**: KEPT as Major weakness—verified. The dual formulation with shadow prices is standard Lagrangian relaxation; paper doesn't clearly distinguish from DRO baselines.

- **Harsh Critic Point 3 (Objective-Metric Misalignment)**: KEPT as Minor weakness—verified. Section 6.1 confirms MMF@K measures bottom 20%, not single worst group.

- **Harsh Critic Point 4 (Jensen Gap Decoupled)**: KEPT as Minor weakness—verified. Section 4.2 confirms simulation uses known true preferences.

- **Harsh Critic Abstract criticism ("sample independence" terminology)**: REMOVED—this is a minor terminological nitpick. The paper's core point about non-linearity causing gradient bias is valid even if phrasing is imprecise.

- **Harsh Critic Section 3 criticism (hybrid formulation)**: REMOVED—partially addressed. The paper does use sampling relaxation (Section 5.2.2), though the theory doesn't account for it (covered in Major weakness).

- **Harsh Critic Section 4 criticism (Jensen gap framing)**: REMOVED—framing non-linear SGD bias as "Jensen gap" is a valid conceptual contribution, even if the phenomenon is general.

- **Harsh Critic Section 5 criticism (frozen embeddings)**: MOVED to Nice-to-Have—this is an ablation request, not a fundamental flaw.

- **Strength Finder "Theoretical quantification of optimization error"**: KEPT but weakened—the quantification exists but has theory-practice gaps.

- **Strength Finder "Provable convergence bound"**: KEPT but weakened—the bound exists but assumes conditions not met in implementation.

- **Strength Finder "Consistent empirical superiority"**: KEPT—supported by Tables 1-2.

## Novel Insights
The paper's core insight—that mini-batch sampling in Max-Min Fairness optimization introduces a Jensen gap due to non-linear loss structure—is conceptually sound and aligns with known issues in constrained optimization. However, this insight is not unique to MMF; similar gaps arise in any non-linear fairness objective (Gini, power-mean welfare) optimized via SGD. The paper's contribution lies in providing explicit bounds for the MMF case, though the practical significance remains unclear without direct measurement during real training.

## Suggestions
1. **Clarify the Top-K relaxation**: Explicitly describe how gradients flow through the Top-K constraint in practice. If sampling or soft Top-K is used, revise Theorems 1-4 to state assumptions about this relaxation and bound the additional approximation error.

2. **Add standard Lagrangian DRO baseline**: Include a baseline implementing standard Lagrangian relaxation for MMF (without FairDual's momentum/freezing) to isolate whether gains come from the dual formulation or specific engineering choices.

3. **Measure Jensen gap during real training**: Report gradient bias or loss discrepancy during actual training on MIND/Amazon datasets to demonstrate the gap exists and is reduced in the target setting.

4. **Report true Max-Min metric**: Include the utility of the single worst-off group (not just bottom-20% average) to verify the Max-Min optimization claim.

## Score and Decision

**Calibration anchors retrieved:**
- **mex3rvs2KX (6.50, Accept)**: RaCO-DP for rate-constrained fairness with dual updates. Stronger theory-practice alignment—convergence analysis explicitly leverages linear structure of dual parameter, and experiments validate on deep learning tasks. FairDual has similar dual formulation but weaker theory-practice connection.
- **cuzWopwoZG (4.67, Accept)**: Differentiable Top-K for diversity optimization. Directly addresses non-differentiability issue that FairDual glosses over. Similar theory-practice gap concerns.
- **L8pyycR4wW (5.50, Accept)**: Pareto optimal fairness-accuracy trade-offs via convex optimization. Comparable theoretical contribution level with empirical validation.
- **vf6FxFj1OK (4.40, Reject)**: Max-min MORL with constraints. Rejected for limited scalability and missing baselines—similar novelty concerns as FairDual.
- **gl2nAqMlII (3.00, Reject)**: Fairness-aware reward optimization with Lagrangian formulation. Rejected for "lack of novelty" over prior Lagrangian fairness methods—directly analogous to FairDual's novelty concern.
- **HTgvEiBKVX (4.00, Reject)**: Fairness-aware recommender with extensive experiments. Rejected despite strong empirical results due to theory-experiment gaps (no online A/B testing).

**Positioning**: FairDual has stronger empirical scale than gl2nAqMlII (3.0) and vf6FxFj1OK (4.40), with consistent improvements across multiple datasets and backbones. However, it shares the theory-practice gap concerns of cuzWopwoZG (4.67) and the novelty-over-Lagrangian concerns of gl2nAqMlII (3.0). Compared to L8pyycR4wW (5.50), FairDual has comparable empirical strength but weaker theoretical novelty. The paper is positioned between the rejected Lagrangian fairness papers (3.0-4.4) and the accepted theory+experiment papers (5.5-6.5).

**Final assessment**: The paper makes a real empirical contribution with consistent improvements, but the theoretical claims are not fully supported by the implementation. This is a borderline paper—the empirical results are solid enough to warrant consideration, but the theory-practice gaps and unclear novelty over standard DRO prevent a strong accept.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>