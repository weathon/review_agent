## Summary

This paper introduces the Implicit Bayesian Markov Decision Process (IBMDP), a framework for sequential assay planning in drug discovery when no environment simulator is available. IBMDP constructs a nonparametric transition model by sampling historical compound outcomes weighted by similarity to the current candidate, enabling Monte Carlo Tree Search planning with Bayesian belief updates. The approach is evaluated on a CNS drug discovery task (N=220 compounds) showing resource savings, and on a synthetic benchmark where the optimal policy is computable.

## Strengths

- **Addresses a genuine practical problem with clear motivation:** Sequential assay planning without simulators is a real constraint in drug discovery. The formulation correctly identifies that traditional RL requires (s, a, s') tuples or simulators, which are unavailable in this setting, and proposes a principled alternative using historical data.

- **Theoretical grounding via POMDP equivalence:** Appendix A formally derives the similarity-weighted belief updates as Bayesian posterior updates over latent historical prototypes (Equation 10, Theorem D.6). This provides a rigorous justification for the sampling mechanism rather than treating it as a heuristic.

- **Two-part experimental design:** Combining a real-world case study with a synthetic benchmark where the optimal policy is computable allows assessment of both practical utility and decision quality. The synthetic benchmark uses VI-Theo with analytically derived conditional variance as a ground-truth baseline.

- **Ensemble approach improves robustness:** Table 2 shows IBMDP Top-2 covers the optimal action in 66% of trials versus 36% for deterministic VI-Sim, demonstrating that stochastic ensemble planning explores near-equivalent high-value actions that deterministic optimization misses.

## Weaknesses

- **Real-world evaluation limited to 4 hand-picked compounds:** Table 1 presents resource savings for only four "representative" scenarios, with no population-level evaluation across the 220-compound dataset. Claims of "up to 92% reduction" rest on these cherry-picked cases. Leave-one-out cross-validation or aggregate metrics across all compounds would be needed to establish generalizability.

- **Baseline comparison appears to use a strawman:** The paper compares to a "traditional approach" that runs all assays ($5,200). However, the described rule-based heuristic (Section 5.1) provides conditional stopping rules based on QSAR predictions. If the heuristic says "promising" or "non-promising," one would not necessarily run all assays. The actual comparison should be IBMDP vs. the heuristic as defined, not vs. running every assay.

- **No decision correctness metric reported:** The paper optimizes uncertainty reduction (H(s) ≤ ε) and reports resource savings, but never evaluates whether IBMDP's final Go/No-Go recommendations are *correct*. In drug discovery, the cost of false positives (pursuing failed compounds) and false negatives (discarding viable ones) is asymmetric and critical. Resource efficiency is meaningless if the resulting decisions are wrong.

- **Synthetic benchmark evaluates only first action:** Appendix D.7 states "compute VI-Theo's optimal first action at the initial state" for each trial. This is an incomplete evaluation for a sequential planning method. First-action alignment (47% match rate) tells us little about multi-step policy quality.

- **47% optimal policy match is mediocre:** Even accepting the single-step evaluation, matching the optimal first action in 47% of trials is underwhelming for a method claiming principled Bayesian planning. The Top-2 coverage (66%) is better, but the 47% Top-1 rate raises questions about practical reliability for high-stakes decisions.

- **Defensive stance on baselines undermines empirical case:** Appendix C spends several pages arguing that comparisons with GP-based methods, Bayesian optimization, and active learning are "fundamentally unfair." This claim is overstated—multi-fidelity Bayesian optimization and sequential experimental design methods can be adapted to historical data settings. The absence of any adapted baseline (even a simple GP acquisition function) leaves the empirical case weaker than it could be.

- **Hyperparameter tuning on evaluation data:** Appendix B.1 states λ_w was tuned in [0.5, 2.0] on the CNS dataset. If the same 220 compounds used for evaluation also informed hyperparameter selection, this introduces overfitting risk. No sensitivity analysis is provided.

## Nice-to-Haves

- **Evaluate decision accuracy, not just resource savings:** Report the fraction of compounds where IBMDP's final recommendation (proceed vs. terminate) matches the true outcome based on held-out data.

- **Add a GP-based sequential acquisition baseline:** Even if imperfect, a simple GP with uncertainty sampling over assay selection would demonstrate whether the case-based sampling provides advantages over standard surrogate models.

- **Report performance as |D| varies:** Test how IBMDP degrades when historical data is limited or distributionally shifted relative to the candidate compound.

## Removed Points

These points are flagged to be removed, please take a look but treat them with caution:

- **Citation formatting complaint:** The harsh critic noted a "broken citation (?)" in Appendix E. This appears to be an author oversight during anonymization for double-blind review, not a substantive flaw.

- **Claim that POMDP equivalence is "not novel":** While the mathematical connection between case-based reasoning and Bayesian updating is established in prior work, the paper's specific contribution is applying this to sequential assay planning with ensemble MCTS. The criticism that "this is just Nadaraya-Watson" dismisses the systems-level integration.

- **Demand for multi-fidelity BO comparison without acknowledging adaptation requirements:** The positive reviewer notes that while Appendix C is defensive, adapting multi-fidelity BO to this setting requires non-trivial modifications (batched actions, stopping rules, state constraints). A fair comparison requires implementing those adaptations—criticizing their absence without acknowledging this complexity is scope creep.

## Novel Insights

The ensemble MCTS approach reveals an interesting trade-off: deterministic value iteration with similarity-based variance estimation (VI-Sim) achieves 36% optimal alignment, while stochastic ensemble MCTS achieves 47%. This ~11 percentage point gain comes not from a better model (both use identical transition estimates) but from the stochastic planner's ability to explore and identify near-equivalent actions. The Top-2 coverage (66% vs. 36%) suggests that in assay selection problems, multiple action sequences often yield similar information gains—a structure that stochastic search exploits but deterministic optimization misses. This observation could inform future work on action equivalence classes in sequential experimental design.

## Suggestions

1. **Run leave-one-out cross-validation on the CNS dataset:** Report aggregate resource savings and decision correctness across all 220 compounds (holding each out from the historical database during planning for it).

2. **Implement the rule-based heuristic fairly:** If the heuristic says "promising" (PgP < 2 AND BCRP < 2), the baseline cost should not be $5,200—compare against what the heuristic would actually recommend, not against running every assay.

3. **Evaluate full policy sequences in the synthetic benchmark:** Extend Table 6 from first-action match to full-trajectory alignment or cumulative reward comparison.

4. **Add hyperparameter sensitivity analysis:** Show how performance varies with λ_w, ensemble size N_e, and thresholds ε, τ. This is essential for practitioners to understand robustness.

5. **Report decision accuracy:** For each compound in the real-world experiment, report whether IBMDP's ultimate recommendation (above/below target threshold) matched the true outcome.