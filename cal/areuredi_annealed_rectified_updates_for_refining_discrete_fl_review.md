=== CALIBRATION EXAMPLE 31 ===

# Final Consolidated Review
## Summary

AReUReDi extends Rectified Discrete Flows (ReDi) to multi-objective optimization for biological sequence design by integrating annealed Tchebycheff scalarization, locally balanced MCMC proposals, and Metropolis-Hastings updates. The framework provides theoretical guarantees of convergence to the Pareto front and demonstrates improved multi-property optimization (up to five therapeutic objectives) on peptide and SMILES sequence generation tasks compared to evolutionary and diffusion-based baselines.

## Strengths

- **Principled integration of ReDi with multi-objective MCMC guidance.** The paper provides a technically coherent framework that combines three mechanisms—annealed Tchebycheff scalarization, locally balanced proposals, and MH updates—each serving a clear role: the scalarization maps the multi-objective problem to a tractable single-objective reward, the locally balanced proposal blends the ReDi generative prior with reward guidance while maintaining reversibility, and MH ensures distributional invariance. This is a genuine algorithmic contribution beyond simply applying guidance to an existing model.

- **Theoretical proofs of invariance and Pareto convergence.** The appendix provides complete proofs that the transition kernel preserves the target distribution $\pi_{\eta,\omega}$ (detailed balance via balancing function symmetry) and that samples concentrate on Pareto-optimal states as $\eta \to \infty$ with interior weight vectors. The Pareto Point Representability theorem explicitly shows how weight vectors map to distinct Pareto points, which is crucial for coverage.

- **Empirical demonstration of multi-objective trade-off navigation across diverse targets.** The ablation in Tables 7–8 shows that removing any single objective guidance causes collapse in that property, confirming AReUReDi targets all objectives simultaneously. The method outperforms four classical MOO algorithms and PepTune+DPLM on eight diverse protein targets (Table 1–2), including structured targets, disordered targets, and targets with/without known binders. The matched wall-clock comparison (Table 11) shows AReUReDi's top-2 samples still achieve substantially better non-fouling, solubility, and half-life than PepTune's top-2 of 100.

## Weaknesses

### Major:

- **Monotonicity constraint used in all experiments invalidates the theoretical convergence guarantees.** Section 4 states: "we introduce a monotonicity constraint that accepts only token updates that increase the weighted sum of the current objective scores. … this monotonicity constraint was involved in all the following experiments." This converts the MCMC sampler into a hill-climber that rejects all downhill moves, directly violating the detailed balance condition required by the Invariance Theorem (Appendix A.2). Consequently, the chain is no longer ergodic with respect to $\pi_{\eta,\omega}$, and the proven convergence to the Pareto front with full coverage does not apply to any empirical result in the paper. Table 6 shows the constraint dramatically improves performance, which suggests the pure MH sampler may be impractically slow—but this underscores the severity of the disconnect: the theory describes an algorithm that is not actually run. The paper must either (a) present results without the monotonicity constraint to validate the theory, or (b) clearly reframe the theoretical claims as describing an idealized limit that the heuristic implementation approximates, with empirical evidence about how closely it tracks.

- **No standard Pareto front quality metrics (Hypervolume, IGD, Spacing).** Tables 1–2 report average scores per objective across 100 generated sequences. In multi-objective optimization, averages across the front are often misleading: a method generating solutions at extreme trade-offs can produce the same averages as one generating balanced solutions. Without Hypervolume or similar metrics that measure both convergence to the front and diversity along it, the claim of "superior trade-off navigation" (Contribution 4) and "full coverage" cannot be empirically validated. This is a critical gap for a paper whose core contribution is Pareto optimality.

- **No Pareto front visualization.** The paper claims full Pareto front coverage theoretically, but no scatter plot or parallel coordinates plot shows the actual distribution of the 100 generated solutions across the objective space. Without this, it is impossible to assess whether AReUReDi truly explores diverse trade-offs or clusters around a single compromise point. Figure 1E shows mean trajectories over iterations but not the front shape.

### Minor:

- **Pareto optimality claims should be explicitly scoped to surrogate objectives.** The score models have modest validation performance: hemolysis F1 = 0.58, affinity Spearman = 0.64, half-life $R^2$ = 0.60 (Appendix E). The theoretical guarantees apply to the Pareto front of these predictors, not to biological reality. The paper mentions "in silico results" in the Discussion but does not clearly flag this limitation in the Abstract or Contribution 2, where "convergence to the Pareto front" is stated without qualification. Given Goodhart's Law, optimizing hard for imperfect surrogates risks generating sequences that exploit predictor blind spots.

- **No ablation of the Metropolis-Hastings acceptance step.** The paper claims MH preserves distributional invariance (and notes automatic acceptance with Barker's function), but never demonstrates what degrades when MH is removed. Since the monotonicity constraint already modifies the acceptance rule in practice, understanding the role of the formal MH step becomes even more important.

- **Mixed rectification ablation results.** Table 9 shows that for target 5AZ8, the unrectified PepReDi achieves higher affinity (6.4391 vs. 6.2792) than the rectified PepReDi[3]. Rectification appears to shift the trade-off (favoring half-life over affinity) rather than strictly dominating. The claim that rectification "enables AReUReDi to achieve stronger Pareto trade-offs" would be more convincing with Hypervolume analysis showing a strictly better frontier.

### Trivial:

- The annealing schedule parameters ($\eta_{min} = 1.0$, $\eta_{max} = 20.0$) are fixed across targets. No guidance is provided for selecting these on new tasks, though Table 10 shows sensitivity to the schedule choice.

## Nice-to-Haves

- Comparison with adapted continuous-space MOO methods (e.g., ParetoFlow with discrete projection) to directly assess the value of operating in discrete token space
- Sensitivity analysis injecting noise into score models to quantify how predictor errors propagate to final sequence quality
- Structural motif or sequence logo analysis to assess whether generated sequences learn meaningful biochemistry vs. exploit predictor artifacts
- Ensemble uncertainty quantification for the property predictors, especially for the half-life model (105 training samples)

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Parser artifact complaints** (formatting/style nitpick, not a substantive issue)
- **Self-created benchmarks as a weakness** — The paper correctly notes no public MOO benchmarks for biological sequences exist; creating new ones is standard practice, and comparisons against multiple established baselines mitigate fairness concerns
- **Demand for wet-lab validation** — Outside the scope of a computational methods paper; the authors explicitly frame results as in silico
- **SMILES "4 vs. 5 objectives" inconsistency** — The abstract says "up to five," which is accurate since wild-type peptides use five and SMILES uses four
- **Missing hyperparameters / reproducibility complaints** — Annealing parameters, sampling steps, and model details are provided in Appendices F and C
- **Demand for comparison with continuous-space methods as a "fundamental" gap** — The paper argues domain mismatch precludes fair comparison, which is reasonable; adapting methods across representations changes too many variables for a clean comparison

## Novel Insights

The monotonicity constraint's dramatic impact (Table 6: half-life jumps from ~2h to 44–54h) suggests that the pure MH sampler with detailed balance may be impractically slow for the annealing schedules used, raising the question of whether the theoretical framework's assumptions about chain length are compatible with feasible computation. This points to a deeper design tension: the theoretical guarantees require ergodicity (which the monotonicity constraint destroys), but practical performance seems to require the constraint. A promising direction would be to analyze whether a softer constraint—e.g., accepting downhill moves with probability proportional to the reward decrease—could partially preserve ergodicity while maintaining most of the practical acceleration, or whether adaptive annealing schedules that start very cold could make the pure MH sampler tractable.

## Suggestions

- Run the core experiments (at least Table 2) both with and without the monotonicity constraint and report Hypervolume for both, so readers can assess the true cost of the constraint on Pareto front quality versus the claimed theoretical guarantees.
- Add a 2D scatter plot of two objectives (e.g., affinity vs. half-life) for AReUReDi and baselines, and compute Hypervolume relative to a reference point, to empirically validate the Pareto convergence and coverage claims.
- Explicitly qualify the theoretical guarantees in the Abstract and Contributions as applying to the surrogate objective landscape, not biological reality, given the known limitations of the score models.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
