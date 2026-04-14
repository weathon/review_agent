## Summary

CMBO (Cost-sensitive Multi-fidelity Bayesian Optimization) proposes a framework for HPO in which a user-defined utility function encodes the trade-off between BO cost (in epochs) and validation performance, rather than optimizing asymptotic performance alone. The method introduces (i) a utility-aware acquisition that maximizes expected utility improvement over a dynamically chosen future horizon Δt, (ii) an adaptive stopping criterion combining normalized regret with a probability-of-improvement (PI) threshold, and (iii) a PFN surrogate trained via a two-stage LC mixup strategy for transfer learning across task families. Extensive experiments on LCBench, TaskSet, PD1, and a real-world object-detection dataset demonstrate consistent improvements over multi-fidelity and transfer-BO baselines under various utility functions and cost penalties.

---

## Strengths

- **Genuinely novel acquisition formulation.** Eq. (2) extends expected improvement from performance to utility, with a joint optimisation over the future continuation length Δt per configuration. This is a non-trivial departure from prior freeze-thaw acquisitions (DyHPO, DPL, iFBO), which either use single-step greedy extensions or maximise performance at the last epoch or randomly chosen epochs. The shift from exploration to exploitation as cost dominates utility (Fig. 7b, showing Δt/T shrinking over BO steps) is an insightful and empirically verified consequence of the formulation.

- **Two-stage LC mixup for PFN transfer learning.** The proposal to apply a shared λ₁ across configurations within a dataset before mixing across configurations is a subtle but important detail: it preserves inter-configuration correlations while generating effectively unlimited synthetic training tasks. Fig. 6a directly shows that mixup reduces overfitting in the surrogate; Fig. 6b shows a downstream BO benefit. This is a principled and practical contribution beyond generic data augmentation.

- **Comprehensive and consistent empirical validation.** The paper evaluates on three standard HPO benchmarks (LCBench: 35 tabular datasets, TaskSet: 9 NLP tasks, PD1: 7 tasks with modern architectures) plus a real-world object-detection dataset with 30 tasks from heterogeneous architectures. Results are consistent across benchmarks, cost penalties, and utility function families (linear, quadratic, square-root, staircase), and ablations in Table 3 cleanly decompose contributions from the stopping criterion, acquisition function, and transfer learning.

- **Empirical demonstration that strong transfer priors can dominate multi-fidelity mechanics.** The observation that FSBO (a black-box transfer-BO that evaluates only at the last epoch) outperforms most multi-fidelity baselines (Fig. 4) is a sharp and practically important finding, underscoring that sample efficiency from transfer learning can outweigh the benefit of multi-fidelity decisions when the surrogate prior is weak.

---

## Weaknesses

### Fatal
None.

### Major

- **No comparison with cost-aware BO baselines.** The paper's central claim is that explicitly representing cost-performance utility improves over conventional BO. However, the baseline set contains only methods that ignore cost (multi-fidelity BO methods optimizing asymptotic performance) or that use a simpler fixed budget. There is no comparison with methods that incorporate cost into the acquisition function (e.g., EI-per-unit-cost, cost-cooled EI, BOCA, or multi-fidelity acquisitions that weight improvement by evaluation cost). Without this, it is impossible to determine whether CMBO's gains come from the principled utility formulation or simply from the superior surrogate and stopping logic. This is a critical omission given the paper's framing.

- **No comparison with simple heuristic stopping baselines.** Practitioners facing limited compute often just "run BO for K epochs." Without a comparison to "fixed shorter budget" or "stop after K steps with no improvement" baselines, it is unclear whether the complex utility + stopping machinery is necessary, or whether one would obtain similar cost-performance trade-offs by simply truncating a standard multi-fidelity BO at a shorter total budget.

- **Stopping criterion is heuristic and under-justified.** Eqs. (3)–(5) combine an approximate normalized-regret estimate (using hand-constructed Û_max and Û_min) with a PI-based threshold modulated by BetaCDF^γ. While the idea is intuitive, the paper provides no theoretical justification for this specific construction, no analysis of when the regret proxy is well-calibrated (it can be severely distorted if U_min is a loose lower bound), and no ablation demonstrating that β=e^{-1} transfers across utility families. The β ablation in Fig. 7d covers only PD1; its stability on LCBench and TaskSet is untested, and there is no principled guidance for setting β on a new problem.

- **Utility estimation from preferences is lightly validated.** The paper presents preference learning as a key contribution (Abstract, §3.1, §4.7), yet experiments almost entirely use hand-specified utility functions. The single "Estimated" condition in Table 2 constructs preferences synthetically by anchoring to iFBO's trajectory—not from actual human annotators. Fig. 2 shows one synthetic recovery example. Critical questions remain unanswered: (a) how many pairwise comparisons are needed in practice, (b) sensitivity to preference noise or misspecification of the utility family, (c) effect of utility estimation error on stopping and acquisition decisions. The "Estimated" condition in Table 4 is described as preference-based but no details of the elicitation protocol are provided.

### Minor

- **Table 3 presentation error.** Rows 3 and 4 of Table 3 have identical checkmarks (p_b=✓, Acq.=✓, T.=✓) but report dramatically different regrets (4.4 vs. 0.9 for α=2e-4). The intended distinction (presumably T.=✗ vs T.=✓ for the last row) is lost in the formatting, making the ablation hard to interpret. This should be corrected.

- **LC mixup over heterogeneous configuration spaces.** The second-stage mixup in §3.3 linearly interpolates configuration–curve pairs: (x'', l'') = λ₂(x_n, l'_n) + (1−λ₂)(x_{n'}, l'_{n'}). For benchmarks with categorical or log-scaled hyperparameters, linear interpolation of raw x-vectors does not correspond to any meaningful intermediate configuration. The paper does not address how x is represented or whether mixup is restricted to continuous embeddings. Since the surrogate quality drives most of CMBO's advantage, this assumption deserves explicit justification or qualification.

- **Algorithm 1, Line 4 notation issue.** Line 4 reads `n* ← arg max_{n ∈ C} A(n)`, but C is defined as a set of (x, t, y) triples (the history), not a set of configuration indices. At the first BO step, C is empty, so the argmax is over an empty set. The intended range is presumably n ∈ [N] or x_n ∈ X. This is likely a notation bug but affects reproducibility.

- **Uniform per-epoch cost assumption.** The method uses BO steps (epochs) as the cost axis. This assumes equal wall-clock cost per epoch across all configurations and tasks. Yet the object-detection dataset trains ResNet-50, HR-Net, and MobileNetv2—architectures with very different per-epoch costs. The paper acknowledges this implicitly via the Quick-Tune† modification (which removes cost-weighting for non-uniform wall-time), but does not discuss what happens to utility optimality when epoch cost is heterogeneous.

- **Computational overhead not reported.** CMBO's acquisition requires MC sampling of full future trajectories for all N configurations at each BO step, plus maximisation over Δt. The paper does not report surrogate training time, per-step inference cost, or comparison with the wall-clock overhead of baselines. For a method that claims cost-sensitivity, this omission is notable.

### Tiny

- Fig. 5 is explicitly labeled "cherry-picked examples." While the aggregate tables (Tables 1–4) are the primary evidence, the paper should at least note that per-task breakdowns appear in the appendix to make it clear that the cherry-picked examples are representative of aggregate trends, not chosen to hide failures.

- The paper says Û_min is approximated by decaying y^{worst}_1 across B steps. The rationale for this specific approximation and its sensitivity are not discussed, even briefly.

---

## Nice-to-Haves

- A controlled robustness study on utility misspecification: inject varying levels of noise into the utility function parameters and measure the degradation in stopping quality and final regret. This would directly validate the preference-learning pipeline's practical limits.

- Visualization of LC extrapolation quality at early BO stages (e.g., after 1, 3, 5 observations) to provide direct evidence that the transfer-learned surrogate is reliable when it matters most for stopping decisions.

- At least one experiment with heterogeneous per-epoch wall-clock costs, where b in the utility function is replaced by cumulative compute time, to demonstrate applicability to the motivating cloud/Slurm scenario.

- A per-task win/tie/loss breakdown or effect-size analysis beyond average rank, to assess whether CMBO's advantage is broad and consistent or driven by a subset of tasks.

- Sensitivity analysis for γ (the exponent in BetaCDF^γ) to complement the β ablation, since γ sets the baseline stopping threshold in the regret-only limit.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Zero standard deviations for deterministic methods (Harsh Critic).** The critic flags "exactly 0.0 standard deviation" as suspicious. The paper explains that FSBO and Quick-Tune† are deterministic methods and that task averaging eliminates all stochasticity for such methods. This is a legitimate and clearly explained design choice; no problem exists here.

- **"Dramatic improvement" wording in the conclusion (Harsh Critic).** This is a stylistic/tone concern with no substantive technical content. The empirical results do support strong improvements; the word choice is a matter of writing style and does not constitute a weakness.

- **U_prev not being the best-so-far utility creates potential for oscillation (Harsh Critic).** The paper explicitly discusses and defends this in §3.2: "the cost of BO that has previously been incurred is not reversible … U_prev can either increase or decrease during the BO, and we need to stop the BO when U_prev starts decreasing monotonically." The design choice is clearly motivated and the acquisition is specifically formulated to handle it. While a deeper theoretical analysis would be welcome, the absence of one does not invalidate the approach.

- **Optimization target max_{n,t} y_{n,t} unusual (Harsh Critic).** BO performance ỹ_b is the best validation seen up to step b. With α=0 this matches standard multi-fidelity HPO. With α>0, it is a natural consequence of optimizing a trajectory-based utility. The paper is clear about this definition; this is not a flaw.

- **Preference learning literature is under-covered in related work (Harsh Critic).** Appendix §A explicitly defers related work on cost-sensitive HPO and BO with user preference. Criticizing related work coverage without access to the appendix is unreliable.

- **Comparison with FSBO as unfair to CMBO (multi-fidelity vs. black-box) (Harsh Critic).** The comparison is intentionally informative: if a simpler black-box transfer-BO can match or exceed complex multi-fidelity methods, it shows the importance of priors. The "unfairness" here favors the baseline (FSBO), not CMBO, making the result a stronger point for CMBO when it wins and an honest admission when it doesn't. This should not be counted as a weakness.

---

## Novel Insights

The most insightful observation arising from this work—beyond the expected contributions—is that **the dominant source of performance improvement in modern multi-fidelity BO appears to be surrogate quality (transfer learning) rather than multi-fidelity decision-making per se**. The finding that FSBO, a black-box method that never exploits intermediate epoch information, outperforms all multi-fidelity baselines on LCBench and nearly so on PD1 (Fig. 4) is a sharp empirical result with important implications: years of algorithmic development in freeze-thaw BO and Hyperband variants may have been bottlenecked primarily by weak priors, not by suboptimal fidelity-selection policies. This motivates a rethinking of the standard narrative that multi-fidelity exploration is the key lever for HPO efficiency. The CMBO results then show that once a strong transfer prior is in place, the utility-aware acquisition and stopping criterion provide substantial additional gains—particularly for aggressive cost penalties—suggesting these two axes (prior quality and cost-sensitivity) are complementary and largely orthogonal.

---

## Suggestions

1. **Add cost-aware BO comparison.** Include at least one established cost-aware baseline (e.g., EI/cost or a cost-adjusted version of iFBO) on the same benchmarks. Even a simple acquisition that divides EI by expected epoch cost would clarify whether the utility framing provides structured gains beyond naive cost-weighting.

2. **Fix Table 3 row labeling.** Correct the duplicate row headers in Table 3 (rows 3 and 4 both showing p_b=✓, Acq.=✓, T.=✓) to reflect the intended ablation difference.

3. **Add a simple-budget baseline.** Compare CMBO against "run iFBO for K epochs, then stop" where K is chosen to match CMBO's average stopping budget. This isolates whether the utility-aware machinery adds value beyond reduced total budget.

4. **Provide computational overhead measurements.** Report per-step inference time for CMBO and representative baselines (iFBO, DPL) on a common hardware setting. This is essential for a cost-sensitive method.

5. **Clarify hyperparameter representation in LC mixup.** Specify how categorical or log-scaled hyperparameters are represented when computing the configuration mixup (x'', l'') = λ₂(x_n, l'_n) + ..., and whether the mixup applies to raw parameter vectors or continuous embeddings.

6. **Extend β ablation to LCBench and TaskSet.** The current ablation (Fig. 7d) only covers PD1. At minimum, a brief table showing robustness of the β=e^{-1} choice across all three benchmarks would substantially strengthen the stopping criterion's credibility.

---

## Evaluation on Key Axes

- **Originality:** High. The utility-aware acquisition with dynamic Δt selection and the combined regret+PI stopping criterion are genuinely novel formulations for multi-fidelity BO. The LC mixup strategy is a creative and practical adaptation of mixup for PFN transfer learning. The overall problem formulation (explicitly optimizing a user-defined cost-performance trajectory utility) is a meaningful departure from the asymptotic-performance focus of prior work.

- **Importance of research question:** High. Cost-sensitive HPO is directly relevant to practitioners operating under cloud budgets or cluster allocations, and the observation that specifying a cost-performance trade-off is easier than specifying a target budget is practically well-motivated.

- **Claims well-supported:** Moderate-to-high. The empirical results are broad, consistent, and supported by ablations. The acquisition analysis (Fig. 7a–c) provides mechanistic insight. The main unsubstantiated claim is the preference-learning pipeline, which is presented as a key contribution but validated only superficially.

- **Soundness of experiments:** Moderate. The benchmarks and baseline set are appropriate and broad. However, the absence of cost-aware BO baselines is a notable gap given the paper's framing, and the stopping criterion comparison is inherently disadvantageous to baselines (who cannot use the PI-based component). The Table 3 error undermines confidence in the ablation.

- **Clarity of writing:** Good overall, with intuitive explanations and well-designed figures. The Algorithm 1 notation issue and the Table 3 presentation error are concrete clarity failures that should be fixed.

- **Value to the research community:** Solid. The empirical finding about transfer learning dominating multi-fidelity mechanics has implications beyond this paper, and the utility framework and surrogate training recipe are directly usable by practitioners.

- **Contextualization relative to prior work:** Adequate in the main text; the paper positions itself clearly against freeze-thaw BO and transfer-BO literature. The positioning against cost-aware BO is underdeveloped, though the appendix apparently contains additional discussion.

MY FINAL SCORE: <pineapple>6.4</pineapple>