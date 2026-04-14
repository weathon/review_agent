## Summary

CMBO (Cost-sensitive Multi-fidelity Bayesian Optimization) introduces a utility function U(b, ỹ_b) that explicitly trades off BO step count (cost) against best-found validation performance, reframing multi-fidelity HPO as maximizing user-defined utility rather than asymptotic validation performance under a fixed budget. The method contributes a utility-aware acquisition function (Eq. 2), a stopping criterion that interpolates between normalized-regret and probability-of-improvement signals (Eq. 3–5), and a PFN training strategy using LC mixup for sample-efficient transfer. Experiments across LCBench, TaskSet, PD1, and a collected real-world object-detection dataset show consistent and substantial improvements over multi-fidelity and transfer-BO baselines.

---

## Strengths

- **Novel and practically grounded problem formulation.** The reframing from "maximize asymptotic performance under a target budget" to "maximize a user-defined utility over the BO trajectory" is a conceptually fresh and practically motivated contribution. The cloud-credit / Slurm motivation is concrete and resonates with real practitioner constraints, not a generic appeal to HPO importance.

- **Utility-aware acquisition with dynamic horizon selection.** Unlike prior freeze-thaw methods that fix the target epoch to T (DPL) or use greedy one-step lookahead (DyHPO, Quick-Tune), Eq. (2) jointly selects which configuration to continue *and* how far to run it (Δt), explicitly guided by utility improvement. The behavioral analysis in Fig. 7b—showing that the chosen Δt starts large (non-greedy) then shrinks as cost pressure grows—provides direct empirical validation that the acquisition acts as intended.

- **LC mixup is a specific, non-obvious data-augmentation strategy for PFNs.** The two-stage interpolation (first across datasets preserving inter-configuration correlations, then across configurations) addresses the finite-sample limitation of training PFNs on real datasets rather than synthetic priors. Fig. 6 provides a clear ablation showing reduced test loss and improved BO regret from this design choice alone.

- **Breadth and consistency of empirical results.** The method is evaluated on three standard tabular LC benchmarks plus a collected real-world object-detection dataset, against eight baselines, across multiple utility families (linear, quadratic, square-root, staircase) and a range of penalty strengths α. CMBO achieves the best average rank in nearly all settings, with a particularly decisive margin for strong penalties (α = 2e-04), which is precisely the regime the method is designed for.

- **Acquisition analysis illuminates the mechanism.** Fig. 7a/b/c directly tracks which configurations are selected and how far, confirming that CMBO focuses on fewer, higher-quality configurations under stronger cost pressure while baselines over-explore — this is an unusually transparent mechanistic analysis for a BO paper.

---

## Weaknesses

### Fatal
None. The core contribution is intact.

### Major

- **Algorithm 1, Line 4 has a likely pseudocode error.** The line reads `n* ← argmax_{n∈C} A(n)`, optimizing over elements of the *observation history* C. At b=1, C is empty (initialized to ∅ in line 2), making this undefined. Conceptually the acquisition should range over configurations in X (or those with at least one observation, which requires clarification of how cold-start configurations are handled). This is not merely a formatting artifact — it reflects a genuine ambiguity in when and how new, never-evaluated configurations enter the optimization. The authors should clarify whether CMBO can ever propose a brand-new configuration, and fix the pseudocode accordingly.

- **Table 3 ablation has duplicate row labels.** Rows 3 and 4 both carry the same checkmark pattern (p_b ✓, Acq. ✓, T. ✓) yet report substantially different numbers (e.g., α=2e-04: 4.4 vs. 0.9). Since these are the two rows that together span most of the improvement, the reader cannot tell what factor is being ablated between them. This must be corrected with explicit labels; as written, the ablation cannot be interpreted.

- **No fixed-budget baseline.** The most natural cost-sensitive alternative is simply running any existing BO method with a smaller budget cap. Without a "BOHB/iFBO run to b* steps" baseline, it is impossible to determine whether the utility framework and stopping criterion add value beyond the trivially obvious intervention of allocating less compute upfront. This is a substantive gap in the experimental story.

- **No raw performance metric reported alongside utility regret.** The evaluation metric (normalized utility regret) inherently rewards cost-awareness, so methods that explicitly optimize utility will naturally score well on it. Without also reporting the best validation accuracy achieved at termination — and comparing it to what baselines achieve when allowed to run longer — readers cannot assess the actual performance sacrifice incurred by early stopping. Whether stopping at utility maximum corresponds to acceptable absolute performance degradation is a key practical question left unanswered.

### Minor

- **Utility learning validation is thin and artificial.** The "Estimated" column in Table 2 proxies user preferences by fitting a utility that improves on iFBO's trajectory, which is not a realistic user-elicitation scenario. §3.1 motivates preference learning from genuine user data as a core contribution, but Appendix B (showing utility recovery with varying datapoints) remains the only experiment touching this claim. More realistic validation — even simulated, with a held-out user utility and noisy pairwise queries — would substantially strengthen the case.

- **Sensitivity to γ is not analyzed.** β is studied in Fig. 7d and a clear optimum at β = e^{-1} is identified. However, γ — which sets the baseline stopping threshold (δ_b = 0.2 when β→0 corresponds to γ = log₂5) — is fixed throughout without any sensitivity experiment. Since γ controls the absolute stopping aggressiveness independent of p_b, its choice materially affects all results.

- **Several CMBO entries report ±0.0 standard deviation.** The paper explains that 5 runs are used for CMBO and 30 for noisier baselines, and that deterministic methods (FSBO, Quick-Tune†) naturally show ±0.0. However, CMBO itself reports ±0.0 in multiple entries (e.g., Table 1: 1.3±0.0 on TaskSet; Table 2: several entries). Standard deviation of exactly 0.0 across 5 independent BO runs is implausible unless the benchmarks are fully deterministic end-to-end. This should be investigated and explained.

- **U_min approximation bias affects all reported numbers.** The paper acknowledges that exact U_min is a difficult combinatorial problem and approximates it with U(B, y₁^worst). Since all normalized regret numbers are divided by (U_max − U_min), a poor lower bound on U_min compresses the denominator and makes all methods appear to have higher regret. The direction of this bias is systematic and could affect ranking. At minimum, a brief sensitivity check on this approximation would add credibility.

- **The acquisition is a single-configuration rollout, not stated explicitly.** Eq. (2) evaluates the utility improvement from continuing one chosen configuration for Δt steps, treating other configurations as frozen. This is a one-step approximation to the true multi-step adaptive policy. The text sometimes implies the acquisition is optimizing over "the future BO process," which is stronger than what is computed. This approximation should be stated plainly rather than left to reader inference.

- **No explicit limitations section.** The paper does not surface its own limitations — finite-pool assumption, dependence on in-domain transfer data, heuristic stopping criterion, preference elicitation burden, or benchmark-dependence. For ICLR standards this is a notable omission.

### Tiny

- U_prev in Eq. (2) is the *most recently achieved* utility (not the running maximum), which is an intentional and important design choice (cost is irreversible). The paper explains this correctly in §3.2, but the rationale could be stated earlier, as readers familiar with standard EI will find this non-obvious.
- The temperature τ in the Bradley-Terry model (Eq. 1) is introduced but its selection — whether fixed, fitted, or tuned — is not described in the main text.

---

## Nice-to-Haves

- **Wall-clock cost validation.** All experiments use step-count as cost, which treats every epoch equally regardless of configuration (e.g., batch size, architecture size). At least one experiment with real model training, where different configurations have different per-epoch wall times, would validate the practical cost-sensitivity story.
- **LC extrapolation calibration.** Since both the acquisition and stopping criterion depend heavily on the quality of probabilistic LC forecasts, an experiment measuring NLL or calibration of the PFN predictions at different stages of BO (especially early, where stopping decisions are most consequential) would strengthen confidence in the approach.
- **Scatter plot of actual vs. oracle stopping points.** Aggregate metrics obscure whether the stopping criterion systematically stops too early or too late on specific task types. A scatter plot across all tasks would directly reveal this.
- **Computational overhead of CMBO.** The paper optimizes HPO cost but does not report how much additional wall-clock overhead CMBO's PFN inference and acquisition rollouts impose compared to lighter baselines like BOHB. Even a rough comparison would be useful for practitioners.
- **Broader utility estimation experiment.** A controlled study where synthetic "users" provide noisy pairwise preferences (with varying noise levels and query counts) would illuminate the robustness of the Bradley-Terry estimation under realistic conditions.

---

## Removed Points
*These points are flagged for removal — treat them with caution.*

- **"Insufficient engagement with cost-aware BO literature in related work"** (Harsh Critic): The paper explicitly defers further related work on cost-sensitive HPO and BO with user preferences to §A (Appendix A), and what remains in §2 is appropriately focused on freeze-thaw BO and LC extrapolation, which are the direct technical ancestors. Not a meaningful gap given the appendix coverage.

- **"Lack of theoretical guarantees for the acquisition/stopping criterion"** (Harsh Critic): This is an empirical systems paper in the freeze-thaw BO tradition; none of the closely related works (DyHPO, iFBO, DPL) provide regret bounds either. Demanding theoretical proofs here imposes a non-standard requirement for this subfield.

- **"Comparison to FSBO is unfair because FSBO is black-box"** (Harsh Critic): FSBO is included precisely to show how a strong black-box transfer method compares; the comparison is informative regardless of fidelity access. The fact that FSBO performs well in some settings is discussed and attributed to transfer quality, which strengthens rather than undermines the paper's analysis.

- **"Baselines adapted with stopping rule by the authors"** (Harsh Critic, framing as unfair): All baselines are given the same regret-based stopping rule (Eq. 3 with δ_b = 0.2) because they cannot use the PI-based component (which depends on CMBO's utility-aware acquisition). This is described transparently in footnote 2. The stopping rule used for baselines is a reasonable default, not a straw-man.

- **"Missing references may not exist"**: Per instructions, any cited paper is assumed to exist.

- **"Requesting confidence intervals for tabular benchmarks"**: Single-run evaluation with mean±std over 5 runs is standard in the multi-fidelity BO literature; demanding formal significance tests against every baseline exceeds community norms.

- **"The problem formulation may collapse to budgeted BO with unknown horizon"** (Harsh Critic): This framing is inaccurate. The utility framework is strictly more expressive than a budget cap: it allows non-monotone stopping criteria, user-defined cost-performance tradeoff shapes (staircase, quadratic, etc.), and automatic stopping without committing to any budget a priori. The distinction is real and the paper demonstrates it empirically with non-linear utility functions in Table 2.

---

## Novel Insights

The most interesting observation emerging from synthesis of all three reviews is that **FSBO — a black-box transfer method that ignores fidelity entirely and trains only on last-epoch performance** — frequently outperforms sophisticated multi-fidelity methods (DyHPO, DPL, BOHB) in the α=0 (no cost penalty) setting. The paper attributes this to the outsized impact of transfer learning on sample efficiency, and this is directly supported by the mixup ablation (Fig. 6). The implication is that for regimes where data from related tasks is available, the benefit of multi-fidelity adaptation can be dominated by transfer quality, and the practical value of cost-sensitive multi-fidelity BO may lie specifically in the *regime of strong cost pressure* (large α) — where CMBO's margin over FSBO is large and where FSBO's black-box nature becomes a liability. This suggests a useful design principle: the justification for freeze-thaw BO over simpler black-box transfer BO is not sample efficiency per se but cost-awareness, and future work should more explicitly test this regime boundary.

---

## Suggestions

1. **Fix Algorithm 1 Line 4**: Change `argmax_{n∈C}` to `argmax_{n∈X}` (or clearly specify the set being optimized), and add an explicit initialization protocol for configurations that have never been evaluated (C is empty at b=1).

2. **Resolve Table 3 ambiguity**: The two rows both labeled (p_b ✓, Acq. ✓, T. ✓) must be differentiated. If one uses a different β or a different stopping variant, label it explicitly.

3. **Add a fixed-budget baseline**: Run iFBO and/or CMBO (without its stopping criterion) capped at exactly b* steps (where b* is the average stopping step of full CMBO), and report utility regret. This directly tests whether the stopping criterion adds value over simply running less.

4. **Report best validation performance at termination**: Add a companion table or figure showing mean best-found validation accuracy (not utility regret) at the stopping point of each method, across tasks, so readers can assess whether the utility-optimal solution is practically acceptable.

5. **Explain or investigate zero-variance CMBO entries**: Verify whether the ±0.0 entries arise from full benchmark determinism, collapsed distributions across tasks, or an artifact of averaging. Report the explanation in the text.

6. **Add γ sensitivity analysis**: Fig. 7d studies β but treats γ as fixed at log₂5. A brief supplementary sensitivity analysis on γ would complete the stopping criterion characterization.

7. **Provide explicit statement of the finite-pool limitation and transfer-data dependency**: Add a short limitations paragraph acknowledging that (a) the method assumes a pre-specified pool X and has not been evaluated in continuous HP spaces, and (b) transfer gains depend on in-domain task availability.

---

**Overall assessment**: The paper presents a **genuinely novel and practically motivated problem formulation** with **solid, consistent empirical results**. Novelty is moderate-to-high for the utility framework and the LC mixup strategy; technical soundness is moderate (the stopping criterion is heuristic and the pseudocode has real ambiguities); empirical support is good in breadth but has notable gaps (no fixed-budget baseline, no absolute performance metrics). Significance is high given the prevalence of resource-constrained HPO in practice. Clarity is adequate overall but requires specific fixes to the ablation table and algorithm pseudocode before the paper is ready for final publication.