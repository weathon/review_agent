## Summary
CMBO (Cost-sensitive Multi-fidelity Bayesian Optimization) reformulates HPO as maximizing a user-specified utility function that trades off BO step cost against best-so-far performance, rather than optimizing asymptotic validation accuracy. The paper introduces a utility-based acquisition function with dynamic lookahead, an adaptive stopping criterion that blends regret-based and probability-of-improvement signals, and a novel two-stage LC mixup strategy for training PFN surrogates on existing learning curve datasets. Across four benchmarks (LCBench, TaskSet, PD1, RoboFlow) and eight baselines, CMBO consistently achieves the best normalized regret under various cost-sensitive settings.

---

## Strengths

- **Genuinely novel utility-based framing for freeze-thaw BO.** Prior freeze-thaw methods (DyHPO, iFBO, DPL) target asymptotic performance or a fixed final epoch; CMBO is the first to treat the joint (cost, performance) trade-off as the primary objective, and this reformulation drives both acquisition design and endogenous stopping in a unified way.

- **Two-stage LC mixup preserves cross-configuration structure.** The key insight—applying the same λ₁ across all configurations in the first mixup step to preserve correlation structure before individual interpolation in the second step—is a principled and non-obvious design choice for augmenting learning curve datasets for PFN training. Fig. 6 shows it measurably reduces test loss and downstream regret on PD1.

- **Acquisition function analysis confirms intended behavior.** Figs. 7a–c show that the selected configurations initially have large optimal Δt (exploratory, non-greedy) but progressively shift toward Δt≈0 (exploitative) as cost dominates, and the method concentrates on fewer configurations under higher cost penalties. This directly confirms that Eq. (2) functions as designed.

- **Comprehensive ablation isolates contributions.** Table 3 cleanly attributes gains to three distinct components (stopping criterion, acquisition, transfer learning), with monotone improvement as each is added—particularly dramatic under strong cost penalties (α = 2e−4).

- **Strong performance across diverse benchmarks and utility forms.** Rank 1.0 across all conditions in Table 2 (various utility functions on PD1) and consistent top rank in Tables 1 and 4 demonstrate that the method's advantage is not confined to a narrow setting.

---

## Weaknesses

### Fatal
None.

### Major

- **Cost is modeled as BO step count, not actual compute.** The utility U(b, ỹ_b) penalizes the integer step index b, but the paper explicitly motivates cloud credits, wall-clock time, and Slurm quotas—all of which are non-uniform across configurations and architectures. In the RoboFlow experiment, three architectures (ResNet-50, HRNet, MobileNetv2) have very different per-epoch compute. The current formulation equates all BO steps as equal-cost, which directly undermines the "cost-sensitive" framing. This is the single largest gap between the paper's motivation and its technical contribution; the utility function and stopping criterion need the cost axis to be meaningfully calibrated for real-world applicability.

- **Table 3 ablation has unexplained duplicate rows.** The last two rows are both labeled p_b ✓, Acq. ✓, T. ✓ but report substantially different results (e.g., 4.4 vs. 0.9 for α=2e−4). Since the paper describes only three binary ablation factors, there is an implicit fourth varying component that is never labeled. As presented, this renders the bottom two rows uninterpretable and weakens the ablation evidence.

- **Transfer learning contributes disproportionately to gains, but cross-benchmark ablation is missing.** The mixup ablation (Fig. 6) is presented only on PD1. Given that transfer learning is one of three claimed core contributions, an ablation on at least one additional benchmark is necessary to confirm generalization of this finding.

### Minor

- **Utility elicitation from user preferences is not validated end-to-end.** All main experiments use analytically specified utilities (linear, quadratic, staircase). The "Estimated" condition in Table 2 is constructed synthetically by assuming the user wants a better trade-off than iFBO—this is a method-relative construction, not a real user preference. The Bradley-Terry preference learning is demonstrated only in isolation (Fig. 2, 1,000 synthetic queries). Since this component is presented as a substantive contribution, its absence from the closed-loop evaluation (estimate utility → run BO → measure outcome) is a notable gap. Appendix §B discusses fewer datapoints but still uses synthetic queries.

- **Algorithm 1 notation bug.** Line 4 reads n* ← argmax_{n ∈ C} A(n), but C is a set of (x, t, y) triples, not a set of configuration indices. This should be n ∈ [N]. The intent is clear from context, but the formal definition is incorrect and should be corrected.

- **β sweet spot varies by dataset but a single value is used globally.** Fig. 7d shows that the optimal β differs for LCBench, TaskSet, and PD1, yet β=e⁻¹ is applied uniformly across all experiments. While the average performance at β=e⁻¹ appears good, there is no discussion of how sensitive results are to this choice and how a practitioner would select β without access to the benchmark. The paper should include a brief sensitivity analysis or criterion for choosing β in new settings.

- **Zero standard deviations for FSBO and Quick-Tune† are not explained.** Several entries in Tables 1–4 for these methods show ±0.0. While FSBO and Quick-Tune† may be deterministic, ±0.0 at the reported precision should be explicitly justified (e.g., confirmed deterministic, or variance rounds below 0.05).

### Tiny

- **Notation inconsistency between ỹ_b (§3.1) and ȳ_b (Algorithm 1).** Both appear to denote best-so-far BO performance but use different symbols. Line 10 of Alg. 1 defines ȳ_b while §3.1 introduces ỹ_b for the same quantity. Similarly, Eq. (5) uses ȳ_{b+Δt} in the indicator while prior sections use ỹ.

- **The U_prev update at line 11 of Alg. 1 clarifies a key design choice, but interaction with the acquisition is not analyzed.** The paper explains that U_prev is not the global best utility but the most recent one (justified by cost irreversibility). However, if utility dips repeatedly, the acquisition threshold decreases, which could allow continued exploration of configurations with modest expected improvement. The paper's discussion is qualitative; a brief quantitative analysis would strengthen the argument.

---

## Nice-to-Haves

- Extend U(b, ỹ_b) to accept a per-step cost function c(x_n, t_n) so that BO steps differing in wall-clock cost are weighted appropriately; this would make the "cloud credits" and "Slurm wall-time" motivations technically consistent with the formulation.
- Provide wall-clock overhead comparison of the PFN surrogate vs. lighter alternatives (GP, deep kernel GP), including PFN pretraining cost, so readers can assess total efficiency.
- End-to-end validation with real user preference queries (even a small user study) to test whether ~100–300 pairwise comparisons suffice for downstream BO quality.
- An oracle stopping comparison scatter plot—actual b* chosen by CMBO vs. the oracle optimal stopping step—would be the most direct visualization validating the stopping criterion.
- Ablation of the BetaCDF stopping form in Eq. (4): compare against simpler monotone squashing functions (logistic, piecewise linear) to assess whether the specific family matters or whether any monotone transform of p_b works similarly.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Eq. (3) stopping rule is directionally odd."** The critic claims the rule stops when normalized regret is large. This is correct and intentional: the LHS (Û_max − U_prev)/(Û_max − Û_min) is large precisely when U_prev has declined far below the best seen utility, signaling the BO has passed its optimum. This is the semantically correct stopping signal. The critic misread the criterion's intent.

- **"Using U_prev rather than best utility so far is unusual and under-analyzed."** The paper explicitly justifies this on page 5: "the cost of BO that has previously been incurred is not reversible." This is a deliberate and sound modeling choice. The stopping criterion compensates for cases where utility has fallen and continues falling.

- **"Stopping rule for baselines is not optimized fairly."** Footnote 2 explicitly explains that the PI-based component of the stopping criterion (Eq. 5) depends on the utility-aware acquisition function, which baselines do not have. The paper correctly gives baselines the best-available regret-only stopping threshold (δ_b = 0.2), and notes this was found to perform well for them. This is a reasonable constraint, not a fairness failure.

- **"The evaluation metric favors the proposed formulation by construction."** Optimizing and being evaluated on the same criterion is expected for any method with a well-defined objective. The paper also reports complementary metrics (average rank, Fig. 5 trajectories, Fig. 7 analyses), providing multiple lines of evidence.

- **"Potential data leakage through task-level interpolation."** The paper uses distinct train/test task splits; linear interpolation between training tasks is a standard augmentation. Without evidence of leakage, this is speculative.

- **"FSBO outperforms multi-fidelity methods is under-analyzed."** The paper does provide a clear explanation: transfer learning substantially improves sample efficiency, making FSBO (which uses the same LC datasets) competitive despite its black-box nature. Quick-Tune† underperforms FSBO due to a greedy acquisition and no data augmentation.

- **"The conclusion is overly confident."** The empirical evidence is comprehensive (4 benchmarks, 8 baselines, ablations, real-world data). The conclusion language is somewhat strong but proportionate to the experimental scope.

- Missing related work critiques (removed per review instructions).

---

## Novel Insights

The paper surfaces an underappreciated failure mode of multi-fidelity BO: even methods that efficiently allocate epochs *within* the optimization loop systematically over-explore because they ignore the cost of the optimization process itself. The insight that cost-sensitivity should act on the meta-level (when to stop the entire BO) rather than only at the object-level (which configuration to evaluate next) is productive and broadly applicable. The finding in Fig. 4 that FSBO—a black-box method that cannot switch configurations mid-run—outperforms all multi-fidelity baselines on most benchmarks is striking: it implies that the sample efficiency gains from dynamic configuration switching are currently smaller than the gains from strong transfer surrogates. This suggests that the field may be somewhat overfocused on sophisticated acquisition designs relative to the surrogate quality, a message the CMBO results reinforce since the transfer component drives the largest share of improvement in Table 3.

---

## Suggestions

- **Fix the Table 3 presentation:** label or describe all four ablation conditions, including the implicit component distinguishing the last two rows. Consider introducing a "T. (no mixup)" vs. "T. (with mixup)" distinction explicitly.
- **State the equal-cost-per-step assumption explicitly** in §3.1 and acknowledge it as a limitation in §5, given that the motivation heavily invokes non-uniform compute scenarios.
- **Expand the mixup ablation to at least one additional benchmark** (LCBench or TaskSet) to confirm that the results in Fig. 6 generalize.
- **Clarify zero-variance entries:** add a footnote or table note explaining which methods are run deterministically and why their variance is below rounding threshold.
- **Address the Algorithm 1 notation bug** (line 4: n ∈ C → n ∈ [N]).
- **Provide a brief discussion of β selection** for practitioners: does the chosen β=e⁻¹ correspond to a principled heuristic, or should users tune it, and if so, how?

---

## Evaluation on Key Axes

**Originality:** High. The utility-based objective for freeze-thaw BO and the two-stage LC mixup are both novel contributions not present in prior work. The adaptive stopping criterion combining normalized regret with probability of improvement is also a new design.

**Importance of research question:** High. Cost-sensitivity in HPO is practically critical and structurally underrepresented in the multi-fidelity BO literature. The shift from "maximize terminal accuracy" to "maximize utility over (cost, performance)" is a meaningful and well-motivated reformulation.

**Claims well-supported:** Mostly. Empirical results are comprehensive across four benchmarks, and the ablation isolates contributions at the component level. However, the utility learning-from-preferences claim—arguably the most novel aspect of the framing—is not validated in the closed-loop BO setting with real user data.

**Soundness of experiments:** Good overall, with one significant gap: the ablation table has an unresolved ambiguity (duplicate rows), and the mixup ablation is restricted to one benchmark. The evaluation metric and baseline stopping setup are well-documented.

**Clarity of writing:** Good. The method is clearly described and the algorithm is easy to follow. Minor notation inconsistencies (ỹ vs. ȳ, C indexing in Alg. 1) should be cleaned up.

**Value to the research community:** High. The framework, transfer surrogate, and stopping criterion are modular and could benefit practitioners running HPO under real budget constraints. The real-world RoboFlow experiment adds credibility beyond tabular benchmarks.

**Contextualization relative to prior work:** Good. The paper accurately positions CMBO against the freeze-thaw BO literature and provides a principled distinction between greedy and non-greedy multi-fidelity methods, black-box transfer BO, and the new cost-sensitive setting.