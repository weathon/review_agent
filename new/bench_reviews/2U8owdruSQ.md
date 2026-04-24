## Summary

This paper introduces **Fidelity to Stochastic Process (F2SP)**, an evaluation criterion for DNNs that forecast spatiotemporal stochastic systems, contrasting it with the standard **Fidelity to Realization (F2R)**. The authors propose that Expected Calibration Error (ECE) is well-suited for assessing F2SP because it can be computed from a single observed realization (Observed-GT) yet targets the underlying process probabilities (Statistic-GT). They support this with synthetic benchmark experiments across three complex systems (forest fire, host-pathogen, stock market) using a controlled S-Level parameter, and with a real-world wildfire case study.

## Strengths

- **Clear conceptual distinction and practical motivation.** The paper correctly diagnoses that threshold-based classification metrics and proper scoring rules conflate mismatches due to stochastic variability with mismatches due to model error (§3.3). The F2R/F2SP framework is intuitive and the synthetic S-Level control provides a clean experimental knob for stress-testing metrics (§2.3, §4).
- **Consistent empirical signature across diverse systems.** Figure 3 demonstrates that ECE exhibits a sharp diagonal pattern (low error only when train and test S-Levels match) across all three synthetic benchmarks, while AUC-PR shows no such structure and MSE shows only partial diagonal trends. Figure 4 further shows ECE remains stable over long prediction horizons when the model matches the test S-Level. These results are specific, reproducible, and visually compelling.
- **Real-world case study with actionable implications.** Table 2 shows that on the NDWS wildfire dataset, ECE improves as fire-map overlap (Dice Coefficient) decreases—i.e., as the forecast becomes more stochastic—while classification metrics degrade. This provides practical evidence that ECE measures a complementary property to standard ranking metrics, justifying the proposed two-axis evaluation framework (Figure 1.b).

## Weaknesses

### Fatal
None.

### Major

- **Unsubstantiated "uniqueness" claims about ECE.** The paper repeatedly asserts that ECE "uniquely captures F2SP" (Abstract, §1, §3.4, §4.2.1, §7) and "possesses the unique ability to test fidelity to Statistic-GT" (§3.4). However, the theoretical justification in §3.4.1 is a standard linearity-of-expectation argument: if \(\hat{p}_{t,(i,j)} = p_{t,(i,j)}\), then the empirical fraction in each bin converges in expectation to the bin's predicted probability. *Any* marginal calibration metric—including Maximum Calibration Error (MCE), reliability diagrams, or the calibration component of the Brier score—satisfies the same property. The paper provides neither a proof of uniqueness nor an empirical comparison to alternative calibration metrics; in §F.3 it compares ECE to classification metrics and scoring rules (BCE, CRPS, Energy Score), but notably omits other calibration baselines. Because the uniqueness claim is central to the paper's marketing and repeated in the abstract and contributions, its lack of substantiation is a significant flaw.
- **Contradictory framing of ECE's sufficiency.** The abstract states that ECE "exclusively assesses F2SP," implying it fully captures process fidelity. Yet §3.4.1 correctly notes that ECE ignores spatial dependencies and is only a "necessary condition... but not the sufficient criterion" for evaluating fidelity to Statistic-GT. This tension between the abstract's strong language and the body text's careful qualification undermines the paper's credibility. A reader relying on the abstract would conclude ECE is sufficient for testing whether a DNN has learned the stochastic process, which the paper's own technical section disclaims.

### Minor

- **No direct validation that ECE tracks true Statistic-GT divergence.** Because the synthetic experiments generate the true Statistic-GT via 1,000 Monte Carlo simulations, the authors could directly measure \(\|\hat{P}_t - P_t\|_1\) or \(\|\hat{P}_t - P_t\|_2\) and correlate it with ECE. Instead, the experiments rely on the indirect proxy of train/test S-Level matching (Figure 3). While this proxy is reasonable, direct error computation would immediately validate whether ECE is a faithful proxy for the intended target rather than merely a proxy for S-Level homogeneity.
- **Real-world interpretation is post-hoc and correlational.** In §5, the divergent trends between ECE and AUC-PR on NDWS (Table 2) are interpreted as evidence that ECE measures F2SP. Without access to the true wildfire stochastic process, this interpretation is speculative: the divergence only confirms that calibration and discrimination are different properties, not that ECE specifically captures the latent process distribution. The analysis would benefit from more cautious phrasing.
- **Missing decomposition of MSE into calibration vs. refinement.** The paper notes in §3.4.2 that MSE can be decomposed into calibration and refinement terms, and that the latter confounds the metric. However, it does not report the calibration (reliability) term of MSE in the main experiments. If that term reproduced the diagonal pattern of Figure 3, it would further weaken the claim that ECE is *unique* among calibration-sensitive metrics.

### Trivial

- Figure 3 uses different color scales for each metric column, which slightly obscures direct visual comparison of diagonal structure (though the paper does acknowledge MSE's partial diagonal trend in the caption).
- Minor terminology tension in §3.4.1: calling the expectation identity a "formal demonstration of ECE's unique suitability" overstates a one-line derivation.

## Nice-to-Haves

- Empirical comparison of ECE against MCE and the reliability component of Brier score/MSE on the synthetic benchmarks to properly scope the contribution.
- Direct pixel-wise \(L_1/L_2\) divergence between predicted probability maps and true Statistic-GT, plotted against ECE, to validate the proxy relationship.
- Spatially resolved calibration maps (e.g., moving-window reliability) to reveal whether global ECE pooling hides structured miscalibration.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"ECE cannot test fidelity to the joint distribution / full stochastic process."** The paper explicitly acknowledges in §3.4.1 that ECE ignores spatial dependencies and is only a necessary condition, not sufficient. While the abstract's stronger claims create a real tension, the body text does not present ECE as sufficient for joint fidelity. This weakness is therefore partially addressed and has been reframed as a contradiction between abstract and body rather than a wholesale invalidation.
- **"Related work omits spatial scoring rules (Energy Score, Variogram Score)."** Per hard rules, missing-related-work criticisms are not included. The paper does mention Energy Score in §F.3.
- **Formatting/parser artifacts** (typos, line breaks, garbled symbols): these are parser errors, not author errors.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

- **Reframe the contribution:** Position the paper as advocating for calibration-based evaluation (of which ECE is one practical instance) in complex-systems forecasting, rather than claiming ECE is *unique*. The synthetic benchmarks and real-world case study already provide strong evidence that calibration metrics capture a useful, distinct property.
- **Add the missing calibration-baseline comparison:** Include MCE and the reliability term of MSE/Brier score in Figure 3 or an ablation table. If these metrics do not reproduce ECE's diagonal pattern, the uniqueness claim is rescued; if they do, the contribution should be reframed accordingly.
- **Add direct Statistic-GT error plots:** On the synthetic data, compute the true per-pixel error between predicted and ground-truth process probabilities, and show its correlation with ECE. This would close the empirical loop.

## Score and Decision

**Calibration anchors used:**
- *High:* `37EXtKCOkn` (avg 7.5, spatiotemporal dynamical systems, spotlight) — novel method, strong experiments, no central overclaim. This paper is below it due to the unsubstantiated uniqueness framing.  
- *High:* `5AtlfHYCPa` (avg 6.75, weather forecasting dataset, poster) — solid empirical contribution with clear scope. This paper is below it because the core theoretical claim is trivial and overstated.  
- *Medium:* `X0epAjg0hd` (avg 5.67, calibration reassessment, poster) — interesting theorems but debatable claims and unclear presentation. This paper is comparable: it has stronger experiments but a more central overclaim.  
- *Medium:* `qjFnENGhDE` (avg 5.0, zero-sum games, reject) — sound but reviewers questioned novelty; small-scale experiments. This paper has more novel framing but similar score-level concerns about contribution scope.  
- *Low:* `FjifPJV2Ol` (avg 3.4, Schrödinger bridge, reject) — known theorems, poor evaluation; `2CxkRDMIG4` (avg 1.5, trivial extension, reject). This paper is far above these due to genuine experiments and a novel problem formulation.

The paper has a real and underappreciated insight: evaluation of stochastic spatiotemporal forecasts should separate process fidelity from realization fidelity. The synthetic benchmark is well-designed and the empirical results are consistent. However, the central claim that ECE *uniquely* captures F2SP is not supported—the proof applies to any marginal calibration metric, and no comparison to MCE, reliability diagrams, or the calibration component of scoring rules is provided. The abstract's claim that ECE "exclusively assesses F2SP" directly contradicts §3.4.1's admission that it is only a necessary condition. These issues are fixable with a reframing, but in the current form they undermine the paper's central marketing and distort its contribution.

**Score: 5.0**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>