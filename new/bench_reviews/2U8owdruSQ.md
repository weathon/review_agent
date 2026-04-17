Now I have sufficient context to write the review. Let me synthesize all the information from the paper and the reviews.

Key observations from my reading of the paper:

1. **The core theoretical claim**: The paper claims ECE satisfies the "necessary condition" for testing F2SP. The formal result (§3.4.1, line 160) states: "a low ECE satisfies the necessary condition for evaluating fidelity to Statistic-GT, but not the sufficient criterion." This is indeed only a trivial property: if the predictor perfectly matches Statistic-GT, then ECE→0. This is exactly what calibration means.

2. **The paper's language is consistently overclaiming**: Despite acknowledging "necessary but not sufficient," the abstract says "ECE uniquely captures F2SP," the conclusion says "ECE uniquely evaluates this capability," and §4.2.2 says "ECE remains low and stable for the DNN trained on S-Level 20, indicating it has learned the correct Statistic-GT." The gap between "necessary" and "sufficient" is enormous and never bridged.

3. **The Statistic-GT definition is actually about marginals**: Line 132 defines Statistic-GT as $P_t = \{p_{t,(i,j)}\}^{H \times W}$ where each $p_{t,(i,j)}$ is a marginal Bernoulli probability. Despite calling this a "joint probability distribution," it's actually a field of marginal probabilities. ECE only tests marginal calibration per bin, not spatial correlations or joint structure.

4. **The claim about proper scoring rules is problematic**: The paper argues MSE/Brier's refinement term makes it unsuitable for F2SP. But by the paper's own definition, if $\hat{p}_{t,(i,j)} = p_{t,(i,j)}$ (perfect Statistic-GT prediction), the Brier score is minimized. The refinement term is a property of the data, not a penalty on correct forecasts.

5. **The empirical evidence is suggestive but not conclusive**: The ECE diagonal pattern in Figure 3 is interesting, but this could simply reflect ECE's sensitivity to distribution shift in marginal probabilities, not uniquely capturing "fidelity to the stochastic process."

6. **The NDWS case study is speculative**: As the reviewers note, the interpretation that ECE improves as DC decreases because the model is "learning the stochastic process" is just one interpretation; an alternative is that the model gets more conservative/uncertain, which coincidentally improves calibration.

Let me now write the consolidated review.</think>Now I have a thorough understanding of the paper and the review inputs. Let me write the final consolidated review.

## Summary

This paper proposes a new evaluation criterion, "Fidelity to Stochastic Process" (F2SP), for assessing whether DNNs have learned the underlying stochastic process of complex systems, as opposed to merely matching a single observed realization (F2R). The authors formalize the "Statistic-GT" as the field of per-cell marginal probabilities across all possible outcomes, argue that traditional metrics (classification metrics, proper scoring rules) cannot measure F2SP, and claim that Expected Calibration Error (ECE) uniquely satisfies the necessary condition for testing F2SP using only the Observed-GT. Empirical validation uses synthetic agent-based models with controllable stochasticity (S-Level) and a real-world wildfire case study.

## Strengths

- **Important problem conceptualization.** The distinction between F2R (fidelity to a single realization) and F2SP (fidelity to the underlying stochastic process) addresses a genuine and underexplored gap in evaluating DNNs for stochastic complex systems. The observation that models can be valuable even when they mismatch Observed-GT—because they may have captured the process-level statistics—is practically important and clearly articulated.

- **Thoughtful synthetic experimental design.** The use of agent-based models with tunable S-Levels to control stochasticity, and the cross-evaluation heatmap paradigm (training on one S-Level, testing on another) is a clean and compelling methodological choice. The diagonal pattern in ECE heatmaps (Figure 3) across three distinct systems is visually striking and demonstrates that ECE behaves differently from AUC-PR and MSE in these settings.

- **Empirically interesting findings.** The observation that ECE remains stable over long prediction horizons when train/test S-Levels match (Figure 4), while AUC-PR degrades regardless, is a noteworthy finding that suggests calibration metrics capture complementary information in stochastic settings.

- **Real-world case study addresses a practical tension.** The NDWS analysis directly addresses the disconnect reported by Huot et al. (2022) between low classification scores and positive qualitative assessments, which practitioners will find relevant.

## Weaknesses

### Major:

- **ECE satisfies only a necessary, not sufficient, condition for F2SP, yet the paper's language and interpretive claims consistently treat low ECE as indicating the model has "learned the stochastic process."** The paper's core theoretical result (§3.4.1) is: if $\hat{p}_{t,(i,j)} = p_{t,(i,j)}$ for all cells, then ECE→0 in expectation. This is simply the well-known property that predicting the true probabilities yields perfect calibration—a tautological restatement. The paper acknowledges "low ECE satisfies the necessary condition... but not the sufficient criterion" (line 160), yet throughout the empirical sections, this critical caveat is forgotten: §4.2.2 states "ECE remains low and stable for the DNN trained on S-Level 20, indicating it has learned the correct Statistic-GT" (line 212); the abstract claims "ECE uniquely captures F2SP"; the conclusion says "ECE uniquely evaluates this capability." A model can achieve low ECE by predicting smoothed or scrambled marginals that preserve per-bin calibration without recovering the true $p_{t,(i,j)}$ field. This gap between necessary and sufficient is not merely a theoretical nicety—it fundamentally undermines the interpretive weight placed on ECE throughout the paper.

- **ECE only tests marginal, per-cell calibration—not fidelity to the "joint stochastic process" the paper defines.** The paper defines Statistic-GT as $P_t = \{p_{t,(i,j)}\}^{H \times W}$ and calls it a "joint probability distribution" (line 132), but each $p_{t,(i,j)}$ is a marginal Bernoulli parameter. ECE bins predictions by predicted probability and compares empirical event frequencies—this is purely a marginal calibration test. It cannot detect whether spatial correlations or interaction dynamics are correctly captured. A trivial model that predicts each cell's marginal probability independently (ignoring all spatial interaction rules) could achieve near-zero ECE while failing to learn anything about the process's joint structure. The paper does not address this limitation, which directly undermines the claim that ECE tests "learning the stochastic process."

- **The claim that proper scoring rules "fail to test F2SP" is not theoretically supported and contradicts standard decision-theoretic results.** Strictly proper scoring rules (Brier/MSE, BCE, CRPS) are, by definition, minimized in expectation when the forecaster predicts the true probability distribution. Under the paper's own framework, if $\hat{p}_{t,(i,j)} = p_{t,(i,j)}$ for all cells, then the expected Brier score is minimized. The paper argues that MSE's "Refinement" component—determined by the data-generating process's variance—makes it unsuitable (§3.4.2, line 172), but this misinterprets the decomposition: the refinement term does not "penalize" correct forecasts; it is a fixed property of the data. The metric still ranks the correct forecast as optimal in expectation. The paper provides no formal argument showing when/why proper scores fail to reflect F2SP while ECE succeeds, nor any experiment where two models with identical proper-score performance have clearly different Statistic-GTs.

- **The empirical results do not establish that ECE uniquely identifies the correct stochastic process.** The diagonal pattern in Figure 3 is consistent with a simpler explanation: ECE is more sensitive to distribution shift in marginal probabilities than AUC-PR, so matching train/test S-Levels (which share similar marginal distributions) yields better calibration. No control experiment tests whether two different processes with identical marginals but different dynamics produce indistinguishable ECE—such an experiment would be essential for validating the "uniqueness" claim. The paper also does not compare ECE to other calibration metrics (e.g., Maximum Calibration Error, Brier's calibration component, or continuous calibration estimators) to test whether the diagonal pattern is unique to ECE.

### Minor:

- **The independence assumption in ECE computation is at odds with spatial dependencies in the studied systems.** As the paper acknowledges (line 160), ECE computation "marginalizes over data points in bin $I_k$, treating them as independent." Grid cells in complex systems are spatially and temporally correlated, potentially making ECE estimates high-variance or unreliable. The paper does not analyze how this affects the validity of its F2SP claims.

- **No direct quantitative evaluation against ESP-derived Statistic-GT in the main paper.** The paper defines Statistic-GT via Monte Carlo ESP (1000 simulations), but the main results never directly measure how close $\hat{P}_t$ is to the true $P_t$ across all cells. The verification is relegated to §E.2 without quantitative details in the main text, making it hard to assess whether the DNNs have actually learned Statistic-GT.

- **The NDWS interpretation is speculative.** In Table 2, ECE improves as Dice Coefficient decreases (more abrupt shifts). The paper attributes this to ECE "capturing F2SP" (line 250), but an equally plausible explanation is that the model predicts conservatively (low probabilities) in high-stochasticity settings, which coincidentally improves calibration while degrading discriminative ability. No analysis of prediction sharpness or comparison to trivial baselines (e.g., climatology forecasts) is provided to disentangle these explanations.

### Trivial:

- The evaluation framework (Figure 1.b) places ECE on the x-axis and AUC-PR on the y-axis, but provides no concrete decision rules for model selection when metrics conflict.

## Nice-to-Haves

- Test a marginal-only baseline (e.g., predicting per-cell base rates) to expose the gap between marginal calibration and process learning.
- Compare ECE against other calibration metrics (MCE, Brier calibration component, kernel-based estimators) to test the "uniqueness" claim.
- Design an experiment where two different stochastic processes share similar marginals but differ in spatial correlations, to more directly test whether ECE can distinguish them.
- Report confidence intervals or bootstrap estimates for ECE, given its sensitivity to binning and sample size.
- Include a "base rate" or climatology baseline in the NDWS analysis to contextualize the ECE values.

## Novel Insights

The paper's most valuable insight is not the ECE-specific claim but the broader observation that evaluation metrics for stochastic complex systems conflate two distinct goals—matching a particular outcome (F2R) vs. capturing the statistical properties of the process (F2SP). The observation that ECE produces a diagnostic "diagonal" pattern when train/test stochasticity levels match, while classification metrics do not, is genuinely interesting. However, this insight is better interpreted as "ECE is a useful complementary metric that is more sensitive to distributional alignment in stochastic settings" rather than as evidence that ECE measures whether a DNN has "learned the stochastic process." Had the paper framed its contribution as opening a new evaluation perspective rather than claiming ECE uniquely solves it, it would be a significantly stronger contribution.

## Suggestions

- **Rewrite the core claims to accurately reflect what ECE provides:** ECE tests marginal calibration, which is a necessary but far-from-sufficient condition for learning the stochastic process. The abstract and conclusion should not claim ECE "uniquely captures F2SP."
- **Explicitly acknowledge the marginal vs. joint gap in the main text** and discuss what additional metrics or conditions would be needed to establish sufficiency.
- **Retract or substantially soften the claim that proper scoring rules "fail to test F2SP."** The current argument contradicts standard decision theory without providing a formal counterargument.
- **Add a base-rate or marginal-only baseline experiment** that would directly demonstrate the limitation of ECE as a proxy for process learning.

## Score and Decision

**Calibration anchors:**

- **SEAL (evaluation framework, Accept Spotlight, scores 8/6/8/6):** Strong evaluation framework paper with clear metrics, rigorous benchmarking, and well-supported claims. This paper under review similarly proposes an evaluation framework but with substantially weaker theoretical grounding and overclaimed results.

- **Reassessing Calibration (Accept Poster, scores 5/6/6):** Solid calibration analysis paper with correct theoretical results. Scores modestly because findings are incremental. The current paper's theoretical contribution is weaker (a trivial necessary condition) but its problem framing is broader.

- **Calibration Benchmark Study (Accept Poster, scores 6/3/6):** Empirical calibration study with comprehensive dataset. The current paper has less comprehensive empirical scope and more overclaimed theoretical claims.

- **Indeterminate Probability Theory (Reject, scores 3/1/6):** Severe overclaiming case. The current paper's overclaiming is less severe than this but follows a similar pattern of ambitious theoretical framing unsupported by the actual results.

- **Stochastic Diffusion/CRPS paper (Accept Poster, scores 5/8/8/3):** Proper scoring rule theory applied to stochastic processes with rigorous formal contributions. The current paper is markedly weaker in its theoretical contribution despite targeting a similar conceptual domain.

**Assessment:** The paper identifies a genuinely important and underexplored problem, but its central claims are substantially overstated relative to what the theory and experiments support. The "necessary but not sufficient" gap is acknowledged once then ignored throughout the rest of the paper, and the claim that proper scoring rules fundamentally cannot test F2SP is unsupported. These are not minor presentation issues—they are central to the paper's contribution. At the same time, the problem framing and empirical observations (particularly the ECE diagonal pattern) have genuine value. The paper sits below the calibration benchmark papers (which have more modest, well-supported claims and got 5-6 range) and well below strong evaluation framework papers like SEAL. It's closer to papers with interesting ideas but overclaimed theoretical contributions that received scores in the 4-5 range.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>