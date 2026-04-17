Now I have a thorough understanding of the paper. Let me write the final review.

## Summary

3CIL proposes a causality-inspired conditional imitation learning method for autonomous driving that addresses spurious correlations (causal confusion) in behavior cloning. The method identifies three traits a robust imitator must possess—sufficient state inference from history (T1), minimal reliance on spurious correlations via action residual prediction (T2), and emphasis on high-divergence scenarios via sample weighting (T3)—and implements them using a conditional VAE with RSSM for future image reconstruction, supervised contrastive learning (RNC loss), action residual prediction, and a residual-error-based sample weighting scheme. Experiments in CARLA across 6 scenarios show improved driving performance over baselines in most settings.

## Strengths

- **Addresses a genuinely important problem.** Causal confusion (inertia/copycat problems) in imitation learning for autonomous driving is well-documented and practically relevant. The paper provides a structured decomposition of the problem into three traits (T1–T3), which is a conceptually useful framework for thinking about robustness in IL.

- **Strong empirical performance in most scenarios.** 3CIL achieves the highest accumulated reward in 5 of 6 CARLA scenarios, demonstrating meaningful improvements over both vanilla CIL and sophisticated baselines like PALR, DIGIC, and Premier-TACO. The evaluation includes distribution shifts (unseen towns, modified weather/traffic/camera), going beyond simple in-distribution testing.

- **Coherent technical design.** The combination of future-frame reconstruction, action residual prediction, and supervised contrastive learning into a representation learning pipeline is well-motivated and coherently integrated. The two-stage training (representation then policy) with frozen representation for the second stage is a clean and reproducible design.

- **Partial ablation via RNC and RAP baselines.** Including RNC (contrastive only) and RAP (residual prediction only) as baselines provides useful insight into individual component contributions, and the discussion in Sec. 4.2 about complementary roles of these components is informative.

## Weaknesses

### Major:

- **The "causality-inspired" framing substantially overstates the formal causal content of the method.** The paper's central contribution is framed as deriving algorithmic design from causal reasoning, but the actual method is a particular combination of standard deep learning components (conditional VAE, supervised contrastive learning, auxiliary regression, exponential weighting). No causal identification, intervention analysis, counterfactual reasoning, or do-calculus is performed. The causal graph in Fig. 1b is used only as motivational scaffolding—not as a formal constraint. Three specific overstated claims: (1) The statement that "directly building a policy π(a_t|o_t,v_t) is inappropriate" (Sec. 3.1) misapplies a causal DAG of the data-generating process to the imitator's decision process; in a POMDP, learning π(a|o,v) or π(a|h) is the standard and valid approach—the causal DAG does not make it "inappropriate." (2) The sample-weighting scheme is described as "akin to inverse probability weighting and doubly robust learning" (Sec. 3.3), but there is no treatment assignment model, no propensity score, and no doubly-robust estimator; the weighting is a heuristic exponential function of one model's prediction error. (3) The claim that bidirectional supervision makes ŝ_t "produce stable causal effects on descendant nodes" (Sec. 3.1) is asserted without formal argument or empirical validation of stability under interventions. The gap between rhetoric and realization is significant because the causal framing is the paper's primary novelty claim—without it, this is a new combination of known ingredients.

- **Scenario 6 performance directly contradicts the core robustness claim.** In Scenario 6 (Town05, unseen), 3CIL achieves only 195.53 reward—less than half of RAP (447.44) and far below DIGIC (409.88) and PALR (389.07). This is the only truly unseen town scenario (Scenario 5 uses Town02, also unseen, where 3CIL does well), and it represents the hardest generalization test. Yet the paper does not discuss or analyze this failure at all. The claim that "3CIL still maintains a robust driving strategy" (Sec. 4.2) and that "the pursuit for T1,T2,T3 does improve the robustness" is selectively supported—ignoring the scenario where the full method dramatically underperforms its own ablations.

- **No evidence that the method actually reduces causal confusion.** The paper motivates the entire approach as addressing inertia/copycat spurious correlations, but no experiment measures whether these specific failure modes are reduced. There is no diagnostic metric (e.g., dependence on past actions, sensitivity to speed perturbation, reaction to interventions breaking speed→braking correlations). The improvement in aggregate reward/collision rate is compatible with many non-causal explanations (better representation learning, regularization, etc.). Given that the causal narrative is the primary selling point, this evidential gap is significant.

- **The sample-weighting mechanism's link to expert-imitator divergence is not established.** The residual prediction error δa measures how well the representation model predicts action differences, not how far the final policy J is from the expert. These are different quantities: a sample hard to predict for f_r could be easy for J, and vice versa. Since G and f_r are frozen when learning J, δa encodes representation-model failure modes rather than behavioral mismatch. No analysis is provided showing that high-δa samples correspond to high-expert-divergence scenarios; without this link, the T3 justification is speculative. Furthermore, the aggressive exponential weighting (γ=6.67) could heavily distort the training distribution with no sensitivity analysis or comparison to simpler reweighting baselines (e.g., direct action prediction error as in Keyframe).

### Minor:

- **No variance or statistical significance reported.** All results in Table 1 are single-point estimates. CARLA evaluations are notoriously high-variance; without standard deviations across multiple seeds, it is unclear whether the moderate differences between methods are robust.

- **Incomplete ablation structure.** The paper provides RNC (T2-contrastive only) and RAP (T2-residual only) as partial ablations, but there is no ablation removing the sample-weighting alone or reconstructing all frames vs. future frames only. A full factorial ablation (as mentioned in the appendix but not in the main text) would strengthen the evidence for each component.

- **High collision rate in Scenario 3.** 3CIL achieves 3.15‰ collision rate in Scenario 3, the worst among all non-PALR methods, which is notable for a safety-critical application. The paper's claim that "3CIL is one of the most cautious drivers" is not uniformly supported.

- **Limited novelty of individual components.** The core building blocks—conditional VAE with RSSM, supervised contrastive learning (RNC), action residual prediction, and prediction-error-based weighting—are all drawn from prior work. The paper's novelty lies in the combination motivated by the causal decomposition, but this is weakened by the overclaiming of the causal framework (see Major weakness above).

### Trivial:

- The claim about maximizing I(ŝ_t, a_t|a_{t-1}) is mentioned but never operationalized; it serves as motivational language rather than a formal objective.

## Nice-to-Haves

- Direct diagnostic experiments for causal confusion (e.g., measuring sensitivity of predictions to speed perturbation while holding images constant, or testing interventional robustness).
- Analysis and discussion of the Scenario 6 failure—why does 3CIL catastrophically fail in Town05?
- Sensitivity analysis for the weighting hyperparameters (b_min, b_max, γ), which control a critical mechanism.
- Multi-seed evaluation with variance estimates.
- Comparison against more recent CARLA baselines (e.g., Transfuser-like methods) to better contextualize improvements.

## Novel Insights

The T1–T3 decomposition (state sufficiency → spurious correlation avoidance → divergence-focused weighting) is a genuinely useful conceptual framework for reasoning about robustness in imitation learning, independent of whether the specific algorithmic realization achieves it through causal mechanisms. The idea of replacing explicit previous-action inputs with action-residual prediction to capture a_{t-1}→s_t→a_t influence while avoiding copycat shortcuts is promising, even though the paper does not conclusively show it works for the stated causal reasons rather than as generic regularization.

## Suggestions

- **Tone down or restructure the causal framing.** Either replace "causality-inspired" with a more accurate description (e.g., "representation-regularized") and present the causal graph as motivation rather than formal justification, or design experiments that directly test causal predictions (intervention-based robustness, measurement of spurious correlation reliance).
- **Add a dedicated paragraph analyzing Scenario 6 failure.** Explain whether it's a representation failure, weighting failure, or data distribution issue—this is essential for the robustness narrative.
- **Add a direct causal confusion diagnostic.** A simple test: measure how much ŝ_t's predicted action changes when speed is artificially perturbed while keeping the image constant, comparing across methods.

## Evaluation on Axes

- **Originality:** Moderate. The combination of known components under a causal framing is the main novelty, but the causal claims are more rhetorical than substantiated.
- **Importance of research question:** High. Causal confusion in IL for autonomous driving is important and well-recognized.
- **Claims support:** Weak-to-moderate. Empirical performance is strong in most scenarios, but the core causal claims lack direct evidence, and Scenario 6 contradicts the robustness narrative.
- **Experimental soundness:** Moderate. Good scope (6 scenarios, 7 baselines) but missing variance estimates, missing causal diagnostics, and unaddressed Scenario 6 failure.
- **Clarity:** Good. The paper is well-structured and the T1–T3 framework is clearly presented.
- **Community value:** Moderate. The T1–T3 framework and the practical method could be useful, but the overclaimed causal framing risks misleading readers.

## Calibration

Papers compared against:
- **Diffusion-Based Planning for AD (Oral, scores 8,8,8,6):** Much stronger—real-world benchmarks (nuPlan), comprehensive ablations, clear methodological contribution. This paper is far below this tier.
- **Copilot4D (Accept poster, scores 6,8,6,10,6):** Strong world model for driving with large performance gains; concerns about causal confusion in open-loop but still accepted for empirical strength. Our paper has weaker empirical scope (CARLA only, no real-world).
- **Causally-Aware Multi-Agent Representations (Reject, scores 5,6,6):** Similar pattern—causal framing but limited validation that the method actually achieves causal robustness. Scored below threshold despite reasonable experiments.
- **BaCE / Class-Incremental with Causal Effects (Reject, scores 6,3,5,6):** Similar overclaim of causal mechanisms where the method is really heuristic, plus experimental issues. Mean ~5, rejected.
- **GAP: Generative Aided Planner (Reject, scores 5,6,5,5):** Good CARLA results but limited evaluation scope and novelty. Mean ~5.25, rejected.
- **DriveE2E (Withdrawn=Reject, scores 3,3,5,3):** CARLA-only evaluation with significant gaps, mean ~3.5.

Our paper has stronger empirical results than GAP and DriveE2E (3CIL wins 5/6 scenarios), but shares the overclaiming pattern with Causally-Aware Multi-Agent and BaCE. The Scenario 6 failure and absence of causal diagnostics are significant weaknesses. It sits roughly between the rejected causal-overclaiming papers (scores 4-5.5) and the accepted driving papers (scores 6+) that have stronger empirical grounding. A fair assessment places it below the acceptance threshold.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>