## Summary

This paper introduces Earliest Disagreement Q-Evaluation (EDQ), a model-free off-policy evaluation method for continuous-time sequential decisions governed by marked point processes. EDQ proposes a dynamic programming recursion that avoids time discretization by regressing Q-values to the earliest time at which the observed and counterfactual treatment intensities diverge. The authors prove a tower property (Theorem 1) for this recursion, ground the estimator in continuous-time causal identification theory, and empirically compare EDQ against discretized FQE and ERM baselines on time-to-failure and tumor-growth simulators.

## Strengths

- **Novel and elegant core idea.** The earliest-disagreement recursion (Section 3.1) is conceptually attractive: it adaptively chooses the lookahead horizon based on treatment events rather than a fixed discretization grid, potentially avoiding the optimization instability and information loss that come with fine-grained time steps. This directly addresses a practically important gap in scalable continuous-time off-policy evaluation.
- **Compatible with modern sequence models.** The implementation uses a GPT-2 architecture with continuous-time positional embeddings and target-network soft updates (Section 5), demonstrating that the recursion can be paired with high-capacity models. This is a meaningful systems contribution relative to prior continuous-time causal estimators that rely on differential equation solvers or importance-weight integrals.
- **Explicit causal identification framework.** The paper formally states ignorability assumptions (Assumption 1) via local independence and eliminability (Definitions 2–3) from the point-process causal inference literature, and proves that the learned Q-function yields valid causal effects under these assumptions (Corollary 1). This rigor is valuable in a space where many scalable methods ignore continuous-time causal validity.
- **Empirical evidence of robustness to timing distribution shift.** On the time-to-failure simulator, EDQ achieves normalized RMSE of 0.20 ± 0.006 when λ_obs = 0.5 and λ_int = 0.1, compared to 0.23 ± 0.04 for FQE and 0.38 ± 0.011 for ERM (Figure 3, red row). This suggests the continuous-time recursion provides some benefit under mismatched treatment intensities.

## Weaknesses

### Fatal
None.

### Major
- **Inconsistent formalization of the augmented history across Definition 4, Theorem 1, and Algorithm 2.** Definition 4 defines δ as the next *observed* treatment time (min over \tilde{\mathcal{H}}^{a_{\text{obs}}}), but the surrounding text and Figure 1 describe it as the minimum of the next observed and next counterfactual treatment times. More critically, Theorem 1 conditions the inner expectation on \mathcal{H}_t \cup \tilde{\mathcal{H}}_{(t,t+\delta]}^{\setminus a_{\text{obs}}} (all events *except* observed treatments), whereas Algorithm 2 line 6 constructs the history as \mathcal{H}_t \cup \tilde{\mathcal{H}}_{(t,t+\delta]}^{a_{\text{obs}}} (only observed treatments). These are contradictory specifications. While the underlying tower-property idea is sound, the paper’s main formal statements and pseudocode are not internally consistent as written, and readers cannot determine which specification the implementation actually follows. This undermines confidence in the theoretical guarantee and reproducibility of the method.
- **Experiments evaluate against single stochastic samples rather than the true conditional expectation.** Section 5.2 computes normalized RMSE between f_\theta(\mathcal{H}_t) and a single sampled label y_t \sim P(\cdot|\mathcal{H}_t). Because the simulators are stochastic (vital dynamics have additive noise, tumor growth is random), squared error against a single realization conflates estimator bias with irreducible variance. Since these are simulators, the true expectations can be computed by Monte Carlo averaging from each test history; without that, the empirical results do not demonstrate that EDQ accurately estimates the causal estimand \mathbb{E}_P[Y|\mathcal{H}_t].
- **Tumor-growth experiment does not test continuous-time treatment intensities.** Section 5.2 explicitly states that this simulator “works in discrete time t \in [T]”; irregularity is induced by missing observations. The treatments are applied at discrete time steps with a policy over a finite action set. This directly contradicts the paper’s central claim of handling interventions on continuous-time treatment intensities λ^a, and standard discrete-time OPE methods are applicable to this task. Using a discrete-time simulator as evidence for continuous-time timing interventions substantially weakens the empirical contribution.
- **FQE baseline underperforms ERM even on-policy, suggesting implementation weakness rather than a fundamental limitation of discretization.** In Figure 3 (right), when λ_obs = λ_int = 0.5, FQE achieves 0.197 ± 0.013 versus ERM/MC at 0.11 ± 0.004. In Figure 4 (right), FQE slightly outperforms EDQ when λ_obs = λ_int = 2. Since ERM is a simple supervised predictor and FQE should match or exceed it on-policy when properly tuned, this pattern indicates the FQE implementation likely suffers from optimization instability (e.g., bootstrap noise) that is not intrinsic to time discretization. Without evidence that the baseline was tuned comparably (e.g., target-network ablation, learning-rate search, stabilization tricks), the performance gap is unconvincing evidence for EDQ’s superiority over a competently implemented discrete-time alternative.

### Minor
- **Algorithm 2 is under-specified regarding model-freeness.** Line 6 instructs the reader to “Draw \tilde{\mathcal{H}} \sim \tilde{P}(\cdot|\mathcal{H}_t),” but the main text never explains how this sampling is implemented without a model of P_obs. In practice the algorithm appears to splice observed trajectory segments with sampled counterfactual treatment times, yet the conditions under which observed post-t segments remain valid after a counterfactual treatment are not made explicit. Clarifying exactly which components of \tilde{P} are sampled and which are taken from data would strengthen the methodological contribution.
- **Time-to-failure simulator has regular observations; it does not fully test irregular observation times.** Section 5.2 notes that in this simulator “observations measured regularly every one time unit; only treatments are irregular.” Consequently, neither experiment fully exercises the paper’s claimed ability to handle irregular *observation* times (N^x) alongside irregular treatments.

### Trivial
- None.

## Nice-to-Have
- A continuous-time simulator where *both* observations and treatments are generated by point processes (e.g., a continuous-time physiological model) would directly test the paper’s core contribution.
- Monte Carlo ground-truth evaluation: for each test history \mathcal{H}_t, averaging many rollouts under the target policy to obtain an accurate estimate of \mathbb{E}_P[Y|\mathcal{H}_t] would resolve the single-sample evaluation concern.
- Estimated effect curves plotting \mathbb{E}_P[Y|\mathcal{H}_t] as a function of proposed treatment time for held-out patients would provide qualitative validation that EDQ learns sensible timing recommendations.

## Removed Points
These points are flagged to be removed; treat them with caution.

- *“Those observations were generated under P_obs without that treatment and are therefore not valid under the interventional distribution P.”* — This claim in the harsh review is factually incorrect. In the augmented process \tilde{P} defined in Definition 4, the intensities of x and y are \lambda_{obs}^e(t|\mathcal{H}_t^a), meaning they are generated *conditional on the counterfactual treatment history* \mathcal{H}^a, not under P_obs. The issue is not that these events are invalid for P, but that Definition 4’s δ does not match the intuitive “earliest disagreement” and that Algorithm 2’s history construction does not match Theorem 1.
- *Criticism about positional embedding differences between EDQ and FQE.* — The difference between continuous and discrete positional embeddings is an inherent consequence of comparing continuous-time versus discretized methods; controlling for it would require changing the core method and is not a standard baseline requirement.
- *Request for ablation of target networks and architecture.* — While useful, this is a standard experimental suggestion rather than a flaw that threatens core claims.

## Novel Insights

None beyond the paper’s own contributions. The earliest-disagreement recursion is a genuinely novel observation about point-process structure that enables model-free bootstrapping without fine discretization, and the paper correctly identifies that standard FQE fails in continuous time because P(\mathbf{x}_{t+2}|\mathcal{H}_t) \neq P_{obs}(\mathbf{x}_{t+2}|\mathcal{H}_t). However, the submitted version does not yet deliver convincing evidence that the implemented estimator accurately recovers interventional expectations.

## Suggestions

- Correct Definition 4 to define δ over the union of observed and counterfactual treatments, and ensure Algorithm 2’s history construction matches Theorem 1 (either both should use ^{\setminus a_{obs}} or the theorem should be revised to reflect the implementation).
- Redesign the evaluation protocol to report error against Monte Carlo estimates of the true conditional expectation, not single-sample labels.
- Add a fully continuous-time simulator (or at least one with genuinely continuous-time treatment intensities *and* observation times) to validate the core capability.
- Tune the FQE baseline comparably to EDQ (target-network ablation, learning-rate search) to ensure observed gaps reflect methodological differences rather than implementation artifacts.

## Score and Decision

**Score: 4.5 (Reject)**

**Calibration reasoning:**
- *High anchor:* `37EXtKCOkn` (avg 7.50, Spotlight) — A technically sound paper with comprehensive experiments and clear methodology. EDQ falls well below this: its theory has internal inconsistencies and its experiments do not validate the core estimand.
- *Medium anchor:* `iMRhuFS0Uz` (avg 5.25, Withdrawn/Reject) — A clearly written paper with a sensible idea and standard benchmark experiments, but reviewers questioned the significance and consistency of improvements. EDQ is comparable in novelty but worse in experimental rigor due to single-sample evaluation and a discrete-time simulator that does not test the claimed capability.
- *Low anchor:* `7AiPfnM73h` (avg 4.00, Reject) — Had well-written theory but poor empirical results and limited improvement over baselines. EDQ has more novelty than this paper, but shares the problem of unconvincing baseline comparisons and theoretical derivations that reviewers found difficult to trust. `or8wkKoBP4` (avg 4.00, Reject) had mathematical mistakes and no experiments; EDQ is stronger because it has some empirical validation, but its theoretical inconsistencies are in the same severity class.

EDQ is positioned between the low and medium bands: it presents a genuinely important and elegant idea, but the submitted version contains significant inconsistencies between its formal statements and algorithm, and the experiments do not robustly demonstrate accurate causal effect estimation. The paper would benefit from a thorough revision of the theoretical definitions and a redesigned experimental protocol before resubmission.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>