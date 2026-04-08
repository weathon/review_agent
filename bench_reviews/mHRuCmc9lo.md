## Summary

This paper develops a minimax-optimal decision-making framework for forecasters that satisfy partial calibration guarantees (H-calibration) rather than full calibration. Using Lagrangian duality, the authors characterize the optimal robust decision rule in closed form (Theorem 3.1) and prove a striking "sharp transition" result: once the test class H contains the decision-calibration indicators, the robust policy collapses to the simple plug-in best response (Theorems 4.1–4.2), recovering the trustworthiness semantics of full calibration at a substantially weaker and more tractable condition.

## Strengths

- **Novel duality characterization of robust decision rules (Theorem 3.1):** The reduction of the minimax problem over policies to a finite-dimensional concave maximization over dual variables, with pointwise computation of the worst-case belief q*(v), provides a concrete and implementable recipe. This is a non-trivial structural result that cleanly separates the global (dual multiplier optimization) from the local (pointwise best-response) computation.

- **Decision calibration collapse result (Theorems 4.1, 4.2):** The finding that the minimax-optimal policy reduces to plug-in best response under decision calibration—and that this is stable under enrichment of H—is the paper's most significant insight. It upgrades decision calibration's previously known swap-regret guarantees to minimax optimality over *all* forecast-based policies, not just swap-type policies. The proof insight is clean: decision calibration makes the expected utility of a_BR invariant to the adversary's choice of q ∈ Q, so the adversary cannot degrade its performance.

- **Self-orthogonality from standard training (Proposition 4.4):** Identifying that first-order stationarity of squared-loss training with a linear head yields a free H-calibration guarantee is practically useful—it means practitioners can apply the robust rule without any post-hoc recalibration, leveraging structure that already exists in standard pipelines.

- **Simultaneous optimality across decision problems (Corollary 4.3):** The result that a single forecaster satisfying combined decision-calibration tests yields plug-in optimality for *all* downstream decision makers simultaneously is a strong practical upshot that goes beyond what prior work on decision calibration established.

## Weaknesses

- **The linearity assumption on utility (Assumption 2.1) is a fundamental restriction.** The proof of Theorem 4.1 critically uses linearity to establish invariance of a_BR's utility under adversarial tilting (Eq. 9: E[u(a, q(f(X)))] = u(a, E[q(f(X))])). For concave (risk-averse) utilities, Jensen's inequality breaks this equality, and the adversary could exploit curvature within decision regions R_a to degrade plug-in performance while respecting calibration constraints. The paper acknowledges this in Section 6 and cites linearization over bases (Gopalan et al., 2024b; Lu et al., 2025), but notes these bases "are not always low dimensional enough to be practical." This gap matters because the introduction explicitly motivates the work for healthcare and finance—domains where risk aversion is the norm. The paper should more prominently flag this as a scope limitation of the core "trustworthiness" claim, not just a future direction.

- **The empirical evaluation does not test the paper's central theoretical result.** The most important contribution—the collapse of the robust policy to plug-in best response under decision calibration—is never empirically demonstrated. All experiments use the self-orthogonality H = {h(v) = v}, which falls short of decision calibration. An experiment post-processing a forecaster to satisfy decision calibration and then verifying that the robust rule matches plug-in best response would directly validate Theorem 4.1 and is an obvious missing piece.

- **Experiments use only constructed adversaries, not real distribution shifts.** The adversarial evaluations in Table 1 are distributions mathematically derived to satisfy H-calibration constraints, which circularly validates the duality theory but does not demonstrate that the robustness helps against *naturally occurring* distribution shifts (e.g., temporal drift on Bike Sharing across years). Whether calibration-preserving adversaries align with real-world failure modes is the key empirical question left unanswered.

- **Limited experimental scope and missing baselines.** Only two 1D regression datasets with 3-action decision problems and a single MLP architecture are tested. There is no evaluation in the high-dimensional multiclass setting that motivates the paper (the intractability of full calibration in high d), and no comparison to other robust decision-making methods (e.g., Wasserstein DRO, conformal prediction-based decisions). Without such baselines, it is unclear whether the calibration-specific robustness structure offers advantages over generic distributional robustness.

- **Gap between population-level self-orthogonality and finite-sample practice.** Proposition 4.4 assumes the model reaches a population-level first-order stationary point. In practice, training involves finite data, early stopping, and approximate SGD, so the self-orthogonality moments hold only approximately. While Appendix B provides epsilon-slack theory, no experiment or analysis connects the epsilon to sample size, network architecture, or training duration, leaving it unclear how large the violation might be in practice and how it scales.

## Nice-to-Haves

- Evaluation under authentic temporal or domain shifts (e.g., testing Bike Sharing on held-out years) to probe whether calibration-preserving adversaries correlate with real distributional changes.
- Comparison against standard DRO or conformal prediction-based decision baselines to contextualize the calibration-specific robustness advantage.
- An experiment in a multiclass (d > 1) setting to demonstrate the framework in the regime where full calibration is actually intractable and the paper's partial calibration approach is most needed.
- Error bars or confidence intervals on the utility numbers in Table 1, particularly since some differences (e.g., 0.474 vs. 0.463) are small.
- Computational overhead analysis comparing the latency of solving the dual + pointwise minimization versus a single forward pass.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "The claim that decision calibration is 'tractable' conflates verifying vs. learning the guarantee."** The paper cites Noarov et al. (2023) for achieving decision calibration in practice. Whether enforcing it during training is expensive is a reasonable concern, but the paper explicitly references prior work showing how to accomplish this. This is more of a nuance than a flaw in the paper's claims.

- **Weakness: "The dual problem's convergence rate and conditioning in high dimensions are not discussed."** The paper describes a standard concave maximization with projected subgradient methods and provides the structure of the dual. For a theory paper, demanding a full convergence rate analysis is scope creep beyond the stated contributions. Moved to nice-to-have territory.

- **Weakness: "The pointwise computation of q*(v) may be prohibitive in latency-sensitive systems."** This is speculative without evidence of actual runtime. The pointwise problem is a small convex program over [0,1]^d, which for moderate d and finite A is fast. Without measurements, this is not a demonstrated weakness.

- **Weakness: "Missing discussion of societal impact / potential under-provisioning in healthcare."** This is a generic responsibility demand not standard for a theoretical ML paper at ICLR. The paper studies decision rules given forecasts; it does not deploy systems.

- **Weakness: "Figure 2 is only schematic; a plot with actual data would be more convincing."** This is a formatting/visual nitpick. The schematic effectively communicates the conceptual point.

- **Weakness: "No comparison to top-label calibration or weaker variants of decision calibration."** This is asking the paper to expand its scope. The paper already studies the hierarchy via H-classes and identifies the sharp transition. Requesting analysis of every intermediate notion is scope creep.

## Novel Insights

The most striking insight emerging from the synthesis of these reviews is that the paper contains an **internal tension between its theoretical center of gravity and its empirical evaluation**. The theory's crown jewel—the decision calibration collapse—receives zero experimental validation, while the experiments exclusively test the weakest instantiation (self-orthogonality under 1D regression). This means the paper empirically demonstrates only the least surprising prediction of its framework (robustness helps under adversarial shifts consistent with weak guarantees) while leaving its most surprising and significant prediction (the sharp transition) as a purely theoretical claim. A single experiment with a decision-calibrated forecaster would have closed this gap and dramatically strengthened the paper. Additionally, the self-orthogonality result creates an interesting practical asymmetry: it gives practitioners robustness "for free" from standard training, but this free robustness is precisely the regime where it is least needed (1D, low-stakes regression), while the high-dimensional, high-stakes settings that motivate the paper require explicit decision-calibration enforcement—exactly the regime left untested.

## Suggestions

- Add one experiment post-processing a forecaster to satisfy decision calibration (e.g., via the batch multicalibration algorithm from prior work) on a multiclass task, and show that the robust rule matches plug-in best response, directly validating Theorem 4.1.
- Evaluate under a real temporal or covariate shift (e.g., train Bike Sharing on 2011–2012, test on 2012–2013 held-out data) to establish whether the robustness helps beyond mathematically constructed adversaries.
- Include at least one baseline from the DRO or conformal prediction literature to contextualize whether the calibration-structured ambiguity set offers practical advantages over generic distributional uncertainty sets.