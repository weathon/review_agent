Now let me run calibration searches to anchor the score appropriately.Now I have all the information needed to write the final consolidated review.

---

## Summary

This paper develops a constraint-based causal discovery framework for continuous-time stochastic processes modeled by acyclic SDEs. The core theoretical contribution is a Markov property (Proposition 3.1) for a novel "lifted dependence graph," enabling a sound and complete causal discovery algorithm (Theorem 3.2) that uniquely identifies the full dependence graph—not merely a Markov equivalence class—by exploiting time directionality. The framework handles partially observed data via an FCI-inspired extension. To make the theory practical, the authors propose a signature kernel CI test with a novel density-free consistency proof. The method is benchmarked against SCOTCH and PCMCI across multiple settings and also evaluated on a real-world pairs trading task.

---

## Strengths

- **Markov property and sound/complete algorithm (Proposition 3.1, Theorem 3.2):** The lifting of the dependence graph to incorporate past/future variables and the corresponding Markov property is a genuinely novel theoretical construction. Unlike classical PC-algorithm outputs, Algorithm 1 uniquely recovers the full DAG (not just an equivalence class) by exploiting temporal structure. The proof only requires a weaker "parent faithfulness" rather than full strong faithfulness.

- **Density-free consistency proof for the signature kernel CI test (Appendix A.14):** Existing consistency proofs for KCIPT/SDCIT require the existence of densities, which do not hold for path-valued random variables. The new proof filling this gap is non-trivial and of independent interest for the functional data and stochastic process communities.

- **Impossibility result for the cyclic setting (Appendix A.3):** Proving that full-graph constraint-based causal discovery is impossible for cyclic SDE models using arbitrary CI expressions of the form in Eq. (2) is an important negative result that clearly motivates and justifies the acyclicity assumption.

- **Partially observed setting (Figure 4, Appendix D):** The ability to handle latent confounders via an FCI-inspired extension is a meaningful advantage over score-based methods like SCOTCH, which are fundamentally challenged by unobserved variables. Figure 4 demonstrates that SCOTCH incorrectly infers a false adjacency in 88/100 runs, while the proposed method does so in only 8/100 runs.

- **Robustness to data missingness and out-of-model distributions (Figures 7, 8):** Empirical evidence showing that SigKer works under high data missingness (Figure 7) and on fractional Brownian motions outside the semi-martingale framework (Figure 8), where no existing CI tests are applicable, establishes the method's broad practical scope.

- **Strong empirical performance in linear, path-dependent, and nonlinear bivariate settings (Table 1, Table 2 for d ≤ 10):** SigKer achieves decisively lower SHD than all baselines (CCM, Granger, PCMCI, and SCOTCH) in these settings. For example, in path-dependence at n=200: SigKer 5±2 vs. next-best SCOTCH 23±12.

---

## Weaknesses

### Fatal
None.

### Major

- **Failure on diffusion dependence — contradicts a core advertised advantage.** Section 1 explicitly lists diffusion dependence (item (d)) as a key advantage over prior work: "causal dependence may stem from the diffusion, not captured by the means of observed paths." However, Table 1 shows SigKer achieving SHD×10² = 72±6 (n=200) and 63±5 (n=400) in this setting — while the best SCOTCH configuration achieves 25±13 (n=200) and 9±8 (n=400), an order of magnitude better. The paper's response — that SCOTCH is hyperparameter-sensitive and cannot be reliably cross-validated — is partially valid but does not explain why SigKer performs at nearly chance-level (CCM, which has no causal model, achieves 100±10). The paper itself acknowledges: "the constraint ⊥⁺_{s,h} may be weaker in this setting as the signature kernel may not efficiently detect diffusion dependence without drift." If the CI test cannot detect diffusion dependence, Theorem 3.2's guarantees are practically vacuous for this setting. This is the most significant empirical gap relative to stated scope.

- **Performance degrades substantially at high dimensions (d ≥ 20), contradicting the broad superiority claim.** Table 2 shows that SCOTCH (200, 2k) achieves SHD×10² = 370±174 at d=20 and 538±70 at d=50, while the best SigKer variant achieves 725±439 and 4593±93 — roughly 2× and 8× worse, respectively. Importantly, SCOTCH (200,2k) *consistently* outperforms SigKer at these dimensions across the table, making it difficult to dismiss as a lucky hyperparameter configuration. The paper's claim in the abstract and Section 4 summary that SigKer "broadly outperforms the state-of-the-art" is not supported for d ≥ 20.

### Minor

- **Path-dependence experiments use Markovian SDEs, not genuine history-dependent dynamics.** The main comparison in Eq. (5) and Table 1 implements "path-dependence" through a 3D Markovian SDE with an auxiliary variable X³ mediating the dependence — not via genuinely non-Markovian coefficients μ^k(X_{[0,t]}). The theoretical framework explicitly covers history-dependent drift/diffusion, and Figure 8 (appendix) shows fBM results for the unconditional test, but the causal discovery algorithm is never tested on a genuinely non-Markovian system (e.g., a delay SDE). This is a gap between stated scope and empirical demonstration.

- **Partially observed evaluation limited to a single example graph.** Figure 4 reports results for one fixed graph structure. A systematic evaluation over multiple graph sizes and confounder configurations would make the partial-observation claim substantially more convincing.

- **Hyperparameter selection concern for s.** While the paper frames SigKer as hyperparameter-free (in contrast to SCOTCH), the parameter s (set to 0.1·T based on Table 5) requires selection. It is not explicitly stated whether this value was chosen on held-out data or on the same experimental data used to produce Tables 1 and 2. If the latter, results for Algorithm 1 may be optimistically biased.

- **Finance experiment provides weak causal discovery evidence.** The pairs trading evaluation (Table 3) covers a 3-year window with 10 stocks in a specific post-crisis market regime. With no statistical significance testing and only one evaluation window, the large Sharpe ratio differences (1.500 vs. 0.242) could plausibly reflect noise rather than superior causal discovery. More fundamentally, the comparison pits cointegration (ADF) against causal direction (SigKer), testing different hypotheses.

### Trivial
None beyond the identified issues.

---

## Nice-to-Haves

- A power analysis for the *conditional* independence test (⊥⁺_{s,h} with non-empty K) analogous to Figure 3 (which only covers the unconditional test) would directly demonstrate the test's utility for the full causal discovery pipeline.
- A delay-SDE or fractional-coefficient benchmark would directly validate the theoretical claim about genuine path-dependence in the causal discovery algorithm.
- Precision/recall or F1 on edge recovery (in addition to SHD) would clarify whether high SHD at large d reflects over-prediction or under-prediction, and whether the method is adding spurious edges or missing true ones.
- An investigation into why the signature kernel CI test fails to detect diffusion-channel dependencies (power issue? Calibration issue?) would help characterize where future improvements are needed.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "no explicit scalability characterization in main text"** — The paper defers complexity to Appendix B.9, and the appendix exists in the original submission. The paper is evaluated up to d=50 empirically. Removed as an appendix-deferral complaint.

- **Harsh Critic: "bootstrapped SDCIT and HSIC chosen in preliminary experiments — sensitivity not reported"** — The model selection is clearly motivated and described; requesting a separate sensitivity analysis for the CI test variant is a nice-to-have, not a material weakness. Removed as minor scope creep.

- **Harsh Critic: "SCOTCH comparison dismissal not credible for SCOTCH (200, 2k) consistently winning at high d"** — Partially valid and retained as a Major weakness above, but the framing that the SCOTCH argument is "not credible" ignores that (200,2k) is one of four configurations and that (100,2k) performs dramatically worse at d=20 (1525 vs 370). Retained as a Major weakness but with a more measured framing.

- **Strength Finder: "Real-world validation in finance substantially outperforming ADF/Granger"** — Removed as an over-strong strength because the evaluation window is very limited (3 years, 10 stocks, no significance testing), and the comparison conflates different statistical hypotheses (cointegration vs. causal direction).

---

## Novel Insights

The most conceptually innovative aspect of this paper—not fully surfaced by either reviewer—is the impossibility result for cyclic SDEs (Appendix A.3). This negative result establishes a fundamental information-theoretic boundary: constraint-based methods using CI expressions of the form Eq. (2), no matter how flexible, cannot distinguish between the many possible cyclic graphs compatible with the same set of independencies. This simultaneously justifies the acyclicity assumption and clarifies that the paper's completeness result is the strongest possible within its class. For practitioners, the "lifted graph" construction (splitting each variable into past and future copies) is an elegant and reusable idea that could inspire time-directed constraint-based methods beyond the SDE setting.

---

## Suggestions

1. **Qualify the abstract's scope claims.** Replace "outperforming existing approaches in SDE models and beyond" with a claim scoped to the settings where the method genuinely works: drift- and path-dependent SDEs at moderate dimension (d ≤ 10), irregular sampling, and partially observed settings.

2. **Investigate the diffusion-dependence failure mechanism.** Is it insufficient kernel power to detect diffusion-channel variation, or is the test statistically well-calibrated but just low-powered? Testing the quadratic variation of paths instead of (or in addition to) the paths themselves may recover the diffusion dependence signal.

3. **Evaluate the partially observed setting more systematically.** Run at least 10–20 distinct graph structures with varying numbers of latent confounders and report aggregate SHD or edge-level metrics.

4. **Clarify s selection procedure.** State explicitly whether s = 0.1·T was selected on held-out data, or on the same samples used for reporting; if the latter, re-run with cross-validated s.

---

## Score and Decision

**Calibration anchors:**

- *SCOTCH (V1GM9xDvIY, avg 7.0, Accept poster):* The current paper is the direct follow-up/competitor. SCOTCH proposes a score-based variational SDE method; the current paper proposes a theoretically sounder constraint-based approach. The current paper has stronger theoretical guarantees (soundness/completeness, density-free consistency, impossibility result, partial observation handling) but fails empirically in SCOTCH's two best scenarios (diffusion dependence, high dimensions). Broadly comparable in practical utility.

- *IDOL (2efNHgYRvM, avg 8.0, Accept oral):* A high-quality causal discovery paper with tight identifiability proofs and strong empirical coverage. The current paper is not quite at this level — IDOL's experimental coverage is more complete relative to its claims.

- *KCI kernel test paper (GPcSYm89wK, avg 4.5, Reject):* Much weaker — only marginal improvements and no new theory. The current paper is substantially above this.

**Assessment:** The paper makes genuine theoretical and methodological contributions — the Markov property for lifted graphs, the density-free consistency proof, the impossibility result, and the FCI-based partial-observation extension are all non-trivial and publishable. Empirically, the method clearly dominates in linear/path-dependent/nonlinear settings at moderate dimension (d ≤ 10) and handles settings (partial observations, irregular sampling, fBM) where no competing method applies. The two Major weaknesses — diffusion dependence failure and high-d degradation — are real and not fully addressed, but they do not invalidate the other settings. The paper's overclaiming in the abstract/introduction should be corrected. Positioned above the SCOTCH anchor (7.0) due to richer theory but below IDOL (8.0) due to incomplete empirical coverage of advertised settings. The honest scope of this contribution is narrower than claimed but genuinely valuable.

**Final score: 7.0** — Accept (poster).

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>