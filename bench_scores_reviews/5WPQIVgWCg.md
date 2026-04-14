## Summary

SELECT is an oracle-based algorithmic template for satisficing regret minimization in bandit optimization. Its central contribution is replacing the *satisficing gap* Δ_S — which is zero in continuous-arm settings like concave and Lipschitz bandits, making all prior algorithms vacuous — with the *exceeding gap* Δ_S\* = r(X\*) − S, which is always positive in the realizable case. By combining oracle arm identification, forced sampling, and LCB-based round termination, SELECT achieves constant (T-independent) satisficing regret in the realizable case and matches the oracle's standard regret in the non-realizable case.

---

## Strengths

- **The Δ_S → Δ_S\* conceptual pivot is the paper's most important contribution.** For concave and Lipschitz bandits, Δ_S = 0 necessarily (Remark 4), meaning prior algorithms such as SAT-UCB (Michel et al., 2023) and Garivier et al. (2019) cannot yield constant satisficing regret in these settings at all. SELECT is the first to obtain constant satisficing regret for infinite-arm continuous settings, making this a meaningful qualitative advance rather than an incremental improvement.

- **Each component of SELECT has a clear, non-trivial mathematical justification.** Remark 2 precisely explains why oracle-based arm identification is necessary over uniform exploration (Step 1), why forced sampling is required before LCB testing — without it, the LCB can drop below S even for a satisficing arm due to insufficient pulls (Step 2) — and why LCB rather than UCB or empirical mean is necessary for avoiding 1/Δ_S dependence in round termination (Step 3). This level of component-wise justification is above average for the genre.

- **Graceful two-regime behavior in Theorem 1.** The bound min(O(C₁^{1/(1−α)} (1/Δ_S\*)^{α/(1−α)} · polylog), C₁T^α · polylog(T)) ensures SELECT automatically adapts to the oracle's performance when T is too small for constant regret to dominate. This is not a trivial design feature.

- **Lower bounds confirm the correct gap dependence.** Theorems 3 and 4 establish Ω(1/Δ) lower bounds for finite-armed and concave settings (with Δ = Δ_S\*), confirming that the inverse-gap dependence in Corollaries 1 and 2 is tight and not an artifact of the analysis.

---

## Weaknesses

- **Proofs reside in an external document, not the submission.** All proofs are deferred to "Appendix A/B of the full version (Feng et al., 2025)" — an external document not included in or linked from the submission. The main text provides only high-level proof sketches for Theorem 1 (via Propositions 1 and 2) and no sketch at all for Theorem 2. This makes independent verification of correctness impossible from the submitted paper alone. For ICLR, the supplementary material should be self-contained.

- **The algorithm requires knowledge of oracle regret parameters (α, β, C₁), with no discussion of robustness to misspecification.** The forcing schedule γ_i = 2^{−i(1−α)/α} is parametrized by the oracle's regret exponent α. While α is known analytically for standard oracles (UCB, Thompson sampling), it may be unavailable or unreliable for complex or learned oracles. The paper makes no mention of robustness to misspecification of α or whether a parameter-free variant is achievable, weakening the "plug-and-play" framing.

- **No lower bound for Lipschitz bandits, and the K factor in finite-armed bandits is unresolved.** Theorems 3–4 provide lower bounds only for finite-armed and concave settings. For Lipschitz bandits in d dimensions, Corollary 3 gives O(L^d / (Δ_S\*/2)^{d+1} · polylog) while no matching lower bound is given; it is unclear whether the (d+1) exponent on 1/Δ_S\* or the L^d factor is tight. Similarly, for finite-armed bandits, the upper bound is O(K/Δ_S\*) but the lower bound (from a 2-arm construction) is Ω(1/Δ_S\*), leaving the K factor unresolved.

- **Algorithm 1 lacks an explicit stopping condition.** The pseudocode takes T as input but contains no check on remaining budget. As written, Algorithm 1 will attempt to initiate a new round even when insufficient time remains to complete the forced sampling phase. A clean budget check (e.g., halt if remaining time < T_i for the next round) is necessary for a complete algorithmic specification.

- **Notation inconsistency between the main text and Algorithm 1.** Step 1 of the algorithm description introduces $\tilde{X}_i$ as the uniformly sampled arm, but Algorithm 1 (line 3 of the pseudocode) uses $\hat{X}_i$ throughout, including in Steps 2 and 3. This inconsistency is never reconciled and creates unnecessary confusion.

- **Non-realizable experiments show unexplained patterns that are not discussed.** In Figures 3b and 4b, SELECT substantially outperforms its own oracle (Convex ALG by ~20%, Uniform UCB by ~75%) in standard regret under non-realizability. Theorem 2 is only an upper bound, so this does not violate theory, but the mechanism is never discussed. More strikingly, in Figure 3b, the discretized SAT-UCB heuristic (~500) and SAT-UCB+ (~600) achieve markedly lower standard regret than SELECT (~800), which the paper does not acknowledge. The discussion simply states SELECT "can also reach good enough distribution," which is insufficient.

---

## Nice-to-Haves

- A regret-vs-Δ_S\* plot (varying S across several values) to empirically validate the 1/Δ_S\* dependence in Theorem 1. The current experiments fix S and vary T, which validates that regret is constant in T but does not verify the gap dependence.
- An ablation removing forced sampling (Step 2) to directly demonstrate Remark 2's claim that LCB tests fail without sufficient initial pulls. This would isolate the forced sampling contribution.
- A brief sensitivity analysis or heuristic for choosing α when the oracle's exact regret exponent is uncertain.
- Lipschitz bandit experiments in d > 2 to probe the curse-of-dimensionality scaling in Corollary 3.
- A direct empirical comparison with Garivier et al. (2019) for finite-armed bandits, since the theoretical relationship is stated as "incomparable" and numerical comparison would clarify practical tradeoffs.
- Discussion of whether SELECT can detect non-realizability (output a flag) during or after execution, which would be valuable in practical applications.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **[REMOVED] Criticism that the 1/4 probability bound in Proposition 2 is arbitrary.** This constant emerges from the proof calculation and is internally consistent. Asking for a "tightness" justification for intermediate lemma constants is not a substantive criticism.

- **[REMOVED] Criticism of the LCB constant factor of 4.** The factor arises from standard subgaussian tail bounds and its derivation in the appendix is expected. Demanding it be re-derived in the main text is a formatting nitpick.

- **[REMOVED] Criticism of oracle data discarding at round restart.** The paper is clear that SELECT restarts ALG each round; Remark 2 explains why this design is needed. Whether warm-starting would be more efficient is a question outside the paper's scope and does not affect the correctness of the stated bounds.

- **[REMOVED] Criticism about loose inventory management example.** The subgaussian noise assumption is a standard modeling choice throughout the bandit literature. Questioning its applicability to inventory management is scope creep into applied modeling, not a theoretical flaw.

- **[REMOVED] Demand for confidence intervals/error bars on the 1000-repetition experiments.** Reporting mean curves without error bars is standard practice in the bandit empirical literature, especially when the curves are smooth and repetition count is high.

- **[REMOVED] Framing SELECT outperforming its oracle in non-realizable cases as "contradicting Theorem 2."** Theorem 2 is an upper bound. SELECT being empirically better than its oracle does not contradict an upper bound. This criticism reflects a misreading of the theorem statement.

- **[REMOVED] Criticism that the abstract fails to mention Δ_S\* > 0 as a qualifier.** The abstract says "realizable case (i.e., a satisficing arm exists)" and Δ_S\* > 0 follows from this by definition. The clarification is immediate and the abstract is not misleading.

---

## Novel Insights

The paper's deepest insight — which the spark finder articulates most clearly — is that Δ_S and Δ_S\* measure fundamentally different things that happen to coincide in finite discrete settings but diverge in continuous ones: Δ_S measures the nearest non-satisficing arm's distance to the threshold (which collapses to zero in dense spaces), while Δ_S\* measures the best arm's margin above the threshold (which is preserved in the realizable case regardless of arm density). Prior satisficing algorithms relied on Δ_S because they identified non-satisficing arms via exclusion; SELECT circumvents this by instead identifying *candidate satisficing arms via reward maximization* (Step 1) and testing them statistically (Step 3). The LCB test is key: it terminates a round within 1 step in expectation for a non-satisficing arm (since the LCB will almost certainly be below S immediately), while a satisficing arm with reward above S + Ω(γ_i) will sustain the LCB above S indefinitely once γ_i ≪ Δ_S\*. This is precisely the mechanism by which Δ_S\* — rather than Δ_S — enters the bound: round stability depends on γ_i relative to Δ_S\*, and once γ_i < Δ_S\* the geometric success probability kicks in (Proposition 2). The lower bounds confirm this is not improvable in gap dependence.

---

## Suggestions

1. **Make the supplementary self-contained.** Move the proofs of Theorems 1 and 2 (and Propositions 1 and 2) into a submission-attached appendix rather than citing an external "full version." This is essential for reviewers and readers to verify correctness.

2. **Add a stopping rule to Algorithm 1.** Include an explicit check such as "if remaining time < T_i + t_i, terminate" at the start of each round. This makes the pseudocode a complete and implementable specification.

3. **Add a regret-vs-Δ_S\* experiment.** Fix T and vary S to change Δ_S\*, and plot the resulting constant regret. A log-log plot should reveal a slope of approximately α/(1−α) = 1 (for α = 1/2), directly validating the core theoretical prediction.

4. **Discuss or add a missing lower bound for Lipschitz bandits in d > 1.** State explicitly whether the (d+1) exponent on 1/Δ_S\* in Corollary 3 is conjectured to be tight and what the primary obstacle to proving it is.

5. **Acknowledge and explain the non-realizable experimental gap.** Add a paragraph in Section 6 explaining why SELECT outperforms its oracle in non-realizable settings (possibly: non-realizable rounds terminate faster via LCB, so SELECT effectively concentrates the oracle's budget on a shorter effective horizon). Also acknowledge that SAT-UCB/SAT-UCB+ heuristics achieve lower standard regret than SELECT in Figure 3b, even if this is expected to be an artifact of the specific instance.

6. **Discuss practical approaches for unknown α.** Even a brief discussion of monotone search over α, or a default conservative choice (e.g., α = 0.5), would substantially improve the paper's practical utility claim.