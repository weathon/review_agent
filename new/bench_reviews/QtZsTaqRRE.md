Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

The paper proposes AuxSS, a method that uses small quantities of expert demonstration states (no actions or rewards required) to form a dynamically updated auxiliary start state distribution guided by episode-length-based state safety, accelerating online RL in resettable simulators. The core idea is that states from which the current policy produces short episodes (low safety) are likely task-critical states that need more training visits, and the algorithm continuously updates a sampling distribution over demonstration states based on this signal.

## Strengths

- **Clean, well-motivated core idea connecting to Go-Explore framework.** Using demonstration states as auxiliary start states with safety-informed weighting is a natural and potentially impactful idea (Section 4, Algorithm 1-2). It extends Go-Explore by providing a principled, general mechanism for selecting exploration frontiers rather than relying on domain-specific heuristics or uniform random selection.

- **Strong ablation study isolating the safety-weighting mechanism (Figure 7).** Section 5.5 compares AuxSS against Ω-SS (static safety-based), showing matching early training trends and validating that episode length approximates the intended safety quantity. Section 5.6 compares against U-SS (static uniform) and GoalDist-SS (dynamic goal-distance), both of which dramatically underperform, demonstrating that not all auxiliary start state distributions are equally effective and that safety-inspired construction specifically matters.

- **Demonstrates practical advantage with dramatically less expert data (Figure 6).** AuxSS with only 0.5K transitions matches or exceeds the robustness and sample efficiency of HySAC and JSRL given 7.5K transitions (15× more data)—and those baselines additionally require expert actions and rewards. This is a clear practical benefit.

- **Dynamic update prevents robustness degradation (Section 5.5).** The comparison with static Ω-SS reveals that while Ω-SS initially learns faster (it uses the true safety distribution), its robustness degrades later in training because it cannot adapt. AuxSS's dynamic update prevents this degradation, providing evidence that adaptivity is important.

- **Scales to image-based state spaces and hard-exploration 3D navigation (Figure 4).** AuxSS is the only method to consistently solve the easier 3D navigation task and the only method to solve the harder variant from the original start state distribution.

- **Evaluation includes both in-distribution (p₀) and out-of-distribution (μ_OOD) metrics**, providing evidence that learned policies generalize beyond the training start state distribution.

## Weaknesses

### Fatal
None.

### Major

- **Abstract overclaims "matching or exceeding" SOTA "even when competing with algorithms with access to expert actions and rewards."** On MuJoCo (Figure 5), AuxSS clearly underperforms HySAC (which has expert actions/rewards) — the figure description places AuxSS and SAC at "intermediate performance" while "HySAC+AuxSS and HySAC achieve the highest rewards." On Lava Bridge (Figure 3), AuxSS alone underperforms IQL+JSRL and HySAC+AuxSS on both p₀ success and μ_OOD success. The claim is only cleanly supported on the 3D Navigation tasks (Figure 4). The paper's own Section 5.3 frames MuJoCo results as "comparable to HySAC," but the figures contradict this characterization. This overclaiming matters because it frames the contribution as more broadly applicable than the evidence supports — the strongest results consistently come from HySAC+AuxSS (which does use expert actions and rewards), not from AuxSS alone.

### Minor

- **Confusing experimental setup description in Section 5.1.** The text states "the number of offline demonstration transitions is set to 10 million, number of online learning steps is 300000, replay buffer size is 10000" while also stating "All hybrid methods have access to 500 transitions of expert demonstration data." The 10 million figure conflicts with 500 transitions and with the replay buffer size of 10,000. The x-axis extending to 3M timesteps also needs reconciliation with the 300K figure. While the key quantities are stated elsewhere (Section 5 setup: 500 transitions for Lava Bridge, 1000 for MuJoCo, max episode length 500), this paragraph is sufficiently confusing to hinder understanding of the experimental conditions.

- **Safety-proxy mechanism provides no benefit in dense-reward environments where termination is rare.** The paper acknowledges this in Section 5.3 ("early episode terminations cease rapidly, resulting in AuxSS becoming a uniform sampling distribution"), but frames it as acceptable rather than as a scope limitation of the core mechanism. The paper's key insight — safety-weighted auxiliary start states — degenerates to uniform sampling over demo states in any environment where survival is easy, which includes many practical RL problems. The paper provides no evidence for how the method behaves in environments with intermediate safety characteristics.

- **GoalDist-SS ablation, while dynamic, uses a curriculum-style schedule rather than feedback-driven dynamics.** GoalDist-SS updates via temperature scaling, not from training outcomes like value estimates or visitation counts. A dynamic baseline that updates based on training feedback (e.g., prioritizing states with high TD-error or low visitation) would more cleanly isolate whether the *safety* mechanism specifically matters versus whether any *feedback-driven* dynamic distribution would suffice. The existing ablations leave this distinction partially open.

### Trivial
- The definition of task-critical states C in Section 3 is informal and policy-dependent, which makes it impossible to compute directly. However, this is used as motivational intuition rather than in proofs, so the algorithm does not depend on its formalization.

## Nice-to-Haves

- Experiments in environments with intermediate safety characteristics (some but not all poor behavior causes termination) would test whether the method provides partial benefit or fails gracefully outside its sweet spot.
- Sensitivity analysis for the smoothing parameter σ² and weight threshold δ, which control the locality and minimum weight of the sampling distribution.
- Visualization of which demonstration states receive high/low sampling weight over training, to verify that AuxSS concentrates on meaningful task-critical states rather than trivially dangerous states.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Arbitrary state resetting is a more restrictive affordance than expert actions/rewards"** — The paper acknowledges the need for arbitrary resets as a limitation (Section 6). The claim that this is "more restrictive" is debatable (many simulators support this) and constitutes scope creep rather than a methodological flaw.

- **"Gaussian smoothing distance metric unspecified for image observations"** — Implementation detail that would not affect the validity of the results; the paper demonstrates the method works with image states in Figure 4.

- **"Harder 3D Navigation variant relegated to appendix"** — Presentation choice, not a substantive weakness.

- **"Missing dynamic non-safety-based distribution ablation"** — The critic claimed this ablation was entirely missing, but GoalDist-SS is explicitly described as "a dynamic distribution" in Section 5.6 that "exponentially weights states based on their distance from the task goal" with "temperature scaling...gradually rising over the course of training." The critic misread the paper on this point.

- **"15× fewer expert samples needs statistical validation"** — This is directly shown in Figure 6, which compares AuxSS@0.5K against methods given 7.5K transitions. The data supports the claim.

- **"Post-hoc and unverified explanation for MuJoCo results"** — The paper's explanation (task-critical states in expert data are under-visited from p₀) is plausible and consistent with the mechanism. Whether it's "post-hoc" is speculative criticism.

- **"Informal definition of C is circular and uncomputable"** — The concept is used as motivational intuition; the algorithm does not depend on computing C. This is not a flaw in the paper's methodology.

## Novel Insights

The most insightful observation from the review is the asymmetry between the paper's strongest and weakest results: the safety-weighted auxiliary start state mechanism is most effective precisely where it is hardest to validate its superiority over simpler alternatives. In sparse-reward environments (Lava Bridge, 3D Navigation), where early termination is the dominant failure mode and the safety proxy provides a strong signal, AuxSS shows clear advantages—but these are also environments where any reasonable auxiliary start state distribution (e.g., uniform over demo states) might help substantially simply by providing access to states beyond the initial distribution. In dense-reward environments (MuJoCo), where the safety signal vanishes and AuxSS degenerates to uniform sampling, the method still helps by providing diverse starting states—but cannot match methods with expert actions/rewards. This suggests the core practical contribution may be "resetting to demo states" rather than "safety-weighted resetting," a distinction the current ablations partially but not fully resolve.

## Suggestions

- Scale back the abstract claim to match the evidence: replace "matching or exceeding state-of-the-art performance... even when competing with algorithms with access to expert actions and rewards" with a more precise statement like "achieving competitive performance on sparse-reward tasks with only state information from demonstrations, and matching methods with expert actions/rewards on 3D navigation tasks."
- Clarify the experimental setup numbers in Section 5.1 — the 10 million, 300K, and 500 figures need reconciliation or clearer specification of what each refers to (e.g., total replay buffer capacity vs. expert data vs. gradient steps vs. environment steps).
- Add a feedback-driven dynamic baseline (e.g., prioritizing demonstration states by visitation count or TD-error) to isolate whether the *safety* mechanism specifically matters versus whether any *training-feedback-driven* dynamic distribution suffices.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Intelligent Go-Explore | apErWGzCAA | 7.00 | Similar Go-Explore extension theme; stronger empirical scope (more diverse tasks) but has prompt-engineering dependency weakness. Our paper has a simpler/cleaner algorithm but narrower evaluation scope. |
| IBRL | Ap344YqCcD | 5.50 | Similar hybrid RL with demonstrations; practical algorithm with good empirical support. Our paper has better ablations but overclaims more. IBRL was withdrawn/rejected at 5.50. |
| Explore-Go | X6W5eqhzDx | 4.67 | Similar start-state-distribution manipulation for RL; weaker baselines and more overclaiming. Our paper has much stronger ablations. |
| RISC | Nq45xeghcL | 6.75 | Reset-free RL with intelligent switching; novel mechanism with SOTA results but limited theory. Comparable novelty level to our paper, but RISC has stronger empirical results without overclaiming. |
| SAC-BC | 2nrn8LRpex | 2.50 | Minimal contribution offline RL. Our paper is clearly much stronger — novel algorithm, good ablations, multiple environments. |
| IRL Variational | VyWv7GSh5i | 2.75 | Flawed theoretical derivation. Our paper has no such fundamental flaw. |

The paper sits above IBRL (5.50, withdrawn) due to stronger ablations and a cleaner algorithmic contribution, but below RISC (6.75, accepted) due to overclaiming and weaker MuJoCo results. The overclaiming in the abstract is the main factor preventing a higher score — the core algorithm is sound and the ablations are informative, but the framing exceeds what the evidence supports. Compared to Explore-Go (4.67, rejected), this paper has substantially better empirical methodology and ablations, justifying a score well above that range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>