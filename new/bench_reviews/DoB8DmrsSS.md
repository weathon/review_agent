Now let me read more of the paper and then run calibration searches.Now I have sufficient information to write the final review. Let me synthesize everything.

---

## Summary
SHIFT (Stealthy History Aligned diFfusiOn aTtack) proposes a novel diffusion-based adversarial state perturbation attack on reinforcement learning agents that transcends traditional ℓp-norm constraints. The paper argues that ℓp-constrained attacks cannot alter the semantic meaning of states, which is why state-of-the-art diffusion-based defenses can neutralize them; SHIFT instead uses a history-conditioned diffusion model with classifier guidance (for target action steering) and autoencoder guidance (for realism) to generate semantically different, realistic, and temporally plausible perturbations. Experiments across four Atari environments with six defenses demonstrate that SHIFT significantly reduces agent rewards where all prior attacks fail.

---

## Strengths

- **Clear fundamental insight with direct visual evidence (Figure 1, Figure 2):** The paper concretely demonstrates that even at ε = 15/255 under ℓ∞, PGD cannot change the position of the Pong ball or paddles, while SHIFT can. Figure 2 further shows that the conditional diffusion model produces states with lower ℓ2 distance to true states than even small-budget PGD (ε = 3/255), directly supporting both the motivation and the realism objective.

- **Consistent empirical domination across six defenses and four environments (Table 1):** SHIFT reduces reward substantially against all six defense methods — from vanilla DQN to the most advanced diffusion-based defenses (DP-DQN: 680 → 2 on BankHeist; Diffusion History: 21 → 6 on Pong). Prior ℓp attacks fail completely against DP-DQN in the Freeway comparison (Figure 3a), clearly isolating the ℓp constraint as the bottleneck.

- **Theorem 1 — non-trivial theoretical contribution (Section 3.2.2):** The proof that classifier guidance and classifier-free guidance can be combined without interference in this RL setting — because ã_t and τ_{t-1} condition on disjoint quantities — is not obvious and is likely reusable beyond this specific attack. The paper correctly notes this only holds due to the Markovian structure of the RL policy.

- **Practical feasibility via EDM (Table 2):** The DDPM → EDM substitution yields a ~25× speedup (5 sec → 0.2 sec per perturbation) with essentially equal attack performance (manipulation rate 76.6% vs. 87.1% on Pong, reward impact unchanged), which is a meaningful contribution to making the attack practical in real-time settings.

- **Preliminary probing defense analysis (Figure 3b):** The monotonic recovery of reward as probing interval decreases provides a concrete, actionable direction for future defense work and demonstrates the authors' interest in both offense and defense.

---

## Weaknesses

### Fatal
None.

### Major

- **Partially circular static stealthiness metric:** The Reconstruction Error metric is the ℓ2 distance between a perturbed state and its autoencoder reconstruction, where the same autoencoder is explicitly used to guide the attack at every reverse diffusion step (Section 3.2.3, Algorithm 2). Reporting that SHIFT achieves the lowest reconstruction error in Figure 3a is therefore self-confirming by construction — the attack was gradient-descended to minimize this very quantity. The paper does use the Wasserstein distance as an independent dynamic stealthiness metric, and SHIFT also achieves the best Wasserstein distance (Figure 3a), which provides legitimate independent evidence of dynamic stealthiness. However, the static stealthiness claim via reconstruction error is circular and cannot be used as independent evidence. The paper presents both metrics as complementary evidence without flagging this circularity, which overstates the support for the "simultaneously stealthy and effective" claim.

- **High variance in key results against the strongest defenses:** Against DP-DQN in Pong, the result is 0.5 ± 11.4 (Table 1) over 10 runs, despite Pong rewards lying in [−21, +21]. This is consistent with a near-bimodal distribution where the attack either completely succeeds or completely fails. Similarly, Diffusion History in RoadRunner yields 1480 ± 788 vs. a no-attack baseline of 13,500 — a high variance result. These are precisely the hardest defenses, and the paper's central claim depends on them. Ten episodes is insufficient to characterize attack performance in this regime; the paper should report per-run results or at minimum a median alongside the mean and std, and use substantially more evaluation rollouts.

### Minor

- **Attack comparison restricted to a single environment (Freeway):** The head-to-head comparison of SHIFT against PGD, MinBest, PA-AD, PGD-TC, Blurred, and Shifting in Figure 3a is conducted only in Freeway. Given that Table 1 shows attack effectiveness varies substantially across games (e.g., DP-DQN achieves mean reward 0.5 in Pong vs. 14.6 in Freeway under SHIFT), restricting this comparison to Freeway limits the generality of the conclusions. Including at least one more game in Figure 3a would strengthen the comparison.

- **True history vs. perturbed history approximation gap (Section 3.2.1):** The paper acknowledges that the attack conditions on the true history τ_{t-1} rather than the projected perturbed history H_{t-1} required by Definition 4, and attributes this to computational expense. Definition 4 is precisely what guarantees undetectability from the victim's viewpoint — the implemented algorithm uses a different conditioning signal. The paper mentions this as a future direction but does not quantify how large a gap this creates in practice (i.e., how much history-alignment degrades from ideal). This is an acknowledged limitation but one that deserves at least a brief empirical characterization.

- **Thresholds δ and ω in Definitions 2 and 5 are never given concrete values in the main text:** These thresholds define what counts as "realistic" and "approximately history-aligned," and without concrete values the formal definitions cannot be applied to the experimental results as stated. The definitions serve as conceptual scaffolding rather than operationally verified criteria.

### Trivial
- Reconstruction Error metric appears in Table 1's four-metric protocol description but is only reported in Figure 3a for Freeway, not across all defense/environment combinations in the main table.

---

## Nice-to-Haves

- **Independent stealthiness evaluation:** Train a held-out anomaly detector (separate from the attack's AE) on clean data and report detection rates for SHIFT vs. ℓp attacks under matched reward-degradation levels. This would independently corroborate the stealthiness claim for the static metric.
- **Adaptive defense evaluation:** Test against a defender who uses its own separately trained diffusion model as an anomaly detector — the current defenses (DP-DQN, Diffusion History) were not designed with SHIFT-type attacks in mind. An adaptive defense that specifically anticipates semantic perturbations would stress-test the attack's real-world viability.
- **Broader Atari coverage:** Extending Table 1 from 4 to 8 games would better support the abstract's claim of "various Atari games."

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic W2 — "Asymmetric threat model comparison invalidates headline result":** The critic argues that SHIFT and ℓp attacks operate under different threat models and that comparing them is unfair. However, the paper's entire thesis IS that ℓp constraints are the fundamental limitation preventing semantic changes — removing the constraint is the contribution, not a confound. The comparison is intentionally asymmetric to prove the paper's point. This should be removed per the Hard Rule on comparisons that favor the baseline and not the author's method.

- **Harsh Critic — missing appendix proofs for Theorem 1:** The critic hints the proof may rely on unstated assumptions and that the main text doesn't fully justify the conditional independence property. The proof is in Appendix C, which is stripped by the parser. Per Hard Rules, complaints about missing appendix content are removed.

- **Harsh Critic — concrete values for δ and ω needed for falsifiability:** Elevated from removed to Trivial/Minor because it does affect interpretability in the main text, but the formal definitions themselves are conceptual tools rather than falsifiable experimental criteria. Kept as a minor note only.

- **Strength Finder S2 — "Dual stealthiness metrics (static and dynamic)":** The static metric (reconstruction error) is partially circular as noted in Major weaknesses. The Wasserstein distance is a genuine independent metric. The strength is therefore half-supported; the Wasserstein component is kept as a legitimate point but the reconstruction error component is removed as independent evidence of stealthiness.

---

## Novel Insights

The paper's most noteworthy insight is conceptual: the failure of ℓp-constrained attacks against diffusion-based defenses is not a failure of attack strength but a structural consequence of the ℓp constraint itself, which prevents semantic change. This reframing suggests that the relevant axis along which to measure adversarial perturbations in RL is semantic content (operationalized through state-space projections and history alignment), not pixel-space distance. Theorem 1's implication — that in Markovian settings with discrete actions, classifier guidance and classifier-free conditioning on history are orthogonal and can be trivially superposed — is technically clean and likely transferable to other sequential prediction settings where action-level and trajectory-level guidance are needed simultaneously.

---

## Suggestions

1. Replace or supplement the autoencoder reconstruction error as a stealthiness metric with an independent held-out detector not used in the attack pipeline (e.g., a separately trained one-class SVM or a separate AE trained on a disjoint data split). Clearly distinguish the in-loop guidance metric from the evaluation metric.
2. Increase evaluation rollouts (at minimum 30, preferably 50+) for high-variance game/defense pairs (DP-DQN/Pong, Diffusion History/RoadRunner) and report median and 25th/75th percentile alongside mean ± std.
3. Extend the Figure 3a comparison to at least one more environment to reduce selection concerns.
4. Add a brief ablation (even with proxy metrics) quantifying the true-history vs. perturbed-history conditioning gap.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Illusory Attacks (RL stealthy attacks, info-theoretic) | `/home/wg25r/review_agent/human_reviews/F5dhGCdyYh.md` | 7.33 (Spotlight) | Closest topical match. More rigorous: formal info-theoretic detectability bounds + human study. Stronger evaluation rigor than SHIFT. |
| Beyond Worst-case RL robustness | `/home/wg25r/review_agent/human_reviews/DFTHW0MyiW.md` | 7.00 (Spotlight) | RL robustness paper with theoretical grounding and empirical validation. Similar scope, stronger theory. |
| Black-box DRL adversarial manipulation | `/home/wg25r/review_agent/human_reviews/UhW2wA1pRV.md` | 5.50 (Reject) | RL adversarial attack paper with similar scope and ambition, rejected for mixed empirical results and limited novelty. |
| Adversarial training + purification combo | `/home/wg25r/review_agent/human_reviews/u7559ZMvwY.md` | 5.67 (Accept poster) | Borderline accept on adversarial robustness with limited scope. |
| Low-scoring adversarial detection paper | `/home/wg25r/review_agent/human_reviews/kz78RIVL7G.md` | 2.6 (Reject) | Overclaimed near-perfect results with insufficient validation — much weaker than SHIFT. |
| Low-scoring RL robustness | `/home/wg25r/review_agent/human_reviews/KncRpAnprQ.md` | 2.0 (Reject) | Unfair baselines + missing key comparisons — significantly weaker than SHIFT. |

**Reasoning:** SHIFT sits clearly above the low-scoring anchors (2.0–2.6), which suffer from fundamentally flawed evaluations or missing key components. Compared to the medium anchors (5.50–5.67), SHIFT has a stronger and more clearly novel conceptual contribution (beyond-lp semantics-aware attack, Theorem 1) and more comprehensive empirical coverage (6 defenses × 4 environments). However, compared to the spotlight anchors (7.0–7.33) — particularly Illusory Attacks, which addresses an almost identical problem with rigorous info-theoretic detectability bounds and human subject validation — SHIFT falls short due to the partially circular stealthiness metric, high variance in key results (Pong/DP-DQN: 0.5 ± 11.4 over 10 runs), and lack of an independent stealthiness evaluation. The paper makes a genuine contribution and the core empirical point (semantics-aware attacks defeat diffusion defenses while ℓp attacks fail) is well-supported, but evaluation rigor issues prevent confident acceptance. Score: **5.5**.

**Axes summary:**
- *Originality*: Good — going beyond ℓp for RL attacks via conditional diffusion is a genuine step, and Theorem 1 is non-trivial.
- *Importance of research question*: High — diffusion-based defenses are emerging as the strongest RL defense category; attacking them is important.
- *Claims well supported*: Moderate — the effectiveness claim is well-supported; the "stealthiness" claim is supported for the dynamic (Wasserstein) metric but circular for the static (AE reconstruction) metric.
- *Soundness of experiments*: Moderate — 4 games and 6 defenses is good coverage, but 10 evaluation runs is insufficient for high-variance outcomes.
- *Clarity of writing*: Good — the paper is organized and readable.
- *Value to research community*: Moderate to high — highlights a real blind spot in current RL defenses.

**Decision: Reject (borderline — revisions to evaluation rigor could flip this)**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>