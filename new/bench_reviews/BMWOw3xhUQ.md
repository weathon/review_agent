Now let me search for calibration papers to anchor my scoring.Now I have enough information to write the final review. Let me synthesize everything.

---

## Summary

This paper proposes **GCReinSL (Goal-Conditioned Reinforced Supervised Learning)**, a framework that embeds Q-function maximization into Outcome-Conditioned Behavioral Cloning (OCBC) methods to endow supervised learning (SL) with trajectory stitching capability. The key idea is to use expectile regression to make the model predict in-distribution maximum Q-values (estimated via a CVAE), and then condition action generation on those predicted Q-values during inference. The paper also claims (via Corollary 1) that this framework is theoretically equivalent to goal data augmentation. Empirical results on Ghugare et al.'s stitching-focused Pointmaze/Antmaze datasets and D4RL Antmaze-v2 show consistent improvements over OCBC and goal data augmentation baselines.

---

## Strengths

- **Addresses an important known failure mode.** The lack of stitching in OCBC methods is a well-documented problem (Yang et al. 2023; Ghugare et al. 2024), and this paper proposes a principled alternative mechanism that does not require dynamic programming or explicit Bellman targets.

- **Compelling empirical results on D4RL Antmaze-v2.** Table 1 shows GCReinSL achieves a total score of 306.4 vs. 174.8 for Reinformer (the best prior SL method) — a nearly 75% relative improvement. On medium-play and medium-diverse, GCReinSL improves from ≈0.7/0.5 (prior SL) to 49.0/51.7, dramatically closing the gap to TD methods. This is among the strongest results of any SL method in this class.

- **Consistent improvements over OCBC and data augmentation baselines.** On Pointmaze datasets (Fig. 4), GCReinSL outperforms OCBC, SGDA, and TGDA across all six (dataset × model) configurations. On Antmaze stitching datasets (Fig. 5), results are similarly consistent.

- **Architecture-agnostic framework.** The method is demonstrated on both a transformer-based (DT) and MLP-based (RvS) model, validating that Q-conditioned maximization is a portable paradigm rather than a trick tied to one backbone.

- **Motivating example is effective.** The maze example in §4.1 clearly illustrates why naive OCBC fails and why naive Q=1 initialization creates OOD problems, providing good intuition for the proposed approach.

---

## Weaknesses

### Fatal
*None identified that completely invalidates the empirical contribution.*

### Major

- **Theorem 4.1 characterizes the wrong quantity for decision-making.** The theorem states `lim_{m→1} Q^m(SG) = Q_max` where `Q_max = max_{s,a,g} Q(s,a,g)` — the *global* maximum Q in the entire dataset. For action selection, the relevant quantity is the *conditional* maximum over actions (or trajectories) given the current (s,g). A theorem about convergence to the dataset-wide scalar maximum cannot directly justify the per-step inference procedure in Eqs. (12)–(13), which assumes the predicted Q̂(s,g) guides contextually appropriate action selection. In practice, the neural network is trained with (s,g) inputs and may learn something more sensible, but the theorem as stated does not characterize this. The stated `Q_max = max_{s,a,g} Q(s,a,g)` would — if taken literally — push every (s,g) to predict the same scalar, which would destroy the signal needed for stitching. The paper's empirical success suggests the network doesn't actually collapse this way, but the theorem does not explain *why*. This is the paper's primary theoretical gap.

- **Contradiction between Section 5.3 text and data.** The text in §5.3 states: *"However, in some datasets such as Antmaze-Medium, GCReinSL is inferior to advanced TGDA method."* The actual Fig. 5 data shows GCReinSL scoring 0.28 for both DT and RvS on Antmaze-Medium, while TGDA scores 0.15 (DT) and 0.25 (RvS) — GCReinSL *outperforms* TGDA in both cases. This factual error in the analysis section is a notable credibility concern.

- **Limited experimental scope — only maze environments.** The method is evaluated exclusively on Pointmaze and Antmaze datasets. Whether the CVAE-based Q-estimation and Q-conditioned maximization generalize to tasks with higher-dimensional action spaces, image goals, or non-navigation task structures (e.g., manipulation) is completely unknown. This limits the confidence in the method's generality.

### Minor

- **Hyperparameter sensitivity without principled guidance.** The ablation in Fig. 6 (right) reveals that for DT, performance is sensitive to the expectile parameter m, and very high m can hurt performance — contradicting the theoretical prediction that m→1 should improve results. The explanation of "overfitting to excessively large Q-function values" is qualitative and inadequate. This tension between theory (monotone improvement) and practice (degradation at high m) is unresolved and environment-dependent (DT vs. RvS behave differently). Practitioners have no principled way to choose m.

- **Ablation does not test key conceptual choices.** The ablations in §5.5 only sweep L (importance sampling samples) and m (expectile parameter). Missing is any comparison against: (i) using a simpler non-probabilistic Q estimator (e.g., direct return regression) instead of the CVAE, (ii) the same architecture with a fixed/random Q conditioning value, or (iii) expectile regression with a simpler Q target. Without these controls, the gain could come from added conditioning capacity rather than the specific proposed mechanism.

- **No validation that CVAE Q-estimates are accurate.** The entire training pipeline depends on the CVAE producing meaningful estimates of p^+_π(g|s,a) (the discounted goal-reaching probability). No evaluation of estimation quality is provided. If the CVAE produces poor estimates — especially likely for larger Antmaze tasks (consistent with the very low 0.12 / 0.02 success rates on Antmaze-Large) — then the expectile target is noise, undermining the theoretical justification.

- **Gap with TD methods remains large.** Despite the paper claiming to "significantly close the gap with TD learning," GCReinSL's total of 306.4 on Antmaze-v2 remains well below IQL (432.0) and CQL (371.2). The paper appropriately acknowledges this in the conclusion, but the abstract/introduction language overstates the result.

### Trivial

- The label `p_π^π(g|s,a)` in §4.2 appears to be a typo/inconsistency (should likely be `p_π^+(g|s,a)` per §3.1).

---

## Nice-to-Haves

- **Evaluate on non-maze offline tasks.** Kitchen, AntPush, or manipulation environments would substantially strengthen generality claims.
- **Visualize Q-value predictions on the motivating maze.** Showing that GCReinSL actually produces Q≈0 at s₀ and Q≈1 near the stitching point (as described in Fig. 1) would directly validate the core mechanism.
- **Validate VAE Q-estimate quality.** A correlation plot between CVAE-estimated Q and empirical returns would ground the framework's theoretical foundations.
- **Compare with SL+TD hybrids** (e.g., Doctor / IQL-conditioned policies) to clarify where GCReinSL sits in the broader SL-TD design space.
- **Discuss the two different Antmaze setups.** The gap between Fig. 5 (Ghugare's Antmaze, ~10-28% success) and Table 1 (D4RL Antmaze-v2, up to 80%+) is confusing without explanation; these are different datasets and should be explicitly distinguished.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**From the Harsh Critic:**

1. **"CVAE does not estimate Q^π for the learned policy"** — *Removed.* The paper builds directly on Eysenbach et al. (2022b)'s Theorem 3.1 (restated as Theorem 3.1 here), which equates Q^π(s,a,g) with the discounted occupancy probability p^+_π(g|s,a). Using a CVAE to estimate this probability from offline trajectories is the standard approach in contrastive/offline GCRL (e.g., GCBC, Quasimetric RL). Criticizing this as "conflating density estimation with policy-specific reachability" ignores that this is an explicitly established theoretical identity the paper is building on.

2. **"§4.1 claim that OCBC can follow τ₂ once Q̂_t=1 assumes the policy has already learned the right conditional action"** — *Removed.* §4.1 is a motivating illustration, not a formal proof. The claim is about what *would* happen if the model had the capability to predict Q=1 at the stitching state — precisely the capability the paper proposes to add. This is a strawman reading.

3. **"Corollary 1 is too strong / assumptions are not explicit in main text"** — *Removed.* The paper states the proof is in Appendix A.3. Per the review rules, we do not challenge the existence of appendix proofs without reading them. The harsh critic demands the assumptions appear in the main body, which is a formatting/style nitpick.

4. **"Fig. 4 does not support the caption that GCReinSL improves in all tasks"** — *Removed, factually wrong.* Checking Table (Fig. 4): GCReinSL achieves the highest value across all 6 (dataset × model) configurations. DT Umaze: 0.50 > TGDA 0.20. DT Medium: 0.70 > TGDA 0.60. DT Large: 0.35 > TGDA 0.25. RvS Umaze: 1.00 > TGDA 0.85. RvS Medium: 0.50 vs OCBC 0.45 (GCReinSL is highest). RvS Large: 0.35 > TGDA 0.15. The caption claim is correct.

5. **"Cross-paper comparison in Table 1 weakens superiority claims"** — *Removed as nitpick.* Cross-paper comparison of established baselines (CQL, IQL, DT, EDT, Reinformer) is standard practice in offline RL papers, and the Reinformer paper's numbers are publicly verifiable.

6. **"Inference complexity is significantly higher"** — *Removed.* Two forward passes per step (predict Q, then predict a) is at most 2× overhead. This is not "significantly higher" and is common in actor-critic-style inference pipelines.

---

## Novel Insights

The most genuinely novel observation from the reviews — not present in the paper itself — is the tension between the global-max formulation of Theorem 4.1 and the practical behavior of neural networks under this loss. If Theorem 4.1 truly causes Q̂ to converge to a single global scalar, the model would lose the ability to distinguish states, and stitching would fail. Yet the method works empirically. This suggests the expectile loss, when applied to a neural function approximator trained on (s,g) → Q̂ mappings, behaves as an *implicit conditional* maximization (each (s,g) is pushed toward the max Q in its neighborhood of the training data), not a literal global collapse. This is a meaningful distinction that the paper's theorem doesn't capture, and characterizing this gap more carefully (e.g., connecting to conditional expectile estimation with neural networks) would substantially strengthen the theoretical narrative.

---

## Suggestions

1. **Fix Theorem 4.1** to characterize what expectile regression actually achieves for a neural function approximator: a state-goal-conditional maximum over Q values within the data distribution near (s,g), not the dataset-wide global max. The current statement `Q_max = max_{s,a,g} Q(s,a,g)` is not the right quantity and does not justify the inference procedure.
2. **Correct the Section 5.3 text** — GCReinSL outperforms TGDA on Antmaze-Medium (both DT and RvS); the current text claims the opposite.
3. **Add ablation on Q-estimator type** — compare CVAE-based Q-estimation against a simpler return-regression baseline to isolate whether the probabilistic formulation is necessary.
4. **Clarify the two Antmaze dataset families** with a brief explanatory note to help readers interpret the large score discrepancy between Fig. 5 and Table 1.
5. **Provide a principled strategy for choosing m** (e.g., validation on a held-out split, or correlation with dataset characteristics) rather than reporting that both m=0.9 and m=0.999 can work or fail depending on the environment.

---

## Score and Decision

**Calibration:**

- **qg5JENs0N4** (*Closing the Gap between TD Learning and SL*, accepted poster, 6/5/5/6 = avg 5.5): The TGDA paper this work directly improves upon. That paper was accepted based on strong theoretical framing and empirical validation on stitching datasets. GCReinSL demonstrates consistent outperformance over TGDA, which is a positive anchor.

- **BxLK1M1f8T** (*Doctor*, rejected, 6/3/5/6 = avg 5.0): A similar SL+TD hybrid that was rejected primarily because stitching evidence was weak and evaluation missed AntMaze. GCReinSL avoids both problems and is clearly stronger.

- **C9BA0T3xhq** (*EIQL*, rejected, 3/1/3/1): Superficially similar (expectile regression for offline RL) but far weaker — no stitching focus, poor theoretical grounding, weak experiments. GCReinSL is substantially above this.

**Assessment:** GCReinSL has stronger empirical results than BxLK1M1f8T and is in the same neighborhood as qg5JENs0N4 in experimental depth. However, it has a genuine theoretical weakness in Theorem 4.1 (global vs. conditional max), a factual error in the results section, and limited scope (maze only). These prevent a confident accept but do not reduce the paper to the level of weak papers. Positioned below qg5JENs0N4 (accepted, 5.5) due to the Theorem 4.1 gap and limited scope, and above BxLK1M1f8T (rejected) due to substantially better empirical validation.

**Originality:** The Q-conditioned maximization SL framework is novel and well-differentiated from prior work. Building on Eysenbach et al.'s Q-probability equivalence to design an expectile-based OCBC is a creative synthesis.
**Importance:** High — stitching in offline GCRL is an important open problem.
**Support for claims:** Moderate — empirical claims are well-supported; theoretical claims are partially supported (Corollary 1 may be sound but the mechanism for Theorem 4.1 is incorrect as stated).
**Soundness:** Moderate — the overall experimental design is sound but the core theorem has a gap.
**Clarity:** Adequate, with the notable exception of the text-data error in §5.3.
**Value to community:** Moderate-to-high given the strong D4RL Antmaze-v2 results.

**Final Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>