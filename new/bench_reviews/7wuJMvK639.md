Now let me read the calibration papers to anchor the score.Now let me read the calibration papers more carefully for scoring context.Now I have all the information needed to write the final consolidated review.

---

## Summary
Puppeteer is a hierarchical world model for visual whole-body humanoid control of a simulated 56-DoF humanoid. The method decouples control into a low-level tracking agent (pretrained on 836 MoCap clips via RL using TD-MPC2) and a high-level puppeteering agent (trained on downstream tasks using visual observations, outputting geometric end-effector commands). The paper introduces a new 8-task benchmark, a user study demonstrating strong preference for natural motions over flat RL baselines, and ablations supporting each design choice.

---

## Strengths

- **Clean and practical hierarchical decomposition.** A single 5M-parameter tracking world model is pretrained once and reused across all downstream tasks, in contrast to MoCapAct's ~2,600 per-clip policies. The geometric command interface (end-effector positions) is interpretable and task-agnostic — a real design win.

- **Compelling naturalness evidence via user study.** 97.8% preference from 51 participants over TD-MPC2 is a strong, hard-to-dismiss result. The qualitative examples (rolling up stairs vs. walking) vividly illustrate what end-to-end reward optimization sacrifices in terms of human-likeness, and the paper's framing of this as "reward hacking" is well-motivated.

- **Thorough and informative ablations.** The experiments isolate data mixture (offline vs. online vs. both), number of MoCap clips, planning level (neither / high only / low only / both), and high-level pretraining. The finding that planning at both levels is critical (43.0 vs. 1.6 normalized score) is striking and directly supports the modeling choices.

- **Impressive zero-shot generalization.** Figure 9 shows that Puppeteer maintains high performance on gap lengths 3× beyond the training range, while TD-MPC2 degrades sharply. This is a genuine empirical contribution suggesting the hierarchy provides real robustness benefits.

- **New benchmark and open-source release.** The 8-task suite (5 visual, 3 proprioceptive) with a highly expressive 56-DoF humanoid fills a gap in the community. Code, model checkpoints, and environments are publicly released.

---

## Weaknesses

### Fatal
*None that invalidate the core claim.*

### Major

- **The "no reward design" framing is incorrect as stated, and this is a headline claim.** The abstract reads: "without any simplifying assumptions, reward design, or skill primitives." Yet Section 3.1 explicitly states: "We label all transitions using the reward function from Hasenclever et al. (2020)" for the low-level tracker, and Section 4.1 specifies that all 5 visual tasks use "a reward function that is proportional to the linear forward velocity." The paper does use reward design — at both levels. What the paper arguably means is "no task-specific shaped reward engineering" or "no adversarial style reward," and that is a weaker but still meaningful claim. As written, however, this is a central advertised contribution that is directly contradicted by the method description. The authors should correct this throughout.

- **Naturalness evidence is too narrow to support the breadth of the central claim.** The user study aggregates across tasks but compares only against TD-MPC2 — not against any method designed for natural motion (e.g., adversarial imitation, AMP-style). The quantitative naturalness proxies in Table 1 (episode length, torso height) are reported only on the *gaps* task, yet the paper generalizes conclusions across the suite. The chosen proxies are weak correlates of human-likeness and could equally reflect gait selection or robustness differences. Showing per-task user study results in the main paper and widening the naturalness proxy to at least two tasks would substantially strengthen this core claim.

- **The claim "several orders of magnitude less interactions" is inaccurate.** MoCapAct uses ~150M environment steps; Puppeteer uses ~3M for downstream tasks. This is approximately 50×, which is roughly 1.7 orders of magnitude — not "several orders of magnitude." This overclaim should be corrected to a factual statement (e.g., "~50× fewer interactions").

### Minor

- **No baseline designed for natural motion is compared against.** The paper frames naturalness as its primary advantage over flat RL, yet the only naturalness comparison is against TD-MPC2, which is not designed for naturalism and "rolls up stairs." A comparison against at least one motion-prior method (e.g., AMP-style discriminator added to TD-MPC2) — even on the three non-visual proprioceptive tasks where such methods are directly applicable — would make the naturalism claim far more credible and grounded. The hierarchical baselines (SAC/DreamerV3 + low-level TD-MPC2) could have included one that uses an adversarial style reward at the high level.

- **k=1 means the architecture is hierarchical in action space only, not in time.** The paper describes k as a hyperparameter "that allows us to trade strong motion prior (large k) for control granularity (small k)," but all experiments use k=1. No ablation explores k>1. Without this, the temporal abstraction framing in the hierarchical RL literature does not apply here, and the hierarchy reduces to action-space factorization. The related work section does acknowledge "our method does not rely on...temporal abstraction," but this tension between the hierarchical framing and k=1 should be discussed more directly.

- **The termination-handling contribution (Section 3.3) is under-specified for a claimed novel methodological contribution.** "We maintain a cumulative weighting (discount) of termination probabilities when rolling out the model (capped at 0)" is described narratively without a formal equation. Given that the paper identifies this as novel and shows (in Section 4.1) that TD-MPC2 degenerates without it, a precise mathematical specification would be appropriate.

- **The ablation does not isolate the MoCap prior from the hierarchical structure.** The paper ablates planning levels and data mixtures but does not include a jointly-trained or scratch-trained hierarchical baseline. Without this, it is impossible to determine how much of the naturalism benefit comes from the MoCap-informed end-effector command space versus from the hierarchical world model architecture per se.

### Trivial

- The 50/50 offline/online training ratio in the tracking stage is reported without ablation. The paper notes "we did not experiment with other ratios," which is fine, but a brief note on sensitivity would be helpful.

---

## Nice-to-Haves

- **Richer naturalness metrics.** Fréchet Motion Distance against MoCap references, foot skating ratio, joint jerk, or other standard metrics from the motion generation community would reduce reliance on a single user study and make the evaluation more self-contained and reproducible.

- **Per-task user study breakdown.** Reporting per-task preference scores (not just the aggregate 97.8%) would reveal whether the preference is consistent across tasks or driven primarily by the conspicuous *stairs* failure mode.

- **Evaluate temporal abstraction (k>1).** Testing k∈{1,2,5,10} would answer an open question implied by the architecture, illuminate the naturalness vs. task-control granularity trade-off, and genuinely strengthen the hierarchical RL contribution.

- **Including non-locomotion tasks.** The paper acknowledges this limitation; adding even one reaching or manipulation task would better demonstrate the reusability claim.

- **Command trajectory visualization.** Plotting the puppeteer's output end-effector commands over time alongside the actual joint trajectories would reveal whether commands are smooth and within the tracking agent's training distribution, providing important mechanistic insight.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic W3] "Benchmark comparison is not strong enough because MoCapAct/DeepMimic are excluded."** → Partially removed. The paper explicitly states these methods do not support visual observations and require ~50× more interactions. Comparing methods with mismatched modalities is not a valid baseline comparison. The kept version is the milder concern about AMP-style comparisons on non-visual tasks (Minor weakness above).

- **[Human Finder W2] Real-world validation is missing.** → Removed. This is a simulation RL paper; real-world deployment is outside its stated scope. Sim-to-real transfer is not standard for this subfield and would constitute a separate contribution.

- **[Human Finder W3] Computational cost / inference time.** → Removed. A 12-day pretraining time on a single consumer GPU is not unreasonable for a paper of this scope, and the paper is transparent about it. This is a reproducibility nitpick.

- **[Harsh Critic — Section 3.1 note about "major inductive bias" being underplayed]** → Removed as strawman. The paper explicitly discusses the end-effector command space as a design choice and provides ablations supporting it. The critics' framing that the paper "underplays" this is not supported by the text.

- **[Harsh Critic — baseline implementation of TD-MPC2 being "unfair"]** → Removed. The paper gives TD-MPC2 the same termination-awareness modification it gives Puppeteer, which is fair. Any remaining implementation gap is not established by the critic.

---

## Novel Insights

The most genuinely novel observation in this paper is the empirical demonstration that **action-space factorization via a MoCap-pretrained end-effector command interface implicitly shapes learned behavior toward human-like motions without any explicit naturalism reward.** The 97.8% user preference result and the stark qualitative failure of end-to-end RL (rolling up stairs) suggest that the choice of command space — not just the hierarchical architecture or planning algorithm — is doing significant work. This insight, that geometric inductive bias in the action interface can substitute for explicit style rewards, has practical implications for the design of hierarchical controllers in physical AI systems. The zero-shot generalization to 3× training-range gaps as an emergent property of the hierarchy is similarly novel and underexplored.

---

## Suggestions

1. **Replace "without any reward design" with an accurate, qualified claim** throughout (abstract, intro, conclusion). A defensible version: "without task-specific reward engineering beyond a simple forward velocity signal."
2. **Correct "several orders of magnitude"** to the actual number (~50×, or ~1.7 orders of magnitude).
3. **Add at least one naturalism-focused baseline** (e.g., TD-MPC2 + AMP discriminator) on the three non-visual tasks to validate that the naturalism benefit comes from the hierarchy and not simply from having a naturalness-oriented reward.
4. **Include per-task user study results** in the main paper or appendix to let readers verify the aggregate result is not driven by a single task.
5. **Provide a formal mathematical specification** of the soft termination truncation in Section 3.3, including the exact functional form of the cumulative discount.

---

## Score and Decision

**Calibration:**
- **H-GAP** (humanoid control, MoCapAct, MPC, 56-DoF, scores 8/8/6, accepted spotlight): Closest architectural cousin. No visual observations, no user study, offline only. Puppeteer adds visual control, online interaction, user study results, and benchmark — substantively stronger in scope, but with more overclaims and narrower task diversity.
- **Universal Humanoid Motion Representations** (scores 8/8/8, accepted spotlight): Broader task coverage, better motion evaluation, stronger baselines. Puppeteer is weaker on task diversity and naturalism metrics but stronger on the visual control axis and zero-shot generalization.
- **HuWo** (humanoid locomotion world model, scores 6/5/6/3, rejected): Thinner methodological contribution, missing ablations, weaker evaluation. Puppeteer is clearly stronger.
- **TD-MPC2** (scores 8/8/8/8, accepted spotlight): The backbone algorithm. Puppeteer is a meaningful hierarchical extension but narrower in scope and with genuine overclaims.

The paper sits between H-GAP and the full spotlight papers. Its core contributions are real — visual whole-body humanoid control, compelling user study, and strong generalization — but the overclaiming in framing ("no reward design," "several orders of magnitude") and the missing naturalism baselines prevent it from reaching spotlight quality. The Limited task scope (all locomotion) and weak naturalness metrics further temper enthusiasm. This is a solid poster-level accept: genuine contributions, real results, fixable presentation issues.

**Score: 6.5 — Marginal Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>