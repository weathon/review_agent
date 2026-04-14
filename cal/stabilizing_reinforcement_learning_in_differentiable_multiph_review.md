=== CALIBRATION EXAMPLE 30 ===

# Final Consolidated Review
## Summary

This paper introduces two complementary contributions: **SAPO** (Soft Analytic Policy Optimization), a maximum entropy first-order model-based RL algorithm that incorporates analytic gradients from differentiable simulation to maximize both expected return and policy entropy, and **Rewarped**, a parallel differentiable multiphysics simulation platform built on NVIDIA Warp that supports rigid bodies, articulations, and deformable materials (elastic, plasticine, fluid) in parallel GPU environments. The authors demonstrate that SAPO outperforms model-free (PPO, SAC) and first-order model-based (APG, SHAC) baselines across six locomotion and manipulation tasks—including four involving soft bodies—using the same number of environment steps.

---

## Strengths

- **Rewarped fills a genuine, well-documented platform gap.** Table 1 systematically shows that no existing parallel simulator simultaneously supports differentiable simulation and the full material spectrum (rigid, articulated, elastic, plasticine, fluid). This is not just a useful tool; it is a necessary prerequisite for the experiments and future work in this area. The MPM parallelization with CUDA-graph-based gradient checkpointing is a concrete and non-trivial engineering contribution.

- **SAPO's ablation directly isolates the entropy contribution.** Figure 3 and Table 3 show that removing entropy from returns (ablation b: w/o $\mathcal{H}_\pi$) drops HandFlip performance from 90 to 59 (+78.8% vs. +172.7% over SHAC), while removing only the soft value function (ablation a) yields a smaller drop. This is more diagnostic than most ablations in FO-MBRL papers, which rarely decompose policy entropy from critic targets. The compound ablation (c) further reveals that architectural improvements from SAC/CrossQ alone explain roughly half the total gain, with entropy regularization responsible for the other half.

- **Empirical breadth across heterogeneous dynamics.** Six tasks spanning purely rigid (AntRun, HandReorient), elastic (SoftJumper), plasticine (RollingFlat, HandFlip), and fluid (FluidMove) settings—with action dimensions ranging from $\mathbb{R}^3$ to $\mathbb{R}^{24}$ and 10 random seeds with 95% CIs throughout—provides considerably broader coverage than prior FO-MBRL work (SHAC, APG) which focused exclusively on rigid-body locomotion.

- **SAPO achieves large and consistent gains on deformable tasks.** The improvements on SoftJumper (853 → 1821, +113%), HandFlip (33 → 90, +173%), and RollingFlat (87 → 100) are substantial and replicated across seeds. Notably, SHAC and APG show near-zero or negative returns on HandReorient and RollingFlat achieves near-ceiling only for SAPO, suggesting a real behavioral difference rather than margin effects.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Missing AHAC (Georgiev et al., 2024) as a baseline.** AHAC is the most direct FO-MBRL competitor: it modifies SHAC specifically to handle gradient instability at contact discontinuities by adaptively truncating the horizon based on Jacobian norms or contact forces—exactly the problem that motivates SAPO's entropy regularization. The paper cites AHAC in related work but provides no explanation for its omission from experiments. Without this comparison, it is impossible to assess whether SAPO's gains over SHAC exceed or are comparable to what AHAC already achieves, and whether entropy is truly a superior stabilization mechanism compared to adaptive horizon adjustment.

- **No wall-clock or compute-cost comparison.** The paper's core motivation is overcoming the "several orders of magnitude slower" runtime of soft-body simulation through parallelism. Yet no wall-clock training times are reported for any method. MPM simulation with full reverse-mode autodifferentiation and gradient checkpointing is substantially more compute-intensive per environment step than non-differentiable simulation. Without quantifying this overhead, the claimed sample efficiency advantage (in environment steps) may not translate to actual time efficiency. For practitioners, the trade-off between analytic gradients and compute cost is the most practically relevant question the paper raises but never answers.

- **One-way rigid–deformable coupling limits physical fidelity.** Section 5 states that Rewarped "supports one-way coupling from kinematic articulated rigid bodies to MPM particles." This means the robot hand in HandFlip and the rolling pin in RollingFlat exert forces on deformables, but the deformable material does not push back on the robot. In physical reality, a hand flipping dough feels substantial resistance from the dough. With one-way coupling, the learned policy optimizes against a physically incorrect simplification, which may inflate measured performance in simulation and makes the sim-to-real gap larger than the limitations section acknowledges (which only mentions privileged particle observations, not mechanical coupling). This is important to state prominently for any reader considering real-world application.

### Minor

- **Ablations are confined to a single task (HandFlip).** The paper claims entropy regularization stabilizes FO-MBRL broadly, yet ablations are run only on one deformable manipulation task. SAPO also improves substantially on AntRun (a pure rigid-body task, 3621 → 4536), and it is unknown how much of that gain is from entropy versus architectural changes (design choices III–V). The paper acknowledges in Appendix F.3 that design choices III–V have minimal impact on DFlex rigid-body tasks, but the entropy contribution on those tasks is not ablated. A brief ablation on AntRun would significantly strengthen the generality claim.

- **HandReorient failure is unanalyzed.** SAPO achieves the highest return (221.7) on HandReorient but the authors acknowledge it "only catches the cube" and fails to reorient it. All first-order methods score near zero or negative. The paper does not investigate *why*: is it gradient explosion through hard contacts, an underspecified reward function (modified to remove non-differentiable terms), insufficient environment steps, or a fundamental limitation of entropy-based stabilization on rigid contact-rich tasks? Understanding this failure mode is as informative as reporting successes.

- **Algorithmic novelty is incremental and should be positioned more carefully.** The "main observation" (Eq. 19) is the substitution of the entropy-augmented $H$-step return into the FOBG formula, which follows directly from adding entropy terms to Eq. 1 and applying Eq. 5. The five design choices (Section 4.2) are largely transferred from SAC, CrossQ, and Ball et al. (2023). This combination is practically valuable and empirically effective, but the paper should be more direct that the algorithmic contribution is a motivated engineering synthesis rather than a theoretically novel derivation. Overstating novelty invites skepticism; framing it as "principled integration with careful component selection" would be more defensible.

- **No simulation throughput or scalability data for Rewarped.** The platform is described as "scalable and easy-to-use" but no throughput numbers (environments per second, memory usage per environment count, maximum parallel environments before OOM) are provided. Given that the core argument for Rewarped is enabling RL at scale on a commodity GPU, quantifying the actual scaling behavior is essential.

- **Entropy smoothing hypothesis is only indirectly validated.** The hypothesis that entropy regularization "smooths the optimization landscape" (cited from Ahmed et al., 2019) is the central mechanistic claim of the paper. The ablation shows entropy improves performance, but does not confirm *why* at a mechanistic level. It is consistent with the data but so are alternative explanations (e.g., entropy simply prevents premature policy collapse, or broadens the action distribution near contact surfaces). This is a gap in understanding rather than a gap in results.

### Tiny

- **Value function consistency under on-policy truncated training.** The paper uses the soft value function $V_\text{soft}$ for TD($\lambda$) bootstrapping (Eq. 20), but $V_\text{soft}$ as defined by the soft Bellman equation (Eq. 14–15) is consistent only under the optimal soft policy $\pi^*$. The on-policy, truncated-horizon training regime of SAPO does not guarantee this. While similar approximations are common in practical max-entropy RL, this is worth a brief acknowledgment.

- **FluidMove provides weak differentiation.** Return differences across all methods on FluidMove are very small (21.7–30.6 range), with SAPO's improvement being modest. This task appears near-trivially solvable by all methods and contributes little diagnostic value to the comparison. The authors should note whether this reflects task design or algorithm design.

- **Particle count sensitivity unanalyzed.** Using ~2500 particles per deformable environment is a design choice with no justification. Sensitivity of results to particle count is relevant for assessing physical realism and reproducibility.

---

## Nice-to-Haves

- **Wall-clock time and GPU-hour comparison** between SAPO and the best non-differentiable baseline (SAC, PPO) to quantify the compute trade-off for practitioners.
- **Ablation on optimization horizon $H$** — FO-MBRL methods are known to be sensitive to horizon length, and demonstrating that SAPO maintains stability across varying $H$ (compared to SHAC's instability) would directly validate the stabilization claim.
- **Gradient norm or entropy trajectory plots over training** to correlate entropy levels with stability improvements and provide mechanistic support for the smoothing hypothesis.
- **Analysis of entropy temperature sensitivity** (fixed vs. auto-tuned $\alpha$, sensitivity to target entropy $\bar{\mathcal{H}}$) to distinguish algorithmic robustness from hyperparameter sensitivity.
- **Basic physics fidelity validation of Rewarped** (e.g., energy conservation in elastic simulation, comparison to established non-differentiable MPM solvers on simple cases) to establish the simulator as a reliable research platform independent of RL performance metrics.
- **Side-by-side rollout visualizations comparing SAPO and SHAC** on deformable tasks to expose specific behavioral differences (e.g., local optima in SHAC vs. exploratory behavior in SAPO).

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **Claim that the abstract's "outperforms" is misleading (HandReorient):** SAPO achieves 221.7 vs. 5.9 (PPO) on HandReorient — it numerically dominates all baselines per Table 2. That the task is not fully solved does not invalidate the "outperforms" language. Removed.
- **Fairness concern about PPO/SAC baselines:** The critic suggests model-free baselines may be "artificially disadvantaged" by not being tuned for Rewarped. However, the paper uses implementations explicitly validated to scale with parallel simulation (Li et al., 2023b), and all methods use standardized architecture/hyperparameters (Appendix C). The comparison is symmetric. Removed.
- **Table 1 asterisks soft-pedaling competitors:** The footnotes correctly document temporary engineering gaps (API breaking changes, active development) rather than overstating permanent fundamental limitations. Noting these is accurate and transparent. Removed.
- **$\Delta\%$ computation critique:** The raw-return percentage changes in Table 3 are displayed alongside absolute values; the reader can assess meaningfulness directly. Removed as a nitpick.
- **Broader impact / energy cost criticism:** Too generic for this type of systems paper; demanding broader impact statements for GPU simulation is not standard practice in this community. Removed.
- **Requesting visual observation experiments in the main text:** The paper includes visual encoder support in Appendix B.1 and positions this as future work. Demanding main-text experiments on a capability the paper explicitly scopes as secondary is scope creep. Moved to nice-to-have.
- **Missing related works:** Per review instructions, not included as the reviewer cannot verify external sources.

---

## Novel Insights

The most underappreciated insight in this paper is the *asymmetric impact* of entropy regularization across task types, partially revealed by the ablation and Table 2 but not fully analyzed. On deformable manipulation tasks (HandFlip), entropy is the critical differentiator—removing it nearly halves the improvement over SHAC. Yet on rigid-body locomotion (AntRun), SAPO still improves substantially over SHAC even though Appendix F.3 shows design choices III–V alone have minimal effect on DFlex rigid-body tasks. This suggests SAPO's entropy term may interact differently with the MPM gradient landscape versus the rigid-body gradient landscape—a phenomenon that, if investigated, could yield principled guidance for when max-entropy objectives are most valuable in differentiable simulation. This is left entirely implicit in the current paper. Additionally, the observation that APG outperforms SHAC on deformable tasks while SHAC outperforms APG on rigid-body tasks (consistent across DaXBench and DFlex respectively) suggests a structural interaction between simulation type and FO-MBRL horizon structure that is noted but unexplained.

---

## Suggestions

1. **Add AHAC as a baseline or provide an explicit justification for its exclusion** (e.g., incompatible contact force interface with Rewarped's MPM). This is the highest priority revision.
2. **Report wall-clock training times** for all methods, ideally as a secondary x-axis on training curves or as a table column — even a rough estimate (GPU-hours per 1M environment steps) would satisfy the compute-efficiency question the paper implicitly raises.
3. **Acknowledge one-way coupling explicitly in the Limitations section** alongside the particle observation issue. Quantify its expected impact or provide an informal argument for which tasks it most affects.
4. **Run at minimum one entropy ablation on a rigid-body task (AntRun or a DFlex task)** to determine whether the entropy contribution generalizes beyond deformable manipulation.
5. **Investigate the HandReorient failure** — at minimum, hypothesize whether the bottleneck is contact-induced gradient explosion (where AHAC's adaptive horizon might help), insufficient training budget, or reward function design. This would strengthen the paper's scientific contribution beyond benchmark reporting.
6. **Include basic Rewarped throughput metrics** (e.g., simulation steps/second at N=32, 64 parallel envs) to substantiate the "scalable" claim in the platform description.

---

**Axis Evaluations:**
- **Novelty:** Moderate. The platform contribution (Rewarped) is genuinely novel and fills a documented gap. The algorithmic contribution (SAPO) is an incremental but well-motivated synthesis whose novelty is primarily in application to differentiable deformable simulation rather than in the core formulation.
- **Technical soundness:** Good. The formulation is correct and the design choices are well-grounded in prior literature, though the theoretical justification for the soft value function under on-policy truncated training could be tightened.
- **Empirical support:** Good, with notable gaps. Six tasks, 10 seeds, and CIs are commendable. The missing AHAC baseline and single-task ablation are the primary weaknesses.
- **Significance:** High for the platform; moderate-to-high for the algorithm. Rewarped enables an entire class of previously infeasible experiments; SAPO's significance depends partly on how it compares to AHAC.
- **Clarity:** Good overall. The paper is well-organized and algorithmic descriptions are precise. The limitations section is honest but incomplete regarding physical fidelity.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
