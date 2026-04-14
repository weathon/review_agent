## Summary
HyPER (Hybrid PDE Predictor with Reinforcement Learning) proposes a framework that uses a REINFORCE-based binary decision policy to selectively invoke a numerical PDE simulator as a "knowledge-guided corrector" during neural surrogate rollouts. The key practical motivation is eliminating the differentiability requirement that burdens existing simulator-in-the-loop approaches, enabling use of black-box legacy solvers. The method is evaluated on 2D Navier-Stokes (in-distribution, changing boundary conditions, noisy inputs) and subsurface flow (Richards equation) benchmarks, reporting substantial reductions in cumulative rollout MSE compared to surrogate-only baselines.

---

## Strengths

- **Removal of the differentiability constraint is a concrete and practically impactful contribution.** Most solver-in-the-loop approaches (Chen et al. 2018; Belbute-Peres et al. 2020; Um et al. 2020) require differentiable simulators as additional layers. HyPER treats the simulator as a pure black box that returns the next state, enabling integration of legacy Fortran/Julia/C codes and any production-grade solver. This broadens applicability significantly compared to prior art.

- **Learned policy demonstrably outperforms random scheduling in challenging regimes.** Table 4 shows that when boundary conditions change (80.5% win rate) or when bimodal noise is injected (up to 84.5% win rate), HyPER's learned policy consistently beats a random policy with the same simulator budget. This establishes that RL is learning something non-trivial in scenarios with temporally structured error growth.

- **The cost-aware formulation with the λ parameter provides explicit, user-controllable budget management.** Unlike fixed hybrid approaches, the design cleanly lets practitioners dial in the fraction of simulator calls, making the efficiency-accuracy tradeoff explicit and tunable rather than implicit in architecture design choices.

- **Demonstrated across two distinct PDEs, simulators, and trajectory lengths.** Using both a Python (ΦFlow) Navier-Stokes simulator and a Julia (DPFEHM) subsurface flow solver demonstrates genuine simulator agnosticism. Trajectories ranging from 20 to 100 steps and grid sizes from 50×50 to 64×64 span meaningfully different physical regimes.

---

## Weaknesses

- **Absence of a fixed-frequency baseline is a critical evaluation gap.** The most directly relevant ablation — calling the simulator at every k-th step deterministically (e.g., every 3rd step to match λ=0.3) — is absent from the experiments. Table 4 compares only against a *random* same-budget policy. In the noise-free in-distribution case, HyPER wins only 57% of trajectories over the random policy; a fixed-stride schedule (which has no variance) may well close this gap entirely. Without this comparison, the marginal value of the learned RL policy over a trivial rule is not established for the cleanest setting. The paper acknowledges the 57% figure but attributes it to "low variance in MSE" — this explanation needs to be validated, not asserted.

- **Subsurface flow baselines are not fine-tuned on the subsurface dataset.** Table 5 compares HyPER (explicitly trained with 200+200 subsurface trajectories) against UNet-P and FNO-P, which are described as pretrained models applied without fine-tuning to the subsurface domain. Since HyPER's RL policy is specifically adapted to the subsurface task, this comparison does not isolate the HyPER contribution from the benefit of task-specific adaptation. Fine-tuned surrogate baselines on the subsurface task should be included in this experiment.

- **The trajectory-level cost function makes the reward non-Markovian, but the MDP formulation does not acknowledge this.** Equation 6 defines C(a, λ, T) as the absolute deviation of the *entire trajectory's* simulator call fraction from λ. This is a trajectory-level penalty, not decomposable per-step. The MDP defined in Section 2.2 implicitly assumes a standard per-step reward, but REINFORCE in Eq. 7 is applied with R(a) as a trajectory return. The paper does not clarify how the cost term is handled — is it subtracted once at the episode end, or spread across steps? This deserves explicit treatment, as incorrect decomposition could compromise gradient signal quality.

- **The RL training stability of REINFORCE is uncharacterized.** REINFORCE is known for high variance and seed sensitivity. No training curves, variance across seeds, or sensitivity to RL-specific hyperparameters (learning rate, baseline estimation) are provided. For a venue emphasizing reproducibility, evidence that training converges reliably across seeds is needed.

- **λ = 0.3 is used throughout without ablation.** The hyperparameter λ fundamentally controls the accuracy-cost tradeoff and is the key design choice a practitioner must make. The paper's efficiency analysis (Figure 5) hints at this tradeoff but never evaluates whether λ = 0.3 is robust: does performance degrade sharply at λ = 0.1 or λ = 0.5? Without this, practitioners cannot calibrate λ for their own simulators without trial-and-error.

- **Abstract/conclusion statistics are aggregated over selectively chosen baselines.** The conclusion's "75.28% improvement for changing physical conditions" is the average of only the pretrained (not adapted) surrogates, UNet-P and FNO-P — averaging 71.36% and 79.20%. The improvement over task-adapted UNet and FNO (42.72% and 45.61%) is materially lower and arguably a fairer comparison. Reporting the simpler average without explaining the choice can mislead readers about the magnitude of the contribution.

- **No learned policy behavior analysis.** The paper visualizes where in trajectories the simulator is called (Figure 1, 4) but provides no systematic characterization of what state features trigger simulator invocations. Does the policy respond to high-gradient regions, energy injection events, or something more opaque? Absence of this analysis means the paper cannot explain *why* the policy works, only that it does.

---

## Nice-to-Haves

- **Uncertainty-based or residual-based reward proxy.** The current reward requires ground-truth next states during RL training. Replacing this with a proxy (e.g., ensemble disagreement, residual magnitude) would enable training in settings where ground truth is unavailable post-training and would also address the conceptual gap between training (oracle-guided) and inference (state-only) regimes.

- **Policy input ablation.** Ablating what input features are provided to π_θ (full field, coarse statistics, uncertainty estimates) would clarify what signal the policy is actually exploiting to anticipate rollout errors.

- **Break-even horizon analysis.** Identifying the minimum trajectory length at which HyPER becomes more efficient than calling the simulator at every step (or at fixed intervals) would define the practical operating regime more precisely.

- **Longer-horizon demonstration.** The 20-step Navier-Stokes and 100-step subsurface experiments are useful but climate/geology applications involve thousands of steps. Even a preliminary stability check at longer horizons would strengthen the significance claim.

- **RL training cost accounting.** The efficiency analysis (Figure 5) shows inference cost, but the simulator calls consumed during RL training are not accounted for in the overall efficiency picture.

---

## Removed Points

*These points were flagged and removed or substantially downgraded; treat them with caution.*

- **[REMOVED] Convection-term labeling inconsistency (Eq. 8).** The harsh critic claimed that Eq. 8 labels the term $-\mathbf{v} \cdot \nabla \mathbf{v}$ as "advection of concentration." This is incorrect: Eq. 8 is the momentum equation for velocity $\mathbf{v}$ and $-\mathbf{v} \cdot \nabla \mathbf{v}$ is the standard convection of velocity, which is correctly labeled. Eq. 11 separately models concentration advection. No inconsistency is present.

- **[REMOVED] The third contribution ("rigorous experimentation") is not a scientific contribution.** This is a style/formatting nitpick with no bearing on the technical content.

- **[REMOVED] Data split asymmetry (surrogate sees 400 trajectories vs. SUG's 800).** The paper explicitly states: "All SUG baselines are trained using 800 trajectories (same as HyPER RL plus surrogate dataset)." The total data budget is identical; HyPER simply splits it across the two-stage training process. The surrogate component of HyPER is deliberately trained on 400 to leave 400 for RL; the comparison is by design and the paper is transparent about this. The resulting HyPER surrogate still contributes to outperforming models trained on 800 trajectories, suggesting the RL training compensates effectively.

- **[REMOVED] No confidence intervals for large-scale benchmarks.** All tables average over 200 test trajectories. Per the review rules, demanding statistical tests for 200-trajectory benchmarks imposes standards not routinely required in this setting.

- **[WEAKENED → Removed] Oracle reward / training-inference mismatch as a fatal concern.** The RL policy $\pi_\theta(a|u_t)$ is conditioned only on the current state $u_t$ at inference, not on any ground-truth error signal. Ground truth is used during training to compute the reward — this is standard practice for model-free RL (the reward exists only during training). The policy generalizes by learning a state-to-action mapping. The authors could discuss this more explicitly in a limitations section, but it is not an undisclosed flaw.

- **[WEAKENED → Nice-to-Have] Scalability to 3D fine grids.** The paper targets 2D modest-resolution problems, and limitations of RL policies for high-dimensional state spaces are real. However, the paper's scope is clearly stated, and evaluating against unstated 3D scaling requirements is scope creep. Moved to nice-to-have.

- **[WEAKENED → Nice-to-Have] Relaxing C(a,λ,T) from deviation penalty to linear cost.** This is a legitimate design alternative but represents a methodological preference, not a flaw in the current design. Moved to nice-to-have.

---

## Novel Insights

The most practically underappreciated observation in the reviews is the asymmetric utility of the learned policy: in *temporally smooth, noise-free* settings, the random policy nearly matches HyPER (57% win rate), whereas in settings with *structured temporal disruptions* (boundary condition changes, bimodal noise), the RL policy yields win rates of 67–84.5%. This suggests that the primary value of learning a policy — rather than using a fixed schedule — is specifically in identifying *when* the physical trajectory is about to depart from the surrogate's training distribution. In smooth regimes, almost any schedule works because the surrogate degrades uniformly; in disrupted regimes, the timing of corrections matters enormously. This has implications for where RL-based hybrid scheduling adds genuine value versus where simpler heuristics suffice, and the paper would be stronger for making this distinction explicit.

---

## Suggestions

1. Add a deterministic fixed-interval baseline (call simulator every ⌊1/λ⌋ steps) to Table 4 to isolate the RL policy's contribution from mere simulator access.
2. Include fine-tuned UNet and FNO baselines in Table 5 (subsurface flow) to give a fair surrogate-agnosticism demonstration.
3. Report results from ≥3 random seeds for the RL training phase, with mean ± std on cumulative MSE, to establish REINFORCE stability.
4. Add an ablation table varying λ ∈ {0.1, 0.2, 0.3, 0.5} on the Navier-Stokes benchmark to characterize the cost-accuracy tradeoff curve.
5. Add a brief limitations section addressing: (a) the training-inference regime (ground truth used for reward only during training), (b) the trajectory-level non-Markovianity of the cost function C, and (c) scope of the current 2D experiments.
6. Clarify the aggregation methodology for the reported statistics: the "75.28%" figure in the conclusion should note that it averages over pretrained (not adapted) baselines only, or provide two separate averages (pretrained-only and adapted-surrogate).

---

**Assessment:** HyPER addresses a genuine and practically important bottleneck (non-differentiable simulator integration for rollout correction) with a technically sound RL formulation. The novelty is moderate but well-targeted; the idea of using REINFORCE to learn a binary switching policy between surrogate and black-box simulator is clean. Empirical support is convincing in challenging settings (OOD boundary conditions, structured noise) but weaker in the baseline in-distribution case where a 57% win rate over random raises questions about the RL contribution. The missing fixed-schedule ablation and the use of non-fine-tuned baselines in the subsurface experiment are the most substantive evaluation gaps. Significance for the SciML community is high if the method holds up under tighter ablation. As submitted, the evidence base is solid but incomplete.

MY FINAL SCORE: <pineapple>5.8</pineapple>