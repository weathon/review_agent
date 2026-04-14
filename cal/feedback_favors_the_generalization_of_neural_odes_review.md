=== CALIBRATION EXAMPLE 49 ===

# Final Consolidated Review
## Summary
The paper proposes **Feedback Neural Networks**, an architecture that augments Neural ODEs with a real-time feedback loop that corrects learned latent dynamics using the deviation between measured and predicted states. Two feedback forms are introduced: (1) a linear form (gain matrix **L**), for which exponential convergence to a bounded error set is proven; and (2) a nonlinear neural form trained via domain randomization while the base Neural ODE is frozen, yielding a two-degree-of-freedom (two-DOF) network that preserves nominal accuracy while gaining generalization. The approach is validated on real-hardware trajectory prediction of an irregular object and quadrotor model-predictive control (MPC) under substantial parametric uncertainties.

---

## Strengths

- **Two-DOF training decomposition with clear empirical support.** Freezing the base Neural ODE and training only the feedback module via domain randomization is a structurally elegant solution that—unlike naïve domain randomization—demonstrably preserves nominal accuracy (Figure 6 vs. Figure 5(c)). This is a specific, non-obvious insight that distinguishes this work from prior approaches that retrain the full network and suffer accuracy–robustness tradeoffs.

- **Substantial real-hardware MPC results against multiple baselines.** The quadrotor flight experiments (Section 5.2) compare six methods on a Lissajous trajectory with 37.6% mass uncertainty, 40% inertia uncertainty, drag coefficient perturbations, and additive disturbances simultaneously. FNN-MPC achieves RMSE 0.093 m vs. 0.151 m for AdapNN-MPC (the next-best adaptive method)—a ~38% improvement under a demanding multi-uncertainty regime. This is not a toy benchmark.

- **Gain decay strategy for multi-step prediction (Eq. 11) is a thoughtful and effective engineering contribution.** Figure 5(g) clearly shows the decay strategy addresses noise amplification in the cascaded prediction scheme. This detail, while simple, is practically critical and non-obvious.

- **Explicit grounding in classical control theory.** The paper correctly situates the linear feedback as an observer-style correction, cites Luenberger (1966), Kalman (1960), and ESO (Guo et al., 2020) as conceptual precursors, and frames the contribution as a principled adaptation of observer design to the neural ODE context. This grounding is intellectually honest.

---

## Weaknesses

### Fatal
None. The core claims are supported by the experiments.

### Major

- **Computational cost of the N-step cascade is unaddressed.** The multi-step prediction strategy requires N separate forward passes through the feedback neural network per prediction horizon (N=50 in experiments). For MPC, the prediction must be computed at each control timestep inside an optimization loop. The paper reports no inference times, no wall-clock MPC solve times, and no comparison of latency relative to Neural-MPC or AdapNN-MPC. For a paper positioning FNN-MPC as a deployable solution, this is a critical omission. If the cascade is 50× slower than a standard Neural ODE, real-time feasibility is in question.

- **The conclusion's "first time" feedback claim is inaccurate and should be corrected.** The conclusion states: *"we proposed to incorporate a feedback loop into the neural network structure for the first time, as far as we know."* The paper itself cites Cheng et al. (2019), O'Connell et al. (2022), and Richards et al. (2023) — all of which incorporate real-time state feedback into neural network-based controllers. The specific contribution is the **integration of an observer-style feedback loop into the latent dynamics of a Neural ODE**, not the first use of feedback in neural networks. This overclaim undermines the paper's credibility.

- **Full-state observability assumption is unacknowledged in the limitations.** The feedback correction **L(x(t) − x̂(t))** requires ground-truth state measurements x(t) at every timestep. In many Neural ODE applications—time-series imputation, vision-based estimation, partially observed systems—states are not directly measurable. The Limitations section mentions gain tuning but is completely silent on this structural requirement. The scope of applicability is narrower than the abstract implies, and this should be clearly stated.

- **FB-MPC (feedback on nominal model, RMSE 0.203 m) underperforms Neural-MPC (0.167 m), and this result is unexplained.** This is an informative failure mode: applying feedback to a poor base model produces worse results than a better learned model without feedback. This suggests the method's effectiveness is coupled to the quality of f_neural, and there is likely a threshold below which feedback becomes destabilizing. The paper does not discuss this interaction at all, which obscures the conditions under which the method is expected to work.

### Minor

- **Theorem 1 is a direct application of standard ISS/Lyapunov analysis** to the linear error dynamics ẋ̃ = −**L**x̃ + Δf. It is correct and provides useful practical guidance, but it does not constitute a novel theoretical result. More importantly, the bound on **B₂** grows proportionally to λ_M(**L**)/λ_m(**L**) — i.e., the condition number of **L** — meaning for a poorly conditioned gain matrix (asymmetric state dimensions), the bound on ẋ̃ is vacuous. This is not discussed. Additionally, the theorem holds in continuous time, but the implementation uses Euler integration (Eq. 8). No discrete-time stability or discretization error analysis is provided; for large T_s, the discrete closed-loop may not inherit the continuous-time convergence property.

- **The neural feedback form (Section 4) lacks any stability or convergence analysis.** The neural feedback is used in the primary application (quadrotor MPC) yet no formal guarantees exist for it. For safety-critical control, the paper should at minimum empirically demonstrate behavior when uncertainties exceed the domain randomization range (e.g., a failure case study), rather than only presenting success cases.

- **Single trajectory in MPC experiment; no cross-run statistics.** Figure 9 reports RMSE for one Lissajous trajectory. The number of independent trials or flights is unclear. For the key quantitative claim (0.093 m vs. 0.151 m), there are no standard deviations or confidence intervals even across the 3 randomly generated validation trajectories mentioned in Section 5.2.1.

- **Ambiguity between simulation and real flight in Section 5.2.** Section 5.2 uses the heading "Flight Tests," and Figure 8 refers to "real fly results" for training data. However, the test uncertainties (precisely specified mass, inertia, drag coefficient percentages, and additive disturbance values) read more like simulation-injected parameters than hardware measurements. The paper should explicitly state whether the MPC results in Figure 9 are from physical quadrotor flights or high-fidelity simulation with injected perturbations—this distinction is essential for assessing the contribution's real-world validity.

- **n_case (number of domain randomization cases) is not specified in the main text** for either the spiral or quadrotor experiments. The sensitivity of the method to this hyperparameter is untested, even though it is a critical design choice.

### Tiny

- Section 3.4 ("Ablation Study on Observer Gain") appears before Section 3's main motivating example (Figure 5), which makes the ordering confusing. Figure 4 references content that is better understood after Figure 5 is shown. Swapping the presentation order would improve readability.

- The term "two-DOF" is used throughout but its standard control-theoretic meaning (independent tuning of setpoint tracking and disturbance rejection channels) is never connected to the architectural definition used here. A brief explicit connection would clarify the framing for readers from the ML community.

---

## Nice-to-Haves

- **Visualization of feedback correction magnitude over time.** A plot of ||h(·)||/||f_neural(·)|| during test trajectories would reveal whether the feedback is making minor perturbations or effectively replacing the base model. This would help distinguish genuine generalization from base-model failure being patched by feedback.

- **Unfrozen-base ablation.** An experiment where f_neural parameters are *not* frozen during feedback training would empirically verify that freezing is necessary for the two-DOF property rather than being a coincidental design choice.

- **Extension discussion for partial observability.** Even a short discussion on how the method could be extended to handle partial state observations (e.g., via a learned encoder or coupled observer) would substantially broaden the paper's applicability claims.

- **Testing under varying uncertainty magnitude.** Sweeping uncertainty levels beyond those used in the domain randomization range would characterize the method's robustness frontier and expose where it breaks down—currently only a single uncertainty regime is tested.

- **Automated gain tuning.** The authors note this as a future direction in the Limitations section. A brief bi-level formulation sketch would strengthen the paper and make the limitation feel more principled.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **"Luenberger observer = not novel" as a fatal criticism.** The harsh critic argues that Eq. (7) is "formally identical to a Luenberger observer" and therefore not novel. However, the paper explicitly cites Luenberger (1966) and ESO (Guo et al., 2020) as inspirations in both the introduction and Section 3.1 — it does not hide the connection. The contribution is the specific integration of this correction mechanism into the Neural ODE latent dynamics, combined with the two-DOF training decomposition. Claiming this is "not novel" because the scalar error dynamics are classical confuses the mechanism with its application. Weakened and absorbed into the Minor section on Theorem 1.

- **Criticism of baselines from 2001–2012 as "outdated."** The trajectory prediction task uses an existing open-source dataset (Jia et al., 2024), and the model-based methods cited are the established baselines for *that specific task*. Demanding newer baselines on a fixed dataset is unreasonable; the comparison with Neural ODE (Chen et al., 2018) is the primary ML baseline.

- **Demand for EKF/UKF augmented network baselines and meta-learning baselines.** These would constitute new related-work comparisons that are not clearly within the paper's stated scope. The paper compares with adaptive last-layer methods (AdapNN-MPC), which are the most structurally similar. EKF/UKF comparison would be valuable context but its absence is not a fatal flaw given the paper's positioning.

- **Claim that 9 test trajectories invalidates results.** Figure 7 shows standard deviations across all 9 trajectories with visible separation between methods. For real-hardware aerodynamic experiments, 9 test trajectories is not an unreasonably small sample. This is weakened rather than kept as a major concern.

- **Criticism that the "two-DOF" framing resembles adapter-based fine-tuning / continual learning.** While superficially similar, the specific application to correcting ODE latent dynamics via a feedback signal (not a static residual) is architecturally distinct. The comparison is imprecise and not actionable.

- **Demands for confidence intervals on MPC results.** For real quadrotor flight experiments, single-run evaluation is standard in the field. Requesting multi-run statistics with confidence intervals is not the community norm for this type of hardware paper. Noted as a Minor concern rather than a Major one.

---

## Novel Insights

The most genuinely interesting observation that emerges from synthesis of the reviews—but is not fully articulated in the paper itself—is the **functional coupling between base model quality and feedback effectiveness**: FB-MPC applies feedback to the nominal (physics-only) model and performs *worse* than Neural-MPC (0.203 m vs. 0.167 m), while FNN-MPC applies feedback to the neural-augmented model and performs substantially better (0.093 m). This suggests the feedback loop is not a universal corrective mechanism but acts as a *multiplicative amplifier* of the base model's residual quality — a good base model with bounded Δf benefits strongly from feedback, while a poor base model may have Δf that violates Assumption 1, causing the feedback to destabilize rather than correct. This interaction deserves explicit theoretical treatment (what bound on Δf is needed for feedback to improve vs. degrade performance?) and would constitute a novel and practically important characterization of the method's operating envelope.

---

## Suggestions

1. **Report MPC inference times and clarify flight test vs. simulation.** Explicitly state whether Figure 9 results are hardware flights or simulation, and provide wall-clock prediction times for FNN-MPC vs. Neural-MPC. These are blockers for assessing real-world deployability.

2. **Correct the "first time" claim in the conclusion.** Replace with a precise statement: *"we integrate, for the first time, a continuously-operating observer-style feedback loop into the latent dynamics of a Neural ODE."*

3. **Add a limitations paragraph on full-state observability** and sketch a path to partial observability (e.g., coupled with a Luenberger-style neural observer for the output equation).

4. **Analyze FB-MPC failure mode.** Discuss theoretically or empirically why feedback on the nominal model underperforms a learned model without feedback. This would validate the claim that FNN-MPC's advantage comes from both the neural base *and* the feedback, not from feedback alone.

5. **Add a failure case study for the neural feedback.** Show at least one scenario where uncertainties exceed the domain randomization range and characterize how gracefully the method degrades.

6. **Specify n_case** for all experiments, and include a sensitivity experiment (varying n_case) in the appendix.

7. **Address condition number sensitivity for gain matrix L.** Discuss the B₂ bound behavior for ill-conditioned **L** and provide practical guidance on gain selection, or show empirically that the gains used in experiments are well-conditioned.

---

**Evaluation axes:**

- **Novelty:** Moderate. The linear feedback form is a Luenberger observer applied to neural ODEs — the theoretical machinery is classical. The two-DOF training decomposition and the multi-step cascade with gain decay are original contributions at the system level. The paper would benefit from tempering its novelty framing while emphasizing the specificity and practical value of the contribution.

- **Technical soundness:** Adequate for the linear case; the neural feedback case lacks formal support. The discrete-time stability gap and the unanalyzed condition number dependence of Theorem 1 are genuine technical omissions, though they do not undermine the experimental findings.

- **Empirical support:** Reasonably strong for the quadrotor MPC task (real hardware, six baselines, large multi-uncertainty regime), but noticeably thin in reporting discipline — single trajectory, no cross-run statistics, ambiguity about simulation vs. hardware. The trajectory prediction experiment is less compelling given only 9 test trajectories and limited baselines.

- **Significance:** High for the robotics/control-learning intersection. The ability to add uncertainty-handling to an already-trained Neural ODE without retraining the base model is a practically important capability.

- **Clarity:** Good overall. The derivation from Eq. (4)–(8) is careful and the notation is consistent. The presentation order in Section 3 (ablation before motivating example) and the unexplained "two-DOF" terminology are the main clarity gaps.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
