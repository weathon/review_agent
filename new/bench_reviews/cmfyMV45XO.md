## Summary

The paper proposes feedback neural networks, which augment neural ODEs with an observer-based feedback loop that uses real-time state prediction errors to correct the learned latent dynamics. A linear feedback form with convergence guarantee (Theorem 1) is introduced first, followed by a nonlinear neural feedback learned via domain randomization within a "two-DOF" framework that freezes the pre-trained neural ODE and trains only the feedback module. Experiments include a spiral toy example, trajectory prediction of an irregular bottle, and model predictive control of a quadrotor under compound uncertainties.

## Strengths

- **Principled observer-based correction for neural ODEs**: The idea of using the tracking error between observer and true state to reveal and correct model mismatch is well-grounded in control theory (Luenberger observers) and novel in its application to neural ODE generalization. Eq. 7 provides a clean, interpretable mechanism: $\hat{f}_{neural}(t) = f_{neural}(t) + \mathbf{L}(\mathbf{x}(t) - \hat{\mathbf{x}}(t))$.

- **Convergence guarantee for the linear feedback form**: Theorem 1 proves that under Assumption 1 (bounded residual $\|\Delta f(t)\| \leq \gamma$), the observation error and its derivative exponentially converge to bounded sets whose size is regulated by feedback gain $\mathbf{L}$, with explicit bounds $\|\tilde{\mathbf{x}}(t)\| \leq \gamma/\lambda_m(\mathbf{L})$.

- **Two-DOF framework demonstrably preserves nominal-task accuracy**: Figure 6 directly shows that domain randomization applied to the entire neural ODE degrades nominal performance (Fig. 6a), while freezing the pre-trained model and training only the feedback part maintains the original accuracy (Fig. 6b) while enabling generalization to randomized cases (Appendix Fig. S10).

- **Substantial empirical improvement on quadrotor MPC under real-world uncertainties**: FNN-MPC achieves RMSE of 0.093m under 37.6% mass uncertainty, 40% inertia uncertainty, 14.3–25% drag coefficient uncertainty, and 0.3N disturbances — a 44% reduction over Neural-MPC (0.167m) and 38% over AdapNN-MPC (0.151m) (Figure 9). In this MPC setting, all methods have access to the current state at each receding-horizon step, making the comparison relatively fair.

- **Systematic ablation linking theory to practice**: Figure 4's heatmap of prediction errors across feedback gain levels and uncertainty levels empirically validates Theorem 1's prediction that error decreases with gain up to a noise-amplification threshold.

## Weaknesses

### Fatal
None.

### Major

- **Theorem 1 does not cover the cascaded multi-step prediction used in all experiments.** Theorem 1's convergence guarantee assumes the feedback term has access to the true state $\mathbf{x}(t)$ (the error dynamics in Eq. 9 derive from substituting the true dynamics Eq. 1 into Eq. 7). However, in the cascaded multi-step prediction (Section 3.3, Figure 3), only the first layer receives the true state; subsequent layers use *predicted* states as the input to the feedback term. The paper acknowledges this partially on line 127: "the convergence of $\hat{f}(t)$ can only be guaranteed as current $t$," but the assertion that "the prediction error will converge from top to bottom in order" (line 129) is made without proof or formal analysis. Since the multi-step cascaded prediction is the primary evaluation protocol across all experiments (Figures 5, 7, 9), this theory-experiment gap is significant. Even a bound on error accumulation in the cascaded setting would substantially strengthen the paper.

- **The nonlinear neural feedback — presented as a key contribution — is only evaluated on the toy spiral example.** Section 4 introduces the learnable nonlinear feedback $h_{neural}$ as the solution to the linear form's limitations (avoiding manual gain tuning, handling complex scenes). This is explicitly framed as advancing beyond the linear form. Yet the nonlinear feedback is tested only on the spiral example (Figure 6). The real-world experiments — bottle trajectory (Section 5.1) and quadrotor MPC (Section 5.2) — appear to use the linear feedback form, as the authors themselves acknowledge: "the presented nonlinear neural form is preliminarily tested in Section 4" (line 336-337). If the headline results rest on the linear form, then the nonlinear neural feedback is an unvalidated extension rather than a contribution the paper's claims can rest on.

### Minor

- **Missing measurement re-initialization baseline for the bottle trajectory experiment.** In Section 5.1, the feedback NN uses intermediate measurements during its 0.5s prediction horizon, while the Neural ODE baseline performs a pure open-loop rollout. A simple Neural ODE with one-step re-initialization from each measurement would isolate whether the improvement comes from the observer mechanism specifically, or simply from having access to measurements that the vanilla rollout does not use. (This concern is less acute for the MPC experiment where all methods re-initialize at each receding horizon step.)

- **The gain decay strategy (Eq. 11) is purely heuristic.** No theoretical justification is provided for why exponential decay $\mathbf{L}_i = \mathbf{L} \odot e^{-\beta i}$ is appropriate, how to choose $\beta$, or what the tradeoffs are. While Figure 5g shows the strategy works empirically, the lack of any analytical guidance limits the method's applicability to new systems.

- **Only 9 test trajectories in the bottle experiment (Section 5.1), without statistical significance tests.** The shaded standard deviation regions in Figure 7 provide some indication of variability, but with only 9 trajectories, the reliability of the reported improvement is uncertain.

- **Observer initialization $\hat{\mathbf{x}}(0)$ is not discussed.** Eq. 6 defines $\hat{\mathbf{x}}(t)$ but the initial condition is critical for the feedback mechanism's transient behavior and effectiveness. This omission affects reproducibility and understanding of the method.

### Trivial
- None worth listing.

## Nice-to-Haves

- A theoretical characterization of error propagation in the cascaded multi-step prediction setting, even a loose bound, would significantly strengthen the paper.
- Evaluate the nonlinear neural feedback on the quadrotor MPC experiment to validate whether the learnable feedback provides benefits over manual gain tuning in complex real-world settings.
- Add a Neural ODE with measurement re-initialization baseline for the bottle trajectory experiment.
- Visualize the feedback correction term $\|\mathbf{L}(\mathbf{x}(t) - \hat{\mathbf{x}}(t))\|$ over time during multi-step prediction to reveal whether the feedback is actively correcting or has collapsed.
- Clarify upfront that the method requires real-time state measurements and is designed for closed-loop prediction/control settings rather than open-loop long-horizon forecasting.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Generalization" vs. "online correction" framing (Harsh Critic's structural issue #1)**: The harsh critic argues this is not "generalization in the standard ML sense." While the distinction is valid, the paper is clearly about closed-loop prediction/control settings where measurements are available — common in robotics/control. The term "generalization" is defensible in this context (performing well on unseen dynamics). This is more of a scoping/clarity issue (minor) than a structural flaw.

- **Biological analogy overstated**: The harsh critic says the biological analogy is "overstated" because biological feedback operates in closed-loop sensorimotor circuits while the method adds an observer to an open-loop predictor. This is a style/presentation concern, not a substantive weakness.

- **MLP-MPC single-step training as unfair comparison**: The harsh critic flags that MLP-MPC uses single-step training, disadvantaging it. The authors acknowledge this on line 308. The main comparison in the MPC experiment is between Neural-MPC (multi-step trained) and FNN-MPC, so the MLP-MPC issue is secondary. Also, per the rules, this asymmetry favors the author's method, but since it's not the primary comparison, it remains a minor note rather than a major weakness.

- **"Two-DOF" framing overlaps with adapters/LoRA**: The harsh critic claims the two-DOF concept is just the well-known frozen-encoder-plus-adapter strategy. While the general strategy is indeed well-known, its specific application to neural ODEs with observer-based feedback — where the feedback has a clear physical/control-theoretic interpretation — adds novelty beyond a generic adapter.

- **Assumption 1 excludes growing residuals**: The harsh critic says bounded $\Delta f$ excludes dynamics where the residual grows over time. This is a standard assumption in observer theory and the paper acknowledges it can cover "common step disturbances." It is a known limitation of Luenberger-type analysis, not a specific flaw of this paper.

- **Missing comparison to EKF, test-time training, adaptive last-layer approaches**: These are requests for additional baselines outside the paper's primary scope. The paper already compares against AdapNN-MPC (Cheng et al., 2019), which is a representative adaptive last-layer approach.

- **No convergence guarantee for the nonlinear case**: The paper explicitly positions Theorem 1 as covering the linear case. The nonlinear case is presented as a practical extension. Demanding theory for the nonlinear case is a nice-to-have, not a weakness.

## Novel Insights

The paper reveals an interesting asymmetry between measurement access and model correction: simply re-initializing predictions from measurements (as baselines implicitly do in MPC) does not leverage the *error signal* between predicted and measured states to correct the underlying dynamics model. The observer structure extracts this information and feeds it back into the dynamics, which is a qualitatively different use of measurements. This distinction between "using measurements as initial conditions" versus "using measurement-prediction gaps as correction signals" is the core conceptual contribution, and it is most cleanly illustrated by the comparison between FB-MPC (0.203m RMSE, feedback on nominal model) and FNN-MPC (0.093m, feedback on neural ODE) in Figure 9 — the same observer mechanism produces different gains depending on the quality of the underlying model being corrected.

## Suggestions

- Add a brief theoretical analysis (even a loose bound) on how errors propagate in the cascaded multi-step prediction setting, or explicitly acknowledge this as an open problem with discussion of what makes it challenging.
- Run at least one real-world experiment (ideally quadrotor MPC) with the nonlinear neural feedback form to validate whether learnable feedback provides practical benefits over manual gain tuning.
- Add a one-sentence clarification in the introduction that the method requires real-time state measurements and targets closed-loop prediction/control settings.

## Score and Decision

**Calibration anchors:**

- **High band (>7):** AoraWUmpLU (8.0, Oral): Neural ODE global convergence — strong theory-experiment alignment, clean contribution. S5Yo6w3n3f (7.25, Spotlight): ODE-based smoothing for RL — control-theoretic approach with good theory+experiments. kNpSUN0uCc (7.33, Poster): Model correction for RL — strong theory-experiment match. This paper is clearly below these — its theory doesn't cover its experimental setting.

- **Medium band (4-6):** GrDne4055L (6.25, Poster): Lyapunov-stabilized neural ODEs — similar control-theory+neural ODE combination, similar weaknesses, scored 6.25. AMegoEnlpS (5.0, Reject): Kalman filtering for fine-tuning — observer-based correction for models, weaker practical validation, scored 5.0. WpQbM1kBuy (4.25, Reject): Strong empirical results but theory doesn't cover experimental setting + missing baselines — similar weakness pattern, scored 4.25. This paper is between AMegoEnlpS and GrDne4055L: it has better practical results than AMegoEnlpS but a worse theory-experiment gap than GrDne4055L.

- **Low band (<3):** XQFSIdKMhJ (2.5, Withdrawn): Overclaimed contribution with single-sensor experiments for multi-sensor claims. 3ZdGSTxKuy (2.0, Withdrawn): Overclaimed "open-world" but only narrow OOD tested. This paper is clearly above these — it has real, substantial contributions, not just overclaimed ones.

The paper sits around 5.0-5.5. Its core idea is genuine and the quadrotor MPC results are substantial, but the theory-experiment gap (Theorem 1 doesn't cover multi-step prediction) and the incomplete evaluation of the nonlinear feedback (the paper's most distinctive contribution over a simple Luenberger observer) are significant weaknesses that prevent confident acceptance. Compared to GrDne4055L (6.25), this paper has a larger theory-experiment gap. Compared to AMegoEnlpS (5.0), it has better practical results but similar observer-based correction novelty.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>