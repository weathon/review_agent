Now let me write the final review.

## Summary

This paper proposes feedback neural networks that incorporate a real-time state feedback loop into neural ODEs to correct learned latent dynamics and improve generalization. A linear feedback mechanism (structured as a Luenberger-like observer) is first presented with convergence guarantees (Theorem 1), then extended to a learnable nonlinear feedback form trained via domain randomization while freezing the base neural ODE, yielding a two-DOF architecture that preserves nominal-task accuracy. The method is evaluated on spiral trajectory prediction, irregular bottle trajectory prediction, and quadrotor MPC under parametric uncertainties and disturbances.

## Strengths

- **The two-DOF training framework (freeze f_neural, train h_neural separately via domain randomization) is a clean and practically effective design.** Figure 6 directly validates that this approach preserves nominal-task accuracy (6(b) matches Figure 5(c)) while naive full-model domain randomization degrades it (6(a)), circumventing the classic robustness–accuracy tradeoff. This is the paper's strongest contribution.

- **Convergence guarantee (Theorem 1) provides formal bounds for the linear feedback form.** The error dynamics $\dot{\tilde{x}}(t) = -L\tilde{x}(t) + \Delta f(t)$ and its bounded convergence sets $B_1$ and $B_2$ give principled guidance on gain selection, which is more than most neural-ODE generalization methods offer.

- **Substantial empirical improvement on quadrotor MPC.** Under 37.6% mass uncertainty, 40% inertia uncertainty, and 0.3N disturbances (Figure 9), FNN-MPC achieves RMSE of 0.093m vs. 0.151m for the next-best adaptive method (AdapNN-MPC), a 38.4% improvement. This demonstrates real-world applicability beyond synthetic benchmarks.

- **Gain decay strategy (Eq. 11) addresses noise amplification in cascaded multi-step prediction.** Figure 5(g) provides concrete evidence that the exponential decay mitigates error accumulation, giving a practical solution to a real engineering problem.

## Weaknesses

### Fatal

None.

### Major

- **Missing observer baseline with equivalent information access makes it impossible to attribute improvement to the specific feedback architecture rather than to measurement access.** The core mechanism in Eq. (7), $\hat{f}_{neural}(t) = f_{neural}(t) + L(x(t) - \hat{x}(t))$, is structurally equivalent to a Luenberger observer — the paper itself cites Luenberger observers as inspiration (Sections 1 and 3.1). The baselines (Neural ODE, model-based methods in §5.1; Nomi-MPC, Neural-MPC in §5.2) do not exploit real-time state measurements. Without comparing against a standard observer (e.g., EKF or Luenberger observer applied to the same Neural ODE model), the paper cannot distinguish whether the performance gain comes from the feedback *architecture* or simply from *having access to real-time measurements*. The paper includes "FB-MPC" (feedback on nominal model), which partially addresses this, but a comparison where the Neural ODE baseline is augmented with a standard observer is the critical missing experiment. This directly impacts the core claim that the *proposed architecture* drives generalization improvement.

- **Multi-step prediction is underspecified about how feedback operates beyond the first step when ground-truth states are unavailable.** Section 3.3 describes cascaded one-step prediction (Figure 3): the output of layer $i$ becomes the input of layer $i+1$. After the first step, the true state $x(t)$ is unavailable. The paper does not explicitly specify how $x(t_i) - \hat{x}(t_i)$ is computed for layers $i > 0$. If the predicted state from the previous layer is used as $x(t_i)$, then both $x(t_i)$ and $\hat{x}(t_i)$ derive from the same prediction chain, making their difference near-zero and rendering the feedback term ineffective. The gain decay strategy (Eq. 11) implicitly acknowledges this degradation but does not resolve the ambiguity. Since the quadrotor MPC results depend on multi-step predictions over a receding horizon, this affects whether the empirical claims reflect what the paper asserts.

### Minor

- **Quadrotor flight test (§5.2) reports results on a single Lissajous trajectory, providing no estimate of variance across different initial conditions or uncertainty realizations.** While the uncertainties are substantial (37.6% mass, 40% inertia, etc.), single-trajectory evaluation without confidence intervals or multiple test runs limits the statistical strength of the claimed 38.4% improvement (RMSE 0.093m vs. 0.151m).

- **Theorem 1's derivative error bound $B_2$ contains $\lambda_M(L)/\lambda_m(L)$ (the condition number of $L$), which is not discussed.** For practical gain matrices where the eigenvalues span a wide range, this bound can be arbitrarily large, suggesting that derivative accuracy may degrade even as state error improves. The paper should note this trade-off.

- **The claim that "the convergence of later layers will not affect the convergence of previous layers" (§3.3) is misleading in a cascaded architecture.** Errors from early layers propagate as inputs to later layers. The statement is technically true in the sense that each layer's convergence is computed independently, but the *prediction quality* of later layers absolutely depends on the errors from earlier ones. This should be stated more precisely.

### Trivial

None.

## Nice-to-Haves

- A comparison against at least one modern online adaptation method (e.g., extended Kalman filter adaptation or gradient-based test-time adaptation) cited in §6.3, to position the method relative to the online adaptation literature.
- Visualization of the learned $h_{neural}$ compared to the known residual $\Delta f$ on a synthetic example with known ground truth, to reveal whether the feedback has learned a meaningful correction or acts as a generic stabilizer.
- Multiple test trajectories with variance reporting in the quadrotor experiment.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Generalization claim conflates online measurement exploitation with true generalization"**: The harsh critic's claim that this is *not* generalization is too strong. In real-time control and estimation systems, using measured state feedback to adapt predictions at test time is a recognized form of adaptation/generalization to unseen conditions (different from i.i.d. generalization, but valid in its own context). The paper's contribution of combining this with a two-DOF architecture that preserves nominal accuracy is real. The concern is better expressed as a *missing baseline* (kept as Major weakness above) rather than a category error.

- **"The linear feedback is just classical observer theory and not novel"**: The paper explicitly acknowledges the connection to Luenberger and Kalman observers (lines 43, 71). The contribution is the integration into a neural ODE with a specific two-DOF training framework, not the observer structure itself. Calling this lack of novelty a "Fatal" issue overstates the case; it's a fair observation that the paper should position more clearly, but the integration and training strategy add real value.

- **"Biological motivation connection is weak"**: This is a minor framing issue, not a substantive weakness. Motivational analogies need not be rigorous derivations.

- **"Two-DOF advantage is trivially true for any residual/additive approach"**: While the principle is well-known (residual learning), the specific application to neural ODE generalization with frozen-then-feedback training has practical value and is empirically validated.

- **"21 training / 9 test trajectories is thin evidence" (§5.1)**: For real-world bottle trajectory data, this is a reasonable scale. The standard deviations are reported.

- **"MLP-MPC uses known poor practice / AdapNN-MPC adapts only the last layer"**: These are the methods as published and cited. Criticizing them as "deliberately limited" is speculative. Per the rules, if the asymmetry favors the baseline, the comparison is fair — the authors are showing their method beats established approaches.

- **"No significance testing in §5.1"**: The standard deviations are shown in Figure 7(Right), which allows visual assessment. Formal significance testing is nice-to-have but not standard for this type of evaluation.

- **"Missing comparison to test-time adaptation methods cited in §6.3"**: Downgraded to Nice-to-Have. These methods address related but distinct problems (distribution shift in classification/generation vs. online state estimation in control). The comparison would strengthen but is not essential.

## Novel Insights

The paper's two-DOF framework — freezing the base neural ODE and training a separately parameterized feedback module via domain randomization — is a clean decomposition that merits attention beyond the specific observer-like mechanism. It suggests a general principle: in settings where both accuracy preservation and generalization are needed, decoupling the "accuracy module" and the "adaptation module" during training is more effective than joint training with domain randomization. This insight is relevant to any neural-network-in-the-loop control system.

## Suggestions

- Add a "Neural ODE + EKF" or "Neural ODE + Luenberger observer" baseline to the quadrotor MPC experiment. This is the single most important addition — it would either validate the architectural contribution or honestly reposition the paper as a practical integration of observer theory with neural ODEs.
- Explicitly clarify in Section 3.3 how $x(t_i) - \hat{x}(t_i)$ is computed for multi-step prediction layers beyond the first step, and discuss the implications for feedback effectiveness.
- Run the quadrotor experiment on multiple test trajectories and report variance to strengthen the empirical claims.

## Evaluation

**Originality**: The integration of observer-style feedback into neural ODEs with a two-DOF training scheme is novel, though the linear feedback mechanism itself is classical. Moderate originality.

**Importance**: The problem (generalization of neural ODEs for control under uncertainty) is important and practical. The real quadrotor experiments demonstrate relevance.

**Claims support**: Partially supported. The two-DOF advantage is well-validated; the attribution of MPC improvements to the specific feedback architecture is undermined by the missing observer baseline.

**Experimental soundness**: The experiments demonstrate the method works, but the lack of an observer baseline and single test trajectory in the quadrotor experiment limit conclusiveness.

**Clarity**: Generally well-written, though the multi-step prediction section could be clearer.

**Community value**: The two-DOF training principle and the practical demonstration on real flight data are valuable for the learning-for-control community.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Phy-DRL (spotlight) | /home/wg25r/review_agent/human_reviews/5Dwqu5urzs.md | 7.5 | Observer-like residual policy with safety guarantees, strong robot experiments. More novel theory, cleaner contribution. This paper is below Phy-DRL. |
| AROS (poster) | /home/wg25r/review_agent/human_reviews/GrDne4055L.md | 6.25 | Neural ODE + Lyapunov stability for OOD detection. Similar pattern of applying control theory to neural ODEs, accepted as poster. This paper is comparable — similar integration of classical theory, similar empirical scope. |
| FNSDA | /home/wg25r/review_agent/human_reviews/SXj1qjFEpQ.md | 5.75 | Domain adaptation for dynamics generalization. Rejected for incremental methodology and incomplete ablations. This paper is above FNSDA due to stronger theory and real experiments. |
| Runtime Learning Machine | /home/wg25r/review_agent/human_reviews/KCTHM2Ffh3.md | 6.33 | Observer-like HA-Teacher correcting HP-Student for safety, evaluated on robots. Rejected despite 8/5/6 scores. More complex system, weaker presentation. This paper is comparable. |
| CLIMODE (oral) | /home/wg25r/review_agent/human_reviews/xuY33XhEGR.md | 8.0 | Strong physics-informed neural ODE with value-conserving architecture. Far above this paper in novelty and experimental scope. |
| Control-based ANNs | /home/wg25r/review_agent/human_reviews/7duh4Ml5rc.md | 1.67 | Deeply flawed control-theory-applied-to-NN paper. Far below this paper. |

This paper sits in the 5–6 range relative to anchors. The two-DOF framework and convergence guarantee are real contributions; the missing observer baseline and multi-step specification gap prevent it from being clearly above the acceptance bar. It is comparable to AROS (6.25, poster) and slightly below Runtime Learning Machine (6.33, rejected). The missing baseline that would control for information access is a significant methodological gap, but the paper does make a real contribution with the two-DOF training framework and real flight data.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>