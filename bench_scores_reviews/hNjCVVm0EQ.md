Now I have sufficient understanding of the paper. Let me synthesize the final review.

---

## Summary

MamKO proposes integrating Mamba's matrix-generation (selection) mechanism with Koopman operator theory to produce time-varying Koopman operators from historical data, enabling efficient convex MPC for nonlinear time-varying systems. By generating the full operator sequence $\{\bar{A}_{k:k+H-1|k}, \bar{B}_{k:k+H-1|k}, C_{k:k+H-1|k}\}$ offline from past observations before each control step, the approach preserves MPC convexity (avoiding the bilinear $B_k u_k$ coupling) while adapting to changing dynamics. The method is evaluated on five benchmark systems spanning time-invariant and time-varying settings.

---

## Strengths

- **Elegant convexity preservation through offline operator generation.** By generating the full future operator sequence from *historical* data rather than online states, MamKO specifically avoids the bilinear term $B_k u_k$ that would arise if $B_k$ depended on the current input. This is a concrete, non-obvious design decision that keeps the MPC problem quadratic. The paper makes this explicit (Section 3.2): "as $B_k$ is generated from the input sequence containing $u_k$, the bilinear term $B_k u_k$ can lead to a non-convex optimization problem... As a substitute, we generate the matrices from the historical data."

- **CELU activation for unstable system representation.** Replacing Mamba's negative-exponential eigenvalue constraint with negative-CELU is a targeted and technically justified modification. It allows continuous-time eigenvalues up to +1 (positive, hence unstable modes), enabling Koopman representation of inherently unstable systems (e.g., CartPole, oscillatory GRN). The ablation in Figure 4 confirms CELU yields the best modeling across all three tested environments, and the improvement over unconstrained eigenvalues (the "None" variant) demonstrates that the constraint is not just permissive but regularizing.

- **Demonstrated advantage that scales with dynamical variation rate.** Figure 2(f) shows that MamKO's advantage over DKO grows monotonically with the angular frequency of parameter variation (0.1, 1, 10 rad/s). This provides principled evidence that the time-varying operator generation directly addresses the problem's core difficulty, rather than incidentally benefiting from extra parameters.

- **Dramatic computational advantage over nonlinear MPC.** MamKO-based MPC reduces solve time by 98%+ over MLP-MPC (nonlinear optimization via IPOPT) across all five systems — e.g., from 3.31 s to 0.0262 s for RSCP (sampling period 18 s) and from 0.743 s to 0.0102 s for CartPole (sampling period 0.02 s). This enables real-time control feasibility where MLP-MPC is not viable.

- **Sample efficiency advantage over model-free RL.** MamKO uses 36,000 labeled samples versus SAC's 1,000,000 environment steps. While both approaches achieve different objectives, this distinction matters practically for data-scarce industrial settings and is a genuine qualitative advantage of the model-based approach.

---

## Weaknesses

- **No comparison against adaptive or online Koopman baselines — the most critical gap.** The paper's primary claim is superiority on time-varying systems, yet the only Koopman baseline is the time-invariant DKO. The Related Work explicitly cites online DMD (Zhang et al., 2019), parameter-adjusting Koopman (Hao et al., 2022), and Fourier-filtered Koopman (Liu et al., 2023) as prior art for time-varying systems. The paper argues these are "time-consuming," but provides no empirical evidence of this claim and no comparison of their accuracy. Without such comparison, gains over DKO on time-varying benchmarks cannot be attributed to the Mamba architecture versus simply using any time-varying operator. Including at least one recurrent (LSTM/GRU) time-varying Koopman baseline would also isolate the contribution of the Mamba selection mechanism specifically.

- **Critical design assumption left unanalyzed: future dynamics predicted from past data.** At time $k$, the network $\phi$ generates the *future* matrix sequence $\bar{A}_{k:k+H-1|k}$, $\bar{B}_{k:k+H-1|k}$, $C_{k:k+H-1|k}$ using only past data $x_{k-H:k-1}$, $u_{k-H:k-1}$. This is a substantive assumption — it presupposes that the historical window provides enough information to forecast how the dynamics will evolve over the next $H$ steps. The paper never states this assumption explicitly, never analyzes when it holds or fails, and provides no experiment stress-testing the prediction horizon or the effect of history length $H$. For slowly varying systems (0.1 rad/s), this may be benign; for rapidly varying ones (10 rad/s), it is non-trivial and is exactly where the method is most claimed to shine.

- **No ablation of history length $H$.** History length is a critical hyperparameter governing the operator-generation quality, yet no sensitivity analysis is provided. It is unknown whether performance degrades gracefully or sharply with shorter windows, which matters greatly for online deployment.

- **MamKO is slower than DKO, yet efficiency is framed as a key contribution.** Table 2 shows MamKO is uniformly slower than DKO-based MPC: 10.2 ms vs. 7.35 ms (CartPole), 26.2 ms vs. 10.6 ms (RSCP), 29.5 ms vs. 12.8 ms (TV-RSCP). The paper's efficiency argument is valid only against MLP-MPC, which requires nonlinear optimization. The legitimate efficiency claim is "MamKO is nearly as fast as DKO while substantially outperforming it," not a general efficiency advantage that the text implies. This should be stated more precisely.

- **Marginal improvement on time-invariant systems without significance testing.** For CartPole (time-invariant), MamKO reduces cost by 5.05% over DKO. Given the confidence intervals visible in Figure 3, it is unclear whether this difference is statistically significant. No formal tests (e.g., paired t-test) are reported. Claiming "superiority" on these margins without statistical support is an overstatement.

- **Notation inconsistency between training (Eq. 9) and MPC (Eq. 10): six improvements cited for "five" systems.** The MPC optimization (Eq. 10a) optimizes over $u_{k|k}^*, \ldots, u_{k+N-1|k}^*$ using index $N$, but constraints use index $H$ (Eq. 10e: $j = k+1, \ldots, k+H-1$). $N$ and $H$ are never equated explicitly. Additionally, Section 5.2 lists six percentage improvements ("5.05%, 3.70%, 92.10%, 6.56%, 14.19%, 84.74%") for what the text refers to as "five systems," because the 10 rad/s time-varying CartPole is treated as a sixth scenario without being explicitly counted. These are writing errors that should be corrected.

- **Eigenvalue stability under long sampling periods.** The paper permits continuous-time eigenvalues up to +1, discretized as $\bar{A} = e^{AT}$. For RSCP with sampling period $T = 18$ s, a continuous eigenvalue of +1 yields a discrete eigenvalue of $e^{18} \approx 6.6 \times 10^7$. There is no discussion of how the training procedure prevents or handles such extreme values, or whether gradient descent remains numerically stable under these conditions.

- **Absence of theoretical stability or feasibility guarantees.** For a control-focused paper, the complete absence of closed-loop stability analysis is significant. While acknowledged as future work, with time-varying NN-generated operators, neither recursive feasibility of the MPC nor stability of the closed-loop system is addressed even informally. Standard MPC stability proofs assume model invariance; this paper's setting explicitly violates that assumption. A bounded-error or robustness-oriented argument, even informal, would substantially strengthen the control-theoretic grounding.

- **Incorrect and recurring "LLM" terminology.** The paper characterizes Mamba as a "large language model" throughout (Abstract, Introduction, Related Works, Conclusion). Mamba is a structured state-space model architecture; it is not an LLM in any precise sense. The only element borrowed from Mamba is the selective matrix-generation (discretization) mechanism, not language modeling, tokenization, or scale. This mischaracterization, while perhaps intended to attract attention, undermines technical precision and will be distracting to informed reviewers.

---

## Nice-to-Haves

- **Ablation isolating the input-dependent selection mechanism.** Compare MamKO against a variant where operator matrices vary with time step index but are not conditioned on historical input/state data (i.e., remove the selection/conditioning, keeping only time-varying operators). This would quantify the value of Mamba's specific input-dependent generation vs. simply learning time-varying operators by any means.

- **Latent-space linearity verification.** Low prediction error does not confirm that dynamics are truly linear in the lifted space. Plotting residuals of $z_{k+1} - \bar{A}_{k|k} z_k - \bar{B}_{k|k} u_k$ or visualizing eigenvalue trajectories of $A_k$ over time would validate that the Koopman assumption is not merely a useful fiction.

- **Sensitivity to observation noise.** Real control systems have noisy sensors. Testing whether the historical-data-conditioned operator generation amplifies noise (destabilizing MPC) would strengthen claims of practical applicability.

- **Eigenvalue trajectory visualization.** Plotting the diagonal entries of $\bar{A}_{k|k}$ over time for a time-varying system would provide interpretable evidence that the network genuinely adapts its represented dynamics rather than learning a fixed average.

- **Higher-dimensional benchmarks.** The largest system in the paper (RSCP) has a modest state dimension. Testing on a higher-dimensional system (e.g., a 7-DoF robot arm or quadrotor) would validate scalability claims implied by the "large model" framing and the efficiency results.

- **Explicit sample efficiency comparison.** The 36k (MamKO) vs. 1000k (SAC) training data gap is mentioned but not highlighted as a primary result. A dedicated comparison of data efficiency would be a compelling selling point.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Diagonal A lacks justification in the Koopman setting"** (Harsh Critic): The diagonal structure is explicitly inherited from Mamba's design and is standard in the SSM literature (S4, Mamba). Its computational advantages are well-understood and its effectiveness is empirically validated. Demanding a Koopman-specific theoretical justification for a design choice borrowed from a mature SSM literature is scope creep.

- **"Missing related works"**: Per review instructions, we do not cite missing related works as we cannot verify their existence.

- **"Validate on hardware or high-fidelity simulator with delays"** (Spark Finder): Demanding hardware experiments is well outside the scope of this type of ML/control venue paper and is not a standard expectation for ICLR contributions of this kind.

- **"Benchmark scalability / larger datasets"** (Reviewers 2 & 3): The current benchmarks are appropriate for demonstrating proof of concept. Larger benchmarks would strengthen claims but are not necessary to validate the core method. Moved to nice-to-haves in abbreviated form.

- **"Unfair comparison: MLP-MPC uses expensive nonlinear optimization"**: This comparison is intentionally asymmetric — MLP-MPC solving a nonlinear program is the cost of using an MLP model in MPC, and the asymmetry benefits the baseline to prove a stronger point about convexity. It is not a flaw.

- **"Requesting confidence intervals / formal significance tests for small-scale benchmarks"**: Unlike large-scale benchmarks where single-run evaluation is the norm, these are small-scale repeated experiments (10 trials each), so confidence intervals are already provided. The concern about formal significance testing on marginal improvements is retained as a weakness (not removed), but demanding it be done via a specific statistical framework (e.g., "Wilcoxon") is too prescriptive.

---

## Novel Insights

The most genuinely novel architectural insight in this paper is the *convexity-preserving historical conditioning* for time-varying Koopman MPC: by generating the full future operator sequence before the MPC solve (using only past data), the authors sidestep the bilinear $B_k u_k$ coupling that would arise if $B_k$ were conditioned on the current state or input. This design pattern — deliberate decoupling of the model-adaptation step from the optimization step — has broader implications for any learning-based MPC scheme that uses input- or state-dependent predictive models. The CELU activation as a generalization of the Mamba negative-exponential constraint to allow positive eigenvalues is a small but practically important modification with clear motivation from the control setting. Beyond these, the insight from Figure 2(f) that advantage over fixed-operator methods scales with the rate of dynamic variation provides a principled, quantifiable criterion for when the architectural complexity of MamKO is justified over simpler alternatives.

---

## Suggestions

1. **Add at least one time-varying Koopman baseline** (e.g., windowed EDMD or an LSTM/GRU-based time-varying Koopman model) to empirically substantiate the claim that the Mamba architecture, not merely any time-varying parameterization, drives the improvements. Even a simple RNN-based operator generator as an ablation point would be informative.

2. **Provide a sensitivity analysis over history length $H$** across at least two systems (one time-invariant, one time-varying). Show how prediction error and control cost change as $H$ varies; this is essential context for practitioners deploying the method.

3. **Revise all instances of "large language model" / "LLM"** when referring to Mamba. The accurate phrase is "state-space model (SSM) architecture" or "selective state-space model." This correction is both technically necessary and avoids the impression of buzzword-chasing.

4. **Correct and clarify the N/H notation** in Eq. 10: either unify to one symbol or explicitly state that $N = H$ in this formulation. Also fix the "five systems / six percentages" inconsistency in Section 5.2.

5. **Add a brief discussion of eigenvalue magnitude under long sampling periods** (the RSCP case with $T=18$ s) and whether the training procedure implicitly constrains eigenvalues to avoid numerically pathological discrete operators.

6. **Include a short analysis or discussion of closed-loop stability under time-varying operators**, even if informal. At a minimum, bound the prediction error of the operator-generation network and discuss how this propagates to MPC suboptimality. This is the most important gap for readers from the control community.

7. **Apply formal significance testing** (e.g., paired t-test across the 10 trials) when reporting control cost improvements in the time-invariant settings where the margins are small (5–6%). Either confirm statistical significance or temper the language of superiority.