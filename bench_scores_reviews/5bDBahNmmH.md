## Summary

Cohesion introduces a diffusion-based forecasting framework for chaotic PDE systems that reframes autoregressive forecasting as trajectory planning. The key contribution is using a Koopman-based reduced-order model (ROM) to efficiently generate long conditioning prior sequences, enabling single-pass (R=T) conditional denoising over full trajectories. Temporal composition allows flexible subsequence-length trade-offs between speed and accuracy (R∈[1,T]), and classifier-free guidance enables zero-shot conditioning without retraining. Experiments on Kolmogorov Flow (Re=10³) and Shallow Water Equation show advantages over probabilistic SFNO variants in RMSE, MS-SSIM, and spectral divergence.

---

## Strengths

- **Koopman ROM enabling trajectory-level single-pass denoising**: The specific use of a deep Koopman encoder-operator-decoder (Eq. 13–15) to generate an entire sequence of conditioning priors C(x) in one forward pass, which is then consumed by a single conditional denoising pass, is a concrete and practically motivated innovation. Figure 12 shows empirical runtime reductions of 7–13× (R=1 vs R=T) on both benchmarks, directly demonstrating the computational benefit.

- **Spectral fidelity as a core design objective**: The paper goes beyond pixel-wise metrics and explicitly evaluates spectral divergence (Figure 8) and power spectrum evolution (Figure 10). Figure 10 uniquely visualizes how Cohesion first recovers low-wavenumber coherent structures before correcting high-wavenumber fluctuations during the denoising trajectory — a mechanistic insight that distinguishes the approach from methods that only minimize reconstruction loss.

- **Zero-shot generalization to partial observations**: Section 4.3 and Figure 11 demonstrate that the classifier-free guidance approach maintains physically consistent forecasts even when the ROM conditioning prior is masked (equally-spaced masking experiment), without any retraining. This is a meaningful practical robustness property that most competing approaches lack by construction.

- **ROM-as-refiner ablation**: Figure 9 provides a direct comparison of "Coherent-only" (ROM standalone) vs "+Cohesion" for both KF and SWE, showing quantitative improvements in RMSE, MAE, and MS-SSIM across rollout horizons. This cleanly isolates the benefit of the diffusion refinement stage over the ROM prior, and partially addresses concerns about ROM quality.

- **Reynolds decomposition taxonomy**: The explicit connection drawn in Section 2 between turbulence theory (Reynolds decomposition) and the two classes of diffusion conditioning approaches (full posterior vs. residual posterior) provides a unifying conceptual lens that organizes a fragmented literature. While not yielding new algorithmic derivations, this framing is clarifying.

---

## Weaknesses

### Fatal
None.

### Major

- **"Orders-of-magnitude speedups" claim is unsupported.** Figure 12 reports 7–13× speedup (R=1 relative to R=T), which is less than one order of magnitude, let alone "orders of magnitude." This is a factually inaccurate statement in the abstract and conclusion. It also compares only Cohesion(R=1) vs Cohesion(R=T) internally — no runtime is reported relative to the SFNO baselines, making it impossible to assess whether Cohesion(R=T) is even faster than a single SFNO forward pass. The efficiency claim needs to be (a) corrected in language (e.g., "up to 13× speedup") and (b) contextualized against baselines.

- **Absence of diffusion-based baselines.** The paper claims to outperform "state-of-the-art probabilistic emulators" (abstract, conclusion), yet all baselines are ad hoc probabilistic wrappers around a deterministic SFNO (checkpoint ensembles, MC-Dropout, IC Perturbation). No diffusion-based baselines — of the kind cited in the introduction (Lippe et al. 2024, Price et al. 2023, Srivastava et al. 2023) — appear in the experiments. Outperforming poorly-suited probabilistic SFNO variants does not substantiate state-of-the-art claims within the diffusion-based forecasting family that the paper positions itself against. At minimum, a standard autoregressive diffusion baseline (e.g., EDM-style or a simple DDPM applied autoregressively) should be included to isolate the benefit of the ROM conditioning strategy.

- **No ablation studies for key design choices.** Three design choices are central to the method — window size W, subsequence length R (beyond the two extremes), and the ROM architecture — yet none are ablated:
  - Only W=5 is used, with no ablation over W∈{1,3,5,10}. The paper claims local agreement translates to global consistency, but this is unverified empirically.
  - Only R=1 and R=T are reported; intermediate values are claimed as a flexibility feature but never shown.
  - The guidance scale γ in the variance term (after Eq. 10) is a critical hyperparameter with no sensitivity analysis.
  Without these, the reader cannot understand which components drive the performance gains.

- **Missing probabilistic calibration metrics.** The paper positions Cohesion as a probabilistic emulator for uncertainty quantification (Introduction, Conclusion). All evaluation metrics — RMSE, MAE, MS-SSIM, spectral divergence — measure reconstruction quality of individual trajectories or the ensemble mean. No calibration metrics (CRPS, energy score, rank histograms, or coverage plots) appear anywhere. For a probabilistic emulator, these are necessary to assess whether the ensemble spread is meaningful. A model with low RMSE can be completely miscalibrated, which would undermine the uncertainty quantification claim.

### Minor

- **Notation collision in Equation 7.** In Eq. 7, `c` is sampled from N(0,I) and serves as the regression target (i.e., it plays the role of the noise variable ε in standard score matching), while `c` is defined throughout Section 3.1 and 3.3 as the conditioning prior c := ũ(x,t). These are fundamentally different objects sharing the same symbol. The training target in Eq. 7 should be renamed (e.g., ε or n) to distinguish it from the conditioning signal. While not fatal, this is a source of genuine confusion in the formulation.

- **Thin and non-operational RL framing.** The paper claims to incorporate "reinforcement learning (RL) principles" (Introduction, abstract) and cites Janner et al. (2022) for trajectory planning. However, no RL machinery appears in the method: no reward signal, no policy, no value function, no policy gradient. The only connection is the use of the phrase "trajectory planning" as a metaphor. This framing is misleading and should either be grounded (e.g., by discussing the structural analogy more precisely) or replaced with more accurate terminology such as "sequence-level conditional generation."

- **ROM trained with 1-step loss for multi-step rollout.** Equation 15 minimizes a 1-step reconstruction loss, yet the ROM is used autoregressively for up to T=32–64 steps in trajectory planning mode. Error accumulation under 1-step training is a known issue for chaotic systems. Figure 9 shows that ROM+Cohesion improves over ROM-only, which provides indirect evidence the diffusion corrects ROM drift — but the magnitude of ROM error at long horizons, and the regime where this correction fails, is not characterized.

- **"Zero-shot" terminology is non-standard.** The paper uses "zero-shot" to mean "no retraining when the conditioning likelihood changes." In standard ML usage, zero-shot denotes generalization to unseen categories/tasks without any task-specific data. The more precise and community-standard terminology for this setting is "plug-and-play guidance," "training-free conditioning," or "zero-shot posterior sampling" (as used in, e.g., Chung et al. 2022 and Rozet & Louppe 2023 from which the method is derived).

- **No limitations section.** The conclusion is purely optimistic and does not discuss failure modes: ROM divergence at high Re or on out-of-distribution initial conditions, scaling to 3D or high-resolution grids, or cases where the Gaussian observation model in Eq. 9 is violated.

### Tiny

- The "unified framework" described in Section 2 is better characterized as an organizing taxonomy rather than a framework enabling new derivations. The abstract should temper this claim.
- Algorithms 1–4 are referenced in the main text but not included in the main paper body, affecting reproducibility from the main text alone.

---

## Nice-to-Haves

- **ROM error propagation analysis**: Plot ROM-only RMSE vs Cohesion RMSE as a function of rollout step T (extending Figure 9 with quantitative error bars) to characterize the "correction capacity" of the diffusion stage. Particularly valuable at high Reynolds numbers or long T.
- **Training compute discussion**: A brief comparison of total training compute (ROM + score network) vs SFNO training would give a more complete efficiency picture.
- **Spectral error evolution over intermediate timesteps**: Currently Figure 10 shows final-step power spectra. Showing spectral divergence at intermediate T would reveal whether high-frequency error accumulates gradually or sharply.
- **Conservation law verification**: Plotting total energy or enstrophy trajectories from Cohesion vs ground truth over long rollouts would provide concrete evidence supporting the "physically-consistent" claim.
- **Validation on additional PDE systems**: An additional 2D benchmark (e.g., Lorenz-96, 2D Boussinesq convection) would broaden the generalizability argument beyond the two closely related incompressible/shallow systems tested.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **"Unified framework is superficial"** (Harsh Critic): While Section 2 is a taxonomy rather than an algorithmic framework, the framing is conceptually genuine and useful. The paper does make the connection between turbulence theory and diffusion conditioning explicit, which has value. Removed as a standalone weakness (retained in Tiny as a wording suggestion only).

- **Statistical significance / confidence intervals** (Harsh Critic): For chaotic PDE benchmarks with trajectory-level evaluation, single-run results are standard in this field. Demanding confidence intervals across seeds for ensemble forecasts over 25 timesteps is not a standard expectation for this community. Removed.

- **Criticizing that SFNO baselines are "poorly tuned"** (Harsh Critic): The paper explicitly states that SFNO parameters were scaled to "match or exceed those of Cohesion" and three different probabilistic modifications were evaluated. Demanding evidence of hyperparameter tuning beyond this is unreasonable. Removed.

- **Missing real-world weather data (ERA5)** (Spark Finder): The paper's stated scope is chaotic PDE emulation on KF and SWE. Requiring ERA5 validation is scope creep that goes beyond the paper's stated contribution. Moved to nice-to-have only.

- **Higher resolution scaling** (Spark Finder): 120×240 for SWE is not a "toy" resolution. Demanding larger resolution benchmarks is a generic scalability request not specific to a methodological flaw in this paper. Removed as a weakness.

- **Climatology preservation (long-term statistics)** (Spark Finder): Reasonable longer-term goal, but outside the stated scope of trajectory-level forecasting. Removed as a weakness; not included as nice-to-have since it is a different task entirely.

- **DPS/Rozet–Louppe not acknowledged as novel** (Harsh Critic): The paper explicitly cites and acknowledges Chung et al. (2022) and Rozet & Louppe (2023) as the basis for its conditional sampling procedure (see Section 3.1: "Following works from Rozet & Louppe (2023); Qu et al. (2024), we improve the numerical stability…"). The paper does not claim novelty in the inference procedure itself. This criticism is factually incorrect. Removed.

---

## Novel Insights

The most genuinely novel observation across the three reviews, extending slightly beyond the paper's own claims: the mechanistic decomposition illustrated in Figure 10 — that Cohesion sequentially recovers low-wavenumber coherent structures first and then high-wavenumber fluctuations during denoising — mirrors the physical coarse-to-fine cascade of energy in turbulent systems. This is not merely a design choice but an emergent property that provides indirect validation that the Reynolds decomposition framing is not just a metaphor but is physically operative in the model's internal dynamics. Future work could exploit this by designing adaptive NFE schedules that allocate more denoising steps to high-wavenumber correction, potentially improving spectral fidelity without increasing total compute.

---

## Suggestions

1. **Correct the runtime claim**: Replace "orders-of-magnitude speedups" with the accurate figure (e.g., "up to 13× speedup") and add a column in Figure 12 showing absolute time relative to SFNO to contextualize the efficiency claim.
2. **Add at least one autoregressive diffusion baseline**: Even a simple DDPM or EDM-based model applied autoregressively at R=1 on KF would clarify how much of the performance benefit comes from the diffusion architecture vs. the ROM conditioning strategy.
3. **Add CRPS or energy score**: Include at least one probabilistic calibration metric (CRPS is standard and cheap to compute for ensemble forecasts) to validate the uncertainty quantification claim.
4. **Rename the noise variable in Eq. 7**: Use ε or η instead of c to eliminate the notation collision with the conditioning prior.
5. **Add W and γ ablations**: Even a simple 2×2 ablation table varying W∈{1,5} and γ∈{0.1,1.0} would substantially strengthen the design justification.
6. **Replace "RL principles"**: Replace with "trajectory planning" or "sequence-level conditional generation" throughout, and clearly scope the RL analogy to the structural parallelism with Janner et al. (2022) rather than implying RL algorithms are used.
7. **Add a limitations section**: Characterize at least ROM error regime limits (e.g., does performance degrade significantly above Re=10³?) and memory scaling for R=T at high resolutions.