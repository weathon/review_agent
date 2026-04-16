Now let me check some calibration papers to inform scoring.## Summary

Cohesion is a diffusion-based framework for long-range probabilistic dynamics forecasting that explicitly connects turbulence theory (Reynolds decomposition) with score-based generative modeling. The core idea is to use a lightweight reduced-order model (Koopman ROM) to cheaply generate conditioning priors for entire trajectories, then perform a single joint denoising pass — framed as "trajectory planning" — rather than stepwise autoregressive denoising. The method further incorporates classifier-free guidance for zero-shot conditioning and a windowed temporal convolution for local-global consistency, and is evaluated on Kolmogorov Flow and Shallow Water Equation benchmarks.

---

## Strengths

- **Conceptually elegant trajectory-planning formulation**: The reframing of autoregressive forecasting as trajectory planning via a cheap ROM prior (Sec. 3.3) is genuinely novel and practically useful. The ability to interpolate between R=1 (autoregressive) and R=T (full-pass) without retraining is a meaningful design contribution.

- **Turbulence-diffusion unification as organizing principle**: Explicitly connecting Reynolds decomposition to the conditioning+denoising split in Sec. 2 (Eqs. 2–4) provides a useful conceptual framework that unifies several disparate prior works and motivates Cohesion's design choices coherently.

- **Real inference speedups within the proposed framework**: Fig. 12 demonstrates that trajectory planning (R=T) is ~7× faster than autoregressive (R=1) for Kolmogorov Flow and ~13× faster for SWE, with maintained or improved accuracy (Figs. 5, 7). This is a legitimate and practically important result for long-range forecasting.

- **Consistent improvements across diverse metrics**: Cohesion outperforms all probabilistic SFNO baselines on RMSE, MAE, MS-SSIM, and spectral divergence across two distinct chaotic systems (Figs. 5, 7, 8). The spectral divergence metric is particularly well-chosen for assessing multi-scale physical fidelity.

- **Insightful decomposition of the method's role**: Figs. 9–10 provide genuine mechanistic understanding — showing Cohesion acts as a refiner improving coarse ROM forecasts, and as a "resolver" that progresses from low- to high-frequency spectral content across denoising steps. These analyses are informative and strengthen understanding of the method.

- **Parameter-fair baseline comparison**: The authors scale SFNO's parameters to match or exceed Cohesion's (Sec. 4), which at least controls for model capacity against the chosen baseline family.

---

## Weaknesses

### Fatal
*(None — the paper makes real contributions and the core mechanism is not fundamentally broken. However, the major weaknesses below collectively constitute a significant bar to acceptance.)*

### Major

- **Baseline comparison is too narrow to support the headline "state-of-the-art" claim.** The Abstract and Sec. 1 explicitly state Cohesion "outperforms state-of-the-art probabilistic emulators," yet Sec. 4 compares only against three probabilistic variants of SFNO (Checkpoints, MC-Dropout, IC Perturbation). The Introduction itself surveys a substantial body of diffusion-based probabilistic forecasters — Price et al. (2023), Lippe et al. (2024), Li et al. (2024), Mardani et al. (2024), among others — yet none appear as baselines. Beating ad hoc probabilistic wrappers around a single deterministic operator model does not establish superiority over the diffusion-based forecasting methods the paper claims to surpass and unify. This gap is critical because it is impossible to determine how much of the gain comes from the Cohesion framework versus simply from using a proper generative model rather than a heuristically probabilistic SFNO.

- **The "orders-of-magnitude speedup" claim is factually unsupported by the paper's own evidence.** The Abstract and Conclusion repeatedly invoke "orders-of-magnitude speedups," but Fig. 12 shows that R=T is ~7× faster than R=1 for KF and ~13× faster for SWE. This is at most one order of magnitude, and the comparison is exclusively *between two modes of the proposed method*, not against any competing probabilistic forecaster. If other diffusion-based methods require T×K NFEs (with K denoising steps per timestep), the comparison against them could support a stronger claim — but this comparison is never made. As written, the claim is an overclaim relative to the evidence presented.

- **Probabilistic evaluation is insufficient for a paper centered on uncertainty quantification.** Despite being framed as a probabilistic emulator for uncertainty quantification (Abstract; Sec. 1), the paper evaluates probabilistic performance with only five ensemble members and no calibration metrics, no proper scoring rules (e.g., CRPS, log-likelihood), no reliability diagrams, and no ensemble spread-skill analysis. RMSE, MAE, MS-SSIM, and spectral divergence are predominantly sample-quality or deterministic fidelity metrics. A probabilistic emulator that is poorly calibrated or systematically over/under-dispersive would score well on these metrics while failing its stated purpose. This is a methodological gap that directly weakens the paper's central claim.

- **"Zero-shot conditioning for a broad range of scenarios" is asserted but not demonstrated.** The paper prominently claims classifier-free guidance enables "zero-shot forecasts given different conditioning scenarios" (Abstract; Sec. 3.1). The only empirical evidence is Fig. 11, which shows a single qualitative experiment with equally-spaced masking — no alternative observation operators, noise levels, sparsity patterns, or quantitative metrics are reported. The approximation in Eq. 9 (Gaussian observation model) underpins the zero-shot claim, but the paper does not characterize the class of conditioning scenarios for which this approximation is reliable. The claim as stated substantially exceeds the evidence.

### Minor

- **ROM is trained on a 1-step reconstruction loss but used autoregressively for long sequences.** Eq. 15 minimizes a 1-step lag loss only. There is no analysis of how ROM forecast quality degrades over time, whether ROM drift corrupts the conditioning prior at long lead times, or whether Cohesion's diffusion stage can recover from a significantly degraded prior. Since the entire trajectory-planning formulation (R=T) depends on the ROM generating useful priors over T steps, understanding its failure modes is necessary.

- **Key design choices are not ablated.** The paper introduces multiple novel components (temporal window size W, Langevin correction steps, CFG guidance strength γ, refinement length R) but provides no systematic ablation isolating their contributions. The comparison in Fig. 9 (coherent-only ROM vs. +Cohesion) is useful but conflates the contributions of the diffusion model, the temporal window, and the trajectory-planning inference scheme. Readers cannot determine which element drives the improvements.

- **Spectral divergence is evaluated only at the final timestep.** Fig. 8 reports spectral divergence at T=Δt only. A model could preserve spectra at intermediate steps while diverging at the end, or vice versa. Reporting spectral divergence trajectories over rollout time would more rigorously support the multi-scale consistency claim.

- **The RL/trajectory-planning framing is superficial.** The paper invokes trajectory planning terminology from RL (citing Janner et al. 2022) but there is no reward function, no planning algorithm, and no policy — it is sequence-level conditional generation. The RL language may mislead readers about the nature of the method.

### Trivial

- **Equation 7 notation collision**: In Eq. 7, `p(c) ~ N(0,I)` appears in the expectation, but `c` is used throughout the paper as the ROM-generated conditioning prior. The expression `‖ε_θ(·) − c‖²` sets `c` as the prediction target in a noise-prediction parameterization context. This collision between `c` (conditioning prior) and `c` (noise target) is confusing and should be clarified (e.g., using `ε` for the noise target).

---

## Nice-to-Haves

- **Real-world benchmark (e.g., WeatherBench/ERA5)**: The paper repeatedly gestures toward weather/climate applicability, but both benchmarks are synthetic PDEs. Even a lightweight ERA5 experiment would substantially strengthen the application claims.
- **Absolute wall-clock times**: Figure 12 reports only relative runtimes within the proposed method. Reporting absolute times (and comparing to at least one other diffusion-based forecaster) would make the computational advantage claim actionable.
- **Quantitative partial-observation evaluation**: Fig. 11 would be far more convincing with RMSE, MS-SSIM, or spectral divergence under partial observation, even just for the masking scenario shown.
- **Ensemble diversity analysis**: Rank histograms or spread-skill diagrams across the rollout horizon would allow readers to judge calibration quality.
- **Scalability discussion**: The R=T mode requires denoising an entire sequence jointly, which scales memory with T. Discussing memory constraints for higher-resolution or longer-horizon applications (e.g., global weather at 0.25°) would help practitioners assess feasibility.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic — Section 2 "unified framework is purely metaphorical"**: The critic claims the turbulence-diffusion connection "yields no new derivations, guarantees, or testable consequences." However, the paper is transparent that this is a *unifying lens* (Sec. 2: "we make the connection explicit"), not a formal theory. The framework does motivate the design choices (Koopman ROM as coherent-flow prior, stochastic refinement) and unifies prior work conceptually. Criticizing expository framing as lacking formal guarantees is scope creep for an empirical systems paper. **Removed as scope creep.**

**Harsh Critic — "learned decomposition does not have the physical meaning implied by Reynolds terminology"**: The paper does not claim the latent Koopman representation is physically equivalent to a Reynolds-filtered field — it uses the decomposition as a structural analogy. Criticizing the analogy for not having formal physical grounding attacks a framing device, not a scientific claim. **Removed.**

**Harsh Critic — "predictor-corrector in Eqs. 11–12 is not clearly marked as standard"**: The paper cites Song et al. (2020) and Zhang & Chen (2022) directly for these equations. The lack of a bold "we did not invent this" disclaimer is normal academic writing. **Removed as style nitpick.**

**Neutral Reviewer — "Missing analysis of conservation laws (mass conservation, divergence-free error)"**: While physical conservation properties are important, neither Kolmogorov Flow nor SWE experiments are set up to require divergence-free enforcement as a primary benchmark criterion in the ML emulation literature. The spectral divergence metric already captures physical fidelity; demanding PDE-specific conservation analysis is outside the paper's stated scope. **Removed as scope creep.**

**Human Finder — "Limited discussion of long-horizon error accumulation" citing very short rollouts**: The paper evaluates T=25 for KF and T=32 for SWE, which constitute the full Δt for these benchmark settings. Characterizing these as "extremely short" mischaracterizes the benchmark setup. **Removed as factually inaccurate criticism.**

**Human Finder — ROM Koopman connection "very loose"**: Valid as a general observation about deep Koopman methods, but not specific to claims the paper makes — the authors do not claim formal Koopman theory guarantees, and the ROM's downstream utility is demonstrated empirically. **Removed as generic criticism not targeted to this paper's claims.**

---

## Novel Insights

The most genuinely novel observation across all three reviews is the identification of a structural asymmetry between the paper's ambitions and its evaluation: Cohesion is framed as unifying and outperforming a family of diffusion-based forecasters, but is only evaluated against a non-diffusion baseline. This means the paper's most important claim — that trajectory planning with ROM priors yields advantages over existing generative forecasting approaches — is precisely the one left untested. The within-method ablation (ROM-only vs. +Cohesion) is informative but orthogonal to the comparative claim. If future work adds the missing baselines and proper probabilistic metrics, the paper's core contribution could either be validated strongly or substantially reframed.

---

## Suggestions

1. **Replace the "orders-of-magnitude" language** with the empirically supported "~7–13×" and add a comparison of wall-clock times against at least one published diffusion-based forecasting method.
2. **Add at least one strong diffusion-based baseline** (e.g., PDE-Refiner/Lippe et al., which is already cited and closely related in problem formulation) to substantiate or appropriately qualify the "state-of-the-art" claim.
3. **Report ensemble calibration metrics**: CRPS or rank histograms with a larger ensemble (≥20 members) would validate the probabilistic emulator claim. Without this, the probabilistic framing cannot be assessed.
4. **Quantify the partial-observation experiment**: Add RMSE and spectral divergence for Fig. 11 under various masking rates to replace the purely qualitative claim.
5. **Ablate W, Langevin correction steps, and γ**: At minimum, show sensitivity to the window size W and the number of predictor-corrector steps, as these are the least-obvious hyperparameters.
6. **Characterize ROM long-horizon drift**: Show how ROM RMSE evolves as a function of autoregressive steps and how Cohesion's refinement quality depends on ROM quality. This would bound the operational regime of the trajectory-planning approach.

---

## Score and Decision

**Calibration against retrieved papers:**

- *1hhja8ZxcP (Turbulent Flow with ACDM, Reject, avg 5.0)*: That paper also applied diffusion to turbulence with limited baselines and was rejected for not going beyond straightforward application. Cohesion is more novel (trajectory planning, temporal composition) but similarly limited in baselines. Cohesion is somewhat above this work.
- *ZhlwoC1XaN (From Zero to Turbulence, Accept Poster, avg 6.75)*: That paper introduced a new 3D turbulence dataset, stronger empirical novelty, and cleaner claims. Cohesion has more methodological novelty but weaker empirical backing and significantly overclaims.
- *fkrYDQaHOJ (Koopman RL, Accept Poster, avg 5.5)*: Accepted despite a weak reviewer (score 3) because other reviewers recognized the theoretical contribution. Cohesion has similar claim-evidence mismatches but fewer formal results.
- *uKZdlihDDn (Graph Diffusion Fluids, Accept Oral, avg 7.6)*: Substantially stronger empirical setup, handles unstructured meshes, real engineering applications. Cohesion does not reach this bar.

**Assessment**: Cohesion occupies the range between the rejected ACDM paper and the weaker poster accepts. The framework idea is interesting and the improvements over SFNO are real, but three major claims are simultaneously overclaimed or unsupported: the speedup magnitude, the state-of-the-art comparison, and the broad zero-shot capability. The probabilistic framing is not validated with appropriate metrics. This combination falls below the acceptance bar.

**Axes:**
- *Originality*: Moderate-to-good (trajectory planning + ROM prior for diffusion is fresh; temporal composition elegant)
- *Importance of research question*: High (long-range probabilistic PDE forecasting is important)
- *Claims well supported*: Weak (three headline claims overclaimed or undervalidated)
- *Soundness of experiments*: Fair (internally consistent but too narrow baseline suite, thin probabilistic evaluation)
- *Clarity of writing*: Good, with one notation issue
- *Value to research community*: Moderate (interesting framework, but the community needs proper comparative evaluation before adoption)

**Final Score: 4.5**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>