## Summary

Cohesion reframes diffusion-based dynamics forecasting as trajectory planning, using a Koopman-based reduced-order model (ROM) to generate long-horizon conditioning priors that enable single-pass conditional denoising of entire forecast sequences. This avoids costly autoregressive rollout of diffusion models and yields substantial internal speedups (7–13×) while maintaining accuracy. The framework unifies prior–residual decomposition under the lens of Reynolds decomposition from turbulence theory and incorporates temporal composition (flexible subsequence length R) and classifier-free guidance for zero-shot conditioning. Experiments on Kolmogorov Flow and Shallow Water Equation show improved RMSE, MS-SSIM, and spectral divergence over probabilistic SFNO baselines.

## Strengths

- **Novel and well-motivated framework formulation**: The trajectory-planning reformulation—conditioning on cheap ROM-generated priors to denoise entire sequences at once rather than autoregressively—is a meaningful architectural insight that directly addresses a well-known bottleneck (autoregressive diffusion cost). The flexibility to interpolate between R=T (full trajectory planning) and R=1 (standard autoregression) via temporal composition is elegant.

- **Strong empirical results vs. probabilistic SFNO**: Cohesion consistently outperforms all three probabilistic SFNO variants on both Kolmogorov Flow and SWE across RMSE, MAE, MS-SSIM, and spectral divergence. Qualitative results (Figures 4, 6) show visibly more stable and detailed long rollouts.

- **Meaningful spectral evaluation**: Including spectral divergence (Figures 8, 10) goes beyond naive pixel metrics and demonstrates that Cohesion better preserves multi-scale physical structures—a critical property for chaotic dynamics emulation.

- **Clear refinement role demonstrated**: Section 4.3 (Figures 9–10) cleanly shows that the diffusion component adds genuine value on top of the ROM prior, both in improving pointwise metrics and correcting the power spectrum. This is one of the paper's most concrete contributions.

## Weaknesses

### Fatal
None.

### Major

- **No comparison with other diffusion-based or generative PDE emulators**: The paper positions itself against "state-of-the-art probabilistic emulators" but only compares against probabilistic modifications (MC-Dropout, checkpoints, IC-perturbation) of a deterministic SFNO. These are not native generative models. The paper cites numerous diffusion-based forecasting methods (GenCast, PDE-Refiner, DYffusion, etc.) but includes none as baselines. Without such comparisons, the claim that Cohesion "outperforms state-of-the-art probabilistic emulators" is unsupported—it remains unclear whether the advantages come from the specific Cohesion design or simply from using any diffusion model versus SFNO+heuristics. This matters because the paper's core contribution is the trajectory-planning design, not the mere act of applying diffusion.

- **Probabilistic evaluation is essentially deterministic**: Despite claiming to be a "probabilistic emulator" producing "ensemble forecasts," all evaluation uses pointwise metrics (RMSE, MAE, MS-SSIM, spectral divergence) against a single ground truth. No proper probabilistic scoring rules (CRPS, spread-skill, rank histograms, calibration) are reported. With only 5 ensemble members and no calibration analysis, the paper cannot verify whether the ensemble spread meaningfully captures forecast uncertainty. This directly undermines the probabilistic-forecasting framing that is central to the paper's motivation.

- **Efficiency claims are overclaimed and incompletely substantiated**: The abstract claims "orders-of-magnitude speedups." Figure 12 shows only an internal comparison (Cohesion R=1 vs. R=T), yielding 7× and 13× speedups—roughly one order of magnitude at most, only in one system. No absolute wall-clock times or NFE counts are reported relative to SFNO or other diffusion baselines. The crucial comparison—an autoregressive diffusion model with the same score network vs. Cohesion's trajectory planning—is entirely absent, so the core efficiency claim relative to the existing paradigms the paper critiques remains unverified.

### Minor

- **Temporal composition ablation is limited**: Only R=1 and R=T are tested. The paper's framework emphasizes flexible subsequence length R, but no intermediate values (e.g., R=4, R=8, R=16) are evaluated. This leaves open how the accuracy–efficiency frontier actually behaves and whether the trajectory-planning advantage degrades for shorter sequences.

- **Partially-observed conditioning experiment is purely qualitative**: Figure 11 shows visual results for masked priors but provides no quantitative metrics (RMSE, spectral divergence) under varying masking ratios. The zero-shot generalization claim for novel conditioning scenarios is unsupported beyond this single visual illustration with one masking pattern.

- **ROM error propagation and failure modes are unanalyzed**: The ROM is trained with a 1-step loss (Eq. 15) but autoregressively rolled out for T steps to produce conditioning priors. No analysis of ROM error growth over the rollout horizon is provided, and no discussion of what happens when the ROM prior degrades substantially (whether the diffusion refinement can still recover or fails).

- **Missing ablations on key design hyperparameters**: The temporal window size W=5, the Tweedie correction coefficient γ, and the number of Langevin correction steps are presented without any sensitivity analysis. It is unclear whether these are robust choices or require careful tuning per system.

- **The turbulence-theoretic framing adds limited theoretical substance**: The mapping from Reynolds decomposition to "prior + stochastic refinement" is conceptually clean but introduces no new modeling constraints, architectural invariants, or training objectives beyond what existing residual diffusion methods already employ. The title and framing suggest a deeper connection to turbulence theory than the method delivers.

### Trivial
- The conclusion does not acknowledge limitations or failure modes.

## Nice-to-Haves

- Compare Cohesion against at least one native generative PDE baseline (e.g., PDE-Refiner, a standard autoregressive diffusion, or flow-matching approach) with matched compute budget.
- Report CRPS or spread-skill ratios across the ensemble to validate probabilistic calibration.
- Test on a higher-resolution or 3D system to assess scalability beyond 2D grids.
- Include an ablation sweep over R ∈ {1, 2, 4, 8, 16, T} to map the full accuracy–efficiency frontier.
- Report wall-clock time and NFEs for all methods on identical hardware.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"SFNO baselines are unfair/unspecified"** — While baseline specification could be clearer, the asymmetry (probabilistic modifications of a deterministic method vs. a native generative model) does not favor the authors' method. Per the hard rules, criticisms about unfair comparison that favor the baseline are removed. The transparency concern about exact hyperparameters is kept as a minor note above.

- **"No deterministic baseline matched to compute budget"** — This is related to the fairness concern. Adding a compute-matched deterministic baseline would strengthen the paper, but criticizing the absence when the stated scope is probabilistic emulation is scope creep. Moved to nice-to-have.

- **"Reproducibility concerns about implementation details"** — Concerns about undisclosed hyperparameters, training details, and missing NFE counts border on reproducibility nitpicks. The missing NFE counts and wall-clock times are kept under the major efficiency weakness because they directly affect the claimed contribution; minor implementation details are removed.

- **"The paper should evaluate physical conservation laws"** — Evaluating energy/enstrophy conservation would strengthen the physics claims, but the paper's stated scope is forecasting accuracy, not conservation-law verification. Moved to nice-to-have.

- **"Formatting/presentation nitpicks"** — Removed per hard rules.

- **"Scale/compute concern about simultaneously denoising full sequences"** — This is a valid scalability question but is more of a nice-to-have for future work than a weakness of the current contribution on 2D systems.

## Novel Insights

The trajectory-planning perspective—generating long coherent priors with a cheap ROM and then refining the entire sequence in a single diffusion pass—is a genuinely useful reframing that, if it scales, could change how diffusion-based PDE emulators are designed. The empirical finding that the score network's temporal receptive window (W=5) suffices for local consistency while accumulated NFEs produce global consistency is an interesting design insight that challenges the assumption that specialized temporal architectures are necessary for coherent long-range forecasts. The classifier-free guidance mechanism enabling zero-shot conditioning on varying priors (including partially-observed ones) is potentially impactful for real-world applications where observation processes change.

## Suggestions

- Add at least one generative PDE baseline (autoregressive diffusion or PDE-Refiner) to isolate the contribution of the trajectory-planning design versus the benefit of using any diffusion model over deterministic methods.
- Report proper probabilistic metrics (CRPS, spread-skill) alongside pointwise metrics to substantiate the probabilistic framing.
- Replace "orders-of-magnitude speedups" with the measured 7–13× figure, and report absolute wall-clock times and NFEs for all methods.
- Add an ablation over R values and W values to demonstrate the design space actually works as described.

## Score and Decision

**Calibration context**: I compared this paper against human reviews of similar papers. Turbulent Flow Simulation using ACDM (diffusion for PDEs, scored 5/5/5/5, Reject) had similar concerns about missing generative baselines and overclaimed efficiency. Continuous Ensemble Weather Forecasting (scored 5/5/5/5, Accept Poster) had similar novelty but better-positioned experiments. PDE-Diffusion (scored 1-3, Reject) was rejected for much weaker methodology and evaluation. Learning Chaos In A Linear Way (scored 5-8, Accept Poster) had strong Koopman contributions but also baseline concerns. SineNet (scored 6-8, Accept Poster) had solid empirical PDE results with some baseline limitations.

Cohesion has genuine contributions—the trajectory-planning insight is valuable, the empirical results are strong against the baselines presented, and the spectral evaluation is meaningful. However, the three major weaknesses (no generative baselines, no probabilistic evaluation, overclaimed efficiency) significantly undermine the stated claims. This paper is stronger than PDE-Diffusion (which had fundamental methodological issues) but weaker than papers that received 6+ scores with more complete evaluations. It sits in the borderline range similar to papers that scored 5.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>