Now I have a good sense of the calibration landscape. Let me now synthesize the review carefully.

Key calibration data:
- PG-Diff (diffusion for CFD super-resolution): Scores 3-6, Reject
- Text2PDE (latent diffusion PDE): Scores 3-6, Accept (Poster)
- Discretization-invariance neural operators: Scores 5-8, Accept (Poster)
- DIFOCON (diffusion control): Scores 3-5, Reject
- Multi-scale wavelet generative modeling: Scores 1-8, Reject
- Physics-enhanced Neural Operator: Scores 3-6, Reject

The WDNO paper has strong empirical results, a well-motivated method, and comprehensive evaluation. The main issues are: (1) the approximate scale invariance argument is heuristic, not rigorous, (2) the control evaluation lacks some details and variance reporting, (3) the wavelet domain's interaction with Gaussian diffusion assumptions is unexamined, (4) some baselines for super-resolution and ERA5 are limited. However, the paper makes genuine contributions and demonstrates substantial improvements on hard problems. It's clearly above the level of rejected papers like DIFOCON or PG-Diff.</think>## Summary

WDNO proposes performing diffusion-based generative modeling in the wavelet domain for PDE simulation and control, paired with a multi-resolution training scheme that exploits approximate scale invariance to enable zero-shot super-resolution. The method is evaluated on five physical systems (1D advection, 1D Burgers', 1D compressible Navier-Stokes, 2D incompressible fluid, ERA5), showing substantial improvements over baselines including DDPM, FNO, WNO, and various RL/IL methods, particularly on tasks involving abrupt changes and indirect control.

## Strengths

- **Well-motivated and coherent dual innovation**: Wavelet-domain diffusion directly addresses the known difficulty of capturing abrupt changes (shocks, discontinuities) that arise in PDE dynamics, and multi-resolution training exploits the locality property of wavelets for cross-resolution generalization. These two components are integrated logically—wavelets provide the basis for multi-resolution decomposition, and the diffusion model leverages the structured wavelet coefficients for generation. The ablation in Figure 4c (multi-resolution in wavelet vs. original domain) provides concrete evidence of their synergy.

- **Substantial empirical improvements on genuinely challenging problems**: The 25× MSE reduction over DDPM on 1D compressible Navier-Stokes (Table 1) and the 78% reduction in smoke leakage on 2D indirect control (Table 2b) are striking results. The 2D control experiment involves 3,584 spatial control variables over 32 timesteps with indirect boundary forcing—a setting far more demanding than typical RL benchmarks. These are not marginal gains on easy tasks.

- **Comprehensive ablation suite**: The paper systematically ablates the wavelet vs. Fourier transform (Figure 5c), per-timestep error analysis near shocks (Figure 6), multi-resolution + wavelet synergy (Figure 4c), long-term error growth (Figure 5a), noise robustness (Figure 5d), and data efficiency (Figure 5b). These collectively support the claim that the benefits arise from the specific design choices rather than general diffusion model capacity.

- **Diverse evaluation scope**: Testing across five systems (smooth advection, shock-forming Burgers', compressible Navier-Stokes, 2D incompressible fluid, ERA5) covering both 1D and 2D problems, simulation and control tasks, and synthetic and real-world data demonstrates breadth.

## Weaknesses

### Major

- **The "approximate scale invariance" justification for zero-shot super-resolution is heuristic and not rigorously validated**: Section 3.2 motivates multi-resolution training by noting that downsampling changes coordinates and rescales the PDE (Eq. 6), but this is a mathematical identity about coordinate transformation, not a statement about how the *solution structure* relates across resolutions. For PDEs with shocks, fine-scale physics, or strong scale-dependent dynamics, the discretized solutions at different resolutions can differ qualitatively (e.g., shock thickness, numerical dissipation). The paper does not provide quantitative bounds on approximation error, nor does it test whether the "learned pattern" genuinely generalizes PDE dynamics beyond the training resolution family. The super-resolution experiments (Section 4.6) evaluate by interpolating all outputs to a single finest grid, but it is unclear whether the finest-resolution ground truth is independently solved at that resolution or merely upsampled. If ground truth at the target resolution is independently computed, this should be stated explicitly; if not, the task reduces to learning an interpolator. This matters because zero-shot super-resolution is presented as a headline contribution and the paper claims it addresses a "fundamental requirement of a neural operator."

- **The control evaluation protocol is under-specified for interpreting the magnitude of claimed improvements**: For the 2D fluid control (Section 4.4, Table 2b), the paper reports WDNO achieves J=0.0679 vs. second-best 0.3124, but no standard deviations or confidence intervals are provided across seeds or initial conditions. More critically, Eq. 4 computes ∇_{W_f} J at every denoising step, but the paper does not specify whether gradients are computed by differentiating through a PDE solver, a learned simulator, or approximately (e.g., finite differences). The number of diffusion steps, the computational cost per step, and whether all baselines use the same rollout environment are not detailed. Given that the paper highlights the 78% reduction as a key result, these omissions weaken confidence in the exact magnitude of improvement.

- **The wavelet domain's interaction with the Gaussian diffusion framework is unexamined**: Wavelet high-frequency coefficients are sparse and non-Gaussian, which conflicts with DDPM's Gaussian noise assumptions (as noted in prior work on wavelet-domain generative modeling). The paper applies noise and denoising uniformly on all wavelet coefficients without analyzing whether this distributional mismatch hurts generation quality. There is no empirical analysis of the coefficient distributions or discussion of whether different noise schedules, coefficient-wise weighting, or separate treatment of scaling vs. detail coefficients would help. This is a substantive gap because the entire method rests on performing diffusion in the wavelet domain.

### Minor

- **Limited baselines for the ERA5 experiment and super-resolution evaluation**: For ERA5, WNO fails to converge and only single-variable (temperature) prediction over a short horizon (12→20 hours) is tested. For super-resolution, the baselines are FNO, WNO, and interpolation—recent resolution-generalizable operators (e.g., SFNO, U-FNO) are not compared. These limitations temper the strength of claims about real-world applicability and resolution generalization.

- **Burgers' simulation shows DDPM slightly outperforming WDNO**: In Table 1, DDPM achieves MSE=0.00013 on 1D Burgers' vs. WDNO's 0.00014, even though Burgers' is the canonical shock-forming equation. The paper attributes WDNO's advantage to better handling of abrupt changes, but the overall MSE comparison on Burgers' does not support this. The per-timestep MAE analysis (Figure 6) partially addresses this, but the mixed signal should be acknowledged more transparently.

- **No computational cost or inference time comparison**: The paper adds wavelet transforms and a super-resolution model on top of DDPM, but no wall-clock time, memory usage, or function evaluation counts are reported in the main text (appendix only). This makes it difficult to assess the practical trade-off between accuracy and efficiency relative to cheaper alternatives like FNO.

### Trivial

- The notation switches between $W_a$ and $W_a$ (both subscripts for wavelet-transformed values and initial conditions) in Section 3.1, which could momentarily confuse readers.

## Nice-to-Haves

- A formal analysis or proposition specifying under what PDE conditions the scale transformation preserves equation form and quantifying approximation error would strengthen the zero-shot super-resolution claims.
- Ablation on wavelet basis choice (bior2.4 and bior1.3 are used without justification or comparison to alternatives).
- Visualization of learned wavelet coefficients vs. DDPM features to illuminate *why* the wavelet domain helps beyond "it works better."
- Testing on non-uniform or irregular grids, even via simple projection, to address the acknowledged limitation.
- Report of variance/standard deviations across runs, especially for control tasks.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that models/tools/benchmarks are unreleased or unavailable**: Several reviews question the availability or verification of baselines or datasets. Since the paper cites them, they are assumed to exist per our rules.

- **Demand for comparison with "recent diffusion-based PDE solvers (e.g., PDE-Refiner, DiffPDE)"**: The paper already compares against DDPM and multiple neural operators. Demands for specific additional baselines that might or might not exist in the literature are speculative and not grounded in verified knowledge.

- **Demand for physics-informed loss terms**: The paper explicitly acknowledges this as a limitation (Section 5). Criticizing its absence is scope creep—the paper's stated contribution is wavelet-domain diffusion and multi-resolution training, not physics-informed regularization.

- **Claim that DDPM outperforming WDNO on Burgers' "directly contradicts" the abrupt changes narrative**: This overstates the issue. The per-timestep analysis in Figure 6 shows WDNO has lower error during shock periods, and the overall MSE difference (0.00013 vs 0.00014) is negligible. The paper's narrative is about better handling of sharp features, which is supported by the compressible Navier-Stokes and 2D fluid results.

- **Demand for "failure cases or worst-case trajectory analysis"**: This is a generic suggestion applicable to any paper and not specific to a demonstrated failure mode.

- **Claim that the approximate scale invariance argument is "essentially vacuous"**: While the theoretical argument is indeed heuristic and informal (which is a valid weakness), calling it "vacuous" goes too far—it correctly identifies that linear coordinate rescaling produces a consistent pattern across resolutions, which is a meaningful observation even without formal bounds.

## Novel Insights

Beyond the paper's own contributions, a cross-cutting insight emerges: the performance gap between WDNO and DDPM widens dramatically on PDEs with strong discontinuities (25× on compressible Navier-Stokes, order-of-magnitude on 2D fluid) but narrows on smooth problems (near-identical on advection and Burgers'). This suggests that the wavelet-domain approach's primary value proposition is specifically for non-smooth dynamics, and future work might productively focus on adaptive or hybrid representations that switch between spatial and wavelet domains depending on the local smoothness of the solution, rather than committing entirely to one.

## Suggestions

- Provide quantitative bounds or at least empirical verification of approximate scale invariance for the specific PDEs used (e.g., measure how well PDE solutions at different resolutions are related by the coordinate rescaling described in Eq. 6). Even a simple table showing the error between rescaled coarse solutions and fine solutions would clarify what the multi-resolution model is actually learning to correct.
- Report inference time and number of diffusion steps for all methods, and acknowledge the inherent speed-accuracy tradeoff relative to deterministic simulators.
- Include standard deviations across multiple runs or seeds for all reported metrics, especially for stochastic methods like diffusion models and RL baselines.
- Analyze and report the distribution of wavelet coefficients (scaling vs. detail) in the training data, and discuss how the Gaussian diffusion process interacts with their non-Gaussian statistics.

## Score and Decision

**Calibration papers considered:**

| Paper | Topic | Scores | Decision |
|-------|-------|--------|----------|
| PG-Diff | Diffusion for CFD super-resolution | 3-6 | Reject |
| Text2PDE | Latent diffusion for PDE | 3-6 | Accept (Poster) |
| Discretization-invariance neural operators | Resolution generalization in neural operators | 5-8 | Accept (Poster) |
| DIFOCON | Diffusion-based control | 3-5 | Reject |
| Multi-Scale Wavelet Generative | Wavelet domain generative modeling | 1-8 | Reject |
| Physics-Enhanced Neural Operator | Neural operator + physics | 3-6 | Reject |

WDNO is substantially stronger than the rejected papers (DIFOCON, Multi-Scale Wavelet Generative, Physics-Enhanced Neural Operator) in terms of empirical rigor, breadth of evaluation, and clear methodological contribution. However, it has meaningful weaknesses: the heuristic theoretical justification for zero-shot super-resolution, the under-specified control evaluation, and the unexamined Gaussian/sparsity mismatch in wavelet diffusion. These prevent it from reaching the level of the top-accepted papers like the discretization-invariance paper (which had formal analysis). The empirical contributions are strong and the method is well-motivated, with results that advance the state of the art on challenging PDE benchmarks. This places it in the range of mid-tier accepted papers—clearly above the bar, but with gaps that limit its impact.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>