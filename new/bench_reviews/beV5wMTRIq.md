Now I have sufficient calibration data. Let me compose the final review.

**Calibration summary:**
- ClimODE (the base model this paper builds upon): Accepted with oral, scores 8/8/8/8. Strong physics-informed approach, but even there reviewers noted limited baselines (only ClimaX and NODE) and limited physical complexity.
- WeatherODE (similar neural ODE weather forecasting, similar overclaim pattern): Rejected, scores 3/6/5/1/3. Key issues: oversimplified physics, weak baselines, overclaiming.
- PACE (similar physics-informed climate emulator, overclaims): Withdrawn (rejected), scores 1/3/3/3/5/3. Key issues: overclaims physics grounding, poor presentation, missing comparisons.
- PhyDL-NWP: Rejected, scores 6/3/3/5. Weak baselines, small-scale experiments, unclear method sections.
- DLWP Benchmark: Rejected, scores 8/3/5/3. Coarse resolution concerns, need stronger baselines.

PA-TFNP shares many weaknesses with the rejected papers: oversimplified physics, weak baselines, overclaiming. It improves upon ClimODE's framework but the core method has significant gaps between claimed and actual contributions.

Let me now verify the key claims against the paper text before writing the review.

Key point to verify: Is the "78.92%" claim traceable? Let me check the paper again.

The abstract says "outperforming ClimODE by 78.92% on global hourly data" - this number is NOT clearly traced in the main text. The experiments show RMSE improvements for individual variables and lead times, but a single 78.92% figure is not shown anywhere.

Key point: Is the claimed rotation equivariance actually substantiated? The equation in Section 3.2 shows:
f_TFN(I[i,c_out]) = I⊗I = Σ W[c_out,c1,c2](I[i,c1]·I[i,c2])

This is a pointwise bilinear interaction on channels - no spatial coupling, no spherical harmonics, no irreducible representations, no Clebsch-Gordan coefficients. As the harsh reviewer notes, this is not how standard TFNs achieve equivariance. The paper doesn't substantiate that this operator is equivariant under SO(3).

Key point: Does the paper truly operate "directly on the sphere"? The paper actually operates on a lat-lon grid with padding and metric-corrected finite differences. The abstract claims "directly on the sphere" which is misleading.

Key point: The "primitive equations" connection: The paper adds a diffusion term α(x)Δq and a momentum correction f_phys = -∇Φ + νΔu - γu. This is NOT the atmospheric primitive equations, which include Coriolis force, hydrostatic balance, equation of state, continuity equation, etc. The overclaim here is substantial.

Now let me finalize the review.</think>Now I have comprehensive calibration data. Let me compose the final review.

## Summary

The paper proposes PA-TFNP (Physics-Aware Tensor Field Neural PDE), a weather forecasting framework building on ClimODE that introduces three modifications: (1) a tensor-field neural network (TFN) architecture claimed to be rotation-equivariant, (2) a spherical coordinate-aware gradient operator with physically motivated boundary conditions, and (3) physics-inspired diffusion and momentum correction terms blended with neural predictions via a time-dependent schedule. Experiments on ERA5 data at 5.625° and 11.25° resolutions with 5 atmospheric variables show improvements over ClimODE, ClimaX, and NODE across global, regional, and monthly-averaged forecasting tasks.

## Strengths

- **Well-motivated problem and sensible direction**: The paper correctly identifies real limitations in ClimODE's treatment of spherical geometry (polar artifacts), absence of physically consistent boundary conditions, and lack of physics-informed dynamics for long-term stability. Each proposed modification addresses a concrete problem.

- **Consistent empirical improvements**: PA-TFNP outperforms ClimODE and ClimaX across multiple settings (global short-term, global long-term, regional, monthly-averaged), with gains that are particularly pronounced at longer lead times for key variables like geopotential height and atmospheric temperature.

- **Honest limitations section**: The paper explicitly acknowledges that rotation equivariance offers limited benefits for regional forecasting and that applying uniform diffusion to all variables is physically suboptimal — a refreshing admission that strengthens credibility.

## Weaknesses

### Major:

- **The claimed rotation equivariance is not substantiated by the described architecture.** The core TFN operator (Section 3.2) is defined as f_TFN(I[i,c_out]) = Σ W[c_out,c1,c2](I[i,c1]·I[i,c2]). This is a pointwise bilinear mixing of channels at each grid point — there is no explicit dependence on spatial positions, relative orientations, spherical harmonics, or Clebsch-Gordan coefficients that are the standard mechanisms for achieving equivariance in published TFN / e3nn architectures (Thomas et al., 2018; Weiler et al., 2018; Kondor et al., 2018). Without specifying which irreducible representations are used, how inputs are decomposed into scalar/vector/tensor fields, and how spherical rotations act on them, the claim of strict SO(3)-equivariance is unsupported. Since rotation-equivariance is the central "geometry-aware" contribution, this gap is not cosmetic — it undermines the interpretation of ablation results attributing gains to "rotational equivariance."

- **Overstated connection to the atmospheric primitive equations.** Throughout the paper (Abstract, Introduction, Section 3.3, Conclusion), the method is described as embedding "diffusion terms derived from the atmospheric primitive equations" and "enforcing core physical constraints" including "divergence-free conditions." In reality, Section 3.3 adds a generic diffusion term α(x)Δq (with learnable, unconstrained α) and a simplified momentum tendency f_phys = −∇Φ + νΔu − γu. The standard primitive equations include the Coriolis force, hydrostatic balance, an equation of state, continuity equation, and pressure-gradient terms coupled across multiple levels — none of which are present. The paper itself acknowledges (Section 5) that "the modification of the model equation should be tailored to each variable, as their physical interpretations differ significantly," further highlighting the gap between the claimed and actual physical modeling. There is also no explicit divergence-free constraint or projection step anywhere in the formulation, contradicting the claim in the Conclusion.

- **The "78.92% improvement" claim in the abstract is not traceable to any result in the paper.** The abstract states PA-TFNP "outperforming ClimODE by 78.92% on global hourly data with a comparable number of parameters," but no single result, metric, variable, or lead time in the paper produces this figure. The actual RMSE improvements across variables and lead times range widely. This is misleading and should be corrected.

- **Narrow and weak baselines limit claims of "state-of-the-art" performance.** The paper compares only against NODE, ClimaX, and ClimODE. Multiple significantly stronger forecasting models are cited in the introduction (GraphCast, Pangu-Weather, FourCastNet, Aurora) but never included in experimental comparisons. The ClimODE reviews themselves noted that "comparing only with ClimaX is definitely not convincing enough" and "ClimaX is a weak baseline marginally better than ResNet." Without comparison to at least one modern large-scale forecasting model, claims of state-of-the-art performance are unsupported.

- **Insufficient ablation to attribute improvements to specific proposed mechanisms.** The paper ablates only two configurations: ClimODE vs. TFNP (assessing rotation-equivariance) and TFNP vs. PA-TFNP (assessing all physics-aware additions together). There is no isolation of: (i) boundary conditions alone, (ii) spherical gradient correction alone, (iii) physics-derived features (wind magnitude, lapse rate, vorticity) alone, (iv) diffusion term alone, or (v) momentum blending alone. Without per-component ablations, it is impossible to determine whether gains come from the purported physics-aware mechanisms or from generic capacity/information additions.

### Minor:

- **Operates on lat-lon grid, not "directly on the sphere."** The abstract claims the model works "directly on the sphere," but the method actually operates on a standard lat-lon grid with padding and metric-corrected finite differences. This is a standard approach in NWP and non-spectral models, not a novel spherical discretization. The "average padding" strategy is heuristic rather than derived from a rigorous discretization of the sphere.

- **Limited variable set and coarse resolution constrain practical significance.** Only 5 variables at 5.625°/11.25° are tested. Operational and modern research models (GraphCast, Pangu-Weather) work with dozens to hundreds of variables at 0.25°–1° resolution. Whether these improvements transfer to practical settings remains untested.

- **The blend factor β_t = 1 − exp(−t/τ₀) lacks justification.** The functional form and timescale τ₀ are introduced without theoretical or empirical justification, and no sensitivity analysis is provided.

## Nice-to-Haves

- Comparison with at least one modern large-scale forecasting baseline (e.g., GraphCast or Pangu-Weather) on standard WeatherBench configurations.

- Per-component ablation study isolating boundary conditions, spherical gradients, physics features, diffusion, and momentum blending.

- Quantitative verification of physical consistency (e.g., tracking globally integrated mass/energy over rollouts) to substantiate "physical fidelity" claims.

- Analysis of learned physical parameters (α(x), ν, γ, τ₀) to verify whether they produce physically meaningful spatial/temporal patterns.

- Scaling experiments to higher-resolution and more variables.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Missing comparison with operational NWP models (IFS, GFS)**: The paper frames itself as bridging physics-driven NWP and data-driven methods, so comparing against NWP would be ideal — but this is outside the paper's stated scope of improving data-driven methods, and NWP baselines require different evaluation protocols. Moved to Nice-to-Haves as a suggestion rather than a required weakness.

- **Missing uncertainty quantification**: While ClimODE provides uncertainty estimates and this is a genuine advantage, UQ is outside PA-TFNP's stated scope of improving deterministic prediction accuracy and physical consistency. It would strengthen the paper but is not a core flaw.

- **Demands for theoretical proofs of equivariance**: The harsh reviewer demands formal mathematical proof that the architecture is equivariant. While the equivariance claim is indeed unsubstantiated (a major point above), demanding a formal proof goes beyond what is standard for empirical ML papers. What is needed is a clear specification of how the architecture implements equivariance and an empirical equivariance test — not a mathematical theorem.

- **Concerns about reproducibility of hyperparameters**: The paper defers some details to an appendix, which is standard practice. The core model equations are specified, and the dataset and training setup follow ClimODE. This is not a substantive weakness.

- **Formatting/style nitpicks**: Removed per rules.

- **Factually incorrect claim that paper "never reports" 78.92% in the body**: While the number is indeed not traceable to a specific result, the paper does present numeric comparisons in tables and figures. The claim is misleading rather than entirely absent.

## Novel Insights

The paper demonstrates that combining three modifications to ClimODE (coordinate-aware gradients, boundary padding, physics-derived diffusion/blending) yields consistent improvements, particularly at longer lead times where stability matters. However, the ablation evidence is insufficient to determine which of these modifications actually drives the gains. This is a meaningful finding for the neural weather modeling community — even simple physics-inspired additions to neural ODE frameworks can improve long-horizon stability — but the community should be skeptical of attributing these gains specifically to rotation equivariance or primitive-equation grounding without cleaner evidence.

## Suggestions

- Either provide rigorous specification of how the TFN architecture achieves equivariance (irrep decomposition, CG coefficients, etc.) with an empirical equivariance test, or scale back the claim to "geometry-informed" or "coordinate-aware."

- Replace the "78.92%" abstract claim with an honest summary of average RMSE improvement across the tested variables and lead times.

- Conduct per-component ablations to identify which of the several proposed changes actually matter.

- Add at least one comparison with a modern large-scale model (even using published checkpoints at comparable resolution) to contextualize results.

- Be precise about the physical modeling: call it "physics-inspired" rather than "derived from the atmospheric primitive equations," and remove the claim about "divergence-free conditions" unless these are explicitly enforced.

## Score and Decision

**Calibration**:
- ClimODE (the base model, accepted oral): Scores 8/8/8/8. Had similar weaknesses (limited baselines, simplified physics) but stronger novelty (first to propose the neural ODE + advection framework) and cleaner formulation.
- WeatherODE (similar pattern — incremental physics additions to ClimODE-like framework, overclaiming): Rejected with scores 3/6/5/1/3. Key issues: oversimplified physics, unfair comparisons, weak baselines, overclaiming improvements.
- PACE (physics-informed climate emulator, severe overclaims and poor formulation): Rejected with scores 1/3/3/3/5/3.
- PhyDL-NWP (physics-guided, weak baselines, unclear method): Rejected with scores 6/3/3/5.

PA-TFNP shares significant weaknesses with the rejected WeatherODE and PhyDL-NWP papers: oversimplified physics relative to claims, weak baselines, and unsubstantiated core architectural claims. While it shows genuine empirical improvements and tackles important problems, the gap between what is claimed (rotation-equivariant, primitive-equation-derived, operating "directly on the sphere") and what is delivered (pointwise channel mixing, ad hoc diffusion on a lat-lon grid with heuristic padding) is substantial. The paper is somewhat better than WeatherODE in that it has clearer experiments and honest limitations, but the core methodological claims remain unverified.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>