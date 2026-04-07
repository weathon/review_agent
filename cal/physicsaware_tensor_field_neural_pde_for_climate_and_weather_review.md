=== CALIBRATION EXAMPLE 27 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title claims "rotation-equivariant tensor-field neural operators directly on the sphere," but the actual implementation works on a discretized latitude–longitude grid — not directly on the sphere. The gap between the high-level claim and the technical reality is significant. Additionally, "spherical transforms" in the abstract implies Legendre/Fourier-based spherical harmonic decomposition, whereas Section 3.3 reveals this amounts to standard finite differences with a latitude-dependent cosine correction factor. This is a meaningful overstatement.

The headline result — "outperforming ClimODE by 78.92% on global hourly data" — is extraordinary by weather-forecasting standards, where even 5–10% RMSE improvements are noteworthy. The abstract provides no context for what "78.92%" means (percent reduction in RMSE?) nor any acknowledgment that this comparison is against a single, narrowly-scoped baseline.

---

### Introduction & Motivation

The motivation is broadly reasonable: purely data-driven approaches lack physical consistency and generalize poorly. However, the contribution bullets repeat the abstract without adding precision. The third bullet — "embed diffusion dynamics informed by the atmospheric Primitive Equations" — obscures the fact that this is a phenomenological diffusion term with a learnable coefficient, not a derivation from the full primitive equations (which include the continuity equation, thermodynamic equation, and geostrophic balance — none of which are properly embedded).

The paper claims advantages in computational efficiency ("demanding significantly fewer computational resources"), but no FLOPs, training-time, or memory comparisons are provided anywhere in the paper.

---

### Method (Section 3)

**3.2 Tensor Field Neural PDE:**  
The tensor product operation described (Equation for f_TFN) is a learnable bilinear form:

> `f_TFN(I[i, c_out]) = ΣΣ W[c_out, c1, c2] (I[i, c1] · I[i, c2])`

This is a straightforward channel-wise outer product with a learned weight tensor. The Thomas et al. (2018) Tensor Field Network from which this is adapted uses spherical harmonics and Clebsch–Gordan coefficients to construct *strictly SO(3)-equivariant* feature representations. The paper never demonstrates that this bilinear formulation inherits that equivariance, nor that it respects the group structure of rotations on the 2-sphere SO(3) vs. the symmetry group of the lat-lon grid. The argument in Figure 1 (partitioning into four regions A,B,C,D and claiming rotation equivariance via reflection-coupling) is informal and lacks a rigorous proof or even a quantitative verification. The key theoretical claim of the paper is thus unsupported.

**3.3 Boundary Conditions:**  
Applying Neumann padding at the poles and circular padding along longitude is standard and physically well-motivated. However, the "average padding" strategy — padding the pole with the mean of the outermost latitude ring — is not derived from any physical boundary condition. At the true pole, the average-of-ring argument treats all longitude values as equivalent contributions, which is geometrically reasonable but not a *Dirichlet* or *Neumann* condition in the PDE sense. The paper does not characterize what PDE boundary condition this approximates.

**Spherical Gradient (Eq. 3):**  
The central-difference formula with a cosine correction for longitude is standard in atmospheric modeling and geo-data processing. Calling this "a numerically rigorous spherical-transform-based gradient operator" elevates standard practice into a claimed contribution. The denominator `R·h·π·cos(ϕ)/180` is simply the metric factor for converting angular grid spacing to arc-length, not a spherical transform.

**Modified Primitive Equation:**  
The time-blending formula  
> `∂u_i/∂t = (1 − β_t) f_η(…) + β_t f_phys(x, t, u_i)`  
where `β_t = 1 − exp(−t/τ_0)` gradually shifts *away from* neural inference and *toward* physical operators over time.

This is physically counterintuitive. At short lead times, data-driven models are typically most accurate (initial conditions are well-constrained), and physical operators can compensate for model drift at longer lead times. Here the model *increases reliance on f_phys* as forecast time grows — but f_phys is a simple pressure-gradient + viscosity + drag operator that knows nothing about baroclinic instability, latent heat release, or any of the mechanisms that dominate weather evolution. There is no ablation of τ_0, no justification for this functional form, and no discussion of what happens in the limit t → ∞ (full reliance on f_phys).

The f_phys operator itself:  
> `f_phys = −∇Φ + ν∆u_i − γu_i`  
is a simplified Ekman-layer momentum equation with linear drag and viscous diffusion. This is a valid simplified model of boundary-layer dynamics, but claiming it is derived from the *primitive equations* is an overreach — the primitive equations include the full Coriolis term, vertical pressure gradient, and hydrostatic balance, none of which appear here.

The scalar diffusion coefficient α(x) ∈ R^{d×H×W} has 5×32×64 = 10,240 learnable parameters. For a model designed for physical consistency, having a spatially unconstrained learnable diffusion coefficient can fit noise. No analysis of learned α(x) patterns or their physical plausibility is provided.

---

### Experiments (Section 4)

**Baseline Selection:**  
The primary comparison throughout is against ClimODE. GraphCast (Lam et al., 2023), FourCastNet (Kurth et al., 2023), Pangu-Weather (Bi et al., 2023), and Aurora (Bodnar et al., 2024) are all cited in the introduction and related work but never evaluated against. These are the state-of-the-art models in the field. Comparing mainly against ClimODE — a model the authors explicitly build upon — substantially narrows the scope of the contribution's validation.

ClimaX appears in some tables but is inconsistently included: it is absent from Figure 3's global forecasting comparison, present in Table 1 (regional), present in Table 2 (monthly), but then the 78.92% improvement claim is stated only relative to ClimODE.

**The 78.92% Claim:**  
This number is unexplained. It is not clear whether this is a mean reduction in RMSE across all variables and lead times, or for a specific variable or horizon. For geopotential height z, even ClimODE substantially outperforms ClimODE in some settings. The aggregate statistic must be clearly defined with its formula. Furthermore, if the improvement is driven primarily by better polar-region predictions (where ClimODE exhibits known artifacts), the 78.92% figure conflates boundary correction with general forecasting improvement.

**Table 1 (Regional):**  
Several cells in the table are garbled by the PDF parser (showing "_._ **8** _._ ±" without leading digits for the z variable rows), but even from the readable rows, the picture is mixed:
- For t2m (ground temperature) at short lead times (6h, 12h, 18h), PA-TFNP underperforms ClimODE in both Australia and South America.
- The paper acknowledges this but attributes it vaguely to "trade-off between local variance sensitivity and longer-horizon stability" without further analysis.

**Table 2 (Monthly):**  
- For t at month 2: PA-TFNP (2.44) is marginally worse than TFNP (2.42) — the physics-aware extension does not help here.
- For u10 at month 1: ClimaX (1.80) outperforms both TFNP (1.86) and PA-TFNP (1.83).
- For v10 at month 2: ClimaX (1.71) outperforms PA-TFNP (1.91).

These cases suggest the gains are selective and the model is not uniformly superior, contrary to the paper's framing.

**Resolution and Scope:**  
All experiments are at 5.625° (32×64 grid) or 11.25° (16×32 grid). These are extremely coarse resolutions with no operational relevance. Modern ML weather models operate at 0.25° (721×1440 grid). The scalability of the tensor product formulation — which has O(N · C²_in · C_out) cost — to higher resolutions is not analyzed.

**Ablation Study (Section 4.4):**  
The paper presents two ablation comparisons: (1) ClimODE vs. TFNP (testing rotation equivariance), and (2) TFNP vs. PA-TFNP (testing physics augmentation). However, PA-TFNP adds *four* simultaneous changes relative to TFNP: boundary conditions, corrected gradients, three physics-derived features, and the modified primitive equation. There is no isolation of individual contributions. It is impossible to determine from the provided ablations whether the gains come from the boundary conditions alone, the diffusion term, the vorticity feature, or the blending mechanism.

**Statistical Significance:**  
Results in the tables are reported as mean ± standard deviation over multiple runs/test sets. However, Figure 3 (the primary global comparison) presents results as curves without error bands. No statistical tests (e.g., paired t-tests) are reported.

---

### Writing & Clarity

Section 3.2 contains a critical exposition gap: the tensor product formula is presented without explaining *how* the bilinear form over a lat-lon grid achieves rotation equivariance. The four-region A/B/C/D argument in Figure 1 is intuitive but not mathematically complete — it describes what should happen but not how fTFN achieves it. A reader trying to understand or replicate the equivariance property would be unable to do so from the paper's description alone.

The abstract states PA-TFNP "learns directly on spherical tensor fields," but the methodology section makes clear the model operates on a Euclidean lat-lon grid with corrections. This inconsistency should be resolved.

---

### Limitations & Broader Impact

The paper acknowledges two important limitations: (1) rotation equivariance offers limited benefits for regional forecasting, and (2) the same diffusion term is applied to all variables regardless of their distinct physical semantics. These are significant constraints given that regional forecasting and variable-specific modeling are core operational needs.

Notably absent from the limitations discussion:
- No comparison to NWP baselines (IFS, GFS) that represent the operational standard
- No discussion of what happens at longer forecast horizons (beyond 138 hours tested)  
- No consideration of ensemble forecasting or uncertainty quantification, which are now standard in operational forecasting
- The 32×64 grid scale means evaluation is far from real-world applicability, and no discussion of what barriers exist to scaling up

---

### Overall Assessment

PA-TFNP introduces several genuinely interesting ideas — spherical boundary treatment, physics-derived input features, and a physics-blended velocity update — but the paper's execution falls short of ICLR's standards on multiple fronts. The central claim of rotation equivariance is not rigorously established: the tensor product bilinear form is not shown to be SO(3)-equivariant, and the four-region symmetry argument in Figure 1 is informal. The "spherical transform gradient" is standard finite differencing with a geographic correction, overstated as a contribution. The headline 78.92% improvement is unexplained in derivation, and the comparison set (almost exclusively ClimODE) ignores all strong contemporary baselines (GraphCast, FourCastNet, Pangu-Weather, Aurora). The ablation is insufficient to attribute gains to any specific component, and the experiments are conducted only at extremely coarse resolutions with no path to operational relevance. The time-blending mechanism (βt) is physically counterintuitive and unablated. The paper's contribution may be positive relative to ClimODE in the narrow tested setting, but the current form does not provide the theoretical rigor, experimental breadth, or methodological transparency needed for confident acceptance at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes the Physics-Aware Tensor Field Neural PDE (PA-TFNP), a framework that combines Tensor Field Networks for rotation-equivariant spherical processing with physical constraints derived from atmospheric primitive equations. The method addresses polar distortions and long-term error accumulation by integrating spherical gradient operators, physically consistent boundary padding, and diffusion terms directly into the neural ODE dynamics. Reported results on the ERA5 dataset indicate state-of-the-art performance compared to ClimODE, with claims of superior efficiency and accuracy in global weather forecasting tasks.

### Strengths
1.  **Geometric Inductive Bias:** The use of Tensor Field Networks (TFN) to enforce rotation equivariance on the sphere is a well-justified architectural choice. This directly addresses the well-known issues of latitude-longitude grid distortion near the poles that plague standard CNNs (Figure 1 and Section 3.2).
2.  **Physically Motivated Dynamics:** Unlike methods that rely solely on auxiliary loss penalties, this work embeds physics directly into the ODE dynamics (e.g., diffusion terms and blending with primitive equation operators). Ablation studies (Figure 4) demonstrate that these terms significantly improve long-term stability beyond 24 hours.
3.  **Computational Efficiency:** The paper claims a significant reduction in trainable parameters and training time compared to ClimODE (Table 5), which is a notable contribution given the high computational cost of climate modeling. The use of specific padding strategies (Neumann/Average) to handle boundaries is a practical and effective solution to the discretization artifacts in spherical-to-planar mappings.

### Weaknesses
1.  **Incomplete Baseline Comparisons:** While ClimODE and ClimaX are compared, the paper lacks comparisons with major Transformer-based operator learning baselines such as FourCastNet, Pangu-Weather, or GraphCast. These are the current dominant architectures in the community, and omitting them weakens the claim of "state-of-the-art" performance.
2.  **Unclear Metric Scaling:** Reported RMSE values in Table 4 (e.g., ClimODE showing 3115.1 vs. TFNP showing 0.2 for geopotential height in global long-term prediction) suggest potential issues with unit scaling or normalization reporting. Such order-of-magnitude differences are physically unrealistic for geopotential height errors and warrant clarification regarding metric definition.
3.  **Limited Regional Applicability:** The authors explicitly admit in the Limitations section (5) that the rotation-equivariant benefits offer "limited benefits for regional forecasting." Given that many applications focus on regional events, reliance on the global architecture may introduce unnecessary overhead if regional performance lags, which is hinted at in Table 1 where some wind components show marginal or mixed improvements.

### Novelty & Significance
The novelty lies primarily in the adaptation of Tensor Field Networks to global weather prediction combined with a hybrid physics/neural ODE formulation. While the use of spherical convolutions is established, the specific integration with atmospheric primitive equation constraints in a Neural ODE solver is a distinct contribution. The significance for ICLR is the intersection of geometric Deep Learning with Scientific Machine Learning; however, the novelty of the "Tensor Field" component itself is incremental (based on existing TFN literature) and relies on the novelty of the domain application. To meet ICLR's bar, the method must demonstrate a clear improvement in learning efficiency or generalization that transcends the specific dataset, which the paper attempts but requires stronger validation against non-ODE baselines.

### Suggestions for Improvement
1.  **Expand Baseline Evaluation:** Include quantitative comparisons against FourCastNet and Pangu-Weather using the same WeatherBench setup. If parameter efficiency is the advantage, explicitly demonstrate inference latency and convergence speed against these Transformer-based models.
2.  **Clarify Metric Reporting:** Provide a detailed definition of the error metrics used, specifically normalizing the RMSE values or explaining why ClimODE's reported errors in Table 4 are orders of magnitude larger than TFNP's. Visualizing the error distribution (e.g., histograms or spatial heatmaps for a sample case) would clarify if the improvements are global or localized.
3.  **Ablate Physics Components:** While the paper ablates "PA-TFNP" vs "TFNP," it would be stronger to ablate individual physical terms (e.g., diffusion coefficient $\alpha$ vs. drag $\gamma$) to quantify which physical constraint contributes most to the gains, ensuring the physics is actually aiding learning rather than just constraining the solution space arbitrarily.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against industry SOTA baselines.** Include GraphCast, Pangu-Weather, and FourCastNet in the evaluation; claiming state-of-the-art performance without comparing to these established leaders invalidates the primary contribution.
2. **Report Anomaly Correlation Coefficient (ACC).** RMSE alone is insufficient for weather forecasting; ACC is the standard metric for assessing forecast skill and must be included to trust the performance claims.
3. **Quantify conservation law adherence.** Measure mass and energy conservation errors explicitly over time; claims of "physical fidelity" are unsubstantiated without metrics demonstrating reduced physical violations compared to baselines.
4. **Evaluate at higher spatial resolutions.** Conduct experiments at standard operational resolutions (e.g., 1.4° or 0.25°) rather than only 5.625°; results at coarse resolutions do not demonstrate practical utility for modern weather prediction.

### Deeper Analysis Needed (top 3-5 only)
1. **Isolate physics component contributions.** Perform ablations on spherical gradients, diffusion terms, and extra features separately; the current ablation lumps all physics additions together, obscuring which mechanism drives performance gains.
2. **Quantify physical constraint violations.** Measure the magnitude of divergence-free errors or energy drift; asserting physical consistency without quantifying the violation gap undermines the method's core value proposition.
3. **Analyze the blending factor dynamics.** Explain why the time-dependent factor $\beta_t$ shifts reliance from neural to physical terms over time; without justification, this suggests the neural component may collapse during long-term integration.
4. **Compare error growth rates.** Analyze error growth relative to climatology and persistence baselines; this is necessary to verify if the model learns true dynamics or simply regresses to the mean over extended horizons.

### Visualizations & Case Studies
1. **Plot kinetic energy spectra.** Visualize energy spectra to verify if the model preserves physical scaling laws (e.g., Kolmogorov); this reveals whether the diffusion terms correctly handle subgrid turbulence or introduce artificial damping.
2. **Show extreme event case studies.** Provide forecast visualizations for specific extreme events (e.g., tropical cyclones) rather than only global averages; global RMSE often hides failures in critical high-impact scenarios.
3. **Detail polar region predictions.** Provide detailed maps of predictions specifically at the poles; claims of rotational equivariance must be visually substantiated in regions where standard CNNs typically fail.

### Obvious Next Steps
1. **Extend forecast horizon to 10 days.** Evaluate medium-range forecasting skill up to 10 days instead of stopping at 42 hours; medium-range skill is the primary benchmark for operational weather models and is missing here.
2. **Report inference latency.** Provide wall-clock inference time per forecast step; training efficiency is less critical than inference speed for real-time weather deployment and reproducibility.
3. **Implement variable-specific equations.** Address the acknowledged limitation that temperature and wind require distinct physical equations; a unified equation contradicts the stated goal of high physical fidelity.

# Final Consolidated Review
## Summary

The paper proposes Physics-Aware Tensor Field Neural PDE (PA-TFNP), a framework that combines Tensor Field Networks with physical constraints for weather forecasting. Key contributions include: (1) rotation-equivariant processing via Tensor Field Networks to handle spherical geometry, (2) spherical-coordinate gradient operators with physically consistent boundary padding, and (3) physics-derived diffusion terms and momentum blending based on atmospheric primitive equations. The method is evaluated on ERA5 data for global, regional, and monthly forecasting tasks.

## Strengths

- **Geometric Inductive Bias**: The use of Tensor Field Networks (Thomas et al., 2018) to handle rotational properties on spherical domains is a principled architectural choice that addresses polar distortion artifacts common in lat-lon grid representations. Figure 6 provides visual evidence that TFNP reduces errors near poles compared to ClimODE.

- **Integrated Physics Constraints**: Unlike methods relying solely on auxiliary loss penalties, the approach embeds physics directly into the neural ODE dynamics—specifically through diffusion terms with learnable coefficients and momentum blending with primitive-equation-inspired operators. Figure 4 demonstrates improved long-term stability beyond 24 hours.

- **Computational Efficiency**: Table 5 shows PA-TFNP uses ~0.196M parameters versus ClimODE's 2.75M (roughly 14× fewer) with faster training times, addressing practical concerns for deployment. The efficiency stems from the tensor product formulation rather than heavier CNN or Transformer backbones.

- **Consistent Empirical Gains**: Tables 1-4 show consistent improvements across geopotential height (z), temperature (t, t2m), and wind components (u10, v10) on global, regional, and monthly tasks, with the strongest gains at longer horizons.

## Weaknesses

- **Missing State-of-the-Art Baselines**: The paper compares primarily against ClimODE and ClimaX, but omits GraphCast (Lam et al., 2023), FourCastNet (Kurth et al., 2023), Pangu-Weather (Bi et., 2023), and Aurora (Bodnar et al., 2024)—all cited in the introduction as transformative approaches. Claiming state-of-the-art performance without comparison to these dominant baselines significantly weakens validation of the contribution.

- **Overstated Terminology**: The abstract claims "spherical-transform-based gradient operator," which suggests spherical harmonic decomposition, but Section 3.3 reveals this is standard central finite differencing with latitude-dependent metric factors. This is standard practice in atmospheric modeling, not a novel spherical transform. Similarly, "learns directly on spherical tensor fields" overstates the case—operations occur on a lat-lon grid with boundary corrections.

- **Unsubstantiated Headline Claim**: The statement "outperforming ClimODE by 78.92% on global hourly data" appears without methodology for deriving this percentage. The formula, aggregation method across variables/times, and justification for this specific metric are absent, making it impossible to evaluate or reproduce.

- **Insufficient Ablation Granularity**: PA-TFNP incorporates four modifications relative to TFNP: boundary conditions, spherical gradient correction, physics-derived features (wind magnitude, lapse rate, vorticity), and modified primitive equation blending. The paper presents only a single TFNP vs PA-TFNP ablation (Figure 4), making it impossible to determine which components drive improvements or whether simpler additions (e.g., boundary padding alone) would suffice.

- **Limited Resolution and Scalability**: Experiments use 5.625° (32×64) and 11.25° (16×32) grids—extremely coarse compared to operational standards (0.25°, 721×1440). The tensor product formulation has O(N·C²_in·C_out) complexity; scalability to higher resolutions is neither analyzed nor demonstrated.

- **Selective Performance Gaps**: Table 2 shows ClimaX outperforming PA-TFNP for u10 at month 1 (1.80 vs 1.83) and v10 at month 2 (1.71 vs 1.91). Table 1 shows PA-TFNP underperforms ClimODE for t2m at short lead times (6h, 12h) in regional tasks. The paper's framing of uniform superiority is not fully supported by the data.

- **Physical Justification Gaps**: The time-blending factor βt = 1 - exp(-t/τ_0) transitions from neural-dominated predictions at short horizons to physics-dominated at long horizons. The paper provides no ablation of τ_0, no justification for this functional form, and no discussion of what happens in the t→∞ limit when the model relies entirely on the simplified f_phys operator, which lacks core atmospheric dynamics (baroclinic instability, latent heat release).

## Nice-to-Haves

- Isolated ablations of individual physics components (boundary conditions vs. gradients vs. diffusion vs. features) to quantify each contribution
- Experiments at higher resolutions (≥1.4°) to assess scalability and operational relevance
- Comparison against GraphCast/FourCastNet/Pangu-Weather using WeatherBench2 protocols
- Quantification of physical constraint violations (mass/energy conservation errors) to substantiate "physical fidelity" claims

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Claim that tensor product is "straightforward channel-wise outer product"**: While the formulation is indeed bilinear, this characterization understates that TFN (Thomas et al. 2018) has established equivariance properties through proper tensor product construction with spherical harmonics, even if the paper's exposition is incomplete.

- **Claim that average padding lacks physical justification**: The paper explains that average padding "transforms the rectangular domain into a sphere-like domain" and represents a physically reasonable treatment for polar boundaries. While not a rigorous Dirichlet/Neumann condition, this is a reasonable approximation for the singularity.

- **Garbled table entries**: These are PDF parsing artifacts, not substantive issues with the paper.

- **Claim that physics-blending is "counterintuitive"**: The design philosophy—neural handles initial transients while physics prevents long-term drift—has merit even if under-justified. The critic's alternative interpretation is not obviously correct.

- **Demand for NWP baseline comparisons (IFS, GFS)**: While valuable, the paper evaluates against neural baselines on a standard benchmark dataset, which is within scope for an ML methods paper.

## Novel Insights

The paper identifies an important design tension in physics-aware neural forecasting: naive integration of physics terms can conflict with learned dynamics, while pure data-driven approaches accumulate errors over long horizons. The hybrid blending mechanism (neural for short-term accuracy, physics for long-term stability) represents a principled attempt to balance these trade-offs. However, the paper's empirical validation does not rigorously isolate whether this mechanism works as intended or whether improvements come from simpler factors like boundary treatment. The finding that rotation-equivariant architectures particularly benefit polar regions (Figure 6) while offering "limited benefits for regional forecasting" (acknowledged in Limitations) suggests the global equivariance property may be overkill for local tasks—an architectural mismatch worth investigating further.

## Suggestions

1. **Add GraphCast/FourCastNet baselines**: At minimum, report RMSE/ACC on the same WeatherBench test period using published checkpoints; parameter efficiency is only valuable if accuracy remains competitive with these baselines.

2. **Define the 78.92% metric**: Provide explicit formula for how this aggregate improvement was computed, including weighting across variables and lead times.

3. **Perform component-wise ablations**: Report separate experiments for (a) boundary conditions alone, (b) spherical gradient alone, (c) physics features alone, and (d) blending mechanism alone. This would clarify whether the diffusion term or simpler boundary handling drives the gains.

4. **Justify or modify the blending schedule**: Provide physical rationale for why transitioning to simplified physics at long horizons should help, and ablate τ_0 values; alternatively, consider a constant mixing weight and compare.

5. **Clarify resolution limitations**: Acknowledge that 5.625° experiments cannot demonstrate operational utility, and discuss computational barriers to scaling or propose architectures for higher resolutions.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
