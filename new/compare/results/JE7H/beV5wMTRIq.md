---
job_id: 227789e3-7fdd-41ea-a172-ff72b2564df7
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: beV5wMTRIq.pdf
paper: Physics-Aware Tensor Field Neural PDE for Climate and Weather Prediction
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is on physics-aware neural PDE / neural ODE models for climate and weather prediction with tensor-field networks and spherical operators, which fits squarely under representation learning on geometries, physics-informed ML, and applications to physical sciences.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments/Results, Conclusion) are present. The work is technically nontrivial, with defined methodology and extensive experiments. While there are important issues in correctness and positioning (detailed in the review), these do not reach the level of a desk reject.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any evidence of prompt injection, hidden instructions to LLM reviewers, or other manipulative content in the main text.

---

# Expected Review Outcome:

## Summary

The paper proposes PA-TFNP, a physics-aware tensor field neural PDE framework for climate and weather prediction. It replaces the CNN-based operator in ClimODE with a tensor-field network plus attention to improve rotational handling on the sphere, introduces physically motivated boundary conditions and a spherical finite-difference gradient, and augments the dynamics with diffusion and primitive-equation–inspired velocity tendencies. Experiments on ERA5-based global and regional forecasting tasks report substantial RMSE improvements over ClimODE and ClimaX, with fewer parameters and faster training.

## Strengths

1. **Clear motivation around geometry and physics on the sphere.**  
   The paper correctly identifies two major shortcomings of many ML weather surrogates: (i) using Euclidean CNNs on a lat-lon grid leads to geometric and rotational artifacts near the poles, and (ii) missing or weak physical constraints harm long-range stability. The qualitative comparison in **Figure 2c** (top: true fields, middle: ClimODE absolute error, bottom: TFNP error) nicely illustrates that ClimODE suffers from large polar errors, while TFNP’s errors are much more uniform, supporting the claim that better treatment of geometry and boundaries matters.

2. **Integrated combination of multiple “physics-aware” components.**  
   Section 3.3 proposes a reasonably coherent package: (a) physical boundary conditions implemented via Neumann and average padding (Figures **2a** and **2b**), (b) a spherical-gradient finite difference scheme (Equation (3) and the follow-up derivation in Appendix D), (c) extra physics-derived input features (wind magnitude, lapse rate, vorticity), and (d) a hybrid neural–physical velocity equation mixing learned tendencies with a Laplacian, geopotential gradient, and drag. While each idea is not individually new, combining them into a single neural PDE framework for global climate prediction is conceptually sensible.

3. **Strong empirical gains over ClimODE in several regimes.**  
   The main global forecasting results are compelling. In **Figure 3**, across both coarse 5.625° 5‑day forecasts and finer 11.25° hourly forecasts, PA‑TFNP consistently yields much lower RMSE for all five variables. The stated “38.12% improvement on daily data and 78.92% on hourly data” is backed by large absolute gaps in the plotted curves, especially for geopotential height and atmospheric temperature. **Table 4** (Page 16) gives concrete numeric values: for example, at 5 days / 5.625°, z RMSE drops from \(1104.0 \pm 104.0\) (ClimODE) to \(220.4 \pm 21.6\) (TFNP), and at 6 hours / 11.25° it drops from \(3115.1 \pm 216.6\) to \(161.2 \pm 17.4\), highlighting very large performance differences.

4. **Evidence that physics-aware terms help long-range stability.**  
   The comparison between TFNP and PA‑TFNP in **Figure 4** (RMSE vs lead time up to 138 hours) and **Table 2** (two‑month monthly-mean forecasts) suggests that adding diffusion and the hybrid physical tendency improves medium/long-range behavior. For instance, in **Table 2**, at 1‑month lead, z RMSE is \(529.44\) for TFNP vs \(502.01\) for PA‑TFNP, and t2m is also slightly improved. In **Figure 4**, PA‑TFNP’s RMSE curves consistently sit below TFNP’s for scalar fields, with divergence that grows with forecast horizon, qualitatively matching the motivation.

5. **Nontrivial efficiency improvements.**  
   **Table 5** shows that for global high-resolution forecasts the proposed model uses only 0.196M parameters vs 2.75M for ClimODE, and trains roughly \(2\times\) faster per epoch (e.g., 11.27 s vs 23.69 s for global long-term). For the monthly task, TFNP/PA‑TFNP are about half to one-third the training time and a small fraction of the parameter count of ClimODE and ClimaX. If the reported numbers are accurate, these are practically meaningful efficiency improvements.

6. **Good use of qualitative diagnostics.**  
   The paper does more than dump RMSE numbers. **Figure 6** (Appendix A) and **Figure 8** (Page 19) visualize spatial error patterns for multiple variables and lead times. For example, **Figure 6** compares absolute errors for z, v10, u10, t2m, and t, with ClimODE having visible high-error patches (especially near polar regions) and TFNP showing more uniformly low errors. These plots help substantiate the geometric/physical claims and are valuable for practitioners.

## Weaknesses

1. **The “Tensor Field Network” formulation is mathematically inconsistent and does not actually implement a rotation-equivariant spherical TFN.**  
   Section 3.2 claims to “parametrize the nonlinear operator with a Tensor Field Network (TFN) [Thomas et al., 2018; Weiler et al., 2018; Kondor et al., 2018]” and emphasizes rotation equivariance on the sphere. However:
   - The given definition of \(f_{TFN}\) is  
     \[
     f_{TFN}(I[i,c_{\text{out}}]) = I \otimes I = \sum_{c_1=1}^{C_{\text{out}}} \sum_{c_2=1}^{C_{\text{out}}} W[c_{\text{out}}, c_1, c_2]\,(I[i,c_1]\,I[i,c_2]),
     \]
     for \(i\in[N]\). This uses \(c_1, c_2\) ranging over \(C_{\text{out}}\) instead of \(C_{\text{in}}\), and it is just a quadratic channel-wise MLP; there is no dependence on spatial coordinates, no spherical harmonics, no Clebsch–Gordan structure, and no specification of irreducible representations. This is not the TFN from Thomas et al., nor is any rotation-equivariance proof or argument provided.  
   - The description in **Figure 1** visually suggests region-wise processing and mentions “rotation equivariant,” but the actual operator as written is permutation invariant over channels at each grid point and entirely local in space.  
   Therefore, the core claimed architectural novelty is under-specified and, as written, incorrect with respect to the cited TFN literature. At minimum the authors need to:
     - Correct the indices and give a mathematically coherent definition of the operator.
     - Explicitly explain how spatial coordinates and spherical harmonics or SO(3) representations enter the computation and guarantee equivariance.
     - Clarify whether they actually use the full TFN machinery or just a quadratic MLP; the current text strongly suggests the latter.

2. **The “physics-aware” PDE modifications are ad hoc and not tightly connected to the primitive equations.**  
   Section 3.3 introduces a diffusion term \(\alpha(\mathbf{x})\Delta q_i\) and a blended velocity tendency  
   \[
   \frac{\partial \mathbf{u}_i}{\partial t} = (1-\beta_t) f_\eta(\cdot) + \beta_t f_{\text{phys}}(\mathbf{x},t,\mathbf{u}_i),
   \quad f_{\text{phys}} = -\nabla \Phi + \nu \Delta \mathbf{u}_i - \gamma \mathbf{u}_i.
   \]
   Concerns:
   - There is no derivation from or explicit mapping to the atmospheric primitive equations, despite the claim in the abstract and Section 3.3 (“diffusion terms derived from the atmospheric primitive equations”). The proposed diffusion and drag terms are generic; similar terms appear in many fluid models, but calling them “derived” is overstated without a clear derivation or scaling argument.
   - The diffusion coefficient \(\alpha(\mathbf{x})\) is said to be “non-negative,” but there is no explanation of how non-negativity is enforced (e.g., via softplus parameterization). Negative effective diffusion would be physically meaningless and numerically unstable.
   - The blending factor \(\beta_t = 1 - \exp(-t/\tau_0)\) is specified in continuous time, but implementation uses a normalized time grid (Appendix C) and a forward Euler solver. There is no description of the chosen \(\tau_0\), or how \(\beta_t\) is evaluated at discrete steps. This matters because the contribution of \(f_{\text{phys}}\) vs \(f_\eta\) can drastically affect stability and conservation.
   - The conclusion claims that the approach “incorporat[es] physically consistent diffusion terms and divergence-free conditions”, but nowhere in the method is a divergence-free constraint on \(\mathbf{u}_i\) enforced or even mentioned (no Helmholtz projection, no constraint term in the loss). This is misleading and should be explicitly corrected.

3. **Key equations show inconsistencies/typos that suggest the dynamical system is not clearly specified.**  
   There are multiple mathematical issues:
   - The system integral on Page 3 (Equation (2)) is written as  
     \[
     \begin{bmatrix}\mathbf{Q}(t)\\ \mathbf{U}(t)\end{bmatrix}
       = \begin{bmatrix}\mathbf{Q}(t_0)\\ \mathbf{U}(t_0)\end{bmatrix}
       + \int_{t_0}^{t}
       \begin{pmatrix}
       \frac{d\mathbf{Q}(s)}{d\mathbf{U}(s)}\\
       \frac{d\mathbf{S}(s)}{ds}
       \end{pmatrix} ds,
     \]
     which is clearly wrong dimensionally: \(\frac{d\mathbf{Q}}{d\mathbf{U}}\) makes no sense here, and \(\mathbf{S}\) is undefined. It ought to be something like \(\frac{d\mathbf{Q}(s)}{ds}, \frac{d\mathbf{U}(s)}{ds}\). Given that this is the core ODE system, a correct and unambiguous formulation is important.
   - In the discrete PDE system just above, \(\widehat{F}\) is written as taking both \(\mathbf{q}\) and \(\mathbf{u}\) neighbors, but its functional form is completely unspecified. Since this is what is approximated by \(f_\eta\), readers need to know at least the structure (e.g., advection plus diffusion plus source terms).
   - As mentioned, the TFN index ranges use \(C_{\text{out}}\) for summation indices that should be \(C_{\text{in}}\). While this might be a typo, in aggregate with the other math issues it undermines confidence in the precise implementation of the operator.

4. **Rotation equivariance and boundary-condition benefits are not rigorously quantified.**  
   The paper leans heavily on the rotation-equivariant story (Figure 1 discussion and Section 3.2), but the actual empirical evidence is limited:
   - The qualitative **Figure 2c** and **Figure 6** show reduced polar errors with TFNP compared to ClimODE, but these could also be explained by other architectural differences (attention, training dynamics) or better padding alone. There is no controlled ablation isolating “pure TFN vs CNN” with identical training and physics settings.
   - There is no experiment where the input field is rotated and the model’s equivariance error is measured (standard practice in equivariant ML). Without this, the claim that the model is truly rotation equivariant remains unvalidated.
   - The paper introduces two padding schemes (Neumann and average padding) but does not clearly specify which one is ultimately used in each experiment, nor does it show a quantitative comparison between them. Figure 2a,b are purely schematic; Table 4 and Figure 3 do not break down performance by padding choice.

5. **Physics-aware ablation is shallow and not component-wise.**  
   The only explicit physics-aware ablation is the TFNP vs PA‑TFNP comparison in **Figure 4** and **Table 2**. However, “PA‑TFNP” bundles together several changes: spherical gradients, diffusion \(\alpha\Delta\), blended \(f_{\text{phys}}\), and three additional physical features. There is no experiment that:
   - Removes the diffusion term while keeping other changes.
   - Uses spherical gradients + padding but no physical velocity blend.
   - Omits the extra physics-derived inputs.  
   As a result, one cannot tell which component is responsible for the long-term gains in Figure 4. Since the headline claim is that “embedding primitive-equation-inspired diffusion and physical operators improves stability and accuracy,” this is a significant omission.

6. **Experimental comparisons to broader SOTA are limited, and some reported numbers raise plausibility questions.**  
   - The main baselines in Tables 1–4 are Neural ODE, ClimaX, and ClimODE, all relatively parameter-limited models. There is no comparison, even at reduced resolution or limited regions, to more recent global ML forecasters like GraphCast, FourCastNet, Fengwu, Aurora, or related physics-aware transformer / WeatherODE-style models. This makes it difficult to judge where PA‑TFNP stands in the current landscape.
   - Some gains are surprisingly large. For example, in **Table 4(b)**, for global short-term prediction at 11.25°, z RMSE at 6 hours is \(3115.1\) for ClimODE vs \(161.2\) for TFNP, and t2m RMSE is around 38 for ClimODE vs ~1 for TFNP. Those are order-of-magnitude differences that warrant more explanation: are the metrics truly the same (latitude-weighted RMSE as defined in Appendix C.2)? Are baselines tuned correctly? Are units consistent? Without more detail, there is a risk that baselines are underperforming or evaluated under a different scaling.
   - For regional tasks, the story is more mixed. In **Table 1** (Australia and South America) and **Table 3** (North America), PA‑TFNP underperforms ClimODE quite substantially for t2m at earlier horizons, and for some regions it also does not dominate on t. The text briefly acknowledges this (“PA‑TFNP underperforms at earlier lead times but catches up at 24h”) but does not analyze why, nor whether physics-inspired modifications may be harming local short-range skill.

7. **Discrepancies between main text and appendix regarding the ODE solver and setup.**  
   - Section 3 (Page 3) says “By integrating Equation (2) using the Runge-Kutta method…”, but Appendix C states that the forward Euler method is used with a time resolution of 1/6 month. These are materially different integrators; given that stability of stiff PDE-like systems is crucial, this inconsistency should be resolved and justified.
   - The main text says training/evaluation follow Verma et al. (2024) with some modifications; but hyperparameters, solver tolerances, and time discretization are only cursorily mentioned. For example, the normalized time step of 0.01 corresponds to ~5 days, which is a very coarse step for a PDE with diffusion and advection. It is not clear how many Euler steps are taken to predict, say, 42 hours in the high-resolution experiment, and whether this is consistent across models.

8. **Limited discussion of physical diagnostics beyond RMSE.**  
   Given the strong physics-aware framing, the evaluation sticks almost entirely to RMSE. There is no check of:
   - Conservation of mass/energy proxies.
   - Spectral energy distributions.
   - Biases in key dynamical regimes (e.g., mid-latitude storm tracks, tropics).  
   Even some simple integrated quantities (global-mean temperature bias, variance) or spectral error plots would strengthen the claim that the method improves “physical fidelity” rather than only pointwise regression error.

9. **Related work omits several closely related physics-informed weather ML models.**  
   The Related Works section covers traditional NWP, classic DL models (GraphCast, FourCastNet, Pangu-Weather, ClimaX), and PINNs/Neural PDEs, but omits fairly direct contemporaries that also integrate physics with neural ODE/PDE formulations for weather, listed below in the “Potentially Missing Related Work” section. This weakens the positioning of PA‑TFNP relative to the rapidly growing literature on physics-informed weather ML.

## Potentially Missing Related Work

1. **Chen et al., “DeepPrim: a Physics-Driven 3D Short-term Weather Forecaster via Primitive Equation Learning,” 2026.**  
   DeepPrim directly learns primitive-equation dynamics with a physics-driven 3D architecture for short-term forecasting, which is conceptually very close to PA‑TFNP’s use of primitive-equation-inspired operators. It should be discussed in Section 2 (Physics-Informed Machine Learning) and compared conceptually with the proposed modified primitive equation in Section 3.3.

2. **Liu et al., “Mitigating Time Discretization Challenges with WeatherODE: A Sandwich Physics-Driven Neural ODE for Weather Forecasting,” 2024.**  
   WeatherODE focuses on physics-driven neural ODEs for weather forecasting, explicitly addressing time discretization issues similar to those discussed in this paper. It should be cited around the Method of Lines / neural ODE formulation in Section 3.1 and contrasted with ClimODE and PA‑TFNP in Section 2 and 4.

3. **Lyu et al., “Physics-Informed Teleconnection-Aware Transformer for Global Subseasonal-to-Seasonal Forecasting,” 2025.**  
   This work integrates multiple physical inductive biases into a transformer for global S2S forecasting. It is highly relevant to the physics-aware forecasting narrative and should be referenced in Section 2 as a related alternative architecture and in the discussion of long-term monthly predictions (Section 4.3).

4. **Chen et al., “Physics-informed generative neural network: an application to troposphere temperature prediction,” 2021.**  
   This is an earlier but directly related example of physics-informed neural networks applied to atmospheric temperature fields. It belongs in Section 2 under Physics-Informed Machine Learning and could be mentioned when introducing the loss function and stochastic modeling in Appendix C.1.

5. **Zeng et al., “PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems,” 2025.**  
   This work emphasizes physics-encoded message passing for PDEs, relevant to the idea of coupling physical operators and neural networks. It should be cited in Related Work (Section 2) and possibly contrasted with the TFNP design in Section 3.2.

6. **Sparey et al., “Physics-Informed Machine Learning under Climate Domain Shift: PDE-Free Physics Regularisation for Cloud Prediction,” 2025.**  
   This paper proposes physics regularization for cloud prediction without explicit PDE solvers. It is relevant to Section 2 as an example of maintaining physical consistency in climate tasks via regularization instead of operator-level modifications.

7. **Bugaev et al., “Physics-Constrained Neural Networks for Improved Short-Term Weather Forecasting: A Case Study over the South Pacific,” 2026.**  
   Focuses on hybrid physics-constrained networks for regional short-term forecasts, directly comparable to the regional experiments in Section 4.2. It should be cited in Related Work and possibly discussed when interpreting regional t2m behavior in Table 1 and Table 3.

8. **Zhao et al., “PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks,” 2024.**  
   Presents a transformer-based PINN framework with physics constraints embedded in a sequence model. It should be acknowledged in Section 2 as an alternative direction for physics-aware neural PDE solvers and contrasted with the ODE-based approach here.

(ClimODE [Verma et al., 2024] is properly cited and used as a baseline.)

## Questions

1. **Clarification of the actual TFN implementation.**  
   - Can you provide the exact mathematical form of the operator used in code, including how spatial coordinates (latitude/longitude or 3D unit vectors) are incorporated, and how rotation equivariance is guaranteed?  
   - Is the operator based on spherical harmonics / Clebsch–Gordan coefficients as in Thomas et al., or is it a simpler quadratic MLP? If the latter, how do you justify calling it a Tensor Field Network?

2. **Details on enforcing non-negative diffusion and physical coefficients.**  
   - How are \(\alpha(\mathbf{x})\), \(\nu\), and \(\gamma\) parameterized to ensure physical sign constraints (e.g., \(\alpha,\nu\ge 0\))? Without such constraints, could the model learn negative diffusion or negative drag?

3. **Exact time discretization and solver choices.**  
   - Do you use Runge-Kutta or forward Euler in the main experiments? If different integrators are used in different tasks, please specify which for each table and figure.  
   - For the short-term 6–42h forecasts at 11.25°, how many integration steps are taken, and what is the time step size? Is this identical for ClimODE and PA‑TFNP?

4. **Component-wise ablation of physics-aware terms.**  
   - Can you provide ablations where you: (a) remove \(\alpha\Delta q_i\); (b) set \(\beta_t = 0\) (no physical operator) but keep spherical gradients and padding; (c) drop the extra physics-derived features one by one?  
   - This would help clarify which components are responsible for the improved long-range behavior in Figure 4 and monthly improvements in Table 2.

5. **Clarification and sanity checks on very large RMSE gaps.**  
   - The orders-of-magnitude gaps in Table 4(b) (e.g., z RMSE 3115 vs 161 at 6h) are striking. Could you confirm that (i) the same normalization / de-normalization and latitude-weighted RMSE metric are used for all models, and (ii) baselines were trained to convergence with the same data and optimizer?  
   - Any additional diagnostic (e.g., plotting the distribution of errors across space/time for ClimODE vs TFNP) would help convince readers the baseline is not inadvertently under-trained or misconfigured.

6. **Equivariance experiment.**  
   - Could you add a simple test where an input snapshot is rotated around, say, the polar axis by a fixed angle, propagate both the original and rotated states through the model, and measure the equivariance error between appropriately rotated outputs?  
   - This would provide concrete evidence that the model is at least approximately rotation-equivariant.

7. **Divergence-free and conservation claims.**  
   - How exactly are divergence-free conditions enforced or encouraged for \(\mathbf{u}_i\)? If they are not, please revise the claims in Section 5 to avoid overstatement.  
   - Have you computed any simple conservation-related diagnostic (e.g., global-mean mass or energy proxies) to see whether PA‑TFNP improves physical consistency beyond RMSE?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The overall framework is plausible and the experiments are extensive, but there are several mathematical inconsistencies (e.g., TFN definition, Equation (2)), incomplete specification of crucial operators (TFN, \(\alpha,\nu,\gamma\), \(\beta_t\)), and a lack of clean ablations that make it hard to fully trust or interpret the claimed mechanisms behind the gains.

## Presentation Rating

2: fair.  
The paper is readable and includes many informative figures and tables, but key technical components (the TFN, the physical blending, solver details) are under-specified or mis-specified, and there are nontrivial notation errors. The related work is incomplete with respect to recent physics-aware weather ML.

## Contribution Rating

2: fair.  
The idea of coupling a more geometry-aware operator with physical boundary conditions and simple primitive-equation-inspired terms is interesting, and the empirical improvements over ClimODE are promising. However, the architectural description, theoretical grounding, and positioning relative to contemporaneous physics-informed weather models are not yet at the level needed for a strong ICLR contribution.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper tackles an important problem with a conceptually appealing blend of geometric and physics-based priors and shows strong empirical improvements over ClimODE, especially in global settings. However, the core TFN operator is not correctly or fully specified, several math/physics claims are overstated or inconsistent, and the experimental analysis does not isolate which components actually drive the gains. With substantial clarification, corrected formulations, and stronger ablations (especially around equivariance and physics-aware terms), this could evolve into a solid contribution, but in its current form it falls short of ICLR’s bar for reliability and clarity.

## Reviewer Confidence

4: confident.  
I am familiar with neural ODE/PDE methods, equivariant networks, and ML for weather/climate, and I have carefully checked the core equations and experimental design. Some implementation details are missing, so there is room for clarifying responses to improve my assessment, but the main concerns are unlikely to vanish completely.