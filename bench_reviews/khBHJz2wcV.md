## Summary

This paper introduces a framework for post-training fine-tuning of flow-matching generative models to enforce parameter-dependent PDE constraints and jointly infer latent physical parameters, without requiring paired parameter–solution training data. The method leverages the Adjoint Matching framework (reformulating fine-tuning as stochastic optimal control), uses weak-form PDE residuals as rewards, and proposes a joint evolution of state and latent parameters via a surrogate base flow derived from a pre-trained inverse predictor. The approach is evaluated on four canonical PDE systems and a natural-image recoloring task.

## Strengths

- **Joint state–parameter evolution is a genuinely novel architectural contribution.** The key idea of evolving α alongside x through a surrogate base flow (derived from the inverse predictor φ) is elegant and addresses a real gap: existing physics-constrained generative methods either assume known parameters or require joint training data. The design—where v_{t,α}^{base} points from current α_t toward φ's one-step terminal estimate—enables physics-aware fine-tuning without paired labels, which is not achieved by prior work.

- **Weak-form residuals as rewards are a principled and well-justified choice.** The use of integration-by-parts to transfer derivatives from x to randomly sampled test functions ψ directly addresses the known instability of strong-form PDE residuals under noisy or misspecified data (Section 3.1). The Wendland-wavelet test function construction with bridge mollifiers (Appendix D.3) is carefully designed and more stable than naive strong-form approaches used in prior physics-informed diffusion work (e.g., Bastek et al., 2024).

- **Scaled memoryless noise schedule with theoretical justification.** Lemma 1 (Appendix D.4) proves that σ²(t) = (1−κ)2η_t retains the memoryless property for 0 ≤ κ < 1, providing a family of valid schedules rather than the single canonical choice. This is a clean theoretical extension of Domingo-Enrich et al. (2025) and offers practical control over the exploration–stability trade-off.

- **Extensive experimental coverage with honest ablations.** The paper evaluates across four physically distinct PDE families (elliptic diffusion, elasticity, wave propagation, incompressible flow) and explicitly shows the residual–distribution trade-off curves (e.g., Fig. 3, Fig. 5) rather than reporting only best-case numbers. The model misspecification studies (e.g., Stokes with F₀ = 2→0, Helmholtz damped→lossless) test genuine out-of-distribution adaptation.

## Weaknesses

### Major:

- **Reliance on inverse predictor φ creates an under-analyzed distributional shift vulnerability.** The surrogate base flow v_{t,α}^{base} and the regularization term f(α) both depend on φ(ẋ₁), where ẋ₁ is a one-step estimate from the current (fine-tuned) state. As fine-tuning progresses, the distribution of ẋ₁ departs from the base distribution on which φ was trained. The regularization f(α) partially mitigates this by anchoring to the base estimate, but the Darcy experiment (Section 4.1) explicitly shows the limitation: "Because α^{base} is itself fragmented, artifact-ridden, some artifacts persist" even with regularization. The paper does not analyze *when* this feedback loop becomes unstable—i.e., how far the fine-tuned distribution can drift before φ's estimates degrade catastrophically. This is a core reliability concern for scientific users deploying the method under severe misspecification.

- **The natural image experiment (Section 4.6) stretches the "physics-constrained" framing.** The abstract promises "cross-domain utility through fine-tuning of natural-image models" in the context of "scientific systems," but the image experiment optimizes a PickScore aesthetic reward via a polynomial color transform. This is a preference-alignment task, not a physics-constrained one. While the *optimization framework* (Adjoint Matching with a latent parameter) applies broadly, framing this as validating "physics-aware" generation is misleading. The experiment validates the *algorithm's generality as a reward-based fine-tuning method*, which is a different contribution than what the title and abstract emphasize. The paper should either reframe this section honestly (as a reward-alignment demonstration) or remove the implication that it supports physics-awareness.

- **The PBFM comparison is acknowledged as asymmetric but still presented prominently without sufficient caveats in the main text.** The paper correctly notes in Appendix E.2 that "such misspecification is inherently challenging for training-time methods like PBFM which naturally places them at a disadvantage." However, the main-text tables (Tables 1, 2) and the Stokes discussion (Section 4.5) feature PBFM's poor performance prominently without reiterating this caveat. PBFM failing to converge on Stokes (strong residuals 1.15×10¹) is presented as a point in favor of the proposed method, when it primarily reflects the fundamental incompatibility of a training-time method with a misspecification scenario. Readers may draw misleading conclusions about relative method quality rather than about the suitability of each approach class for this specific problem setting.

### Minor:

- **Multiple interacting hyperparameters (λx, λα, λf, κ, q) without principled selection guidance.** The ablations in Figure 3 and the appendix tables show that these parameters govern critical trade-offs (residual reduction vs. diversity, residual vs. distributional fidelity, stability vs. exploration). While the sweeps demonstrate the method's controllability, practitioners face a non-trivial tuning burden when adapting to new PDE systems. The paper provides no heuristic or automated strategy for initial hyperparameter selection beyond empirical trial.

- **Total computational cost is not fully contextualized.** The "20 gradient steps, under 15 minutes" claim for Darcy fine-tuning is impressive but omits the upstream costs: base FM pre-training (300 epochs, ~12 hours on RTXA6000; Table 4), inverse predictor pre-training, and the per-epoch cost of lean adjoint solves during fine-tuning. For a fair assessment of efficiency, a comparison of total wall-clock time (base training + φ training + fine-tuning) versus training a physics-constrained model from scratch (e.g., PBFM) to equivalent residual levels would be informative.

### Trivial:

- The notation density around η_t in Equation 1 and the definition of the reference flow coefficients could be clearer, but this does not impede understanding for the target audience.

## Nice-to-Haves

- **Comparison with established inverse solvers** (e.g., MCMC or PINN-based inversion) on the parameter recovery task, to contextualize whether the generative approach achieves competitive inference quality.
- **Validation on real observational data or 3D domains**, as all experiments use synthetic 2D 64×64 grids. This would strengthen claims about utility for "scientific systems."
- **Uncertainty calibration analysis** of the recovered parameter distributions against ground truth, given the stated goal of addressing "ill-posed inverse problems."
- **Visualization of failure modes** or limits of misspecification tolerance (e.g., at what point does the base model's support become too disjoint from the target physics for fine-tuning to succeed?).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Weak residual normalization blow-up near α→0**: The critic suggested normalization by local mean coefficient could blow up in voids. However, all datasets use strictly positive coefficient fields (permeability ∈ {3, 12}, Young's modulus ∈ [1.0, 10.0], etc.; Appendix B). This concern does not apply to the actual experiments.
- **Statistical significance of ± values**: The critic questioned whether reported standard deviations reflect training variance or sample variance. With 256 generated samples per configuration (Section 4), reporting sample variance is standard practice for generative model evaluation. This is a reproducibility nitpick.
- **κ as evidence of instability**: The critic framed κ as a "fix" for an unstable method. The paper presents κ as a generalization of the noise schedule family (Lemma 1 proves consistency), not as a patch. The original κ=0 schedule also works; κ>0 provides additional flexibility.
- **Missing comparison with guided diffusion approaches**: The paper compares against guided sampling (ECI) and training-time physics (PBFM). Demanding comparison with every recent guided-diffusion variant is a generic scope expansion.
- **Formatting/style complaints** about Equation 1 notation density and Figure 1 visual clarity.

## Novel Insights

The joint evolution design reveals an interesting symmetry: the latent parameter α has no ground-truth flow (unlike x, which has the base FM trajectory), so the method *manufactures* a surrogate base flow from the inverse predictor. This effectively turns a pure inference tool (φ) into a generative trajectory regularizer. The success of this approach—particularly the Stokes result where joint evolution achieves MMD_α ≈ 0.07–0.13 versus ablations at 0.22–0.28 despite similar residuals—suggests that the *path structure* of the parameter trajectory matters more for distributional fidelity than the endpoint quality alone. This observation, that regularizing the generative path (not just the terminal state) improves distributional metrics, could inform other settings where latent variables must be jointly sampled with observables.

## Suggestions

- Add a short subsection or paragraph explicitly analyzing the robustness of φ to distributional shift under fine-tuning. Even a simple experiment measuring φ's prediction error on fine-tuned vs. base samples would quantify the risk.
- Rebrand the image experiment (Section 4.6) as "Reward-Based Fine-Tuning Beyond Physics" or move it to an appendix, to prevent readers from interpreting it as evidence of physics-awareness. The current framing creates a mismatch between title/abstract promises and delivered content.
- Add a "Limitations" paragraph in the conclusion explicitly discussing the boundaries of applicability: under what degree of misspecification does the method break down? The Stokes forcing ablation (Figure 13) provides partial evidence, but a principled characterization (even qualitative) of when φ becomes unreliable would significantly help practitioners.
- Report total pipeline cost (base training + φ training + fine-tuning) alongside the fine-tuning-only cost, to give readers a complete efficiency picture.