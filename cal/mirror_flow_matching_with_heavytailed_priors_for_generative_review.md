=== CALIBRATION EXAMPLE 30 ===

# Harsh Critic Review
Now I have enough of the paper read to write a thorough review. Let me compile my assessment.

---

## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately reflects the paper's dual contribution: mirror flow matching (constrained generation) combined with heavy-tailed priors. The abstract is honest about what is proven versus observed empirically. It promises (1) a regularized mirror map with finite-moment guarantees, (2) Wasserstein convergence rates, and (3) primal-space guarantees under ε-accurate velocity fields. All three appear in the body. However, the abstract characterizes the empirical performance as "competitive" on real-world tasks—an oddly modest qualifier, and one that turns out to be accurate: the method is only competitive (not clearly superior) in the random-initialization setting (CMMD 0.177 vs. MDM's 0.152). This tension between the abstract's framing and the actual experimental story deserves sharper honesty.

---

### Introduction & Motivation

The motivation is clearly articulated. The two identified challenges—(1) heavy-tailed dual distributions induced by log-barrier maps, and (2) mismatch between Gaussian priors and heavy-tailed targets—are genuine and well-illustrated by Figure 1 and Example 2. The contributions are enumerated in Section 1.1 and are specific enough to be checked against the later content.

One subtle issue: the paper frames mirror flow matching as being distinct from prior mirror diffusion models (MDM, Liu et al., 2023a) primarily via the regularized mirror map and Student-t prior, but the conceptual framework (map to dual, do flow/diffusion, map back) is the same. The novelty is in the choices of mirror map and prior, rather than in the architecture. This should be stated more plainly to avoid an impression of a more fundamental departure.

---

### Section 2: Ingredients (Mirror Map & Prior)

**Mirror Map (Section 2.1).** The regularized potential Ψ(x) = −(1/(1−κ)) Σ_i (−ϕ_i(x))^(1−κ) + (1/2)||x||² is a clean modification of the log-barrier. The key properties proved in Proposition 2.2 are (i) ∇²Ψ ≽ I (strong convexity, giving W₂ ≤ W_{2,Ψ}), and (ii) a tail bound P(||∇Ψ(X)|| ≥ R) ≤ C/R^{β/κ} from which p-th moment existence follows for κ < β/p.

*Critical concerns:*
- **Bounded-gradient assumption.** Proposition 2.2 explicitly requires "ϕ_i are smooth convex functions with bounded gradient." This immediately excludes many natural convex bodies—e.g., those defined by quadratic inequalities with unbounded domains, or more general semi-algebraic sets. While the paper verifies the assumption for polytopes and the L₂ ball, the restriction should be clearly stated as a limitation.
- **The parameter β.** For the cube under uniform measure, β = 1 (Example 1). For the L₂ ball, similar calculations would give β = 1 as well. To ensure finite second moments (p = 2) in the dual, one needs κ < β/2, so κ < 1/2 for these standard bodies. Theorem 4 further requires κ ≤ γ/(2d + ν + 2), which is a very stringent constraint in high dimensions. The paper does not discuss how restrictive this is in practice for moderate-to-large d.
- **Example 5 (the ill-shaped triangle).** This example demonstrates that classical log-barrier can have arbitrarily large L_Ψ. However, the example computes the Lipschitz ratio of ∇ψ itself (the *inverse* map), not ∇Ψ*. For the *new* regularized map, the authors claim strong convexity (∇²Ψ ≽ I) ensures L_Ψ = 1. But the example for the log barrier uses a very elongated triangle to make L_Ψ blow up; one could ask whether in such geometries the regularized map is *practically* well-conditioned too (e.g., whether the condition number of ∇²Ψ is still huge even if ∇²Ψ ≽ I strictly).

**Student-t Prior (Section 2.2).** Example 2 gives a vivid illustration of the pathology with Gaussian priors when the target has heavier tails. The argument—that the Gaussian prior induces a bimodal conditional distribution causing super-exponential velocity blow-up—is intuitive and the visualization in Appendix C is helpful. The claim that "the velocity field scales as exp(x²) for some small values of t" is asserted but is not formally proved in the body; only the conditional density formula is displayed, and the blow-up is somewhat hand-wavy. A formal Lemma stating the growth rate would sharpen this central motivation.

---

### Section 3: Mirror Flow Matching

The connection between dual Euclidean training and primal geodesic interpolation under the squared Hessian metric is a nice framing; it unifies the algorithmic procedure with a geometric interpretation. Proposition 3.1 (equivalence of primal and dual objectives) is standard but useful to state explicitly. The proof in Appendix F is carried out correctly via the Banach-space conditional expectation framework, though the heavily fragmented notation in the appendix (clearly an artifact of PDF parsing) makes it hard to follow in places.

**Algorithm 1** has what appear to be formatting artifacts that leave the loop bounds ambiguous (the index runs from k = 0 to T/h − 1 but this is rendered inconsistently). This should be fixed in any revision.

One conceptual gap: the algorithm maps all *data* to dual space (Line 1), but during inference one must sample from the *prior* in dual space. The paper states "prior π₀(x) ~ t_{d,ν}" and maps data to dual space, but the prior in dual space π₀^D is the t-distribution itself only if one samples Z₀ ~ t_{d,ν} directly (without passing through ∇Ψ). This is fine when the prior is placed directly in dual space, but it raises the question of what distribution this corresponds to in primal space—since ∇Ψ* (t_{d,ν}) will generally not be a standard distribution supported on K. This distinction is never made explicit, which could confuse practitioners implementing the method.

---

### Section 4: Theoretical Results

**Proposition 4.1 (Spatial and Temporal Lipschitzness).** This is the core theoretical contribution. The result establishes that, under Assumption 3 with α ≥ 2d + ν + 2, the velocity field is L₁-Lipschitz in z with L₁ = (d+ν)B₁/(1−T)², and the temporal derivative is bounded. This is more general than Zhou & Liu (2025)'s bounded-support requirement.

*Critical concerns:*
- **The tail condition α ≥ 2d + ν + 2.** For high-dimensional problems (say d = 100) and any reasonable ν (e.g., ν = 4 for well-defined variance), this requires α ≥ 210. This means the dual-space distribution must have a polynomial tail of degree at least 210. In practice, most distributions won't satisfy this, and even checking whether they do is non-trivial. This severely limits the practical scope of the guarantee.
- **The L₁ dependence on (1−T)⁻².** The Lipschitz constant blows up as T → 1, with L₁ ~ B₁(d+ν)/(1−T)². The error bound in Theorem 3 has a factor exp(6L₁/L₁) = exp(6), but looking more carefully at the bound: W₂(π₁^D, π̂_T^D) ≤ (e^{6L₁}/L₁)[h² D₃ + ε² + (1−T)M]. So the exponential factor in the numerator is exp(6L₁) ~ exp(6(d+ν)B₁/(1−T)²), which grows super-polynomially as T → 1. Meanwhile, the early-stopping error term (1−T)M shrinks. Balancing these yields an optimal T that is far from 1 (i.e., very early stopping), which would leave a large (1−T)M residual. The paper does not derive the optimal choice of T or the resulting convergence rate, nor does it explore the tradeoff explicitly. This makes the bound hard to interpret as a quantitative guarantee.
- **The constants D₃, B₁, B₂.** These are described as depending "polynomially" on certain quantities, but the exact polynomial degrees and dependencies on dimension are not spelled out. A more explicit characterization (even in a remark) would clarify whether the bound is dimension-free or inherits a curse of dimensionality.

**Theorem 3 (Discretization Error).** Given the concerns above, the bound structure is W₂(π₁^D, π̂_T^D) ≤ exp(6L₁)/L₁ · [h²D₃ + ε² + (1−T)M]. The authors acknowledge (below Theorem 3) that the exponential dependency on L₁ "is plausible to improve via probabilistic couplings," but this is a significant caveat. The bound in practice could be vacuous for moderate T due to the exp(6L₁) term growing as (1−T)⁻². The comparison to existing work (Benton et al., 2024; Bansal et al., 2024; Zhou & Liu, 2025) in the paragraph after the theorem is helpful for positioning but does not resolve the quantitative concern.

**Theorem 4 (Primal-Space Guarantee).** This follows by composing Proposition 2.2 and Theorem 3. The conditions imposed are quite complex (Assumptions 1, 4, plus κ ≤ γ/(2d+ν+2) and κ < β/2), and the resulting bound has the same exponential dependence inherited from Theorem 3. Assumption 4 (smoothness and near-boundary density decay of the primal PDF) is stated but not connected to the conditions verifiable from first principles; there is no example showing a concrete distribution that satisfies both Assumption 4 and the condition on κ in Theorem 4 simultaneously.

---

### Section 5: Experiments

**5.1 Synthetic Experiments.** The method is tested on a 10-dimensional polytope and a 6-dimensional L₂ ball. The setup follows Li et al. (2025), which allows direct comparison with Gauge Flow Matching (GFM) and RFM. Both KL divergence and MMD are reported, with 10 runs of 10,000 samples—reasonable statistical power for synthetic tasks.

*Critical concerns:*
- **Table 1 and 2:** The numbers for "Mirror t-Flow" are garbled in the parsed version (showing "0._±0._") and it's therefore impossible to evaluate the actual magnitude of improvement from the review copy. Assuming these are parser artifacts, the claim of superiority needs to be evaluated from the actual values.
- **MDM comparison:** The authors note they "implemented MDM with regularized log-barrier" for the polytope task because the original MDM only provides closed-form mirror map inverses for specific polytopes. This is a methodological choice that makes the MDM comparison non-standard. Since they modified MDM's mirror map, the results reflect their *own implementation* of MDM rather than the published method. This should be made explicit and a discussion of whether this modification helps or hurts MDM should be included.
- **Ablation in Figure 3:** The sensitivity analysis of κ and ν in Figure 3 is helpful. The observation that "a large ν would require a smaller κ" is consistent with the theory, which adds credibility.

**5.2 Watermarked Image Generation.** This is the more practically relevant experiment.

*Critical concerns:*
- **Two-setting presentation (random vs. EDM checkpoint initialization):** In the random-initialization setting (Section 5.2 paragraph 1), the authors' CMMD (0.177) is *worse* than MDM's (0.152). Only with EDM checkpoint initialization does the method win (Table 3: FID 4.27 vs. 7.29; CMMD 0.023 vs. 0.170). The paper smooths over this by noting "strong potential" from the random-initialization result, but this is optimistic framing. If the method requires a pre-trained checkpoint to be competitive, that is an important practical limitation that should be foregrounded.
- **Training time comparison:** The claim of 3 hours vs. 13 hours is striking, but both start from the same EDM checkpoint. The training efficiency advantage likely stems from the flow-matching framework itself (which is known to require fewer training steps than diffusion) rather than from the mirror or t-distribution contribution per se. This conflation should be disentangled.
- **No unconditional generation baseline without watermarking:** The paper compares against MDM (also constrained) but does not compare against standard unconditional flow matching to gauge the quality degradation due to the constraint. This would help contextualize how much the watermarking constraint hurts generation quality.
- **The FID-50k of 3.14 via flow matching initialization:** This number is mentioned in the text (not in a table) and is called "similar to 3.05 from Liu et al. (2023a)." However, the training time of "1.5 hours" vs. "several hundred hours estimated" is an informal comparison. The authors should present this more carefully.

---

### Writing & Clarity

The main body is mostly well-written. Section 2.1 provides a logical path from challenges to the regularized map. However, Section 4's discussion of the theoretical results lacks a proper discussion of how the bounds scale with dimension d and the choice of T, leaving practitioners unable to tune the method based on theory. The algorithm description in Algorithm 1 (as parsed) has contradictory loop bounds; while this may be a parser artifact, it should be checked.

One genuine clarity issue: the paper never explicitly defines what "early stopping" means in the context of the primal distribution. The early stopping error term (1−T)M in Theorem 3 goes to zero as T → 1, but as noted, L₁ also grows as T → 1. The optimal T choice is never discussed, making the bound difficult to operationalize.

---

### Limitations & Broader Impact

The paper's conclusion identifies several future directions (adaptive ν, non-convex domains, improved Lipschitz constants). Missing from the limitations discussion:
1. **The high-dimensional scaling of the tail condition** α ≥ 2d + ν + 2 (addressed above).
2. **Dependence on bounded-gradient constraint functions**, which excludes certain natural domains.
3. **The non-standard MDM comparison** in experiments.
4. **The performance gap in the random-initialization setting** of the image experiment.
5. **The exponential growth of error bounds** with the Lipschitz constant as T → 1 and the lack of a practical T-selection strategy.

---

### Overall Assessment

This paper makes a genuine methodological and theoretical contribution to flow-based generative modeling on convex domains. The two key ideas—a regularized mirror map that tames dual tail behavior while preserving strong convexity, and a Student-t prior that prevents velocity field blow-up—are well-motivated and clearly distinct from prior work. The theoretical results (Proposition 4.1, Theorem 3) are more general than existing analyses in that they handle polynomial-tail distributions without bounded support. However, the paper has several significant weaknesses that limit the strength of the contribution at its current state. The tail condition α ≥ 2d + ν + 2 is restrictive in high dimensions and not discussed as a limitation. The error bound in Theorem 3 has an exponential factor exp(6L₁) where L₁ ~ (1−T)⁻², making the bound vacuous unless T is kept far from 1—but neither the optimal T nor the resulting convergence rate is analyzed. The experimental comparison uses a modified MDM baseline (with the authors' own mirror map substituted in), and in the random-initialization setting the method is actually outperformed by MDM on CMMD, a fact that is underemphasized. For ICLR, the theoretical contributions are of legitimate interest, but the gap between the theoretical story (uniform improvement, polynomial-tail generality) and what is actually demonstrated (exponentially growing bounds, restricted tail conditions, mixed empirical results) needs to be addressed more honestly. The paper is borderline; acceptance would require the authors to provide a more transparent analysis of the T-selection tradeoff, clarify the dimensional scaling of all guarantees, and present the image generation results without cherry-picking the initialization regime.

# Neutral Reviewer
## Balanced Review

### Summary
The paper addresses flow matching on constrained convex domains by identifying two core bottlenecks: classical log-barrier mirror maps induce heavy-tailed dual distributions that destabilize ODE dynamics, and Gaussian priors mismatch these heavy tails. The authors propose a regularized mirror map that guarantees finite moments and strong convexity, paired with a Student-t prior to control velocity field growth and stabilize training. They provide rigorous theoretical guarantees on spatial Lipschitzness, temporal regularity, and Wasserstein discretization error under polynomial tail assumptions, alongside primal-space convergence bounds, and validate the method on synthetic convex sets and watermarked image generation.

### Strengths
1. **Well-Motivated Co-Design of Geometry and Priors:** The paper correctly diagnoses why standard mirror maps and Gaussian priors fail in constrained settings (heavy tails → ill-posed velocity fields and mismatched conditional expectations, Example 2). The joint design of a regularized mirror map (Prop. 2.2) and Student-t prior directly addresses these pathologies, showing strong methodological insight.
2. **Rigorous and Meaningful Theoretical Contributions:** The analysis in Proposition 4.1 establishes spatial Lipschitzness and temporal regularity of the velocity field under polynomial tail bounds (Assumption 3), a notable relaxation compared to prior work that required bounded support. The resulting discretization error bound (Theorem 3) and primal-space guarantee (Theorem 4) are explicit, track standard assumptions in the literature (ε-accurate networks, Assumption 1), and cleanly separate approximation, discretization, and early-stopping errors.
3. **Clear Empirical Validation and Reproducibility:** Experiments span controlled synthetic benchmarks (10D polytope, 6D $L_2$ ball) and a real constrained task (AFHQv2 watermarked images). The method consistently outperforms baselines in MMD, KL, and feasibility (Tables 1-2), with competitive training efficiency reported (Table 3). Algorithm 1, hyperparameter choices, and network architectures are clearly specified, and code is promised, meeting ICLR's reproducibility expectations.

### Weaknesses
1. **Narrow Real-World Empirical Scope:** While synthetic results are thorough, the real-data evaluation is limited to a single watermarking application on 64×64 AFHQv2 images. ICLR typically expects broader empirical validation to establish general utility. The method's performance on other constrained domains (e.g., probability simplices, PSD cones for covariance generation) or higher-dimensional, high-resolution datasets remains untested.
2. **Unmitigated Exponential Lipschitz Dependency:** The error bound in Theorem 3 scales as $e^{6L_1}$, a common issue in deterministic flow matching analyses but still theoretically limiting for high Lipschitz constants or complex geometries. While the authors acknowledge this and cite possible probabilistic coupling alternatives, the bound remains loose, and no empirical analysis is provided to show that $L_1$ stays moderate in practice for the tested regimes.
3. **Lack of Principled Hyperparameter Selection:** The regularization parameter $\kappa$ and Student-t degrees of freedom $\nu$ significantly impact performance (Figure 3), yet the paper lacks a systematic or data-driven strategy for selecting them. The theoretical condition $\kappa < \beta/p$ and $\kappa \leq 2/(d+\gamma\nu+2)$ provides existence guarantees but does not translate into practical guidance for unseen datasets.

### Novelty & Significance
**Novelty:** The integration of a regularized mirror map with a Student-t prior for flow matching on convex domains is novel. The theoretical derivation of velocity field regularity under polynomial tail assumptions (rather than bounded support) represents a meaningful advance in flow matching theory, addressing a known gap in the literature.
**Significance:** The work has high significance for constrained generative modeling, with direct relevance to safety-critical domains, watermarking, and geometry-aware synthesis. The co-design framework offers a principled alternative to projection/reflection-based methods and is likely to influence subsequent research in Riemannian/constrained flows.
**Clarity & Reproducibility:** The manuscript is logically structured, with clear progression from motivation to theory to experiments. Notation is consistent, and the appendix contains complete proofs. Reproducibility is strong given the explicit algorithmic steps, network specifications, and code availability commitment.

### Suggestions for Improvement
1. **Broaden Empirical Validation & Scalability Analysis:** Add experiments on at least one additional constrained domain type (e.g., probability simplex or symmetric positive-definite cone) and evaluate scaling with dimension $d$ to empirically verify the polynomial dependence on $d$ stated in the theoretical bounds. This would strengthen claims of generality.
2. **Develop Practical Hyperparameter Selection Guidelines:** Propose a validation scheme or heuristic for choosing $(\kappa, \nu)$. For example, estimating the empirical tail index of dual-space samples could inform $\nu$, while $\kappa$ could be tied to constraint margin statistics or selected via light-weight validation on a held-out subset.
3. **Clarify Early Stopping ($T$) and Computational Overhead:** The theory and algorithm rely on $T < 1$ for stability, but practical guidance for selecting $T$ is minimal. Provide a sensitivity analysis or rule-of-thumb for choosing $T$ relative to the target distribution's support and $\nu$. Additionally, quantify the computational overhead of evaluating the regularized mirror map and its inverse compared to the standard log-barrier, as this impacts real-world adoption.
4. **Empirically Characterize $L_1$ in Practice:** Since the error bound depends exponentially on $L_1$, report measured or estimated spatial Lipschitz constants of the learned velocity field across training epochs. This would contextualize the theoretical bound and demonstrate that the Student-t prior effectively keeps $L_1$ within a manageable regime.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Unconstrained heavy-tailed benchmarks:** Add experiments on unconstrained datasets with known heavy tails to isolate the Student-t prior contribution from the mirror map geometry. Without this, it is unclear if performance gains stem from the prior or the constraint handling.
2. **Standard image generation benchmarks:** Evaluate on CIFAR-10 or ImageNet (even unconstrained) to demonstrate general generative modeling utility beyond the niche watermarking application. ICLR reviewers expect validation on standard benchmarks to claim broader relevance.
3. **Inverse map computational cost:** Measure and report the wall-clock time for solving the inverse mirror map $\nabla \Psi^*$ during sampling compared to projection steps in Reflected Flow Matching. This is critical to assess scalability, as inverse maps can be computationally expensive.
4. **Early stopping ablation:** Systematically vary the early stopping time $T$ to quantify the trade-off between theoretical stability and sample quality (FID/CMMD). The theory requires $T < 1$, but the empirical cost of this truncation is not analyzed.
5. **Hyperparameter selection strategy:** Provide a heuristic or guideline for selecting $\nu$ and $\kappa$ without exhaustive grid search. Currently, the method relies on tuning these sensitive parameters per dataset, limiting practical usability.

### Deeper Analysis Needed (top 3-5 only)
1. **Empirical dual tail verification:** Plot the empirical decay rate of the dual space distribution to validate Proposition 2.2's claim about finite moments. The core theoretical contribution rests on this tail behavior being controlled, which needs empirical confirmation.
2. **Numerical feasibility precision:** Audit the "100% feasibility" claim under floating-point arithmetic to distinguish between theoretical guarantee and numerical precision. Small violations due to solver error should be quantified rather than claimed as exactly 100%.
3. **Lipschitz constant estimation:** Estimate the empirical Lipschitz constant of the learned velocity field to validate the magnitude of the theoretical error bounds in Theorem 3. Without this, the theoretical guarantees remain abstract and unverified.
4. **Convergence rate validation:** Plot convergence metrics against iteration steps to verify if the empirical scaling matches the rates predicted in Theorem 3. This connects the theoretical analysis directly to observed training dynamics.
5. **Sensitivity to constraint tightness:** Analyze performance degradation as the convex domain becomes increasingly narrow or ill-conditioned. This tests the robustness of the regularized mirror map compared to standard log-barriers.

### Visualizations & Case Studies
1. **Velocity field magnitude heatmaps:** Visualize the norm of the velocity field $\|v(x,t)\|$ for Gaussian vs. Student-t priors to directly evidence the claimed "blow-up" mitigation. This provides intuitive proof of the core methodological motivation.
2. **Primal trajectory boundary plots:** Show sample trajectories in primal space near the constraint boundaries to visually confirm that the mirror map prevents boundary crossing. This validates the constraint satisfaction mechanism visually.
3. **PCA/t-SNE of dual samples:** Provide dimensionality-reduced visualizations of the dual space distribution to verify tail behavior beyond the limited 2D slices currently shown. This ensures the tail control generalizes across dimensions.

### Obvious Next Steps
1. **Adaptive degrees of freedom:** Develop a mechanism to adapt the Student-t degrees of freedom $\nu$ during training rather than fixing it statically. This would allow the model to adjust tail heaviness based on local data geometry.
2. **Efficient inverse map solvers:** Investigate approximate or neural solvers for the inverse mirror map $\nabla \Psi^*$ to reduce sampling latency. This is essential for scaling to higher-dimensional problems where exact inversion is costly.
3. **Extension to non-convex domains:** Explore leveraging local convexity or landing techniques to extend the framework beyond strictly convex domains. This would significantly broaden the applicability of the method to real-world constraints.

# Final Consolidated Review
## Summary
The paper addresses flow matching on convex domains by identifying two fundamental issues: standard log-barrier mirror maps induce heavy-tailed dual distributions that destabilize ODE dynamics, and Gaussian priors poorly match heavy-tailed targets. The authors propose a regularized mirror map that ensures finite moments and strong convexity, paired with a Student-t prior to control velocity field growth. They provide theoretical guarantees on spatial Lipschitzness, temporal regularity, and Wasserstein convergence under polynomial tail assumptions, and validate the method on synthetic benchmarks and a watermarked image generation task.

## Strengths
- **Well-motivated co-design of mirror maps and priors:** The paper correctly diagnoses why standard mirror maps (heavy tails in dual space) and Gaussian priors (velocity field blow-up) fail in constrained settings. Example 2 provides a clear formal illustration of the pathology with Gaussian priors when targets have heavier tails, and the joint solution (regularized mirror map + Student-t prior) directly addresses both issues.
- **Theoretical advances under polynomial tail assumptions:** Proposition 4.1 establishes spatial Lipschitzness and temporal regularity of the velocity field under Assumption 3 (polynomial tail bounds), which relaxes the bounded-support requirement in prior work (Zhou & Liu, 2025; Benton et al., 2024). This is a meaningful theoretical contribution for flow matching analysis.
- **Explicit primal-space guarantees:** Theorem 4 provides Wasserstein error bounds in the primal space by composing the dual-space analysis (Theorem 3) with the regularized mirror map properties (Proposition 2.2). The bound cleanly separates approximation error (ε²), discretization error (h²D₃), and early-stopping error (1−T)M.
- **Empirical demonstration of feasibility:** On synthetic tasks (10D polytope, 6D L₂ ball), the method achieves 100% constraint satisfaction with improved MMD and KL divergence compared to Gauge Flow Matching and Reflected Flow Matching baselines (Tables 1-2).

## Weaknesses
- **Stringent tail condition in high dimensions:** Proposition 4.1 requires α ≥ 2d + ν + 2 for the dual-space distribution tail decay. For d = 100 and ν = 4, this demands α ≥ 210—meaning the dual distribution must have extremely light tails. The paper does not discuss whether common data distributions or the transformed distributions in practice satisfy this, nor how the theoretical guarantees degrade when the condition is violated. The parameter κ in Theorem 4 must also satisfy κ ≤ γ/(2d + ν + 2), becoming extremely small in high dimensions, with no analysis of the practical implications.
- **Exponential dependency in error bound with unclear practical relevance:** Theorem 3's bound includes exp(6L₁) where L₁ ~ (d+ν)B₁/(1−T)². As T → 1, this factor grows super-polynomially, while the early-stopping error (1−T)M vanishes. The paper acknowledges (page 7) that exponential dependence "is plausible to improve via probabilistic couplings" but provides no analysis of the optimal T or the resulting convergence rate under this trade-off, making the bound difficult to operationalize.
- **Mixed empirical results without transparent discussion:** In the watermarked image generation task (Section 5.2), the method achieves CMMD 0.177 versus MDM's 0.152 under random initialization—*worse* than the baseline. Only with EDM checkpoint initialization does it achieve superior performance (FID 4.27 vs. 7.29; CMMD 0.023 vs. 0.170). The paper frames this as "strong potential" rather than acknowledging a genuine limitation: the method's advantage appears to depend on pre-trained initialization, reducing practical accessibility.
- **No principled hyperparameter guidance:** The parameters κ (mirror map regularization) and ν (Student-t degrees of freedom) significantly impact performance (Figure 3), but the theoretical conditions (κ < β/p, κ ≤ γ/(2d+ν+2)) are existence guarantees with no operationalization. The paper provides no validation scheme, heuristic, or data-driven method for selecting these parameters on new datasets, limiting practical adoption.

## Nice-to-Haves
- **Empirical Lipschitz constant estimation:** Report estimated spatial Lipschitz constants of the learned velocity field across training epochs to contextualize the magnitude of the exp(6L₁) factor in Theorem 3.
- **Early stopping sensitivity analysis:** Systematically vary T to quantify the trade-off between theoretical stability and sample quality (CMMD/FID), providing practical guidance for T selection.
- **Computational overhead comparison:** Quantify the wall-clock time for inverse mirror map evaluation versus projection steps in Reflected Flow Matching, as the inverse map may be computationally expensive for complex domains.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **Weakness: MDM uses modified mirror map for polytopes.** The paper reasonably explains (page 9) that the original MDM only provides closed-form mirror map inverses for specific polytopes, and implementing with log-barrier directly is computationally difficult. Using the regularized log-barrier for both methods enables fair comparison on the same geometry. This is a methodological necessity rather than a flaw.

- **Weakness: Algorithm 1 has contradictory loop bounds.** This appears to be a PDF parsing artifact. The actual algorithm specifies T/h discretization steps clearly in the mathematical context.

- **Weakness: Bounded-gradient assumption excludes natural domains.** The condition that ϕᵢ have bounded gradients holds for polytopes and L₂ balls (the tested cases), which cover important practical domains. While quadratic constraints with unbounded domains are excluded, this is a reasonable assumption for the class of problems targeted.

- **Weakness: No CIFAR-10/ImageNet comparison.** The paper's scope is explicitly constrained generative modeling. Evaluating on unconstrained standard benchmarks would not directly test the claimed contributions about constraint handling and heavy-tailed priors.

## Novel Insights
The co-design insight—that the mirror map and prior distribution must be jointly chosen to match the tail behavior—is more fundamental than prior work suggests. The core observation is that log-barrier maps transform bounded primal distributions into heavy-tailed dual distributions, and Gaussian priors cannot match these tails, causing the conditional E[Z₁|Z_t = z] to develop a spurious mode near z/t that creates velocity field singularities. The Student-t prior's heavier tails suppress this spurious mode, a principle that extends beyond flow matching to any generative model using mirror maps. The paper's theoretical contribution of replacing bounded-support assumptions with polynomial tail bounds suggests a broader research direction: developing convergence analyses that characterize tail behavior rather than relying on compactness.

## Suggestions
1. **Add a "Practical Hyperparameter Selection" subsection:** Provide concrete heuristics for choosing κ and ν, such as: (a) estimate the tail index of dual-space samples via empirical moment ratios to inform ν, and (b) select κ based on constraint margin statistics or via validation on a held-out subset. Even approximate guidance would significantly improve usability.

2. **Address the T-selection trade-off explicitly:** Either derive an approximate optimal T from balancing exp(6L₁) and (1−T)M, or provide empirical sensitivity analysis showing how sample quality varies with T in practice. This connects the theory to implementation.

3. **Discuss the random-initialization limitation transparently:** If the method requires strong pre-trained initialization to outperform baselines, acknowledge this as a practical constraint and discuss what minimal pre-training might suffice, or investigate whether longer random-initialization training can close the gap.

4. **Report dimension-dependent theoretical constants explicitly:** The bounds depend on polynomial expressions involving d, ν, B₁, B₂. Provide explicit formulas for these constants (at minimum in an appendix) so readers can assess scaling behavior without deriving from scratch.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Accept
