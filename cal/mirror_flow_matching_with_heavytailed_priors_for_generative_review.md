=== CALIBRATION EXAMPLE 52 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately reflects the two central contributions (mirror flow matching + heavy-tailed priors). The abstract is dense but informative. One concern: the abstract claims "competitive sample quality on real-world constrained generative tasks," which actually *undersells* the results — Table 3 shows the proposed method achieves FID 4.27 vs. MDM's 7.29, a significant improvement, not merely competitive performance. This discrepancy between cautious abstract language and actual strong results may confuse readers.

---

### Introduction & Motivation (Sections 1, 1.1)

The two identified challenges (heavy-tailed dual distributions from log-barrier maps, and Gaussian prior mismatch) are clearly motivated with Figure 1 and the verbal argument in Section 1.1. The contributions are explicitly listed and matched to specific propositions and theorems.

**Concern**: The claim "no framework yet ensures constraint satisfaction while providing convergence rates for flow matching" (p. 2) is the primary differentiator asserted over prior work, but the paper only establishes this for *convex* domains under an ε-approximation assumption on the neural network. The scope of this claim should be more carefully delimited.

---

### Ingredient 1: Regularized Mirror Map (Section 2.1)

Lemma 2.1 (tail condition for moment existence) is a standard application of layer integration and is correctly stated and proved (Appendix E). Proposition 2.2 establishes that the modified log-barrier Ψ(x) = −(1−κ)⁻¹ Σᵢ(−ϕᵢ(x))^(1−κ) + ½‖x‖² achieves:
1. Strong convexity (L_Ψ ≤ 1, so W₂ ≤ W₂,Ψ)
2. Controlled tail behavior P(‖∇Ψ(X)‖ ≥ R) ≤ C/R^(β/κ)

**Concern 1 — κ selection**: The condition for finite p-th moment is κ < β/p (stated in Proposition 2.2). For a polytope under uniform measure, β = 1 (Example 1), so for p = 2, κ < 0.5. The paper uses κ = 0.3 empirically. However, for target distributions with smaller β (distributions with less mass near the boundary), this constraint becomes tighter. There is no practical guidance for how to choose κ given a specific (β, p) pair, beyond the theoretical condition.

**Concern 2 — Computational cost of ∇Ψ\***: The inverse mirror map ∇Ψ* is needed at inference time (Algorithm 1, Line 10) to map dual samples back to primal space. For the L₂ ball and polytope, closed forms may be tractable, but for general convex sets defined by smooth convex inequalities ϕᵢ, computing ∇Ψ* requires solving a convex optimization problem at each step. This computational cost is never discussed, which is a notable omission given that the paper targets practical applications.

**Concern 3 — Example 5 (Appendix E)**: The example shows L_ψ can blow up for the log-barrier on an ill-conditioned triangle, motivating the strongly convex regularization. The example is instructive, but the claim that "L_Ψ may blow up" for the proposed map is never quantified. How large can L_Ψ become for the regularized map on ill-conditioned polytopes?

---

### Ingredient 2: Student-t Prior (Section 2.2)

Example 2 is the core motivation for using Student-t priors. The analysis shows that with a Gaussian prior, the conditional density E[X₁|Xₜ = x] can blow up super-exponentially for heavy-tailed targets and large ‖x‖. The Student-t prior suppresses this by making the data distribution dominate the tails.

**Concern — Specificity of Student-t**: The paper motivates Student-t qualitatively but does not explain why Student-t specifically is preferred over other heavy-tailed distributions (e.g., Lévy stable, Cauchy). The key property used in the proofs is the polynomial tail decay of the Student-t density, but this is shared by many distributions. The choice of ν (degrees of freedom) is treated entirely empirically (ν = 10 is used in main experiments), yet the theory depends on ν through the condition α ≥ 2d + ν + 2: larger ν (closer to Gaussian) requires stronger tail decay of the target. This tension between the practical desire for a mild prior and the theoretical requirement is acknowledged in Section 1.1 but not resolved.

---

### Mirror Flow Matching (Section 3)

Proposition 3.1 establishes that dual-space (Euclidean) flow matching is equivalent to primal-space flow matching under the squared Hessian metric. The proof in Appendix F is complete and correct. Figure 2 nicely illustrates how straight-line interpolation in dual space corresponds to geodesic interpolation in primal space.

**Concern — Algorithm 1 ambiguity**: Due to what appears to be PDF rendering issues, Algorithm 1 contains duplicated and conflicting entries for steps 3 and 6. Step 3 appears as both "Choose T ∈ Z" and "satisfying h ∈ Z," and step 6 has both "for k = 0 to T/h − 1" and "for k = 0 to T/h − 1." The algorithm as presented is ambiguous about whether h is the step size or the number of steps, and whether T is the stopping time or an integer. This must be clarified in any revision.

---

### Theoretical Results (Section 4)

**Assumption 3 (Polynomial Tail Bound)**: This requires π₁ᴰ(x) ≤ C/‖x‖^α for ‖x‖ ≥ 1, with α ≥ 2d + ν + 2. In d = 64×64 = 4096 dimensions (the watermarking experiment), this requires α ≥ 8196, an astronomical tail-decay rate that no realistic data distribution satisfies. The paper presents theoretical guarantees but applies them to high-dimensional image generation without acknowledging that all guarantees are vacuous in that regime. This is a serious gap.

**Proposition 4.1 (Lipschitz + Temporal Regularity)**: The result establishes spatial Lipschitzness L₁ = (1+ν)/(1−T)² × B₁ and a temporal derivative bound for t ∈ [0, T]. The dependence L₁ = O(1/(1−T)²) means the Lipschitz constant diverges as T → 1 (longer trajectories require larger Lipschitz constant). This is technically correct but creates a fundamental tension:

**Concern — T-tradeoff not optimized**: In Theorem 3, the error bound is:
W₂(π₁ᴰ, π̂ᵀᴰ) ≤ e^(const × L₁) × √(h²D₃ + ε² + (1−T)²M)

Since L₁ = O(1/(1−T)²), the exponential prefactor grows like exp(C/(1−T)²) while the early-stopping error (1−T)M → 0 as T → 1. No optimal T is derived. The bound becomes trivial for T close to 1. This fundamental tension is acknowledged (p. 6: "early stopping error which decreases to zero as T → 1") but not resolved analytically, leaving practitioners without actionable guidance for setting T.

**Exponential dependence on L₁**: The bound exp(6L₁/L₁)√(...) — while standard in the flow-matching literature — is acknowledged as potentially improvable (p. 7). The exponential factor arises from a Gronwall-type argument on the ODE error. The paper correctly notes this can potentially be improved to polynomial via probabilistic couplings (Chen et al., 2023), but does not pursue this.

**Theorem 4 (Primal Space Guarantee)**: The result follows immediately from Proposition 2.2 and Theorem 3. The conditions require κ ≤ (2d + γ)/(ν + 2) and κ < β/2, where β and γ are problem-dependent constants from Proposition 2.2 and Assumption 4. These constants are not estimable without knowledge of the data distribution geometry near the boundary. The practical relevance of Theorem 4 is therefore limited to settings where β and γ can be bounded.

---

### Experiments (Section 5)

**Critical issue — Missing values in Tables 1 and 2**: Due to what appears to be PDF parsing failures, the numerical values for "Mirror t-Flow" in both tables are incomplete (showing only "0._±0._" in Table 1 and "5._±0._" in Table 2). This makes it impossible to assess the quantitative improvement over baselines from the submitted manuscript. This must be verified in the actual submission.

**Synthetic experiments (Section 5.1)**: The experimental setup follows Li et al. (2025), which provides a fair comparison ground. The 10D polytope and 6D L₂ ball tasks are appropriate validation for a method targeting low-to-moderate dimensional constrained domains. Including MDM as a baseline and explaining the implementation choices (using regularized log-barrier for MDM since closed-form inverse is unavailable for general polytopes) is appreciated.

**Ablation concerns (Figure 3)**: The ablation varies κ and ν but does not fully disentangle the two contributions:
- The paper never shows Mirror G-Flow (proposed mirror map + Gaussian prior) vs. Mirror t-Flow (proposed mirror map + t-prior) in the main quantitative tables. This comparison is implicit in Table 1 (Mirror G-Flow is listed), but it's not highlighted as an ablation. The individual contributions of the mirror map regularization vs. the t-prior should be more clearly separated.
- The observation that "a large ν would require a smaller κ" is consistent with the theory, but no quantitative relationship is derived.

**Watermarked image generation (Section 5.2)**: The comparison in Table 3 is potentially confounded by the choice of initialization. Both methods are initialized at EDM (Karras et al., 2022) checkpoint, and the proposed method trains for 3 hours vs. 13 hours for MDM. The FID improvement (4.27 vs. 7.29) is substantial, but it's unclear whether this reflects:
(a) inherent advantages of flow matching over diffusion (unrelated to the proposed contributions),
(b) better exploitation of the EDM checkpoint by flow matching, or
(c) genuine advantages of the proposed mirror map and t-prior.

A comparison of proposed mirror flow vs. standard (unconstrained) flow matching from the same initialization would help isolate (c).

**Missing baselines**: No comparison to neural approximate mirror maps (Feng et al., 2025, ICLR 2025), which is perhaps the closest methodological prior in terms of using mirror maps for constrained generation.

---

### Related Work

The related work coverage is thorough. The connection to mirror Langevin (Li et al., 2022) for the Wasserstein metric analysis is appropriate. The paper correctly positions itself relative to Zhou & Liu (2025) (bounded support) and Gao et al. (2024) (Gaussian-like targets) for temporal Lipschitzness.

---

### Limitations & Broader Impact

The conclusion acknowledges non-convex domains and the exponential-to-polynomial improvement as future work. However, the paper does not explicitly address:
1. The high-dimensional gap between theoretical guarantees (vacuous for d ~ 4096) and experimental practice.
2. The computational cost of the inverse mirror map at inference time.
3. The sensitivity of the method to the hyperparameter κ in practical settings where β is unknown.

---

## Overall Assessment

This paper makes a technically sound and well-motivated contribution to constrained generative modeling. The regularized mirror map (Proposition 2.2) addresses a genuine limitation of log-barrier transforms, and the Student-t prior analysis (Proposition 4.1, Theorem 3) provides meaningful improvement over prior theoretical results that required bounded support. The synthetic experiments demonstrate consistent improvement over existing methods.

However, three concerns are significant at ICLR standards. First, the theoretical guarantees require α ≥ 2d + ν + 2, which is vacuous for high-dimensional applications — the real-data experiment (d ≈ 4096) operates entirely outside the regime covered by the theory, and this gap is not acknowledged. Second, the error bound in Theorem 3 has an exponential prefactor that grows as exp(C/(1−T)²) while the early-stopping error requires T → 1; no optimal T is derived, leaving the bound practically uninterpretable. Third, the apparently missing numerical values in Tables 1 and 2 prevent verification of the paper's quantitative claims (likely a rendering artifact, but it must be confirmed). The watermarked image generation comparison (Section 5.2) also conflates the benefits of the proposed method with those of flow matching vs. diffusion in general. These issues collectively weaken what is otherwise a solid paper; they are addressable in revision, and the core contribution is sufficiently interesting to merit acceptance if adequately addressed.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes "Mirror Flow Matching," a framework for generative modeling on convex domains that combines a regularized mirror map with Student-$t$ priors to address heavy-tailed dynamics and instability in the dual space. The authors provide theoretical guarantees establishing spatial Lipschitzness and temporal regularity of the velocity field under polynomial tail assumptions, extending prior results that required bounded support. Empirically, the method demonstrates superior sample quality and feasibility compared to Mirror Diffusion Models and other constrained flow baselines on synthetic and real-world tasks.

### Strengths
1.  **Theoretical Rigor on Velocity Field Regularity:** The paper provides a significant theoretical advancement by proving spatial Lipschitzness and temporal regularity in the dual space under polynomial tail assumptions (Theorem 3). This extends recent work (e.g., Zhou & Liu 2025) which restricted the data distribution to have bounded support to obtain similar Lipschitz guarantees.
2.  **Systematic Handling of Heavy Tails:** The dual insight to couple a *regularized mirror map* (Proposition 2.2) with a *Student-$t$ prior* explicitly addresses the "heavy-tailed dual distribution" problem identified in Section 1.1. The paper demonstrates how this co-design ensures finite moments and prevents the velocity field from blowing up, which is a known failure mode in standard log-barrier approaches (Figure 1).
3.  **Strong Empirical Validation:** The evaluation is comprehensive, covering both controlled synthetic tasks (polytopes, $L_2$ balls) and a practical application (watermarked image generation on AFHQv2). The method consistently achieves 100% feasibility and competitive MMD/FID scores compared to MDM and Gauge Flow Matching (Tables 1 & 2, Section 5.2).

### Weaknesses
1.  **Restrictive Assumptions for Theory:** Theorem 3 and Proposition 4.1 require the data tail decay exponent $\alpha$ to satisfy $\alpha \geq 2d + \nu + 2$. In high dimensions ($d$ large), this effectively demands a very heavy-tailed prior relative to the data decay, which may not always be natural or easily verifiable in practice. Additionally, the error bound retains an exponential dependence on the Lipschitz constant ($e^{L_1}$), similar to prior flow matching literature, limiting the scalability of the rate.
2.  **Hyperparameter Sensitivity:** Section 5.1 and Figure 3 indicate that performance depends on the regularization parameter $\kappa$ and degrees of freedom $\nu$. While the paper shows the trade-off, it lacks a concrete guideline or adaptive mechanism for selecting $\kappa$ and $\nu$ for arbitrary domains, which could hinder broader adoption. The text notes "larger values of $\kappa$ would induce a tail that is heavier," but practical rules for balancing this against the mirror map's strong convexity are not fully fleshed out.
3.  **Computational Cost of Mirror Map:** The regularized mirror map involves $\kappa$ and potentially complex inverse mappings compared to standard log-barriers. While the paper mentions the inversion difficulty for MDM implementations (Section 5), it does not quantify the overhead of the proposed regularized map's inversion or Hessian computation during the sampling stage, which is crucial for ODE integration speed.

### Novelty & Significance
The novelty lies in the specific integration of Student-$t$ priors into the Flow Matching framework constrained by regularized mirror maps. While Mirror Descent and Flow Matching are established separately, their combination with heavy-tailed analysis for constrained domains is new. The paper effectively bridges the gap between theoretical requirements (Lipschitz continuity of the vector field) and practical modeling choices (priors and potentials). The significance is high for applications requiring hard constraints (e.g., robotics, physical laws) where standard diffusion or flow models struggle with feasibility without projection artifacts. By relaxing the bounded support assumption, it opens the door to more robust error analysis in generative modeling.

### Suggestions for Improvement
1.  **Clarify Hyperparameter Selection:** Provide a strategy or empirical rule for choosing $\kappa$ and $\nu$ based on the dimension $d$ and the geometry of $K$. An ablation study specifically on the relationship between $\kappa$ and the stability of the numerical ODE integration would be valuable.
2.  **Detail Implementation Costs:** Include a discussion or table on the computational overhead introduced by the regularized mirror map compared to standard barriers. Specifically, does the inversion of $\nabla \Psi^*$ require iterative solvers, and how does that impact the sampling speed in the real-world experiments?
3.  **Non-Convex Extensions:** The theory is strictly limited to convex domains. A brief discussion on potential challenges or preliminary results regarding non-convex constraints (e.g., using multiple local mirror maps) could broaden the paper's impact, as the conclusion hints at this but lacks substance.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Evaluate on higher-resolution datasets (e.g., 256x256) because 64x64 results are insufficient to claim competitiveness with state-of-the-art generative models at ICLR.
2. Compare against projection-based flow matching to isolate whether performance gains arise from the mirror map geometry or simply better optimization dynamics.
3. Report feasibility rates under strict numerical tolerances (e.g., $10^{-8}$) since "100% feasibility" is impossible to verify without specifying floating-point precision thresholds.
4. Evaluate on simplex constraints to verify generality beyond polytopes and L2 balls, as these are common in constrained generative tasks like probability modeling.
5. Benchmark sampling latency, as inverse mirror map computations may introduce significant overhead compared to projection baselines that undermines practical utility.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify the impact of the exponential Lipschitz constant in Theorem 3, as this dependence renders the convergence guarantee vacuous for ill-conditioned domains.
2. Analyze the restrictiveness of the tail assumption $\alpha \ge 2d + \nu + 2$ in high dimensions, since it requires impractically light tails as $d$ increases.
3. Provide a sensitivity analysis for the Student-t degrees of freedom $\nu$, as fixed values may fail to generalize across diverse target distributions.
4. Discuss the scaling of the neural network approximation error $\epsilon$ with respect to the dual-space dimension and map conditioning.
5. Clarify the condition number of the regularized mirror map Hessian, as this directly impacts the numerical stability of the inverse mapping during sampling.

### Visualizations & Case Studies
1. Plot dual-space norm histograms for generated samples to empirically verify the suppression of heavy tails claimed in Figure 1.
2. Visualize flow trajectories near boundaries to demonstrate how the method avoids the boundary singularities common in log-barrier approaches.
3. Show failure cases where the inverse mirror map fails to recover primal samples, exposing limits of the regularized map under extreme constraints.

### Obvious Next Steps
1. Include an ablation on adaptive $\nu$ scheduling to justify the choice of a fixed degrees-of-freedom parameter rather than treating it as a hyperparameter.
2. Provide a preliminary extension to non-convex domains via local convexification to address the stated limitation in the conclusion.
3. Implement and benchmark an efficient approximation for the inverse mirror map to validate the method's computational viability for real-time applications.

# Final Consolidated Review
## Summary

This paper proposes Mirror Flow Matching, a framework for generative modeling on convex domains that combines a regularized mirror map with Student-t priors. The authors address two challenges: standard log-barrier mirror maps induce heavy-tailed dual distributions that violate moment conditions, and Gaussian priors mismatch heavy-tailed targets. Theoretical contributions include establishing spatial Lipschitzness and temporal regularity of the velocity field under polynomial tail assumptions (extending prior work that required bounded support), and deriving Wasserstein convergence rates for flow matching with Student-t priors.

## Strengths

- **Novel theoretical contribution**: Proposition 4.1 establishes both spatial Lipschitzness and temporal regularity of the velocity field under polynomial tail assumptions (α ≥ 2d + ν + 2), which extends prior results by Zhou & Liu (2025) and Benton et al. (2024) that required bounded support or Gaussian-like targets. The proof in Appendix G carefully bounds the conditional expectations needed for the Lipschitz analysis.

- **Principled co-design of mirror map and prior**: The paper correctly identifies that log-barrier mirror maps create heavy-tailed dual distributions (Figure 1 demonstrates this empirically), and that Student-t priors better match heavy-tailed targets. Example 2 provides concrete analytical motivation showing that Gaussian priors can cause the conditional velocity field to blow up super-exponentially, while Student-t priors remain controlled.

- **Strong empirical results on synthetic benchmarks**: Tables 1 and 2 show consistent improvements over Gauge Flow Matching, Reflected Flow Matching, and Mirror Diffusion Models on 10D polytope and 6D L₂ ball tasks, achieving 100% feasibility (by construction) with lower KL divergence and MMD. The regularized mirror map successfully avoids the heavy-tail issues that cause MDM to fail on L₂ ball tasks (Table 2, where MDM shows KL = 8.017).

- **Clear theoretical guarantees for primal space**: Theorem 4 provides error bounds in primal space by combining the dual-space guarantees (Theorem 3) with the Wasserstein distance inequality from Proposition 2.2, giving explicit conditions under which generated samples satisfy the constraint set K by construction.

## Weaknesses

- **High-dimensional vacuity of theoretical assumptions**: The tail condition α ≥ 2d + ν + 2 becomes extremely restrictive in high dimensions. For the watermarking experiment (d ≈ 4096 for 64×64 images), this would require α ≥ 8200, which no realistic image distribution satisfies. While the synthetic experiments (d = 10 and d = 6) fall within the theory's scope, the paper applies the method to image generation without acknowledging this theory-practice gap. The paper should explicitly state that high-dimensional applications operate outside the proven guarantees.

- **No analytical guidance for hyperparameter selection**: The paper demonstrates sensitivity to κ and ν (Figure 3) and notes that "a large ν would require a smaller κ," but provides no principled method for selecting these parameters. Proposition 2.2 gives κ < β/p as a sufficient condition for moment existence, yet β depends on the data distribution's behavior near the boundary, which is typically unknown. Practitioners must resort to empirical tuning.

- **Missing comparison to neural approximate mirror maps**: The paper cites Feng et al. (2025) for neural approximate mirror maps but does not compare against this method, which represents a closely related approach to constrained generation. A discussion of the relative merits (learned vs. analytically constructed mirror maps) would strengthen the positioning.

- **Computational cost of inverse mirror map unquantified**: Line 10 of Algorithm 1 requires computing ∇Ψ*(z) to map dual samples back to primal space. For general convex sets defined by smooth inequalities ϕᵢ, this requires solving a convex optimization problem at each integration step. The paper notes this difficulty for MDM implementation (Section 5.1) but does not quantify the overhead for their own method, which is essential for assessing practical utility.

- **Exponential Lipschitz dependence in error bound**: Theorem 3 includes a factor exp(6L₁) where L₁ = O(1/(1-T)²), acknowledged on p. 7 as "arising due to non-convexity" and potentially improvable via probabilistic couplings. This exponential dependence is inherited from prior flow matching analyses, but the paper does not pursue improvements despite noting them.

## Nice-to-Haves

- Evaluation on simplex constraints beyond polytopes and L₂ balls, as simplex constraints are common in probability modeling applications.

- Benchmarking of sampling latency/throughput to assess practical deployment feasibility, particularly comparing the cost of inverse mirror map computation against simpler projection-based approaches.

- Comparison at higher resolution (e.g., 256×256) to validate scalability beyond 64×64.

- Ablation isolating the contribution of the regularized mirror map from the Student-t prior more clearly (Table 1 includes Mirror G-Flow, but the comparison to Gauge/Reflected baselines conflates both contributions).

## Removed Points

- **Claim of missing numerical values in Tables 1 and 2**: This is incorrect. The values are present but with formatting artifacts (e.g., "0._±0._" represents approximately 0 ± 0). The MMD and KL divergence values are readable from the tables.

- **Claim that T-selection is unaddressed**: The paper discusses early stopping and the (1-T)M term in Theorem 3, noting that early stopping error decreases as T→1. While no optimal T formula is derived, this tradeoff is acknowledged.

- **Request for experiments at 256×256 resolution**: While higher resolution experiments would be valuable, the 64×64 experiments already demonstrate the method's effectiveness on real images. This is a nice-to-have, not a core weakness.

## Novel Insights

The paper's central insight—that heavy-tailed dual distributions require both mirror map regularization AND heavy-tailed priors—is a substantive contribution. The analytical connection between the boundary measure condition P(K\Kδ) ≤ CKδ^β and the tail decay of the dual distribution (Proposition 2.2) provides a principled foundation for choosing the regularization parameter κ. This bridges the gap between the geometric structure of convex constraint sets and the statistical properties needed for stable ODE dynamics.

## Suggestions

- Add explicit discussion of the dimension-dependent scaling of the tail condition, noting that high-dimensional applications operate outside the proven guarantees.

- Provide a rule of thumb or heuristic for κ selection based on problem geometry (e.g., κ = β/(2p) for moment conditions, with practical guidance for estimating β from data or constraint structure).

- Include wall-clock timing comparison for the inverse mirror map computation against alternatives (projection, reflection).

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Accept
