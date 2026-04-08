=== CALIBRATION EXAMPLE 20 ===

# Final Consolidated Review
##Summary

The paper proposes Mirror Flow Matching for generative modeling on convex domains, combining two key innovations: (1) a regularized mirror map that controls dual-space tail behavior while guaranteeing strong convexity (∇²Ψ ⪰ I), and (2) a Student-*t* prior that aligns with heavy-tailed dual distributions and stabilizes velocity field dynamics. The paper provides the first convergence guarantees for flow matching under polynomial tail assumptions (rather than bounded support), establishing spatial Lipschitzness, temporal regularity, Wasserstein convergence rates, and primal-space feasibility guarantees under ε-accurate learned velocity fields.

## Strengths

- **Principled co-design of mirror map and prior**: The paper identifies two concrete, interacting failure modes—log-barrier maps inducing heavy-tailed dual distributions (violating moment conditions) and Gaussian priors mismatching heavy-tailed targets (causing velocity field blow-up)—and resolves both with a jointly designed regularized mirror map and Student-*t* prior. The interplay between these choices is well-motivated both theoretically (Proposition 2.2, Example 2) and empirically (Figure 1, Figure 5).

- **First convergence guarantees under polynomial tail assumptions**: Proposition 4.1 establishes both spatial Lipschitzness and temporal regularity of the velocity field for Student-*t* priors, enabling Theorem 3's Wasserstein error bounds for heavy-tailed target distributions. This extends prior analyses (Benton et al., 2024; Zhou & Liu, 2025) that required bounded support, and the temporal regularity result goes beyond Cordero-Encinar et al. (2025), which only addressed spatial Lipschitzness.

- **Elegant strong convexity guarantee**: The regularized mirror map achieves ∇²Ψ(x) ⪰ I (proven in Appendix E), which simultaneously ensures finite dual moments via the tail bound in Proposition 2.2 and guarantees L_Ψ ≤ 1 for the primal-space metric transfer in Theorem 4. This resolves the fundamental issue with standard log-barriers where L_Ψ can blow up (Example 5).

## Weaknesses

### Major:

- **Stringent tail decay requirement**: Assumption 3 requires the target density to satisfy α ≥ 2d + ν + 2 for Proposition 4.1 and Theorem 3 to hold. For practical settings (e.g., d=10, ν=10), this demands decay as O(‖x‖⁻³²), which excludes many genuinely heavy-tailed distributions the paper aims to handle. The paper does not discuss whether this condition is tight or could be relaxed with alternative proof techniques, nor does it empirically verify that the dual-space distributions in its experiments actually satisfy this condition.

- **Missing systematic ablation of the two proposed components**: The paper introduces two innovations jointly, but does not isolate their contributions. Figure 3 varies κ and ν within the proposed method and compares t-Flow vs. G-Flow with the proposed mirror map, but critical combinations are missing: (i) the proposed mirror map with Gaussian prior (G-flow proposed map is shown in Figure 5(d), but only qualitatively), and (ii) the standard log-barrier with Student-*t* prior. Without these, it is unclear whether both components are necessary or whether one dominates the improvement.

- **Exponential dependence on Lipschitz constant renders bounds potentially vacuous**: Theorem 3's error bound scales as e^{6L₁}, where L₁ ∝ (1−T)⁻². The paper acknowledges this mirrors prior work (Bansal et al., 2024; Zhou & Liu, 2025), but the practical implications deserve more prominence: for moderate T, L₁ can be large enough to make the bound uninformative. No empirical estimation of L₁ is provided to assess whether the theoretical guarantee has any practical bite on the experimental tasks.

### Minor:

- **Limited real-world evaluation scope**: The only real-world experiment is 64×64 watermarked image generation on AFHQv2, a task adopted from Liu et al. (2023a). Other constrained generation tasks highlighted in the introduction—molecular generation, robotics, preference alignment—are not empirically validated, leaving the method's generality to those settings unconfirmed.

- **Computational cost of the inverse mirror map**: The paper notes that standard log-barriers lack closed-form inverses for arbitrary polytopes, motivating MDM's implementation difficulties. However, the computational cost of computing ∇Ψ* for the proposed regularized map during sampling (Algorithm 1, Line 10) is not analyzed. If numerical inversion is required at each Euler step, this could significantly impact inference time relative to reflection-based methods.

- **No principled guidance for hyperparameter selection**: The method introduces κ (mirror map regularization) and ν (Student-*t* degrees of freedom) with theory constraining κ < β/p and κ ≤ (2d+ν+2)/(γ+2), but provides no practical strategy for choosing these on new problems. Figure 3 shows sensitivity to these parameters but offers no adaptive or heuristic selection rule.

- **Low-dimensional synthetic experiments**: The polytope (d=10) and L₂ ball (d=6) experiments are low-dimensional, and it remains unclear how the method scales to the high-dimensional constrained problems where it would be most impactful. The exponential Lipschitz dependence in Theorem 3 could become more severe in higher dimensions.

### Trivial:

- The MDM baseline uses a modified implementation (regularized log-barrier instead of the original's special-case closed form), which the paper transparently acknowledges. This is a reasonable experimental choice given MDM's design limitations, not an unfair comparison.

## Nice-to-Haves

- Empirical estimation of the spatial Lipschitz constant L₁ on the experimental tasks to assess whether Theorem 3's bound is informative in practice
- Evaluation on an additional real-world constrained task (e.g., simplex constraints for probability distributions) to demonstrate generality
- Reflection-based baselines (RFM, Gauge FM) on the image generation task, since only MDM is compared there
- Quantitative analysis of the early stopping trade-off (sample quality vs. T) on the real-data task
- An adaptive strategy or heuristic for selecting ν based on estimated tail behavior of the dual distribution

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Corrupted table values in Tables 1–2**: The apparent garbling ("0._±0._") is a PDF parser artifact, not a paper error. The original paper contains properly formatted numerical values.

- **Lemma 2.1 being "not novel"**: The paper does not claim Lemma 2.1 as a contribution; it is a supporting result establishing the connection between tail bounds and moment existence. Standard supporting results need not be novel.

- **L_Ψ blow-up for the proposed mirror map**: The harsh critic raised concern that L_Ψ could blow up for the proposed map as it does for the log-barrier. However, since ∇²Ψ(x) ⪰ I (proven in Appendix E), the inverse Hessian satisfies ∇²Ψ*(z) ⪯ I, guaranteeing L_Ψ ≤ 1 uniformly. This is precisely what the regularization achieves and is the core motivation for the modified log-barrier.

- **Non-smooth constraints requiring smoothing**: The critic suggested polytope facets are non-smooth, but the paper defines polytopes via affine constraints ϕ_i(x) = a_i^T x − b_i, which are smooth (infinitely differentiable) convex functions satisfying the bounded gradient assumption.

- **Missing broader impact discussion**: This is scope creep; ICLR does not require broader impact statements, and the paper's scope is clearly defined as a methodological and theoretical contribution.

- **Demand for statistical significance testing**: The paper reports mean ± std over 10 runs, which is standard practice in the field. Requesting formal hypothesis testing for synthetic benchmark comparisons is beyond community norms.

- **Inconsistency between text and Table 3**: Section 5.2 discusses random initialization results (CMMD 0.177 vs. 0.152), while Table 3 reports EDM checkpoint initialization (CMMD 0.023 vs. 0.170). These are clearly different experimental settings explained in the text, not an inconsistency.

## Novel Insights

The paper's most insightful observation is that the two challenges of mirror flow matching—heavy-tailed dual distributions and velocity field blow-up—are *coupled* through the prior choice. Using a Student-*t* prior does not merely match heavy tails better; it fundamentally alters the conditional distribution p(X₁|Xₜ=x) by ensuring the dominant mode remains near x₁≈0 rather than near x₁≈x/t for large ‖x‖, which is what prevents the velocity field from exploding. This explains why simply changing the mirror map (without changing the prior) or simply changing the prior (without regularizing the mirror map) may be insufficient—both are needed because the mirror map controls *which* distribution appears in dual space, and the prior must then be matched to *that* distribution's tail behavior. This co-design principle extends beyond the specific choices in this paper.

## Suggestions

- Add a 2×2 ablation table (log-barrier vs. proposed mirror map × Gaussian vs. Student-*t*) on the synthetic tasks, reporting MMD/KL/feasibility, to conclusively demonstrate both components are needed.
- Empirically measure and report ‖v(x,t)‖ as a function of ‖x‖ for learned velocity fields under both priors on the actual training data, directly validating the central claim that Student-*t* priors suppress explosive gradients in practice.
- Discuss whether the α ≥ 2d + ν + 2 condition is tight or if it could be relaxed; even a brief remark comparing with the α values empirically observed in the dual-space distributions would help practitioners assess whether the theory applies to their setting.
- Report wall-clock inference time (including ∇Ψ* computation) alongside training time to address the computational overhead concern.

---

**Evaluation on axes:**

- **Novelty**: High. The combination of regularized mirror maps with Student-*t* priors specifically designed for constrained flow matching is novel, and the theoretical analysis extending flow matching convergence to polynomial tail assumptions fills a genuine gap.

- **Technical soundness**: Moderate-to-high. The theoretical framework is rigorous, with correct proofs and well-stated assumptions. However, the stringent tail decay requirement (α ≥ 2d + ν + 2) and the exponential Lipschitz dependence limit the practical applicability of the guarantees, and the paper could be more transparent about these limitations.

- **Empirical support**: Moderate. Synthetic experiments demonstrate clear improvements over baselines with 100% feasibility, and the real-data application is competitive. However, the lack of systematic ablation, low synthetic dimensions, and single real-world task leave the empirical case less comprehensive than desired.

- **Significance**: Moderate-to-high. Constrained generative modeling is practically important, and providing the first convergence guarantees with constraint satisfaction is a meaningful advance. The practical impact depends on whether the method scales well and whether the theoretical assumptions hold in realistic settings.

- **Clarity**: Good. The paper clearly motivates the two challenges, develops the solutions systematically, and connects theory to algorithm. Some notational inconsistencies and dense proof sections in the appendix could be improved, but the main text is well-organized.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Accept
