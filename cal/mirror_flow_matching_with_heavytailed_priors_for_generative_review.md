=== CALIBRATION EXAMPLE 35 ===

# Final Consolidated Review
## Summary

This paper introduces Mirror Flow Matching, a framework for generative modeling on convex domains that addresses two identified challenges: (1) standard log-barrier mirror maps induce heavy-tailed dual distributions violating moment conditions, and (2) Gaussian priors poorly match these heavy-tailed dual targets. The authors propose a regularized mirror map (based on a modified log-barrier) that ensures finite moments and strong convexity, paired with a Student-*t* prior that stabilizes velocity fields by suppressing blow-up modes. The paper provides theoretical guarantees including spatial Lipschitzness, temporal regularity, Wasserstein convergence rates under polynomial tail assumptions, and primal-space feasibility guarantees—all under ε-accurate learned velocity fields. Empirical results on synthetic convex-domain problems and watermarked image generation demonstrate improvements over baselines.

## Strengths

- **Principled co-design identifying a real failure mode:** The paper identifies a concrete, previously under-appreciated problem—the coupling between log-barrier-induced heavy tails in the dual space and Gaussian prior mismatch—and proposes a unified solution (regularized mirror map + Student-*t* prior). Example 2 clearly illustrates why Gaussian priors cause velocity field blow-ups with heavy-tailed targets, and how Student-*t* priors prevent this by making the data distribution dominate the conditional's tail behavior. This is a genuine insight, not an incremental patch.

- **First error bounds for flow matching under polynomial tail assumptions:** Theorem 3 provides convergence guarantees for t-Flow in Euclidean space under Assumption 3 (polynomial tail bound), extending prior results (Zhou & Liu, 2025; Benton et al., 2024) that required bounded support. Proposition 4.1 establishes both spatial Lipschitzness and temporal regularity of the velocity field—prior work typically only addressed spatial regularity or required stronger distributional assumptions. This is a substantive theoretical advance.

- **Strong convexity enables primal-space transfer:** The key insight that regularizing the mirror map to be strongly convex (∇²Ψ ⪰ I) allows bounding primal Wasserstein distances via dual Wasserstein distances (Proposition 2.2, Equation 1) is clean and effective. Example 5 in the appendix concretely demonstrates that log-barrier's LΨ can blow up even for simple 2D polytopes, motivating the modification.

- **Consistent empirical gains with 100% feasibility:** Tables 1–2 show the method outperforms all baselines (RFM, Gauge FM, MDM) in MMD and KL divergence on polytope and L₂ ball tasks, while guaranteeing constraint satisfaction by construction. The real-world watermarked image generation experiment (Table 3) demonstrates practical applicability with superior FID and CMMD under EDM initialization, at lower training cost.

## Weaknesses

- **Exponential dependence on L₁ creates an unresolved tension with early stopping:** Theorem 3's error bound scales as e^{6L₁}, where L₁ ∝ 1/(1−T). Early stopping error scales as (1−T)·M, creating a fundamental tension: making T → 1 reduces early-stopping bias but inflates the discretization/approximation error bound exponentially. The paper acknowledges this exponential dependence as shared with prior work (Bansal et al., 2024; Zhou & Liu, 2025) and mentions probabilistic coupling as a potential fix, but does not empirically analyze how this tension manifests in practice (e.g., how the bound's actual tightness varies with T, or what T is chosen in experiments and why).

- **Hyperparameter selection lacks practical guidance:** The method introduces two key parameters—κ (mirror map regularization) and ν (Student-*t* degrees of freedom)—subject to multiple theoretical constraints: κ < β/p for moment existence (Proposition 2.2), κ ≤ 2/(d + γν + 2) for Assumption 3 to hold (Lemma G.5), and κ < β/2 for Theorem 4. These depend on unknown distribution properties (β, γ). Figure 3 shows sensitivity to these choices but provides no systematic heuristic. A practitioner cannot determine κ or ν from data alone using the current presentation.

- **Insufficient ablation between the two innovations:** The contribution combines a regularized mirror map and a Student-*t* prior, but their individual contributions are not cleanly isolated. Figure 3 partially compares t-Flow vs. G-Flow for different κ values, but the paper lacks a full 2×2 ablation (log-barrier vs. regularized map) × (Gaussian vs. Student-*t* prior) that would demonstrate each component's necessity. The 2D visualization in Figure 5 hints at this but only qualitatively.

- **Inverse mirror map computational cost unanalyzed:** Algorithm 1 (Line 10) requires computing x̂ = ∇Ψ*(ẑ) at sampling time. For general polytopes, no closed form exists (the paper itself notes this difficulty for MDM). While strong convexity of Ψ should make the inversion well-conditioned, the computational overhead per sample—especially relative to projection-based methods—is not discussed or benchmarked. This matters for practical adoption.

- **Dimension scaling of theoretical bounds unclear:** The constants B₁, B₂ in Proposition 4.1 and D₃ in Theorem 3 depend polynomially on d, 1/(1−T), ν, and moments of the target/prior. But L₁ itself depends on B₁, which may grow with dimension through the tail bounds in Proposition G.1. The paper does not analyze or empirically evaluate how L₁ scales with d or constraint complexity, leaving it unclear whether the bounds remain meaningful in high dimensions (e.g., d ≫ 100).

- **Assumption 4 (boundary density condition) lacks verification for common cases:** Assumption 4 requires sup_{x∈K\Kδ} πᴱᵘᶜᴾ(x) ≤ C_{pdf} δ^γ near boundaries. Example 1 verifies the boundary *probability mass* condition P(K\Kδ) ≈ dδ, but this is about mass, not density supremum. For uniform distributions the density is bounded, but for distributions with density concentration near boundaries (e.g., certain truncated Gaussians with modes near constraints), this assumption may fail, and the paper doesn't discuss when or provide verification beyond the simple uniform case.

## Nice-to-Haves

- **Evaluation on unconstrained heavy-tailed targets:** Since the t-Flow analysis in Section 4.1 applies to general Euclidean flow matching (not just the mirror setting), testing the Student-*t* prior on unconstrained heavy-tailed benchmarks would isolate its benefit from the mirror map's geometry handling and strengthen the paper's broader claim.

- **Broader real-world evaluation beyond watermarking:** The introduction mentions molecular generation, policy optimization, and physical constraints, but the real-world experiment is limited to watermarked image generation. One additional application (e.g., a physical constraint task) would better validate generalizability.

- **Non-convex extension discussion:** The paper scopes to convex K and mentions non-convex extension as future work. While this is reasonable, a brief discussion of the fundamental obstacles (e.g., loss of strong convexity, mirror map invertibility) would help readers understand the scope limitations.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **CMMD numerical inconsistency (Harsh Critic):** The critic claimed Section 5.2 text (CMMD = 0.177 vs 0.152) contradicts Table 3 (0.023 vs 0.170). In reality, the text reports results under *random initialization* while Table 3 reports results under *EDM checkpoint initialization*—these are different experimental conditions explicitly distinguished in the paper ("We first report the CMMD metric...With 10,000 generated images" vs. "in Table 3 we next report results when the models are initialized at EDM checkpoint").

- **Corrupted table values as author error (Harsh Critic):** The "0._±0._" entries in Tables 1–2 are clearly PDF parser artifacts (digits lost during extraction), not errors in the original paper. The code is provided for reproducibility.

- **Algorithm 1 notation issues (Harsh Critic):** The conflicting statements in Lines 3 and loop notation are parser artifacts from the PDF extraction, not logical errors in the paper's algorithm.

- **"Guarantees finite moments" claimed as unconditional (Harsh Critic):** The abstract's statement is appropriately qualified by the context of the paper; the specific conditions are detailed in Proposition 2.2 and Theorem 4. This is standard theoretical practice.

- **Negative social impact of watermarking (Harsh Critic):** This is scope creep; the paper's contribution is methodological, and watermarking is an application example, not the paper's focus.

- **Unfair baseline comparison for MDM (Harsh Critic):** The paper transparently explains why MDM is implemented with regularized log-barrier (the original closed-form doesn't apply to arbitrary polytopes) and implements it with the closed form for the L₂ ball case. This is a reasonable and honest comparison, not an unfair one.

- **Missing standard benchmarks like CIFAR-10/ImageNet (Spark Finder):** The paper's focus is constrained generation, where such benchmarks don't have natural constraint structures. The watermarking task is a more appropriate evaluation for the method's target setting.

- **Does 1D Example 2 pathology persist in high dimensions? (Harsh Critic):** The theoretical analysis in Section 4 handles the general d-dimensional case explicitly (Proposition 4.1, Theorem 3), so the pathology is addressed theoretically. The 1D example serves as motivation, not proof.

## Novel Insights

The paper reveals a subtle but important coupling between the *choice of mirror map* and the *choice of prior* in constrained generative modeling that has been overlooked: it's not just that log-barriers create heavy tails in the dual, but that these heavy tails interact catastrophically with Gaussian priors by creating spurious modes near x/t in the conditional distribution, leading to super-exponential velocity field growth. The Student-*t* prior resolves this not merely by "matching tails" but by fundamentally changing the competition between the likelihood and prior terms—making the data-dominated mode near zero prevail over the prior-dominated mode near x/t. This is a deeper structural insight than "heavy-tailed priors for heavy-tailed targets." Furthermore, the strong convexity regularization of the mirror map serves a dual purpose: it controls tail behavior *and* ensures primal-dual Wasserstein distance transfer via the inequality ∥x−y∥ ≤ L_Ψ∥∇Ψ(x)−∇Ψ(y)∥, a connection that ties optimization geometry (strong convexity) directly to sampling quality (primal error bounds).

## Suggestions

- Add a 2×2 ablation table (log-barrier vs. regularized map) × (Gaussian vs. Student-*t* prior) on at least one synthetic task to cleanly isolate each component's contribution and validate the co-design claim.

- Empirically quantify the early stopping bias: plot a metric (MMD or KL) versus T ∈ (0.8, 0.99) to show whether the (1−T) bias term in Theorem 3 is tight in practice, and report the T values used in all experiments.

- Provide a practical heuristic or procedure for selecting κ and ν (e.g., based on estimated tail index of the dual distribution from training data), even if approximate, to make the method more accessible.

- Include wall-clock time comparisons for the inverse mirror map computation at sampling time versus projection-based alternatives, to clarify the practical computational trade-off.

## Evaluation

- **Novelty:** Strong. The identification of the mirror map–prior coupling failure mode and the co-designed solution (regularized map + Student-*t* prior) is genuinely new. The theoretical results extend the state of the art beyond bounded-support assumptions.

- **Technical soundness:** Good, with caveats. The proofs are rigorous and the theoretical framework is well-constructed. The exponential L₁ dependence and the tension with early stopping are real limitations that are acknowledged but not fully resolved.

- **Empirical support:** Adequate but could be stronger. Synthetic experiments convincingly demonstrate improvements, and the watermarking task shows practical viability. However, the limited real-world evaluation, lack of systematic ablation, and missing computational cost analysis leave some claims under-supported.

- **Significance:** Meaningful. The paper provides the first convergence guarantees for flow matching with heavy-tailed targets on convex domains, addressing a gap that the community will likely find relevant. Practical impact depends on demonstrating scalability and ease-of-use.

- **Clarity:** Good. The main ideas are clearly presented with helpful illustrations (Figures 1, 2, 5). The theoretical sections are dense but accessible. Notation could be more consistent (e.g., π^D vs dual-space distribution references).

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Accept
