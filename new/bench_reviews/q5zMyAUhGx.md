Let me now synthesize the review based on all the information I've gathered, cross-checking the reviewers' claims against the paper content provided. Let me carefully consider each major criticism.

**Harsh Critic Point 1: "No combinatorial dependence" claim is misleading**

Looking at the paper: The claim in the abstract says "no dependence on combinatorial parameters (e.g., number of nodes) outside of logarithmic factors." Theorem 1 states log N ≤ (α̃³ log(2d̃p̃))/ε². The key is that d̃ = max_i d_i and p̃ = max_i p_i appear only inside a logarithm. However, the bound depends on α̃ which contains B_i and C_i. The critic argues that B_i and C_i in practice scale with d_i and p_i, making the "no combinatorial dependence" claim misleading.

This is a valid and substantive point. The claim is technically correct on its face—the bound has no *explicit* polynomial dependence on width—but it's achieved by assuming the ℓ₁ norms B_l are bounded. If B_l grows with d_l * p_l (as it naturally would without explicit regularization), the dependence reappears inside B_l. The paper does note in Remark 4 that ||B_l||₁ ≤ B_max ||B_l||₀, which acknowledges sparsity helps, but it doesn't discuss how B_l relates to architecture size in practice. This is a legitimate criticism of the *framing* of the contribution, though not of the mathematical result itself.

**Harsh Critic Point 2: Empirical validation is tautological**

The critic argues that the correlation in Figure 2 is essentially tautological because both the complexity measure and the excess loss are governed by the same norms/Lipschitz constants that evolve during training. This is partially valid - the normalization scheme where complexity is scaled so its maximum equals the final excess loss value is problematic. However, the claim isn't entirely tautological - showing that a specific combination of norms tracks generalization gap is non-trivial. But the experimental design is indeed weak: no variation in architecture, no comparison with simpler complexity measures, no absolute bound values.

**Harsh Critic Point 3: Loss assumptions vs. claimed generality**

The paper claims the result "does not require the boundedness assumption on the loss function" (from the abstract). Theorem 3 does handle unbounded losses via truncation, but requires Assumption 4 which includes moment conditions on G(x,y) and B(y). This is a valid concern that the scope is narrower than the framing suggests, but it's somewhat standard in the learning theory literature. The paper does state these assumptions clearly in Assumption 4.

**Harsh Critic Point 4: Low-rank RKHS setting is underspecified**

The critic notes that Section 2.3 results are not empirically validated and no connection is made to practical KAN implementations. This is valid - the RKHS results are mathematically consistent but disconnected from practical KAN usage and the empirical section.

**Neutral Reviewer Points:**
- Product of Lipschitz constants: valid concern about potential explosion in depth
- Normalization of complexity measure: valid point
- Dimensionality in RKHS bound: valid concern about curse of dimensionality
- B_i and ρ_i relationship: noted in Remark 5 but could be further explored

**Human Finder Points:**
- Insufficient comparison with MLP bounds (appendix comparison): this is valid but somewhat addressed in Section A.1 (mentioned in paper)
- Tightness concerns and lack of lower bounds: valid
- Limited empirical evaluation: valid
- Unclear practical relevance of assumptions: valid
- Exponential depth dependence: valid but standard for norm-based bounds

Now let me calibrate against similar papers:

- **UvpuGrd6ey** (DNNs breaking curse of dimensionality): Scores 6,8,5,6 (~6.25 avg). This paper had novel compositionality results with clean proofs, but was criticized for issues with Theorem 4.
- **q6zrZbth1F** (How many samples for DNNs): Scores 8,6,8,6 (~7 avg). Established minimax lower bounds, but had concerns about experimental validation.
- **ydlDRUuGm9** (KAN expressiveness): Scores 6,6,8,5 (~6.25 avg). KAN-theory paper, limited theoretical novelty, limited experiments.
- **JN7TcCm9LF** (Koopman generalization): Scores 6,8,5 (~6.3 avg). Novel direction but concerns about hidden constants and vague conclusions.
- **hiHZVUIYik** (path-norm toolkit): Scores 6,8,8 (~7.3 avg). Clean theoretical framework, but bounds still vacuous in practice.
- **fOOOyVhTYV** (multicover): Scores 1,3,5,5,5 (~3.8 avg). Very weak paper with poor writing.

This paper makes a genuine technical contribution (extending covering number bounds to KANs), is well-written, and the mathematics appears sound. However, the framing overclaims, the empirical validation is weak, and the RKHS section lacks empirical grounding. It's comparable to the KAN expressiveness paper (ydlDRUuGm9) and Koopman bound paper (JN7TcCm9LF) in terms of novelty and issues. Solidly in the 5-6 range seems appropriate.

Let me finalize:</think>## Summary

This paper derives generalization bounds for Kolmogorov–Arnold Networks (KANs) under two settings for the activation functions: (1) linear combinations of basis functions, yielding bounds scaling with the ℓ₁ norm of coefficient matrices and Lipschitz constants, with only logarithmic dependence on the number of nodes; and (2) functions lying in a low-rank Reproducing Kernel Hilbert Space (RKHS), yielding bounds scaling polynomially with rank and Lipschitz constants. The basis-function results are extended to unbounded regression-type losses via truncation arguments. Numerical experiments on simulated and real datasets show correlation between the proposed complexity measure and excess loss during training.

## Strengths

- **First rigorous generalization bounds for KANs**: The paper provides the first formal covering-number-based generalization bounds for the KAN architecture, a timely and relevant contribution given the rapid growth of KAN applications. The adaptation of Maurey's sparsification lemma and iterative covering-number arguments to the KAN structure (with vector-valued compositions and layerwise Lipschitz constants) is technically competent.

- **Accommodates unbounded losses**: The extension via truncation (Theorems 3 and 5) to cover regression losses like squared loss, pinball loss, and Huber loss—without requiring boundedness—is a genuine improvement over margin-based bounds that require bounded losses (e.g., Bartlett et al., 2017). This broadens the applicability of the theory.

- **Low-rank RKHS extension**: Section 2.3 for low-rank RKHS activations is a novel theoretical generalization that does not appear to have precedents for MLPs or KANs, and provides a natural connection to fine-tuning/LoRA-style scenarios (Remark 6).

- **Clear and precise mathematical presentation**: Assumptions, propositions, and theorems are stated precisely. The relationship to prior MLP bounds is discussed (Section A.1), and the proof techniques are well-organized.

## Weaknesses

### Major:

- **The "no combinatorial dependence" claim is misleadingly framed.** The paper repeatedly highlights that the bound has "no dependence on combinatorial parameters (e.g., number of nodes) outside of logarithmic factors" as a key selling point. This is technically true as stated—the width parameters d_i and p_i appear only inside log(2d̃p̃) in Theorem 1. However, the complexity measure α̃ contains B_i = ||B_l||₁, which in practice scales with d_l · p_l unless strong ℓ₁ regularization is imposed. Remark 4 acknowledges that ||B_l||₁ ≤ B_max ||B_l||₀, but the paper never discusses how B_l relates to architecture size for SGD-trained KANs. Once B_l is expressed in terms of architecture parameters, the polynomial dependence reappears. The claim is thus correct only under a strong norm-boundedness assumption whose relationship to real KAN training is unexamined. This is a framing issue, not a mathematical error, but it significantly inflates the perceived novelty.

- **Empirical validation is weak and partially tautological.** The experiments show that a normalized version of the complexity measure α̃ tracks the excess loss along a single training trajectory. However: (a) The complexity measure is *normalized so its maximum equals the last excess-loss value*, which artificially forces visual alignment without testing absolute tightness; (b) Both quantities are governed by the same norms/Lipschitz constants that evolve monotonically during training, making correlation unsurprising; (c) There is no variation of architecture (width, depth, number of basis functions) to test whether combinatorial parameters truly enter only logarithmically; (d) No comparison with simpler complexity measures (e.g., product of spectral norms, path norms) to establish that α̃ captures something KAN-specific; (e) The RKHS-based bounds (Theorems 4–5) receive no empirical evaluation. The claim that the results "demonstrate the practical relevance of these bounds" goes well beyond what the experiments support.

- **The low-rank RKHS results (Section 2.3) are disconnected from practice.** Assumption 5 posits that KAN activations lie in a low-rank RKHS with bounded RKHS norm and Lipschitz constant, but there is no discussion of how this constraint is enforced or approximately satisfied in any concrete KAN implementation. The experiments use only the basis-function setting. The RKHS section is mathematically valid but floats without empirical grounding or concrete instantiation.

### Minor:

- **Exponential depth dependence via product of Lipschitz constants**: The complexity α̃ involves products ∏ρ_j across layers. While this is standard in norm-based bounds (analogous to Bartlett et al., 2017), the paper does not discuss whether KAN-specific structure might mitigate this. An empirical analysis of how ρ_j grows during training would strengthen the paper.

- **The B_l and ρ_l relationship is underexplored**: Remark 5 shows ρ* ≤ ||A||_σ c_l √(b_l), linking the Lipschitz constant to the spectral norm and basis Lipschitz constants. Since ||A||_σ is related to B_l, treating them as independent in the bound may lead to double-counting. A refined bound integrating these dependencies could be tighter.

- **Dimensionality curse in RKHS bound**: In Theorem 4, the exponent (d_{i-1}/ν) ∨ 1 can be very large when hidden-layer dimensions d_{i-1} are large relative to the smoothness ν, which is typical for practical KANs. The paper does not discuss when this bound is meaningful.

- **Assumptions on loss generality**: While Theorem 3 lifts the boundedness requirement, Assumption 4 still imposes moment conditions on a uniform envelope G(x,y) = sup_{f∈M} |L(f(x),y)|. For unbounded, high-capacity KANs without output constraints, this envelope may not have finite moments. The paper does not discuss when these conditions concretely hold.

## Nice-to-Haves

- Compare the actual (unnormalized) numerical bound values against excess loss to assess tightness, not just trend correlation.
- Vary architecture parameters (width, depth, basis count) systematically and test whether generalization trends match the theoretical predictions.
- Compare with MLP generalization bounds (e.g., Bartlett et al., 2017) on the same tasks to determine whether KAN-specific analysis offers tighter guarantees.
- Include experiments testing the RKHS-based bounds or at minimum instantiate the RKHS framework with a concrete KAN architecture.
- Investigate whether regularizing the complexity measure α̃ during training causally improves generalization, as suggested in Section 4.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Insufficient comparison with existing MLP bounds in main text"** (Human Finder): The paper does reference Section A.1 for comparison with Bartlett et al. (2017). While more detailed comparison in the main text would be preferable, this is a presentation preference rather than a fundamental flaw.

- **"Missing related works"** (Human Finder): Per the hard rules, I should not flag missing references.

- **"Reproducibility concerns about implementation details"** (Spark): Exact basis function choices and training hyperparameters are standard implementation details that need not be exhaustively specified in the main text of a theory paper.

- **"The paper does not verify that SGD solutions satisfy the constraints defining M"** (Spark, Harsh Critic Section 3): This is a valid theoretical concern, but it applies to virtually all norm-based generalization bounds in deep learning. The paper's bounds are uniform over M, which is the standard framework in this literature.

- **"SGD may not satisfy the ERM condition ĥ ≤ f*"**: This is a standard caveat in ERM-style generalization theory and does not undermine the results. The authors acknowledge in Corollary 1 that only empirical risk no larger than f* is needed.

## Novel Insights

The key insight of this paper—that KANs admit covering-number-based generalization bounds with only logarithmic dependence on width when ℓ₁ norms of coefficient matrices are controlled—is genuinely interesting, even if the framing overclaims. The connection between the low-rank RKHS framework and LoRA-style fine-tuning (Remark 6) is a creative observation that could stimulate future work. However, the most important finding may be a negative one: the complexity measure α̃ is dominated by products of Lipschitz constants across layers, suggesting that the practical relevance of these bounds depends critically on whether KAN training implicitly controls these norms, a question the paper leaves unresolved.

## Suggestions

- Reframe the "no combinatorial dependence" claim to explicitly state that the dependence is pushed into the norm bounds B_l, C_l, and discuss what this means architecturally. This would make the contribution more honest without diminishing it.
- Report unnormalized bound values and quantitative correlation metrics (e.g., R²) rather than only visually aligned, normalized plots.
- Add even one experiment varying width or number of basis functions to test the logarithmic-dependence claim.
- Add a brief discussion of when Assumption 4's moment conditions concretely hold for KANs with specific loss functions (e.g., squared loss with Gaussian noise).

## Score and Decision

**Calibration**: I compared this paper against several related works:
- *How DNNs break the Curse of Dimensionality* (UvpuGrd6ey, avg score ~6.25): Novel compositionality framework with clean proofs, but had issues with Theorem 4 and limited empirical validation. Comparable technical contribution but arguably cleaner framing.
- *How many samples are needed to train a deep neural network?* (q6zrZbth1F, avg score ~7): Established both upper and lower bounds, with concerns about experimental validation of asymptotic rates. Stronger contribution due to lower bounds.
- *KAN expressiveness paper* (ydlDRUuGm9, avg score ~6.25): KAN theory paper with limited novelty in analysis and limited experiments. Most directly comparable in topic.
- *Koopman-based generalization bound* (JN7TcCm9LF, avg score ~6.3): Novel direction but concerns about hidden constants and vague conclusions. Similar pattern of novel framework with practical relevance concerns.
- *Path-norm toolkit* (hiHZVUIYik, avg score ~7.3): More thorough and impactful contribution with sharper bounds and real-scale experiments, but bounds still vacuous.
- *Multicover approach* (fOOOyVhTYV, avg score ~3.8): Very weak paper with poor writing and no real contribution.

This paper makes a genuine technical contribution (first generalization bounds for KANs, competent covering-number analysis, unbounded-loss extension) but overclaims the practical relevance and the significance of the "no combinatorial dependence" feature. The empirical validation is weak, the RKHS section is disconnected from practice, and the framing needs significant toning down. It falls below the path-norm toolkit paper and below "How many samples" (which had both upper and lower bounds), and is comparable to the Koopman and KAN expressiveness papers in contribution level, but with more overclaiming.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>