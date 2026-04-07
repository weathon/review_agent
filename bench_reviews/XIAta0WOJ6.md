## Summary
This paper introduces F²SA-p, a class of fully first-order methods for stochastic bilevel optimization that generalizes prior work by using p-th order finite differences to approximate the hyper-gradient. For problems with p-th order smoothness in the lower-level variable, it achieves an improved SFO complexity of Õ(pϵ^{-4-2/p}), and establishes an Ω(ϵ⁻⁴) lower bound, showing near-optimality when p is large.

## Strengths
- **Novel algorithmic insight**: The paper provides a fresh interpretation of existing first-order bilevel methods as forward-difference approximations, naturally motivating extensions to higher-order finite differences. This elegant perspective leads to a generalizable family of algorithms (F²SA-p) and is clearly presented in Section 3.1 and Lemma 3.1.
- **Strong theoretical contributions**: The authors prove improved upper bounds that beat prior Õ(ϵ⁻⁶) complexity for first-order smooth problems, and complement this with an Ω(ϵ⁻⁴) lower bound via a reduction to single-level optimization, demonstrating near-optimality in ϵ for large p. These results are formally stated in Theorem 3.1 and Theorem 4.1.

## Weaknesses
- **High-order smoothness assumption limits applicability**: The improved rates require Assumption 2.5 (p-th order smoothness in the lower-level variable y), which may not hold in many practical bilevel problems, e.g., those involving non-smooth activations like ReLU. While justified with examples like logistic regression, this restricts the direct practical relevance of the theoretical acceleration.
- **Insufficient empirical validation of theoretical scaling**: Experiments are conducted on a smooth logistic regression problem but lack systematic verification of how complexity scales with p or direct comparison to Hessian-vector-product methods under the same smoothness assumptions. Without this, the core claim of faster rates for higher p is not fully empirically substantiated (Figure 1 shows performance but no ablation on p or oracle counts).
- **Loose condition number dependence**: The upper bound scales as κ^{9+2/p}, which is large and may hinder efficiency for ill-conditioned problems. Although concurrent works improving this are cited, the paper’s own dependence remains loose, and the gap is acknowledged but not resolved (Table 1 and open problems).

## Nice-to-Haves
- More extensive experiments on synthetic problems to directly verify the O(ϵ^{-4-2/p}) scaling with p and the effect of the finite-difference parameter ν.
- Discussion on the per-iteration cost of F²SA-p (which requires solving p lower-level subproblems) compared to F²SA in wall-clock time.
- Inclusion of error bars or multiple runs in experiments to account for stochastic variability.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- Formatting nitpicks about reference placeholders (e.g., mention of "ICLR" in citations).
- Demands for experiments on non-smooth problems (e.g., ReLU networks) or non-convex lower-level problems, which are outside the paper's stated scope of high-order smooth bilevel optimization.
- Requests for deep derivation of variance and bias of the hyper-gradient estimator, which goes beyond standard complexity analysis in this literature.
- Criticisms about missing broader impact statements, as these are not required for technical evaluation at ICLR.

## Novel Insights
The paper provides a novel perspective by linking bilevel optimization to finite difference approximations, which not only unifies prior first-order methods but also naturally leads to accelerated algorithms for higher-order smooth problems. The insight that smoothness only in the lower-level variable y (not jointly in x and y) suffices for acceleration is non-trivial and contrasts with existing joint smoothness assumptions in the literature.

## Suggestions
- In the experiments, include a controlled synthetic study to plot gradient norm vs. SFO calls for different p values, directly validating the theoretical complexity improvement.
- Clarify in the main text or caption for Appendix F that the MLP experiments are exploratory and outside the theoretical assumptions, to avoid misleading readers about the method's applicability to non-smooth settings.