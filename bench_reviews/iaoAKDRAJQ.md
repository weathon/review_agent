## Summary

This paper extends the theory of adaptive smoothness—previously established for convex settings—to nonconvex optimization, showing it precisely characterizes the convergence of adaptive optimizers (Adam, AdaGrad, Shampoo) under a unified framework of "well-structured preconditioner sets." It establishes two key benefits of adaptive geometry over standard geometry: (1) adaptive smoothness enables an accelerated O(T⁻²) rate for adaptive optimizers with Nesterov momentum in the convex setting, a rate impossible under standard ℓ∞-smoothness; and (2) an analogous "adaptive variance" assumption yields dimension-free convergence for NSD in the stochastic nonconvex setting, whereas standard variance inevitably introduces dimension dependence.

## Strengths

- **Clean theoretical separation between adaptive methods and NSD.** The paper formalizes that both families exploit the same non-Euclidean geometry but through fundamentally different smoothness notions (adaptive vs. standard). The convex acceleration result (Theorem 4.3) combined with the Guzmán & Nemirovski lower bound provides a sharp separation: adaptive smoothness enables O(T⁻²) while standard ℓ∞-smoothness permits at best Ω(T⁻¹). This is a concrete, provable advantage, not just a notational distinction.

- **Novel matrix inequality (Lemma 3.3) for general preconditioner sets.** Extending the nonconvex convergence analysis from diagonal to arbitrary well-structured preconditioner sets requires handling noncommutativity of matrix preconditioners. Lemma 3.3 and its supporting Lemma C.1 (relating differences of positive definite matrices to differences of their logarithms) provide a general bound with a log d penalty for noncommutative cases and no penalty for commutative (diagonal) cases. This is a technical contribution of independent interest.

- **Adaptive variance and dimension-free NSD rates.** The parallel between adaptive smoothness (enabling acceleration) and adaptive variance (enabling dimension-free rates) is conceptually elegant. Theorem 4.5 gives a dimension-free NSD rate under adaptive variance, while Theorem 4.7 proves dimension-dependence is unavoidable under standard variance for ℓ∞ geometry—establishing a genuine separation, not merely an artifact of the analysis.

- **Unified algorithmic framework.** Algorithm 1, parameterized by a well-structured preconditioner set H, recovers AdaGrad, Adam, AdaGrad-Norm, full-matrix AdaGrad, and one-sided Shampoo as special cases. The convergence theorems (3.1, 3.2, D.2, D.7, D.8) apply uniformly across these methods.

## Weaknesses

### Major:

- **The strongest benefit (acceleration) is restricted to convex losses.** The paper motivates itself through the lens of deep learning optimizers (Adam, Muon, Lion), yet the O(T⁻²) acceleration in Theorem 4.3 applies only to convex functions. For the nonconvex setting—which is the practically relevant regime—the paper shows only that adaptive and standard smoothness differ, but does not establish that adaptive smoothness confers any rate advantage. This creates a gap between the motivation (explaining deep learning optimizer success) and the delivered theoretical benefit. The paper should explicitly acknowledge this limitation and discuss whether the convex separation suggests analogous (but unproven) benefits in nonconvex settings, or whether there are fundamental barriers.

- **The adaptive smoothness constant can be up to d times larger than standard smoothness (Proposition 2.5).** Since Λ_H(f) ≤ d · L_{∥·∥_H}(f), the adaptive smoothness bound in the nonconvex convergence rate could be substantially worse than the NSD rate in terms of problem-dependent constants. The paper argues adaptive methods "automatically identify the best geometry," but if Λ_H(f) ≈ d · L_{∥·∥_H}(f) in practice, the asymptotic advantage is offset by constant factors. There is no discussion of when or whether Λ_H(f) is close to L_{∥·∥_H}(f) for realistic loss landscapes, leaving the practical significance of the theoretical framework unclear.

### Minor:

- **The adaptive variance assumption (Definition 4.1) is stronger than standard variance, and its practical validity is unverified.** Adaptive variance requires uniform control of noise over all preconditioners H ∈ H with Tr(H) ≤ 1. The paper shows it is weaker than bounded covariance (Proposition B.10), but does not discuss whether common noise sources (mini-batch sampling, label noise) actually satisfy the adaptive variance bound in practice. Without some evidence— even qualitative—that this assumption is realistic for neural network training, the dimension-free result remains a theoretical curiosity.

- **The log d factor in the nonconvex convergence rate for general well-structured H (Theorem 3.1).** For non-diagonal (noncommutative) preconditioner sets, the convergence rate picks up a log d factor that is absent in the diagonal case. The paper identifies this as arising from noncommutativity but does not establish whether this factor is tight or an artifact of the proof technique. Given that methods like Shampoo involve non-diagonal preconditioners, the log d factor could be significant in high dimensions.

### Trivial:

- **Equivalence between weighted, cumulative, and EMA variants** is stated briefly in Section 3.2 but the precise hyperparameter mappings (e.g., η_W = η_E/√(1−β)) are deferred to the appendix. Stating these in the main text would improve readability for practitioners.

## Nice-to-Haves

- **Empirical validation of the adaptive vs. standard smoothness gap.** Even a small-scale experiment (e.g., on logistic regression or a simple neural network) measuring Λ_H(f) vs. L_{∥·∥_H}(f) and adaptive vs. standard variance would substantially strengthen the practical relevance of the theoretical framework.

- **Experiments confirming the acceleration separation (Theorem 4.3).** A convex benchmark (e.g., ℓ∞-smooth logistic regression) comparing Algorithm 2 against NSD would directly validate the paper's core claim that adaptive smoothness enables acceleration unattainable under standard smoothness.

- **Discussion of whether the convex acceleration insight extends to nonconvex settings.** Even without a formal theorem, qualitative reasoning about what barriers exist for nonconvex acceleration under adaptive smoothness would guide future work.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Abstract lacks precision about dimension-free claim.** The abstract already qualifies the statement with "for certain non-Euclidean geometry," which is precise. Removed as a nitpick.

- **Weakness: Algorithm 2 requires knowledge of D.** The paper addresses this in Remark 4.4 and Appendix E.2 with a projected variant (Algorithm 8, Theorem E.5) that removes the dependence on D. The concern is already handled.

- **Weakness: Comparison with Kovalev (2025a/b) may be apples-to-oranges.** The paper correctly notes that using standard smoothness (which is ≤ adaptive smoothness) yields a tighter bound. This is a valid, precise comparison—removed as unfounded.

- **Weakness: Computational cost of computing gradients of the modified loss f_{α_t,x̄_t}.** This is a standard Nesterov acceleration construction; the gradient of the modified loss requires one gradient evaluation of f at a shifted point, which is no more expensive than standard Nesterov momentum. Removed as factually wrong.

- **Weakness: Citation density and positioning as extension of Xie et al. (2025b).** Building on prior work with genuine novel contributions (nonconvex extension, acceleration, adaptive variance) is standard practice. Removed as not a real weakness.

- **Weakness: Missing related works.** Per rules, not included.

- **Weakness: Formatting and parser artifacts.** Per rules, removed as style nitpick.

- **Weakness: Reproducibility of hyperparameters.** Per rules, removed.

- **Weakness: Generalizing beyond well-structured preconditioner sets.** The paper's scope is explicitly about well-structured sets; demanding generalization beyond this is scope creep.

- **Weakness: Complexity/density of proofs.** Generic criticism applicable to any mathematical paper; removed as not specific.

## Novel Insights

The duality between "adaptive smoothness enables acceleration" and "adaptive variance enables dimension-free rates" reveals a deeper structural principle: under non-Euclidean geometry, averaging (of iterates for acceleration, of gradients for variance reduction) can fail to reduce norms effectively, because the dual norm ∥·∥_{H,*} is the *infimum* of individual dual norms rather than the norm at any fixed H. Adaptive assumptions circumvent this by ensuring uniform geometric control that makes averaging meaningful again. This suggests that the practical success of adaptive methods may stem not from any single geometric alignment but from a stronger "uniform adaptivity" property of the loss landscape that simultaneously enables both acceleration and variance reduction—a property that standard smoothness/variance simply cannot capture.

## Suggestions

- Add an explicit "Limitations" paragraph in the main text acknowledging that (1) acceleration is proven only for convex losses, (2) the magnitude of the adaptive-smoothness-vs-standard-smoothness gap in practice is unknown, and (3) the adaptive variance assumption requires empirical validation.

- State the hyperparameter equivalences between weighted/cumulative/EMA variants in the main text (Section 3.2), not just the appendix, since this directly affects how readers interpret the convergence guarantees for standard Adam.

- In Section 4.2, add a brief discussion of the computational overhead (if any) of the projected variant (Algorithm 8) versus standard Adam, to help practitioners assess feasibility.

- Consider adding a simple 2D visualization (extending Figure 1) that contrasts convergence trajectories of adaptive methods vs. NSD on a function where Λ_H(f) ≪ d · L_{∥·∥_H}(f), to build intuition for when adaptive smoothness provides a practical advantage.