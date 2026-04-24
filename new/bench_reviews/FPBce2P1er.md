## Summary

This paper develops a theory of compositional generalization in kernel models with fixed, compositionally structured representations. Its core theoretical contribution is Theorem 4.2, which proves that such models are constrained to *conjunction-wise additive* computations—summing values over conjunctions of components seen during training. The paper further derives an exact closed-form prediction for the “memorization leak” failure mode (Proposition 5.1) and introduces a representational salience metric $S(k;C)$ to characterize how architectural choices affect compositional geometry. Finally, the authors test whether these kernel-theory predictions qualitatively describe the behavior of end-to-end trained deep networks (ConvNets, ResNets, ViTs) on natural-image versions of symbolic addition and context-dependence tasks.

## Strengths

- **Exact finite-sample characterization of kernel-model constraints.** Theorem 4.2 provides a novel and precise functional characterization: any kernel model with a compositionally structured representation decomposes into a sum over familiar conjunctions (Eq. 2). This advances beyond prior asymptotic or task-specific analyses (Abbe et al., 2023; Lippel et al., 2024) by giving exact constraints for finite numbers of components.
- **Closed-form quantitative prediction of a failure mode.** Proposition 5.1 derives an exact formula (Eq. 3) for the memorization-leak slope in symbolic addition, showing that test-set predictions are compressed by a factor $m$ depending only on salience $S(1;2)$ and training-set size $p$. Figures 4a-b confirm this predicted linear distortion.
- **Useful conceptual and methodological tools.** The salience metric $S(k;C)$ compresses the space of compositionally structured kernels into interpretable parameters, and the distinction between “memorization leak” (inductive bias toward full conjunctions) and “shortcut bias” (exploitation of spurious correlations) provides a productive typology for organizing future work.

## Weaknesses

### Fatal
None.

### Major
- **Empirical validation lacks kernel-regime diagnostics and baselines.** Section 6 trains finite-width, end-to-end networks with backpropagation—architectures operating in the feature-learning regime—yet frames these experiments as validating a theory that assumes fixed representations and kernel-regime learning (Section 2, Section 3.2). The paper does not verify kernel-regime conditions (e.g., via NTK/NNGP baselines, lazy-training diagnostics, or checks that representation gradients are small), nor does it justify why learned representations should remain compositionally structured. Because the central empirical claim is that the theory “captures the behavior of deep neural networks” (Abstract, Section 6), this omission weakens the asserted bridge from kernel theory to deep learning. The authors’ honest note that networks did not perfectly fit the training split (Section 6) further underscores that the experiments test a regime beyond the theory’s stated assumptions.
- **Abstract and introduction overclaim empirical scope.** Phrases such as “captures the behavior of deep neural networks” and “we empirically validate our theory” frame the deep-network experiments as direct validation. While the Discussion (Section 7) appropriately softens this to “captures many qualitative phenomena,” the headline framing in the abstract and introduction is stronger than the evidence supports. The gap between the theory’s kernel-regime assumptions and the feature-learning experiments makes the deep-network claims speculative rather than established.

### Minor
- **Conjunction-wise additivity fit is not quantified in the main text.** The paper asserts that “a conjunction-wise additive model was highly predictive of the model responses” (Section 6), but defers quantitative metrics (e.g., $R^2$) to Appendix D.3. Without at least a summary statistic in the main text, readers cannot assess whether the additive decomposition is a tight constraint or a loose first-order approximation.
- **Shortcut-bias analysis lacks formal rigor comparable to the memorization-leak analysis.** While Proposition 5.1 gives an exact result for memorization leak, the shortcut-bias claim for context dependence (CD-3) rests on empirical weight magnitudes (Fig. 4d) showing correlation but not causal necessity. A formal proposition characterizing when shortcut bias arises would strengthen the paper’s treatment of this failure mode.
- **Transitive equivalence is not tested in deep networks.** This task is highlighted as a fundamental theoretical limitation (Section 4.3), so an empirical demonstration in deep networks—showing whether they also fail, as the theory predicts—would strengthen the argument that the identified restrictions identify real deep-network failures.

### Trivial
- Figure captions and some cross-references could be tightened for clarity (e.g., distinguishing training-set sizes in Fig. 5c legend).

## Nice-to-Haves
- A direct comparison between the trained deep networks and exact kernel regression (NTK or NNGP) using the same architectures and data would clarify where and how feature learning causes deviation from the kernel predictions.
- Explicit test cases showing where deep networks deviate from conjunction-wise additivity would help bound the approximation.
- Analysis or discussion of when end-to-end trained networks deviate from the fixed-representation kernel theory, and whether architectural or training modifications that push networks out of the kernel regime remediate the identified failure modes.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that deep networks are “constrained” to conjunction-wise additivity.** The harsh critic attributes the word “constrained” to the deep-network analysis, but the paper actually uses the more careful phrasing “tend to implement” (Section 6: “Deep networks tend to implement conjunction-wise additive computations”). This is a misreading.
- **Claim that intermediate-layer salience $S(1;2)$ is “unexplained.”** The paper explicitly justifies measuring $S(1;2)$ in an intermediate ConvNet layer by noting that “the ConvNets’ local weight structure should produce a more conjunctive representation for digits that are closer together” (Section 6). The harsh critic overlooked this explanation.
- **Criticism about missing appendix proofs.** The parser strips appendices from all papers; they exist in the original submission. Complaints about missing appendix material should be disregarded.
- **Request for NTK/NNGP as a “missing baseline” in the sense of unfair comparison.** The asymmetry here would favor the baseline (exact kernel) over the author’s method, so this is not a standard missing-baseline criticism; rather, it is a request for regime validation, which is already captured above.

## Novel Insights

None beyond the paper’s own contributions. The reviews do not surface any genuinely novel observations independent of what the paper already presents; rather, they debate the strength of the evidence bridging kernel theory to deep-network behavior.

## Suggestions

1. **Add kernel-regime baselines.** Run exact kernel regression (NTK or NNGP) with the same architectures on the same compositional tasks. This would directly test whether the deep networks behave like the kernel models the theory describes, and would quantify the deviation introduced by feature learning.
2. **Soften abstract/introduction claims.** Replace “captures the behavior of deep neural networks” with language closer to the Discussion, e.g., “captures qualitative phenomena in deep neural networks” or “predicts key behavioral signatures in deep networks trained on compositional tasks.”
3. **Quantify the additive decomposition in the main text.** Report the variance explained ($R^2$) of the conjunction-wise additive fit to deep-network outputs, ideally with a comparison to a flexible non-additive baseline, so readers can assess the tightness of the constraint.
4. **Derive or formally characterize shortcut-bias conditions.** A proposition analogous to Proposition 5.1 for context dependence would make the paper’s treatment of shortcut bias as rigorous as its treatment of memorization leak.

## Score and Decision

**Calibration comparison:**
- **High anchor:** `7VPTUWkiDQ` (avg 7.33, Accept oral) — strong compositional-generalization theory with limited but well-aligned synthetic experiments. The paper under review has comparably strong theory but less well-aligned empirical validation.
- **High anchor:** `3SJE1WLB4M` (avg 8.00, Accept spotlight) — pure kernel theory with minimal experiments. The paper under review has stronger empirical content but also more empirical tension.
- **Medium anchor:** `1op5YGZu8X` (avg 6.40, Accept poster) — NTK theory extended to adversarial training with real-world experiments; reviewers raised concerns about regime validity (lazy training vs. adversarial robustness). The paper under review has a similar regime-validity tension but does not derive an explicit kernel algorithm for its experiments.
- **Medium anchor:** `e1ETy9XW0T` (avg 5.50, Reject) — strong theory but narrow, noisy, and unconvincing empirical validation. The paper under review has broader and more coherent experiments, though still mismatched to theory.
- **Low anchor:** `QUkgKycSl3` (avg 3.67, Reject) — ambiguous claims that do not hold for standard methods. The paper under review does not suffer from this; its theoretical claims are precise and well-supported.

The paper’s theoretical core (Theorem 4.2, Proposition 5.1) is sound and genuinely advances the understanding of compositional generalization in kernel models. However, the empirical bridge to deep networks is presented more strongly than the evidence supports: the experiments use feature-learning networks without kernel-regime diagnostics or exact kernel baselines, and the abstract frames this as “capturing the behavior of deep neural networks.” This mismatch is a real weakness that weighs against acceptance, but it does not invalidate the theoretical contribution. Relative to the anchors, the paper sits between the accepted poster with regime concerns (`1op5YGZu8X`, 6.40) and the rejected paper with weak empirics (`e1ETy9XW0T`, 5.50). The theory is stronger than the latter, but the empirical overreach is more pronounced than the former.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>