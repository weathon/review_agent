Now let me do calibration searches to anchor the score.Now I have enough material to write the final consolidated review.

---

## Summary

This paper establishes the first generalization bounds for Kolmogorov–Arnold Networks (KANs), covering two settings: (1) activation functions expressed as linear combinations of basis functions (Theorems 1–3), yielding bounds that scale with the $\ell_1$ norm of coefficient matrices and Lipschitz constants with only logarithmic dependence on combinatorial parameters; and (2) activation functions lying in a low-rank Reproducing Kernel Hilbert Space (Theorems 4–5), yielding polynomial dependence on the underlying ranks. The paper also presents empirical studies tracking the derived complexity measure $\tilde{\alpha}$ against excess loss during training on simulated and real datasets.

---

## Strengths

- **First generalization bounds for KANs with logarithmic combinatorial dependence (Theorem 1):** The bound $\log \mathcal{N}(\mathcal{H}_L(\mathbf{X}), \epsilon, \|\cdot\|_2) \leq \tilde{\alpha}^3 \log(2\tilde{d}\tilde{p})/\epsilon^2$ places the number of nodes and basis functions only inside the logarithm. This fills a genuine gap—no prior generalization theory existed for this recently prominent architecture.

- **Extension to unbounded loss functions (Theorem 3):** Unlike Bartlett et al. (2017), which requires bounded (ramp) loss, Theorem 3 uses a truncation argument to accommodate squared loss, pinball loss, and Huber loss via Assumption 4, meaningfully broadening the applicable problem classes.

- **Novel low-rank RKHS bounds (Theorems 4–5):** The RKHS setting is explicitly flagged in Section 1.2 as having no comparable result in the MLP literature ("we are not aware of comparable results for MLPs in the recent literature"). Composing low-rank RKHS covering number bounds (via Sobolev space equivalence) through a multi-layer KAN structure is a more technically original contribution than the basis-function case.

- **Computable complexity measure:** The quantity $\tilde{\alpha} = (D\prod_{j=1}^L \rho_j)^{2/3}\sum_{i=1}^L (B_i c_i)^{2/3}$ (for the $C=0$ case) depends only on $\ell_1$ norms of coefficient matrices and estimated Lipschitz constants—quantities directly extractable from a trained KAN—making the theory operationally applicable.

- **Flexible operator-norm framework (Proposition 1):** The chain-rule for covering numbers is adapted to layer-dependent norms, generalizing the iterative argument of Anthony et al. (1999) and Bartlett et al. (2017) to the KAN setting.

- **Connection to fine-tuning (Remark 6):** The RKHS framework explicitly covers LoRA-style fine-tuning where update directions lie in a low-rank subspace, connecting the theory to a practical and widely-used paradigm.

---

## Weaknesses

### Fatal
None.

### Major

- **Empirical validation weakened by normalization procedure:** The paper states (Section 3): "we normalize the values of the complexity measures so that the maximum value of the complexity measure is equal to the last value of the excess loss." This normalization forces the complexity curve and the excess loss curve to share one endpoint by construction. Since both excess loss and $\tilde{\alpha}$ tend to increase monotonically during training (as the model overfits), two monotone-increasing series normalized to share a terminal value will visually co-move. No unscaled correlation statistics (Pearson, Spearman), no formal test, and no residual analysis are provided. The claim in Section 1.1 that the complexity measure is "tightly correlated with the excess loss, demonstrating the complexity measure's practical relevance" is therefore overstated. The theoretical results are valid on their own, but the empirical section fails to deliver rigorous support for practical relevance.

- **No convergence rate analysis for Theorem 3:** The dominant term in Theorem 3's bound involves $\zeta_0 = \tilde{\alpha}^3 \log(2\tilde{d}\tilde{p})(nC''/\tau)^{2/s'}$, which scales as $n^{2/s'}$. The bound's leading term thus scales as $n^{1/s'-1}$, which converges to zero only when $s' > 1$—yet Assumption 4 only requires $s' > 0$. The paper neither derives the resulting convergence rate, nor specifies how $\tau$ should scale with $n$ to balance probability and rate, nor states the conditions under which the bound is non-trivial. Without this analysis, it is unclear whether Theorem 3 implies consistent learning in any concrete regime. This is a standard component of learning-theory papers.

### Minor

- **Theoretical contribution incremental for the basis-function case:** The paper honestly acknowledges (Section 1.2) that the proof strategy for Theorems 1–3 is "closely related to" Anthony et al. (1999) and Bartlett et al. (2017), applying Maurey's lemma to the feature-expanded linear structure. Once one observes that KAN activations $\psi_i(\mathbf{x}) = \sum_k \beta_{i,j,k} g_{j,k}(x_j)$ are linear in $\beta$ for fixed basis functions, the covering number bound follows the MLP pathway. The paper does not formally characterize settings where the KAN bound is provably tighter than an analogous MLP bound (a point the Discussion honestly flags as open). The RKHS case is more novel, partially compensating for this. The paper should more precisely characterize where KAN-specific structure yields a genuine advantage.

- **Corollary 2 references incorrect assumptions:** Corollary 2 (in the RKHS Section 2.3) states "Suppose Assumptions 1, 2 and 4 hold," but Assumption 2 is specific to the basis-function parameterization of Section 2.2; it has no role in the RKHS setting. The correct reference should be Assumptions 4 and 5. This is likely a copy-paste error but points to insufficient proofreading of the RKHS section.

- **Theorem 4 restricted to $\tilde{d} > \nu$:** The constraint $\tilde{d} := \max_i d_i > \nu$ is stated without analysis of how binding it is in practice. For deep, narrow networks or highly smooth functions (large $\nu$), the theorem is inapplicable. At minimum, the implications of this restriction deserve discussion.

### Trivial

- **"No combinatorial dependence" claim requires mild qualification:** While technically accurate, the claim that the bound has no combinatorial dependence outside logarithms holds only when $B_l$ is treated as a fixed constant. Remark 4 notes $\|\mathbf{B}_l\|_1 \leq B_{\max}\|\mathbf{B}_l\|_0$, so for dense networks $B_l$ grows with width. This is the standard trade-off in norm-based bounds but warrants a brief qualifying remark in the abstract/introduction.

---

## Nice-to-Haves

- **Unnormalized empirical correlation:** Reporting $\tilde{\alpha}$ and excess loss on their natural scales, along with a Pearson or Spearman correlation coefficient across training epochs, would turn the visual illustration into rigorous evidence for practical relevance.
- **Regularization experiment:** Section 1.1 suggests the complexity measure "could be used as a regularizer." A simple comparison of $\lambda\tilde{\alpha}$-penalized training vs. unpenalized training would validate this suggestion.
- **Comparison between KAN and MLP bounds on matched networks:** A single numerical experiment computing both the KAN bound (Theorem 3) and the Bartlett et al. (2017) MLP bound for networks of equivalent empirical capacity would clarify whether the KAN-specific analysis provides any practical advantage.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "KAN activations are mathematically equivalent to MLP with feature expansion, making the contribution trivial."** While the observation is correct, the paper is explicit that this extension is non-trivial (different norms, unbounded losses, RKHS case). The extension to the RKHS setting explicitly breaks from the MLP analogy. Kept as a Minor rather than structural/fatal weakness.

- **Harsh Critic: "The experimental section is the paper's ONLY validation and is structurally invalid."** The theoretical contributions stand independently of the empirical section. A weak empirical section is a Major weakness, not a fatal one in a theory paper. Adjusted to Major.

- **Strength Finder: "Strong empirical correlation between complexity and excess loss (Figure 2)."** The normalization concern weakens this claim. Dropped as an unqualified strength; the valid part (that curves show qualitatively similar shapes) is folded into the discussion.

---

## Novel Insights

The paper's most genuinely novel observation—underappreciated even in the strength analyses—is the connection between the RKHS low-rank KAN bounds and LoRA-style fine-tuning (Remark 6): if parameter updates during fine-tuning live in a low-rank RKHS subspace, the generalization error of the fine-tuned model is controlled by those ranks. This positions the KAN RKHS framework as a potential theoretical foundation for understanding low-rank adaptation of network activations, which has no known parallel in MLP generalization theory.

---

## Suggestions

1. Remove the endpoint normalization from Figure 2 and add a quantitative correlation statistic (Spearman $\rho$ across epochs) for each of the six experimental settings.
2. In Theorem 3, explicitly state the range of $s'$ for which the bound is non-trivial, derive the resulting convergence rate when $s' > 1$, and discuss how $\tau$ should be chosen as a function of $n$.
3. Fix the assumption reference in Corollary 2 (Section 2.3) from "Assumptions 1, 2, and 4" to "Assumptions 4 and 5."
4. Add a paragraph in Section 2.2 giving a worked example comparing the KAN complexity $\tilde{\alpha}$ to the MLP spectral-norm product bound for networks with equivalent empirical performance, even qualitatively, to support the claim of KAN-specific advantage.

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg Score | Relevance to paper under review |
|------|-----------|--------------------------------|
| `hiHZVUIYik.md` | 7.33 (Accept spotlight) | Path-norm toolkit for generalization bounds on general ReLU networks — stronger novelty (full DAG networks, skip connections), more rigorous experiments on ImageNet; paper under review is narrower. |
| `NkmJotfL42.md` | 7.0 (Accept poster) | Studies tightness of generalization bounds — conceptually bolder question, strong negative results; more impactful than providing first-of-kind bounds. |
| `q6zrZbth1F.md` | 7.0 (Accept poster) | Sample complexity and entropy bounds for ReLU networks with new lower-bound insights — comparable style, comparable novelty, but derives convergence rates (which this paper does not). |
| `vsLohTBH4h.md` | 4.5 (Reject) | Refined generalization bounds for DRM/PINNs — also incremental theory, also lacks depth in novelty arguments; however, the KAN paper addresses a clearer novelty gap (first results for a brand-new architecture). |
| `Y7lc4aZ4iP.md` | 4.0 (Reject) | Rademacher bounds for CNNs — more incremental than this paper, weaker motivation. |
| `Hh0Cg4epYY.md` | 2.33 (Reject) | Unclear presentation, insufficient detail — much weaker than this paper, which is clearly written and technically sound. |

**Positioning:** The paper under review is better than the 4.0–4.5 anchors (fills a genuine novelty gap, has a more novel RKHS contribution, is more clearly written), but falls short of the 7.0 anchors (those derive convergence rates, have stronger experimental validation, or address more conceptually ambitious questions). The missing convergence rate analysis and the weakened empirical section are real gaps. The basis-function case is solid but incremental; the RKHS case is more original but has the assumption-reference error and the $\tilde{d}>\nu$ restriction. Overall this sits in the upper-medium band.

**Score: 5.5 / 10 — Borderline Reject**

The paper makes a genuine first-of-kind contribution in applying learning theory tools to KANs, and the RKHS results are novel. However, the absence of convergence rate analysis (standard for learning theory submissions), the compromised empirical validation (endpoint normalization), and the incremental methodology of the main basis-function result collectively keep it below the ICLR acceptance threshold. With a rigorous empirical section and a convergence rate derivation, this would be a solid accept.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>