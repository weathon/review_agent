## Summary

This paper provides the first generalization bounds for Kolmogorov–Arnold Networks (KANs). For activation functions represented as linear combinations of basis functions, Theorem 1 establishes a covering-number bound that scales with an $l_1$/Lipschitz complexity measure $\tilde{\alpha}$ and depends on width only logarithmically. Theorem 3 extends this to unbounded Lipschitz losses. For low-rank RKHS activations, Theorems 4 and 5 give polynomial bounds in the ranks. The authors also present SGD experiments on simulated and real data correlating the complexity measure with excess loss.

## Strengths

- **Timely and needed contribution.** KANs have attracted substantial recent interest, yet no prior work had established learning-theoretic generalization guarantees for this architecture. The paper fills a clear gap.
- **Clean covering-number bound for basis-function KANs (Theorem 1).** The bound $\log \mathcal{N}(\mathcal{H}_L(\mathbf{X}), \epsilon, \|\cdot\|_2) \leq \tilde{\alpha}^3 \log(2\tilde{d}\tilde{p}) / \epsilon^2$ depends on the $l_1$ norms of coefficient matrices and per-layer Lipschitz constants, with layer widths and basis counts entering only inside a logarithm. This is a meaningful structural result.
- **Natural complexity measure.** The quantity $\tilde{\alpha}$ is interpretable and provides a candidate regularizer for KAN training.

## Weaknesses

### Major

- **Theorem 5 (low-rank, unbounded loss) has a structural scaling issue that undermines its utility.** The complexity term $\xi_0$ in Equation (6) contains the factor $(nC''/\tau)^{2/s'}$ raised to the layer-dependent power $(d_{i-1}/\nu)\vee 1$. For the dominant layer with $d_{i-1} = \bar{d} > \nu$, the main term in Theorem 5 scales roughly as $n^{2/s' - \nu/(2\bar{d})}$. Because the theorem assumes only $\bar{d} > \nu$ and Assumption 4 allows any $s' > 0$, the bound is non-decaying—and in fact grows with $n$—unless $s'$ exceeds $4\bar{d}/\nu$. Since $\bar{d}$ is the maximum layer width, this requires the number of moments of $B(y)$ to grow linearly with network width, which is unrealistic. The paper neither states this requirement nor acknowledges that the bound is vacuous under standard assumptions.
- **Corollary 2 contains multiple copy-paste errors.** It states "Suppose Assumptions 1, 2 and 4 hold," but the low-rank results require Assumption 5, not Assumption 2. It also writes $(C')^{2/\epsilon}$ where $(C')^{2/s}$ or $(C')^{2/s'}$ is intended (compare Corollary 1, which correctly uses $2/s$). These errors suggest insufficient proofreading of the low-rank section.
- **Empirical validation is weak and overclaimed.** The abstract states that "numerical results demonstrate the practical relevance of these bounds." However, the experiments (Section 3 / Figure 2) normalize the complexity curve so that its maximum equals the final excess loss, which makes visual correlation largely inevitable and precludes any assessment of whether the bound is non-vacuous or predictive in absolute terms. The study tracks only a single SGD trajectory per dataset, uses AlexNet-extracted features for MNIST/CIFAR-10 (which obscures end-to-end KAN behavior), and does not compare against simpler baseline complexity measures (e.g., product of Frobenius norms, path norms, or parameter count) or across varying architectures. The correlation shown is therefore generic and does not support the advertised "practical relevance."

### Minor

- **Theorem 3 requires unstated moment conditions.** The leading complexity term in Theorem 3 is $O(\sqrt{\zeta_0}/n)$ with $\zeta_0 \propto (nC''/\tau)^{2/s'}$, giving a rate of $n^{-(1-1/s')}$. For $0 < s' \leq 1$ the bound does not converge; Assumption 4 only requires $s' > 0$. The paper should explicitly state that $s' > 1$ is needed for a non-vacuous rate.
- **Low-rank bounds do depend on combinatorial parameters.** Section 2.3 claims that Theorem 4 has "no explicit dependence on combinatorial parameters," yet $\xi$ depends linearly on $d_i$ and the exponent $\nu/\bar{d}$ depends explicitly on the maximum width $\bar{d}$. This claim should be qualified.
- **Lipschitz proxies are not quantified.** The experiments estimate Lipschitz constants via the loose upper bound in Remark 5, but the paper does not quantify how much slack this introduces relative to the true empirical constants.

### Trivial

- Experimental protocol details (learning rates, batch sizes, initialization, random seeds) are not reported.

## Nice-to-Have

- Cross-architecture scatter plots showing generalization gap vs. complexity measure across multiple widths, depths, or ranks would provide much stronger evidence that $\tilde{\alpha}$ predicts generalization.
- A direct MLP comparison using the same data and an analogous norm product would clarify whether the KAN-specific measure offers any practical advantage.
- An actual numerical evaluation of the bound (not just a normalized curve) would allow readers to assess tightness.

## Removed Points

These points are flagged to be removed, treat them with caution:
- **Criticism that the analysis is a "direct adaptation" of Bartlett et al. (2017).** The paper explicitly acknowledges its relation to Bartlett et al. (2017) and Anthony et al. (1999), and it highlights genuine differences (unbounded loss, low-rank structure, different parameterization). The novelty is incremental in technique but applied to a genuinely new architecture. This is a fair characterization of scope, not a flaw.
- **Complaint that training may leave the restricted class $\mathcal{M}$.** This is a standard caveat in statistical learning theory. The paper notes it explicitly, and it is not a unique weakness of this work.
- **Request for MLP comparison on identical tasks as an obligatory experiment.** While such a comparison would strengthen the paper, omitting it is not a core flaw for a paper whose primary contribution is a *KAN-specific* bound.
- **Nitpicks about experimental protocol underspecification and use of AlexNet features.** These are minor issues that do not threaten the core theoretical contribution.

## Novel Insights

The paper's most valuable insight is that width-independent (up to logarithms) generalization bounds are achievable for basis-function KANs using $l_1$ coefficient norms and layer-wise Lipschitz constants, mirroring the spectral-norm story for MLPs but adapted to KANs' edge-based activation structure. However, the reviews reveal that this insight is currently undermined by insufficient scrutiny of the low-rank extension and by experimental choices that obscure rather than validate the bound's predictive power. If the authors can repair Theorem 5's scaling, state the necessary moment conditions transparently, and provide controlled experiments that compare across architectures and baseline measures, the contribution could become a standard reference for KAN learning theory.

## Suggestions

1. **Fix Theorem 5.** Either correct the scaling of $\xi_0$ if there is a typesetting error, or explicitly state the required moment growth with width and discuss whether the bound can be made non-vacuous for realistic networks.
2. **State convergence conditions.** Add explicit assumptions on $s$ and $s'$ (e.g., $s' > 1$ for Theorem 3; $s' > 4\bar{d}/\nu$ or similar for Theorem 5) so readers know when the bounds yield decaying rates.
3. **Correct Corollary 2.** Replace Assumption 2 with Assumption 5 and fix the exponent from $2/\epsilon$ to $2/s$ (or $2/s'$).
4. **Strengthen experiments.** Provide scatter plots across varying architectures, compare against at least one baseline complexity measure, and report unnormalized bound values to assess absolute tightness.

## Score and Decision

**Calibration reasoning.** I compared this paper against several anchors:
- `/home/wg25r/review_agent/human_reviews/ydlDRUuGm9.md` (avg 6.25, Accept): Another first theory paper for KANs (expressiveness/spectral bias). It had limited experiments but cleaner theory without scaling gaps or corollary typos. The current paper is weaker due to Theorem 5's vacuous scaling and Corollary 2 errors.
- `/home/wg25r/review_agent/human_reviews/hiHZVUIYik.md` (avg 7.33, Accept spotlight): Very comprehensive path-norm toolkit with ResNet/ImageNet evaluation. Much stronger empirically and technically.
- `/home/wg25r/review_agent/human_reviews/e1ETy9XW0T.md` (avg 5.50, Reject): Strong theory but narrow/limited empirical validation. The current paper has similarly weak experiments and additional technical gaps.
- `/home/wg25r/review_agent/human_reviews/vsLohTBH4h.md` (avg 4.50, Reject): Refined PINN bounds with novelty concerns relative to prior work. The current paper is more clearly novel but has comparable technical sloppiness.
- `/home/wg25r/review_agent/human_reviews/fOOOyVhTYV.md` (avg 3.80, Reject): Poorly written and recovered known results. The current paper is substantially better.

The paper sits below the accepted 6.25 KAN theory anchor because of its technical gaps and weaker empirical support, but above the rejected 4.50 PINN anchor because its core contribution (first KAN generalization bounds) is more clearly delineated. A score of **5.0** reflects a borderline submission: the basis-case theory is sound and worthwhile, but the low-rank extension contains a serious scaling limitation, the corollaries have copy-paste errors, and the experiments do not substantiate the claim of practical relevance.

**Score:** 5.0  
**Decision:** Borderline / Weak Reject

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>