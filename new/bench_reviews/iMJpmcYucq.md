Now let me do calibration searches to anchor the score:Now I have enough to write the final review. Let me synthesize everything.

---

## Summary

This paper proposes SVRGVI, a variance-reduced estimator for Gaussian variational inference on the Bures–Wasserstein (BW) manifold. The core idea is to use the Stein/Hyvärinen score of the current variational distribution $\Sigma_k^{-1}(X_k - m_k)$ as a control variate for the Monte Carlo gradient estimator $\nabla V(X_k)$, exploiting the fact that $\mu_k \approx \pi$ as optimization proceeds. The method is a single-line modification to SGVI, supported by rigorous variance reduction theorems and improved convergence bounds.

---

## Strengths

- **Principled control variate construction** (Section 3): The estimator $\tilde{b}_k = \nabla V(X_k) - c_k \Sigma_k^{-1}(X_k - m_k)$ follows naturally from the Stein score of the variational distribution. The intuition that $\nabla \log \pi(x) \approx \nabla \log \mu_k(x)$ near the optimum is clean and directly motivates the construction.

- **Exact variance decomposition via Stein's lemma (Lemma 1)**: The variance of the proposed estimator is derived in closed form as the MC variance plus $c^2\text{Tr}(\Sigma^{-1}) - 2c\,\text{Tr}(\mathbb{E}\nabla^2 V)$, making variance comparison transparent. This also yields the optimal adaptive $c_k^* = \text{Tr}(\mathbb{E}_{\mu_k}\nabla^2 V)/\text{Tr}(\Sigma_k^{-1})$, theoretically grounding the practical choice of $c \approx 0.9$.

- **Guaranteed variance reduction in two meaningful regimes** (Theorems 1–2): Theorem 1 proves variance reduction in a neighborhood of $\hat\pi$ for any smooth target; Theorem 2 proves it for all $\mu$ with sufficiently large covariance when $V$ is strongly convex. Together they cover both early and late optimization.

- **Convergence bounds strictly improved** (Theorems 3–4): Both the convex and strongly convex convergence bounds from Diao et al. (2023) are improved by factors involving $(1 - \tau)$, directly linking variance reduction to faster convergence and a lower noise floor.

- **Negligible computational overhead** (Section 3): The Cholesky factor already computed for Gaussian sampling is reused to compute $\Sigma_k^{-1}(X_k - m_k)$ at $O(d^2)$ cost, dominated by the $O(d^3)$ existing cost — making this a strict improvement per iteration within BW methods.

- **Consistent within-class empirical gains** (Figures 3–4): Across Gaussian, Student's $t$, and logistic regression targets, SVRGVI consistently and substantially outperforms SGVI and BWGD at the same step size, with the gap widening with dimension.

- **Honest identification of Hessian control variate being vacuous** (Section 3): The paper correctly notes that applying the same reasoning to $\tilde{S}_k$ yields the deterministic $W_k = \Sigma_k^{-1}$, providing no variance reduction and hence leaving the standard Hessian estimator unchanged. This avoids a tempting but futile extension.

---

## Weaknesses

### Fatal
None.

### Major

- **Gaussian target is the idealized edge case for BW variance reduction, yet it is the headline showcase.** For a Gaussian target $V(x) = \frac{1}{2}(x-m^*)^\top\Sigma^{*-1}(x-m^*)$, the Hessian $\nabla^2 V = \Sigma^{*-1}$ is constant, so $S_k = \nabla^2 V(X_k)$ has *zero* variance regardless of algorithm — only gradient noise can be reduced. The "5 orders of magnitude" improvement at $d=200$ (Fig. 3c) therefore reflects variance reduction under *zero* Hessian noise, the best possible case for BW methods generally. The paper's abstract claim of "order-of-magnitude improvements" is accurate, but the headline figure is not representative of general targets where both noise sources are active. The non-Gaussian experiments (Fig. 4) do show consistent improvements but with a notably more modest margin. The paper does not acknowledge this structural difference between the experimental settings, leaving the reader with an inflated impression of typical gains.

### Minor

- **Hessian noise ($S_k$) is the remaining noise floor but receives little analysis for non-Gaussian targets.** The paper honestly notes in Remark 3 that "even when we set $\tau_{\max,\infty} = 0$, the noise terms in the bounds of Thm. 3 and Thm. 4 would not disappear because of another source of randomness coming from $S_k$." However, it does not analyze or empirically characterize the relative magnitude of Hessian noise vs. gradient noise for non-Gaussian targets. For the Student's $t$ or logistic regression cases where Hessian noise is nonzero, understanding whether the Hessian is the dominant residual bottleneck would strengthen the paper's narrative and guide future work.

- **Theoretical $c_k$ prescription vs. empirical fixed $c = 0.9$.** Remark 1 derives the optimal adaptive $c_k^* = \text{Tr}(\mathbb{E}_{\mu_k}\nabla^2 V)/\text{Tr}(\Sigma_k^{-1})$. The experiments fix $c = 0.9$ throughout, but no experiment compares fixed $c = 0.9$ to the adaptive schedule, especially in early iterations where $c^*$ may deviate substantially from 1. The theoretical prescription and the empirical default are not bridged experimentally.

- **EVI comparison lacks wall-clock context.** BW methods require full Hessian evaluation ($O(d^2)$ per evaluation, $O(d^3)$ for Cholesky) while EVI with reparameterization gradients requires only gradient evaluations. The paper correctly handles this by reporting only EVI's final accuracy from a carefully-optimized run (Section 5), and does not claim a per-iteration fair comparison. However, the conclusion "We also clearly outperform EVI in higher dimensions" would benefit from a note that this comparison is at equal final-accuracy rather than equal computational budget — the current phrasing overstates the practical advantage relative to gradient-only baselines.

- **All experiments are on synthetic targets.** The paper acknowledges this briefly in Section 6 ("our experiments focused on synthetic targets"). The logistic regression is on synthetic covariates $X_i \sim \mathcal{N}(0, I_d)$ with no real data. While this expands on prior BW-VI work, the claim that BW methods are now "practical" and that SVRGVI "should always be used" goes somewhat beyond what synthetic benchmarks can establish.

- **The "minor correction to SGVI's bound" is relegated to a footnote.** Footnote 1 reads "With a minor correction to the coefficients in SGVI's bound." A correction to a prior result's theorem should be stated explicitly in the text with the corrected formula, not buried in a footnote.

### Trivial
- The $\mu_\text{best}$ reference distribution in the logistic regression experiment is defined as "the distribution that obtains the smallest $\mathcal{F}$ among all iterations of all algorithms." The paper should clarify which algorithm dominates this best-found distribution, to confirm the metric is not inadvertently biased toward SVRGVI.

---

## Nice-to-Haves

- **Wall-clock or FLOP-controlled comparison against EVI**, at least for one dimension, to provide a complete picture of where BW methods stand in practical compute terms.
- **Variance-over-iterations plot for a non-Gaussian target in the main paper** (currently in Appendix B.4), to directly confirm the variance reduction claim for the more challenging setting without requiring readers to consult the appendix.
- **Real posterior inference task** (e.g., hierarchical model, Gaussian process regression on real data) as a single additional experiment to substantiate the practicality claim.
- **Comparison between fixed $c=0.9$ and adaptive $c_k^*$**, especially in early iterations, to demonstrate the practical value of the closed-form optimal coefficient.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic — "EVI comparison is methodologically invalid":** The paper already explicitly acknowledges the per-iteration cost difference (Section 5: "as the per-iteration cost of these methods is different from the BW methods, we only report the final accuracy for carefully optimized approximations"). The comparison is presented correctly; the framing criticism is partially valid (retained as a minor weakness) but the core methodological complaint is addressed by the paper.

- **Harsh Critic — "μ_best is biased toward SVRGVI by construction":** This misreads the paper. μ_best is defined as the best distribution found by *any* algorithm at *any* iteration, not exclusively by SVRGVI. There is no inherent bias by construction; it is just an unusual reference point. Retained only as a trivial note to clarify.

- **Strength Finder — "Figure 2 provides intuitive validation" (generic):** Kept implicitly in Lemma 1 strength, but the standalone strength claim about Figure 2 as a presentation strength is not independently notable enough to list separately.

---

## Novel Insights

The key structural observation — latent in the paper but underemphasized — is that the BW-VI gradient noise decomposes into two structurally different components: gradient noise (reducible by the proposed control variate, which becomes exact at the Gaussian optimum) and Hessian noise (irreducible by the same control variate since the Gaussian control variate for $\nabla^2 V$ is already deterministic). This decomposition clarifies *why* the Gaussian target is the ideal showcase (zero Hessian noise), *why* non-Gaussian improvements are more modest, and *where* future variance reduction efforts should focus. The connection noted in Section 6 — that BW methods and Roeder et al. (2017)'s Euclidean reparameterization gradient are different discretizations of the same continuous-time flow (Särkkä's ODEs) — is a genuinely interesting structural result that deserves more prominence; it suggests that variance reduction techniques may transfer between these two seemingly different frameworks.

---

## Suggestions

1. Add a sentence or two in Section 5 contextualizing the Gaussian target results: note that Hessian noise is zero for this target and that the non-Gaussian results provide a more representative picture of general performance.
2. Promote footnote 1 to the main text with the corrected formula side-by-side with the original.
3. Include one variance-decomposition plot (gradient noise vs. Hessian noise along iterates for the Student's $t$ target) in the main paper to directly support the claim that gradient variance is the bottleneck and SVRGVI resolves it.
4. Either add a real posterior inference experiment or soften the "should always be used" conclusion to "should always be preferred among BW methods."

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Human Score | Comparison to paper under review |
|------|----------------|----------------------------------|
| `/home/wg25r/review_agent/human_reviews/zlkXLb3wpF.md` | 7.5 (Accept) | Closest thematic match — variance reduction by minimal algorithmic change for flow-based VI; stronger due to broader experimental scope (natural sciences real data) and applicability to both forward/backward KL. |
| `/home/wg25r/review_agent/human_reviews/PP1rudnxiW.md` | 7.2 (Accept) | Transport + variational inference framework; broader scope and real experiments but somewhat more incremental theory. |
| `/home/wg25r/review_agent/human_reviews/Re4Z3Wt2DS.md` | 6.8 (Reject) | Wasserstein-based VI with Gaussian mixtures; rejected partly for experimental and clarity issues. Paper under review is theoretically stronger and cleaner. |
| `/home/wg25r/review_agent/human_reviews/gFBTNDNDUG.md` | 6.0 (Reject) | Deep learning + Wasserstein gradient flows; rejected for experimental scope limitations similar to this paper. |
| `/home/wg25r/review_agent/human_reviews/fjf3YenThE.md` | 5.3 (Accept) | Zero-order variance reduction — less elegant theory and similar synthetic-only evaluation; paper under review has significantly stronger theory. |
| `/home/wg25r/review_agent/human_reviews/tuuEvgfxr5.md` | 2.5 (Reject) | Bayesian pseudo-coresets; clearly weaker paper with fundamental methodological issues — serves as low anchor only. |

**Reasoning:** The paper sits clearly above the medium anchor cluster (5.3–6.0) due to its principled theory and clean algorithmic contribution. It falls below the high-scoring anchors (7.2–7.5) primarily because its experimental scope is entirely synthetic, preventing the "practical BW-VI" claim from being fully validated. The key limitation — that the headline 5-order-of-magnitude result is from a Gaussian target (zero Hessian noise) — is real but does not invalidate the contribution; the non-Gaussian results still show consistent improvement. The paper is closest in spirit and quality to zlkXLb3wpF.md and PP1rudnxiW.md but with narrower experiments, placing it below those at approximately 6.5.

**Axes summary:** *Originality*: Good — principled application of control variates to BW-VI gradient, novel and well-motivated. *Importance*: Moderate — addresses a genuine bottleneck in an emerging sub-field of probabilistic inference. *Claims well-supported*: Mostly — within-class comparisons are rigorous; cross-class (vs. EVI) and practical claims slightly overclaimed. *Soundness of experiments*: Adequate for a theory paper but limited to synthetic settings. *Clarity*: Good — the paper is well-written and the algorithm is a transparent single-line change. *Value to community*: Solid — provides an immediately adoptable, theoretically justified improvement to BW-VI methods.

**Decision: Accept (poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>