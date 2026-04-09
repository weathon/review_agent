## Summary

This paper introduces a framework for deriving worst-case generalization bounds over data-dependent random sets (e.g., optimization trajectories) by combining a novel "random set stability" notion with Rademacher complexity and topological complexity measures. The key advance is replacing the intractable mutual information terms that appear in prior topological/fractal generalization bounds (e.g., Simsekli et al., 2020; Andreeva et al., 2024) with a stability parameter $\beta_n$, yielding the first such bounds that are in principle fully computable. The framework recovers classical stability bounds and uniform convergence as special cases and is validated empirically on ViT and GraphSAGE.

## Strengths

- **Novel and well-motivated stability formulation:** Random set stability (Assumption 3.1) explicitly accounts for algorithmic randomness $U$ in trajectory-based analysis, addressing a real gap in Foster et al. (2019)'s hypothesis set stability which ignores $U$. The connection to classical uniform argument stability via Lemma 3.2 provides a systematic path to verifying the assumption, and Corollary 3.3 demonstrates it concretely for projected SGD.
- **Removal of intractable IT terms:** The framework successfully replaces mutual information terms — which can be infinite and are computationally intractable in general — with a stability parameter that is interpretable and empirically estimable. This is a genuine advance over the PAC-Bayesian random set bounds of Dupuis et al. (2024) and the topological bounds of Andreeva et al. (2024).
- **Unified framework recovering classical results:** The free parameter $J$ interpolates between single-iterate stability bounds ($J=1$, Corollary 3.5) and uniform convergence over fixed hypothesis sets ($J=n$, Corollary 3.6), demonstrating that the framework is not ad hoc but subsumes known settings.

## Weaknesses

### Major:

- **Gap between "fully computable" claim and empirical validation:** The paper claims to provide "the first fully computable topological bounds" (Abstract, Section 4.1), yet the numerical bounds in Table 1 bypass the specific topological coefficients from Theorems 4.3/4.4. Section 5.1 explicitly states: "To avoid the computationally costly evaluation of Lipschitz constants, we estimate a simple upper bound on the Rademacher complexity... we use Massart's lemma." This means the topological quantities $\mathbf{E}_\alpha(W_{S,U})$ and $\mathrm{PMag}(s(\lambda) \cdot W_{S,U})$ appear in the theoretical bounds but are never plugged into a numerical bound evaluation — they are only analyzed for correlation. The "fully computable" characterization is accurate in the sense that no term is fundamentally intractable (unlike mutual information), but the empirical section does not validate the tightness of the topological bounds themselves.

- **Theory-experiment algorithm mismatch:** Corollary 3.3 establishes random set stability specifically for projected SGD under Lipschitz/smooth assumptions. However, all experiments use the AdamW optimizer (Section 5, Appendix C.1), whose adaptive step sizes violate the fixed step-size regime assumed in the stability proof. While the general framework (Assumption 3.1) is optimizer-agnostic, no theoretical result establishes that ADAM satisfies random set stability. The empirical $\beta_n$ estimates thus lack the theoretical grounding that would connect them to the generalization bound.

- **Stability parameter scaling with trajectory length $T$:** Lemma 3.2 sums iterate-level stability $\delta_k$ over $k=1\ldots T$, and the paper notes the worst case yields $\beta_n = O(T^2/n)$. In modern deep learning, $T$ is typically very large (many epochs). If $\beta_n$ grows with $T$, the bound may become vacuous precisely for the long training runs where generalization is most interesting. The paper does not discuss whether $\beta_n$ can remain bounded or decrease as $T \to \infty$ (e.g., due to convergence), which limits the practical applicability of the framework.

### Minor:

- **No baseline bound comparison:** Table 1 reports the proposed bound values but does not compare them against standard uniform stability bounds (e.g., Hardt et al., 2016) or simple Rademacher bounds without topological terms. Without this ablation, it is unclear whether the topological complexity terms actually improve the bound or whether the bound's informativeness is driven entirely by the stability parameter.

- **Optimistic stability estimation without sensitivity analysis:** Algorithm 1 approximates the supremum over $\mathcal{Z}$ using 500 held-out points, which the authors acknowledge yields an "optimistic estimation" of $\beta_n$. Since $\beta_n^{1/3}$ multiplies the entire bound, underestimation could significantly overstate tightness. No sensitivity analysis (e.g., doubling $\beta_n$) is provided to assess robustness.

- **Slower rate trade-off insufficiently justified:** The bound scales as $\beta_n^{1/3}$, yielding roughly $O(n^{-1/3})$ when $\beta_n = O(1/n)$, which is slower than the classical $O(n^{-1/2})$ Rademacher rate or $O(1/n)$ stability rate. Section 4.1 calls this "a deliberate trade-off to maintain boundedness" but does not identify concrete regimes where the IT terms in prior bounds are provably infinite, which would be the strongest motivation for accepting the slower rate.

### Trivial:

- The assumption that $\beta_n^{-2/3}$ is an integer divisor of $n$ (Theorems 4.3, 4.4) is a proof convenience that could be handled by rounding with minor constant adjustments, rather than imposed as a condition on $\beta_n$ and $n$.

## Nice-to-Haves

- High-probability bounds (currently only in-expectation), which would increase practical utility for single-run training guarantees
- A random-labels / overfitting experiment to verify that the bound degrades appropriately when generalization fails
- Sensitivity analysis for the $\beta_n$ estimation bias
- Decomposition plot separating the stability term contribution from the topological term contribution in the bound
- Discussion of computational scalability of persistent homology / magnitude computations for very large models

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Weakness: Dataset scale too small (CIFAR-100, n up to 10,000).** Generic weakness requesting larger datasets when the current scale is sufficient for validating a theoretical framework's structural claims. (Soft rule: weaken generic "need more data" criticism.)
- **Weakness: Missing related works on randomized stability extensions beyond Foster et al. (2019).** Cannot confirm existence of specific works without external sources. (Hard rule: no missing related works.)
- **Weakness: Code not yet released / reproducibility concerns.** Paper states implementation will be available upon publication and provides seeds, hyperparameters, and detailed appendices. (Hard rule: do not question availability of cited/referenced resources.)
- **Weakness: Formatting issues in equations (parser artifacts).** Explicitly excluded per instructions. (Hard rule: remove formatting nitpicks.)
- **Weakness: Expectation-only bounds are a fundamental flaw.** The paper explicitly acknowledges this limitation (Section 6), and expected bounds are standard in the stability literature (Bousquet & Elisseeff, 2002). Demanding high-probability bounds is scope creep for this contribution. (Soft rule: weaken criticisms demanding practices outside paper's scope; moved to nice-to-have.)

## Novel Insights

The interplay between stability and topological complexity uncovered in this paper is genuinely insightful. The product structure $\beta_n^{1/3} \cdot \log \mathbf{C}(W_{S,U})$ in Theorem 4.4 reveals that topological complexity becomes more relevant to the bound as $n$ grows (because $\beta_n$ decreases), while stability dominates at small $n$. This is empirically confirmed by the changing slope of the topological complexity vs. generalization gap regression as $n$ varies (Figures 2–3). This coupling suggests that the informativeness of topological measures for generalization is not universal but depends on the stability regime — a nuanced finding that prior work using only correlations could not articulate.

## Suggestions

- In Table 1, add a column showing the bound value using only the stability term ($2\beta_n$ from Corollary 3.5) alongside the full bound, so readers can directly assess the value added by the Rademacher/topological component.
- Provide at least one experiment using SGD (not ADAM) to close the theory-experiment loop, or add a brief theoretical remark on conditions under which adaptive optimizers satisfy random set stability.
- Discuss concrete examples or regimes where mutual information terms in prior topological bounds are provably infinite or unbounded, to sharpen the motivation for accepting the slower $O(n^{-1/3})$ rate.

---

**Axis Evaluations:**

- **Novelty:** Strong. Random set stability (Assumption 3.1) is a genuine conceptual advance that plausibly bridges two previously disconnected lines of work (stability theory and topological generalization bounds).
- **Technical soundness:** Good with caveats. The theoretical derivations are rigorous and the recovery of classical results is clean. The main concern is the theory-experiment gap (ADAM vs. SGD) and the $T$-dependence of $\beta_n$, which are not adequately addressed.
- **Empirical support:** Moderate. The correlations between topological complexity and generalization are convincing and the stability parameter shows meaningful variation with hyperparameters. However, the specific topological bound values from Theorems 4.3/4.4 are never numerically evaluated, and no baseline comparison is provided.
- **Significance:** Strong, conditional on the framework being extended to cover the optimizers actually used in practice. Removing IT terms from topological bounds is a significant step toward making this line of theory practically relevant.
- **Clarity:** Good. The progression from classical stability → random set stability → Rademacher bound → topological applications is logical. Notation is heavy but generally well-defined.