Now I have thoroughly reviewed the paper, cross-checked the harsh critic's claims against the actual text, and assembled calibration anchors. Let me compose the final review.

## Summary

This paper derives generalization bounds for Kolmogorov–Arnold Networks (KANs) under two activation function regimes: (1) when activations are linear combinations of basis functions, yielding covering-number bounds scaling as $\tilde{\alpha}^3 \log(2\tilde{d}\tilde{p})/\epsilon^2$ where $\tilde{d},\tilde{p}$ appear only logarithmically, and (2) when activations lie in a low-rank RKHS (specifically Sobolev spaces), yielding bounds polynomial in ranks and Lipschitz constants. A truncation argument removes the bounded-loss assumption, extending applicability to regression losses. Experiments on simulated and real datasets show correlation between the derived complexity measure and excess loss.

## Strengths

- **Removal of the bounded-loss assumption (Theorem 3 vs. Theorem 2):** The truncation argument in Section 2.2 relaxes the boundedness requirement to the much weaker Assumption 4 (finite moments), covering squared loss, Huber loss, and pinball loss. This is a genuine improvement over prior MLP generalization work that assumes bounded loss (e.g., Bartlett et al., 2017's ramp loss).

- **Concrete Lipschitz constant estimates for KANs (Remark 5):** The derivation connecting $\rho^* \leq \|A\|_\sigma c_l\sqrt{b_l}$ to the additive structure $\Psi(\mathbf{x}) = A\mathbf{g}(\mathbf{x})$ provides practitioners a computable bound on layer-wise Lipschitz constants, directly linking spectral norms of coefficient matrices and basis function smoothness to the complexity measure.

- **Low-rank RKHS analysis (Theorem 4, Remark 6):** The low-rank RKHS bounds scale polynomially with ranks $r_i$ and Lipschitz constants, with no explicit combinatorial parameter dependence. Remark 6's extension to fine-tuning ($\Phi_l + \Psi$ with $\Psi$ in a low-rank space) directly connects to LoRA-style adaptation, making this section practically relevant.

- **Excess risk bounds for approximate minimizers (Corollary 1):** The condition $\sum_i \mathcal{L}(\hat{f}(\mathbf{x}_i), y_i) \leq \sum_i \mathcal{L}(f^*(\mathbf{x}_i), y_i)$ rather than requiring exact ERM is a meaningful relaxation applicable to SGD solutions.

## Weaknesses

### Fatal

None.

### Major

- **Experimental validation with normalized complexity is nearly vacuous (Section 3, Figure 2).** The paper normalizes the complexity measure "so that the maximum value of the complexity measure is equal to the last value of the excess loss." Since both the complexity measure and excess loss are monotonically decreasing over training epochs (weights shrink, training improves), rescaling one decreasing curve to match an endpoint of another decreasing curve and then claiming "tight correlation" provides almost no empirical evidence for the bound's validity. Meaningful validation would require: (a) showing the *un-normalized* absolute value bears a reasonable relationship to excess loss; (b) varying architectures (depth, width, number of basis functions) to show the complexity measure discriminates between models; (c) comparison against MLP norms on the same tasks. No variance estimation across seeds or architecture sweeps are provided; each plot shows a single training trajectory.

- **The analysis does not exploit KANs' defining additive/univariate structure, yielding bounds that may not differ qualitatively from MLP bounds.** KANs are distinguished from MLPs by their additive structure: each activation is $\psi_i(\mathbf{x}) = \sum_j \psi_{i,j}(x_j)$, a sum of *univariate* functions (Equation 1). However, Assumption 2 permits general *multivariate* basis functions $g_{ij}^{(l)}(\cdot)$, and Remark 2 merely notes this subsumes the KAN structure as a special case, without deriving tighter bounds specific to it. The additive/univariate structure could potentially yield dimension-free bounds via separate covering of each univariate component, which would be a qualitatively different result from MLP bounds that depend on spectral norms of weight matrices. Without exploiting this structure, the paper's bounds are structurally identical to MLP bounds (norm-based covering numbers with logarithmic combinatorial factors), and the "no combinatorial dependence" claim (Section 1.1) is the same type of result as Bartlett et al. (2017) — it means the bound depends on *norms* rather than *dimensions*, not something qualitatively new.

### Minor

- **No quantitative comparison with MLP generalization bounds in the main text.** The paper motivates KANs as "fundamentally different from traditional MLPs" (Section 1.2) but provides no quantitative head-to-head comparison. Section A.1 in the appendix contains such a comparison, but this is critical for evaluating whether KAN-specific theory yields new insights versus simply applying existing MLP theory to another architecture.

- **The low-rank RKHS bound (Theorem 4) has an exponential dependence on $d_{i-1}/\nu$ when $\tilde{d} > \nu$.** The covering number exponent scales as $(d_{i-1}/\nu)^{d_{i-1}/\nu}$, which can be extremely large. The paper does not discuss when this bound is useful or how restrictive the $\tilde{d} > \nu$ condition is.

- **The $\tilde{\alpha}$ formula depends on quantities ($B_i$, $\rho_i$) that likely scale with network dimensions.** While $\tilde{d}$ and $\tilde{p}$ appear only logarithmically, the claim of "no combinatorial dependence" should be qualified — as Remark 5 shows, $\rho^* \leq \|A\|_\sigma c_l\sqrt{b_l}$ where $A \in \mathbb{R}^{d_l \times (d_{l-1}b_l)}$, and spectral norms generally scale with dimensions. This is the same norm-vs-dimension distinction as Bartlett et al. (2017).

### Trivial

None.

## Nice-to-Haves

- Derive tighter bounds exploiting KAN's additive/univariate structure (e.g., separate covering of each univariate component), potentially yielding dimension-dependent improvements that genuinely distinguish KAN bounds from MLP bounds.

- Test the complexity measure as a regularizer during training, as suggested in the discussion, to validate the claim that it "could be used as a regularizer."

- Compute un-normalized bound values to assess quantitative meaningfulness.

- Show failure cases where the complexity measure does not track excess loss (e.g., $\Psi_i(0) \neq 0$, different basis function choices).

## Removed Points

These points are flagged to be removed; treat them with caution:

- *"The covering number analysis is a direct adaptation of Anthony et al. (1999) and Bartlett et al. (2017) with Maurey's sparsification lemma"* — While the techniques are standard, this characterizes the nature of the contribution (incremental adaptation) rather than identifying a flaw. The paper acknowledges the connection explicitly in Section 1.2.

- *"Section A.1 (appendix) apparently contains [the MLP comparison], but it's not in the main text"* — Parser strips appendices; the comparison exists in the original submission. Downgraded to minor since the comparison is present but not prominently featured.

- *"The choice of norms at each layer is discussed only in the appendix — a key technical detail that should appear in the main text"* — Presentation nit; removed per rules.

- *"No variance estimation across seeds"* — The paper's experimental design (single training run per configuration) is standard for learning theory validation papers. This is a nice-to-have, not a fundamental flaw.

- *"The acknowledgment that 'additional empirical studies and theoretical investigation are needed to understand the superiority of KANs over MLPs' implicitly concedes the paper does not establish such superiority"* — This is honest scoping, not a weakness. The paper studies generalization bounds, not KAN vs. MLP superiority.

- *Strength claim: "Strong empirical correlation" from Strength Finder* — Conflicts with the verified major weakness about normalization vacuity. Moved to Removed Points.

## Novel Insights

The truncation argument in Theorem 3, which replaces the bounded-loss assumption with a finite-moment condition (Assumption 4), is a genuinely useful contribution that extends generalization bounds to regression settings. However, the most striking observation is that the paper's analysis treats KAN activations as general multivariate function representations rather than exploiting their defining additive/univariate structure — meaning the bounds are structurally indistinguishable from what one would obtain by applying MLP covering number theory to a network with multivariate activations. The low-rank RKHS analysis (Theorem 4) and its connection to LoRA-style fine-tuning (Remark 6) is a genuine novelty not present in prior MLP generalization work, but it is underexploited: the exponential dependence on $d_{i-1}/\nu$ limits its utility in high-dimensional settings where it would matter most.

## Suggestions

- Re-run experiments with un-normalized complexity measures and include at least a few architecture variants (varying width, depth, or number of basis functions) to demonstrate that the complexity measure discriminates between models, not just tracks a single trajectory.

- Derive covering numbers specifically for the additive structure $\psi_i(\mathbf{x}) = \sum_j \psi_{i,j}(x_j)$ rather than the general multivariate form, which could yield dimension-independent bounds and a genuine theoretical distinction from MLP results.

- Include the MLP comparison from the appendix in the main text and add a paragraph explicitly discussing how $\tilde{\alpha}$ relates to Bartlett et al. (2017)'s spectral norm product.

## Evaluation

**Originality:** Moderate. The adaptation of covering number/Maurey sparsification to KANs is competent but follows the same blueprint as Bartlett et al. (2017). The truncation argument and low-rank RKHS extension add novelty, but the core technical framework is incremental. The paper does not exploit the structural property (additive/univariate) that uniquely defines KANs.

**Importance of research question:** High. Generalization bounds for new architectures like KANs are timely and important for the community.

**Claims support:** The theoretical claims (Theorems 1–5) appear sound. The empirical claim of "tight correlation" between the complexity measure and excess loss is undermined by the normalization issue. The "no combinatorial dependence" claim, while technically correct, is the same norm-vs-dimension insight as existing MLP theory.

**Experimental soundness:** Weak. The normalization procedure and lack of architecture variation make the experiments unconvincing as evidence for the bounds' practical relevance.

**Clarity:** Good. The paper is well-organized and the theoretical exposition is self-contained.

**Community value:** Moderate. Provides a starting point for KAN generalization theory but does not demonstrate that KAN-specific analysis yields insights beyond applying MLP theory.

**Calibration analysis:**
- *High anchors* (avg >7): Path-norm toolkit (7.33) — novel contraction lemmas, recovers/beats known bounds, practical ImageNet evaluation. How many samples (7.0) — first minimax lower bounds for deep ReLU networks. This paper is below these: it adapts rather than invents techniques, and its experiments are weaker.
- *Medium anchors* (avg 4–6): LLM generalization bounds (6.0) — novel empirical Freedman inequality but limited validation. Rademacher CNN bounds (4.0) — extends contraction lemma but vacuous in practice, limited significance. DRM/PINN generalization (4.5) — incremental, limited novelty over prior work. This paper is comparable to the DRM/PINN paper: sound theory but incremental methodology and weak empirical validation.
- *Low anchors* (avg <3): Neural ODE generalization (3.0) — vacuous bounds, plagiarism concerns. This paper is above these: it has genuine theoretical content without plagiarism or triviality.

The paper sits in the borderline range. Its theoretical contributions are sound but incremental, and the experimental validation has a significant weakness. I place it between the DRM/PINN generalization paper (4.5) and the LLM generalization paper (6.0), closer to the former due to the incremental nature and experimental issues.

**Score: 4.5**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>