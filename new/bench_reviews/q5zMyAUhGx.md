Now I have a good sense of the calibration landscape. Let me now carefully synthesize the review.

Key observations from calibration:
- Papers with covering number-based generalization bounds for neural networks (UvpuGrd6ey = compositionality/DNN, scores 5-8, median ~6; JN7TcCm9LF = Koopman bound, scores 5-8, median ~6; q6zrZbth1F = sample complexity, scores 6-8, median ~7) tend to receive moderate-to-good scores, especially when they have clear theoretical contributions.
- The KAN expressiveness paper (ydlDRUuGm9, scores 5-8, median ~6.5) had similar concerns about limited experiments and theoretical scope.
- The compositionality paper (UvpuGrd6ey) had significant concern from Reviewer 3 about loose bounds and limited comparison with MLP bounds, yet was accepted as poster with median ~6.
- The Koopman paper (JN7TcCm9LF) had concerns about hidden parameter dependence in bounds and unclear practical implications, yet was accepted as poster with scores 5-8.

Now let me organize my assessment of this paper:

**Core contribution**: First rigorous generalization bounds for KANs, extending covering number techniques from MLP theory. Two settings: (1) basis function activations, (2) low-rank RKHS activations.

**Strengths**:
1. First theoretical analysis of KAN generalization - timely and important
2. Technically sound covering number analysis with proper extension to KAN architecture
3. No combinatorial dependence beyond logarithmic factors (following Bartlett et al. 2017 style)
4. Treatment of unbounded losses via truncation
5. Novel low-rank RKHS extension

**Weaknesses** (verified against paper):
1. **Empirical validation is weak** - The paper claims "practical relevance" and "tight correlation" but:
   - Only tracks normalized complexity vs excess loss for single architectures
   - Normalization forces endpoint matching
   - No baselines (simpler proxies, MLP comparisons)
   - No variance, no quantitative correlation measures
   
2. **Experimental complexity measure doesn't match theory** - They drop D (input norm) and normalize to match final excess loss. This breaks the quantitative link.

3. **Gap between theory and trained models** - Assumptions (bounded $\|B_l\|_1$, bounded Lipschitz constants) constrain the function class but SGD doesn't enforce these. No verification that trained models satisfy them.

4. **"No combinatorial dependence" claim is nuanced** - $B_l$, $\rho_l$ can implicitly grow with width/depth (as noted in Remark 4). The claim is technically correct but misleading without discussion of implicit scaling.

5. **Unbounded loss claim is overstated** - Assumption 4 still requires a uniform envelope over the model class, which effectively bounds function values.

6. **No lower bounds** - Authors acknowledge this; makes tightness hard to assess.

Let me now look at whether the harsh critic's points are valid:

Point 1 (overstated empirical claims): VALID - The normalization and lack of baselines is a real issue. But I should not overstate - showing correlation is a form of evidence, just not conclusive evidence of "practical relevance."

Point 2 (complexity measure doesn't match theory): PARTIALLY VALID - Dropping D is problematic since it's in the theory. But it's proportional to $\tilde{\alpha}$ (they state this), and they note C=0 by construction. The normalization is more concerning.

Point 3 (unbounded loss claim): PARTIALLY VALID - Assumption 4 is nontrivial but genuinely weaker than bounded loss. The claim is somewhat overstated but has real content.

Now I need to determine how severe these are. The theoretical contribution is genuine and substantial. The experimental section is weak but the paper could stand on its theory. The overclaiming about practical relevance is a real issue but not fatal to the theoretical contribution.

Score calibration: This paper's theoretical contribution is comparable to JN7TcCm9LF (Koopman bound, median ~6.5) and slightly below q6zrZbth1F (sample complexity, median ~7). The experimental section is weaker than both. The UvpuGrd6ey (compositionality) paper was accepted with similar theoretical contributions but also somewhat limited experiments (median ~6).

I'd place this paper solidly in the 5.5-6.5 range. The theoretical contribution is real and significant (first generalization bounds for KANs), but the empirical overclaiming and the gap between theory and practice pull it down from a higher score.</think>## Summary

This paper derives the first generalization bounds for Kolmogorov–Arnold Networks (KANs) under two settings: (1) when activation functions are linear combinations of basis functions, yielding bounds scaling with the $\ell_1$ norm of coefficient matrices and Lipschitz constants with only logarithmic dependence on combinatorial parameters; and (2) when activation functions belong to low-rank RKHS spaces, yielding bounds scaling polynomially with ranks and Lipschitz constants. The bounds are empirically investigated by tracking a complexity proxy against excess loss during SGD training.

## Strengths

- **First rigorous generalization analysis for KANs.** KANs have seen rapid adoption in scientific applications, yet no prior work has provided comprehensive generalization bounds. This paper fills an important gap, extending covering-number techniques from MLP theory (Bartlett et al., 2017; Anthony et al., 1999) to the more flexible KAN architecture where activation functions are learned rather than fixed.

- **Well-structured theoretical results.** Theorem 1's covering number bound has no explicit dependence on combinatorial parameters (width, basis count) beyond logarithmic factors—a property analogous to Bartlett et al. (2017) for MLPs but extended to KANs with learnable activations. The progression from Propositions 1–2 to Theorems 1–5 is logically clean and well-organized.

- **Treatment of unbounded losses.** Theorem 3 uses a truncation argument (Assumption 4) to handle regression-type losses (squared, pinball, Huber) without requiring boundedness, which is a genuine improvement over prior margin-based bounds restricted to ramp loss.

- **Novel low-rank RKHS extension.** Section 2.3 extends kernel-based generalization theory to compositional KANs with low-rank activation constraints. Remark 6's connection to LoRA-style fine-tuning is insightful and timely. This extension appears to be new even for MLPs.

- **Adaptivity to operator norms.** The analysis accommodates different choices of operator norms at each layer, following the spirit of Bartlett et al. (2017)—this allows users to choose norms (e.g., spectral vs. Frobenius) best suited to the problem.

## Weaknesses

### Fatal
None.

### Major

- **The empirical validation does not support the claimed "practical relevance."** This is one of three advertised contributions (Abstract: "numerical results demonstrate the practical relevance"; §1.1: "complexity measure's practical relevance"; §4: "strong correlation between the excess loss and the complexity measure, demonstrating the practical relevance"). However: (i) The complexity measure tracked in experiments omits the input norm $D$ (from $\tilde{\alpha}$) and is then *normalized* so its maximum equals the final excess loss—this breaks the quantitative link to the theory. (ii) No alternative complexity proxies (parameter count, weight norms, MLP-derived bounds) are compared, so it is impossible to assess whether $\tilde{\alpha}$ captures anything KAN-specific. (iii) Only one architecture per dataset is tested, with no variance estimates. (iv) The observed behavior—that a norm-like quantity increases during training as overfitting grows—is expected for virtually any complexity proxy. The "practical relevance" and "tight correlation" claims are therefore not supported by the presented evidence. The empirical section is more of an illustration than a validation.

- **Gap between the constrained function class $\mathcal{M}$ and SGD-trained KANs.** The bounds hold uniformly over $\mathcal{M}$, which requires bounded $\ell_1$ norms ($\|\mathbf{B}_l\|_1 \leq B_l$), bounded Lipschitz constants ($\rho_l$), and bounded basis-function Lipschitz constants ($c_{ij}^{(l)}$). SGD does not enforce any of these constraints. No experiment or argument verifies that trained KANs remain within these constrained classes, making it unclear whether the theory applies to standard training practice. This is a pervasive issue in generalization theory for neural networks, but it is particularly important here because the function class is defined through properties of learned activation functions, not just weight matrices.

### Minor

- **The "no combinatorial dependence" claim deserves nuance.** Theorems 1–3 correctly show no explicit width/basis-count dependence beyond $\log(2\tilde{d}\tilde{p})$. However, as Remark 4 acknowledges, $\|\mathbf{B}_l\|_1 \leq B_{\max} \|\mathbf{B}_l\|_0$, and the Lipschitz bound in Remark 5 includes $\sqrt{b_l}$, so the key complexity parameters ($B_l$, $\rho_l$) can implicitly grow with architecture size. The paper notes this but does not discuss typical scaling, which would clarify whether the bounds are non-vacuous for realistic architectures.

- **The unbounded-loss generality is stated more broadly than warranted.** The Abstract says the bound "does not require the boundedness assumption on the loss function." Assumption 4, which replaces boundedness, still requires a uniform envelope $G$ bounding $\sup_{f \in \mathcal{M}} |\mathcal{L}(f(\cdot), \cdot)|$, which effectively constrains function values over the entire model class—a condition that, for typical unbounded losses (e.g., squared loss), implicitly requires norm bounds on the parameterized functions. This is a real improvement over requiring bounded loss, but the qualitative leap in generality is somewhat overstated.

- **The low-rank RKHS section (§2.3) would benefit from concrete examples.** The condition $\tilde{d} > \nu$ in Theorem 4 and the rank constraint $\dim(\text{span}(\psi_1, \dots, \psi_m)) \leq r_l$ are quite abstract. No example is worked out showing what typical KAN architectures satisfy these conditions or what realistic values of $\nu$ and $r_l$ look like.

- **No lower bounds.** As the authors acknowledge in §4, the absence of lower bounds makes it impossible to determine whether the $\tilde{\alpha}^{3/2}$ scaling in the generalization bound or the polynomial rank dependence in Theorem 4 are tight or artifacts of the proof technique.

### Trivial
None.

## Nice-to-Haves

- Experiments varying architecture (width, depth, basis size) and showing how $\tilde{\alpha}$ and excess loss co-vary, with comparison to simpler baselines (parameter count, weight norms).
- Computing the actual numerical value of the generalization bound from Theorems 2/3 for at least one trained model and comparing it to the observed generalization gap, even if the bound is loose by orders of magnitude.
- Explicit regularization experiments using $\tilde{\alpha}$ as a penalty term, as suggested in §1.1 and §4.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No comparison with MLP baselines."** (Spark reviewer) While a comparison with MLP generalization would be informative, this paper's stated scope is establishing *KAN-specific* bounds. The theoretical contribution stands independently; MLP comparison is a nice-to-have, not a core flaw.

- **"Assumption 1 ($\|\mathbf{X}\|_2 \leq D$) should be probabilistic in the theorem statements."** (Harsh critic) Remark 1 explicitly notes this relaxation is possible, and the current deterministic statement is standard for covering-number bounds in the literature (cf. Bartlett et al., 2017). This is a presentation preference, not a methodological error.

- **"Incomplete justification for Assumption 2 generality"** (Neutral reviewer) — the paper already acknowledges in Remark 2 that the general form encompasses the additive KAN structure as a special case. The generality is an analytic convenience, not a flaw.

- **"Report experimental hyperparameters"** (Neutral reviewer) — reproducibility details are a minor concern not affecting the core contribution of a theoretical paper with illustrative experiments.

## Novel Insights

The low-rank RKHS analysis in §2.3, combined with Remark 6's connection to LoRA-style fine-tuning, points to an interesting and potentially impactful direction: if KAN activations after pre-training have low effective rank, the generalization bounds in Theorem 4 become much tighter, providing a theoretical justification for low-rank adaptation strategies in KANs analogous to those developed for large language models. This connection between generalization theory for KANs and practical fine-tuning techniques is underexplored and could drive future work.

## Suggestions

- Tone down the "practical relevance" claims throughout the paper. The experiments illustrate that $\tilde{\alpha}$ tracks excess loss, but do not validate the bounds or demonstrate practical utility for regularization or architecture design. Changing "demonstrate" to "suggest" or "illustrate" would be more accurate.
- Add an explicit discussion of how $B_l$ and $\rho_l$ typically scale with architecture size in practice, to clarify when the "no combinatorial dependence" property yields meaningful advantages.
- Verify (even approximately) whether trained KANs satisfy Assumption 2's constraints, or discuss what implicit regularization SGD provides toward these constraints.

## Score and Decision Calibration

Calibration papers examined:
- **UvpuGrd6ey** (Compositionality/DNN generalization): Covering-number-based bounds, similar structure to this paper. Median score ~6. Accepted as poster. Weaknesses included limited experimental validation and unclear comparison with existing bounds.
- **JN7TcCm9LF** (Koopman-based bound): Novel generalization bound for full-rank weights. Scores 5-8, median ~6.5. Accepted as poster. Concerns about hidden parameter dependence and unclear practical implications.
- **q6zrZbth1F** (Sample complexity DNN): Lower bounds for ReLU networks. Scores 6-8, median ~7. Accepted as poster. Stronger theoretical contribution (both upper and lower bounds) and simpler experiments.
- **ydlDRUuGm9** (KAN expressiveness): Directly related KAN theory paper. Scores 5-8, median ~6.5. Accepted as poster with concerns about limited theoretical scope and experiments.

This paper provides the first systematic generalization analysis for KANs, which is genuinely novel and timely. The theoretical development is sound and follows established proof templates (Maurey sparsification, covering number composition). However, the empirical claims are notably overstated relative to the evidence, and the gap between the constrained function class and SGD-trained models is not discussed. Compared to the Koopman paper (median ~6.5) and compositionality paper (median ~6), this paper has comparable theoretical novelty but weaker empirical validation and more overclaiming. It is somewhat stronger than the KAN expressiveness paper (median ~6.5) in theoretical depth. I place it slightly below the median of these related papers.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>