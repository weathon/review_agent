Now let me search for calibration papers to properly score this paper.Now I have enough information to write the final review.

---

## Summary

This paper proposes an identifiability theory for latent variables in nonlinear causal models using single-domain data. The core contributions are: (1) a structural reduction of any SCM (with no directed paths among observed variables) to an equivalent powerset bipartite graph (PBG-SCM) by clustering exogenous variables with identical observed descendant sets; (2) a novel "minimality" condition on shared latent variables that, together with invertibility and independence, guarantees identifiability up to invertible transformation; and (3) small-scale synthetic experiments ablating the necessity of each condition.

---

## Strengths

- **PBG Reduction (Section 4, Figure 1):** The formal reduction of any admissible SCM to an equivalent PBG-SCM — by clustering exogenous variables indexed by the binary representation of their observed descendant sets — is a conceptually clean and genuinely novel structural insight. It provides a canonical form for identifiability analysis that is applicable to general causal graphs.

- **Novel minimality condition and its analysis (Assumption 1.iii, Proposition 5.1, Corollary 5.1):** The minimality condition formalizes why a shared latent variable with "oversized" intrinsic dimension causes identifiability failure, and Proposition 5.1 gives a precise characterization of what goes wrong. Corollary 5.1 provides a practically useful equivalent criterion (match latent dimension to IDim(z)) and explains why minimality has been implicitly satisfied — but not recognized — in prior experimental practice.

- **Constructive proof structure (Theorem 2, Section 5.2, Figure 4b):** The extension from the basis model to general PBG-SCMs via iterative application of basis models is elegant, and Figure 4b concretely validates that the iterative procedure achieves high R² across all 7 latent variables in a size-3 PBG-SCM.

- **Ablation experiments confirm necessity of each condition (Table 1, Figure 3/4a):** Across three qualitatively distinct datasets (Concatenation, Split, Fusion), removing CLUB (violating independence) or inflating dimension (violating minimality) substantially degrades R², providing empirical support that each assumption is necessary.

---

## Weaknesses

### Fatal
None.

### Major

- **The minimality condition is not operationally checkable without ground-truth knowledge, limiting the theoretical contribution's practical impact.** Assumption 1.iii (and its generalization in Assumption 2.iii) requires existential quantification over all models satisfying invertibility and independence — a condition that cannot be verified from data alone. As Corollary 5.1 makes explicit, minimality is automatically satisfied when the practitioner sets dim(z̃) = IDim(z), i.e., when the true intrinsic dimension is known in advance. The paper correctly acknowledges this in Section 7: *"the succeeded algorithms in our experiments still need pre-known knowledge of the intrinsic dimension of latent variables."* The result therefore characterizes *why* the standard practice of using the correct dimension works, rather than enabling new identification in genuinely unknown-dimension settings. Papers such as p60Y6o85Cj address this gap directly (identifiability under unknown latent dimensions); the paper under review does not. The contribution is real but its scope is narrower than the framing suggests.

- **Experiments contain no comparison with any prior method and use exclusively toy-scale synthetic data.** The entire experimental section is an ablation of the proposed method across two hyperparameters (CLUB on/off, dimension inflated/correct). There is no evaluation against Lachapelle et al. (2024), Kong et al. (2024), Brady et al. (2023), or any other method on shared benchmarks. The basis-model experiments use a 2-observed-variable setup with at most d_v = 10, d_z = 5; the general model uses 3 observed variables with d_s_i = 2 each. Without competitive comparison on established benchmarks, the paper cannot substantiate its claim that minimality is "easier to satisfy" or more generally applicable than compositionality or subspace-span conditions.

### Minor

- **The differentiability condition (Assumption 1.i) is technically violated by the Split dataset.** Assumption 1.i requires a *differentiable* invertible function g with differentiable inverse. The Split dataset constructs z⁺ = max(z, 0) and z⁻ = min(z, 0), which are non-differentiable at z = 0. The paper states these datasets are "globally invertible but not locally invertible" (Section 6.1) but does not address the differentiability failure. While the measure-zero nature of the issue may render this practically harmless, the paper claims the theory is validated on datasets that technically violate one of its own assumptions.

- **The full-rank weight check does not guarantee global invertibility of a deep nonlinear MLP.** Section 6.1 states: *"We checked the rank of weight matrices in each linear layer to ensure they are of full rank, therefore any fi is guaranteed to be invertible."* This argument is invalid: layer-wise full-rank matrices in a nonlinear network do not imply global invertibility of the composed nonlinear function. This is a gap in the dataset construction argument, though the empirical results are consistent with the datasets behaving as intended.

- **The hierarchical minimality condition (Assumption 2.iii) uses an ordering that is not fully justified in the main text.** The condition requires s'_k ~ s_k for k satisfying k ≠ i and k & i = 0. This "lower variable" restriction implicitly relies on a topological ordering that is not explicitly motivated in the main text. The full justification is deferred to the appendix, making Section 5.2 difficult to evaluate on its own.

### Trivial
None worth mentioning.

---

## Nice-to-Haves

- An algorithm or heuristic for estimating IDim(z) without ground-truth knowledge — even a simple elbow-curve approach — would make the theory actionable and significantly strengthen the practical relevance of the minimality condition.
- Evaluation on a benchmark dataset used in the disentanglement literature (e.g., 3DShapes, Sprites, or the Causal3DIdent benchmark) would demonstrate that the theory's conditions are achievable outside pure toy settings.
- A formal comparison showing a concrete case where minimality holds but the subspace-span condition of Kong et al. (2024) fails (or vice versa) would substantiate the claimed advantage over prior work.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "The SCM reduction preserves marginal distributions but not full interventional structure."** The paper explicitly scopes its equivalence claim to marginal distributions over observed and concatenated exogenous variables; this is an intentional scope restriction for observational identifiability, not an oversight.
- **Harsh Critic: "AE(dz=5) without CLUB achieving R²≈0.81 is not theoretically analyzed."** This is a valid observation but is actually shown empirically in Figure 3 and Table 1 and is discussed in Section 6.2. Demanding a full theoretical analysis of this gap goes beyond the paper's stated scope.
- **Strength Finder: "Clear notation and formalism."** Removed as generic; not grounded in a specific section or figure citation.
- **Harsh Critic: "The claim that minimality is weaker than Kong et al.'s condition is unsubstantiated."** While this could be strengthened with a formal counterexample, it is a claim about comparative ease of satisfaction, not a proof claim. Moved to Nice-to-Haves rather than a main weakness.

---

## Novel Insights

The most genuinely novel observation in this paper — beyond standard identifiability claims — is the precise characterization of *why* the standard experimental practice of pre-specifying the true latent dimension implicitly satisfies a non-trivial theoretical condition. Corollary 5.1 shows that dim(z̃) = IDim(z) is a sufficient substitute for minimality, which explains why minimality has not previously appeared in the literature: prior experiments have enforced it unknowingly. This is a methodological insight with implications for how identifiability experiments should be designed (i.e., using unknown-dimension settings as a harder test). However, the paper identifies this gap without closing it — no dimension-estimation procedure is proposed — so the insight is diagnostic rather than constructive.

---

## Suggestions

1. Extend the experimental section with at least one comparison against a prior single-domain method (e.g., Lachapelle et al. 2024) on a shared dataset, with both methods evaluated at their best configurations.
2. Correct or qualify the MLP invertibility argument in Section 6.1; consider using normalizing flows or coupling layers where global invertibility is architecturally guaranteed.
3. Address the differentiability issue with the Split dataset explicitly — either by smoothing max/min with a soft approximation (SoftPlus) or by noting that a.e. differentiability suffices and citing the appropriate theorem.
4. Strengthen Section 5.2 by moving the topological ordering justification into the main text rather than fully deferring to the appendix.

---

## Score and Decision

**Calibration anchors reviewed:**

| Paper | Avg Score | Comparison to this paper |
|---|---|---|
| `2efNHgYRvM` (IDOL, temporal causal representation) | 8.0 | Much stronger: rigorous theory + real-world motion-forecasting benchmarks + addresses practical gap (instantaneous dependencies) |
| `3cuJwmPxXj` (Intervention extrapolation identifiability) | 8.0 | Much stronger: clean theory + real downstream task validation; comparable theory depth but better empirical scope |
| `lk2Qk5xjeu` (Unifying CRL with invariance) | 7.0 | Stronger: unifies existing literature under a principled framework with stronger generalizability |
| `p60Y6o85Cj` (Content-style under unknown latent dimension) | 6.6 | Stronger: directly addresses the key limitation of this paper (unknown latent dimension) with multi-domain data |
| `v1VvCWJAL8` (Domain counterfactuals for invertible latent models) | 5.75 | Comparable: theory + limited synthetic validation; accepted, but has slightly broader real-world framing |
| `nzgvkQM3EH` (Nonparametric dynamic causal model) | 5.75 | Comparable: theory paper with limited experiments; rejected |
| `kkQSwtx0p3` (Task structure identifiability) | 5.25 | Comparable: identifiability theory with narrow experiments; rejected; this paper is slightly cleaner theoretically |
| `0sO2euxhUQ` (Learning latent SCMs) | 4.0 | Weaker anchor: this paper clearly exceeds it in theoretical rigor and cleanliness |

**Assessment:** This paper sits in the 4.5–5.5 range. The PBG reduction and minimality condition are genuinely new and cleanly formulated, placing it above the 4.0-level papers. However, it falls short of the 5.75–6.6 accepted papers in two key ways: (1) the minimality condition requires knowing the true intrinsic dimension — a limitation that the comparable paper p60Y6o85Cj actually resolves — and (2) experiments are purely ablative and toy-scale, with no competitive comparison. The contribution is real but its practical scope is narrow and its experimental support is thin. The paper most closely resembles the borderline-reject zone of the 5.0–5.5 papers. I settle on **4.5**.

**Originality:** Moderate — the PBG reduction and minimality condition are new, but the setting and assumptions (invertibility + independence) are standard.  
**Importance:** Moderate — the research question (single-domain identifiability) is important, but the answer only applies when the true latent dimension is known.  
**Claims well-supported:** Partially — the theory appears formally correct, but experiments are insufficient to support claims of advantage over prior work.  
**Soundness of experiments:** Weak — toy-scale ablations only; MLP invertibility claim is technically invalid; one dataset violates differentiability.  
**Clarity:** Good — notation is consistent and the paper is generally readable.  
**Value to community:** Limited at current stage — the diagnostic insight about minimality is useful, but without an actionable dimension-selection procedure or competitive experiments, the contribution is primarily theoretical/descriptive.

**Decision: Reject.**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>