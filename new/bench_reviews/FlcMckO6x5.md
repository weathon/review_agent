Now I have all the information needed. Let me compose the final review.

## Summary

This paper establishes the first comprehensive theoretical foundation for separable neural networks (SepNNs) across three contributions: (1) a universal approximation theorem (Theorem 1) proving CP, TT, and Tucker SepNNs can approximate any continuous multivariate function, (2) NTK regime characterizations showing deterministic convergence under infinite width-and-rank and random convergence under fixed rank (Theorem 2, Corollary 1), and (3) an efficient separable preconditioned gradient descent (SepPGD) method with O(nD) complexity that claims to provably alleviate spectral bias by adjusting the NTK spectrum.

## Strengths

- **Universal approximation theorem covering multiple SepNN architectures (Theorem 1):** Establishes that CP, TT, and Tucker SepNNs can approximate any continuous multivariate function on compact sets. This strictly generalizes the only prior result by Cho et al. (2023), which was limited to D=2 CP SepNNs. The unified Stone-Weierstrass proof strategy (Section 2, proof sketch around line 90–98) is elegant and avoids the constructive orthogonal-basis approach of prior work.

- **NTK decomposition revealing structural properties of SepNNs (Lemma 1):** The result that SepNN's NTK decomposes as a weighted sum of factor NTKs (Eq. 4) is insightful and structurally distinct from standard MLP NTKs. This directly motivates the design of SepPGD.

- **O(nD) complexity of SepPGD is a genuine practical improvement (Table 1, Remark 4):** Reducing preconditioner application from O(n^D) to O(nD) and construction from O(n^{3D}+n^{2D}P) to O(D(n³+n²P)) makes preconditioning practical for grid-structured problems where n^D is prohibitively large. The complexity analysis is clear and correct.

- **Lemma 2 proves SepPGD ≡ classical PGD for D=2:** The equivalence between SepPGD and NTK-based PGD with Kronecker preconditioner S̃ = (S₁⊗Iₙ + Iₙ⊗S₂) is rigorously established for D=2 (Section 4, Lemma 2). The key identity (C^⊤⊗A)vec(B) = vec(ABC) enables the efficient decomposition.

- **Broad empirical validation across diverse tasks:** Experiments span KRR (Fig. 2a), image representation and inpainting (Fig. 2b, Fig. 3 left), surface representation (Fig. 3 right), and three PDE types with PINNs (Fig. 4), showing consistent practical improvements.

- **Convergence curves plotted against wall-clock time (Fig. 2):** This is the appropriate metric for evaluating methods claiming computational efficiency, fairly accounting for per-iteration cost differences.

## Weaknesses

### Fatal
None.

### Major

- **The "provably" claim for SepPGD's spectral bias alleviation is unsupported.** The abstract states SepPGD "provably adjusts the NTK spectrum to alleviate spectral bias," but the theoretical support falls far short:
  - Lemma 2 (the SepPGD ≡ PGD equivalence) is proven only for D=2. The extension to D>2 is stated as "it is believed that the result... can be readily extended" (line 349), which is not a proof. Most experiments (image INR with D=2 being a special case; PINNs with D≥3) use D>2.
  - Even for D=2, the spectral improvement argument that KS̃ has better spectrum than K is heuristic, not proven. The paper writes "This can possibly be verified" and "We can ultimately show that KS̃ has better spectrum than K" (lines 346–349), but no proof follows. Crucially, S̃ = (S₁⊗Iₙ + Iₙ⊗S₂) is a *sum* of Kronecker products, and the eigenvalue-product property applies to individual Kronecker products, not sums — so the hand-waving about eigenvalue products is insufficient for S̃. The gap between the word "provably" and what is actually proven is too large for the abstract's central algorithmic claim.
  - Convergence guarantees are explicitly deferred: "This is left for future research" (line 349).

- **Theory-practice disconnect in the fixed-rank NTK regime.** The paper acknowledges (Remark 3, line 211) that under fixed rank — the practical regime since small R is used for efficiency — the NTK converges to a random kernel and "the training dynamic can not be characterized uniformly using a fixed NTK matrix as in (5)." This means the convergence and spectral bias analysis in Eq. (5), which assumes a fixed deterministic NTK, does not apply in the practical setting. The paper offers no alternative characterization, stating only that "the random NTK can at least characterize the training dynamic within a small range of training time" as future work, and that "even with small rank, the proposed SepPGD method is effective" — supported only empirically (Appendix Table 3). This leaves the core theory disconnected from the regime in which experiments operate.

### Minor

- **Missing direct SepNN+MSK vs. SepNN+SepPGD comparison for INR/PINN tasks.** The KRR experiment (Fig. 2a) compares MSK on both MLP and SepNN architectures, but the main application experiments (image/surface INR, PINNs) compare SepPGD on SepNN primarily against standard GD on SepNN and MSK on MLP. A direct matched-architecture comparison (MSK applied to SepNN vs. SepPGD on SepNN) for these tasks would more cleanly isolate the contribution of SepPGD's preconditioning from SepNN's architectural advantage. Since the paper already demonstrates this comparison for KRR, extending it to other tasks should be straightforward.

- **No error bars or variance reported in experiments.** The convergence curves and quantitative results (e.g., IoU in Fig. 3) report no uncertainty estimates, making it difficult to assess the statistical significance of improvements, especially when margins appear modest in some settings.

### Trivial
None.

## Nice-to-Haves

- **Empirical eigenvalue spectra visualization of K vs. KS̃:** The paper claims S̃ improves the condition number of K. Directly computing and plotting the eigenvalue spectra of K and KS̃ at various training stages would empirically validate (or invalidate) the spectral bias alleviation claim, providing evidence that the heuristic theoretical argument gets right.

- **Ablation on rank R:** Systematically varying R and showing how SepPGD's effectiveness scales would directly address the theory-practice gap around fixed vs. infinite rank, and guide practitioners in choosing R.

- **Quantitative approximation bounds:** Theorem 1 is existential. Quantitative bounds on how rank R and width W trade off with approximation error ε for representative function classes would guide practical architecture choices.

- **Extension of Lemma 2 to D>2:** Without this, the theoretical backing for SepPGD in the multivariate setting (where most experiments operate) remains incomplete.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that the approximation error bound may need large R and widths (Section 2 notes):** The harsh critic notes the Cauchy-Schwarz bound across factor MLPs may require large R and widths. However, Theorem 1 is an existential result (as is standard for universal approximation theorems), and the paper does not make quantitative claims about rank-error tradeoffs. This is a nice-to-have request, not a weakness of what the paper claims.

- **Harsh critic's claim of unfair comparison with MSK (Critical Issue 3):** The critic suggests the comparison might be unfair because MSK may not exploit SepNN structure. However, this asymmetry actually favors the baselines (MSK on standard architectures that it was designed for), not the authors' method. Per the hard rules, this is not a valid criticism. The KRR experiment does include the direct comparison on both architectures.

- **Strength finder's claim about "provable equivalence to classical PGD" as a core strength:** While Lemma 2 is correctly proven for D=2, elevating this as a "provable" strength supporting the broader "provably alleviates spectral bias" claim conflates equivalence (which is proven for D=2) with spectral improvement (which is not). The equivalence is a legitimate strength but only for D=2; it does not substantiate the broader "provably" claim in the abstract.

- **Harsh critic's demand for convergence to same solution verification:** The paper already shows SepPGD converges faster; whether it converges to the same or better solution is visible in the convergence curves reaching lower error. This is a minor presentation suggestion, not a substantive weakness.

- **Harsh critic's demand for hyperparameter sensitivity analysis of k:** This is a standard minor ablation request, not a core flaw. Moved to trivial/nice-to-have.

## Novel Insights

The paper reveals a fundamental tension in the theory of SepNNs that prior work did not confront: the NTK regimes split into two qualitatively different asymptotic regimes depending on rank, and the practical regime (small R) falls outside the deterministic NTK theory that motivates SepPGD. This theory-practice gap is not unique to SepNNs — it mirrors a broader challenge in NTK theory — but the paper's explicit derivation of the random NTK under fixed rank (Corollary 1) makes the gap unusually visible, which is scientifically honest even though the paper does not resolve it. The SepPGD design insight — decomposing the preconditioner along factor dimensions to match SepNN's separable structure, yielding O(nD) instead of O(n^D) — is conceptually clean and the Lemma 2 equivalence for D=2 confirms the decomposition is mathematically sound, even though the spectral improvement argument remains unproven.

## Suggestions

- Replace "provably adjusts the NTK spectrum to alleviate spectral bias" in the abstract with a more accurate formulation such as "is equivalent to NTK-based preconditioned gradient descent (proven for D=2), which is expected to alleviate spectral bias by adjusting the NTK spectrum." The current wording claims more than is proven.

- Add a direct SepNN+MSK vs. SepNN+SepPGD comparison for at least one INR or PINN task, as the paper already demonstrates this for KRR. This would isolate the preconditioning contribution from the architectural advantage.

- Plot eigenvalue spectra of K vs. KS̃ at several training steps to empirically verify the spectral improvement claim, which would substantially strengthen the paper even without completing the theoretical proof.

## Evaluation on Key Axes

- **Originality:** Moderate-to-high. The UAT (Theorem 1) properly generalizes prior work from D=2 to general D and across CP/TT/Tucker. The NTK decomposition (Lemma 1) and SepPGD design are novel. However, the SepPGD idea (per-factor preconditioning) is a natural consequence of the SepNN structure, reducing some novelty.

- **Importance of research question:** High. SepNNs are increasingly used in INRs and PINNs, and understanding their theoretical properties and optimization behavior addresses a real gap.

- **Claims well supported:** Mixed. The UAT and NTK regime results are well supported. The central "provably alleviates spectral bias" claim is not adequately supported by the proofs provided — the spectral improvement argument is heuristic even for D=2, and the D>2 extension and convergence guarantees are deferred.

- **Soundness of experiments:** Adequate. Experiments are broad and show consistent improvements. Missing error bars and the missing matched-architecture comparison for INR/PINN tasks are gaps but not fatal.

- **Clarity of writing:** Good. The narrative arc (approximation → NTK → preconditioning) is coherent. Some equations suffer from parser artifacts but the logical flow is clear.

- **Value to research community:** Moderate-to-high. The UAT and NTK decomposition provide a useful theoretical foundation for the SepNN community. SepPGD provides a practical optimization tool. The overclaims need correction to avoid misleading future work.

## Calibration

Anchors used:
- **Tucker-FNO (avg 5.0, Accept Poster):** Also provides tensor decomposition + UAT for neural operators. This paper has a more complete NTK analysis and an efficient preconditioning algorithm, but Tucker-FNO's theory is more complete (no overclaims). This paper is roughly comparable.
- **NdLinear (avg 4.0, Reject):** Also uses Tucker decomposition for efficiency with VC-dimension theory. Had ambiguous theorems and limited insight. This paper's theoretical contributions are stronger and more rigorous.
- **Frozen-PINN (avg 7.0, Accept Oral):** Strong practical PINN method with thorough experiments. This paper's empirical contributions are less dramatic but its theoretical scope is broader.
- **Empirical NTK rank (avg 3.5, Reject):** Had theory-experiment disconnect and vacuous bounds. This paper's theory is stronger and the experiments are more convincing, though it shares the issue of theory not fully covering the practical regime.
- **Convergence guarantees for neural PDE solvers (avg 3.0, Reject):** Had overclaimed provability with restricted assumptions. This paper's overclaim is similar in nature but less severe — the paper proves equivalence for D=2 and the experiments are more extensive.

This paper sits above the 3.0–3.5 range (stronger theory, broader experiments) and below the 7.0 range (incomplete proof for central claim). It is comparable to Tucker-FNO (5.0) with stronger NTK analysis but a more significant gap between claims and proofs. The "provably" overclaim is a real issue that prevents a higher score.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>