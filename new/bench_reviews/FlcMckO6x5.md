Now I have a thorough understanding of the paper and calibration papers. Let me write the final review.

## Summary

This paper develops theoretical foundations for Separable Neural Networks (SepNNs) in three directions: (1) proving universal approximation theorems for CP, TT, and Tucker SepNNs via Stone-Weierstrass and universal approximation theory; (2) deriving NTK regimes showing convergence to deterministic kernels under infinite width/rank and to random kernels under infinite width/fixed rank; and (3) proposing an efficient Separable Preconditioned Gradient Descent (SepPGD) method that exploits the separable structure to achieve O(nD) complexity for n^D grid-structured training samples, compared to O(n^D) for standard NTK-based preconditioners.

## Strengths

- **Unified approximation theory for SepNNs.** Theorem 1 provides a clean, unified proof covering CP, TT, and Tucker SepNNs using the Stone-Weierstrass theorem, generalizing prior results that only covered the bivariate CP case (Cho et al., 2023). The verification of the three Stone-Weierstrass conditions for the separable function class is well-executed.

- **Nontrivial NTK structural decomposition.** Lemma 1 showing the SepNN NTK decomposes as a weighted sum of factor MLP NTKs is a useful and non-trivial structural result. The distinction between deterministic (infinite width & rank) and random (infinite width & fixed rank) NTK regimes (Theorem 2, Corollary 1) provides genuine insight—in particular, that fixed-rank SepNNs do not enjoy the deterministic NTK regime, which has practical implications for lazy-training arguments.

- **Significant efficiency gain from SepPGD.** The algorithm achieves O(nD) per-iteration complexity versus O(n^D) for standard NTK-PGD (Table 1), which is a substantial and practically meaningful improvement. Lemma 2 establishing the equivalence between SepPGD and NTK-based PGD for the bivariate case (D=2) via Kronecker structure is a clean technical result.

- **Broad experimental evaluation.** Testing across KRR, image INRs, surface INRs, and PINNs demonstrates that SepPGD is not tied to a single toy problem and provides consistent improvements. Wall-clock-time comparisons (rather than iteration counts) appropriately reflect the claimed efficiency advantages.

- **The NTK on grid inputs admits Kronecker product structure.** This structural observation (Appendix A.3) enables efficient NTK computation for SepNNs beyond the algorithmic contribution, which is a useful byproduct.

## Weaknesses

### Major:

- **The "provably adjusts" spectral bias claim is not rigorously established.** The paper's central algorithmic claim is that SepPGD "provably adjusts" the NTK spectrum and "provably alleviates spectral bias" (abstract, contributions, Section 4). However, the spectral argument after Lemma 2 is informal: it states "Suppose that K̃ is close to the true NTK matrix K..." and argues heuristically that eigenvalues of Kronecker products factor and therefore S̃ improves the spectrum of K̃, and hence by assumption K·S̃ improves the spectrum of K. No theorem or quantitative bound establishes (i) that K̃ ≈ K in any spectral norm, (ii) that K·S̃ has a better condition number than K, or (iii) that the transfer from K̃·S̃ to K·S̃ preserves the spectral improvement. The paper uses the word "provably" in the abstract and conclusions (lines 8, 63, 349, 367) for claims that are only heuristically argued. This is a meaningful gap between the claims and what is proved—the paper demonstrates an efficient algorithm equivalent to NTK-PGD for a particular Kronecker-structured approximation, but does not provably establish spectral improvement of the true SepNN NTK.

- **Lemma 2 proved only for D=2 grid inputs; D>2 and non-grid cases left unproven.** The paper states "It is believed that the result in Lemma 2 (and the analysis following) can be readily extended to multivariate cases D > 2" (line 349). Since practical applications (e.g., PINN experiments in Fig. 4 use D=3), this gap between what is proved and what is claimed is significant. The non-grid extension is mentioned briefly with an einsum construction but without formal analysis. This weakens the theoretical backing for the algorithm's operation in the settings where it is actually empirically tested.

- **Gap between NTK theory and practical regime.** The deterministic NTK regime (Theorem 2) requires both infinite width and infinite rank, while the practical regime uses small fixed rank. The paper acknowledges this (Remark 3, Corollary 1) and notes that training dynamics "can not be characterized uniformly using a fixed NTK matrix" in the fixed-rank case. However, the spectral bias analysis and SepPGD both rely on the fixed-NTK regime (Eq. 5). While the paper empirically demonstrates SepPGD's effectiveness with small rank (Appendix Table 3), the theoretical narrative has an acknowledged gap between the regime where the results hold and the regime where the method is actually used.

### Minor:

- **Universal approximation theorem provides no quantitative bounds.** Theorem 1 is existential—it guarantees approximation to arbitrary precision but provides no bound on how rank R or width W must scale with error ε and dimension D. This limits the theorem's practical guidance for architecture selection, though this is a standard limitation of universal approximation results in general.

- **NTK analysis limited to CP formulation.** Lemma 1, Theorem 2, and Corollary 1 are developed only for CP SepNNs. The approximation theorem covers TT and Tucker, but the NTK theory does not extend to these formulations. Extending the NTK theory to TT/Tucker would require non-trivial derivation and is identified in Remark 1 as "straightforward" for multi-layer MLPs but left unstated for other decomposition structures.

- **Limited baselines for spectral bias mitigation.** Experiments compare against MSK (Geifman et al., 2024; Shi et al., 2025) but not against other common spectral bias approaches such as Fourier feature encodings (Tancik et al.) or alternative activation functions (SIREN). While MSK is the most direct NTK-based comparison, additional baselines would better position SepPGD relative to the broader spectral bias literature.

- **No ablation on rank R.** The theory distinguishes infinite-rank (deterministic NTK) from fixed-rank (random NTK) regimes, yet no experiment systematically varies R to examine how SepPGD's effectiveness depends on rank—directly connecting theory to practice.

### Trivial:
- Notation in Definition 1/Eq. (7)-(8) is dense; a small illustrative example (e.g., D=2, R=2, n=3) would aid readability.

## Nice-to-Haves

- **Formal convergence guarantee for SepPGD.** The paper notes this is "left for future research" (line 349). A convergence rate bound would substantially strengthen the theoretical contribution, though the empirical evidence is already compelling.
- **Visualization of NTK eigenvalue spectra before/after SepPGD.** The paper claims spectral improvement but never directly visualizes eigenvalue decay of K versus K·S̃, which would be the most direct empirical evidence.
- **Approximation rates for Theorem 1.** Even loose bounds relating rank R to approximation error ε for specific function classes (e.g., smooth functions) would elevate the result from existential to quantitative.
- **Generalization analysis.** The entire spectral discussion focuses on training convergence; whether SepPGD helps or harms test-time generalization is not analyzed.

## Removed Points

These points are flagged to be removed—treat them with caution:

- **"NTK theory for TT/Tucker SepNNs missing."** The paper explicitly scopes its NTK analysis to the CP formulation and does not claim TT/Tucker NTK results. Criticizing the absence of results the paper never claimed is scope creep. That said, the gap between the approximation theory (which covers all three) and the NTK theory (CP only) is worth noting as a minor point.

- **"Factually wrong claims about NTK convergence assumptions."** The harsh reviewer suggested Theorem 2's assumptions differ from typical NTK settings (e.g., sine activations, non-Gaussian init). But the paper uses standard NTK assumptions (differentiable σ, Gaussian init) in line with Jacot et al. and Arora et al., and explicitly notes multi-layer extensions in Remark 1.

- **"Reproducibility concerns about hyperparameters."** The choice of k (number of eigenvalues to flatten) and preconditioner update frequency are implementation details that, while relevant to practical performance, are not unusual omissions. The paper does describe these choices.

- **"Insufficient novelty over Geifman et al. (2024) since the preconditioning idea is inherited."** The paper clearly positions its contribution as exploiting the separable structure to achieve O(nD) efficiency—the core algorithmic novelty is the factor-wise decomposition of the preconditioner, not the eigenvalue flattening idea itself. This is appropriately acknowledged in the prior arts discussion.

- **"Demanding comparison with instant-NGP, hash-grid encodings."** These are architectural alternatives to SepNNs, not alternative spectral bias mitigation methods. The paper's contribution is an optimization method for SepNNs, not a replacement for instant-NGP.

## Novel Insights

The decomposition of the SepNN NTK into a weighted sum of factor NTKs (Lemma 1) is an insightful structural result that reveals how separability manifests in the kernel space: the factor interactions appear as inner products of factor-function products, not as additive terms. This insight directly motivates the factor-wise preconditioning strategy. The observation that fixed-rank SepNNs yield random NTKs (Corollary 1) is practically important—it explains why lazy-training guarantees may not apply in the regime where SepNNs are most useful, and cautions against over-reliance on deterministic NTK analysis for these architectures.

## Suggestions

1. **Dial back "provably" language.** Replace "provably adjusts" and "provably alleviates" with "motivates adjusting" or "is designed to adjust" throughout the paper. Reserve "provably" for claims that are formally proved (Lemma 2, Theorem 1, Theorem 2).
2. **Provide at least a sketch or formal statement for the D>2 extension.** Even without a full proof, a precise conjecture or theorem statement for D>2 would significantly strengthen the paper, given that the PINN experiments rely on it.
3. **Add NTK eigenvalue spectrum visualizations** (before/after SepPGD) as direct empirical evidence for spectral improvement, which would partially compensate for the lack of formal spectral bounds.
4. **Include a rank ablation study** that varies R to examine how SepPGD's effectiveness changes with rank, directly connecting the theoretical NTK regimes to empirical observations.

## Score and Decision

**Calibration comparisons:**
- TNYLCF7vZA (Inductive gradient adjustment for INR spectral bias): similar NTK-based spectral bias mitigation topic, but overclaimed relative to prior art and had reproducibility concerns. Mean score ~5, rejected.
- WWlxFtR5sV (Operator preconditioning for PINNs): theoretical preconditioning paper with limited experiments. Mean score ~6.3, accepted poster.
- ydlDRUuGm9 (KAN expressiveness and spectral bias): theoretical analysis + spectral bias study, existential approximation results, moderate experiments. Mean score ~6.25, accepted poster.
- aVlDNbvmCK (Architectural insights for PINNs via NTK): mixed bag of results, weak coherence. Mean score ~3.5, rejected.

This paper has stronger theoretical contributions than the rejected aVlDNbvmCK and pv2U1BeC5Z papers, and more comprehensive experiments. Its NTK decomposition and efficient algorithm are genuine contributions comparable to the accepted ydlDRUuGm9 and WWlxFtR5sV. However, the overclaiming of "provably alleviates spectral bias" when only heuristic arguments are provided, combined with the D=2 limitation of the key lemma and the NTK regime/practice gap, places it below the cleanly positioned accepted papers. The paper would be significantly stronger if it accurately represented the gap between what is proved and what is conjectured.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>