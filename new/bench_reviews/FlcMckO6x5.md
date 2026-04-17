Now I have enough context from calibration papers. Let me synthesize the final review.

## Summary

This paper provides a theoretical study of Separable Neural Networks (SepNNs)—architectures that factorize multivariate functions into linear combinations of univariate factor networks—addressing three questions: (1) universal approximation capacity, (2) NTK regime characterization (deterministic vs. random under different asymptotic regimes), and (3) an efficient Separable Preconditioned Gradient Descent (SepPGD) method that exploits separable structure to achieve O(nD) complexity for spectral bias alleviation on grid data, compared to O(n^D) for standard preconditioning.

## Strengths

1. **Comprehensive theoretical treatment of a practically important architecture class**: The paper addresses a genuine gap—SepNNs are widely used in INRs and PINNs but lacked unified approximation theory and NTK characterization. The extension from the bivariate result of Cho et al. (2023) to general multivariate (CP, TT, Tucker) cases is meaningful, and the proof via Stone-Weierstrass is cleaner than prior orthogonal-basis constructions.

2. **Novel NTK factorization insight for SepNNs**: Lemma 1 showing the SepNN's NTK decomposes as a weighted sum of factor NTKs is both theoretically clean and practically valuable—it directly enables the Kronecker-product structure over grids, which is the foundation of SepPGD's efficiency. The distinction between deterministic (infinite rank+width) and random (infinite width, fixed rank) regimes is conceptually insightful and aligns with practice where rank is kept small.

3. **Substantial computational gain of SepPGD**: The reduction from O(n^D) to O(nD) for preconditioner application on grid-structured data (Table 1) is a clear and significant improvement. Lemma 2's Kronecker-product equivalence connects SepPGD to the established NTK-PGD framework of Geifman et al. (2024), providing a principled theoretical grounding.

4. **Broad experimental validation**: Experiments span KRR, image/surface representation via INRs, and PINNs across multiple PDEs, consistently showing SepPGD accelerates convergence. Measuring convergence vs. wall-clock time (rather than iteration count) is appropriate given the efficiency claims.

## Weaknesses

### Major:

- **The "provably" spectral bias alleviation claim is overstated for D>2**: The abstract and introduction describe SepPGD as "provably adjusting the NTK spectrum." However, the formal equivalence between SepPGD and NTK-PGD (Lemma 2) is proven only for D=2. The D>2 extension is stated as "believed to be readily extendable" (Section 4) and "left for future research." Since SepNNs' efficiency advantage is greatest for high-dimensional problems (D≥3), and all PINN experiments use D=3, this gap directly undermines the paper's primary practical contribution. The informal spectral argument after Lemma 2 uses phrases like "can possibly be verified" and would have "better spectrum," without formal definition of "better spectrum" or quantitative bounds on how conditioning improves. This is not a "provable" result for the general case the paper actually deploys.

- **NTK theoretical regime mismatches practical SepNN deployment**: Theorem 2 (deterministic NTK convergence) requires both infinite width AND infinite rank, but practical SepNNs use small rank for efficiency—the very motivation for SepNNs. Under the practically relevant fixed-rank regime, Corollary 1 shows the NTK is random, and the paper acknowledges (Remark 3) that training dynamics "can not be characterized uniformly" in this setting. This means the spectral bias analysis (Eq. 5) and the convergence rate characterization—both assuming a fixed kernel—do not formally apply to the regime where SepPGD is actually used.

- **Approximation theorem provides no quantitative approximation rates**: Theorem 1 is purely existential—it confirms SepNNs are universal approximators but gives no relationship between the required rank R, factor network width, and approximation error ε. Without such bounds, one cannot assess whether the rank-scaling is favorable or whether exponential rank might be needed for certain function classes, which would negate the practical efficiency advantage. For a theory paper whose first main claim is an approximation result, the absence of any quantitative rate is a notable gap.

### Minor:

- **Scope mismatch between theory and algorithm**: Theorem 1 and the NTK results cover CP, TT, and Tucker SepNN variants, but SepPGD is developed and analyzed exclusively for CP SepNN. The introduction suggests all three are addressed, but the algorithmic contribution does not extend to TT or Tucker forms.

- **No ablation on rank R**: Despite R being the central structural parameter and the infinite-vs-fixed-rank distinction being a key theoretical finding, the experiments do not systematically vary R to study its impact on NTK spectra, spectral bias, or SepPGD effectiveness. This disconnects the theoretical finding (Corollary 1) from empirical practice.

- **No visualization of NTK eigenvalue spectra pre-/post-conditioning**: The core mechanism claimed is eigenvalue flattening, yet no figure directly shows the NTK spectrum before vs. after SepPGD preconditioning. Such a plot would directly validate the claimed mechanism.

## Nice-to-Haves

- Convergence guarantees for SepPGD (even asymptotic, under NTK regime assumptions) would strengthen the theoretical contribution significantly. The paper leaves this as "future research."
- Comparison with Fourier feature mappings or other spectral bias mitigation methods (beyond MSK) would better contextualize SepPGD's contribution.
- Generalization analysis: faster training convergence via spectral bias alleviation can come at the cost of overfitting noisy data, which is not addressed.

## Removed Points

- **Claim that the paper should compare SepNN+SepPGD vs. standard MLP+PGD head-to-head**: This ignores that the paper's contribution is SepPGD for SepNNs specifically; the efficiency advantages of SepNN over MLP were established in prior work (Liang et al., 2022; Cho et al., 2023). A head-to-head comparison would conflate architectural and optimizer contributions.

- **Claim about missing baselines like Adam or learning rate scheduling**: These are not spectral bias mitigation methods and address a different question. The relevant baselines (NTK-PGD, MSK) are correctly included.

- **Concern about "not yet released" or availability of models/datasets**: All cited models (SepNN architectures, KRR setups, PDE test problems) are standard.

- **Request for generalization analysis as a major weakness**: While relevant to practical deployment, the paper's stated scope is understanding and improving training dynamics (spectral bias and convergence). Generalization is outside this scope, and the experiments do show test-set results for several tasks.

- **Nitpick about notation density in Definition 1**: This is a formatting/style concern; the notation is standard for tensor decomposition papers.

- **Concern about the Hessian-based method complexity in Table 1 being O(P)**: This is background context, not the authors' contribution, and the stated assumption "n^D < P" is reasonable for overparameterized networks.

- **Claim that the NTK derivation should include multi-layer MLP details in the main text**: Theorem 2 specifies two-layer MLPs and Remark 1 explicitly notes extendability; the details in the appendix follow standard practice.

## Novel Insights

The most interesting structural insight is that SepNN's NTK admits an additive decomposition over factor NTKs (Lemma 1), which combined with the Kronecker-product structure over grids, transforms an O(n^D) preconditioning problem into D independent O(n^2) problems. This factorization insight—that separable architecture yields not just parameter efficiency but also spectral decomposability of the training dynamics—is the paper's most valuable conceptual contribution and goes beyond what prior NTK preconditioning work (Geifman et al., 2024; Shi et al., 2025) offers.

## Suggestions

1. **Soften the "provably" claim** to "provably for D=2" or provide the full proof for D>2. The gap between the claim and the proven result is the single most damaging issue for the paper's credibility.
2. **Add at minimum a sketch of quantitative approximation bounds** (even polynomial rates under smoothness assumptions), which would give practitioners guidance on rank selection.
3. **Include an ablation on rank R** with NTK eigenvalue spectrum plots, connecting the random-NTK theory to empirical observations.
4. **Add a small worked example** (e.g., D=2, R=2, n=3) to make Definition 1 accessible.

## Score and Decision

**Calibration**: I compared this paper against several anchors:
- **TNYLCF7vZA** (Inductive Gradient Adjustment/IGA for spectral bias in INRs, scores 8/3/5/3, withdrawn): Very similar topic (NTK preconditioning for spectral bias). IGA was criticized for insufficient attribution to Geifman et al. and lack of novelty in the core preconditioning idea. This paper shares some overlap concerns but offers a genuinely novel structural insight (SepNN's NTK decomposability) and a concrete O(nD) efficiency gain.
- **WWlxFtR5sV** (Operator preconditioning for PIML, scores 5/6/5/6/8/8, accept poster): Similar theoretical positioning but with preconditioning for PDE operators. That paper had strong theory but limited practical applicability (only linear PDEs).
- **dpDw5U04SU** (Minimum width for universal approximation, scores 8/8/5, accept poster): Clean theoretical result with tight bounds. More precise and complete than this paper's approximation theorem, which lacks rates.
- **G2Lnqs4eMJ** (Optimal NN approximation for high-dim functions, scores 3/5/1/1, reject): Existential approximation result with weak proofs and overclaimed significance—this paper's Theorem 1 is better positioned but still shares the existential-only limitation.

This paper has real substance: the NTK factorization, the efficiency gain, and the experimental results are solid. However, the theoretical claims (especially "provably") outpace what is actually proven, the approximation result is purely existential, and the random-NTK regime (the practically relevant one) has no convergence theory. The paper falls between a solid empirical+insight contribution and a complete theoretical treatment, landing in a territory where the theoretical claims are more aspirational than established.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>