Now I have enough calibration context. Let me carefully analyze the paper and write the final review.

Key issues to verify from the harsh critic:

1. **"Provably" claim overreach**: The abstract says "provably adjusting its NTK spectrum" and contribution 3 says "provably adjusts the eigenvalue distribution of NTK matrix." Looking at Section 4 text: "This can possibly be verified...We can ultimately show that KS͂ has better spectrum than K." and "It is believed that the result in Lemma 2...can be readily extended to multivariate cases D > 2." This confirms the "provably" claim is overreached - the proof is only for D=2 equivalence, and the spectral improvement argument is informal/heuristic.

2. **Architecture vs optimizer confounding**: The experiments compare SepPGD+SepNN vs PGD/MSK+MLP. The paper does have SepNN+GD baselines showing improvement. The paper acknowledges this: "Because the efficiency advantage of SepNN and SepPGD comes from the lower complexity in an iteration, we plot the convergence curve w.r.t. execution time rather than iteration number." However, the paper does show SepPGD improving over SepNN with standard GD in some figures. This is a valid concern but somewhat addressed.

3. **D>2 theory gap**: Lemma 2 only covers D=2, all experiments use D>2. This is a genuine gap.

Let me now compose the final review.</think>Now I have enough information to compose the final review. Let me carefully synthesize the findings.

## Summary

This paper provides theoretical foundations for Separable Neural Networks (SepNNs) and proposes an efficient optimization method. It makes three claimed contributions: (1) a universal approximation theorem for CP, TT, and Tucker SepNNs, confirmed via Stone-Weierstrass arguments; (2) NTK regime characterizations showing deterministic vs. random kernel limits under infinite vs. fixed rank; and (3) the SepPGD method that applies factor-wise preconditioning to alleviate spectral bias, with an O(nD) complexity per step versus O(n^D) for standard NTK-based PGD.

## Strengths

- **Universal approximation theorem (Theorem 1):** A clean and unified proof covering CP, TT, and Tucker variants that subsumes the prior bivariate result of Cho et al. (2023). The proof strategy—using Stone-Weierstrass to establish density of the separable function class, then applying universal approximation to each factor—is elegant and avoids the orthogonal-basis construction of prior work.

- **NTK decomposition and regime characterization (Lemma 1, Theorem 2, Corollary 1):** The NTK decomposition as a sum of factor NTK contributions (Lemma 1) naturally leverages the separable structure. The distinction between deterministic NTK (infinite width + infinite rank) and random NTK (infinite width + fixed rank) is both theoretically clean and practically relevant, since practical deployments use small ranks. The empirical validation in Fig. 1 directly confirms these two regimes.

- **Lemma 2 establishing SepPGD-PGD equivalence for D=2:** This is a non-trivial structural result showing that factor-wise preconditioning is equivalent to a Kronecker-product-structured global preconditioner. The computational complexity reduction from O(n^D) to O(nD) per step is dramatic and well-argued (Table 1).

- **Competitive empirical results:** The experiments across KRR, image/surface representation, and PINNs show consistent speedups, and SepPGD improves upon SepNN trained with standard GD, demonstrating some optimizer-level benefit.

## Weaknesses

### Fatal
None.

### Major

- **The "provably" claim for spectral bias alleviation is overstated relative to what is actually proven.** The abstract and contribution statement claim SepPGD "provably adjusts" the NTK spectrum, but what Lemma 2 actually proves is the *equivalence of form* between SepPGD and a structured PGD for D=2—specifically, that SepPGD equals PGD with preconditioner Ŝ = S₁⊗Iₙ + Iₙ⊗S₂. The subsequent argument that KŜ has a better spectrum than K is informal: the paper uses language like "This can possibly be verified" and "We can ultimately show" without completing the proof. The critical missing step is that better spectra of individual S_d versus K_{Θ_d} does not automatically imply a better spectrum for KŜ, since the eigenvalues of a product of non-commuting matrices depend on eigenvector alignment, not just individual eigenvalue distributions. Similarly, the D>2 extension is stated as "It is believed that...can be readily extended" rather than proven, yet all experiments use D>2 settings. This gap between the "provably" framing and the actual theoretical content is significant.

- **Experimental comparisons conflate architecture and optimizer efficiency.** All headline comparisons in Figures 2–4 compare SepPGD+SepNN against PGD/MSK+MLP, mixing both the architecture (SepNN vs. MLP) and the optimizer (SepPGD vs. standard PGD). While SepPGD+SepNN vs. SepNN+GD is shown in some panels (e.g., Fig. 2(a)), the primary metric is wall-clock time, making it impossible to attribute speedups to the optimizer's convergence-per-iteration improvement versus the architecture's lower cost per forward/backward pass. A clean ablation—standard PGD applied to SepNN compared against SepPGD on the same architecture, measured in iterations—would isolate the optimizer-level benefit but is absent.

### Minor

- **No ablation on modulation hyperparameter k:** The modulation function g(λᵢ) = λ^k for i ≤ k depends on a cutoff rank k, which controls how many eigenvalues are flattened. This hyperparameter directly affects preconditioner quality, yet no ablation or sensitivity analysis on k is provided, leaving practitioners without guidance on its selection.

- **SepPGD convergence guarantee deferred to future work:** The paper notes that convergence and solution consistency of SepPGD "can also be deduced" from the equivalence with NTK-based PGD and representer theorem arguments, but this is "left for future research." While reasonable given the existing results, a formal convergence rate would strengthen the practical claims.

- **NTK spectral properties of SepNN vs. standard MLP are not compared:** The paper characterizes spectral bias of SepNN through its NTK eigenvalues but never compares the SepNN NTK's eigenvalue decay rate to a standard MLP's NTK on the same data. If SepNN inherently has more severe spectral bias (faster eigenvalue decay), this would motivate preconditioning more strongly; if less, the need for specialized preconditioning is less clear.

### Trivial
None.

## Nice-to-Haves

- A formal convergence rate or condition number bound for SepPGD (even just for D=2), which would substantiate the spectral improvement claim.
- Proving or disproving the D>2 extension of Lemma 2, since all major experiments operate in D>2.
- Visualization of eigenvalue spectra comparing K, KŜ (SepPGD), and KS (standard PGD) on actual problem instances.

## Removed Points

- **Reproducibility/code availability concern:** The paper provides a GitHub link. Removed per hard rule (questioning availability of cited resources).

- **Missing related works on alternative SepNN/efficient architectures:** Per hard rule, no flagging of missing citations.

- **Formatting/presentation nitpicks (notation inconsistencies, parser artifacts):** Removed per hard rule on formatting issues.

- **Claim that SepPGD theoretical guarantees are only for D=2 while experiments use D>2 (separate from the "provably" issue):** This is already captured in the Major weakness above about the overclaim; listing it twice would be redundant. The specific gap about D>2 extension being unproven is already noted.

## Novel Insights

The paper identifies an insightful structural connection: that factor-wise preconditioning in SepPGD is equivalent (for D=2) to a global PGD with a Kronecker-product-structured preconditioner whose cost scales as O(nD) rather than O(n^D). This reframes preconditioning for this architecture class as exploiting the tensor structure of the NTK itself, rather than resorting to mini-batch approximations. The NTK regime distinction (deterministic under joint infinite limits vs. random under fixed rank) is also a practically relevant insight for practitioners choosing rank parameters.

## Suggestions

- Tone down "provably adjusting" to "designed to adjust" or "aimed at adjusting" in the abstract and contribution statements, unless and until the spectral improvement is formally proven.
- Add an ablation comparing standard PGD applied to SepNN vs. SepPGD on SepNN (same architecture), measuring convergence per iteration, to isolate the optimizer's contribution from the architecture's efficiency.
- Include a simple ablation or sensitivity study on the modulation hyperparameter k.

## Score and Decision

**Calibration anchors:**

- **High band (≥6):** WWlxFtR5sV (operator preconditioning for PINNs, rigorous theory, avg 6.33); AoraWUmpLU (NTK global convergence for Neural ODEs, avg 8.0); SyVPiehSbg (ActNet/KST universal approximation, avg 7.5); ydlDRUuGm9 (KAN expressiveness/spectral bias, avg 6.25)

- **Medium band (~5):** eQggPqESBr (optimization threshold in two-layer ReLU, avg 5.50); 5EtSvYUU0v (NTK/NNGP unification, avg 6.0); aVlDNbvmCK ( preconditioned neural arch for PINNs, overclaimed theory, avg 3.5)

- **Low band (≤4):** 2XdRkRHBT9 (overclaimed "provably" barren plateau avoidance, avg 4.0); a8XwgTZzE0 (vague grokking theory, avg 2.0); WpQbM1kBuy (overclaimed "provably" with theory-practice gap, avg 4.25)

This paper is significantly stronger than the low-band papers (it has genuine technical content: a real approximation theorem, real NTK analysis, real experiments, and Lemma 2 is a real structural result). However, the overclaim on "provably" spectral improvement and the D=2 vs. D>2 theory-experiment gap place it below the high-band papers like AoraWUmpLU which have rigorous proofs matching their claims. Compared to ydlDRUuGm9 (KAN spectral bias, avg 6.25) which also studies spectral bias of alternative architectures, this paper has a comparable scope but with the overclaim issue. Compared to WWlxFtR5sV (preconditioning for physics-informed ML, avg 6.33), this paper has a similar optimizer perspective but weaker theoretical guarantees for its core claim. The overclaim pushes it below the 6.25 range but the substantive contributions (approximation theory, NTK decomposition, Lemma 2, empirical results) keep it solidly in the 5–6 range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>