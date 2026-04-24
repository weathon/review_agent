Now I have sufficient calibration data to write the final review. Let me synthesize everything.

---

## Summary

This paper provides a comprehensive theoretical foundation for separable neural networks (SepNNs), making three contributions: (1) a universal approximation theorem (UAT) for CP, TT, and Tucker SepNNs for general dimension D ≥ 2 via Stone-Weierstrass and classical UAT; (2) derivation of NTK regimes under two asymptotic limits (deterministic at infinite width+rank; random at infinite width but fixed rank); and (3) a separable preconditioned gradient descent (SepPGD) that reduces preconditioning complexity from O(n^D) to O(nD) for n^D training samples. Experiments span kernel ridge regression, image/surface implicit neural representations, and 3D PINNs.

---

## Strengths

- **Theorem 1 provides a unified and rigorous UAT for all three SepNN decomposition types (CP, TT, Tucker) for arbitrary D ≥ 2**, extending the prior bivariate-only result of Cho et al. (2023). The proof technique—Stone-Weierstrass density + standard MLP UAT on the dense subclass—is clean and general.

- **The O(nD) complexity reduction for SepPGD is a concrete and substantial algorithmic contribution**, clearly documented in Table 1. The key insight of decomposing the large Kronecker preconditioner into D smaller n×n factor preconditioners via vec(ABC) = (C^⊤ ⊗ A)vec(B) is elegant and directly measurable.

- **Lemma 2 formally establishes that SepPGD is equivalent to classical NTK-based PGD with S̃ = S₁⊗I_n + I_n⊗S₂ for D=2**, providing a rigorous link between the separable computation and prior preconditioned-gradient methods (Geifman et al., 2024). This is the paper's sharpest theoretical result.

- **The distinction between deterministic NTK (W,R→∞) and stochastic NTK (W→∞, fixed R) is scientifically honest and useful** (Theorem 2 vs Corollary 1). Acknowledging that the deterministic NTK regime—which underpins training dynamics and spectral bias theory—requires infinite rank, while practical SepNNs use finite rank, is a notable instance of theoretical transparency rather than hand-waving.

- **Broad experimental validation across diverse domains** (KRR, image INR, surface representation, 3D PINNs with diffusion, Klein-Gordon, and Helmholtz equations) establishes that SepPGD generalizes across application types, not just a single setting.

---

## Weaknesses

### Fatal
None.

### Major

- **Lemma 2 (the theoretical bridge between SepPGD and NTK-based PGD) is proved only for D=2, yet the core experiments use D=3.** The PINN experiments (3D diffusion, Klein-Gordon, Helmholtz) operate at D=3, and the paper states Lemma 2 "can be readily extended" to D>2 without providing a proof or sketch. For D=3, the Kronecker structure becomes S̃ = S₁⊗I⊗I + I⊗S₂⊗I + I⊗I⊗S₃, and verifying that SepPGD remains equivalent to this structured preconditioner is non-trivial. The theoretical justification for the primary experimental setting is therefore absent. This is not a missing appendix; it is a genuine gap between the theorem's scope and the experiments it is supposed to support.

- **The abstract's claim that SepPGD "provably adjusts its NTK spectrum" is partially overstated.** Lemma 2 establishes that SepPGD is equivalent to NTK-based PGD with a Kronecker preconditioner S̃ — that is provable. However, the claim that this *actually adjusts the spectrum of the true NTK K* requires that K̃ = KΘ₁⊗I + I⊗KΘ₂ is a good approximation to K, i.e., the cross-dimension NTK terms are small. The paper says this "can possibly be verified" using Lemma 3, and then explicitly defers convergence guarantees to "future research." The spectral improvement is thus conditional on an unverified approximation, and "provably" is too strong a word for the current state of the argument.

### Minor

- **Theory-practice regime mismatch.** The spectral bias characterization in Section 3 (and the motivation for SepPGD) rests on the deterministic NTK (W,R→∞), but Corollary 1 and Remark 3 concede that for practical finite rank, the NTK is random and the deterministic training dynamics do not apply. The paper cites Appendix Table 3 as empirical evidence that "even with small rank, SepPGD is effective," which is reasonable empirical support, but the theoretical chain from spectral bias characterization to SepPGD's design has a known gap that deserves more prominent acknowledgment (not just a remark).

- **The KRR experiment compares different models (SepNN+SepPGD vs. MLP+MSK), not just different optimizers.** The performance gain cannot be attributed purely to preconditioning vs. architecture. An ablation with the same SepNN under Adam, standard GD, and SepPGD would isolate the optimizer contribution.

- **No complexity scaling study.** The central algorithmic claim is O(nD) vs O(n^D). No figure shows wall-clock time as a function of n or D, which would provide direct empirical confirmation of the theoretical advantage at the heart of Table 1.

### Trivial

- The update frequency heuristic ("every ten iterations") is stated without justification; its sensitivity to this choice is not discussed.

---

## Nice-to-Haves

- Plotting the eigenvalue distribution of KS̃ vs K before/after preconditioning would directly validate the spectral adjustment mechanism empirically.
- A rank ablation (performance as a function of R) would help delineate the practical regime where SepPGD is effective.
- A proof sketch for the D>2 generalization of Lemma 2 would significantly strengthen the theoretical coverage.
- A convergence theorem for SepPGD, even under simplifying assumptions, given that Lemma 2 plus Geifman et al.'s existing convergence result likely provides a tractable path.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "the paper should not be accepted in its current form; Lemma 2 must be extended or experiments restricted to D=2."** This overclaims the severity. The D>2 extension issue is a major gap, but the paper has three independent contributions; the UAT (Theorem 1) and the NTK regime analysis are complete and valid independently of the D=2 restriction on Lemma 2. The algorithmic contribution and empirical results remain meaningful. Framing this as a reason to fully reject is too strong given the remainder of the contributions.

- **Harsh Critic: "The claim it includes Cho et al. as a special case is an overstatement because different norms."** This is a minor precision issue (sup-norm vs. L² norm), but the substance—generalizing from bivariate to D-variate and covering all three decomposition types—is clearly novel and correctly framed. The norm distinction is not material to the paper's claim of broader coverage.

- **Harsh Critic: Section 3 spectral bias is "essentially a restatement of known NTK spectral bias results."** The specific derivation of the SepNN NTK kernel structure (Lemma 1) and its factored Kronecker form, which yields the regime analysis in Theorem 2 and Corollary 1, is specific to SepNNs and not a trivial restatement. The spectral bias phenomenon itself is known, but its characterization for SepNNs is new.

- **Strength Finder: "Kronecker product structure of SepNN NTK matrix for grid inputs."** This is valid but is subsumed by the broader Lemma 1 and Lemma 2 discussion already credited in the Strengths section; repeating it as a separate point would be redundant.

---

## Novel Insights

The most insightful contribution, which is not fully highlighted in the paper itself, is the combination of Lemma 2 and the complexity argument: rather than the efficiency gain being ad hoc, the paper shows that SepPGD's separable structure is *algebraically equivalent* to applying a structured Kronecker preconditioner to the full NTK-based PGD. This is a principled explanation for *why* separable preconditioning works, not merely an observation that it does. The paper also makes an honest and rare observation that the practical SepNN regime (finite rank) yields a *random* NTK, which is theoretically distinct from the regime used for spectral analysis—a distinction most NTK papers gloss over.

---

## Calibration Anchors

| Path | Avg score | Comparison |
|------|-----------|------------|
| `/home/wg25r/review_agent/human_reviews/WWlxFtR5sV.md` | 6.33 | Similar theory+preconditioning for PINNs; narrower experiments (linear PDEs only), accepted poster |
| `/home/wg25r/review_agent/human_reviews/1op5YGZu8X.md` | 6.40 | NTK theory applied to a specific problem (adversarial training); comparable theory-practice gaps, accepted poster |
| `/home/wg25r/review_agent/human_reviews/xGvPKAiOhq.md` | 8.00 | Stronger theoretical completeness with novel lower bounds; sets the high bar |
| `/home/wg25r/review_agent/human_reviews/TBh4XQAXEb.md` | 3.50 | Theory only valid for simplified settings, weaker contributions overall |
| `/home/wg25r/review_agent/human_reviews/C85eSjKenO.md` | 5.25 | Tensor decomposition for neural network training; rejected, comparable breadth but less theoretical depth |
| `/home/wg25r/review_agent/human_reviews/6K81ILDnuv.md` | 5.25 | PINNs with enhanced loss formulation; moderate theoretical content, rejected |

The paper under review is most comparable to WWlxFtR5sV (6.33) and 1op5YGZu8X (6.40): both have real theoretical contributions applied to neural network optimization, both have acknowledged theory-practice gaps (this paper's is slightly more significant—Lemma 2 for D=2 only), and both were accepted as posters. The present paper's experimental breadth exceeds WWlxFtR5sV. However, the D=2 restriction on the core theoretical lemma (used for D=3 experiments) and the overstated "provably" claim pull the score slightly below these anchors. I place the paper at **5.5** — just below the borderline of acceptance in this cohort, needing the Lemma 2 extension to reach poster-acceptance level.

---

## Score and Decision

**Originality:** Moderate-high. The UAT extension and NTK regime analysis for SepNNs are genuinely new, and the Kronecker preconditioning insight is non-trivial.

**Importance of research question:** High. SepNNs are increasingly used in INRs, PINNs, and scientific ML; theoretical foundations are genuinely needed.

**Claims well-supported:** Partially. UAT and NTK regimes are fully supported. The "provably adjusts NTK spectrum" claim is partially supported (for D=2, conditional on K≈K̃). The D=3 experimental results are not fully covered by the theory.

**Soundness of experiments:** Good but with gaps (no scaling study, model-optimizer confound in KRR).

**Clarity of writing:** Good overall, with some notation overload in Section 4.

**Value to research community:** Concrete algorithmic improvement (O(nD) complexity) and theoretical framework for an increasingly important architecture class.

## Score and Decision

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>