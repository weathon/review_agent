Now I have enough information to write the final consolidated review. Let me compose it.

## Summary

This paper develops constraint-based causal discovery algorithms for stochastic processes modeled by SDEs with acyclic dependence graphs (allowing self-loops). The core contributions are: (1) a novel global Markov property for a "lifted dependence graph" that exploits time directionality to recover the full DAG rather than just an equivalence class (Proposition 3.1, Theorem 3.2); (2) a consistent signature kernel-based conditional independence test for path-valued random variables that does not require density assumptions; and (3) extensions to partially observed systems with unobserved confounders, including an impossibility result motivating the acyclicity assumption.

## Strengths

- **The lifted graph construction and Markov property (Proposition 3.1) is genuinely novel and elegant.** Splitting paths into past/future intervals and constructing a lifted graph where d-separation corresponds to the h-local independence relation ⊥⁺ₛ,ₕ is a clean theoretical device that enables full DAG recovery. The proof (Appendix A.6) leverages the factorization structure of the SDE generating mechanism with arguments from stochastic analysis, going beyond standard graphical model theory for Euclidean-valued variables.

- **Sound and complete causal discovery algorithm uniquely recovering the full DAG (Theorem 3.2).** Unlike the PC algorithm which only recovers a CPDAG equivalence class, Algorithm 1 provably identifies the entire dependence graph including loops under parent faithfulness—a weaker assumption than strong faithfulness. The alternative CPDAG post-processing (Corollary 3.3, Algorithm 2) provides a potentially more robust practical variant, and Table 2 confirms ⊥ₛᵧₘ tp.p. outperforms ⊥⁺ₛ,ₕ across all dimensions.

- **Consistent CI test for path-valued random variables without density assumptions (Section 3.2, Appendix A.14).** Existing consistency proofs for KCIPT/SDCIT require densities, which path-valued random variables lack. The new proof requires novel arguments and makes the test applicable beyond the SDE model (e.g., fractional Brownian motions, functional data). This is a legitimate methodological contribution of independent interest.

- **Simultaneous handling of multiple challenging settings.** The framework addresses (a) irregular sampling, (b) partial observations, (c) path-dependent drift/diffusion, and (d) diffusion-driven causal dependence—none of which are simultaneously addressed by existing methods. The impossibility result for cyclic SDEs (Appendix A.3) provides principled motivation for the acyclicity restriction.

- **Strong empirical performance in most settings.** Table 1 shows SigKer decisively outperforms all baselines in linear, path-dependent, and nonlinear settings. Table 2 shows clear superiority over PCMCI and most SCOTCH configurations for d ≤ 10. The unconditional test reaches near-perfect power at moderate sample sizes (Figure 3).

## Weaknesses

### Fatal
None.

### Major

- **Conditional CI test lacks dedicated power and type-I error analysis.** Figure 3 evaluates only the *unconditional* independence test ($X^1 \perp\!\!\!\perp X^2$). The conditional case ($X \perp\!\!\!\perp Y \mid Z$ with path-valued $Z$) is the actual bottleneck for causal discovery, as the paper itself acknowledges by citing Shah & Peters (2020) and Lundborg et al. (2022) on fundamental limits of CI testing. While the paper mentions appendix experiments on "forks, colliders, chains" (Appendix B.2), and the conditional test is indirectly evaluated through the end-to-end causal discovery results in Tables 1–2, there is no dedicated power curve or type-I error calibration for the conditional test. This gap matters because the permutation-based CI test's power degrades as the conditioning set grows—a well-known difficulty the paper should confront directly, especially since the method's practical reliability hinges on the conditional test's performance.

- **Misleading "hyperparameter-free" claim with asymmetric comparison to SCOTCH.** The paper states "SigKer—free of any hyperparameter choices" (Section 4), yet $s = 0.1 \cdot T$ was selected because it "performed best (Table 5)" and the choice of SDCIT vs. HSIC vs. KCIPT was also tuned (Appendix B.2, Figure 6). Meanwhile, SCOTCH's hyperparameter sensitivity is foregrounded as evidence of its unreliability. A fairer presentation would either: (a) acknowledge SigKer's own hyperparameter sensitivity (showing results across $s$ and $h$ values in the main text), or (b) give SCOTCH a single recommended default alongside SigKer's. The paper's argument that SCOTCH's optimal $\lambda$ varies unpredictably across settings is valid, but the asymmetry in presentation undermines confidence in the comparison.

### Minor

- **Partially observed setting evaluation is thin.** Figure 4 evaluates only a single 4-node graph (with one unobserved confounder) over 100 runs. While the result is encouraging (8 false adjacencies vs. SCOTCH's 88), one example does not constitute a systematic evaluation. Given that partial observations are one of the four claimed contributions and a major advantage over score-based methods, varying the number of latent variables, confounding strength, and graph size would significantly strengthen the evidence.

- **Underperformance in diffusion dependence setting.** In Table 1, SigKer achieves SHD of 72/63 in diffusion dependence, while SCOTCH's best configurations achieve 25/9. The paper's explanation—that SCOTCH's results vary unpredictably with hyperparameters—is partially valid, but the diffusion term is precisely where SDE-based methods should have the largest advantage. The paper should more directly acknowledge this as a current limitation rather than explaining it away entirely.

- **Pairs trading example does not validate causal direction.** The real-world experiment (Table 3) uses trading P&L as a proxy for correctness, but pairs trading strategies can profit from statistical dependence without requiring correct causal direction. The paper appropriately calls this a "proof-of-concept," but the framing still risks conflating downstream task performance with causal discovery accuracy.

- **Parent faithfulness assumption lacks plausibility analysis.** The paper claims parent faithfulness is weaker than standard faithfulness (Appendix A.8), but provides no argument for why it is more plausible in the SDE setting. Near-canceling causal effects in SDE coefficients could violate it, analogous to known faithfulness violations in linear models. This deserves at least brief discussion in the main text.

### Trivial
None.

## Nice-to-Haves

- Power curves for the conditional CI test (at least fork, collider, and chain structures) would significantly strengthen the empirical case and help practitioners understand the method's practical limits.
- Breakdown of SHD into edge additions, deletions, and reversals would reveal whether the method struggles more with detection or orientation.
- Wall-clock times for the causal discovery experiments would help practitioners assess feasibility for their problem sizes.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Abstract omits CI oracle caveat"**: Incorrect. The abstract explicitly states "assuming a CI oracle" (line 43).
- **"The procedure is just computing signature kernel Gram matrices then running existing permutation tests, so novelty is overstated"**: This mischaracterizes the contribution. The consistency proof without densities (Appendix A.14) requires genuinely novel arguments since existing proofs rely on density existence. The application to path-valued variables with different time intervals is also non-trivial.
- **"SHD ~49 at d=50 means roughly half the edges are wrong"**: This interpretation is unsupported without knowing the true number of edges. For sparse Erdős-Rényi graphs at d=50, an SHD of ~49 could represent a relatively small fraction of errors. The paper also acknowledges it outperforms SCOTCH only "up to d=10."
- **"Acyclicity is a severe restriction for dynamical systems where feedback is ubiquitous"**: The paper explicitly discusses this limitation (Section 1), provides an impossibility result for the cyclic case (Appendix A.3), and identifies it as future work. Criticizing the scope of what the paper explicitly scopes out is scope creep.
- **"The single example in Figure 4 does not constitute systematic evaluation"**: Kept as a minor weakness above, but the harsh critic's framing as a "half paragraph" understates the contribution—Appendix D contains full details.
- **"SCOTCH's best configuration achieves SHD 25/9 in diffusion dependence, the paper should acknowledge this"**: Kept as a minor weakness above, but the harsh critic's framing that the paper "dismisses" this is overstated—the paper does mention it and provides a reasoned (if incomplete) explanation.
- **Requests for missing related works**: Removed per instructions—cannot verify existence of cited works not in the paper.
- **Formatting/notation nitpicks**: Removed per instructions.

## Novel Insights

The paper's lifted graph construction reveals a general principle: when the data-generating process has a natural temporal ordering, constructing a "doubled" graph that separates past and future versions of each variable can turn undirected equivalence-class edges into directed ones via d-separation. This is distinct from, and arguably more principled than, simply adding time-indexed variables as in discrete-time unrolling, because the Markov property is proven from the SDE's factorization structure rather than assumed. The insight that one can recover the full DAG (not just CPDAG) using constraint-based methods when time directionality is available could generalize beyond the SDE setting to other ordered domains.

## Suggestions

- Add conditional CI test power curves (for at least forks, colliders, chains) to the main text, ideally with type-I error calibration at different conditioning set sizes. This is the single most impactful improvement for the paper.
- Replace the "hyperparameter-free" claim with a transparent discussion of SigKer's own sensitivity to $s$, $h$, and the choice of permutation test (SDCIT vs. KCIPT), and show results across these settings in the main text alongside SCOTCH's hyperparameter sensitivity.
- Expand the partially observed evaluation beyond the single example in Figure 4—vary graph size, number of latent variables, and confounding strength.

## Evaluation

**Originality:** High. The lifted graph Markov property, the sound/complete algorithm exploiting time directionality, and the consistency proof without densities are all novel contributions not present in prior work.

**Importance of research question:** High. Causal discovery for continuous-time stochastic processes with realistic complications (irregular sampling, path/diffusion dependence, partial observations) is an important and underaddressed problem.

**Claim support:** Moderate-to-good for theoretical claims; the main gap is empirical support for the conditional CI test. The end-to-end causal discovery results (Tables 1–2) provide indirect evidence, but dedicated conditional test evaluation is missing.

**Soundness of experiments:** Moderate. Experiments cover diverse settings and compare against relevant baselines, but the conditional test evaluation is thin and the SCOTCH comparison has the asymmetry noted above.

**Clarity:** Good. The paper is well-structured with clear algorithm descriptions and a logical flow from theory (CI oracle) to practice (signature kernel test). The lifted graph concept is illustrated effectively (Figure 2).

**Value to community:** High. The CI test alone fills a genuine gap (no existing consistent CI test for path-valued variables), and the framework opens a new approach to causal discovery in continuous-time systems.

## Calibration Anchors

- **SCOTCH (V1GM9xDvIY, avg 7.0, Accept Poster):** Directly comparable—SCOTCH is the primary baseline. This paper makes stronger theoretical contributions (Markov property, soundness/completeness, consistency proof) but has less polished empirical evaluation. Slightly below SCOTCH due to empirical gaps.
- **IDOL (2efNHgYRvM, avg 8.0, Accept Oral):** Similar level of theoretical contribution (identifiability results) with more polished experiments. This paper is clearly below IDOL due to the conditional CI test evaluation gap and asymmetric comparison.
- **Kernel CI test paper (GPcSYm89wK, avg 4.5, Reject):** Much weaker contribution—only marginal improvements in test power via kernel parameter selection. This paper is clearly above it.
- **Faithfulness violations (or8wkKoBP4, avg 4.0, Reject):** Purely theoretical with no experiments. This paper has substantial empirical results in addition to theory.
- **Low-scoring papers (2–3 range):** Fundamentally flawed or overclaimed. This paper has genuine, verifiable theoretical contributions and reasonable experiments—clearly above this tier.
- **CausalRivers (wmV4cIbgl6, avg 7.33, Accept Spotlight):** Large-scale benchmarking contribution with strong empirical results but less theoretical depth. This paper has more theoretical depth but less comprehensive evaluation.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>