=== CALIBRATION EXAMPLE 36 ===

# Final Consolidated Review
## Summary

This paper proposes a constraint-based causal discovery framework for continuous-time stochastic processes governed by acyclic (except for self-loops) SDE models. The central theoretical contribution is the construction of a "lifted dependence graph" $\tilde{\mathcal{G}}$ that splits observed paths into past and future segments, combined with a Markov property (Proposition 3.1) establishing that $d$-separation in $\tilde{\mathcal{G}}$ characterizes the conditional independence structure. Building on this, the authors prove sound and complete causal discovery algorithms (Theorem 3.2, Corollary 3.3) that, crucially, recover the *full* underlying DAG—not just a Markov equivalence class—by exploiting the arrow of time, with extension to partially observed settings via an FCI-inspired approach. To make this practical, they introduce a signature kernel-based conditional independence test for path-valued random variables with a novel consistency proof that does not require densities, empirically demonstrating superior performance in bivariate settings and on functional data beyond SDEs.

---

## Strengths

- **Full DAG identification via time directionality**: Most constraint-based causal discovery algorithms recover only an equivalence class (CPDAG or PAG). This paper proves that exploiting temporal asymmetry—conditioning on past path increments and testing against future increments via $\perp_{s,h}^+$—yields unique identification of the full underlying DAG (Theorem 3.2). This is a substantive improvement over Laumann et al. (2023), who neither exploit the arrow of time nor handle partial observations.

- **Weaker faithfulness suffices**: The completeness proof relies only on "parent faithfulness" rather than the standard full faithfulness assumption (Appendix A.8). This is a non-trivial improvement: full faithfulness is known to be fragile in dense graphs, making the weaker requirement practically important.

- **Novel consistency proof for path-space CI testing**: Existing proofs of consistency for KCIPT/SDCIT rely on the existence of densities. The authors provide new arguments (Appendix A.14) applicable to path-valued random variables without densities. The test also handles irregular sampling and fractional Brownian motions (Figure 8), settings for which no comparable test previously existed.

- **Provable handling of partial observations**: The authors prove that their symmetric CI constraint $\perp_\text{sym}$ satisfies a global Markov property with respect to the induced ADMG, enabling an FCI-inspired algorithm that recovers a unique ancestral graph rather than an equivalence class of PAGs. Score-based methods such as SCOTCH are fundamentally incapable of handling latent confounders in this manner; Figure 4 concretely demonstrates this advantage (88 vs. 8 false adjacencies over 100 runs).

- **Empirical strength in bivariate and low-to-mid dimensional settings**: Table 1 shows decisive margins over all baselines in linear, path-dependent, and nonlinear bivariate SDE settings. Table 2 shows consistent superiority over PCMCI across all dimensions and over most SCOTCH configurations up to $d=10$. The power analysis (Figure 3) shows the signature kernel test reaching near-perfect power at $n \geq 40$, substantially outperforming the only existing alternative (Laumann et al., 2023).

---

## Weaknesses

### Fatal
None.

### Major

- **High-dimensional performance is substantially worse than SCOTCH (Table 2)**: At $d=20$ and $d=50$, SCOTCH (200, 2k) achieves SHD of $370 \pm 174$ and $538 \pm 70$ respectively, while the best SigKer variant ($\perp_\text{sym}$ + pp) achieves $725 \pm 439$ and $4593 \pm 93$—roughly a $2\times$ to $8\times$ gap. The authors argue that SCOTCH's hyperparameter (200, 2k) cannot be reliably selected without ground truth. While this argument is valid in principle (no principled cross-validation is available for causal discovery tasks), it is weakened by the fact that the *same* hyperparameter setting (200, 2k) consistently yields the best SCOTCH performance across both $d=20$ and $d=50$, suggesting it is a robustly good setting rather than a lucky coincidence. The paper's claim that "SigKer broadly outperforms the state-of-the-art, such as SCOTCH" is not supported by Table 2 for $d \geq 20$, and the authors should characterize this limitation more honestly. A supplementary analysis of what SHD SCOTCH achieves under a reasonable prior over hyperparameter choices (not just best-case vs. worst-case) would strengthen or fairly contextualize the comparison.

- **Type I error / test size control is not empirically verified**: For any CI test used as the backbone of a constraint-based algorithm, controlling the false positive rate under the null is as fundamental as test power. The paper reports power curves (Figure 3) but never reports empirical rejection rates under $H_0$, calibration plots, or Type I error at the nominal significance level. In constraint-based methods, inflated Type I error directly causes spurious edges in the recovered graph. This is a critical omission for a paper whose primary deliverable is a CI test.

- **Misleading "hyperparameter-free" claim**: The paper asserts "SigKer—free of any hyperparameter choices" (Section 4), but the method uses: (a) an RBF base kernel with length scale set by the median heuristic, (b) the time-split parameters $s$ and $h$ (fixed to $s=0.1T$ after Table 5 in the appendix), (c) signature truncation depth, and (d) number of permutations. While these are less sensitive than SCOTCH's $\lambda$, the claim of being hyperparameter-free is factually inaccurate and unfairly frames the comparison. The text should be revised to acknowledge that SigKer has hyperparameters but is substantially less sensitive to them than SCOTCH, particularly for $\lambda$.

### Minor

- **Partial observation evaluation is severely under-powered**: The evaluation of the partial observation setting consists of a single graph (Figure 4). While the result is impressive, a single example provides insufficient evidence to substantiate broad claims of superiority. A systematic table varying the number of latent confounders, graph size, and observation rate would meaningfully strengthen this section.

- **Finance case study lacks rigor for the claims made**: The pairs trading case study spans only three years (2010–2012) of post-financial-crisis data on 10 stocks, with no comparison against SCOTCH (which is available), no correction for multiple testing across pairs, and no reporting across multiple time windows to verify the result is not period-specific. The authors present it as a "proof-of-concept," which is reasonable, but the language around Sharpe ratio of 1.5 vs. 0.09–0.24 implies a stronger conclusion than the evidence supports. The causal interpretation of trading P&L is also speculative—a method that better captures statistical path dependencies (not necessarily causal ones) could achieve the same gain.

- **CI test power analysis is restricted to the linear setting**: Figure 3 reports power only for a linear SDE (Eq. 5 with $B \equiv 0$). Since the paper's primary claims of generality extend to path-dependent and nonlinear SDEs, power curves in those settings should also be included or discussed.

- **Sensitivity analysis for $s, h$ is deferred to the appendix with minimal main-text discussion**: The paper acknowledges that $s=0.1T$ was selected based on Table 5 (appendix) but provides no theoretical guidance. Even a brief intuition for why $s=0.1T$ works and what regime it exploits would aid practitioners and clarify the practical robustness of the algorithm.

### Tiny

- The relationship between Algorithm 1 (direct recovery via $\perp_{s,h}^+$) and Algorithm 2 ($\perp_\text{sym}$ + post-processing), and the trade-offs guiding when to use each, is discussed in prose but could be summarized in a small decision table or figure for practitioners.
- The post-processing rule (Corollary 3.3) relies on the independence of initial conditions $X_0^k$ from noise processes $W^k$. For real data where initial conditions may reflect pre-observation history, this assumption deserves a sentence of explicit discussion about potential violations.

---

## Nice-to-Haves

- **Runtime and scalability comparison**: A wall-clock time plot (SigKer vs. SCOTCH vs. PCMCI) across dimensions $d$ would allow users to understand computational trade-offs. The paper's claim of practical usability is plausible given the signature kernel's efficient PDE-based computation, but without timing data this remains unverified.

- **Cycle detection diagnostic**: Given the strict acyclicity assumption (and the impossibility result for cyclic graphs in Appendix A.3), a brief discussion of potential diagnostics or heuristics for detecting when the model may be misspecified (i.e., when cycles are present) would be practically valuable for users.

- **Signature depth ablation**: An experiment varying signature truncation level would help practitioners understand the power-versus-cost trade-off of the core feature extractor and confirm that the default depth is well-chosen.

- **Test under the null for path-dependent and irregular sampling settings**: While Figure 3 tests power in the linear setting, a corresponding null-rejection rate plot in the same or harder settings would simultaneously address the missing Type I error verification and demonstrate robustness to the irregular sampling the authors claim to handle.

- **Empirical demonstration of faithfulness violation behavior**: A brief simulation showing how the method degrades (or degrades gracefully) when faithfulness is violated would help users understand practical risk.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **Duplicate line numbers in Algorithm 1 (lines 7 and 8)**: Pure formatting nitpick; does not affect correctness.
- **Contributions listed twice in abstract and contributions paragraph**: Pure style concern.
- **The impossibility result for cyclic graphs "underplayed"**: The paper explicitly discusses this in Section 1 under "Limitations and requirements" and provides the formal result in Appendix A.3. Its placement is reasonable and not a meaningful omission.
- **Proof of Markov property for path-dependent coefficients**: The harsh critic flagged concern about whether the factorization argument holds for non-Markovian, path-dependent coefficients. The paper explicitly states its SDE model is path-dependent (Eq. 1), and the proofs in Appendix A.6 are stated to handle this generality. We do not remove the edge case but no evidence of an error was found.
- **Claim that SCOTCH's best-case performance for diffusion dependence (Table 1, $\lambda=50, n_e=2k$ and $\lambda=100, n_e=2k$) challenges SigKer unfairly**: This is actually an asymmetric comparison that *favors the baseline* (best-case SCOTCH vs. hyperparameter-free SigKer), so pointing it out as a weakness of SigKer is unfair to the authors.

---

## Novel Insights

The most genuinely novel conceptual contribution of this paper is the construction of the lifted dependence graph $\tilde{\mathcal{G}}$ as a principled device to translate the continuous-time Markov property of SDE models into a graphical criterion amenable to CI-based discovery. The key insight—splitting paths at a common time point $s$ so that the "past" ($V_0$) and "future" ($V_1$) variables inherit a DAG structure, in which d-separation on $\tilde{\mathcal{G}}$ recovers the CI structure of the original SDE—is elegant and has broader implications. In particular, it shows that the arrow of time in SDEs provides *strictly more* identifiability than in static SCMs: the temporal asymmetry breaks Markov equivalences that would otherwise leave edges undirected, achieving unique graph identification without additional distributional assumptions (e.g., non-Gaussianity or equal noise variances). The impossibility result for cyclic graphs (Appendix A.3) is equally sharp: it establishes that no CI-based method of the form studied here can recover the full graph when cycles exist, clarifying the fundamental boundary of constraint-based causal discovery for dynamical systems.

---

## Suggestions

1. **Revise the "hyperparameter-free" claim**: Replace with an accurate statement such as "SigKer has hyperparameters (base kernel length scale via median heuristic, time split $s=0.1T$, signature truncation depth), but is substantially more robust to their specification than SCOTCH, particularly for the critical sparsity parameter $\lambda$."

2. **Report Type I error control**: Add a table or figure showing empirical rejection rates under the null $H_0: X \perp\!\!\!\perp Y$ (or conditional null) at the nominal significance level $\alpha \in \{0.01, 0.05\}$ for both the linear and at least one nonlinear SDE setting. This is essential to validate the CI test as scientifically sound.

3. **Provide a more honest characterization of high-dimensional performance**: Revise Section 4 to clearly state that at $d \geq 20$, SigKer is outperformed by SCOTCH under its empirically best (though not oracle-selected) hyperparameter setting, and discuss whether this gap is expected to shrink with more samples or is structural.

4. **Expand the partial observation experiment**: Replace Figure 4 with a table varying graph size, latent variable count, and sample size, reporting SHD mean ± SE over multiple graph instances. This is the minimum needed to substantiate the partial-observation claims.

5. **Expand the finance case study or reframe its claims**: Either test across multiple non-overlapping time windows (2004–2006, 2007–2009, 2013–2015) to assess whether the Sharpe ratio advantage is robust, or explicitly describe Table 3 as a single-instance exploratory result with no claim of statistical significance.

6. **Move key scalability discussion to main text**: At minimum, add a one-paragraph summary of the complexity analysis from Appendix B.9 and either a runtime table or a brief comparison of wall-clock times for representative configurations, alongside Table 2.

# Actual Human Scores
Individual reviewer scores: [8.0, 10.0, 6.0, 8.0]
Average score: 8.0
Binary outcome: Accept
