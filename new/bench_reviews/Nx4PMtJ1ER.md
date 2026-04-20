## Summary
The paper introduces a rigorous framework for causal discovery in continuous-time stochastic processes modeled by SDEs. By defining a lifted dependence graph and a future-extended local independence relation, the authors derive a constraint-based algorithm that uniquely recovers the full DAG (rather than just a Markov equivalence class) by exploiting the direction of time. The method is made practical through a novel signature kernel-based conditional independence (CI) test, which is proven to be consistent for path-valued random variables. Extensive experiments demonstrate that the approach robustly outperforms established baselines (PCMCI, SCOTCH, CCM, Granger) across varying dimensions, data modalities, and partial observation settings.

## Strengths
- **Novel Markov property enabling unique full-DAG recovery**: Proposition 3.1 and Theorem 3.2 establish that the proposed future-extended $h$-local independence satisfies the global Markov property on a lifted graph, allowing the algorithm to recover the exact dependence graph rather than a Markov equivalence class. This is a significant theoretical advance over standard constraint-based methods that can only output CPDAGs or PAGs.
- **First consistent CI test on path space without density assumptions**: Section 3.2 addresses a critical gap by providing a consistent kernel-based CI test for path-valued variables (Appendix A.14), which requires novel arguments since standard proofs rely on the existence of densities that do not exist on path space. This contribution is of independent interest beyond causal discovery.
- **Comprehensive empirical superiority and hyperparameter-free operation**: Table 1 and Table 2 demonstrate that the proposed CI test consistently outperforms all baselines (including PCMCI and SCOTCH) in SHD across multiple dimensions ($d \in \{3, 5, 10, 20, 50\}$) and bivariate settings. Unlike SCOTCH, whose performance heavily depends on unselectable sparsity parameters ($\lambda$), SigKer requires no such tuning, making it reliably applicable in practice.
- **Robust handling of complex data modalities**: The method successfully handles irregular sampling, path-dependence, diffusion-dependence, and partially observed systems simultaneously (Section 1 criteria a–d), with the partial observation results (Figure 4) correctly handling unobserved confounders where score-based methods fail.

## Weaknesses

### Fatal
None.

### Major
None.

### Minor
- **Unquantified scaling of CI test with conditioning set size $|K|$**: While Table 2 shows the method degrades in structural Hamming distance as $d$ increases (e.g., SHD of $725 \times 10^{-2}$ at $d=20$ for $\perp_{\text{sym}}$), the paper does not explicitly ablate how the signature kernel CI test's power and Type I error scale with the dimensionality of the conditioning set $|K|$. Quantifying this would provide clearer limits on the algorithm's practical scalability.
- **Real-world case study is anecdotal rather than validating**: The pairs trading example (Table 3) uses P&L as a proxy for causal correctness. As profits are heavily confounded by market microstructure, transaction costs, and non-causal correlations, this serves as an interesting proof-of-concept but does not rigorously validate the discovered causal structures on real data.

### Trivial
- **Clarification needed on $p$-variation and Brownian motion**: Section 2 defines the signature kernel on paths of finite $p$-variation for $1 \leq p < 3$. While Brownian motion has infinite 1-variation, its sample paths have finite $p$-variation for any $p > 2$ almost surely, easily fitting within the stated range. A brief clarifying remark would prevent confusion for readers unfamiliar with the precise variation properties of stochastic processes.

## Nice-to-Haves
- Adding an explicit comparison of computational runtime against SCOTCH and PCMCI for large $d$ (e.g., $d=50$) would be beneficial, as constraint-based methods with kernel tests scale differently than variational neural SDE approaches.

## Removed Points
*These points are flagged to be removed or are significantly weakened based on verification against the paper:*
- **Theoretical mismatch between signature kernel and SDE path regularity**: The critic claims Brownian paths violate the $1 \leq p < 3$ finite $p$-variation assumption, breaking the kernel's characteristic property. This is incorrect: Brownian sample paths have finite $p$-variation for any $p > 2$ almost surely, fully contained within the paper's defined range. The PDE solver trick (Eq. 4) correctly computes the kernel for these paths without relying on a rough path lift.
- **Curse of dimensionality renders the method non-viable beyond small graphs**: The critic claims the method breaks down at $d \geq 20$, but Table 2 clearly shows it outperforms PCMCI ($3530 \times 10^{-2}$) and the best SCOTCH settings ($1525 \times 10^{-2}$) at $d=20$, proving its viability even as SHD naturally increases with dimension.
- **Overclaim on identifiability in partially observed settings**: The critic argues that unique edge orientation requires unproven assumptions about latent processes. The paper defers the formal proof to Appendix D, which provides the necessary conditions under the standard causal discovery assumptions. The claim is supported by the provided appendix.

## Novel Insights
The paper's core insight—using the arrow of time not merely to orient edges but to formally construct a global Markov property on a temporally *lifted* graph—is a substantial theoretical step. It cleanly bridges the gap between static constraint-based discovery, which is fundamentally limited to Markov equivalence classes, and continuous-time dynamics. By proving that local independence constraints on split time intervals ($X_{[0,s]}$ vs. $X_{[s,s+h]}$) uniquely identify the DAG, the work transforms a temporal property into a rigorous graph-theoretic identifiability condition. Furthermore, bypassing the need for densities in path-space CI testing opens a new theoretical avenue for hypothesis testing on stochastic processes.

## Suggestions
- Include an ablation study plotting the CI test's statistical power and Type I error rate as a function of the conditioning set size $|K|$ to clearly establish the method's operational limits.
- Explicitly clarify in Section 2 that Brownian motion sample paths (with $p$-variation $>2$ but $<3$) are natively supported by the signature kernel definition, preempting confusion regarding rough path theory prerequisites.

## Score and Decision
I compared this paper against similar causal discovery and SDE-themed papers in the calibration set. Compared to SCOTCH (V1GM9xDvIY, Scores: 6, 6, 8, 8, Accept Poster), which uses variational inference for continuous-time discovery but faces hyperparameter sensitivity, this paper provides a tighter theoretical guarantee (sound and complete algorithm) and empirical robustness. Compared to FoRDE (nLWiR5P3wr, Scores: 6, 8, 6, 8, Accept Spotlight), which uses kernel methods for functional data, this paper matches the empirical rigor while offering stronger theoretical identifiability results. The paper's theoretical contributions and comprehensive benchmarks position it at the level of the higher-scoring anchors.

MY FINAL SCORE: <pineapple>8.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>