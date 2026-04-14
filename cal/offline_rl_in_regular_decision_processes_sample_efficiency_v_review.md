=== CALIBRATION EXAMPLE 22 ===

# Final Consolidated Review
## Summary

This paper studies offline RL in Regular Decision Processes (RDPs), a class of non-Markovian environments whose dynamics can be captured by a hidden finite-state automaton. The authors propose two improvements over the prior state-of-the-art algorithm RegORL/ADACT-H: (1) a novel language-theoretic metric $L_X$ grounded in the dot-depth hierarchy of formal language theory, which avoids exponential decay of state distinguishability suffered by the standard $L_\infty^p$ metric in structured domains (e.g., the T-maze), yielding provably exponential sample complexity gains; and (2) a Count-Min-Sketch (CMS) variant that preserves the same asymptotic sample complexity while substantially reducing memory requirements. Both contributions are supported by PAC sample complexity bounds and validated experimentally.

---

## Strengths

- **Exponential separation theorem with a clean separation witness.** Theorem 1 formally establishes that the $L_{\mathcal{X}_{2,1}}$-distinguishability in the T-maze family is $\Omega(1)$, while the $L_\infty^\ell$-distinguishability is $\mathcal{O}(2^{-N})$. This is not a vague claim but a proved exponential separation backed by an explicit construction of the language ($\{x \text{ North } \mathcal{O}/r\, y \mid \ldots\} \in \mathcal{X}_{2,1}$) that witnesses the separation. Most prior work either claims intuitive improvements without formal separation results or works only with specific metric families.

- **Unifying language metric framework.** Definition 2 provides a genuine unification: $L_X$ reduces to $L_\infty$, $L_1$, $L_\infty^p$, and total variation as special cases of $\mathcal{X}$. The two-dimensional hierarchy $\mathcal{X}_{i,j}$ built from the $C_k^\ell$ operator offers a principled continuum between computational tractability (small $j$) and metric strength (large $j$), grounded in the dot-depth hierarchy from formal language theory.

- **Correction of prior work with quantitative impact.** The paper identifies and corrects a mistake in Cipollone et al. (2023)'s proof that adds a multiplicative $\sqrt{H}/\mu_0$ factor to both the previous and the new bounds. This is an honest and rigorous contribution that affects the quantitative conclusions of the predecessor work.

- **Compelling empirical scalability demonstration.** Figure 2 directly validates Theorem 1's exponential gap: the language metric achieves linear growth in both runtime and automaton size with corridor length $N$, while CMS grows exponentially—this matches the theoretical prediction and is a clean, falsifiable claim. Table 1 further shows the language metric reaching near-optimal policies in T-maze(c), Cheese, and Mini-hall where CMS and FlexFringe fail or time out.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing RegORL/ADACT-H baseline.** The paper explicitly claims to improve upon RegORL (Cipollone et al., 2023), but Table 1 does not include ADACT-H with the standard $L_\infty^p$ distance as a direct comparison. CMS is positioned as an improvement over plain ADACT-H (same sample complexity, lower memory), but there is no experiment that shows plain ADACT-H's performance, making it impossible to isolate the contribution of each proposed technique empirically. The claimed improvement over the SOTA for RDPs is thus unquantified in Table 1.

- **No sample efficiency curves (performance vs. dataset size).** The title promises "sample efficiency," and the central theoretical claim is that the language metric requires fewer samples than the $L_\infty^p$-based approach. Yet all experiments are run at a fixed $K=100$ episodes with no variation. There is no plot of return vs. number of episodes to empirically validate the claimed reduction in sample complexity. Without such a plot, it is impossible to confirm whether the language metric achieves the same policy quality with fewer data points in practice.

- **No principled guidance for selecting the hierarchy parameters $i$ and $j$.** Assumption 1 states that $\pi^b$ ensures $L_{\mathcal{X}_{i,j}}$-distinguishability of at least $\mu_0 > 0$, with $\mathcal{X}_{i,j}$ given as an input to the algorithm. All experiments use $\mathcal{X}_{3,1}$, but no guidance is provided on how a practitioner should choose $i$ and $j$ for a new problem. If $j$ is too small, states may be indistinguishable and Assumption 1 is violated silently (since $\mu_0 > 0$ cannot be verified from data without knowing the true automaton). If $j$ is too large, computational cost increases. This practical gap is the main barrier to deploying the method.

### Minor

- **$d_m^*$ bottleneck is acknowledged but not resolved.** Both Theorems 2 and 3 include $1/d_m^*$ where $d_m^* = \min_{u,a,o} d_t^*(u,a,o)$. The paper explicitly notes (line 220) that this "depends exponentially on $H$ if there exists an RDP state that is very hard to reach." However, Example 4's claim of "exponential gain" focuses exclusively on the $1/\mu_0^2$ term improving from $\Omega(4^N)$ to $O(1)$, without verifying that $d_m^*$ is favorable in the T-maze family. If $d_m^*$ is also exponentially small, the overall bound remains exponential. The authors should verify or discuss the behavior of $d_m^*$ in the T-maze to complete the argument.

- **CMS failure in Cheese is unexplained.** In the Cheese domain ($H=6$), CMS achieves only $0.40 \pm 0.05$ while the language metric achieves $0.87 \pm 0.03$. The paper provides no explanation for this large gap. Since CMS is meant to preserve the statistical test with only a memory tradeoff, a nearly 50% gap in reward at such a small horizon ($H=6$) suggests hash collision effects may be corrupting the statistical test in practice. This deserves investigation.

- **Bug correction in Cipollone et al. (2023) is underexplained.** The identification of the error (line 212) is a notable contribution, but the paper mentions it only in a single sentence. What was the nature of the mistake? Does it affect the qualitative conclusions of RegORL (e.g., does the extra $\sqrt{H}/\mu_0$ factor change the regime in which RegORL is preferred)? A clearly labeled remark or appendix subsection would significantly strengthen transparency.

- **Distinguishability gap not empirically measured.** The exponential improvement from $L_\infty^\ell$ to $L_{\mathcal{X}}$ distinguishability is the core theoretical claim, but it is not measured empirically in domains like Cheese or Mini-hall where performance differences are reported. A plot of both distinguishability values vs. corridor length (or another problem parameter) would directly validate Theorem 1's significance for the practical domains tested.

- **Computational cost of membership testing unanalyzed.** While $|\mathcal{X}_{i,j}| = \mathcal{O}((AOR)^j)$ for fixed $j$, the cost of testing whether a trace belongs to a language in $C_k^\ell(\mathcal{G}_i)$ (which involves ordered subsequence matching) is non-trivial. The paper does not analyze this per-trace cost, which is needed to fully characterize algorithmic runtime and to interpret the empirical timing comparisons.

- **$L_\infty^\circ$ vs. $L_\infty^p$ notation shift.** The paper defines $L_\infty^\circ$ in Section 2 (line 116) using prefix probabilities, and then switches to $L_\infty^p$ in Section 3 (lines 142–146) without explicitly reconciling the two. They appear to be the same object, but the relationship should be made explicit.

### Tiny

- **T-maze vs. T-maze(c) distinction not highlighted in experiments.** Table 1 uses "T-maze(c)" (Bakker 2001 original) while Figure 2 uses "T-maze" (the paper's variant from Example 1). These are different parameterizations; the distinction should be noted in the experimental section more clearly for readers who do not cross-reference the appendix.

- **The $\mathcal{O}(\cdot)$ hiding poly-logarithmic terms is non-standard** (line 56) and complicates reading the sample complexity bounds. This should be flagged more prominently where the bounds are stated.

---

## Nice-to-Haves

- **Adaptive or data-driven selection of $\mathcal{X}_{i,j}$.** An incremental scheme that starts with $j=1$ and increases until distinguishability is empirically sufficient (or a held-out validation is maximized) would greatly improve practical usability without changing the theoretical guarantees.

- **Ablation on $\mathcal{X}_{i,j}$ parameters.** Systematically varying $i$ and $j$ in one or two domains would show the trade-off between expressiveness and cost, motivating why $\mathcal{X}_{3,1}$ is a good default.

- **Behavior policy coverage study.** Varying the coverage of the offline dataset (e.g., random vs. near-optimal behavior policy) would empirically test the sensitivity to the concentrability assumption $C_R^* < \infty$.

- **Learned automata visualizations.** Showing the RDP automata learned by the language metric vs. FlexFringe in Mini-hall would clarify whether the improvement comes from correctly capturing long-term dependencies.

---

## Removed Points

*These points are flagged as removed; treat them with caution.*

- **"CMS runtime contradicts its motivation" (negative framing from Reviews 2 & 3).** The paper explicitly explains in Section 5 (line 253) that CMS's memory savings do not translate to time savings because "the statistical test still has to iterate over all suffixes, which is exponential in $H$." The paper is honest that CMS addresses memory, not runtime. The "contradiction" raised by the reviewers does not hold against the paper as written. The CMS contribution is correctly framed as a memory improvement. *(Minor: the explanation appears late and could be moved earlier to manage reader expectations — this is a presentation note, not a substantive flaw.)*

- **"Distinction between model learning and RL is unclear" (Review 2).** The paper is clear that ADACT-H is the model-learning component and that an optimal policy is subsequently derived by solving the learned MDP (via RegORL in Appendix A). Framing the contribution as "offline RL" is standard in the literature where learning the model and planning over it are jointly evaluated. This is not a substantive weakness.

- **"FlexFringe comparison is unfair" (Review 1).** The paper explicitly acknowledges that FlexFringe uses heuristics and can produce cyclic RDPs (line 224). The comparison is presented as a practical benchmark, not a theoretical comparison. The asymmetry (FlexFringe lacks guarantees) favors FlexFringe in easy domains and is declared upfront. This is a reasonable practical comparison.

- **"Limited domain complexity / noisy observations" (Review 2).** The paper studies episodic acyclic RDPs as a formal class; this is the paper's explicitly stated scope. Criticizing the absence of continuous observation spaces or noisy domains falls outside this scope.

- **"Request for regret/sub-optimality gap vs. raw reward in Table 1" (Review 2).** Since the optimal policy reward is known for these benchmarks and reported (max reward = 1, or 4 for T-maze(c)), the raw reward already encodes near-optimality. Requiring a formal regret number is not a standard expectation for this type of empirical evaluation.

- **Concentrability verification criticism (Review 3).** The assumption $C_R^* < \infty$ is standard in offline RL across MDPs and RDPs. Criticizing its non-verification in experiments is not specific to this paper.

---

## Novel Insights

The paper's most genuinely novel insight is the observation that standard suffix-probability metrics ($L_\infty^p$, $L_1^p$) are "point tests" that ignore the internal algebraic structure of the trace alphabet. By lifting from individual traces to *languages* as the unit of comparison, and grounding the hierarchy in the dot-depth classification of star-free regular languages, the paper recovers nearly $L_1$-strength distinguishability at $L_\infty$-level sample cost in domains where the relevant pattern is a subsequence occurrence rather than an exact suffix match. This insight — that the combinatorial structure of the trace (e.g., "does a North action appear before a rewarded observation?") is the informative signal, not the exact suffix — is the key bridge between automata-theoretic complexity and RL sample complexity. The $C_k^\ell$ operator is a clean formalism for this idea. A related implication, noted informally but not fully developed, is that the choice of language family is essentially a form of inductive bias: selecting $\mathcal{X}_{i,j}$ encodes prior knowledge about which pattern types are relevant, and the hierarchy provides a spectrum from bias-heavy (small $j$) to bias-free (large $j$). This framing may be useful beyond RDPs.

---

## Suggestions

1. **Add ADACT-H (plain $L_\infty^p$) as a baseline in Table 1.** This is the direct predecessor and the most natural comparison point for both CMS and the language metric. Without it, the incremental contribution over prior work is not empirically visible.

2. **Generate learning curves (return vs. $K$ episodes) for at least the T-maze and one other domain.** Plot language metric, CMS, and ADACT-H as $K$ varies. This directly validates the core sample efficiency claim.

3. **Add a dedicated remark (or appendix section) on the error in Cipollone et al. (2023).** Describe the nature of the mistake, the magnitude of the correction ($\sqrt{H}/\mu_0$), and whether it changes any qualitative conclusion of RegORL. This is important for the community.

4. **Discuss $d_m^*$ in the T-maze explicitly.** Either show analytically that $d_m^*$ is $\Omega(1/\text{poly}(H))$ in the T-maze under the behavior policy used, or acknowledge that the exponential gain claim is conditional on $d_m^*$ being well-behaved. Currently the claim of "exponential gain" is incomplete.

5. **Provide practical guidance on selecting $\mathcal{X}_{i,j}$.** Even a simple heuristic such as "start with $\mathcal{X}_{3,1}$ and increase $j$ if the learned automaton has too many states" would make the method significantly more accessible, and could be tested empirically.

6. **Explain the CMS underperformance in Cheese.** Investigate whether hash collisions at $H=6$ are large enough to distort the statistical test, and report CMS's collision rate or approximation error in this domain.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
