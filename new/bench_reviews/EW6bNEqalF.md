Now I have enough calibration data. Let me synthesize my final review.

**Calibration summary:**
- Papers on offline RL with PAC bounds (GnOLWS4Llt) got scores of 5,5,5 (accepted as poster)
- Papers on POMDP sample complexity (1hsVvgW0rU) got 6,6,6,6 (accepted poster)
- Papers on POMDP hardness theory (Q00CO1Tm6M) got 6,5,6,6 (rejected)
- Theoretical POMDP/representation papers with weak experiments (B5kAfAC7hO) got 5,5,6 (rejected)
- Papers on offline RL with general function approximation and good theory (JSS9rKHySk) got 8,8,6,6,6 (spotlight)

The paper under review is a theoretical contribution on offline RL in RDPs with PAC bounds and small-scale experiments. It has genuine theoretical novelty (language metric) but also some significant limitations. Let me assess where it falls.

Key assessment:
- The language metric is a genuine theoretical contribution, well-grounded in formal language theory
- The separation result (T-maze) is compelling
- The CMS contribution is theoretically fine but practically weak
- The "exponential gain" claim is overstated — it applies to a specific family of RDPs
- Missing RegORL baseline is a notable experimental gap
- The choice of $\mathcal{X}_{i,j}$ requires a priori knowledge (Assumption 1)
- The $1/d_m^*$ dependence means worst-case exponential behavior persists
- The step from model identification to policy quality is not explicit in main text

Compared to calibration papers: This is stronger than the rejected POMDP papers (B5kAfAC7hO at 5,5,6; Q00CO1Tm6M at 6,5,6,6) because it has a clearer and more novel theoretical contribution. It's probably comparable to or slightly above the offline RL with observation histories paper (GnOLWS4Llt at 5,5,5) which was accepted as poster — that paper also had theoretical contributions on sample complexity in non-Markov settings with limited experiments. I'd place this around a 6 — above the acceptance threshold but not a strong accept, given the overclaiming issues and experimental gaps.</think>Now I have sufficient calibration data. Let me write the final review.

## Summary

This paper proposes two techniques for improving offline reinforcement learning in Regular Decision Processes (RDPs): (1) a novel language metric $L_X$ grounded in formal language theory that replaces the $L_\infty^p$-distinguishability used in prior work, provably yielding better sample complexity (with an exponential improvement in specific domains like T-maze); and (2) a Count-Min-Sketch-based approach that reduces memory requirements. The authors provide PAC sample complexity bounds for both variants and validate them experimentally on five small benchmark domains.

## Strengths

- **Novel and well-motivated language metric.** The introduction of $L_X$ through the hierarchy $\mathcal{X}_{i,j}$ (inspired by the dot-depth hierarchy of star-free regular languages) is a genuinely creative contribution that bridges formal language theory and statistical learning in RDPs. The key insight — that evaluating probabilities over structured language families rather than individual strings can yield exponentially better distinguishability — is both novel and principled.

- **Formal exponential separation result.** Theorem 1 constructs a concrete family of RDPs (the T-maze) where $L_\infty^p$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$. This is a clean, concrete separation that justifies the theoretical motivation. The experimental results in Figure 2 corroborate this with clear linear vs. exponential scaling trends.

- **Transparent correction of prior work.** The paper identifies and corrects a mistake in Cipollone et al. (2023), noting an additional $\sqrt{H}/\mu_0$ factor. This is handled openly and strengthens the paper's reliability.

- **Clear theoretical results.** Theorems 2 and 3 provide explicit, clean PAC sample complexity bounds incorporating the new distinguishability parameter and language family size. The notation and proofs are rigorous.

## Weaknesses

### Major:

- **Overclaiming of "exponential gain" generality.** The abstract states that the gain "is showcased to be exponential in the episode length," and the introduction frames this as broadly overcoming prior exponential blow-ups. However, the exponential improvement is demonstrated only for the T-maze family (Theorem 1 and Example 4). The paper does not characterize the class of RDPs for which $L_X$-distinguishability is substantially larger than $L_\infty^p$-distinguishability, nor does it provide examples beyond T-maze. This makes the central generality claim ("in domains admitting low complexity languages") under-supported — it is a plausible qualitative claim but not formally characterized beyond one example family.

- **Assumption 1 requires prior knowledge of $\mathcal{X}_{i,j}$.** The PAC guarantee in Theorem 3 depends on $L_X$-distinguishability $\mu_0 > 0$ for a chosen language family $\mathcal{X}_{i,j}$, which is an input to the algorithm. The paper does not discuss how to select $i$ and $j$ in practice without knowledge of the RDP structure, nor the consequences of a poor choice. The experiments only test $\mathcal{X}_{3,1}$. The algorithm is not fully specified without this choice, which creates a gap between the theoretical improvement and practical realizability.

- **Residual exponential dependence on $H$ via $1/d_m^*$.** The bound in Theorem 3 still depends on $1/d_m^*$, which can be exponential in $H$ when some RDP states are very hard to reach under the optimal policy. The paper acknowledges this (Section 4.2: "The constant $1/d_m^*$ depends exponentially on $H$ if there exists an RDP state that is very hard to reach"), but this means that the "removal of exponential blow-up" in the headline claim is real only for RDPs where $1/d_m^*$ is small. The paper claims to "overcome the limitations of existing offline RL algorithms for RDPs" — this is true for the distinguishability parameter specifically, but the overall sample complexity can still be exponential in $H$ through $1/d_m^*$.

- **No comparison with RegORL (Cipollone et al., 2023).** The paper's entire motivation is improving upon RegORL, yet experiments only compare against FlexFringe. Without benchmarking against the directly competing algorithm whose theoretical guarantees are being improved, the practical impact of the proposed techniques cannot be properly assessed.

### Minor:

- **The bridge from RDP identification to policy optimality is not explicit in the main text.** The central task is offline RL (learning an $\varepsilon$-optimal policy), but Theorems 2–3 only guarantee recovery of the minimal RDP. The step from RDP identification to policy quality is deferred to RegORL in the appendix, making it harder for readers to assess the end-to-end offline RL guarantee.

- **The CMS contribution is theoretically clean but practically weak.** Figure 2 and Table 1 show that CMS scales exponentially in time and consistently underperforms the language-based approach. CMS times out on Mini-hall. The paper frames CMS as "alleviating memory requirements for long planning horizons," but since the statistical test still iterates over exponentially many suffixes for $L_\infty^p$, it does not actually address the computational bottleneck for large $H$.

- **No experiments varying dataset size.** All experiments use a fixed $K=100$ episodes. Since the main theoretical contribution is improved sample efficiency, experiments varying dataset size would more directly validate the theoretical claims about $\mu_0$ and $|\mathcal{X}|$ dependencies.

- **No experiments varying the hierarchy parameters $(i,j)$.** Only $\mathcal{X}_{3,1}$ is tested, leaving the utility of the full hierarchy and the sensitivity to parameter choice unevaluated.

### Trivial:

- The $C_k^\ell$ operator definition (Definition 1) is dense; a simpler worked example before Example 2 would improve readability. However, this is a presentation issue, not a substantive one.

## Nice-to-Haves

- Provide adaptive or principled guidance for choosing $\mathcal{X}_{i,j}$, even as a heuristic or cross-validation scheme.
- Include RegORL as a direct experimental baseline.
- Add experiments varying dataset size ($K$) to validate sample complexity claims.
- Characterize conditions on RDP structure under which $L_X$-distinguishability is substantially better than $L_\infty^p$-distinguishability beyond the T-maze family.
- Discuss the computational cost of evaluating the $L_X$ statistical test per comparison (membership testing in languages $X \in \mathcal{X}_{i,j}$).

## Removed Points

- **Criticisms of limited baselines beyond RegORL (e.g., deep RL, RNN-based methods).** The paper is about RDP learning with provable guarantees; deep RL methods operate in a fundamentally different paradigm without such guarantees and are not appropriate baselines.

- **Criticisms of "notation-heavy" presentation.** While the notation is dense, it is necessary for the formal language theory contribution. This is a style issue, not a substantive weakness.

- **Demands for experiments on larger/continuous domains.** The paper's scope is provably efficient offline RL in episodic RDPs, which are inherently finite-state; large-scale continuous domains are outside scope.

- **Criticisms that the FlexFringe comparison is "apples-to-oranges."** The paper transparently acknowledges the differences between the algorithms, and FlexFringe is the most relevant available baseline for learning probabilistic deterministic finite automata from data.

- **Requests for theoretical proofs of near-optimality for the hierarchy choice.** This would be a significant new contribution beyond the paper's scope.

## Novel Insights

The paper reveals that the gap between $L_1$-based and $L_\infty$-based distinguishability metrics for RDP state-merging algorithms is not merely quantitative but can be exponential in the episode length, and that this gap can be closed by replacing pointwise string-level comparisons with comparisons over structured language families drawn from the dot-depth hierarchy. This is a genuine insight at the intersection of formal language theory and statistical learning theory for RL.

## Suggestions

- Reframe the main claim from "exponential improvement in sample efficiency" to "exponential improvement in distinguishability for structured RDPs, leading to improved sample complexity when $1/d_m^*$ is manageable." This is more accurate and still compelling.
- Add a brief discussion in Section 4.2 about the end-to-end offline RL PAC guarantee, explicitly stating how the improved RDP identification bound translates to a policy optimality bound via RegORL.
- Include at least one experiment comparing against RegORL to close the loop on the claimed improvement.

## Score and Decision

**Calibration comparison:**
- Offline RL with Observation Histories (GnOLWS4Llt): scores 5,5,5, accepted as poster — theoretical + limited empirical, similar profile but less novel theoretical contribution
- Provable Representation for POMDPs (B5kAfAC7hO): scores 5,5,6, rejected — weak theoretical contribution with overclaimed practical impact
- Sample-Efficient POMDP Learning (1hsVvgW0rU): scores 6,6,6,6, accepted poster — solid theoretical contribution identifying learnable POMDP subclasses
- General Function Approximation in Offline RL (JSS9rKHySk): scores 8,8,6,6,6, spotlight — strong theoretical contribution with broad impact

This paper has a genuinely novel theoretical contribution (language metric for RDP state merging) that is well-grounded and yields a clean separation result. However, its claims are somewhat overstated (the exponential improvement is specific to structured examples, and $1/d_m^*$ can still dominate), the experiments lack the most direct baseline (RegORL), and the algorithm requires a priori knowledge of the language family. It is above the POMDP papers that were rejected (B5kAfAC7hO) because the theoretical contribution is more novel and better executed, and roughly comparable to (but slightly better than) the offline RL with observation histories paper (GnOLWS4Llt) that was accepted as poster, since this work has a clearer and more targeted theoretical advance.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>