## Summary

This paper introduces *Markovian compression operators* for distributed optimization, specifically two sparsification schemes—**BanLast** and **KAWASAKI**—whose coordinate-selection probabilities evolve as Markov chains depending on previous iterations. The authors integrate these compressors into vanilla QSGD (**MQSGD**, Algorithm 1) and a multi-momentum accelerated variant (**AMQSGD**, Algorithm 2), proving convergence for MQSGD under non-convex and Polyak-Łojasiewicz (PL) assumptions (Theorem 2) and for AMQSGD under strong convexity (Theorem 3). Empirical evaluations on MNIST logistic regression and ResNet-18/CIFAR-10 demonstrate that KAWASAKI can outperform standard random sparsification.

## Strengths

- **Novel compressor designs with rigorous Markov-chain analysis.** BanLast and KAWASAKI are genuinely new instantiations of history-dependent stochastic sparsification. Theorem 1 proves they are ergodic with uniform stationary distributions and provides explicit mixing parameters (e.g., $\rho = \sqrt{1 - \left( \frac{C_{d-2Km}^m}{(C_{d-Km}^m)^2} \right)^K}$ for BanLast), which is essential for applying the convergence theory.
- **Non-trivial analysis of non-i.i.d. compression.** The “stepping back” proof strategy (Section 2.2) that bounds the Markovian bias via mixing time is technically sound and appropriately imports techniques from the Markovian SGD literature. The resulting polynomial dependence on mixing time $\tau$ compares favorably to prior works with exponential-in-$\tau$ bounds (Section 2.4).
- **Honest theoretical discussion.** Section 2.4 contains an unusually frank discussion of why the theory yields a $d^2/m^2$ penalty (vs. $d/m$ for unbiased compressors) and explicitly notes the tension between larger $K$ (better empirical performance) and larger $\tau$ (worse theoretical constants).
- **Proper statistical evidence in Table 1.** On ResNet-18/CIFAR-10, Table 1 reports means $\pm$ standard deviations over 5 runs, showing KAWASAKI achieves roughly half the training loss (0.0305 vs. 0.0743) and gradient norm (0.745 vs. 1.403) of Rand-5%, along with higher test accuracy (89.05% vs. 87.9%). This constitutes credible empirical evidence for the compressor designs.

## Weaknesses

### Fatal
None.

### Major
- **The central acceleration claim is unsubstantiated.** The abstract states: “we prove convergence results for our algorithms … and show theoretically that the accelerated method converges faster than the basic version.” This claim is not supported by the theorems. Theorem 2 (MQSGD) covers only non-convex and PL cases; Theorem 3 (AMQSGD) covers only strong convexity. No convergence bound for the basic method is given under strong convexity, so no direct, rigorous comparison is possible. Even using the PL bound from Corollary 1 as a proxy, the accelerated complexity in Corollary 2 scales as $(L/\mu)^{2/3}\tau^{4/3}$ versus the basic $(L/\mu)\tau$; the accelerated rate is actually *worse* whenever $\tau > L/\mu$. Because the paper presents no head-to-head regime where acceleration is unambiguously and rigorously proven, the headline theoretical claim is invalid.
- **“Best runs” cherry-picking invalidates key visual evidence.** Figure 1 and Figure 2 explicitly state that “best runs are selected” and “Best runs for each method are displayed.” Selecting best runs after hyperparameter tuning is a form of cherry-picking that invalidates the statistical basis for claiming superiority from trajectory plots. While Table 1 mitigates this with proper statistics, the main visual evidence in Figures 1–2 is unreliable. Furthermore, there is no experiment directly comparing MQSGD against AMQSGD to validate the acceleration claim even empirically.

### Minor
- **Example 1 is a transient, contrived motivating scenario.** Example 1 argues that BanLast yields a 3× speedup over Rand for a static one-hot gradient. Because BanLast’s stationary distribution is uniform, its asymptotic per-iteration expected progress equals Rand; the apparent speedup is purely an initialization artifact that disappears once the chain mixes. Using this transient scenario to motivate the algorithm is conceptually misleading, though it does not invalidate the broader empirical results.
- **Theorem 3 likely contains a typographical error.** The left-hand side reads $F_{\tau+1}$ while the right-hand side depends on the total horizon $T$, suggesting the bound should likely be $F_T$ or similar. This is a presentation issue but creates confusion about what the theorem actually states.

### Trivial
None.

## Nice-to-Have
- A direct empirical comparison of MQSGD vs. AMQSGD on logistic regression with matched hyperparameters and mean curves over multiple seeds, to test whether acceleration actually materializes in practice.
- An analysis of time-averaged variance or non-stationary gradient arguments explaining why the methods work well despite the pessimistic $d^2/m^2$ bounds.
- Empirical coordinate-selection frequency plots to verify that BanLast/KAWASAKI achieve more uniform historical coverage than Rand.

## Removed Points
These points are flagged to be removed, treat them with caution.
- **Novelty mischaracterization regarding PermK and EF21.** The criticism that “history-dependent compression already exists” conflates deterministic cycling (PermK) and error-feedback state (EF21) with *Markovian stochasticity of the compressor itself*. The paper’s specific claim is about compressors whose coordinate-selection distribution evolves as a Markov chain, which is distinct from deterministic statefulness or separate error compensation. This criticism misunderstands the scope of the novelty claim.
- **Missing appendix material.** Per instructions, proofs deferred to the appendix are not weaknesses; the original submission contains appendices that were stripped by the parser.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Remove or severely weaken the claim that the accelerated method is “theoretically shown to converge faster than the basic version” from the abstract and introduction, unless a basic-method bound is provided under the same assumptions as Theorem 3 (or a rigorous comparison is made explicit).
- Replace “best run” selections in all figures with mean $\pm$ standard deviation or confidence intervals over at least 5 independent runs, using a fixed hyperparameter selection protocol.
- Add a basic strongly convex convergence bound for MQSGD (Algorithm 1), or explicitly state that the PL bound serves as the proxy and acknowledge the $\tau$-dependent comparison.

## Score and Decision

**Calibration comparison:**
- **High anchors:** `jj5ZjZsWJe` (8.00, Spotlight) and `PpYy0dR3Qw` (7.50, Spotlight) feature rigorous theory with clear, substantiated acceleration or communication complexity claims and solid experiments. This paper falls well below them due to the unsubstantiated acceleration claim and cherry-picked figures.
- **Medium anchors:** `ZEZ0CPmoSI` (5.00, Accept poster) had mixed reviews with theoretical concerns but real contributions; `ogIFNo2bQw` (4.80, Reject) lacked convergence analysis and had limited novelty. This paper is comparable—its core compressor designs and basic convergence theory are real contributions, but the central acceleration overclaim and experimental cherry-picking are serious liabilities.
- **Low anchors:** `l2odw7OiNw` (2.50, Withdrawn) made an unsubstantiated acceleration claim based on loose bounds; `ER1VDuwWvB` (3.67, Reject) introduced a method with weak theoretical grounding. This paper is above these because Theorem 2 is sound, Table 1 provides proper statistical evidence, and the compressor designs are genuinely novel.

Relative to these anchors, the paper sits at the lower end of the medium band: it has real merit (novel compressors, non-trivial basic theory, honest discussion, credible results in Table 1), but the headline acceleration claim is not supported by the theorems, and the “best runs” protocol undermines the visual evidence. A score around the medium-low range is appropriate.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>