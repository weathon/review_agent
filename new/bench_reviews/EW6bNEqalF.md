Now I have enough information to write the final review. Let me synthesize everything.

## Summary

This paper introduces two techniques for improving offline RL in Regular Decision Processes (RDPs): a language-theoretic metric $L_X$ parameterized by a hierarchy of language families $\mathcal{X}_{i,j}$ that generalizes existing $L_1$ and $L_\infty$ metrics, and a Count-Min-Sketch (CMS) data structure for reducing memory requirements. The language metric provably yields exponential improvements in state distinguishability over prior work (Theorem 1), and PAC sample complexity bounds are derived for both approaches (Theorems 2–3). The paper also identifies and corrects a bug in the prior RegORL algorithm's analysis.

## Strengths

- **The language metric $L_X$ is a genuine and elegant conceptual contribution.** The pseudometric parameterized by a set of languages, placing $L_1$ and $L_\infty$ as special cases (Definition 2, Section 4.1), provides a principled and well-motivated framework. The two-dimensional hierarchy $\mathcal{X}_{i,j}$ constructed via the $C_k^\ell$ operator (Definition 1) cleanly interpolates between coarse-and-cheap and fine-and-expensive metrics.

- **Theorem 1 provides a concrete exponential separation.** It proves that $L_\infty^p$-distinguishability is $\mathcal{O}(2^{-N})$ while $L_{\mathcal{X}_{2,1}}$-distinguishability is $\Omega(1)$ in the T-maze domain, directly justifying the language metric's advantage. This is not just asymptotic hand-waving—it is a specific, verified theoretical gain.

- **Honest correction of an error in prior work.** The paper identifies that an additional $\sqrt{H}/\mu_0$ factor applies to the RegORL bounds (Section 4.2), which strengthens the paper's credibility and the motivation for the new approach.

- **The PAC bounds in Theorem 3 are meaningful.** When $j$ is a small constant, $\log|\mathcal{X}_{i,j}| = \tilde{\mathcal{O}}(1)$, which represents a genuine improvement over the exponential dependence of prior bounds via the $1/\mu_0$ term.

- **Empirical validation is informative.** Figure 2 clearly demonstrates that the language metric approach scales linearly with corridor length while CMS scales exponentially, validating the theoretical prediction.

## Weaknesses

### Fatal
None.

### Major

- **Assumption 1 ($\mu_0 > 0$ for $\mathcal{X}_{i,j}$) is an unverifiable assumption about the unknown RDP.** The algorithm's correctness requires the practitioner to choose $\mathcal{X}_{i,j}$ such that the $L_{\mathcal{X}_{i,j}}$-distinguishability of the *unknown* RDP is positive. Choosing too coarse a language family can cause incorrect state merges, producing a non-minimal RDP and a policy with no guarantees. There is no diagnostic or adaptive procedure for selecting $\mathcal{X}_{i,j}$ from data, and the paper provides no general characterization of when $\mathcal{X}_{i,j}$-distinguishability is positive beyond the T-maze example. While unverifiable assumptions about the unknown environment are common in PAC-RL (e.g., concentrability), here the assumption directly governs which language family to input—without guidance, a practitioner cannot use the algorithm with confidence. An adaptive schedule (e.g., starting with $\mathcal{X}_{1,1}$ and increasing complexity) with theoretical backing would substantially strengthen the paper.

- **RegORL (Cipollone et al., 2023) is absent from the experimental comparison.** RegORL is the direct prior state-of-the-art that this paper builds on and whose analysis it corrects. Comparing only against FlexFringe—which uses different heuristics and can learn cyclic automata—makes it difficult to assess the empirical improvement from the language metric over the $L_\infty^p$-based approach it replaces. Since the paper already has a re-implementation of ADACT-H (which RegORL uses), including RegORL as a baseline should be straightforward.

- **The CMS contribution is presented alongside the language metric as jointly solving the paper's stated motivating question, but only the language metric delivers computational tractability.** The introduction asks for "sample efficiency" *and* "computationally tractable implementation" (Section 1). CMS addresses only memory; the statistical test for CMS still iterates over all suffixes in $(\mathcal{AO}/\mathcal{R})^{H-t}$, exponential in $H$, which the paper acknowledges in Section 5. The two techniques cannot be combined to achieve both goals simultaneously. The abstract and contributions section should make it unambiguous that CMS provides memory efficiency, not computational efficiency, and that only the language metric resolves the exponential-computation bottleneck.

### Minor

- **The $d_m^*$ term in Theorem 3 can still have exponential dependence on $H$.** The paper notes this ("The constant $1/d_m^*$ depends exponentially on $H$ if there exists an RDP state that is very hard to reach," Section 4.2), but the qualifier is buried. The "exponential gain" claim in the abstract and title is conditional on *both* $\mu_0 = \Omega(1)$ and $d_m^*$ not being exponentially small—stating these conditions more prominently would prevent over-reading of the headline result.

- **The experimental evaluation does not vary dataset size $K$ to test sample efficiency claims.** All experiments fix $K=100$, and Figure 2 varies corridor length. Since the paper's core theoretical contribution is about sample complexity, varying $K$ and measuring recovered reward/policy quality would directly validate the sample efficiency claims.

- **The estimator notation in Section 4.1 is ambiguous.** The expression $\hat{p}_1 := \sum_{e \in \mathcal{Z}_1} \mathbb{I}(e \in \mathcal{X}_{i,j}) / |\mathcal{Z}_1|$ treats $\mathcal{X}_{i,j}$ as a single language, but the statistical test in Theorem 3 takes a maximum over languages in $\mathcal{X}$. The per-language estimator $\hat{p}_1(X)$ for each $X \in \mathcal{X}_{i,j}$ should be defined explicitly.

### Trivial
None.

## Nice-to-Haves

- An adaptive or diagnostic procedure for selecting $\mathcal{X}_{i,j}$ from data (even a simple incremental schedule with theoretical guarantees) would transform the algorithm from requiring oracle knowledge to being self-configuring.
- Experiments on RDPs where even $\mathcal{X}_{3,1}$-distinguishability is small, to illustrate the algorithm's limitations and validate that the choice of $(i,j)$ matters in practice.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"What happens when Assumption 1 is violated"—over-merge/under-merge analysis.** This is essentially the same concern as the Assumption 1 unverifiability point already captured above. Analyzing failure modes in detail goes beyond the paper's scope and would be a separate research contribution. KEPT the core unverifiability concern; REMOVED the demand for a full failure-mode analysis.

- **"The $\mu_0$ phrasing as 'maximum $\mu_0$' is confusing in the distinguishability definition."** This is a minor notation clarity concern that doesn't affect correctness—the definition is logically sound ("maximum $\mu_0$ such that the inequality holds for all pairs" is equivalent to "minimum over pairs of the distance"). Not substantive enough to list as a weakness.

- **"The nature of the mistake in Cipollone et al. (2023) is not described."** The paper states that it found an additional $\sqrt{H}/\mu_0$ factor; a detailed proof audit of a specific bug in another paper is not the role of this paper. The correction is honestly reported.

- **Missing confidence intervals/variance in some experimental results.** This is a minor presentation issue—some entries in Table 1 do report standard deviations. Requesting more is a nice-to-have, not a weakness.

- **Request for experiments on $\mathcal{X}_{3,1}$-distinguishable domains with small $\mu_0$.** This would indeed strengthen the paper but is beyond its stated scope. Moved to Nice-to-Haves.

- **"Not yet released" or "cannot be independently verified" concerns about benchmarks/models.** Per the hard rules, these are removed.

## Novel Insights

The language metric framework reveals an underappreciated structural insight: RDP state distinguishability is not an intrinsic property of the RDP itself, but depends on the metric used to assess it. By parameterizing the metric through language families, the paper shows that a judicious choice—grounded in the theory of star-free languages and the dot-depth hierarchy—can yield exponential improvements. This suggests that the right abstraction for RDP learning is not just "how different are the distributions" but "what kinds of patterns make them different," framing distinguishability as a resource that can be amplified by choosing the right representation rather than a fixed obstacle imposed by the environment.

## Suggestions

- Add a paragraph in Section 4 discussing what happens when Assumption 1 is violated (incorrect state merges) and propose at least a heuristic adaptive procedure for selecting $\mathcal{X}_{i,j}$ from data.
- Add RegORL as an experimental baseline using the corrected bounds the paper identifies.
- Restate the paper's motivating question to make it clear that CMS and the language metric address different bottlenecks—memory vs. computation—rather than jointly solving a single problem.
- Clarify the estimator notation in Section 4.1 to define per-language estimates $\hat{p}_1(X)$ for each $X \in \mathcal{X}_{i,j}$.

## Calibration Summary

| Anchor Paper | Avg Score | Comparison |
|---|---|---|
| POMDP distinguishable (1hsVvgW0rU) | 6.00 | Similar unverifiable environment assumption concern; this paper has a more novel theoretical contribution (language hierarchy) but weaker experimental evaluation |
| Confounded POMDP PG (8BAkNCqpGW) | 8.00 | Stronger theoretical novelty and completeness; this paper has a genuine conceptual contribution but more practical gaps |
| Laplacian RL (7xf50qWFGP) | 4.50 | Had unverifiable reachability assumption as a key weakness; this paper's assumption is more standard in PAC-RL and the contribution is more substantive |
| EIQL (C9BA0T3xhq) | 2.00 | Poor theoretical grounding; this paper is far beyond this level |
| Low-scoring (fvTaoyH96Z) | 2.33 | Accumulated unverifiable assumptions with weak theory; this paper has a solid core contribution that fundamentally works |

The paper sits above the Laplacian RL anchor (4.5) because its core theoretical contribution is genuinely novel and well-proven, and the unverifiable assumption is standard for PAC-RL rather than fatal. It sits below the confounded POMDP anchor (8.0) because the experimental evaluation is less complete and the paper has notable framing and practical deployment issues. Compared to the POMDP-distinguishable paper (6.0), which also had unverifiable assumptions about the environment but was accepted, this paper has a more creative theoretical framework but weaker experiments. I place it at approximately the same level, slightly above given the stronger theoretical contribution.

## Score and Decision

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>