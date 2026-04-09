=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary

This paper studies metric $k$-clustering in the Rank-model (R-model), where only a noisy quadruplet oracle (answering "is pair (A,B) closer than pair (C,D)?") is available—no distance computations are permitted. The authors design randomized algorithms that construct an $O(1)$-Coreset+ of size $O(k \cdot \text{polylog}(n))$ using $O(nk \cdot \text{polylog}(n))$ queries for general metrics, improving to $O((n+k^2) \cdot \text{polylog}(n))$ for bounded doubling dimension. For $k$-median in doubling metrics, they further achieve $(1+\varepsilon)$-approximation with the same query complexity. This removes the distance-oracle dependency required by prior RM-model work.

## Strengths

- **Removes the distance-oracle requirement for $k$-median/means clustering.** Prior work (Galhotra et al. 2024; Raychaudhury et al. 2025) showed that no $o(n)$-approximation is possible for $k$-median/means using only quadruplet oracles *and required access to a distance oracle*. This paper overcomes that barrier by outputting $O(k \cdot \text{polylog}(n))$ centers with a mapping, backed by a lower bound (Appendix A) showing $2k-1$ centers are necessary—making the bicriteria output essentially optimal in structure.

- **The ALG-TESTER emulation of adversarial from probabilistic noise is a clever technical contribution.** By using kernel and guard sets to ensure vertices are well-separated from sample points before comparison, the tester's behavior can be treated as an adversarial oracle with $\mu=1$, enabling the use of ADVSORT. This reduction (Lemma B.5) is the linchpin of the approach and a non-trivial insight.

- **Near-optimal query complexity with a complete theoretical story.** The progression from general metrics (ALG-G) → doubling dimension (ALG-D, with lazy filtering and degenerate-graph coloring reducing queries) → refined approximation (ALG-DI) is well-structured, and the bounds match prior RM-model results up to polylog factors while eliminating all distance queries.

## Weaknesses

### Major:

- **The experimental evaluation fundamentally misaligns with the R-model premise.** Section 4.1 explicitly states: *"After constructing the weighted coreset, we assume access to a distance oracle, as in the RM-model... We then apply k-means++ algorithm on the weighted coreset to obtain the final set of k centers, invoking the distance oracle when necessary."* The paper's core theoretical contribution is enabling clustering *without* distance oracles, yet the experiments evaluate a pipeline that requires them for the final step. The Coreset+ mapping cost $\sum_v d(v, M(v))$—which IS the quantity the theory bounds and does NOT require distance access—is never reported. This makes it impossible to assess whether the R-model construction alone produces a practically useful output.

- **No query-count reporting in experiments.** For an oracle-based paper where query complexity is the primary resource, the absence of any plot or table showing actual quadruplet query consumption versus $n$, $k$, or noise level $\phi$ is a critical gap. Figures 3 and 4 show clustering cost but not the oracle cost to achieve it. Without this, the practical efficiency of the method cannot be evaluated and the theoretical $O(nk \cdot \text{polylog}(n))$ bound cannot be empirically verified.

- **The baseline is trivially weak.** The baseline "assumes every response is correct" (Section 4.1), essentially comparing a noise-robust algorithm against one that ignores noise entirely. A more informative baseline would repeat each quadruplet query $O(\log n)$ times with majority vote before running a standard procedure. This would isolate whether the performance gain comes from the novel kernel/guard/ALG-TESTER structure or simply from standard noise-reduction via repetition.

### Minor:

- **Persistent noise assumption may not hold for practical oracles.** The model assumes that repeated queries to the same edge pair always return the same answer (persistence). LLMs and embedding services—cited as motivating examples—often produce non-deterministic outputs across repeated calls. The paper does not discuss how non-persistence affects correctness or how to enforce persistence (e.g., caching).

- **Small-scale experimental datasets.** The real datasets are downsampled to 2,000 points via Meyerson sampling, and the synthetic dataset has $10^4$ points. For a method whose primary motivation is scalability when distances are expensive, this scale is insufficient to stress-test the approach or demonstrate the coreset's compression benefit at realistic sizes.

- **LLM/embedding-based oracle validation is absent.** The introduction motivates the work with LLMs and embedding services, but all experiments use simulated noise on ground-truth Euclidean distances. No experiment validates the quadruplet oracle abstraction against a real learned model or human annotator.

### Trivial:

- The exact constants hidden in $O(\cdot)$ and the polylog factors are not empirically analyzed. This is standard for theory-first papers but limits immediate practical adoption.

## Nice-to-Haves

- **Evaluate the Coreset+ mapping cost directly** (i.e., $\sum_v d(v, M(v))$) without invoking a distance oracle for $k$-means++. This is the quantity the theory actually bounds and would fully validate the R-model contribution.

- **Plot clustering cost vs. number of quadruplet queries** to empirically demonstrate the query-efficiency tradeoff.

- **Test with a real quadruplet oracle** (e.g., CLIP embedding similarities or an LLM API) even at small scale, to validate the abstraction.

- **Analyze sensitivity near the $\phi = 1/4$ boundary** to show how the algorithm degrades as noise approaches the theoretical limit.

- **Propose a mechanism to reduce $O(k \cdot \text{polylog}(n))$ centers to exactly $k$** using only quadruplet comparisons, closing the gap with standard $k$-clustering outputs.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Bicriteria output as a fundamental flaw.** The paper is extremely explicit (Introduction, paragraph beginning "We emphasize that our algorithms do not directly output exactly $k$ centers") and provides a lower bound (Appendix A, Theorem A.1: $2k-1$ centers necessary) justifying this. This is not a hidden weakness but a known information-theoretic barrier.

- **Non-metric space evaluation.** The paper explicitly scopes to metric spaces. Generalizing to non-metric graphs is listed as future work. Criticizing its absence is scope creep.

- **Formatting artifacts in equations.** These are parser errors per instructions.

- **Missing related works.** Cannot verify existence per rules.

- **Reproducibility concerns about undisclosed hyperparameters.** Per rules, these are removed.

- **Demand for confidence intervals or multiple runs on large-scale benchmarks.** Single-run evaluation is the norm for this type of theoretical-empirical paper.

## Novel Insights

The key structural insight—kernel and guard sets enable the emulation of an adversarial oracle from a probabilistic one—could have broader applicability beyond clustering. Any problem requiring approximate sorting or selection under probabilistic noise where a "safe zone" can be identified might benefit from this filtering-then-testing paradigm. The lazy filtering strategy in ALG-D (checking proximity only when needed during traversal rather than globally) is also a noteworthy design pattern for reducing oracle queries in structured metric spaces.

## Suggestions

- **Report Coreset+ mapping cost in experiments.** Compute and plot $\sum_v d(v, M(v))$ normalized by OPT, which directly tests the theoretical guarantee without requiring a distance oracle for final center selection.

- **Add a query-count figure.** Plot total quadruplet queries vs. $n$ (on the synthetic dataset with varying $n$) and vs. $k$ to empirically validate the scaling behavior.

- **Strengthen the baseline.** Implement a majority-vote baseline that repeats each quadruplet query $\lceil c \log n \rceil$ times and takes the majority, then runs the generic algorithm. This controls for the trivial effect of noise reduction.

- **Add a brief discussion of the persistence assumption.** Note that in practice, persistence can be enforced by caching oracle responses, and discuss any implications if caching is imperfect.

## Axis Assessments

- **Novelty:** High. First algorithm for $k$-median/means clustering in the pure R-model; the adversarial-from-probabilistic emulation is a non-trivial technical contribution.

- **Technical soundness:** Good. Proofs are rigorous; the algorithmic framework is well-constructed with clear lemmas supporting each component.

- **Empirical support:** Weak. The evaluation does not test the R-model output directly, lacks query-count reporting, and uses an overly weak baseline. The experiments validate that the Coreset+ is useful as a *preprocessing step* for distance-based clustering, but not that it stands alone as an R-model output.

- **Significance:** High theoretical significance for removing the distance-oracle barrier; practical significance is promising but currently unvalidated due to the experimental gaps.

- **Clarity:** Good. The paper is well-organized with a clear technical overview; the progression from general to doubling to refined approximation is logical. Some definitions suffer from parser artifacts but the prose is clear.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0, 6.0]
Average score: 7.2
Binary outcome: Accept
