Now I have enough calibration data. Let me compose the final review.

## Summary

VISTA is a modular divide-and-conquer framework for causal structure learning that (1) decomposes the global DAG problem into Markov Blanket subgraphs, (2) aggregates local edge-level evidence via an exponential-decay weighted voting mechanism, and (3) enforces acyclicity through a Feedback Arc Set heuristic. The framework is model-agnostic with respect to base learners and MB identification, supports full parallelization, and comes with finite-sample error bounds and asymptotic consistency guarantees.

## Strengths

- **Clean, practical modular design.** The separation into MB identification, local learning, weighted voting, and FAS post-processing is intuitive and easily implementable. The pseudocode (Figure 2) and overview diagram (Figure 3) make the pipeline transparent.

- **Substantial and credible runtime gains.** Table 3 shows 3–11x speedups across base learners as n grows (e.g., NOTEARS: 12516s→2137s at n=300; GraN-DAG: 25206s→2336s). These gains directly result from the parallel decomposition into small neighborhoods and are a genuine contribution to scalable causal discovery.

- **Consistent F1 improvements under weighted voting.** On synthetic data (Table 1, n=100, h=5), VISTA-WV improves F1 over standalone baselines in nearly all settings: NOTEARS 0.76→0.79, GOLEM 0.35→0.60, DAG-GNN 0.35→0.59, GraN-DAG 0.06→0.17, SCORE 0.14→0.31 on ER5. This holds across both linear and nonlinear SEMs and different graph families.

- **Responsible fixed hyperparameter strategy.** Using λ=0.5 and t=0.7 across all experiments without per-dataset tuning, and providing full precision-recall curves (Figure 4) rather than cherry-picking operating points, avoids overfitting concerns and demonstrates the framework's stability.

- **Honest discussion of limitations.** The conclusion explicitly acknowledges latent confounding from variable subsets and potential pruning of correct weak edges by FAS.

## Weaknesses

### Major:

- **Core theoretical guarantees are misaligned with the actual VISTA pipeline.** Theorems 3.2, 3.4, and 3.5 assume votes from different local subgraphs are *independent* and that each edge receives m = C log n subgraph votes with fixed success probabilities p > t and q < t. In reality, (a) overlapping MB subgraphs trained on the same data produce strongly correlated votes, not i.i.d. Bernoulli trials; (b) the number of subgraphs containing a given edge is determined by graph degree structure and is bounded for sparse graphs—it does not scale with n; (c) the success probabilities p, q likely deteriorate as graph size grows. The paper acknowledges the independence assumption is "idealized" (around line 302) but then presents Theorem 3.5 as showing VISTA is "asymptotically consistent" under "mild" conditions, which is misleading given these conditions do not hold for the MB-based decomposition as instantiated. The theoretical contribution, while correct as an abstract voting analysis, does not legitimately establish guarantees for the actual algorithm, yet it is framed as a key contribution of the paper.

- **No theoretical treatment of Markov Blanket estimation error.** Proposition 3.1 guarantees edge coverage only under *correct* MB identification, and the voting theory operates on the output of local subgraphs. In practice, MBs are estimated from finite samples, and missing an endpoint from an estimated MB irrevocably loses the corresponding edge. The theory never propagates MB estimation error into the probability bounds, despite this being the most fragile upstream component—particularly in the high-dimensional regime the paper targets. Figure 1 shows MB accuracy is "relatively stable," but this is empirical, not theoretical.

- **Limited real-world evaluation for a method targeting large-scale settings.** Only the Sachs network (11 nodes, 17 edges) is used for real data validation. This is too small to test the framework's scalability advantage and is a well-studied benchmark with known limitations. Without evaluation on larger real-world networks (e.g., gene regulatory networks with 50+ nodes), the practical utility of VISTA beyond synthetic benchmarks remains unsubstantiated.

### Minor:

- **NV variant produces catastrophically high FDR.** In Table 1, VISTA-NV achieves FDR of 0.84–0.95 and SHD values orders of magnitude worse than baselines (e.g., NOTEARS NV: SHD=3172 vs. baseline SHD=209). While NV serves as a conceptual stepping stone, this shows the decomposition alone is harmful and the framework's success depends almost entirely on the weighted voting + thresholding—meaning the method's robustness is sensitive to the choice of λ and t. The paper could more clearly acknowledge that the aggregation (not the decomposition) is doing the heavy lifting.

- **"Model-agnostic" claim is overstated for CPDAG-outputting learners.** The method requires base learners to output directed edges; undirected adjacencies are treated as "no directional vote" (Section 3). Many widely-used methods (PC, GES, FCI) output CPDAGs or PAGs with undirected edges, and treating these as non-evidence could significantly weaken performance. The model-agnostic claim should be qualified to apply to methods that produce directed local output.

- **DCILP comparison relegated to appendix.** DCILP (Dong et al., 2024) is the most directly competing divide-and-conquer framework, yet the comparison appears only in Appendix F.2 with no summary in the main text. Given that VISTA's primary motivation is improving over such modular approaches, this comparison deserves prominence.

### Trivial:

- The GraN-DAG baseline performs very poorly across all variants (F1 ≤ 0.18), suggesting it may not be well-suited for these experimental settings. This is not a paper flaw per se, but including it as evidence of VISTA's generality is weak.

## Nice-to-Haves

- Extend the theoretical analysis to handle weakly dependent votes (e.g., via concentration inequalities for dependent random variables or martingale-based arguments), which would make the theory directly applicable to VISTA's actual operation.

- Provide a data-driven or stability-selection-based procedure for choosing (λ, t) rather than relying on a single fixed operating point, to improve practical applicability.

- Evaluate on larger real-world benchmarks (e.g., gene regulatory networks) and at larger synthetic scales (n > 300) to substantiate the large-scale scalability claim.

- Report per-stage error decomposition (edges lost at each pipeline stage: MB estimation → local learning → voting → FAS → thresholding) to shed light on failure modes.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Critic claimed the independence assumption makes the theorems "not valid for VISTA."** The theorems *are* correct mathematical statements about a voting rule; the issue is that the assumptions don't match the algorithm's instantiation, not that the proofs themselves are wrong. The distinction matters: the theory provides qualitative insight but not quantitative guarantees for the pipeline.

- **Critic claimed the framework's success "depends critically on the weighted voting and threshold" as a weakness.** This is by design—WV is the proposed method. The NV variant exists for conceptual clarity, and the paper positions WV as the main contribution. Criticizing the method for relying on its own mechanism is circular.

- **Demands for experiments at n > 300 or n > 1000.** While larger-scale experiments would be nice, n=300 with h=5 already represents 1500 potential edges and demonstrates clear scalability advantages. Requesting n=1000+ is a generic scaling demand that doesn't invalidate the existing evidence.

- **Critic claimed p and q "likely deteriorate" as n grows.** This is speculative without evidence. The paper's empirical results don't show systematic degradation, and MB-based decomposition specifically aims to keep local problems small as n grows.

## Novel Insights

The most interesting observation across the reviews is that VISTA's primary value may be more architectural than algorithmic: the framework demonstrates that a simple, theoretically-motivated edge-level aggregation can serve as a universal "stabilizer" for diverse base learners—reducing their characteristic failure modes (high FDR for score-based methods, low recall for others) without requiring any internal modification. The real theoretical gap is not in the voting analysis per se, but in the absence of an end-to-end analysis that connects MB estimation quality, local subgraph confounding, and vote correlation into a unified error model. Such an analysis would likely reveal that VISTA's practical effectiveness stems from the *averaging* of correlated but biased votes (reducing variance) rather than from the independence-based concentration the current theory describes.

## Suggestions

- Reframe the theoretical contribution as providing *design principles* (the role of λ in controlling precision-recall, the logarithmic scaling requirement) rather than operational guarantees for the full pipeline, and add clear caveats that the independence and m-scaling assumptions are not satisfied by MB-based decomposition.

- Expand the DCILP comparison into the main text with a side-by-side table on accuracy, runtime, and memory, as this is the most informative point of comparison.

- Add at least one real-world experiment on a network with 30+ nodes to test the framework beyond the trivially small Sachs network.

## Score and Decision

**Calibration references:**
- DDCD (Z756zcjNcC): causal discovery with relaxed theoretical claims, limited real evaluation → scores 6,6,3,3, rejected
- Exact Distributed Structure-Learning (DUfwD5yiN4): distributed BN learning with theory-practice gap, limited baselines → scores 5,3,8,5, rejected  
- Node-based Multiple Graph Learning (YtGtIAYDV3): theoretical guarantees with overclaim, narrow scope → scores 3,5,3, rejected
- Versatile Causal Discovery (FhQSGhBlqv): tight theory-practice alignment, strong experiments → scores 8,6,8,8, accepted

VISTA sits between the weak-theory rejected papers and the strong-theory accepted papers. Its practical contributions (runtime, modularity, F1 gains) are genuine and significant, placing it above papers like YtGtIAYDV3 (score 3-4) that overclaim with limited empirical backing. However, the theory-reality gap is substantial and the real-world evaluation is extremely thin, placing it below papers with aligned theory and experiments (score 7-8). Compared to DUfwD5yiN4 (mean ~5.25, rejected), VISTA has stronger empirical results but a similar theory-practice disconnect. The paper makes a real contribution but the overclaim on the theoretical side is a significant weakness that undermines a core advertised contribution.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>