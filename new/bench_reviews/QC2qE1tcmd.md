Now I have sufficient calibration information. Let me compose the final review.

## Summary

The paper proposes a unifying axiomatic framework that recasts simplicial message-passing as relational message-passing, enabling the extension of graph-theoretic oversquashing analyses (sensitivity bounds, curvature, depth effects, hidden dimension effects) to higher-order structures. The key technical object is the *influence graph* derived from an aggregated influence matrix over all relations, and the paper also proposes a practical rewiring heuristic that collapses the relational structure to a graph, applies existing graph rewiring, and adds new connections back as a new relation.

## Strengths

- **Addresses a genuine open problem.** The paper directly tackles research directions 2 and 9 from Papamakarios et al. (2024) on oversquashing and rewiring in TDL, filling a recognized gap in the literature.
- **Systematic and mathematically sound framework.** The mapping from simplicial complexes to relational structures (Section 2) is well-defined, and the influence-graph construction (Definition 3.1) correctly identifies the objects needed to extend graph-theoretic tools. Lemma 3.2, Proposition 3.4, and Theorem 3.5 are technically correct and follow rigorous proof structures.
- **Covers multiple axes of oversquashing analysis.** The paper analyzes sensitivity (Lemma 3.2), local geometry via extended Forman curvature (Definition 3.3, Proposition 3.4), depth effects (Theorem 3.5), and hidden dimensions (Section 3.4), providing a comprehensive suite of theoretical tools for the relational setting.
- **Practical rewiring heuristic.** Algorithm 1 is simple, modular, and broadly applicable—it allows any graph rewiring method to be used as a plug-in for relational structures, which lowers the barrier to adoption.
- **Comprehensive experimental suite.** The evaluation covers multiple model families (graph, relational graph, and topological), multiple graph liftings (none, clique, ring), five TUDatasets, RINGTRANSFER synthetic validation, and additional experiments in appendices.

## Weaknesses

### Fatal
None.

### Major

- **Theoretical results are incremental—each is a direct lift of an existing graph result to a derived graph.** Lemma 3.2 mirrors Topping et al. (2022) and Di Giovanni et al. (2023); Proposition 3.4 mirrors Fesser & Weber (2023, Prop 3.4); Theorem 3.5 mirrors Di Giovanni et al. (2023, Thm 4.1). The key step in each case is forming the aggregated influence matrix Å and then applying known graph analysis to the influence graph. While the paper makes this systematic, no result exploits the higher-order structure in a way that cannot be captured by the flat influence graph. The paper does not identify phenomena unique to the topological/relational setting (e.g., oversquashing patterns that arise from particular incidence structures and are invisible in the aggregated adjacency), which limits the depth of the theoretical contribution. — This matters because the framing promises to "demystify" topological message-passing and provide novel insights specific to higher-order structures, but the actual analysis reduces to standard graph analysis on a derived structure.

- **Empirical evidence does not convincingly demonstrate that oversquashing is being addressed in practice.** The real-world experiments (Table 1) use standard TUDataset benchmarks that are not designed to exhibit long-range dependency bottlenecks, and the paper itself uses fixed, untuned hyperparameters, acknowledging that "hyperparameter tuning can significantly impact performance." The results are mixed: rewiring sometimes helps but often hurts (many red entries in Table 1), and no statistical testing is provided. On RINGTRANSFER (Figure 2), the basic GIN model consistently outperforms the relational/topological models, which raises questions about whether the topological formulation provides practical advantages. Without experiments on genuinely long-range tasks (e.g., LRGB) or direct oversquashing metrics (Jacobian estimation, effective resistance), the claim that the framework advances oversquashing mitigation remains empirically under-supported.

- **The rewiring heuristic discards topological structure during the critical step.** Algorithm 1 collapses all relations into a single undirected count-based adjacency (Equation 14), applies a graph rewiring method on this flattened graph, then adds the new edges as a single untyped relation R_{k+1}. This means the rewiring step itself operates without any awareness of simplex dimension, relation type, or topological semantics—exactly the information that distinguishes topological from graph message-passing. The paper acknowledges this but does not analyze whether this design choice limits effectiveness or introduces spurious structure.

### Minor

- **The extended Forman curvature (Definition 3.3) is not well-justified for the directed weighted setting.** The coefficients (4, -1, -1, +3, +2) are taken from the augmented Forman curvature defined for undirected unweighted graphs. The paper does not discuss whether the geometric interpretations that motivate these coefficients carry over to directed weighted influence graphs, nor does it explore alternative curvature definitions that might be more suitable.

- **The paper does not compare with the most direct baseline: a GNN applied to the influence graph G(S, B).** If the influence graph is the right object for oversquashing analysis, a natural sanity check is whether a simple GCN/GIN on this graph matches or exceeds more complex relational/simplicial models. Such a comparison would clarify whether the topological machinery adds value beyond what the flattened influence graph already provides.

### Trivial
- Notation is sometimes heavy; the aggregated influence matrix (Equation 5) sums over all positions and auxiliary indices, making it somewhat cumbersome to parse on first reading.

## Nice-to-Haves

- Experiments on tasks with genuine higher-order structure (e.g., mesh data, molecular dynamics) to validate that the framework captures oversquashing phenomena specific to topological message-passing that are absent in graph settings.
- Analysis of per-relation contributions to the influence graph, revealing which types of adjacency (boundary, co-boundary, lower, upper) alleviate or exacerbate oversquashing.
- A rewiring strategy that operates directly on the weighted directed influence graph rather than collapsing to an undirected adjacency, which would better leverage the theoretical framework.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The framework is just a repackaging of the augmented Hasse diagram."** While the critic argues the relational view adds nothing beyond what's already implicit in Hasse diagrams, the paper explicitly identifies the right derived objects (influence graph, aggregated influence matrix) and systematically extends results that have not been applied to topological settings before. The contribution is in making the connection precise and actionable, not in the individual components. Whether this amounts to "demystifying" is debatable, but the systematic framework does facilitate future work.

- **"Claims broader applicability to cellular complexes, sheaves, etc. but only experiments on simplicial complexes."** The paper's primary claim and title scope is to simplicial message-passing. The broader applicability is stated as a possibility (Remark 2.7) rather than an established result. Removing this as a major weakness since the paper scopes itself clearly.

- **"No comparison with virtual nodes, graph transformers, or other oversquashing-mitigation methods."** The paper's focus is on extending curvature-based rewiring to relational structures, not on comparing all possible oversquashing mitigations. Such comparison would broaden the paper but is beyond scope.

- **"Performance improvements may stem from added parameters."** The paper compares models of different architectures, but the primary comparison is between the same model with and without rewiring (Table 1), which is a controlled comparison. Cross-architecture parameter matching would be informative but is not essential.

## Novel Insights

The influence graph construction is a genuinely useful conceptual tool: by collapsing multi-arity relations into a single aggregated adjacency, it provides a principled "reduction" from topological message-passing oversquashing analysis to graph-based analysis. However, this reduction also reveals a fundamental limitation—once the aggregation is done, no result in the paper exploits information that is lost in the flattening (relation types, arity, dimensional structure). This suggests that the most impactful future direction may not be further graph-theoretic analysis on the influence graph, but rather developing analysis that preserves and exploits relational structure during the oversquashing analysis itself.

## Suggestions

- Add a direct comparison of a standard GNN operating on the influence graph vs. simplicial/relational models, to clarify whether the topological architecture provides benefits beyond the flattened analysis.
- Evaluate on at least one benchmark designed for long-range dependencies (e.g., LRGB Peptides) to more directly test oversquashing mitigation claims.
- Analyze the EFC curvature on the influence graph for concrete examples, examining how it differs from standard Forman curvature on the base graph and whether it reveals topologically meaningful bottlenecks.

## Score and Decision

**Calibration references:**

- **EcrdmRT99M** (Curvature Rewiring Effectiveness, scores 6/5/6/6, mean ~5.75, Accept Poster): Similar topic (rewiring for oversquashing), modest experimental results, but provides important reevaluation. This paper has more theoretical content but less conclusive experiments.
- **NmcOAwRyH5** (Virtual Nodes & Oversquashing, scores 5/3/6/6/8, mean ~5.6, Accept Poster): Similarly extends oversquashing theory to a new setting (virtual nodes), with incremental theory. Our paper has a broader scope but less novel results per result.
- **LkB80Rw6Ap** (Curvature MPNNs, scores 5/5/3/1, mean ~3.5, Reject): Much weaker—unclear experiments, no clear advantage. Our paper is clearly above this.
- **Ax3uliEBVR** (E(n) Equivariant Topological NNs, scores 6/6/6/6, mean 6, Accept Poster): Incremental but solid TDL contribution. Comparable scope and novelty.
- **scI9307PLG** (BuNN, scores 8/6/8, mean ~7.3, Accept Spotlight): Much stronger theoretical contribution (universality result). Our paper is below this.

The paper makes a genuine conceptual contribution (systematizing the extension of oversquashing analysis to topological structures via relational frameworks) but with incremental theoretical results and mixed empirical validation. It falls in the same tier as EcrdmRT99M and NmcOAwRyH5—useful contributions that extend oversquashing analysis to new settings but don't fully deliver on their framing's promise. I score it slightly above those due to the breadth of the framework and the number of axes covered, but the incremental nature of the theory and lack of compelling empirical evidence for oversquashing mitigation keep the score moderate.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>