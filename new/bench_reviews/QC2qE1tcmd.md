## Summary

This paper introduces an axiomatic framework that reformulates simplicial complexes and their message-passing schemes as relational structures, enabling the extension of graph-theoretic oversquashing analyses (sensitivity bounds, curvature, depth effects, hidden dimension analysis) to topological deep learning. The key technical device is the "influence graph" derived from the aggregated influence matrix of shift operators, which collapses relational message-passing dynamics into a weighted directed graph on which existing graph oversquashing results can be applied. The paper also proposes a heuristic rewiring algorithm (Algorithm 1) that collapses a relational structure to a graph, applies any off-the-shelf graph rewiring method, and feeds the added edges back as a new relation.

## Strengths

- **Timely and important problem formulation**: The paper addresses a genuine gap—oversquashing in topological message-passing has been identified as a pressing open question in the TDL community (research directions 2 and 9 of Papamakarios et al., 2024)—and provides a systematic starting point for its analysis.

- **Clean and systematic theoretical framework**: The influence graph construction (Definitions 3.1, Equations 5–7) is well-defined, and the paper systematically derives extensions of key oversquashing results—sensitivity bounds (Lemma 3.2), extended Forman curvature (Definition 3.3, Proposition 3.4), depth effects (Theorem 3.5), and hidden dimension analysis (Section 3.4)—in a coherent and self-contained manner.

- **Broad empirical coverage**: The experiments span multiple model types (SGC, GCN, GIN, RGCN, RGIN, SIN, CIN, CIN++), multiple lifting strategies (none, clique, ring), multiple rewiring algorithms (SDRF, FoSR, AFRC), five TUDataset benchmarks, and a synthetic RINGTRANSFER task, providing a reasonably comprehensive evaluation landscape.

- **Practical utility**: The idea of applying existing graph rewiring as a preprocessing step for topological models is simple and potentially useful for practitioners, regardless of the depth of the underlying theory.

## Weaknesses

### Major

- **Theoretical contributions are largely direct transfers of existing graph results via a graph reduction**: The central conceptual step is constructing the influence graph from a relational structure, after which all subsequent analysis (Lemma 3.2, Proposition 3.4, Theorem 3.5) closely parallels existing GNN results (Topping et al., 2022; Di Giovanni et al., 2023; Fesser & Weber, 2023). The paper acknowledges this structural similarity in passing ("makes it possible to leverage and extend graph-theoretic concepts"), but the framing as "novel extensions" and "addressing settings where prior results are not applicable" overclaims: the results are obtained by reducing relational structures to a weighted directed graph and reapplying existing arguments almost verbatim. No regime is identified where the higher-order structure (e.g., orientation, incidence algebra, homology) produces qualitatively different oversquashing behavior than what the collapsed graph analysis captures. This is a meaningful gap between the paper's ambitious framing and the actual technical content.

- **Empirical evaluation does not substantively validate topological/simplicial-specific claims about oversquashing**: All real-world benchmarks (ENZYMES, IMDB-B, MUTAG, NCI1, PROTEINS) are graph classification tasks where graphs are artificially lifted into simplicial complexes. The paper does not evaluate on any naturally higher-order datasets (e.g., meshes, molecular datasets with ring structure) where topological structure is inherent rather than synthetic. The RINGTRANSFER task is also a graph benchmark, not one where oversquashing arises specifically from higher-order adjacency structure. Without such experiments, it remains unclear whether the framework provides new insight into oversquashing in genuinely topological settings, or simply repackages graph-level analysis.

- **The rewiring heuristic (Algorithm 1) is disconnected from the theoretical framework and is essentially a thin wrapper**: The algorithm collapses the relational structure to an unweighted graph via the collapsed adjacency matrix (Definition 4.1), applies any graph rewiring algorithm, and adds the new edges as a separate relation $R_{k+1}$. This procedure: (a) ignores edge weights and directionality present in the influence graph; (b) does not use the curvature or sensitivity machinery developed in Section 3; (c) does not reason about which relations or dimensional interactions are actual bottlenecks. The paper itself acknowledges "the rewiring algorithms we applied our relational rewiring heuristic to were not originally designed with weighted directed influence graphs in mind." The connection between the theory and practice is therefore weak, and the algorithm does not leverage the unique aspects of the relational formalization in a meaningful way.

### Minor

- **Fixed, dataset-agnostic hyperparameters for rewiring limit interpretability of results**: The authors use "fixed, dataset- and model-agnostic hyperparameters" for rewirling, diverging from prior work. While understandable from a resource perspective, this makes it difficult to attribute observed performance changes specifically to the rewiring strategy versus hyperparameter sensitivity, especially given that many improvements in Table 1 are within 1–3 standard errors.

- **No analysis of information lost in the influence graph reduction**: The aggregated influence matrix (Equation 6) and collapsed adjacency (Definition 4.1) discard arity and relational type information. The paper does not analyze what structural properties of the original relational/topological structure are lost and whether this loss matters for oversquashing analysis or rewiring effectiveness. The Appendix D.2 finding of a "statistically significant linear relationship between the weighted curvature of graphs and their lifted clique complexes" actually suggests that the graph-level curvature already captures much of what matters, which somewhat undermines claims that the topological perspective adds unique insight.

- **Quantitative validation of theoretical bounds is absent**: The RINGTRANSFER experiments validate qualitative trends (performance drops with ring size, recovers with width/rewiring), consistent with intuition and prior work. However, the paper does not verify whether the actual Jacobians match the theoretical bounds (e.g., the exponential decay in Theorem 3.5, the dependence on $\omega_\ell(\sigma,\tau)$ and $M$), leaving it unclear whether the bounds are predictive or vacuously loose.

### Trivial

- None warranting mention.

## Nice-to-Haves

- Experiments on naturally higher-order datasets (e.g., mesh classification, molecular property prediction with ring structure) where topological structure is inherent rather than synthetically generated, to demonstrate that the framework provides value beyond what graph-level analysis already offers.

- A direct comparison between relational rewiring (Algorithm 1) and the simpler baseline of rewiring the base graph first, then lifting into a simplicial complex. This would clarify whether the relational framework adds practical value.

- A simplicial-specific synthetic task where oversquashing arises from the higher-order adjacency structure (e.g., information must propagate across boundary/co-boundary/lower/upper adjacencies in a way that has no graph analog), demonstrating genuinely novel oversquashing phenomena in topological settings.

- Experiments verifying the quantitative tightness of the theoretical bounds on trained models, to establish that the bounds are not merely formally correct but also predictive.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Claim that the relational message passing model (Definition 2.5) is under-specified or lacks equivalence proof**: The harsh critic claims the "equivalence statement... is plausible but not formally proved; there is an implicit assumption that aggregation over $\mathbf{A}^{R_i}$ recovers exactly the sums in Eq. (1)." However, the paper provides clear definitions of shift operators $\mathbf{A}^{R_i}$ and explicitly states in the paragraph following Remark 2.6 that "the message functions $\psi_i^{(t)}$ correspond to $M_B, M_C, M_\downarrow, M_\uparrow$, the update function $\phi^{(t)}$ to UPDATE, and aggregation uses shift operators $\mathbf{A}^{R_i}$." The correspondence is sufficiently specified for the claims made; demanding a formal equivalence proof is an excessive requirement for this type of framework paper.

- **Demand for cellular/CW complex examples in the main text**: The harsh critic argues that "at least one non-simplicial concrete example in the main text would strengthen the case." The paper explicitly claims applicability to these settings and provides details in appendices. Given the paper's focus on simplicial complexes (as stated in the title), this is a scope creep request.

- **Criticisms about unfair comparison with other methods**: The harsh critic suggests comparing against more recent topological architectures and asking whether simpler baselines (e.g., graph-level rewiring before lifting) achieve similar results. While these are valid experimental suggestions (moved to Nice-to-Haves), they do not constitute unfairness in the current comparisons—the paper compares like-with-like (same models with and without rewiring) rather than giving its own method an advantage.

- **Formatting and presentation nitpicks**: Removed per instructions (e.g., table layout concerns, color scheme in Table 1).

- **Criticism that "existing methods for analysis do not apply" is overstated**: The harsh critic argues that once higher-order structures are converted to graphs (via Hasse diagrams, etc.), graph-based analyses do apply. This is precisely the paper's point—their framework enables this conversion systematically. The paper does not claim these methods were impossible before; it claims they were not directly applied to topological message-passing settings, which is accurate.

## Novel Insights

The paper's most novel contribution is conceptual rather than technical: the insight that the "influence graph" $\mathcal{G}(\mathcal{S}, \mathbf{B})$ acts as a sufficient reduction for analyzing oversquashing in arbitrary relational message-passing schemes, capturing all relevant message-passing dynamics regardless of the arity or type of the underlying relations. This has an important corollary that the paper does not fully draw out: if oversquashing in a simplicial complex can be fully characterized by the influence graph (a weighted directed graph on simplices), then any oversquashing phenomenon in topological message-passing that is distinctive from graph oversquashing must involve dynamics that cannot be captured by this pairwise aggregation. The empirical finding in Appendix D.2 that weighted curvatures of graphs and their clique complexes exhibit a statistically significant linear relationship further hints that, at least for the weighted Forman curvature measure, the graph-level analysis may already subsume much of what the topological perspective offers—a finding that could be interpreted as either supporting the framework's graph reduction or questioning the unique value added by topological formalisms for oversquashing analysis.

## Suggestions

- Directly compare Algorithm 1 against the baseline of: (1) rewiring the base graph, then lifting, versus (2) lifting, then applying Algorithm 1. If the results are comparable, the relational perspective adds less practical value than claimed.

- Analyze what structural information about the original relational/topological structure is lost in the influence graph reduction and whether this loss affects oversquashing analysis in specific regimes (e.g., when certain relations are much more structurally important than others).

- Add experiments on at least one dataset where higher-order structure is inherent (e.g., mesh classification, molecular datasets with ring structure) rather than synthetically generated via lifting, to validate the claim that the framework genuinely advances TDL rather than just graph methods applied to lifted structures.

## Score and Decision

I calibrated against several papers in the oversquashing and TDL space:

- **Virtual Nodes / Oversquashing** (NmcOAwRyH5): Scores 5,3,6,6,8 (avg ~5.6), accepted poster. Solid theoretical contribution on VN effects on oversquashing, limited empirical improvement.
- **Curvature-based rewiring effectiveness** (EcrdmRT99M): Scores 6,5,6,6 (avg ~5.75), accepted poster. Re-evaluates existing methods, raises important methodological concerns about rewiring benchmarks.
- **Locality-Aware Rewiring** (4Ua4hKiAJX): Scores 8,5,3,6 (avg ~5.5), accepted poster. Novel framework for rewiring with empirical improvements, but concerns about hyperparameter fairness and limited baselines.
- **Curvature MPNNs** (LkB80Rw6Ap): Scores 5,5,3,1 (avg ~3.5), rejected. Mathematical errors and unclear methodology.

This paper is comparable to the virtual nodes paper in scope (a theoretical framework extending oversquashing analysis to a new setting), but with more incremental theoretical contributions (direct transfers via the influence graph reduction rather than genuinely new characterizations) and weaker empirical validation (no naturally higher-order datasets, limited connection between theory and practice). It is stronger than the Curvature MPNNs paper (which had mathematical errors) but weaker than the locality-aware rewiring paper (which proposed a genuinely novel rewiring framework with clearer empirical benefits). The conceptual contribution is real but modest, and the execution largely repackages existing graph-theoretic ideas through a sensible but conceptually straightforward reduction.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>