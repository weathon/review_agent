## Summary
This paper proposes a unifying relational-structure view of simplicial and related higher-order message passing, then uses that view to extend several graph oversquashing analyses to the topological setting via aggregated influence matrices and influence graphs. The main contribution is therefore conceptual/theoretical: it provides a clean bridge from graph oversquashing theory to simplicial/topological message passing, and supplements it with an initial relational rewiring heuristic plus broad but only partially convincing experiments.

## Strengths
- **Clear and useful unifying framework.** Section 2 gives a principled formalization of simplicial message passing as relational message passing. The mapping from simplicial adjacencies to relations is explicit in Section 2.2, and the paper cleanly positions simplicial, relational graph, higher-order, and cellular message passing under one umbrella.
- **A meaningful technical bridge for oversquashing analysis.** Lemma 3.2 is the core contribution: it generalizes sensitivity-style oversquashing bounds from graphs to relational structures using the aggregated influence matrix and the augmented matrix \( \mathbf B \). This is a reusable abstraction rather than a one-off proof trick.
- **Influence graphs are an intuitive abstraction.** Definition 3.1 gives a concrete graph object on which existing graph-theoretic tools can operate. This makes the extensions of curvature- and path-based reasoning technically plausible and conceptually transparent.
- **The paper addresses an important and timely question.** Oversquashing in topological deep learning is indeed underexplored, and the paper is well motivated in trying to give TDL a comparable theoretical toolkit to what GNNs already have.
- **Theoretical scope is fairly broad.** Beyond sensitivity, the paper discusses local geometry, depth, and width/hidden-dimension effects. Even where some parts are lighter than others, the paper covers the main axes along which oversquashing is typically analyzed.
- **The limitations discussion is commendably candid.** Section 6 explicitly notes that direct graph-vs-complex comparisons are only proxy-level and that the rewiring methods used were not designed for the weighted directed influence graphs produced by the theory. This honesty improves trust in the paper.
- **Empirical coverage is broad.** The experiments include graph, relational graph, and topological models, multiple liftings, several rewiring algorithms, and a synthetic long-range benchmark. The breadth is a strength even if the conclusions from it should be stated more cautiously.

## Weaknesses

###: Fatal
None.

### Major:
- **The practical claim that the proposed heuristic “mitigates oversquashing” is only weakly supported by the experiments.** The paper’s empirical evidence is mostly task accuracy after rewiring, not direct evidence that oversquashing itself was reduced. On real datasets, rewiring is mixed and often hurts performance; the paper itself says, “the impact of rewiring ... varies across datasets” (Section 5.1). On the synthetic benchmark, the main text provides qualitative trend descriptions rather than direct measurements of the sensitivity quantities the theory analyzes. As written, the theory supports an oversquashing *analysis* framework much better than the experiments support a robust *mitigation* claim.
- **The empirical setup does not cleanly isolate the contribution of the relational formalization from architecture and lifting confounds.** Table 1 compares graph, relational graph, and topological models across different lifted representations (none/clique/ring), which substantially change the number of entities and connectivity structure. Section 6 itself acknowledges that “the significant differences in size and structure make direct empirical comparisons ... less theoretically rigorous.” This means the experiments do not cleanly establish whether the relational viewpoint itself is responsible for the observed behavior, or whether representation/lifting effects dominate.
- **The rewiring heuristic is conceptually weaker than the theory that motivates it.** Section 3 develops weighted directed influence graphs and relation-aware machinery, but Algorithm 1 in Section 4 rewires a collapsed adjacency matrix. This discards relation type, arity, and much of the higher-order structure that the framework carefully introduces. The paper acknowledges this mismatch in Section 6 (“the rewiring algorithms ... were not originally designed with weighted directed influence graphs in mind”), which is fair, but it also means the practical method is an initial baseline heuristic rather than a principled consequence of the full relational analysis.
- **The experimental “validation” of the depth/width theory is indirect.** Theorem 3.5 and Section 3.4 concern sensitivity bounds and quantities derived from the influence graph, yet Section 5.2 validates them via downstream accuracy trends as ring size, rewiring iterations, and hidden dimension vary. Those trends are compatible with the theory, but they do not directly test the mechanism. There are no main-paper measurements of Jacobians, \((\mathbf B^t)_{\sigma,\tau}\), path-count terms, or related quantities, so the empirical support for the specific theoretical interpretation is limited.

### Minor
- **The paper occasionally overstates the practical significance of the empirical results.** For example, the abstract and takeaway statements suggest stronger practical conclusions than the mixed experimental outcomes justify. The strongest contribution is the theoretical framing, not conclusive rewiring gains.
- **Section 3.4 is materially weaker than the formal theorem/proposition results.** The hidden-dimension discussion is based on substituting width-dependent big-O estimates into the sensitivity bound and drawing qualitative conclusions. This is useful intuition, but it is more heuristic than the surrounding formal results and should be framed as such.
- **Some assumptions are presented a bit too strongly.** In Section 3.3, Assumption 2 is described as “non-restrictive,” but row-normalization choices can materially affect message-passing dynamics in practice. The assumption is acceptable, but the phrasing slightly oversells its innocuousness.
- **The aggregation in the influence matrix reduces diagnostic granularity.** By summing over all relations into \(\tilde{\mathbf A}\), the framework gains portability but loses the ability to say which particular relation types are chiefly responsible for bottlenecks. This is not a flaw in the basic theory, but it limits the interpretability of the analysis.

### Trivial
- **Novelty is more in the unification than in entirely new higher-order phenomena.** The theory is valuable, but many results are extensions of known graph arguments once the influence-graph abstraction is in place. This does not negate the contribution, but it slightly limits the level of technical surprise.

## Nice-to-Haves
- A relation-aware rewiring method operating directly on the weighted directed influence graph, rather than the collapsed graph, would better match the theory.
- A decomposition of sensitivity by relation type could reveal whether boundary, co-boundary, lower, or upper adjacencies contribute differently to oversquashing.
- Direct measurements of Jacobian norms or influence quantities on synthetic tasks would make the theoretical claims much more compelling empirically.
- A controlled comparison between rewiring the base graph before lifting and the proposed relational wrapper after lifting would clarify the added value of the relational approach.
- More rigorous per-dataset/per-model tuning for rewiring would strengthen the empirical story, though this is not required to assess the theoretical contribution.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper should include additional external benchmarks such as LRGB/MANTRA.”** This is scope creep. It would strengthen the paper, but the current synthetic long-range benchmark plus broad real-data study are enough for an initial theory-forward paper.
- **“The paper must include actual simplicial models on RingTransfer or else the claim is unsupported.”** This is too strong. The paper’s main target is relational/topological message passing broadly, and Section 5.2 does test lifted relational variants relevant to the framework. Still, using more explicit simplicial architectures there would indeed improve the empirical story, so this is better treated as a nice-to-have.
- **“The framework extends to cellular complexes/sheaves but does not fully demonstrate them, so the claim is invalid.”** The paper carefully says the connection extends broadly and gives the simplicial case as the case study; criticizing the absence of a full multi-structure treatment would be beyond scope.
- **Any criticism based on formatting/parser artifacts.** The extracted text clearly contains artifacts and should not be used against the paper.

## Novel Insights
The most interesting synthesis is that the paper is strongest precisely where it is most modest: as a transfer principle from graph oversquashing theory to higher-order message passing. The influence-graph formalism is not just a notational convenience; it identifies the real unit of analysis as the induced message-passing dependency structure rather than the original topological object itself. At the same time, the practical section reveals an important tension: once one collapses higher-order relations enough to reuse existing graph rewiring tools, one also gives up much of what is distinctive about the higher-order representation. That tension suggests the paper’s real long-term value is as a foundation for future relation-aware diagnostics and rewiring methods, more than as a finished practical solution.

## Suggestions
- Recast the paper more explicitly as a **theory-first contribution** and soften the practical language around “mitigation” unless supported by direct oversquashing diagnostics.
- Add a direct empirical probe of the theoretical mechanism: Jacobian norms, sensitivity surrogates, or influence-graph quantities on RingTransfer-style settings.
- Compare rewiring on the collapsed graph to a simpler baseline such as rewiring the original graph before lifting, to isolate what the relational wrapper adds.
- Develop at least one relation-aware rewiring variant that uses the weighted directed influence graph rather than discarding higher-order structure.
- Separate the formal results from the heuristic discussion more clearly, especially in Section 3.4.
- Quantify the “respond similarly” claim in Section 5 rather than stating it qualitatively from a heterogeneous table.

## Score and Decision
**Overall evaluation by axis:**  
- **Originality:** moderate-to-good. The central novelty is the unifying relational/influence-graph formulation rather than entirely new oversquashing phenomena.  
- **Importance of the research question:** high. Oversquashing in topological message passing is a real and worthwhile problem.  
- **Whether the claims are well supported:** mixed. The theoretical claims are reasonably supported; the practical mitigation claims are overstated relative to the evidence.  
- **Soundness of experiments:** adequate but not decisive. Broad coverage, but confounded comparisons and indirect validation limit the strength of the conclusions.  
- **Clarity of writing:** generally good and conceptually clear.  
- **Value to the research community:** good, especially for researchers interested in theory and unification across graph and higher-order message passing.

**Calibration against human-reviewed anchors:**  
I compared this paper primarily to:
- **EcrdmRT99M.md** (“The Effectiveness of Curvature-Based Rewiring... Revisited”, scores 6/5/6/6, Accept Poster): that paper also studies rewiring/oversquashing with mixed empirical conclusions. The current paper is somewhat stronger conceptually/theoretically but similarly weaker on definitive practical claims.
- **NmcOAwRyH5.md** (“Understanding Virtual Nodes: Oversquashing and Node Heterogeneity”, scores 5/3/6/6/8, Accept Poster): that paper is a useful anchor for theory-heavy oversquashing work with mixed reviewer reactions. The current submission feels in a similar accept-range, though with a narrower and more transfer-style theoretical contribution.
- **4Ua4hKiAJX.md** (“Locality-Aware Graph Rewiring in GNNs”, scores 8/5/3/6, Accept Poster): this anchor is stronger on having a concrete rewiring method, while the current paper is stronger on unifying topological/relational theory but weaker on practical demonstration.
- **K0oFDAPnU4.md** (“A Unified Framework for Hierarchical Diffusion via Simplicial Complexes”, scores 3/3/1/8, Reject): this serves as a lower-end anchor for higher-order/simplicial work whose core claims were not convincingly established. The present paper is clearly above that bar because it has a coherent technical core and a credible, useful theoretical framing.

Relative to these anchors, this paper looks like a **borderline-to-solid poster accept** rather than a high-confidence accept or a reject. Its theoretical contribution is real and useful, but the practical framing should be toned down.

**Final score: 6.5 / 10**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>