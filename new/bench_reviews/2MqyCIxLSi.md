## Summary

The paper introduces Generalized Combinatorial Complex Neural Networks (GCCNs), a family of topological deep learning (TDL) architectures that decompose a combinatorial complex into an ensemble of strictly augmented Hasse graphs—one per neighborhood function—each processed by an independently chosen base model (GNN, Transformer, etc.) as the neighborhood message function ω_N. The paper also introduces per-rank neighborhoods, which restrict neighborhood functions to cells of specific ranks, and TopoTune, a lightweight configuration-driven software module integrated into TopoBenchmark for defining and training GCCNs. The authors prove GCCNs generalize CCNNs (Proposition 1), are permutation equivariant (Proposition 2), and claim GCCNs are strictly more expressive than CCNNs (Proposition 3), supported by experiments showing GCCNs often outperform CCNNs with fewer parameters.

## Strengths

- **Clean and useful architectural pattern.** Decomposing a combinatorial complex into an ensemble of strictly augmented Hasse graphs—each processed by a dedicated base model—is a principled way to bridge TDL and standard GNN libraries. This directly addresses the practical limitation that existing TDL tools can only replicate specific CCNN message-passing schemes. The comparison in Table 1 between the ensemble approach and the "Best GNN, 1 Aug. Hasse graph" baseline validates this design choice (e.g., ZINC cellular: 0.19 vs. 0.31).

- **TopoTune is a genuine software contribution.** A configuration-driven interface that lets practitioners define GCCNs by specifying only a neighborhood collection, ω_N (importable from PyTorch Geometric / DGL), and architectural parameters substantially lowers the barrier to TDL experimentation. Integration with TopoBenchmark ensures standardized data processing and training pipelines (Section 5).

- **Per-rank neighborhoods provide practical value.** The mechanism (Eqs. 6–7) allows selective activation of neighborhood functions for specific ranks, reducing computational cost and model size. The parameter efficiency results are concrete: on MUTAG, a cellular GCCN with per-rank neighborhoods is 19% the size of the best cellular CCNN while outperforming it (Section 6.2); on PROTEINS, per-rank GCCNs achieve within 2% of best performance with 48% of parameters (Fig. 5).

- **Formal subsumption of CCNNs.** Proposition 1 proves that for any CCNN there exists a GCCN that exactly reproduces its computation. This is a meaningful theoretical result distinguishing GCCNs from prior graph-expansion approaches (Jogl et al.) that could only simulate CCNNs under restrictive conditions (same intra/inter aggregations, no rank/neighborhood-dependent message functions).

- **Broad empirical coverage.** Experiments span 8 datasets, 3 topological domains (cellular, simplicial, hypergraph), 5 base models, and both graph-level and node-level tasks. The consistent pattern of GCCNs matching or outperforming CCNNs across this space, often with fewer parameters, is informative.

## Weaknesses

### Fatal

None.

### Major

- **Proposition 3's expressivity claim is likely trivially true and misleadingly presented.** The proposition states "GCCNs are strictly more expressive than CCNNs," but GCCNs allow arbitrary ω_N architectures (GIN, Transformers, multi-layer GNNs) while CCNNs are defined with single-layer message-passing functions ψ_{N,rk(·)}. A model class that subsumes Transformers being more expressive than one restricted to single-layer message-passing is expected—the strictness almost certainly derives from the unconstrained ω_N class rather than from the framework's structural properties (ensemble decomposition, per-rank neighborhoods). The meaningful expressivity question—whether GCCNs with comparable-capacity message-passing ω_N are strictly more expressive—is never isolated. Notably, the paper's own contribution bullet (line 100) states GCCNs are "as expressive as CCNNs," contradicting Proposition 3's "strictly more expressive" claim. This inconsistency, combined with the inaccessible appendix proof, prevents readers from assessing what genuinely drives the strictness. The result as presented implies a substantive structural finding that it likely does not deliver.

- **Experimental comparisons conflate framework contribution with base model power.** When GCCN(ω_N=GIN) outperforms a CCNN with standard message-passing, it is impossible to attribute the improvement to the GCCN framework rather than GIN's inherently greater expressivity over basic message-passing. The paper lacks an ablation that isolates the framework's contribution: for instance, GCCNs with simple message-passing ω_N (comparable to CCNNs) vs. CCNNs, or CCNNs augmented with equivalent-capacity base models. Table 2 partially addresses this by replicating SCNN/CWN neighborhood structures, but even there, ω_N is swept over five choices, making attribution unclear. The paper's claim that "GCCN's architectural novelties contribute to this performance" (Section 6.2) is therefore not rigorously supported. The one informative comparison is the "Best GNN, 1 Aug. Hasse graph" vs. ensemble rows in Table 1, which does suggest the ensemble structure itself helps—but this is only one dimension of the claimed contribution.

### Minor

- **Inconsistency between contribution statement and formal proposition.** The contributions section (line 100) says GCCNs are "as expressive as CCNNs," while Proposition 3 (line 235) says "strictly more expressive." This creates confusion about what the paper actually claims and suggests the strict expressivity claim may have been added without updating the earlier framing.

- **Statistical significance is weak on several datasets.** On MUTAG (188 graphs), standard deviations reach 6–8 points, making rankings essentially noise. The claim of outperforming by ">1σ in 11 of 16 cases" is a low bar on small datasets with high variance—many of these "outperformances" are within the overlap of confidence intervals. For example, PROTEINS cellular: GCCN(GCN) at 74.41 ± 1.77 vs. CCNN at 76.13 ± 2.70—the CCNN actually wins here, and several "wins" elsewhere are within overlapping uncertainty ranges.

- **Asymmetric hyperparameter tuning.** CCNNs benefit from "extensive hyperparameter tuning" (line 261) while GCCNs use "the TopoBenchmark default configuration." This actually makes GCCN improvements more impressive if anything (they're less tuned), but it also means neither comparison direction is fully controlled. The paper should acknowledge this limitation explicitly.

### Trivial

None.

## Nice-to-Haves

- Comparison with standard GNNs on original (non-lifted) graphs would help establish whether topological lifting provides value beyond what standard graph methods achieve—a question implicit in the paper's motivation but never directly tested.

- A systematic analysis of which per-rank neighborhood configurations work best for which tasks and why, going beyond the current parameter-efficiency demonstration.

- Ablation experiments with GCCNs using simple message-passing ω_N (matching CCNN capacity) to isolate the framework's structural contribution from the base model's contribution.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "systematic transformation" claim is overstated.** The critic says this is "a specific architectural template, not a general transformation that preserves the inductive biases." But the paper accurately describes what it does: plug any GNN into ω_N on the strictly augmented Hasse graphs. It never claims to preserve inductive biases. This criticism misreads the claim.

- **Harsh critic: per-rank neighborhoods are "just masking/pruning, not a conceptual advance."** The paper presents per-rank neighborhoods as a practical innovation for parameter efficiency and selective activation, not as a deep theoretical contribution. The empirical results (19% model size at comparable performance) justify it on its own terms.

- **Harsh critic: missing GNN baselines on original graphs as a damaging omission.** The paper's explicit scope is comparing GCCNs to CCNNs within TDL. While comparing to standard GNNs would strengthen the paper, its absence doesn't undermine the claimed contributions about GCCN vs. CCNN generalization and performance.

- **Harsh critic: Proposition 1 proof is inaccessible, raising reproduction concerns.** The proof is in the appendix, which the parser strips from all papers. Per our rules, we assume the appendix exists and contains the proof.

- **Harsh critic: formatting/presentation nitpicks.** Minor notation and figure concerns are trivial and removed per rules.

- **Strength finder: "Proposition 3 proves strictly greater expressivity" as a core strength.** This is removed because the strictness likely derives from allowing arbitrary ω_N rather than from the framework's structural properties, as discussed in the Major weakness above. It conflicts with a verified weakness.

## Novel Insights

The paper reveals an interesting architectural trade-off space in TDL that hasn't been systematically explored: the separation between neighborhood structure (which defines the topology) and the message function ω_N (which defines the computation). By decoupling these, GCCNs expose that much of the performance variation in TDL may come from the choice of base model rather than the topological framework itself—a possibility the paper's experiments inadvertently demonstrate but never explicitly confront. The parameter efficiency results from per-rank neighborhoods suggest that not all topological relationships are equally useful for learning, hinting that "topological completeness" may be less important than "topological relevance"—an insight that could inform future TDL architecture design beyond GCCNs.

## Suggestions

- Moderate the Proposition 3 claim: either reframe it as "GCCNs with unrestricted ω_N are strictly more expressive" (acknowledging the trivial source of strictness) or add an ablation testing whether GCCNs with message-passing-only ω_N are also strictly more expressive.

- Add one focused ablation: GCCN with simple GCN-level ω_N vs. CCNN, controlling for base model capacity, to cleanly demonstrate the framework's structural contribution.

- Fix the inconsistency between the contribution bullet ("as expressive as CCNNs") and Proposition 3 ("strictly more expressive than CCNNs").

## Evaluation

**Originality:** The ensemble-of-strictly-augmented-Hasse-graphs decomposition is a natural but non-obvious extension of prior graph-expansion work. Per-rank neighborhoods are a useful practical innovation. TopoTune as a unifying software framework is original in the TDL context. The theoretical contributions (Props 1-3) are incremental—Prop 1 is a standard subsumption proof, and Prop 3 is likely trivially true.

**Importance of research question:** Standardizing and democratizing TDL is an important community goal. The paper directly addresses acknowledged open problems in the field.

**Claim support:** The subsumption claim (Prop 1) is well-supported. The strict expressivity claim (Prop 3) is likely trivially true. The empirical superiority claims are confounded by the base model power asymmetry.

**Soundness of experiments:** Broad but confounded. The comparison with "1 Aug. Hasse graph" is the cleanest evidence for the ensemble structure's value, but other claimed contributions lack clean ablations.

**Clarity:** The paper is well-written and well-motivated, with clear figures and a logical structure. The Proposition 3 / contribution bullet inconsistency is a notable lapse.

**Value to community:** TopoTune and the GCCN framework have genuine practical value for lowering the barrier to TDL experimentation. The paper addresses real infrastructure gaps in the field.

## Score and Decision

**Calibration anchors:**

- **High band (>7):** EzjsoomYEb (avg 8.0, TDL expressivity / MCN/SMCN) — deeper theoretical analysis of HOMP expressivity blindspots with new architectures and benchmarks; substantially more rigorous than this paper. 0JsRZEGZ7L (avg 8.0, Differentiable Cell Complex Module) — novel method for latent topology inference with extensive experiments. o2Igqm95SJ (avg 8.0, CAX software) — clean software contribution with 2000× speedups and novel experiments. The paper under review has weaker theory and confounded experiments compared to these.

- **Medium band (4-6):** WpQbM1kBuy (avg 5.25, Prodigy) — strong empirical results but overclaimed theoretical contribution and confounded experiments. This paper is stronger than Prodigy because: (1) the software contribution (TopoTune) adds independent value, (2) the subsumption proof (Prop 1) is substantive, and (3) the parameter efficiency results are informative even without clean ablations. BOQpRtI4F5 (avg 6.75, GNN generalization/expressivity) — bridges generalization and expressivity with benchmark experiments. This paper is comparable but has a more practical orientation via software.

- **Low band (<3):** 63r6HyqyRm (avg 2.33, LLM vs. non-pretrained) — fundamentally unfair comparison using pretrained vs. non-pretrained models. This paper's comparison issues are much less severe—the GCCN vs. CCNN comparison is between models in the same domain with the same training pipeline.

The paper sits above the medium-band papers with confounded experiments (Prodigy at 5.25) due to its genuine software contribution and useful framework, but below the high-band TDL papers (8.0) due to overclaimed theory and the lack of clean ablations. Relative to the 6.5-range generalization papers, this paper is roughly comparable—its software contribution offsets its weaker theory, while its confounded experiments offset its broader empirical coverage.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>