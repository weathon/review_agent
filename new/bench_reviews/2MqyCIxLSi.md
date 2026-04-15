## Summary

This paper introduces Generalized Combinatorial Complex Neural Networks (GCCNs), a unified framework that decomposes a combinatorial complex into an ensemble of *strictly augmented Hasse graphs*—one per neighborhood function—and processes each with an arbitrary neural network (GNN, Transformer, etc.), then aggregates outputs rank-wise. Three propositions formalize the framework's properties: GCCNs subsume CCNNs (Prop 1), are conditionally cell permutation equivariant (Prop 2), and are strictly more expressive than CCNNs (Prop 3). The paper also introduces TopoTune, a lightweight software integrated into TopoBenchmark that enables rapid prototyping of GCCN architectures by importing models directly from PyG/DGL. Experiments across simplicial and cellular domains on eight datasets compare a large family of GCCNs against existing CCNN baselines.

---

## Claims and Support

| Claim | Verdict | Notes |
|---|---|---|
| GCCNs systematically generalize *any* GNN to a TDL counterpart | **Partially supported** | The construction is a flexible template, but "any neural network" in the abstract/intro overstates things; Prop 2's equivariance guarantee is conditional on the base model being permutation equivariant. |
| GCCNs formally subsume CCNNs (Prop 1) | **Supported** | Prop 1 is stated clearly with a constructive argument; proof is in Appendix B.1. The key design choices (Eq. 8 vs Eq. 3) make the simulation plausible. |
| GCCNs are cell permutation equivariant (Prop 2) | **Conditionally supported** | The paper correctly states the conditions in Prop 2, but the Introduction contribution bullet ("*are* cell permutation equivariant") drops those conditions, creating a misleading impression. |
| GCCNs are "as expressive as CCNNs" (Intro bullet) vs. "strictly more expressive" (Prop 3) | **Internally inconsistent** | The Introduction says "(iii) are as expressive as CCNNs" while Prop 3 in Section 4 says "strictly more expressive." These are not equivalent and only the stronger version is formally claimed in propositions. |
| GCCNs "consistently match or outperform CCNNs" | **Partially supported** | Table 1 shows many overlapping intervals at 1σ; the GCCN class covers a very large search space (10 neighborhoods × 5 base models) while CCNNs are fixed "best" baselines. Competitive is a fairer characterization than "consistently outperform." |
| Ensemble of strictly augmented Hasse graphs improves over single augmented Hasse graph | **Partially supported** | Table 2 shows the comparison for SCNN/CWN counterparts. The improvement is real in some cases but not uniform or fully controlled. |
| Hypergraph domain coverage | **Not demonstrated** | Table 1 includes a Hypergraph section showing only CCNN results with no GCCN rows. The paper claims the framework works "across many topological domains" but never evaluates GCCNs in the hypergraph domain. |

---

## Strengths

- **Principled and practically useful unification.** The strictly augmented Hasse graph ensemble approach is conceptually clean: it separates per-neighborhood information flows (correcting the known limitation of single augmented Hasse graphs that conflate neighborhoods) while remaining implementable with standard GNN libraries. This is a genuine architectural innovation over prior work (Jogl et al., Hajij et al.).

- **Per-rank neighborhoods.** The formal introduction of rank-restricted neighborhoods (Eq. 6–7) allows practitioners to select exactly which ranks participate in each message-passing step, reducing unnecessary computation. The paper demonstrates a real case where this enables an 19%-sized model on MUTAG with competitive accuracy—a non-trivial finding.

- **Software contribution (TopoTune).** The integration into TopoBenchmark with direct PyG/DGL import capability addresses a genuine pain point: the lack of a standardized implementation vehicle in TDL. The configuration-file–based paradigm (specify neighborhoods + ω_N) is concretely described and likely to lower barriers for non-experts.

- **Breadth of experiments.** The coverage across 8 datasets, 2 topological domains, 5 base architectures, 10 neighborhood structures, and both graph-level and node-level tasks gives meaningful evidence that the framework is versatile and not dataset-specific.

- **Theoretical grounding.** Propositions 1–3, even if imperfectly stated in the main text, provide formal justification that the GCCN framework is not merely an engineering wrapper but sits in a well-defined relationship to CCNNs in terms of expressivity and symmetry.

---

## Weaknesses

### Fatal
*None identified.* The paper's core contributions—the GCCN construction, TopoTune, and the theoretical framework—are sound and useful. No single issue invalidates the central claims.

### Major

- **Internal inconsistency in expressivity claim.** The Introduction contribution bullet states GCCNs "(iii) are as expressive as CCNNs," while Proposition 3 asserts "GCCNs are *strictly* more expressive than CCNNs." These are contradictory claims. "Strictly more expressive" is a qualitatively stronger statement requiring a formal separation argument (showing a pair of complexes that GCCNs distinguish but no CCNN can). The proof is delegated to Appendix B.3 with no intuition in the main text. The main text should resolve this contradiction, pick the correct statement, and provide at minimum a concrete witness pair illustrating the strict gap—especially since the strict expressivity advantage may stem entirely from using non-message-passing ω_N (e.g., Transformers), which would substantially narrow the practical scope of the claim for the message-passing configurations evaluated in experiments.

- **Missing GCCN results in the hypergraph domain.** Table 1's hypergraph section contains only CCNN baselines; there are no GCCN rows. This directly contradicts the paper's claim that the framework "works across many topological domains" and that it provides a "first method designed to work across many topological domains." Without GCCN results for hypergraphs, this claim is unsubstantiated for an entire domain.

- **Comparison protocol overstates consistency of outperformance.** GCCNs are evaluated by sweeping 10 neighborhood structures × 5 base models (and reporting the best per-ω_N or overall best), while CCNN baselines are fixed "best" configurations from TopoBenchmark. With 5 seeds and high variance on several datasets (e.g., MUTAG: ±4–7%), many reported gains are within one standard deviation. The statement "GCCNs outperform the best counterpart CCNN by >1σ in 11 cases" uses one-standard-deviation overlap as a significance proxy, which is not a statistical test. The paper's framing of "consistently outperform" is too strong; "competitive with and often comparable or better than" is what the data supports.

- **Conditional equivariance presented as unconditional in the introduction.** The introduction contribution bullet says GCCNs "are cell permutation equivariant" without conditions, while Proposition 2 correctly qualifies this: the base model must be node permutation equivariant and the inter-neighborhood aggregator must be cell permutation invariant. Since the paper actively encourages using arbitrary architectures including Transformers, this distinction matters in practice. The abstract also uses the unconditional phrasing "any neural network."

### Minor

- **Proposition 3's justification is entirely absent from the main text.** A headline theoretical result—strict expressivity advantage over all CCNNs—requires at least an intuitive example or proof sketch in the main body, not solely an appendix reference. Readers cannot evaluate the validity of the claim from the paper as written.

- **High variance on MUTAG weakens graph-level conclusions.** Standard deviations of ±6–8% (e.g., GCCN-GIN cellular 86.38 ± 6.49, GCCN-GCN cellular 85.11 ± 6.73) on a dataset where competing methods differ by 5–10% make it impossible to draw firm conclusions about comparative performance from 5 seeds alone. Increasing seeds or reporting confidence intervals would help.

- **Ablation for ensemble vs. single graph is not fully controlled.** Table 2 compares ensemble vs. single-Hasse-graph GCCNs for SCNN/CWN counterparts but reports only the best ω_N per approach rather than isolating the ensemble contribution under identical base models and parameter budgets. The paper's claim that "representing complexes as ensembles consistently improves results" would be stronger with a matched-model ablation.

### Trivial

- **Terminology inconsistency (GCCN/GCNN/GCN).** The method is called GCCN in the abstract/introduction/theory, but appears as "GCN" throughout the experiments section and table headers. Table 1 labels all proposed models as "GCN ω_N = ..." This requires disambiguation even though readers can infer the meaning.

---

## Nice-to-Haves

- **Comparison with plain GNNs on unlifted graphs.** Since all datasets are originally graphs lifted to topological domains, understanding whether the topological processing provides genuine benefit over base GNNs on the original structure would be informative. This is not a requirement for a TDL framework paper but would strengthen the value proposition.

- **Computational overhead quantification in the main text.** The paper acknowledges in Section 6.2 that GCCNs "slow down for larger datasets, most likely due to TopoTune's on-the-fly graph expansion" but defers wall-clock times to Appendix G. Moving a summary table of training times to the main text would help practitioners assess when GCCNs are practical.

- **Explicit mapping to the 11 open problems.** The conclusion claims to address "7 of the 11 open problems" but provides only inline references. A concise mapping table (problem → addressed by which contribution) would make this claim more transparent and useful.

- **A concrete illustrative example of per-rank neighborhoods improving over standard neighborhoods.** A small molecule case study showing where restricting message passing to a specific rank preserves relevant information while a full-rank neighborhood introduces noise would make the per-rank concept intuitive.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "the paper never formalizes what counts as the 'TDL counterpart' of an arbitrary architecture."** Removed because the framework is explicitly presented as a construction template (not a bijective canonical mapping). The paper makes no claim to a unique canonical counterpart; it claims to provide a systematic way to *produce* a valid topological architecture from a given graph module. The criticism sets up a standard the paper never claimed to meet.

- **Harsh Critic: "the headline empirical claim is not supported by a fair comparison protocol" (framed as equivalent to overclaiming)** Partially removed as a "fatal" framing. The comparison fairness issue is real (kept as Major), but the paper does explicitly state "CCNN results reflect extensive hyperparameter tuning... while we fix GCN training hyperparameters using the TopoBenchmark default configuration"—i.e., the paper is transparent about the asymmetry, and the asymmetry intentionally disfavors GCCNs on training hyperparameters. The search over architecture variants (neighborhoods, ω_N) is a legitimate design space exploration, not hidden. The weakness is kept but de-escalated from the harsh critic's near-fatal framing.

- **Human Finder: "Datasets are not intrinsically higher-order, limiting conclusions about TDL benefits."** Removed because this is a universal limitation of the TDL benchmarking ecosystem, not a flaw specific to this paper. The paper does not claim to prove that topology helps; it claims to provide a better framework for TDL architectures. Moreover, all competitors in Table 1 use the same lifted datasets, so the comparison is internally consistent.

- **Spark Reviewer: "Comparison against plain GNNs on original graphs is missing and the value of the topological pipeline is unsubstantiated."** Removed as a weakness. The paper's stated goal is to provide a principled way to generalize GNNs to TDL counterparts, not to argue that TDL outperforms plain GNNs. Evaluating the paper on whether it proves TDL generally superior is scope creep.

- **Harsh Critic: "the correctness of Proposition 1–3 cannot be verified since Appendix B is removed."** Removed per hard rule—the appendix exists in the full paper; the parsing artifact of the text file ending at "Rest of paper (reference and Appendix) is removed" does not mean the appendix was not submitted. Questioning the existence of appendix content is a reviewer knowledge gap.

---

## Novel Insights

The most genuinely novel observation across all three reviewers (confirmed by the paper) is the distinction between *single* augmented Hasse graphs (which collapse all neighborhood and rank distinctions into one graph) and an *ensemble* of strictly augmented Hasse graphs (one per neighborhood function). Prior work by Jogl et al. and Hajij et al. operated on the collapsed representation and acknowledged the loss of topological symmetry. The GCCN ensemble construction is the first to recover full cell permutation equivariance in a graph-based TDL framework while simultaneously enabling non-message-passing neighborhood modules. The introduction of per-rank neighborhoods as a first-class modeling concept—allowing selective message passing across ranks—is a small but practically impactful contribution that reduces model size without sacrificing expressivity on several benchmarks.

---

## Suggestions

1. **Resolve the expressivity inconsistency immediately**: pick either "at least as expressive" (which follows from Prop 1 alone) or "strictly more expressive" (which requires a proper separation proof), and make this consistent between the abstract, intro bullet, and Proposition 3. Add a concrete example of a distinguishable complex pair to justify the strict claim.

2. **Add hypergraph GCCN experiments**: run at least one GCCN configuration on the hypergraph domain and include results in Table 1. This directly validates a central claimed property of the framework.

3. **Add Proposition 2's conditions to all high-level descriptions**: both the abstract and the intro contribution bullet should say "are conditionally cell permutation equivariant (when ω_N is equivariant and ⊗ is invariant)" to avoid overclaiming.

4. **Strengthen the comparison analysis**: either increase seeds (from 5 to 10+) for high-variance datasets like MUTAG, or report bootstrap/Wilcoxon significance intervals when claiming "outperforms by >1σ." Alternatively, frame Table 1's conclusion more carefully as showing the GCCN family is competitive rather than consistently superior.

5. **Add a controlled ablation for ensemble vs. single-graph decomposition**: same ω_N, same parameter budget, same dataset—varying only single-union augmented Hasse graph vs. per-neighborhood ensemble. This would isolate the core architectural contribution of the paper cleanly.

---

## Overall Assessment

**Originality:** Moderate-high. The combination of strictly-per-neighborhood graph decomposition, per-rank neighborhoods, and plug-and-play GNN modules is a fresh synthesis; individual components are not new, but the integration addresses a real gap.

**Importance of research question:** High. Standardizing TDL architecture development is a genuine bottleneck as identified by Papamarkou et al. (2024); TopoTune directly targets this.

**Support for claims:** Mixed. The framework and software claims are well-supported. The theoretical claims are plausible but imprecisely stated (expressivity inconsistency). The empirical claim of "consistent outperformance" is overstated given the comparison design.

**Soundness of experiments:** Moderate. Good breadth but the missing hypergraph GCCN results, high variance on MUTAG, and the large search space vs. fixed baseline issue limit confidence in the stronger empirical claims.

**Clarity:** Generally clear with the notable exception of the expressivity inconsistency and the GCCN/GCNN/GCN terminology drift.

**Value to the research community:** Solid. TopoTune is likely to be adopted and cited; the GCCN framework gives TDL practitioners a principled, accessible starting point for architectural exploration.

## Score and Decision

The paper makes a concrete and useful framework contribution with sound theoretical foundations (modulo the expressivity inconsistency), a practical software artifact, and reasonable empirical evidence of competitiveness. The weaknesses—expressivity presentation inconsistency, missing hypergraph GCCN results, and over-interpreted empirical comparisons—are real but addressable without fundamental redesign. The core contribution is original and valuable for the TDL community.

**Score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>