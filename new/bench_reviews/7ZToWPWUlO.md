## Summary

The paper proposes Wedge and Ring Transformers (WRT), a Transformer+PPO-based method for graph partitioning that explicitly constrains partitions to ring and wedge shapes around a chosen center on planar graphs. It introduces polar-coordinate–based graph transformations, a two-stage RL training scheme, and Cheeger-type bounds for ring/wedge partitions on idealized spider-web graphs, and reports strong empirical performance versus METIS, spectral clustering, ClusterNet, and NeuroCUT on synthetic “spider web” graphs and a real traffic network.

## Strengths

- **Clear, application-driven motivation and problem formulation.** The paper identifies a concrete, underexplored setting: when partitions must respect domain-specific shapes (rings/wedges) as in road network simulation. Section 3 formalizes ring and wedge partitions and a ring–wedge partition pipeline (Fig. 2) that matches the traffic-simulation story well.

- **Clever polar-coordinate graph transformations enabling Transformers.** The ring and wedge transformations (Sec. 5.2, Fig. 3) reduce the partition problem to 1D/circular orderings, making it natural to feed into a Transformer while preserving the equivalence of ring/wedge partitions. This is a neat, technically sound idea that cleanly exploits the geometry of the problem.

- **Thoughtful RL and architecture design.**  
  - The two-stage RL training strategy (Sec. 5.5.1) – train wedges with randomized rings and wedge-only reward, then fix the wedge policy and train rings – directly targets policy interference and sparse-reward issues in this coupled decision problem. This is a nontrivial, well-motivated design.  
  - Partition Aware MHA (PAMHA, Sec. 5.4, Fig. 4b) uses the precomputed volume matrix and current partition to mask attention to the relevant subgraph, a reasonable and problem-aware modification of standard MHA.  
  - The integration of dynamic programming for ring partitioning (Sec. 5.3) inside the RL loop is also an interesting hybrid design.

- **Strong quantitative results on the chosen distributions.**  
  - On all three dataset types (predefined-weight, random-weight, city traffic) and for multiple sizes and partition counts, WRT consistently achieves the lowest normalized cut among the methods tested (Table 1), and shows good transfer across graph sizes (Table 2).  
  - On the city graphs, WRT also achieves the highest Ringness and Wedgeness (Table 3), aligning performance with the stated shape constraints.

- **Additional theoretical perspective.** Proposition 1 in Sec. 4 establishes Cheeger-type upper bounds for normalized cut under ring and wedge partitions on unweighted spider-web graphs, providing at least some theoretical context for why ring/wedge structures are not pathological with respect to NC.

- **Clarity of exposition for the core method.** The methodology section is generally well written: the environment, transformations, WRT architecture, and training/testing pipeline (Fig. 4) are described clearly enough that a reader familiar with RL and Transformers could plausibly reimplement the approach.

## Weaknesses

### Fatal

None rise to the level of “not even a paper” or a clear mathematical flaw, but there are structural issues in the empirical evaluation that, in my view, undermine the central comparative claims.

### Major

- **Mismatch between constrained task and unconstrained baselines in the main claims.**  
  The problem is explicitly formulated as *constrained* normalized cut: “we restrict our attention to partitions … where each partition is either ring-shaped or wedge-shaped” (Sec. 3). WRT hard-codes this via its action space and transformations, and all synthetic data is spider-web–like. Yet the headline empirical claim is that WRT “outperforms existing RL-based and traditional methods” on normalized cut (Abstract, Sec. 6.3.1) compared to METIS, spectral clustering, ClusterNet, and NeuroCUT – all run in the **unconstrained** partition space. Those baselines are neither restricted to ring/wedge shapes nor given access to the polar structure.  

  On such data, WRT’s gains in normalized cut are very plausibly explained by its strong shape prior being aligned with the graph family, rather than by its learning algorithm being intrinsically better at optimizing NC in a fair setting. If the true task is “best NC under ring–wedge constraints,” fair baselines should operate in that same constrained family; if the task is unconstrained NC, WRT’s action space should not be restricted to rings/wedges. As it stands, the evaluation supports a weaker, but accurate, statement: *given graphs strongly shaped like rings/wedges, and if we want ring/wedge partitions, then a model architecturally restricted to ring/wedge partitions does better (both in NC and in ringness/wedgeness) than generic, unconstrained methods.* It does **not** convincingly show that WRT is a superior general normalized-cut optimizer.

- **Synthetic “predefined-weight” graphs are tuned to WRT’s inductive bias.**  
  In Sec. 6.1, Predefined-weight Graphs are generated by first sampling a valid ring–wedge partition, then assigning “lower weights to edges that cross different partitions and higher weights to edges within the same partition.” This directly encodes the ground-truth ring–wedge structure into the weights. WRT is architecturally constrained to this class, and trained on hundreds of thousands of such graphs. Baselines are not informed of this generative model or tailored to it. Unsurprisingly, Table 1 then shows large margins for WRT on this distribution.  

  That setup strongly biases the evaluation: the data is constructed so that the optimal partition lies in precisely the family enforced by WRT. The resulting superiority mostly demonstrates “a model whose hypothesis class matches the data generator, trained on that data, beats generic baselines on that distribution,” which is useful, but weaker than the claims made. This limits how much weight can be placed on the predefined-weight results as evidence of algorithmic strength.

- **No strong algorithmic baseline restricted to the same ring–wedge family.**  
  The key conceptual contribution is the shape-constrained partitioning scheme; however, the only “shape-aware” baselines are described as:
  - **Bruteforce**: “enumerate possible ring and wedge partitions,” but then the text also says it and Random “do not consider the differences of edge weights, and only do random partitioning” (Sec. 6.3.1), which is contradictory and suggests the brute force baseline is intentionally handicapped.  
  - **Random**: 10,000 random ring–wedge partitions, also described as ignoring edge weights structurally.  

  There is no well-designed, weight-aware search heuristic over ring–wedge partitions (e.g., DP or greedy over radii/angles) used as a baseline – despite DP being implemented internally in Sec. 5.3. Without at least one strong non-learning baseline in the **same constrained search space** that fully exploits weights, it’s impossible to separate “benefit from shape restriction” from “benefit from having any decent optimizer in that space.” This confounds the claimed contribution of the RL+Transformer machinery.

- **Theoretical section is largely disconnected from the practical method and data.**  
  Sec. 4 proves Cheeger-type bounds for unweighted spider-web graphs with exact regular structure and a specialized version of NC. WRT operates on **weighted**, often irregular graphs (random weights, real traffic graphs), and there is no indication that the algorithm uses any spectral information or that these theoretical bounds meaningfully predict or constrain WRT’s behavior in practice. The paper asserts that spider-web graphs “give a theoretical justification of the normalized cut definition and the ring-wedge shaped partition,” but there is no bridge from the simplified graph model to the weighted, noisy, and possibly non-ideal real graphs. The result is mathematically fine but feels bolted on rather than supporting the central algorithmic claims.

- **Real-graph setting and shape prior suitability are not convincingly justified.**  
  The method requires: (i) an underlying planar graph, and (ii) a meaningful center o for ring/wedge construction. In Sec. 6.1, the real dataset is “sub-graphs randomly extracted from a comprehensive city traffic map,” but the paper does not demonstrate that these subgraphs truly resemble spider-web structures, are planar in the graph-theoretic sense, or possess a unique meaningful center. Yet the evaluation emphasizes high Ringness/Wedgeness as inherently positive (Table 3), which by construction favors the inductive bias. There is no analysis of cases where the polar decomposition is a poor fit, nor of sensitivity to the choice of center o. This weakens the claim of robust utility for “real city traffic graphs.”

### Minor

- **Normalization and Cheeger connection not fully justified.**  
  The normalized cut is defined as the maximum over partitions (Eq. 2), rather than the more standard sum. This is a legitimate design choice, particularly if one cares about the worst partition in parallel simulation, but the paper does not articulate why max is preferable to sum for the stated use case, nor how this aligns with their Cheeger-like interpretation.

- **“Fuzzy” rings/wedges are under-specified.**  
  Sec. 3 mentions allowing some nodes to be swapped to adjacent partitions to reduce NC, with Figure 4(c) (Stage ④) illustrating a local correction, but this is described in prose only. There is no formal constraint on how many nodes can be “fuzzy,” nor a careful analysis of how post-refinement deviates from pure ring/wedge partitions. For a paper whose core appeal is explicit shape control, a more precise treatment of these deviations would help.

- **Handling of weighted baselines is unclear.**  
  Sec. 2.2 states that NeuroCUT and ClusterNet “do not handle weighted graphs, making them unsuitable in our scenarios,” yet those same methods are used as baselines on weighted graphs (Table 1) without clear description of how they are adapted. If they ignore weights or only use them as features without objective alignment, their competitive standing is ambiguous. The paper does not appear to misrepresent their results numerically, but the fairness and interpretation of these comparisons are under-explained.

- **Key metric definitions are in the appendix.**  
  Ringness and Wedgeness are central to the narrative (Fig. 1, Table 3) but defined only in the appendix; readers cannot assess their properties, scale, or robustness from the main text. Bringing at least the core formulas and some intuition into Sec. 3 would improve clarity.

- **Scalability and runtime not discussed.**  
  Experiments are on fairly modest sizes (e.g., up to 200 nodes per ring; total node counts are not heavily stressed), and there is no reporting of wall-clock time or memory usage versus METIS or spectral clustering. Given that the method combines a Transformer and an O(n²k) DP subroutine, more explicit discussion of scaling to large city-wide graphs would be valuable.

### Trivial

- Some typos, minor grammar issues, and repetition in figure captions (e.g., duplicated Figure 4 description) are present but do not affect scientific content.

## Nice-to-Haves

- A sensitivity study for the choice of center o on real graphs, showing how NC and Ringness/Wedgeness vary if the center is perturbed or chosen via different heuristics.

- Visualization of failure modes: examples where WRT’s ring–wedge constraint yields notably worse NC than an unconstrained optimum, to clarify the trade-off between shape control and cut quality.

- Ablation of Partition Aware MHA vs vanilla MHA to quantify how much benefit the structural attention mask contributes.

- Explicit use of the internal ring DP algorithm as a standalone heuristic baseline in the same constrained space, to clarify what RL+Transformer adds on top of DP.

## Removed Points

These points are flagged to be removed; treat them with caution if encountered elsewhere.

- **Claims that models/datasets/baselines “do not exist” or are unreleased.** None of the reviewers made these, and under the instructions they would be inappropriate if they had.

- **Criticism asserting that NeuroCUT or ClusterNet “cannot be run” on weighted graphs.** While the paper notes they are not designed for weighted graphs, it *does* provide results for adapted versions; any suggestion that they cannot be used at all would contradict the text and is removed.

- **Overly speculative concerns about exact planarity violations (e.g., overpasses making the traffic graph non-planar in a strict sense).** The paper treats its city subgraphs as planar for modeling; unless contradicted internally, questioning that assumption in detail would exceed what can be established from the text alone.

- **Extreme statements that the Cheeger bounds are “wrong” or mathematically invalid.** The paper states the proposition and refers to proofs in the appendix; without seeing the proofs, we cannot declare them incorrect, only note the limited scope and missing connections.

## Novel Insights

The genuinely interesting conceptual move in this paper is to recast a structured graph-partitioning problem into a 1D (ring) or circular (wedge) sequence via polar-coordinate transformations, thereby enabling the use of standard sequence models like Transformers on a combinatorial graph problem with constrained shapes. Combined with a carefully staged RL training scheme to handle coupled ring and wedge decisions, this illustrates a broader pattern: when the application domain implies strong geometric priors, aggressively encoding those into the action space and representation can substantially simplify learning, at the cost of generality. The paper stops short of fully disentangling how much of the observed gain comes from the prior vs the optimizer, but it offers a concrete, well-engineered example of this design philosophy.

## Suggestions

- **Reframe the core claims to emphasize the constrained setting.** Rather than positioning WRT as a generally superior NC optimizer, explicitly state that it targets *shape-constrained* partitioning on graphs with approximate radial structure, and that its advantage stems from exploiting this prior. This would bring the claims in line with what the experiments actually support.

- **Add at least one strong, weight-aware baseline in the same ring–wedge search space.**  
  For example, use the dynamic programming scheme from Sec. 5.3 + a simple outer heuristic over radii/angles, or a greedy algorithm, as a separate baseline. Ensure it fully uses edge weights and operates over the same constrained family as WRT. This would greatly clarify the incremental value of RL+Transformer vis-à-vis algorithmic search within the ring–wedge parameterization.

- **Revisit the synthetic data design.** To avoid tautological advantages, consider additional synthetic distributions where the optimal partition is *not* exactly ring–wedge, or where the weights are not constructed from a ground-truth ring–wedge partition. Alternatively, clearly separate “in-distribution” (generator-aligned) and “off-distribution” experiments, and temper claims accordingly.

- **Move definitions and key experimental details into the main text.** Include formulas and some intuition for Ringness/Wedgeness in Sec. 3; summarize the graph size and parameter ranges for the synthetic and city graphs in the main body (not just Table 4 in the appendix). Clarify how NeuroCUT and ClusterNet are adapted to handle weighted graphs.

- **Clarify the role and impact of theory.** Either (a) strengthen the connection between the Cheeger bounds and the algorithm (e.g., show experiments on unweighted spider-web graphs where the empirical NC closely tracks the bounds, or use spectra in some design choice), or (b) present the theory more modestly as a side note limited to an idealized model, without implying it justifies behavior on real weighted graphs.

- **Provide some scalability numbers.** Report runtime and memory usage versus METIS and spectral clustering for the largest graphs you can handle, and discuss how sequence length (number of nodes) affects attention cost and DP overhead. This will help readers assess whether WRT is practical for large-scale traffic simulations.

- **Analyze robustness to center selection.** For the city graphs, show how performance changes if the center is chosen differently (e.g., geometric centroid vs traffic hub vs random) to give practitioners guidance on deploying the method.

## Score and Decision

For calibration, I compared with:

- **FneYHZU19U – “Constrained Graph Clustering with Signed Laplacians.”** This paper had nontrivial theory (Cheeger inequality) but weak baseline comparisons and limited practical impact; human scores were around 3–6 with a reject decision. The present paper is somewhat similar: interesting constrained partition setting and some theory, but evaluation and baselines are not fully convincing.

- **gCSEQIgbWH – RL+GNN for the k-server problem, “Generalist Policy for k-Server Problem.”** This used standard RL+GNN machinery for a new CO setting, with concerns about limited novelty and incomplete ablations; scores were around 3–5 and rejected. Our paper has a somewhat stronger domain story and more careful architecture design (two-stage RL, PAMHA), so I view it as slightly stronger than that anchor.

- **CpiJWKFdHN – “ROS: GNN-based Max-k-Cut.”** Also in RL/graph CO, with reasonable results but missing baselines and insufficient ablations; scores roughly 5–6 and rejected. Relative to that, this submission has a more specialized, well-motivated constrained setting but more problematic evaluation bias (synthetic generator aligned to model) and lack of fair constrained baselines. I judge it slightly weaker overall in evidential rigor.

Positioning between these, I see this as a **borderline but ultimately below-threshold paper**: interesting idea and solid engineering, but with evaluation that does not yet substantiate the strongest claims and lacks key like-for-like baselines. I therefore lean to a reject with a mid-range score.

MY FINAL SCORE: <pineapple>5.0</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>