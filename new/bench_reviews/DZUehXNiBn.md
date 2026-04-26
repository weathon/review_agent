Now I have enough context to write the review. Let me synthesize everything.

## Summary

VISTA proposes a modular, model-agnostic framework for large-scale causal structure learning that decomposes the global DAG problem into local Markov Blanket subgraphs, aggregates edge-level votes using a weighted voting scheme with exponential confidence weighting, and enforces acyclicity via a Feedback Arc Set heuristic. The paper provides theoretical finite-sample error bounds and asymptotic consistency guarantees for the voting mechanism under independence assumptions, and demonstrates empirical improvements in F1 and runtime across multiple base learners on synthetic and real data.

## Strengths

- **Modular, practically useful framework**: VISTA genuinely decomposes causal discovery into parallel subproblems. Table 3 shows substantial and consistent runtime speedups (3–5x for most methods at n=300; SCORE from >10,000s to 199s), which is a clear engineering contribution for scalability.

- **Meaningful F1 improvements for some base learners**: In Table 2 (normalized setting), VISTA-WV improves F1 notably for GOLEM (0.35→0.60 on ER5), DAG-GNN (0.35→0.59), and NOTEARS (0.76→0.79). On the Sachs dataset (Table 4), FDR drops consistently across all methods.

- **Proposition 3.1 is clean and correct**: The coverage guarantee that the union of MB subgraphs contains all true edges under correct MB identification is a simple, useful result that justifies the divide step.

- **Principled aggregation mechanism**: The weighted voting score s(X→Y) = (1−e^{−λm})·A/m provides a theoretically motivated way to penalize low-support edges, and Theorem 3.4 derives a feasible interval for λ.

## Weaknesses

### Major

- **Theory-practice disconnect on independence assumption**: Theorems 3.2 and 3.5 derive finite-sample bounds and asymptotic consistency assuming votes across local subgraphs are independent (A ∼ Binomial(m, p)). The paper acknowledges on p. 8 that "subgraphs learned from the same dataset can induce correlations among votes." Since all subgraphs are estimated from overlapping variable sets drawn from the same observational data, these correlations are structurally inherent—not a corner case. The paper frames the bounds as "qualitative guidance," but the theoretical claims (finite-sample error bounds, asymptotic consistency) are the primary theoretical contribution and they formally do not apply to the deployed method. No sensitivity analysis or argument about the degree of dependence is provided to assess how loose the bounds become.

- **Naive Voting inflates false positives massively, and it is unclear whether WV's gains come from principled aggregation or from compensating for artifacts introduced by the divide step**: In Table 1, VISTA-NV increases NOTEARS FDR from 0.21 to 0.87 (ER5) while dropping F1 from 0.76 to 0.23. Similar catastrophic degradations occur for GOLEM, DAG-GNN, and SCORE. VISTA-WV then recovers performance primarily by suppressing these inflated false positives. This pattern suggests the MB decomposition introduces systematic spurious edges (via confounding from variable subsetting, which the paper explicitly acknowledges), and WV's exponential weighting + threshold filtering partially compensates. The critical missing ablation is: what happens if one applies the same weighted-voting-and-thresholding directly to base learner outputs on the full variable set (without decomposition)? Without this, one cannot determine whether the divide step contributes any value beyond creating artifacts that the conquer step then fixes.

- **Missing comparisons to simpler aggregation baselines**: VISTA simultaneously introduces MB decomposition, weighted voting, and FAS-based acyclicity enforcement. No ablation isolates the contribution of each component, and no comparison is made to simpler alternatives like majority voting with a threshold, simple ensemble averaging, or applying exponential weighting directly to base learner outputs. The DCILP comparison is relegated to an appendix. This makes it impossible to assess whether the specific design choices of VISTA matter versus whether any reasonable aggregation+filtering would produce similar gains.

### Minor

- **Overclaimed "model-agnostic" property**: The paper states VISTA "imposes no assumptions on the inductive biases of base learners" (Abstract) and is "strictly model-agnostic" (p. 15). However, the framework requires base learners to output directed edges; undirected adjacencies are discarded ("treated as providing no directional vote," p. 73). This exclusion of constraint-based methods that output CPDAGs is a meaningful limitation. The claim should be qualified.

- **Fixed λ=0.5 may not satisfy Theorem 3.4 conditions for all edges**: The feasible interval for λ in Eq. (5) depends on m (number of subgraphs containing the edge), t, and ε. Since different edges have different m values, no single λ satisfies the condition for all edges simultaneously. The paper uses λ=0.5 globally without discussing edges with very small m, where the theoretical condition could be violated.

- **Mixed results on Sachs and for some base learners**: On the Sachs dataset, SCORE loses TPR (0.18→0.12) and GraN-DAG loses TPR (0.53→0.29) with VISTA. For GraN-DAG and SCORE in Table 1 (the better-performing standalone baselines), VISTA-WV actually reduces F1 compared to the standalone baseline (SCORE: 0.14→0.09 on ER5 with WV, though F1 improves slightly on SF5). The largest gains come from weaker baselines, raising questions about how VISTA behaves when the base learner is already strong.

- **Large standard deviations in Table 1**: Some entries have enormous variability (e.g., NOTEARS SHD 208.80 ± 199.71 on ER5; SCORE FDR 0.92 ± 0.10). The conclusions about consistent improvements are not supported by statistical significance testing.

### Trivial

- None significant.

## Nice-to-Haves

- Analysis of vote correlation structure across subgraphs would help bridge the theory-practice gap and assess how loose the independence-based bounds are in practice.
- Ablation without MB decomposition (applying WV + thresholding directly to base learner outputs on the full variable set) to isolate the contribution of the divide step.
- Analysis of how VISTA's performance degrades with MB estimation errors, beyond the "relatively stable" observation in Figure 1.
- Extension to handle undirected edge outputs from constraint-based methods.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that WV only filters artifacts from the divide step without evidence**: While the missing ablation makes this a valid major concern, the Harsh Critic's assertion that WV "merely filters" artifacts goes beyond what can be concluded from the evidence—it's a plausible hypothesis, not an established fact. The proper framing is that the question cannot be answered without the ablation.

- **Proposition 3.1 assumes correct MB identification**: This is true but standard across divide-and-conquer causal discovery methods. The paper acknowledges this and empirically investigates MB accuracy in Figure 1. This is a limitation, not a fatal flaw.

- **The λ conflates frequency with confidence**: The Harsh Critic argues that WV conflates vote frequency and confidence. However, this is an intentional design choice—the paper explicitly formulates s(X→Y) = (1−e^{−λm})·(A/m) to jointly penalize low-support (small m) and low-agreement (low A/m) edges. Whether this is "conflation" or "design" is debatable; it's a reasonable engineering choice.

- **Harsh Critic's claim that Table 4 Sachs results are "marginal" and "a weak test"**: The Sachs network is a small (11 node) standard benchmark. While not the ideal test of large-scale efficiency, it's one of the few real-world benchmarks available, and showing consistent FDR reduction across methods is informative.

- **Harsh Critic's critique about "systematic biases in specific directions that voting would amplify rather than cancel"**: This is speculative speculation about failure modes not demonstrated empirically.

- **Strength Finder's claim about "50-80% FDR reduction"**: This is somewhat misleading as the absolute FDR values and trade-offs with TPR matter. For strong baselines (SCORE, GraN-DAG), the F1 improvements are inconsistent. This strength is partially tempered.

## Novel Insights

The paper's most insightful observation—which is not fully developed—is that the divide step in Markov-Blanket-based causal discovery systematically introduces latent confounding (since variable subsetting creates unobserved parents), and that weighted voting with FAS filtering can partially compensate for this. This creates an inherent tension: the decomposition that enables scalability also creates systematic errors that the aggregation must fix. Whether this cycle ultimately produces net positive value depends on whether the MB-based decomposition provides enough true signal to overcome the confounding noise, which the current experiments cannot definitively answer due to the missing ablation.

## Suggestions

- Add the critical ablation: apply WV + threshold filtering directly to base learner outputs on the full variable set (without MB decomposition). This is the single most important experiment to validate the framework's core claim.
- Qualify the "model-agnostic" claim to acknowledge the directed-edge requirement, e.g., "model-agnostic for base learners that produce directed edge outputs."
- Discuss the per-edge λ feasibility issue and its practical implications, or show empirically that the global λ=0.5 is within the feasible interval for the vast majority of edges.
- Add at least one comparison to a simpler aggregation baseline (e.g., uniform averaging + threshold) to demonstrate the value of the specific WV formulation.

## Score and Decision

**Calibration anchors:**

1. **UAkVjK00Wv** (avg 4.75, Reject): Divide-and-conquer ensemble for BN structure learning—similar topic, but more incremental and with less theoretical grounding. VISTA is more rigorous but has the same core class of criticism (insufficient ablation, limited novelty).

2. **grM2Yv49cI** (avg 6.0, Accept-Poster): Model aggregation framework (MEVA) that is model-agnostic and demonstrates empirical gains across applications. Has similar weaknesses about assumption validity and limited comparisons. VISTA is comparable in its model-agnostic aggregation framing but has a larger theory-practice gap.

3. **or8wkKoBP4** (avg 4.0, Reject): Causal structure learning with impractical theoretical assumptions (minimal dependence faithfulness) and no experimental evaluation. VISTA is much stronger empirically but shares the concern about theory not matching practice.

4. **Js5PJPHDyY** (avg 6.0, Accept-Poster): Simple training-free baseline with strong empirical results but questions about fair comparison. VISTA has more substantive contributions but also more serious theoretical concerns.

5. **KstDMYkfj4** (avg 3.8, Reject): Theory paper where the main theoretical claims are "basically vacuous" for practical settings. VISTA is not this bad—the theory provides qualitative guidance—but the independence assumption gap is real.

The paper's core contribution—a practical, modular framework for scaling causal discovery—is genuine and useful. The empirical results show real improvements, particularly for weaker base learners. However, the theoretical claims rest on an independence assumption that is structurally violated, and the experimental validation cannot distinguish VISTA's value from post-hoc filtering due to missing ablations. This is a paper with a real and useful idea, but with significant gaps in both theory and empirical attribution. Compared to the 4.75-scoring Auto-Ensemble BN paper (similar topic, similar issues), VISTA has more theoretical depth but also more serious theory-practice disconnect. Compared to MEVA at 6.0, VISTA has a larger gap between claims and evidence.

I place this paper slightly below the borderline. The framework is practical and the runtime gains are real, but the accuracy improvements are inconsistently demonstrated and not well-attributed, and the theoretical contribution has a fundamental limitation.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>