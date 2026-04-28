## Summary
This paper proposes a unifying axiomatic framework that models simplicial complexes as relational structures, enabling the extension of graph-theoretic tools (specifically oversquashing analysis and rewiring heuristics) to topological deep learning. The theoretical contributions include sensitivity bounds (Lemma 3.2), extended Forman curvature for directed weighted graphs (Proposition 3.4), and depth analysis (Theorem 3.5). The practical contribution is a relational rewiring algorithm that adapts graph rewiring techniques to higher-order structures via a collapsed adjacency matrix.

## Strengths
- **Unified formalism bridging GNN and TDL**: Section 2 establishes a rigorous mapping between simplicial message passing and relational structures (Definitions 2.4–2.5), successfully bridging notation gaps between communities. This enables the systematic extension of GNN sensitivity analysis to higher-order domains where prior methods do not directly apply, as demonstrated by Lemma 3.2 (Equation 8) which derives sensitivity bounds using the augmented influence matrix **B**.

- **Novel theoretical extensions for higher-order oversquashing**: The paper derives specific analytical results addressing a genuine gap in TDL literature. Theorem 3.5 (Equation 12) quantifies exponential sensitivity decay based on combinatorial distance in the influence graph, and Proposition 3.4 (Equation 10) defines an extended Forman curvature for weighted directed relational graphs—tools that prior topological deep learning work lacked.

- **Problem relevance**: Addressing oversquashing in TDL is a significant open problem, as topological layers often exacerbate path lengths compared to the 1-skeleton. The paper directly addresses research directions 2 and 9 from Papamakarios et al. (2024) as claimed.

## Weaknesses

### Fatal
None

### Major
- **Theory-experiment mismatch in synthetic validation**: The theoretical analysis (Section 3) is explicitly motivated by simplicial message passing (SIN, CIN), yet the primary synthetic validation (Section 5.2, RINGTRANSFER) uses RGCN on lifted graphs rather than simplicial architectures. While Remark 2.7 correctly states the relational framework encompasses RGCN, RGCN treats relations as independent channels without the weight-sharing and dimensional coupling inherent to simplicial operators (Eq. 1). Consequently, the experiments do not directly verify that *simplicial* oversquashing behaves as the theory predicts. This gap between the simplicial motivation and RGCN-based validation undermines the claim that the theory explains simplicial-specific phenomena.

- **Overstated empirical claims about rewiring effectiveness**: The paper claims "rewiring generally boosts performance... with relational and topological models performance responding to rewiring similarly to graph models" (Section 5.1), but Table 1 shows rewiring *decreases* performance for topological models in multiple cases (SIN on ENZYMES: 47.5→46.8; CIN on ENZYMES: 50.0→49.9; CIN++ on MUTAG with Ring lifting: 90.5→84.5). The "Best Rew" column also obscures which algorithm (SDRF, FoSR, or AFRC) was selected per dataset, preventing assessment of the proposed heuristic's reliability and suggesting potential cherry-picking.

### Minor
- **Tension between "topological" framing and relational rewiring**: Algorithm 1 (Section 4) adds new relations R_{k+1} based on a collapsed graph without respecting simplicial incidence constraints. While valid within the relational structure formalism, this approach effectively abandons topological constraints to fix a topological problem. The paper should more explicitly acknowledge that this step transforms the simplicial complex into a general relational structure, which may concern readers expecting topology-preserving methods.

- **Inconsistent citation naming**: Section 1 references "Papamakarios et al., 2024" while Section 6 spells it "Papamakou et al." This inconsistency should be corrected.

### Trivial
- **Clarification needed on "Best Rew" selection**: The paper should explicitly state whether "Best Rew" represents a single fixed algorithm or a dataset-specific oracle selection. If the latter, reporting results for each algorithm individually would improve reproducibility.

## Nice-to-Haves
- **Ablation of added relation types**: Reporting what kinds of relations Algorithm 1 adds (e.g., vertex-to-vertex vs. vertex-to-triangle) would help assess whether the method preserves any topological meaning and guide future topology-preserving variants.

- **Analysis of rewiring failures**: Investigating why rewiring decreases performance for SIN/CIN on certain datasets (e.g., ENZYMES) would strengthen the paper—distinguishing whether this is due to over-squashing of noise or disruption of topological features.

- **Visualization of rewired influence graphs**: Showing before/after influence graphs for small complexes would reveal how the topology is altered by new relations and aid intuition.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point 1 (Structural: rewiring violates topological constraints)**: Removed. The paper explicitly frames its contribution as viewing simplicial complexes *through the lens of relational structures* (Section 2, Takeaway Message 1). The rewiring algorithm operates on this relational view, not on the simplicial complex directly. The critic misunderstands the paper's scope—the paper does not claim to preserve simplicial incidence axioms during rewiring; it claims to extend the relational framework to enable graph techniques. This is a feature, not a bug, of the proposed approach.

- **Strength Finder Point 3 (Effective adaptation of rewiring heuristics)**: Removed. This strength conflicts with verified weaknesses. Table 1 shows rewiring often harms topological model performance, and the "Best Rew" column's opacity prevents claiming the adaptation is "effective." The empirical evidence does not support this strength claim.

- **Harsh Critic Point on Remark 2.6 (weighted vs. binary relations)**: Removed as a trivial nitpick. The paper's relational framework naturally accommodates weighted relations; this is standard in the shift operator literature (Remark 2.6 cites Mateos et al., 2019; Gama et al., 2020). The base complex typically uses binary weights, but the framework's generality is a strength.

- **Harsh Critic Point on Influence Graph losing directional semantics**: Removed as scope creep. The influence graph aggregates all relations for sensitivity analysis, which is appropriate for bounding worst-case information flow. The paper's theory is about *whether* information can propagate, not the directional semantics of specific message types. This is a design choice, not a flaw.

## Novel Insights
The paper's core insight—that simplicial complexes can be rigorously modeled as relational structures to enable transfer of GNN theoretical tools—is genuinely novel and valuable for the TDL community. However, the tension this creates (using relational rewiring that abandons topological constraints to solve a topological problem) reveals a deeper question: whether "topological" deep learning should prioritize preserving topological structure or leveraging topological representations as intermediate formalisms. The paper implicitly argues for the latter, but this philosophical stance deserves more explicit discussion.

## Suggestions
1. **Run RINGTRANSFER with SIN/CIN**: The synthetic benchmark should evaluate simplicial architectures (SIN, CIN) in addition to RGCN to validate that simplicial-specific oversquashing behaves as predicted by the theory.

2. **Clarify "Best Rew" methodology**: Explicitly state whether results use a fixed rewiring algorithm or per-dataset selection. If the latter, report individual algorithm results to enable reproducibility.

3. **Reframe empirical claims**: Temper claims about rewiring effectiveness for topological models given the mixed results in Table 1. Acknowledge cases where rewiring harms performance and discuss potential causes.

4. **Discuss topology preservation trade-offs**: Add a paragraph in Section 4 or 6 explicitly acknowledging that Algorithm 1 does not preserve simplicial incidence constraints, and discuss when this is acceptable versus when topology-preserving rewiring would be preferable.

---

## Calibration and Scoring

I calibrated this paper against the following anchors:

**High-scoring anchors (avg ≥ 6):**
- **QYtmqCoilk.md (6.80)**: Directly addresses oversquashing with theoretical counterexamples and proposes a new curvature metric with strong empirical validation. This paper has stronger empirical backing than the current submission.
- **YR3CNvFfCr.md (6.67)**: Introduces Semi-Simplicial Neural Networks with theoretical expressivity proofs and comprehensive experiments on 13 datasets. More complete theory-experiment alignment than the current paper.
- **1taAXRcm21.md (6.00)**: Unifies diffusion frameworks with experimental gains; reviewers noted the genetics-simplex connection lacked intuitive exposition but accepted due to strong theory.

**Medium-scoring anchors (avg ~5):**
- **AHpexliCTM.md (5.50)**: Cooperative Sheaf NNs with receptive-field proofs but lacks stability analysis and clear novelty boundaries. Similar theory-experiment gap as the current paper.
- **eC89CbINIw.md (5.33)**: Learnable graph lifting with inconsistent benefits on heterophilic datasets; reviewers questioned why TNN baselines don't outperform GNNs. Similar mixed empirical results.
- **VxBh4rtg9Y.md (5.00)**: Theoretical sensitivity bounds against oversquashing but weak empirical case studies that don't clearly demonstrate theoretical guarantees. Directly comparable weakness pattern.

**Low-scoring anchors (avg ≤ 4):**
- **5RbpF0U3aQ.md (4.00)**: GNN long-range diagnostics with missing long-range benchmarks, weak correlation results, and insufficient baseline comparisons. More severe empirical gaps than current paper.
- **Q3MisVkuTu.md (4.00)**: Demystifies oversquashing beliefs via counterexamples but uses only toy problems with insufficient experimental analysis. Similar theory-heavy, empirical-light pattern.
- **qtjLwGTvGR.md (3.50)**: HKS-based topological learning with missing related works and insufficient experiments on higher-order complexes. More severe empirical shortcomings.

**Positioning**: This paper's strength pattern (rigorous theoretical framework, unified formalism) matches the high-scoring anchors (QYtmqCoilk, YR3CNvFfCr), but its weakness pattern (theory-experiment mismatch, mixed/overstated empirical results) aligns with medium-scoring anchors (VxBh4rtg9Y, AHpexliCTM). Unlike the low-scoring anchors, this paper has genuine theoretical contributions and a coherent framework—the empirical issues are inconsistencies rather than fundamental absences.

The paper is stronger than Q3MisVkuTu (4.0) because it provides novel theoretical bounds rather than just counterexamples, and has more substantial real-world experiments. It is weaker than QYtmqCoilk (6.8) because the empirical validation does not match the theoretical ambition (RGCN vs. simplicial models, mixed rewiring results). The closest match is VxBh4rtg9Y (5.0), which also has strong theory but weak empirical demonstration of theoretical claims. However, this paper's unified formalism and real-world benchmarks are stronger than VxBh4rtg9Y's case studies.

Given the genuine theoretical contributions and the fact that the empirical issues are overstated claims rather than missing experiments, I score this at **5.5**, positioning it between VxBh4rtg9Y (5.0) and AHpexliCTM (5.5), slightly below YR3CNvFfCr (6.67) due to the theory-experiment mismatch.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>