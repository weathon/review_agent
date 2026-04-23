Now I have a thorough understanding of the paper and the calibration landscape. Let me write the final review.

## Summary

MTGRN introduces a novel formulation of gene regulatory network (GRN inference as multivariate time series forecasting: cells are ordered along pseudotime trajectories (via Slingshot), and a transformer with temporal and spatial attention blocks is trained to forecast future gene expression from past windows. Spatial attention scores are extracted as the inferred GRN. The method outperforms six baselines across five benchmark datasets on AUPRC and F1, including the multimodal CellOracle method, and provides biological validation through key TFG identification (Nanog, Sox2, Pou5f1 in mESC) and in silico Gata1 knockout simulation in mHSC-E.

## Strengths

- **Novel continuous formulation of GRN inference**: Unlike prior methods that discretize cells by type or cluster, MTGRN assigns pseudotime to each cell and formulates GRN inference as MTS forecasting (Equation 1, Section 3.1). This addresses a recognized limitation in the field and is claimed as the first such formulation, which is a genuine conceptual contribution.

- **Strong quantitative performance, including over a multimodal method with only unimodal input**: MTGRN achieves the highest AUPRC and F1 across all five benchmark datasets (Table 1), e.g., AUPRC of 0.849 on mHSC-GM versus the next best 0.634 (CEFCON). It outperforms CellOracle—which requires both scRNA-seq and scATAC-seq—on every dataset and metric, using only scRNA-seq data. This demonstrates practical advantage where multimodal data is unavailable.

- **Biological validation of inferred GRN**: On the mESC dataset, MTGRN identifies Nanog, Sox2, and Pou5f1 as key TFGs whose activity scores decline during differentiation (Figure 5c, Section 5.2), consistent with their established role as pluripotency factors. The Gata1 knockout simulation in mHSC-E correctly predicts blocked erythrocyte differentiation (Figure 7c, Section 5.3).

- **Principled temporal mask design**: The upper-triangular temporal mask (Equation 2) enforces that cells can only attend to ancestral information, reflecting the biological constraint that descendant cells do not yet exist—a well-motivated architectural choice.

- **Insightful analysis of metric appropriateness under class imbalance**: The paper identifies that positive edges constitute only ~0.3–1.4% of all possible edges (Table 2) and correctly argues that AUPRC is more appropriate than AUROC in this regime, citing Davis & Goadrich (2006).

## Weaknesses

### Fatal

None.

### Major

- **The spatial mask restricts the model to only outputting edges present in the NicheNet prior knowledge, making novel regulatory discovery impossible.** Equation 4 sets $M^s_{ij} = -\infty$ when $P_{ij} = 0$, and Section 3.4 confirms: "the regulatory scores in the masked regions are zero, as no regulatory edges exist between those gene pairs in the prior knowledge." This means the model's output GRN is a strict subset of prior knowledge edges, re-ranked. The paper claims to "infer" GRN (Abstract, Section 1, Section 6) but actually refines/re-ranks an existing network. While CEFCON and NetREX also refine prior knowledge, they are more explicit about this; MTGRN's contribution framing as "GRN inference" overstates what the method achieves. The paper devotes one sentence to this limitation without discussing its implications for the claimed contributions. This is not fatal because the continuous MTS formulation and attention-based refinement still represent a genuine approach, but the framing needs substantial correction.

- **No analysis of overlap between NicheNet prior knowledge and ground truth networks.** Since the model can only output edges from the NicheNet prior knowledge (above), the degree of overlap between NicheNet and the benchmark ground truth networks directly determines the upper bound of achievable performance. If NicheNet already contains most true edges, even trivial re-ranking would score well. The paper provides no analysis of this overlap, making it impossible to assess whether MTGRN's strong AUPRC numbers reflect genuine learning or simply the informativeness of the prior knowledge. At minimum, the recall of ground truth edges within the prior knowledge alone should be reported.

### Minor

- **No ablation testing whether the forecasting objective produces attention weights corresponding to regulatory relationships.** The method hinges on the assumption that attention weights learned via MSE forecasting loss capture gene-gene regulatory interactions. Attention useful for prediction need not correspond to direct regulatory interactions—it may attend to co-regulated (correlated) genes rather than causal regulators. The paper provides no ablation (e.g., removing the spatial mask, comparing attention-derived GRN against a co-expression baseline on the same data) to test this core assumption. The biological validation (Sections 5.2–5.3) provides indirect support but does not isolate this claim.

- **GENIE3 and GRNBoost2 use TF gene lists (not prior knowledge networks), creating an asymmetry in comparisons.** The paper classifies GENIE3, GRNBoost2, NetREX, and CEFCON together as "methods that use scRNA-seq data combined with prior knowledge" (Section 4), but GENIE3/GRNBoost2 only use knowledge of which genes are TFs to restrict candidate regulators, while NetREX/CEFCON use a full prior knowledge interaction network comparable to MTGRN's NicheNet constraint. This distinction is not clearly acknowledged. The fairer comparison is with CEFCON and NetREX, where MTGRN still shows substantial improvements.

- **Factual error in AUROC result reporting.** The paper states "MTGRN achieved the highest performance on four out of the six datasets" for AUROC (Section 5.1), but Table 1 shows only five datasets and MTGRN wins AUROC on three (hHep, mESC, mHSC-GM), not four. The same paragraph refers to "all six datasets" when there are five. This appears to conflate the number of baselines (6) with the number of datasets (5).

- **Perturbation validation is qualitative and assumes linear propagation.** The perturbation mechanism ($\Delta X \times H$) assumes linear propagation through the attention matrix, which is biologically unrealistic given the nonlinear nature of gene regulation. The validation is purely qualitative (transition vectors pointing toward HSC in Figure 7c), with no quantitative comparison against actual experimental knockout data.

- **Pseudotime-as-time framing treats each cell as a separate time point (T=C).** With thousands of cells, each "time step" is a single noisy observation. The paper does not discuss noise implications or whether binning cells at similar pseudotimes might improve signal-to-noise.

### Trivial

- The paper describes NicheNet as "highly comprehensive" (Section 3.3) without providing statistics about its coverage relative to the evaluated gene sets.

## Nice-to-Haves

- **Ablation without the spatial mask or with a relaxed (soft penalty) mask** to quantify how much performance depends on the prior knowledge constraint vs. learned representations. This would directly test whether the forecasting objective actually learns regulatory structure and would transform the method from re-ranking to genuine inference.

- **Quantitative perturbation validation** comparing perturbation predictions against experimental knockout data (e.g., Perturb-seq) rather than only qualitative consistency with known biology.

- **Analysis of which prior knowledge edges the model correctly downweights** (removing false positives from prior knowledge) vs. incorrectly removes, to demonstrate meaningful refinement beyond simple re-ranking.

- **Robustness analysis to different TI methods** (e.g., PAGA, Waterfall instead of Slingshot) to assess sensitivity of the pipeline to pseudotime computation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that the spatial mask issue makes the paper fundamentally not GRN inference and should not be accepted**: While the spatial mask constraint is a real limitation, CEFCON and NetREX also refine prior knowledge networks and are legitimate GRN methods. The continuous MTS formulation is a genuine contribution. This is a major weakness in framing/overclaiming, not a fatal flaw invalidating the entire paper.

- **Harsh critic's claim that the comparison with GENIE3/GRNBoost2 is "unfair as an evaluation of the method's inference capability"**: The asymmetry does favor MTGRN, but the paper also compares against CEFCON and NetREX which use prior knowledge networks. The comparison is informative, even if not perfectly symmetric. Downgraded to minor rather than major.

- **Harsh critic's claim that the paper's contribution is "not fixable by additional experiments alone — they require fundamental redesign"**: The continuous formulation and strong benchmark results (even against prior-knowledge-using baselines) represent real contributions. Additional experiments (overlap analysis, ablations) could substantially clarify the contribution.

- **Harsh critic's criticism that identifying Nanog/Sox2/Pou5f1 is "among the most well-known pluripotency factors" and therefore unconvincing**: This is an unreasonable standard—validating against well-known biology is standard practice in the field and does provide evidence that the model captures real biology, even if it doesn't demonstrate novelty of discovery.

- **Harsh critic's concern about "autoregressive masking on independent samples could force the model to learn spurious temporal dependencies"**: This is speculative and not supported by evidence. The model's strong performance suggests the temporal structure is being leveraged productively.

- **Strength finder's claim that "consistently superior quantitative performance across benchmarks" is a core strength without noting the prior knowledge constraint**: The quantitative results are strong but must be interpreted in light of the prior knowledge advantage. Kept the strength but contextualized it.

## Novel Insights

The key tension in this paper—between its genuinely novel continuous MTS formulation and its heavy dependence on prior knowledge for constraining the search space—is not fully recognized by either reviewer. The paper's strongest evidence that the forecasting objective learns real regulatory structure (rather than just predictive correlations) comes from comparing against CEFCON and NetREX, which also use prior knowledge networks. MTGRN's substantial margin over these methods (e.g., AUPRC 0.849 vs 0.634 on mHSC-GM) suggests the continuous formulation and temporal attention provide genuine value beyond what prior knowledge alone offers. However, without the overlap analysis and ablations, this inference remains circumstantial.

## Suggestions

- Report the recall of ground truth edges within the NicheNet prior knowledge alone for each dataset, and include a simple baseline that ranks prior knowledge edges by a heuristic (e.g., co-expression magnitude) to quantify MTGRN's added value beyond prior knowledge.
- Run an ablation with the spatial mask removed (or softened to a penalty rather than $-\infty$) to test whether the forecasting objective alone can learn regulatory structure. This would also allow novel edge discovery.
- Correct the factual error in Section 5.1: MTGRN achieves highest AUROC on 3 out of 5 datasets (not "4 out of 6").
- Explicitly acknowledge in the abstract and introduction that MTGRN refines a prior knowledge network rather than performing unconstrained GRN inference, and reframe the contribution accordingly.

## Score and Decision

**Calibration anchors:**

| Paper | Score | Topic | Comparison |
|-------|-------|-------|-----------|
| gFR4QwK53h | 7.33 | GRN inference from scRNA-seq, causal model for dropouts | Strong theoretical grounding; MTGRN lacks this rigor |
| bcTjW5kS4W (NetFormer) | 7.50 | Attention-based connectivity inference in neuroscience | Analytical justification for attention=structure; MTGRN has none |
| gQlxd3Mtru | 8.67 | Continuous stochastic dynamics from snapshots | Very strong theoretical contribution; far above MTGRN |
| IT33VLRJuS (Marlene) | 4.25 | Self-attention for dynamic GRN from scRNA-seq | Similar idea but weaker validation; MTGRN is stronger |
| 7zsWni0qzC (PerturbODE) | 4.50 | Neural ODEs for GRN under perturbations | Weak experiments; MTGRN has better evaluation |
| 3Fgylj4uqL | 6.00 | Prior knowledge + causal representation learning | Accepted poster; similar use of prior knowledge |
| 44IKUSdbUD | 3.0 | Transformer + sampling for gene-gene interaction | Weak novelty, insufficient baselines; far below MTGRN |
| n9CqhWGK4o | 2.5 | Single-cell GNN for cell-cell interaction | Weak methodology; far below MTGRN |

MTGRN is clearly stronger than the low-scoring anchors (3.0 range), which had fundamental novelty/evaluation issues. It is stronger than Marlene (4.25) and PerturbODE (4.50), which had similar ideas but weaker validation. It falls well below the high-scoring anchors (7+), which had strong theoretical grounding or analytical justification for their core claims. The 6.0 anchor (prior knowledge + causal learning, Accept poster) is perhaps the closest comparison—similar use of prior knowledge but with more explicit acknowledgment of limitations. MTGRN has stronger benchmark results but the prior knowledge constraint and lack of overlap analysis/ablations are significant gaps that prevent it from reaching the 6+ range. I place it at 5.0: genuine contributions in formulation and results, but with major gaps in what the results can be attributed to.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>