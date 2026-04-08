=== CALIBRATION EXAMPLE 43 ===

# Final Consolidated Review
## Summary

DenseFace proposes a post-training bias mitigation method for face recognition that models embeddings as von Mises-Fisher (MF) distributions and adjusts similarity scores based on local embedding density. By estimating inter-class densities from a balanced anchor set and applying a margin-based "local distortion" to handle near-orthogonal embeddings, the method contracts dense regions and expands sparse regions of the embedding space, equalizing false positive rates across demographic groups without retraining the backbone. The paper also advocates for NIST-style FPR-based bias metrics over the commonly used RFW accuracy standard deviation.

## Strengths

- **Practical post-training paradigm:** DenseFace operates on pre-trained models without retraining, making it immediately applicable to deployed systems. This is a genuinely valuable contribution in a field where most methods require modifying the training pipeline.
- **Density-bias correlation insight:** Figure 3 provides an important empirical observation that inter-class embedding density correlates with demographic attributes and, in turn, with FPR disparities. This observation motivates the method and offers a new lens for understanding the geometry of biased embedding spaces—something most prior work does not analyze.
- **Improved evaluation rigor:** The paper's argument (Section 4.3, Figure 5) that RFW accuracy standard deviation is an inconsistent bias measure, and its adoption of NIST-style FPR at fixed similarity thresholds, raises the bar for fairness evaluation in this area. The demonstration that models with lower Std can actually be *more* biased by FPR criteria is a concrete contribution to how the community should measure bias.
- **Consistent empirical gains across diverse backbones:** Tables 2–3 show substantial FPR ratio improvements (toward 1.00) for non-Caucasian groups across AdaFace and CosFace models trained on four different datasets, with Table 4 indicating TPR is maintained. The learning-based variant (DenseFace†, Table 5) even outperforms the anchor-set version while adding only marginal overhead.

## Weaknesses

### Major:

- **No comparison to existing post-training calibration baselines:** This is the most significant gap. The related work (Section 2.3) explicitly discusses post-training methods—Terhörst et al. (2020a;b) propose fair score normalization and a classifier-based replacement of similarity; Conti et al. (2022) use MF loss for embedding post-processing; Linghu et al. (2024) propose score normalization methods—yet none of these appear in Tables 2–5 as baselines. The paper only compares against cosine similarity. Without this comparison, it is impossible to determine whether the gains come from the density-aware MF modeling specifically, or from *any* reasonable post-hoc calibration. For a paper claiming to advance post-training debiasing at ICLR, this omission undermines the core empirical contribution.

- **Anchor set dependency is under-analyzed and under-disclosed:** The method critically depends on a demographically balanced anchor set of 54K identities constructed using pre-trained race and gender classifiers on Glint360K (Section 4.2). Two sub-concerns arise: (a) The paper claims "no attribute annotation during inference" (Section 2.3), which is technically accurate for the matching phase, but the anchor set construction *does* require demographic labels. This distinction is not clearly surfaced and could mislead readers about the method's data requirements. (b) There is no sensitivity analysis on anchor set composition—what happens if the anchor set is imbalanced, smaller, or drawn from a different distribution than the deployment environment? Given that the anchor set is the sole mechanism through which density estimates capture demographic structure (Figure 6 shows the nearest-neighbor race distribution is largely same-race), the method's robustness to anchor set characteristics is a critical open question.

### Minor:

- **Local distortion (Equation 7) is heuristic:** The piecewise margin function $f(\theta_{kl}^i)$ that replaces cosine values below threshold $m$ with $\cos(\theta_{kl}^i - m)$ is introduced to fix near-orthogonality issues in anchor set embeddings, but it is not derived from any statistical principle. While the engineering motivation is understandable (Figure 4 shows the shift), the paper provides no theoretical bound or empirical validation showing this specific formulation recovers density better than alternatives. This makes the most novel technical component of the pipeline feel ad hoc.

- **No ablation on key hyperparameters $K$ and $m$:** The method uses $K=128$ nearest neighbors and an angular margin $m$ (value not clearly stated in the main text), yet no ablation study explores sensitivity to these choices. In high-dimensional hyperspherical spaces, density estimation via K-NN can be sensitive to $K$, and the margin $m$ directly controls the "distortion" magnitude. Without this analysis, it is unclear how carefully these need to be tuned for new models or datasets.

- **Cross-racial matching evaluation is claimed but not clearly delivered:** The abstract states the method "also assesses the verification accuracy on multi-racial and cross-racial pairs," and Section 4.3 motivates cross-racial matching as a key limitation of RFW. However, the experimental tables do not clearly present cross-racial verification results separate from within-race results. If this evaluation was conducted, it should be explicitly presented; if not, the claim should be revised.

- **Derivation of matching score is not self-contained:** Equation (9) is presented by citation of Li et al. (2021) without even a sketch of the integration steps from Equation (8). For a method whose core contribution is the matching formula, providing the derivation (or at least an outline) would improve verifiability and reader understanding.

### Trivial:

- The claim "preserves accuracy" in the abstract could be slightly nuanced to "preserves or slightly improves verification accuracy" to match the more precise language in the experiments, though Table 4 does appear to support the claim.

## Nice-to-Haves

- **Statistical significance testing or confidence intervals** on FPR and TPR metrics, particularly for the learning-based variant where training randomness could affect results. This is not standard practice in large-scale face recognition benchmarks, but would strengthen confidence in the reported improvements.
- **FNR/FNMR analysis** alongside FPR, since demographic bias can manifest differently in false non-match rates. The NIST-style FPR focus is reasonable for security applications, but a more complete picture of both error types would be valuable.
- **t-SNE or similar visualizations** of embedding distributions before and after DenseFace, to provide geometric intuition for the claimed "expansion and contraction" of dense and sparse regions.
- **Threshold sensitivity analysis** showing how bias metrics vary across different operating points, not just at the single Caucasian FPR = 10⁻³ threshold.
- **Failure case analysis** identifying specific subgroups or image conditions where DenseFace does not reduce bias or degrades accuracy.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Computational cost not discussed" (Harsh Critic):** Factually incorrect. Section 4.8 explicitly discusses runtime and memory, reports 1.5× slowdown with optimizations, and references Appendix G for detailed tables. The concern is already addressed in the paper.
- **"No statistical significance testing" as a major weakness:** For large-scale face recognition benchmarks with millions of pairs (the paper notes ~14K positive and ~50M negative pairs per group on RFW), single-run evaluation is the community norm. Moved to nice-to-have.
- **"Missing related works" suggestions:** Per hard rules, we cannot confirm existence of uncited works and should not flag missing references.
- **"The RFW protocol dismissal is abrupt" (Harsh Critic):** The paper provides substantial justification in Section 4.3 and Figure 5 for why Std of accuracy is an inconsistent measure. The criticism that the paper should "ensure claims are contextualized against the bulk of existing literature" is a generic demand; the paper does include Table 1 with RFW Std results for comparability.
- **"Claims about large unbalanced datasets having lesser bias contradict prior intuition" (Harsh Critic):** The paper provides empirical evidence (Table 1) for this claim and aligns it with Gwilliam et al. (2021). Disagreeing with "prior intuition" when supported by data is not a weakness—it is a contribution.
- **"Reproducibility concerns about anchor set" (Spark Finder):** The anchor set construction is described in sufficient detail in Section 4.2 (54K balanced identities from Glint360K using pre-trained classifiers). This is a methodology description, not a reproducibility gap.
- **Demand for comparison with Z-norm/T-norm (Harsh Critic, Spark Finder):** These are speaker verification techniques not standard in face recognition bias literature. Requesting comparison with methods outside the paper's community standards is scope creep.

## Novel Insights

The observation that inter-class embedding density (not intra-class) carries demographic signal is a meaningful and underexplored insight. While prior work has modeled face embeddings with MF distributions, the key realization—that intra-class density is uninformative about demographics (because same-identity embeddings are dense regardless of race, e.g., from video frames) while inter-class density reveals systematic demographic disparities—reframes the problem geometrically. The further observation (Figure 6) that nearest neighbors from a balanced anchor set act as an implicit race classifier, and that this property is what enables the method to avoid explicit demographic labels at inference, provides a principled explanation for why the approach works without per-query attribute annotation. This density-demographic link could inspire future work on whether similar density-based calibration applies to other modalities or protected attributes beyond race and gender.

## Suggestions

1. **Add comparisons to at least 2–3 existing post-training debiasing methods** (e.g., Terhörst et al. 2020a/b score normalization, Conti et al. 2022 MF projection, Linghu et al. 2024 score normalization) under the same NIST protocol. This is the single most important revision for establishing the marginal value of density-aware matching over simpler calibration.

2. **Include an ablation study on $K$ and $m$** in the supplementary material, showing how FPR ratios and TPR change across a reasonable range. This would address robustness concerns and provide guidance for practitioners.

3. **Explicitly acknowledge and discuss the anchor set's demographic label requirement** as a practical limitation, and test at least one alternative anchor set composition (e.g., naturally imbalanced vs. balanced) to quantify sensitivity.

4. **Present cross-racial verification results explicitly** if they were computed, or remove the claim from the abstract/Section 4.3 if they were not.

5. **Provide a brief derivation sketch** from Equation (8) to (9) in an appendix, since the matching score is the method's core output and the derivation involves non-trivial integration on the hypersphere.

# Actual Human Scores
Individual reviewer scores: [2.0, 6.0, 2.0]
Average score: 3.3
Binary outcome: Reject
