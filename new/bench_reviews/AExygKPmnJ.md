## Summary

This paper proposes VN-EGNN, an E(3)-equivariant graph neural network extended with virtual nodes and a three-phase heterogeneous message passing scheme for protein binding site identification. The central idea is that virtual nodes, connected to all physical (residue) nodes, can simultaneously mitigate oversquashing and serve as direct predictors of binding site centers—rather than post-hoc centroid pooling from segmentation maps used by prior methods. Experiments report new state-of-the-art DCC success rates on COACH420, HOLO4K, and PDBbind2020, with ablation studies supporting each architectural component.

## Strengths

- **Clear empirical gains across multiple benchmarks.** Table 1 shows VN-EGNN achieving 0.605 DCC on COACH420 vs. 0.423 for EquiPocket (the previous geometric DL leader), a +0.182 margin that is large and well beyond the reported standard deviation of 0.009. Similar gaps appear on HOLO4K and PDBbind2020, and results are reported across multiple training re-runs with standard deviations.
- **Practical design choices.** Representing proteins at the residue level (α-carbons) rather than atom-level significantly reduces graph size and computational overhead, while ESM-2 embeddings provide strong sequence priors. This mirrors a trend in the field and is validated empirically.
- **Novel message passing formulation.** The three-phase heterogeneous scheme (physical-to-physical, physical-to-virtual, virtual-to-physical) with explicit coordinate update equations for virtual nodes (Eqs. 13, 17) is clearly formalized and differs from prior virtual node approaches that often collapse to global pooling.
- **Useful self-confidence module.** The self-confidence scoring (Eq. 20) enables ranking of predicted binding sites and integration with downstream docking pipelines, mirroring successful object detection paradigms.
- **Theoretical grounding.** The oversquashing argument (bounding maximal shortest-path distance to 2 via virtual nodes) is a valid theoretical connection to Topping et al. (2022), though the causal link is not experimentally isolated.

## Weaknesses

### Fatal
None.

### Major

- **Unfair baseline comparison due to different training objectives.** VN-EGNN's full objective includes $\mathcal{L}_{\text{bsc}}$ (Eq. 19), which directly supervises virtual node coordinates to match ground-truth binding centers. By contrast, all graph-based baselines in Table 1 (EGNN, EquiPocket) were trained on segmentation losses (CE/Dice) and evaluated on post-hoc centroids of predicted segments. The evaluation metric (DCC success rate: whether a predicted center is within 4Å of a true center) is inherently aligned with what $\mathcal{L}_{\text{bsc}}$ optimizes. This creates an apples-to-oranges comparison: a model trained on center regression will naturally outperform models trained on per-node classification evaluated via geometric centroids. The paper argues this is the point of the contribution ("We attribute our improvement to the direct prediction of binding site centers, rather than inferring them from the geometric center of segmented areas," Section 4), but the SOTA claim cannot be fully substantiated without comparing either (a) an equivalent baseline also trained with a center-prediction loss, or (b) VN-EGNN evaluated under a segmentation-based protocol. The performance gap may conflate architectural improvements with supervision advantages.

### Minor

- **Equivariance claim requires clarification.** Proposition 1 states that VN-EGNN is "equivariant with respect to roto-translations and reflections of the input and virtual node coordinates" (E(3)-equivariant). However, Section 2.6 explicitly states: "We break the equivariance property for mirroring through feature encoding... consequently breaks E(3) symmetry to SE(3)." While the message passing operations (Eqs. 7–18) are individually E(3)-equivariant, the complete model—including the amino acid feature encoding that distinguishes chirality—is only SE(3)-equivariant. This should be stated clearly in the paper to avoid confusion. The claim is not invalidated but needs refinement.

- **Ablation study does not isolate architectural contribution from loss function.** Table 2 ablates virtual nodes, message passing variants, and embeddings, but never isolates the impact of $\mathcal{L}_{\text{bsc}}$. The first row "EGNN+VN (Satorras et al., 2021)" achieves 0.156 DCC on COACH420—the exact same as plain EGNN in Table 1—suggesting this row does not use $\mathcal{L}_{\text{bsc}}$. Without training a standard EGNN with $\mathcal{L}_{\text{bsc}}$ for comparison, it remains unclear how much of the improvement is due to the virtual node architecture versus the center-prediction loss. A clean ablation holding the loss constant while toggling virtual nodes would strengthen the causal claim.

- **Hard min in $\mathcal{L}_{\text{bsc}}$ creates sparse gradients without discussion.** The loss $\mathcal{L}_{\text{bsc}} = \frac{1}{M}\sum_{m=1}^M \min_k \|\mathbf{y}_m - \hat{\mathbf{y}}_k\|^2$ assigns gradients to only the single closest virtual node per ground-truth pocket. During training, this typically leads to suboptimal assignment and potential virtual node collapse (multiple nodes converging to the same location), which the authors mitigate post-hoc with MeanShift clustering at inference. The paper does not address whether alternative matching strategies (e.g., Hungarian matching, optimal transport, or a repulsion term) would improve stability or reduce reliance on post-processing.

### Trivial

- **Virtual node count vs. target count protocol.** The paper uses $K=5$ virtual nodes per protein but evaluates by selecting $M$ predictions (where $M$ = number of known binding sites). When $K \neq M$, the matching protocol for the DCC/DCA metric is underspecified, though MeanShift clustering at inference partially addresses this.

## Nice-to-Haves

- Visualizations of virtual node coordinate trajectories across message passing layers (from initialization to convergence) would provide intuitive evidence for the claim that virtual nodes converge to physical binding positions.
- Reporting a version of VN-EGNN evaluated on a segmentation-based protocol (e.g., thresholding virtual node-based predictions to produce segment maps) would help disentangle architecture vs. objective contributions.
- A brief discussion of the expressiveness argument backed by existing experimental evidence in the appendix (App. K) would strengthen the main text's oversquashing and expressiveness claims.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Structural: Direct violation of equivariance properties invalidates core geometric claim."** — The paper's operations (Eqs. 7–18) are indeed E(3)-equivariant; the feature encoding (L/D amino acids) breaks E(3) to SE(3). This is a standard convention in molecular ML (all protein models distinguish chirality) and does not "invalidate" the claim—it merely requires clarifying the boundary between architectural equivariance and full-model equivariance. Downgraded to Minor.
- **"Evaluation protocol fundamentally flawed / SOTA claim cannot stand."** — The paper explicitly frames direct center prediction as the contribution and compares fairly under the established DCC/DCA benchmark protocol (same as Zhang et al., 2023b). While the comparison could be strengthened with matched baselines, the SOTA claim is not invalid. Downgraded to Major (unfair comparison) rather than Fatal.
- **"Hard reviewer's Section 2.5 relaxed initialization breaks per-sample equivariance."** — The paper explicitly acknowledges the relaxed initialization in Section 2.4 and discusses the trade-off. This is a minor clarification issue.
- **"Matching protocol for K predictions to M targets is underspecified."** — The paper describes MeanShift clustering, which handles the mismatch. This is a minor detail.
- **"The EGNN+VN row in Table 2 does not use $\mathcal{L}_{\text{bsc}}$, making it impossible to verify gains."** — Partially valid observation but not a fundamental flaw in the ablation; each row demonstrates incremental improvements. Addressed in the Minor section above.

## Novel Insights

The paper's most interesting conceptual contribution is reframing binding site identification not as a segmentation problem with post-hoc centroid extraction, but as a direct center-prediction task enabled by structurally embedding virtual nodes into the message passing graph. This bypasses the well-known brittleness of centroid-from-segmentation approaches (where a single misclassified region can shift the center arbitrarily) and instead makes the center a first-class learnable entity. The idea that virtual nodes naturally converge to physically meaningful geometric locations (binding pockets) during training is an observation that merits further investigation across other geometric ML domains.

## Suggestions

1. **Clarify equivariance scope.** State explicitly in the main text that the architectural operations are E(3)-equivariant (Proposition 1), but the full model including sequence-based feature encoding is SE(3)-equivariant, and discuss the practical impact of this distinction.
2. **Add a loss-matched baseline.** Train EquiPocket or a standard EGNN with a simple center-prediction readout and $\mathcal{L}_{\text{bsc}}$ to establish whether the improvement is architectural or objective-driven.
3. **Discuss $\mathcal{L}_{\text{bsc}}$ optimization properties.** Briefly address the sparse gradient issue from the hard min operator and explain why MeanShift clustering suffices post-hoc, or consider alternatives like softmin with temperature or bipartite matching.
4. **Specify K and matching protocol.** Explicitly state the number of virtual nodes used and the protocol for handling $K \neq M$ cases during evaluation.

## Score and Decision

I compared this paper against several calibration anchors:
- **Similar topic anchors:** Protein/binding site papers in the corpus received 5–8. The AtomSurf paper (ARQIJXFcTH) scored 6,8,5,8 with similar benchmark-based protein structure learning contributions; ProFSA (uMAujpVi9m) scored uniformly 6. SE(3) equivariance papers ranged from 5–8.
- **Quality anchors:** Strong SOTA papers with well-supported ablations in other domains (cJs4oE4m9Q, lgsyLSsDRe) scored 8s—these papers had clean experimental isolation without confounded comparisons. Papers with major comparison/equivalence concerns (BXMoS69LLR) scored 3–5.
- **Anchor positioning:** This paper sits between these ranges. The empirical results are genuinely strong (large margins with standard deviations), the architecture is novel and well-formulated, and the ablation supports design choices. However, the baseline comparison confound is a meaningful concern that prevents a top-tier score, and the equivariance claim needs clarification. This is roughly comparable to the equivariance/symmetry-breaking paper (yr7PjzmkQ6) that received 5–6, but with stronger empirical results.

I place this at **6.5**—a solid paper with real contributions and meaningful results, but with a notable comparison fairness issue and some claims that need tightening. It falls below the 8-range papers that had cleaner experimental isolation, but above papers with fundamental flaws.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>