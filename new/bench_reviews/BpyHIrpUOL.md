## Summary

PolyhedronNet introduces a Surface-Attributed Graph (SAG) formalism with face-hyperedges to encode vertices, edges, faces, and their geometric relationships, decomposing these into rotation/translation-invariant local rigid representations and aggregating them via a heterogeneous PolyhedronGNN with intra-face and inter-face message passing. The method demonstrates consistent gains over five baselines across four datasets in both classification and retrieval tasks.

## Strengths

- **Novel SAG formulation with explicit face-level semantics**: Definition 4.1 introduces face-hyperedges that encode both connectivity order and face attributes, formally characterizing that adjacent faces share edges in opposite directions. This is a substantively more structured approach than prior work that treats polyhedra as unstructured point clouds or polygon sequences. The key structural insight ($\exists e_{o,r} \in f_i, e_{r,o} \in f_j$) is sound and enables the heterogeneous message passing design.

- **Rotation/translation-invariant local rigid representation**: The five-tuple representation $s(\pi_{i,j,k}) = (d_{i,j}, d_{j,k}, \theta_{i,j,k}, \phi_{i,j,k}, \psi_{i,j,k})$ (Eq. 1) uses edge distances, planar angles, dihedral angles, and face indices — all geometrically invariant quantities. Theorem 4.5 claims this representation can reconstruct an equivalent SAG, grounding the design.

- **Heterogeneous intra-face/cross-face message passing**: PolyhedronGNN (Section 4.3) distinguishes paths within a planar face from paths across face boundaries via separate MLPs $\Psi^{(l)}$ weighted by path type, which is geometrically appropriate for polyhedra.

- **Face-attribute ablation confirms SAG value**: Table 3 shows removing face attributes drops MNIST-C accuracy from 0.858 to 0.360, providing direct evidence that the face-attribute modeling is a meaningful contributor to performance, not a peripheral detail.

## Weaknesses

### Major

- **Baseline selection excludes all standard 3D geometric deep learning methods**: The paper positions PolyhedronNet as a contribution to "3D polyhedral representation learning" and claims to "significantly outperform state-of-the-art approaches by a substantial margin" (Section 3). Yet Section 5.2 compares exclusively against 2D/1D polygon encoders (ResNet1D, VeerCNN, NUFT-DDSL, NUFT-IFFT) and a 2D visibility-graph GNN (PolygonGNN), which the paper itself notes is "designed specifically for 2D shapes" (Section 2.2). No comparison is made against any standard 3D representation learning baseline (e.g., PointNet, MeshCNN, DGCNN) trained on vertices or face graphs from the same polyhedra. The large margins in Tables 1-2 (e.g., MNIST-C Acc: 0.858 vs. 0.435) are largely expected when comparing a 3D-aware model to 2D-only methods. Without at least minimal comparisons against established 3D baselines, the "state-of-the-art" claim cannot be substantiated.

- **Rotation invariance claim incompletely validated**: The paper emphasizes that PolyhedronNet maintains "rotation and translation invariance" throughout (Abstract, Sections 1, 4.2). However, the Building dataset (Section 5.1) is explicitly "not subjected to random rotations due to the original lack of alignment." On one of four datasets — and the one where the method achieves perfect AUC — the rotation invariance property is not tested at all. This leaves a significant gap between the theoretical claim and the empirical validation.

### Minor

- **Overstated framing of theoretical guarantees vs. practical architecture**: The paper states the framework "minimizes information loss" (Abstract) and that SAG provides "ensuring no information is lost" (Section 3). However, the final readout is sum-pooling over all nodes (Eq. 4), which is inherently lossy. Theorem 4.6's proof sketch acknowledges this by describing a discretization into "small granules," which is a heuristic universal approximation argument rather than an information-preservation guarantee. The SAG transformation itself (Lemma 4.2) is proven invertible, but the learned GNN readout is not. The authors should clarify the boundary between what is theoretically preserved (the SAG representation construction) and what is practically approximate (the GNN aggregation).

- **Ablation study is narrowly scoped**: Table 3 only masks face attributes. It does not ablate the core architectural contributions — the dihedral angle $\phi$, the separation of intra-face vs. cross-face message paths, or the two-hop path decomposition. This limits understanding of which design elements drive performance.

- **Datasets partially synthetic and derived through opaque preprocessing**: MNIST-C and Building are created by extruding 2D polygons along the Z-axis, introducing degenerate geometry. ShapeNet-P and ModelNet-P are derived via a "mesh merge algorithm" with limited description of how non-watertight or degenerate outputs are handled relative to the SAG construction requirements (e.g., consistent CCW face ordering). No analysis of how many samples fail the manifold/enclosure assumptions is provided.

### Trivial

- None beyond the above.

## Nice-to-Haves

- t-SNE/UM visualization of final graph embeddings to verify whether representations cluster by geometric class or by injected face attributes.
- Evaluation on natively 3D, watertight polyhedral datasets (e.g., procedural 3D primitives without extrusion) to demonstrate genuine volumetric geometry learning.
- Sensitivity analysis on the hyperparameters and number of GNN layers on ShapeNet-P and ModelNet-P (currently only evaluated on MNIST-C, per Figure 3).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Artificial dataset construction invalidates evaluation"**: While the datasets have limitations (extruded 2D shapes, mesh-merged polyhedra), this is common in early work on new representation formalisms. The Building dataset is derived from OpenStreetMap (real building footprints), and ShapeNet-ModelNet are standard benchmarks with modified preprocessing. This weakens scope coverage but does not invalidate the paper's contribution.
- **"Contradiction between theoretical claims and lossy architecture is fatal"**: The critic conflates Lemma 4.2's SAG transformation invertibility (which is about the representation, not the learned model) with Theorem 4.6's universal approximation claim (which is a "with sufficient dimension" approximation result, not strictly lossless). The gap is worth clarifying but does not invalidate the approach, as universal approximation theorems for GNN readouts are well-established in the literature.
- **"Missing normalization in GNN aggregation causes gradient explosion"**: Standard GNN practice; the paper includes hyperparameter sensitivity analysis showing stable training. This is an implementation detail and not a structural flaw.
- **"Face indexing dependence before permutation-invariant aggregation"**: The five-tuple representation uses face indices only as identifiers for face-hyperedges; the permutation invariance comes from the set-based message summation (Eq. 3). This is a standard GNN design pattern.

## Novel Insights

One paragraph synthesizing genuinely novel observations.

The paper's core contribution is genuinely structural rather than incremental: the Surface-Attributed Graph bridges the gap between graph-based polyhedral modeling and face-level semantic reasoning, which prior polygon sequence methods fundamentally cannot support. The key insight — that polyhedral geometry naturally decomposes into local rigid bodies parameterized by distance-angle-dihedral tuples — provides an invariant representation that is neither voxel-dependent nor coordinate-frame dependent. However, the empirical validation falls short of the theoretical ambition: the baselines and datasets do not adequately stress-test whether the method learns true 3D volumetric geometry or primarily exploits surface attributes. The paper would be substantially stronger if positioned explicitly as introducing a new representation formalism and demonstrating its attribute-driven advantages, rather than claiming broad 3D SOTA without 3D baseline comparisons.

## Suggestions

- Compare PolyhedronNet against at least two standard 3D baselines (e.g., PointNet or a vertex-graph GNN) on the ShapeNet-P and ModelNet-P datasets. Even if PolyhedronNet underperforms, this would ground the claims and clarify where the SAG approach excels and where it does not.
- Evaluate rotation invariance on the Building dataset by applying random rotations and comparing performance to the unrotated results.
- Expand the ablation to isolate the dihedral angle $\phi$ and the cross-face message path, since these are central architectural choices.
- Clarify in the text that Lemma 4.2 establishes invertibility of the SAG *construction*, while the GNN readout (Theorem 4.6) provides a universal approximation guarantee — these are distinct claims with different guarantees, and conflating them weakens the paper's theoretical rigor.
- Report the fraction of ShapeNet-P/ModelNet-P samples that pass the watertight/manifold assumptions after mesh merging, to validate that the SAG construction requirements are met in practice.

## Score and Decision

For calibration, I compared this paper against several anchor groups:

- **High-scoring anchors (avg 7+):** vVCHWVBsLH (polyhedron decomposition, scores 8,8,8,5) and gxhRR8vUQb (mesh metric, scores 6,6,8,8) are both accepted and feature rigorous validation with strong baselines appropriate to their domain. This paper falls notably below them in experimental thoroughness.
- **Mid-scoring anchors (avg 5-6):** 7vVWiCrFnd (novel GNN from probabilistic inference, scores 5,6,8,6,8) and kat8uANDlU (heterogeneous GNN, scores 5,6,6,5,6) both have novel formalisms but face criticism for limited/weak baselines — similar to this paper. They were ultimately accepted despite baseline weaknesses because the core ideas were compelling.
- **Low-scoring anchors (avg 3):** rEQ8OiBxbZ (tetrahedron-based molecular pretraining, scores 3,3,3,3) had insufficient experiments to support its claims. Our paper exceeds this — its results are consistent and the SAG formalism is clearly motivated, just under-validated.

This paper's situation is structurally closest to the 5-6 anchors: a novel representation formalism with meaningful empirical gains, but baseline selection that does not match the stated scope of the contribution. The missing-3D-baselines concern is real and significant, but the core idea (SAG with face-hyperedges) is genuinely novel and the empirical results on the chosen baselines are substantial. I position this paper slightly above the 5-6 cluster since the novelty of the SAG framework is particularly clean and the theoretical framing (while imperfectly articulated) is substantively nontrivial.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>