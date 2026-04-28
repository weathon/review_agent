Now I have enough calibration anchors. Let me write the final review.

## Summary
This paper proposes PolyhedronNet, a framework for 3D polyhedral representation learning that introduces Surface-Attributed Graphs (SAG) with face-hyperedges and Local Rigid Representations for rotation/translation invariance. The method is evaluated on four datasets (MNIST-C, Building, ShapeNet-P, ModelNet-P) for classification and retrieval tasks, reporting substantial improvements over polygon-based baselines.

## Strengths

- **Novel Surface-Attributed Graph formulation**: The SAG representation (Definition 4.1) explicitly models face-hyperedges with attributes, extending beyond vertex/edge-only graph representations. This is a conceptually meaningful abstraction for domains where face semantics matter (e.g., architectural modeling with material data), as formalized in Section 4.1.

- **Built-in geometric invariance via Local Rigid Representation**: The method achieves rotation and translation invariance by design through relative geometric metrics (distances and angles in Equation 1) rather than relying on data augmentation or absolute coordinates. Theorem 4.5 claims this representation retains sufficient information for graph reconstruction, providing theoretical grounding.

- **Strong empirical performance across multiple datasets**: Tables 1 and 2 show PolyhedronNet achieving state-of-the-art results on both classification (e.g., 0.858 Accuracy on MNIST-C vs. 0.435 for PolygonGNN) and retrieval tasks (e.g., 0.945 NDCG on MNIST-C) across four datasets, demonstrating consistent improvements over baselines.

- **Ablation study validates face attribute importance**: Table 3 provides concrete evidence that semantic face attributes contribute to performance (MNIST-C accuracy drops from 0.858 to 0.360 without attributes), supporting the paper's motivation that face semantics help disambiguate geometrically similar shapes (Figures 4-5).

## Weaknesses

### Fatal
None

### Major

- **Missing comparison against true 3D baselines**: The experimental evaluation compares PolyhedronNet against polygon/sequence encoders (ResNet1D, VeerCNN, NUFT-DDSL, NUFT-IFFT, PolygonGNN) that are explicitly designed for 2D polygons or 1D inputs (Section 5.2 describes these as "polygon encoders"). Section 2.2 notes PolygonGNN is "designed specifically for 2D shapes." The paper does not explain how these 2D/1D models process 3D polyhedral data, nor does it compare against standard 3D representation learning methods (e.g., PointNet++, DGCNN, MeshCNN, or mesh-based GNNs) that would be appropriate baselines for 3D polyhedra. This undermines the claim in Section 5.3 that PolyhedronNet "significantly outperform[s] state-of-the-art approaches" for 3D polyhedral learning. Calibration anchor: Papers with missing 3D baselines in geometry tasks typically score 3-4 (e.g., 2cq8FyBfDk.md avg 3.20, Snf7vos1Xp.md avg 3.50), while papers with appropriate baseline comparisons score 5-6 (e.g., Uf8X57bQIr.md avg 6.00).

- **Performance heavily dependent on synthetic face attributes**: The ablation study (Table 3) reveals that removing face attributes causes MNIST-C accuracy to collapse from 0.858 to 0.360, and Section 5.1 admits MNIST-C faces are "color-coded...to highlight directional identification" (synthetic labels). Section 5.5 states "Building and ModelNet-P datasets do not possess face attributes," and ShapeNet-P attributes appear artificially assigned (Figure 5 shows "orange and black" vs. "grey" faces). This raises concerns about whether the model learns genuine geometric structure or primarily exploits artificially injected color codes. The claim in the Abstract that the method is effective for "real-world polyhedral objects" is not well-supported, as standard 3D benchmarks do not provide the per-face semantic labels required for the reported performance levels.

### Minor

- **Unclear permutation invariance in Local Rigid Representation**: Definition 4.4 includes ψ_{i,j,k} as "indices of the face-hyperedge," but the paper does not explain how these indices are assigned canonically to ensure permutation invariance. If ψ relies on arbitrary ordering, the representation is not permutation invariant; if ψ is omitted, the reconstruction claim (Theorem 4.5) may fail for complex polyhedra. The proofs are deferred to the appendix (which I cannot verify), but this theoretical gap should be addressed in the main text.

- **Selection bias in dataset preprocessing**: Section 5.1 states that for ShapeNet-P and ModelNet-P, "files that still retain numerous mesh faces after merging are dropped." This introduces selection bias toward simplified polyhedra, potentially limiting the method's demonstrated applicability to "complex 3D shapes" as claimed in Section 5.3. The performance on truly complex polyhedra remains unverified.

- **Limited hyperparameter sensitivity analysis**: Section 5.6 evaluates hyperparameter sensitivity only on MNIST-C. Given the claim of robustness on complex shapes, sensitivity analysis on ShapeNet-P or ModelNet-P would strengthen the evaluation, especially since these datasets have different characteristics (no natural face attributes, more complex geometries).

### Trivial
None

## Nice-to-Haves

- **Complexity and runtime analysis**: The number of 2-hop paths scales with the square of node degree. Providing computational complexity analysis and runtime comparisons against baselines would help assess practical scalability, especially for large polyhedra.

- **Discussion of generalization to non-polyhedral meshes**: Since polyhedra are a subset of meshes, discussing how SAG might handle non-planar faces or general mesh topologies would broaden the paper's applicability and impact.

- **Evaluation on datasets with natural face attributes**: Testing on CAD models or architectural datasets with inherent material/semantic tags (rather than synthetically colored faces) would better validate the method's utility for real-world applications.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point 3 (Theoretical Flaw in SAG Invertibility)**: While the permutation invariance concern is valid (moved to Minor), the claim that this "undermines the foundational claim" is overstated. The paper provides proofs in the appendix (Lemma 4.2, Theorem 4.5), and the core methodology remains sound even if the indexing mechanism needs clarification. This is a presentation/theoretical clarity issue, not a fundamental flaw.

- **Harsh Critic's claim that baselines are "fundamentally mismatched" making comparison "invalid"**: While the missing 3D baselines is a Major weakness, the comparison is not entirely invalid—the paper is comparing against prior work in polygon representation learning extended to 3D. The issue is the absence of stronger 3D baselines, not that the existing comparisons are meaningless. Calibration shows papers with baseline limitations often score 4-5 if empirical results are strong (e.g., YPPD3Gf8mc.md avg 4.00, 4siOgDfJn1.md avg 4.00).

- **Strength Finder's claim about "Built-in Geometric Invariance" being fully validated**: The invariance claim depends on resolving the permutation invariance question, so this strength is partially contingent on addressing the Minor weakness.

- **Generic strengths about "clarity" and "well-written"**: These are subjective and not substantive evidence of contribution quality.

## Novel Insights
The paper's core insight—that face attributes (semantics, materials) provide crucial disambiguation for geometrically similar polyhedra—is genuinely interesting and underexplored in 3D representation learning. However, the reliance on synthetic attributes in current benchmarks reveals a broader community gap: standard 3D datasets (ShapeNet, ModelNet) lack the per-face semantic annotations that would enable methods like PolyhedronNet to demonstrate their full potential. This suggests an opportunity for the community to develop polyhedral benchmarks with natural face attributes, rather than treating this as a limitation of the method alone.

## Suggestions

1. **Add comparisons with 3D GNN/mesh baselines**: Include at least 2-3 standard 3D representation learning methods (e.g., DGCNN, PointNet++, or a mesh-based GNN adapted for polyhedra) to validate the SOTA claim. Even if these require adaptation, this is essential for establishing the method's contribution to 3D (not just polygon) learning.

2. **Clarify the face indexing mechanism**: Explain how ψ_{i,j,k} indices are assigned to ensure permutation invariance, or provide a canonicalization algorithm. If the appendix contains this proof, summarize the key insight in the main text.

3. **Report results without synthetic attributes**: For MNIST-C and ShapeNet-P, include performance metrics without the artificial face coloring to demonstrate what the model learns from geometry alone. This would help distinguish geometric learning from attribute exploitation.

4. **Discuss dataset limitations more transparently**: Acknowledge that Building and ModelNet-P lack face attributes, and clarify how the method handles these cases (e.g., are attributes zero-initialized?). Discuss the selection bias from dropping complex meshes and its implications for claimed robustness.

5. **Extend hyperparameter analysis**: Include sensitivity analysis on ShapeNet-P or ModelNet-P to demonstrate robustness across datasets with different characteristics.

## Score and Decision

**Calibration Analysis:**

I retrieved anchors across the score spectrum:

- **High-scoring anchors (≥6)**: Uf8X57bQIr.md (avg 6.00) is a polyhedron benchmark paper with comprehensive evaluation and clear contributions. mGxtoQY3GA.md (avg 6.00) has strong GNN experiments with proper baselines. These papers have appropriate baseline comparisons and don't rely on synthetic data injections.

- **Medium-scoring anchors (4-6)**: A8JuOtpAUM.md (avg 5.00) has strong tables but was rejected for methodological issues. YPPD3Gf8mc.md (avg 4.00) has baseline comparison concerns similar to this paper. 4siOgDfJn1.md (avg 4.00) applies 2D methods to 3D tasks with missing baseline comparisons. iFPUEBwwuT.md (avg 5.00) and PSgps4JXTb.md (avg 5.33) are geometry papers with solid methodology.

- **Low-scoring anchors (≤4)**: 2cq8FyBfDk.md (avg 3.20) was rejected for missing 3D baselines in a structure-aware task. Snf7vos1Xp.md (avg 3.50) lacks specialized baselines. Eq8NRBIRaZ.md (avg 3.00) criticized for missing ablation studies. kK1DwkBzN6.md (avg 3.00) has unfair baseline comparisons.

**Positioning**: This paper has stronger empirical results than the low-scoring anchors (consistent SOTA across 4 datasets with large margins), but the missing 3D baselines is a significant flaw shared with papers scoring 3-4. The synthetic attribute dependency is concerning but the paper is transparent about it (unlike papers that hide such limitations). The SAG formulation is novel, but the evaluation doesn't fully validate it against appropriate 3D methods.

Compared to YPPD3Gf8mc.md (avg 4.00, rejected for baseline issues), this paper has stronger empirical results and a more novel formulation. Compared to Uf8X57bQIr.md (avg 6.00, accepted), this paper lacks the comprehensive benchmark design and has more significant baseline gaps. The paper sits between the 4-5 range anchors.

Given the strong empirical performance but significant baseline limitations, I position this at **4.5**, slightly above papers with similar baseline issues (4.0) but below papers with comprehensive evaluations (5.0-6.0). The synthetic attribute concern prevents scoring higher, as it limits real-world applicability claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>