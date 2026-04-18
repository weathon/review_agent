Now I have enough calibration data. Let me synthesize the final review. Let me carefully verify each claimed weakness against the actual paper content.

Key calibration anchors:
- CGAE (polygon retrieval, limited baselines): scores 3,8,5,5 → rejected
- ComFormer (geometric graph representation, claimed completeness): scores 6,5,3,6 → poster
- Invariant Attention (provable invariance, limited experiments): scores 6,3,3,5 → rejected
- Simplicial k-forms (geometric representation, novel): scores 8,6,5,6,5 → poster
- CARE (equivariant contrastive learning): scores 6,8,6,8 → poster

Now let me carefully evaluate each claimed weakness:

1. **Lack of 3D baselines**: This is factually correct. All baselines are 2D polygon encoders. The paper claims to address 3D polyhedra but doesn't compare against any 3D deep learning methods (PointNet, DGCNN, MeshCNN, etc.). This is a legitimate concern, though the paper might argue these operate on different data modalities (points/meshes vs. polyhedra).

2. **Theoretical claims not convincingly established**: The paper does defer proofs to appendices. The main text states Theorems 4.5 and 4.6 but only gives brief intuitions. Let me check whether this is fatal or just needs better presentation.

3. **Rotation/translation invariance not properly grounded**: The paper does claim invariance and states three datasets are randomly rotated. The baselines are 2D encoders, so this criticism is essentially the same as point 1.

4. **Limited/contrived benchmarks**: MNIST-C is indeed synthetic extruded digits. The paper does acknowledge this in Section 5.1. The filtering of ShapeNet-P/ModelNet-P is acknowledged.

5. **Overstated Theorem 4.6**: This is a valid minor concern - the theorem is about universal approximation, not about distinguishing non-isomorphic SAGs specifically.

6. **Naming inconsistency (PolyhedronCNN vs PolyhedronGNN)**: Let me check the paper... The abstract says "PolyhedronCNN" but Section 4.3 says "PolyhedronGNN." This is a genuine inconsistency.

7. **No error bars/statistical significance**: Valid concern. The paper reports single numbers.

8. **Low absolute performance on ShapeNet-P/ModelNet-P**: Valid observation.

9. **No complexity/scalability analysis**: Valid concern about two-hop path enumeration.

10. **Insufficient ablations**: Only face attributes are ablated. No ablation of individual components of the local rigid representation or the heterogeneous message passing.

11. **Face attribute dependency**: The ablation shows MNIST-C dropping from 0.858 to 0.360 without face attributes. This suggests the model heavily relies on face attributes rather than geometric reasoning.

Let me also verify the harsh critic's claim about invertibility. Looking at the paper: Lemma 4.2 says SAG construction is invertible, which is stated in the main text with proof in Appendix B. Theorem 4.5 says local rigid representation allows reconstruction - proof in Appendix C. The harsh critic is right that these proofs are deferred and the main text lacks sufficient detail. However, the SAG invertibility claim seems plausible (it's essentially a half-edge-like data structure), and the local rigid representation reconstruction is plausible too if you have all 2-hop paths with distances and angles. The real question is whether there are degenerate cases, but this is a reasonable theoretical framework.

For the naming inconsistency: The abstract says "we propose PolyhedronCNN" but Section 4.3 introduces "PolyhedronGNN architecture." Let me verify this... Looking at the paper again: "Subsequently, we propose PolyhedronGNN to hierarchically aggregate..." (abstract). Wait, actually let me re-read. The abstract says "PolyhedronCNN" - let me check again. Actually, looking more carefully at the paper content: "Subsequently, we propose PolyhedronGNN to hierarchically aggregate the local rigid representation via intra-face and inter-face geometric message passing modules" - so it says PolyhedronGNN in the abstract. The neutral reviewer says the abstract says "PolyhedronCNN" - let me check... The abstract text provided says: "Subsequently, we propose PolyhedronGNN to hierarchically aggregate" - so actually the abstract uses PolyhedronGNN. But wait, the neutral reviewer says "The abstract states 'we propose PolyhedronCNN'" - this seems to be a typo claim by the neutral reviewer. Let me re-read the abstract carefully again... "Subsequently, we propose PolyhedronGNN to hierarchically aggregate" - it says GNN. So this claimed naming inconsistency may be fabricated. Actually, I need to check the full abstract - it says both "PolyhedronNet" throughout. Let me look: The abstract says "This study proposes PolyhedronNet" and "we propose PolyhedronGNN." So it consistently uses PolyhedronNet for the framework and PolyhedronGNN for the specific GNN architecture. The neutral reviewer's claim about "PolyhedronCNN" in the abstract seems incorrect based on the paper text provided. This is a factually wrong criticism.

Actually, wait. Let me look very carefully. The neutral reviewer says: "The abstract states 'we propose PolyhedronCNN'" - but in the paper text I see: "Subsequently, we propose PolyhedronGNN." So either the reviewer misread, or there's a version discrepancy. Based on the text provided, I'll treat this as a factual error by the reviewer.

Now let me finalize my assessment:

**Major weaknesses (keep):**
1. Lack of 3D baselines - all baselines are 2D polygon encoders, no comparison with mainstream 3D methods
2. Insufficient ablations - only face attributes ablated, not individual geometric components or message passing types
3. The model's strong performance appears driven by face attributes rather than geometric reasoning (MNIST-C drops from 0.858 to 0.360 without face attributes)
4. Low absolute performance on realistic 3D datasets (ModelNet-P: 43.5% accuracy)

**Minor weaknesses (keep):**
- No statistical significance/error bars
- No complexity/scalability analysis
- Datasets are somewhat artificial (MNIST-C: extruded digits; Building: extruded 2D shapes)
- Theoretical claims deferred to appendix with insufficient detail in main text

**Remove/adjust:**
- Naming inconsistency claim appears to be factually wrong
- Harsh critic's claim about theoretical non-uniqueness is speculative - the representations store distances, angles, dihedral angles, and face indices, which is a fairly complete encoding for 2-hop neighborhoods. The reconstruction from these is plausible for manifold polyhedra.
- The unfair comparison claim needs careful treatment: it IS true that baselines are 2D and PolyhedronNet is 3D, but this isn't necessarily unfair in PolyhedronNet's FAVOR since the baselines are being compared on their own turf (polygon encoding) extended to 3D. The concern is more about missing 3D baselines, not about unfair comparison.

After careful consideration, this paper has genuine novelty in the problem formulation and the SAG + local rigid representation framework. However, the experimental evaluation is substantially weakened by the absence of mainstream 3D baselines, the apparent heavy reliance on face attributes rather than geometric reasoning, and the low absolute performance on the most realistic datasets. These are significant but not fatal issues. The paper would have been much stronger with comparisons to PointNet/DGCNN on the same data and more thorough ablations.

Score calibration:
- CGAE (limited baselines, niche domain) → avg 5.25, rejected
- ComFormer (geometric completeness claims, limited verification) → avg 5, poster
- Invariant Attention (invariance claims, limited experiments) → avg 4.25, rejected

This paper is somewhat similar to CGAE (limited baselines, polygon domain) but has stronger theoretical foundations. It's also similar to ComFormer in that it claims completeness/invertibility of representation. The experimental weaknesses are significant but the contribution is more novel than CGAE. I'd place this around 4.5-5.</think>Now I have sufficient calibration data. Let me synthesize the final review.

## Summary

PolyhedronNet proposes a framework for learning representations of 3D polyhedral objects by introducing three components: (1) the Surface-Attributed Graph (SAG), a directed graph with face-hyperedges that captures vertex, edge, and face relationships plus semantic face attributes; (2) a local rigid representation that decomposes the SAG into five-tuple geometric features (distances, angles, dihedral angles, face indices) over 2-hop paths, achieving rotation/translation invariance by construction; and (3) PolyhedronGNN, a message-passing architecture with heterogeneous intra-face and inter-face aggregation modules. Experiments on four datasets for classification and retrieval show consistent improvements over 2D polygon encoder baselines.

## Strengths

- **Novel and principled problem formulation.** The SAG data structure cleanly unifies vertices, edges, faces, and face attributes into a single graph representation for polyhedra, going beyond prior work on 2D polygon sequences. Lemma 4.2 claims invertibility of the polyhedron-to-SAG transformation, and Theorem 4.5 claims reconstructability of the SAG from local rigid representations, providing theoretical grounding for the information-preserving claim.

- **Geometrically motivated design.** The local rigid representation uses physically meaningful geometric invariants (distances, angles, dihedral angles) rather than raw coordinates, naturally achieving SE(3) invariance without data augmentation. The distinction between intra-face and inter-face message passing captures a meaningful structural property of polyhedra.

- **Consistent improvements across tasks.** PolyhedronNet achieves the best results on all four datasets and both tasks (classification and retrieval), with particularly large margins on MNIST-C (e.g., accuracy 0.858 vs. 0.435 for the next-best baseline).

- **Ablation demonstrating face attribute importance.** The face-attribute ablation shows substantial performance drops when attributes are removed, validating that face semantics carry crucial information (Section 5.5, Table 3–4).

## Weaknesses

### Major

- **No comparison with 3D deep learning baselines.** All five comparison methods (ResNet1D, VeerCNN, NUFT-DDSL, NUFT-IFFT, PolygonGNN) are 2D polygon encoders operating on vertex sequences. Since ShapeNet-P and ModelNet-P are derived from canonical 3D shape benchmarks, the absence of any mainstream 3D method (PointNet, PointNet++, DGCNN, MeshCNN, or other point cloud/mesh/voxel approaches) is a significant gap. The claim that PolyhedronNet provides "comprehensive and informative representations of 3D polyhedral objects" is not meaningfully tested against the natural competitor methods. The current results establish that PolyhedronNet's SAG + GNN pipeline outperforms 2D polygon encoders on 3D data—which is expected—but they do not establish superiority or even competitiveness with standard 3D shape representation methods.

- **Strong performance appears driven by face attributes rather than geometric reasoning.** The MNIST-C ablation (Table 3) shows accuracy collapsing from 0.858 to 0.360 when face attributes are masked—a 49.8 percentage point drop. Since face attributes in MNIST-C encode digit direction via color (each face type has a distinct color indicating orientation), this suggests the model leans heavily on categorical surface labels rather than geometric structure. On ShapeNet-P, the ablation drop is much smaller (0.627→0.578 accuracy), suggesting geometry matters more there, but the absolute performance on ShapeNet-P (0.627) and especially ModelNet-P (0.435) remains low. This pattern raises questions about whether the sophisticated geometric representation (local rigid tuples, dihedral angles) is the primary driver of performance, or whether access to discriminative face attributes is the core advantage.

- **Low absolute performance on the most realistic datasets.** On ModelNet-P classification, PolyhedronNet achieves only 43.5% accuracy and 0.824 AUC across 14 categories—barely above the PolygonGNN baseline (43.0%, 0.821 AUC). On ShapeNet-P, retrieval precision is only 0.322. These are modest absolute numbers that suggest the method struggles with complex, multi-category 3D shapes, undermining the claim of a general polyhedra representation framework.

### Minor

- **Insufficient ablation granularity.** Only face attributes are ablated. There is no ablation of individual local rigid representation components (distances, angles, dihedral angles, face indices), no comparison of intra-face vs. inter-face message passing, and no comparison of the heterogeneous path-type MLPs vs. a single shared MLP. Without these, it is unclear which architectural choices actually matter.

- **No error bars or statistical significance tests.** All reported metrics are single-point estimates. Given the relatively small sizes of ShapeNet-P (2,122) and ModelNet-P (1,303), the modest improvements over PolygonGNN on these datasets (e.g., 0.435 vs. 0.430 accuracy on ModelNet-P, 0.576 vs. 0.575 NDCG) may not be statistically meaningful.

- **Datasets have limited realism.** MNIST-C and Building are synthetic 3D extrusions of 2D data. ShapeNet-P and ModelNet-P filter out objects with "numerous" mesh faces, biasing evaluation toward simple polyhedra. The paper does not report how many objects were dropped or how this affects dataset difficulty.

- **No computational complexity or scalability analysis.** The two-hop path enumeration grows as O(|V|·d²) per graph, but no runtime, memory, or scaling analysis is provided. Real-world polyhedra in CAD applications can have thousands of faces.

### Trivial

- None

## Nice-to-Haves

- Comparison with at least one 3D point cloud or mesh method (e.g., PointNet, DGCNN) on the same ShapeNet-P/ModelNet-P data, even with point sampling from the polyhedral surfaces.
- Empirical verification of rotation/translation invariance (e.g., systematic evaluation under varying 3D rotations).
- Reconstruction experiment from local rigid representations to validate Theorem 4.5 empirically.
- Ablations of individual geometric features and the intra-/inter-face message passing modules.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Naming inconsistency (PolyhedronCNN vs. PolyhedronGNN):** The neutral reviewer claims the abstract says "PolyhedronCNN" while Section 4.3 says "PolyhedronGNN." Upon verification, the abstract text reads "we propose PolyhedronGNN"—consistent with the rest of the paper. This appears to be a misreading by the reviewer.

- **Claim that baselines are unfairly handicapped by lacking rotation invariance:** While true that baselines lack 3D rotation invariance, the concern about fairness cuts both ways—the paper's method was designed precisely to address this challenge. The legitimate weakness is the *absence* of fair 3D competitors, not that existing baselines are weak.

- **Theoretical non-uniqueness concerns about local rigid representation (harsh critic, point 2):** The harsh critic argues that reconstruction from local rigid tuples may not be unique due to "well-known non-uniqueness issues." However, the five-tuple stores distances, angles, dihedral angles, and face indices—it encodes the local geometry quite completely for manifold polyhedra. The reconstruction claim (Theorem 4.5) is plausible even if the proof detail is deferred to the appendix. While the main text could be more explicit about assumptions (e.g., manifoldness, non-degenerate configurations), this is a presentation gap rather than a fundamental error.

- **Demand for tasks beyond classification/retrieval (generation, optimization):** The paper explicitly scopes itself to representation learning for classification and retrieval. Requesting generation/optimization tasks goes beyond the stated scope.

- **Demand for confidence intervals as a fatal flaw:** While desirable, single-run evaluation without error bars is common in this community, especially when datasets and training details are newly introduced. Flagged as minor, not major.

## Novel Insights

The most revealing finding is the sharp performance asymmetry between face-attribute-dependent and geometry-dependent performance: MNIST-C classification drops by ~50% without face attributes, while ShapeNet-P drops only ~5%. This suggests that the SAG's main practical advantage in current experiments comes from its ability to incorporate heterogeneous face attributes (material, color) that 2D polygon encoders cannot easily consume—not necessarily from superior geometric modeling. This tension between the theoretical emphasis on geometric completeness and the empirical reliance on semantic attributes is underexplored in the paper and deserves explicit discussion.

## Suggestions

- **Compare against at least one 3D baseline on the same data.** Sample points from the polyhedral surfaces and run PointNet/PointNet++ as a straightforward comparison. This would establish whether the polyhedral representation itself provides advantages over standard 3D representations, or whether the gains come primarily from face attributes.

- **Add component-wise ablations of the local rigid representation.** Remove distances, angles, and dihedral angles individually to clarify which geometric features drive performance on datasets without strong face attribute cues (ShapeNet-P, ModelNet-P).

- **Explicitly discuss the face attribute dependency.** Acknowledge that current performance heavily relies on surface semantic labels, discuss how the framework would perform on polyhedra without such attributes (as is common in many CAD/mesh datasets), and propose strategies (e.g., learning face embeddings from geometry) for handling attributeless polyhedra.

## Evaluation

**Originality:** The problem formulation (SAG for polyhedra, local rigid representation with face-hyperedges) is genuinely novel and addresses an underexplored niche. The PolyhedronGNN architecture, while competent, bears similarity to prior geometric GNNs (DimeNet-like two-hop messages with angular features), making its architectural novelty moderate.

**Importance:** The research question—adequate representation of 3D polyhedra—is timely and relevant for CAD, robotics, and computational geometry. However, the current experimental evidence does not establish the method's practical value against established 3D representations.

**Claims support:** The core claim of superior polyhedra representation is partially supported against 2D baselines but unsupported against 3D competitors. The theoretical claims (invertibility, reconstructability) are plausible but insufficiently detailed in the main text.

**Soundness of experiments:** The experiments are well-structured (four datasets, two tasks, multiple metrics, ablation) but fundamentally limited by the choice of baselines and the revealed dependency on face attributes.

**Clarity:** The paper is well-organized and clearly written. Figures effectively convey the framework.

**Value to community:** The framework provides a useful direction for polyhedral representation, but its practical impact depends on future comparison with 3D methods.

## Score and Decision

**Calibration comparisons:**

- **CGAE** (polygon retrieval, limited baselines, same domain) → avg 5.25, rejected. PolyhedronNet has stronger theory but similar baseline issues.
- **ComFormer** (geometric completeness for crystals) → avg 5.0, poster. PolyhedronNet shares the completeness/reconstructability angle but has weaker experimental validation.
- **Invariant Attention** (provable invariance, limited experiments) → avg 4.25, rejected. Similar pattern of theoretical claims with limited empirical grounding.
- **Simplicial k-forms** (novel geometric representation, solid theory) → avg 6.0, poster. More novel and better validated than PolyhedronNet.

PolyhedronNet sits between these anchors. It has real novelty and a coherent framework, but the experimental evaluation is structurally flawed by the absence of 3D baselines, and the revealed face-attribute dependency undermines the "geometric representation" narrative. The absolute performance on real 3D datasets is underwhelming. These are meaningful but not fatal issues—the method is sound and the ideas are promising, just not yet convincingly validated.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>