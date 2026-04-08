=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary

π³ introduces a fully permutation-equivariant feed-forward architecture for visual geometry reconstruction that eliminates the reliance on a designated reference view. By predicting affine-invariant camera poses and scale-invariant local point maps defined in each view's own coordinate system, and by removing order-dependent components like positional embeddings and reference tokens, the model achieves robustness to input ordering while attaining state-of-the-art performance on camera pose estimation, depth estimation, and point map reconstruction across numerous benchmarks.

## Strengths

- **Principled elimination of a pervasive inductive bias.** The paper identifies that anchoring reconstruction to a fixed reference view—an assumption inherited from classical SfM and adopted by all modern feed-forward methods—introduces instability when the reference is suboptimal. The solution (removing all order-dependent components and supervising on relative quantities) is clean, well-motivated, and directly addresses the identified problem. This is a genuine architectural contribution rather than an incremental tweak.

- **Compelling robustness evidence.** Table 6 demonstrates near-zero standard deviation across input permutations on DTU and ETH3D, outperforming VGGT by an order of magnitude (e.g., DTU Acc. std: 0.003 vs. 0.033; ETH3D metrics effectively zero vs. 0.049). This is the strongest empirical validation of the core claim and directly ties the architectural design to its intended benefit.

- **Broad and strong empirical performance.** The method achieves new SOTA on multiple benchmarks simultaneously—camera pose ATE on Sintel drops from 0.167 (VGGT) to 0.074, video depth Abs Rel on Sintel improves from 0.299 to 0.233, and point map estimation leads on DTU and ETH3D—demonstrating that the equivariant design does not sacrifice accuracy for robustness.

- **Interesting empirical finding on pose distribution structure.** Figure 4/6 shows that the permutation-equivariant model learns camera pose distributions with clear low-dimensional structure compared to VGGT's scattered distribution, suggesting the architecture implicitly regularizes toward geometrically plausible trajectories. This is a non-obvious emergent property worth highlighting.

## Weaknesses

### Major:

- **VGGT initialization dependency clouds attribution of gains.** Appendix A.4 reveals the final model initializes encoder and alternating-attention weights from pretrained VGGT. Table 8 shows π³ trained from scratch *without* a proxy task underperforms VGGT on 7-Scenes (Acc. 0.064 vs. 0.057) and is only competitive with the proxy. This raises a fundamental question: to what extent do the reported SOTA results reflect the superiority of the permutation-equivariant architecture versus transfer of priors from a reference-view teacher? The paper frames VGGT initialization as an efficiency choice ("to maximize computational efficiency"), but the from-scratch results suggest it may be necessary for competitive performance. This should be discussed more candidly in the main text—currently it is buried in an appendix—as it directly affects how one interprets the contribution. A comparison where both π³ and VGGT use identical pretraining strategies (e.g., both from scratch with the proxy, as partially shown in Table 8) would substantially strengthen the architectural claim.

### Minor:

- **Imprecise "no global coordinate system" framing.** The abstract claims the model operates "without any reference frames" and the introduction says it "completely removing the need for a global coordinate system." Yet Eq. 7 computes relative poses as $\hat{\mathbf{T}}_{i \leftarrow j} = \hat{\mathbf{T}}_i^{-1} \hat{\mathbf{T}}_j$, which requires $\hat{\mathbf{T}}_i$ and $\hat{\mathbf{T}}_j$ to reside in a common frame. Section 3.3 acknowledges the poses are "defined up to an arbitrary similarity transformation"—there IS a coordinate system, just not one anchored to a specific input index. The paper should distinguish between "reference-index-free" (no designated input view as anchor) and "reference-frame-free" (no coordinate system at all), as the former is accurate while the latter is misleading.

- **Inference-time point cloud fusion process underspecified.** Each point map $\mathbf{X}_i$ is defined in its own camera coordinate system. For scene-level evaluation (Table 2–3), these must be fused into a global point cloud, but the paper does not explicitly describe how $\{\mathbf{X}_i\}$ and $\{\mathbf{T}_i\}$ are combined at inference to produce the final reconstruction. The evaluation uses Umeyama + ICP alignment, but the practical pipeline from model output to fused scene is unclear.

- **FPS comparison does not control for model depth.** Table 4 reports 57.4 FPS vs. VGGT's 43.2 FPS, but Appendix A.1 reveals π³ uses 36 alternating-attention layers while VGGT uses 48. A significant portion of the speed advantage likely comes from having 25% fewer layers rather than from the permutation-equivariant design per se. The paper should acknowledge this confound or provide a depth-normalized comparison.

- **Training instability with relative supervision is underanalyzed.** Appendix A.4 identifies a "cold start" problem: the N×N coupled relative constraints are harder to optimize from random initialization than anchor-based formulations. This is a meaningful practical limitation of the equivariant approach that receives only brief discussion. Understanding *why* the coupled constraints are unstable and whether this is fundamental to equivariant formulations or an artifact of the specific loss design would strengthen the paper.

### Trivial:

- **O(N²) training loss for camera poses.** Eq. 8 sums over all ordered view pairs. With training batches of 2–24 images (Appendix A.2), this is computationally manageable in practice, but it is not discussed as a scalability consideration for longer sequences.

## Nice-to-Haves

- Ablation on the alternating view-wise/global attention pattern versus simpler equivariant architectures (e.g., pure global attention) to justify this specific design choice for achieving equivariance.
- Isolated quantitative evaluation on dynamic scenes, given the explicit claim of handling "both static and dynamic scenes" (Figure 1, abstract). Sintel contains dynamic content but results are not disaggregated.
- Visual failure case gallery (transparent objects, textureless regions, extreme baselines) beyond the brief textual listing in Appendix A.8.
- FLOPs and peak memory usage alongside FPS to give a more complete efficiency picture.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Data leakage" from VGGT initialization on zero-shot benchmarks.** The critic argued that since π³ initializes from VGGT (trained on Co3Dv2 etc.), it inherits dataset priors that give unfair advantage on zero-shot claims. However, π³ is explicitly *trained* on 15 datasets covering the same domains—initialization affects optimization, not data access. All compared methods train on similar data pools. This is not data leakage.

- **Metric scale recovery at inference.** The critic argued that the cited robotics applications require metric scale, which the scale-invariant model cannot provide. The paper explicitly scopes its output as scale-invariant (Abstract, Section 3.2–3.3). Criticizing a paper for not doing something outside its stated design is scope creep. The mention of robotics in the Introduction is aspirational context, not a claim of metric-scale output.

- **"Unfair" baseline comparison due to training data scale.** The transferred review argued training on 15 datasets gives an unfair advantage. All modern feed-forward geometry methods (VGGT, CUT3R, FLARE) use large-scale multi-dataset training. This is standard practice, not an unfair comparison.

- **Demand for confidence intervals or statistical significance.** Single-run evaluation is the norm for large-scale 3D reconstruction benchmarks. Requesting confidence intervals is a nice-to-have, not a core flaw.

- **Formatting/style nitpicks.** Removed per hard rules.

## Novel Insights

The most interesting non-obvious finding is the *low-dimensional structure* of predicted pose distributions (Figure 4/6): the permutation-equivariant model's predicted camera poses concentrate along far fewer principal components than VGGT's, suggesting that removing the reference-view anchor implicitly regularizes the network toward geometrically structured solutions. This connects to a deeper observation: the "cold start" problem (Appendix A.4) reveals a fundamental tension in equivariant formulations—N×N coupled relative constraints provide richer geometric signal but create a harder optimization landscape than anchor-based decompositions. The fact that the proxy task (which re-introduces a reference frame during training only) resolves this suggests that equivariant architectures may benefit from a "curriculum" that first learns in an anchored regime before transitioning to relative supervision. This has implications beyond this paper for how equivariant geometric models should be trained.

## Suggestions

- **Move the VGGT initialization discussion and from-scratch comparison into the main paper** (Section 4.5 or a dedicated paragraph in Section 3.4), with honest framing of what the from-scratch results imply about the architecture's standalone viability versus the efficiency benefits of initialization.

- **Add a single clarifying paragraph** after Eq. 7 explicitly describing how per-view point maps and poses are combined at inference to form a scene-level reconstruction, and acknowledging that a latent coordinate system exists (defined by the network's internal representations) even though no input view is designated as its origin.

- **Footnote or parenthetical in Table 4** noting that π³ uses 36 attention layers vs. VGGT's 48, so the FPS advantage partially reflects reduced depth. This costs one sentence and greatly improves honesty of the comparison.

- **Report the "from scratch + proxy" result from Table 8 in the main text** alongside the final model results, to show that the architecture can match or exceed VGGT without VGGT initialization when properly stabilized. This would substantially mitigate the major weakness above.

# Actual Human Scores
Individual reviewer scores: [8.0, 10.0, 6.0]
Average score: 8.0
Binary outcome: Accept
