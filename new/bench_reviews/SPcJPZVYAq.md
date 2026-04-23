## Summary

YOSO proposes applying compressed sensing (CS) to GNN node sampling: sample nodes once at the input layer into an M-dimensional measurement space (M ≪ N), propagate through the network using ΦÂΦ^T in this reduced space, and reconstruct the full N-dimensional output embedding at the final layer. The method jointly optimizes the GNN parameters Θ, a sparsifying basis U (with Stiefel manifold orthogonality constraints), and a sparse representation Ĥ^(L) via a combined reconstruction + task loss. Experiments across 5 datasets on node classification and link prediction report ~75% average training time reduction with competitive accuracy.

## Strengths

- **Substantial and consistent training time reduction**: Figure 2 demonstrates ~75% average training time reduction across all 5 datasets and both tasks. For example, on Reddit, YOSO takes 213.63s vs. 376.5s for the next-best (FastGCN); on ogbn-products, 498.8s vs. 1054.45s (GraphSage). The time breakdown shows the reduction comes primarily from near-elimination of per-epoch resampling overhead.
- **Competitive accuracy maintained**: Table 1 shows YOSO matches or closely approaches top baselines — 0.967 on Reddit (matching GraphSAINT-RW), 0.72 on ogbn-arxiv (matching GraphSage), 0.787 on ogbn-products (within 0.005 of GraphSAINT-EDGE).
- **Genuinely novel framing**: The application of compressed sensing theory to GNN sampling is creative and opens a new direction. The idea of one-time sampling at the input layer with output-layer-only reconstruction is a distinctive architectural contribution.
- **Graph-aware sampling matrix construction**: Section 5.3 designs Φ = S̃ ⊗ Σ using eigenvalue-weighted sampling probabilities derived from the graph Laplacian, combined with Gaussian randomness for RIP satisfaction. This structurally incorporates graph information into the CS framework.

## Weaknesses

### Fatal

None.

### Major

- **Overclaimed "lossless reconstruction" undermines the paper's central thesis**: The abstract and Section 4 prominently claim "lossless reconstruction" equivalent to "full node participation." Standard CS lossless recovery guarantees require a fixed, known sparsifying basis U (Candes & Tao, 2006). In YOSO, U is a free optimization variable jointly learned with GNN parameters (Algorithm 1, Lines 12, 15–16; Eq. 9). When U is free, the sparsity constraint ‖Ĥ^(L)‖_{2,1} can be trivially satisfied — for any H^(L), one can choose U and a sparse Ĥ^(L) such that H = UĤ (e.g., any matrix can be decomposed to have a single non-zero row in some orthonormal basis). The paper cites graph signal processing literature (Isufi et al., 2024; Bo et al., 2023) proving U *exists* for graph data, but existence of a fixed U derived from graph structure is fundamentally different from freely optimizing U. This gap invalidates the direct application of RIP-based "lossless" guarantees to YOSO's learned U. The method may work well in practice, but calling it "lossless" is misleading. This matters because the "lossless" claim is the paper's primary differentiator from other sampling methods.

- **Inconsistency in U's dimensionality and unaddressed scalability concerns**: The paper states UU^T = U^TU = I (Eq. 8, 9), which requires U ∈ R^{N×N}. However, Algorithm 1 (Line 16) enforces only U^TU = I, consistent with U ∈ R^{N×K}. If U is N×N, then for ogbn-products (N=2.4M), storing U requires ~23 TB and each Stiefel projection via SVD costs O(N³) — clearly infeasible, yet the paper reports experiments on this dataset. If U is N×K (with K < N), the orthogonality constraint UU^T = I in the equations is incorrect (since UU^T ≠ I_N for K < N), and the paper's CS proofs (referenced in Appendices D.2, D.3) that assume UU^T = I would not directly apply. This dimension inconsistency is not discussed anywhere, and the paper never clarifies what size U is actually used, how it is stored, or how Stiefel projection is performed for large graphs. This is a critical gap between the presented theory and the practical implementation.

- **Intermediate-layer computation (ΦÂΦ^T) is an unverified approximation, not CS-based reconstruction**: In the forward pass (Eq. 7, Lines 4–7), YOSO computes T^(l) = σ(ΦÂΦ^T T^(l-1) W^(l)), operating entirely in the M-dimensional measurement space. This is NOT equivalent to the original GNN computation H^(l) = σ(ÂH^(l-1)W^(l)) because Φ^T Φ ≠ I when M < N (Φ^T is not an inverse of Φ). The paper acknowledges this tradeoff as "Obstacle II" and claims the error is "controllable with a known upper bound" (Section 4), deferring the proof to Appendix D.4. However, this bound cannot follow from standard CS theory because the intermediate computations do not satisfy CS's requirements (no sparsity constraint, no RIP guarantee for Φ^T as a "reconstruction" operator). Without the appendix available for verification, this is a significant unsupported claim. The paper's own motivation — that YOSO is equivalent to full-batch training — rests on this bound being meaningful.

### Minor

- **Questionable baseline accuracy for some methods**: In Table 1, FastGCN achieves only 0.438 on ogbn-arxiv and 0.404 on ogbn-products, which are substantially below typical reported results (usually 0.65–0.70 on ogbn-arxiv). AS-GCN at 0.51 on ogbn-products is also low. The paper notes "several baseline models lacked implementations for link prediction, prompting us to modify them accordingly" (Section 6.1), raising questions about whether these are fairly tuned implementations. While this doesn't invalidate YOSO's own results, it makes the relative comparison less convincing for these baselines.

- **No reconstruction quality evidence in the main text**: The core claim of the paper is "lossless" or "near-zero bias and variance" reconstruction. Section 6.4 promises heatmaps comparing H^(L) and H̄^(L), but these are deferred to the appendix. Quantitative reconstruction error (e.g., ‖H^(L)_full − H̄^(L)_YOSO‖) is not reported in the main text. This is the most direct evidence for the paper's central claim and should be in the main text.

- **Suspiciously identical baseline training times for ogbl-ppa**: In Figure 2(d), five baselines report nearly identical training times (43.67–45.45s), suggesting sampling overhead is negligible for this dataset. YOSO's speedup on ogbl-ppa (21.42s vs ~44s) likely stems from computing on fewer nodes rather than from the CS framework's one-time-sampling advantage, making the speedup claim for this dataset less informative.

### Trivial

- Convergence analysis (Figure 3) shows training loss curves, but validation/test accuracy would be more informative for generalization assessment.

## Nice-to-Haves

- An ablation removing the reconstruction loss (α = 0) would reveal whether the CS component actually contributes or whether YOSO's success is mainly due to training on a fixed sampled subset with the GNN loss.
- Compare against full-batch GCN on graphs where it fits in memory — if YOSO truly achieves "equivalent to full node participation," this is the most natural baseline.
- Report actual sparsity levels of Ĥ^(L) at convergence. If ‖Ĥ^(L)‖_{0,row} is not much smaller than N, then CS cannot work with M ≪ N, and the method's success must be attributed to other factors.
- Include the eigenvalue preprocessing time for S̃ construction in timing comparisons, or clarify that approximate eigendecomposition is used.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Harsh Critic Claim: "U is N×N requiring 23TB for ogbn-products"* — While the dimension inconsistency is real (see Major weakness above), the paper DOES run on large graphs, suggesting a practical N×K implementation exists. The issue is the lack of clarity, not necessarily that the method is impossible.
- *Harsh Critic Claim: "Figure 1(b) undermines the need for YOSO since simple methods like GraphSAGE have low sampling overhead"* — This misreads the paper. Figure 1(b) shows that methods achieving high accuracy (GraphSAINT, ClusterGCN) DO have high sampling overhead. YOSO targets the accuracy-efficiency tradeoff, not just efficiency.
- *Harsh Critic Claim: "Section 3 presents U as known, then Section 4 reveals it's unknown — this obscures the inconsistency"* — This is a deliberate pedagogical choice (present the ideal CS framework, then address practical obstacles), not obfuscation. The paper is transparent about the transition.
- *Harsh Critic Claim: "No convergence analysis for joint non-convex optimization"* — Convergence proofs for non-convex Stiefel optimization are not standard in empirical ML papers. This is a nice-to-have, not a weakness.
- *Harsh Critic Claim: "Eigenvalue computation for S̃ is O(N³)"* — For sparse Laplacians, iterative eigendecomposition methods (Lanczos) are much more efficient than O(N³), and this is a one-time preprocessing cost. This is a nice-to-have, not a weakness.
- *Harsh Critic Claim: "Missing full-batch GCN baseline is conspicuous"* — Full-batch GCN may not fit in memory for large graphs; this is a valid suggestion but not a critical omission.
- *Strength Finder Claim: "Principled solution to the unknown orthonormal basis problem"* — This conflicts with the verified Major weakness that making U learnable undermines CS guarantees. The "solution" creates a new theoretical gap.
- *Strength Finder Claim: "Faster and more stable convergence"* — Training loss convergence (Figure 3) does not directly demonstrate better generalization; this strength is weakened by the lack of validation accuracy curves.

## Novel Insights

The paper's most interesting contribution — beyond the CS framing — may be the observation that message passing can be effectively approximated in a low-dimensional measurement space via ΦÂΦ^T. This is essentially a graph-structure-aware dimensionality reduction for GNNs, where Φ acts as a structured projection and ΦÂΦ^T captures the essential message-passing dynamics. Whether this works because of CS theory or simply because the graph's information is low-rank (as spectral GNN theory would suggest) is an open question that the paper does not disentangle.

## Suggestions

- Replace "lossless reconstruction" with "approximate reconstruction" or "bounded-error reconstruction" throughout the paper. The current claim is the most prominent and most problematic overstatement.
- Explicitly state the dimensions of U used in practice (N×K), clarify that U^TU = I_K is the enforced constraint (not UU^T = I_N), and discuss how this affects the CS guarantees. This would resolve the dimension inconsistency.
- Add a single table or figure in the main text showing reconstruction error ‖H^(L) − H̄^(L)‖ across datasets and sampling sizes M. This is the most direct evidence for the paper's core claim.

## Score and Decision

**Calibration anchors:**
- **High (7+)**: ScaleGUN (8.0, rigorous theory + billion-edge experiments), MAST (7.0, sound sparsification theory + convergence proofs) — YOSO is well below these due to significant theoretical gaps.
- **Medium (4–6)**: SEIGNN (5.75, honest claims + practical scalability), StructDrop (4.25, simple but consistent), S3 (4.0, limited gains + flawed theorem) — YOSO has a more novel idea and stronger speedups than most, but also more serious theoretical overclaims.
- **Low (<3)**: HashGIN (2.67, plagiarism), FlashSampling (2.50, weak contribution) — YOSO is well above these; it has real empirical results and a genuine contribution.

YOSO sits in the 3.5–4.5 range. Its novelty and empirical speedups push it above the low-scoring anchors, but the overclaimed "lossless reconstruction" and the U dimension inconsistency are significant issues that separate it from accepted papers like SEIGNN (5.75) or MAST (7.0). The theoretical framework as presented does not support the paper's strongest claims, and the gap between the N×N U formulation in the equations and the N×K implementation implied by the experiments is a transparency concern. The practical method likely works for good engineering reasons (low-rank graph structure, structured sampling), but the paper attributes its success to CS theory that doesn't directly apply when U is jointly optimized.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>