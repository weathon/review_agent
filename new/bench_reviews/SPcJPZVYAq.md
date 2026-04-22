Now I have all the information I need to write the comprehensive final review. Let me also check one more thing - does the paper mention the ablation study details about reconstruction error quantitatively?

The ablation study (Section 6.4) mentions comparing H^{(L)} "generated without sampling" to Ĥ^{(L)} "produced by YOSO's sampling-reconstruction process" with "differences visualized with heatmaps." This is interesting — the paper does partially address the full-batch comparison through its ablation, but the quantitative details are sparse (no pun intended) in the main text.

Now let me write the final review.

## Summary

YOSO proposes applying compressed sensing (CS) to GNN training by sampling nodes once at the input layer and reconstructing embeddings at the output layer. The key idea is to treat the embedding matrix as a multi-channel signal with a sparse representation in some orthonormal basis U, then use a CS measurement matrix Φ to compress the signal to M ≪ N nodes during forward propagation and reconstruct the full N-node output after the final layer. This eliminates per-epoch resampling overhead, which the paper shows can account for up to 64% of training time in existing methods.

## Strengths

- **Clear empirical motivation (Figure 1):** The paper provides a compelling breakdown showing sampling overhead accounts for 35.7%–64% of total training time across seven GNN sampling methods on Reddit, making a strong case that sampling—not computation—is the primary efficiency bottleneck.
- **Substantial training time reduction (Figure 2, Table 1):** YOSO achieves ~75% average training time reduction across 5 datasets, with sampling time reductions up to 99% on Reddit (1149s → 15s). Accuracy remains competitive with the best sampling baselines within ~0.005 on most datasets.
- **Novel application of CS to GNN sampling:** Framing GNN node sampling as a compressed sensing problem is genuinely creative. The identification of two concrete obstacles (unknown U, universal Φ) with corresponding solutions in Sections 5.2–5.3 provides a clear technical roadmap.
- **Evaluation across two tasks:** The paper validates on both node classification (3 datasets) and link prediction (2 datasets), demonstrating generality beyond a single task type.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed "lossless" and "near-zero bias and variance" guarantees (Abstract, Section 4, Eq. 8–9):** The paper's central theoretical claim of "lossless reconstruction" depends on ΦU satisfying the RIP (Eq. 4). Standard CS theory requires U to be a *fixed, known* basis. In YOSO, U is a *learned parameter updated every epoch* (Algorithm 1, Lines 15–16). Gradient updates to U will generally not preserve the RIP of ΦU, and the joint loss (Eq. 9) includes no explicit RIP constraint. The paper provides no mechanism to maintain RIP during optimization. Calling this "lossless" or "equivalent to full node participation" (Abstract, Section 1) is therefore a significant overclaim. The method may learn something useful empirically, but invoking CS reconstruction theory to describe what it does is not justified. This overclaim directly undermines the paper's central selling point.

- **Inconsistent dimensionality of U creates ambiguity between theory and practice (Section 3, Eq. 8, Algorithm 1):** The paper states U^{(l)} ∈ R^{M×N} in Section 3 (which is dimensionally incompatible with H = UĤ where H ∈ R^{N×d}), while Eq. 8 imposes UU^T = U^T U = I (requiring a square N×N matrix), and Algorithm 1 Line 16 enforces only U^T U = I (compatible with a rectangular N×K Stiefel manifold matrix). If U is truly N×N, learning and projecting it is infeasible at scale (ogbn-products has N ≈ 2.4M nodes). If U is N×K with K ≪ N, the constraint UU^T = I_N cannot hold. The paper never clarifies the actual dimensions of U in the implementation, leaving a critical gap between the theoretical framework and the reported experiments on million-node graphs.

- **Missing full-batch GCN baseline (Table 1):** The paper claims YOSO achieves results "equivalent to full node participation" and "near-zero bias and variance," yet never reports accuracy for full-batch (unsampled) GCN training. All comparisons are against other sampling methods. While the ablation (Section 6.4) mentions comparing the reconstructed H^{(L)} with a version "generated without sampling," no quantitative comparison of final test accuracy against full-batch training appears in the main text. Without this baseline, the "lossless" claim is empirically unverified.

### Minor

- **ΦĀΦ^T as a compressed graph operator lacks analysis (Eq. 7):** The forward propagation T^(l) = σ(ΦĀΦ^T T^(l-1) W) replaces the full adjacency Ā ∈ R^{N×N} with ΦĀΦ^T ∈ R^{M×M}. Since Φ^T Φ cannot approximate I when M ≪ N, this substitution introduces unanalyzed spectral distortion. The paper acknowledges this trade-off as Obstacle II and defers a bound to Appendix D.4, but the severity of this approximation directly affects the method's viability and is not adequately discussed in the main text.

- **"You-Only-Sample-Once" branding is somewhat misleading:** Algorithm 1 Line 3 re-computes T^(0) = ΦUĤ every epoch inside the while loop. Since U is updated each iteration, the effective information passing through the sampling process changes every epoch. What is computed once is Φ (the sampling matrix structure), not the actual sampling operation. The branding obscures this distinction.

- **Storing Ĥ^(L) ∈ R^{N×d} partially negates memory savings:** Algorithm 1 Line 17 updates Ĥ^(L) via gradient descent, requiring storage and updates of an N×d matrix — the same size as the full embedding matrix. If N is very large (e.g., 2.4M for ogbn-products), this memory footprint may be problematic.

- **Preprocessing cost for eigenvalue computation not included in timing:** The construction of Φ requires computing all N eigenvalues of the normalized Laplacian (Section 5.3). For large graphs (e.g., ogbn-products with 2.4M nodes), this is a non-trivial one-time cost that is not reflected in the reported "sampling time" in Figure 2.

- **Weak baseline tuning for some methods:** In Table 1, FastGCN achieves 0.438 on ogbn-arxiv while GraphSAGE achieves 0.72 — a gap far larger than typically reported for these methods on this dataset. This raises concerns about whether all baselines were competitively tuned.

### Trivial
None.

## Nice-to-Haves

- Report reconstruction error ‖H^(L) − Ĥ^(L)‖_F / ‖H^(L)‖_F per epoch during training to verify whether reconstruction quality degrades as U drifts from initialization.
- Include a full-batch GCN baseline in Table 1 to directly test the "lossless" claim.
- Clarify the actual dimensions of U used in the implementation (N×K? N×N?) and analyze the corresponding scalability implications.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"U is N×N making Stiefel projection O(N³) and completely infeasible" (Harsh Critic Issue 1):** The critic assumed U must be N×N based on the double-sided orthogonality constraint UU^T = U^T U = I. While the paper does state both constraints in Eq. 8, Algorithm 1 Line 16 only enforces U^T U = I, which is compatible with U ∈ R^{N×K} on the Stiefel manifold (projection cost O(NK²), not O(N³)). The dimensionality inconsistency is real (and noted as a Major weakness above), but the critic's conclusion that the implementation "must do something fundamentally different" is overstated — a rectangular U with K ≪ N is a plausible and feasible implementation that simply contradicts one part of the stated constraint.

- **"Forward propagation equation is mathematically inconsistent with the CS measurement model" (Harsh Critic Issue 2):** The paper explicitly designed ΦĀΦ^T as a compressed-domain operator and acknowledges this as Obstacle II (Section 4), trading per-layer reconstruction (faithful but slow) for output-layer-only reconstruction. This is a methodological design choice, not an error. The concern about unanalyzed spectral distortion is valid and included as a Minor weakness, but calling it "mathematically inconsistent" with the CS model treats the per-layer reconstruction approach as the only valid option, when the paper explicitly discusses and motivates the alternative.

- **"Storing Ĥ^(L) has the same memory footprint as the full embedding matrix, negating memory advantage" — kept as Minor since it does partially negate memory savings, but removed the implication that it negates *all* advantages. The memory savings from not storing intermediate H^(l) for l=1,...,L-1 still apply.

- **RIP violations from changing U (Harsh Critic Issue 3):** Kept as Major overclaim, but removed the implication that this makes the entire CS framework "decorative." The CS framing still motivates the sampling design (Φ construction) and provides useful inductive biases, even if the theoretical lossless guarantee does not formally transfer.

## Novel Insights

The paper reveals an interesting tension in applying compressed sensing to learned representations: CS theory assumes fixed, known bases, but GNN embeddings evolve during training, making the sparse representation basis inherently dynamic. YOSO's solution — learning U jointly with GNN parameters — is an elegant way to align representation sparsity with the model's needs, but it fundamentally breaks the CS contract that guarantees lossless recovery. This suggests that "compressed sensing-inspired GNN sampling" and "theoretically lossless CS reconstruction" are two separate goals that cannot be simultaneously achieved with the current formulation, and future work would need to either constrain U's evolution or develop new CS-style guarantees for jointly optimized bases.

## Suggestions

- Replace "lossless" with "near-lossless" or "high-fidelity" throughout, and explicitly state that the CS reconstruction guarantee motivates but does not formally ensure the method's accuracy preservation, since U is learned rather than fixed.
- Add a full-batch GCN baseline (even on smaller datasets) to directly validate accuracy preservation claims.
- Explicitly declare the dimensions of U used in experiments (presumably N×K for some K) and discuss how K affects the accuracy-scalability tradeoff.
- Report the Laplacian eigenvalue computation time as a separate preprocessing line in Figure 2's timing breakdown.

## Score and Decision

**Calibration comparison:**

- **High anchors (avg > 7):** pPyJyeLriR (7.50, scalable graph propagation), SG1R2H3fa1 (7.50, random walks for efficient GNNs), C61sk5LsK6 (7.00, lossless data pruning). These papers have clear theoretical grounding matching their empirical claims, with no gap between theory and implementation. YOSO is significantly weaker on theoretical soundness.
- **Medium anchors (avg 4–6):** 2soZBUoG3n (4.25, StructDrop GNN sampling — split reviews between theoretical novelty concerns and practical efficiency), ghH6YYDs15 (4.67, CS theory applied to sparse autoencoders — similar pattern of CS theory overclaimed given evolving bases), QcMdPYBwTu (5.75, scalable implicit GNN with efficient mini-batch). YOSO most closely resembles ghH6YYDs15 and 2soZBUoG3n — strong empirical efficiency results with overclaimed theoretical guarantees and baseline tuning concerns.
- **Low anchors (avg < 3):** vAoyZWyDEc (2.50, incomputability theory vs. numerical experiments gap), bEgDEyy2Yk (1.0, O(n²) implementation claims but impractical theoretical algorithm). YOSO is above these — it has real empirical results and a workable implementation, not a purely theoretical paper with fatal flaws.

YOSO sits squarely in the medium band. It has genuine empirical contributions (~75% training time reduction, competitive accuracy) but overclaimed theoretical guarantees and notable gaps (U dimensionality, missing full-batch baseline, misleading "lossless" branding). This is very similar to 2soZBUoG3n (4.25) and ghH6YYDs15 (4.67). The overclaim is more severe in YOSO given the "lossless" language, but the empirical gains are also more clearly demonstrated across more datasets.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>