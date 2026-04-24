Now let me search for calibration anchors.Now I have enough information to write the final review. Let me compile everything.

---

## Summary

YOSO proposes applying compressed sensing (CS) to GNN mini-batch training, sampling the node feature matrix once using a fixed matrix Φ, propagating the compressed representation through all GNN layers, and reconstructing the full embedding only at the final output layer. The stated goal is to eliminate per-batch/per-epoch resampling overhead while preserving accuracy equivalent to full-node training. Experiments on five OGB/standard datasets across node classification and link prediction show ~75% average total-training-time reduction with accuracy competitive against seven 2017–2019 baselines.

---

## Strengths

- **Concrete and informative bottleneck analysis (Figure 1(b)):** The paper clearly decomposes training time into Sampling, Mem2GPU, and Computation across all major baselines, providing quantified motivation. The finding that AS-GCN spends 55.6% of 3376 s on sampling while still underperforming simpler methods is a crisp and useful empirical observation.
- **Strong empirical efficiency gains (Figure 2, Table 1):** YOSO reduces average total training time by ~75% across five datasets and two tasks, with sampling overhead cut by ~96% on Reddit and ~80% on link-prediction datasets. Accuracy is competitive: matches the best methods on Reddit (0.967), ogbn-arxiv (0.72), and nearly so on ogbl-ppa (0.2238 vs 0.2263 best).
- **Breadth of evaluation:** The paper tests node classification and link prediction on datasets of varying scales (Reddit, ogbn-arxiv, ogbn-products, ogbl-ppa, ogbl-citation2), giving a reasonable picture of the method's behavior.
- **Convergence stability (Figure 3):** YOSO shows faster and smoother convergence curves than GraphSAGE, GraphSAINT, and FastGCN on ogbn-arxiv and ogbl-ppa, supporting the claim that the reconstruction-guided loss stabilizes training.

---

## Weaknesses

### Fatal

None. The empirical results are genuine, and while there are methodological issues, they do not fully invalidate the experimental contributions.

### Major

- **Eigenvalue-to-node sampling probability mapping is conceptually incorrect (Section 5.3).** The paper assigns sampling probability P(i) = λ_i / Σλ_j to node i, where λ_i is the i-th eigenvalue of the normalized Laplacian: *"The N nodes correspond to N eigenvalues from the spectral decomposition of Ã… These eigenvalues capture important structural properties of the graph, where larger eigenvalues correspond to more influential nodes."* However, the eigenvalues of the normalized Laplacian are ordered by frequency (from smooth/global at λ=0 to oscillatory/local near λ=2) and indexed by frequency mode, not by node. Node ordering in the graph is arbitrary and has no correspondence to spectral eigenvalue ordering. This conflation of frequency eigenvalues with per-node importance invalidates the claimed graph-structure-aware design of **S̃** and undermines the theoretical motivation for the structure-aware component of Φ. If the sampling probabilities are effectively arbitrary due to the meaningless node-to-eigenvalue mapping, the paper loses its key differentiator from a plain random Gaussian Φ — a comparison that is never performed.

- **Abstract claims "lossless" reconstruction while the body explicitly acknowledges accuracy loss.** The abstract states: *"nodes are sampled once at the input layer, followed by a **lossless** reconstruction at the output layer."* Yet Section 4 (Obstacle II) explicitly says: *"it may introduce some accuracy loss due to reduced intermediate layer information, it remains efficient if this loss is controllable."* Section 5.2 similarly notes: *"achieves improved efficiency at the cost of a slight reduction in accuracy."* This contradiction between the abstract's claim and the body's candid admissions constitutes a material overclaim. The paper does reference a bound in Appendix D.4, but that bound covers only the final reconstruction step, not the forward-propagation approximation error accumulated over L layers.

- **Forward-propagation approximation (Φ Ã Φ^T) lacks justification for the compressed-domain computation.** YOSO propagates T^(l) = σ(Φ Ã Φ^T T^(l-1) W^(l)) (Eq. 7). Full-domain propagation would give H^(l) = σ(Ã H^(l-1) W^(l)), but since H^(l-1) ≈ Φ^T T^(l-1), the compressed-domain analog requires Φ Ã Φ^T ≈ Φ Ã Φ^T, which holds only if Φ^T Φ ≈ I — impossible when M ≪ N. The paper does note this introduces accuracy loss (Obstacle II), but provides no error bound for the per-layer propagation approximation. The appendix bound (D.4) applies to the final reconstruction, not to the compounding layer-by-layer approximation error. This gap means the efficiency claims are credible but the theoretical claims about approximating full-node training are overstated.

- **Baselines are 5–7 years old (2017–2019).** All seven baseline sampling methods are from 2017–2019. The paper explicitly mentions linearization methods (SIGN, GAMLP) and historical-embedding methods in Section 2.1 and Appendix A but excludes them from comparison on the grounds that they "focus on sampling-based methods." While the scoping is stated, the effect is that the "state-of-the-art" claim is relative to a very dated comparison set. Many of these baselines would themselves be beaten by more recent efficient GNN training methods, weakening the paper's positioning.

### Minor

- **"You-Only-Sample-Once" framing is misleading regarding per-epoch computation.** Algorithm 1 places "Compute T^(0) = Φ U X̂" (Line 3) inside the "while not converged" loop (Line 2). Since **U** is updated every epoch, **T^(0)** is recomputed every epoch. The Φ matrix is fixed once, but the paper's core computation T^(0) changes at every epoch. The "once" in the title/abstract refers to once-per-epoch rather than once per training run, which is a more modest novelty than the name implies.

- **Inconsistency between Equation 7 and Algorithm 1 Line 5 in operand ordering.** Equation 7 writes σ(Φ Ã Φ^T **T^(l-1) W^(l)**) — W right-multiplies the aggregate. Algorithm 1 Line 5 writes σ(Φ Â Φ^T **W^(l) T^(l-1)**) — W left-multiplies. These are different operations and the ordering matters dimensionally. This is an internal inconsistency that makes the method difficult to reproduce as written.

- **Suspicious and uniform baseline times for ogbl-ppa (Figure 2(d)).** For the ogbl-ppa dataset, all baselines except GraphSage and VR-GCN show identical or near-identical training times (~44.57 s each), despite spanning structurally different sampling paradigms (node-wise, layer-wise, subgraph). This uniformity is implausible and suggests a measurement or configuration error for this dataset. YOSO's ~50% reduction on ogbl-ppa is a meaningful component of its headline "72.13% average" for link prediction; if the baseline numbers are unreliable, this figure is questionable.

- **No comparison to full-graph GCN as ground truth.** The theoretical claim is that YOSO emulates "full node participation." Without a row for full-graph GCN in Table 1, it is impossible to assess whether YOSO achieves the claimed equivalence or simply converges to a different solution with similar accuracy to other sampling methods.

### Trivial

- Section 5.3 asserts Φ is "row full rank" via a brief argument about Gaussian randomness; the argument that non-zero rows suffice for full rank is incomplete (linear dependence between rows is not precluded by nonzero rows alone).

---

## Nice-to-Haves

- **Ablation of structure-aware Φ vs. plain random Gaussian Φ:** Given the mathematical concerns with the eigenvalue-to-node mapping, an experiment testing whether the proposed Φ actually outperforms a pure Gaussian random Φ would directly validate the key design choice.
- **Visualization of learned U:** Showing whether the converged **U** aligns with Laplacian eigenvectors or is a purely data-driven artifact would clarify whether the CS motivation is substantive or decorative.
- **Comparison with more recent scalable GNN methods** (e.g., SIGN, GAMLP, or more recent historical-embedding approaches) as an optional extended comparison to strengthen the positioning.
- **Scaling to larger graphs** (MAG240M, IGB-large) to stress-test the memory footprint of storing U ∈ R^{N×N}.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Algorithm 1 places T^(0) computation inside the loop = the entire CS guarantee is broken"** — This is overstated. The paper explicitly acknowledges in Obstacle II that sampling-once and reconstructing at the end introduces bounded accuracy loss. The appendix (D.4) provides a proof that the paper points to. The concern about T^(0) being recomputed is real but only a framing/minor issue, not a structural failure. Retained as minor rather than major/fatal.
- **"The unknown-U approach abandons all CS guarantees"** — The paper proactively identifies this as its core challenge (Obstacle I) and addresses it with a joint optimization approach. The approach may not provide the same guarantees as classical CS, but the paper does not silently abandon this concern; it reformulates the problem. The abstract's "lossless" language is the real error (kept as major above), but the methodological choice per se is reasonable and acknowledged.
- **Criticisms about missing appendix proofs (D.1, D.2, D.3, D.4, C.4)** — Removed per policy; appendix sections are stripped by the parser and exist in the original submission.
- **Criticism about reproducibility (hyperparameter choices, full training logs)** — Removed per policy.
- **Strength: "Convergence stability supports the claim that reconstruction process stabilizes the learning path"** — Retained above as this is supported by Figure 3 with specific dataset citations.
- **Strength: "Robust construction of universal sampling matrix Φ satisfies RIP"** — Removed from strengths because the eigenvalue-to-node mapping (a major weakness verified above) undermines the claimed theoretical justification for this construction.

---

## Novel Insights

The most genuine insight in YOSO is the empirical observation, quantified in Figure 1(b), that sampling overhead is the dominant training-time bottleneck for modern GNN training methods (35–64% of total time), and that existing methods increase this overhead as they improve accuracy. The paper's response — fix the sampling matrix once and propagate in the compressed domain — is a creative reframing of the problem, even if the theoretical justification for the compressed-domain propagation is incomplete. The joint optimization of the sparse basis **U** alongside GNN weights via Stiefel manifold projection is also a practically interesting technique that allows the CS framework to operate without expensive explicit eigendecomposition. These insights are genuinely novel, though the theoretical apparatus built around them overpromises relative to what it delivers.

---

## Suggestions

1. **Fix the eigenvalue-to-node assignment** or replace it with a defensible node-importance measure (e.g., node degree, PageRank, or actual diagonal entries of the Laplacian). Provide an ablation comparing the proposed Φ against a plain random Gaussian baseline.
2. **Correct the abstract**: Replace "lossless" with "near-lossless with provable error bounds" and make clear that T^(0) is recomputed per epoch as U evolves.
3. **Resolve the Eq. 7 / Algorithm 1 ordering inconsistency** and provide one clear, dimensionally-verified description of the forward pass.
4. **Provide per-layer error analysis** or at least an empirical measurement of embedding error accumulation with depth to substantiate the "bounded accuracy loss" claim beyond the final reconstruction step.
5. **Include at least one more recent baseline** (post-2020 sampling method) to contextualize the SOTA claim.
6. **Report full-graph GCN performance** in Table 1 as a ground truth for the "full node participation" claim.

---

## Evaluation Summary

**Originality:** Moderate — applying CS to eliminate GNN resampling overhead is a creative and under-explored idea, but the theoretical framework has identifiable flaws. **Importance of research question:** High — training efficiency for large-scale GNNs is a pressing need. **Claims vs. support:** Weak-to-moderate — the efficiency claims are well-supported empirically, but the "lossless" and "full-node equivalence" claims are not. **Soundness of experiments:** Moderate — strong results across 5 datasets, but the suspicious ogbl-ppa baselines and absence of full-graph comparison reduce confidence. **Clarity of writing:** Fair — the main algorithm is described, but the Eq. 7 / Algorithm 1 inconsistency and the incorrect eigenvalue-to-node mapping indicate that the paper was not carefully proofread at the methodological level. **Value to community:** Moderate — the empirical efficiency gains are real and the CS framing is novel, but the theoretical justification needs to be substantially strengthened.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg score | Comparison |
|---|---|---|
| `/home/wg25r/review_agent/human_reviews/2soZBUoG3n.md` (StructDrop) | 4.25 (Reject) | Most topically similar: large-scale GNN training efficiency, speedup claims, somewhat questionable methodological aspects; rejected despite strong results from one reviewer. YOSO has broader evaluation but more fundamental theoretical issues. |
| `/home/wg25r/review_agent/human_reviews/H7z1gHsaZ0.md` (Staleness-based subgraph) | 4.00 (Reject) | GNN sampling efficiency; weak theoretical proof, limited gains. YOSO has stronger empirical gains but also stronger theoretical overclaims. |
| `/home/wg25r/review_agent/human_reviews/UqrFPhcmFp.md` (TOP / Message Invariance) | 6.25 (Accept Poster) | GNN large-scale training with approximation; accepted with strong results and reasonable theory. YOSO falls below this: outdated baselines, major eigenvalue-mapping flaw, abstract overclaim. |
| `/home/wg25r/review_agent/human_reviews/Gq7RDMeZi4.md` (MuseGNN) | 6.25 (Accept Poster) | Scalable GNN training with theoretical backing; above YOSO in theoretical grounding. |
| `/home/wg25r/review_agent/human_reviews/RzEWcuZQcA.md` (HashGIN) | 2.67 (Withdrawn) | GNN with random weights for 1-epoch training; seriously flawed. YOSO is well above this. |

YOSO sits between the rejected GNN efficiency papers (~4.0–4.25) and the accepted posters (~6.25). The empirical contributions (75% speedup, 5 datasets, two tasks) are more substantial than StructDrop or the staleness paper, but the major methodological issues (eigenvalue-to-node mapping error, "lossless" overclaim, outdated baselines, approximation without layer-by-layer error analysis) prevent it from reaching the level of TOP or MuseGNN. The paper's theoretical apparatus purports to justify a CS-based approach but has identifiable errors in the core construction (Section 5.3). Given these anchors, a score of **4.5** is appropriate — above the clearly flawed papers but below what would warrant acceptance as a poster without significant revision.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>