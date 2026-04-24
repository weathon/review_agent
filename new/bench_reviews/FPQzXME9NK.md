Now let me search for calibration papers.Now I have sufficient information to write the final review. Let me also check the medium-scoring calibration anchors.Now I have all the information I need. Let me write the final review.

---

## Summary

This paper introduces the Spherical Tree-Sliced Wasserstein (STSW) distance, a novel metric for comparing probability measures on the hypersphere. By replacing the traditional one-dimensional projection domains (great circles or semicircles) of prior spherical sliced Wasserstein methods with *spherical trees* — quotient spaces formed by gluing k spherical rays at a common root — the method retains closed-form OT computation while capturing richer topological structure. The paper provides a complete theoretical foundation (metric property, Radon transform injectivity under O(d+1)-invariant splitting maps, orthogonal invariance) and evaluates STSW on four tasks: gradient flow, self-supervised learning, earth density estimation, and sliced-Wasserstein autoencoding.

---

## Strengths

- **Theorem 3.3 and closed-form computation (Eq. 19)**: The construction of spherical trees as tree metric spaces via stereographic projection and quotient topology is carefully executed. Theorem 3.3 guarantees that OT on spherical trees admits the closed-form tree-Wasserstein expression of Eq. (3), and Eq. (19) makes this entirely explicit as a cumulative-sum scan — enabling full GPU parallelization without linear programming or binary search, unlike semicircle-based methods.

- **Theorem 4.3 (injectivity) and Theorem 5.2 (orthogonal invariance)**: The proof that the spherical Radon transform is injective for O(d+1)-invariant splitting maps (Theorem 4.3) is a non-trivial and necessary result for STSW to be a proper metric rather than a pseudo-metric. Theorem 5.2's O(d+1)-invariance of STSW is a geometrically meaningful property that prior S3W variants do not trivially enjoy, and is obtained cleanly through the O(d+1)-invariance of the splitting map.

- **Consistent empirical improvement across all four tasks**: STSW achieves the best results on log W₂ and NLL in the gradient flow experiment (Table 1: −4.69 vs. −4.39 for ARI-S3W(30)), best SSL accuracy on both encoded and projected features (Table 2: 80.53% vs. 80.08%), best NLL across all three earth density estimation datasets (Table 3), and best log W₂ and NLL in SWAE (Table 4). Notably, STSW achieves 1.89s vs. ARI-S3W(30)'s 20.25s runtime in Table 1.

- **Practically grounded splitting map (Eq. 14–15)**: The proposed β and softmax-based splitting map are O(d+1)-invariant (verified formally), handle the antipodal degeneracy explicitly (β = 0 at y = ±x), and come with a tunable sparsity parameter ζ. The code is publicly available.

---

## Weaknesses

### Fatal
None.

### Major

- **Inconsistent rotation budgets for ARI-S3W across tables**: ARI-S3W is compared with 30 rotations in Table 1 (gradient flow), 50 rotations in Table 3 (earth density estimation), and only 5 rotations in Table 4 (SWAE). Since ARI-S3W's performance scales with the number of rotations (Table 1 shows RI-S3W(1) at −3.12 and ARI-S3W(30) at −4.39), using ARI-S3W(5) in Table 4 artificially weakens the baseline. The margin by which STSW outperforms ARI-S3W in Table 4 (log W₂: −3.4191 vs. −3.3935) is narrow, and it is plausible that ARI-S3W at 20–30 rotations could match STSW. The paper does not justify why 5 rotations are used for SWAE when the gradient flow experiment used 30. This inconsistency undermines the strength of the SWAE comparison.

### Minor

- **Missing ablation on ζ and k in the main paper**: The softmax temperature ζ and the number of tree edges k are the two hyperparameters that most directly control STSW's behavior. The paper notes that "as |ζ| increases, the resulting value of α tends to become more sparse" (§4) but provides no ablation showing sensitivity to these choices. Since the appendix (which the paper explicitly references for experimental details in Appendix B) is unavailable to reviewers in stripped form, it is unclear whether such ablation exists. If not, understanding of the method's sensitivity to these parameters is incomplete.

- **Computational hyperparameters of STSW (L, k) absent from table captions**: Table 1 reports N_R = 30 for ARI-S3W but gives no comparable information for STSW's L (number of projecting trees) and k (number of edges per tree). Since the runtime of STSW scales as O(nkL), the comparison of 1.89s vs. 20.25s is opaque without knowing the effective computational budget of STSW. The paper directs readers to Appendix B, which is stripped, but ideally this would appear in the table caption.

- **SSL experiment is narrow in scope**: Table 2 reports results only on CIFAR-10 with ResNet18. The performance gap over ARI-S3W(5) is 0.45% on encoded features and 1.66% on projected features. While this follows the convention of prior work in the same line (Bonet et al. 2022, Tran et al. 2024b similarly use CIFAR-10 + ResNet18), a single dataset with no confidence intervals does not constitute strong evidence for consistent SSL improvement.

- **SWAE reconstruction quality (BCE) is not discussed**: STSW's BCE in Table 4 (0.6341) underperforms SSW (0.6309, the best) and SW (0.6314). The paper notes this only briefly ("though its BCE slightly underperforms the others"). Since BCE measures reconstruction fidelity — the primary purpose of an autoencoder — this trade-off between latent-space regularization quality and reconstruction quality deserves a more substantive discussion.

### Trivial
None beyond what is addressed above.

---

## Nice-to-Haves

- An ablation varying k ∈ {2, 3, 5, 10} and ζ ∈ {large negative, 0, large positive} on at least one task would significantly strengthen understanding of the method's sensitivity and help practitioners set these hyperparameters.
- Reporting STSW's (L, k) values alongside baselines' rotation counts in every table would make computational comparisons transparent.
- Including one additional dataset in the SSL experiment (e.g., STL-10) would substantially strengthen the claim of consistent SSL improvement.
- An experiment or geometric argument quantifying the practical impact of the antipodal degeneracy (β = 0 for y near −x) would address a non-trivial design choice currently acknowledged but unexplored.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: Missing Quellmalz et al. (2023) baselines.** Per rules, missing related work comparisons are not cited — cannot confirm whether these methods are directly applicable as baselines in the same experimental protocol.

- **Harsh Critic: Root-antipode degeneracy as a major weakness.** The paper explicitly handles this case in Eq. (14) with β = 0 and notes it in Figure 2's caption. This is a design choice that is acknowledged, not an oversight. Removed as a major issue (retained only as a nice-to-have for deeper investigation).

- **Harsh Critic: "Claim that spherical trees 'better capture topological information' is never formally grounded."** This is a conceptual motivation, not a formal claim. The paper does not state a theorem about information preservation vs. SSW; it is an informal motivation. Criticizing an informal motivating claim as if it were a theorem is not fair.

- **Harsh Critic: No statistical analysis (sample complexity, etc.).** This is appropriate methodology for an empirical systems paper in this space. Per soft rules, moved to nice-to-have territory; per hard rules (demanding theory for empirical paper outside field norms), removed as a weakness.

- **Harsh Critic/Strength Finder: "Topological expressiveness is a key conceptual advance."** This is too informal and not supported by a formal result comparing information content. Removed as a listed strength; it is a motivating intuition, not a demonstrated contribution.

---

## Novel Insights

The most genuinely novel observation from this review, going beyond the paper's stated contributions, is the following: the use of O(d+1)-invariant splitting maps as a **sufficient condition** for both (a) injectivity of the spherical Radon transform and (b) orthogonal invariance of the resulting distance is a tightly connected two-for-one result. The same algebraic structure that makes STSW a proper metric also makes it rotationally symmetric — these are not independent design goals but consequences of a single structural choice. This coupling is under-emphasized in the paper and represents the deepest theoretical insight: constructing a splitting map that respects the symmetry group of the sphere automatically yields a metrically well-behaved and geometrically natural distance. Future spherical OT methods could exploit this principle more broadly.

---

## Suggestions

1. **Ensure ARI-S3W is run at a computationally matched budget** (e.g., with N_R scaled so that runtime ≈ STSW's runtime) in Table 4, and either justify the choice of N_R = 5 for SWAE or supplement with higher-rotation results.
2. **Report L and k for STSW** alongside N_R for ARI-S3W in all table captions, or at minimum in the text near each table.
3. **Add ablation on ζ and k** to main paper (even a single figure/table varying these on one task).
4. **Expand the discussion of the reconstruction-regularization trade-off** in §6.4 with one or two sentences examining why STSW trades slightly worse BCE for better distributional metrics.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Decision | Relevance |
|---|---|---|---|---|
| TSW-SL (flat predecessor) | EKaVO0ceh8.md | 6.0 | **Reject** | Highly relevant: same framework, flat space; STSW is the sphere extension with additional theory and more experiments |
| LSSOT (spherical sliced OT) | fgUFZAxywx.md | 7.5 | **Accept Spotlight** | Very relevant: spherical sliced OT, similarly rigorous theory; LSSOT has linear embedding, STSW has tree embedding — comparable depth |
| IFGW (GW for graphs, low quality) | Aku2I3z4aV.md | 2.6 | **Reject** | Low anchor: clearly weaker than STSW — no meaningful theoretical depth, poor empirical results |
| Dynamic OT (low scoring OT paper) | ueQ6T58ZAK.md | 4.0 | **Reject** | Low-medium anchor: interesting idea but experiments not convincing |

**Reasoning:** STSW is clearly above the low anchors (2.6, 4.0) and shares structure with the medium anchor TSW-SL (6.0). The key differentiator is that STSW adapts the framework to the sphere — a non-trivial topological challenge — adds O(d+1)-invariance as a new theoretical result, and demonstrates improvement on four sphere-specific tasks rather than TSW-SL's three general tasks. This places STSW above the rejected TSW-SL. Compared to LSSOT (7.5, Spotlight), STSW is somewhat narrower in experimental scope (single-dataset SSL, inconsistency in baseline rotations) and lacks ablations. I place STSW between the two at **6.5**, representing a moderate accept: the theoretical contributions are genuine, non-trivial, and carefully executed; the experiments broadly support the claims; but the empirical evaluation has real gaps that prevent it from reaching Spotlight-level confidence.

**Originality**: Good — spherical tree construction is novel and non-trivial.
**Importance**: Good — spherical data appear in many practical domains.
**Claim support**: Adequate — strong across 3 of 4 tasks, with concern about SWAE comparison.
**Experimental soundness**: Adequate with noted inconsistency in rotation budgets.
**Clarity**: Good — theory is well-presented; experimental setup could be more transparent.
**Value to community**: Good — provides public code, solid theory, broad empirical validation.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>