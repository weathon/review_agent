=== CALIBRATION EXAMPLE 51 ===

# Final Consolidated Review
## Summary
The paper proposes SIIHPC (also written as SIHPC throughout), an Incomplete Multi-View Clustering (IMVC) method that addresses two limitations of prior work: (1) ignoring missing samples when constructing bipartite similarity, replaced here by a Similarity-Level Imputation (SLI) module that recovers missing similarity entries via a shared consensus graph; and (2) using a single prototype count per view, replaced by a group of hybrid prototype quantities {1k, 2k, 3k, 4k, 5k} (where k is cluster count) per view. The combined objective is optimized via an alternating four-step scheme, with a monotonicity-guaranteed inner-loop auxiliary function (Theorem 1). Experiments on six datasets (up to 70k samples) demonstrate favorable results and dramatically lower memory usage than competing methods.

---

## Strengths

- **Compelling scalability and memory efficiency with strong empirical evidence.** Table 3 shows SIIHPC consuming 5.13 GB and 6.01 GB on FASHMINST and YOUTUBEFACE respectively, against 123.42 GB and 126.28 GB for OSLFIMVC and 99.01 GB and 82.96 GB for SAGL — reductions of 20–24×. Many competing methods cannot run at all on these datasets. This is a concrete and differentiating practical contribution.

- **Significant and consistent benefit of the SLI module, quantitatively demonstrated.** Table 4 shows the no-SLI ablation collapsing badly on large-scale datasets (e.g., YOUTUBEFACE 30%: ACC 46.19 → 76.29; VGGFACEFIFTY 30%: ACC 6.71 → 12.52). The gains span all six datasets and all three missing ratios, isolating a clear contribution of the imputation component.

- **Theoretically grounded inner-loop optimization.** Lemmas 1–2 and Theorem 1 rigorously establish monotonic increase of the auxiliary function g for the H_{v,s} sub-problem, and Figures 3–4 experimentally confirm this property across multiple iterations of the outer loop. The approach of using SVD-based closed-form updates while maintaining orthogonality is technically clean.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Stopping criteria in Algorithm 1 and Algorithm 2 are logically inverted.** Algorithm 1 (line 109) reads: `while (g(H^{r+1}) - g(H^r)) / g(H^r) ≤ 1e-3 do`. Since Theorem 1 proves g is monotonically increasing and the improvement is always non-negative, this condition instructs the loop to *continue* when the relative gain is small (converged) and to *stop* when it is large (early in optimization). The correct convergence criterion should terminate the loop once improvement is below the threshold, i.e., the condition should be `≥ 1e-3` (continue while still making meaningful progress). Algorithm 2 (line 162) has the same inversion for the outer-loop decrease condition. This is either a systematic pseudocode error or a fundamental logical mistake in the algorithm description — in either case it must be resolved, as it calls into question whether the implementation matches the manuscript.

- **Table 5 SPQ rows for MR=70% (lines 351–356) are exactly identical to the MR=50% rows (lines 345–350).** All six datasets and all five prototype sizes have the same numerical values. This is almost certainly a copy-paste error and calls the reliability of the ablation into question. The authors must provide correct 70% results.

- **No formal convergence proof for the outer alternating optimization (Algorithm 2).** Theorem 1 and Lemmas 1–2 apply only to the inner-loop H_{v,s} sub-problem. No analogous result is provided for Steps 2–4 or for the joint alternating scheme. The paper shows convergence empirically for three datasets (Fig. 2), but this is not a proof. At minimum, the paper should argue that each sub-step is non-increasing in the joint objective, which would follow straightforwardly from the structure of Steps 2–4 if the QP sub-problems are solved exactly — but this argument is absent.

- **No hyperparameter sensitivity analysis.** The method introduces two hyperparameters (λ and β) that appear in the core objective (Eq. 2) and the weight update (Eq. 9), respectively. There is no figure or table showing how performance varies across a range of λ and β values on any dataset. Without this, the reproducibility and practical deployability of the method are undermined.

### Minor

- **Notational inconsistency in J_{v,s}.** Eq. (6) and the surrounding text (line 126) give J_{v,s} = G_s − H_{v,s}^T D_v W_v W_v^T **M_v**, but Remark 2 simplifies the construction of J as H_{v,s}^T D_v ⊙ B_v (omitting M_v), and the structural dimensions of the full expression with M_v ∈ R^{n×(n−n_v)} are inconsistent with G_s ∈ R^{m_s×n}. The most likely intent is J_{v,s} = G_s − H_{v,s}^T D_v W_v W_v^T (both m_s×n), from which missing entries are extracted via M_v indexing as a separate step. This should be stated unambiguously.

- **Table 2 third-column group is unlabeled.** For each dataset, three groups of columns are shown (presumably 30%, 50%, 70% missing rates), but only the first two groups are labeled in the header. The third group is implied but never stated. This should be labeled.

- **Table 6: one counterexample to the claimed uniform improvement.** The paper states "AW-PQ scheme makes performance improvement" (line 358). However, for FASHMINST at 50% MR, ETPQ achieves PUR=64.84 while "Ours" achieves PUR=64.64 — a marginal regression on one metric. While this does not undermine the overall conclusion, the paper should acknowledge this exception rather than claiming unconditional uniform improvement.

- **Missing baseline entries unexplained.** For large datasets (VGGFACEHUND, YOUTUBEFACE, FASHMINST), many baseline methods show blank cells in Tables 2 and 3. Line 275 says they "cannot normally run," but the reason (OOM, timeout, numerical failure) is never stated. This matters for assessing whether the comparison is fair.

### Tiny

- **Inconsistent method name: SIIHPC vs. SIHPC.** The abstract and introduction use "SIIHPC" while the experiments section (line 274), conclusion (line 400), and Table 3/Section 5.3 consistently use "SIHPC." The final paper should adopt one name throughout.

- **O(n) complexity constant factor.** Remark 5 claims O(n) overall complexity, which is technically correct when m_s and S are fixed constants. However, for large k (e.g., VGGFACEHUND with k=100 and m_s up to 500), Step 3 requires solving n=36,287 QPs each of dimension 500, with a hidden O(m_s^3) constant (O(1.25×10^8) per pass). The paper acknowledges this in Remark 3 but should be more explicit about the practical constant's dependence on k.

---

## Nice-to-Haves

- Provide justification for the prototype group grid {1k, 2k, 3k, 4k, 5k} and test sensitivity to S (number of groups). The current choice is purely empirical.
- Evaluate under non-random (MAR/MNAR) missing mechanisms to characterize generalization beyond MCAR.
- Test extreme missing rates (e.g., 90%) to characterize the breakdown point of the SLI module.
- Include recent scalable deep IMVC baselines (2023–2024) as additional comparison points, even if the method itself is shallow.
- Provide convergence curves for all six datasets and all missing rates rather than just three dataset/one-rate examples.
- Publicly release source code, which would substantially strengthen the reproducibility argument.
- An imputation quality check (e.g., comparing imputed similarity values to held-out observed entries) would make the "leveraging latent information from missing samples" claim more concrete.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Venue fit (shallow vs. deep learning):** The reviewer argues the paper lacks deep neural networks and is therefore a poor fit for ICLR. This is scope creep — ICLR accepts work across broad ML topics including shallow methods, and the paper should be evaluated on the quality of its own contribution, not whether it uses deep learning.
- **"Regularization destroys clustering structure":** The term λ‖G_s‖²_F is constrained by −1 ≤ G_s ≤ 1, which prevents the zero-collapse the harsh critic worries about. The regularizer is standard, and while its specific role deserves brief discussion, the criticism as stated is overstated.
- **Statistical significance testing:** The harsh critic demands standard deviations and significance tests. For large-scale clustering benchmarks with costly experiments, single-run evaluation is the accepted norm in this subfield. This should not be held against the paper.
- **Circular dependency in imputation:** Alternating optimization with interdependent variables (Q via G, G via Q) is standard in matrix factorization and graph learning. The paper explicitly adopts an alternating framework, which is the field-standard approach. This is not a flaw unique to this paper.
- **"Abstract claim is overstated (uses information from missing samples)":** The critic argues that the imputation uses only other views' observed data, not the missing sample's own data. This is by definition what cross-view imputation does — the method correctly leverages the consensus graph (built from all available views) to fill in missing entries. The claim is appropriate for the mechanism described.
- **Only traditional baselines:** The reviewer demands deep learning baselines. While this would be strengthening (moved to Nice-to-Haves), its absence is not a fatal flaw given the method's own paradigm is shallow/spectral.

---

## Novel Insights

The observation that similarity-level (rather than feature-level) imputation can leverage cross-view structural consistency without ever recovering the raw missing features is interesting and underexplored. The consensus graph G_s acts as a cross-view bridge at the similarity rather than the data level, which is a conceptually clean separation that avoids the noise amplification typical of feature-space imputation. The substantial collapse of performance when SLI is removed (Table 4, especially on YOUTUBEFACE and FASHMINST where the no-SLI baseline achieves roughly half the ACC) suggests this graph-level recovery is doing substantial work, making it a genuine contribution worth examining in future work. The adaptive prototype weighting scheme (Eq. 11 closed-form solution) that automatically down-weights poorly-fitting prototype scales is also a clean design choice that generalizes straightforwardly to any bipartite prototype framework.

---

## Suggestions

1. **Fix the while-loop conditions in both algorithms:** Change `≤ 1e-3` to `≥ 1e-3` (and analogously in Algorithm 2) so the loop continues while the improvement is above the threshold and terminates when converged. Verify the implementation matches.

2. **Correct Table 5:** Provide actual 70% MR results for the SPQ ablation rather than copying the 50% values.

3. **Add a convergence argument for the outer loop:** Even an informal argument showing each sub-step is non-increasing in the joint objective would suffice; Steps 2 (direct assignment), 3 (QP minimization), and 4 (closed-form minimization) all obviously decrease or maintain the objective, making this argument available and worth including.

4. **Add a hyperparameter sensitivity figure** (e.g., ACC vs. log-λ and log-β on two representative datasets) to support reproducibility.

5. **Unify the method name** to SIIHPC or SIHPC consistently across the full paper.

6. **Fix the J_{v,s} definition** to make its dimensions unambiguous and consistent with Remark 2.

7. **Label all three missing-rate column groups in Table 2** and provide footnotes explaining why specific baselines are absent from large-dataset rows.

---

**Evaluation summary:**
- *Novelty:* Moderate — the two components (similarity-level imputation + hybrid prototype counts) are individually grounded in prior work, but their integration into a unified linear-complexity framework with a clear empirical payoff is a genuine contribution.
- *Technical soundness:* Moderate — the inner-loop analysis is rigorous, but the outer-loop convergence is unproven, and the apparently inverted stopping criteria are a serious presentation problem.
- *Empirical support:* Good — six datasets, three missing rates, 13 baselines, and meaningful ablation coverage; the scalability advantage is particularly convincing.
- *Significance:* Moderate-to-good — the practical impact for large-scale incomplete multi-view clustering is real, though the theoretical depth is limited.
- *Clarity:* Below average — many undefined or inconsistent symbols, inverted algorithm conditions, unlabeled table columns, and inconsistent naming weaken readability substantially.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
