Now let me read the human reviews directory and fetch calibration data.Now I have enough information to write the final consolidated review.---

## Summary
GLIDE proposes a causal graph discovery framework grounded in the distributional invariance of P(effect | cause) under changes to P(cause). The paper makes three contributions: (1) an invariance test via optimal downsampling that varies source variable priors without changing causal conditionals; (2) an O(d²) parent search using maximal cliques over the Markov blanket augmented bidirectional graph G'(X); and (3) extensive empirical evaluation on synthetic and real-world datasets up to 1041 variables, consistently outperforming existing baselines in SHD and spurious rate.

---

## Strengths

- **Principled invariance test (Theorem 1, Eq. 1):** The formal result that V[P₊(X|Z)] > 0 ⟹ Z ≠ Pa[X] provides a theoretically grounded, assumption-light causal test. Unlike NOTEARS and model-based methods, GLIDE imposes no functional-form assumptions, explaining its robustness across L-G and nL-nG settings in Figures 2–4.

- **Provably optimal downsampling construction (Theorems 4 and 5):** The closed-form minimum downsampling rate (Eq. 4) and sampling-without-replacement procedure give a principled, non-arbitrary way to construct synthetic environments. This is technically precise and avoids the ad hoc data-splitting found in related invariance-based work.

- **Convex hull characterization of informative priors (Theorem 6, Eq. 5):** The explicit characterization of the feasible source prior subspace with γᵢ ≥ γ₀ enables principled Dirichlet sampling and K-means clustering. This connects the abstract theory cleanly to the practical implementation.

- **Maximum basis construction (Theorem 3):** An O(d²) greedy algorithm identifies the maximum basis — a practical surrogate for source variables — without requiring prior knowledge of graph structure. This is a non-trivial algorithmic contribution that makes the framework operational.

- **Empirical breadth and large-scale results (Table 2, Figures 2–4):** GLIDE achieves best SHD across all 7 real-world datasets and best spurious rate on 5/7. On Munin (1041 nodes), GLIDE reaches 1.8% spurious rate vs. 42.4% for GIES and 883 SHD vs. 1235 for GIES, a substantial improvement at a scale few prior methods attempt.

---

## Weaknesses

### Fatal
None.

### Major

- **Completeness of the maximal clique search is unproven (Theorem 7, Section 4.3):** Theorem 7 establishes that Pa[X] corresponds to *a* clique in G'(X), not necessarily a *maximal* clique. The algorithm enumerates only maximal cliques. If Pa[X] ⊂ C for a strictly larger maximal clique C, Pa[X] is never directly tested. The concern is concrete: if C = Pa[X] ∪ {v} where v is a co-parent of X's child, then by the local Markov condition X ⊥ v | Pa[X], so P₊(X|C) = P₊(X|Pa[X]) and both achieve ≈0 variance under Eq. (3). In this scenario, the algorithm reports C instead of Pa[X], producing spurious edges. The paper does not prove Pa[X] is always maximal in G'(X), nor does it characterize graph configurations where the superset degeneracy fires. This undermines the completeness claim for the parent-finding algorithm. The non-zero spurious rates observed empirically (e.g., 23% on Water, Table 2) are consistent with this failure mode occasionally occurring.

- **Sufficient condition for unique minimum-variance identification is established only for infinite m (Section 4.1, Eq. 3):** Theorem 1 proves the necessary direction: V > 0 ⟹ Z ≠ Pa[X]. The parent-selection rule Eq. (3) requires the sufficient direction: V ≈ 0 uniquely at Z = Pa[X]. The paper states this bidrectionality holds "when m is infinitely large" and is "highly accurate" for large finite m, but no finite-sample guarantee or convergence rate is provided. The degeneracy described above (supersets that also achieve ≈0 variance) is exactly the natural failure mode. The theoretical justification for Eq. (3) as a reliable minimum-variance selector is thus incomplete.

### Minor

- **Scalability framing in the abstract may mislead (Abstract, Table 2):** The abstract states "reducing the processing time by up to 25×." This is accurate when comparing against MLP-NOTEARS on the extreme continuous benchmark, but GIES completes Munin (1041 nodes) in 61.5 seconds while GLIDE requires 6,200 seconds — over 100× slower. The paper correctly acknowledges the accuracy-runtime tradeoff in Section 5.3, but the abstract's unqualified "up to 25×" claim invites misreading that GLIDE is generally faster than the state of the art. It would be more accurate to specify that this speedup applies relative to model-based continuous baselines.

- **PC absent from continuous data experiments without explanation (Table 1 vs. Figures 2–3):** Table 1 marks PC as applicable to L-G and nL-nG data, yet PC does not appear in Figures 2 or 3. The paper gives no justification (e.g., timeout). An informative alternative would be to report the node count at which PC times out, which would actually strengthen the scalability narrative.

- **Identifiability assumptions not stated:** The method claims to recover "the true causal graph" (Section 3), but the approach — like PC — can at best recover a graph up to Markov equivalence under faithfulness from observational data. Footnote 1 acknowledges this limitation but the main text does not explicitly state the faithfulness assumption or note that the output may represent a CPDAG rather than a unique DAG. Clarifying this would improve precision without affecting any experimental claims.

### Trivial
None beyond what is already addressed.

---

## Nice-to-Haves

- **Ablation on basis variable quality vs. source variables:** The method's theory is stated for source variables; the algorithm uses the maximum basis as a surrogate. A targeted experiment showing performance as a function of how close the basis is to the true sources (e.g., on small graphs where sources are known) would close the gap between theory and implementation.

- **Statistical power characterization:** A study of how false-positive/negative rates of the invariance test vary with m and n would clarify whether GLIDE's empirical gains depend on a favorable n/m regime specific to the tested benchmarks.

- **Edge-level confidence scores:** The variance value from Eq. (3) is a natural edge-level confidence proxy. Outputting per-edge scores (e.g., gap between winning and runner-up candidate variances) would increase practical utility and make borderline edges visible.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic — FCI absent from Figure 2:** Factually incorrect. The figure caption explicitly lists "FCI (cyan plus)" as one of seven compared methods. The 64.17% SHD reduction over FCI is valid. Removed.

- **Harsh Critic — Basis variable substitution insufficiently established in main text:** The proof is deferred to Appendix B.1. Per rules, this is removed because the appendix exists in the original submission; the parser strips it.

- **Harsh Critic — Missing proofs for Theorems 2, 3, 4, 5, 6, 7:** All proofs are deferred to Appendices A.1–A.7. Removed per the rule prohibiting criticism of missing appendix content.

- **Harsh Critic — Binning hyperparameter sensitivity analysis absent from main text:** This is a reproducibility/hyperparameter nitpick that belongs in an appendix; the paper does use binning transparently. Removed.

- **Harsh Critic — O(d²) complexity assumes p constant, which could be large in adversarial settings:** The paper empirically validates p ≤ 13 across all tested benchmark graphs and in Appendix C.2.3 across varying topologies. Treating this as a fatal or major flaw misrepresents an empirically verified assumption. Downgraded/removed.

- **Harsh Critic — Definition 1 identifiability issue:** The paper explicitly acknowledges in footnote 1 that exact graph recovery is impossible from observational data alone. While the identifiability assumption (faithfulness) should be stated more prominently in the main text (captured under Minor above), the underlying criticism that the paper's problem formulation is fundamentally flawed is removed.

- **Strength Finder — "publicly available code" strength:** Generic and non-specific to this paper's technical contribution. Removed as a standalone strength.

---

## Novel Insights

The most genuinely novel observation in GLIDE is the explicit characterization of the *convex hull* of valid source priors (Theorem 6) and the minimum-downsampling-rate resampling (Theorems 4–5). Most prior invariance-based causal discovery work (e.g., ICP) assumes access to explicit environments; GLIDE constructs those environments internally from a single dataset by shifting the prior over basis variables. The connection between this shift and a minimum-information-loss downsampling problem — solved in closed form — is original and practically impactful, as it explains why GLIDE can work reliably even with modest sample sizes. The theoretical gap between the clique-existence guarantee (Theorem 7) and the maximal-clique search strategy is the main unresolved tension and would benefit from either a proof that Pa[X] is maximal under faithfulness + causal sufficiency, or an explicit characterization of when it is not.

---

## Calibration Anchors

| Path | Avg Score | Decision | Comparison to GLIDE |
|------|-----------|----------|---------------------|
| `/human_reviews/xByvdb3DCm.md` | 8.0 | Oral | Stronger theoretical grounding and broader claims; GLIDE below this bar |
| `/human_reviews/M0xK8nPGvt.md` | 7.5 | Poster | Hierarchical Bayesian approach with tighter theory; GLIDE below this bar |
| `/human_reviews/wmV4cIbgl6.md` | 7.33 | Spotlight | Benchmark contribution with real-world complexity; different type |
| `/human_reviews/eeJz7eDWKO.md` | 6.0 | Poster | Similar causal discovery with ML novelty; GLIDE is empirically stronger but has comparable theoretical gaps → closest anchor |
| `/human_reviews/iaP7yHRq1l.md` | 5.5 | Poster | Benchmarking paper; less novel algorithmically than GLIDE |
| `/human_reviews/0cZDnlw0WL.md` | 4.5 | Withdrawn | Causal discovery with discretization issues and unproven conditions; empirically weaker than GLIDE |
| `/human_reviews/or8wkKoBP4.md` | 4.0 | Reject | Purely theoretical, no experiments, mathematical mistakes — clearly weaker than GLIDE |
| `/human_reviews/4P76wCt9N5.md` | 3.0 | Withdrawn | Vague methodology, unsubstantiated claims — much weaker than GLIDE |

**Calibration reasoning:** GLIDE sits above eeJz7eDWKO (6.0) in empirical breadth (larger scale, more datasets, stronger accuracy gains) and in the maturity of the downsampling construction, but has comparable theoretical completeness gaps. The completeness gap (Issue 2) is more structurally concerning than the gaps in eeJz7eDWKO, pulling the score back toward 6.0. The 0cZDnlw0WL paper (4.5) had similar theoretical issues but weaker empirical results, confirming GLIDE is solidly above the low band. I anchor at 6.0.

---

## Score and Decision

**Originality:** High — the use of source prior perturbation via downsampling as an invariance test is genuinely new and connects disparate ideas (distributional invariance, minimal resampling, clique enumeration) in a principled way.

**Importance:** High — scalable, assumption-light causal discovery for large graphs is a pressing need.

**Claims support:** Moderate — the core empirical claims are well-supported; the theoretical completeness claims are not fully proven.

**Soundness of experiments:** Good — diverse benchmarks, confidence intervals, multiple graph types and sizes. Some baselines excluded from continuous experiments without clear justification.

**Clarity of writing:** Good — the high-level framework is well-explained; the mathematical exposition of Theorems 4–6 is precise.

**Value to community:** High — the Munin (1041 nodes) result with 1.8% spurious rate is a concrete contribution that practitioners can use.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>