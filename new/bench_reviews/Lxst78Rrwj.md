Now let me search for calibration papers.Now I have enough information to write the final consolidated review. Let me compose it.

---

## Summary
GLIDE is a causal graph discovery framework exploiting the insight that P(X | Pa[X]) remains invariant to changes in the prior P(B) over source variables. It constructs synthetic environments via a downsampling scheme over observational data to test this invariance, then pairs the test with a DFS algorithm that restricts candidate parent sets to maximal cliques in a Markov-blanket-induced bidirectional graph, achieving claimed O(d²) complexity. The method is evaluated on synthetic continuous/categorical and seven real-world bnlearn benchmarks up to 1041 nodes, demonstrating competitive or superior performance relative to PC, GIES, FCI, NOTEARS, MLP-NOTEARS, DAS, and SCORE.

---

## Strengths

- **Novel operationalization of causal invariance (Theorem 1, Section 4.2):** The core insight—that P(X | Pa[X]) is invariant to source prior shifts—is well-grounded in the causal Markov factorization, and translating it into a practical test via Theorem 5's downsampling is non-trivial. This circumvents the need for interventional data or multiple real-world environments, which is a genuine methodological advance.

- **Elegant resampling scheme with formal characterization (Theorems 4–6):** The minimum-downsampling-rate guarantee (Theorem 4), the constructive resampling procedure (Theorem 5), and the characterization of valid source priors as a convex polytope (Theorem 6, Eq. 5–6) are concrete, well-formalized contributions.

- **Strong empirical results at large scale (Table 2):** On the 1041-node Munin dataset, GLIDE achieves 1.8% spurious rate vs. 42.4% for GIES, while PC times out entirely. Across all seven real-world datasets GLIDE achieves best SHD in 7/7 and best spurious rate in 5/7. This scale of evaluation is substantial for an observational causal discovery paper.

- **Versatility across data modalities (Table 1):** Unlike NOTEARS/MLP-NOTEARS/DAS/SCORE, which fail on categorical data, GLIDE handles continuous (L-G, nL-nG) and categorical data within a single framework, giving it broader practical applicability.

---

## Weaknesses

### Fatal
None.

### Major

- **Theoretical gap: basis variables used as surrogates for source variables without a formal bridge theorem.** Theorem 1 establishes invariance of P(X | Pa[X]) under shifts of the prior over *source* variables B = {B | Pa[B] = ∅}, because these have no parents and hence resampling their marginal leaves every structural conditional P(X_i | Pa[X_i]) undisturbed. However, Section 4.2.1 explicitly states "as finding the true sources is not possible without intervention, we will use a basis set as a surrogate," and the algorithm operates on a basis B̃ that may contain non-source variables. For a basis variable B̃_j with true parents, jointly resampling P(B̃_j) while "preserving P(X \ B̃ | B̃)" does not correspond to any valid causal operation—the parents of B̃_j's structural equation are implicitly disturbed. The justification provided is informal: same mutual independence as sources, same maximum size (Theorem 2), same "dependence set" (Φ(X)). These are size and correlation-structure properties, not causal-mechanism properties, and they do not formally extend Theorem 1's guarantee to basis variables. No theorem in the main paper closes this gap. Since the practical invariance test hinges entirely on this substitution, the core claim that the test "reliably determines if Z = Pa[X]" (Section 4.1, Contribution 1) lacks a complete theoretical backing.

- **Completeness gap in the parent-finding algorithm: maximal vs. non-maximal cliques.** Theorem 7 states that Pa[X] *corresponds to a clique* in G′(X). Section 4.3 then restricts the candidate search to *maximal* cliques for efficiency. For this restriction to be complete, Pa[X] must itself be a maximal clique in G′(X). This is not proved in the main paper. If Pa[X] = {A, B} is a proper subset of a larger clique {A, B, C}, the algorithm will never test {A, B} and may spuriously assign C as a parent of X. The paper provides no argument (formal or constructive) ruling out this configuration in the main text. As a completeness claim for causal recovery, this gap matters.

### Minor

- **O(d²) complexity is an empirically conditioned bound, not a formal guarantee.** The abstract and introduction present "quadratic complexity" unconditionally, but the proof in Section 4.3 relies on the degeneracy constant p being treated as a small constant: "Consider p as a small constant compared to d." The bound p ≤ 13 is stated only as an empirical observation across benchmark graphs (Section 4.3). For other graph families or at larger scales, p could grow with d. The paper should either present this as a conditional complexity result (O(d · p · 3^{p/3}) with p empirically bounded) or explicitly caveat that O(d²) is an empirically motivated claim, not a theorem.

- **PC omitted from continuous data experiments (Figures 2–3) without explanation.** PC is listed as applicable to L-G and nL-nG data in Table 1 and is included as a baseline in the categorical experiments (Figures 4, Table 2). However, it does not appear in Figures 2–3. If PC timed out on the continuous benchmarks, this should be stated explicitly—and GLIDE's speed advantage over PC on continuous data would be the most directly interpretable comparison against the canonical constraint-based method.

- **No finite-sample analysis of the invariance test's false-positive rate or power.** The paper notes (Footnote 2 and Section 4.1) that the implication in Eq. (1) is one-directional for finite m and relies on m being "sufficiently large and diversified." For finite datasets and large conditioning sets Z, the empirical variance of P_i(X | Z) could be near-zero for both Z = Pa[X] and for supersets of Pa[X], making the argmin in Eq. (3) unreliable. There is no analysis of the test's power or false-positive rate as a function of m or dataset size n.

### Trivial

- The abstract's "up to 25× reduction in processing time compared to state-of-the-art methods" corresponds to one specific comparison (GLIDE vs. MLP-NOTEARS in the nL-nG extreme setting). On real-world data (Table 2), GLIDE is 100–1000× *slower* than GIES. The "up to" qualifier is technically accurate but the unqualified framing of the abstract is misleading about the typical case.

---

## Nice-to-Haves

- A controlled ablation using graphs where the true basis set diverges meaningfully from the true source set would directly probe whether the basis substitution causes failures. Currently no such experiment exists.
- A formal or empirical argument in the main body that Pa[X] is (at least generically) a maximal clique in G′(X) would substantially strengthen the theoretical coverage of the parent-finding algorithm.
- A sensitivity analysis on m (number of augmented datasets) in the main paper (the current ablation is deferred to the appendix) would substantiate the test reliability claims for finite m.
- Discussion of how the invariance principle degrades under latent confounders (violation of causal sufficiency) would be valuable, as this is a common real-world concern.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"25x speedup overstates the contribution"** *(from Harsh Critic, Abstract)*: The "up to 25×" qualifier makes this technically accurate. Moved to Trivial.
- **"Definition 1 is a perfect map, not a causal model"**: While Definition 1 is stronger than typical causal model definitions (it requires bidirectional implication of CI and d-separation), this is an internal definitional choice that is self-consistent throughout the paper. The faithfulness assumption is implicitly required, as is standard in this subfield. This is a minor precision issue, not a substantive flaw.
- **"Categorical data spurious rate is higher than PC by 4%"** *(Harsh Critic, Section 5.2)*: The paper explicitly acknowledges this trade-off and demonstrates a 30× speedup in exchange. The framing is honest. Removing as a standalone weakness.
- **"Real-world runtime: GIES is faster"**: The paper includes full runtime tables (Table 2) and the authors correctly frame this as a trade-off. GIES's speed advantage comes at a large accuracy cost. Not a substantive weakness.
- **"K-means source prior selection is a heuristic without analysis"** *(Harsh Critic, Section 4.2.4)*: Sensitivity analysis on m is deferred to the appendix. Under the rule against penalizing appendix-deferred material, this is removed as a standalone major weakness (partially subsumed by the finite-sample Minor weakness above).
- **Strength: "Publicly accessible code"**: Generic; does not cite a specific section/result. Removed from Strengths.

---

## Novel Insights

The key synthesis insight not emphasized by either reviewer: the two major theoretical gaps (basis substitution and maximal clique restriction) are not independent—they compound. The basis substitution gap means the synthetic environments may not cleanly separate cause from correlation; the maximal clique gap means the search might not even test the correct hypothesis. Yet the algorithm still performs well empirically, which suggests either (a) the gaps are practically benign for the tested graph families, or (b) the invariance signal is robust to these approximations for reasons the theory has not articulated. This disjunction is the most important open question the paper leaves unresolved, and future theoretical work should aim to characterize when the basis approximation is valid and when Pa[X] is guaranteed to be a maximal clique in G′(X).

---

## Suggestions

1. **Prove or formally qualify the basis substitution:** Add a theorem showing the invariance test remains valid (or approximately valid under identifiable conditions) when basis variables replace source variables. Alternatively, clearly demarcate which results are theoretical guarantees and which are heuristic justifications.
2. **Address the maximal clique completeness gap:** Either prove that Pa[X] is always (or with high probability under faithfulness) a maximal clique in G′(X), or adjust the algorithm to also test non-maximal cliques when computational budget allows.
3. **Include PC in Figures 2–3 or explain its absence:** If PC times out on continuous data, note this explicitly (as is done for Pathfinder/Munin in Table 2), which would strengthen the scalability argument.
4. **Restate the complexity bound conditionally:** "O(d · p · 3^{p/3}) where empirically p ≤ 13 across all tested graphs, giving an effective O(d²) complexity" is more accurate than an unqualified "O(d²)."

---

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Relation to GLIDE |
|---|---|---|
| `/human_reviews/xByvdb3DCm.md` ("Selection Meets Intervention") | 8.0 | Oral acceptance; fully rigorous theory + provably sound algorithm + solid experiments. GLIDE is below this—theoretical gaps would need to be resolved to reach this level. |
| `/human_reviews/u63OVngeSp.md` ("Deriving Causal Order from Interventions") | 7.0 | Poster acceptance; novel faithfulness variant, theoretical guarantees, good empirics. Similar scope to GLIDE but tighter theory. |
| `/human_reviews/8muemqlnG3.md` ("Causal Discovery via Bayesian Optimization") | 6.5 | Poster acceptance; novel approach, decent empirics, some theoretical limitations. GLIDE has broader evaluation and a more principled core insight, but has more significant theoretical gaps. |
| `/human_reviews/iaP7yHRq1l.md` ("Robustness of Differentiable Causal Discovery") | 5.5 | Poster acceptance; primarily empirical benchmarking, limited novel theory. GLIDE has more novel theoretical framing. |
| `/human_reviews/or8wkKoBP4.md` ("Structure Learning for Unfaithful Distributions") | 4.0 | Rejected; theoretical framework but no experiments + gaps. GLIDE's extensive empirics put it well above this. |
| `/human_reviews/4P76wCt9N5.md` ("DAG-based Generative Regression") | 3.0 | Rejected; weak methodology, questionable experiments. GLIDE far exceeds this. |

GLIDE sits between the DrBO anchor (6.5, accepted poster with novel idea + some theoretical limitations) and the causal discovery benchmarking paper (5.5, accepted poster but mainly empirical). GLIDE has a more original core insight than DrBO and significantly stronger empirical evaluation, but its two major theoretical gaps are more substantive than DrBO's limitations. The basis-substitution gap in particular is the crux of the algorithm's correctness and is not formally resolved. On balance, I score GLIDE at **5.5**—borderline, leaning toward marginal accept if the authors can provide even a partial formal argument for the basis substitution and maximal-clique completeness claims.

**Axes summary:**
- *Originality*: High — the distributional invariance principle for observational causal discovery is novel and principled.
- *Importance of research question*: High — scalable, reliable causal discovery from observational data is an important open problem.
- *Claims well-supported*: Moderate — empirically well-supported, but the two core theoretical claims (invariance carries over to basis variables; maximal cliques suffice) are unproven or only informally justified.
- *Soundness of experiments*: Good — diverse benchmarks, confident intervals, large-scale real-world evaluation up to 1041 nodes.
- *Clarity of writing*: Good — the paper is clear and well-organized; the informal justification for basis substitution is readable but incomplete.
- *Value to community*: Moderate-to-high — the method is practical and the Munin result is impressive, but theoretical gaps limit confidence in when the method will and won't work.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>