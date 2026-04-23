Now I have all the information needed to write the final consolidated review. Let me carefully synthesize everything, verifying each weakness against the actual paper.

## Summary

This paper provides the first graphical characterization of distributional equivalence for linear non-Gaussian latent-variable cyclic causal models—determining when two graphs with arbitrary latent structure and cycles induce the same set of observed distributions. The central technical contribution is a local decomposition (Theorem 2) that reduces equivalence checking from an exponential number of subset conditions to checking singletons, enabled by a newly introduced tool called edge ranks and their duality with path ranks (Theorem 1). A transformational characterization (Theorem 3) enables traversal of entire equivalence classes, and the glvLiNG algorithm demonstrates proof-of-concept recovery from data.

## Strengths

- **The local decomposition in Theorem 2 is genuinely surprising and powerful.** Going from checking all subsets of X (exponential) to checking each singleton Xi independently (linear) is a non-obvious simplification. This is the paper's strongest technical contribution and directly enables the algorithm's tractability in Phase 2 (edge recovery from observed variables).

- **The edge rank framework and its duality with path ranks (Theorem 1) is a clean and novel tool.** The relation min(|Z|,|Y|) − ρ(Z,Y) = |V| − max(|Z|,|Y|) − r(V\Y, V\Z) connects the global path-based perspective (well-known in causal discovery) with the local edge-based perspective (new), and the paper appropriately credits the matroid theory origins (Kőnig, Perfect, Ingleton & Piff). This duality is not merely instrumental—the local decomposability of edge ranks is precisely what enables Theorem 2.

- **The transformational characterization (Theorem 3) provides an operational mechanism for equivalence class traversal**, analogous to covered edge reversals for Markov equivalence (Meek, 1997). The result that at most one cycle reversal is needed is clean and the admissible edge addition/deletion criterion (Lemma 7) is concrete.

- **Comprehensive evaluation across five aspects**, including exhaustive equivalence class quantification (Table 3: 783 classes from 480,640 irreducible models for n=5 with 2 latent), runtime benchmarks (Table 4: glvLiNG solves n=10 in <5s vs. hours for LP baseline), and oracle evaluation of existing methods under structural misspecification (Table 5: LaHiCaSl and PO-LiNGAM misidentify over half of edges).

- **Honest positioning of glvLiNG as a proof of concept** with explicit acknowledgment of OICA limitations and clear future directions (partial rank information methods, ancestral relation pruning). The interactive demo at https://equiv.cc supports pedagogical accessibility.

## Weaknesses

### Fatal
None.

### Major

- **The Zariski closure step (Section 3.1) is load-bearing but insufficiently justified in the main text.** The paper progresses from the actual mixing matrix set A(G,X) to its Zariski closure, stating only: "this does not affect our results" (line 109). Since the core proof path goes: distributional equivalence → A(G,X) equality (Lemma 1) → Zariski closure equality (Lemma 3) → graphical criterion (Theorem 2), the gap between Zariski closure equality and A(G,X) equality must be argued. In algebraic statistics, this typically relies on the fact that the "extra" points in the Zariski closure lie on a measure-zero subset (where I−B is singular), and distributions arising from this locus do not enlarge the distribution set P(G,X). However, this argument is not even sketched in the main text, despite supporting every subsequent result. If the appendix proof is sound (as is likely given standard algebraic statistics techniques), this is a presentation concern; if not, it invalidates the main theorems. At minimum, a one-paragraph sketch belongs in Section 3.1.

- **The finite-sample evaluation (aspect 4) is extremely thin in the main text**, reduced to a single sentence: "glvLiNG performs particularly better than baselines on denser graphs and stays more robust to latent dimensionality, likely due to avoiding model misspecification, while baselines perform better on sparser graphs" (Section 5). This acknowledges that glvLiNG *loses* to existing methods when their structural assumptions hold (the sparse regime most practitioners encounter), without quantifying the gap, reporting precision/recall/F1, or providing confidence intervals. While the paper positions glvLiNG as a proof of concept, the theoretical contribution's practical relevance depends on whether the generality comes at an acceptable cost. The full results in Appendix D.4 presumably contain this information, but the main text should summarize key quantitative findings, not just qualitative observations.

### Minor

- **The proof that Theorem 2's local decomposition follows from Lemma 5 is entirely deferred to the appendix.** Given that this is the paper's central result and the decomposition from exponential to linear is non-obvious, at least a proof sketch or high-level intuition should appear in the main text to allow readers to evaluate the result without consulting the appendix.

- **The faithfulness assumption (Assumption 1, in appendix) deserves discussion in the main text.** It rules out coincidental rank drops beyond those structurally entailed. With the number of rank conditions growing with the graph size, the probability of near-violations under finite samples increases, which could affect the reliability of the method. This is particularly relevant for the practical deployment of glvLiNG.

- **The interpretation of the irreducible form H recovered by glvLiNG warrants clarification.** The reduction in Proposition 2 may produce an irreducible form with edges not present in the true model (e.g., when a redundant latent L with parents P1, P2 and sole child C is reduced, edges P1→C and P2→C are added if not already present, while L is removed). Practitioners need guidance on interpreting these "shortcut" edges: do they represent genuine direct causal pathways, or are they artifacts of the canonicalization?

### Trivial
None.

## Nice-to-Haves

- **Systematic analysis of equivalence class sizes as a function of graph structure.** The paper provides scattered examples (17, 783, 872, 1,024) but no systematic understanding of when classes are small (informative) versus large (uninformative). This directly determines the practical value of the structural-assumption-free approach.

- **An OICA-free variant using partial rank information**, even as a heuristic sketch. The paper suggests this direction; developing it would dramatically increase practical relevance.

- **A worked end-to-end example** of the full glvLiNG pipeline on a small non-trivial model, showing intermediate results at each step (OICA output → digraph construction → equivalence class traversal).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that "applying the reduction in Proposition 2 does not increase the number of edges or cycles" is incorrect (Harsh Critic Issue 3).** The critic argues Step 4 "clearly adds edges." However, this is a misreading: Step 4 adds edges from parents of l to c *and then removes the latent vertex l along with all its edges*. The net edge count always decreases (removing at least |paH(l)| + 1 edges while adding at most |paH(l)| − 1). The paper's claim is correct for edges. The cycles claim requires more careful analysis but is presented as a "side note" and is not load-bearing for any main result.

- **Concern about Lemma 2 needing more justification for cyclic digraphs.** The paper cites the generalization by Talaska (2012) and states the result holds "for generic choice of A ∈ A(G)." The cyclic case is handled through the rational function structure of (I−B)⁻¹. This is a standard result and the criticism demands more detail than the main text of a conference paper typically provides.

- **Concern about "missing related works."** Per the rules, these are removed since we cannot confirm their existence.

- **Concern about equivalence classes being "not much more informative than saying 'we don't know'" when large.** This misunderstands the paper's contribution: the equivalence class IS the best one can identify from data without structural assumptions. The paper provides tools to understand class structure (maximal digraph, invariant edges via Theorem 4). This is analogous to how large Markov equivalence classes in FCI do not invalidate FCI's contribution.

- **Request for confidence intervals and large-scale benchmarks for the theoretical paper's proof-of-concept algorithm.** Demanding extensive statistical validation for a paper whose primary contribution is a theoretical characterization, with the algorithm explicitly positioned as a proof of concept, is scope creep.

- **Nitpicks about notation and presentation** (e.g., MEC abbreviation, expression formatting) are removed per formatting rules.

## Novel Insights

The edge rank framework's duality with path ranks (Theorem 1) reveals that the well-known bottleneck structure in causal discovery—previously understood only through the lens of vertex-disjoint paths and d-/t-separation—has a complementary, purely local edge-based characterization. This is not just a rephrasing: the local decomposability of edge ranks is what enables the exponential-to-linear reduction in Theorem 2, something path ranks alone could not provide despite being known for decades. The implication is that other problems in latent-variable causal discovery that have been approached through path ranks may benefit from reformulation in terms of edge ranks, extending beyond the specific linear non-Gaussian setting of this paper.

## Suggestions

- Add a one-paragraph sketch in Section 3.1 explaining why Zariski closure equality suffices for distributional equivalence (the measure-zero locus where I−B is singular does not enlarge P(G,X)), even if the full formal argument remains in the appendix.

- Include a summary table or figure from Appendix D.4 in the main text quantifying the finite-sample comparison (at least precision/recall for edge recovery at a representative setting), so readers can assess the practical cost of generality.

- Add a brief remark clarifying the interpretation of "shortcut" edges in the irreducible form recovered by Proposition 2—are they direct causal effects in the reduced model, or should they be interpreted as indirect pathways through removed latents?

## Evaluation

**Originality:** High. This is the first distributional equivalence characterization for latent-variable models in any parametric setting without structural assumptions. The edge rank framework and the local decomposition are genuinely novel.

**Importance of research question:** High. The lack of equivalence characterization has been identified as a core obstacle to structural-assumption-free latent-variable causal discovery, and this paper directly addresses it.

**Claims well supported:** Mostly yes, with the caveat that the Zariski closure argument is load-bearing and deferred to the appendix. The theoretical framework is rigorous and the algorithm is honestly evaluated.

**Soundness of experiments:** Moderate. The evaluation covers five aspects but the critical finite-sample comparison (aspect 4) is thin in the main text. Aspects 1–3 (equivalence class quantification, runtime, oracle evaluation of baselines) are well-executed.

**Clarity of writing:** Good. The paper is well-organized with clear motivations and effective historical analogies (PC/CPDAG, FCI/MAG). Some key proof ideas could be more accessible (Zariski closure, local decomposition proof).

**Value to research community:** High. The edge rank tool has potential beyond this specific setting, and the equivalence characterization provides the foundation for future structural-assumption-free methods.

## Calibration

**Anchors compared:**
- mA78uXqcnl (avg 7.0, Accept Oral): Similar—first identification conditions for latent confounders in Hawkes processes. The paper under review has arguably a more fundamental theoretical contribution (equivalence characterization vs. identification conditions). Both have proof-of-concept algorithms with acknowledged limitations.
- bZqCBgm2N0 (avg 6.5, Accept Poster): Novel latent causal model with identifiability result. The paper under review has a deeper theoretical framework and more comprehensive evaluation.
- BNHplerBYE (avg 5.33, Accept Poster): Score-based search for partially observed models. The paper under review has a more fundamental and novel theoretical contribution (first equivalence characterization vs. extending GES).
- ssYeoL4ksl (avg 5.5, Reject): Higher-order cumulants for cyclic latent models. The paper under review has much stronger theoretical depth (full equivalence characterization vs. pairwise relationships) and a more complete framework.
- Xa9E195Eym (avg 2.5, Reject): Overclaimed thin experiments. The paper under review is clearly far above this—genuine theoretical contributions, not overclaimed SOTA.

The paper under review sits above the medium-scoring causal discovery anchors (5.3–6.0) due to the fundamental nature and novelty of its theoretical contribution, and is comparable to the high-scoring anchors (7.0). Its main drag factors are the thin empirical summary in the main text and the OICA dependency, both of which the paper honestly acknowledges. The theoretical contribution—the first equivalence characterization with local decomposition—is sufficiently strong to place it in the 6.5–7.0 range.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>