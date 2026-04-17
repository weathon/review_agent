Now let me carefully verify a few specific claims against the paper text before synthesizing.

The paper content is provided in the user message. Let me check specific claims about: (1) the definition of distributional equivalence vs. generic reasoning, (2) the faithfulness assumption placement, (3) the "structural-assumption-free" claim context, and (4) the treatment of singular I−B.

From the paper:

- **Definition 1**: Exact equality of distribution sets P(G,X) = P(H,X).
- **§3.1**: The authors explicitly move to Zariski closures: "So for now, let us progress with the Zariski closure of A(G,X), an algebraic variety..."
- **Lemma 2**: "for generic choice of A ∈ A(G)"
- **Assumption 1** is deferred to Appendix A (mentioned briefly in §5).
- **§1**: "to our knowledge, this is the first equivalence characterization with latent variables in any parametric setting *without structural assumptions*, and hence the first structural-assumption-free discovery method."
- The paper's stated scope is linear non-Gaussian models throughout.

Now let me synthesize the final review.

## Summary

This paper establishes the first graphical characterization of distributional equivalence for linear non-Gaussian latent-variable causal models with arbitrary latent structure and cycles. Central to the approach is the introduction of "edge rank" constraints—a local counterpart to the global "path rank" constraints—connected via a duality theorem (Theorem 1). This enables an efficient equivalence criterion (Theorem 2), a transformational characterization (Theorem 3), and a constraint-based algorithm (glvLiNG) for recovering models up to equivalence without imposing structural assumptions on the graph.

## Strengths

- **Genuinely novel theoretical contribution**: This is the first distributional equivalence characterization for any parametric latent-variable causal model setting without structural assumptions. The analogy to CPDAGs and MAGs is apt: just as PC required CPDAGs and FCI required MAGs, the field has lacked the analogous characterization for latent-variable non-Gaussian models. This paper fills that gap, which is a significant conceptual advance (§1, §4).

- **Edge rank duality is elegant and potentially impactful**: Theorem 1 establishing the duality between path ranks and edge ranks is a genuine conceptual addition to the rank-based causal discovery toolbox. It transforms global path-rank constraints into local edge-rank constraints that can be independently checked per variable (Theorem 2), which is what ultimately makes the characterization tractable (§3.3, §4).

- **Clean theoretical pipeline**: The progression from mixing-matrix closure (Lemma 1) → path-rank equivalence (Lemma 3) → edge-rank equivalence (Lemma 5) → children-bases criterion (Theorem 2) → transformational characterization (Theorem 3) is logical and well-motivated. The reduction from checking exponentially many subsets (Lemma 3) to checking only singletons (Theorem 2) is non-trivial and practically important.

- **Comprehensive structural results**: Not only a recognition criterion (Theorem 2), but also a generative transformational characterization (Theorem 3) analogous to Meek's conjecture for Markov equivalence, and a maximal-graph representation (Theorem 4, appendix). This suite of results mirrors the classical CPDAG theory at a higher level of complexity.

- **Honest about existing methods' failures under misspecification**: Table 5 showing that LaHiCaSl and PO-LiNGAM misidentify over half of edges when their structural assumptions are violated is a compelling empirical argument for why assumption-free methods are needed (§5).

- **Interactive demo and code**: Providing https://equiv.cc for equivalence class traversal is valuable for understanding the otherwise abstract theory.

## Weaknesses

### Major:

- **Mismatch between the stated definition of distributional equivalence and the generic/algebraic reasoning actually used**: Definition 1 defines equivalence as exact equality of observed distribution sets P(G,X) = P(H,X). However, the core theoretical arguments operate on Zariski closures and generic parameter choices (§3.1: "let us progress with the Zariski closure of A(G,X)"; Lemma 2: "for generic choice of A"). The gap between exact distributional equivalence and generic algebraic equivalence is acknowledged only informally ("this does not affect our results") rather than rigorously closed. If two graphs yield identical distributions on a Zariski-open subset but differ on a measure-zero set, Definition 1's exact equality may not hold even though they are generically equivalent. This matters because the paper's main theorems (Theorems 2 and 3) are proved via the generic/algebraic route, while the headline claims use the word "distributional equivalence" as defined exactly. The paper should either (a) explicitly redefine equivalence as generic/almost-everywhere, or (b) provide a rigorous argument that the Zariski closure and measure-zero arguments do not affect the final set equality under the stated Definition 1. This is not just a terminology issue—it concerns what the theorems actually prove.

- **The "structural-assumption-free" claim overstates what is achieved**: The paper claims to be "the first structural-assumption-free discovery method" (§1, §5). It is indeed free of *graphical structural assumptions* (pure measurements, bow-free, etc.), which is significant. However, the method requires: (i) a linear non-Gaussian parametric assumption; (ii) OICA identifiability conditions (sufficient observed dimensions, non-Gaussianity); (iii) a faithfulness-like assumption on rank patterns (Assumption 1, deferred to Appendix A); and (iv) irreducibility (which, while presented as canonicalization, does constrain the model class by merging latents with fewer than two external children). These are not structural in the graph-topology sense, but they are nontrivial assumptions that limit the scope. The paper acknowledges (i) and partially (iii), but the headline claim could mislead readers into thinking no assumptions of any kind are needed.

- **OICA as a practical bottleneck, with unexplored sensitivity**: The glvLiNG algorithm relies on overcomplete ICA (OICA) as its first step. While the paper candidly acknowledges that OICA is "known inefficiency in practice" and frames glvLiNG as "more as a proof of concept" (§5), this significantly limits practical applicability. More importantly, the paper does not analyze how errors in OICA estimation—whether in the number of latent sources, their mixing matrix, or near-rank deficiencies—propagate through the rank-computation and graph-construction steps. This matters because rank-based decisions are discrete and potentially fragile: a small perturbation that changes an estimated rank can alter the entire equivalence class. The finite-sample experiments (Appendix D.4) evaluate performance but do not systematically study this sensitivity.

- **Equivalence classes can be very large, raising questions about practical informativeness**: Example 1 notes that with 4 observed and 2 latent variables in a structured graph, equivalence classes can contain 872 or 1,024 digraphs. While this is an honest characterization, the paper does not analyze what fraction of edges or orientations are actually identifiable within these large classes. For practitioners, knowing that the true model lies in a class of 1,024 graphs may not be much more useful than knowing it lies in the much larger space of all possible graphs. The paper would be strengthened by analyzing edge-level or path-level identifiability within equivalence classes.

### Minor:

- **Assumption 1 (faithfulness) is deferred to the appendix**: This assumption is critical for the algorithm's guarantees—it states that no coincidental low ranks occur beyond those structurally entailed—yet is not discussed in the main text. This is related to, but distinct from, standard CI-faithfulness, and its strength and testability deserve main-text discussion.

- **Proofs of central results are entirely in appendices**: For a theory-heavy paper, key results like Lemma 3 (rank constraints alone suffice for equivalence), Theorem 1 (duality), and Theorem 2 (children-bases criterion) would benefit from at least proof sketches in the main text. Currently, the reader must trust these assertions without any structural intuition about *why* they hold.

### Trivial:

- The real-data application (Hong Kong stock returns, Appendix D.5) is illustrative but speculative; no ground truth is available for validation. This is acceptable for a theory paper but should not be overstated as empirical validation.

## Nice-to-Haves

- **Edge-level identifiability analysis**: Report what fraction of edges/orientations are determined across all members of the equivalence class, not just the class size. This would show practitioners what partial causal queries remain answerable.

- **OICA-free rank estimation variant**: Even integrating existing partial rank tests (cumulant-based, independence-based) as an alternative to full OICA would improve practical viability.

- **Analysis of when equivalence classes are small**: Characterize graph structures that yield small vs. large equivalence classes, connecting to Adams et al. (2021) on unique identifiability conditions.

- **Sensitivity analysis for rank estimation errors**: Study how small perturbations to estimated ranks affect the recovered equivalence class.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"No comparison with rank-based methods (e.g., GIN-based approaches)"**: The paper compares with LaHiCaSl and PO-LiNGAM, which are the most directly comparable methods (latent-variable LiNG methods). The suggested GIN-based methods operate under different settings (covariance-based, often Gaussian). The comparison gap is not as critical as claimed since the paper's contribution is primarily theoretical rather than benchmarking.

- **"Irreducibility is a structural assumption in disguise"**: The paper explicitly discusses this point (§2.2) and explains that irreducibility is a canonicalization that removes trivially unidentifiable latents, not a structural assumption like pure-measurements or bow-free. Two arbitrary models are equivalent if and only if their irreducible forms are equivalent, so nothing is lost. This criticism misunderstands the nature of irreducibility.

- **"Real-world application provides limited validation"**: While true, this is standard for theory-oriented causal discovery papers and the paper does not make strong claims about the real-data results.

- **"Missing proof-level detail is fatal"**: For a conference paper with appendices, deferring proofs is standard. The concern is valid for readability but does not constitute a weakness of the results themselves.

- **"Singular I-B means the definition is ill-defined"**: The paper addresses this by noting that invertibility is required in the definition of A(G) (Eq. 4) and that the pathological locus is measure-zero. While the gap between Zariski closure and exact distribution sets is real (noted above), the singular-I-B issue specifically is standard in the literature and handled by assumption.

- **"Irreducibility removes scientifically meaningful latents"**: The paper acknowledges this is about identifiability, not substantive importance. Latents with fewer than two external children are provably non-identifiable regardless of method—this is a fact about the model class, not an arbitrary restriction of this method.

## Novel Insights

The edge-rank/path-rank duality (Theorem 1) reveals a fundamental but previously unnoticed connection in the rank-based causal discovery toolbox: every path-rank constraint—a global, combinatorial quantity defined via max-flow—has an exact local counterpart in edge ranks defined via bipartite matching. This is not merely a translation but a complementary perspective that makes previously intractable equivalence characterizations (requiring exponentially many rank checks) decompose into independent per-variable checks (Theorem 2). The insight that this duality arises from classical matroid theory (König, Perfect, Ingleton & Piff) but was entirely unknown in the causal discovery community suggests that further matroid-theoretic tools may unlock additional compositional structure in latent-variable models.

## Suggestions

- Explicitly redefine or clarify the notion of distributional equivalence as *generic* distributional equivalence in the main text, or provide a rigorous argument bridging Zariski closures to exact distribution set equality under Definition 1. This is the most important precision issue.

- Bring Assumption 1 into the main text and discuss its relationship to standard faithfulness, when it is more/less likely to hold, and what happens when it is violated.

- Report edge-level identifiability statistics (fraction of edges present/absent/oriented across all graphs in the equivalence class) alongside class sizes in the evaluation section.

## Score and Decision

**Calibration anchors:**

- *Linear SCM Identification with Confounders and Gaussian Noise* (bjxuqI4KwU): Strong theory paper on identifiability of linear SCMs, scores 6-8, accepted as Spotlight. This paper has a narrower scope (identifiability under known structure) but is technically clean. The current paper has broader scope but with the exact-vs-generic equivalence gap.

- *RLCD* (FhQSGhBlqv): Latent-variable causal discovery using rank information, strong theoretical results with experiments, scores 6-8, accepted as poster. This paper is comparable in setting but requires triangle-free + two-pure-children assumptions, making the current paper's structural-assumption-free approach a clear theoretical advance.

- *Causal Structure Recovery under Milder Assumptions* (MukGKGtgnr): Milder structural assumptions for latent variable recovery, scores 5-8, accepted as poster. The current paper goes further by removing structural assumptions entirely.

- *Efficient and Trustworthy Causal Discovery* (BZYIEw4mcY): Latent variable causal discovery with pure children assumption, scores 6, accepted as poster. The current paper's theoretical contribution is stronger, though its practical algorithm is less immediately deployable.

The current paper makes a genuinely significant theoretical contribution—the first equivalence characterization without structural assumptions for latent-variable causal models, plus a new tool (edge ranks). This is comparable in importance to the theoretical foundations that enabled PC (CPDAGs) and FCI (MAGs). The main weaknesses—the exact-vs-generic equivalence gap and OICA's practical limitations—are real but do not undermine the core theoretical contribution. They warrant discussion but not rejection. The paper is above the quality threshold for acceptance, though the precision issues should be addressed in a camera-ready version.

MY FINAL SCORE: <pineapple>7</pineapple>
MY FINAL DECISION: <orange>Accept</orange>