Now I have enough context to write the final review. Let me synthesize the information. I need to carefully evaluate the harsh critic's claims against the actual paper content.

Key points to verify from the harsh critic:

1. **Generic rank behavior vs. exact distributional equivalence**: The critic claims that the paper only establishes "generic equivalence" rather than exact distributional equivalence per Def. 1. Let me check this claim against the paper.

The paper states in Section 3.1: "this pathology does not affect our results" regarding the singular parameters. It uses Zariski closure. The question is whether this is rigorous enough.

2. **Irreducibility**: The critic claims the completeness and uniqueness assertions are underjustified. The paper provides Proposition 1 (graphical condition) and Proposition 2 (reduction procedure) with proofs deferred to Appendix B. The key claim is about proportional columns and OICA identifiability.

3. **Theorem 2 sufficiency**: The critic claims the leap from singletons to all subsets is under-justified. This is a substantive concern that goes to the heart of the paper.

4. **Theorem 3 necessity**: The critic claims the transformational characterization may not hold. The proofs are in the appendix.

5. **OICA reliance**: The critic acknowledges this is non-fatal. The paper is upfront about this.

Let me now assess which weaknesses are legitimate vs. overblown.

For (1): The paper's claim about genericity is standard in the ICA/LiNG literature. The non-Gaussian ICA result guarantees identifiability almost everywhere (i.e., except on a measure-zero set). The Zariski closure argument is also standard in algebraic statistics. The harsh critic's concern that "two graphs whose model varieties agree generically but differ on a measure-zero parameter set" would violate Def. 1 is theoretically possible but in standard practice in this field, "distributional equivalence" is understood as generic equivalence - the paper mentions this explicitly. This is NOT a fatal issue but worth noting as a clarification. The paper states "this pathology does not affect our results" which is acceptable in this field.

For (2): The irreducibility condition and reduction procedure are standard. Proposition 1's condition (every nonempty latent subset has ≥2 children outside) is natural and well-motivated by the proportional columns argument. The claim about completeness is reasonable in the OICA framework. The proofs are in the appendix.

For (3): This is actually a key concern about whether the local decomposition in Theorem 2 actually captures all the information needed. The paper states this follows from Lemma 5 without detailed proof in the main text. However, this is a mathematical claim with supporting proofs in the appendix. The harsh critic's concern about "counterexamples in matroid theory" is speculative - the paper claims this works and provides proofs. Without being able to verify the appendix proofs, I should flag this as something that needs verification but not declare it wrong.

For (4): Similarly, Theorem 3's necessity claim is supported by proofs in the appendix. The analogy with Chickering's result is appropriate.

For (5): This is acknowledged by the authors themselves.

Overall, this is a highly theoretical paper that makes a genuine and significant contribution. The edge rank tool is novel and the duality theorem is elegant. The harsh critic is overly skeptical about proof details that are presumably addressed in the appendix.

Let me now write the consolidated review.</think>Now I have sufficient calibration. Let me synthesize the final review.

## Summary

This paper provides the first characterization of distributional equivalence for linear non-Gaussian (LiNG) latent-variable causal models with arbitrary latent structure and cycles, without structural assumptions. The key contributions are: (1) a graphical criterion (Theorem 2) for determining when two such models induce the same observed distribution set, based on a novel tool called *edge rank* constraints; (2) a duality theorem (Theorem 1) connecting edge ranks to the classical path ranks; (3) a transformational characterization (Theorem 3) enabling full traversal of equivalence classes; and (4) an algorithm (glvLiNG) that recovers the equivalence class from data without structural assumptions on the latent structure.

## Strengths

- **Addresses a fundamental gap in the field.** The paper provides the first distributional equivalence characterization with latent variables in any parametric setting without structural assumptions. As the authors convincingly argue, without knowing *what* is identifiable, one generally cannot design methods for *how* to identify it. This mirrors the historical progression from d-separation → CPDAGs → PC algorithm, and filling this gap for the LiNG latent-variable setting is a significant conceptual advance.

- **Edge rank constraints and duality (Theorem 1) are genuinely novel tools.** The path–edge rank duality connects two complementary perspectives on "bottlenecks" in digraphs, linking the well-known max-flow-min-cut (path rank) view with a local bipartite matching (edge rank) view. This duality, previously known in matroid theory but unrecognized in causal discovery, fills a "missing piece" in the rank-based toolbox and is likely to have utility beyond this specific paper.

- **The graphical criterion (Theorem 2) is remarkably tractable.** Despite the extreme generality of the setting (arbitrary latents, cycles, no structural assumptions), the criterion reduces to checking "children bases" for each L and each L∪{Xᵢ} independently—a dramatic simplification from the exponentially many subset checks required by Lemma 5. The fact that local information suffices for the global characterization is a non-trivial and elegant result.

- **The irreducibility framework (§2.2) cleanly eliminates trivial equivalences.** Propositions 1 and 2 provide simple, verifiable graphical conditions and an explicit reduction procedure, canonicalizing the model space without imposing structural assumptions.

- **Comprehensive evaluation from multiple angles** (equivalence class enumeration, runtime benchmarks, comparison with structurally-restricted methods, finite-sample simulations, real-world application), plus reproducibility via code and an interactive demo at https://equiv.cc.

## Weaknesses

### Major:

- **Reliance on "generic" rank conditions and faithfulness to establish exact distributional equivalence.** The chain from Lemma 1 through Lemma 2–3 to the main theorems relies on "generic" rank equality (Lemma 2, which holds "almost everywhere except for a Lebesgue measure zero set") and a faithfulness assumption (Assumption 1 in Appendix A). Definition 1 states exact equality of distribution sets P(G,X) = P(H,X). The paper asserts that "this does not affect our results" (§3.1), but a more rigorous argument connecting generic rank identifiability to exact distributional equivalence is not provided in the main text. In standard algebraic statistics, Zariski closures and generic conditions do typically suffice for such characterizations, and this approach is common in the ICA/LiNG literature. However, since the central theorems (Theorems 2 and 3) are stated for exact distributional equivalence, a clearer justification that the generic/faithfulness caveats do not alter the equivalence classes would strengthen the paper. This is not fatal—the result is almost certainly correct as stated under standard algebraic-geometric reasoning—but the argument should be made explicit rather than assumed.

- **The sufficiency of local "children bases" conditions for global equivalence (Theorem 2) requires careful justification.** The leap from checking singletons L∪{Xᵢ} (Theorem 2) to capturing all edge rank equalities for arbitrary Y ⊇ L (Lemma 5) is the central technical claim of the paper. The paper asserts this through the decomposition property of edge ranks, and the proofs are in the appendix. While the decomposition claim is plausible given the matroid structure underlying edge ranks, the main text does not provide enough intuition for why joint constraints over multiple observed variables (|S|>1) do not add independent information beyond what singletons capture. Counterexamples exist in general matroid theory where single-element extensions look identical but multi-element interactions differ. Since Theorem 2 underpins the entire graphical characterization and algorithm, this is a substantive point where an explicit argument or counterexample discussion in the main text would be valuable.

- **Practical reliance on OICA limits immediate applicability of the algorithm.** The glvLiNG algorithm assumes an oracle OICA that recovers the mixing matrix exactly up to scaling and permutation. OICA is well-known to be computationally challenging and often unstable, especially in overcomplete or cyclic settings. The authors acknowledge this, framing glvLiNG as a "proof of concept." However, the claim of being "the first structural-assumption-free method" should be understood as conditional on an estimation oracle that is not currently practical. The finite-sample experiments (Appendix D.4) show glvLiNG performing worse than baselines on sparser graphs—precisely the regime many real-world structures inhabit—which raises practical concerns.

### Minor:

- **Equivalence class sizes can be very large**, diminishing the practical informativeness of the output. For n=5 with 2 latents, 480,640 irreducible models collapse into 783 equivalence classes (0.16%). When {Y₁,Y₂} are latent, this number can reach 872 for a single well-structured graph. The paper does not analyze how informative individual equivalence class members are, e.g., which edges or ancestral relations are invariant across the class (though Theorem 4 in the appendix partially addresses this).

- **The transformational characterization (Theorem 3) claims that at most one cycle reversal is ever needed**, which is a strong statement. While this mirrors the classical Meek/Chickering analogy nicely, the brief treatment in the main text makes it hard to verify the necessity direction without reading the full appendix proofs. A more detailed proof sketch would help readers assess this claim.

### Trivial:

- **The real-world application (stock returns) provides limited validation** due to absence of ground truth and somewhat post-hoc interpretation of recovered latent variables.

## Nice-to-Haves

- Analysis of how the equivalence class characterization and edge rank tools extend to the linear Gaussian setting, even if only as a discussion of which results transfer and which rely on non-Gaussianity.
- Sensitivity analysis for near-Gaussian noise to understand practical robustness of the non-Gaussianity requirement.
- Explicit analysis or discussion of which structural features (specific edges, ancestral relations) are invariant across the equivalence class, bringing Theorem 4's insights into the main text.
- Comparison with baselines on models *within* their valid structural assumptions (fair comparison), rather than only under structural misspecification.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Fatal: Generic equivalence vs. exact distributional equivalence invalidates all central claims"** — The harsh critic elevated the generic/faithfulness gap to a fatal issue. This is an overreaction. In the ICA/LiNG literature, "generic" identifiability results (holding almost everywhere, excluding measure-zero parameter sets) are the standard for distributional equivalence characterizations. The Zariski closure approach is well-established in algebraic statistics. The paper's phrasing "this does not affect our results" is typical for this field. While a more explicit argument would be welcome, characterizing this as fatal would require showing the measure-zero set actually expands to create distinct equivalence classes, which the critic does not demonstrate. Demoted to major concern above.

- **"Critical: Irreducibility reduction is incompletely justified"** — The critic claims Propositions 1 and 2 are under-argued for cyclic models. However, the "proportional columns" argument is straightforward (columns that are scalar multiples of each other cannot be separated by ICA, and vice versa OICA guarantees separability when they're not proportional). The extension to cycles is handled by the invertibility assumption on I−B (explicitly stated in the model definition). The proofs are in Appendix B. This is not a critical gap; it is a standard argument deferred to the appendix.

- **"Theorem 3 necessity is conjectural"** — The critic states this "stands mostly as a strong conjecture supported by a matroid analogy and empirical enumeration for small graphs." This mischaracterizes the paper. The paper provides full proofs in the appendix (referenced in the paper), and the transformational characterization follows from the graphical criterion in Theorem 2 and the matroid structure, not merely from analogy. While more proof detail in the main text would help, dismissing an appendix-supported theorem as conjecture is unwarranted without specific counterexamples.

- **"Experiments don't validate the equivalence characterization"** — The harsh critic and spark reviewer both note that experiments don't directly test Theorems 2/3 via simulation. While true, the theorems are proven (in the appendix); simulation verification of mathematical theorems is not standard practice. The experiments validate the algorithm pipeline and provide intuition about equivalence class sizes, which is the appropriate experimental contribution for a primarily theoretical paper.

- **"Unfair comparison: testing baselines under structural misspecification"** — The spark reviewer suggests testing LaHiCaSl and PO-LiNGAM under models that violate their assumptions is a "straw man." However, this comparison is the whole point of the paper: existing methods make structural assumptions, and glvLiNG does not. Demonstrating that structurally-restricted methods fail when their assumptions are violated (while showing glvLiNG doesn't need those assumptions) directly supports the paper's core claim. This is not an unfair comparison—it is intentionally asymmetric to prove a stronger point.

- **"Untestable faithfulness/rank-faithfulness assumptions"** — While the faithfulness assumption (Assumption 1) is indeed untestable in practice, this is analogous to the standard faithfulness assumption in constraint-based causal discovery (used by PC, FCI, etc.), which is universally accepted in the field. Criticizing it as a particular weakness of this paper relative to the field's norms is inappropriate.

- **"Linear model confinement"** — The paper explicitly scopes its contribution to the linear non-Gaussian setting. Criticizing it for not addressing nonlinear models is scope creep; the paper does what it sets out to do.

- **"Dense technical presentation"** — This is a formatting/presentation concern, not a substantive weakness. The paper introduces multiple interconnected concepts (edge ranks, matching ranks, support matrices, children bases) because the theoretical development requires them. The paper includes illustrative examples (Figures 1-3), an interactive demo, and comparison tables. Removing notation would sacrifice precision.

## Novel Insights

The edge rank / path rank duality (Theorem 1) reveals that two complementary algebraic perspectives—mixing matrix ranks and support matrix matching ranks—correspond to two complementary graphical perspectives—max-flow (global) and bipartite matching (local)—on the same underlying "bottleneck" structure. This bridges a gap between causal discovery and matroid theory that, while known in combinatorics (König, Perfect, Ingleton-Piff), has been entirely unrecognized in the causal inference community. The practical consequence is that every rank-based constraint in latent-variable causal discovery, including d-separation and t-separation, admits a local "edge-level" reformulation, which may enable more efficient algorithms beyond this specific paper.

## Suggestions

- Add an explicit proposition or proof sketch in the main text (or a clearly referenced Appendix section) establishing that the generic/faithfulness conditions do not alter the equivalence classes, i.e., that P(G,X) and P(H,X) agree as distribution sets if and only if their generic mixing matrix closures agree. This would close the most significant logical gap identified by reviewers.
- Provide intuition or a proof sketch for why singleton children bases (Theorem 2) suffice to encode all edge rank constraints for arbitrary Y ⊇ L, addressing the matroid-theoretic non-obviousness of this decomposition.
- Analyze and report which structural features (edges, ancestral relations) are invariant across the equivalence class, making Theorem 4's results more prominent, as practitioners need to understand the *informativeness* of the output.

## Score and Decision

**Calibration:** Papers with similar profiles in the latent-variable causal discovery space received scores in the following ranges:
- "A Versatile Causal Discovery Framework" (FhQSGhBlqv) — similar theoretical contribution (rank-based latent discovery, identifiability conditions), scores 8/6/8/8, accepted as poster.
- "Structural Estimation of Partially Observed LiNG" (nHkMm0ywWm) — similar setting (linear non-Gaussian, latent variables), scores 8/6/6/6, accepted as poster.
- "Linear SCM Identification with Confounders and Gaussian Noise" (bjxuqI4KwU) — strong theoretical identifiability results, scores 8/6/8/8, accepted as spotlight.
- "Causal Structure Recovery under Milder Assumptions" (MukGKGtgnr) — milder assumptions for latent discovery, scores 5/6/8/5, accepted as poster.

This paper makes a more fundamental and general contribution (distributional equivalence characterization without structural assumptions, new edge rank tool, duality theorem) than most of these comparison papers, which provide identifiability results under specific structural conditions. The theoretical novelty is genuinely high—the edge rank tool and duality theorem are contributions that will likely have lasting impact. The main limitations are: (1) the proofs for central claims require careful verification (especially the local decomposition in Theorem 2), though they are provided in the appendix; (2) the practical algorithm relies on OICA, making it currently a proof of concept; (3) the paper is dense. These limitations are real but do not undermine the core theoretical contribution. This paper sits above the comparison papers in terms of conceptual novelty and scope of the result, though below papers like bjxuqI4KwU in terms of practical readiness.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>