---
job_id: 012babdf-f9e5-4251-9c03-4874f77a83f7
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: b8TlYh6PN6.pdf
paper: Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models: Characterization and Learning
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is on causal discovery, identifiability, and graphical characterization in linear non-Gaussian models, which fits squarely within causal reasoning, learning theory, and general machine learning.

## Minimum Quality
Pass ✅.  
The paper is complete (abstract, introduction, related-work-style discussion, methodology, theory, algorithm, experiments, and conclusion are all present), technically deep, and the arguments are rigorous. I see no fatal methodological or theoretical flaw, nor any egregious issue in the experiments that would warrant desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, instructions to reviewers/LLMs, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary
The paper studies distributional equivalence of linear non-Gaussian causal models with arbitrary latent variables and cycles. It introduces a graphical characterization of when two such latent-variable models induce the same set of observational distributions, via path/edge rank conditions and a new notion of edge ranks, and then derives a transformational characterization of the resulting equivalence class. Building on this, the authors propose glvLiNG, a structural-assumption-free algorithm that, given mixing matrices estimated by overcomplete ICA, reconstructs the equivalence class of causal graphs and demonstrate it on synthetic and real data.

## Strengths

1. **Conceptual contribution: full equivalence characterization in a hard setting.**  
   The paper tackles a long-standing gap: characterizing distributional equivalence in linear non-Gaussian models with *both* latent variables and cycles, without structural constraints on the latent structure. The main criterion (Theorem 2) and the transformational characterization (Theorem 3) together play the role that CPDAGs and covered edge reversals play for fully observed DAGs. This is conceptually important for clarifying what can be identified in LiNG models with latent structure.

2. **Edge rank as a genuinely useful new tool.**  
   Introducing edge ranks (Definition 4) and linking them to matching rank in the binary support matrix (Lemma 4) is elegant and practically helpful. The duality between path ranks and edge ranks in Theorem 1 connects two strands of matroid theory that have been unevenly used in causal discovery, and the paper shows that edge ranks are often the more local and manipulable view. This is not just cosmetic; the entire simplification from the global path-rank criterion (Lemma 3) to the local bases conditions in Theorem 2 hinges on this.

3. **Matroid-theoretic backbone is sound and well integrated.**  
   The proofs and constructions in Appendix A/B lean heavily on transversal matroids and strict gammoids, with correct references to Ingleton & Piff, Mason, Brylawski, etc. The matrix/matroid manipulations (e.g., Lemma 8 on how matroids behave under column augmentation, Lemma 10 on constructing a particular augmentation, and Lemma 12 on connectivity of the “column augmentation” Hasse diagram) are nontrivial but appear consistent and are carefully justified. Equation-heavy arguments (e.g., Equations (A.21), (B.40), (B.41)) are logically tracked and not handwavy.

4. **Nontrivial reduction from global rank conditions to local bases checks.**  
   Lemma 5 states equivalence via edge ranks over all \(Z,Y\) sets with \(L\subseteq Y\); naively this is still exponential. The key technical simplification is Lemma 9 and Theorem 2: instead of checking all subsets of \(X\), it suffices to match child-bases of \(L\) and of each \(L\cup\{X_i\}\) separately. This is a strong structural insight about how the rank constraints “factor” across observed variables and makes the criterion actually usable.

5. **Transformational characterization is clearly specified and matches intuition.**  
   Theorem 3 shows that two irreducible models are equivalent iff they can be connected (up to latent relabeling) by (i) admissible cycle reversals (Lemma 6) and (ii) admissible edge additions/deletions (Lemma 7) defined via edge ranks. Figure 3 and Figure 5 visualize equivalence classes and the transitions (solid vs. dashed edges for edge-add/delete vs. cycle reversals), which makes the rather abstract statement concrete. The condition in Lemma 7, expressed via edge ranks like
   \[
   r_{\mathcal{G}}(\text{nonchildren}\backslash\{V_j\},L\backslash\{V_i\}) < r_{\mathcal{G}}(\text{nonchildren},L\backslash\{V_i\}),
   \]
   is nontrivial but well motivated, and Example 2 walks through how it works in a small graph from Figure 3.

6. **Irreducibility notion is natural and well-justified.**  
   The irreducibility condition (Proposition 1) gives a crisp graph property in terms of child sets of latent subsets; the reduction procedure in Proposition 2 is constructive and illustrated in Figure 1. This nicely ties back to OICA identifiability: removing latent “components” which lead to proportional columns in the mixing matrix. The irreducible reduction is clearly canonization, not an extra structural restriction.

7. **Algorithmic instantiation is nontrivial and reasonably efficient.**  
   Given only implicit access to matching ranks via an OICA-estimated mixing matrix, glvLiNG reconstructs a compatible support matrix \(Q\) (the bipartite graph) in two phases. Phase 1 realizes the transversal matroid on the latent columns via an \(\alpha\)-system construction (Appendix A.3). Phase 2 augments the columns for observed variables, with the key insight from Lemma 9 that singleton augmentations suffice. Lemma 10 gives a constructive way to set column entries via circuit information. Table 4 shows reasonable runtimes compared to a MILP baseline, scaling up to \(n=13\) within tens of seconds, with Phase 2 being a minor cost relative to Phase 1. This demonstrates that the theory is not purely existential.

8. **Figures and tables are informative and aligned with claims.**  
   - **Figure 2** nicely illustrates the difference between path ranks and edge ranks, and the duality: the min-cut \(\rho_{\mathcal{G}}(Z,Y)=2\) on the left vs. the maximum bipartite matching \(r_{\mathcal{G}}(V\setminus Y, V\setminus Z)=4\) on the right, with corresponding red entries in the support matrix confirming \(\text{mrank}=4\). This directly complements Theorem 1 and clarifies the bottleneck notion.  
   - **Figure 3** and **Figure 5** explicitly show small equivalence classes and allowed moves (edge +/- vs cycle reversals), making Theorem 3 very tangible.  
   - **Figure 6**’s “CP(\(\mathcal{G}\))” representation (solid vs dashed edges over a maximal graph) illustrates how the proposed presentation summarizes the class, analogous to CPDAGs.  
   - **Figure 7** presents SHD vs sample size for multiple \(n,\ell,d\) settings; glvLiNG’s better performance on denser graphs is visible and matches the textual explanation.  
   - **Figure 8**’s sector-colored stock network clarifies the interpretability claim (e.g., bank nodes as sources, real estate as sinks, utilities in cycles, plausible latent).

9. **Experimental evaluation is modest but appropriately targeted.**  
   The experiments are not large-scale but are well-chosen for a theory-heavy paper: exhaustive enumeration up to 6 nodes to characterize equivalence-class sizes (Table 3), runtime evaluation against MILP (Table 4), and robustness to misspecified structural assumptions by running LaHiCaSl and PO-LiNGAM under oracle conditions (Table 5). The simulation study in Figure 7 probes where glvLiNG’s structural-assumption-free nature helps (denser graphs, more latents). The real-data example (stock returns) is plausible and supports that the algorithm can yield interpretable structures.

10. **Positioning within the literature is solid and self-aware.**  
    The related work section and Appendix C.5 show familiarity with both causal graph equivalence literature (MAGs, PAGs, CPDAGs, Meek rules, etc.) and LiNG/LvLiNGAM work. The authors are explicit about the limitation that OICA is used as an oracle in the consistency guarantee and discuss directions for more practical estimators.

## Weaknesses

1. **Strong reliance on OICA and faithfulness, with limited practical discussion.**  
   The guarantees of glvLiNG assume access to an oracle OICA and a strong faithfulness condition (Assumption 1) that all relevant minors of the mixing matrix reflect generic path ranks. In practice, OICA is notoriously fragile and slow in high dimensions. The paper acknowledges this, but the experimental section does not critically examine how violations of these assumptions affect the reconstructed equivalence class. The heuristic treatment of empirical ranks (Section D.4, “full-rank confidence score”) and the ad hoc thresholding seem under-motivated and are not stress tested. For example, there is no experiment quantifying how frequently the inferred matroid violates matroid axioms or how sensitive the graph recovery is to small perturbations in singular values.

2. **Empirical evaluation is thin relative to the ambition and complexity of the theory.**  
   While I do not expect a benchmark-style empirical tour for such a theoretical paper, the experiments feel somewhat narrow.  
   - For equivalence-class statistics (Table 3), the presented histogram fields are almost all zeros in the excerpt, making it impossible to interpret distribution shapes or typical class sizes; the narrative only extracts one or two numbers (e.g., 783 classes for \(n=5,l=2\)).  
   - For glvLiNG’s performance (Figure 7), results are in terms of SHD to *some* graph in the ground-truth equivalence class, but there is no breakdown of (a) how often the true equivalence class is fully recovered, (b) how many extra equivalent graphs are produced, or (c) errors in ancestral relations among observed variables, which Theorem 3 suggests should be particularly well-identified.  
   - The stock-market example (Figure 8) is interesting, but no alternative methods or simpler LiNG-based baselines are compared on this dataset, nor are stability analyses provided.

3. **Some central constructions are only sketched and quite opaque to non-matroid experts.**  
   The algorithmic Phase 1 (Appendix A.3) relies on the \(\alpha\)-system of the dual matroid (Equation (A.17)) and the Mason–Ingleton–Piff pipeline. For readers not already comfortable with transversal/gammoid dualities and matroid minors, the description will feel like a black box. The main text simply asserts “this is equivalent to reconstructing the digraph representation of the strict gammoid… based on Mason (1972) and Ingleton & Piff (1973)” and then moves on. While formally acceptable, this limits the accessibility of the core realization step. Similarly, the notion of cocircuits and fundamental cocircuits in Appendix A.1 is dense; the main algorithmic implications (e.g., in Lemma 10 and Corollary 2) could benefit from a more intuitive explanation in the main body.

4. **Transformational characterization is not fully translated into a practical traverser.**  
   Theorem 3 is elegant, but the paper does not explore how BFS/DFS over admissible operations scales in practice beyond toy cases. Figure 3 and Figure 5 show small equivalence classes, and the web demo is mentioned, but there is no explicit complexity discussion or empirical scaling curve for equivalence-class traversal (e.g., number of graphs visited as function of \(n\) for typical densities). In large systems, even enumerating the class may be infeasible; some insight into how one might sample a few representative graphs, or just identify invariant edges (as in Theorem 4), would make the results more actionable.

5. **Irreducibility reduction, while natural, deserves more nuance.**  
   Proposition 1 and Proposition 2 formalize irreducibility and the corresponding reduction, but the discussion assumes that removing non-ancestors and merging redundant latents is harmless for all downstream tasks. In some applications, scientists may actually care that a latent is known to have only one observed child (e.g., instrument-like variables or nearly pure measurement). The paper is clear that irreducibility is a canonicalization, not a modeling restriction, but the presentation is a bit dismissive about reintroducing such latents post hoc. It would help to say more explicitly how one could reconstruct plausible non-irreducible models from the irreducible equivalence class, and to what degree such reconstructions are underdetermined.

6. **Limited comparison to other LiNG latent-structure works that also weaken assumptions.**  
   While many references are listed, some immediately relevant linear non-Gaussian latent-variable works are not really contrasted against in a focused way. For instance, Chen, Cai & Zhang (2022) (see below) propose causal discovery in LiNG acyclic models with multiple latent confounders and weaker structural constraints than classical measurement models. The paper mentions numerous LvLiNGAM variants (e.g., Salehkaleybar et al. 2020, Wang & Drton 2023, etc.) but does not systematically explain, in a small table or paragraph, where exactly their assumptions differ from the fully assumption-free setting here and what those methods can or cannot recover compared to glvLiNG.

7. **Figures/Tables could more directly support some key interpretive claims.**  
   - **Figure 7**: while it shows SHD vs. sample size, the axes and legends in the provided snippet are rather condensed; it is hard to see numerical scales and whether differences are statistically significant across runs. Also, the text claims that glvLiNG is “more robust to latent dimensionality” (Section D.4); this is only supported by one or two comparisons (e.g., \(n=13,d=3\)) and not systematically summarized in a table. A small table summarizing SHD as function of \(\ell\) for fixed \(n,d,N\) would make the claim more convincing.  
   - **Table 4**: the runtime breakdown gives Phase 1 vs Phase 2 times, but there is no sense of the number or complexity of rank queries issued to the OICA mixing matrix; since these are potentially expensive, the overall cost should be decomposed more fully. Also, glvLiNG is only compared against a relatively naive MILP baseline; alternative constraint solvers (CP-SAT, specialized matroid or graph realization algorithms) might be more informative competitors.

8. **The faithfulness assumption is nontrivial but under-discussed.**  
   Assumption 1 requires that all \(d_X \times d_X\) and \((d_X-1)\times(d_X-1)\) minors behave generically. This is a strong and non-testable assumption in real data. The paper could be clearer on what kinds of parameter degeneracies would violate it and how often they might occur in practice. For instance, in cyclic systems, subtle cancellations in \((I-B)^{-1}\) can happen; Appendix C.4 hints at “non-rank” constraints in mixing matrices but does not thoroughly discuss whether these present practically relevant faithfulness violations. As the ultimate guarantees of equivalence recovery hinge on this, a more explicit caveat is warranted.

9. **Algorithm specification is partially relegated to the appendix.**  
   Section 5 in the main text describes glvLiNG at a rather high level, referring repeatedly to Appendix A. For reproducibility and comprehension, at least a pseudocode-level description of the full pipeline (OICA, Phase 1, Phase 2, traversal) in the main paper would help, even if proofs stay in the appendix. For example, exactly how the “full-rank confidence score” heuristics in Section D.4 feed into Phase 1’s matroid fitting is only lightly sketched.

10. **Minor technical nitpicks and notation density.**  
    - There is heavy overloading of \(r\) for edge ranks (Definition 4) and matroid ranks (Appendix A.1, Equation (A.1)), which is formally clarified but conceptually a bit confusing.  
    - Some key path/edge rank equalities are stated “for generic \(A\)” without a clear statement in the main text about how degenerate parameters behave; the proof in Lemma 2 references Talaska (2012) but is omitted.  
    - In Lemma 7 and Lemma 13, the introduction of \(V_i\)’s “nonchildren” and re-expression in terms of submatrices \(Q_{R,Y}\), \(Q_{R\setminus\{V_j\},Y}\) is correct but terse; a schematic example analogous to Figure 2, with highlighted rows/columns indicating these sets, would help.

Overall, the weaknesses are about practical applicability and clarity/coverage, not about the core correctness or novelty of the theoretical contributions.

## Potentially Missing Related Work

1. **Wei Chen, Ruichu Cai, Kun Zhang (2022). “Causal Discovery in Linear Non-Gaussian Acyclic Model With Multiple Latent Confounders.”**  
   This work addresses causal discovery in LiNG acyclic models with multiple latent confounders and more flexible structures than pure indicator measurement models. It is directly relevant as another attempt to move away from strong measurement assumptions in LiNG latent-variable settings. It should be cited in the related work discussion (Appendix C.5 or Section 1) and compared against glvLiNG regarding (a) whether latents can have arbitrary children/parents, (b) whether cycles are allowed, and (c) what aspects of the latent structure are identifiable.

(The Tashiro et al. 2014 work is already cited in the paper as a robustness method against latent confounders.)

## Questions

1. **Practical robustness of rank estimation:**  
   Can the authors provide additional experiments (or at least a discussion) on how sensitive glvLiNG is to noise in the estimated mixing matrix? For example, if one randomly perturbs the OICA matrix such that, say, 10% of theoretically full-rank minors become numerically close to singular, how often does Phase 1 recover the correct matroid? Are there principled statistical tests for distinguishing coincidental low singular values from true rank drops?

2. **Partial access to ranks without full OICA:**  
   The conclusion mentions that some methods provide partial access to rank information without full OICA. Can the authors be more concrete: which methods (e.g., regression-invariance based, cumulant-based) could be plugged into glvLiNG, and which parts of the theory would need to change if only some subset of ranks were reliably estimable?

3. **Scalability of equivalence-class traversal:**  
   Beyond small examples (Figures 3 and 5), what are realistic limits on equivalence-class traversal using Theorem 3 in terms of \(n\) and average degree, based on experiments? Could the authors report, for a few random models, (i) the number of admissible operations, (ii) the size of the equivalence class found by BFS up to some depth, and (iii) how often multiple cycle-reversal configurations exist?

4. **Recoverability of non-irreducible structure:**  
   Suppose in an application domain the scientist believes there is a latent with a single observed child (which irreducibility removes). Given only the irreducible equivalence class, can the authors clarify how many non-irreducible supergraphs are observationally indistinguishable and whether any of their features (e.g., sign constraints or number of such latents) can be inferred?

5. **Edge-rank interpretation for practitioners:**  
   Edge ranks and matroid notions (circuits, cocircuits, flats) may be intimidating to practitioners. Can the authors provide a brief “engineering view” of Lemma 7 and Lemma 10, perhaps in the supplement, with a worked-out tiny example that explicitly shows the support matrices and matchings, beyond the high-level Example 2?

6. **Non-rank constraints:**  
   Appendix C.4 gives a concrete example of non-rank equality constraints in the mixing matrix. Do such constraints ever distinguish graphs that are indistinguishable by ranks, or are they always redundant with rank information in the LiNG non-Gaussian case? Clarifying whether Lemma 3 is “if and only if” even in the presence of those constraints would strengthen the theoretical story.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating
4: excellent — The technical development (rank/edge-rank duality, equivalence criteria, irreducibility, matroid-based reconstruction, and transformational characterization) is careful, grounded in well-known matroid theory, and internally consistent. Remaining concerns are about practical robustness and coverage, not about core correctness.

## Presentation Rating
3: good — The paper is dense but generally well written, with clear organization and many helpful figures/tables. However, matroid-heavy parts and the algorithmic details are mostly in the appendix and will be hard to digest for non-experts; more intuition and streamlined notation would help.

## Contribution Rating
4: excellent — Providing the first structural-assumption-free distributional equivalence characterization for linear non-Gaussian models with latent variables and cycles, plus a working recovery algorithm and new edge-rank tools, is a substantial contribution to causal discovery theory.

## Overall Rating
8: Accept, good paper (poster).  
The work makes a strong and timely theoretical contribution by closing an important gap in equivalence characterization for LiNG models with latents and cycles, backed by nontrivial matroid machinery and a concrete algorithmic instantiation. While empirical evaluation and practical aspects (especially OICA reliance and robustness) are not fully developed, the theoretical depth and clarity of the core results merit acceptance.

## Reviewer Confidence
4: confident — I am familiar with causal discovery and LiNG/LvLiNGAM literature and have traced the key derivations and matroid constructions, though I have not formally verified every detail of the more intricate proofs.