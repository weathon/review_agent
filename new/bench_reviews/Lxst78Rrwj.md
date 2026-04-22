Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

This paper introduces GLIDE, a causal discovery framework that exploits the invariance of the effect-cause conditional distribution P(X|Pa[X]) to changes in the prior distribution P(Sources) of source variables. Instead of imposing functional-form assumptions, GLIDE constructs augmented datasets via principled downsampling to vary P(Sources), then selects parent sets by finding maximal cliques in a Markov blanket-derived graph whose conditional distributions exhibit minimal variance across augmentations. The method achieves O(d²) complexity under an empirically-verified sparsity assumption and demonstrates strong performance on benchmarks including the 1041-variable Munin dataset.

## Strengths

- **Novel invariance-based framework:** The core insight—that P(effect|cause) is invariant to shifts in P(cause)—provides a model-agnostic approach to causal parent identification that differs meaningfully from standard constraint- and score-based methods (Section 4.1, Theorem 1). This perspective is genuinely novel and well-motivated.

- **Principled data augmentation pipeline:** Theorems 4–6 (Section 4.2) provide a complete, theoretically grounded resampling procedure: minimum downsampling rate (Thm 4), optimal resampling achieving that rate (Thm 5), and convex hull characterization of valid source priors (Thm 6). These results are correct and potentially useful beyond this application.

- **Model-agnostic applicability:** GLIDE handles both continuous and categorical data without imposing parametric assumptions (Table 1), unlike NOTEARS/SCORE/DAS which are restricted to specific data models. This is a real practical advantage.

- **Strong empirical performance on spurious rate:** On real-world datasets, GLIDE achieves dramatically lower spurious rates (e.g., 1.8% vs. 42.4% for GIES on Munin in Table 2), demonstrating that the invariance test yields more reliable causal predictions than score-based methods.

- **Scalability demonstrated on large graphs:** GLIDE successfully processes the 1041-node Munin dataset where PC fails entirely (N/A at 48 hours), showing practical viability for larger-scale problems.

## Weaknesses

### Fatal
None.

### Major

- **The bidirectional implication claim is incorrect, introducing a theoretical gap in the parent identification mechanism.** The paper states (Section 4.1, line 103): "When m is infinitely large, the implication in Eq. (1) becomes bi-directional and V[P₊(X|Z)] = 0 definitively implies Z = Pa[X]." This is false. By the local Markov property (Eq. II, line 75), X ⊥ (non-descendants \ Pa[X]) | Pa[X], so P(X|Z) = P(X|Pa[X]) for any Z ⊇ Pa[X] that adds only non-descendants d-separated from X given Pa[X]. For example, a co-parent (spouse) S of X in a v-structure (S → C ← X) is in M(X) but satisfies X ⊥ S | Pa[X], giving P(X|{A,S}) = P(X|{A}) and hence zero variance. The invariance test cannot distinguish Pa[X] from such supersets. Since the algorithm restricts candidates to maximal cliques (Section 4.3, line 195) and Pa[X] may not be a maximal clique, this can cause GLIDE to select a superset of the true parents, producing spurious edges. This is a substantive gap in the theoretical foundation, though estimation noise may partially mitigate it in practice (as suggested by the low empirical spurious rates).

- **The basis-vs-source gap is acknowledged but not formally resolved.** The paper replaces true source variables with "basis" variables (Definition 3, Section 4.2.1) for practicality, since sources cannot be identified without intervention. While Theorem 2 establishes equal cardinality and informal arguments are given about "similar properties" (line 137), no formal guarantee ensures that varying P(Basis) induces sufficient variation in P(X|Z) for Z ≠ Pa[X] to be detected. When Basis ≠ Sources, source variables absent from the basis may not have their priors adequately varied, potentially making P(X|Z) approximately invariant across augmentations even when Z ≠ Pa[X], causing false negatives (missing edges). The lack of formal analysis here leaves the soundness of the method's implementation uncertain.

### Minor

- **The O(d²) complexity claim depends on an empirically-verified but theoretically unbounded sparsity assumption.** The Bron-Kerbosch complexity is O(d·p·3^{p/3}), and the paper reports p ≤ 13 on all benchmarks (Section 4.3). However, these benchmarks use sparse graphs (E = V in the normal setting, E ≤ 2V in the extreme setting). For graphs with many v-structures, p can grow with d, making O(d²) unsubstantiated in general. The paper is transparent about this ("empirically verified assumption on causal graph sparsity," line 115), but O(d²) is stated as a definitive claim in the abstract and title area.

- **The "25× faster" and "73.3% reduction" claims are cherry-picked.** The 25× figure is the best speedup against NOTEARS in specific synthetic settings; on Munin (the largest real-world dataset), GLIDE is ~100× slower than GIES (6200s vs. 61.5s). The 73.3% spurious rate reduction is a maximum against the worst baseline in one setting, not a representative average. The "up to" qualifier partially mitigates this, but the framing in the abstract is misleading.

- **P(X|Z) estimation and norm specification are underspecified.** The invariance test in Eqs. (2)–(3) computes ‖P_i(X|Z) − P̄(X|Z)‖², but the norm type, estimation method for conditional distributions, and bin width selection for continuous data (briefly mentioned on line 161) are not specified. These choices directly affect the test's sensitivity and specificity.

### Trivial

- The causal sufficiency assumption (no confounders) is stated (line 79) but its implications for the invariance test are not analyzed. When confounders exist, both the basis finding and the invariance test could break down.

## Nice-to-Haves

- Report how often Pa[X] is a proper subset of the selected maximal clique in experiments, to directly measure the severity of the superset identification issue.

- Add a minimality selection step: after finding the minimum-variance maximal clique, search its subsets for a minimal set with comparable variance.

- Test on denser graphs (E ≥ 3V or 5V) to validate the sparsity assumption for the O(d²) complexity claim.

- Provide formal conditions under which the basis-to-source transfer preserves the invariance test's properties.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Critic's claim about "O(d²) complexity claim rests on unsupported sparsity assumptions"**: The paper is transparent that this is an empirical assumption. Downgraded to minor rather than a major/fatal issue since the paper explicitly qualifies it.

- **Critic's claim about missing NOTEARS/DAS/SCORE baselines for categorical data**: The paper (Table 1 and line 228–229) explains these baselines produce "very poor performance that is not meaningful for comparison." This is a legitimate design choice, not a missing baseline—it would not meaningfully improve the paper to include nonsensical results.

- **Criticism about "no statistical significance tests despite 10 runs"**: Confidence intervals are reported (e.g., Table 2). This is standard practice and sufficient.

- **Critic's claim about reproducibility (unspecified estimation details)**: While the estimation details could be more complete, the code is publicly available. The gap is minor enough to be a minor issue, not a major one.

- **Strength Finder's claim that "quadratic-complexity parent search via maximal cliques" is a core strength**: While the clique-based reduction is clever, this is undercut by the major weakness that the maximal clique selection may identify supersets rather than exact parent sets. Keeping this as a supporting contribution rather than a core strength.

## Novel Insights

The paper's invariance-based perspective on causal discovery is genuinely novel and well-motivated. The key theoretical insight—Theorem 1's one-directional guarantee that non-zero variance implies non-parent status—is correct and useful. However, the paper's bidirectional claim (zero variance ⟹ parent status) is incorrect, and the method's practical viability may depend on estimation noise acting as an implicit regularizer that distinguishes true parents from supersets. This creates an interesting paradox: the method works well empirically (low spurious rates) precisely because the theoretical ideal is never achieved in finite samples, and the residual variance from estimation acts as a differentiator. A more honest framing would acknowledge the one-directional guarantee and treat the parent-finding as a heuristic selection among minimum-variance candidates rather than a definitive identification.

## Suggestions

- Correct the bidirectional claim in Section 4.1 to state only the proven one-directional implication; add a discussion of when zero-variance supersets can occur and how estimation noise mitigates this.

- Add an explicit minimality step to the parent-finding algorithm: after selecting the minimum-variance maximal clique, prune unnecessary variables by testing whether removing each one increases the variance significantly.

- Frame the O(d²) complexity as conditional on the sparsity assumption rather than as a general guarantee.

- Report per-variable parent-set analysis showing how often the selected maximal clique differs from the true Pa[X] on synthetic graphs where the ground truth is known.

## Score and Decision Calibration

**Anchors compared:**

- High (avg > 7): *Unifying Causal Representation Learning with Invariance Principle* (7.0, Accept Poster) — strong theoretical + empirical contribution on invariance in causality; *Deriving Causal Order from Single-Variable Interventions* (7.0, Accept Poster) — novel invariance-based causal discovery with theoretical guarantees.
- Medium (4–6): *Doubly Robust Structure Identification* (4.8, Reject) — has a gap between proven theorem and practical algorithm; *Beyond DAGs: Latent Partial Causal Model* (5.75, Reject) — interesting framing but theoretical claims overreach; *Learning Identifiable Causal Structures with Pairwise Representation* (4.25, Reject) — overclaimed advantages, theoretical gaps.
- Low (< 3): *IRIS* (2.5, Reject) — lacks theoretical grounding entirely; *DAG-based Generative Regression* (3.0, Reject) — large empirical margins but theoretical incompleteness and weak grounding.

This paper is significantly stronger than the low anchors: it has a correct one-directional theoretical result (Theorem 1), correct resampling theory (Theorems 4–6), and strong empirical performance. It is weaker than the high anchors: those papers have complete theoretical guarantees whereas this paper has a significant gap in the bidirectional implication claim. Compared to the medium anchors, this paper shares the profile of having good empirical results and partial theoretical foundations but overclaimed theoretical completeness. The bidirectional implication error is a substantive theoretical flaw, but the paper's overall contribution (novel framework, correct core theorem, principled resampling, strong experiments) is meaningful. The paper is somewhat stronger than the medium anchors because it has a correct core result and much stronger empirical evidence.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>