## Summary

This paper introduces GLIDE, a causal discovery framework that tests for direct parent-child relationships by checking whether the conditional distribution $P(X \mid Z)$ is invariant across synthetic data augmentations that alter the prior over source/basis variables. The method couples this non-parametric invariance test with a quadratic-time search over candidate parent sets derived from maximal cliques in an augmented bidirectional graph. Empirically, GLIDE achieves strong accuracy (low SHD and spurious rate) across synthetic and real-world benchmarks, including the 1,041-variable Munin network.

## Strengths

- **Principled non-parametric invariance test.** The paper derives a causal test (Theorem 1) grounded in the modularity of causal mechanisms, requiring no parametric assumptions on the structural equations. The downsampling scheme (Theorems 4–6) to synthesize informative source priors from a single observational dataset is clever and well-motivated.
- **Strong empirical accuracy.** On real-world categorical benchmarks (Table 2), GLIDE achieves the best SHD on all seven datasets and dramatically outperforms GIES in spurious rate (e.g., 1.8% vs. 42.4% on Munin). On synthetic continuous data, it is competitive with or better than NOTEARS and MLP-NOTEARS in accuracy while being faster in several settings (Figures 2–3).
- **Broad applicability.** Unlike several baselines, GLIDE is evaluated on both continuous and categorical data, and it scales to problems where PC fails to finish within 48 hours (Table 2).

## Weaknesses

### Major

- **Maximal-clique restriction can exclude the true parent set, and the invariance test identifies supersets, not exact parents.** Theorem 7 correctly states that $\text{Pa}[X]$ is a *clique* in $G'(X)$, but Section 4.3 restricts candidate parent sets to *maximal* cliques. As a counterexample, if $X$ has parents $\{B_1,B_2\}$ and a child $C$ whose parents are $\{X,B_1,B_2\}$, then $\mathbb{M}(X)=\{B_1,B_2,C\}$, $G'(X)$ is a triangle, and the only maximal clique is $\{B_1,B_2,C\}$. The true parent set $\{B_1,B_2\}$ is never enumerated. Moreover, Section 4.1 incorrectly claims that in the limit of infinite augmentations, $\mathbb{V}[P_+(X\mid Z)]=0$ implies $Z=\text{Pa}[X]$. This is false: if $Z\supseteq \text{Pa}[X]$ and the extra variables are descendants whose mechanisms are also invariant, the variance is still zero (e.g., $P(X\mid B_1,B_2,C)$ is invariant in the counterexample above). Because Eq. (3) selects a single $\arg\min$ without a minimality principle, the algorithm is not guaranteed to return $\text{Pa}[X]$ even when the correct set is present. These two issues are structural: they break the correctness guarantee for parent recovery and are not mere clarifications.
- **Scalability claims are contradicted by the largest real-world benchmark.** The abstract and introduction claim that GLIDE achieves “up to $25\times$ reduction in processing time” and “generally better scalability” than state-of-the-art methods. However, on the largest real-world dataset (Munin, $d=1041$), GLIDE takes $\sim$6,200 seconds while GIES takes $\sim$62 seconds—two orders of magnitude *slower* (Table 2). While GLIDE is faster than PC and some continuous baselines in other settings, the headline scalability claim is overstated and undermined by the very benchmark held up as evidence of large-scale applicability.

### Minor

- **Handling of continuous data is underspecified.** Section 4.2.2 notes that “to accommodate for continuous data, we use binning with fixed bin width to make the data categorical,” but provides no discussion of bin count, edge effects, empty cells, or how binning interacts with the variance of the test statistic. For parent sets of moderate size, histogram-based estimation of $P(X\mid Z)$ is exponential in $|Z|$; the $O(md^2)$ complexity bound assumes evaluating a candidate is cheap, which is not justified for this estimation step.
- **Missing conceptual comparison with Invariant Causal Prediction (ICP).** The core idea—testing whether $P(\text{effect}\mid\text{cause})$ is invariant across distributional shifts—is closely related to ICP (Peters et al., 2016, already cited). The paper does not discuss how its downsampling-based environment generation differs from existing invariance-based methods, which weakens the novelty claim.
- **Absolute SHD is hard to interpret across graphs of different sizes.** Table 2 reports raw SHD without normalizing by the number of edges in the ground-truth graph, making it difficult to assess whether an SHD of 883 on Munin is large or small in relative terms.

### Trivial

- None.

## Nice-to-Haves

- Include FGES (a fast score-based baseline) in the real-world experiments to clarify the true scalability–accuracy trade-off.
- Add an ablation studying bin width and parent-set cardinality for continuous data.
- Add a minimality principle (e.g., selecting the smallest set with variance below threshold) to the parent-finding step, and justify or bound the additional cost.

## Removed Points

These points are flagged to be removed, treat them with caution:
- Criticisms about missing appendix, missing proofs in appendix, or absent references. The parser strips those sections from all papers; they exist in the original submission.
- Pure formatting/style nitpicks, typos, spelling, grammar, or broken characters—these are parser artifacts, not author errors.
- Claims that cited models, benchmarks, or datasets do not exist or are unreleased. Everything cited in the paper is assumed to exist.
- Generic reproducibility nitpicks such as undisclosed hyperparameters or missing training logs.

## Novel Insights

The downsampling scheme to generate synthetic interventional-like environments from purely observational data is genuinely novel and practically appealing. The idea of using the convex hull of admissible source priors (Theorem 6) to ensure augmentations are sufficiently different from the original data, while preserving enough samples to estimate conditionals, is an interesting contribution that could be useful beyond this specific algorithm. However, the paper's core algorithmic logic is undermined by the interaction between the maximal-clique search restriction and the superset-accepting invariance test.

## Suggestions

- Fix the search space to enumerate *all* cliques (or prove maximality under additional assumptions) and integrate a minimality principle into the parent-finding criterion. Without this, the theoretical justification for correctness collapses.
- Temper the scalability claims in the abstract and introduction to reflect that GLIDE is not universally faster than all score-based baselines, particularly on large sparse real-world graphs.
- Add normalized metrics (e.g., SHD per edge, F1) to the real-world benchmark table.

## Score and Decision

**Calibration anchors used:**
- `/home/wg25r/review_agent/human_reviews/wmV4cIbgl6.md` (Avg 7.33, Accept Spotlight): A large-scale benchmark paper with strong real-world experiments and clear utility. GLIDE has comparably strong empirical breadth but lacks its theoretical solidity.
- `/home/wg25r/review_agent/human_reviews/KWO8LSUC5W.md` (Avg 5.60, Accept Poster): A method paper with overclaimed novelty and limited experimental scope. GLIDE has stronger experiments but more fundamental algorithmic flaws.
- `/home/wg25r/review_agent/human_reviews/ZKRHiu5kE4.md` (Avg 4.25, Withdrawn): Strong synthetic and real-world experiments undermined by incorrect theoretical proofs. GLIDE is analogous—strong empirics, serious theoretical issues—but its flaws are in the core algorithmic logic rather than auxiliary proofs, placing it slightly below this anchor.
- `/home/wg25r/review_agent/human_reviews/or8wkKoBP4.md` (Avg 4.00, Reject): Mathematical mistakes and no experiments. GLIDE is well above this due to its extensive and positive empirical results.
- `/home/wg25r/review_agent/human_reviews/JzFLBOFMZ2.md` (Avg 3.20, Reject): Fundamental theoretical flaws and no guarantees. GLIDE is above this because its empirical results are genuinely strong and broad.

GLIDE presents an appealing conceptual idea and impressive empirical numbers, but its central algorithmic claims are structurally flawed: the maximal-clique restriction can exclude true parents, and the invariance test admits supersets without a minimality safeguard. Combined with overstated scalability claims that are directly contradicted by the largest real-world benchmark, the paper is below the acceptance threshold in its current form. If the authors can fix the search space and parent-selection criterion to provide correctness guarantees, and temper the scalability claims, the paper could become a strong contribution.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>