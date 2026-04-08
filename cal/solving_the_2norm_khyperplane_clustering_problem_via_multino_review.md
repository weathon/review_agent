=== CALIBRATION EXAMPLE 50 ===

# Final Consolidated Review
## Summary

The paper proposes strengthening the classical MI-QCQP formulation for the NP-hard $k$-Hyperplane Clustering problem ($k$-HC2) by intersecting it with constraints derived from polyhedral norm ($p = 1, \infty$) formulations. The key theoretical result is that including $\infty$-norm constraints reduces the number of SBB nodes needed to obtain a nonzero global lower bound from exponential $2^{k(n-1)}$ to polynomial $k(n-1)$. Experiments on synthetic benchmarks show median speedups up to 41× and up to 63% more instances solved.

## Strengths

- **Provably polynomial SBB convergence for nonzero bounds (Proposition 4):** The result that adding $\|\mathbf{w}_j\|_\infty \geq 1/\sqrt{n}$ constraints and branching on the associated binary variables $u_{jh}$ yields a nonzero lower bound in only $k(n-1)$ nodes—versus exponential $2^{k(n-1)}$ for the baseline—is a strong and uncommon theoretical contribution for nonlinear integer programming. Establishing lower bounds on the number of branching operations required to compute a nonzero lower bound is a type of result the paper itself notes is "extremely rare in integer programming."

- **Novel multi-norm intersection framework (Theorem 1, Corollaries 1–2):** The technique of deriving valid relaxations by exploiting norm congruence inequalities and then intersecting constraints from different norms to tighten formulations is creative. The derivation of an approximation factor for the combined multi-norm relaxation (Corollary 2, with factor $1/\sqrt[4]{n}$) via the construction of a new norm $\max\{\|\mathbf{x}\|_\infty, \frac{1}{\sqrt{n}}\|\mathbf{x}\|_1\}$ (Lemmas 3–4) is a nontrivial and well-executed piece of analysis.

- **Substantial empirical speedups on tested instances:** Tables 1–4 show consistent and large speedups across both testbeds. On Low-dim, 21 instances unsolvable in 46 hours with the baseline are solved by the strengthened formulations—some in under 10 seconds. The Wilcoxon signed-rank tests with Holm correction confirm statistical significance ($p$-values as low as $5.6 \times 10^{-9}$). These are not marginal improvements.

## Weaknesses

### Major:

- **Experimental evaluation limited to small synthetic instances with no real-world validation.** Both testbeds use synthetically generated data (points on random hyperplanes plus Gaussian noise) with at most $m = 30$ points in $n \leq 5$ dimensions with $k \leq 5$ hyperplanes. The introduction cites applications including LiDAR classification, medical prognosis, and image analysis—domains where datasets have hundreds or thousands of points in much higher dimensions. The gap between the problem scales where the method is demonstrated and those of the claimed applications is substantial. While exact global optimization of an NP-hard problem naturally limits scale, the paper would be significantly strengthened by (a) experiments on at least one real-world dataset, even if small, and (b) analysis of how the method degrades as problem size increases beyond the tested range.

- **Assumption 1 (midpoint branching) creates a meaningful theory-practice gap.** Propositions 2–4 rely on Assumption 1, which posits that branching on variables with symmetric domains occurs at the midpoint. Modern SBB solvers like Gurobi employ strong branching, pseudo-cost branching, and other heuristics that do not follow this rule. The paper uses Gurobi experimentally and obtains good results, which partially validates the practical relevance of the insight, but the theoretical guarantees themselves do not directly apply to the solver's actual behavior. The paper should more explicitly acknowledge this gap and discuss whether the empirical speedups are consistent with the theory's predictions (e.g., does the ∞-norm formulation indeed obtain nonzero bounds earlier in practice?).

- **No comparison with heuristic or approximate methods.** The paper only compares strengthened formulations against the baseline exact formulation. There is no comparison with the $k$-planes heuristic of Bradley & Mangasarian (2000) or other approximate methods. For ML practitioners, the value proposition of exact global optimization depends on how much better the exact solutions are than fast heuristics. Reporting the optimality gap of heuristic solutions on the solved instances would quantify when the computational cost of exact solving is justified.

### Minor:

- **Multi-norm formulation underperforms individual norms in High-dim (Table 2), and the explanation could be sharper.** The multi-norm formulation achieves only 5.6× speedup versus 11.5× and 10.1× for the single-norm variants on High-dim. The text attributes this to the exponential versus polynomial node bounds of Propositions 3 and 4, but the deeper structural reason—that the multi-norm approximation factor ($1/\sqrt[4]{n}$ from Corollary 2) is strictly weaker than the individual norm factors ($1/n$ from Corollary 1)—is not clearly connected to the experimental observation. This matters because it reveals that adding more constraints can worsen the relaxation quality when the intersection's geometry is less favorable, a counterintuitive result that deserves explicit discussion.

- **Proposition 3 shows the 1-norm strengthening still requires $2^{k(n-1)}$ nodes for a nonzero bound.** While the paper states this, it could be more prominently noted: adding $\|\mathbf{w}_j\|_1 \geq 1$ constraints does not reduce the exponential node complexity for obtaining a nonzero bound—it only helps after the full branching tree on the $s_{jh}$ variables is built. The practical benefit of the 1-norm formulation comes from the fact that, once the tree is complete, the polyhedral constraint is satisfied in every leaf (no further branching on $\mathbf{w}$ needed), not from reducing the initial tree size.

### Trivial:

- The abstract phrase "formulating the problem in $p$-norms with $p = 2$" is slightly confusing; it should read $p \neq 2$ to clarify that the strengthening comes from other norms.

## Nice-to-Haves

- **Lower bound convergence curves:** Plotting the global lower bound vs. time for all four formulations on a representative hard instance would visually confirm the central claim that the strengthened formulations obtain nonzero bounds earlier, directly linking theory to practice.

- **Analysis of time spent before vs. after first nonzero bound:** Table 5 shows millions of total nodes even for the best formulation. Dissecting what fraction of total solve time occurs before versus after the first nonzero bound would clarify whether the theoretical contribution (faster nonzero bounds) is the primary driver of empirical speedups, or whether other factors (tighter subsequent bounds, better pruning) also play major roles.

- **Warm-starting from heuristic solutions:** Using the $k$-planes heuristic to provide initial incumbent solutions could further reduce the SBB tree size and is standard practice in MIQP. This could amplify the already substantial speedups.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **W: OCR artifacts impact readability of equations.** This is a PDF parsing artifact, not a paper problem. (Formatting nitpick — removed.)

- **W: NP-hardness claim applies to feasibility, not optimization.** The distinction is not consequential for the paper's contribution, and the standard interpretation in the literature treats $k$-HC2 as NP-hard in the optimization sense as well. (Factually misleading criticism — removed.)

- **W: Proposition 1 nonconvexity proof only handles $n=2$.** A counterexample in $\mathbb{R}^2$ is sufficient to disprove convexity for any $n \geq 2$, since the 2-dimensional case can be embedded in higher dimensions. (Factually wrong criticism — removed.)

- **W: Random seeds for instance generation not specified.** The dataset generation procedure is fully described (uniform random hyperplane parameters, Gaussian noise with specified variance range), and the code is publicly available. Specifying exact seeds is a trivial reproducibility concern. (Nitpick — removed.)

- **W: Missing comparison with other global optimization approaches (RLT, SDP relaxations).** The paper's stated contribution is strengthening the standard MI-QCQP formulation via multi-norm constraints. Demanding comparison with an entirely different class of global optimization techniques is scope creep. The comparison against the baseline formulation directly measures the effect of the proposed strengthening. (Scope creep — removed.)

- **W: Broader impact / ethics discussion is generic.** The paper addresses a mathematical optimization problem. Demanding detailed discussion of surveillance or biased clustering risks for an algorithmic optimization paper is beyond the paper's scope. (Scope creep — removed.)

- **W: Memory footprint of additional binary variables not analyzed.** The paper adds $O(nk)$ binary variables—a modest increase. No instances in the experiments showed memory-related failures, and the overhead is implicitly accounted for in the solve times. (Generic weakness not harmful to core claim — removed.)

## Novel Insights

The multi-norm intersection approach reveals a counterintuitive structural lesson: adding more valid constraints to an optimization formulation can *worsen* practical performance when the intersection of the corresponding feasible regions has worse geometric properties (here, a weaker approximation factor of $1/\sqrt[4]{n}$ vs. $1/n$). This happens because the multi-norm $\min\{\|\mathbf{w}\|_1, \sqrt{n}\|\mathbf{w}\|_\infty\}$, while dominating both individual norms, yields a "hybrid" norm whose congruence gap with the 2-norm is larger than either individual gap. For SBB, the key insight is that the *geometry of the constraint boundary* (2^n facets for the 1-norm ball vs. 2n facets for the ∞-norm ball) is more important for branching efficiency than the *tightness* of the relaxation—polynomial branching on a weaker but simpler constraint (∞-norm) outperforms exponential branching on a tighter but complex constraint (1-norm). This has implications beyond $k$-HC: when designing strengthening constraints for SBB, the combinatorial structure of the constraint (number of disjunctions needed) may matter more than its relaxation quality.

## Suggestions

- Run experiments on at least one real-world dataset (e.g., a small LiDAR segment or UCI regression dataset with $n \leq 5$) to demonstrate the method handles real noise distributions and feature correlations, even if the problem size remains modest.

- Report the optimality gap of the Bradley & Mangasarian (2000) $k$-planes heuristic on the solved instances, to quantify when exact global optimization provides meaningfully better solutions.

- Add a brief explicit discussion acknowledging that Assumption 1 is idealized and that the empirical results, while consistent with the theoretical predictions, are obtained with a solver that does not follow midpoint branching. Even a single sentence noting this would strengthen the paper's candor.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 4.0]
Average score: 5.3
Binary outcome: Accept
