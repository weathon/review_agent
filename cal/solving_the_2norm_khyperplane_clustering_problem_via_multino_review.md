=== CALIBRATION EXAMPLE 56 ===

# Harsh Critic Review
Now I have read the paper in full, including the appendix with all proofs. Here is my section-by-section critical review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is accurate and descriptive. The abstract correctly summarises the method (multi-norm SBB strengthening), the theoretical guarantee (linear node count for a nonzero lower bound via polyhedral norms), and the experimental claims (41× median speedup, 63% more solved instances). These are all substantiated in the body. One minor concern: the abstract says "for every (suitably scaled) p ∈ N ∪ {∞}, one obtains a variant of k-HC2 whose optimal solutions yield lower bounds within a multiplicative approximation factor"—this is true but the 1/n factor for polyhedral norms is quite loose, and the reader does not learn until Section 3.3 that the multi-norm relaxation tightens this to 1/√n.

---

### Introduction & Motivation

The problem is well-motivated with a diverse set of applications (LiDAR, piecewise-affine system identification, dictionary learning, etc.). The NP-hardness citation (Megiddo & Tamir, 1982) is appropriate. The Contributions paragraph accurately previews what is proved and achieved. The Novelty paragraph is appropriately modest, calling out what is new relative to prior work. The single-author status is not a concern, but the community and citation base is almost entirely from operations research (INFORMS, EJOR, Math. Progm.), which raises the question of ICLR fit (discussed in Overall Assessment).

---

### Section 2: Preliminaries

The presentation of p-norm point-to-hyperplane distance using dual-norm conventions is careful and correct. Proposition 1 (nonconvexity of dp(a,H) in (w,γ)) is proved in the appendix via a concrete 2-dimensional counterexample for finite p′, and a separate case for p′=∞; both are valid. This foundational result well-motivates the SBB approach.

---

### Section 3: Approximating k-HC2 Using Different Norms

**Theorem 1** (cross-norm optimal-value inequality) is the paper's core theoretical engine. The proof is tight and correct: norm equivalence of p-norms → congruence of dp distances → congruence of the per-point minimisation → sum-over-points inequality → pass to minimisation over hyperplane parameters → apply quadratic homogeneity. Each step is logically sound.

**Corollary 1** correctly specialises Theorem 1 to the standard inequalities ‖x‖₂/√n ≤ ‖x‖∞ ≤ ‖x‖₂ and ‖x‖₂/√n ≤ ‖x‖₁/√n ≤ ‖x‖₂, giving approximation factor 1/n for each polyhedral norm. These are tight (Proposition 5 in the appendix shows the factors √n and 1/√n are tight), so no improvement is possible here.

**Multi-norm relaxation (Section 3.3)**: The key insight—combining both polyhedral constraints to obtain a tighter feasible region and better approximation factor—is original and non-trivial because ‖w‖_multi = min{‖w‖₁, √n‖w‖∞} is not a norm (its sub-level sets are non-convex). Lemma 2 (equivalence to max{d∞, d₁/√n} distance), Lemma 3 (this is a norm), and Lemma 4 (congruence n^{-1/4} ‖x‖₂ ≤ max{‖x‖∞, ‖x‖₁/√n} ≤ ‖x‖₂) are all proved correctly. Corollary 2 gives the 1/√n approximation factor improvement, which is the theoretical highlight of Section 3.

**A concern with Lemma 4 / Corollary 2**: The paper derives the congruence constant n^{-1/4} using a maximisation argument (problem (P) in the appendix proof of Lemma 4). The analysis relies on the optimal solution to (P) having the specific form x* = (1,…,1,r,0,…,0), established by a convexity argument (steps (i)–(iii)). This is rigorous and correct. However, Corollary 2 then reports the approximation factor as 1/√n (not n^{-1/2} = 1/√n applied squarely via Theorem 1). Let me verify: Theorem 1 uses (α/δ)² as the approximation factor. With α = n^{-1/4} and δ = 1, the factor is (n^{-1/4})² = n^{-1/2} = 1/√n. ✓ This is correct.

**Comparison**: Individual polyhedral norms give factor 1/n, multi-norm gives 1/√n—a strict improvement. This is a genuine theoretical contribution.

---

### Section 4: Solving via SBB

**Section 4.2 (Baseline formulation)**: The MI-QCQP formulation for k-HC(2,1) is presented with clear explanations of variables xij, di, and bounds. The symmetry-breaking choices (wj1 ≥ 0 for the 2-norm and 1-norm cases, restricted one-sided disjunction for the ∞-norm) are discussed. However, there is an important incompatibility: when the ∞-norm constraints are present, the wj1 ≥ 0 symmetry-breaking must be dropped (Section 4.4, last paragraph). This means the ∞-norm and multi-norm formulations are operating with **weaker symmetry breaking** than the baseline and the 1-norm formulation. This is a confounding factor in the experimental comparison, and the paper does not quantify how much of the performance difference can be attributed to this weaker symmetry reduction versus the norm constraints themselves. An ablation isolating this effect would strengthen the experimental section.

**Proposition 2** (exponential lower bound for 2-norm): The result is proved cleanly under Assumption 1. The key geometric argument is that the SBB convex envelope makes the origin (wj=0) feasible until every coordinate of every wj has been sign-determined. With the symmetry constraint wj1 ≥ 0 removing one sign decision per hyperplane, this requires 2^{k(n-1)} nodes. The proof is correct.

**Assumption 1** (branching at the midpoint of symmetric domains): This is stated as "holds in most SBB codes." In practice, Gurobi—the solver used in the experiments—employs sophisticated branching heuristics (pseudo-costs, strong branching) that often deviate from midpoint branching. The propositions are stated as lower bounds on node counts, which remain valid under any strategy, but the practical relevance of the assumption to the *experimental* setup is not discussed. The paper should note whether Gurobi's actual branching strategy was inspected or constrained.

**Proposition 3** (1-norm): The result is that a nonzero lower bound is obtained only after ≥ 2^{k(n-1)} nodes, i.e., **the same exponential count as Proposition 2**. The benefit of the 1-norm constraint is not in reducing the number of nodes for the first nonzero lower bound, but in that the leaf nodes of this (exponential) tree completely satisfy ‖wj‖₁ ≥ 1 and require no further branching on w. This distinction is subtle and could be stated more explicitly. A reader might otherwise expect the 1-norm to give a polynomial node bound analogous to Proposition 4.

**Proposition 4** (∞-norm): This is the paper's strongest structural result. The binary tree has depth k(n-1) and only needs 2k(n-1)+1 total nodes (not 2^{k(n-1)}). However, the proposition title claims "k(n-1) nodes suffice"—this refers to the depth, not the total node count. The paper uses "nodes" ambiguously; the appendix proof describes a tree with "two nodes per level," which implies total count ≈ 2k(n-1)+1, which is linear in kn. This should be clarified.

The tree argument in the proof is slightly informal: it assumes a specific branching schedule (branch on u_{jh} before anything else, always left-branch first). In an actual SBB run, the solver controls the branching order. It's unclear that Gurobi will actually follow this schedule, and the paper does not force this in experiments. The theoretical guarantee under Assumption 1 + the specific u_{jh}-first ordering needs to be more carefully qualified.

---

### Section 5: Computational Results

**Dataset size and representativeness**: The testbeds are very small (m ≤ 30, n ≤ 5, k ≤ 5). Even the hard instances barely exceed n=5 hyperplanes in 5 dimensions with 17 points. Real ML applications of k-HC typically involve thousands of points in higher dimensions. The paper makes no claim about what happens for larger instances, but the scaling limitation should be discussed more prominently—especially given that even 46 hours is not enough to solve some of the included instances.

**Instance generation**: All instances are synthetically generated with points lying exactly on hyperplanes plus Gaussian noise. This controlled generation ensures well-separated hyperplanes, which may make the problem easier than adversarial instances. No real-world datasets are tested.

**Comparison basis**: Tables 1 and 2 report median times on the subset solved by **all four formulations**. For LowDim, this subset is 20/43 instances (all 20 that the baseline can solve). For HighDim, it is 30/43 instances. This is the correct methodology for comparing speeds, but the "# solved" line (20/42/40/42 for LowDim; 31/41/40/37 for HighDim) is very informative and shows the strengthened formulations dramatically widen the solvability frontier. However, a shift penalty plot or virtual best comparison would better capture the full picture.

**Counterintuitive performance of multi-norm on HighDim**: On the HighDim testbed, the multi-norm formulation (5.6× speedup) is substantially worse than either individual norm (11.5× for 1-norm, 10.1× for ∞-norm), even though it has the best theoretical approximation factor (1/√n vs 1/n). The explanation offered—that the 1-norm branching has exponential cost that "dilutes" the benefit of the ∞-norm branching—is plausible but not fully convincing: if the ∞-norm subproblem is solved first and the ∞-norm constraints fully determine w in k(n-1) nodes, why does adding 1-norm constraints (which should only tighten the formulation) worsen performance? The likely cause is the additional kn binary variables sjh and kn continuous variables w+,w- introduced by the 1-norm MILP encoding, which inflates the problem size and potentially disrupts Gurobi's internal heuristics. This overhead is not analysed.

**Multi-norm solves fewer instances on HighDim (37 vs 40-41)**: The multi-norm formulation actually solves 3-4 fewer instances than the individual norms on HighDim. This directly contradicts the expectation from Corollary 2 (better approximation ⟹ tighter relaxation ⟹ more pruning ⟹ faster solving). The paper does not flag this as a finding or offer an explanation.

**Statistical testing**: Wilcoxon signed-rank tests with Holm correction are reported, which is appropriate. All p-values are highly significant.

**Missing ablation**: No experiment isolates the effect of the norm constraints from (a) the symmetry-breaking difference and (b) the variable-count overhead. A formulation that adds the ∞-norm variables but restores the original symmetry breaking (even at the cost of possible over-restriction, as discussed) would help disentangle these effects.

**No comparison with heuristics**: The paper does not compare against the k-means heuristic for k-HC, the distance-based reassignment heuristic from Amaldi & Coniglio (2013), or more recent subspace clustering methods (e.g., Tsakiris & Vidal, 2017, which is cited). Including such comparisons would help contextualize the cost of exact optimality.

---

### Section 6: Concluding Remarks

The conclusion is concise and accurate. The future direction on coresets is well-chosen. One missed future direction worth mentioning: investigating whether the norm-strengthening idea can be applied to other families of globally-solved nonconvex problems (e.g., mixed-integer QCQP more broadly).

---

### Appendix

The proofs are provided in full and are generally rigorous. The proof of Lemma 4 (congruence for the multi-norm) is the most technically involved and is correct. The proof of Proposition 4 deserves clarification on the node count vs. depth distinction (see above).

The detailed per-instance tables (Tables 3–5) are valuable. Table 5 (node counts for HighDim) clearly shows the reduction: 7.9M nodes on average for the baseline vs. 2.7M for the ∞-norm—roughly a 3× reduction in nodes, compared to the 10× speedup, suggesting Gurobi's time per node is also reduced with the stronger formulation.

---

### Overall Assessment

This paper makes a genuine and technically sound contribution to the exact solution of the k-hyperplane clustering problem. The theoretical framework—using norm-equivalence to derive cross-norm lower bounds, strengthening an MI-QCQP with MILP-representable polyhedral-norm constraints, and proving exponential vs. polynomial node-count gaps—is original and carefully worked out. Propositions 2–4 constitute a rare type of result: a concrete lower bound on SBB tree size for a nonlinear integer program. The experiments confirm that the proposed strengthening yields substantial practical speedups (10–41×) and dramatically increases the number of solvable instances.

However, several concerns weaken the paper's case for ICLR. First, and most fundamentally: the contribution is squarely in mathematical programming and combinatorial optimization, not machine learning. The experiments are entirely on randomly-generated synthetic data; no real ML task is solved or improved. The paper offers no comparison with heuristics, no application to a downstream learning task, and no engagement with the scale of datasets typical in ML. At ICLR, a paper addressing an NP-hard ML-related problem via exact optimization would need to demonstrate relevance to ML practice much more concretely. Second, there is a gap between theory and experiments: the best-theoretically-motivated formulation (multi-norm, with 1/√n approximation factor) is dominated by simpler alternatives in practice, especially on HighDim—and the paper's explanation for this is incomplete. Third, the confounding effect of different symmetry-breaking strategies across formulations is not isolated experimentally. This paper is excellent work for a mathematical programming venue (Math. Progm. Computation, INFORMS JoC, SIAM SODA/OPT), but needs stronger ML relevance and a more careful experiment design to compete at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a Spatial Branch-and-Bound (SBB) method to solve the $k$-Hyperplane Clustering problem with 2-norm distances ($k$-HC2) to global optimality. The core contribution involves strengthening the classical MI-QCQP formulation by intersecting it with scaled constraints from polyhedral norms ($p=1, \infty$). The authors theoretically prove that these multi-norm constraints reduce the number of SBB nodes required to compute a nonzero lower bound from exponential to linear/polynomial under specific branching assumptions, which is validated by experimental results showing significant speedups and improved solvability.

### Strengths
1.  **Theoretical Rigor and Novelty of Analysis:** The paper provides a rigorous theoretical analysis connecting the geometry of different norms to the complexity of the SBB tree. Specifically, Propositions 2, 3, and 4 offer a rare theoretical bound on branching operations required to prune the feasible region for nonlinear integer programming, establishing a clear link between norm geometry ($1, \infty$ vs. 2) and solver behavior.
2.  **Significant Practical Performance:** The experimental evaluation demonstrates substantial improvements over the baseline formulation. On the LowDim testbed, median solve times are reduced by up to 41x, and the number of solved instances increases by up to 63%. These are concrete, measurable benefits that validate the theoretical claims.
3.  **Reproducibility and Rigor:** The authors provide a reproducibility statement and make the code available under an MIT license. The paper includes detailed proofs in the appendix (Sections B and C), allowing for verification of the approximation factors and branching complexity claims.

### Weaknesses
1.  **ICLR Scope and Scalability:** While $k$-hyperplane clustering has ML applications, the instances tested are small ($m \le 30, n \le 5$). ICLR focuses heavily on scalable machine learning methods. The exact global optimization of combinatorial problems often does not scale to the dimensions typical of modern deep learning datasets (e.g., $n > 100$). The paper does not sufficiently discuss how this exact method fits into the broader context of approximate ML where scalability is prioritized over global optimality.
2.  **Dependence on Solver Heuristics for Theoretical Bounds:** Proposition 4 claims polynomial node counts ($k(n-1)$) but explicitly states "Assume that... branching takes place on the $u_{jh}$ variables first." Standard solvers like Gurobi employ general branching heuristics. The paper does not demonstrate empirically that Gurobi (without specific tuning instructions) actually follows this theoretical path, though the speedup suggests it. This creates a slight divergence between the tight theoretical bound and practical solver behavior.
3.  **Formatting Artifacts (Parser Issues):** Although I am instructed not to treat parser artifacts as weaknesses, the garbled mathematical notation (e.g., broken equations in Section 2 and Lemma proofs) makes it difficult to parse the precise definitions of the multi-norm constraints ($\| \cdot \|_{multi}$) without extensive reconstruction. This slightly lowers the clarity compared to standard accepted ICLR formatting.

### Novelty & Significance
The paper introduces a novel approach to strengthening global optimization formulations by intersecting constraints derived from different norm relaxations. The derivation of approximation factors between $k$-HC formulations under different norms (Theorem 1) and the specific application of these multi-norm constraints to improve SBB efficiency is technically novel. However, the significance is somewhat niche within the broader ICLR community; it addresses a specific exact optimization subroutine rather than proposing a new learning algorithm or representation. It is highly significant for the sub-field of global optimization in ML, but its impact on general deep learning research may be limited.

### Suggestions for Improvement
1.  **Contextualize Scalability:** Given the small instance sizes (HighDim only has up to $n=5$), the authors should explicitly discuss the limitations of exact optimization in ML. A discussion comparing this method to coreset-based or approximate clustering methods (referenced in the conclusion) regarding trade-offs between accuracy and computational cost for larger $n$ would strengthen the positioning.
2.  **Clarify Solver Behavior:** Provide data or arguments explaining why the default Gurobi SBB solver behaves in a way that aligns with the theoretical assumption (branching on $u_{jh}$ first). If the speedup comes from the constraints tightening the LP relaxation itself rather than the specific branching order, clarify this distinction.
3.  **Human-Readable Formatting:** Ensure the final submission renders the mathematical notation clearly without OCR/parser artifacts. The current garbled text (e.g., Lemma 3 proof) obscures the mathematical derivation, making it harder for reviewers to verify the tightness of the congruence inequalities.
4.  **Expand Application Discussion:** Briefly elaborate on specific ML scenarios (e.g., piecewise affine systems, signal processing) where this *exact* solution is critical compared to heuristics, to better justify the focus on global optimality over speed for large-scale problems.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Evaluate on real-world datasets (e.g., LiDAR, image line detection) mentioned in the Introduction, as synthetic data does not validate practical utility for ICLR.
2. Compare against standard heuristic $k$-plane clustering algorithms to quantify the trade-off between exactness and computational cost.
3. Test on larger instances ($m > 50$) to demonstrate scalability limits, as current $m \le 30$ is trivial for modern ML tasks.
4. Include instances with varying noise levels and outliers to assess robustness, as current data uses fixed Gaussian variance ranges.
5. Report results on instances where the heuristic fails to find the global optimum to highlight the specific value of the exact method.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze whether Gurobi's actual branching strategy adheres to Assumption 1 (midpoint branching), as the theoretical node count bounds depend on this specific behavior.
2. Ablate the symmetry-breaking constraints separately from the norm constraints to verify the speedup originates from the multi-norm formulation rather than symmetry handling.
3. Report the optimality gap of heuristic solutions on these instances to justify the need for global optimality in the first place.
4. Discuss the computational overhead of adding the auxiliary binary variables for the 1-norm constraints versus the benefit gained in bound tightness.
5. Analyze the sensitivity of the speedup to the dimension $n$, as the theoretical benefit scales with $n$ but experimental $n$ is very low ($n \le 5$).

### Visualizations & Case Studies
1. Plot dual bound progression over time to empirically verify the claim that nonzero lower bounds are achieved earlier with the strengthened formulation.
2. Visualize clustering results on real-world data to demonstrate that exact solutions yield meaningfully different structures than heuristics.
3. Show histograms of SBB tree sizes across instances to validate the theoretical linear vs. exponential node count distinction.
4. Provide a scatter plot of solve time vs. instance difficulty (e.g., separation between hyperplanes) to identify failure modes.
5. Include a visualization of the feasible region intersection in higher dimensions or via projection to illustrate the tightening effect of the multi-norm constraints.

### Obvious Next Steps
1. Provide a complexity analysis relative to $m$ (samples), not just $n$ and $k$, as $m$ is the primary scaling factor in ML.
2. Discuss integration with warm-starting from heuristic solutions to mitigate the cold-start cost of SBB for practical deployment.
3. Address the limitation of fixed $k$ by discussing strategies for model selection within the exact framework.
4. Investigate whether the multi-norm strengthening applies to other non-convex clustering objectives beyond $k$-hyperplane clustering.
5. Provide a guideline for practitioners on when to use this exact method versus heuristics based on instance characteristics.

# Final Consolidated Review
## Summary
This paper proposes a method to solve the k-Hyperplane Clustering problem with 2-norm distances (k-HC2) to global optimality via spatial branch-and-bound (SBB). The key innovation is strengthening the classical MI-QCQP formulation by intersecting it with constraints derived from polyhedral norms (p=1 and p=∞). The authors prove that these constraints change the complexity of obtaining a nonzero lower bound from exponential (2^{k(n-1)} nodes) to polynomial (O(kn) nodes) under specific branching assumptions, and demonstrate substantial empirical speedups (up to 41× median, 63% more instances solved).

## Strengths
- **Theoretical contribution establishing SBB complexity bounds:** Propositions 2, 3, and 4 provide rare concrete bounds on SBB tree size for a nonlinear integer program—specifically, proving that the 2-norm formulation requires exponential branching to obtain a nonzero lower bound, while the ∞-norm formulation achieves this in O(kn) nodes. This is a non-trivial result connecting norm geometry to branch-and-bound behavior.
- **Multi-norm relaxation with provably tighter approximation:** Lemma 4 and Corollary 2 show that combining 1-norm and ∞-norm constraints yields approximation factor 1/√n (better than the 1/n factor from individual polyhedral norms), which is a genuine theoretical improvement over using a single norm relaxation.
- **Substantial empirical improvements:** On the LowDim testbed, the strengthened formulations solve 21 instances that the baseline cannot solve within 46 hours, and achieve 28-41× median speedup on commonly-solved instances. The node count data (Table 5) confirms the theoretical prediction of reduced tree sizes.
- **Reproducibility:** Code is provided under MIT license, and all proofs are included in the appendix with sufficient detail for verification.

## Weaknesses
- **Instance scale far below typical ML applications:** All experiments use m ≤ 30 points in n ≤ 5 dimensions with k ≤ 5 hyperplanes. Real ML applications (LiDAR, image analysis, dictionary learning) typically involve hundreds or thousands of points in much higher dimensions. The paper does not demonstrate that the method scales to regime relevant to ML practice.
- **Multi-norm underperforms empirically despite better theory:** The multi-norm formulation has the best approximation factor (1/√n) but is dominated by simpler formulations on HighDim (5.6× speedup vs. 10-12× for single norms, and 37 instances solved vs. 40-41). This gap between theory and practice is acknowledged but not fully explained—likely due to the overhead of additional binary variables (s_jh and u_jh) disrupting solver heuristics.
- **No comparison with heuristic methods:** The paper cites k-means-based heuristics (Bradley & Mangasarian 2000) and distance-based reassignment heuristics (Amaldi & Coniglio 2013) but provides no comparison. For ICLR, understanding the trade-off between exact optimality and computational cost is essential—the value proposition of global optimality requires showing where heuristics fail.
- **Confounding symmetry-breaking differences:** The 2-norm and 1-norm formulations use w_j1 ≥ 0 for symmetry breaking, while the ∞-norm formulation uses a restricted one-sided disjunction (requiring dropping w_j1 ≥ 0). The multi-norm formulation inherits this weaker symmetry breaking. This difference is not isolated experimentally, making it unclear how much speedup comes from norm constraints versus symmetry handling.
- **Synthetic data only:** All instances are generated by placing points exactly on hyperplanes and adding Gaussian noise. This controlled setup may be substantially easier than real-world data with outliers, varying cluster densities, or adversarial structure.

## Nice-to-Haves
- Ablation experiments isolating symmetry-breaking from norm constraint effects.
- Comparison with heuristics to quantify when exact optimality matters.
- Dual bound progression plots to empirically validate the claim of earlier nonzero bounds.
- Discussion of how to choose among formulations given that the theoretically-best option (multi-norm) underperforms in practice.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- "Parser formatting artifacts obscure the math"—explicitly instructed to ignore parser issues as not paper's fault.
- "Gurobi may not follow Assumption 1 (midpoint branching)"—Propositions 2-4 establish lower bounds that remain valid under any branching strategy; the empirical speedups demonstrate practical relevance regardless of solver internals.
- "Proposition 4 uses 'nodes' ambiguously (depth vs. count)"—The appendix proof correctly describes a tree with "two nodes per level," and the practical results confirm polynomial behavior; this is a minor presentation issue.
- "Should compare to subspace clustering or deep learning methods"—Scope creep; the paper addresses exact k-HC2 optimization, not approximate clustering methods outside its stated contribution.

## Novel Insights
The geometric insight that polyhedral norm constraints induce MILP-representable disjunctions that "break" the origin-feasibility problem in SBB is elegant: the 2-norm ball's complement is a sphere (requiring exhaustive binary subdivision to exclude the origin), while the ∞-norm ball's complement is a union of half-spaces (detectable after O(kn) binary decisions). The practical observation that tighter theoretical bounds don't always yield faster solving—due to formulation overhead—provides a cautionary note for the global optimization community: approximation factor tightness is not the only metric for formulation quality.

## Suggestions
- Run experiments on at least one real-world dataset from the cited applications (e.g., LiDAR data for line detection) to demonstrate practical utility beyond synthetic instances.
- Add a comparison with the k-means heuristic showing where exact solutions differ meaningfully from heuristic ones, quantifying when global optimality is worth the computational cost.
- Provide an ablation that applies the same symmetry-breaking to all formulations to isolate the contribution of norm constraints from symmetry handling.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 4.0]
Average score: 5.3
Binary outcome: Accept
