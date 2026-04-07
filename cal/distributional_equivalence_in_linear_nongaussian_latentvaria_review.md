=== CALIBRATION EXAMPLE 84 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**  
The title accurately reflects the paper’s scope and contributions. The abstract clearly states the core problem (lack of equivalence characterization for latent-variable models) and summarizes the key results: a graphical criterion for distributional equivalence in linear non-Gaussian models with arbitrary latents and cycles, the introduction of edge rank constraints, a transformational characterization, and a structural-assumption-free discovery algorithm. The claim that this is the first such characterization in any parametric setting without structural assumptions is bold and, if correct, represents a significant advance.

**Introduction & Motivation**  
The introduction thoroughly motivates the problem, highlighting the limitations of existing methods that rely on strong structural assumptions (e.g., pure children, hierarchical latents, acyclicity). It convincingly argues that the absence of an equivalence characterization has hindered the development of general methods. The contributions are clearly listed and align with the paper’s claims. The related work discussion is comprehensive, though a more explicit comparison to the closest work (Adams et al., 2021) would strengthen the narrative.

**Problem Setup (Section 2)**  
The definitions of linear non-Gaussian models, latent-variable models, and distributional equivalence are precise and standard. The introduction of irreducibility (Proposition 1) as a canonicalization to eliminate trivial unidentifiable latents is well-justified. Proposition 2 provides a constructive reduction procedure. One minor concern: the condition for irreducibility (every non-empty set of latents has at least two children outside) is intuitive, but its necessity and sufficiency rely on OICA identifiability; the proof is deferred to the appendix, which is acceptable.

**Developing Graphical Tools (Section 3)**  
This section builds the theoretical machinery. Lemma 3 reduces distributional equivalence to equality of path ranks up to permutation—a known but powerful connection. The discussion on the complexity of manipulating path ranks (Section 3.2) motivates the need for a more local tool. The introduction of edge ranks (Definition 4) and their duality with path ranks (Theorem 1) is a key conceptual contribution, bridging graph theory and matroid theory. Lemma 5 rephrases equivalence in terms of edge ranks. The exposition is clear, though the technical definitions (matching rank, support matrix) require careful reading. The duality theorem is stated without proof in the main text; its correctness is critical to subsequent results, but the proof is provided in the appendix.

**Graphical Characterization (Section 4)**  
Theorem 2 provides a tractable graphical criterion: equivalence reduces to checking that the “children bases” for the latent set \(L\) and for each \(L \cup \{X_i\}\) coincide up to permutation. This is a major simplification over checking all subsets of observed variables. However, computing bases\(_G(Y)\) for a set \(Y\) may still require enumerating all perfect matchings from \(Y\), which could be exponential. The paper does not discuss the computational complexity of this check explicitly, though it might be feasible via matroid isomorphism tests.  

Theorem 3 gives a transformational characterization (admissible cycle reversals and edge additions/deletions), analogous to the Meek conjecture for Markov equivalence. This provides a practical way to traverse the equivalence class. Lemma 7 details the condition for edge addition/deletion, which involves an edge rank inequality that is local to the column of the source vertex. While local, it still requires solving bipartite matching problems. The presentation of the equivalence class (Theorem 4, in appendix) is a valuable addition, offering a compact representation akin to a CPDAG.

**Algorithm and Evaluation (Section 5)**  
The glvLiNG algorithm leverages OICA to estimate a mixing matrix, then constructs a digraph satisfying the observed rank constraints, and finally traverses the equivalence class. The rank realization step is decomposed into two phases: recovering latent-to-all edges (a bipartite realization problem) and recovering observed-to-all edges (via independent column augmentations). This design is clever and exploits the local decomposition from Theorem 2.  

The evaluation is extensive:  
1. Quantifying equivalence class sizes (Table 3) gives insight into inherent uncertainty.  
2. Runtime comparisons (Table 4) show glvLiNG is significantly faster than an MILP baseline.  
3. Benchmarking under oracle inputs (Table 5) reveals that existing methods (LaHiCaSl, PO-LiNGAM) suffer under model misspecification.  
4. Finite-sample simulations (Figure 7) demonstrate that glvLiNG performs particularly well on denser graphs and is more robust to latent dimensionality, while baselines excel on sparser graphs.  
5. The real-world stock market analysis yields plausible interpretations.  

**Major concerns:**  
- The algorithm relies heavily on OICA, which is notoriously unstable and computationally demanding in practice. The authors acknowledge this and treat glvLiNG as a proof-of-concept, but it limits immediate applicability.  
- The rank realization step assumes accurate rank queries from OICA. In practice, ranks must be estimated from finite data (e.g., via singular value thresholds), which introduces noise. The appendix describes a heuristic using confidence scores, but its robustness is not thoroughly evaluated.  
- The scalability of the overall approach is not tested beyond small graphs (up to 13 variables). Given the combinatorial nature of equivalence classes and OICA’s poor scaling, practical use on larger problems may be challenging.

**Writing & Clarity**  
The paper is well-organized and clearly written, though necessarily dense due to the technical material. Definitions are carefully stated, and examples/figures aid understanding. Some sections (e.g., the matroid-based algorithm details in Appendix A) are highly technical and may be difficult for a general audience, but the main ideas are accessible. The interactive demo is a nice supplement.

**Limitations & Broader Impact**  
The conclusion appropriately discusses limitations: reliance on OICA, linearity and non-Gaussianity assumptions, and the faithfulness assumption. The authors outline promising future directions (OICA-free algorithms, extension to linear Gaussian systems). There are no apparent ethical concerns. A broader limitation is the inherent uncertainty when equivalence classes are large, which may limit practical utility without additional constraints (e.g., sparsity, interventional data).

### Overall Assessment
This paper makes a substantial theoretical contribution to causal discovery by providing the first complete characterization of distributional equivalence for linear non-Gaussian latent-variable models with cycles. The introduction of edge ranks and their duality with path ranks is a novel tool that enriches the rank-based framework. The graphical criterion and transformational characterization are elegant and provide new insights. The proposed algorithm, while limited by its dependence on OICA, demonstrates that the equivalence class is recoverable in principle without structural assumptions. The work is technically sound, thoroughly evaluated, and well-presented. It meets ICLR’s standards for novelty, significance, and rigor. Despite practical limitations, the theoretical advances are likely to influence future research. I recommend acceptance, provided the authors address minor concerns (e.g., clarifying computational complexity of checking Theorem 2, discussing robustness to rank estimation errors more explicitly).

# Neutral Reviewer
## Balanced Review

### Summary
This paper establishes the first graphical characterization of distributional equivalence for linear non-Gaussian (LiNG) causal models with arbitrary latent variables and cycles. It introduces a novel tool called "edge rank constraints," which dualizes traditional path rank constraints and enables a local, graph-theoretic criterion for equivalence. Based on this characterization, the authors develop a constraint-based algorithm (glvLiNG) to recover the entire equivalence class from data without imposing structural assumptions like pure children, acyclicity, or hierarchical latents.

### Strengths
1. **Fundamental theoretical contribution**: The paper solves a long-standing open problem—characterizing distributional equivalence in latent-variable cyclic models—which is a prerequisite for designing general, assumption-free discovery methods. The results fill a critical gap in causal discovery theory.
2. **Novel tool (edge rank constraints)**: The introduction of edge ranks and their duality with path ranks (Theorem 1) provides a new perspective in rank-based causal discovery. This tool simplifies derivations and may have broader applications beyond the current setting.
3. **Comprehensive characterization**: The work offers multiple interconnected results: a graphical criterion for equivalence (Theorem 2), a transformational characterization for traversing the equivalence class (Theorem 3), and an explicit reduction to irreducible forms (Proposition 2). Together, these provide a complete picture analogous to classical equivalence results (e.g., CPDAGs and Meek rules).
4. **Practical algorithm and evaluation**: The glvLiNG algorithm translates theory into practice, and the authors evaluate it thoroughly: quantifying equivalence class sizes, benchmarking runtime against MILP, comparing with existing methods under oracle and finite-sample settings, and applying it to real stock-market data. The interactive demo enhances reproducibility and understanding.
5. **Clarity and structure**: Despite technical depth, the paper is well-organized, with clear definitions, lemmas, and examples. The appendix provides detailed proofs and additional discussions.

### Weaknesses
1. **Reliance on OICA**: glvLiNG depends on overcomplete ICA to estimate the mixing matrix, which is notoriously challenging in practice, especially with many latents. While the authors acknowledge this and suggest future improvements, the current algorithm’s robustness and scalability may be limited in high dimensions.
2. **High technical complexity**: The heavy use of matroid theory and combinatorial arguments may hinder accessibility for a broader audience. Key concepts like edge ranks and transversal matroids are introduced quickly, and the proofs are highly condensed.
3. **Limited real-world validation**: The real-data experiment uses only one dataset (14 stocks) with speculative interpretation of latents. More extensive validation on diverse benchmarks (e.g., biological networks) would strengthen practical impact.
4. **Restricted to linear non-Gaussian setting**: The results are specific to LiNG models; extension to other parametric families (e.g., linear Gaussian, discrete) is non-trivial and left as future work. The discussion in the appendix is preliminary.
5. **Computational scalability**: Although glvLiNG scales better than MILP, traversing equivalence classes via BFS/DFS could become prohibitive for large graphs (beyond ~10 vertices). The paper does not deeply analyze worst-case complexity or suggest approximation heuristics.

### Novelty & Significance
The work is highly novel: it provides the first equivalence characterization for latent-variable cyclic models in any parametric setting without structural assumptions. The introduction of edge rank constraints is an original contribution that enriches the rank-based toolbox. The significance lies in laying a foundation for assumption-free causal discovery, potentially influencing future method design and theoretical developments. The paper meets ICLR’s standards for novelty, technical rigor, and potential impact.

### Suggestions for Improvement
1. **Mitigate OICA dependency**: Explore integrating more robust OICA estimators or, as hinted, design OICA-free variants that use partial rank information (e.g., via cumulant-based tests). A discussion of practical OICA choices or hyperparameter tuning would help users.
2. **Improve accessibility**: Add an intuitive tutorial-style example early on to illustrate edge ranks and their duality. Provide a high-level proof sketch in the main text to guide readers through the technical machinery.
3. **Extend empirical validation**: Apply glvLiNG to more real-world benchmarks (e.g., gene regulatory networks with known latent confounders) and report quantitative metrics where possible. Compare with more baselines (e.g., FCI, RFCI) in simulations.
4. **Discuss scalability and approximations**: Analyze the computational complexity of equivalence class traversal and suggest pruning strategies (e.g., using identifiable ancestral relations among observed variables) for larger graphs.
5. **Clarify limitations and extensions**: Expand the discussion on extending to linear Gaussian and discrete settings, outlining concrete challenges (e.g., trek separation, tensor ranks) and potential pathways. This could motivate follow-up work.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No synthetic evaluation of glvLiNG's ability to recover the true equivalence class from data.** The paper only benchmarks baselines under oracle inputs and shows runtime. To validate the core claim of a "structural-assumption-free discovery method," experiments must show that glvLiNG, given finite samples, outputs an equivalence class that contains (or is close to) the true model. Metrics like precision/recall for invariant edges or coverage of the true equivalence class are needed.
2. **No sensitivity analysis to violations of faithfulness or non-Gaussianity.** The algorithm relies on OICA and rank constraints under faithfulness (Assumption 1). Without experiments testing robustness to near-faithfulness violations, different noise distributions (e.g., near-Gaussian), or small sample sizes, the practical utility is unclear.
3. **No ablation study on glvLiNG's components.** The algorithm has multiple steps (irreducibility reduction, Phase 1/2, traversal). Ablations are needed to show which parts are critical for performance and to diagnose failure modes.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of edge ranks versus path ranks for estimation and identifiability.** The paper introduces edge ranks as a new tool but does not analyze whether they are easier to estimate from data or lead to tighter equivalence classes compared to path ranks. This analysis is needed to justify their utility beyond theoretical convenience.
2. **Scaling analysis of equivalence class sizes.** Table 3 counts classes for small graphs, but no analysis is given on how class size grows with graph parameters (e.g., number of latents, cycles, density). This is crucial for understanding the remaining uncertainty after applying the method.
3. **Sensitivity analysis to OICA estimation errors.** The algorithm's first step is OICA, which is known to be unstable. Without an analysis of how errors in the estimated mixing matrix propagate to the recovered equivalence class, the method's practical reliability is in doubt.

### Visualizations & Case Studies
1. **Visual case studies of glvLiNG recovering equivalence classes from simulated data.** The paper shows example equivalence classes but not how glvLiNG recovers them from data. Visualizing the true versus recovered class (e.g., showing invariant edges and uncertainties) would demonstrate whether the method works as claimed.
2. **Real-world case studies with known latent variables.** The stock market example lacks validation. A case study on data where latent variables are known (e.g., from domain knowledge) would help assess whether the recovered latents and edges are meaningful.

### Obvious Next Steps
1. **Benchmark glvLiNG against baselines on synthetic data with known equivalence classes.** This is a standard evaluation for causal discovery methods and is necessary to substantiate the paper's claims.
2. **Develop or integrate rank tests that bypass OICA.** Given OICA's inefficiency, the paper should have explored using direct rank tests on data (e.g., via non-Gaussianity) to estimate the required constraints, making the method more practical.
3. **Provide statistical tests for edge ranks.** Since edge ranks are central to the theory, a natural next step is to develop hypothesis tests for edge rank constraints from data, which would enable a more robust constraint-based implementation.

# Final Consolidated Review
## Summary
This paper establishes the first graphical characterization of distributional equivalence for linear non-Gaussian latent-variable causal models with arbitrary cycles. It introduces edge rank constraints, proves their duality with path ranks, and uses them to derive a local criterion for equivalence. Based on this, the authors provide a transformational characterization for traversing equivalence classes and develop an algorithm (glvLiNG) to recover such classes from data without structural assumptions.

## Strengths
- **Fundamental theoretical advance:** Solves a long-standing open problem by characterizing distributional equivalence in a general latent-variable cyclic setting, which is a prerequisite for assumption-free causal discovery.
- **Novel tool (edge ranks):** Introduces edge rank constraints and their duality with path ranks (Theorem 1), enriching the rank-based toolbox and enabling simpler derivations.
- **Comprehensive characterization:** Offers multiple interconnected results: a graphical criterion for equivalence (Theorem 2), a transformational characterization for traversal (Theorem 3), and a procedure for reduction to irreducible forms (Proposition 2), analogous to classical equivalence theory.
- **Algorithmic proof-of-concept:** Develops glvLiNG, an algorithm that demonstrates recoverability of the equivalence class from data, supported by extensive evaluation including runtime comparisons, benchmarking under oracle and finite-sample settings, and a real-world application.

## Weaknesses
- **Dependence on OICA:** The algorithm relies on overcomplete ICA to estimate mixing matrices, which is notoriously unstable and computationally demanding in practice. While acknowledged as a limitation, this hinders immediate practical utility and scalability.
- **Computational complexity unanalyzed:** The paper does not discuss the computational complexity of checking the graphical criterion (Theorem 2) or traversing equivalence classes for larger graphs, leaving scalability concerns for real-world applications.
- **Insufficient robustness evaluation:** Robustness to estimation errors in ranks (e.g., from finite samples) and violations of faithfulness or non-Gaussianity is not thoroughly tested; the heuristic confidence scores in Appendix D.4 lack comprehensive validation.
- **Limited empirical validation:** Only one real-world dataset is used, and synthetic evaluations could more directly measure recovery of the true equivalence class (e.g., coverage metrics for invariant edges) rather than relying solely on structural Hamming distance to a closest graph.

## Nice-to-Haves
- Extend the characterization to linear Gaussian or discrete parametric settings, as preliminarily discussed in the appendix.
- Provide intuitive tutorials or examples to improve accessibility for a broader audience, given the technical depth.
- Conduct ablation studies to understand the contribution of each algorithm component (e.g., irreducibility reduction, phase 1 vs. phase 2).
- Analyze how equivalence class sizes scale with graph parameters like number of latents, cycles, or density.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- Criticisms that the paper is restricted to linear non-Gaussian models are scope creep; the paper explicitly focuses on this setting as a necessary first step.
- Comments on high technical complexity are stylistic and not a substantive flaw in the work itself.
- Claims of "missing experiments" are partially addressed by existing simulations in Section D.4, where glvLiNG is evaluated against baselines with equivalence-aware metrics.

## Novel Insights
The introduction of edge ranks provides a dual perspective to path ranks, enabling a local decomposition for equivalence that reduces global rank checks to per-variable conditions. This insight—that equivalence can be determined by examining children bases for the latent set and each observed variable individually—likely extends beyond the current setting to other rank-based causal discovery problems.

## Suggestions
- Develop or integrate more robust rank estimation methods that bypass OICA's instability, such as direct cumulant-based tests or partial rank queries.
- Include explicit complexity analysis of the equivalence checking (Theorem 2) and traversal procedures (Theorem 3) to inform practical use.
- Expand real-world validation to domains with known latent structure (e.g., biological networks with confirmed confounders) to strengthen practical impact.
- In simulations, report metrics that directly assess equivalence class recovery, such as the fraction of true invariant edges identified or the precision/recall for edges that must appear in all equivalent graphs.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
