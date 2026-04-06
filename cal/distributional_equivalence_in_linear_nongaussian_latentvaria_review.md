=== CALIBRATION EXAMPLE 79 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the paper's focus: characterizing distributional equivalence in linear non-Gaussian latent-variable cyclic models. The abstract clearly states the problem (lack of general equivalence characterization), the core contributions (graphical criterion, new edge rank tool, traversal procedure, and recovery algorithm), and the claim of being the first structural-assumption-free method. All claims appear supported by the paper's content. The mention of code and an interactive demo is a strength for reproducibility.

### Introduction & Motivation
The introduction is thorough and well-structured. It effectively motivates the challenge of latent-variable causal discovery without restrictive structural assumptions, positioning the lack of an equivalence characterization as a core obstacle. The historical context (FCI, parametric methods) and the gap in the literature (no equivalence characterization with latents in parametric settings) are clearly explained. The three research questions and corresponding contributions are stated precisely, setting clear expectations for the paper.

### Method / Approach
**Section 2 (Problem Setup):** The definitions of distributional equivalence and irreducibility are clear and necessary. Proposition 1 and 2 provide a clean way to rule out trivial unidentifiability. The presentation is rigorous.

**Section 3 (Graphical Tools):** This section is the theoretical core. The progression from distributional equivalence to mixing matrix closure (Lemma 1), to path ranks (Lemmas 2, 3), and finally to edge ranks (Definition 4, Lemma 4, Lemma 5) is logical. The introduction of "edge ranks" and the duality theorem (Theorem 1) is a significant conceptual contribution, reframing a global constraint (path rank) into a more local, combinatorial one. This duality is elegantly connected to matroid theory, filling a recognized gap in the rank-based toolbox.

**Potential Concern:** While the mathematical derivations are in the appendix, the main text relies heavily on concepts from matroid theory (transversal matroids, strict gammoids, duality). For a general ML audience, this might pose a significant barrier to understanding. The paper could benefit from more intuitive explanations alongside the formal definitions, even if brief.

**Section 4 (Graphical Characterization):** The main results (Theorems 2 and 3) are presented clearly. Theorem 2 provides an efficient, local graphical criterion for equivalence, which is a major improvement over checking all path ranks. Theorem 3 offers a transformational characterization (analogous to Meek's conjecture) that enables practical traversal of the equivalence class. The connection to classical results (e.g., reduction to Lacerda et al. (2008) when \(L=\emptyset\)) is helpful for context. The discussion of a potential "CPDAG-like" presentation (Theorem 4 in Appendix) is a thoughtful addition.

**Logical Gaps/Questions:** The proofs are deferred, so a high-level verification is needed.
1.  Lemma 3 states equivalence is characterized by path ranks under a vertex permutation. The proof relies on the identifiability of OICA and the duality in Theorem 1. This appears sound, but the critical step is the application of the Fundamental Lemma (Ingleton & Piff, 1973) to link isomorphic strict gammoids to equivalent mixing matrix closures. This is a non-trivial bridge between graph theory, matroid theory, and algebraic geometry. While the authors are experts in this area, the argument is highly compressed in the main text.
2.  The transformational characterization (Theorem 3) hinges on Lemmas 6 and 7. Lemma 7's criterion for admissible edge addition/deletion is derived from matroid-preserving column augmentation (Appendix B.3). The connection is made via Lemma 13. The overall logic is coherent but complex.

### Experiments & Results
**Section 5 (Algorithm and Evaluation):** The glvLiNG algorithm is clearly outlined in three steps: OICA estimation, rank realization, and class traversal. The constraint-based approach for rank realization (Phases 1 & 2) is ingenious, leveraging the local decomposition from Theorem 2 and the matroid construction lemmas (8-10).

**Evaluation:**
1.  *Equivalence Class Sizes (Table 3):* Provides a useful empirical sense of the inherent uncertainty. The numbers are large, underscoring the importance of characterizing the whole class.
2.  *Runtime (Table 4):* Demonstrates a massive speedup over a MILP baseline, validating the efficiency of the constraint-based design.
3.  *Benchmarking under Oracle (Table 5):* Shows that existing methods (LaHiCaSl, PO-LiNGAM) fail significantly under model misspecification, reinforcing the need for an assumption-free method.
4.  *Simulations (Figure 7):* The evaluation is comprehensive, varying graph size, density, latent count, and sample size. The results are honest: glvLiNG performs comparably or better on denser graphs and is more robust to latent dimensionality, while existing methods are better on very sparse graphs (where their assumptions are less violated). This is a fair assessment.
5.  *Real-World Data:* The application to stock returns is illustrative and yields plausible interpretations (banks as upstream causes, latent representing a conglomerate group). It demonstrates the method's applicability.

**Major Concern - OICA Dependency:** The authors openly acknowledge that glvLiNG's practical performance is tethered to the reliability of OICA estimation. While the SDP-ICA implementation is used, OICA is known to be challenging with finite samples, especially as dimension grows. The "Handling empirical ranks" subsection in Appendix D.4 describes a heuristic scoring and thresholding procedure to cope with noisy ranks. **This is a significant practical limitation.** The simulation results (Figure 7) show glvLiNG's performance improves with sample size, but it would be valuable to see a more direct ablation of OICA's performance (e.g., how often does it correctly identify the number of latents? How sensitive are the recovered ranks to estimation error?) under the studied conditions. The claim that glvLiNG is "structural-assumption-free" is true in principle, but in practice, the faithfulness of OICA estimation becomes a critical, non-trivial assumption.

**Missing Ablation:** An ablation study on the individual components of glvLiNG (e.g., the impact of the heuristic rank correction in Phase 1/2) would strengthen the empirical section.

### Writing & Clarity
The paper is generally well-written for a theoretical audience. The structure is logical, and the use of examples (Figures 1, 2, 3) and analogies (to CPDAGs, Meek conjecture) is very effective. However, as noted, the heavy use of matroid terminology (bases, circuits, cocircuits, coloops) in Sections 3, 4, and the Appendix, without much pedagogical scaffolding, will make the paper dense and difficult for readers not versed in this area. A short intuitive primer on the relevant matroid concepts (even in a footnote) would greatly improve accessibility.

### Limitations & Broader Impact
The limitation section correctly identifies the reliance on OICA as the main practical bottleneck and suggests promising future directions (OICA-free algorithms, extensions to Gaussian settings). The broader impact statement is appropriate for a foundational methodology paper. The societal impact is neutral, and the authors note the use of LLMs for writing polish.

### Overall Assessment
This paper makes a fundamental and novel contribution to causal discovery theory. It successfully provides the first graphical characterization of distributional equivalence for linear non-Gaussian models with arbitrary latent variables and cycles, a long-standing open problem. The introduction of "edge ranks" and the duality theorem are elegant and likely to influence future work beyond this specific setting. The proposed algorithm, glvLiNG, is a principled proof-of-concept that equivalence is recoverable without structural assumptions. The main weaknesses are practical: the algorithm's dependency on reliable OICA estimation and the inherent complexity of the equivalence classes (which is a property of the problem, not the method). The theoretical analysis is deep and appears sound, though the reliance on advanced matroid theory may limit the audience. For ICLR, which values significant theoretical advances, this paper is a strong candidate for acceptance, provided the authors can address the concerns about clarity and the practical limitations of OICA more thoroughly in a revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper provides the first characterization of distributional equivalence for linear non-Gaussian latent-variable cyclic causal models. It introduces edge rank constraints as a new graphical tool, establishes necessary and sufficient conditions for equivalence, and develops an algorithm (glvLiNG) to recover the equivalence class from data without imposing structural assumptions like pure measurement models or acyclicity.

### Strengths
1. **High Novelty**: This is the first equivalence characterization for latent-variable models in any parametric setting without structural assumptions, addressing a long-standing gap in causal discovery. The introduction of edge ranks as a dual to path ranks is a novel theoretical contribution.
2. **Theoretical Rigor and Completeness**: The paper provides a comprehensive theoretical analysis, including graphical criteria for equivalence (Theorem 2), a transformational characterization for traversing equivalence classes (Theorem 3), and proofs of correctness under faithfulness. The appendix contains detailed proofs and additional results.
3. **Practical Implementation and Evaluation**: The authors provide an open-source implementation and an interactive demo. The algorithm is evaluated on synthetic data, showing competitive performance, and applied to a real-world stock market dataset, yielding interpretable results.
4. **Clarity and Presentation**: The paper is well-structured, with clear motivations, definitions, and illustrative examples. The analogy to classical equivalence results (e.g., CPDAGs, Meek rules) helps contextualize the contributions.

### Weaknesses
1. **Reliance on OICA**: The glvLiNG algorithm depends on overcomplete independent component analysis (OICA), which is known to be statistically and computationally challenging in practice. While the paper acknowledges this and suggests future improvements, it remains a significant limitation for immediate practical application.
2. **Limited Empirical Depth**: The experimental evaluation, while demonstrating feasibility, is somewhat limited. The real-data analysis is brief and more illustrative than conclusive. Simulation comparisons show mixed results, and the benefits of being assumption-free are not overwhelmingly demonstrated against baselines that do make assumptions.
3. **Scalability and Interpretability of Equivalence Classes**: The equivalence classes can be extremely large (e.g., 19,008 graphs in the stock example), which may overwhelm practitioners. The paper does not deeply address how to navigate or interpret such large classes in practice, beyond presenting the maximal graph.
4. **Faithfulness Assumption**: Theoretical guarantees require a faithfulness assumption (no coincidental rank cancellations). While standard, the algorithm's robustness to violations is not thoroughly tested, and real data may deviate from this genericity condition.

### Novelty & Significance
The work is highly novel, providing the first general equivalence characterization for latent-variable causal models without structural restrictions. The introduction of edge ranks enriches the rank-based causal discovery toolbox. The significance lies in enabling a principled, assumption-free approach to latent-variable discovery, a fundamental challenge. The results lay a foundation for extensions to other parametric families (e.g., linear Gaussian) and inspire new algorithms.

### Suggestions for Improvement
1. **Mitigate OICA Dependence**: Explore and integrate more robust or efficient alternatives to OICA for rank estimation, such as methods based on cumulants or regression, to enhance practical applicability.
2. **Deepen Empirical Evaluation**: Conduct more extensive simulations under varied conditions (e.g., stronger confounding, non-faithfulness) and include additional real-world benchmarks to better illustrate the advantages and limitations of the assumption-free approach.
3. **Develop Practical Summaries of Equivalence Classes**: Investigate methods to present large equivalence classes more intuitively, perhaps by extracting common invariant features or incorporating user-specified constraints (e.g., sparsity, known ancestral relations) to reduce ambiguity.
4. **Discuss Incorporation of Prior Knowledge**: Extend the discussion on how domain knowledge (e.g., known causal orders, absence of cycles) could be integrated to shrink equivalence classes and improve interpretability, potentially as a future work direction.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against constraint-based methods that allow cycles and latents.** The paper only compares against LaHiCaSl and PO-LiNGAM, which have restrictive structural assumptions. A direct comparison against FCI or its cyclic extensions (e.g., Cyclic Causal Discovery algorithms) is missing. Without this, the practical advantage of the proposed method over existing general-purpose constraint-based methods is unclear.
2. **Quantify recovery of the true equivalence class.** The paper does not measure whether the algorithm's output actually contains the true graph (or its irreducible form). There should be an experiment reporting the proportion of simulations where the true graph is within the recovered equivalence class.
3. **Evaluate on real-world benchmarks with known latent structure.** The stock market analysis is purely exploratory. To demonstrate real-world utility, the method should be tested on datasets where latent variables are known (e.g., from domain knowledge or simulated benchmarks), and the recovered latent relations should be evaluated.
4. **Test robustness to model violations.** The method assumes linearity and exact non-Gaussianity. No experiments assess performance under mild nonlinearities, near-Gaussian noise, or measurement error, which are critical for real-world applicability.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze the practical informativeness of the equivalence classes.** For realistic graph sizes (e.g., 10-20 variables), how large are the equivalence classes on average? If classes are enormous, the method's output is uninformative. The paper only enumerates tiny graphs; a scalability analysis is needed.
2. **Investigate identifiability of causal effects.** Even if the graph is identified only up to equivalence, can total causal effects between observed variables be uniquely determined? This is crucial for applications but not discussed.
3. **Analyze sensitivity to OICA estimation errors.** The algorithm hinges on OICA to estimate the mixing matrix. There is no analysis of how errors in OICA (e.g., from finite samples or algorithmic limitations) propagate to errors in graph recovery. A sensitivity study is required.

### Visualizations & Case Studies
1. **Visualize the equivalence class for a moderate-sized graph.** The paper shows tiny examples (e.g., 3 observed, 2 latent). A compelling case study would visualize the equivalence class (e.g., as a Hasse diagram or a set of graphs) for a larger simulated graph (e.g., 5 latents, 10 observed) to illustrate the remaining uncertainty.
2. **Demonstrate how the recovered class converges with sample size.** Plot the size or structural accuracy of the recovered equivalence class as sample size increases, to show statistical consistency and sample efficiency in practice.
3. **Illustrate a typical failure mode.** Show a scenario where OICA fails or faithfulness is violated, and illustrate how the algorithm breaks down. This would help users understand the limitations.

### Obvious Next Steps
1. **Integrate more robust rank estimation techniques.** Instead of relying solely on OICA, incorporate recent methods for testing rank constraints directly (e.g., from linear Gaussian literature) to improve efficiency and robustness.
2. **Develop and implement a compact representation of the equivalence class.** The paper mentions a "presentation" (Appendix C.3) but does not implement it in the algorithm or evaluation. Providing a graphical representation (like a PAG for latent variables) would make the output interpretable.
3. **Handle unknown number of latents automatically.** The current algorithm requires the number of latents as input (from OICA). A complete method should include a principled model selection procedure to determine the number of latents from data.

# Final Consolidated Review
## Summary
This paper provides the first graphical characterization of distributional equivalence for linear non-Gaussian latent-variable causal models, allowing arbitrary latent structure and cycles. It introduces a new tool, edge ranks, which are dual to path ranks, leading to a local graphical criterion for equivalence and a transformational characterization for traversing equivalence classes. The authors also develop an algorithm, glvLiNG, to recover the equivalence class from data without imposing common structural assumptions like pure measurement models or acyclicity.

## Strengths
- **First general equivalence characterization for latent-variable models.** The paper solves a long-standing open problem by providing necessary and sufficient conditions for when two linear non-Gaussian models with arbitrary latents and cycles are observationally indistinguishable, without relying on structural assumptions common in prior work.
- **Introduction of edge ranks and their duality.** The novel concept of edge ranks, and the duality theorem linking them to the well-known path ranks, provides a more local and manipulable tool. This not only enables the paper's main results but also enriches the broader rank-based causal discovery toolbox.
- **Complete theoretical framework with practical algorithm.** The work includes a graphical criterion (Theorem 2), a transformational characterization for class traversal (Theorem 3), and a constraint-based algorithm (glvLiNG) that is guaranteed to recover the correct equivalence class under faithfulness, with demonstrated computational efficiency over a naive baseline.

## Weaknesses
- **Practical performance hinges on reliable OICA estimation.** The glvLiNG algorithm requires estimating a mixing matrix via overcomplete independent component analysis (OICA), which is statistically and computationally challenging with finite samples. While the paper uses heuristics to handle noisy ranks, this dependency is a significant practical bottleneck, as acknowledged in Section 5 and the limitations.
- **Missing comparison to general constraint-based baselines.** The empirical evaluation compares against methods (LaHiCaSl, PO-LiNGAM) that rely on specific structural assumptions. A comparison to general constraint-based methods that allow cycles and latents (e.g., FCI or its cyclic extensions) is absent, making the practical advantage of being "structural-assumption-free" less clear against the most general available alternatives.

## Nice-to-Haves
- A more extensive analysis of the practical informativeness of equivalence classes for moderate-sized graphs, including how often the true graph is contained within the recovered class.
- Investigation into the identifiability of causal effects between observed variables within an equivalence class, which is crucial for many applications.

## Novel Insights
The core novel insight is the introduction and utilization of edge rank constraints. This tool provides a local, combinatorial perspective (via bipartite matchings) that is dual to the global, algebraic perspective of path ranks (via vertex-disjoint paths). This duality, rooted in matroid theory, allows the complex problem of distributional equivalence to be decomposed into local checks on children's bases, leading to an efficient graphical criterion and a clean transformational characterization via admissible edge additions/deletions and cycle reversals.

## Suggestions
- Implement and evaluate the proposed compact presentation of the equivalence class (mentioned in Appendix C.3) within the glvLiNG algorithm to make its output more interpretable for practitioners facing large equivalence classes.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
