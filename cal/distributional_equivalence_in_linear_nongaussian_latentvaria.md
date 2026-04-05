=== CALIBRATION EXAMPLE 90 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is accurate and fairly specific: it signals both the main theoretical object (“distributional equivalence”) and the model class (linear non-Gaussian latent-variable cyclic causal models).
- The abstract clearly states the motivation, the central gap, and the main deliverables: a graphical equivalence characterization, a new “edge rank” tool, a traversal procedure, and a learning algorithm.
- The strongest claim in the abstract is that this is “the first structural-assumption-free discovery method” and “the first equivalence characterization with latent variables in any parametric setting without structural assumptions.” Given the breadth of latent-variable causal discovery literature, this is a very high bar and would need extremely careful qualification. The paper does argue novelty, but the abstract’s wording is stronger than what is fully established in the main text, especially because the algorithm still relies on OICA and a faithfulness/genericity assumption. So the claim is directionally supported but somewhat over-broad.

### Introduction & Motivation
- The problem is well-motivated. The introduction correctly identifies that most latent-variable causal discovery methods rely on structural assumptions and that cycles plus latents make equivalence much harder.
- The gap in prior work is stated clearly: there is no known general equivalence characterization for latent-variable linear non-Gaussian cyclic models.
- The contributions are stated cleanly, and the paper does a good job of positioning itself relative to FCI/MAG/CPDAG-style equivalence frameworks.
- That said, the introduction slightly over-sells “structural-assumption-free” discovery. The paper’s main theorem is equivalence characterization, while the algorithmic result depends on an oracle OICA or an estimation pipeline with additional practical heuristics. So the introduction should more carefully distinguish the theorem from the implemented method.

### Method / Approach
- The overall method is ambitious and potentially significant: it defines distributional equivalence, reduces trivial cases via irreducibility, derives graphical conditions via path ranks, introduces edge ranks and a duality result, and then derives a transformational characterization plus an algorithm.
- The high-level structure is coherent, but several logical steps need stronger justification or clearer separation between theorem and proof sketch.
- A major concern is the reliance on OICA identifiability and “faithfulness” assumptions in the learning pipeline. The paper claims a structural-assumption-free method, but the algorithmic recovery guarantee is not assumption-free in the usual sense; it depends on oracle access to rank information via OICA and on genericity assumptions in the mixing matrix.
- The use of Zariski closure/genericity in the equivalence argument is plausible, but the paper needs to be clearer about exactly which claims are generic and which are exact. The text says pathological singular loci “do not affect our results,” but this needs to be stated with precision, especially because distributional equivalence is a statement about the full observed distribution set.
- The proof architecture is sophisticated, but some key statements seem stronger than the exposition justifies. For example, Theorem 1’s duality between path ranks and edge ranks is central, yet the paper gives limited intuition about why this duality should hold in the stated generality for arbitrary digraphs with overlaps between \(Z\) and \(Y\).
- Failure modes are not discussed enough. The algorithm’s practical step of thresholding empirical singular values to infer matroid structure is heuristic; the paper should be more explicit about when this can fail and how sensitive the downstream graph reconstruction is to rank misestimation.

### Experiments & Results
- The experiments do test the main claims reasonably well:
  - Table 3 quantifies equivalence-class sizes, supporting the claim that latent-variable equivalence can be large.
  - Table 4 evaluates runtime of the proposed graph construction procedure.
  - Table 5 tests existing methods under oracle inputs and shows brittleness under model misspecification.
  - The simulation section compares glvLiNG against relevant latent-variable baselines.
  - The stock-market case study demonstrates a plausible application.
- The baselines are relevant, but the comparison is not completely balanced. For example, Table 5 gives PO-LiNGAM and LaHiCaSl oracle tests, but then evaluates against the ground truth “best possible result” over the equivalence class, which is favorable to the baselines in one sense but also not the most standard comparison protocol. More importantly, the baselines are methods with strong assumptions that may not be designed for the general model class of this paper, so the results should be interpreted as robustness-to-misspecification rather than direct superiority.
- A key missing ablation is the effect of the rank-recovery heuristics in glvLiNG. The paper emphasizes OICA as an oracle-like first step, but the actual finite-sample procedure uses score thresholds and matroid approximation heuristics. It would be materially helpful to isolate the impact of:
  1. OICA estimation quality,
  2. the matroid realization step, and
  3. the traversal/classification step.
- Error bars are reported in some tables and figures, which is good. However, I did not see statistical significance analyses for the main claims, and the simulation results would benefit from more explicit variance characterization, especially when comparing multiple random model settings.
- The datasets and metrics are mostly appropriate. Using SHD against the best graph in the true equivalence class is sensible for this setting. The stock-price application is illustrative, but it is more of a qualitative demonstration than a rigorous external validation.

### Writing & Clarity
- The paper is generally well organized, but several parts are quite difficult to follow, even for a technical audience. This is especially true in the derivation chain from mixing-matrix equivalence to path-rank equivalence to edge-rank equivalence.
- The most serious clarity issue is that some of the key results are stated in ways that are conceptually dense and hard to parse on first reading, especially Theorems 1–3 and the matroid-based Appendix A/B arguments.
- Figures and tables are conceptually useful, especially Figures 2–3 and the summary tables. Table 1 does a good job of contrasting path ranks and edge ranks. The equivalence-class diagrams are helpful.
- The paper would benefit from more intuitive explanation of:
  - why edge ranks are the “right” dual notion,
  - how the support-matrix/matching-rank viewpoint maps back to causal graphs,
  - and why the singleton decomposition in Lemma 9 is valid.
- The appendix is extensive and seems to contain many of the needed proofs, but because the main text leans heavily on these results, the paper’s readability depends on the appendix more than ideal for an ICLR submission.

### Limitations & Broader Impact
- The limitations section acknowledges the use of OICA and suggests future OICA-free work, which is good but incomplete.
- A more substantive limitation is that the learning algorithm is still very much a proof-of-concept rather than a fully practical method. It depends on estimated rank structure and matroid realization procedures that may be brittle in realistic noisy settings.
- The paper also does not sufficiently discuss when linear non-Gaussian assumptions are likely to fail in practice, nor the consequences of misspecification for the equivalence characterization.
- On broader impact, the paper states no ethical concerns. That is acceptable, though in causal discovery there is a general risk of over-interpreting learned graphs in high-stakes domains. This could be briefly acknowledged, especially given the real-data stock-market example and the paper’s claims of practical applicability.

### Overall Assessment
This is a technically ambitious and potentially important paper. Its strongest contribution is the attempt to characterize distributional equivalence for a very general class of latent-variable linear non-Gaussian cyclic models, and the introduction of edge ranks as a dual tool is novel and potentially useful. That said, the paper’s practical learning claim is weaker than its rhetoric suggests: the algorithm still depends on OICA and genericity/faithfulness assumptions, and the empirical evaluation does not fully isolate how robust the proposed procedure is to finite-sample rank-estimation errors. For ICLR, I would regard the theory as the main contribution and the algorithm as a promising proof-of-concept. The work likely stands as a meaningful advance, but the authors should tighten the distinction between exact equivalence theory and approximate learning, and be more careful about “structural-assumption-free” claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies distributional equivalence for linear non-Gaussian causal models with arbitrary latent variables and cycles. Its main claim is a graphical characterization of equivalence, a transformational characterization for traversing an equivalence class, and an algorithm (glvLiNG) that reconstructs models up to equivalence from data using overcomplete ICA plus matroid-style rank reasoning.

### Strengths
1. **Addresses an important and underexplored problem.**  
   The paper tackles latent-variable causal discovery without imposing common structural assumptions such as pure children, hierarchy, bow-freeness, or acyclicity. This is a meaningful direction for ICLR, since the work aims at a more general identifiability theory for causal discovery.

2. **Claims a strong theoretical contribution.**  
   The paper proposes an equivalence characterization that is intended to be valid for arbitrary latent structure and cycles in the linear non-Gaussian setting, and then derives a traversal characterization and a presentation of equivalence classes. If correct, this would be a substantial advance over prior work focused on restricted graph families.

3. **Connects causal discovery with graph/matroid tools in a novel way.**  
   The introduction of “edge ranks” as a complement to path ranks is conceptually interesting. The duality between path ranks and edge ranks, and the use of transversal matroids, suggests a potentially useful bridge between causal graphical models and combinatorial optimization.

4. **Provides an algorithmic pipeline, not only theory.**  
   glvLiNG is presented as a constructive method that leverages OICA estimates and then reconstructs a graph/equivalence class via rank constraints. The paper also includes runtime comparisons and simulation experiments, which is important for ICLR-style evaluation.

5. **Attempts to go beyond mere identifiability into equivalence-class traversal.**  
   The transformational characterization and the claim of an equivalence-class presentation are valuable because ICLR typically rewards methods that provide actionable structure, not just existence proofs.

### Weaknesses
1. **The presentation is extremely dense and difficult to verify from the main text.**  
   Although the paper is ambitious, many core claims are introduced with heavy notation, long proof sketches, and multiple reductions. For an ICLR audience, the main results are hard to independently parse, and the paper often requires jumping between sections to understand what is actually being proven. This hurts clarity and makes it difficult to assess correctness.

2. **The algorithmic contribution appears to depend on strong oracle-like assumptions.**  
   glvLiNG relies on OICA, and the paper explicitly states the algorithm is proved under “access to an oracle OICA” and faithfulness/genericity assumptions. This makes the practical impact less clear, because the method’s strongest guarantees are not from a fully data-driven, finite-sample procedure but from an idealized estimation pipeline.

3. **Finite-sample robustness and real-world validation are limited.**  
   The simulations compare against two baselines, but the evaluation does not establish broad robustness across realistic misspecification, noise, or difficult latent structures. The real-data example on stock returns is interesting, but it is largely qualitative and does not provide external validation of the recovered graph.

4. **Novelty relative to prior combinatorial and algebraic work is not fully isolated.**  
   The paper claims several “first” results, but the boundary with prior work on latent-variable ICA, transversal matroids, rank constraints, and cyclic equivalence is not always crisply delineated. In particular, it is hard to tell from the text how much is fundamentally new versus a synthesis/repackaging of existing matroidal and rank-based ideas.

5. **The empirical comparison is somewhat narrow.**  
   The baselines are limited to LaHiCaSl and PO-LiNGAM, which are relevant but not exhaustive for latent-variable causal discovery. Given the broad theoretical claims, the experimental section would ideally include more diverse baselines, sensitivity analyses, and ablations separating the contribution of OICA, the rank-realization step, and the equivalence-class traversal step.

6. **Potential scalability concerns remain.**  
   The paper argues that the method scales better than a MILP baseline, but the dependence on OICA and combinatorial graph traversal may still limit scalability. The runtime evidence is encouraging but only modestly probes the larger-scale regime and does not fully justify the “efficient” claim for broad use.

### Novelty & Significance
**Novelty:** High, if the equivalence characterization is correct as stated. A graphical equivalence theory for arbitrary latent structure and cycles in linear non-Gaussian models would be quite novel and likely of interest to the ICLR causal discovery community. The edge-rank viewpoint also appears genuinely fresh as a modeling/analysis tool.

**Significance:** Potentially high, because equivalence characterizations often enable downstream discovery algorithms, class enumeration, and canonical representations. However, the significance is somewhat tempered by the strong idealized assumptions and by the fact that practical gains over existing methods are not yet convincingly demonstrated.

**Clarity:** Moderate to low. The paper’s core ideas are interesting, but the exposition is very challenging, with many technical definitions and long proof-oriented sections in the main text.

**Reproducibility:** Moderate. Code and a demo are provided, which is a plus. But reproducing the theoretical pipeline likely depends on OICA implementation details, rank-thresholding heuristics, and nontrivial combinatorial procedures that are not fully transparent from the main text alone.

### Suggestions for Improvement
1. **Sharpen the main theorem statements and their assumptions.**  
   Provide a concise theorem table early in the paper summarizing exactly what is identified, under what assumptions, and whether the result is generic, oracle-based, or finite-sample.

2. **Separate the conceptual contribution from the technical machinery.**  
   Add a simpler, high-level explanation of why edge ranks are needed, how they differ from path ranks, and how they enable the final equivalence characterization.

3. **Strengthen the practical learning story.**  
   The paper should more clearly distinguish the theoretical identifiability result from the actual implemented algorithm, and clarify what is guaranteed in finite samples versus only under oracle/OICA assumptions.

4. **Expand and diversify experiments.**  
   Include more baselines, ablations, and sensitivity analyses. In particular, test robustness to OICA estimation errors, latent-number misspecification, and graph types beyond Erdős–Rényi.

5. **Provide more transparent algorithmic complexity analysis.**  
   Report the computational complexity of each stage of glvLiNG, especially the traversal procedure and any rank/matroid subroutines, so readers can better judge scalability.

6. **Improve the real-data analysis with validation.**  
   The stock-market case study would be stronger if the recovered relations were compared with known sectoral relationships, temporal validation, or out-of-sample predictive checks.

7. **Clarify the novelty relative to existing matroid and rank literature.**  
   Explicitly identify which results are new theorem-level contributions and which are derivations or translations of known matroid facts into the causal-discovery setting.

8. **Add a compact “how to use this result” section.**  
   ICLR readers would benefit from a concise guide: given data, what steps are run, what assumptions are needed, and what object is returned at the end?

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to the strongest latent-variable and cyclic causal discovery baselines that operate under weaker assumptions or recover equivalence classes, not just PO-LiNGAM and LaHiCaSl. For ICLR, the claim of a “first structural-assumption-free method” is not credible without benchmarking against methods like FCI/PAG-based approaches, cyclic LiNGAM variants, GES-style latent methods, and recent causal representation learning baselines on the same synthetic regimes.

2. Add ablations that isolate what actually drives performance: OICA quality, Theorem 2’s local decomposition, and the traversal procedure from Theorem 3. Without these, it is unclear whether gains come from the new theory or simply from a strong ICA estimator plus a graph post-processing heuristic.

3. Add experiments on data generated outside the paper’s exact linear non-Gaussian assumptions, especially mild nonlinearity, near-Gaussian noise, and misspecified latent counts. The paper claims broad applicability and robustness, but the method depends heavily on OICA identifiability and rank faithfulness; the boundary of failure needs to be measured.

4. Add a scaling study on larger graphs and denser latent connectivity for both runtime and recovery quality. Current results stop at small to medium graphs, which is insufficient for ICLR-level claims about a practical “efficient algorithm” when the underlying search space is combinatorial.

5. Add a controlled study showing how often the algorithm recovers the correct equivalence class versus only a graph in the class that matches SHD weakly. Because the output is an equivalence class, plain SHD to one representative can hide whether the method actually recovers the right unidentified structure.

### Deeper Analysis Needed (top 3-5 only)
1. Prove or empirically validate the key assumption that OICA returns a uniquely usable mixing matrix up to permutation/scaling in the presence of cycles and arbitrary latent structure. The paper’s main recovery pipeline rests on this, but the identifiability conditions and failure modes are not sufficiently quantified for the core claim to be trustworthy.

2. Analyze the necessity and practical testability of “faithfulness” in the rank-based sense used here. The method assumes no accidental rank cancellations, but the paper does not show how sensitive the equivalence characterization is to finite-sample rank errors or near-degenerate parameterizations.

3. Provide a complexity analysis of the equivalence traversal and class presentation in terms of number of vertices/latents. ICLR reviewers will expect more than qualitative “efficient” claims; the paper needs explicit worst-case or amortized complexity bounds for the main reconstruction and traversal steps.

4. Clarify what is and is not identifiable from observational data alone under this model class. The paper states broad equivalence characterization, but it still needs a sharper statement of identifiability of causal direction, latent structure, and cycle orientation to avoid overclaiming.

5. Analyze how the proposed edge-rank criterion relates to existing algebraic invariants beyond ranks. The discussion hints at additional polynomial constraints, but without a systematic explanation of why ranks are sufficient for equivalence in this setting, the theoretical contribution feels under-justified.

### Visualizations & Case Studies
1. Add side-by-side failure cases where the method and baselines disagree, with the true graph, recovered equivalence class, and the specific rank/matching constraints that caused each decision. This would expose whether the proposed criteria genuinely explain recoveries or merely post-hoc encode them.

2. Add a visualization of the equivalence class traversal on a nontrivial example, showing the admissible cycle reversals and edge additions/deletions step by step. The current examples are too schematic; an explicit trace would make the transformational characterization believable.

3. Add a visualization of uncertainty under finite samples: confidence in each inferred edge and how often the recovered class changes across bootstrap resamples. Since the method is rank-sensitive, this is essential to judge whether the output is stable or brittle.

4. Add a case study on the stock data that tests whether the inferred latent factors are actually interpretable across perturbations or subsets of stocks. One real-world example is not enough to support claims of practical usefulness; the result needs sensitivity checks.

### Obvious Next Steps
1. Add an OICA-free or rank-estimation-robust variant of glvLiNG. The paper itself identifies OICA as a limitation; without removing this dependency, the method is not yet a genuinely practical discovery pipeline.

2. Extend the characterization to Gaussian or mixed Gaussian/non-Gaussian settings, or explicitly delineate why the current theory cannot transfer. This is the most obvious theoretical next step if the authors want the work to matter beyond one narrow parametric family.

3. Develop a finite-sample theory linking rank test errors to equivalence-class recovery guarantees. For ICLR, a method paper needs more than generic identifiability; it needs a statement about when the algorithm succeeds under realistic sample sizes.

4. Provide an algorithmic implementation for enumeration/sampling of all graphs in the equivalence class, not just traversal in principle. The paper’s transformational result is interesting only if it yields an actually usable output mechanism, which is currently underdeveloped.

# Final Consolidated Review
## Summary
This paper studies distributional equivalence for linear non-Gaussian causal models with arbitrary latent variables and cycles. Its central contribution is a graphical characterization of equivalence, supported by a new “edge rank” tool and a transformational view of equivalence-class traversal; it also proposes glvLiNG, an algorithm that reconstructs models up to this equivalence class from OICA-estimated mixing matrices.

## Strengths
- The main theoretical contribution is genuinely ambitious and, if correct, substantial: the paper claims a distributional equivalence characterization for latent-variable LiNG models with arbitrary latent structure and cycles, which is a noticeably broader setting than most prior work.
- The edge-rank perspective is novel and useful. The paper does more than rephrase old results: it introduces a combinatorial dual to path ranks, proves a duality theorem, and uses it to derive a more local equivalence criterion and a traversal procedure for the equivalence class.
- The paper is not purely existential; it gives an actual pipeline, glvLiNG, and backs it with runtime experiments, synthetic evaluations, and a real-data case study. The code/demo availability is also a plus.

## Weaknesses
- The practical learning claim is overstated relative to the assumptions. glvLiNG still relies on OICA, genericity/faithfulness of rank patterns, and heuristic thresholding of empirical singular values. That is not “structural-assumption-free” in any practical sense; the theorem is about equivalence, but the implemented recovery pipeline is much more brittle than the rhetoric suggests.
- The exposition is extremely dense and hard to verify. The paper front-loads many definitions and reductions, and the main results depend on long chains of algebraic/matroid arguments that are difficult to assess from the main text alone. This makes it hard to judge whether the claimed equivalence characterization is as clean and general as advertised.
- The empirical evaluation is too narrow for the strength of the claims. The baselines are limited, there are no ablations isolating OICA quality versus the matroid/rank reconstruction and traversal steps, and the real-data analysis is qualitative. The paper does not convincingly show robustness under finite-sample rank errors or misspecified latent structure.
- The novelty boundary with existing rank/matroid literature is not fully crisp. The paper likely synthesizes several known ideas in a new setting, but the main text does not cleanly separate what is truly new from what is a translation of matroidal facts into the causal-discovery setting.

## Nice-to-Haves
- A compact theorem table early in the paper summarizing exactly what is identified, under which assumptions, and whether the result is generic, oracle-based, or finite-sample.
- More explicit complexity bounds for each stage of glvLiNG, especially the rank-realization and traversal components.
- A clearer explanation of how the proposed equivalence class representation should be used in practice, given an estimated mixing matrix.

## Novel Insights
The paper’s most interesting conceptual move is to recast latent-variable causal equivalence through a matroidal lens: path-rank constraints are not the end of the story, and edge ranks provide a local combinatorial dual that makes the final criterion and traversal results much more operational. That said, the algorithmic side does not fully live up to the elegance of the theory: the recovery pipeline still leans on oracle-like ICA and fragile rank estimation, so the work is best viewed as a strong theoretical characterization with a proof-of-concept learner rather than a mature practical discovery method.

## Potentially Missed Related Work
- FCI / MAG / PAG literature — relevant as the classic equivalence-class framework for latent-variable causal discovery, useful for comparing what “equivalence presentation” means in weaker nonparametric settings.
- Existing cyclic LiNGAM equivalence results — relevant because the paper extends cycle-only equivalence ideas into the latent-variable setting.
- Matroid-based structure learning work — relevant because the paper’s traversal and rank-realization arguments are deeply matroidal and should be situated more precisely in that literature.

## Suggestions
- Add ablations separating OICA estimation error, rank-to-graph realization, and equivalence-class traversal, since the current experiments do not show which part of the pipeline actually works.
- Provide a short “given data, what exactly do I run?” algorithmic summary with explicit assumptions and failure modes.
- Strengthen the finite-sample story: quantify how sensitive the method is to near-degenerate ranks and latent-count misspecification, rather than relying on oracle-style rank access.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
