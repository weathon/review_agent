=== CALIBRATION EXAMPLE 9 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title "ALIGNED TEXTUAL SCORING RULES" accurately reflects the paper's contribution. The abstract clearly states the goal: to design a proper scoring rule for text that aligns with human preference. However, it lacks crucial details about the method's assumptions and limitations. Specifically, it does not mention the restrictive "know-it-or-not" report space (reports limited to 0, 1, or ⊥) or the use of separate (weighted sum) scoring rules, which define the hypothesis space for optimization. The claim that ASR "outperforms previous methods" is supported in the experiments, but the abstract should briefly quantify the improvement (e.g., correlation coefficients) to meet ICLR's standards for concrete claims.

### Introduction & Motivation
The introduction effectively motivates the problem of aligning truthful textual elicitation with human preferences, building nicely on Wu & Hartline (2024). The contributions are stated but could be more explicitly enumerated. A significant omission is the lack of motivation for the key **Assumption 2.2 (know-it-or-not reports)**, which is central to the technical approach but only introduced later. The introduction should foreshadow and justify this assumption, as it critically limits the generality of the method (e.g., it may not handle nuanced uncertainty in text). The writing is clear, but the scope of the contribution should be more precisely framed from the start.

### Preliminaries (Section 2)
This section defines the elicitation setting clearly. However, the transition from general proper scoring rules to the restricted "know-it-or-not" setting (Assumption 2.2) is abrupt and under-motivated. The assumption that an agent's posterior belief is either 0, 1, or the prior is a strong simplification of textual reporting; the paper argues it holds in their dataset but does not discuss its general applicability. This is a major limitation that should be explicitly acknowledged and justified here. Definition 2.3 and Definition 2.5 are consistent, but the relationship between the general proper scoring rule *S* and the derived rule *S_p* could be clarified. The definitions of V-shaped, separate, and max-over-separate scoring rules are clear.

### Aligned Scoring Rule: Algorithm (Section 3)
**3.1 Provable Properness:** The reduction's properness relies on the question-answering oracle being *non-inverting* (Definition 3.1). The paper does not discuss how to verify this property in practice or its plausibility when using real LLMs. This is a significant assumption for the theoretical guarantee.
**3.2 Optimization for Alignment:** The optimization problem (Program 2) is well-formulated. The choice of separate scoring rules ensures convexity, but it restricts the hypothesis space. The paper correctly notes that other aggregations (e.g., max-over-separate) do not yield convex problems, but it does not discuss the potential expressiveness lost by not considering them. A more serious concern is that the properness constraints in Program 2 are enforced only for the *specific prior p_i* estimated from the data. While this suffices under Assumption 2.2, the paper should explicitly argue why this is sufficient and highlight that properness guarantees are conditional on that assumption. Corollary 3.4 claims convexity without proof; a brief justification (linear constraints, convex objective) should be provided.

### Implementation of Language Oracles (Section 4)
The implementation details are thorough, and prompts are provided in the appendix, aiding reproducibility. However, there is **no evaluation of the oracle's accuracy or the non-inverting property**. For a method that relies on these oracles for properness, it is essential to analyze their error rates and discuss how errors might affect alignment and truthfulness. The summarization process involving clustering of negative/positive pairs is innovative but its robustness across different domains is untested.

### Empirical Evaluation (Section 5)
**Dataset and Metrics:** The peer grading dataset is appropriate but relatively small (516 reviews across 22 assignments). The paper does not specify the train/test split or cross-validation procedure, making it hard to assess overfitting. The metrics (MSE, Pearson, Spearman) are standard.
**Results:** Table 1 shows ASR outperforms baselines in alignment metrics. However, the baselines are weak: a constant score and non-aligned proper rules (AV, MV). It is unsurprising that optimizing for alignment improves alignment. A stronger baseline would be a non-proper LLM-judge score itself, to see the cost of imposing properness. The correlation of ASR with instructor scores (~0.71) is moderate, indicating room for improvement.
**Missing Analyses:** 
1. No direct test of truthfulness: while properness is proven theoretically under assumptions, an empirical simulation with strategic agents would strengthen the claim.
2. No ablation study: How important is the separate scoring rule restriction? What if the know-it-or-not assumption is relaxed?
3. Sensitivity analysis: The results depend on the LLM used for oracles and judging (Gemini vs. GPT-4.1 in Appendix B shows variability). This should be discussed as a limitation.
4. Interpretability claim (Appendix C) is only demonstrated with one example; a more systematic analysis of learned weights across assignments would be more convincing.

### Limitations & Broader Impact
A dedicated limitations section is **absent**, which is a major flaw for an ICLR submission. Key limitations that must be acknowledged include:
- Strong reliance on Assumption 2.2 (know-it-or-not reports), limiting applicability to textual reports expressing more nuanced uncertainty.
- Hypothesis space restricted to separate scoring rules, potentially missing more expressive aligned rules.
- Dependence on the quality and non-inverting property of LLM oracles, with no empirical validation.
- Evaluation on a single, narrow domain (peer grading in algorithm classes); generalizability is unclear.
- The reference scores (instructor or LLM-judge) may contain biases; aligning with them could perpetuate those biases without discussion.

Broader impact should discuss potential benefits (scalable, truthful peer assessment) and risks (gaming the system if assumptions fail, fairness concerns if biased references are used).

### Writing & Clarity
Overall, the paper is well-organized and clearly written. Some technical passages could be smoother (e.g., the transition from numerical to textual elicitation). The figures and tables are helpful. The appendix provides necessary details. However, the lack of a limitations section and insufficient motivation for key assumptions hinder the overall clarity of the contribution's scope.

### Overall Assessment
The paper proposes a novel and theoretically sound method to align proper scoring rules with human preferences for textual elicitation. The core idea—optimizing over a space of proper rules to minimize deviation from a reference score—is valuable and well-executed within the chosen framework. Empirical results show improved alignment over non-optimized proper rules. However, the work is significantly hampered by strong, under-motivated assumptions (know-it-or-not reports, separate scoring rules) and a relatively narrow evaluation that does not fully address the method's robustness or limitations. The absence of a limitations section is a critical omission for ICLR. While the contribution is promising, the paper in its current form does not fully meet ICLR's high standards for technical rigor and comprehensive evaluation. **Major revisions** are needed to address the concerns above, particularly by thoroughly discussing limitations, strengthening the empirical validation, and providing a more nuanced analysis of the assumptions.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Aligned Scoring Rules (ASR) for textual information elicitation, building upon a recent reduction from textual to numerical elicitation. The core idea is to optimize over a space of proper scoring rules (specifically, separate scoring rules) to minimize the mean squared error between the rule's output and a reference score (e.g., from a human instructor or an LLM judge). This yields a scoring mechanism that is provably truthful (proper) while being better aligned with human preferences than non-optimized proper rules. Empirical evaluation on peer-grading datasets demonstrates improved alignment metrics compared to baseline proper scoring rules.

### Strengths
1. **Well-Motivated and Practical Problem**: The paper addresses a clear and timely issue: how to design scoring rules for text that are both strategically truthful (proper) and produce scores that match human preferences. The application to scalable, LLM-assisted peer grading is impactful and well-argued.
2. **Solid Technical Foundation**: The work cleanly builds upon the reduction framework of Wu & Hartline (2024), inheriting its properness guarantees. The formulation of alignment as a convex optimization problem (for separate scoring rules) is sound and computationally tractable.
3. **Comprehensive Empirical Evaluation**: The experiments are thorough, using real-world peer-grading data from two algorithm classes. The evaluation employs multiple relevant metrics (MSE, Pearson, Spearman) and compares against meaningful baselines (constant score, prior non-aligned proper rules). The results consistently show ASR's superior alignment.
4. **Interpretability and Analysis**: The paper provides a case study (in the appendix) visualizing the learned single-dimensional scoring rules, offering interpretability into which rubric points the aligned rule deems important—a valuable feature for real-world deployment.

### Weaknesses
1. **Restrictive "Know-It-or-Not" Assumption**: The theoretical and empirical framework relies on Assumption 2.2, which restricts agent beliefs and reports to be either 0, 1, or the prior (⊥). This significantly simplifies the problem but may not hold in many realistic textual elicitation settings where reports can express nuanced, continuous uncertainties.
2. **Limited Hypothesis Space**: The optimization is performed over the space of *separate* scoring rules (weighted averages of per-dimension rules). While this yields convexity, it may be less expressive than other aggregations (e.g., max-over-separate). The paper does not deeply explore the performance trade-off induced by this architectural choice.
3. **Domain-Specific Evaluation**: The empirical validation is conducted solely on peer-grading datasets from undergraduate algorithm courses. While appropriate for a proof-of-concept, the generalizability of the method to other domains (e.g., creative writing, legal analysis, open-ended QA) remains an open question and is not discussed.
4. **Dependence on LLM Oracles**: The method's end-to-end performance hinges on the quality of the LLM-based summarization and question-answering oracles. Although the paper cites robustness results from prior work, it does not empirically analyze how errors in these oracles propagate to affect alignment or properness in its own optimization pipeline.

### Novelty & Significance
**Novelty**: The core novelty lies in the *optimization for alignment* within the constrained space of proper scoring rules for textual elicitation. While the reduction framework and the use of proper scoring rules are not new, formulating and solving the alignment problem as a convex optimization over this space is a novel and meaningful contribution. It effectively bridges the goals of strategic truthfulness and preference matching.

**Significance**: The work has significant practical implications for areas like education (scalable peer grading) and AI alignment, where we need automated systems that are both robust to manipulation and produce ratings that humans find sensible. By providing a method to "convert" a non-proper reference score (like an instructor's grade) into a proper proxy, it offers a principled path towards more reliable and scalable evaluation mechanisms. The approach is conceptually clear and demonstrates strong empirical results on its target domain.

### Suggestions for Improvement
1. **Relax the "Know-It-or-Not" Assumption**: Explore extensions to handle continuous belief reports (e.g., \( r_i \in [0,1] \)). This would increase the model's realism and applicability. A discussion on the computational and statistical challenges of this extension would be valuable.
2. **Expand the Hypothesis Space**: Experiment with optimizing over a broader class of proper scoring rules beyond separate aggregations, perhaps using techniques from differentiable economics. While convexity might be lost, this could lead to better alignment and reveal interesting trade-offs between expressivity and optimization stability.
3. **Evaluate on More Diverse Tasks**: To bolster claims of generalizability, apply ASR to at least one other textual elicitation domain (e.g., summarizing news articles, providing feedback on essays). This would test the robustness of the summarization/OA pipeline and the alignment method itself.
4. **Analyze Oracle Error Propagation**: Conduct a sensitivity analysis or ablation study to understand how errors in the LLM-based summarization and question-answering steps affect the final aligned scoring rule's properness and alignment. This would provide crucial insights for practical deployment.
5. **Compare with a Strong, Non-Truthful Baseline**: Include a comparison with a high-performing non-proper scoring method (e.g., a fine-tuned LLM judge) to quantitatively illustrate the *cost* of imposing properness. This would help situate the contribution by showing the alignment gap that properness necessitates closing.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Empirical test of strategic robustness.** The paper claims properness but provides no simulation with strategic agents (e.g., using LLMs to generate adversarial or deceptive reports) to show ASR is harder to manipulate than the non-proper reference scores. Without this, the practical benefit of properness is unsubstantiated.
2. **Comparison against strong non-proper alignment baselines.** The baselines are only other proper rules and a constant. To claim alignment is effective, you must compare to state-of-the-art non-proper methods (e.g., direct LLM-as-Judge scoring, embedding similarity) to show ASR does not sacrifice too much alignment for properness.
3. **Ablation on the separate scoring rule restriction.** The optimization is limited to separate (linear combination) scoring rules. You must compare to optimizing over a richer class (e.g., including max-over-separate) to show this choice does not severely limit alignment performance.
4. **Validation on diverse textual elicitation tasks.** The evaluation is solely on peer grading. To demonstrate general applicability, test on other tasks like summarization quality assessment or factual correctness elicitation to show the method generalizes beyond educational reviews.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of the alignment–properness trade-off.** Measure how much the optimization for alignment degrades approximate properness (e.g., compute the maximum expected gain from deviating from truth). Without this, it’s unclear if the optimized rule remains sufficiently proper in practice.
2. **Systematic interpretation of learned scoring rule weights.** The case study is anecdotal. Perform a quantitative analysis correlating the learned weights/curvatures with independent human judgments of rubric importance to validate that ASR identifies semantically meaningful dimensions.
3. **Sensitivity to LLM oracle errors.** The theory assumes a non-inverting oracle. Analyze how ASR’s alignment and properness degrade when the QA oracle makes systematic or random errors (e.g., via simulated noise). This is critical because real LLMs are imperfect.
4. **Statistical significance testing.** Report confidence intervals or statistical tests for the improvements in MSE and correlation over baselines. With only 22 assignments, it’s necessary to show differences are not due to chance.

### Visualizations & Case Studies
1. **Residual plots or failure case analysis.** Plot the residuals (reference score minus ASR score) against review features (e.g., length, sentiment) to reveal systematic biases or failure modes where alignment breaks down.
2. **Strategic manipulation case study.** Select a few reviews and show how a strategic agent could gain by misreporting under the reference score (e.g., LLM-judge) but not under ASR, visually demonstrating the properness advantage.
3. **Visualization of scoring rule adaptation across assignments.** Plot the learned weights or curvature parameters for each assignment to show if ASR consistently identifies similar important rubrics or adapts to context, indicating overfitting or generalization.

### Obvious Next Steps
1. **Extend to continuous or more expressive reports.** The know-it-or-not (0,1,⊥) report space is restrictive. Implement and test with continuous belief reports to handle nuanced uncertainties, which is a natural next step for textual elicitation.
2. **Incorporate flexible aggregation via neural networks.** Mentioned as future work, but a preliminary experiment using a differentiable properness-preserving architecture (e.g., from differentiable economics) could show potential for better alignment without convex restriction.
3. **Human evaluation of alignment.** Conduct a user study where instructors rank or prefer scores from ASR vs. baselines. Correlation metrics are indirect; direct human preference would strengthen the alignment claim.
4. **Computational scalability analysis.** Provide runtime and scaling results (e.g., time vs. number of summary points, reviews) for the convex optimization, as practical deployment requires efficiency.

# Final Consolidated Review
## Summary
This paper proposes Aligned Scoring Rules (ASR), a method for designing proper (truthful) scoring rules for textual information elicitation that are also aligned with human preferences. Building on a prior reduction from textual to numerical elicitation, the authors frame alignment as a convex optimization problem—minimizing the mean squared error between a proper scoring rule (from the space of separate scoring rules) and a reference score (e.g., from an instructor or an LLM judge). Experiments on peer-grading data show ASR achieves better alignment metrics than non-optimized proper baselines while maintaining provable properness under stated assumptions.

## Strengths
- **Well-motivated and practical contribution:** The paper addresses a timely and important problem: designing automated scoring mechanisms for text that are both strategically robust (proper) and produce scores that match human judgments. The application to scalable, LLM-assisted peer grading is clearly impactful.
- **Solid technical foundation:** The work cleanly builds upon the established reduction framework of Wu & Hartline (2024), inheriting its properness guarantees. Formulating the alignment objective as a convex optimization over the space of separate scoring rules is sound and computationally tractable.
- **Thorough empirical evaluation on real-world data:** Experiments use peer-grading datasets from two algorithm courses. Evaluation employs multiple relevant metrics (MSE, Pearson, Spearman) and shows consistent improvement over meaningful proper baselines (constant score, prior non-aligned rules). The appendix provides a detailed case study demonstrating the interpretability of the learned scoring rules.

## Weaknesses
- **Restrictive "Know-It-or-Not" reporting assumption:** The core technical approach relies on Assumption 2.2, which restricts an agent's posterior belief (and thus reportable content) to be either 0, 1, or the prior (⊥). This is a strong simplification of textual reporting that may not hold in many settings where nuanced, continuous uncertainty is expressed. The paper motivates this based on observed data but does not discuss its general applicability, which limits the method's scope.
- **Constrained hypothesis space may limit expressiveness:** Optimization is performed only over the space of *separate* scoring rules (weighted averages of per-dimension rules). While this ensures convexity, it may be less expressive than other proper aggregations (e.g., max-over-separate). The paper does not analyze the potential performance trade-off incurred by this architectural choice.
- **Lack of a dedicated limitations section:** A significant omission is the absence of a section systematically discussing the method's limitations. Key issues—such as the domain-specific evaluation, dependence on LLM oracle quality without empirical error analysis, and the potential for propagating biases from the reference scores—are noted but not synthesized, which hinders a complete assessment of the work's boundaries.

## Nice-to-Haves
- **Comparison against strong non-proper baselines:** Including a comparison with a high-performing, non-proper alignment method (e.g., a direct LLM-as-Judge score) would help quantify the *cost* of imposing properness and better contextualize the alignment performance.
- **Empirical analysis of oracle error propagation:** A sensitivity analysis or ablation studying how errors in the LLM-based summarization and question-answering oracles affect final alignment and properness would provide valuable insights for practical deployment.
- **Evaluation on additional textual domains:** Testing the method on at least one other textual elicitation task (beyond peer grading) would strengthen claims about generalizability and reveal domain-specific challenges.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"Abstract lacks crucial details about assumptions"**: The abstract succinctly states the paper's goal; detailed assumptions are appropriately presented in the main text.
- **"No evaluation of the oracle's accuracy or the non-inverting property"**: The paper explicitly relies on the prior theoretical framework (Wu & Hartline, 2024) for robustness guarantees; empirical validation of the oracle is not a stated contribution of this work.
- **"No direct test of truthfulness... an empirical simulation with strategic agents would strengthen the claim"**: The paper provides a theoretical properness guarantee under its assumptions; an empirical test, while interesting, is not required to establish the core claim.
- **"No ablation study... What if the know-it-or-not assumption is relaxed?"**: This demands an extension beyond the paper's clearly scoped contribution.
- **"No statistical significance testing"**: The results show clear, consistent quantitative improvements across multiple metrics; formal significance testing, while good practice, is not standardly required for this type of empirical comparison in the field.

## Novel Insights
The core novel insight is the formulation of alignment for textual scoring rules as a convex optimization problem over a constrained space of proper rules. This provides a principled method to transform a non-proper reference score (e.g., an instructor's grade) into a proper proxy, effectively bridging the objectives of strategic truthfulness and preference matching. The interpretable, separate structure of the optimized rule also offers a means to identify which rubric points the alignment process deems important.

## Suggestions
- **Add a dedicated "Limitations" section** to explicitly discuss the impact of the know-it-or-not assumption, the separate scoring rule restriction, domain specificity, and dependence on LLM oracles.
- **Conduct a simple but illustrative experiment comparing ASR to a strong non-proper baseline** (e.g., the raw LLM-Judge score) to clearly demonstrate the alignment-properness trade-off.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 0.0, 0.0]
Average score: 0.5
Binary outcome: Reject
