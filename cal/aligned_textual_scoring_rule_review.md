=== CALIBRATION EXAMPLE 10 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title "Aligned Textual Scoring Rules" accurately reflects the paper's contribution. The abstract is clear, stating the problem (proper scoring rules may not align with preferences), the proposed solution (ASR optimizes MSE to a reference score while maintaining properness), and the key empirical claim (ASR outperforms baselines). The abstract correctly sets expectations.

**Introduction & Motivation**
The introduction effectively motivates the problem, transitioning from numerical proper scoring rules to textual elicitation with LLMs. It correctly identifies a gap in prior work (Wu & Hartline, 2024): ensuring properness but not alignment. The goal of aligning a proper scoring rule with human preference (e.g., instructor scores) is well-justified for applications like peer grading. The contributions are clearly stated: a simple, truthful, interpretable scoring rule optimized for alignment. The introduction is strong.

**Preliminaries (Section 2)**
The definitions for numerical and textual elicitation are precise. However, several assumptions and modeling choices are impactful and require more justification:
*   **Assumption 2.2 (Know-it-or-not):** This is a strong restriction, limiting posterior beliefs to {0, 1, prior}. The paper states it's based on the observed dataset, but its general applicability is questionable. The paper should discuss the implications of this assumption and what happens if agents have more nuanced beliefs (e.g., a probability of 0.7). This is a significant limitation.
*   **Language Oracle Model:** The reduction relies on Summarization and QA oracles. The assumption that the QA oracle is perfect for the ground truth but can have (non-inverting) errors on the agent's report is central to the properness guarantees. This is a reasonable modeling choice but should be explicitly justified in the context of real LLM behavior.
*   **Report Space:** The mapping of textual reports to the ternary space {0,1,⊥} and the treatment of ⊥ as the prior is a design decision. The paper should briefly discuss alternatives.

**Method / Approach (Section 3)**
The core method is clearly described: optimize over separate scoring rules to minimize MSE to a reference score subject to properness constraints.
*   **Convexity & Expressivity:** Corollary 3.4 correctly states the optimization is convex for separate scoring rules. However, the choice of the *separate* scoring rule space (weighted average of per-dimension scores) is a potentially limiting design decision. The paper compares to the max-over-separate (M) aggregation as a baseline but does not optimize over it due to non-convexity. A critical question is: **Is the separate scoring rule space sufficiently expressive to achieve good alignment?** An ablation study comparing the optimized separate rule to the (non-optimized) separate V-shaped rule (AV) would help disentangle the benefit of optimization from the choice of hypothesis space.
*   **Optimization Details:** Program 2 presents the formulation. Key details are missing for reproducibility:
    1.  **Boundedness Constraint:** The constraint ∑_i S_i(r_i, θ_i) ∈ [0,1] is global and non-linear. How is it enforced during gradient descent? (e.g., via projection, penalty, or Lagrange multipliers?)
    2.  **Weights in Separate Rule:** Definition 2.7 includes weights w_i, but Program 2 uses an unweighted sum. Are weights w_i implicitly set to 1/m? If so, this should be stated. If they are also optimized, the formulation needs to be updated.
    3.  **Empirical Expectations:** The properness constraints in Program 2 involve expectations E_{θ_i∼p_i}. How are these computed? From the empirical prior p_i? This should be clarified.
*   **Generalization & Overfitting:** The optimization is performed on a dataset. The paper does not mention a train/validation/test split or cross-validation. With a limited number of assignments (22), there is a risk of overfitting. The experimental setup must be clarified: is the rule learned per-assignment? If so, how many data points are used for learning per assignment? A discussion of sample complexity or use of cross-validation is needed.

**Implementation of Language Oracles (Section 4)**
The detailed prompts in the appendix are a strength for reproducibility. However:
*   **Clustering Robustness:** The summarization oracle involves a complex multi-step LLM clustering process. While the paper correctly notes that summarization errors do not affect *properness*, they could significantly impact *alignment* because the scoring rule is defined over the identified summary points. If clustering merges distinct points or creates noisy dimensions, alignment performance could degrade. Some analysis of clustering quality (e.g., inter-annotator agreement with a human) would strengthen the empirical validation.
*   **QA Oracle Verification:** Theorem 3.2's properness guarantee requires the QA oracle to be non-inverting on reports. The paper does not provide any evaluation to verify that the implemented LLM-based QA satisfies this property. This is a critical assumption that should be empirically checked or at least discussed as a potential failure mode.

**Empirical Evaluation (Section 5)**
The experiments on peer grading data are the main evidence for the alignment claim.
*   **Dataset & Setup Clarity:** The dataset description is vague. Key details are missing: total number of peer reviews, how data is split for optimization and evaluation, and whether scoring rules are learned per-assignment or globally. This makes it difficult to assess statistical significance and generalizability.
*   **Metric Scaling and Interpretation:** There is confusion regarding score scaling. The text states reference scores are normalized to [0,1] (Program 1), but Table 1 shows MSE values like 1.73 for ASR and 3.74 for the constant baseline. If scores are normalized to [0,1], an MSE of 1.73 is impossible. This suggests MSE is reported on the original scale (e.g., instructor scores in [0,10]). This must be clarified. The constant baseline MSE (variance of the reference score) provides a good reference point.
*   **Baseline Comparison:** Comparisons to the constant score and the ElicitationGPT (AV, MV) baselines are appropriate. The results show ASR achieves much lower MSE and higher correlation. However:
    *   **Statistical Significance:** No measures of variance (e.g., standard errors over assignments) or statistical tests are reported. Given the likely small sample size per assignment, the improvements, while large, need statistical validation.
    *   **Ablation Missing:** The most critical ablation is missing: **What is the performance gain from optimization itself versus just using a proper scoring rule?** A baseline that optimizes a separate scoring rule but without the properness constraints (i.e., a simple regression) could show the cost of the truthfulness constraint. Similarly, comparing optimized separate rules to the fixed V-shaped separate rule (AV) would isolate the benefit of optimization within the same hypothesis space.
*   **Case Demonstration (Appendix C):** The interpretability analysis is a nice plus, showing which summary points receive more convex scoring rules. This aligns with the claim of interpretability.

**Writing & Clarity**
The paper is generally well-written. The flow is logical. Some sections could be clearer:
*   Section 2 could better connect the ternary report space to real textual reports.
*   The formulation of Program 2 needs clarification regarding weights and the handling of the boundedness constraint.
*   Figures 1 and 2 are referenced but not included in the parsed text, slightly hampering understanding.

**Limitations & Broader Impact**
The paper lacks a dedicated limitations section. Key limitations that must be explicitly discussed include:
1.  The restrictive **Know-it-or-not assumption** (Assumption 2.2).
2.  Dependence on **language oracles** and the associated assumptions (perfect QA on ground truth, non-inverting QA on reports).
3.  The potential **limited expressivity** of the separate scoring rule space.
4.  **Experimental scope:** Results are on a specific peer grading dataset; generalizability to other textual elicitation domains is unknown.
5.  **Optimization practicalities:** Requirement for a dataset with reference scores and known priors.
Broader impacts are briefly implied (improving peer grading) but not deeply discussed. Potential negative societal impacts (e.g., gaming the aligned scoring rule) are not addressed.

### Overall Assessment

The paper presents a novel and well-motivated idea: aligning proper scoring rules for text with human preferences via convex optimization. The core contribution—a method to make a truthful scoring rule also match a reference score—is valuable for applications like peer grading. The theoretical grounding is sound, building correctly on prior work. However, the empirical evaluation has significant weaknesses: the experimental setup is unclear, key ablations are missing, statistical rigor is lacking, and the impact of strong modeling assumptions (Know-it-or-not) is underexplored. Furthermore, the reliance on LLM oracles without verification of their required properties (non-inverting QA) is a concern. The paper's contribution is promising and likely of interest to the ICLR community, but in its current form, it does not meet the high bar for acceptance. **Major revisions are required**, primarily to strengthen the empirical evaluation (clarify setup, add ablations, provide statistical analysis) and to thoroughly discuss limitations and assumptions.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Aligned Scoring Rules (ASR), a method for designing proper (truthful) scoring rules for textual information elicitation that are also aligned with human preferences. Building on the reduction from textual to numerical elicitation by Wu & Hartline (2024), ASR optimizes over a space of separate proper scoring rules to minimize the mean squared error to a reference score (e.g., instructor or LLM-judge scores). Experiments on peer grading datasets demonstrate improved alignment over non-aligned proper baselines while maintaining properness.

### Strengths
1. **Addresses a meaningful gap**: The paper identifies that proper scoring rules for text may not align with human preferences and proposes a principled optimization framework to bridge this gap. This is a novel and practical contribution at the intersection of mechanism design and NLP.
2. **Theoretically grounded and computationally tractable**: The method leverages the properness guarantees of the underlying reduction and formulates alignment as a convex optimization problem (for separate scoring rules), ensuring efficient solvability.
3. **Convincing empirical validation**: The paper provides thorough experiments on real peer grading data, showing ASR achieves lower MSE and higher correlation with reference scores compared to baselines. The near-identity linear regression fit strongly indicates successful alignment.
4. **Interpretability and case study**: The separate scoring rule structure allows for interpretation of rubric point importance, and the case demonstration (Appendix C) gives concrete insight into how ASR weights different aspects.

### Weaknesses
1. **Limited report space**: The method is restricted to a ternary report space {0,1,⊥} (know-it-or-not) based on an empirical observation from the dataset. This limits generalizability to settings where agents can report continuous probabilities or more nuanced beliefs.
2. **Heavy reliance on LLM oracles**: The approach depends on LLMs for summarization and question-answering. While the paper cites robustness results from prior work, it does not empirically test how oracle errors impact alignment or properness in practice.
3. **Narrow empirical scope**: Evaluation is solely on peer grading datasets. Broader validation on other textual elicitation tasks (e.g., crowdsourcing, forecasting) would strengthen the claim of general applicability.
4. **Baseline comparisons could be deeper**: The baselines are relatively simple (constant score and non-aligned ElicitationGPT). Comparing to more advanced methods from differentiable economics or automated mechanism design would better contextualize the performance gains.

### Novelty & Significance
The paper’s novelty lies in combining proper scoring rules with preference alignment for textual elicitation, an underexplored area. The significance is both theoretical (extending the scoring rule optimization literature to text) and practical (enabling scalable, truthful, and human-aligned evaluation in settings like peer grading). While the core reduction framework is from prior work, the alignment optimization and its application to text with LLM oracles represent a clear advance.

### Suggestions for Improvement
1. **Generalize the report space**: Explore extensions to continuous or more expressive report spaces to enhance applicability beyond the know-it-or-not setting.
2. **Evaluate robustness to oracle errors**: Conduct ablation studies with noisy or imperfect LLM oracles to understand the sensitivity of alignment and properness.
3. **Broaden empirical evaluation**: Test ASR on additional textual elicitation benchmarks (e.g., crowdsourced fact-checking, subjective judgment aggregation) to demonstrate wider utility.
4. **Include more competitive baselines**: Compare against state-of-the-art methods from differentiable economics or automated mechanism design that could also be adapted for alignment.
5. **Clarify theoretical contributions**: More explicitly distinguish the novel optimization formulation from the underlying reduction framework, perhaps by discussing the convexity and interpretability advantages of separate scoring rules in greater depth.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Test on diverse textual tasks beyond peer grading.** The claims of a general textual elicitation method are undermined without validation on other domains (e.g., forecasting, QA). This is critical for ICLR's expectation of broad applicability.
2. **Empirically verify properness under realistic oracle errors.** The theoretical guarantees assume a non-inverting QA oracle; an experiment with strategic agents (simulated or human) using the actual LLM oracles is needed to confirm truthful incentives hold in practice.
3. **Compare against an unconstrained predictor of the reference score.** To show the cost of imposing properness, a baseline that directly predicts the reference score from reports (without properness constraints) should be included. This establishes an alignment upper bound.
4. **Ablation on the separate scoring rule structure.** The choice of separate scoring rules is justified by convexity, but no comparison is made to optimizing other proper aggregations (e.g., max-over-separate) via non-convex methods. This omission leaves it unclear if the hypothesis space is optimal.

### Deeper Analysis Needed (top 3-5 only)
1. **Systematic analysis of the learned scoring rules' interpretability.** The paper claims interpretability but only provides one anecdotal example. A quantitative analysis linking learned weights/curvature to rubric importance across all assignments is necessary to substantiate this claim.
2. **Investigate the tension between alignment and properness.** Analyze how much alignment error is attributable to the properness constraint by comparing the MSE of ASR to that of an unconstrained model. Without this, it's unclear if the alignment loss is inherent to properness.
3. **Sensitivity to the summarization granularity.** The method depends on the number and quality of summary points from the LLM. An analysis of how clustering parameters (e.g., number of clusters) affect alignment and stability is missing and critical for reproducibility.

### Visualizations & Case Studies
1. **Qualitative case studies of successes and failures.** Show concrete examples of peer reviews where ASR aligns well or poorly with the reference score, alongside baseline scores. This would reveal failure modes (e.g., misaligned rubric points) and strengthen the empirical narrative.
2. **Visualization of strategic robustness.** For a set of simulated agent beliefs, plot the expected score under ASR for truthful vs. strategic reports. This visual proof would bolster the properness claim beyond theory.

### Obvious Next Steps
1. **Extend optimization to non-separable proper scoring rules.** The paper restricts to separate rules for convexity, but exploring optimization over broader classes of proper rules (e.g., via differentiable economics) is a direct next step to improve alignment.
2. **Deploy in a live peer-grading experiment.** The ultimate test is whether ASR actually elicits higher-quality reviews in a controlled setting with real students. This should have been a validation step in the paper.
3. **Handle continuous or more nuanced beliefs.** The "know-it-or-not" assumption is a major limitation. The method should be extended to allow continuous reports to handle more general textual elicitation.

# Final Consolidated Review
## Summary
This paper proposes Aligned Scoring Rules (ASR), a method for designing proper (truthful) scoring rules for textual information elicitation that are also aligned with human preferences. Building on a prior reduction from textual to numerical elicitation, ASR performs convex optimization over a space of separate proper scoring rules to minimize the mean squared error to a reference score (e.g., an instructor's grade). Experiments on peer grading data show improved alignment over non-optimized proper baselines.

## Strengths
- **Addresses a clear and practical gap**: The paper identifies that proper scoring rules for text may not align with human preferences and provides a principled optimization framework to bridge this gap, a novel contribution at the intersection of mechanism design and NLP.
- **Convex and interpretable optimization**: The method restricts its hypothesis space to separate scoring rules, which yields a convex optimization problem for efficient solving and provides inherent interpretability, as demonstrated by visualizing the importance of different rubric points.
- **Empirical validation on a real task**: The paper provides convincing experiments on peer grading datasets, showing ASR achieves lower MSE and higher correlation with reference scores than non-aligned proper baselines, with a near-identity linear fit indicating successful alignment.

## Weaknesses
- **Limited report space restricts generality**: The method is built on the "know-it-or-not" assumption (Assumption 2.2), restricting beliefs and reports to {0, 1, prior}. This is justified by the specific dataset but significantly limits the method's applicability to general textual elicitation where agents hold nuanced, continuous beliefs.
- **Experimental setup lacks rigor and clarity**: The empirical evaluation omits standard practices, including statistical significance tests, a clear train/test or cross-validation procedure, and a description of how the global boundedness constraint is enforced during optimization. This makes it difficult to assess the robustness of the reported improvements and the risk of overfitting on the limited number of assignments.
- **Potential expressivity limitation of the hypothesis space**: The optimization is performed over the space of separate (weighted average) scoring rules. While this enables convexity, the paper does not demonstrate that this space is sufficiently expressive, nor does it ablate the benefit of optimization within this space versus using a fixed proper rule (e.g., the averaged V-shaped rule).

## Nice-to-Haves
- An analysis quantifying the "cost of properness" by comparing ASR's alignment error to that of an unconstrained model predicting the reference score.
- Broader validation on other textual elicitation tasks (e.g., forecasting, crowdsourcing) to demonstrate general applicability beyond peer grading.
- An empirical check of the critical assumption that the implemented LLM-based QA oracle is "non-inverting" on reports, as required for the theoretical properness guarantee.

## Novel Insights
The paper's core novel insight is that for textual elicitation, one can decouple the guarantee of truthfulness (via a prior reduction framework) from the goal of preference alignment, and then computationally search for a scoring rule within the truthful hypothesis space that best matches a reference signal. This provides a principled pathway to convert non-proper but aligned evaluation metrics (like instructor or LLM-judge scores) into proper mechanisms.

## Suggestions
- Clarify the experimental setup in a revision: specify the data split for optimization/evaluation, report standard errors or confidence intervals for the metrics, and describe how the boundedness and properness constraints are implemented numerically.
- Conduct an ablation study within the separate scoring rule space (e.g., compare the optimized rule to the fixed V-shaped separate rule) to isolate the performance gain from optimization itself.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 0.0, 0.0]
Average score: 0.5
Binary outcome: Reject
