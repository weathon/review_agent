=== CALIBRATION EXAMPLE 30 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the paper's contribution. The abstract clearly states the motivation (overlooking combination effects in LLM-based rule learning), the proposed RLIE framework, and the key finding that direct logistic regression inference outperforms LLM-augmented strategies. However, the claim of "superior performance" is somewhat overstated given that RLIE is not always the absolute best in Table 1 (e.g., HypoGeniC outperforms RLIE on some datasets). The abstract should more precisely characterize the results (e.g., "competitive and robust performance").

### Introduction & Motivation
The introduction effectively motivates the need for interpretable rule sets and identifies a clear gap: existing LLM-based methods do not jointly learn and combine rules probabilistically. The contributions are well-articulated and align with the paper’s content.

### Method / Approach
The four-stage framework is clearly described. However, several details critical for reproducibility are missing or ambiguous:
- **Rule generation:** The prompt templates are in the appendix, but the main text does not describe how the LLM is prompted to generate candidate rules initially (e.g., the exact instruction). 
- **Iterative refinement:** When the rule set exceeds capacity \(H\), the paper states rules are discarded "based on their individual accuracy on the validation set." This is vague: Is accuracy computed as the accuracy of using that single rule as a classifier? How is the ternary judgment (+1/0/-1) converted to a binary prediction for this calculation? This needs clarification.
- **Hard example selection:** The parameter \(k\) (number of hard examples) is set to 20, but no justification or sensitivity analysis is provided. This could affect refinement behavior.
- **Logistic regression:** The use of Elastic Net is appropriate, but the paper does not specify how the hyperparameters (\(\lambda, \alpha\)) are tuned (e.g., search ranges, cross-validation details). This is important for reproducibility.

Overall, the method is conceptually sound, but the missing details could hinder exact replication.

### Experiments & Results
This section has significant issues that undermine the validity of the claims:
- **Inconsistent LLM backbones:** The paper states in Section 4.3 that "All experiments involving LLMs utilized gpt-4o-mini." However, Table 1 shows baselines using DeepSeek-V3, while RLIE is evaluated with Qwen3-Next-80B, Qwen3-235B, and DeepSeek-V3. This inconsistency makes comparisons unfair. If RLIE uses stronger backbones than the baselines, its superior performance may be attributed to the model rather than the framework. The authors must either re-run all methods with the same backbone or provide a rigorous justification for using different models.
- **Missing baseline:** Table 1 includes "Few-shot (ICL)" which is not described in the baseline section (4.2). Its inclusion without explanation is confusing.
- **LoRA fine-tuning baseline:** The LoRA baseline uses Qwen3-8B, while other baselines use DeepSeek-V3. This further confounds the comparison. The paper notes LoRA fails on complex tasks, but this may be due to model size/capability differences rather than the method itself.
- **Dataset size:** The small fixed splits (200 train, 200 validation, 300 test) may not provide robust statistical conclusions, especially for iterative methods. The authors should discuss the impact of sample size on rule learning.
- **Statistical significance:** While means and standard deviations over three runs are reported, no statistical significance tests (e.g., paired t-tests) are performed to compare RLIE with baselines or inference strategies. This is important given the observed variances.
- **Ablation studies:** The paper lacks an ablation study on the iterative refinement component. How much does iterative refinement contribute versus the initial rule generation? The case study in Appendix B is qualitative; a quantitative ablation (e.g., performance after each iteration) would strengthen the claim.
- **Parameter sensitivity:** Appendix C shows sensitivity to coverage threshold \(\gamma\), but this is not discussed in the main text. The robustness to other key parameters (e.g., rule capacity \(H\), number of hard examples \(k\)) should be analyzed.

The inference strategy comparison (Table 2) is a highlight, showing that "Linear-only" consistently outperforms LLM-augmented strategies. However, the same backbone inconsistency applies here: which LLM is used for strategies E2-E4? If different from the rule-generation LLM, this should be clarified.

### Writing & Clarity
The paper is generally well-written and logically structured. However, the experimental section is confusing due to the inconsistent model usage and missing details. The description of baselines and model choices needs to be clarified to allow readers to understand exactly what is being compared.

### Limitations & Broader Impact
The discussion appropriately notes that LLMs struggle with fine-grained probabilistic integration, which is a key limitation. The ethics statement acknowledges potential biases in rule sets. However, the paper could more explicitly discuss limitations such as: dependency on the quality of the LLM for rule generation, computational cost of iterative LLM calls, and the small dataset sizes used in evaluation.

### Overall Assessment
The paper presents a novel and well-motivated framework (RLIE) that combines LLM-based rule generation with probabilistic combination via logistic regression. The central finding—that direct logistic regression inference outperforms strategies that inject rules back into the LLM—is valuable and aligns with emerging observations about LLM limitations. However, the experimental evaluation has critical flaws, primarily the inconsistent use of LLM backbones across baselines and the proposed method, which severely undermines the fairness of comparisons. Additionally, the small dataset sizes and lack of statistical significance tests weaken the empirical support. The contribution is promising, but the paper in its current form does not meet ICLR's standards for rigorous evaluation. Major revisions to the experimental setup (e.g., consistent backbones, larger-scale validation, proper statistical analysis) are required to substantiate the claims.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces RLIE, a framework that integrates Large Language Models (LLMs) for generating natural language rules with a logistic regression model for probabilistically weighting and selecting those rules. The framework includes iterative refinement based on prediction errors and a systematic evaluation comparing direct inference using the weighted rule set against various methods of injecting rules into an LLM. A key empirical finding is that using the logistic regression model directly (Linear-only) outperforms strategies where the LLM is prompted with the rules and their weights.

### Strengths
1. **Unified and Well-Motivated Framework**: The paper clearly identifies a gap—existing LLM-based rule learning methods often overlook rule combination effects and probabilistic calibration—and proposes a cohesive four-stage pipeline (Rule generation, Logistic regression, Iterative refinement, Evaluation) to address it. The motivation is grounded in both classical rule learning and modern LLM capabilities.
2. **Rigorous and Hierarchical Evaluation**: The paper goes beyond standard performance comparison by designing a structured ablation on inference strategies (E1-E4). This provides valuable empirical evidence for the central claim that LLMs struggle with fine-grained probabilistic integration, a finding with practical significance for neuro-symbolic system design.
3. **Strong Empirical Results**: RLIE demonstrates robust and competitive performance across six diverse real-world datasets (from HypoBench), often ranking in the top two. The results are presented with appropriate metrics (Accuracy, F1) and standard deviations, showing stability. The case study (Table 3) qualitatively illustrates the iterative refinement process.

### Weaknesses
1. **Limited Analysis of the Core Negative Finding**: While the result that LLM-augmented inference (E2-E4) underperforms Linear-only (E1) is striking, the analysis remains somewhat superficial. The discussion (Section 6) offers plausible high-level explanations (e.g., LLMs are less reliable at "fine-grained, controlled probabilistic integration") but lacks a deeper investigation. For instance, there is no error analysis examining whether specific types of rules or weight patterns lead the LLM astray, or whether prompt design could be improved to mitigate this.
2. **Baseline Comparisons and Task Complexity**: The selected baselines (Zero-shot, IO Refinement, HypoGeniC) are appropriate, but the paper notes that "Zero-shot Inference outperforms many of the more complex baselines in several scenarios." This raises questions about the absolute difficulty of the chosen tasks and whether the gains from RLIE are substantial enough over a very simple baseline. Furthermore, the comparison to LoRA fine-tuning (Table 1) is somewhat apples-to-oranges, as LoRA is a full-model fine-tuning method, not a rule-learning method.
3. **Assumption of LLM as a Reliable Rule "Judge"**: The method relies on the LLM to provide consistent ternary judgments (abstain, positive, negative) for each rule on every data point (Φ mapping). The paper does not discuss the potential noise, inconsistency, or bias in these LLM-based judgments, which form the foundational features for the logistic regression. The sensitivity of the overall framework to this noise is not evaluated.

### Novelty & Significance
**Novelty**: The core novelty lies in the explicit integration of LLM-based natural language rule generation with a classic probabilistic combiner (elastic-net logistic regression) for global rule weighting and selection. The hierarchical evaluation of inference strategies is also a novel contribution for understanding how to best use LLM-generated rules.

**Significance**: The work is significant for the neuro-symbolic reasoning and interpretable ML communities. It provides a practical, reproducible framework for learning interpretable rule sets from unstructured text. The counterintuitive finding about LLMs' limitations in probabilistic reasoning is an important cautionary note that could steer future research towards hybrid systems with a clearer division of labor, as the authors advocate.

### Suggestions for Improvement
1. **Deepen the Analysis of LLM Inference Failure**: Conduct a detailed error analysis to understand *why* providing rules and weights harms LLM performance. For example, categorize failure cases by rule complexity, weight magnitude, or conflict between rules. Consider experiments with simplified/improved prompts for the E2-E4 strategies to see if the gap can be closed.
2. **Strengthen the Baseline Section and Address Task Difficulty**: Include a more comprehensive suite of baselines, such as other LLM-based prompting techniques (e.g., Chain-of-Thought) or a simple "LLM-as-judge" majority vote over rules. Discuss the datasets' difficulty more explicitly—perhaps report human performance or ceiling estimates to contextualize the absolute scores.
3. **Quantify Sensitivity to LLM Judgment Noise**: Evaluate the robustness of the logistic regression weights to perturbations in the ternary feature vectors (Φ). A simple experiment could involve adding synthetic noise to these judgments or using a different LLM for the judging step to see how performance varies. This would strengthen the claim about the reliability of the overall pipeline.
4. **Clarify Computational and Cost Analysis**: The paper uses large commercial LLMs (GPT-4o-mini, DeepSeek-V3, Qwen3-235B) for both generation and judgment. A discussion of the API cost, latency, and a comparison of performance vs. cost across different backbones would be valuable for practitioners considering adoption.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation study on iterative refinement vs. random example selection.** The paper claims iterative refinement via hard examples improves rule quality, but no experiment compares it to using random examples for rule generation in each iteration. Without this, it's unclear whether the "hard example" selection is driving improvement or if any additional data would suffice.

2. **Comparison to a simple logistic regression on LLM-generated rule judgments without refinement.** The core claim is that the full RLIE pipeline (generation + logistic regression + iterative refinement) is superior. A baseline where logistic regression is applied to an initial set of LLM-generated rules (without refinement) is missing. This would isolate the contribution of the iterative loop.

3. **Evaluation on tasks with more complex, compositional rule structures.** The six binary classification tasks may not require sophisticated rule interactions. Testing on tasks where rules are inherently interdependent (e.g., multi-hop reasoning, hierarchical classification) is needed to substantiate claims about learning a "collaborative rule set."

4. **Sensitivity analysis on the rule capacity (H) and the number of hard examples (k).** The paper fixes H=10 and k=20. The impact of these critical hyperparameters on performance, rule diversity, and overfitting is not studied, undermining claims about producing a "compact" set.

5. **Direct comparison to a traditional rule learning method (e.g., decision trees, rule lists) on a tabular version of the same tasks.** The paper argues LLMs overcome predicate limitations, but without showing RLIE's superiority over classical methods on structured data (if available), the added value of LLM-generated natural language rules is not convincingly demonstrated.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of rule diversity and redundancy.** The paper claims the rule set is "compact and semantically clearer," but provides no metrics (e.g., semantic similarity, coverage overlap) to show the rules are non-redundant and cover distinct patterns. Without this, the logistic regression's sparsity may just be deleting redundant rules, not learning meaningful interactions.

2. **Error analysis on why LLM-augmented inference degrades performance.** The key finding is that providing rules/weights to an LLM hurts performance. A systematic analysis of failure cases (e.g., does the LLM ignore weights, misinterpret rule semantics, or contradict the linear prediction?) is essential to understand this limitation and support the "division of labor" claim.

3. **Analysis of the stability and consistency of LLM rule judgments.** The entire framework relies on LLMs to produce ternary judgments (rule applicability). No analysis is provided on the intra- or inter-LLM consistency of these judgments, which is critical for trusting the logistic regression features.

4. **Calibration analysis for the linear combiner vs. LLM-augmented strategies.** The paper discusses calibration but provides no calibration metrics (e.g., ECE, reliability diagrams). Without this, claims about the probabilistic combiner being "robust and calibratable" are unsupported.

### Visualizations & Case Studies
1. **Visualization of rule weight trajectories across refinement iterations.** A plot showing how weights evolve for individual rules as new rules are added and hard examples are addressed would visually demonstrate refinement and global selection.

2. **Case studies contrasting successful linear-only predictions vs. failed LLM-augmented predictions.** Concrete examples where the linear model is correct but the LLM (given rules, weights, and the linear prediction) makes an error would powerfully illustrate the LLM's inability to integrate probabilistic signals.

3. **Heatmap of rule activations (judgments) on misclassified examples.** Showing which rules fire (and how) on hard examples before and after refinement would reveal whether the iterative process actually closes coverage gaps.

### Obvious Next Steps
1. **Human evaluation of rule interpretability and utility.** Since interpretability is a core motivation, the paper should have included a human study where domain experts rate the generated rules for clarity, correctness, and usefulness for decision-making.

2. **Extension to multi-class classification.** The framework is presented for binary tasks only. A natural next step is to adapt it to multi-class settings (e.g., using multinomial logistic regression), which would significantly broaden applicability.

3. **Experiments with cheaper/smaller LLMs for rule generation and judgment.** The paper uses large, proprietary models (GPT-4o-mini, Qwen3-235B). Testing with smaller open-source LLMs is necessary to assess the framework's cost-effectiveness and accessibility.

4. **Exploration of a more expressive global combiner (e.g., GAMs) as mentioned in the discussion.** The discussion suggests extending the linear combiner, but no experiments are conducted. Implementing at least one such variant (e.g., a GAM) and comparing performance would strengthen the claim that the framework is extensible.

# Final Consolidated Review
## Summary
This paper introduces RLIE, a unified framework that integrates Large Language Models for generating natural language rules with a logistic regression model for probabilistically weighting and selecting those rules. It includes iterative refinement based on prediction errors and a systematic evaluation comparing direct inference using the weighted rule set against various methods of injecting rules back into an LLM. A key empirical finding is that the direct logistic regression classifier ("Linear-only") consistently outperforms strategies where the LLM is prompted with the rules and their weights.

## Strengths
- **Novel and well-motivated hybrid framework:** The paper clearly identifies a gap in existing LLM-based rule learning—the lack of principled, probabilistic combination of multiple rules—and proposes a cohesive four-stage pipeline (Rule generation, Logistic regression, Iterative refinement, Evaluation) that integrates LLMs' generative strengths with classical, calibratable probabilistic modeling.
- **Rigorous and insightful evaluation of inference strategies:** The paper designs a structured hierarchy of inference methods (E1-E4), providing strong empirical evidence for the central, counterintuitive finding that LLMs struggle with fine-grained probabilistic integration of weighted rules. This is a valuable contribution with practical implications for neuro-symbolic system design.

## Weaknesses
- **Unfair and inconsistent experimental comparisons:** Table 1 shows that RLIE is evaluated using powerful backbones like Qwen3-235B and DeepSeek-V3, while key baselines (Zero-shot, IO Refinement, HypoGeniC) are evaluated using DeepSeek-V3. This inconsistency in model capability confounds the comparison and undermines claims about the framework's superiority. The performance advantage could be attributed to the stronger model rather than the method.
- **Superficial analysis of the core negative finding:** While the result that LLM-augmented inference underperforms the linear classifier is striking, the analysis remains high-level. The paper lacks a deeper investigation (e.g., error categorization, analysis of rule/weight patterns that mislead the LLM) into *why* providing rules and weights degrades LLM performance, missing an opportunity to provide more actionable insights into LLMs' reasoning limitations.

## Nice-to-Haves
- A quantitative ablation study isolating the contribution of the iterative refinement component versus random example selection.
- A sensitivity analysis of key hyperparameters like rule capacity (`H`) and the number of hard examples (`k`).
- An exploration of a more expressive global combiner (e.g., a Generalized Additive Model as mentioned in the discussion) to validate the framework's extensibility.
- Human evaluation of the generated rules' interpretability, given that it is a core motivation.

## Novel Insights
The paper's primary novel insight is the demonstration that a clear division of labor is effective: LLMs excel at the semantic tasks of generating and judging individual natural language rules, but a transparent, classic probabilistic model (logistic regression) is superior for the global task of weighting and combining those rules. The counterintuitive finding—that providing an LLM with explicit rule weights and even the linear model's correct prediction often degrades its performance—provides concrete, empirical evidence for LLMs' limitations in controlled, fine-grained probabilistic integration, steering the neuro-symbolic reasoning community toward more robust hybrid architectures.

## Suggestions
- Re-run the primary baseline comparisons (IO Refinement, HypoGeniC) using the *same* LLM backbone as used for RLIE's rule generation to ensure a fair assessment of the framework's contribution.
- Conduct a detailed error analysis on the test set to categorize and understand the failure modes of the LLM-augmented inference strategies (E2-E4), providing concrete examples of where and how the LLM's reasoning diverges from the linear model's correct judgment.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
