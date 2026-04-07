=== CALIBRATION EXAMPLE 26 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:**
The title "RLIE: RULE GENERATION WITH LOGISTIC REGRESSION, ITERATIVE REFINEMENT, AND EVALUATION FOR LARGE LANGUAGE MODELS" accurately reflects the method's components. The abstract clearly states the problem (overlooking rule combination effects in LLM-based rule learning) and contribution (the RLIE framework). The surprising finding—that injecting weighted rules into the LLM degrades performance—is highlighted, setting up an interesting narrative. However, the abstract could more sharply articulate the *novelty* relative to classical probabilistic rule learning (e.g., Markov Logic Networks) which already combines logic and weights, but uses symbolic predicates. The claim that the framework "paves the way for more reliable neuro-symbolic reasoning systems" is slightly overstated without more concrete evidence of this broader impact in the paper.

**Introduction & Motivation:**
The introduction effectively motivates the need for verifiable, composable theories and contrasts deterministic vs. probabilistic rule aggregation. It clearly identifies the gap: LLMs can generate rules but lack principled, global combination methods. The contributions are listed. A minor weakness is that the related classical work (e.g., Markov Logic Networks, Bayesian Rule Lists) is not directly contrasted; the paper positions RLIE against other LLM-based methods but could more explicitly distinguish its "natural language predicates evaluated by LLM" approach from symbolic predicate-based probabilistic models. The spam detection example is helpful.

**Method / Approach:**
The four-stage pipeline is clearly described and illustrated in Figure 1. However, several aspects lack necessary detail or raise methodological concerns:
1.  **Rule Generation & Ternary Judgment (Section 3.1):** The process where an LLM judges a rule's applicability (outputting -1, 0, +1) is central but underspecified. What prompts are used? How is the LLM guided to produce these three discrete outputs reliably? Potential biases or inconsistencies in this "local judgment" are a critical point of failure for the entire probabilistic aggregation, yet no analysis or validation of this step is provided.
2.  **Logistic Regression (Section 3.2):** Using logistic regression on the ternary features is straightforward. However, a key assumption is that the rule satisfaction indicators \( z_{i,j} \) are *conditionally independent* given the input, or at least that their linear combination is sufficient. This contradicts the motivation that rules need to "work in concert." The Elastic Net regularizer encourages sparsity but does not model interactions. The paper does not discuss this limitation or potential extensions (e.g., including interaction terms) in the method section, though they are mentioned later in the discussion.
3.  **Iterative Refinement (Section 3.3):** The process of selecting hard examples and prompting for new rules is sensible. However, the pruning step ("ranking them based on their individual accuracy on the validation set") seems to contradict the global weighting philosophy. Pruning based on individual accuracy might discard rules that are weak alone but valuable in combination. A pruning strategy based on the regularized logistic weights (e.g., removing rules with zero weight) would be more coherent with the framework.
4.  **Reproducibility:** The method description is mostly reproducible, but the crucial prompts for rule generation, ternary judgment, and refinement are only provided in the appendix for one task. The prompts for the ternary judgment are not shown at all in the main text or appendix snippets (Figures 2-8 show other prompts). Their design is a significant hyperparameter.

**Experiments & Results:**
1.  **Baselines & Comparisons (Section 4.2, 5.1):** The choice of baselines (Zero-shot, IO Refinement, HypoGeniC) is appropriate for comparing LLM-based rule learning. However, a stronger and more relevant baseline is missing: **a standard logistic regression or random forest trained on bag-of-words or sentence embeddings from the same training data.** This would answer: Does the overhead of generating and judging natural language rules provide any benefit over a simple discriminative model on the raw text? The comparison with LoRA fine-tuning is interesting but shown as non-generalizable; a more systematic comparison with in-context learning (ICL) or supervised fine-tuning (SFT) of the LLM itself would strengthen the claim about RLIE's efficiency/value.
2.  **Main Results (Table 1):** The results show RLIE is competitive and often best among rule-learning methods. The variance discussion is good. However, the claim that RLIE "consistently ranks within the top two" is not fully supported by the data (e.g., on Citations, HypoGeniC is significantly better with DeepSeek-V3). The analysis attributing IO Refinement's occasional superiority to single-rule generalization is speculative and not validated.
3.  **Inference Strategy Analysis (Table 2, Section 5.2):** This is a core contribution. The finding that Linear-only (E1) outperforms LLM-augmented strategies (E2-E4) is compelling and well-discussed. However, the experimental design could be tighter: Are the LLMs used for inference the *same* as the one used for rule generation/judgment? If so, there's a potential contamination or bias. Using a different LLM family for inference would make the finding more robust. Also, the prompts for E2-E4 (Figures 6-8) are complex and ask the LLM to "think step by step." The performance drop might be due to prompt design rather than a fundamental LLM limitation. An ablation on prompt simplicity is needed.
4.  **Ablation & Sensitivity:** The parameter study on coverage threshold \( \gamma \) (Appendix Table 4) is good. However, critical ablations are missing: (a) The importance of the iterative refinement loop vs. just initial generation + logistic regression. (b) The impact of the logistic regression's \( L_1 \) penalty (rule selection) versus just using all generated rules.
5.  **Case Study (Appendix B):** The qualitative example is useful but limited. It shows rule evolution but doesn't demonstrate *why* the new rules are better or how the weights reflect collaboration. Showing how the rule satisfaction vectors \( \mathbf{z}_i \) change for hard examples across iterations would be more insightful.

**Writing & Clarity:**
The paper is generally well-written and logically structured. The figures and tables are clear. Some minor clarifications are needed: In Section 3.3, "the union of newly generated rules and old rules can be directly used as \(H^{(t+1)}\), if it doesn’t exceed the capacity limit \(H\)" – this seems to conflict with the later sentence "otherwise we need to discard some rules based on their classification accuracy." Which rule set is pruned: \(H_{tmp}^{(t+1)}\) or the old \(H^{(t)}\)? The pseudocode is ambiguous.

**Discussion & Limitations:**
The discussion (Section 6) is a strength. It thoughtfully analyzes the division of labor between LLMs (semantics/local judgment) and probabilistic combiners (global aggregation). The suggestions for extending the combiner (GAMs, factor graphs, Bayesian regression) are excellent and set a clear research agenda. The limitation that LLMs struggle with fine-grained probabilistic integration is empirically supported.
However, the **stated limitations are too brief.** The paper should explicitly discuss: (1) The computational cost of multiple LLM calls for rule generation, ternary judgment on all training data each iteration, and inference. (2) The dependence on the quality and bias of the underlying LLM's judgments, which are treated as ground truth features. (3) The potential for the iterative refinement to overfit to hard examples, especially with small datasets (N_tr=200). (4) The restriction to binary classification. The broader impact statement is adequate.

### Overall Assessment
This paper presents a well-motivated and empirically solid framework (RLIE) that meaningfully integrates LLMs and probabilistic learning for rule induction. Its core finding—that a simple logistic combiner outperforms using the LLM for weighted rule aggregation—is valuable and thought-provoking for the neuro-symbolic community. However, to meet ICLR's high bar, several methodological gaps need addressing: justifying the rule independence assumption in the logistic model, adding stronger baselines (simple supervised models on text), providing a more rigorous ablation study, and thoroughly analyzing the limitations (computational cost, reliance on LLM judgment fidelity). The discussion is excellent and elevates the work. With revisions that solidify the methodological foundations and experimental rigor, this could be a strong contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes RLIE, a framework that integrates Large Language Models (LLMs) for generating natural language rules with logistic regression for probabilistically weighting and combining those rules. It features an iterative refinement loop that uses prediction errors to improve the rule set. A key empirical finding is that directly using the weighted logistic model for inference outperforms strategies that inject the rules (and their weights) back into an LLM for final prediction, highlighting a limitation in LLMs' ability to perform controlled probabilistic integration.

### Strengths
1. **Novel Integration**: The work presents a clear and novel synthesis of LLM-based rule generation with classical probabilistic modeling (logistic regression with Elastic Net). This addresses a recognized gap in the literature, as prior LLM-based rule learning methods often neglect global rule combination and calibration.
2.  **Comprehensive Evaluation Design**: The paper systematically evaluates four distinct inference strategies (Linear-only, LLM+Rules, LLM+Rules+Weights, LLM+Rules+Weights+Linear Prediction) across six diverse real-world datasets. This hierarchical analysis provides valuable empirical insights into how best to utilize learned rules and reveals the counter-intuitive finding that LLM-augmented inference can degrade performance.
3.  **Methodological Clarity and Reproducibility**: The framework's four-stage pipeline (Rule generation, Logistic regression, Iterative refinement, Evaluation) is clearly described. The paper includes a reproducibility statement, details on prompts (in the appendix), hyperparameters, and comparisons against established baselines (IO Refinement, HypoGeniC), facilitating validation and follow-up work.

### Weaknesses
1. **Limited Scale and Statistical Robustness**: Experiments are conducted on relatively small, fixed-size splits (200 train, 200 validation, 300 test). While useful for proof-of-concept, this scale limits the statistical power of the conclusions and raises questions about the framework's scalability and performance on larger, more complex datasets.
2. **Superficial Analysis of Core Finding**: The paper identifies a significant result—LLM-augmented inference underperforms the simple linear combiner—but offers only speculative, high-level discussion for why this occurs (e.g., LLMs are "less reliable at fine-grained, controlled probabilistic integration"). A deeper analysis, such as error categorization or probing the LLM's reasoning process when given weights, is missing and is critical for a top-tier conference.
3. **Clarity and Presentation Issues**: The writing contains minor errors (e.g., "Large Lange Models" in the abstract) and occasionally overly complex sentences that hinder readability. Some technical details, like the precise mechanism for obtaining the LLM's ternary judgment (`z ∈ {-1, 0, +1}`), are underexplored, leaving ambiguity about potential prompt sensitivity or judgment consistency.

### Novelty & Significance
The core novelty lies in the structured integration of LLMs (for semantic rule generation and local judgment) with a probabilistic global combiner (for weight learning and selection), coupled with an error-driven iterative refinement loop. This represents a meaningful step towards practical neuro-symbolic reasoning systems. The significance is bolstered by the empirical demonstration that a transparent, calibratable model can outperform a powerful LLM at synthesizing its own generated rules, offering an important cautionary note for the field and a clear design principle (division of labor between neural and symbolic components).

### Suggestions for Improvement
1. **Deepen the Analysis of LLM Inference Failure**: Conduct a targeted analysis to understand why providing rules and weights to the LLM hurts performance. This could involve analyzing cases where the linear model is correct but the LLM overrides it, or examining if the LLM misinterprets the weight semantics. This analysis is crucial for the paper's impact.
2. **Strengthen Empirical Evaluation**: Run experiments on at least one larger-scale dataset to better assess scalability. Perform more rigorous statistical testing (e.g., significance tests) on the reported results to bolster claims of superiority and robustness. Include a runtime/efficiency comparison between inference strategies.
3. **Improve Exposition and Technical Detail**: Correct typographical errors and streamline the writing for clarity. Expand the explanation of the "ternary judgment" process in Section 3.1, possibly including an example prompt and discussing the reliability of this step. Move critical implementation details (like the coverage threshold study from Appendix C) into the main body to improve methodological transparency.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare with traditional interpretable models (e.g., decision trees, rule lists, sparse logistic regression on bag-of-words).** Without this, the claim that RLIE produces superior interpretable rule sets is unsubstantiated; it may be outperformed by simpler, established methods.
2. **Ablation study on iterative refinement.** The paper lacks a quantitative assessment of how much refinement contributes to final performance. This is critical to support the claim that the iterative loop is necessary.
3. **Sensitivity analysis on key hyperparameters (rule capacity H, coverage threshold γ).** The fixed choice of H=10 is arbitrary; showing how performance scales with capacity is essential to understand the method's trade-offs and robustness.
4. **Human evaluation of rule interpretability.** The paper claims rules are "semantically clearer" and aid "human-AI consensus," but provides no human assessment. This is a major gap for an interpretability-focused method.

### Deeper Analysis Needed (top 3-5 only)
1. **Failure analysis for LLM-based inference strategies.** The paper observes performance degrades when rules/weights are injected into the LLM but does not analyze why (e.g., instruction-following errors, overwriting correct judgments). This is central to the claim about LLMs' limitations in probabilistic integration.
2. **Analysis of rule selection by logistic regression.** How many rules are typically retained after regularization? What are the characteristics (coverage, individual accuracy) of kept vs. pruned rules? This is needed to validate the claim of learning a "compact" set.
3. **Calibration analysis.** The paper discusses calibration but provides no metrics (e.g., ECE). For a method combining probabilistic weights, assessing whether predicted probabilities are trustworthy is crucial.

### Visualizations & Case Studies
1. **Side-by-side examples of correct/incorrect predictions across inference strategies.** Show cases where the linear model is correct but LLM-based inference fails (and vice versa) to concretely illustrate the strengths/weaknesses of each strategy.
2. **Visualization of rule activation patterns across the dataset.** A heatmap showing which rules fire on which examples would help assess rule diversity, coverage, and potential redundancy.

### Obvious Next Steps
1. **Benchmark against traditional interpretable models.** This is a glaring omission for a paper on rule learning.
2. **Conduct human evaluation of rule quality.** Claims of interpretability and human-AI consensus require empirical validation.
3. **Experiment with more sophisticated probabilistic combiners** (e.g., GAMs, Bayesian logistic regression) mentioned in the discussion, to see if they improve upon the basic logistic regression.

# Final Consolidated Review
## Summary
This paper introduces RLIE, a framework that integrates LLMs for generating natural language rules with logistic regression for global probabilistic weighting and selection, followed by iterative error-driven refinement. A core finding is that using the learned weighted rules directly via logistic regression outperforms injecting them back into an LLM for final inference, suggesting a division of labor where LLMs handle semantic generation/local judgment while classical models manage global probabilistic aggregation.

## Strengths
- **Novel and well-motivated integration:** The framework meaningfully bridges LLM-based natural language rule generation with classical probabilistic combination (logistic regression with Elastic Net), addressing a clear gap in prior work that treated rules independently or used simple aggregation.
- **Systematic and insightful evaluation design:** The paper conducts a hierarchical comparison of four inference strategies, providing strong empirical evidence that the simplest "Linear-only" strategy is most effective. This yields a valuable, counter-intuitive finding about LLMs' limitations in fine-grained probabilistic reasoning.
- **Clear methodological pipeline and discussion:** The four-stage process (Rule generation, Logistic regression, Iterative refinement, Evaluation) is clearly described. The discussion thoughtfully articulates a principled neuro-symbolic division of labor and suggests concrete extensions (e.g., GAMs, factor graphs), setting a clear research direction.

## Weaknesses
- **Limited scale and statistical depth:** Experiments use small, fixed-size splits (200 train/200 validation/300 test), which limits the statistical power of the conclusions and raises questions about scalability and robustness on larger, more complex datasets. Reporting standard deviations is insufficient; statistical significance testing or confidence intervals would strengthen the claims.
- **Superficial analysis of the core finding:** The paper identifies the important result that LLM-augmented inference underperforms the linear combiner but offers only high-level speculation for the cause. A deeper failure analysis (e.g., categorizing error types, examining cases where the LLM overrides a correct linear prediction) is missing and is critical for a top-tier contribution.
- **Underexplored dependency on LLM judgment fidelity:** The entire framework relies on the LLM's ternary judgments (-1,0,+1) as features. The consistency, potential bias, and prompt sensitivity of this critical "local judgment" step are not analyzed, creating a vulnerability in the methodological foundation.

## Nice-to-Haves
- **Comparison with traditional interpretable models:** Including a baseline like logistic regression on bag-of-words or a decision tree would better contextualize the performance and interpretability benefits of the LLM-generated rule paradigm.
- **Human evaluation of rule quality:** Claims about improved interpretability and "human-AI consensus" would be strengthened by a small-scale human study assessing the clarity and utility of the generated rules.
- **Ablation on the iterative loop:** A quantitative ablation showing the performance gain attributable to the iterative refinement stage versus just initial generation + logistic regression would clarify the contribution of each component.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength:** "The paper is well-written" - This is generic and applies to many papers.
- **Weakness (Misreading):** "Prompts for ternary judgment are not provided" - The appendix (Figures 2-8) contains prompts for various stages; the exact ternary judgment prompt can be inferred from the described process and the provided examples.
- **Weakness (Scope Creep/Demanding Non-Standard Practice):** "Missing baseline of supervised fine-tuning (SFT) or in-context learning (ICL) of the LLM" - The paper's scope is rule learning and combination methods; comparing to general-purpose LLM adaptation techniques is a different paradigm.
- **Weakness (Overly Critical/Addressable):** "Logistic regression assumes conditional independence of rule features" - The paper acknowledges this limitation in the Discussion (Section 6) and proposes extensions like GAMs to model interactions, showing awareness and a reasonable path forward.
- **Weakness (Nitpick):** "Pruning based on individual accuracy contradicts global weighting" - The method description states pruning occurs only if the temporary rule set exceeds capacity *H*. Using a simple heuristic like individual accuracy for this contingency is a reasonable engineering choice, not a fundamental flaw.
- **Weakness (Unfair Demand):** "Need for an ablation on prompt simplicity for LLM inference strategies" - The prompts provided are detailed and representative of common practice for complex LLM tasks; demanding a full prompt ablation is beyond standard evaluation expectations.

## Novel Insights
The primary novel insight is the empirical demonstration of a effective division of labor in neuro-symbolic reasoning: LLMs excel at the semantic tasks of generating and locally evaluating natural language rules, but a transparent, calibratable probabilistic model (logistic regression) is superior for the global task of weighting and combining those rules. The finding that providing rules, their learned weights, and even the linear model's prediction to an LLM often degrades performance is a significant cautionary insight for the field, challenging the assumption that LLMs can robustly internalize and act on explicit probabilistic guidance.

## Suggestions
- Conduct a targeted error analysis to understand *why* LLM-augmented inference (strategies E2-E4) fails relative to the linear combiner. Categorize failure modes (e.g., instruction misunderstanding, overwriting correct references) and report examples.
- Include statistical significance testing (e.g., paired t-tests) for the main comparative results in Table 1 to bolster claims of robustness and superiority.
- Expand the limitation section to explicitly discuss the computational cost of multiple LLM calls and the framework's reliance on the underlying LLM's judgment consistency for the ternary feature generation.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 4.0]
Average score: 2.5
Binary outcome: Reject
