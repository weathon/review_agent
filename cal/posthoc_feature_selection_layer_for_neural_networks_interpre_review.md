=== CALIBRATION EXAMPLE 32 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title and abstract clearly state the paper's goal: a *post-hoc* variant of the Feature Selection Layer (FSL) for interpretability of pre-trained models. The abstract claims advantages over other post-hoc methods in terms of "visual and clustering-based interpretability" (weighted t-SNE, silhouette score) while acknowledging a trade-off in some stability metrics (Jaccard, Spearman, Pearson). This sets up a clear, testable contribution. A minor concern is the phrase "maintaining the predictive power" – the method is applied to a *frozen* model, so predictive power is, by design, not altered (though the weighted input could theoretically affect it). The abstract should more precisely state that the method aims to interpret the model's existing function without degrading it.

**Introduction & Motivation**
The motivation is well-established, highlighting the need for interpretability in high-stakes domains like healthcare. The gap identified—that the original FSL is an *embedded* method requiring training from scratch and thus cannot be applied post-hoc to existing models—is clear and valid. The proposed solution (a post-hoc FSL) is directly motivated by this gap. The contributions, however, could be stated more explicitly and formally in this section. Currently, they are somewhat woven into the narrative. For ICLR, a bulleted or bolded list of contributions (novel method, comprehensive evaluation against SOTA, analysis on tabular/HDLSS data) would strengthen the paper.

**Method / Approach**
The proposed method is simple and clearly described. The core idea is to insert a trainable, dense layer with ReLU activation and L1 regularization between the input and a frozen pre-trained model. The training procedure (only the new layer's weights are updated) is standard for adapter-based fine-tuning.

**Key Concerns:**
1.  **Theoretical Justification & Novelty:** The methodological novelty is relatively modest. Adapting a linear feature-weighting layer to function in a post-hoc setting is a straightforward engineering adaptation of the original FSL concept. The paper would benefit from a deeper discussion of *why* this simple approach works, or what its theoretical properties are compared to gradient-based attribution methods. Is it approximating a specific form of explanation? The lack of a strong theoretical grounding is a potential weakness for ICLR.
2.  **Hyperparameters & Sensitivity:** The method introduces hyperparameters: the strength of L1 regularization (λ) and the learning rate for training the new layer. The paper does not discuss how these were chosen or their impact on the resulting feature weights and stability. For a robust and reproducible method, an ablation study on these hyperparameters is important, especially since stability is a noted weakness.
3.  **Class-Specific Explanations:** The method produces a single, global set of feature weights. As correctly noted in Appendix D (Limitations), this is a major drawback compared to many post-hoc attribution methods (e.g., Integrated Gradients, SHAP) that can produce instance-specific or class-specific explanations. This limitation should be highlighted earlier, perhaps in the Method section, not just in the appendix. It significantly narrows the scope of applicability.
4.  **Equation 1:** There is a formatting artifact: the regularization term `r(W^FSL) = λ * Σ|w_i - 1/n| / n` appears garbled. The intended formula seems to be from the original FSL paper, but it's not clear if this exact form is used in the post-hoc version (which uses standard L1). This should be clarified.

**Experiments & Results**
This is the most critical section. The evaluation is extensive, using synthetic and real-world tabular datasets and a diverse set of metrics (predictive, selection accuracy, visual, stability).

**Strengths:**
*   Use of synthetic data with ground truth (PIFS, PSFI) is excellent for direct evaluation.
*   Inclusion of stability analysis via k-fold validation is a good practice.
*   Testing on a real pre-trained model (TabPFN) is a valuable addition that strengthens the "post-hoc for any model" claim.

**Major Concerns:**
1.  **Baseline Comparison & Goal of Interpretability:** Table 5 shows that the **Baseline** model (no FSL) often has the best predictive performance. For the Spam dataset, the post-hoc FSL even slightly outperforms the baseline. This raises a critical question: **If the primary goal is *interpretability*, why is the post-hoc FSL sometimes changing the model's predictions?** The frozen network's parameters are fixed, but the weighted input `w⊙x` changes the effective input distribution. The paper claims the method is "non-invasive," but if the accuracy/F1 changes, it is technically altering the model's function on the given input. The authors must clarify the objective: Is it to (a) explain the *original* model's predictions faithfully, or (b) create a new, *more interpretable model* that approximates the original one? Most post-hoc attribution methods aim for (a). This result suggests the post-hoc FSL may be doing (b), which is a different, though still valid, contribution. This distinction must be explicitly addressed.
2.  **Cherry-Picking Results & Contribution Clarity:** The results are mixed. On SynthA, post-hoc FSL is clearly worse than FSL and other post-hoc methods in predictive performance, selection accuracy (PIFS/PSFI), and stability (Table 3, 4, 8). The authors attribute this to the dataset's structure. On real-world datasets, it shows good visual metrics (silhouette score) but poor stability (Table 9). The abstract and conclusion emphasize the superiority in "visual and clustering-based interpretability." However, **weighted t-SNE and silhouette score are highly indirect, specialized metrics for interpretability.** The community standard for evaluating feature attribution is typically based on ground truth (for synthetic data) or fidelity measures like insertion/deletion curves (for real data). The heavy reliance on a visual metric weakens the claim of "superior interpretability." The paper needs to justify why improved silhouette score on a weighted t-SNE projection is a better measure of interpretability than, e.g., higher correlation with ground truth features (where it often underperforms).
3.  **Statistical Significance:** The tables mention statistical significance tests (Kruskal-Wallis, Dunn's) but only mark "best" results in orange. It's unclear if the post-hoc FSL's advantages in silhouette scores are statistically significant compared to all other methods. This should be explicitly reported.
4.  **Stability Results:** The very low stability scores (Spearman ~0.3 for Spam, Table 9) for post-hoc FSL are a serious concern. If the feature rankings are not stable across data resamples, the reliability of the interpretation is questionable. The discussion in Appendix D is good, but this fundamental weakness should be more prominently discussed in the main results and limitations.
5.  **Missing Ablation:** An important ablation is missing: **What is the effect of the L1 regularization strength (λ)?** A key claimed benefit is sparsity (highlighting top features). The degree of sparsity and its impact on the various metrics should be analyzed.

**Writing & Clarity**
Overall, the paper is well-structured and clear. The figures, despite some formatting artifacts from the parser, seem intended to show t-SNE visualizations. A significant clarity issue is in **Section 5.1, discussing TabPFN**. The text states "both post-hoc techniques successfully identified the most relevant features, yielding identical PSFI and PIFS scores of 0.966". However, it then says both methods made the same error (overweighting a noisy feature and underweighting an informative one). This contradiction (0.966 score but mis-ranking) is confusing and needs clarification. It seems the top-30 set had good coverage but the *ranking* within that set was imperfect.

**Limitations & Broader Impact**
The limitations section is too brief and placed in the conclusion. It correctly identifies key issues: instability in multi-class settings (global vs. local explanations) and limitation to tabular data. However, it misses the major limitation discussed above: the potential alteration of the model's predictive function and the consequent ambiguity in the interpretability goal (faithfulness vs. approximation). The broader impact statement is generic but acceptable.

### Overall Assessment
This paper presents a straightforward and practically useful adaptation of the Feature Selection Layer for post-hoc interpretation of tabular neural networks. The experimental evaluation is comprehensive in breadth, covering multiple datasets and metrics. However, the **contribution is significantly undermined by critical issues in the experimental narrative and evaluation methodology.** The core finding—that the method improves silhouette scores on weighted t-SNE—is not strongly persuasive as a primary measure of interpretability, especially when the method underperforms on standard selection accuracy (PIFS/PSFI) and stability metrics in several cases. The most serious concern is the ambiguity about whether the method explains the original model or creates a new, slightly different one, as evidenced by changes in predictive metrics. For ICLR, where novelty, theoretical grounding, and rigorous evaluation are paramount, the paper in its current form likely falls below the acceptance bar. **Major revisions** are required to: 1) Clarify and defend the core objective (faithfulness vs. approximation), 2) Provide a more convincing and standard evaluation of attribution quality beyond visual metrics, 3) Include hyperparameter sensitivity and ablation studies, and 4) Discuss the stability weaknesses and their implications more thoroughly in the main text.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a post-hoc adaptation of the Feature Selection Layer (FSL) to enhance the interpretability of pre-trained neural networks on tabular data. The method attaches a lightweight, trainable layer with one weight per input feature, fine-tunes only these weights while keeping the original model frozen, and uses the learned weights as feature importance scores. The approach is evaluated on synthetic and real-world datasets using predictive performance, feature selection accuracy (where ground truth exists), visual metrics (weighted t-SNE and silhouette score), and stability metrics, and is compared to several existing post-hoc attribution methods and the original embedded FSL.

### Strengths
1. **Practical and Non-Invasive Contribution**: The method addresses a real-world need by enabling interpretability of already deployed models without altering their parameters. The post-hoc approach is lightweight, easy to implement, and preserves model integrity, which is valuable for high-stakes domains like healthcare.
2. **Comprehensive and Multifaceted Evaluation**: The paper employs a broad set of evaluation metrics, including predictive performance (accuracy, F1, etc.), feature selection accuracy (PIFS, PSFI on synthetic data), visual interpretability (weighted t-SNE, silhouette score), and stability (Jaccard, Spearman, Pearson). This thorough analysis across multiple dimensions is a significant strength.
3. **Clear Methodological Description**: The architecture (a dense layer with one-to-one mapping, ReLU activation, L1 regularization), training procedure (freezing pre-trained model, updating only FSL weights), and design choices (initialization to 1.0) are clearly explained, enhancing reproducibility.
4. **Strong Reproducibility Commitment**: The paper provides a detailed reproducibility statement with links to code, datasets, and dependencies, aligning well with ICLR’s standards and facilitating future research.

### Weaknesses
1. **Limited Novelty and Positioning**: The core idea of attaching a trainable linear layer to a frozen model for feature attribution is straightforward and not fundamentally novel. While adapting FSL to a post-hoc setting is new, the paper does not sufficiently differentiate it from other post-hoc feature weighting or surrogate modeling approaches (e.g., LIME, SHAP surrogates), nor does it compare to methods that also learn explainers via fine-tuning.
2. **Mixed Empirical Results**:
   - On the SynthA dataset, post-hoc FSL underperforms in predictive metrics, stability, and visual separability compared to other methods (Tables 3, 4, 2). The authors hypothesize dataset structure as a cause but provide no deeper analysis, weakening confidence in the method’s robustness.
   - Stability metrics (especially Spearman and Jaccard) for post-hoc FSL are often lower than for competing methods, particularly on real-world data (Table 9). The trade-off between stability and visual metrics is noted but not adequately explained or justified.
   - The method is limited to global feature importance and performs less effectively on multi-class problems (Breast dataset), where class-specific attributions are often needed. This is acknowledged as a limitation, but the evaluation does not include class-specific analyses or adaptations.
3. **Lack of Theoretical Justification**: There is no theoretical motivation or analysis for why optimizing the FSL weights against the frozen model’s output should yield meaningful feature importances. A discussion of convergence properties, connections to gradient-based methods, or identifiability conditions would strengthen the methodological foundation.
4. **Evaluation Metrics and Significance**:
   - The use of weighted t-SNE and silhouette score as primary interpretability metrics is non-standard and their direct relevance to feature importance is not thoroughly justified. While visually intuitive, they may not capture attribution correctness as directly as ground-truth-based metrics.
   - Statistical significance testing (Kruskal-Wallis and Dunn’s tests) is mentioned but results are only highlighted in tables without reporting p-values or effect sizes, making it difficult to assess the robustness of claimed advantages.

### Novelty & Significance
The novelty is moderate: repurposing an embedded feature selection layer for post-hoc interpretation is a new application, but the underlying concept of learning linear weights to explain a frozen model is not groundbreaking. The significance lies in its practicality for interpreting existing models in tabular domains, especially where non-invasiveness is critical. However, the mixed empirical performance and limitations in stability and multi-class settings may constrain its broader impact. For ICLR, the paper presents a solid incremental contribution but falls short of presenting a major algorithmic or theoretical advance.

### Suggestions for Improvement
1. **Enhance Novelty and Related Work**: Clearly differentiate the approach from other post-hoc methods that learn explainers (e.g., surrogate models, attention-based explainers) and justify why this simple linear adaptation is advantageous. Discuss potential connections to influence functions or linear probes.
2. **Deeper Analysis of Results**:
   - Investigate why post-hoc FSL underperforms on SynthA—consider optimization challenges (e.g., loss landscape of frozen model), hyperparameter sensitivity, or dataset characteristics. Ablation studies on initialization, regularization, and activation functions could provide insights.
   - For multi-class problems, propose and evaluate an extension to produce class-specific weights (e.g., via multiple FSL heads or a modified loss) and compare to class-wise attributions from other methods.
3. **Theoretical Foundation**: Provide a theoretical analysis, even if simplified (e.g., under linear model assumptions), linking the learned weights to feature importance. Discuss convergence guarantees or interpretability guarantees (e.g., fidelity to the frozen model’s predictions).
4. **Strengthen Evaluation**:
   - Include standard feature selection evaluation such as model performance when retraining using only top-ranked features, or correlation with domain expert judgments for real-world data.
   - Report full statistical test results (p-values, effect sizes) for all comparative metrics, not just predictive ones, to substantiate claims of superiority.
   - Expand experiments to more diverse tabular datasets (e.g., with categorical features, larger sample sizes) and consider runtime efficiency comparisons.
5. **Improve Clarity and Presentation**:
   - Ensure final version cleans up parser-induced formatting artifacts (broken equations, garbled tables). Improve figure readability (e.g., Figure 8) and provide more descriptive captions.
   - Clarify how top-n features are selected for Jaccard index in the absence of ground truth, and justify the choice of n (e.g., via thresholding or variance explained).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare with permutation importance and KernelSHAP as standard baselines for tabular data.** The paper only uses gradient-based methods and Feature Ablation. Without comparison to these fundamental, model-agnostic techniques, the claimed advantage of post-hoc FSL is not convincing.
2. **Ablation study on initialization, activation, and regularization choices.** The paper sets weights to 1.0, uses ReLU and L1 without justification. Showing how these design choices affect performance and stability is necessary to validate the method’s design.
3. **Test on tabular datasets with categorical features and missing values.** All used datasets are fully numerical without missing values. The method’s applicability to realistic tabular data—a core claim—remains unverified, undermining its practical utility.
4. **More synthetic datasets with varied complexities (e.g., linear, non-linear, correlated features).** Relying only on XOR and SynthA is insufficient to demonstrate robust feature recovery. Additional controlled experiments are needed to assess sensitivity to data characteristics.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify how the FSL layer alters the pre-trained model’s predictions.** The claim of “non-invasive” interpretation requires showing that predictions with and without the FSL layer remain highly consistent (e.g., via accuracy/agreement metrics). Without this, the method may change model behavior, compromising its fidelity.
2. **Analyze the failure mode on the SynthA dataset.** Post-hoc FSL underperforms significantly on SynthA compared to other methods. A thorough investigation (e.g., examining gradient flow, loss landscape, or dataset structure) is needed to explain when and why the method fails.
3. **Investigate the root cause of low stability metrics.** Post-hoc FSL shows poor Jaccard/Spearman stability. Analysis should determine if this is due to optimization instability, data sensitivity, or the global weighting scheme, and whether it materially affects interpretability.

### Visualizations & Case Studies
1. **Show actual features selected for real-world datasets with domain relevance.** For the spam dataset, list high-weight words; for microarray data, list top genes and discuss biological plausibility. Without this, the claim of “identifying relevant features” is not substantiated.
2. **Visualize per-sample feature attributions versus other methods on critical examples.** Case studies showing how attributions differ between post-hoc FSL and, e.g., SHAP on specific instances would reveal whether the method produces coherent local explanations or erratic ones.

### Obvious Next Steps
1. **Extend to class-specific feature importance for multi-class problems.** The paper notes this as a limitation but does not attempt a solution (e.g., training separate FSL heads per class). Given the Breast dataset is multi-class, this adaptation is a necessary next step for the method’s generality.
2. **Apply to a wider range of pre-trained model architectures.** The experiments use simple custom networks. Testing on diverse pre-trained models (e.g., ResNets for tabular, or more complex transformers) would demonstrate broader applicability.
3. **Evaluate the impact on model robustness (e.g., adversarial susceptibility).** Adding the FSL layer could change the model’s sensitivity to input perturbations. Assessing this is important for deployment in high-stakes domains.

# Final Consolidated Review
## Summary
This paper introduces a post-hoc variant of the Feature Selection Layer (FSL) for interpreting pre-trained neural networks on tabular data. The method attaches a trainable linear layer with one weight per input feature, fine-tunes only these weights while keeping the original model frozen, and uses the learned weights as feature importance scores. Evaluation on synthetic and real-world datasets compares the approach to embedded FSL and several post-hoc attribution methods using predictive, feature selection, visual, and stability metrics.

## Strengths
- **Practical, non-invasive adaptation**: The method enables interpretability for already deployed models without modifying their parameters, addressing a real need in high-stakes domains like healthcare. This is evidenced by the clear architectural description and frozen training procedure.
- **Comprehensive evaluation**: The paper assesses performance across multiple dimensions—predictive metrics (accuracy, F1), feature selection accuracy (PIFS, PSFI on synthetic data), visual interpretability (weighted t-SNE, silhouette score), and stability (Jaccard, Spearman, Pearson)—providing a thorough analysis of the method's behavior.
- **Strong reproducibility**: Code, datasets, and dependencies are openly provided, facilitating replication and future work.

## Weaknesses
- **Incremental novelty and lack of theoretical grounding**: The adaptation of FSL to a post-hoc setting is straightforward, and the paper does not provide theoretical justification for why optimizing these weights yields meaningful feature importance, limiting its conceptual contribution.
- **Faithfulness to the original model is compromised**: The method scales input features, which alters the model's predictions (e.g., Table 5 shows changed accuracy/F1 scores). This raises concerns about whether it explains the original model or creates a new approximation, undermining its post-hoc interpretability claim.
- **Overreliance on non-standard visual metrics**: The primary advantages are claimed based on weighted t-SNE and silhouette scores, which are indirect and not established standards for evaluating feature attribution. The paper does not sufficiently justify why these metrics are preferable to ground-truth correlation or fidelity measures like insertion/deletion curves.
- **Poor stability and unstudied hyperparameter sensitivity**: Post-hoc FSL exhibits low stability scores (e.g., Spearman ~0.3 on Spam data, Table 9) compared to baselines, and the impact of key hyperparameters (e.g., L1 regularization strength λ) on performance and sparsity is not analyzed, affecting reproducibility and reliability.
- **Inconsistent performance without deep analysis**: The method underperforms on the SynthA dataset in predictive metrics, selection accuracy, and stability (Tables 3, 4, 8), with only a superficial hypothesis provided. This lack of root-cause analysis weakens confidence in its robustness.

## Nice-to-Haves
- Comparison to additional standard baselines like permutation importance or full Kernel SHAP (beyond the limited TabPFN experiment).
- Testing on tabular datasets with categorical features or missing values to broaden applicability claims.
- Visualization of top-ranked features for real-world datasets (e.g., high-weight words for spam) to concretely demonstrate relevance.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- Criticism about garbled Equation 1 formatting: this is a parser artifact, not a paper flaw.
- Claim of contradiction in the TabPFN discussion: the text consistently reports high but imperfect selection scores with ranking errors, not a contradiction.
- Generic strengths like "the paper is well-written" or "the topic is important" are omitted as they apply broadly.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct ablation studies to analyze the impact of hyperparameters (e.g., λ, initialization) on feature weights, sparsity, and stability.
- Provide full statistical significance details (p-values, effect sizes) for all comparative metrics to substantiate claims of superiority.
- Explore extensions to produce class-specific feature weights for multi-class problems, addressing a noted limitation.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
