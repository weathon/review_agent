=== CALIBRATION EXAMPLE 26 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly indicates the contribution: a post-hoc adaptation of a Feature Selection Layer for interpretability. The abstract effectively summarizes the method, motivation, and key results. However, it presents a claim of "distinct advantages over other state-of-the-art methods," which seems slightly overreaching given the mixed results presented later (e.g., lower stability metrics, inconsistent performance on SynthA). The abstract should more accurately reflect the trade-offs observed.

### Introduction & Motivation
The introduction successfully motivates the problem of interpretability in high-stakes domains and the limitation of existing embedded methods like FSL. The core contribution—a post-hoc variant that works with frozen, pre-trained models—is clearly stated. The scope (tabular data, post-hoc interpretability) is well-defined. A minor point: the claim that the method might "improve predictive performance" is not strongly supported by the results (performance is largely maintained, not improved).

### Method / Approach
The proposed method is simple and clear: attach a trainable, dense layer with ReLU activation and L1 regularization before a frozen pre-trained model and optimize its weights. This raises significant conceptual and technical questions:
1.  **Novelty Gap:** The method is essentially fine-tuning a single new input layer. How does this differ conceptually from performing a few steps of gradient descent on the first layer of a network while freezing the rest? The distinction from simply interpreting the weights of a first layer after fine-tuning is unclear.
2.  **Mechanism Justification:** The authors state the final weights "represent feature importance." However, optimizing a layer to minimize loss on a frozen model's output could simply learn to scale inputs to match the pre-trained model's expected distribution. A stronger justification is needed for why these learned scaling factors correlate with *feature importance for the original model's reasoning* rather than being a compensatory transformation. An ablation studying the effect of the FSL on intermediate representations would be insightful.
3.  **Multi-class Limitation:** The authors correctly note the limitation for multi-class problems in Section 6.1, as the method produces a single global weight per feature. This is a fundamental drawback compared to attribution methods that can produce per-class explanations. This limitation should be acknowledged earlier (e.g., in the Method section) as it defines the scope of the claimed contribution.

### Experiments & Results
The experimental design is extensive, using synthetic and real-world datasets and a broad suite of metrics (predictive, selection accuracy, visual, stability). However, several issues weaken the claims:
1.  **Mixed Results Undercut Claims:** The central claim is that post-hoc FSL offers advantages. Yet, on the SynthA dataset, it is consistently outperformed by the original FSL and all other post-hoc methods in predictive performance, feature selection accuracy (PIFS/PSFI), stability, and visual metrics (Table 3, 4, 8, Figure 5). The authors hypothesize the dataset structure is problematic but do not investigate this further. This significant failure case needs deeper analysis, not dismissal. It suggests the method may not be robust.
2.  **Stability is a Major Weakness:** Across all datasets, the stability metrics (Jaccard, Spearman, Pearson) for post-hoc FSL are often the worst or among the worst (Tables 1, 4, 9). For the Spam dataset, its Spearman correlation is ~0.3, indicating very low rank-order consistency across folds. The authors argue it still identifies "meaningful" features, but low stability fundamentally undermines the reliability of any single explanation produced by the method—a critical concern for interpretability in high-stakes domains. This major drawback is not given enough weight in the discussion.
3.  **Evaluation Metric Choice:** The use of weighted t-SNE and silhouette score as primary "interpretability metrics" is unconventional. While these may measure cluster separability under a specific weighting, they are several steps removed from standard evaluations of feature attribution (e.g., faithfulness, robustness, alignment with ground truth). The superior performance on these metrics is noted, but their direct relevance to *interpretability* is not thoroughly justified. Are well-separated clusters in a weighted t-SNE plot a valid proxy for a good explanation?
4.  **Baseline Comparison:** The comparison against strong post-hoc baselines (Integrated Gradients, SHAP, etc.) is appropriate. However, the paper lacks a discussion on the *computational cost* comparison. Training the post-hoc FSL layer likely requires more forward/backward passes than a single backward pass for gradient-based methods. This trade-off should be analyzed.
5.  **Statistical Significance:** The use of statistical tests (Kruskal-Wallis, Dunn's) is good practice. However, the results (marked in orange) often show the *baseline* or *FSL* as the best, not post-hoc FSL. This pattern doesn't support the claim of superiority.

### Writing & Clarity
The paper is generally well-structured and clear. The figures, while suffering from parser artifacts, appear intended to be informative. The flow from motivation to method to results is logical. The limitations section is appropriately placed but could be more forthright about the core weaknesses identified above.

### Limitations & Broader Impact
The stated limitations (multi-class, tabular-only data) are appropriate. The broader impact statement is minimal but acceptable. The critical limitation regarding **poor stability** is buried in Appendix D and not featured prominently in the main limitations section. This should be a primary, upfront limitation. The societal impact of using an unstable interpretability method in healthcare (a mentioned domain) could be negative, as it may provide inconsistent justifications for decisions.

### Overall Assessment
The paper presents a straightforward and intuitively appealing idea: making an embedded feature selection method work post-hoc. The experiments are comprehensive in scope. However, the work faces a high novelty bar for ICLR. The core method is a simple adaptation, and the empirical evidence presents a mixed picture: while the method works well on some datasets (XOR, real-world microarrays), it shows clear and significant weaknesses on others (SynthA) and exhibits critically low stability across the board. The reliance on non-standard metrics (weighted t-SNE silhouette) to claim superiority further complicates the assessment. For ICLR, where a significant and robust advance is expected, the contribution is likely insufficient. The paper identifies an interesting application but does not convincingly demonstrate that the proposed method is a reliable, stable, and conceptually novel improvement over existing post-hoc attribution techniques.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a post-hoc adaptation of the Feature Selection Layer (FSL) for interpreting pre-trained neural networks on tabular data. The method attaches a lightweight, trainable layer in front of a frozen model to learn global feature importance weights via fine-tuning. The primary contributions are a non-invasive interpretability technique and an extensive empirical evaluation against multiple post-hoc baselines using statistical, visual, and stability metrics on synthetic and real-world datasets.

### Strengths
1.  **Clear Practical Motivation:** The paper addresses a genuine need for interpreting already-deployed ("frozen") models in high-stakes domains like healthcare, positioning the work within a relevant and important problem space.
2.  **Comprehensive Empirical Evaluation:** The experimental design is thorough, using diverse datasets (synthetic, spam, microarray) and a wide array of evaluation metrics (predictive performance, PIFS/PSFI, silhouette scores, weighted t-SNE, Jaccard/Spearman/Pearson stability). The inclusion of a pre-trained model (TabPFN) for an additional experiment adds credibility.
3.  **Effective Visual Interpretation:** The method demonstrates strong performance on visual interpretability metrics (weighted t-SNE and silhouette score), often matching or exceeding state-of-the-art post-hoc methods on real-world datasets, which is a concrete and useful result.

### Weaknesses
1.  **Limited Scope and Generality:** The method is explicitly designed for and evaluated only on tabular data. The authors note it performs poorly on multi-class problems (as shown in the Breast dataset stability analysis) and is not applicable to modalities like images. This significantly limits its applicability compared to more general post-hoc attribution methods.
2.  **Poor Stability Performance:** The stability metrics (Jaccard, Spearman, Pearson) for the proposed method are frequently the worst or among the worst across experiments (e.g., Tables 4 and 9). This indicates that the feature rankings are highly sensitive to data perturbations, which undermines reliability—a critical concern for interpretability methods. The analysis in Appendix D acknowledges this but doesn't resolve it.
3.  **Insufficient Theoretical Grounding and Novelty:** The core idea—adding a trainable input layer to a frozen network—is simple and derived directly from the original embedded FSL. The paper lacks a deep theoretical analysis of why this post-hoc adaptation should work or how its learned weights relate to other attribution concepts (e.g., Shapley values, gradients). The novelty is therefore incremental.
4.  **Missing Computational Analysis:** While claimed to be "lightweight," no computational cost or runtime comparisons with other post-hoc methods (which can be expensive) are provided. This is a missed opportunity to highlight a potential practical advantage.

### Novelty & Significance
**Novelty:** Moderate. The adaptation of an embedded feature selection layer (FSL) to a post-hoc setting is a clear and logical technical contribution, but it is a relatively straightforward extension of prior work.
**Significance:** The method provides a new, model-specific tool for post-hoc interpretability on tabular data. Its strong performance on visual clustering metrics is promising for exploratory data analysis. However, its limitations in stability, multi-class settings, and domain specificity reduce its potential impact compared to more robust and general attribution frameworks.

### Suggestions for Improvement
1.  **Address Stability:** Investigate and propose modifications (e.g., different regularization, training protocols, or ensembling) to improve the consistency (Spearman, Jaccard) of the feature rankings across data splits. This is the most critical weakness.
2.  **Deepen the Analysis:** Include a theoretical discussion or proof sketch on the relationship between the post-hoc FSL weights and other attribution concepts. Analyze the optimization landscape—why does fine-tuning only this layer converge to meaningful weights?
3.  **Expand Scope and Evaluation:** To increase impact, explore initial adaptations or discussions for multi-class scenarios (e.g., per-class weighting) or structured data. Include a computational efficiency benchmark comparing training time against other post-hoc methods.
4.  **Improve Presentation of Visual Results:** The weighted t-SNE figures are central but hard to interpret in the text-only format. In a camera-ready version, ensure these are high-quality and accompanied by clearer captions explaining what "better" separation looks like.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison to strong, simple baselines:** The paper lacks comparisons to model-agnostic feature importance methods like permutation importance or Lasso-based selection. Without this, it's unclear if the added complexity of post-hoc FSL is justified over simpler, established techniques.
2. **Ablation on layer configuration:** The paper changes the FSL's initialization and uses ReLU but does not ablate these design choices against the original FSL setup. This gap undermines the claim that these modifications are necessary or beneficial for the post-hoc setting.
3. **Rigorous multi-class evaluation:** The method's noted instability on multi-class problems is a major limitation. The paper should include a dedicated synthetic multi-class dataset with known feature importance to systematically quantify and analyze this failure mode.
4. **Computational cost analysis:** No comparison of training time or inference overhead relative to other post-hoc methods (e.g., SHAP, Integrated Gradients) is provided. For a "lightweight" method proposed for deployed models, efficiency is a critical practical claim that remains unverified.

### Deeper Analysis Needed (top 3-5 only)
1. **Root cause of instability:** The results show post-hoc FSL has poor Spearman stability versus other methods, especially on real data. The paper only hypothesizes but does not investigate *why* (e.g., is it due to optimization landscape, interaction with the frozen network, or data noise?). This analysis is crucial for trusting the method's reliability.
2. **Meaning of "global" feature weights:** The paper claims post-hoc FSL provides global importance scores. A deeper analysis is needed to show how these single-weight-per-feature attributions relate to, or conflict with, local (instance-specific) importance scores from other methods, especially for multi-class datasets where feature relevance is class-dependent.
3. **Sensitivity to hyperparameters:** The performance depends on the regularization strength (λ) and learning rate. There is no analysis of how sensitive the method's stability and accuracy are to these choices, making reproducibility and robustness questionable.

### Visualizations & Case Studies
1. **Feature weight trajectories during training:** Plotting how the FSL weights evolve over epochs for different features (relevant vs. noisy) would reveal if the method stably converges or oscillates, directly explaining the stability results.
2. **Case study on real data with expert validation:** For a dataset like the Liver microarray, list the top 10 features selected by post-hoc FSL and provide a brief biological justification (from literature) for their relevance to the disease. This would ground the interpretability claim in domain knowledge, not just silhouette scores.

### Obvious Next Steps
1. **Address the multi-class problem directly:** The limitation is acknowledged, but the paper should have proposed and tested a straightforward extension (e.g., using a separate FSL weight vector per class) to demonstrate the method's potential adaptability, rather than leaving it as future work.
2. **Ensemble for stability:** Given the low stability scores, an obvious improvement is to ensemble feature weights from multiple training runs or data perturbations. This simple step should have been implemented and tested to see if it bridges the stability gap with other methods.
3. **Test on a broader class of models:** The experiments use standard feedforward networks and one Transformer (TabPFN). The core claim of generality for "any compatible pre-trained network" should be tested on other architectures common in tabular data (e.g., ResNets, SNNs) to ensure the method is not architecture-sensitive.

# Final Consolidated Review
## Summary
This paper introduces a post-hoc adaptation of the Feature Selection Layer (FSL) to interpret frozen, pre-trained neural networks on tabular data. The method attaches a lightweight, trainable layer with ReLU activation and L1 regularization in front of a frozen model, optimizing its weights to indicate global feature importance. Evaluation across synthetic and real-world datasets shows the method maintains predictive performance and achieves competitive visual cluster separability, but exhibits notably lower stability compared to other post-hoc attribution techniques.

## Strengths
- The work addresses a clear practical need: interpreting already-deployed models without altering their parameters, which is crucial for high-stakes domains like healthcare.
- The experimental evaluation is comprehensive, using diverse datasets (synthetic, spam, microarray) and a broad suite of metrics (predictive, feature-selection accuracy, visual clustering, and stability).
- The method demonstrates strong performance on visual interpretability metrics (weighted t‑SNE and silhouette score), often matching or exceeding other post-hoc methods on real‑world datasets, providing a tangible way to visualize feature importance.

## Weaknesses
- The stability of the feature rankings is critically low, as measured by Spearman and Jaccard correlations across data splits (e.g., Tables 4 and 9). This undermines the reliability of any single explanation—a major concern for interpretability in high‑stakes applications.
- On the SynthA synthetic dataset, the method underperforms the original FSL and other post‑hoc baselines in predictive performance, feature‑selection accuracy (PIFS/PSFI), and stability (Tables 3, 4, 8), indicating a robustness gap that is not fully explained.
- The interpretability evaluation relies heavily on weighted t‑SNE and silhouette scores, which are non‑standard metrics for feature attribution. Their direct connection to explanation quality (e.g., faithfulness, robustness) is not thoroughly justified, making it difficult to assess the method’s interpretability against established criteria.

## Nice-to-Haves
- A computational cost comparison with other post‑hoc methods would strengthen the claim of being “lightweight,” especially since the method requires training.
- An ablation study on design choices (e.g., ReLU activation, weight initialization) would clarify their necessity for the post‑hoc setting.
- Comparing with simple model‑agnostic baselines (e.g., permutation importance) would better contextualize the added value of the proposed approach.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- To address the stability issue, consider ensembling feature weights from multiple training runs or data perturbations, and evaluate whether this closes the gap with more stable baselines.
- For multi‑class datasets, explore a straightforward extension (e.g., learning separate weight vectors per class) to provide class‑specific explanations, as the current global weights may not capture local feature importance.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
