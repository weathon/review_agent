# Splines-Based Feature Importance in Kolmogorov-Arnold Networks: A Framework for Supervised Tabular Data Dimensionality Reduction

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Feature selection is a key step in many tabular prediction problems, where multiple candidate variables may be redundant, noisy, or weakly informative. We investigate feature selection based on Kolmogorov-Arnold Networks (KANs), which parameterize feature transformations with splines and expose per-feature importance scores in a natural way. From this idea we derive four KAN-based selection criteria (coefficient norms, gradient-based saliency, and knockout scores) and compare them with standard methods such as LASSO, Random Forest feature importance, Mutual Information, and SVM-RFE on a suite of real and synthetic classification and regression datasets. Using average F1 and $R^2$ scores across three feature-retention levels (20\%, 40\%, 60\%), we find that KAN-based selectors are generally competitive with, and sometimes superior to, classical baselines. In classification, KAN criteria often match or exceed existing methods on multi-class tasks by removing redundant features and capturing nonlinear interactions. In regression, KAN-based scores provide robust performance on noisy and heterogeneous datasets, closely tracking strong ensemble predictors; we also observe characteristic failure modes, such as overly aggressive pruning with an $\ell_1$ criterion. Stability and redundancy analyses further show that KAN-based selectors yield reproducible feature subsets across folds while avoiding unnecessary correlation inflation, ensuring reliable and non-redundant variable selection. Overall, our findings demonstrate that KAN-based feature selection provides a 
powerful and interpretable alternative to traditional methods, 
capable of uncovering nonlinear and multivariate feature relevance 
beyond sparsity or impurity-based measures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose four KAN-based selectors: KAN-L1 and KAN-L2 (based on spline coefficient norms), KAN-SI (sensitivity integral approach), and KAN-KO (knock-out strategy). These methods are systematically evaluated against classical baselines (LASSO, Random Forest, Mutual Information, SVM-RFE) across multiple classification and regression benchmarks. The results demonstrate that KAN-based selectors, particularly KAN-L2, KAN-L1, KAN-SI, and KAN-KO, are competitive with or superior to traditional methods in structured and synthetic datasets. However, KAN-L1 exhibits overly aggressive pruning in regression tasks, while KAN-L2 underperforms in classification tasks with complex feature interactions.

### Strengths
The paper presents the first systematic investigation of KANs for feature selection, introducing four complementary importance measures that leverage spline parameterizations for interpretability. This addresses a significant gap in the literature and offers a fresh perspective on supervised dimensionality reduction.

By utilizing KANs' spline-based architecture, the proposed methods provide direct access to interpretable feature importance measures. This is particularly valuable in domains where understanding feature contributions is critical, such as bioinformatics or finance.

### Weaknesses
Key experimental parameters are inadequately specified. The paper lacks details on KAN architecture choices (e.g., number of layers, spline degree $p$, number of knots $K$), optimization procedures (learning rate, regularization), and computational infrastructure. This hinders reproducibility and makes it difficult to assess whether results generalize beyond the reported settings. The work also lacks comparisons with the KAN-related feature selection methods mentioned in the introduction (Zheng et al. 2025, Wang et al. 2025), while comparing with classical non-KAN methods only fails to verify its ability to address the limitations of existing KAN-based attempts.

The datasets used (e.g., Wine: 178 samples, 13 features; Diabetes: 442 samples, 10 features) are relatively small-scale. There is no evaluation on truly high-dimensional datasets (e.g., genomics with $d > 10^3$ features) or large-sample scenarios ($n > 10^6$), where feature selection is most challenging and computationally demanding. The synthetic datasets (Make Classification/Regression with 500 samples) are also simplistic and may not reflect real-world complexity.

There is no discussion or empirical or theoretical evaluation of computational scalability.

### Questions
Detailed analysis of algorithms or experimental settings is also lacking.

1. Why were 20%, 40%, and 60% chosen as feature retention levels? Did you test other thresholds (e.g., 10%, 30%) and observe meaningful changes in performance? How would your selectors adapt to use cases where the "optimal" retention level is unknown (a common real-world scenario)?

2. What impact do different normalization choices have on KAN-L1 and KAN-L2

Moreover, this paper appears to be incomplete, ending abruptly at "A APPENDIX" without any actual appendix content. It suggests a lack of preparation and consideration for the review process.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a novel KAN-based method for selecting important features and reducing dimensionality in tabular data. By leveraging the notable interpretability advantages of KANs, the authors propose three different feature importance selection approaches: (i) coefficient-based importance of spline weights, (ii) knock-out importance, and (iii) sensitivity-based importance. The authors conduct a wide range of experiments on classification and regression problems to validate the proposed approach.

### Strengths
The motivation for using KANs to select feature importance is well-posed. Combining KAN's features and standard feature importance selection methods is appropriate. The paper is written with sufficient background and is easy to understand its contributions.

### Weaknesses
Using KAN's interpretability and feature importance is not novel and has been explored across different problems [1,2,3]. The experiments rely on outdated datasets, some dating back to 1993; perhaps more relevant or practical datasets could strengthen the paper. The experiences include relatively simple baselines and no comparison with existing KAN-based methods. The performance gains across different approaches are either inconsistent or fail to highlight the advantages of using KANs for feature importance or dimensionality reduction. In particular, the main paper's claims about interpretability, feature importance, and dimensionality reduction are not clearly reflected in the experiments (see Questions).

[1] (Phan et al.) GeKAN: Enhancing High-Dimensional Gene Expression Classification with Feature-Selected Kolmogorov-Arnold Networks. ISDS 2025 

[2] (Alharbi et al.) Interpretable graph Kolmogorov–Arnold networks for multi-cancer classification and biomarker identification using multi-omics data. In 	arXiv:2503.22939

[3] (Chen et al.) Kolmogorov-Arnold Networks with Trainable Activation Functions for Data Regression and Classification. In ICAIIC 2025.

### Questions
1. In Figure 1 (upper left), what would be the relative advantages of KAN-based approaches compared to the simple baselines (LASSO, Random Forest, and SVM)? Even KAN-KO significantly underperforms the standard baselines.

2. In practice, with KAN features, does the dimensionality reduction actually work? It appears that using KAN with all features yields the best results across benchmarks (Figures 1 and 2).

3. What would be the KAN's interpretability on feature importance selection that the authors claim? There are no such analyses in the experiments.

4. As far as I understand, KANs can not completely replace MLPs. Would it be possible to extend the same analysis to MLP features, combining standard feature importance selection methods (coefficient-based, knock-out, and sensitivity) with MLP features? This could help to highlight the advantages of KANs against MLPs in this problem setting.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This manuscript proposes and evaluates feature selection methods based on Kolmogorov-Arvold Networks (KANs). The authors introduce four variants of such KANs-based models (KANL1, KAN-L2, KAN-SI, KAN-KO) and compare them against classical baselines across multiple classification and regression benchmarks. The evaluation results show that the proposed methods are competitive with or superior to classical baselines in structured and synthetic datasets.

### Strengths
1. The evaluations are comprehensive, covering different tasks, different choices of classifiers/regressors, and different evaluation metrics.
2. This paper demonstrates good readability and is easy to follow.

### Weaknesses
1. The proposed methods do not demonstrate better performance compared to existing approaches. In addition, the performance of the proposed methods are quite unstable, showing high variance across different tasks/datasets.
2. There are multiple variants of the proposed methods, however, different variant performs better for different tasks with different classifiers on different evaluation metrics. This indicates a significant loss of generalizability, increasing complexity without substantial performance gains.
3. The organization and presentation of the paper need to be further improved. For example, (1) the abstract is not concise, containing too many details about different model variants' characteristics and the comparison among them on different tasks. (2) The axis labels and texts are too small to read for some figures.

### Questions
Please see "Weaknesses"

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper dives into feature selection methods based on Kolmogorov-Arnold Networks (KANs), which is comprised of four KAN-based importance measures (KAN-L1, KAN-L2, KAN-SI, KAN-KO).
This paper also compares proposed methods to classical baseline of machine learning, demonstrating excellent performance. However, some weakness inevitably persist.

### Strengths
1. Novelty of Exploration: The paper represents the first systematic attempt to leverage KANs for feature selection, exploring a promising research direction that combines the mathematical foundations of Kolmogorov-Arnold representation theorem with practical feature selection needs.
2. Excellent Performance on Classical  Benchmarks: The experimental design includes comparisons with six classical feature selection methods across seven datasets, evaluating performance at multiple feature retention levels (20%, 40%, 60%). The selected features perform the same as all features, which demonstrates the effectiveness of proposed methods.

### Weaknesses
1. The paper lacks sufficient theoretical justification for the proposed KAN-based importance measures. The mathematical rationale and intuitive explanations for why these specific formulations (L1/L2 norms of spline weights, knockout strategy, sensitivity integrals) are appropriate for feature importance in KANs are underdeveloped. At the very least, the intuitive reasoning behind why this should be done has not made sense to me.
2. Since the author denotes "they involve trade-offs among computational cost, predictive performance, stability, and interpretability." in Introduction section, it seems that the evaluation in this paper focuses primarily on predictive performance but neglects critical aspects such as computational efficiency and selection stability.
3. The reliance on classic machine learning datasets raises questions about the methods' applicability to modern high-dimensional data challenges. The absence of contemporary, complex datasets limits the generalizability claims.
4. The experimental results indicate that the predictive performance of features selected via KAN remains nearly consistent with that of the full feature set. This only demonstrates that the selected features adequately represent the full feature set, but does not confirm whether redundant features still exist among the selected ones.

### Questions
1. Further theoretical analysis or intuitive explanation is required.
2. More analyses based on experiments are required.
3. More experiments on challenging and contemporary datasets are required.
4. Illustration of whether redundant features still exist after feature selection is required.

### Soundness
2

### Presentation
3

### Contribution
2
