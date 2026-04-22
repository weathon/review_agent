# INTERPRETABLE COMPACT CATEGORICAL FEATURES ENCODING FOR SUPERVISED LEARNING

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
In supervised learning, encoding techniques for continuous features are well studied. However, few are specific for categorical features. The categorical encoding approaches that are widely used are one-hot encoding, target encoding and its variant ordered target encoding. In many cases categorical features carry significant, if not a dominant portion of, feature information in a supervised learning problem. Therefore they are key to improve model fit. One hot encoding is known for its curse of dimensionality issue, especially when categorical features are of high cardinality and/or sparse. Such problem not only increases the problem size but may also introduce instability for numerical solvers. Target encoding and its variant ordered target encoding are often used to address such data issues. It is fast and compact. The downside is that target encoding tend to overfit due to the way it is implemented. To our knowledge, the other categorical encoding methods used in machine learning and deep learning algorithms do not preserve interpretability. Our goal is to bridge the gap between dimension reduction, accuracy, feature interpretability, and scalability. In this paper, we introduce a polynomial algorithm called Interpretable Compact Categorical Feature Encoding for Supervised Learning (ICFESL). Under reasonable assumption, our encoding technique ensures no information loss for regression and minimum information loss for classification. At the core, it leverages L2 regularized linear models to efficiently calculate coefficients for one-hot-encoded categorical features and group them together without transforming them. We prove that applying K-means clustering for the grouping problem yields optimal solutions. We test our algorithms on simulations and real-world datasets both in regression and classification to validate the assumption and demonstrate the encoding method’s performance. The results show that for regressions, ICFESL enabled linear models and xgBoost models often significantly outperform state-of-the-art algorithms such as CatBoost and TabNet in terms of RMSE. The results also show that for classifications, ICFESL has comparable performance and outperforms CatBoost measured by AUC when ordered target encoding shows significant overfitting. We demonstrate how interpretability is preserved with example clusters from one of the experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ICFESL, a polynomial-time algorithm for compressing high-cardinality categorical features by clustering their levels based on the coefficients from a preliminary OLS or Logit model. The core idea is to use K-means on these coefficients, weighted by class frequencies or the derivative of the logistic function, to find an optimal grouping that minimizes information loss. The authors provide theoretical guarantees for this approach under an orthogonality assumption of one-hot encoded vectors. The method is evaluated on multiple datasets for both regression and classification tasks, demonstrating performance comparable or superior to one-hot and target encoding while significantly reducing dimensionality and maintaining interpretability.

### Strengths
**Interpretability**: The method directly compresses the original feature levels, preserving the meaning of the features, which is a significant advantage over black-box embedding methods.

**Theoretical Grounding**: Provides proofs of optimality under specified conditions, which is rare for practical encoding schemes.

**Practicality and Scalability**: The algorithm has polynomial time complexity and is demonstrated on real-world, high-cardinality datasets, showing its potential for industrial use.

### Weaknesses
**Reliance on Strong Assumptions**: The optimality guarantees depend on the orthogonality of OHE vectors, which is an idealized condition. The practical workaround (Hamming clustering) is mentioned but not deeply evaluated.

**Hyper-parameter Sensitivity**: The performance depends on p-value thresholds and the choice of stopping criterion, which requires tuning and may not be fully automated.

**Limited Baseline Comparison**: While compared to OHE and target encoding, a comparison with other advanced methods like feature hashing or SOTA encoding methods would strengthen the empirical validation.

### Questions
1. How sensitive is the algorithm's performance to the violation of the orthogonality assumption?

2. Could you provide an ablation study showing the performance gain/loss with and without the Hamming distance pre-processing step?

3. The p-value threshold is a key hyper-parameter. Did you explore automated ways to set it, perhaps based on the desired level of compression, rather than relying on a decision plot?

4. Have you considered applying this method to tree-based models like LightGBM or CatBoost directly, which have their own built-in mechanisms for handling categorical features? How would ICFESL complement or compete with these native methods?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
### Summary

The paper addresses encoding high-cardinality categorical features. One-hot encoding explodes dimensionality, target encoding is ad-hoc. The core  Fit OLS on one-hot encoded data, then cluster resulting coefficient values using K-means.

### Main contributions:

For regression: proves clustering is lossless under orthogonality assumption across all feature levels. The weighted average is the exact OLS solution for the collapsed problem.
For classification: no exact solution - minimizes prediction squared error instead of likelihood. Admits "measurable information loss."
Algorithm is polynomial O(|P|mn²) with automatic K selection via inertia stopping criterion. Requires tuning p-value threshold manually via decision plots.

The authors test on 5 datasets with OLS/Logit, XGBoost, TabNet. Results are okay. Sometimes their method performs better than target encoding, often it is comparable, and occasionally worse. They show that the dimension of the problem reduces but interpretability of actual clusters are never validated.

### Strengths
- The paper is clear and easy to understand. The theoretical claims and proves seem valid.
- The algorithm presented is polynomial time O(|P|mn²) is reasonable. Not trying to solve some NP-hard problem with exponential search like prior work (GRASP). However, the work doesn't directly compare with GRASP.
- Experiments across model types & datasets are good to have.

### Weaknesses
- My biggest issue with the paper is that there is no validation of interpretability despite it being the core claim. The authors assume that categorical clustering preserves interpretability. While I agree with it on surface, I don't think it is always true. The clustering algorithm can make meaningless uninterpretable groups across feature levels which have a similar coefficient. I would have liked to see analysis on what kind of clusters does their algorithm produce or does the algorithm improve clustering or interpretability compared to other sensible baselines.
   - What's needed: Show actual clusters, compare to natural hierarchies, or measure cluster coherence. Report interpretability metrics across baselines.
    - Embedding features in vector space is a strong baseline and there are papers that try to interpret those features. A comparison to such methods would be nice as well. 
-  More directly comparable baselines are needed to assess the strengths of the algorithm. Some suggestions:
    - CatBoost with native categorical handling - directly addresses the same problem, widely used
    - Entity embeddings - standard deep learning approach for categoricals
    - GRASP (Carrizosa et al., 2021) - most directly comparable prior work
  The excuse that GRASP solves a "different problem" (single feature vs. all features) is weak since ICFESL also processes features independently (Algorithm 1, lines 280-294). The real issue is GRASP is expensive, but a limited comparison on smaller features would still be valuable.
Without these comparisons, we can't assess if coefficient clustering is competitive with other principled approaches.

### Questions
1. Can you provide concrete examples of clusters from your experiments? For instance, for the US AQI dataset where "Local Site Name" goes from 1,033 levels to 103 clusters, what do clusters 1, 5, and 10 actually contain? Do the grouped monitoring sites share geographic proximity, pollution sources, or other interpretable attributes?
2. What exactly happens to categorical levels filtered out by the p-value threshold? Looking at Table 2, Fraud UW goes from 59 OHE features to 34 ICFESL features. Are the remaining 25 dropped entirely, kept as individual levels, or something else?
3. Can you quantify what happens when orthogonality is violated? Even a simple simulation would help: generate correlated categorical features, apply your method, measure the actual vs. theoretical information loss. How robust is the "lossless" claim in practice?

### Soundness
3

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
4

### Summary
This paper proposes a method named ICFESL (Interpretable Compact Categorical Feature Encoding for Supervised Learning), a polynomial-time algorithm for encoding categorical features by clustering feature levels based on OLS/MLE coefficients. The approach addresses the dimensionality curse of one-hot encoding (OHE) and the ad-hoc nature of target encoding. The authors prove that applying K-means clustering to regression coefficients (weighted by observation counts) or classification coefficients (weighted by logistic derivatives) yields optimal clustering schemas. The method is evaluated on five real-world datasets using OLS, XGBoost, and TabNet models, demonstrating comparable or superior performance to existing encoding methods.

### Strengths
- Addresses real problem: High-cardinality categorical encoding is genuinely challenging in practice
- Polynomial complexity: More tractable than GRASP-based approaches (exponential)
- Automatic cluster number selection: Unlike methods requiring pre-specified K, uses stopping criteria to determine cluster counts
- Interpretability focus: Unlike many neural embedding approaches, the proposed one maintains coefficient interpretability

### Weaknesses
- Limited novelty: The core idea of clustering coefficients is relatively straightforward; applying K-means to regression coefficients is not a significant algorithmic innovation.
- Incremental improvement: Results show ICFESL is comparable to target encoding in most cases, with only marginal improvements in select scenarios.
- Theoretical-practical gap: The orthogonality assumption is acknowledged to "rarely hold by default" yet no systematic study of when violations matter
-  Weak baselines: No comparison with learned embeddings (entity embeddings, CatBoost's ordered target encoding variants). No comparison with recent categorical encoding methods from the literature
Hamming distance clustering is presented as a baseline but is actually a preprocessing step for ICFESL
- Limited experimental scope: only 5 datasets (3 classification, 2 regression)
- No datasets with truly high cardinality (US AQI has 1033 unique values but after filtering may be much smaller)
- Missing analysis: No runtime comparisons with other methods. No study of performance scalability with cardinality. No ablation study on the impact of Hamming distance preprocessing

### Questions
- Can you provide empirical analysis of how often the assumption X^T_{i,j}X_{s,t} = 0 holds in your datasets before and after Hamming clustering?
- Why does ICFESL underperform target encoding on XGBoost for most datasets? Can you explain this pattern?
- Is Hamming clustering mandatory or optional?
- What is the "Min Obs" parameter and how do you set it?
- Can you show results on datasets with higher cardinality?  let's say about 100k levels

### Soundness
3

### Presentation
2

### Contribution
2
