# Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants

- Decision: Reject
- Scores: 8, 5, 5, 3

## Abstract
Synthetic Minority Oversampling Technique (SMOTE) is a common rebalancing strategy for handling imbalanced tabular data sets. However, few works analyze SMOTE theoretically. In this paper, we prove that SMOTE  (with default parameter) tends to copy the original minority samples asymptotically. We also prove that SMOTE exhibits boundary artifacts, thus justifying existing SMOTE variants. Then we introduce two new SMOTE-related strategies, and compare them with state-of-the-art rebalancing procedures. Surprisingly, for most data sets, we observe that applying no rebalancing strategy is competitive in terms of predictive performances, with tuned random forests, logistic regression or LightGBM. For highly imbalanced data sets, our new methods, named CV-SMOTE and Multivariate Gaussian SMOTE, are competitive. Besides, our analysis sheds some lights on the behavior of common rebalancing strategies, when used in conjunction with random forests.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates the effectiveness of rebalancing strategies for handling imbalanced datasets in binary classification, specifically focusing on the Synthetic Minority Oversampling Technique (SMOTE) and its variants. The authors provide a theoretical analysis demonstrating that default SMOTE tends to asymptotically copy original minority samples and exhibits boundary artifacts. They introduce two new strategies, CV-SMOTE and Multivariate Gaussian SMOTE (MGS), and compare their performance against traditional methods like Random Under Sampling (RUS), Random Over Sampling (ROS), and Class Weighting (CW). The empirical results indicate that for many datasets, applying no rebalancing strategy is competitive, while the proposed methods show promise for highly imbalanced datasets.

### Strengths
1. The paper is well-organized, with clear sections that guide the reader through the theoretical analysis, empirical evaluation, and results.
2. The paper provides a solid theoretical foundation for understanding the behavior of SMOTE, including its limitations and boundary artifacts.
3. The authors conduct a thorough empirical evaluation across multiple datasets.

### Weaknesses
1. The diversity in terms of data characteristics (e.g., feature types, distributions) could be expanded.
2. More recent references and baselines should be included.

### Questions
What criteria should practitioners use to select the hyperparameter grid for CV-SMOTE? Are there specific guidelines or best practices that can be derived from the experiments?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper analyzes SMOTE theoretically. Its contributions include:

(1) It proves that, without tuning the hyperparameter K (usually set to 5), SMOTE asymptotically copies the original minority samples, therefore lacking the intrinsic variability required in any synthetic generative procedure. It provides numerical illustrations of this limitation.

(2) It proves that SMOTE density vanishes near the boundary of the support of the minority distribution, therefore justifying the introduction of SMOTE variants such as BorderLine SMOTE.

(3) It introduces two SMOTE alternatives, CVSMOTE and Multivariate Gaussian SMOTE (MGS). It evaluates these two new strategies and state-of-the-art rebalancing strategies on several real-world data sets using random forests/logistic regression/LightGBM. These experiments show that applying no strategy is competitive for most data sets. For the remaining data sets, the proposed strategies are among the best strategies in terms of predictive performance. The analysis of experiment results also provides some explanations about the good behavior of RUS, due to an implicit regularization in presence of random forests classifiers.

### Strengths
1. This paper theoretically analyzes Synthetic Minority Oversampling Technique (SMOTE), which is a widely adopted rebalancing strategy for handling imbalanced tabular data sets and has numerous variants. The theoretical analysis is comprehensive.
2. The paper proposes two SMOTE alternatives: CVSMOTE and Multivariate Gaussian SMOTE (MGS), thus making minor technical innovations.
3. Through extensive experiments on various datasets, the paper demonstrates that applying no rebalancing strategy can be competitive for some data sets. This finding is insightful.

### Weaknesses
Please refer to questions.

### Questions
1. It is recommended to clearly explain the implications of the metric \(\bar{C}(Z, X)/\bar{C}(\tilde{X}, X)\), such as what it signifies when the value approaches 1, 0, or exceeds 1, as well as the impact of the corresponding synthetic samples on the classification model performance, which currently lacks discussion.

2. Regarding Theorem 3.2, the author's claim that "This highlights a good behavior of the default setting of SMOTE (K = 5), as it can create more data points, different from the original sample, and distributed as the original sample" needs further justification. The reasoning for "create more data points" is unclear, and with finite \(n\) in practice, the conclusion "distributed as the original sample" may require additional support.

3. In the statement, “Choosing \(K\) that increases with \(n\) leads to larger characteristic distances: SMOTE observations are more distant from their central points. Corollary 3.6 leads us to choose \(K\) such that \(K/n\) does not tend too fast to zero, so that SMOTE observations are not too close to the original minority samples,” it's hard to understand “Choosing \(K\) that increases with \(n\)” and “tend too fast to zero.” These informal expressions can be confusing. It is suggested to explain the practical meaning of larger characteristic distances, such as diversity, to aid understanding.

4. In Section 3.2, THEORETICAL RESULTS ON SMOTE, could explicit ranges for \(K\) values be provided? Is there a trade-off between "regenerating the distribution of the minority class," avoiding "boundary bias," and achieving "more diversity"?

5. The term “boundary artifacts” appears only in the abstract and conclusion but not elsewhere in the paper. Furthermore, no reference is found in the original BorderLine SMOTE paper. Additionally, the author claims to “prove that SMOTE exhibits boundary artifacts, therefore justifying the introduction of SMOTE variants such as BorderLine SMOTE (Section 3),” so it is recommended to provide relevant citations and explain how BorderLine SMOTE addresses this issue.

6. Concerns regarding Section 3.3, NUMERICAL ILLUSTRATIONS:
   - (1) In the first result analysis, the author states, “Note that, for the other asymptotics in \(K\), the diversity of SMOTE observations increases with \(n\), meaning \(\bar{C}(Z, X)\) gets closer to \(\bar{C}(\tilde{X}, X)\). This behavior in terms of average distance is ideal since \(\tilde{X}\) is drawn from the same theoretical distribution as \(X\). On the contrary, \(K = 5\) maintains a lower average distance, indicating a lack of diversity in generated points.” Based on previous descriptions, higher diversity is considered better, so \(K = 5\) with low diversity should be undesirable, which conflicts with the statement “This highlights a good behavior of the default setting of SMOTE (K = 5).”
   - (2) How should “this diversity is asymptotically more important for xxx” be understood?
   - (3) How should “By construction, SMOTE data points are close to central points, which may explain why the quantity of interest in Figure 1 is smaller than 1” be interpreted? Furthermore, in the second result analysis, some values of the metric \(\bar{C}(Z, X)/\bar{C}(\tilde{X}, X)\) exceed 1; what implications would this have?

7. The conclusion that “applying no strategy is competitive for most data sets” based solely on “the imbalance ratio is not high enough or the learning task not difficult enough” is not rigorous enough. A more detailed analysis of dataset characteristics would help identify when it’s appropriate to apply no strategy versus when to use rebalancing strategies.

8. It is recommended to compare some newer SMOTE variants, as some of the rebalancing strategies used in the experiments are somewhat outdated.

9. More precise and standardized descriptions would improve readability. For example, phrases like “guarantees asymptotically” and “smoothed out” require clarification. Consistent terminology should be used instead of mixing “SMOTE data points,” “SMOTE samples,” and “SMOTE observations.” Additionally, the title of Table 2 could specify the experimental module for clarity.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a comprehensive study of synthetic rebalancing strategies from theoretical and computational perspectives, further proposing two promising approaches specifically designed for highly imbalanced datasets. The findings reveal that for most mildly imbalanced datasets, applying no rebalancing strategy can be competitive, while rebalancing methods show clear benefits in scenarios with high imbalance.

### Strengths
1. The paper provides an in-depth theoretical analysis of SMOTE, offering valuable insights into its behavior and mechanics, which enhances the understanding of its effectiveness in handling imbalanced datasets.
2. The paper is well-written, with a clear articulation of its objectives and contributions. In particular, the Introduction and Related Works sections effectively outline the development of solutions for imbalanced datasets, providing a strong foundation for the study's motivation and purpose.

### Weaknesses
1. It would be incorrect and confusing about the definition of imbalanced ratio. For example, in line 147, the authors define the imbalance ratio as the proportion of minority samples to the total samples (minority samples/total samples). However, in line 400, the authors mention that "applying no strategy is the best, probably highlighting that the imbalance ratio is not high enough". This wording suggest low-imbalanced datasets while not high imbalance ratio in your definition indicates highly imbalanced datasets.
2. The novelty of the paper appears limited in several aspects. a. While the paper offers a thorough theoretical examination of SMOTE, the insights may not extend significantly beyond those of earlier works. I may not have the complete picture, however, and remain open to further clarification from the authors on their theoretical contributions. b. CV-SMOTE addresses hyperparameter selection via cross validation, while MGS introduces Gaussian sampling. Could the authors provide any unique aspects of their implementation or theoretical justifications that set the approaches apart from existing methods? c. The observation that applying no rebalancing strategy performs competitively on mildly imbalanced datasets is consistent with existing knowledge. Could the authors discuss how their empirical results add to or refine existing knowledge?
3. The study focuses solely on binary classification and tabular data. It would be beneficial to discuss how the findings might extend to multiclass problems to improve the paper's generalizability.

### Questions
My opinions and questions are outlined in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper makes a theretical analysis of the well-known SMOTE method for imbalance classification, and proposes two simplest variants.

### Strengths
This paper involves both theoretical analysis and method development

### Weaknesses
1) The theretical analysis is based on the assumption that $n$ is sufficiently large such that $K/n$ tends to zero, but this assumption is totally impractical in the setting of imbalanced learning where minority class has few examples.
2) The theretical analysis provides little information with respect to imbalanced learning, and it simply discusses the SMOTE method in the general learning setting. The result does not help understand the essence of imbalanced learning, and does not motivate how to solve the problem.
3) The connection between theoretical analysis and the method proposal is not well established. 
4) The two proposed methods are not attractive. CV SMOTE is simply a hyperparameter tuning based on cross-validation, which is usually effective and commonly used technique as an engineering approach, without any theoretical or methodological contribution. MGS requires that there are at least $d+1$ minority-class examples, which may be impractical for high-dimensional data. In addition, even there are more than $d+1$ minority-class examples available, it has to assume that the covariance matrix is diagonal, which is also not the case in practice and thus over-generalized samples would be generated, which may do harm to the performance of imbalanced learning.  
5) The experimental results are not significant.

### Questions
Please refer to the weaknesses 1, 2, 3 and 4.

More specifically, 
1) Could you provide bounds or approximations that hold for small $n$, instead of huge $n$?
2) Could you analyze how SMOTE affects the decision boundary or classification margins in imbalanced settings? 
3) Could you explain the reasons for designing these two variants based on the theoretical analysis in your paper and your responses to the questions 1) and 2)?
4) Have you considered any theoretical guarantees for CV SMOTE? Have you explored regularization techniques for MGS to handle high-dimensional data?

### Soundness
2

### Presentation
2

### Contribution
2
