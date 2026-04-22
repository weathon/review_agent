# CFA: Causal Feature Augmentation for High-Dimensional Linear Regression

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
High-dimensional prediction with limited samples poses a significant challenge due to severe overfitting. While existing approaches tackle this via regularization, clustering, or representation learning, we introduce a novel framework inspired by causal inference that is designed to exploit latent structure linking predictors and responses. Our approach employs a new similarity-based clustering procedure guided by a metric that quantifies shared predictor-response dependencies, which tends to group variables that play similar roles with respect to (possibly latent) mediators or confounders. The resulting causality-inspired features are then incorporated into an augmented regression model, yielding sparser, more robust, and more generalizable predictions without attempting to recover the underlying causal graph. Experiments across synthetic and real-world datasets, including S&P 500 market data, demonstrate that our method achieves higher regression performance and markedly reduces overfitting compared to existing baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce Causal Feature Augmentation (CFA), a framework for high-dimensional linear regression. The key idea is to consider the effect of mediators and confounders that affect the relationship between X and Y. The authors propose an algorithm to separate the predictors based on pairwise similarity (how to predictors jointly affect the response) and construct augmented confounding and mediation features. The augmented features [X,Z_c,Z_m] are then fitted with an Elastic net as a standard regression problem with L1 and L2 penalties. CFA shows better correlation results compared to those commonly used high-dimensional regression methods.

### Strengths
1.	Theoretical analysis, especially the explanation of why counting shared Y-dependencies can reveal latent grouping structure, is solid.
2.	The performance gain is pretty big especially under the scenarios with small sample size.
3.	The consideration of The mediator/ confounder effect is pretty intuitive, and the features construction is also straightforward.

### Weaknesses
1.	It seems that the similarity matrix replies on a quite number of dependency test. This will be quite inefficient especially when dimension of X and Y are large.
2.	The performance on real data (S&P 500) shows some improvement, but the correlation is low on test set, which makes little practical use.
3.	Recovering each confounder using only the first principal component per cluster can be quite restrictive. Sometimes the true shared variability can be multi factors, which may require more PCs.

### Questions
1.	Could the author provide results on more commonly used regression datasets? e.g., from UCI Machine Learning Repository. 
2.	How would Z_m Z_c be sensitive to those important hyper parameters, such as he number of clusters, p-value thresholds etc.
3.	The sample sizes in all experiments are very small, which does fit the many problems in modern data science. The authors should demonstrate clear under what scenarios this method can be useful. Perhaps a real dateset (e.g., a rare disease) with limited sample size may help.
4.	Some important baselines are missed, such as group lasso, multi-task lasso. More modern AI methods based on representation learning shall also be considered, such as a fully-connect network.
5.	Detailed scalability experiments are needed for different sample size and different dimension of X,Y.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses high-dimensional, small-sample linear regression tasks by introducing a method inspired by causal inference concepts. The authors cluster predictors into groups that share latent mechanisms (either confounder-based or mediator-based) and use these latent variables to improve regression performance. The paper defines a similarity metric between predictors, classifies clusters based on intra-cluster correlations, and constructs latent variables either by averaging (for confounder clusters) or via PCA (for mediator clusters). The final regression model is trained with Elastic Net. Experiments on synthetic datasets and S&P 500 stock data demonstrate gains.

### Strengths
1. The paper presents a clear motivation and a coherent framework.
2. The proposed distinction between confounder and mediator clusters is intuitively appealing and nicely connected to causal concepts.

### Weaknesses
1. Limited novelty: The causal framing adds interpretability but little methodological innovation.
2. Sparse and outdated citations: The reference list includes fewer than 30 works, most from before 2010, raising concerns that relevant modern literature was overlooked.
3. Unconvincing experiments: Synthetic data favor the proposed structure by design, while the S&P 500 results are very weak.

### Questions
1. Could the authors clarify whether the method’s advantage persists when the true latent structure does not align with the causal assumptions?
2. Why were the four additional datasets mentioned in the appendix not included in the main paper? They seem to perform better than S&P 500.
3. Have the authors compared their method to modern alternatives? Did the authors conduct a detailed investigation of this field and compare with more advanced baselines?

### Soundness
3

### Presentation
2

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
The paper proposes Causal Feature Augmentation (CFA), a pragmatic pipeline that engineers a small set of proxy mediator and confounder features from the original predictors and appends them to a standard linear model (Elastic Net) for better multi-response prediction. Concretely, CFA (i) detects which predictors relate to which responses via thresholded two-tailed correlation t-tests, (ii) builds a predictor–predictor similarity score $s(i,j)$ from how often two predictors co-associate with the same responses, then (iii) clusters predictors using bottom-up hierarchical clustering (average linkage). Within each cluster, CFA labels mediator-type vs confounder-type via intra-cluster correlation tests. Augmented features are then constructed to enhance model prediction. Experiments include synthetic linear settings matching the design assumptions and a real S&P 500 prediction task; both show improvements over a plain Elastic Net and several ablations of the augmentation choices.

### Strengths
1. The proposed method slots into standard Elastic Net workflows; no bespoke learner or complex training loop is required. 
2. The constructed features are easy to interpret using ideas from causality.
3. The similarity $s(i,j)$ seems to be novel and explicitly leverages multiple responses to decide which predictors likely share causal roles.

### Weaknesses
1. In causality, identifiability is one of the most important things. The author borrowed the terminology, but unfortunately can not provide any guarantees on the identifiability of the mediator and the latent variables in their method in any case (even if the ground-truth is Figure 1). Therefore, the title and the abstract are misleading. I would suggest that the author change it to be casality-inspired feature augmentation to make a clearer separation. 

2. In the setting, all predictors are assumed to be non-descendants of the response variable. I wonder why it is the case and how anti-causal relationships ($Y\rightarrow X$) would affect the proposed framework. After all, if the response is the cause of a predictor $X$, then $X$ should also be included as an important feature.

3. The mediator is simply the average of the variables in the cluster. Although it might be meaningful for some data sources, it is in general a simple approximation. I wonder whether it can be extended to an arbitrary linear transformation.

4. The calculation in Table 1 is not clear to me. The probability $\pi_m,\pi_c$ for linking $X$ and $Z$ is missing. Shouldn't there also be a comparison with the probability of not having any mediator/confounder, i.e. direct link with the response? That would help to better assess the similarity metric.

5. Since the proposed method has multiple parameters to tune, a direct comparison of the runtime is necessary to assess the practicality of the method.

6. The experiments are not sufficient. For example, in the synthetic datasets, there is no test for the robustness of the method under assumption violations, e.g. alternative mediator aggregation, anticausal predictors. Only one real dataset is given, making it difficult to evaluate the performance of the method. The author could include more datasets, e.g. gene datasets like GTEx / eQTL, TCGA multi-omics, etc.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
