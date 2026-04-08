## Human Reviewer 1

### Summary
The paper investigates the impact of the bootstrap rate (BR) greater than 1.0 on Random Forest (RF) performance, concluding that higher BR values may lead to statistically significant improvements. The paper is purely empirical. I have a few concerns primarily regarding the lack of insight and the generalizability of the findings.

### Strengths
Extensive experimental studies

### Weaknesses
1. Lack of Insight: The study would greatly benefit from a discussion on the underlying rationale behind the observed improvements in classification accuracy with BR > 1. Although the (very limited) empirical results suggest that higher BR values could potentially improve performance, the paper does not offer insights into why this might be the case. Introducing a simplified, theoretical case, such as a linear Gaussian model, could help illustrate the mechanisms at play and provide a foundation for these empirical observations. This addition would make the findings more interpretable and provide value beyond experimental results.

2. Experimental Setup and Generalizability: While the authors employ a diverse dataset collection, the setup appears somewhat arbitrary, raising questions about the generalizability of the results. For example, RF performance is known to vary significantly with the number and complexity of categorical variables, which can greatly influence tree-based methods. Controlling for these factors or providing more context on dataset selection criteria would help clarify the scope of the findings and allow for a better assessment of their relevance to broader cases.

3. Classifier for BR Selection: The approach to developing a classifier to predict optimal BR usage lacks a solid rationale. The selection of 18 RF configurations and 10 BR values per dataset does not align with the principles of cross-validation, where data should be permutable to make the validation results meaningful/generalizable. This choice weakens the validity of the classification model and its potential for reliable application across diverse datasets.

### Questions
None

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
1

### Confidence
5

---

## Human Reviewer 2

### Summary
The paper explores the use of bootstrap sampling in Random Forests, specifically focusing on the bootstrap rate (BR), which is the ratio of the number of observations in each bootstrap sample to the total number of training instances. While traditional methods use a BR of 1 (sampling equal to the original dataset size), the authors investigate the effects of higher BR values, ranging from 1.2 to 5.0, on classification accuracy. Additionally, the authors create a binary classifier to predict whether the optimal BR for a given dataset is less than or equal to 1 or greater than 1.

### Strengths
1)	Overall, this work re-examines the bootstrap rate in random forests, considering a wide range of BR values and multiple datasets, which throw light on focusing some basic machine learning configs.

2)	Sufficient experiments on multiple datasets are carried out, which involves various aspect on BR values, such as the relationship between BR and classification accuracy, the influence of different RF configurations, and the dependence of the optimal BR on the dataset.

### Weaknesses
1)	Although different BR values have been extensively studied, the explanations for why certain datasets exhibit specific behaviors at specific BR values may not be deep and comprehensive enough, especially for some complex patterns and anomalies. For example, for a model like RF (nf all) that exhibits special behaviors, although some hypotheses have been proposed, more research may be needed to understand the exact mechanism behind it.

2)	When analyzing the relationship between the optimal BR value and the dataset, although some attributes and characteristics of the dataset are taken into account, there may be other undiscovered factors that affect the selection of the optimal BR value, which may play an important role in different datasets and application scenarios.

3)	The article does not analyze time performance related issues in detail. In practical applications, different BR values and model configurations may have a significant impact on training and prediction time, which may be a direction for future research.

### Questions
1)	Relationship between BR value and classification accuracy should be supported by more theoretical analysis and illustrated in a more rigorous form.

2)	It is presumed that “RF(nf all) will perform well on datasets with a high proportion of insignificant or less significant features, as it may avoid building trees primarily based on these features”. Could you provide some stronger evidence or more in-depth analysis?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
This article investigates the impact of the bootstrap rate (BR) on the predictive performances of random forests. BR is a hyperparameter of random forests: each tree of a random forests are built by drawing with replacement BR*n observations among the n original ones. 
Previous works have tested the impact of BR values ranging from 0.2 to 1.2. In this paper, the authors compare the performance of various RF configurations (varying the number of trees, maximal depth, number of features randomly selected in each cell...) together with different BR values ranging from 0.2 to 5. 

Experimental results over 36 open source data sets show that a BR larger than one is associated to good predictive performances for most data sets. The authors explain these results by introducing a statistics measure over a data set, corresponding to the inhomogeneity of the data set. More precisely, the parameter $k\_l$ stands for the number of observations in the data set that have $l$ observations with the given label among their $k$ nearest neighbors. The authors demonstrate that the optimal BR is correlated to this measure. 

Finally, the authors also introduce two binary classifiers that predict if a given data set has an optimal BR greater or lower than 1. These classifiers are respectively trained on 36 and 24 data sets.

### Strengths
The paper studies the impact of BR on random forests performances and provide insights about the characteristics of data sets for which BR >1 should be preferred. Extensive simulations are done to corroborate the proposed statements.

### Weaknesses
- Several related works are missing. Besides, it is not clear from the related work section which contributions are theoretical or practical. See for example: 
    - For empirical performances, see Section 4 in Random Forests for Big Data https://arxiv.org/abs/1511.08327
    - For theoretical results quantifying the uncertainty of random forests, depending on BR, see https://jmlr.org/papers/volume17/14-168/14-168.pdf or https://arxiv.org/pdf/1405.0352
    - In https://www.esaim-ps.org/articles/ps/pdf/2018/01/ps170099.pdf, the authors provide the expression of the optimal subsampling rate for median forests
    - Regarding Breiman's forests, the analysis of https://arxiv.org/pdf/2006.06998 takes into account bootstrap

- Experiments are done for classification data sets only. The paper would be strengthened if experiments were done also for regression problems. 

I have several concerns related to the design of experiments, and more specifically with Table 1: 
- No standard deviations are displayed. Thus, we are not sure if the best method is really the best or if many other RF configurations have performances close to the best one. 
- Looking at Table 2 in Martinez-Munoz and Suarez (2010), which displays the average test error for different BR ratios, the standard deviations are quite important. For example, the standard deviation of the error (and thus the accuracy) is between 7 and 8 for the audio data set, whereas you display the accuracy with 3 decimals. 
- Besides, the test set is used for two different objectives: selecting the best RF configuration and computing its error. Thus, the error of the best RF configuration is rather optimistic. It would be more sound to evaluate the performance on a different data set (or to use the out-of-bag error of the forest). Thus, t-test results do not seem conclusive. Besides, as you consider several t-test, you should probably correct the significance level $\alpha$ accordingly, taking into account multiple testing issues.
- Given the performances displayed in Table 1, I would like to see the same experiment run with a Baseline RF whose number of trees is set to 500. Indeed, this parameter seems to be the most important one, whereas in practice it is fairly easy to tune (the higher the better in terms of accuracy). 
- A Table with all values of RF configuration / BR rate with standard deviations would give more precise information. 
- l.335 The fact that optimal performances are reached for two very different values of BR (0.2 and 5) by slightly changing the data set is worrying. One reason can be that the two values are good for the two data sets, which is in contradiction with your study, showing that BR >1 is better than the opposite. This is why metrics for all configurations with standard deviations should be displayed.

### Questions
- l.47 "we expect 36.8% of observations to be absent in each bootstrap sample." We know that the probability for one observation to be absent of a bootstrap sample is 0.368. However, I am not sure that this corresponds to the percentage of observations absent from the bootstrap sample, as the events $B_i$ ("presence of the $i$th observation in the bootstrap sample") are not independent. Can you prove the statement or give a reference? 
- Looking at the paper of Martinez-Munoz and Suarez (2010), it appears that they do not use split randomization in their random forests implementation. This could explain the difference between their results and the results obtained in the present paper. This should be clearly mentioned somewhere. 
- l.98 what is the rationale behind standardizing one-hot encoded features? Tree based methods do not usually need input standardization to work. 
- The ranges of values for the different hyperparameters are not consistent: the max depth varies between 10 and 25 (which results in leaf containing between n/2^10 and n/2^25 observations if the tree is balanced) while the minimum number of observations per leaf varies between 2 and 5. These ranges of values should be harmonized.  
- A twofold cross-validation is performed 200 times. Usually, a 5-fold cross validation is performed. Is there any reason for choosing this value? Can you display standard deviation averaged across runs? 
- A BR value much larger than one would lead to use almost all observations for each tree construction. This would probably damage the computation of the out-of-bag error, which is an important feature of random forests, as it allows us not to divide the data set into a train and a test set to compute the RF error. Could you study how the out-of-bag error is influenced by the BR value?  
- l.206 "Finally, BR= 1, defined in the original formulation of the bootstrapping procedure and the most frequently used
value, performed relatively poorly." This is not shown by the experiments: the only things shown here with respect to BR=1 is that it is often not the best configuration. But it can be the second-best most of the time. 
- l.340 notation $k\_l$ is uncommon and misleading as it counts a number of observations. I would rename it $n_{k,l}$ for clarity.
- Table 2. What is the rationale for studying very low values of $k$ and $l$ with respect to the data set size? Choosing a fix number of nearest neighbors $k$ for all data sets corresponds to considering neighborhood of different sizes across data sets. Thus, I wonder if such a measure $k\_l$ corresponds to a characteristic of the data set.
- l.431 Regarding the bootstrap rate prediction, I am not sure to understand the final methodology. More precisely, I do not understand how features are selected: "based on the absolute value of the Spearman rank-order coefficient, calculated separately for each run on the training instances". Does it mean that, at first, 12,000 features are computed, then the correlation with the best BR is computed, and only the ten best are kept? Since computing the best BR requires a validation set, performances should be computed on a test set. Besides, does the procedure require running random forest with all BR values (in order to compute the best ten values among the 12,000 original features) in order to build a classifier to choose the best BR value? 
- The resulting classifier predict whether the best BR is larger than one. More practically, I would be more interested in estimating the best BR. How does your analysis help the practitioners?

### Soundness
3

### Presentation
3

### Contribution
1

### Rating
1

### Confidence
5

---

## Human Reviewer 4

### Summary
The paper studies the effects of the Bootstrap Rate (BR) parameter on the test set accuracy of random forest classifiers. The authors claim that it is possible to obtain improvements (not always) by setting BR > 1, which is rarely done in practice. The authors develop a classifier to predict if the optimal BR is larger than 1 based on the dataset.

### Strengths
* Good scope of study in terms of datasets and covering a wide range of hyperparameters
* Exploration of the effects of a bootstrap rate that is generally understudied
* The idea of determining the BR based on data statistics is interesting

### Weaknesses
* The analysis carried out in table 1 is not explained in a clear enough way. For instance, the paired t-test is not fully described and is hard to understand. For a result listed as a key contribution in the paper, the procedure should be explicitly described.
* Figure two does not include error bars, and in almost all cases the effect of the BR parameter seems very small in terms of accuracy.  
* The classifier only predicts if the optimal BR (a term that need to be formally defined) is larger than 1. It would be more useful for practitioners to predict the actual rate.
* Generally I feel as if the findings in this study align well with the hypothesis that BR > 1 is as good, but not better, than BR=1. When making a claim that the optimal BR is equal to a certain value, the authors rely on finite sample analysis, which might be susceptible to random fluctuations.

### Questions
* By "2-fold stratified cross-validation, repeated 200 times, was applied, yielding 400 results", do the authors mean that they did a 50/50 (train/test) split and ran two experiments for each configuration?
* Can the authors please fully describe their hypothesis testing procedure carried out in Table 1? 
* Can the authors please formally define the optimal BR?

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4