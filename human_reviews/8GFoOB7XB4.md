# SEMF: Supervised Expectation-Maximization Framework for Predicting Intervals

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
This work introduces the Supervised Expectation-Maximization Framework (SEMF), a versatile and model-agnostic approach for generating prediction intervals in datasets with complete or missing data. SEMF extends the Expectation-Maximization algorithm, traditionally used in unsupervised learning, to a supervised context, leveraging latent variable modeling for uncertainty estimation. Extensive empirical evaluations across 11 tabular datasets show that SEMF often achieves narrower normalized prediction intervals and higher coverage rates than traditional quantile regression methods. Furthermore, SEMF can be integrated with machine learning models like gradient-boosted trees and neural networks, highlighting its practical applicability. The results indicate that SEMF enhances uncertainty quantification, particularly in scenarios with complete data.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes an Expectation Maximization (EM) Approach for supervised learning problems. The algorithm is described and evaluated empirically on several benchmark models and datasets. I am not yet convinced by the approach but happy to change my mind during the discussion period.

### Strengths
- The approach model agnostic and so can in theory easily applied on any supervised learning baseline model.
- The overall algorithm is well described and easy to follow, at least for someone who has worked on EM algorithms.

### Weaknesses
- Using MC sampling to approximate the posterior over the latent variables z is a potentially inaccurate approach, especially as z increases in dimensionality. This is the reason, in Bayesian inference, we would e.g. use MCMC sampling not MC sampling from the prior. Can the authors comment on why this is not a problem in their approach? Asked differently, how large did the number of samples R have to be in their cases to produce good results? How does R affect the quality of the results?
- The results not super convincing across the board. Especially, I was surprised to see the point estimation performance to drop when using the EM approach. Can the authors explain that?
- The missing value setting is strange to me. Why investigate an MCAR setting, which implies ignorable missingness. I believe a MAR setting where missingness is in principle recoverable would be a much more relevant scenario.

### Questions
- Why do we need the double index r, s when sampling first z and then y in EQ 17?  Just sampling z_r and then y_r would be conceptually enough I believe.
- Why does mini-batching make your results unstable? Is that a common reason or something specific to your method?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a novel prediction interval generation framework called SEMF (Supervised Expectation-Maximization Framework). SEMF is a general, model-agnostic approach that can be applied to complete datasets or datasets containing missing data. It extends the traditional EM (Expectation-Maximization) algorithm, which is typically used for unsupervised learning, by applying it to supervised learning for uncertainty estimation through latent variable modeling.

### Strengths
- The SEMF proposed by the authors is a model-agnostic method, meaning it can be integrated with various machine learning models, providing high applicability and flexibility.
- The problem addressed by the authors is often overlooked in the real world, specifically the presence of missing data and the uncertainty estimation of provided predictions.
- The authors conducted experiments on a large number of datasets to validate the effectiveness of their method.

### Weaknesses
- The writing in this paper is unclear. I suggest that the authors introduce the research problem setting either before Section 2 or at the beginning of Section 2, rather than listing formulas.
- The theoretical analysis in the paper assumes independent distributions among the variables, but real-world situations are often more complex. I believe the authors' investigation of this issue is not sufficiently thorough.
- There appears to be a substantial amount of prior research [1, 2, 3] on interval data prediction, which is not mentioned in this paper.

[1] Billard, L. and Diday, E. Regression analysis for interval valued data. In Data Analysis, Classification, and Related Methods, pp. 369–374. Springer, 2000.


[2] Sadeghi, J., De Angelis, M., and Patelli, E. Efficient training of interval neural networks for imprecise training data. Neural Networks, 118:338–351, 2019.


[3] Yang, Z., Lin, D. K., and Zhang, A. Interval-valued data prediction via regularized artificial neural network. Neurocomputing, 331:336–345, 2019.

### Questions
- The authors assume independent distributions among the variables. Based on this assumption, is the hypothesis of the latent variable $z$ not that important? Could we establish the relationship between 
$y$ and $x$ directly instead?
- There is a significant amount of research on interval prediction (which the authors also mention), but the authors do not seem to compare their method with these existing approaches (in the absence of missing data). For cases with missing data, using simple methods (such as interpolation) for comparison would also be straightforward.
- The SEMF method seems to require tuning multiple hyperparameters, such as the number of Monte Carlo samples, the number of latent nodes, and the standard deviation. This may demand considerable experimental and computational resources. Are there any general empirical settings that could be recommended?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces an algorithm for producing prediction intervals (PI) by adapting the EM algorithm in a supervised learning framework. The modeling takes into consideration possible missing input features. The empirical evaluation considers the 95% PI (its coverage and width) against baseline models.

### Strengths
- Adapting the EM algorithm seems pretty novel and the algorithm is well-motivated and sound. 
- There have been many methods introduced in the uncertainty literature which aim to produce prediction intervals, and this work would fit alongside those methods. 
- Consideration for missing data is an interesting application setting which prior works in uncertainty to not consider often. 
- Overall, the writing quality is good and mostly easy to follow

### Weaknesses
- The proposed method incorporates conformal prediction at the end, which guarantees correct coverage. the only other metric is the PI width. In that case, is the benefit of the method just in producing more tightly clustered samples which lead to tighter PI?
- Is it correct to understand the proposed method as just producing samples from the modeled underlying distribution? There are other metrics which are sample-based that could provide a more holistic evaluation of the quality of the predictive distribution, like the energy score.
- As I understand it, this method requires a separate model per input feature dimension, which seems prohibitively expensive, either for large models or large input dimensions.
- Despite the emphasis on adaptability to missing data in the methods sections, its performance on missing data seems a bit unfortunate, but I appreciate the authors' frankness in providing the results.

### Questions
- In L153, why is the $x_1[j]$ notation needed? Isn't this already expressed as $x_{1,j}$?
- In equation 12, what is $x_{1}^{(nm)}[j]$? I don't think $x_{1}^{(nm)}$ is defined.
- I don't quite understand how the missing values were implemented - in L299, when you say "except for the first feature", does that mean the first feature is never missing, and half of the rest of the features are masked out, chosen at random?

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
3

### Summary
Quantification of uncertainty of a given prediction made by the models is critical for a variety of downstream applications. One way to measure the uncertainty of a prediction is to use quantile estimates and corresponding prediction intervals. The paper develops and explores SEMF: a Semi-supervised EM Framework for generating prediction intervals that are model agnostic and can work with incomplete data. The gist of the framework is to convert inputs to a latent space, that is in turn used to predict the outputs, much like autoencoder architectures. The proposed interval estimation framework is tested on 11 problems and three different baseline prediction models.

### Strengths
Importance: Developing methods and models for improving predictions with their uncertainty estimates is significant and important for applications of predicted models especially when incorrect predictions may carry a high risk. Model agnostic framework the work aims to develop would be a great advantage as it does not have to be tailored to specifics of individual models. 

Novelty: The proposed EM framework is novel in context of uncertainty assessment. The methodology of EM and its use in unsupervised and semi-supervised tasks settings is not new.  In addition, the paper introduces a new evaluation metric Coverage-Width Ratio (CWR) that accounts for both the coverage and the precision of the prediction intervals.

Experiments: the experiments are done on 11 problems. Three different prediction methods, and their existing prediction interval estimates are considered. However, the number of baseline models considered does not appear to be sufficient to support the model-agnostic objective. 

Code: Authors provide the code for the reproducibility of the reported results

### Weaknesses
[W1] Intuition. Lack of intuition and argument supporting the benefit of the SEMF framework. The paper says it aims to leverage latent variable modeling framework for uncertainty estimation in predictions. The intuition and justification of this step and design is however not very well argued in the paper. Adding the text covering the intuition aspect would greatly enhance the readability and clarity of the paper and its steps. Along the same lines it would be great to see arguments or intuition why this approach could improve upon alternative quantile estimation methods. 


[W2] Experiment – baselines. The evaluation is limited as it only considers quantile regression as the baseline of comparison that generates prediction interval. Perhaps [1,2], or other relevant methods can be considered as additional baselines. Also, the evaluation is presented on relatively simple datasets: number of features in [7,22] and number of samples in [768, 21K]. It is hard to justify the applicability of the model at scale i.e., with a greater number of features and/or number of samples.

[W3] Experiment- metrics. The results for interval estimates consider PICP, PIW (NPIW) and new CWR metrics, where CWR attempts to combine two different aspects of the interval estimates,  However, there is another existing combined interval score (Gneiting , 2007) that attempts to combine two aspects of the interval estimates and could have been used instead   
T. Gneiting, F. Balabdaoui, AE Raftery. Probabilistic forecasts, calibration and sharpness. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69 (2):243–268, 2007 

[W4] Interpretation of results and conclusions: The results and their discussion are limited, and it is unclear whether the objectives of the development are supported by the experiments and results.  First, using only three model baselines is limiting in terms of support of the model agnostic objective. Second, somewhat surprising are missing data experiments where latent variables in the model should adapt and handle better the data missingness.  What is the insight for these results?  

[W5] Running time: As authors note in the limitations, the usefulness of this approach is limited by its high computation complexity. However, the paper does not report the train and inference time. For completeness, the paper can benefit from including these times to assess the extent of the slowdown caused by SEMF.

### Questions
[W2] Can you elaborate on why you have decided to propose a new score CWR instead of using the existing interval score as suggested in the above comments? 

[W4] Can you please comment on why the results of SEMF on missing data are inferior to results on complete data when compared to baseline methods.

### Soundness
2

### Presentation
3

### Contribution
2
