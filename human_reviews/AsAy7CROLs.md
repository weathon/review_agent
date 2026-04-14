# Prediction Risk and Estimation Risk of the Ridgeless Least Squares Estimator under General Assumptions on Regression Errors

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
In recent years, there has been a significant growth in research focusing on minimum $\ell_2$ norm (ridgeless) interpolation least squares estimators. However, the majority of these analyses have been limited to an unrealistic regression error structure, assuming independent and identically distributed errors with zero mean and common variance. In this paper, we explore prediction risk as well as estimation risk under more general regression error assumptions, highlighting the benefits of overparameterization in a more realistic setting that allows for clustered or serial dependence. Notably, we establish that the estimation difficulties associated with the variance components of both risks can be summarized through the trace of the variance-covariance matrix of the regression errors. Our findings suggest that the benefits of overparameterization can extend to time series, panel and grouped data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the prediction and estimation risks of the ridgeless least square regression under a more general assumption on the noise. Specifically, consider the data
$$
y_i = x_i^\top\beta + \epsilon_i,\ i=1,...,n
$$
with target $\beta\in\mathbb{R}^p$ and $ \epsilon_i$ is the noise independent of $x$ with $\mathbb{E}[\epsilon]=0$ and $\mathbb{E}[\epsilon\epsilon^\top]=\Omega$ finite and positive definite. This includes the case of i.i.d. Gaussian noise, autoregressive noise and cluster noise.

This paper applies the classical bias-variance decomposition onto the risks and finds a closed form expression for the variance term for both risks.

### Strengths
This paper provides rigorous proof and experiments to validate their claim. The assumption on the noise and the input is more general than previous works.

### Weaknesses
However, my biggest concern is the significance of the contribution provided by this paper. 

As mentioned in line 132-133, this paper is not the first to consider non-i.i.d. Gaussian noise. Although this paper requires less assumptions on the noise than [1], it does not contain enough illustrations or interpretations on their main results (Theorem 3.4, 3.5) for "allowing potentially adversarial errors" as promised in line 142-143. Indeed, this paper does not explain how their main results could recover previous result with i.i.d. Gaussian noise or gain new insights with more general noise.

Also, it seems that the techniques used in the main results are rather standard and can be extended easily to more general settings like kernel ridge regression or kernel gradient flow, which could potentially increase the significance of this work.


Reference:
[1] Geoffrey Chinot and Matthieu Lerasle. On the robustness of the minimum ℓ2 interpolator.
Bernoulli, 2023.

### Questions
I believe this paper could improve its significance if it can answer the following questions:

1. What new insights could one gain from Theorem 3.4, 3.5?

2. Related to Q1, the true noise covariance $\Omega$ is not observable in reality. Could the authors provide any examples or algorithms to approximate such noise covariance in real-world datasets? If it is impractical to approximate it, how could we still analyse the risk with your risk expression in $\Omega$? 

3. Could the authors extend their results to more general settings like kernel ridge regression or kernel gradient flow? Or at least explain what are the technical difficulties that might hinder the extension?

4. One Central idea of regularization is to balance out the effect of the noise in the labels. With the ridgeless linear regression setting, this paper unfortunately misses the opportunity to discuss the important interplay between the non-i.i.d. noise and regularization, which I find quite disappointing.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper extends the analysis of ridgeless least squares estimators to more
realistic error structures beyond the traditional i.i.d. assumptions. It
addresses both prediction and estimation risks in settings where regression
errors may be correlated or exhibit heteroscedasticity.

The authors provide exact finite-sample expressions for both types of risk,
which are decomposed into bias and variance components. The variance in
prediction risk is shown to be the product of two distinct terms: one related to
the error covariance matrix and the other dependent on the feature distribution.

The paper also conducts a systematic asymptotic analysis of prediction and
estimation risks. Numerical experiments with autoregressive and clustered
regression errors illustrate the theoretical findings.

### Strengths
Originality:

This paper extends the analysis of ridgeless least squares estimators by
relaxing the i.i.d. assumption on the regression errors. Correlated and
heteroscedastic errors are frequently encountered in practice but are not
investigated in prior works in the high-dimensional setting. This work fills a
gap in understanding the performance of least squares estimators, extending the
observation of double descent to more general settings.

Quality:

The paper is technically sound and comprehensive in its treatment of both
prediction and estimation risks. The finite-sample and asymptotic behaviors
finite-sample results are rigorously presented. The numerical result in Section
1 demonstrates the motivation of the generalized assumptions and the results in
Section 3.3 help validate the theoretical findings.

Clarity:

The paper is overall well written with clear presentation.

Significance

The paper extends the theory of least squares estimators to include the
situation with non i.i.d. errors in the context of overparameterization. It fill
a gap in the literature, so the results deserve to be documented.

### Weaknesses
For the high-dimensional setting considered in the paper, ridgeless least
squares estimators are seldom applied in practice. Regularization is almost
always helpful, and there is a large body of literature demonstrating the
advantages of regularized least squares. The paper should at least discuss this
line of research. In particular, it has been shown that optimized ridge
regression avoids bias inflation. Why, then, would practitioners care about
ridgeless least squares estimators in this scenario?

While the assumptions are more general than the i.i.d. case, the scenario
considered is still much simpler than what is encountered with real, practical
data. For example, the assumption of left-spherical symmetry for the design
matrix is limiting in practical scenarios. Most real-world datasets feature
asymmetrical or skewed distributions, and many variables in real data are
actually categorical, rather than numerical. It would help to add discussions on
the limitations of the current investigation in these contexts.

Most of the theoretical results, especially the finite-sample results, appear
similar to the low-dimensional or fixed-$p$ case. It would be helpful to include
discussions on this point. If the results are not the same, do they reduce to
the low-dimensional results? Even if they have the same expression, it would be
insightful to explain why it is nontrivial to derive these results in the
high-dimensional setting.

### Questions
1. Include discussions on relevant regularized least squares estimators, and
   provide scenarios where the ridgeless least squares estimator excels or
   underperforms.
2. Provide comparative experiments with other regularization techniques to
   demonstrate where ridgeless least squares excels or underperforms.
3. Add discussions on the limitations of the current investigation, particularly
   in terms of the assumptions.
4. Provide discussions connecting the results in the high-dimensional setting to
   those in the low-dimensional setting.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors investigate prediction risk and  estimation risk under more general regression error assumptions beyond i.i.d. errors. In particlar, they explore the benefits of overparameterization in a more realistic setting, which allows for clustered or serial dependence. This paper  demonstrated   that the estimation difficulties associated with the variance components  can be summarized through the trace of the variance-covariance matrix of the regression errors.

### Strengths
The paper is written clearly.  I appreciate the significant contribution to high-dimensial data analysis.  The new approach is  promising for 
 more  broad framework. This paper attacked the very challenging problem involved in least-squares estimators beyond the  assumiption with  i.i.d.  errors. The new idea about  benefits of  over-parameterization could be extended  to time series, panel and grouped data, etc.  with the broad impact.

### Weaknesses
The paper proved several  good theoretical results and properties. The experiemts of data set may not be comprehensive, Since there is no comparison of  the  proposed method with existing methods.   In addition,  the proposed approach only works for  the case  p>n. In practice,    the ultra-high dimensional case with p=exp(n) is very  common.  This is a more realistic setting in the big data era.

### Questions
I have several comments and suggestions for the authors to address.


1. The paper is not complete. Please add the conclusion section in the revision. 

2. It is worthwhile to extend the proposed approach from the case  p>n   to the ultra-high dimensional case with p=exp(n). 

3. It is of interest for the authors to  compare   proposed method with existing ones  (including  Chinot et al. (2022) and Chinot & Lerasle (2023), etc.)  in the experiments.

4. The new idea about benefits of over-parameterization could be extended to time series, panel and grouped data.  It is helpful  for the authors to  elaborate it by proving more details.

5. There are some  typos, grammatical errors,  etc. in the paper. Please check it in the revision carefully. 

in page 1, line 022, "can extend" -> "can be extended". 

in page 5, line 312, "appendix" -> "Appendix".

in page 9, line 432, "appendix" -> "Appendix".

in page 10, line 500, "appendix" -> "Appendix".

in page 14, line 704, "Figure" -> "Figures".

in page 17, line 905, "6" -> "(6)".

### Soundness
3

### Presentation
3

### Contribution
3
