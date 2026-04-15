# Prediction Risk and Estimation Risk of the Ridgeless Least Squares Estimator under General Assumptions on Regression Errors

- Decision: Reject
- Scores: 5, 5, 6

## Abstract
In recent years, there has been a significant growth in research focusing on minimum $\ell_2$ norm (ridgeless) interpolation least squares estimators. However, the majority of these analyses have been limited to a simple regression error structure, assuming independent and identically distributed errors with zero mean and common variance. In this paper, we explore prediction risk as well as estimation risk under more general regression error assumptions, highlighting the benefits of overparameterization in a \emph{finite} sample. We find that including a large number of \emph{unimportant} parameters relative to the sample size can effectively reduce both risks. Notably, we establish that the estimation difficulties associated with the variance components of both risks 
can be summarized through the trace of the variance-covariance matrix of the regression errors.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the prediction risk and estimation risk of the ridgeless least squares estimator. The main contribution is that the i.i.d. assumption is dropped in the theoretical analysis. The critical assumption is left-spherical symmetry for the distribution of the design matrix. Under those assumptions, the authors derived an accurate formula for the prediction error for ridgeless LSE with the high dimensional model and finite data set. Some numerical experiments show that the numerical results agree with the theoretical findings.

### Strengths
- The rigorous evaluation of the prediction and estimation errors is presented for the high-dimensional model using finite samples.
- The authors introduced the left-spherical symmetry as a critical assumption in the theoretical analysis.

### Weaknesses
- The non-i.i.d. noise seems a minor extension of existing works.

- The left-spherical symmetry is an interesting assumption to analyze the ridgeless LSE, the relationship between the left-spherical symmetry and the double descent phenomenon is not sufficiently investigated. Surely, the left-spherical symmetry is useful to derive the explicit expression of the risk. However, the interpretation or meaning of the assumption of the double descent is not sufficiently elucidated.

- In numerical experiments, only the models that agree to the assumption for Theorem 3.2, and 3.3 are used. The readers may be interested in how much the theoretical analysis matches numerical experiments. In other words, the authors could investigate how robust is the theoretical findings to the violation of the assumption.

### Questions
- Please make clear the technical difficulty of dealing with the non-i.i.d. noise assumption. 

- The left-spherical symmetry is an interesting assumption to analyze the ridgeless LSE; the relationship between the left-spherical symmetry and the double descent phenomenon is not sufficiently investigated. Surely, the left-spherical symmetry is useful to derive the explicit expression of the risk. However, the interpretation or meaning of the assumption of the double descent is not sufficiently elucidated. Is it possible to provide a more detailed description of the relationship between the left-spherical symmetry and the double descent phenomenon? 

- In numerical experiments, only the models that agree to the assumptions for Theorem 3.2 and 3,3 are used. The readers may be interested in how much the theoretical analysis matches numerical experiments. In other words, the authors could investigate how robust the theoretical findings are to violating the assumption. Is it possible to add numerical experiments for checking the robustness of the theoretical findings to the violation of the assumption?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper delves into the assessment of prediction risk and estimation risk, expanding the scope to accommodate more general regression error assumptions. It underscores the advantages of overparameterization in the context of finite samples, revealing that the inclusion of a substantial number of seemingly inconsequential parameters relative to the sample size can effectively mitigate both types of risk.

Despite the paper's technical nature and the wealth of analytical content, some aspects warrant further attention. Notably, main results such as the ones presented in Theorems 3.2, 3.3, and Corollary 4.1 lack comprehensive elucidation, leaving it unclear how these outcomes relate to the core assertion regarding the benefits of overparameterization or unimportant parameters. Moreover, there are instances of imprecise writing, such as the reference issue in Section 4.2, where "we can obtain a similar result with 4.1" appears to be a misreference.


====

I acknowledge that I have considered the authors' response, yet after careful deliberation, I have chosen to maintain the current score.

### Strengths
The paper seems to be technically sound.

### Weaknesses
main results such as the ones presented in Theorems 3.2, 3.3, and Corollary 4.1 lack comprehensive elucidation, leaving it unclear how these outcomes relate to the core assertion regarding the benefits of overparameterization or unimportant parameters.

### Questions
See "weakness".

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates the prediction risk and the estimation risk of ridgeless least squares estimator in an overparametrized regime where the number of samples $n$ is less than the number of variables $p$. The main interest of this work is that it addresses non i.i.d. regression errors. Notably the expected value of the estimator variance at finite $n<p$ is found to depend on the sum of the variances of the regression errors, ignoring the correlations between regression errors. As the bias of the estimator is independent of the regression errors, the prediction risk and the estimation risk exhibit the same behavior.

### Strengths
- The article is well motivated. Removing the i.i.d. condition on the regression errors is indeed interesting for studying data such as time series.

- The presentation is sufficiently clear although the nature of the theoretical findings could be better explained (see Weaknesses).

- The theoretical results are confirmed by experiments.

### Weaknesses
- It seems that the main theorems do not give direct access to the relations between the deterministic parameters underlying the data generating process and the learning risks, except in the asymptotic regime of $n,p\to\infty$. If that is the case, the nature of the contributions should be made clearer to stress that point.

- It appears that while allowing dependences between regression errors, the proposed analysis in the ridgeless overparametrized setting shows that the performance stays the same whether or not the regression errors are independent,  as long as the sum of their variances is unchanged. The limitations of this work and the possible extensions should be better discussed in that regard. For instance, what would be the main technical difficulties to extend the analysis to ridge regularization and underparametrized regime, and would the dependences between regression errors have an impact on the learning performance in those settings?

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
