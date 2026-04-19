# Fitting Networks with a Cancellation Trick

- Decision: Accept (Poster)
- Scores: 5, 5, 8, 5

## Abstract
The degree-corrected block model (DCBM), latent space model (LSM), and $\beta$-model are all popular network models. We combine their modeling ideas and propose the logit-DCBM as a new model. Similar as the $\beta$-model and LSM, the logit-DCBM contains nonlinear factors, where fitting the parameters is a challenging open problem. We resolve this problem by introducing a cancellation trick. We also propose R-SCORE as a recursive community detection algorithm, where in each iteration, we first use the idea above to update our parameter estimation, and then use the results to remove the nonlinear factors in the logit-DCBM so the renormalized model approximately satisfies a low-rank model, just like the DCBM. Our numerical study suggests that R-SCORE significantly improves over existing spectral approaches in many cases. Also, theoretically, we show that  the Hamming error rate of R-SCORE is faster than that of SCORE in a specific sparse region, and is at least as fast outside this region.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes logit-DCBM (Degree Corrected Block Model) for community detection in graphs. Authors propose a method to overcome the presence of non-linear terms in the estimation problem. Authors refer to their method as the "cancelation trick". They propose an iterative method dubbed R-score (R for Recursive) which recursively applies the cancellation trick to the estimate. Authors provide theoretical results that guarantee the estimate is close to the true value.

### Strengths
Literature is well reviewed and contributions are well positioned with respect to existing art. Authors introduce their contributions with thorough explanations that are explained in easy-to-read language, despite the large amount of notation and math involved. Theoretical results are well presented and seem to indicate the merit of the method.

### Weaknesses
Presentation of this rather theoretical paper could benefit from a more formal statement of results with a Theorem which goes earlier in the paper and the body of the paper being devoted to explaining the intuition and implications of the result. 

I do not find a clear indication of stopping criteria in the paper, neither did I find any discussion on computational complexity and scalability of the method. 

Numerical experiments are not well presented. At the very least figures could have x and y labels.

### Questions
Numerical experiments do not show any benefit in iterating: after a single iteration the method seems to have converged. Can you explain this? was the test data too easy? did you try running your model on more challenging problems?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces the logit-DCBM as an extension of popular network models DCBM, LSM, and $\beta$-model. The authors propose a "cancellation trick" to tackle the nonlinearity challenges of the logit-DCBM, which allows the model to retain a form of low-rank approximation similar to DCBM. They also present an algorithm, R-SCORE, for community detection in networks based on the logit-DCBM. R-SCORE iteratively updates the model's parameter estimates, removing nonlinear factors and approximating a low-rank model for clustering. The paper shows that R-SCORE achieves a faster Hamming error rate in certain sparse regions than standard SCORE.

### Strengths
The logit-DCMB model provides a nice combination of the DCMB, LSM, and $\beta$ models. 

The "cancellation trick" allows disregarding of nonlinear terms by means of a ratio of two large sums used as estimators. This is an interesting idea that can be applied to various other settings.

### Weaknesses
The scope of the paper is [quote] "to propose a nonlinear version of DCMB so that it will hopefully be more acceptable" and extend SCORE to it. I find this goal somewhat incremental and only marginally related to the ICLR community. 

The paper is hard to follow, with many acronyms that are not spelt out and sentences that are ambiguous. An example: line 085, "The logit-DCBM (and all other models mentioned above) are so-called latent variable models, where Π is the matrix of latent variables. For latent variable models, the EM algorithm (e.g., Dempster et al. (1977)) is a well-known approach." What is EM? "approach" to do what?

### Questions
In eq (16), how can you guarantee |\lambda_{min}|>0?

In Thm 3.1, Lemma 3.1, Thm 3.2, Corollary 3.1: What is C? Are the C used the same across the three results?

In the simulations, can't you compare with additional baselines based on e.g. a MLE?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes R-SCORE (Recursive-SCORE), a generalization of the SCORE spectral algorithm to estimate the parameters and communities in a newly introduced "nonlinear" extension of the DCBM model [3] where a logit function is added to the probability. The R-SCORE algorithm alternatingly optimizes the estimates of the core matrix $\Pi$ and the offset terms $\theta$ with the SCORE algorithm and a custom refitting step respectively.  Theoretical results are shown both for the application of the original SCORE algorithm to the proposed extension logit-DCBM and to the application of the newly proposed R-DCBM algorithm





***References***

[1] Jin J. 2015 Annals of Stats.  "Fast community detection by SCORE"
[2] Jiashun Jin, Zheng Tracy Ke, Shengming Luo, and Minzhe Wang. "Optimal estimation of the number of network communities". Journal of the American Statistical association, 2023. 

[3] Karrer, B. and Newman, M. E. J. (2011). Stochastic blockmodels and community structure in networks

### Strengths
1. Well-written
2. Appears rigorous and professional
3. A generalization of the DCBM model to the logit case is an interesting direction

### Weaknesses
Accessibility to readers less familiar with the literature on community detection

No table of notations

### Questions
Could you compare your rates with the classic rates for the simpler SBM model of Abbe et al. when the thetas are equal?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors first propose the logit-DCBM model as a nonlinear variant of the degree-corrected block model (DCBM), which essentially replaces the log link function with the logit link function. Replacing the link from log brings some nice properties to the model, such as removing some constraints on latent parameters. However, at the cost of introducing non-linearity and the parameter estimation becomes difficult.

The main technique is the cancellation trick, used to estimate nonlinear terms in the logit-DCBM, which simplifies parameter estimation and the authors claim that this addresses an open problem in the $\beta$-model parameter estimation (another one community variant which is the special case of their logit-DCBM). Additionally, the paper introduces R-SCORE for community detection, a recursive adaptation of the SCORE algorithm for the logit-DCBM. Finally, they derive upper bounds on the classification error rates for SCORE and R-SCORE for their proposed model logit-DCBM.

### Strengths
See summary.

### Weaknesses
Overall, I am not too sure why the logit-DCBM model is worth studying. The authors mention some motivation in lines 58-60 and lines 69-71, but I did not find it compelling enough. I found the motivation argued hand-heavily without any concrete and interesting reasons.
Line 58, ",to many statisticians, (5) is preferred.":  Which statisticians? References?
Line 59-60: I don't think the logistic regression is the only "recommended" model by text bool for binary data. Also, what does "recommending" even mean? Where does Hastie et al. 2009 recommend it?

See also questions. Questions 1 and 2 in particular, where I further raised my concern.

It is perhaps interesting to study logit-DCBM, but it is not clear to me right now based on what is written on the paper, and that's why I am currently leaning towards a borderline reject.

### Questions
1. I could not exactly make sense of the open question in the references given e.g. (Goldenberg et al., 2010). What is the open problem precisely? To analyze MLE? Or just to come up with an estimate? Or to come up with an estimate that satisfies certain properties? What is the precise technical open question?
2. What properties does your estimate have? As far as I can think, it is not unbiased, or is it? Is it consistent? What can you say further? For me, right now, it is just some estimate.
3. Ultimately, I really want to get a better understanding of Lemma 3 and how good this estimate is. I understood the statement and algebra used therein. But is there something more interesting? What is the effect of $m$ on the quality of the estimate? Does it not change anything in terms of quality? Or even if it does, we are looking at the statistics for $m=3$ for computational efficiency? Also, is there an intuitive way of thinking about what that statistics is capturing in (10) and its equivalent for higher $m$? And why I can expect it to be a reasonable estimate? Essentially, trying to get intuition on why cancellation happens. 
4. Does the cancellation trick also apply to other link functions beyond logit? In either yes or no, could you please walk me through why for some standard link functions (e.g. probit or other relevant ones)? 

Typos:
-In Lemma 2.1 Proof, is there an extra summation over $i$ again in $(I)$?
-In Line (222): ",the sum is over..",the index $i_2$ is repeated twice.

### Soundness
3

### Presentation
2

### Contribution
2
