# STABLE ESTIMATION OF SURVIVAL CAUSAL EFFECTS

- Decision: Reject
- Scores: 5, 6, 3

## Abstract
We study the problem of estimating survival causal effects, where the aim is to characterize the impact of an intervention on survival times, i.e., how long it takes for an event to occur. Applications include determining if a drug reduces the time to ICU discharge or if an advertising campaign increases customer dwell time. Historically, the most popular estimates have been based on parametric or semiparametric (e.g. proportional hazards) models; however, these methods suffer from problematic levels of bias. Recently debiased machine learning approaches are becoming increasingly popular, especially in applications to large datasets. However, despite their appealing theoretical properties, these estimators tend to be unstable because the debiasing step involves the use of the inverses of small estimated probabilities---small errors in the estimated probabilities can result in huge changes in their inverses and therefore the resulting estimator. This problem is exacerbated in survival settings where probabilities are a product of treatment assignment and censoring probabilities. We propose a covariate balancing approach to estimating these inverses directly, sidestepping this problem. The result is an estimator that is stable in practice and enjoys many of the same theoretical properties. In particular, under overlap and asymptotic equicontinuity conditions, our estimator is asymptotically normal with negligible bias and optimal variance. Our experiments on synthetic and semi-synthetic data demonstrate that our method has competitive bias and smaller variance than debiased machine learning approaches.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors explored the causal survival effects problem. They contended that conventional approaches relying on inverse weighting were prone to instability, where minor inaccuracies in weight estimation could result in substantial errors in the overall estimation. To enhance their estimations, they explored the expansion of the survival function via Taylor's expansion and leveraged the Reisz representation theorem for derivative estimation.They also demonstrated that their method exhibits asymptotic efficiency under a specific set of conditions.

### Strengths
The utilization of the Reisz representation theorem for estimating the derivative of the survival function is a clever innovation. Their definition of the Reisz representer in the context of inverse probability weights serves to circumvent the issues encountered in methodologies reliant on inverse weighting.

### Weaknesses
- The proposed method appears to operate on a population level, yet all steps involve conditioning the hazard function on covariates. I expected to see individual-level results, and it's unclear how well the method performs at the individual level.
- In section 2.2 (identification), there appears to be a typo in A3.
- It's not evident which distribution the expectations are calculated on in the text.
- The natural proxy used for hazard estimation can be quite noisy, especially in small datasets.
- In section 2.3, there is a change in notation from \lambda(x, a) to \lambda(a, x). Consistent notation would be preferable.
- The proof of Lemma 2.1 requires a correction; the survival function should be denoted as P(T > t).

### Questions
- Could you please explain the significance of the second metric, bias/stde?
- I noticed that in both the synthetic experiments (prior to t=15) and in the twins dataset, your bias exceeds the odds ratio (OR). What accounts for this discrepancy?
- If the odds ratio (OR) is already displaying high bias, why did you introduce the concept of relative RMSE? Why not employ RMSE and compare it to the OR in addition?
-In the section on covariate balancing, the paper references Xue et al. 2023, which assumes independent censoring. I believe your method makes a similar assumption. If this is not the case, could you please clarify?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors study the problem of estimating survival causal effects, where the aim is to characterize the impact of an intervention on survival times, i.e., how long it takes for an event to occur. This paper in general is interesting.

### Strengths
In causal survival effect estimation, classic estimators involve inverse probabilities are a product of treatment assignment and censoring probabilities. The authors propose a covariate balancing approach to estimating these inverses directly, sidestepping this problem.

### Weaknesses
I have a couple of comments:

The notation might not be fully rigorous. For example, P sometimes refers to probability, and sometimes refers to density.

Theorem 3.5 provides asymptotical linearity of $\hat \psi$ with balancing weights estimated by (19). Is the choice of (19) in some sense optimal? Could the authors comment on the variance of $\hat \psi$? Is it optimal/minimal?

A recently proposed causal survival forest (Estimating heterogeneous treatment effects with right-censored data via causal survival forests, Cui et al. 2023 JRSSB) can also be used to evaluate $\psi^{a,t}$. A comparison in simulations would be helpful.

By covariate balancing, Hirshberg & Wager (2021) are able to handle high dimensional data, can the proposed method deal with high-dimensionality?

### Questions
Please refer to the Weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new way to estimate the causal effect in survival analysis.

### Strengths
The idea of this paper is interesting to investigate.

### Weaknesses
This paper is written in a very confusing way, which greatly reduces its value. Overall, the structure of the paper is confusing, the language is vague and uninformative, the proposed methodology is poorly described, and the technical details are not well explained. I list several specific points:
1. Abstract: the authors spend too much time discussing other things and make little mention of the proposed methodology and describes it vaguely.
2. Introduction: related work (section 4) should be moved to introduction.  I also recommend that authors only cite and discuss literature that is truly relevant/important to their work.
3. Setting: section 2.3 is really confusing. A lot of notation without explanation. It is more natural to introduce notation only when it needs to be used.
4. Approach: it reads like a stack of math techniques. Unimportant derivations or formulas can be omitted. More explanation would be helpful.
5. Experiment: the settings should be describe explicitly instead of citing other papers.

### Questions
1. Why do you focus on discrete survival analysis? Can your method be generalized to continuous time?
2. Is the asymptotic normality of your estimator examined in your experiments?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor
