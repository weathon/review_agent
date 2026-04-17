# Double/Debiased Machine Learning for Time-to-Event Outcomes Under Poor Overlap

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
In empirical studies with time-to-event outcomes, investigators often leverage observational data to conduct causal inference on the effect of exposure when randomized controlled trial data is unavailable. Model misspecification and lack of overlap are common issues in observational studies, and they often lead to inconsistent and inefficient estimators of the average treatment effect. Estimators targeting overlap weighted effects have been proposed to address the challenge of poor overlap, and methods enabling flexible machine learning for nuisance models address model misspecification. However, the approaches that allow machine learning for nuisance models have not been extended to the setting of weighted average treatment effects for time-to-event outcomes. In this work, we propose a class of one-step cross-fitted double/debiased machine learning estimators for the weighted cumulative causal effect as a function of restriction time. We prove that the proposed estimators are consistent, asymptotically linear, and reach semiparametric efficiency bounds under regularity conditions. Our simulations show that the proposed estimators using nonparametric machine learning nuisance models perform as well as established methods given access to correctly-specified parametric nuisance models, illustrating that our estimators mitigate the need for oracle parametric nuisance models. We apply the proposed methods to real-world observational data from a UK primary care database to compare the effects of anti-diabetic drugs on cancer clinical outcomes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a DML-based framework for the WATEs under a survival setting, addressing the competing risks setting by considering the cause-specific restricted mean time lost. The idea is intuitive and well-motivated. I am mainly concerned about the paper's innovation.

However, I feel that DML for any parameter of interest which is a **function of the survival function** (for example, the WATE, which is the weighted RMST estimate depending on the survival curves) should be trivial, as the efficient influence function (i.e., the centered IF) for the survival functions is well-established. Therefore, the corresponding efficient influence function for the parameter of interest can be obtained by applying the **same function** to the efficient influence function. This is not a challenging part of the problem in my view.

Further, the asymptotic properties of the estimator should also be well-established as long as the **same function** is well-controlled (e.g., satisfying Lipschitz continuity).

Thirdly, I do not see any other approach specific to the DML framework for survival outcomes with competing risks being used to address poor overlap, aside from overlapping weighting, which is also well-established.

Therefore, I have reservations about accepting the paper.

### Strengths
The flow and presentation of the paper are well-organized. As a workflow, the authors provide the identification of the parameter of interest (Section 2), the efficient influence function of the parameter (Section 3), and the asymptotic properties in (Section 4), including consistency, maximized variance reduction, and asymptotic linearity. **Although I did not find the claim of asymptotic normality, I suspect it should hold as well.**

In the simulation study (Section 5), the paper compares its method with other singly robust estimators and parametric doubly robust estimators to showcase its efficiency and double robustness to model misspecification with the use of SuperLearner. The subsequent real-data application also aligns with the claims in previous sections.

In summary, the overall paper flow is clear and easy to follow.

### Weaknesses
As I mentioned in the summary, my major concern is the paper's innovation, as the centered IF is well-established for survival functions. Since the WATE parameter is a **known function** of the survival functions, its centered IF should be easily inferred and the asymptotic properties would follow as expected. Therefore, I think the paper may not meet the originality standard for ICLR.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel class of double/debiased machine learning (DML) estimators for weighted average treatment effects (WATEs) in time-to-event outcomes, addressing two major challenges in observational causal inference: poor overlap and nuisance model misspecification. The proposed method leverages cross-fitted DML with influence function-based estimation to achieve consistency, asymptotic linearity, and semiparametric efficiency under regularity conditions. The authors provide an evaluation on simulated data in a variety of settings

### Strengths
This work provides a strong contribution, well motivated and theoretically rigorous. The proposed approach behaves well on simulated and real-world data, which seems promising.

### Weaknesses
The paper is hard to read, and not standalone without the supplementary materials: the references are at the very end, and not with the main text, the abbreviations and details (at least a part) about the baseline methods are also in the supplementary. The article should be readable without the supplementary. It would also have been good to submit the code as an additional material. Without code, the contribution is less valuable as it is much less likely to be used in practice.

### Questions
1. How does the proposed method compare to the baseline methods on the UK real data?
2. What are the resources needed to run the method (time and memory)? how does it scale with the number of observations?
3. Could the authors be explicit about whether they make the proportional hazard assumption?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a DML estimator with poor overlap from observational time-to-event data. The authors derive theoretically rigorous estimators for weighted Restricted Mean Survival Time (RMST) and Restricted Mean Time Lost (RMTL), proving they are consistent, asymptotically linear, and achieve semiparametric efficiency bounds.

### Strengths
1. The paper tackles a highly practical and important problem. The estimator in this setting is novel and rigrous.
2. The work fills a clearly identified and non-trivial gap in the literature. While DML, WATEs, and survival analysis are established fields, their synthesis is novel.
3. The simulation study is exceptionally well-designed.

### Weaknesses
1. I'm a little curious whether the method is practical, especially for the estimation of nested conditional expectation over time.
2. What are the exact nuisance parameters? In lines 270-271, I don't find Eqn. 3 and Eqn. 4. In Section 3.1, it seems that many nuisances need to be estimated. So how should the estimator be "doubly-robust"? Where the product of nuisance model convergence rates must be at least root-n?
3. How to choose the tilting function? I think this is also an important quantity that might influence the estimator's statistical property and implementation details. But in line 120, the paper mentions it briefly in passing.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a class of double machine learning estimators for weighted average treatment effects. Specifically, it targets the weighted cumulative causal effect as a function of time in terms of the event-free survival time. The method is also extended to the competing risks setting. The proposed method is shown to be consistent, asymptotically linear, and semi-parametrically efficient.

### Strengths
- The paper provides mathematical guarantees on consistency, asymptotic linearity, and semi-parametric efficiency

### Weaknesses
- Motivation: In which setting would the practitioner be interested in the WATE instead of the individualized CATE? The practical value of the method is difficult to assess.
- Mathematical notations are not introduced appropriately. For example, Theorem 1 states many quantities that are neither explained before nor after the theorem.
Therefore, the theorem is unclear to the reader. Furthermore, the correctness and the meaning of the theorem cannot be checked. The same holds for Section 3.2. 
- Important mathematical derivations are pushed to the appendix. For example, the expression for (and meaning of) $\phi$ is never stated in the main paper.
- The paper completely lacks a discussion of related work. This does not only include DML for the WATE, but also approaches for the ATE and CATE, as in these settings, DML for poor overlap (not only treatment overlap, but also censoring and survival overlap) has already been developed. Therefore, it is difficult to evaluate the novelty and originality of the method.
- Theorems are not proven in the main paper, and neither are proofs in the Appendix directly referenced. 
- The empirical evaluation is weak, not well explained, and does not necessarily support the proposed method (e.g., Fig.2)

### Questions
- Theorem 1: What is $\phi$? What do the different superscripts refer to? 
- Section 3.2: What is meant by "correcting for shape constraints thereafter" in step 4? This is not explained in the paper.
- Section 3.2: What exactly is needed to estimate here? The stated formulas are never described. It is not possible to check the correctness of the procedure.
- Choice of function h: How can one choose h in practice? What is the effect of different (potentially suboptimal) choices on the target and the estimated effect?

### Soundness
2

### Presentation
1

### Contribution
2
