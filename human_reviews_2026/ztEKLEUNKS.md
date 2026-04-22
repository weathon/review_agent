# Conformal Prediction with Corrupted Labels: Uncertain Imputation and Robust Re-weighting

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
We introduce a framework for robust uncertainty quantification in situations where labeled training data are corrupted, through noisy or missing labels. We build on conformal prediction, a statistical tool for generating prediction sets that cover the test label with a pre-specified probability. The validity of conformal prediction, however, holds under the i.i.d assumption, which does not hold in our setting due to the corruptions in the data. To account for this distribution shift, the privileged conformal prediction (PCP) method proposed leveraging privileged information (PI)---additional features available only during training---to re-weight the data distribution, yielding valid prediction sets under the assumption that the weights are accurate. In this work, we analyze the robustness of PCP to inaccuracies in the weights. Our analysis indicates that PCP can still yield valid uncertainty estimates even when the weights are poorly estimated. Furthermore, we introduce uncertain imputation (UI), a new conformal method that does not rely on weight estimation. Instead, we impute corrupted labels in a way that preserves their uncertainty. Our approach is supported by theoretical guarantees and validated empirically on both synthetic and real benchmarks. Finally, we show that these techniques can be integrated into a triply robust framework, ensuring statistically valid predictions as long as at least one underlying method is valid.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the challenge of preserving the validity guarantees of Conformal Prediction (CP) when training and test data are non-exchangeable due to label corruption or missing information. The authors propose three complementary approaches to restore reliable coverage: Privileged Conformal Prediction (PCP), which reweights calibration samples using privileged information to correct for corrupted labels; Uncertain Imputation (UI), which imputes missing or noisy labels while injecting uncertainty to prevent overconfident intervals; and a Triply Robust CP method that ensures valid coverage if any of these assumptions hold. The paper provides theoretical proofs of coverage guarantees under each setting and supports them with synthetic and real-world experiments. Overall, it extends CP to corrupted-label scenarios and offers a practical framework for robust uncertainty quantification beyond the standard exchangeability assumption.

### Strengths
Conformal Prediction (CP) to data settings with label corruption and non-exchangeability. It also explores solutions for cases where the fundamental exchangeability assumption of traditional CP is violated. This is a problem that has received little attention in previous research.


The proposed combination of Privileged Conformal Prediction (PCP), Uncertain Imputation (UI), and Triply Robust CP creatively integrates ideas from conformal prediction, causal inference, and robust statistics, building a novel framework for uncertainty quantification under data corruption.


In terms of research quality, the theoretical analysis in the paper is rigorous, the assumptions are clearly stated, and precise coverage guarantees are provided under different robustness conditions. In terms of clarity, the paper is well-structured, with clear motivation and consistent notation. Although the technical content is deep, the core ideas are clearly expressed and easy to understand.

### Weaknesses
First, the distinction between exchangeability and i.i.d. should be made explicit early in the paper, particularly in the abstract. Currently, the text occasionally treats these terms as interchangeable, but this is conceptually inaccurate: exchangeability is a strictly weaker assumption than i.i.d., and making this distinction clear would improve both theoretical rigor and readability.

Second, the robustness assumptions underlying each proposed method (e.g., $((X,Y)\perp M|Z$) for PCP and accurate conditional modeling $(Y|Z)$ for UI) are relatively strong and may not hold in many real-world corruption processes. A more detailed discussion or empirical sensitivity analysis would help clarify how performance degrades when these assumptions are violated.

Third, while the theoretical analysis is rigorous, the implementation aspects (such as how to estimate the weighting function ($w(Z)$) or residual noise in high-dimensional settings) are under-explained and may limit reproducibility.

Finally, the experimental evaluation, though adequate, could be expanded to include additional diagnostics such as calibration error or coverage-interval efficiency, and more intuitive visualizations would help convey how weighting and imputation mechanisms restore validity.

### Questions
Exchangeability vs. i.i.d.: Please clarify the precise assumption used in your theoretical results. Some parts (e.g., the abstract) seem to equate exchangeability with i.i.d., though the former is weaker.

Assumption robustness:
How sensitive are PCP and UI to violations of their key assumptions? A short sensitivity or ablation analysis would clarify how performance degrades when these assumptions fail.

Implementation details:
Provide more detail on how the weighting function ($w(Z)$) and residual noise in UI are estimated in practice, particularly in high-dimensional settings, to improve reproducibility.

Evaluation metrics:
Consider adding calibration error or conditional coverage metrics to complement coverage and interval width, giving a fuller view of empirical performance.

Relationship to Literature: How do the authors' findings compare with the existing literature for CP with ambiguous ground truth in the calibration data, that is https://openreview.net/forum?id=L7sQ8CW2FY and https://proceedings.neurips.cc/paper_files/paper/2024/hash/d42a8bf2f40555d4a5120300f98c88f6-Abstract-Conference.html ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the label corruption in conformal prediction. While conformal prediction assumes exchangeability between the calibration points and the test point, if a subset of calibration points have corrupted labels (specifically if the corruption is non-random or dependent to a feature) then the exchangeability breaks. The paper is focused in the case that there is a privileged information either indicating the faulty label or carrying information about the true label. 

For the case where the information indicates the corruption label, one existing approach - a.k.a. PCP - is to treat each calibration point as a test point and calibrate on the rest of non-corrupted points. With those scores we can find a threshold for the left out calibration point. Then the conformal threshold will be the 1 - beta quantile of those threshold. For this case the authors evaluate the robustness of PCP under various noise in the weights of the weighted conformal prediction. They further characterize under which cases does the PCP sets violate the coverage guarantee.

Additionally they also propose a label imputation setup by splitting the calibration and adding the label-estimation error to the labels.

### Strengths
1. The problem is applicable in many cases when there is any type of label noise. 
2. The authors approached the problem in an organized way. They clearly break down the cases where the labels are faulty, and address each case separately.
3. The theoretical contribution of the paper is considerable. 

In total while the problem is not clearly defined and solved I think the contribution in theory is above the standard for acceptance.

### Weaknesses
1. In general I could not connect the theorems in Section 3.1 to derive a robustness guarantee and a well delivered understanding of what the procedure is and what the guarantee would be. This is a shortcoming in the application as it is not clear what assumption holds.
2. Minor writing points: Line 176 the term indicator is used twice, maybe you can drop the second one. 
3. Why the authors even discuss the setup with the constant noise on the weights? Isn’t it too unrealistic?
4. The definition of sufficiently accurate does not conform with the term used. Here the only requirement for the error is to be independent from conditional to Z.

### Questions
1. I can not understand the PCP setup. What is the Q(Z_i)? Do you compute WCP with 1 - alpha threshold for it? Does your theorem 1 hold for any values of beta?
2. Are you sure about line 190? Shouldn’t it be the training and calibration distribution?
3. I am not sure about it, but seems that the theorems in Section 3 are relying on the conditional assessment of error. If so, is this possible in real datasets? Because this means that you are intuitively assuming that you already have access to the clean label that you can estimate its corruption.
4. Is the entire requirement in Theorem 4 for the noise of the classifier to be independent from X conditional to Z? Does any classifier work? Is this even possible to be independent to X conditional to Z?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzes the robustness of existing weighted and privileged conformal prediction methods under inaccurate weight estimation and introduces a new Uncertain Imputation approach for corrupted labels.

### Strengths
1- The paper presents a solid and technically sound theoretical analysis of conformal prediction methods under label corruption, with clear insights into how PCP and WCP behave when weight estimates are inaccurate.

2- The manuscript is clearly written, logically structured, and easy to follow despite the technical content.

### Weaknesses
Please check questions!

### Questions
1- While the paper provides solid theoretical analysis, it would strengthen the work if the authors could elaborate on the computational overhead of combining PCP, UI, and CP in the Triply Robust scheme

2-  The assumptions underlying Theorem 4,may not hold in many practical scenarios. It would be valuable for the authors to discuss how realistic these assumptions are and whether the proposed method remains approximately valid when they are violated.

3- It would be helpful if the authors clarified what predictive models were used in their experiments

4- Can the authors comment on whether their theoretical results extend to more general or data-dependent weight estimation errors, beyond the fixed bias or bounded error settings analyzed in Theorems 2 and 3?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors investigate how to construct prediction sets with$1-\alpha$coverage for the test points, using datasets with corrupted labels. While the existing PCP method relies on a distribution shift$w(z)$to achieve the target coverage, its practical application is limited when $w(z)$ is not precisely known. To bridge this gap, the authors first analyze the theoretical behavior of PCP under scenarios where the shift$w(z)$deviates from the true value. Furthermore, they introduce a new approach called UI, which imputes corrupted labels and directly constructs valid prediction sets without requiring explicit knowledge of the distribution shift. Finally, the authors present extensive experimental results, demonstrating that the UI method achieves shorter prediction sets while maintaining the desired coverage guarantee.

### Strengths
First, the authors investigate the theoretical properties of the PCP method when the distribution shift$w(z)$deviates from the true value, thereby enriching the existing theoretical results. Second, the authors propose a novel UI method. By incorporating the idea of imputation, this method constructs reliable and effective prediction sets for test points without requiring the shift$w(z)$. The paper is clearly written and provides a new solution for constructing prediction sets using datasets with corrupted labels, thereby extending the existing body of work.

### Weaknesses
The paper could be strengthened by a more thorough discussion of the proposed UI method. For instance, Theorem 4 assumes that "the residual errors are independent of the predictions of $g^{*}$ and of$C^{UI}$given the PI $Z$." It would be important to clarify the practical scenarios in which this condition can be reasonably expected to hold. Furthermore, there is no clear evidence that the UI method consistently outperforms the PCP method. When facing a practical problem, what characteristics should one consider to determine which method is more appropriate? Finally, the experimental section would benefit from including the prediction set lengths of the TriplyRobust method for a complete performance assessment, and some notations could be revised for better clarity.

### Questions
1. Theorem 4 requires that "the residual errors are independent of the prediction of $g^{*}$and of $C^{UI}$ given the PI$Z$". However, under what specific settings can this condition be reasonably expected to hold? 
2. When dealing with a practical problem, what specific characteristics should one consider to determine which method is more suitable? For instance, would the UI method outperform the PCP method when $Z$ is highly correlated with $Y$?
3. In line 318, is the residual $E_i$ computed solely for the uncorrupted samples, or for all samples in the reference set? If it's computed for all samples, the true $Y_i$ for the corrupted samples is unobservable. The notation here appears ambiguous.
4. In lines 343 and 344, regarding the condition for the peak of the conditional distribution within the interval $[a(x), b(x)]$, would the symbol "$\geq$" be more appropriately replaced by "$\leq$"?
5. In the experimental section, would the prediction interval lengths produced by the TriplyRobust method be significantly longer than those of the other methods?

### Soundness
3

### Presentation
2

### Contribution
2
