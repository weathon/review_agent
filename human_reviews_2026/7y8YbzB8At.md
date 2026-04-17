# Enhancing Conditional Risk Control in Image Segmentation with Adaptive Conformal Prediction

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Uncertainty quantification is crucial in high-stakes image segmentation, yet existing conformal risk control methods often exhibit highly variable conditional risk: some images suffer extreme false negative rates while others show minimal errors. We introduce Conformal Risk Adaptation (CRA), a framework that employs a novel score function, inspired by adaptive prediction sets, to create image-specific uncertainty regions. We formalize the risk control problem as a weighted quantile estimation task, which enables a computationally efficient, grid-search-free algorithm for threshold calculation. To ensure the reliability of our adaptive score function, we integrate a specialized non-parametric calibration method that enhances pixel-wise probability estimates. Experiments on polyp and crack segmentation demonstrate that CRA maintains valid marginal risk guarantees while delivering substantially more consistent conditional risk control across diverse images. This advancement provides practitioners with a principled approach to uncertainty quantification that adapts to individual cases while maintaining rigorous statistical guarantees, which is critical for personalized medical applications. We have provided code with implementation details in the repository below: https://anonymous.4open.science/r/conformal-risk-adaptation-3BB2.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of improving image-conditional coverage in conformal image segmentation, ensuring that each individual image achieves a reliable coverage rate rather than only a marginal dataset-level guarantee. Existing Conformal Risk Control (CRC) methods provide only marginal control of false negative rates (FNR), which can vary significantly across images, leading to unreliability in safety-critical applications such as medical imaging.

The authors propose two methods.

1. AT (Adaptive Thresholding) learns an image-specific segmentation threshold through supervised regression of optimal thresholds computed offline.
2. COAT (Conditional Optimization for Adaptive Thresholding) is an end-to-end differentiable framework that directly optimizes a soft approximation of the True Positive Rate (TPR) to minimize miscoverage loss, enabling gradient-based learning of image-specific thresholds.

Theoretically, the authors provide finite-sample marginal guarantees following CRC theory and show asymptotic conditional validity. Empirically, COAT consistently reduces coverage gap, the deviation of per-image coverage from the target, across multiple segmentation datasets (Polyp, Skin, Fire) and backbone models (UNet, PSPNet, DeepLabV3+, SINet), while maintaining marginal coverage close to target levels (for example 0.9 or 0.8).

### Strengths
+ **Addresses a critical reliability issue**: tackles the gap between marginal and image-conditional coverage, which is crucial for trustworthy segmentation in medical or safety-critical domains.
+ **Clear methodological design**: the AT and COAT frameworks are conceptually simple, well-motivated, and neatly extend CRC to image-specific threshold adaptation.
+ **End-to-end differentiable innovation**: the proposed differentiable miscoverage loss allows conditional coverage optimization via gradient descent, eliminating the need for non-differentiable calibration loops.
+ **Extensive empirical validation**: experiments across three datasets, multiple segmentation architectures, and two risk levels (α=0.1, 0.2) demonstrate consistent performance gains and robustness.
+ **High reproducibility**: the paper provides detailed algorithm pseudocode, hyperparameters, and sensitivity analyses (Appendix A.6), supporting replicability.

### Weaknesses
+ **Limited theoretical novelty**: while the differentiable loss is useful, the theoretical framework relies almost entirely on prior CRC theory (Angelopoulos et al., 2024); the proofs do not introduce new risk-control guarantees beyond finite-sample marginal bounds.
+ **Empirical interpretation**: although Coverage Gap is reduced, the paper does not report other reliability metrics (e.g., pixel-level confidence calibration, variance of TPR across difficulty strata), which could further substantiate image-conditional consistency.
+ **Ablation clarity**: AT is presented as a baseline, but the quantitative contribution of the differentiable loss relative to supervised regression could be more explicitly isolated.

### Questions
1. Could the authors provide per-image coverage histograms or scatter plots to visualize how COAT stabilizes conditional coverage compared to CRC?
2. The theoretical guarantee in Theorem 1 ensures marginal validity. Can the authors quantify the empirical gap between marginal and conditional coverage under finite-sample conditions  
3. Would COAT’s framework extend naturally to multi-class segmentation or detection settings with multiple thresholds per class?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the critical problem of variable conditional risk in conformal risk control (CRC) for image segmentation. The authors note that while standard CRC methods can guarantee an average (marginal) risk, such as the false negative rate, their performance on individual images can be highly erratic. This inconsistency is a major barrier to adoption in high-stakes fields like medical diagnostics.

To address this, the paper introduces Conformal Risk Adaptation (CRA), a comprehensive framework. CRA's core is a novel adaptive score function that adapts the principle of Adaptive Prediction Sets (APS) to segmentation. Experiments on polyp and crack segmentation datasets demonstrate that CRA successfully maintains the target marginal risk guarantee while achieving substantially more consistent conditional risk (measured by a lower "Coverage Gap") compared to standard CRC and a recent adaptive baseline (AA-CRC) .

### Strengths
1. The paper addresses a well-known and highly important limitation of conformal prediction—the gap between marginal and conditional guarantees. Improving conditional reliability is essential for deploying segmentation models in safety-critical applications, which the authors use as a strong motivation.

2. The CRA framework is a technically sound and intelligent combination of several powerful ideas (APS, non-parametric calibration, and group-conditional CP). The authors correctly identify that their adaptive score (unlike standard CRC) requires accurate probability estimates, and they appropriately add a calibration step to address this.

3. The weighted quantile formulation is also a strong practical contribution. It not only provides theoretical clarity but also yields a massive computational speedup (over 27x vs. CRC in their experiment ) and avoids the precision issues of grid-search-based methods.

4. The experimental setup is strong. The authors validate their method on two distinct tasks (medical and engineering ), compare against relevant baselines (CRC and AA-CRC), and use the correct primary metric (Coverage Gap) to support their core claim . The results in Table 1 and Figure 2 convincingly show that CRA achieves a statistically significant improvement in conditional risk control.

### Weaknesses
1. The paper's most significant weakness is an incomplete evaluation due to a broken ablation study. The text in Appendix A.1 describes an essential ablation study to measure the impact of the individual components (CRA w/o calib, CRA w/o strat, etc.) 16. This study is necessary to validate the paper's claim that the components "work synergistically". However, Table 2, which allegedly contains these results, is an exact duplicate of Table 1. As a result, this key claim is currently unsubstantiated by any data in the paper. This is a significant oversight.

2. The paper's notation for risk levels and thresholds is confusing. The target risk level is $\alpha$. The adaptive set is defined with a parameter $\alpha^{\prime}$. The final stratified threshold is called $\alpha_{k}^{\prime}$. It is unclear how these relate, or if they are meant to be the same variable used in different contexts. This should be clarified. 

3. The explanation for Figure 1 (Left) is opaque. The caption "Distribution of probability mass proportion required to achieve $(1-\alpha)$ coverage" is not clearly explained in the main text and its relevance to the overall argument is not as clear as the right panel (the calibration curve).

### Questions
1. My most critical question concerns the ablation study in Appendix A.1, which is missing its data; Table 2 is a copy of Table 1. Can you please provide the correct results table for the ablation study described in the text 17? This is essential for me to evaluate the individual contributions of the adaptive score, the probability calibration, and the stratification.

2. Can you please clarify the notation used for $\alpha$, $\alpha^{\prime}$, and $\alpha_{k}^{\prime}$? Is the parameter $\alpha^{\prime}$ in Equation 4 the same as the calibrated threshold $\alpha_{k}^{\prime}$ found in Equation 7? How do these relate to the overall target risk level $\alpha$?

3. The methodology has two parts that rely on the sum of probabilities $\sum \hat{p}_{j'}(X_{i})$: the CRA score calculation (Section 3.2) and the stratification (Section 3.4 18). Is the same set of calibrated probabilities (from Section 3.3 19) used for both of these steps?

4. Table 1 shows that while CRA wins on Coverage Gap, it sometimes loses on Precision (e.g., Polyp $\alpha=0.1$, 0.548 vs 0.714 for AA-CRC) or Normalized Size (e.g., Crack $\alpha=0.2$, 0.178 vs 0.132 for AA-CRC). Could you comment on this trade-off? How should a practitioner weigh the significant gain in conditional fairness against a potential loss in precision or efficiency?

5. Regarding the impressive computational speedup (Table 4), does the 0.48s runtime for CRA include the time for all components: the isotonic regression (Section 3.3), the stratification (Section 3.4), and solving the $K=5$ weighted quantile estimations (Section 3.1)?

### Soundness
2

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
4

### Summary
The submission studies conditional risk control in imaging segmentation problems. The submission couples adaptive prediction sets with class-conditional conformal risk control to achieve better conditional control of the false positive rate.

Experiments compare the proposed method (CRA) with existing approaches based on conformal risk control, both on a medical imaging dataset and on an engineering dataset.

### Strengths
- Conditional risk control in imaging segmentation problems is an important topic
- Proposed method is intuitive and computationally inexpensive
- Experiments support the claim that the proposed method improves conditional coverage

### Weaknesses
- Apparent typos and mistakes in the theorems?
- It is unclear whether improved conditional coverage comes from the proposed method or the use of class-conditional calibration.

I will expand on these points below and I am looking forward to discussing with the authors!

### Questions
**Theorem 2**

My main concern about the current submission is the validity of Theorem 2. In particular:

- The optimization problem in Eq. (4) main admit multiple global minimizers, but the set $S$ in the proof of the theorem is unique?

- If I understand the construction and statement correctly, I think we can show a counterexample where the statement does not hold. Consider:

$j = 1,2,3,4,5$

$\hat{p}_j = 0.3, 0.25, 0.20, 0.15, 0.10$, such that $\sum_j \hat{p}_j = 1$

Then,

$s_j = 1.0, 0.7, 0.45, 0.25, 0.10$

and set $1 - \alpha' = 0.40$.

We have $\hat{C} (\alpha') = \\{\\{1,2\\}, \\{2,3\\}, \\{2, 4\\}\\}$ but $S(\alpha') = \\{1,2,3\\}$.

Could the authors clarify where this misunderstanding about the statement of the theorem is coming from?

**Theorem 1**

For the CRC procedure to be valid, monotonicity is required. I assume this is what the authors mean by "ranked by likelihood of inclusion". Could the authors make this statement more clear and precise?

**Probability calibration**

I follow the argument that the proposed calibration procedure depends on the scale of the predicted probabilities. However, I am not sure I follow how post-processing the predictor with cross-entropy achieves calibration. Could the authors expand on this? This is at odds with knowledge that classifiers trained with cross-entropy loss tend to be overconfident and miscalibrated. Possible alternatives from the broader conformal prediction literature might include Venn-Abers predictors.

Figure 1, right panel does not necessarily imply that the probability distribution of the post-processed predictor is more calibrated. Other measures such as distance to calibration or expected calibration error might be employed to support this claim?

**Experiments**

Results show that the proposed method reduces the "coverage gap", which implies better conditional coverage. Do I understand correctly that the stratification procedure was employed for CRA only? This may be unfair to the baseline methods, which were not designed for conditional coverage. Comparing with the stratified version of the baselines would provide stronger evidence in support of the adaptive construction of the prediction sets.

---

**Minor comments**

- Typo on Lines 158 - 159 of the proof of Theorem 1? The last equation reads $1 - \tau'$, but Eq. (3) reads $\tau'$.
- Equations are references in the text with repetitions, e.g. "Equation equation 3".
- Why is there a hyperlink on the bottom of page 6?
- Figure 2: the claim that CRA produces more concentrated results is difficult to verify without quantitative measures.

### Soundness
2

### Presentation
3

### Contribution
2
