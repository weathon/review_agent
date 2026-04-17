# Fairness via Independence: A General Regularization Framework for Machine Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Fairness in machine learning has emerged as a central concern, as predictive models frequently inherit or even amplify biases present in training data. Such biases often manifest as unintended correlations between model outcomes and sensitive attributes, leading to systematic disparities across demographic groups. Existing approaches to fair learning largely fall into two directions: incorporating fairness constraints tailored to specific definitions, which limits their generalizability, or reducing the statistical dependence between predictions and sensitive attributes, which is more flexible but highly sensitive to the choice of distance measure. The latter strategy in particular raises the challenge of finding a principled and reliable measure of dependence that can perform consistently across tasks. In this work, we present a general and model-agnostic approach to address this challenge. The method is based on encouraging independence between predictions and sensitive features through an optimization framework that leverages the Cauchy–Schwarz (CS) Divergence as a principled measure of dependence. Prior studies suggest that CS Divergence provides a tighter theoretical bound compared to alternative distance measures used in earlier fairness methods, offering a stronger foundation for fairness-oriented optimization. Our framework, therefore, unifies prior efforts under a simple yet effective principle and highlights the value of carefully chosen statistical measures in fair learning. Through extensive empirical evaluation on four tabular datasets and one image dataset, we show that our approach consistently improves multiple fairness metrics while maintaining competitive accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel, general, and model-agnostic regularization framework based on the Cauchy-Schwarz (CS) Divergence. The core idea is to encourage independence between model predictions and sensitive attributes by using CS divergence as the fairness loss term $\mathcal{L}_{fairness}$. The authors motivate this choice by arguing that CS divergence provides a tighter theoretical bound compared to other measures like Kullback-Leibler (KL) divergence and gap parity, which they posit leads to improved generalizability and robustness.

### Strengths
1. The paper is well structured and easy to follow.

2. The proposed methods are well presented and evaluated with extensive experiments.

3. The choice of the Cauchy-Schwarz divergence is well-motivated. It is justified by theoretical properties, and the paper highlights its advantages, which provide a strong foundation for the method.

### Weaknesses
1. The paper claims the CS regularizer is more "robust", which is partially supported by the smoother trade-off curves (Obs. 5 ). However, robustness to model parameters is also claimed. Figure 2 is presented as evidence for this, but its caption is vague, and it is not well-integrated into the main text. The parameter sensitivity analysis in Figure 5 is good for the CS method itself, but it doesn't include a comparative analysis against the baselines.

2. The authors explicitly state that they exclude adversarial debiasing methods to avoid complexity. While this is an understandable simplification, adversarial methods (e.g., Zhang et al., 2018 ) are a very common, powerful, and closely related family of in-processing techniques for achieving independence. Including at least one such baseline would have provided a more complete picture of the method's performance relative to the state-of-the-art.

3. The paper introduces CS divergence as a novel regularizer for fair machine learning. While this application appears novel, the paper itself notes that CS divergence is used in other ML domains like deep clustering and representation learning. The "Related Work" section (in the appendix ) is quite general.

### Questions
see above

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper modifies the typical "performance+ fairness regularizer" framework by using the Cauchy-Schwarz divergence as the fairness regularizer.

### Strengths
- The paper offers a new viewpoint on measuring fairness and thereby is applicable to existing fairness optimization frameworks.

- Using Cauchy-Schwarz divergence as a fairness regularizer offers a mathematically simple and interpretable alternative to traditional measures like KL divergence or MMD and the empirical estimation of CS divergence is often more stable than KL divergence under small or skewed samples, reducing the risk of gradient explosion or extreme behaviors during fairness optimization.

### Weaknesses
- The overall contribution is rather incremental, the main framework simply replaces existing discrepancy measures (KL, MMD, HSIC, f-divergences) with the Cauchy–Schwarz divergence. This substitution alone does not introduce a new conceptual insight, making the work less exciting. Moreover, CS divergence has already been widely adopted in areas such as representation learning, domain adaptation, and clustering and this paper mainly transfers an existing metric into a new application area (fairness), rather than introducing a fundamentally new fairness paradigm.

- The empirical estimation of the CS divergence relies on kernels (Eq. 11). However, the theoretical analysis (e.g., Proposition 4.2) only holds under the Gaussian distribution assumption. Could the authors clarify whether this assumption is necessary for the empirical version as well? If not, why did the paper not explore or compare other kernel functions (e.g., Laplacian, polynomial), or analyze the impact of kernel choice on fairness performance in the experiment section? Since the kernel implicitly defines a density estimator, the fairness performance could be highly sensitive to kernel type and bandwidth. A kernel ablation or sensitivity analysis would strengthen the empirical validity of the claim.

- It is unclear whether the choice of CS divergence is practically applicable or scalable, the complexity of using CS is $O(n^2)$, however, the paper does not include any runtime or complexity comparison, nor any discussion of stability during optimization. Without such analysis, the claimed robustness of CS divergence remains questionable in practice. 

- Some experimental claims appear not well supported. For instance, in obs 1 (page 7), the authors state that "...CS consistently achieves the best $\Delta_{EO}$ and ranks among the top four for $\Delta_{DP}$...” However, the baselines do not include any method that explicitly optimizes EO as a fairness objective, making this comparison incomplete. Moreover, the evaluation omits several recent works that use f-divergence or information-theoretic measures for fairness regularization, which weakens the empirical significance of the conclusions.

- Table 2 is overly compact and hard to read. The authors could consider presenting results in a clearer format such as $85.63_{\pm 0.34}$ instead of the current layout. (minor)

- Appendix C.3, all the figures are left without any explanations. The authors could provide some brief explanations of what each figure illustrates and conclude how they relate to the main fairness claims. (minor)

### Questions
- Is the proposed method applicable to scenarios involving multiple sensitive attributes?
- What is the impact of using different kernel functions on the method’s performance?
- In Section 5.4, the parameter sensitivity analysis is conducted only on $\alpha$ and $\beta$, not including the $\sigma$ used in the Gaussian kernel. How sensitive is the performance to the kernel parameters?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In this paper, the authors explore fairness via independence. The method is based on encouraging independence between predictions and sensitive features through an optimization framework that leverages the Cauchy–Schwarz (CS) Divergence as a principled measure of dependence.

### Strengths
- The authors introduce Cauchy–Schwarz (CS) Divergence as a principled measure of dependence.
- The authors give three tubular datasets to validate.

### Weaknesses
- This paper gives a comparison between cs divergence and kl divergence under Gaussian distribution. In fact, Gaussian distribution is very special, I suggest the authors discuss more advantages and provide comparisons with other regularizers, hgr, MI, dc or empirical dc for instance. 
-

### Questions
- This paper gives a comparison between cs divergence and kl divergence under Gaussian distribution. In fact, Gaussian distribution is very special, I suggest the authors discuss more advantages and provide comparisons with other regularizers, hgr, MI, dc or empirical dc for instance. 

- Line 084, you should add citations for gap parity and MMD.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
It argues that CS divergence offers a tighter bound compared to existing measures like KL divergence and MMD, leading to more robust and generalizable fairness. The authors present empirical results on several datasets demonstrating improved fairness metrics while maintaining competitive accuracy. The paper is well-written, technically sound, and addresses an important problem in the field of fair machine learning.

### Strengths
1. The application of CS divergence to fairness regularization is a novel contribution. The paper provides a good theoretical justification for why CS divergence might be superior to other measures.
2. The paper is generally well-written and easy to follow, clearly explaining the proposed method and its advantages. The inclusion of algorithm implementation details enhances reproducibility.
3. The empirical evaluation is comprehensive, covering multiple datasets, fairness metrics, and comparisons to existing methods. The analysis of the trade-off between accuracy and fairness is particularly insightful. The paper takes considerable care in describing the dataset parameters and the rationales that underpinned their selection.

### Weaknesses
1. While the paper highlights the advantages of CS divergence, a more thorough discussion of its limitations would be beneficial. For example, are there specific types of datasets or scenarios where CS divergence might not perform well? Are there computational costs associated with using CS divergence compared to simpler measures?
2. While the paper compares against several standard fairness methods, including more recent and sophisticated techniques from the fairness literature, it could further strengthen the empirical evaluation.
3. There are nine figures in the paper. These help provide greater clarity on the data and results. It would also be helpful if the visualisations were briefly discussed in the text of the paper.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
