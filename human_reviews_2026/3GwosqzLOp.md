# Learning Conformal Explainers for Image Classifiers

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Feature attribution methods are widely used for explaining image-based predictions, as they provide feature-level insights that can be intuitively visualized. However, such explanations often vary in their robustness and may fail to faithfully reflect the reasoning of the underlying black-box model. To address these limitations, we propose a novel conformal prediction–based approach that enables users to directly control the fidelity of the generated explanations. The method identifies a subset of salient features that is sufficient to preserve the model’s prediction, regardless of the information carried by the excluded features, and without demanding access to ground-truth explanations for calibration. Four conformity functions are proposed to quantify the extent to which explanations conform to the model’s predictions. The approach is empirically evaluated using five explainers across six image datasets. The empirical results demonstrate that FastSHAP consistently outperforms the competing methods in terms of both fidelity and informational efficiency, the latter measured by the size of the explanation regions. Furthermore, the results reveal that conformity measures based on super-pixels are more effective than their pixel-wise counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the explainability problem of image classifiers. Building on existing feature attribution methods, the authors propose an approach that determines the minimal sufficient feature subset to maintain the original model prediction. This minimal subset provides a qualitative aspect that complements the quantitative assessments on feature influences provided by feature attribution methods. In addition to the proposed algorithm, the authors further introduce variants of the conformity function, which exhibit different behaviors in terms of confidence and fidelity during empirical evaluation.

### Strengths
- This work provides a new perspective for presenting the results of feature attributions, which is generally orthogonal to existing explainability research.
- The main results are supported by detailed quantitative evaluations presented in the appendix.
- The provided pseudocode facilitates a clearer understanding of the proposed algorithm and enhances the reproducibility of the presented results.

### Weaknesses
- It is unclear how this work should be positioned in the established explainability literature. I personally interpret it as a plugin that complements existing feature attribution methods, but its connection and contribution should be better clarified and communicated.
- The motivation and the reported results are somewhat disconnected. This work begins with a critique of the robustness and faithfulness of feature attribution methods; however, it remains unclear how the conformal explainer addresses these limitations. In fact, the proposed method will inherit, if any, the weaknesses of the underlying explainer that produces $\Phi$, given its direct reliance on feature attributions.
- The writing and overall presentation of this work represent another major weakness. Many notations collected from different sources are presented together without alignment. Some discussions and even entire sections appear contextually unfit, missing the necessary motivation or clarification for the particular designs.
- Although the experimental results are detailed, they only focus on comparisons among the variants within the conformal explainer framework. The results do not show how the proposed approach contributes beyond the existing literature, which limits its potential impact.

### Questions
- The extended discussion on marginal and conditional expectations is a bit confusing. These are neither new proposals nor essential for the following discussion. It appears that the preliminary section would be a better fit to include general information about available baseline choices.
- Both sections 3.3 and 3.4 read as rather isolated. What is the purpose, motivation, or intuition behind the approaches described there?
- As mentioned in the weakness part, the symbol system is not well aligned. Related to this point:
  - The symbols $y_i$, $h(\cdot)$, and $f(\cdot)$ all refer to model outcomes but are represented differently.
  - The symbol $f_i$ (line 199) denotes an input feature but conflicts with the use of $f(\cdot)$ (line 184). Is there any inherent connection between them?
  - The calibration dataset $\mathbb{Z}$ refers to input-prediction tuples (Section 2.1, line 90) but is redefined as a set of input-attribution tuples, where $\Phi$ represents attribution vectors.
- Why are some plots dashed and others solid?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposed a metric to measure and control the degree of explanation for  post-hoc explainable AI (xAI) methods that measures the importance of attributions via the conformal prediction score, which basically reflects the variance of the classification probability under specified perturbation.

### Strengths
Compared with traditional Shapley value approaches which is in similar framework, the proposed method provide more statistical plausible measurement that can compare the degree of explanability over different situations.  Also, the method can be applied on different levels of features from pixel to super-pixel of vectors, which is more flexible.

### Weaknesses
Although the authors claim their method can be applied to all xAI models that generates feature attributions. The formulation of xAI models in Sect. 2.2 only fits primitive explanation models (perturbation or simple gradient methods) , but cannot be adopted to more advanced approaches such as class activation mapping, counter-factual generation or latent space disentanglement, which are more popular now.  The baseline explanation model they used in the experiments all fall in to the category of perturbation or simple gradient ones, and most of them are quite out-dated.
On the other hand, the experiments in Fig. 3 and 4 showed confidence vs. SE and confidence vs. fidelity, both are positively correlated. It doesn't justifies how the proposed measurement help with trading-off different critical characters of a baseline method, so it cannot persuade that the measurement is useful.

### Questions
The exact calculation of "fidelity" is completely missing in the manuscript. How did you define and calculate it? Also, between "fidelity" and "confidence", which one exactly defines "compactness"? As both are positively correlated but none of them are directly related to compactness (i.e.,  low dimensionality).

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
Brief summary: This paper proposes an approach to calibrate a global threshold based on existing feature attribution approaches, so that keeping the highest attributed features preserves the confidence for a predicted class.

### Strengths
- This paper borrows the concept of conformal prediction and applies it in the context of feature attribution. The developed method determines a global threshold over attributions derived from existing explainers, which enables a qualitative presentation of explanation results by showing the minimal components that preserve the original prediction.

### Weaknesses
- While the authors claim the development of a method that “generates explanations”, the conformal explainer leans towards a post-hoc processing of the explanations, as it relies on existing information and does not determine attributions itself.
- The symbol system lacks internal consistency, making it sometimes difficult to associate the abstractions with their corresponding concepts and to follow the description of the method. Particularly, the use of $f_j$ to denote features is very confusing, as the same is used, and commonly used, to denote functions in lines 174~186.
- While conformal prediction is well-defined with model predictions $y_i$ as described by Eq. (1), there is no inherent guarantee in a conformal explainer that translates the threshold $\tau$ for attributions to the prediction outcome space. The authors enforce label preservation in Eq. (3) through an explicit constraint; yet, the quality of the presented subsets is still dominated by the underlying feature attribution methods.
- The time costs are not reported, which could be another concern given the repeated queries required to determine $\sigma_i$ as specified in Eq. (3).

### Questions
- Could the authors elaborate further on the motivation for identifying a global threshold for subset presentation? The tested feature attribution methods all provide local explanations, with patterns that can vary across different test cases (depending on the analyzed classes and the local decision boundaries at a particular location of the manifold ). Given the “local” nature of those explanations, wouldn’t determining a specific threshold for each explanation be more effective and also easier?

### Soundness
3

### Presentation
2

### Contribution
2
