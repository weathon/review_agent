# Uncovering Locally-Persistent Bias via Ceteris Paribus Fairness

- Avg Score: 3.00
- Decision: Reject
- Scores: 0, 4, 4, 4

## Abstract
Fairness is a key concern in Machine Learning, requiring careful consideration of how models treat individuals from different demographic groups. In this paper, we propose Ceteris Paribus Persistent-Bias-Aware (PBA) fairness and an approach to formally quantify it. PBA fairness captures the relative neural network’s confidence between an input and its counterfactual (where the sensitive attribute of the original input is flipped), while numerical features are jointly perturbed within a local neighborhood, but kept identical across both instances. As such, PBA fairness allows us to isolate the effect of the sensitive attribute, enabling formal identification of disparities that are consistently present in the model behavior within an entire neighborhood. We evaluate our proposed approach under both fairness-agnostic and fairness-aware training methods and compare it to several well-established fairness metrics on three benchmark datasets: Adult, COMPAS, and German Credit. The results demonstrate that our proposed approach identifies formally-proved disparities present in the model behavior, but overlooked by other approaches, and offers additional insights into model behavior.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The authors of this paper introduce 'Ceteris Paribus Persistent-Bias-Aware (PBA) Fairness', an approach for evaluating individual-level fairness in neural networks. The proposed approach compares a model’s confidence between an input and its counterfactual, while jointly perturbing numerical features within a small local neighbourhood held constant across both instances. By examining differences in logit-level confidence across these matched pairs, the authors aim to quantify persistent local bias that may not affect classification labels but could reveal latent disparities in model behaviour.

### Strengths
The strengths of this paper are:
- The paper is clear in terms of notation and formalisation. It is easy to follow along and read for a general audience and experts.
- It does a good job of analysing the result and interpreting what they actually mean for a reader, basically, in terms of clarity it is good.

### Weaknesses
The weaknesses of this paper are:
- The proposed method is highly derivative and therefore the novelty is low. The method: flips the sensitive attribute only to measure the effect of the model's output (Counterfactual Fairness- Kusner et al. 2017), the method also applies small pertubations to the data (Fairness Through Robustness- Nanda et al. 2021).This method as a whole seems like a trivial combination of multiple significant papers in the literature.
- The paper also clearly fails to preserve the strong logic of Nanda et al. 2021. If I am correct, by conditioning \hat{y} = \hat{y}' = c_i, this excludes instances where flipping the sensitive attribute changes the model’s decision: the very cases that define individual bias in both counterfactual and robustness-based formulations. As a result, the metric reduces to a confidence-difference statistic over already-consistent predictions, losing the core insight of Nanda et al.’s framework, which measures groupwise disparities in decision vulnerability. The method, thus replicates existing fairness ideas incorrectly, yielding a less meaningful method.
- Equation 9 (which is unlabelled) is a logit difference interpreted as confidence. I am worried the resulting comparisons in the evaluation. If I recall correctly, CertiFair and Fair-N use additional constraints in their training objective. Thus, the logit mangitudes will change between comparable models, making comparison unfair. To further clarify, without temperature scaling or normalisation then I can't be sure the reported metrics are correct.

### Questions
- Could you clarify what specific methodological or conceptual contribution is new beyond this combination? In what way does CPPBA offer capabilities that are not already achievable by existing counterfactual or robustness-based fairness frameworks?
- Can you explain the rationale for the \hat{y} = \hat{y}' = c_i restriction, given that such decision-flip cases are typically the most consequential indicators of individual unfairness?
- Equation (9) (the logit difference used as a “confidence” measure) appears to depend directly on raw logit magnitudes. Since models such as CertiFair and Fair-N are trained with additional constraints or penalties that alter the logit scale, how do you ensure that CPPBA scores remain comparable across models?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new fairness metric called Ceteris Paribus Persistent-Bias-Aware (PBA) Fairness. The core idea is to detect subtle biases in a model's confidence, even when its final predictions for different groups seem fair. The authors test PBA on standard fairness datasets (Adult, COMPAS, German Credit) and show that it successfully identifies these confidence-based disparities, which are often overlooked by metrics like Statistical Parity and Equalized Odds.

### Strengths
1. Individual fairness is an important topic in fairness studies.

2.  The overall presentation is smooth and easy to follow.

3. Extensive experiments are presented.

### Weaknesses
1. While the paper presents an interesting approach, the novelty of the contribution appears incremental. The proposed PBA fairness metric can be viewed as a synthesis of existing concepts, primarily extending Counterfactual Fairness (CFA) by incorporating a confidence-based analysis within a perturbed neighborhood. As the paper's own related work section suggests, using confidence scores to uncover subtle biases is not a new idea, which makes the overall contribution feel more like a refinement than a foundational shift.

2. The authors rightly motivate their work by highlighting the scalability issues of prior methods in high-dimensional settings. However, the evaluation is confined to small, fully-connected networks on low-dimensional tabular data. The paper does not provide evidence that the proposed method, which relies on linear relaxation, would itself scale to the complex architectures (e.g., CNNs) and high-dimensional inputs (e.g., images).

3. The practical application of the method is limited by the lack of clear guidance on selecting the perturbation bound, $\delta$. This hyperparameter is critical as it defines the "local neighborhood" for verification. While the paper uses fixed values, it does not offer a principled process for how a practitioner should select an appropriate $\delta$ for a new dataset or model, which could impact the reliability and comparability of the results.

4. A significant methodological concern is the explicit exclusion of counterfactuals that result in a class change ($\hat{y} \ne \hat{y}'$). By design, the metric overlooks a critical form of unfairness related to model robustness. For instance, a prediction for one demographic group might be fragile and easily flipped by minor perturbations, while remaining stable for another group. This disparity in robustness is a crucial fairness issue that the current formulation cannot capture.

### Questions
1. The final metric is reported as the proportion of inputs exhibiting bias, which treats all instances of unfairness equally, regardless of magnitude. This raises the question: Is a model with a widespread, low-magnitude bias (e.g., 90% of inputs show a tiny confidence gap) less fair than a model with a concentrated, high-magnitude bias (e.g., 10% of inputs show a massive gap)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes PBA fairness, a metric that detects confidence-based bias: when a model gives the same prediction to an input and its counterfactual (flipped sensitive attribute), but with different confidence. It computes the worst-case confidence gap in a locally perturbed neighborhood, isolating the effect of the sensitive attribute.

### Strengths
The paper presents an original fairness perspective by focusing on confidence disparities rather than prediction labels as traditionally done. The method is formally defined and clearly explained, and the comparison with existing fairness metrics across multiple datasets is well-executed.

### Weaknesses
- The method flips the sensitive attribute s while keeping the other features z fixed and applying the same local perturbations to the numerical parts of z, which isolates the marginal effect of s. However, this can produce causally implausible counterfactuals when s influences aspects of z (e.g., age affecting credit-history length). As a result, the measured confidence gaps may not reflect meaningful unfairness, but rather effects arising from counterfactuals that do not correspond to any plausible individual. Without modeling these causal links, it is unclear whether high PBA scores indicate unfair outcomes or simply unrealistic interventions.
- PBA is defined only on non-flip cases, so it evaluates a restricted subset of the data and excludes instances where flipping the sensitive attribute changes the decision. When that subset is small, a high PBA mainly reflects behavior on a narrow slice, making standalone interpretation uncertain.
- The paper does not fully justify why confidence bias matters, especially if the final decisions are equal. It would benefit from concrete examples or applications where differences in model confidence lead to tangible downstream impacts.
- The paper only detects confidence bias, but doesn’t propose a way to reduce it.

### Questions
1) How do you ensure that flipped counterfactuals correspond to plausible individuals, rather than off-manifold inputs that could inflate the confidence gaps?
2) How do you account for the metric’s coverage (since PBA is computed only on non-flip cases)? What proportion of the dataset is excluded due to decision flips, and how should PBA scores be interpreted when coverage is low?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Ceteris Paribus Persistent-Bias-Aware (PBA) fairness, a new formalism for measuring fairness in neural networks. Unlike existing group-level metrics or individual fairness metrics that rely on predefined similarity measures, PBA fairness quantifies local, confidence-based bias by comparing a model’s confidence between an input and its counterfactual, while jointly perturbing all numerical features within a bounded neighborhood. The authors show that this method identifies persistent confidence disparities overlooked by classical fairness metrics, using formal verification and linear relaxations for ReLU networks. Experiments on Adult, COMPAS, and German Credit datasets demonstrate that PBA fairness reveals hidden local biases, even in fairness-aware models.

### Strengths
1. PBA fairness identifies confidence-level disparities that standard metrics miss. The introduction of locally persistent bias offers an interpretable lens to analyze fairness beyond label-level metrics, contributing to nuanced fairness assessment.
2. The study includes three major datasets and compares fairness-aware vs. agnostic training under multiple metrics.
3. The notations and the mathematical formulation are clear and consistent.
4. The writing of the paper is easy to follow. The presentation/visualization of the introduced metric and the experimental results is very clear.

### Weaknesses
1. My biggest concern lies in the experimental design. The current experiments are confined to small feed-forward networks. There’s no indication of feasibility for modern deep architectures or larger inputs.
2. The current experiments are limited to small tabular datasets. Larger datasets and data with other modalities should be tested. In addition, the used Adult dataset is outdated [1]. 
3. The compared fairness-aware baselines are limited. There has been a surge of fair ML research recently. Is there any reason to select CertiFair and Fair-N, and only the two?
4. In the presentation of the results, an additional column showing the difference between the two groups would make it easier for readers to compare. Also, the results would be more convincing with the standard deviation.
5. There is limited discussion on the real-world impact or interpretive significance of the confidence disparities uncovered.


[1] Ding, Frances, et al. "Retiring adult: New datasets for fair machine learning." Advances in neural information processing systems 34 (2021): 6478-6490.

### Questions
1. See my questions in Weaknesses.
2. How sensitive is PBA fairness to the choice of perturbation bounds? Is there any suggestions on how to select proper perturbation bounds in your method? 
3. How do PBA fairness disparities correlate with real-world outcomes? Is there any example where high-confidence unfair regions are also where models make most errors?

### Soundness
2

### Presentation
3

### Contribution
2
