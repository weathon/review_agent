# Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction

- Decision: Reject
- Scores: 4, 6, 8, 2

## Abstract
Deep learning–based weather forecasting (DLWF) models leverage past weather observations to generate future forecasts, supporting a wide range of downstream tasks, including tropical cyclone (TC) trajectory prediction. In this paper, we investigate their vulnerability to adversarial attacks, where subtle perturbations to the upstream weather forecasts can alter the downstream TC trajectory predictions. Although research on adversarial attacks in DLWF models has grown recently, generating perturbed upstream forecasts that reliably steer downstream output toward attacker-specified trajectories remains a challenge. First, conventional TC detection systems are opaque, non-differentiable black boxes, making standard gradient-based attacks infeasible. Second, the extreme rarity of TC events leads to severe class imbalance problem, making it difficult to develop efficient attack methods that will produce the attacker's target trajectories. Furthermore, maintaining physical consistency in adversarially generated forecasts presents another significant challenge. To overcome these limitations, we propose Cyc-Attack, a novel method that perturbs the upstream forecasts of DLWF models to generate adversarial trajectories. First, we pre-train a differentiable surrogate model to approximate the TC detector's output, enabling the construction of gradient-based attacks. Cyc-Attack also employs skewness-aware loss function with kernel dilation strategy to address the imbalance problem. Finally, a distance-based gradient weighting scheme and regularization are used to constrain the perturbations and eliminate spurious trajectories to ensure the adversarial forecasts are realistic and not easily detectable. Our experimental results show that Cyc-Attack achieves higher targeted TC trajectory detection rates, lower false alarm rates, and stealthier perturbations than conventional gradient-based attack methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Cyc-Attack, an adversarial attack method that manipulates deep learning-based weather forecasts to alter downstream tropical cyclone trajectory predictions. The paper focuses on addressing the black-box nature of cyclone detectors and the extreme class imbalance issue by employing a differentiable surrogate model, a skewness-aware loss with kernel dilation, and distance-based gradient weighting to generate stealthy and effective perturbations.

### Strengths
1. It is the first work to demonstrate how to attack a downstream weather application (cyclone tracking) by perturbing upstream forecasts.

2.  The method outperforms several baselines in terms of both attack success rate (higher trajectory detection rate) and stealth (lower false alarm rate and smaller perturbations).

### Weaknesses
1.  The attack's effectiveness may highly rely on the accuracy of the pre-trained surrogate model in approximating the black-box detector; inaccuracies here could degrade performance.

2. The process involves pre-training a surrogate model and running an iterative, gradient-based attack, which can be computationally expensive

3. The technical contribution of the paper is a little limited as adversarial attack on time series problems has been well studied. 

4. The real-world impact of the proposed method is not very clear since how an attacker can add the perturbation in practice is not very clear.

### Questions
Please refer to the weakness part.

### Soundness
2

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
3

### Summary
This paper proposes Cyc-Attack, a novel adversarial attack method that perturbs the outputs of upstream weather forecasting models to manipulate downstream tropical cyclone trajectory predictions. The method effectively addresses challenges such as data sparsity and extreme class imbalance through the use of a differentiable surrogate model, skewness-aware loss, kernel dilation, and distance-based gradient weighting, enabling successful attacks in black-box settings while generating realistic and stealthy adversarial trajectories.

### Strengths
1、The focus on tropical cyclone trajectory prediction provides a highly relevant and impactful setting for studying the vulnerabilities of deep learning weather forecasting models in downstream applications, offering valuable insights for improving real-world robustness.

2、The paper systematically addresses several critical challenges, including the non-differentiability of black-box TC detectors, severe class imbalance, premature attack termination due to surrogate model errors, and the generation of unrealistic zigzag trajectories, demonstrating technical depth and completeness.

### Weaknesses
1、Several critical choices, especially in experimental design and parameter selection, lack in-depth justification, which affects the clarity and persuasiveness of the paper. Specific issues are listed below under "Questions."

### Questions
1、In Section 5.2, when R=0, FPR is 0.0067 and TPR is 0.9896; when R=2, FPR drops to 0.0002, but TPR decreases significantly to 0.8131. Why is the reduction in FPR from 0.0067 to 0.0002 considered more important than the drop in TPR from 0.9896 to 0.8131? Has the risk trade-off between false alarms and missed detections been evaluated in the context of meteorological early warning systems? Is this choice supported by domain expertise or operational requirements?
2、In the first row of Figure 4, the adversarial trajectories generated by baseline methods (e.g., ALA, TAAOWPF, AOWF) are not shown. Is this because these methods failed to produce complete and coherent trajectories, making visualization impossible? If so, does this indicate that these methods are ineffective at the trajectory level? The authors should clarify this in the text and, if possible, include their outputs in the appendix for a more comprehensive comparison.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the vulnerability of deep learning–based weather forecasting systems to adversarial attacks, focusing on the downstream task of tropical cyclone trajectory prediction. Overall it is a timely and well-executed study on adversarial vulnerabilities in downstream weather forecasting pipelines. The method is well-motivated and empirically validated. While the evaluation scope and realism analysis could be expanded, the work is technically sound, clearly written, and potentially impactful for both the adversarial ML and climate science communities.

### Strengths
1. Overall the paper is well-written and easy to follow. The algorithm design is also technically sound.
2. The paper extends adversarial robustness analysis from general DLWF models to downstream applications like TC trajectory prediction, which is societally relevant and technically distinct from prior pixel-level attacks.
3. The study uses real-world datasets (ERA5, IBTrACS) and provides thorough quantitative comparisons, ablations, and visualization (e.g., Hurricane Delta, Typhoon Haiyan). Metrics at both location and trajectory levels are carefully defined.

### Weaknesses
1. While the authors constrain perturbations via distance weighting, physical realism is not formally validated. The “stealthiness” metric is purely statistical (ℓ1 distance).
2. It seems that the defenses discussed in the paper are more of detections rather than technques that can make the forecasting models more robust to attacks. Would appreciate more discussions on the defense side.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This submission focuses on adversarial attacks against weather forecasting models. It identifies three key research challenges: the black-box constraints of tropical cyclone (TC) detection systems, the sparsity of TC events, and the need to maintain physical consistency in the generated perturbations. To address these challenges, the paper proposes Cyc-Attack, a surrogate model–based black-box adversarial attack framework designed to effectively evaluate and exploit the vulnerabilities of TC forecasting systems.

### Strengths
**i.** The research topic is interesting and important. While adversarial attacks have been extensively studied in static domains such as image and text classification, their impact on dynamic applications like time series forecasting remains underexplored. 

**ii.** The paper is easy-to-follow. 

**iii.** The setting that integrates time series forecasting with a downstream tropical cyclone (TC) detection system is interesting and well-motivated, as it reflects a more realistic and application-driven scenario.

### Weaknesses
**i.** The research gaps are not accurately identified. Specifically, the third and fourth “challenges” mentioned in the paper should not be considered unsolved research problems. For example, ensuring that adversarial perturbations are imperceptible and realistic is a fundamental property of adversarial examples, not a unique challenge. Similarly, developing a precise surrogate model is an inherent prerequisite for any surrogate model–based black-box attack, rather than a novel research gap. The paper should better clarify which aspects of these challenges are truly new or unexplored in the context of adversarial studies on weather forecasting models.

**ii.** The motivation for adopting a surrogate model–based transfer attack is not sufficiently supported. Prior studies, such as [1] (multi-query) and [2] (one-query), have already proposed zero-order optimization–based black-box attacks for time series forecasting models. Although the manuscript claims that such methods require queries to the black-box system, it does not clearly explain why constructing a surrogate model is preferable to directly querying the true system. This justification is particularly important because the sparsity of tropical cyclone events can significantly degrade the similarity between the surrogate and the target systems, potentially limiting the transferability and practical effectiveness of the proposed approach.

**iii.** The attack pipeline is conceptually unclear. More importantly, the pipeline shown in Figure 2 does not align with the formulation described in Section 4. In this setup, the weather forecast serves as an intermediate output between the deep-learning-based forecasting model and the downstream tropical cyclone (TC) detection system. However, the paper defines the attack as manipulating the prediction \(Y\) rather than the input \(X\), which is inconsistent. If the attack manipulates the forecast output \(Y\) as described in Section 4, then the forecasting model becomes unnecessary, since one could directly perturb \(Y\) before passing it to the TC detector. Conversely, if the attack perturbs the input \(X\) as shown in Figure 2, then the optimization should compute and update \(X'\) instead of \(Y'\) in Section 4. This inconsistency makes the attack pipeline difficult to interpret and weakens the overall clarity of the experimental design.

**iv.** The technical contribution is relatively marginal. The submission trains a surrogate model to perform a transfer-based black-box attack, with the primary challenge attributed to data imbalance. However, neither the use of surrogate model–based attacks nor the approach to address imbalance represents a novel contribution. Both have been well-studied in prior literature, and the manuscript does not provide sufficient methodological innovation or theoretical advancement beyond existing work.

**v.** The experimental evaluation involves only one tropical cyclone (TC) detection system, which limits the assessment of the proposed method’s generalization capability. Without testing on multiple detection systems or model architectures, it remains unclear whether the surrogate model trained in this study can generalize effectively to other TC detection frameworks or forecasting setups.












**References**

[1] Zhu, Lyuyi, et al. "Adversarial diffusion attacks on graph-based traffic prediction models." IEEE Internet of Things Journal 11.1 (2023): 1481-1495.

[2] Liu, Fuqiang, et al. "Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting." International Conference on Artificial Intelligence and Statistics. PMLR, 2025.

### Questions
The most interesting aspect of this submission lies in its attempt to integrate time series forecasting models with downstream decision-making systems within an adversarial framework. However, the current implementation appears to manipulate intermediate prediction values rather than the raw inputs to the forecasting models, which weakens the conceptual alignment between the attack formulation and the intended end-to-end adversarial setting.

**Which to attack (three options)**  

I can identify how to compute Y' as in Equation 5, but I cannot find how to compute X', even though it is mentioned in Figure 2. You must choose and state one threat model clearly in the paper:

- **(1) Raw-input attack (recommended):** attacker perturbs the raw observations \(X\) that are fed into the forecasting model. This is the most realistic end-to-end setting (sensor spoofing, data-assimilation tampering).  
- **(2) Output/forecast attack:** attacker perturbs the published forecast \(Y\) directly (e.g., intercept/modify forecast products). This is a valid but different threat model and must be justified operationally.  
- **(3) Joint attack:** attacker can perturb both \(X\) and \(Y\). If used, consistency between \(X'\) and \(Y'\) must be enforced.

### Soundness
1

### Presentation
3

### Contribution
1
