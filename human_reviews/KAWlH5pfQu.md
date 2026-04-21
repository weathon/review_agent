# Detecting Adversarial Examples

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 5, 3, 3

## Abstract
Deep Neural Networks (DNNs) have been shown to be vulnerable to adversarial examples. While numerous successful adversarial attacks have been proposed, defenses against these attacks remain relatively understudied. Existing defense approaches either focus on negating the effects of perturbations caused by the attacks to restore the DNNs' original predictions or use a secondary model to detect adversarial examples. However, these methods often become ineffective due to the continuous advancements in attack techniques. We propose a novel universal and lightweight method to detect adversarial examples by analyzing the layer outputs of DNNs. Through theoretical justification and extensive experiments, we demonstrate that our detection method is highly effective, compatible with any DNN architecture, and applicable across different domains, such as image, video, and audio.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This paper proposes a universal method for detecting adversarial examples by analyzing the layer outputs of deep neural networks. The authors claim to have theoretical justifications and experiments to demonstrate the effectiveness of their method on any DNN architecture and its applicability across multiple domains, including image, video, and audio. Empirical experiments are done on CIFAR-100 and ImageNet.

### Strengths
I tried, but it's difficult to write down any points that deserve to be called Strengths for an ICLR-submitted paper.

### Weaknesses
The weaknesses of this paper include:

- **The proof of Theorem 1 is wrong.** The proof bases on that "Finally, the perturbation aligned with DNN weights is amplified as it sequentially moves through the DNN layers (Goodfellow et al. 2014)", which is an *empirical observation*, not a theoretical conclusion. I was shocked that the authors treat an empirical observation as a formal Theorem, and it's easy to construct a counter-example DNN that violate Theorem 1.

- **Non-adaptive evaluations.** In Line 290-292, the authors claim that "We consider the most challenging case for the detector, the white-box attack scenario, in which the attacker has full access to the classification model". This is NOT a white-box scenario for detection, because *the authors did not assume that the attacker has access to the detection model*. The authors should design adaptive attacks [1,2], where the attacker has full access to both the classification and detection models.

- **False AUC values.** As indicated in [3], a detection AUC value can be converted into a classification defense accuracy. Up to now, the state-of-the-art defenses as listed on RobustBench [4] is less than 45% acc (CIFAR-100), which cannot match the mostly >0.99 AUC values reported in this paper.

- **The baselines are weak.** The majority of defense baselines in this paper are simple input processing methods like JPEG, which is far from a fair comparison for a paper submitted in 2024.


References:\
[1] Adversarial examples are not easily detected: Bypassing ten detection methods. AISec 2017\
[2] Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. ICML 2018\
[3] Detecting Adversarial Examples Is (Nearly) As Hard As Classifying Them. ICML 2022\
[4] https://robustbench.github.io/

### Questions
Some questions:

- In the abstract, the authors claim that their method is "applicable across different domains, such as image, video, and audio." However, I can only find experiment results on CIFAR-100 and ImageNet, so where are the results on video and audio?

- The authors claim that their LR method can detect unseen attacks. Where are related experiments? For example, can LR method trained on $\ell_{\infty}$ attacks detect $\ell_{2}$ or $\ell_{0}$ attacks?

- Why there is no experiments on CIFAR-10? CIFAR-10 is the most commonly used dataset in the adversarial literature.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces a technique to detect adversarial examples. The defense works by studying the activations of the model to notice suspicious patters that are not present in the clean data. It presents evidence of its efficacy by evaluating against a suite of attacks, and compares to several prior defenses from the literature.

### Strengths
Defending against adversarial examples is an important and interesting challenge.

### Weaknesses
Unfortunately, it appears unfamiliar with the (vast) literature on this topic and the paper does not present convincing evidence that it will be robust to adaptive attacks.

I would particularly recommend the authors begin by reviewing Carlini & Wagner "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods", and Tramer "Detecting Adversarial Examples Is (Nearly) As Hard As Classifying Them". The latter paper, in particular, shows that the results claimed here would imply a nearly perfectly robust classifier. Then, I would recommend the authors read related papers on detecting adversarial examples ["The Odds are Odd", "Asymmetrical Adversarial Training"]. After this, it could be then useful to study how these defenses were broken in "On adaptive attacks to adversarial example defenses".

What this paper presents---an evaluation against a large set of fixed attacks---is not sufficient for arguing adversarial robustness, and I can not recommend acceptance at this point.

(I would also recommend changing the title of this paper. There are 50+ papers on detecting adversarial examples.)

### Questions
Do you believe this paper will be robust to adaptive attacks, as discussed above?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a detection method named Layer Regression (LR), which assumes that the impact of adversarial examples on the final layer is much larger than the initial layer. The proposed LR can achieves high detection performance, and experimental results show that it can be applied in different tasks.

### Strengths
1. The motivation of the proposed LR is clearly stated.
2. According to the experimental results, LR detects adversarial examples with high efficiency.
3. Experiments in other domains are implemented to prove the universality of LR.

### Weaknesses
1. The detection baselines are not strong enough. Some strong baselines, like [1][2][3], are not included, which makes the experimental results less convincing.
2. There is a lack of adaptive attack against LR, and the adaptive attack is important to evaluate the detection performance.
3. According to Section C in the Appendix, the subset of layer vectors needs to be selected for each model, and the dataset may sometimes influence the choice of layers, reducing the practicality of LR.

[1] Tian, Jinyu, Jiantao Zhou, Yuanman Li, and Jia Duan. "Detecting adversarial examples from sensitivity inconsistency of spatial-transform domain." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 11, pp. 9877-9885. 2021.
[2] Yang, Yijun, Ruiyuan Gao, Yu Li, Qiuxia Lai, and Qiang Xu. "What you see is not what the network infers: Detecting adversarial examples based on semantic contradiction." NDSS 2022.
[3] Zhang, Shuhai, Feng Liu, Jiahao Yang, Yifan Yang, Changsheng Li, Bo Han, and Mingkui Tan. "Detecting adversarial data by probing multiple perturbations using expected perturbation score." In International conference on machine learning, pp. 41429-41451. PMLR, 2023.

### Questions
1. What will the detection performance of LR be when facing targeted attacks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This study proposes a universal and lightweight detection method for adversarial examples to defend deep neural networks from the threat of adversarial examples. The proposed detector is trained to predict the difference in feature vectors of benign inputs among layers. When the loss of the input is higher than a pre-defined threshold, the detector treats the input as an adversarial example. The effectiveness of the proposed detector is justified through theoretical analysis and extensive experiments.

### Strengths
- The proposed method empirically shows meaningful improvement in detection AUC compared to other baseline defenses among various architectures.
- The motivation is well explained and partially justified through theoretical analysis.

### Weaknesses
- The proof of theorem 1 seems to miss the important assumption in the referred paper (Goodfellow et al., 2014). Goodfellow et al. claim that the perturbation is linearly amplified as it moves through linear models, but there are no theoretical results for nonlinear models.
- The effectiveness of the proposed detector could be further emphasized by investigating their detection performance against attacks that produce adversarial examples with minimal perturbation norms because the tested attacks are limited to maximize attack success rates under a given attack budget.

### Questions
- Does the theorem 1 hold for arbitrary deep neural networks? I would like to show you a strict mathematical proof of theorem 1.
- Is the proposed method able to detect adversarial examples with a minimal norm of adversarial perturbation?

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes Layer Regression (LR), a universal and lightweight method for detecting adversarial examples in DNNs. The key innovation lies in analyzing the changes in DNN's internal layer outputs rather than focusing on input modifications or output comparisons like previous approaches. The authors provide theoretical justification through a theorem showing that adversarial impacts are stronger in the final layers compared to the initial layers.

### Strengths
1. **Theoretical Foundation.** Mathematical proof supporting the core concept and has a clear theoretical justification for why the method works.

2. **Lightweight:** Using a relatively small MLP for regression makes LR computationally efficient, and suitable for real-time detection.

### Weaknesses
See questions.

### Questions
1. **Input Selection Heuristic:** The selection of early layer activations for the regression model's input seems somewhat arbitrary. While the paper mentions a trade-off between proximity of clean and adversarial inputs and the accuracy of the estimator, a more principled approach for input selection would strengthen the method.
2. **Lack of Robustness Analysis:** The paper primarily focuses on white-box attacks. The performance against more realistic ***black-box attacks*** is not evaluated. Furthermore, the robustness of LR itself against ***adaptive attacks***, where the adversary is aware of the defense mechanism, is not discussed. An attacker could potentially craft perturbations that minimize the change in the selected early layer activations while still maximizing the classification loss.
3. **Overfitting Potential:** Training the MLP on clean data only might lead to overfitting and poor generalization to unseen adversarial examples, especially considering the high dimensionality of the feature vectors.
4. **Hyperparameter sensitivity:** The method requires setting a threshold value, which could be sensitive to different scenarios.
5. **Inadequate Evaluation:** The baseline methodology used is outdated and not compared to the most recent SOTA **[R1], [R2]**.

**[R1]** What You See in Not What the Network Infers: Detecting Adversarial Examples Based on Semantic Contradiction.   
**[R2]** Detecting adversarial data by probing multiple perturbations using expected perturbation score.

### Soundness
2

### Presentation
2

### Contribution
1
