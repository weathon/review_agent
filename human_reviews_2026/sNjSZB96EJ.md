# Calibrating Uncertainty for Zero-Shot Adversarial CLIP

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 2, 6

## Abstract
CLIP delivers strong zero-shot classification but remains highly vulnerable to adversarial attacks. Previous work of adversarial fine-tuning largely focuses on matching the predicted logits between clean and adversarial examples, which overlooks uncertainty calibration and may degrade the zero-shot generalization. A common expectation in reliable uncertainty estimation is that predictive uncertainty should increase as inputs become more difficult or shift away from the training distribution. However, we frequently observe the opposite in the adversarial setting: perturbations not only degrade accuracy but also suppress uncertainty, leading to severe miscalibration and unreliable over-confidence. This overlooked phenomenon highlights a critical reliability gap beyond robustness. To bridge this gap, we propose a novel adversarial fine-tuning objective for CLIP considering both prediction accuracy and uncertainty alignments. By reparameterizing the output of CLIP as the concentration parameter of a Dirichlet distribution, we propose a unified representation that captures relative semantic structure and the magnitude of predictive confidence. Our objective aligns these distributions holistically under perturbations, moving beyond single-logit anchoring and restoring calibrated uncertainty. Experiments on multiple zero-shot classification benchmarks demonstrate that our approach effectively restores calibrated uncertainty and achieves competitive adversarial robustness while maintaining clean accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addresses the issue that adversarial perturbations not only degrade accuracy but also suppress uncertainty, leading to severe miscalibration and unreliable overconfidence. This overlooked phenomenon highlights a critical reliability gap beyond robustness. The authors reformulate CLIP’s logits as concentration parameters of a Dirichlet distribution and propose a novel uncertainty-calibrated adversarial fine-tuning method that regularizes entire Dirichlet distributions to jointly preserve inter-class relations and calibrate evidence strength. The results demonstrate that the proposed method effectively calibrates uncertainty under attack while maintaining strong clean accuracy and competitive adversarial robustness.

### Strengths
1. Well-motivated research question: The paper addresses an important and overlooked issue regarding the impact of adversarial perturbations on uncertainty calibration. The study is generally well-organized and clearly motivated.

2. Technical soundness: Reformulating CLIP’s logits as concentration parameters of a Dirichlet distribution, coupled with the proposed uncertainty-calibrated adversarial fine-tuning method, is technically sound and well-executed.

3. Comprehensive evaluation: The evaluation is thorough, covering different types of attacks, varying attack strengths, and multiple datasets. Additionally, aablation studies and analyses of key hyperparameters are provided.

### Weaknesses
1. A key concern is that the method is evaluated on only a single variant of CLIP—CLIP-B/32. It would be valuable to see results on additional CLIP variants or applications to other VLMs better demonstrate generalizability.
2. Could the authors provide some explanations for the comparison between FARE2 and the proposed method in the results of Figure 3(b)? Specifically, why does FARE2 achieve better uncertainty calibration but worse robustness accuracy, while the proposed method strikes a better balance?

### Questions
See weaknesses above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to enhance adversarial robustness and calibration of CLIP in the zero-shot evaluation setting.

Based on the insight that adversarial examples not only hurt model performance but also suppress predictive uncertainty, 
This paper proposes to calibrate uncertainty during adversarial training.

Specifically, the Dirichlet parameterization technique is borrowed from the evidential deep learning community. Applying KL divergence loss on the Dirichlet adversarial distribution and the Dirichlet clean distribution can implicitly calibrate the predictive uncertainty meanwhile enhancing adversarial robustness.

### Strengths
(1) The paper writes clearly and is easy to follow.

(2) Incorporating the Dirichlet parameterization technique to adversarial training is interesting.

### Weaknesses
(1) Beyond CLIP, the Dirichlet parameterization is a general technique and could also be applied to traditional adversarial training on the image classification task.
    In the community of adversarial training, there are lots of existing work to improve adversarial robustness, like DKL [ref1], ACAT [ref2], and TRADES.
    Would it be possible to include experiments on a standard benchmark such as CIFAR-100 or CIFAR-10 under DKL, ACAT, and TRADES setups to demonstrate that the proposed Dirichlet uncertainty calibration generalizes beyond CLIP? Otherwise, the paper should clarify why it is inherently tied to CLIP.

[ref1] Decoupled Kullback-Leibler Divergence Loss. NeurIPS 2024.                           
[ref2] Efficient and effective augmentation strategy for adversarial training. NeurIPS 2022..

(2) Although calibration is the central claim, the paper does not compare against previously established uncertainty calibration methods.

(3) Regarding Fig. 3(b), what data are used to evaluate the models?

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

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
1.The paper identifies that CLIP becomes overconfident under adversarial perturbations, leading to miscalibrated uncertainty and unreliable predictions.

2.It introduces a Dirichlet-based reformulation of CLIP’s logits, enabling a unified representation of inter-class relationships and predictive confidence.

3.The authors propose UCAT, an uncertainty-calibrated adversarial fine-tuning objective that aligns full Dirichlet distributions between clean and adversarial examples rather than single-class logits.

4.Experimental results across several zero-shot benchmarks and MS-COCO demonstrate that the proposed method restores calibrated uncertainty, maintains clean accuracy, and achieves competitive adversarial robustness.

### Strengths
1.The paper introduces a Dirichlet-based reformulation of CLIP’s logits, which provides a theoretically grounded way to capture both inter-class relationships and predictive confidence.

2.The theoretical analysis and derivations are presented clearly and are easy to follow, making the methodology accessible to readers.

3.The paper is well-structured, with a logical flow from motivation to method to experiments, which helps communicate the ideas effectively.

4.The experiments are extensive, covering multiple zero-shot benchmarks and MS-COCO, demonstrating both the practical applicability and robustness of the approach.

### Weaknesses
1.The paper is motivated by the observation that CLIP can produce overconfident predictions under adversarial attacks, revealing a gap between accuracy and predictive uncertainty. However, this motivation is not sufficiently novel to fully justify the proposed solution.

2.The paper focuses on calibrating uncertainty for zero-shot adversarial CLIP, but it does not clearly explain why the proposed method is specific to CLIP or zero-shot learning. It appears that similar results could be achieved on standard image classification tasks, which raises questions about the task-specific significance of the approach.

3.The proposed method is only evaluated on CLIP. Although the theoretical analysis explains how the Dirichlet reformulation relates to traditional softmax logits and its applicability to CLIP, the paper does not clarify why CLIP is chosen for this method or what specific benefits arise from applying it to CLIP compared to other models.

4.In the ablation study, the adversarial attacks used for comparison are outdated, which limits the ability to fully demonstrate the advantages of the proposed method.

5.The analyses in Sections 3 and 4 mainly demonstrate the validity of applying the Dirichlet reformulation, but the author do not clearly show the advantages of the proposed method. The paper lacks discussion or evidence on how it outperforms existing approaches or what specific benefits it provides compared to other methods.

### Questions
1.The authors should clarify the connection between their method and CLIP as well as zero-shot learning. It would be helpful to explain why calibrating uncertainty is particularly meaningful in the context of zero-shot CLIP, and whether the proposed approach is specific to this setting. If the method can also be applied to other tasks, the authors could include additional experiments or comparisons to demonstrate its broader applicability and to justify the claimed contributions.

2.I hope the author could include comparisons with other vision–language models or adversarially trained models.This would help clarify whether the proposed method offers unique advantages for CLIP or if similar benefits can be achieved on other models.

3.I hope the author could include more recent and stronger attack methods to provide a more convincing evaluation of their approach.

4.The authors should clarify why using a Dirichlet distribution leads to improvements and what advantages it offers over the traditional softmax approach. They should also explain why this particular distribution is necessary for aligning the distributions of clean and adversarial samples, and  I am curious whether other distributions could be used here instead.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes to enhance the zero-shot classification robustness of the CLIP model, an Uncertainty-Calibrated Adversarial fine-Tuning framework for CLIP (UCAT). The authors introduce a loss function for adversarial training that considers both prediction accuracy and uncertainty alignments. The method is evaluated on multiple datasets and shows improved robustness against adversarial attacks compared to existing methods.

### Strengths
The motivation for improving CLIP's zero-shot robustness is well articulated, and the proposed method is supported by thorough experiments. The authors provide comprehensive evaluations on various datasets and attack methods, demonstrating the effectiveness of their approach. The ablation studies further validate the contributions of different components of the proposed loss function.

### Weaknesses
1. The choice of concentration parameter alpha for the Dirichlet distribution in Definition 4.1 is not well justified. The authors should provide insights into how this parameter is chosen and its sensitivity to performance.

2. The proposed method shows lower performance on certain datasets (SUN397 and PCAM) as seen in Table 1. The authors should discuss potential reasons for this discrepancy.

See the questions.

### Questions
1. Since the goal of this paper is to improve the zero-shot robustness of CLIP, when the labels of the downstream tasks are unknown during training, how does the loss term encourage uncertainty alignment? Is there any insight into why this helps improve robustness on those tasks?

2. Since the Dirichlet distribution is a Bayesian prior over categorical distributions, it is usually sensitive to the choice of concentration parameters. Definition 4.1 introduces a concentration parameter alpha for the Dirichlet distribution. Do the authors have any insights on why they chose this parameter and how sensitive the performance is to this choice? Also, parameters such as tau prime should be considered in ablation studies.

3. The proposed methods outperform existing methods on most datasets, but there are some cases where the performance is consistently lower (SUN397 and PCAM in Table 1). Do the authors have any insights into why this happens?

4. In Table 2 and Table 5, why is the clean average accuracy not consistent？ Since they are all evaluated on the same datasets, shouldn't the clean accuracies be identical?

### Soundness
3

### Presentation
3

### Contribution
3
