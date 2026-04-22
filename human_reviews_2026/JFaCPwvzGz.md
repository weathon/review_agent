# Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Adversarial distillation in the standard min–max adversarial training framework aims to transfer adversarial robustness from a large, robust teacher network to a compact student. However, existing work often neglects to incorporate state-of-the-art robust teachers. Through extensive analysis, we find that stronger teachers do not necessarily yield more robust students–a phenomenon known as robust saturation. While typically attributed to capacity gaps, we show that such explanations are incomplete. Instead, we identify adversarial transferability–the fraction of student-crafted adversarial examples that remain effective against the teacher–as a key factor in successful robustness transfer. Based on this insight, we propose Sample-wise Adaptive Adversarial Distillation (SAAD), which reweights training examples by their measured transferability without incurring additional computational cost. Experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet show that SAAD consistently improves AutoAttack robustness over prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper finds that adversarial transferability plays an important role in adversarial robust distillation(ARD), and further proposes a Sample-wise Adaptive Adversarial Distillation to enhance the effectiveness of ARD.

### Strengths
1. The paper is easy to follow.
2. The experiment seems to be effective.

### Weaknesses
1. The innovation of the argued point is limited. To explain the shortcomings of ARD,  two important baseline methods: B-MTARD[1] and ABSLD[2] should be discussed. The adjustment from the view of teacher entropy has been applied in B-MTARD, which controls the training process via adjustment for the temperature, and ABSLD also applies the weight corresponding entropy in ARD to enhance the class-wise fairness. Meanwhile, the adjustment towards the clean teacher also exists in B-MTARD. However, this paper seems to ingore those baselines, leading to limited innovation and insufficient contribution. 

2. This insihgt lacks theoretical explanation. The paper aruges that overconfident soft labels induce high adversarial variance, and high adversarial variance causes robust overfitting. A detailed theoretical explanation towards those two findings are supposed.

3. The data selection (Open discussion). I wonder if general data enhancement method can alleviate this phenomenon. 

[1] Mitigating accuracy-robustness trade-off via balanced multi-teacher adversarial distillation, TPAMI2024.

[2] Improving Adversarial Robust Fairness via Anti-Bias Soft Label Distillation, NuerIPS2024.

### Questions
See the weakness

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work a new adversarial distillation technique is proposed which is called Sample-wise Adaptive Adversarial Distillation. The main idea behind this new training regime is that the transferability of adversarial examples actually plays a role in how robust a student model is. Experiments are done with respect to CIFAR-10, CIFAR-100 and Tiny-ImageNet.

### Strengths
The concept presented in the paper is interesting and the experiments span multiple datasets. The paper is written in a manner that is easy to understand and follow.

### Weaknesses
=In terms of discussion of related work there are a few missing references. It would be good if 1-2 sentences can be used in the main paper to denote how the paper fits within the scope of existing literature. 

First regarding transferability, the authors do not cite any of the key works in this field, which is a major oversight. These works include some of the original transferability studies and more recent work on the study of transformer transferability:

0. One of the first Transferability works: https://arxiv.org/pdf/1704.03453
1. Delving into Transferability (CNNs): https://openreview.net/pdf?id=Sys6GJqxl
2. Vision Transformer Transferability: https://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html?utm_source=apperceptive&utm_medium=email&utm_campaign=why-those-training-data-poisoning-gimmicks-dont

In addition, there are a few knowledge distillation papers that are either a fundamental part of the field or very related to the concept the authors are proposing: 

3. Fundamental work (MTARD): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640577.pdf

4. Similar work that also leverages architecture diversity and transferability for adversarial knowledge distillation:  https://dl.acm.org/doi/10.1016/j.procs.2025.07.114

Specifically, in regards to reference #4, I am concerned about the overlap between that work and the work the authors are proposing. If the authors are able to adequately include more discussions with citations to the appropriate work in the paper, I would be inclined to increase my score. 

=There are no studies done on state-of-the-art black-box attacks as far as I can tell (beside what is already part of the set of attacks in auto-attack). Would the authors be willing to at least consider some experiments on a boundary attack like Rays?

https://arxiv.org/abs/2006.12792

### Questions
1. Could you add in discussions of the references I mentioned in the weakness section to your paper in an appropriate way. This could further improve the writing and also you can properly cite the original Transferability concept in the literature. 

2. Can you explain why you don't really have any SOTA black-box attacks or references to black-box attacks in your paper? I am aware that black-box attacks generally perform worse than white-box attacks in this field, but it would be good to have some baseline measurements none the less. Or if you believe this is not relevant please offer an explanation in the rebuttal.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper first shows that a stronger robust teacher does not necessarily lead to a stronger student and can, in fact, degrade the student’s robustness. The authors then explore various factors contributing to this issue and propose a hypothesis for this saturation, suggesting that overconfident soft labels from the teacher induce robust overfitting. Finally, they introduce a sample weighting method that compares the adversarial perturbation differences between the student and the teacher for effective label distillation.

### Strengths
Overall, the paper is well written and easy to follow. It identifies an interesting problem of robustness transfer saturation and follows through with a compelling hypothesis and a interesting method.

### Weaknesses
Several important prior works are omitted [1, 2, 3, 4].

In lines 136–137, the authors mention that most existing studies rely on older models:

> Although stronger robust models are now readily available through resources such as RobustBench (Croce et al., 2021), most existing AD studies still rely on older and less robust teacher networks.

This statement is not entirely accurate, as many works, even those published a few years earlier, have demonstrated robustness transfer across larger models. For instance, Chan et al. [1] and Awais et al. [2] employed large Wide ResNets in their studies.

While discussing robust distillation in the introduction and the early parts of Section 3, the authors do not clearly mention which distillation method they used. The specific distillation method is more influential compared to other factors proposed to explain the saturation problem. In general, distillation performance is highly sensitive to several aspects, including hyperparameters and the type of distillation (e.g., feature-based vs. logit-based).

Several works have explored robustness distillation through the feature space. For example, Chen et al. [1] investigated gradient-based (effectively feature) distillation, while Awais et al. [2] utilized intermediate feature representations. Therefore, it is important that the authors differentiate their explanation from these prior works. A more plausible hypothesis for robustness saturation could be the limited utility of logits in robustness transfer. Prior studies [5, 6] have extensively shown that features learned through adversarial training differ significantly from those learned through standard training. For these reasons, I strongly encourage the authors to revise the paper to focus on the limitations of robustness transfer via logits, rather than presenting this as a general explanation for robustness transfer.

Including results on ImageNet would make the paper more comprehensive and strengthen the empirical evaluation.

[1] Chan, A., Tay, Y., & Ong, Y. S. (2020). What it thinks is important is important: Robustness transfers through input gradients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 332-341).

[2] Awais. M., Zhou, F., Xie, C., Li, J., Bae, S. H., & Li, Z. (2021). Mixacm: Mixup-based robustness transfer via distillation of activated channel maps. Advances in neural information processing systems, 34, 4555-4569.

[3] Shao, R., Yi, J., Chen, P. Y., & Hsieh, C. J. (2021). How and when adversarial robustness transfers in knowledge distillation?. arXiv preprint arXiv:2110.12072.

[4] Awais, M., Zhou, F., Xu, H., Hong, L., Luo, P., Bae, S. H., & Li, Z. (2021). Adversarial robustness for unsupervised domain adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 8568-8577).

[5] Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., & Madry, A. (2019). Adversarial examples are not bugs, they are features. Advances in neural information processing systems, 32.

[6] Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., & Madry, A. Robustness May Be at Odds with Accuracy. In International Conference on Learning Representations.

### Questions
On line 337, authors mention: 
> Based on this insight, SAAD assigns sample-wise weights proportional to the entropy of fT (x + δS), effectively prioritizing transferable adversarial examples without incurring additional computational cost.

It is unclear how the proposed method incurs no additional computational cost, given that it requires the generation of adversarial examples for both the student and the teacher. If I understand correctly, the approach requires generating adversarial examples for the student at each training step, as well as at least one-time generation and storage of adversarial examples for the teacher model. This could be quite expensive, especially when using a large teacher model.
To clarify this claim, I suggest including a training time comparison with standard training, adversarial training, adversarial robustness distillation, and your proposed method.

It would also be helpful to specify the teacher architecture in the results tables (e.g., Table 4) for better reproducibility and clarity.

### Soundness
2

### Presentation
4

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
This paper proposes Sample-wise Adaptive Adversarial Distillation (SAAD), a framework to improve adversarial knowledge transfer from a robust teacher to a compact student. The authors revisit the so-called robust saturation, the observation that stronger teachers do not always yield more robust students, and attribute this to low adversarial transferability, defined as the proportion of student-crafted adversarial examples that also fool the teacher. Authors leverage this finding by weighting each training sample based on the entropy of the teacher’s prediction on student-generated adversarial examples, assigning higher weights to transferable samples and inverse weights for clean distillation. This unified weighting scheme adaptively balances robust and clean knowledge transfer.

### Strengths
- Clear diagnostic analysis: The paper convincingly shows that capacity gap alone cannot explain robustness transfer failure, and introduces adversarial transferability as a key explanatory factor.
- Simple yet effective weighting mechanism: The entropy-based adaptive weight is computationally light and easily applicable to other AD frameworks.
- Strong empirical results: SAAD demonstrates consistent improvements in both clean and robust accuracy across datasets and architectures, with clear reductions in adversarial variance and robust overfitting.

### Weaknesses
- Overemphasis on entropy reliability: The assumption that high entropy teacher outputs always indicate beneficial transferability may not hold when entropy arises from noisy or uncertain predictions. Assigning large weights to such samples could amplify noise and harm stability.
- Limited diversity of teacher-student pairs: The evaluation mainly uses WRN-based teachers and ResNet/MobileNet students. It remains unclear whether the conclusions generalize to transformer-based or large-scale settings.
- Underexplored theoretical justification: While the empirical correlation between entropy and transferability is convincing, the theoretical connection from (4) to (5) could be elaborated further to strengthen generality.
- Minor clarity issues: Some details, such as the “interpolation coefficient” in Fig. 3(b) and the normalization of entropy weights, could be described more clearly.

### Questions
- How does SAAD perform when entropy is dominated by label noise rather than genuine transfer uncertainty?
- Could the entropy-based weighting be replaced or complemented by other measures of teacher reliability (e.g., teacher-student disagreement or gradient alignment)?


** Overall Evaluation **

This paper provides an insightful analysis of the limitations of adversarial distillation and introduces a lightweight method that substantially improves robustness transfer. The contribution is empirically strong and conceptually well-motivated, though a deeper examination of noise sensitivity and broader validation would further promote the impact of the paper.

### Soundness
3

### Presentation
3

### Contribution
3
