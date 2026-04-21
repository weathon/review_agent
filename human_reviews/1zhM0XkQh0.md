# ProFeAT: Projected Feature Adversarial Training for Self-Supervised Learning of Robust Representations

- Avg Score: 5.75
- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Supervised adversarial training has been the most successful approach for improving the robustness of Deep Neural Networks against adversarial attacks. While several recent works have attempted to overcome the need for supervision or labeled training data by integrating adversarial training with contrastive Self-Supervised Learning (SSL) approaches such as SimCLR, their performance has been sub-optimal due to the increased training complexity. A recent approach mitigates this by utilizing supervision from a standard self-supervised trained model in a teacher-student setting that mimics supervised adversarial training. However, we find that there is still a large gap in performance when compared to supervised training, specifically on larger capacity models. We show that this is a result of mismatch in training objectives of the teacher and student, and propose Projected Feature Adversarial Training (ProFeAT) to bridge this gap by using a projection head in the adversarial training step. We further propose appropriate attack and defense losses at the feature and projector spaces, coupled with a  combination of weak and strong augmentations for the teacher and student respectively, to improve generalization without increasing the training complexity. We demonstrate significant improvements in performance when compared to existing SSL methods, and performance on par with TRADES, a popular supervised adversarial training method, on several benchmark datasets and models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed to bridge the gap between the self-supervised and supervised adversarial training methods, with good scalability for larger models.

This paper is well-written and easy to follow.

### Strengths
The topic of the paper, bridging the gap between supervised and self-supervised adversarial training, is interesting. 

The result is convincing. 

This paper applied the proposed method to different DNN models, including popular VIT.

Extensive ablation studies are provided, which shows the insight of the proposed method.

Extensive results are shown in the appendices, which are helpful for the readers to understand the whole story.

### Weaknesses
First, the author should give some basic explanation for the results, which will be appreciated. For example:
1. In Table 3, the author should at least explain what is "SA" (standard accuracy?). 
2. In Table 4, the author should at least highlight which method has better results for each network structure (each row). 
3. what is the adversarial perturbation upper bound for testing in the table 3 and 4?

In Table 3, for DeACL (reproduced), the author modified the original teacher model. In this case, the word "reproduced" is misleading. In fact. it is not "reproduced", but modified. 

"DynACL is run for 500 epochs rather than....":
In this case, the reported result of DynACL should not be used for comparison. Because the result is not from the proposed settings, the result is not convincing. The explanation in Appendix D cannot overcome this problem.

### Questions
1. In Table 3, why not highlight the best result in column PGD-20?

2. According to the results, the proposed method gives a better improvement on the larger model (WRN-34-10) than the smaller one (RES-18). This is very interesting. Can the author give some explanation and insight about this?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a novel self-supervised learning (SSL) method named ProFeAT to enhance the robustness of Deep Neural Networks (DNNs) against adversarial attacks. While supervised adversarial training has proven effective, it demands extensive labeled data, leading to high costs. Previous SSL attempts, including SimCLR and Decoupled Adversarial Contrastive Learning (DeACL), have shown limitations in performance and increased training complexity, particularly with larger models. ProFeAT addresses these issues by incorporating a projection head in the adversarial training step, defining specific attack and defense losses, and employing a mix of weak and strong augmentations for the teacher-student setting. This strategy aims to close the performance gap between self-supervised and supervised adversarial training, enhancing generalization without adding to the training complexity.

### Strengths
1. ProFeAT introduces a novel method to improve the robustness of DNNs through self-supervised learning, addressing the challenges of previous SSL adversarial training methods.
2. The paper provides extensive experimental results, demonstrating the effectiveness of ProFeAT across different datasets and model architectures.

### Weaknesses
1. The paper primarily relies on linear probing for evaluation, which is just one of several methods to assess the quality of a trained encoder. It is crucial to explore alternative evaluation techniques such as K-nearest neighbors (KNN) to validate the model's performance comprehensively. Additionally, the effectiveness of the pretraining method in downstream tasks with the finetuning method should be rigorously verified.

2. Section 4.1 lacks compelling evidence and in-depth analysis. The paper lacks a thorough explanation of **why** objective matching leads to improved performance, and there is insufficient exploration of **how** aligning the linear probing objective with pretraining aids in distillation. The correlation between high cosine similarity and low performance is demonstrated in both Table 1, and 2 but lacks meaningful context.

3. While the empirical exploration of simple/difficult augmentation combinations is commendable, the paper lacks a robust analysis or rationale behind why the proposed combinations are deemed the most effective.

4. Although the paper explores various approaches, the methodology lacks strong justification, making it challenging to establish the credibility of the proposed methods. A more thorough and convincing demonstration of these approaches is needed.

### Questions
In my view, this paper demonstrates that the use of the freeze teacher projector can provide a slightly more robust constraint, promoting alignment between the student's feature space and that of the teacher, thus facilitating stable student learning. Consequently, the results show only marginal improvements compared to DeACL for smaller models, while proving more effective in scenarios where distillation regularization becomes challenging, particularly as the model scales up in size. I wonder what the authors think about this interpretation.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes Projected Feature Adversarial Training (ProFeAT) to address the gap between the target of self-supervised training and adversarial training in the teacher-student distillation setting. ProFeAT uses a frozen pretrained projection head from the teacher to isolate impact of distillation loss and prevent the overfitting of the student to the teacher training objective. The performance of ProFeAT is better than existing SSL adversarial training method especially on larger models.

### Strengths
The paper is clearly written.

The experimental result is solid, including models of different sizes and extensive ablation study.

The performance of ProFeAT is competitive on large-scale models.

### Weaknesses
1. There are multiple components in the proposed method, including an additional loss of projection layers, a new attack generation method and weak data augmentation. Although the ablation study includes variants of each component, it is still unknown the specific contribution of each component, or the combination of any two components. A mechanism for why the combination of the three components works is needed. 

2. The experiment is done on CIFAR10 and CIFAR100, which contains sufficient data to train adversarially robust models even without distillation and pre-trained models [1]. It is more interesting to see the performance of ProFeAT on low-data tasks such as Caltech [2] as [3] shows that the benefit of pre-trained models mainly manefests in low-data tasks. 

[1] Wu, Dongxian, Shu-Tao Xia, and Yisen Wang. "Adversarial weight perturbation helps robust generalization." Advances in Neural Information Processing Systems 33 (2020): 2958-2969.
[2] Griffin, Gregory, Alex Holub, and Pietro Perona. "Caltech-256 object category dataset." (2007).
[3] Liu, Ziquan, et al. "TWINS: A Fine-Tuning Framework for Improved Transferability of Adversarial Robustness and Generalization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
1. Could the authors show the effect of each component in the proposed method?

2. It is more convincing to show the effectiveness of the proposed method on small datasets such as Caltech. I believe the benefit of using self-supervised pre-training would be significant on such small datasets.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper analyzed and improved the robustness of representation learned by self-supervised contrastive learning methods against adversarial attacks. They find a significant performance gap in the more prominent models (e.g., WideResNet-34-10) for the existing techniques. They attribute this to the mismatch between the objectives of the teacher and student models. And they find that an additional projector layer with some losses can help to mitigate such discrepancies and improve performance. The evaluations have been done on CIFAR-10 and CIFAR-100.

### Strengths
- The method is simple with the use of a projector network for adversarial training
- The effectiveness in the metrics standard accuracy (SA) and robustness against AutoAttack (AA) is encouraging

### Weaknesses
•	The novelty is marginal: The projector is widely used in SSL which can significant boost performance when evaluating the representations after the backbone network. Here, the use of such projector can improve the representation (as demonstrated in SA metric) is not surprising in self-supervised learning for adversarial attack. Also, the use of weak augmentation for the teacher and strong one for student is exploited in several prior works. 
•	The robust accuracy (RA) is an important metric in evaluation of the robustness but it seems to be omitted in the main paper (table 3,4,5,6), could the author provide the results and analysis of this metric side by side, too?
•	The PGD-20 metric of the proposed method is pretty worse than the other SOTA in most cases but it is not adequately discussed or mentioned. Could the authors provide some intuitions why does such degradation on PGD-20 happen, any investigation to address that drawback?
•	Clarity: 1) It should be consistent in the style, for example, table 3, 5 where the bold results show the best performance but table 4, 6, … are not highlighted, making it hard to follow which one is better. 2) It should be also consistent to report the metric in table 3,4,5,6, for example, table 3 used the mixture “SA” and “AutoAttack”, while table 4 used the full metric “Standard Accuracy”, etc… the consistency should be done for all tables. 3) Since AA has been used for “AutoAttack”, it should be used differently for AA when referring the AutoAugment (table 8) to avoid confusing.

### Questions
See weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
