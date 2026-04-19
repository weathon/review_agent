# Adaptive Energy Alignment for Accelerating Test-Time Adaptation

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 3

## Abstract
In response to the increasing demand for tackling out-of-domain (OOD) scenarios, test-time adaptation (TTA) has garnered significant research attention in recent years. To adapt a source pre-trained model to target samples without getting access to their labels, existing approaches have typically employed entropy minimization (EM) loss as a primary objective function. In this paper, we propose an adaptive energy alignment (AEA) solution that achieves fast online TTA. We start from the re-interpretation of the EM loss by decomposing it into two energy-based terms with conflicting roles, showing that the EM loss can potentially hinder the assertive model adaptation. Our AEA addresses this challenge by strategically reducing the energy gap between the source and target domains during TTA, aiming to  effectively align the target domain with the source domains and thus to accelerate adaptation. We specifically propose two novel strategies, each contributing a necessary component for TTA: (i) aligning the energy level of each target sample with the energy zone of the source domain that the pre-trained model is already familiar with, and (ii) precisely guiding the direction of the energy alignment by matching the class-wise correlations between the source and target domains. Our approach demonstrates its effectiveness on various domain shift datasets including CIFAR10-C, CIFAR100-C, and TinyImageNet-C.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Adaptive Energy Alignment (AEA) for fast test-time adaptation (TTA) in out-of-domain scenarios. It reinterprets entropy minimization (EM) loss and shows that its conflicting energy-based components can hinder adaptation. AEA reduces the energy gap between source and target domains by aligning target sample energy levels with familiar source domains and matching class-wise correlations. The approach achieves efficient TTA with minimal computational overhead, demonstrating strong results on datasets like CIFAR10-C, CIFAR100-C, and TinyImageNet-C.

### Strengths
*	This paper is well-written, well-organized, and easy to follow.
*	The insight of leveraging the direction of energy alignment with the guidance of the structural relations between different classes is convince, and provide a new solution for accelerating test-time adaptation
*	The proposed SFEA and LCS loss is novel to me, effectively reduce the energy gap and aligns the class-wise correlation across source and target domains.
*	Overall, the quality of the paper is commendable. The authors have conducted a thorough comparison experiments to examine the effectiveness of the proposed method. Additionally, they have made the code and datasets available in the supplementary material.

### Weaknesses
*	While the proposed AEA significantly accelerates test-time adaptation (as observed in Figure 1c), the authors provide insufficient discussion on how AEA achieves this acceleration. Further analysis and discussion are needed.
*	I noticed that in TEA, the backbone for CIFAR10-C is WRN-28-10, whereas ResNet-26 is used in this work. If the experimental results are reproduced using publicly available code, please provide the relevant details in the table caption.

### Questions
Please refer to the weakness part.

---
I am open to discussing these points with the authors during the response period. If the concerns and questions are adequately addressed, I will consider raising my score.

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
This paper proposes an Adaptive Energy Alignment (AEA) method that enables fast online test-time adaptation (TTA). The introduced Source-Free Energy Alignment (SFEA) loss strategically aligns the overall energy magnitudes between source and target domains. Additionally, the authors propose a Logit Cosine Similarity (LCS) loss to ensure that class-wise correlations in the target domain align well with those in the source domain during energy alignment. Extensive experiments on three datasets demonstrate the advantages of the proposed AEA method.

### Strengths
*	The proposed SFEA loss provide novel insight for TTA, effectively reduce the domain gap.
*	The authors have conducted a thorough ablation study to examine the impact of various components. Additionally, they have made the code and datasets available in the supplementary material.
*	The paper is well-written, well-organized, and easy to read.

### Weaknesses
*	While the authors demonstrate the superiority of the proposed method on TTA, it would be interesting to evaluate its performance in a similar scenario, such as source-free domain adaptation (SFDA), where source data is also inaccessible. Additional evaluations could further illustrate the versatility of the proposed method.
*	The authors did not clearly discuss the distinction between online test-time adaptation (OTTA) and TTA. In line 142, they claim to address the OTTA problem, but in other parts of the paper, the main focus appears to be on TTA.
*	A minor issue concerns the punctuation of equations. Eqs. (3), (4), and (9) should include the appropriate punctuation.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper targeted at accelerating test-time adaptation from a new perspective, i.e., reducing the energy gap between the source and target domains. There are two key losses in the proposed method. The one is the source-free energy alignment (SFEA) loss to align the overall energy magnitude and the other is the logit cosine similarity (LCS) loss to guide the class-wise alignment direction. Experimental results on three common benckmarks (CIFAR10-C, CIFAR100-C, and TinyImageNet-C) have demonstrated the effectiveness of the proposed method.

### Strengths
The paper is easy-to-follow. The topic is essential yet the idea is moderate. Both test-time adaptation and energy-based models are important topics for the community and the paper addresses two formulations simultaneously.

### Weaknesses
- **Limited motivation.** My major concern is about the motivation of this work. First, the necessity of the proposed logit cosine similarity (LCS) loss is not clear. Second, though the authors have tried to describe the correlation between energy gap reduction and better adaptation, it's not clear why TTA needs to do this. Especially, many previous works have proven its effectivebess in addressing distribution shift problems .Third, the proposed method is complicated but the improvements in Tables 1,2,3 is minor. 

- **Insufficient justifications.** From main results of Table 1,2,3, it can be observed that the proposed AEA obtains limited improvements in three small-scale image classification datasets (CIFAR10-C, CIFAR100-C, and TinyImageNet-C). Large-scale datasets and other tasks like segmentation or detection might be evaluated to demonstrate the AEA's utlization in various scenarios and applications.

### Questions
- How to define the directional aspects of energy alignment?

- It seems that the source-free energy alignment loss ($\mathcal{L}_{SFEA}$) is calculated in each target batch. How can this calculation method avoid the noise caused by bath sampling?

Minor comments:
- Actually, the foundamental of this work is to answer why does energy alignment lead to better adaptation. Thus, in my view, Sec. A.4 should be moved to the main context and the revisit of EMBs can be moved to the appendix.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces an energy-based approach for addressing the test-time adaptation (TTA) problem. The authors analyze the classical entropy minimization loss widely adopted by existing TTA methods through the lens of energy-based model, pinpointing the potential issue lies within the entropy minimization. To address the issue, two alignment objectives termed source-free energy alignment (SFEA) and logit cosine similarity (LCS) are proposed. SFEA aligns the overall energy magnitudes between source and target domains to reduce the energy gap during adaptation, whereas LCS ensures the class-wise correlation of the target domain maintains consistent during energy alignment. Experiments are conducted on several common TTA benchmarks including CIFAR10-C, CIFAR100-C, TinyImageNet-C and ImageNet-C, as well as style shift dataset PACS. The results of the proposed method generally surpass the baselines.

### Strengths
1.	As stated in the paper, Test-time adaptation is an important direction to mitigate the out-of-distribution issue of inference data and worths exploration.
2.	The experiment is extensive on several benchmarks, and the proposed method achieves good results on most tasks within these benchmarks.

### Weaknesses
1.	In both the summary and the introduction, the authors argue that “the EM loss can be decomposed into two energy-based terms with conflicting roles” and such confliction “hinders the energy gap reduction when the EM loss is used alone.” However, the authors do not address this issue in the proposed method. Minimizing the target energy does not resolve the conflict. On the contrary, the reviewer thinks that the free-energy maximization term in equation (4) is necessary for avoiding the explosion of the magnitude of logits. The reviewer’s opinion is supported by the fact that the final objective of the proposed method still incorporates EM loss. 
2.	The authors highlight the strength of the proposed method that can reduce the energy gap more quickly than EM loss in the early batches. However, it appears from fig. 2 that the proposed method performs similarly to other baselines at the beginning. The result makes the author’s claim questionable.
3.	Given that previous works [1][2] adopt hinge loss for energy alignment, the reviewer suggests that the authors should conduct ablation study on the choice of loss to support their design.
4.	For each target batch, the proposed method computes a new approximated energy of the source domain, which means that the estimated source energy varies over time. This is a seemingly strange design without explanation. In fact, the reviewer suspects that this particular design is the actual reason for performance enhancement and distinguishes it from the previous energy alignment method. In reviewer’s opinion, the target energy is not actually aligned with the source energy, but rather is minimized towards an adaptively changing goal. This allows the energy minimization process to be more flexible and friendly to the model training.
5.	The second line of equation (4) is mistaken. It should include parentheses over the two energy terms. 

Ref: 
[1] Energy-based out-of-distribution detection.
[2] Active learning for domain adaptation: An energy-based approach.

### Questions
1.	In the caption of fig. 3, the ‘relative distance in a logit space’ is confusing. Please elaborate on the meaning of the x-axis in detail.
2.	From the experiment results, it appears that SHOT achieves better performance than most of the TTA methods. What is the reason?

### Soundness
2

### Presentation
2

### Contribution
2
