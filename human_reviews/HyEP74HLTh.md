# Adversarial Robust Representation Learning via Contrast and Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 8, 5, 3

## Abstract
Deep neural networks are vulnerable to adversarial noise. Adversarial training (AT) has been demonstrated to be the most effective defense strategy to protect neural networks from being fooled. However, we find AT omits to learning robust features, resulting in poor performance of adversarial robustness. To address this issue, we highlight two characteristics of robust representation: 
$(1) exclusion$: the feature of natural examples keeps away from that of other classes; $(2) alignment$: the feature of natural and corresponding adversarial examples is close to each other. These motivate us to propose a generic framework of AT to gain robust representation, by the asymmetric negative contrast and reverse attention. Specifically, we design an asymmetric negative contrast based on predicted probabilities, to push away examples of different classes in the feature space. Moreover, we propose to weight feature by parameters of the linear classifier as the reverse attention, to obtain class-aware feature and pull close the feature of the same class. Empirical evaluations on three benchmark datasets show our methods greatly advance the robustness of AT and achieve state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes two characteristics for learning robust features: exclusion and alignment. Correspondingly the paper proposes two techniques" asymmetric negative contrast to push away features from differetn class, and reverse attention to align features of the same class.

### Strengths
1. This paper propose novel training objectives sepcifically designed for robust feature optimization
2. The optimization objective of exclusion and alignment is well motivated
3. Significant robustness improvement is achieved by the proposed method, especially with reverse attention

### Weaknesses
1. In the introduction, the paper claims "The overlook may lead to potential threats in the feature space of AT models, which harms robust classification", but does not illustrate what threat exactly. Since this is the motivation of the paper it would be better to provide some examples or citations.
2. The motivation of having asymetric negative contrast is not clear. For the example in Figure 2, given the randomness in negative pair selection, the clean car should be away from both dog and cat feature distributions, rather than just away from dog and close to cat. Performing asymetric negative contrast also does not lead to much difference in the ablation comparing to symetric loss
3. The formulation of the reverse attention is not clear. In Equ. (5), $z_i$ and $\omega^i$ appears to be scaler values, so the Hadamard product isn't needed. Also it is unclear how exactly is the reverse attention strengthen the confidence of the model. It will be better to formulate the entire forward pass from z to p' will help, or directly visualize the comparison of p and p'
4. From the ablation study, significant performance improvement is brought by reverse attention. However, it is uncertain if the improved robustness is brought by gradient masking, where the existance of reverse attention smoehow blocks the gradient propagation for whitebox attack generation. Providing results on blackbox transfer attacks from other models is needed.

Minor point: The use of abbreviation (OE, NP, PP etc.) is very nonstandard and make the paper hard to follow. If the abbreviations have to be used then having a table summarizing these abbreviations will be helpful.

### Questions
See weakness

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work aims to improve the adversarial training (AT) techniques from the perspective of learning robust representation. The authors find previous AT method suffer from poor representation, and highlight two characteristics of robust features: Exclusion and Alignment. These two attributes ensure each natural and adversarial example sets close in the corresponding class, and maintains a large margin with other classes. With the guidance of two characteristics, this work proposes two techniques. First, an asymmetric negative contrast based on probabilities (ANC) is proposed to meet Exclusion. It increases the gap between the natural examples and negative examples from other classes. Second, reverse attention (RA) is proposed to boost Alignment, which aligns the features of natural and adversarial examples by weighting. The framework can be used in a plug-and-play manner with existing AT methods. Empirical evaluations on various datasets and models have proved the validity of the method.

### Strengths
1.	This work proposes Exclusion and Alignment as two targets to explicitly enhance robust feature. The idea of improving AT by learning robust representation is novel and interesting.
2.	ANC and RA are intuitive and well-designed, which can combined with other algorithms in a plug-and-play manner to induce robust representation.
3.	Extensive experiments on three datasets and substantial diagrams of feature visualization show it has an impressive performance, compared to previous state-of-the-art methods.

### Weaknesses
1.	As shown in Limitation, wrong predicted labels may lead to wrong weighting, which will have a negative effect on classification.
2.	In Tab 6 in the ablation studies, the auxiliary probability vectors p^0 and p^1 have achieved the same performance as the final logits, so how does RA work in the inference process?

### Questions
1.	As shown in Limitation, wrong predicted labels may lead to wrong weighting, which will have a negative effect on classification.
2.	In Tab 6 in the ablation studies, the auxiliary probability vectors p^0 and p^1 have achieved the same performance as the final logits, so how does RA work in the inference process?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper provides an adversarial training (AT) framework to enhance the exclusion and alignment of robust features to gain robustness. It designs an asymmetric negative contrast based on predicted probabilities to push away (exclude) the features of natural examples and other-class examples (OEs), and proposes the reverse attention to align features of natural examples with adversarial examples (AEs).

### Strengths
1. This paper proposes an asymmetric negative contrast to put the natural examples away from the examples of other classes.
2. This paper designs a reverse attention block to make the features of natural and corresponding adversarial samples close to each other.
3. The experimental results show the effectiveness of the proposed methods.

### Weaknesses
1. The so-called exclusion and alignment are not new, these names are reinventing the wheels. Similar ideas with different names have already been proposed in [1]-[3] below.

[1] Metric Learning for Adversarial Robustness. (2019)
[2] Adversarial logit pairing. (2018)
[3] Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction. (2020)

2. Figure 1 only shows the distance between natural example and AE/OE, lacking the distances among same-class natural examples. Why not try t-SNE visualization to show the distributions of these three class points?

3. How to avoid robust feature collapse (i.e. all features converge to 0)? In this case, the features of natural examples and AEs are well-aligned.

4.  The methods listed in this paper for comparison, AT, TRADES, MART, AWP, etc., are too low in accuracy and robustness. (SAT, S2O, UDR are too weak and shouldn't be used as baselines at all, I suggest to just remove them.)  For example, taking ResNet-18 on CIFAR-10 against PGD-40, they should have a robustness higher than 51% (44% in this paper) and an accuracy higher than 84% (76% of AWP in this paper). Getting an increase from a reasonable baseline is more convincing for the validity of the proposed method.

5. In Section 3.3, what does it mean for the n blocks in the last layer. The last layer is the FC layer. The number of blocks means the number of reverse attention blocks or n iterations of reverse attention block? It is also confusing about the auxiliary probability vector p used in the training and testing stages.

6. During the training, original output of the true label j before softmax is \Omega_j which contains z_i w^{(i,j)}. After reverse attention, the latter term becomes z_i (w^{(i,j)})^2? How exactly does this help with robustness from a math perspective? Correct me if I have some misunderstanding regarding Fig. 3. 

7. Can you explain more about the auxiliary probability vector p. It makes me confused about the white-box and adaptive attacks. For the so-called “white-box” attack in this paper, the robust accuracy against PGD-40 (89%) is even higher than the natural accuracy (85%), which is quite non-trivial and hard to understand. Can you explain?

8. For TRADES-ANCRA, WideResNet performs worse than ResNet18, which is inconsistent with the conclusions of almost all previous experiments. Is it possible that the auxiliary probability vector p brings some kind of a priori information that only applies to ResNet18?

### Questions
See above weaknesses for questions to the authors. Also, please provide checkpoints trained with the proposed method and the codes for evaluating the robustness against various attacks. Since I remain skeptical of some numbers in the table (e.g. PGD-AT-ANCRA, TRADES-ANCRA and MART-ANCRA, i.e. the bottom 3 rows in Table 1).

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper highlights that models trained using existing adversarial training methods do not learn robust feature representation. Following this, the paper presents the criteria for learning robust features (i.e., exclusion and alignment criteria). An adversarial training with Asymmetric Negative Contrast and Reverse Attention (ANCRA) is proposed to learn robust features. The asymmetric negative contrast loss pushes away features of inter-class samples, and the reverse attention pulls close the features of natural and their adversarial samples. The paper considers CIFAR-10, CIFAR-100, and Tiny-Image datasets for experiments.

### Strengths
The paper proposes a novel adversarial training method that can yield robust models. The proposed method explicitly enforces the large inter-class feature distance and small intra-class feature distance.

### Weaknesses
1. The paper fails to demonstrate the robustness of the models trained using the proposed method. The empirical results highlight the presence of obfuscated gradient or gradient masking [1][2]. Models that exhibit obfuscated gradient or gradient masking are pseudo-robustness[1][2].  
- Accuracy on clean samples should be greater than that on adversarial samples. However, for the models trained using the proposed method, the accuracy on clean samples is lesser than the accuracy on adversarial samples (PGD, FGSM, and C&W). Refer to Table 1, CIFAR-10.
 - Iterative attacks should be stronger than non-iterative attacks, i.e., the accuracy of the models on PGD/C&W adversarial samples should be less than that on FGSM adversarial samples. Results indicate iterative attacks (PGD, C&W)  are weaker than non-iterative attacks (FGSM). Refer Table-1  (PGD/C&W vs FGSM) and Table-3 (FGSM vs C&W).
 -  Huge drop in the accuracy of the model for AutoAttack.

2. Missing discussion on existing methods (such as ALP[3] and TRADES[4]) that explicitly enforce feature distance consistency.  Furthermore, the paper fails to answer the following questions: (a) Why does cross-entropy loss fail to achieve exclusion criteria? (b) How does reverse attention achieve alignment between features of natural and their adversarial samples?

[1] Carlini et al. "On Evaluating Adversarial Robustness" arxiv 2019

[2] Athalye et al. "Obfuscated Gradients Give a False Sense of Security" ICML 2018

[3]Kannan et al. "Adversarial logit pairing." arXiv 2018

[4] Zhang et al. "Theoretically principled trade-off between robustness and accuracy" ICML 2019

### Questions
Address weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
