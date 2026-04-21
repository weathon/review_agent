# Multiple Positive Views in Self-Supervised Learning

- Avg Score: 5.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 5

## Abstract
Contrastive learning is a potent technique for self-supervised learning (SSL) that maintains invariance between two views. Advancements such as the ''core view'' (Tian et al., 2020a) or multi-cropping have harnessed insights from multiple views, culminating in the latest state-of-the-art performance. However, the complexities of multiview learning remain partially unexplored. In this paper, we introduce a ''plug-and-play'' multi-positive-views ($\geq3$) learning approach seamlessly integrated with existing two-view SSL architectures. Theoretical and empirical analyses underscore the feasibility of enhancing traditional SSL models by incorporating multiple positive views. By mitigating the intrinsic biases towards sufficiency and minimality in the embeddings, our method achieves improvements in average accuracy (2% on CIFAR-10 and 26% on Tiny ImageNet) and significant speed-ups (3--4 times) across five datasets and eight architectures. Our research reveals and improves the double-edged nature of conventional assumptions tied to two-view suitability, thereby paving the way for future investigations in multiview SSL.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work analyzes the limitations of contrastive learning with two-views and extend it to multiple-views through the lens of information theory. They provide theory to show their objective is a lower bound of the information bottleneck. Finally, their experiments show improved performance and speed-up.

### Strengths
The problem is underexplored and the novelty of the paper is strong.

The theoretical analysis is rigorous with clear definitions.

### Weaknesses
The presentation can be improved.

In method section,
1) The assumptions of the proposed method are not clearly stated. 
2) What are the failure cases for the method?
3) What is an intuitive example of view-invisible bias? Could you please clarify "a mutually exclusive state of Z owing to the invariant nature of label and view information in optimization"? What are the "certain approaches" that " assume sharable task information ..."?  Are there experiments to support eq 10? Adding concrete examples could help to better digest the theorems.

In experiment section,

4) Could you clarify why "we adopted unselected data augmentations that slightly deviate from a Sweet Spot"? Does it affect the improvements over baselines?
5) Is the linear accuracy of the main experiment, table 2&3, omitted?
6) Is the method scalable to medium size datasets such as ImageNet?
 
Ablation study is not a key contributions to the SSL.

For three variables MI is defined as I(x; y, z) = I(y; z) - I(y; z | x). There are typos in eq 12.

### Questions
This is an interesting paper in multiple aspects, however the presentation can be significantly improved.

### Soundness
3 good

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers using multiple positive views in training the SSL model. The paper derives certain theoretical aspects and advantages of introducing the multi view self-supervised learning method. Empirical evidence demonstrates the superiority of the proposed method.

### Strengths
S1. The paper explores the integration of multiple positive views during the training of the SSL model. 

S2. It establishes specific theoretical implications and benefits associated with the implementation of the multi-view self-supervised learning approach. 

S3. Empirical findings affirm the effectiveness of the proposed methodology.

### Weaknesses
W1. The paper lacks novelty. There are many works introducing multiple views in the SSL pre-training, including DINO, SwAV  and so on, The paper does not compare with these well known SOTA methods. 

W2: It is unclear what is the explicit loss function of the proposed method (although it seems Eq. (11) is the loss), and how it hears advantages over SOTA or how it distinguishes in motivation between the existing multiview method such as SwAV (clustering based method with multi-views).

W3: It is unclear if there are fairness issues during the training (empirical evidence), i.e., does the proposed multi-view contrastive learning simply benefits from more "effective epochs" because of its multi-view training (more data in each batch) in comparison to other SOTA methods? 

Please help clarify the above concerns. 

[A] Mathilde Caron et al., SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. 

[B] Mathilde Caron et al., . Emerging Properties in Self-Supervised Vision Transformers

### Questions
Please see the above weakness for the questions to be addressed. Please correct me during rebuttal, if there is any misunderstanding.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a "plug-and-play" approach to multi-positive-views learning, seamlessly integrating with existing two-view self-supervised learning (SSL) architectures. The authors challenge traditional assumptions about multiview learning and explore its complexities. The proposed method incorporates multiple positive views to enhance traditional SSL models, improving accuracy and speed across various benchmarks and SSL architectures.

### Strengths
- This paper explores the complexities of multi-positive-views learning and provides an alternative way to understand multiview learning.
- Extensive experiments support the effectiveness of multiview learning.
- The paper is well-organized and easy to follow.

### Weaknesses
- Although the proposed strategy (Eq. 11) is an alternative way for multiple positive view contrastive learning, its novelty is limited.
- Extensive experiments are conducted. However, I can hardly find insights different from previous multiple positive view contrastive learning methods.
- In Table 2, the training epochs for each setting are not clear. If all methods share the same training epoch, the comparison is not fair since 4-view models observe more data than 2-view models.

### Questions
- Could you please highlight unique insights different from existing multi-positive view methods?
- In Table 2, do all the methods share the same training epochs? If yes, could you please conduct additional 2-view experiments with double training epochs for fairness?
- In Figure 7, could you please explain why GPU usage decreases as the number of views increases?

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
2 fair
