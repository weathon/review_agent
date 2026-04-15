# Make Small Data Great Again: Learning from Partially Annotated Data via Policy Gradient for Multi-Label Classification Tasks

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 6, 3

## Abstract
Traditional supervised learning methods are heavily reliant on human-annotated datasets. However, obtaining comprehensive human annotations proves challenging in numerous tasks, especially multi-label tasks.
Therefore, we investigate the understudied problem of partially annotated multi-label classification. This scenario involves learning from a multi-label dataset where only a subset of positive classes is annotated. This task encounters challenges associated with a scarcity of positive annotations and severe label imbalance.
To overcome these challenges, we propose Partially Annotated reinforcement learning with a Policy Gradient algorithm (PAPG), a framework combining the exploration capabilities of reinforcement learning with the exploitation strengths of supervised learning. By introducing local and global rewards to address class imbalance issues and employing an iterative training strategy equipped with data enhancement, our framework showcases its effectiveness and superiority across diverse classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this submission, authors investigate the understudied problem of partially annotated multi-label classifcation, where  only a subset of positive classes is annotated. To deal with this problem, authors propose a new method named Partially Annotated reinforcement learning with a Policy Gradient algorithm (PAPG) which can overcome the challenges associated with a scarcity of positive annotations and severe label imbalance.

### Strengths
1. The problem, partially annotated multi-label classifcation, is novel.

2. A novel method named PAPG is proposed, which can overcome the challenges associated with a scarcity of positive annotations and severe label imbalance. 

3. Experiments validate the effectiveness of the proposed PAPG method.

### Weaknesses
1. The motivation might be unreasonable. As stated in abstract, "This task encounters challenges associated with a scarcity of positive annotations and severe label imbalance". These challenges are indeed problem to deal with, but in my opinion, the main challenges correspond to that the negative samples are unknown or we do not know which are negative samples among the remaining unlabeled samples. In other words, the main challenges are the same as PU learning.

2. For the two challenges mentioned in this paper, i.e., a scarcity of positive annotations and severe label imbalance, it is unclear how can the proposed PAPG method overcome them.

3. As stated in Introduction, "Consequently, many advanced methods of PU learning (Su et al., 2021; Acharya et al., 2022; Luo et al., 2021) cannot readily adapt to our multi-label settings". In my opinion, if we focus each label one by one, it is a PU learning problem. Thus, the partially annotated multi-label classifcation problem can be solved via binary relevance strategy where the base classifier is trained with any off-the-shelf PU learning methods.

4. For multi-label classification, it is very important to model the correlations among labels. But it is unclear how can the proposed PAPG method model label correlations.

5. The writtting can be greatly improved. For exampe:
(1) The title is inappropriate.
(2) Errors exist in notations (especially the first paragraph in section 3.1).
(3) What does RARL mean in the last sentence of this paper?

### Questions
If author disagree with my comments in Weaknesses, please clarify them in the rebuttal phase.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the problem of partially annotated multi-label classification, where only a subset of positive classes is annotated, leading to imbalanced and challenging learning scenarios. The proposed idea exploits reinforcement learning and designs local rewards assessed by a value network and global rewards assessed by recall functions to mitigate class imbalance issues. The proposed approach is evaluated across various classification tasks and demonstrates its effectiveness in improving upon previous methods.

### Strengths
- Multi-label annotations are challenging. Partial labeling is a great way to alleviate labeling overheads.

- The approach introduced in the paper is straightforward, aligning with intuitive problem-solving strategies. It offers a clear and understandable solution to the challenges at hand.

### Weaknesses
- The literature survey for weakly supervised learning in this paper is incomplete. The survey only covers some settings under multi-class single-label scanrio and misses the weak supervion under multi-label learning, such as missing-label learning. This miss is critical as the major novelty of this paper comes from proposing method in this setting. The comparisions with such methods are also missing, so that the effectiveness of the proposed method is not well supported. The baselines are considerably inadequate, and the results on some famous multi-label classification dataset, such as NUS-WIDE dataset, are missing.

- The proposed method itself is simple and straightforward. Therefore, it would be better to further analyze the design choices of the proposed model to claim the impact of the proposed method. For example, why is the loss function used? Is there any better option for the authors to try for the loss function?

### Questions
It would be better to add several more results in the main paper, and the novelty is debatable.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The proposed methods explore a new approach on partially annotated multi-label classification by using RL-based framework. The approach contains a local rewards assessed by a value network and global rewards assessed by recall functions to guide the learning of policy network. The experiments on binary image classification, multi-label image classification, and document-level relation extraction show the effectiveness of the proposed method.

### Strengths
+ The proposed methods explore a pretty interesting direction (RL based design) on partially annotated multi-label classification.

+ the paper is well-organized and have clear figures and demonstrations.

+ The paper is technically sound and provide necessary analysis.

+ The author test the algorithm on multiple tasks.

### Weaknesses
- usually the algorithms contain the DRL methods will have a relatively higher variance on the performance. May I know what is the variance of your model's F1 and mAP in multi-label image classification?

- How many time did you repeat your experiments?

### Questions
- How large is the computation overhead in algorithm compared with the standard supervised learning?

- What will be the obstacles if you run this algorithm on a very large image dataset compared to the standard supervised learning? For example, 9 million partially annotated training images (OpenImages V6)? 

- How many time did you repeat your experiments? May I know what is the variance of your model's F1 and mAP in multi-label image classification?

- It would be better if you can show a figure reflecting the training process, e.g., x-axis: training steps, y-axis: accumulated reward or mAP.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes the PAPG framework to deal with the partially annotated multi-label classification.
It first discussed the partially annotated learning in the multi-label classification tasks, which bears great significance.
And then proposed the local and global rewards which is the main contribution of this paper.
With an iterative training strategy, PAPG gains good results.

### Strengths
1. This paper touches on the partially annotated multi-label classification, which is an area of great significance.
2. This paper has some experiments to support its conclusions.
3. This paper is easy to follow, especially the descriptions of the modelling process of RL in Sec. 3.2, which is very clear.
4. The proposed PAPG gains promising results on multiple benchmarks.

### Weaknesses
1. Methods do not support assertions.
This paper claims the local reward is the immediate reward in Sec. 1. and it provides immediate value estimation of each action in Sec. 3.2
However,  these immediate rewards are summed up in dimension $C$ to get the final reward according to Eq. 3.
It does not have immediate properties.
It has the same frequency as the global reward.
If this problem is not well explained, then this problem will be fatal.
I see no reason to use RL for this task.

2. Poor formula expression. It is shown in "Question" part of my review.

### Questions
1. Whether $\theta$ or $\theta^*$ is used in getting $\overline{\mathcal{y}}$ in line13 of Algorithm 1.
2. I am confused with this formula, $\hat{y}^c_i=V_\lambda(x_i)_c=1$. Please explain the process and meaning of it.
3. I think the dataset for the training value network is $(X, Y, \overline{Y})$, not $(X, Y \cup  \overline{Y})$, according to Eq. 4.
It uses $ Y$ and $\overline{Y})$ , not $Y \cup  \overline{Y}$.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
