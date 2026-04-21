# Referring Expression Matters: Multi-referring Feature Aggregation for Referring Video Object Segmentation

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 1, 3

## Abstract
Referring Video Object Segmentation aims to segment object instances referred to by natural language referring expressions in a video sequence. This interaction style is quite simple and flexible, being capable of producing high quality segmentation masks. However, the referring expression variation occurs due to the randomness of expressions provided by users, making the existing state-of-the-art models still face the problem of wrongly identifying the referred object. To address this issue, we present a novel referring video object segmentation network fed with multiple referring expressions. Specifically, a simple but effective neural expression generation module is proposed to map the features of multiple referring expressions to complementary features with less redundancy. This interaction of multiple referring expressions not only is beneficial to identify the referred object but also speeds up the training convergence. We make evaluations of the proposed method on the popular referring video object segmentation datasets, and experimental results demonstrate that our method outperforms the state-of-the-arts by a significant margin in terms of segmentation quality and achieves considerable gains in terms of training convergence speed. Our code and pre-trained models will be available.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper focuses the task of Referring Video Object Segmentation and introduces to integrate multiple referring expressions to boost performance. A neural expression generation module is proposed to create complementary features from these expressions, which not only improves object identification accuracy but also accelerates training convergence. Experimental results on popular RVOS datasets are presented.

### Strengths
(1)	The paper explores the effect of multiple referring expressions for RVOS, which is interesting.

(2)	This paper is well-written and easy to follow.

### Weaknesses
(1) Although the authors present an interesting motivation, suggesting that adjusting referring expressions could enhance segmentation performance, the method proposed does not fully align with this motivation. The reviewer, after going through the introduction, expect to find how the unclear parts within referring expressions are identified and improved. However, the authors merely concatenate multiple referring expressions.

(2) The paper's contribution mainly involves adding an MLP to ReferFormer to merge multiple referring expressions. However, this incremental addition lacks further in-depth consideration, i.e., what kind of scenarios need multiple inputs, how the extent of overlap and divergence between referring expressions affects final performance. Consequently, the contribution of the paper is limited.

(3) The experimental comparisons are unfair. While the proposed method uses multiple referring expressions as input, the compared methods utilize only one expression. To truly demonstrate the impact of the integration of MRE, a more comprehensive comparison should involve merging results from different expressions in other methods. This would effectively showcase the performance gains derived from exploring relationships within referring expressions.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
A Referring Video Object Segmentation method is proposed. However, the motivation is not clear. The details of most of the methods are not explained.

### Strengths
The picture of the model architecture of the proposed method is clear.

### Weaknesses
The writing is too bad. The details of the Multi-modal Fusion are not explained clearly. No referenced paper is mentioned in Deformable Transformer and Instance Sequence Segmentation. It's quite hard to understand the paper.

### Questions
How to do Multi-modal Fusion? What's the structure of the Deformable Transformer? What is Cross-Modal Feature Pyramid Network (CM-FPN) ?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a referring video object segmentation method via a multi-referring feature aggregation mechanism. This mechanism can effectively obtain complementary features with less redundancy, which is not only helpful in identifying the referred object, but also speeds up the training convergence. Experimental results show the effectiveness and superiority of the proposed method.

### Strengths
+ The multiple referring expressions can generate a complete and concise linguistic feature, experimental results also show the effectiveness of the proposed strategy.
+ The proposed method can achieve better training convergence. 
+ The proposed method achieves the new SOTA and outperforms the second-best by a large margin

### Weaknesses
- The novelty of the proposed method is somewhat limited. The main contribution is the neural expression generation via multiple-referring expressions. It seems that this aggregation strategy is simple and lacks insights. 
- The authors declare that they proposed different sampling strategies in cross-modal attention for pre-training and fine-tuning to boost the model performance. However, the illustration of this sampling strategy is unclear, and the differences with existing sampling strategies are also unclear. Also, there are no experimental results to support this assertion. 
- In Eq.3, the authors used the concat operation but in Table 2(c), the proposed NEG is different MRE Cocat, so the reason is unclear.
- The authors do not show the training convergence in the pre-training strategy, So, it is hard to assert the proposed method achieves faster convergence only by verifying it in the fine-tuning stage. 
 - I think the comparison is somewhat unfair. The batch size is different. It mainly influences the training convergence and even the performance.

### Questions
Please seeing the weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
