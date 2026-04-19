# Supervised Dimension Contrastive Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
Self-supervised learning has emerged as an effective pre-training strategy for representation learning using large-scale unlabeled data. However, models pre-trained with self-supervised learning still require supervised fine-tuning to achieve optimal task-specific performance. Due to the lack of label utilization, it is difficult to accurately distinguish between positive and hard negative samples. Supervised contrastive learning methods address the limitation by leveraging labels, but they focus on global representations, leading to limited feature diversity and high cross-correlation between representation dimensions. To address these challenges, we propose Supervised Dimension Contrastive Learning, a novel approach that combines supervision with dimension-wise contrastive learning. Inspired by redundancy reduction techniques like Barlow Twins, this approach reduces cross-correlation between embedding dimensions while enhancing class discriminability. The aggregate function combines the embedding dimensions to generate predicted class variables, which are optimized to correlate with their corresponding class labels. Orthogonal regularization is applied to ensure the full utilization of all dimensions by enforcing full-rankness in the aggregate function. We evaluate our method on both in-domain supervised classification tasks and out-of-domain transfer learning tasks, demonstrating its superior performance compared to traditional supervised learning, supervised contrastive learning, and self-supervised learning methods. Our results show that the proposed method effectively reduces inter-dimensional correlation and enhances class discriminability, proving its generalizability across various downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
Motivated by Barlow Twins, this paper proposes a method to constrain the feature diversity in supervised contrastive learning. Experimental results show the effectiveness of the proposed method.

### Strengths
From experimental section, we can see that the proposed method performance well in OOD scenario. This is interesting and validates to some extent that feature diversity does correlate with the generalizability of feature representations.

### Weaknesses
1. The novelty of this paper is really limited. This article does not explain the motivation for CLASS CORRELATION and FULL RANK AGGREGATION to be proposed. At the same time, no intuitive insight or theoretical analysis is given as to why CLASS CORRELATION and FULL RANK AGGREGATION promote supervised comparative learning performance.

2. The related work section is also quite weak. The authors have not done a good job of highlighting the developmental lineage of self-supervised learning, supervised comparative learning, and the connections and differences between the different algorithms. And also did not highlight the advantages and footholds of the method proposed in this paper.

### Questions
1. Why can CLASS CORRELATION encourage the embedding dimensions to be not only augmentation-invariant and decorrelated but also discriminative for the supervised classification task? As we can see, CLASS CORRELATION is only a metric of Correlational Relationship.

2. Please give the theoretical analysis or related references to the statement: To guarantee that all embedding dimensions contribute to class separability, every embedding dimension should play a role in generating the predicted class variables through the aggregate function. It can be achieved by enforcing aggregate function is full-rank.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper enhances Barlow Twins by using the class-correlation loss and orthogonal regularization loss. The class correlation loss uses supervision information, and the orthogonal regularization loss enforces the aggregate function to be full-rank. These additional losses positively affect the overall performance. Through numerical experiments on in-domain classification and out-domain transfer learning tasks, the proposed SupDCL demonstrated superior performance.

### Strengths
- The proposed method demonstrated superior performance on in-domain classification and out-domain transfer learning tasks.

### Weaknesses
- The class correlation and orthogonal regularization losses have not been investigated in Barlow Twins, but such losses are often investigated in other methods or areas.
- According to Table 4, most of the performance gain of the proposed method compared with Barlow Twins comes from supervision.

### Questions
- What is an intuition that the proposed method improved performance on out-domain transfer learning tasks? 
- Based on Figure 1, the proposed method inherits the superior out-domain performance from Barlow Twins. The other regularization or loss terms are likely to obtain improvement similar to the proposed method from Barlow twins.
- Are there any theoretical insights about the performance of the proposed method?
- Can we use the proposed method without the class correlation loss as a self-supervised learning method? If so, does this outperform the existing methods?
- Regarding Table 4, is there any discussion of the performance of Barlow Twins plus the orthogonal regularization loss?
- In Table 3(b), the existing methods increase performance as the projector dimensionality increases, but the proposed method decreases after 128. Is there any interpretation for that?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper introduces Supervised Dimension Contrastive Learning (SupDCL), which combines supervision with dimension-wise contrastive learning. It targets limitations in supervised contrastive learning (high cross-correlation and limited feature diversity) by reducing redundancy in embedding dimensions while enhancing class discriminability.

### Strengths
- The integration of dimension contrastive learning with supervision to enhance feature diversity makes sense. SupDCL’s approach to reducing cross-correlation among dimensions to improve generalization, particularly in transfer learning tasks, is a well-justified.
- The evaluation covers both in-domain and out-domain tasks on different datasets, providing a clear comparison of SupDCL against multiple baselines, including self-supervised and supervised learning methods.
- Extensive ablation studies that analyze each component of SupDCL’s loss function.
- The paper acknowledges the limitations of SupDCL, specifically its reliance on fixed-form class labels. It provides suggestions for handling these limitations through future work.

### Weaknesses
- The paper introduces orthogonal regularization and decorrelation strategies to improve feature diversity, but it lacks a theoretical foundation to show why these particular choices are necessary. While the empirical results show improved performance, a lack of formal justification leaves the effectiveness of these strategies unclear.
- While SupDCL outperforms SupCon in out-doman tasks, it underperforms SupCon in in-doman tasks.
- SupDCL borrows heavily from existing redundancy reduction techniques (Barlow Twins) without significantly advancing the state of contrastive learning. The contribution of introducing supervision to dimension contrastive learning is incremental and could be viewed as a straightforward application of existing techniques.

### Questions
- Why is feature discriminative inportant for supervised contrastive learning? Figure 1 and experiments show that SupDCL underperforms SupCon in in-doman tasks.
- What is $L_{inv}$ and $L_{decorr}$ is Eq (7)? They are not explained.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper argues that representations learned by supervised contrastive learning methods have limited feature diversity and high cross-correlation between dimensions. The authors propose an approach to learn representations that are both augmentation-invariant and dimension-wise decorrelated, while also being discriminative for classification. They achieve this by extending Barlow Twins, a dimension contrastive learning model, to supervised dimension contrastive learning. Specifically, they introduce a discriminativeness loss instead of the cross-entropy loss, consisting of a class correlation loss and an orthogonal regularization loss. Experimental results show that the proposed SupDCL performs well on both in-domain supervised classification tasks and out-of-domain transfer learning tasks.

### Strengths
The idea of this paper is to combine SupCon (supervised contrastive learning) with Barlow Twins (decorrelated features). The specific approach involves introducing a discriminativeness loss to leverage the supervised labels. The motivation and approach are presented very clearly. The authors also provide experiments, analysis, and evaluation metrics to demonstrate the effectiveness of the proposed method.

### Weaknesses
I believe the experiments could be strengthened, and there are some minor points that would enhance the clarity of the paper. These are outlined in more detail in the Questions section.

### Questions
1. This work aims to extend Barlow Twins to a supervised version. A straightforward extension model could be a combination of Barlow Twins and SupCon, where, instead of generating two augmentations $X^A$ and $X^B$ from image $X$, $X^B$ is selected from another image that shares the same class as image $X$. In Table 2, I would like to see the results of this model.

2. Another straightforward extension is to use a cross-entropy loss function instead of the proposed discriminativeness loss function to leverage the supervised labels. While you have compared this model in Table 3, comparing the results in Table 1 and Table 2 would also be worthwhile.

3. In Table 3, you mention the representation space and embedding space; however, in the methods section and Figure 2, there is no reference to the representation space. You have defined the embedding space, $Z$, as an $M-dimensional$ vector, but it is unclear which vector you refer to as the representation vector.

4. Similarly, Section 4.2 does not clarify what the representation $Y$, a $D-dimensional$ vector, refers to.

5. In lines 371-374, when $t_d = 0.1$, Barlow Twins achieves 1320, which seems to be the lowest count rather than the highest as described. Is there a mistake here?

6. Visualization: Could you visualize the embedding space or representation space (with clearer references if possible) of different methods, along with the different dimensions of those representations? This would help to better understand how the proposed method achieves improved decorrelation and discriminative representations.


Minors:
1. It would be better to mark all the best numbers in all tables in bold.

2. In Table 2, add arrows to indicate whether larger or smaller values are better, and mark the best value in bold.

3. Figure 1 can be split into two separate figures, as the third sub-figure is not on the same scale as the other two.

4. In lines 353-358, 645 (SupDCL) vs. 532 (SupCon) is described as outperforming, while 645 (SupDCL) vs. 904 (Barlow Twins) is described as comparable?

### Soundness
3

### Presentation
3

### Contribution
3
