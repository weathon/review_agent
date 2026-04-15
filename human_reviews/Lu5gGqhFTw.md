# RelationMatch: Matching In-batch Relationships for Semi-supervised Learning

- Decision: Reject
- Scores: 8, 3, 5, 5

## Abstract
Semi-supervised learning has gained prominence for its ability to utilize limited labeled data alongside abundant unlabeled data. However, prevailing algorithms often neglect the relationships among data points within a batch, focusing instead on augmentations from identical sources. This paper presents RelationMatch, an innovative semi-supervised learning framework that capitalizes on these relationships through a novel Matrix Cross-Entropy (MCE) loss function. We rigorously derive MCE from both matrix analysis and information geometry perspectives. Our extensive empirical evaluations, including a 15.21% accuracy improvement over FlexMatch on the STL-10 dataset, demonstrate that RelationMatch consistently outperforms existing state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes the matrix cross-entropy (MCE) loss for semi-supervised learning (SSL). In addition to matching the output of the strong augmentation with the pseudo-label of the weak augmentation, they also match the pairwise product of the output of the strong augmentation with that of the weak augmentation. Extensive theoretical analysis on the MCE loss reveals the nice theoretical property of the proposed approach. Experiments on benchmark datasets validate the effectiveness of the proposal.

### Strengths
- The proposed MCE loss is novel and interesting. As far as I know, this is the first time it has been applied to the SSL literature. 
- The theoretical analysis is very comprehensive and sound. The nice theoretical property can promote further investigation of MCE in the community.
- The empirical performance is very strong since the compared methods are very recent and strong methods in SSL.

### Weaknesses
My small comment concerns the details of the writing, especially the notations. There may be some typos or unclear statements. 

- In section 2.1, it should read $\log q_1$ instead of $\log q_i$. 
- In section 2.1, what's Eq.(2.1)?
- In Eq. 4, is $\tilde{Y}_s$ the model output of weak augmentations? In a line above it is written as $\tilde{Y}$. If they mean the same thing, the notation should be the same. 
- In Definition 4.2, it should be $n=0$ instead of $i=0$. 

The author should check the notation carefully.

### Questions
Since MCE can be simplified with a $l_2$-normalized matrix, what is the loss function used in the experiments? Is it still equation (1) with a non-normalized matrix?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper investigates the challenges in semi-supervised learning. The authors highlight that prior research has often overlooked the interconnections between data points in a batch. To address this gap, they introduce RelationMatch, an approach designed to harness the consistency of relationships within a batch of unlabeled data.

### Strengths
1. The paper is well written and easy to understand. 
2. The authors present the derivation of the proposed MCE Loss through the lenses of matrix analysis and information geometry, showcasing its advantageous characteristics such as convexity, boundedness from below, and optimizable properties.

### Weaknesses
1. My primary concern pertains to the paper's novelty. SimMatch[1] has previously addressed the relationship between data points by applying consistency regularization at both the semantic and instance levels, promoting identical class predictions and maintaining similarity relations with other instances for different augmentations of the same instance. A detailed discussion and comparison between SimMatch and RelationMatch are essential to elucidate the distinct contributions of the latter.

2. The benchmark comparison appears outdated. The most recent method evaluated in the paper is from 2021, and although the authors mention some methods from 2022 and 2023, such as FreeMatch, MaxMatch, and NP-Match, these have not been included in the experimental comparisons. When compared with the latest methods, RelationMatch does not seem to meet the state-of-the-art standard.

3. The experimental scope of the paper is limited to toy datasets. To bolster the findings, it is recommended to extend the experiments to more complex, real-world datasets, such as ImageNet.


[1] Zheng, Mingkai, et al. "Simmatch: Semi-supervised learning with similarity matching." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces the consistency between each pair of weak and strong augmentation within a batch in semi-supervised learning.

### Strengths
1. the paper proposes a novel idea, which consider in-batch relationship in SSL.
2. The paper proposes matrix cross-entropy, which has a theoretical foundation and interpretations.
3. Good writing, easy to follow, I appreciate the warm-up example, which is helpful for understanding.

### Weaknesses
1. Figure 1 can be improved. There are too many lines, which are confusing.
2. Large dataset experiments are missing, e.g., ImageNet 
3. Ablation studies on $\mu$ and $\gamma$ are missing.
4. Formulations and notations are not clear. What's the definition of $Y_s$ and $X_s$ in eq(4)? 
5. How MCE connect with Relation in the introduction?

### Questions
1. Would you consider a relation (strongeaug dog, strongaug cat) > relation(weakaug dog, weakaug cat)? Intuitively, this relation more close to nature's rule.
2. Is there an intuitive explanation of matrix cross-extropy? 
3. MCE(P, Q) = tr(−P log Q + Q). For matrix cross-entropy, why +Q?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the problem of semi-supervised learning, which is a common and interesting area. The author proposes RelationMatch, an innovative semi-supervised learning framework that capitalizes on these relationships through a novel Matrix Cross-Entropy (MCE) loss function. Extensive empirical evaluations, including a 15.21% accuracy improvement over FlexMatch on the STL-10 dataset, have demonstrated that RelationMatch consistently outperforms existing state-of-the-art methods.corruptions.

### Strengths
1. This paper is well-written, well-organized, and easy to follow.
2. The paper addresses a novel and important problem, i.e., the relationships among data points within a batch, which has not been well-studied in the literature. 
3. This method can be easily incorporated with other works

### Weaknesses
1. The experiment appears somewhat insufficient, as only two experiments were conducted in the main text, and they were tested on just two to three datasets. Additionally, I am curious as to why the STL-10 dataset was omitted from Table 1.
2. Based on the results presented in Table 1, the displayed accuracy results show limited differentiation. The matrix cross-entropy outperformed by a margin of less than 0.3%. This could potentially be attributed to randomization and perturbations.
3. Potential failure modes or limitations not discussed.

### Questions
The primary questions for the rebuttal primarily arise from the "weaknesses" section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
