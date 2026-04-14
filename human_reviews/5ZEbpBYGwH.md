# COPER: Correlation-based Permutations for Multi-View Clustering

- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 5

## Abstract
Combining data from different sources can improve data analysis tasks such as clustering. However, most of the current multi-view clustering methods are limited to specific domains or rely on a suboptimal and computationally intensive two-stage process of representation learning and clustering. We propose an end-to-end deep learning-based multi-view clustering framework for general data types (such as images and tables). Our approach involves generating meaningful fused representations using a novel permutation-based canonical correlation objective. We provide a theoretical analysis showing how the learned embeddings approximate those obtained by supervised linear discriminant analysis (LDA). Cluster assignments are learned by identifying consistent pseudo-labels across multiple views. Additionally, we establish a theoretical bound on the error caused by incorrect pseudo-labels in the unsupervised representations compared to LDA. Extensive experiments on ten multi-view clustering benchmark datasets provide empirical evidence for the effectiveness of the proposed model.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed an end-to-end MVC method that leveraged CCA-based correlation maximization and self-supervised pseudo-labels to learn multi-view representations and clusters jointly. In the proposed method, the key components are the Sample Selection and Label Refinement-Agreement of multi-view pseudo-labelling, which follows some technologies of semi-supervised learning and yields a simple unsupervised MVC architecture. Then, the authors present experimental and theoretical results to support the effectiveness of their method.

### Strengths
1. This paper presents a deep MVC method which has the advantages of simpleness and end-to-end. The method leverages a cross-entropy loss and a CCA-like maximization loss to train the deep model. It transfers the two-step training schedule in previous methods into step-by-step one, among which it conducts sample selection by high confidence and label correction by multi-view agreement.

2. The paper is well-written and easy to follow, which introduces theoretical proofs and visualization to support its method.

3. The illustration of cluster permutations is interesting, and it can enhance the embeddings learned by CCA (verified by ablation study).

### Weaknesses
I have the following concern and hope they are useful for improving this manuscript:

As for unsupervised clustering task, the robust model is needed when it processes different datasets in practical scenarios. However, we can observe that the proposed method is sensitive to model architecture settings (Table 7&9). For different datasets, the proposed method has different settings of model architecture and batch size. Moreover, in D.4 GRADUAL TRAINING, the training epochs in each step are determined by human. Since we have no labelled data to tune the model settings in practical applications, the proposed might have limited practical application value.

### Questions
1. Please see above weakness. It is encouraged to use a uniform model architecture to test the clustering performance of the proposed method on different datasets, for a fair comparison and availability.

2. It is encouraged to compare some latest self-supervised deep MVC approaches, e.g., On the effects of self-supervision and contrastive alignment in deep multi-view clustering [CVPR 2023], Investigating and Mitigating the Side Effects of Noisy Views for Self-Supervised Clustering Algorithms in Practical Multi-View Scenarios [CVPR 2024]...

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
In literature, most of the current multi-view clustering methods are limited to specific domains or rely on a sub-optimal and computationally intensive two-stage process of representation learning and clustering. To address this issue, the authors propose an end-to-end deep learning-based multi-view clustering framework which is validated to be effective in experiments.

### Strengths
1. The theory of LDA is analyzed. 
2. The proposed method is validated to be effective in experiments.

### Weaknesses
1. The authors say that most of existing multi-view clustering methods are composed of two-stage process of representation learning and clustering, therefore they propose an end-to-end method. However, the authors also claim that a few end-to-end methods are proposed in literature. So, the motivation should be further clarified. 

2. The paper [1] is an classical end-to-end multi-view clustering method and should be compared or discussed.

3. The parameter study should be included.

[1] Liu J, et al. One-pass Multi-view Clustering for Large-scale Data, ICCV 2021.

### Questions
Please see Weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a deep learning model for multi-view clustering framework, namely, COPER. The proposed model integrates clustering and representation tasks into an end-to-end framework, eliminating the need for a separate clustering step. Extensive experimental evaluation across various benchmark datasets validates the efficiency of the proposed algorithm.

### Strengths
1.	This paper is well organized, and the proposed methodology is enlightening.
2.	The motivation behind the paper is clear, and the theoretical analysis is complete.
3.	The comparison experiments are comprehensive, encompassing datasets of varying sizes and multiple types of baseline methods.

### Weaknesses
1.	Unlike general methods, the proposed approach generates pseudo-labels for each view to enable self-supervised learning. However, in multi-view clustering, aligning the labels across views can pose challenges that may impact subsequent self-supervised learning.
2.	Although the paper includes theoretical analysis, the proposed method offers limited innovation. The correlation maximization loss has already been proposed, and generating pseudo-labels by estimating the probability matrix is also a common approach.

### Questions
As mentioned above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The proposed approach involves generating meaningful fused representations using a novel permutation-based canonical correlation objective. Cluster assignments are learned by identifying consistent pseudo-labels across multiple views.

### Strengths
The proposed approach involves a novel permutation-based canonical correlation objective. Simulanteously, the authors provide a theoretical analysis showing how the learned embeddings approximate those obtained by supervised linear discriminant analysis (LDA).

### Weaknesses
1）In Line 41, multi-view clustering holds immense potential in various applications, however, the methods mentioned are not updated to recent literature. Please update these references to ensure your work is current.

2）The selected comparison methods are not enough. It is recommended to add some comparison methods, otherwise this may have a negative impact on the reliability of the experimental results. 

3）Some of the selected comparison methods are for incomplete multi-view data, and some are for noise correspondence. These methods have special properties and are not recommended as comparison methods.

4）Considering that the proposed method is derived from the CCA objective, it is recommended to classify the compared methods into CCA-derived methods and other non-CCA-derived methods. This can directly demonstrate the effectiveness of the new elements introduced by the proposed method compared to previous similar frameworks and its competitiveness compared to other MVC paradigms.

5) Check for all possible errors in the statement, e.g. the missing serial number for Figure in Line 244,“using using within-cluster permutations”in Line 292.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
