# Explainable $ K $-means Neural Networks for Multi-view Clustering

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Despite multi-view clustering has achieved great progress in past decades, it is still a challenge to balance the effectiveness, efficiency, completeness and consistency of nonlinearly separable clustering for the data from different views. To address this challenge, we show that multi-view clustering can be regarded as a three-level optimization problem. To be specific, we divide the multi-view clustering into three sub-problems based on $ K $-means or kernel $ K $-means, i.e., linear clustering on the original multi-view dataset, nonlinear clustering on the set of obtained linear clusters and multi-view clustering by integrating partition matrices from different views obtained by linear and nonlinear clustering based on reconstruction. We propose Explainable $ K $-means Neural Networks (EKNN) and present how to unify these three sub-problems into a framework based on EKNN. It is able to simultaneously consider the effectiveness, efficiency, completeness and consistency for the nonlinearly multi-view clustering and can be optimized by an iterative algorithm. EKNN is explainable since the effect of each layer is known. To the best of our knowledge, this is the first attempt to balance the effectiveness, efficiency, completeness and consistency by dividing the multi-view clustering into three different sub-problems. Extensive experimental results demonstrate the effectiveness and efficiency of EKNN compared with other methods for multi-view clustering on different datasets in terms of different metrics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work gives a novel insight to the multi-view clustering community, i.e., the multi-view clustering can be regarded as a three-level optimization problem. The authors propose to divide the multi-view clustering into three sub-problems: the linear clustering on the set of original data points based on $ K $-means, the nonlinear clustering of each view on the set of linear clusters based on kernel $ K $-means, and multi-view clustering by integrating information from different views for the set of partitions in terms of $ K $-means.

### Strengths
The authors propose to divide the multi-view clustering into three sub-problems: the linear clustering on the set of original data points based on $ K $-means, the nonlinear clustering of each view on the set of linear clusters based on kernel $ K $-means, and multi-view clustering by integrating information from different views for the set of partitions in terms of $ K $-means.

### Weaknesses
1. Section 3 describes the proposed method. It is not easy to follow, partly because of the necessary notations for multi-view data, but also because the motivation of the approaches is not well explained. For example, why one combines Wv and Uˆv together in Eq. (10), where Wv comes from the linear clustering (7), while Uˆv comes from the nonlinear clustering (9).
2. Following the above point, the motivation of the proposed approach also needs further justification. For example, Eq. (8) applies K-means on the subspace representation. Why one needs to apply K-means on the subspace representation instead of directly applying subspace clustering techniques?
3. Also, Eq. (9) then applies kernel K-means on the K-means coefficients of the above approach. What is the motivation behind this step?
4. Why do we need to further cluster the coefficients?

### Questions
Following the above point, the motivation of the proposed approach also needs further justification. For example, Eq. (8) applies K-means on the subspace representation. Why one needs to apply K-means on the subspace representation instead of directly applying subspace clustering techniques?

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
5

### Summary
This paper shows that multi-view clustering can be regarded as a three-level optimization problem. The authors flexibly divide the multi-view clustering into three sub-problems based on $ K $-means or kernel $ K $-means, i.e., the linear clustering on the set of original data points, the nonlinear clustering of each view on the set of linear clusters, and multi-view clustering by incorporating partition matrices from different views obtained by linear and nonlinear clustering. The proposed EKNN unifies three different sub-problems into a framework and the authors use the iterative algorithm to optimize the formulated problem. EKNN is able to effectively and rapidly cluster data points from different views. The authors also extend EKNN for multi-view clustering to the case of multi-view subspace learning, which is able to learn a satisfied latent representation shared by different views.

### Strengths
The authors flexibly divide the multi-view clustering into three sub-problems based on $ K $-means or kernel $ K $-means, i.e., the linear clustering on the set of original data points, the nonlinear clustering of each view on the set of linear clusters, and multi-view clustering by incorporating partition matrices from different views obtained by linear and nonlinear clustering. The proposed EKNN unifies three different sub-problems into a framework and the authors use the iterative algorithm to optimize the formulated problem. EKNN is able to effectively and rapidly cluster data points from different views.

### Weaknesses
1) Overall, the claim of balancing the effectiveness, efficiency, completeness, and consistency of the approach is not fully supported by the current presentation. Likewise for explainability. The effect of each layer in a classical deep neural network is also explainable, i.e., doing convolution plus nonlinear function. The composition of many layers makes the entire network lake of explainability. The proposed approach of combing several clustering steps seems to suffer from the same issue. 

2) There is no constraint on the coefficient $\Theta_v$ in Eq. (8). So there exists a trivial solution of $\Theta_v = I$. 

3) What is the kernel used in Eq. (9)? 

4) There are many typos can be corrected in the whole paper.

### Questions
Overall, the claim of balancing the effectiveness, efficiency, completeness, and consistency of the approach is not fully supported by the current presentation. Likewise for explainability. The effect of each layer in a classical deep neural network is also explainable, i.e., doing convolution plus nonlinear function. The composition of many layers makes the entire network lake of explainability. The proposed approach of combing several clustering steps seems to suffer from the same issue.

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
This paper proposes Explainable $ K $-means Neural Networks (EKNN) and present how to unify these three sub-problems into a framework based on EKNN. It is able to simultaneously consider the effectiveness, efficiency, completeness and consistency for the nonlinearly multi-view clustering and can be optimized by an iterative algorithm. EKNN is explainable since the effect of each layer is known. To the best of our knowledge, this is the first attempt to balance the effectiveness, efficiency, completeness and consistency by dividing the multi-view clustering into three different sub-problems.

### Strengths
EKNN is explainable since the effect of each layer is known. To the best of our knowledge, this is the first attempt to balance the effectiveness, efficiency, completeness and consistency by dividing the multi-view clustering into three different sub-problems.

### Weaknesses
1. One of the main motivations is on efficiency, yet how the proposed method could advance in this regard can be further illustrated.

2. The reason why existing methods in improving efficiency do not suffice is not very clear. In particular, every two algorithms differs in some sense, thus integrating one into the other would increase complexity.

3. Moreover, to compute a n × n matrix Θ directly differs from the motivation that some clustering methods need to compute pairwise similarity between points.

4. Lastly, Eq. (7) and Eq. (8) are very different: Eq. (7) is essentially k-means on the v-th view Xv of data, whereas (8) is doing k-means on the self-representation matrix. Why would the self representation Θ give clusters V that is non-linearly separable as expected ?

### Questions
One of the main motivations is on efficiency, yet how the proposed method could advance in this regard can be further illustrated.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Explainable $ K $-means Neural Networks (EKNN) based on $K$-means for multi-view clustering and use the iterative method in $K$-means to optimize the networks. Besides, it is explainable since the effect of each layer in EKNN is knowable, i.e., the layer of kernel $K$-means is adopted to obtain nonlinear clusters. We can also observe that $K$-means and kernel $K$-means are special cases of EKNN. The authors also extend EKNN for multi-view clustering to the case of multi-view subspace learning, which is able to learn a desired latent representation shared by different views. The shared subspace representation can be obtained by the latent representation based on self-expressiveness, which is used to achieve the final results by the existing clustering algorithms, i.e., spectral clustering algorithm.

### Strengths
1. This paper proposes Explainable $K$-means Neural Networks (EKNN) based on $K$-means for multi-view clustering and use the iterative method in $K$-means to optimize the networks. 
2. Besides, it is explainable since the effect of each layer in EKNN is knowable, i.e., the layer of kernel $K$-means is adopted to obtain nonlinear clusters. We can also observe that $ K $-means and kernel $ K $-means are special cases of EKNN.

### Weaknesses
1. Note that in the case when the non-linear feature map from the original data space to the high-dimensional space is explicit, one only need to compute n × k  point cluster distance as opposed to n × n kernel matrix, thus the proposed method is not useful in this case. Hence, an experiment where the features can not be computed and one need to compute the kernel matrix instead is needed. However, the authors did not specify which kernel is used for experiments.
2. The clarity of this paper can be improved. The title and the abstract mentioned this method as Explainable $K$-means Neural Networks. However, there is no usage of neural network in the method.
3. The authors mention in the paper that “the quality of nonlinear multi-view clustering results is guaranteed”. Therefore, more theory can be provided to support this claim.
4. The authors are expected to care for the typo error and check the whole paper to avoid such issues in the whole paper.

### Questions
The clarity of this paper can be improved. The title and the abstract mentioned this method as Explainable K-means Neural Networks. However, there is no usage of neural network in the method.

### Soundness
3

### Presentation
3

### Contribution
3
