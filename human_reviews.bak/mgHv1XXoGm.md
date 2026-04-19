# Regularized Conditional Optimal Transport for Feature Learning and Generalization Bounds

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 5

## Abstract
This paper develops the regularized conditional optimal transport for feature learning in an embedding space.  Instead of using joint distributions of data, we introduce conditional distributions to some reference conditional distributions in terms of the Kullback-Leibler (KL) divergence. Using conditional distributions provides the flexibility in controlling the transferring range of given data points. When the alternating optimization technique is employed to solve our model,  it is interesting to find that conditional and marginal distributions have closed-form solutions. Moreover, the use of conditional distributions facilitates the derivation of the generalization bound of our model via the Rademacher complexity, which characterizes its convergence speed in terms of the number of samples. By optimizing the anchors (centroids)  defined in the model, we also employ optimal transport and  autoencoders to explore an embedding space of samples in the clustering problem. In the experimental part, we demonstrate that the proposed model achieves promising performance on some learning tasks. Moreover, we construct a conditional Wasserstein classifier to classify set-valued objects.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a regularized conditional optimal transport model for feature learning in an embedding space, leveraging conditional distributions with Kullback-Leibler (KL) divergence for flexible data transfer. The model uses alternating optimization to derive closed-form solutions for conditional and marginal distributions, and the generalization bound is established through Rademacher complexity. Experiments show the model’s strong performance in clustering tasks, and a conditional Wasserstein classifier is introduced for classifying set-valued objects.

### Strengths
1. Leverage conditional distributions with Kullback-Leibler (KL) divergence for flexible data transfer.
2. Divide variables in optimization into several groups to derive closed-form solutions for probability variables.
3. Present a theoretical analysis of the generalization bound through Rademacher complexity.

### Weaknesses
See Questions.

### Questions
My primary concern lies in the motivation and the problem settings.

1. In Figure 1, the authors explore the relationship between two spaces, $X$ and $Y$, where a element from $X$ are projected onto a set of elements in $Y$. What is the rationale for measuring the Wasserstein distance between the Dirac distribution $\delta_{x_1}$ and the conditional distribution of the projection y, i.e. $p(y|x_1)$? Additionally, in Equation (3), the authors compute the Wasserstein distance between the projections of two encoders over spaces $X$ and $Y$ separately, which differs from the setting in Figure 1.  Could the authors clarify the reasoning behind this change in methodology? From my understanding, the Wasserstein distance generally measures the discrepancy between two probability measures that are defined on the same metric space. Hence, I am unclear about what the author intends when referring to two spaces $X$ and $Y$.

2. In Section 2.2, PROBLEM FORMULATION, I believe that considering two spaces is a new setting, as the authors do not mention any relevant works. Could the authors provide more details or practical examples for this setting and clarify the relationship between spaces $X$ and $Y$? Could the authors provide more details on how to obtain $Y$ regrading the statement "In experiments, $x_1,\cdots,x_n$" consist of training sets and $y_{j|i}$...." in lines 383-384?

3. The authors use KL divergence instead of the information entropy to introduce prior knowledge.  Could the authors provide practical examples of when and why prior knowledge is necessary?

4. In lines 289-290, the statement "Here, we let $f_\theta=g_\phi$ and......." suggests that the authors consider the two autoencoders to be identical in their theoretical analysis. I am not sure if this assumption is valid. Are there any references or motivations for this choice?

5. There is a lack of relevant works for feature learning methods and generalization bounds.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the representation/feature learning under the optimal transport (OT) theory and graph embedding framework. Specifically, this work tries to measure the point-cluster distance via Wasserstein distance and further employ it to facilitate the representation learner. Equivalently, the learned Wasserstein distance can be taken as a graph embedding with transport plan as edge weights. Considering different constructions of clusters, i.e., neighbor set or class prototypes, the proposed methods can be applied to supervised learning and unsupervised learning tasks, e.g., classification and clustering. Generalization bound is derived to show the representation learning objective convergences with the sample-size.

### Strengths
1. The presentation is good, where the methodological parts are clearly stated and the proposed method is technically sound.

2. Theoretical bounds are provided to ensure the convergences of estimations under different learning scenarios. 

3. The superior experiment results on classification and clustering tasks.

### Weaknesses
1. The significance of proposed method is not well-explained; especially, the motivation and advantages for adopting the Wasserstein distance for learning discriminability should be further clarified.

2. The merit from methodological aspect seems to be incremental; compared with Wasserstein discriminant analysis (WDA) that extends linear discriminant analysis, this work can be roughly taken as an extension of marginal fisher analysis (MFA).

3. The empirical validation could be improved, where SOTA representation/metric learning methods for discriminability and complex evaluation datasets/tasks are necessary to verify the significance of proposed method.

### Questions
1. The key of the proposed method is to characterize the conditional discrepancy via the Wasserstein metric. However, the existing works, e.g., WDA-based methods or kernel-based discriminant methods, indeed also characterize the conditional distributions. Specifically, the class-wise means in WDA/Kernel-MFA characterize the class-conditional distribution $p(x|y)$, where kernel statistics ensure stronger property, e.g., embedding property that equivalently characterizes the conditional distributions. Therefore, it would be appreciated to clarify the essential significance of the proposed method.

2. Following Q1, since there are works that also focus on the conditional distributions, the major motivation should be further clarified. As the basic problem is to learn discriminative representations/features, it seems that the introduction does not present new perspectives compared with existing literature.

3. To characterize the discriminability via conditional distributions, there are also considerable efforts from the views of conditional discrepancy and conditional independence, e.g., kernel conditional embedding theory [r1, r2], the expectation of conditional difference measure [r3]. The discussion on related works and the advantage analysis are highly expected.

4. As shown in Eqs. (7)-(11), the Wasserstein distance mainly provides the weights for distance metric learning in the representation space parameterized by $\theta$. Specifically, once the optimal plans $p_{j|i}$ and $p_{i}$ are computed, the representation learning with Eq. (11) indeed shows an equivalent form of MFA (resp. LDA), where $\{ y_{j|i} \}_j$ is the neighbor set (resp. y_{j|i} is class prototype). Thus, further clarification on the significance of the proposed method is necessary.

5. Following Q4, the main difference seems that the MFA/LDA mainly uses 0-1 coding for the edge values, while the proposed method provides soft weights with kernel-like matrices Eqs. (8) and (10) (which are admitted by the entropic regularization and Sinkhorn scaling). Such a soft-weight framework is also considered in the literature (Su et al., IEEE TPAMI’22). Justifications to clarify the significance are expected.

6. As the basic task is to learn discriminative representations, which is a general topic in machine learning and pattern recognition, the current experiments seem to be insufficient to validate the superiority. 1) The methods that also focus on learning discriminant representations/metrics should also be considered, e.g., large margin-based metric learning [r4]; 2) the complex/large/updated datasets for comparison, since the used datasets are generally solved by pioneer works. 

----

[r1] Wang, Meihong, Fei Sha, and Michael Jordan. "Unsupervised kernel dimension reduction." Advances in neural information processing systems 23 (2010).

[r2] Sheng, Tianhong, and Bharath K. Sriperumbudur. "On distance and kernel measures of conditional dependence." Journal of Machine Learning Research 24.7 (2023): 1-16.

[r3] Sheng, Wenhui, and Qingcong Yuan. "Dimension reduction with expectation of conditional difference measure." Statistical Theory and Related Fields 7.3 (2023): 188-201.

[r4] Kim, Minchul, Anil K. Jain, and Xiaoming Liu. "Adaface: Quality adaptive margin for face recognition." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a regularized conditional optimal transport model for feature learning. An autoencoder is employed to learn an embedding space, in which local information is extracted with the guidance of neighbors in the original space. A theoretical analysis is provided, and different kinds of applications are used to evaluate the performance of the proposed method.

### Strengths
1. Both theoretical and empirical results are provided.

2. Different kinds of applications are considered, including clustering, classification, and set-valued object classification.

### Weaknesses
1. Figure 1 is misleading.

(1) Optimal transport (actually, here is not a rigorous optimal transport model, which will be discussed in the following comments) is performed in the embedding space rather than the original space.

(2) For $W(\delta x_1, p(y|x_1))$, it is not an optimal tranpsort model. Optimal transport finds the best mapping between two sets of samples with the minimal cost. However, the set of $x$ only includes one sample $x_i$.  

2. Similar to the above concern in Figure, Problem (3) is not an optimal transport model, and $\bar{W}_i^{\bar{r}}$ is not a Wasserstein distance. Optimal transport finds the best mapping between two sets of samples with the minimal cost, in which the mapping includes variables to optimize. However, Problem (3) only includes one sample $x_i$ in the set of $x$. In addition, $\bar{W}_i^{\bar{r}}$ does not involve variables of the mapping.

3. Problem (3) aims to learn low-dimensional embeddings with a preserved neighbor structure that is constructed based on original features. This idea is similar to manifold learning or graph Laplacian regularization, which also leverages neighbors for feature learning. Therefore, some manifold learning or graph regularization methods should be compared in the experiments.

4. What is the reason to adopt the student's t distribution for $q_{j|i}$? What if adopt other choices?

5. For the continuous version of Problem (6) given in the appendix, it seems that no extra insight regarding the problem or the model can be drawn from the continuous version.

6. For the results on some datasets in Table 1, the linear version LCWSL does not show a significant advantage compared with other methods. The further improvement of DCWSL could come from the deep learning model rather than the proposed method. Is there more discussion or explanation regarding this?

7. It would be better to discuss more about the effect of the number of neighbors. Usually, too many neighbors could introduce noisy or weak relations, which may not be helpful for classification or clustering. What is the effect of the number of neighbors on other tasks or datasets?

8. The ablation study is not sufficient. The objective function consists of several components. The effects of some terms are not evaluated in the experiments.

9. The compared methods are out-of-the-date. Some feature learning methods published recently can be adopted as a comparison. For example, some methods leverage similarity based graph to learn embeddings, which also extract information from neighbors.

10. The results of different values of hyper-parameters $\lambda_2$ and $\lambda_4$ are missing.

11. It would be better to provide visualization results of the solution, such as the probabilities learned by the model. Some interesting observations could be drawn from the visualization results.

### Questions
1. What is the reason to adopt the student's t distribution for $q_{j|i}$? What if adopt other choices?

2. For the continuous version of Problem (6) given in the appendix, what extra insight can be obtained?

3. It would be better to provide more results and discussions. The details can be obtained from the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes using the optimal transport method on conditional distributions to capture features of data in embedded space. The method requires autoencoders to encode and decode features and also uses Kullback-Leibler divergence as a regulation term in the optimization formulation. The paper proves a generalization bound on the loss function for the proposed method in classification and clustering tasks. The paper compares the proposed method with multiple kernel-based methods to see how accurately we can classify and cluster features in the latent space.

### Strengths
1. The paper proposes a novel method that applies optimal transport on conditional distributions and proves upper bounds on the loss terms.

2. The paper does many comparison experiments with other kernel-based methods and shows the proposed method has the lowest error rate and NMI metric. The experiments also contain multiple datasets, such as medical datasets and *NIST datasets, and contain multiple tasks.

### Weaknesses
1. The theorems are not closely related to the experiments. It is unclear how the generalization bound can help in terms of method convergence and empirical experiments.

2.  There are many hyperparameter tuning, for example, $\lambda_1, \lambda_2, \lambda_3$. The paper does not provide guidelines on tuning these parameters. There is no ablation study of these hyperparameters. Thus, it is not clear if the method is scalable to larger-scale and higher-dimension datasets.

### Questions
1. Can you add a citation to the statement that conditional and marginal distributions have closed-form solutions in line 100? 

2. How do you compare your method with other kernel-based methods on computation cost?

3. Is it possible to use the autoencoder to decode the features and compare the decoded images?

### Soundness
3

### Presentation
3

### Contribution
3
