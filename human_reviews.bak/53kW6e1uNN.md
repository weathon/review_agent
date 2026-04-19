# AFDGCF: Adaptive Feature De-correlation Graph Collaborative Filtering for Recommendations

- Decision: Reject
- Scores: 8, 5, 5, 8

## Abstract
Collaborative filtering methods based on graph neural networks (GNNs) have witnessed significant success in recommender systems (RS), capitalizing on their ability to capture collaborative signals within intricate user-item relationships via message-passing mechanisms. However, these GNN-based RS inadvertently introduce a linear correlation between user and item embeddings, contradicting the goal of providing personalized recommendations. While existing research predominantly ascribes this flaw to the over-smoothing problem, this paper underscores the critical, often overlooked role of the over-correlation issue in diminishing the effectiveness of GNN representations and subsequent recommendation performance. The unclear relationship between over-correlation and over-smoothing in RS, coupled with the challenge of adaptively minimizing the impact of over-correlation while preserving collaborative filtering signals, is quite challenging. To this end, this paper aims to address the aforementioned gap by undertaking a comprehensive study of the over-correlation issue in graph collaborative filtering models. Empirical evidence substantiates the widespread prevalence of over-correlation in these models. Furthermore, a theoretical analysis establishes a pivotal connection between the over-correlation and over-smoothing predicaments. Leveraging these insights, we introduce the Adaptive Feature De-correlation Graph Collaborative Filtering (AFDGCF) Framework, which dynamically applies correlation penalties to the feature dimensions of the representation matrix, effectively alleviating both over-correlation and over-smoothing challenges. The efficacy of the proposed framework is corroborated through extensive experiments conducted with four different graph collaborative filtering models across four publicly available datasets, demonstrating the superiority of AFDGCF in enhancing the performance landscape of graph collaborative filtering models.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper draws attention to the challenges of over-smoothing and over-correlation in GNN-based collaborative filtering methods. In particular, the paper provides a detailed analysis of the over-correlation problem, which has been largely overlooked in existing works. Through rigorous theoretical analysis, the paper establishes a proportional association between the over-smoothing issue and the over-correlation issue, shedding light on their interconnected nature.

To tackle these issues, the paper proposes a model-agnostic constraint with adaptive weights. This constraint is designed to effectively mitigate over-smoothing and over-correlation problems in GNN-based collaborative filtering. The adaptive weights allow the constraint to dynamically adjust and optimize the learning process.

Comprehensive experiments are conducted to validate the effectiveness of the proposed constraint. The results demonstrate significant improvements in overall performance, enhanced training efficiency, and the efficacy of the adaptive approach. These findings provide strong evidence for the practical benefits of the proposed constraint in addressing the over-smoothing and over-correlation challenges in GNN-based collaborative filtering methods.

### Strengths
- The paper highlights the issue of decorrelation in collaborative filtering, which has received little attention in previous works.
- Through a comprehensive theoretical analysis, the paper establishes a clear association between the over-smoothing problem and the decorrelation issue.
- To address the challenges of over-smoothing and decorrelation, the paper proposes an effective solution. The proposed scheme is extensively evaluated through rigorous experiments, demonstrating its effectiveness.
- The paper is well-written and provides clear explanations. It includes illustrative figures and pilot experiments that enhance understanding and readability.

### Weaknesses
- I have reservations regarding the dataset preprocessing approach employed in the paper. The authors chose to exclude users and items with fewer than 15/10 interactions in some datasets. However, in my experience, this approach has the potential to create highly dense datasets and introduce bias.
- It would have been beneficial if the paper had explored the recent advancements in self-supervised learning for collaborative filtering, as these techniques have demonstrated superior performance in related studies.

### Questions
I would expect the authors to clarify the two issues mentioned in the weaknesses part.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors focus on analyzing feature over-correlation in graph-based collaborative filtering, and propose an adaptive feature de-correlation regularization in graph-based collaborative filtering. Column-wise feature over-correlation will introduce redundant information for representation learning, the proposed feature de-correlation regularization can significantly improve the representation quality. Besides, the proposed feature de-correlation is very flexible and lightweight, which can coupled with representation-based CF. Experiments on several benchmarks show the effectiveness of the proposed method.

### Strengths
1. Interesting research topic of this paper, tacking feature over-correlation in collaborative filtering is an effective direction.
2. The proposed feature de-correlation regularization is flexible and effective in graph-based collaborative filtering. De-correlation is helpful in learning more high-quality representation for collaborative filtering.
3. Experiments conducted on several graph-based backbones demonstrate the effectiveness of the proposed de-correlation regularization.

### Weaknesses
1. The motivation of this paper should be highlighted. Why do the authors analyze over-correlation combined with over-smoothing? Does feature over-correlation only occur on graph-based collaborative filtering non other methods such as Matrix Factorization? 
2. The reason for existing over-correlation in low-dimensional collaborative filtering is not clear. It will be more interesting if the authors deeply explain the behind reasons. Besides, does alleviating over-correlation can help to reduce over-smoothing issues in graph-based collaborative filtering? The authors should give a more explanatory illustration.
3. Lacking comparisons of related works, disentangled collaborative filtering should be involved. Besides, column-wise de-correlation can be also viewed as self-supervised learning[1]. The authors should discuss with current self-supervised graph collaborative filtering method[2,3,4].
[1]Wang X, Jin H, Zhang A, et al. Disentangled graph collaborative filtering[C]//Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval. 2020: 1001-1010.
[2]Wu J, Wang X, Feng F, et al. Self-supervised graph learning for recommendation[C]//Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 2021: 726-735.
[3]Yu J, Yin H, Xia X, et al. Are graph augmentations necessary? simple graph contrastive learning for recommendation[C]//Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval. 2022: 1294-1303.
[4]Yang, Y., Wu, Z., Wu, L., Zhang, K., Hong, R., Zhang, Z., ... & Wang, M. (2023). Generative-Contrastive Graph Learning for Recommendation.

### Questions
Mentioned as the weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper analyzes the feature correlation issues in graph collaborative filtering. The author(s) present empirical studies on the smoothness and correlation of each layer of various graph collaborative filtering methods. Then, the author(s) propose AFDGCF that incorporates an auxiliary loss function to explicitly optimize the over-correlation issue. Extensive experiments on four public datasets and four popular GCF backbones show the effectiveness of the proposed method. Code is available and the author(s) promise to release all the code after the reviewing phase.

### Strengths
1. The paper studies an important task, i.e., graph collaborative filtering.
2. The proposed model is implemented by an open-source framework, making it easy to reproduce. Code is available during the reviewing phase.
3. Extensive experiments on four public datasets and four popular GCF backbones show the effectiveness of the proposed method.

### Weaknesses
1. Limited novelty. The paper seems like a straightforward application of existing literature, specifically the DeCorr [1] that focuses on general deep graph neural networks, in a specific application domain. The contribution of this study is mainly the transposition of DeCorr's insights into graph collaborative filtering, with different datasets and backbones. Although modifications like different penalty coefficients for users and items are also proposed, the whole paper still lack enough insights about what are unique challenges of overcorrelation in recommender systems.

2. It could be better if one additional figure could be illustrated, i.e., how Corr and SMV metrics evolve with the application of additional network layers—mirroring the Figure 2, but explicitly showcasing the effects of the proposed method—the authors could convincingly validate their auxiliary loss function's efficacy.

3. Presentation issues. The y-axis labels of Figure 2 lack standardization, e.g., 0.26 vs. 0.260 vs. 2600 vs. .2600.

[1] Jin et al. Feature overcorrelation in deep graph neural networks: A new perspective. KDD 2022.

### Questions
According to Theorem 1, there exists a proportional relationship between column correlation and row correlation of a matrix. So whether existing works on alleviating row correlation issues like contrastive learning also solve the correlation issues? Once the row correlation is alleviated, according to the proportional relationship, the column correlation should be alleviated as well. If so, why do we need the proposed auxiliary loss to explicitly alleviate the column correlation issue?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper discusses the possible connections between over-smoothing and over-correlation in graph neural networks-based recommender systems. Indeed, while over-smoothing has been debated in graph-based recommendation for quite some time now, the authors claim over-correlation is still not properly analysed as happening in graph representation learning. Through an initial empirical study, the authors demonstrate that the negative effects of the two issues seem to be directly dependent and go along with the performance degradation of the models (i.e., usually after the third message-passing layer). After that, the paper underlines how over-smoothing and over-correlation may present a direct mapping to rows and columns in the node embedding matrix, respectively, and mathematically proves that the two are proportional. In this respect, as alleviating one of the two would tackle also the other, the authors propose a loss function named adaptive feature decorrelation, that comes into a static and dynamic version. An extensive experimental setting comprising four recommendation datasets and nine baselines demonstrates the efficacy of the proposed approach. Indeed, when applied to existing graph-based recommender systems, the adaptive feature decorrelation loss function is beneficial to improve the performance in terms of recommendation accuracy and requiring much less epochs to reach convergence. Finally, an ablation study justifies the soundness of the proposed architectural choices.

### Strengths
+ The addressed problem (i.e., over-smoothing and over-correlation in graph-based recommendation) is relatively new to the literature.
+ The empirical analysis supported by the mathematical proofs help justifying the existing problem and opening to possible solutions.
+ The experimental setting is extensive with numerous evaluation dimensions.
+ The code and datasets are released at review time.

### Weaknesses
- Some details about the introduced methodology need to be clarified.
- The authors may have not considered other graph-based recommendation baselines whose solutions are like the proposed one.

**After the rebuttal.** The rebuttal clarified all weaknesses.

### Questions
* To the best of my understanding, I cannot find the reason why the authors state that “it is crucial to maintain the smoothness of deep representations while restricting the feature correlations of the model’s representations” (beginning of page 7). The paper seems to claim that when reducing over-correlation for deeper representations, also over-smoothing will be tackled. In this sense, I cannot see the point in the quoted statement. Would you please elaborate on that?
* Did the authors consider graph-based recommendation approaches which leverage decorrelation in a similar manner to the proposed one (e.g., disentangled graph collaborative filtering, DGCF [1]). In authors’ opinion, what would it be (even intuitively) the effect of performing a double decorrelation if the proposed loss function was applied to DGCF? Would it have a positive or a negative impact, and why?

[1] Xiang Wang, Hongye Jin, An Zhang, Xiangnan He, Tong Xu, Tat-Seng Chua: Disentangled Graph Collaborative Filtering. SIGIR 2020: 1001-1010

**After the rebuttal.** The rebuttal answered all questions.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
