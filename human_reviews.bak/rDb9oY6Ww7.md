# Robust Consensus Anchor Learning for Efficient Multi-view Subspace Clustering

- Decision: Reject
- Scores: 8, 8, 8, 3

## Abstract
As a leading unsupervised classification algorithm in artificial intelligence, multi-view subspace clustering segments unlabeled data from different subspaces. Recent works based on the anchor have been proposed to decrease the computation complexity for the datasets with large scales in multi-view clustering. The major differences among these methods lie on the objective functions they define. Despite considerable success, these works pay few attention to guaranting the robustness of learned consensus anchors via effective manner for efficient multi-view clustering and investigating the specific local distribution of cluster in the affine subspace. Besides, the robust consensus anchors as well as the common cluster structure shared by different views are not able to be simultaneously learned. In this paper, we propose Robust Consensus anchors learning for efficient multi-view Subspace Clustering (RCSC). We first show that if the data are sufficiently sampled from independent subspaces, and the objective function meets some conditions, the achieved anchor graph has the block-diagonal structure. As a special case, we provide a model based on Frobenius norm, non-negative and affine constraints in consensus anchors learning, which guarantees the robustness of learned consensus anchors for efficient multi-view clustering and investigates the specific local distribution of cluster in the affine subspace. While it is simple, we theoretically give the geometric analysis regarding the formulated RCSC. The union of these three constraints is able to restrict how each data point is described in the affine subspace with specific local distribution of cluster for guaranting the robustness of learned consensus anchors. RCSC takes full advantages of correlation among consensus anchors, which encourages the grouping effect and groups highly correlated consensus anchors together with the guidance of view-specific projection. The anchor graph construction, partition and robust anchor learning are jointly integrated into a unified framework. It ensures the mutual enhancement for these procedures and helps lead to more discriminative consensus anchors as well as the cluster indicator. We then adopt an alternative optimization strategy for solving the formulated problem. Experiments performed on eight multi-view datasets confirm the superiority of RCSC based on the effectiveness and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
To improve the scalability of the multi-view subspace cluserting to large-scale data, this paper proposes Robust Consensus anchors learning for efficient multiview Subspace Clustering (RCSC), which joints the robust anchor learning, anchor graph construction, and partition into a unified framework. This paper theoretically demonstrates that a optimal anchor graph with block-diagonal structure is obtained if the objective function satisfies certain conditions. The RCSC is simple and effective in terms of performance.

### Strengths
1. The proposed RCSC is simple and effective in terms of performance.
2. The theoretical analysis of the paper is sufficient and the paper is easy to read.

### Weaknesses
The innovation of the paper needs to be further condensed and analyzed. Specifically, it is important to emphasize the connection and differences between the following works:
1. Fast Parameter-free Multi-view Subspace Clustering with Consensus Anchor Guidance.
2. Scalable Multi-view Subspace Clustering with Unified Anchors.

### Questions
There are several issues that still require detailed explanation.
1. Authors claim that "Besides, the robust consensus anchors and the common cluster structure shared by different views are not able to be simultaneously learned in a unified framework". However, some methods can already achieve this, for example, FPMVS-CAG(Fast Parameter-free Multi-view Subspace Clustering with Consensus Anchor Guidance) and SMVSC(Scalable Multi-view Subspace Clustering with Unified Anchors). It is crucial to analyze and explain the connection and differences with the above methods, especially the SMVSC.
2. In Lemma 1, the symbol ||A||_* is the nuclear norm? What is the function of Lemma 1? It seems that there is no nuclear norm in the objective function.
3. Some symbol definitions are unclear, such as X_i^P, A_i.
4. The data used in the experiment is unclear, such as the scale of the YoutubeFace.
5. The experimental section lacks the latest and SOTA methods for comparison.
6. In Figure 4, the horizontal axis is Corrupted ratio, but why is 2k, 3k, 4k, 5k,6k, 7k in (c) and (d)?

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
This paper proposes Robust Consensus anchors learning for efficient multi-view Subspace Clustering (RCSC). The authors first show that if the data are sufficiently sampled from independent subspaces, and the objective function meets some conditions, the achieved anchor graph has the block-diagonal structure. As a special case, the authors provide a model based on Frobenius norm, non-negative and affine constraints in consensus anchors learning, which guarantees the robustness of learned consensus anchors for efficient multi-view clustering and investigates the specific local distribution of cluster in the affine subspace. While it is simple, we theoretically give the geometric analysis regarding the formulated RCSC. The union of these three constraints is able to restrict how each data point is described in the affine subspace with specific local distribution of cluster for guaranting the robustness of learned consensus anchors.

### Strengths
1. Novelty. This paper first shows that if the data are sufficiently sampled from independent subspaces, and the objective function meets some conditions, the achieved anchor graph has the block-diagonal structure. As a special case, the authors provide a model based on Frobenius norm, non-negative and affine constraints in consensus anchors learning, which guarantees the robustness of learned consensus anchors for efficient multi-view clustering and investigates the specific local distribution of cluster in the affine subspace.

2. Quality and Clarity. There are no technical errors, and the presentation and writing are clear.

3. Significance. It is able to ensure the mutual enhancement for these procedures and helps lead to more discriminative consensus anchors as well as the cluster indicator. We then adopt an alternative optimization strategy for solving the formulated problem.

### Weaknesses
1. The authors adopt the orthogonal and nonnegative factorization to directly assign clusters to the data for integrating the partition into the unified framework. The authors should explain why the partition can be integrated into the unified framework based on the orthogonal and nonnegative factorization.

2. The authors report the clustering results with respect to ACC, NMI and F1-score of all multi-view clustering methods in Tables 1-3, respectively. Then they adopt N/A to indicate that the method is not able to be computationally feasible on the dataset caused by out of memory. The authors are expected to highlight the second best clustering performance in Tables 1-3 to make the performance gains of the proposed method more obvious.

3. The authors fix the shared dimension and conduct the sensity analysis for the number of anchors on several datasets in terms of different metrics. According to Fig. 3, The authors find that the proposed method is not significantly influenced by the number of anchors and the clustering results with different number of anchors are relatively stable. In this part, the authors should give more detailed analysis regarding how the total number of anchors impacts the clustering results in this part.

4. The authors give the conclusion including stating that Extensive experiments verify the effectiveness and efficiency of the proposed method on different multi-view datasets under three metrics. In this part, the authors are expected to give more detailed specific values of the performance gains of the proposed method.

### Questions
1. The authors report the execution times of the compared methods and theirs on different datasets. As shown in Fig. 5, it is observed that the proposed method has shown comparable logarithm of running time cost to the existing efficient methods on most of the multi-view datasets, i.e., MSGL. However, the authors do not give the memory value of the used device, which is important in running time analysis part.

2. In this reference part, some names of publication are termed for short, i.e., Artif. Intell., however, some names of publications are fully termed, i.e., IEEE Conference on Computer Vision and Pattern Recognition. I think the authors should carefully check this issue to improve the whole presentation of this paper.

### Soundness
3

### Presentation
4

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
The authors propose a novel Robust Consensus anchor learning for efficient multi-view Subspace Clustering. They first theoretically demonstrate that an anchor graph with a block-diagonal structure can be achieved if the objective function satisfies certain conditions. Then, they impose the orthogonal constraints on the actual bases and constrain a factor matrix to be the cluster indicator matrix built on the rigorous clustering interpretation.

### Strengths
1. The writing style is generally standard and free of obvious errors.
2. This paper fully explores the correlation among the learned consensus anchors with the guidance of view-specific projection, which encourages the grouping effect and tends to group highly correlated anchors together. 
3. It imposes orthogonal constraints on the actual bases and constrains a factor matrix to be the cluster indicator matrix built on the rigorous clustering interpretation. 
4. Extensive experiments validate its effectiveness and efficiency.

### Weaknesses
1. The authors adopt $ A\in R^{d\times l} $ to represent the unified anchors, $ l $ and $ d $ are the number of anchors and shared dimension across views, respectively. $ U^{p}A\in R^{d_{p}\times l} $ represents the basis matrix. The authors then theoretically demonstrate that a block-diagonal anchor graph can be achieved if the corresponding objective function satisfies certain conditions based on the independent subspace assumption, which is shown as Theorem 1 in the following. However, the authors do not give the explanation why $ U^{p}A\in R^{d_{p}\times l} $ can represent the basis matrix and the related analysis can be given here.
2. The authors state that the data are usually contaminated with the possible noise in real applications and then adopt the Frobenius norm for penalizing the noise based on affine and non-negative constraint in Eq. (5). Here, the authors need to explain what kind of noise is dealt with by the Frobenius norm.
3. The authors state that the objective function monotonically decreases in each iteration until convergence since the convex property for each sub-problem and then list the procedure of RCSC in Algorithm 1. The authors are expected to give some explanation for the procedure of RCSC in Algorithm 1.
4. The authors tune the number of the proposed method in the range of $ [2k,3k,\cdots,7k] $, where $ k $ denotes the total number of clusters in dataset. The authors are expected to give some analysis why choose $ [2k,3k,\cdots,7k] $ as the range instead of the others in this part.

### Questions
1. The authors use $ (U^{p}A)^{t} $ as the basis matrix lying in the $ t $-th affine subspace. Here, the dimension of $ (U^{p}A)^{t} $ is expected to be given to better show the reason why it can be adopted as the basis matrix lying in the $ t $-th affine subspace.
2. The authors do not give the notations table throughout the paper and the physical meaning of each variable is not very clear. Then the authors need to give the notation table in this paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a multi-view clustering methods, termed as Robust Consensus anchors learning for efficient multiview Subspace Clustering (RCSC). RCSC aims to address two problems: (1) the existing works pay few attention to guaranting the robustness of learned consensus anchors via effective manner for efficient multi-view clustering and investigating the specific local distribution of cluster in the affine subspace; (2) the robust consensus anchors as well as the common cluster structure shared by different views are not able to be simultaneously learned.

### Strengths
(1) This work theoretically demonstrates that an anchor graph with block-diagonal structure can be achieved if the objective function satifies certain conditions.
(2) The proposed method fully explores the correlation among the learned consensus anchors with the guidance of view-specific projection.
(3) Extentive experiments on different multi-view datasets validate the effectivenss and efficiency of RCSC.

### Weaknesses
(1) The novelty of this work is limited, since these involved components have been widely used for anchor learning and clustering. The objective function is incremental.
(2) The authors lack comparison with some recently proposed SOTA methods, only one within three years, so the experimental results are not convincing.
(3) The authors do not compare the proposed method with theses popular deep learning ones.
(4) The proposed method involves too many hyperparameters, so it is difficult for tuning it and the experiment results are not convincing.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
