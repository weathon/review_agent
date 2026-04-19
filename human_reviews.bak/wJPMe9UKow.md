# Posterior Label Smoothing for Node Classification

- Decision: Reject
- Scores: 6, 8, 5, 3

## Abstract
Soft labels can improve the generalization of a neural network classifier in many domains, such as image classification. Despite its success, the current literature has overlooked the efficiency of label smoothing in node classification with graph-structured data. In this work, we propose a simple yet effective label smoothing for the transductive node classification task. We design the soft label to encapsulate the local context of the target node through the neighborhood label distribution. We apply the smoothing method for seven baseline models to show its effectiveness. The label smoothing methods improve the classification accuracy in 10 node classification datasets in most cases. In the following analysis, we find that incorporating global label statistics in posterior computation is the key to the success of label smoothing. Further investigation reveals that the soft labels mitigate overfitting during training, leading to better generalization performance. Our code is available at https://anonymous.4open.science/r/PosteL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a label-smoothing method called PosteL for the transductive node classification task. This method generates soft labels by integrating local neighborhood information and global label statistics, enhancing generalization and reducing overfitting. Specifically, it computes the posterior distribution of node labels by estimating conditional distributions through neighbour nodes and prior distributions through all labeled nodes, and then interpolates the posterior distribution with the ground truth label to obtain a soft label.Experiments on 10 node classification datasets with 7 baseline models demonstrate its effectiveness.

### Strengths
1. This paper introduces a simple yet effective label smoothing method for transductive node classification.

2. This paper investigates the performance of the proposed label smoothing method on different datasets and models, and analyzes the key factors for its success.

3. The authors present their ideas in a well-structured manner, making it easy for readers to follow the flow of the research. The figures in the paper are presented clearly and effectively support the text.

### Weaknesses
1. After claiming the effective handling of heterophilic graphs as a highlight, the description of the proposed method fails to emphasize its treatment of heterophilic graphs. For instance, it is not clear why the proposed method is suitable for handling heterophilic graphs and where the efficiency of this treatment is manifested.

2. The motivation for proposing the method and the effects achieved by the method itself seem to be contradictory. In Section 1, the authors state "...their performance on heterophilic graphs, where nodes tend to connect with others that are dissimilar or belong to different classes, still remains questionable." However, the proposed method estimates the posterior of one node based on label frequency of its neighbours, which leads to its prediction similar to the classes of its neighbours. In addition, the estimation of the prior mitigates the problem of class imbalance but does not address the stated problem of heterogeneity.

3. The idea in the method section of the article is similar to that of a naive Bayes approach, but some assumptions of conditional independence need to be clearly stated in Section 3.

4. In the case where the neighbors are not labeled and the neighbors of the neighbors are also not labeled, an iterative process of generating pseudo-labels needs to be reflected in this algorithm part.

### Questions
1. Please explain how the method proposed in the article addresses the stated problem of heterogeneity.

2. In the setting of this paper,  the connectivity between all nodes, including the test nodes, is assumed to be observed. Is this assumption a bit too strong? Please give some explanations.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The manuscript proposes PosteL, a label smoothing method utilizing posterior distribution for node classification in graph-structured data. It is basically a preprocessing method for GNNs, generating soft labels based on neighborhood context and global label statistics before the training phase.

### Strengths
1. The manuscript is well-structured and easy to comprehend. For example, Fig. 1 is clear and intuitive.
2. The method is simple yet effective. PosteL can be combined seamlessly with existing methods.
3. The manuscript presents a wealth of experimental results that highlight the potential of PosteL for performance enhancement.

### Weaknesses
The significant drawback I perceive in this manuscript is that while it explores the usefulness of smoothing in the graph domain, the underlying principles that contribute to its effectiveness are not sufficiently clarified. Providing a theoretical analysis of its effectiveness would significantly strengthen the authors' claims.

### Questions
1. According to Section 2.2, PosteL differs from existing works focused on smoothing in the graph domain in terms of assumptions. However, is there any analysis that demonstrates how these assumptions influence performance improvement? Are there specific examples that support the manuscript's assumptions? Providing this would be crucial for substantiating the novelty of the work.
2. According to the appendix, the datasets are all structured data. Can PosteL be applied to a broader range of research in the graph domain?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
1

### Summary
This paper proposes a posterior label smoothing method (PosteL) that enhances node classification accuracy by incorporating both local neighborhood information and global label statistics. The approach demonstrates performance improvements on both homogeneous and heterogeneous graphs. Experimental results show that this method prevents overfitting and exhibits generalization capabilities across multiple datasets and baseline models.

### Strengths
1. This paper introduces a new idea in label smoothing by using posterior distribution calculations for label smoothing. Unlike traditional uniform noise smoothing methods, the authors propose combining local neighborhood information with global label distribution, allowing labels to reflect the local graph structure of nodes.
2. The method was tested on both standard homogeneous graphs and heterogeneous graphs, highlighting its general applicability. Experiments show that PosteL label smoothing can prevents overfitting and enhances model generalization without increasing model complexity.

### Weaknesses
1. While the posterior label smoothing method achieves notable improvements on graph-structured data, it relies on the neighborhood label distribution of nodes, which may lead to unstable posterior distributions in sparse graphs or scenarios with very few labels, resulting in reduced label quality. Although the authors propose a strategy of re-estimating the posterior with pseudo labels, limitations remain, especially on heterogeneous graphs with sparse labels.
2. Although the paper mentions that the computational complexity is linearly related to the number of edges and categories, it lacks sufficient discussion on efficient computation and optimization for large-scale graph structures, especially heterogeneous graphs. In real-world applications, such as large-scale social networks or e-commerce recommendation systems, this method may encounter efficiency bottlenecks.
3. Although the paper compares various existing label smoothing methods, it lacks adequate comparison with the latest graph classification models, particularly on large-scale heterogeneous graphs. This limitation suggests that PosteL’s effectiveness on more complex or dynamic graph tasks may require further validation.

### Questions
1 The author should clarify how different neighborhood configurations impact performance in heterophilic vs. homophilic settings.
2 How does the method perform with highly sparse labels, particularly below 10% labeled data? Are there specific mitigations for sparsity?
3  The aushor should provide details on the computational cost of PosteL relative to other smoothing methods, especially in terms of training duration across datasets?

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
4

### Summary
The paper introduces a method called Posterior Label Smoothing (PosteL) designed to improve node classification tasks in graph-structured data. By integrating both local neighborhood information and global label statistics, PosteL generates soft labels that enhance the generalization capabilities of models and mitigate overfitting. The method is applied to various datasets and models, showing that it consistently outperforms traditional label smoothing and other baseline methods.

### Strengths
1. The paper is well-written and the content is presented clearly, making it easy to follow.

2. The experiments are comprehensive and include a variety of different settings.

3. The proposed method is probabilistically driven, offering a more rational approach compared to existing heuristics.

### Weaknesses
1. The contribution of this paper is somewhat incremental - the idea of using neighborhood information to perform label smoothing has already been extensively studied [1, 2]. 

2. "Under the assumption that the neighborhood labels are conditionally independent given the label of the node to be relabeled...", I fail to comprehend this statement, hope authors can further explain this. But in any case, I think the assumption of two adjacent nodes are conditioally independent is too strong. 

3. While the proposed method shows superior emprical performance, this paper fail to provide an in-depth explaination on why label-smoothing performs so well in graph-structured data - does it stems from the same reason as i.i.d. data? What makes the proposed method superior than other label smoothing methods on graph?

4. The improvements over the most competitive baselines are marginal - for most cases the differences are within the standard deviation.

[1] Adaptive Label Smoothing To Regularize Large-Scale Graph Training, Zhou et al., 2021.

[2] Structure-Aware Label Smoothing for Graph Neural Networks, Wang et al., 2021.

### Questions
please refer to the weakness

### Soundness
3

### Presentation
2

### Contribution
2
