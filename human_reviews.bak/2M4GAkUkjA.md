# Efficient Link Prediction via GNN Layers Induced by Negative Sampling

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
Graph neural networks (GNNs) for link prediction can loosely be divided into two broad categories.  First, \emph{node-wise} architectures pre-compute individual embeddings for each node that are later combined by a simple decoder to make predictions.  While extremely efficient at inference time (since node embeddings are only computed once and repeatedly reused), model expressiveness is limited such that isomorphic nodes contributing to candidate edges may not be distinguishable, compromising accuracy.  In contrast, \emph{edge-wise} methods rely on the formation of edge-specific subgraph embeddings to enrich the representation of pair-wise relationships, disambiguating isomorphic nodes to improve accuracy, but with the cost of increased model complexity.  To better navigate this trade-off, we propose a novel GNN architecture whereby the \emph{forward pass} explicitly depends on \emph{both} positive (as is typical) and negative (unique to our approach) edges to inform more flexible, yet still cheap node-wise embeddings.  This is achieved by recasting the embeddings themselves as minimizers of a forward-pass-specific energy function (distinct from the actual training loss) that favors separation of positive and negative samples.  As demonstrated by extensive empirical evaluations, the resulting architecture retains the inference speed of node-wise models, while producing competitive accuracy with edge-wise alternatives.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The proposed method tackles the link prediction tasks by combining positive and negative edges in the forward pass, resulting in a fast and accurate model.

### Strengths
1. The authors provide the time complexity analysis and show the convergence of the proposed method.

2. The experimental results show that the proposed method outperforms most of the baseline methods across multiple benchmark datasets.

### Weaknesses
1. The major contribution of this paper is to include the negative subgraph during the node representation learning. However, my major concern is how to avoid sampling the false-negative edges during the sampling strategy. These false-negative edges will greatly influence the performance of the proposed method.

2. The presentation of this paper could be improved by breaking some very long sentences into two or more. For instance, "However, these desiderata alone are inadequate for the link prediction task, where we would also like to drive individual nodes towards regions of the embedding space where they are maximally discriminative with respect to their contributions to positive and negative edges.". In addition, the comma is missing in many sentences. For instance, "However, when negative samples/edges are included the isomorphism no longer holds."

3. Only part of the code is provided. It's impossible to reproduce the experimental results.

4. The experimental setting is not clear. See question 1 below.

5. In Table 2, the highlighted MRR score (SEAL) on the Citation2 dataset is not the best one.

### Questions
1. I am curious about the experimental setup. How do you determine the training set, validation set, and the test set? Do you mask a certain percentage of edges as validation and test sets in the experiment? If so, how do you avoid sampling the positive edges using negative sampling?

2. Table 3 shows the ablation study over negative sampling. I am curious why "W/O Negative" is still significantly better than most other baseline methods listed in Table 2 across multiple datasets, especially for those edge-based methods. Since the major contribution of the proposed method is to embed the negative samples into the learned representations for the link prediction, the proposed method with removed negative sampling should have a similar performance to node-wise methods.

3. Should $j'\in V^1_{(i,j)}$ be $j'\in V^k_{(i,j)}$ right above equation 9?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
There are 2 main categories of GNNs for link prediction, node-wise and edge-wise architectures. Node-wise architectures are efficient at test time, model expressiveness is limited such that isomorphic nodes contributing to candidate edges may not be distinguishable. Edge-wise methods form edge-specific subgraph embeddings to enrich representation of pair-wise relationships, disambiguating isomorphic nodes to improve accuracy, but they are computationally expensive. To navigate this trade-off, the authors propose a GNN architecture where nodes are updated following a contrastive and a link prediction loss.

### Strengths
-	The authors address an important tradeoff in GNNs for link prediction (predictive power vs computational complexity)
-	The approach has a small time complexity (similar to GCN)

### Weaknesses
First of all, thanks for sending your work for review. 
I have two general comments that would like you to comment on.

1.	The use of a loss function to generate embeddings that minimize the distance between connected nodes and maximize the distance between disconnected nodes is not new. It’s typically used in the context of self-supervised learning/pre-training, and it has been previously used in the context of link prediction. See: 
- You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020). Graph Contrastive Learning with Augmentations. http://arxiv.org/abs/2010.13902
- Qiu, J., Chen, Q., Dong, Y., Zhang, J., Yang, H., Ding, M., Wang, K., & Tang, J. (2020). GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1150–1160. https://doi.org/10.1145/3394486.3403168
- Shiao, W., Guo, Z., Zhao, T., Papalexakis, E. E., Liu, Y., & Shah, N. (2023). Link Prediction with Non-Contrastive Learning. International Conference on Learning Representations (ICLR). https://openreview.net/pdf?id=9Jaz4APHtWD
- Bordes, A., Usunier, N., Garcia-Durán, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. Advances in Neural Information Processing Systems (NeurIPS). https://papers.nips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

My question is, what is the novelty of your approach given that the notion of contrastive learning has previously been used to compute node embeddings?

2.	What you mention as “including negative edges in the forward pass” seems to me as actually just having 2 steps in the backward pass. A first step where node embeddings are updated based on the contrastive loss, and a second step where node embeddings are further updated based on the link prediction loss.

### Questions
See weaknesses section

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studied the research problem of link prediction. The authors discussed the pros and cons of node-wise and link-wise link prediction literature as well as their trade-offs, and then proposed YinYanGNN which has better expressiveness than the existing node-wise methods while still preserving the fast inference speed of node-wise methods.

### Strengths
1. The research problem is very interesting given that the link prediction literature has been gradually separated into two groups.
2. The proposed method is reasonably designed and showed promising results across multiple benchmarks, including several larger ones.
3. The authors provided complexity analysis as well as inference time comparison of the proposed method, illustrating the efficiency of the proposed method.

### Weaknesses
1. I appreciate the complexity analysis and inference time comparison. Nonetheless, I'd suggest the authors to also include total runtime comparison or even wall times of different procedures for more comprehensive understanding of the efficiency. E.g., Table 3 in [1]
2. Results in Table 2 included strong edge-wise methods with only basic node-wise methods. Even the ones in Table 6 are not specifically designed for link prediction. I'd suggest including more recent / stronger node-wise methods to make the effectiveness comparison more compelling.
3. I found the method section hard to follow even with careful reading

[1] Graph Neural Networks for Link Prediction with Subgraph Sketching, ICLR 2023

### Questions
please see above

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new link prediction framework that aims at improving the expressive capability of the node-based link predictors. The proposed method, namely YinYanGNN, incorporates an additional graph that is constructed from pure negative edges. While generating node embeddings from both the original graph and the additional graph, YinYanGNN can acquire better expressive capability. The authors connect the reason behind why such an implementation would work from the perspective of energy functions. On top of the energy functions explored by existing works, the energy function explored in this work incorporates an additional term that enforces dissimilarities between neighbors on the negative graph. Experiments over multiple datasets are conducted and the proposed YinYanGNN exhibits promising results.

### Strengths
1. I think the overall idea of enhancing node-based link predictors by heuristics from graph-based link predictors is meaningful. 

2. The implementation of the proposed  framework seems easy to implement.

### Weaknesses
1. The introduction is motivated by improving the expressive capability of node-based link predictors, but the proposed method does not seem to improve that. I think the expressive capability discussed by in frameworks like BUDDY or SEAL is related to isomorphism. And the proposed YinYanGNN does not seem to be related to this concept. Section 5.3 tries to showcase this connection, but I think YinYanGNN only leverages this connection implicitly. I might be understanding this wrong due to the second weakness next. 

2. The method section is a little bit difficult to understand. I think the authors should concisely conclude how YinYanGNN works (e.g., how do the forward pass and backward pass look like? what is the model architecture, etc) before introducing the energy function. I feel like the authors explain the overall idea through a series of vague terms (e.g., before section 4.1) that are difficult to understand without sufficient background knowledges. 

3. It seems like the training requires learning over multiple negative graphs. This operation seems to introduce a lot of computational overheads during the training. 

4. The time complexity analysis seems problematic. A lot of overheads during the training can be moved to the pre-processing.

### Questions
Q1: Why does equation (3) need not generally be convex or even lower bounded?

Q2: Do operations described around Equation (13) supports mini-batch training?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
