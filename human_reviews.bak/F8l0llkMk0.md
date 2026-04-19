# The Map Equation goes Neural

- Decision: Reject
- Scores: 6, 1, 3

## Abstract
Community detection and graph clustering are essential for unsupervised data exploration and understanding the high-level organisation of networked systems.
Recently, graph clustering has been highlighted as an under-explored primary task for graph neural networks.
While hierarchical graph pooling has been shown to improve performance in graph and node classification tasks, it performs poorly in identifying meaningful clusters.
Community detection has a long history in network science, but typically relies on optimising objective functions with custom-tailored search algorithms, not leveraging recent advances in deep learning, particularly from graph neural networks.
In this paper, we narrow this gap between the deep learning and network science communities.
We consider the map equation, an information-theoretic objective function for community detection.
Expressing it in a fully differentiable tensor form that produces soft cluster assignments, we optimise the map equation with deep learning through gradient descent.
More specifically, the reformulated map equation is a loss function compatible with any graph neural network architecture, enabling flexible clustering and graph pooling that clusters both graph structure and data features in an end-to-end way, automatically finding an optimum number of clusters without explicit regularisation.
We evaluate our approach experimentally using different neural network architectures for unsupervised clustering in synthetic and real data.
Our results show that our approach achieves competitive performance against baselines, naturally detects overlapping communities, and avoids over-partitioning sparse graphs.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors formulate the well-known MAP equation for community detection as an unsupervised objective for graph clustering with GNNs. The implement this "soft" neural MAP equation in various GNN architectures, showing reasonable performance on both synthetic and real-world graph clustering tasks.

### Strengths
S1: The port of the MAP equation to a NN graph clustering objective is good to have in the modern-day toolkit of neural clustering techniques.

S2: The paper is well-written and easy to follow.

S3: The experiments are sufficient and easy to understand.

### Weaknesses
W1: The contribution itself is marginal. The authors seem to simply replace the objective of Tsitsulin et al. 2023 with the MAP equation.

W2: The authors claim that the MAP equation avoids over-partitioning, but do not provide any theoretical justification.

W3: The authors claim the ability to detect overlapping communities as a contribution of their work, but this is also true of any "soft clustering" neural method including Tsitsulin et al. 2023.

### Questions
My questions are as follows:

(1) re W1, Can the authors claim any technical novelty beyond deriving the MAP equation as a neural objective and using the approach of Tsitsulin et al. 2023?

(2) re W2, on page 4, the authors claim "the map equation naturally incorporates Occam's razor: minimising the map equation requires a trade-off between choosing small modules for low module-level codelength and choosing a small number of modules for low index-level codelength".

This is a strong claim but no theoretical justification was given. It is not clear nor obvious how the Occam's razor concept can be rigorously formulated in (or satisfied by) a neural clustering objective. As was done in Tsitsulin et al. 2023, the authors should formally argue how their objective avoids the collapse condition (all nodes in singleton clusters or in the unity cluster).

(3) The authors claim that a contribution of their approach is the ability to return overlapping cluster assignments. However, this is true of any neural clustering method with soft clustering assignments, including that of Tsitsulin et al. 2023. Can the authors compare the results in Fig 2 with those obtained by DMoN? If those obtained by NeuroMAP appear better, intuitive explanation of the improvement should also be stated.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new community detection algorithm based on the map equation that is the objective function of the well-known Infomap algorithm (Rosvall and Bergstrom, 2008). It treats the map equation as the (differentiable) loss function of graph neural networks for hard and soft clustering. Experimental results demonstrate the effectiveness of this method.

### Strengths
The idea of combining an information-theoretic cost function for clustering with neural networks is new.

### Weaknesses
(1) The presentation of this paper is quite poor. The notations in the Map Equation Loss section, which is the most significant part of this paper, are totally confusing.

(2) The description of Neuromap is too compressed. The details of GNNs with the map equation loss are missing.

(3) The experimental results are not convincing. The results of Neuromap in Figure 1, Tables 2 and 3 are hard to say competitive. It seems that the original Infomap algorithm performs better on many benchmarks.

### Questions
(1) In the Map Equation Loss section, what does the boldface $\textbf{A}_{i,j}$ mean? Is $\textbf{p}$ a vector or matrix? What is the definition of flow matrix? What does $\propto$ mean?

(2) Can you provide more details on the neural networks?

(3) How do you identify the overlapping communities in your algorithm?

(4) What is the efficiency of Neuromap in the experiments?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper discusses the application of deep learning and graph neural networks (GNNs) to community detection and graph clustering tasks. It highlights the under-explored nature of graph clustering as a primary task for GNNs and the limitations of existing approaches in identifying meaningful clusters. The authors propose a method that bridges the gap between deep learning and network science by optimizing the map equation, an information-theoretic objective function for community detection. The method proposed by the paper is generally novel to me, but the overall way that the paper conveys its idea remains a lot of ambiguity and the results need to be discussed more comprehensively.

### Strengths
(1)	The paper tries to use a novel approach to tackle the graph clustering problem, which is significant and has many real-world applications. The paper has addressed the significance of the problem properly. Related works are discussed properly also.
(2)	The paper tries to employ the map equation to solve the conventional graph clustering problem. In this process, the paper makes the optimization process differential to adapt the advanced GNNs to this process. The method is generally novel to me.
(3)	The experiments show that the performance of the proposed model is roughly good.

### Weaknesses
(1)	Paragraph 3 of “Introduction”: I don’t think the community detection using GNNs is “under explored”. There are a few works for this task such as [1], [2], [3], [4] and those discussed in the first paragraph of “Related work”.
(2)	Paragraph 1 and 2 of “Background”: I’m still confused about the goal of the map function. For example, what is the “per-step description length”? What is “Huffman code”? I would suggest maybe the author could introduce this in more detail in Appendix.
(3)	Paragraph 3 of “Background”: I would suggest the author to add a figure to illustrate the whole process discussed in the paragraph to make it more readable.
(4)	In “The map equation goes neural”, the paper introduces “S_{n x s}” without introducing s. I would encourage the author to define s the first time they use it.
(5)	In “The map equation goes neural”, I’m still confused about how the model learns S. The paper claims that S is learned via MLP or GNN, but S is a soft cluster assignment matrix. 
How could we learn a matrix using MLP or GNN? Is it an output from MLP or GNN? If so, what is the input?
(6)	What is the advantage of the proposed model over traditional ones such as KNN and DeepWalk? The paper discusses the existing approaches in paragraph 2 of “Introduction”, but does not mention the motivation of the proposed one. To me the complexity of KNN is O(nd), where d is the feature dimension, whereas the proposed method has the complexity of O(n^2), which is worse than KNN.
(7)	The results in Table 2 show that DmoN has superior performance than the proposed method in many settings. Why? The paper should discuss this. Also, the proposed method performs badly in “arXiv” dataset, which is also not discussed.
(8)	I would suggest the authors put the caption of the table on the top to make the presentation more formal.
[1] Bruna and Li, Community detection with graph neural networks
[2] Sun et al., Graph neural network encoding for community detection in attribute networks
[3] Luo et al., Detecting communities from heterogeneous graphs: A context path-based graph neural network model
[4] Yuan et al., Community detection with graph neural network using Markov stability

### Questions
Please refer to my comments in “Weakness”.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
