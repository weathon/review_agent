# Optimal spherical codes for locality-sensitive hashing

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 1, 3, 3

## Abstract
In the realm of Locality-Sensitive Hashing (LSH), striking the right balance between computational efficiency and accuracy has been a persistent challenge. Most existing unsupervised methods rely on dense representations, which can lead to inefficiencies. To tackle this, we advocate for the adoption of sparse representations and introduce the use of quasi-Optimal Spherical Codes (OSCs) to minimise space distortion. OSCs strive to maximise the minimum angle between any pair of points on the hypersphere, ensuring that the relative angular information between data points is preserved in the representation, which is particularly valuable in tasks involving cosine similarity. We employ Adam-based optimisation to obtain these codes and use them to  partition the space into a $k^\text{th}$-order Voronoi diagram. This approach consistently outperforms existing methods across four datasets on $K$-nearest neighbors search with cosine similarity, while capping the query time for a given embedding size.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method that partitions the surface of the unit hypersphere into a into a quasi-centroidal hyperspherical voronoi diagram with maximally separated generator points, which is beneficial for ANN search especially with cosine similarity. Experiments on four datasets show that OSC-LSH aperforms better than the conventional LSH approaches, such as PR-LSH and FlyLSH.

### Strengths
The idea of optimal spherical codes for LSH is a natural advantage for approximate nearest neighbor search.

### Weaknesses
The writing is so bad with many missings or minor errors, or unclear-ness.

### Questions
What is the relationship between OSC and Spherical Hashing [1]?

[1] Jae-Pil Heo, Youngwoon Lee, Junfeng He, Shih-Fu Chang, Sung-Eui Yoon: Spherical hashing. CVPR 2012: 2957-2964

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes to construct LSH for nearest neighbor search by finding a set of points on the unit sphere. As separating these points on the sphere is difficult, the authors propose to learn these points by gradient descent.

### Strengths
1.	The idea is presented clearly.
2.	Using optimal spherical codes to construct LSH makes sense.

### Weaknesses
1.	Graph-based methods and vector quantization methods are widely used for vector search now and show significantly better performance than LSH. The authors state that “Whereas graph-based techniques perform very well in low dimensions, their performance is suboptimal in higher dimensions, and cannot be used in distributed settings.” This is NOT true! The HNSW paper shows that proximity graph works quite well for high dimension vectors, and the Pyramid paper uses proximity graph for distributed vector search. Also, the FAISS library uses many variants of PQ and the IVF index for vector search. I am not convinced of the practical impact of this paper unless the authors show that the proposed LSH achieves better time-recall curve than graph and quantization methods. Note that time-recall curve is a standard method to evaluate the performance of ANNS, and examples can be found in the HNSW paper.
[1] Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (HNSW)
[2] Pyramid: A general framework for distributed similarity search on large-scale datasets (Pyramid)
[3] Product Quantization for Nearest Neighbor Search (PQ)

2.	Experiments are very limited. (1) Nowadays, million or even billion scale datasets are common for vector search (check the Pyramid paper) but the authors use a datasets of 10,000, which are extremely small. (2) The proposed method needs to check every vector in the dataset to compute Hamming distance, which is infeasible for very large datasets. (3) Pls use standard time-recall curve to compare the performance of the authors in the main experiments. The influence of m/d can be used as a micro experiment to check the influence of parameters. 

3.	I think the paper is not ready for publication is its current state and would benefit from substantial revision for another conference. For instance, there are multiple missing references even in the introduction. Some statements are not logical or scientific due to the lack of checking. For example, “LSH…, where the imperative is to swiftly identify similar items without sacrificing accuracy.”, well, LSH actually sacrifices accuracy to identify similar items swiftly; “To balance between accuracy and query time, we can make sure the centroids used for the space partitioning are maximally separated”, balance is a vague term, and how does maximal separation leads to such balance?; “Existing LSH has …(ii) limited use cases”, what exactly are the use cases your proposed method can handle and existing LSH cannot?; “These methods however are limited for use with a specific hash length or embedding size”, why is the case? To my knowledge, at least random projection allows to handle arbitrarily long embeddings and flexibly set the length of the embeddings. 

4.	How does the method handle more general cases, i.e., when dataset vectors are not on the unit sphere? The optimization objective in (1) is different from the max-min objective of OSC. What is the relation of the solution produced by Algorithm 1 w.r.t. the solution of OSC? Can you show that it is a good approximation?

### Questions
I would suggest the authors to better prepare the manuscript for a serious submission.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Locality-sensitive hashing (LSH) is one of the most vital techniques for solving the approximate nearest neighbor (ANN) search. 
In this paper, the authors introduce OSC-LSH, a new LSH scheme to enhance search efficiency using Optimal Spherical Codes (OSCs). 
They argue that by employing optimal spherical codes, one can achieve better hypersphere partitioning and more discriminative hash codes for ANN search in high-dimensional spaces. 
They develop an optimization algorithm to find the approximate OSC for any embedding size $m$, comparing OSC-LSH with two baselines RP-LSH and FlyHash, and showcasing its benefits through empirical results.

### Strengths
1. Novel Approach: Using spherical codes as a foundation to improve the efficiency of LSH is a novel idea that has been less explored in prior work. Even though some researchers have considered cross-polytype, they did not provide a systematic investigation.

2. Practicality: An optimization method is proposed to identify the approximate OSC for any embedding size $m$, making OSC-LSH practical for different scenarios.

3. Empirical Results: The empirical results in the experiments demonstrate the superiority of the proposed method over traditional LSH methods in terms of both quality and speed.

### Weaknesses
W1. Dependency on Optimal Spherical Codes: The efficacy of OSC-LSH is tied to the quality of the spherical codes. However, as pointed out by the authors, the identification of the optimal spherical codes is computationally prohibitive, especially when $d$ (and $m$) is large. In cases where optimal codes are hard to find, this method might not offer significant improvements.

W2. Ambiguity in Distortion Definition: The paper frequently references the concept of "distortion" when discussing the benefits of using Optimal Spherical Codes (OSC) in LSH. However, a clear and concise definition of distortion is missing, making it difficult to fully understand and evaluate the proposed method's effectiveness. While Figure 1 shows the minimum distance in a self-comparison context, it doesn't provide a comprehensive view of the distortion in the mapping process from the input space to the output space.

W3. Concern of the Optimization Objective: Under the assumption that all data points are normalized, the Euclidean distance between any $c_i$ and $c_j$ is inversely proportional to their inner product. Thus, it is problematic to directly replace Euclidean distance in Equation (1) with the inner product in Equation (2). Is there any typo in Equation (2), e.g., removing the "$-$" or maximizing $L$?

W4. Unconvincing Empirical Results.

W4.1. Lack of End-to-End Comparison: While the paper compares the new approach with RP-LSH and FlyHash, it would be beneficial to see the end-to-end comparisons with other state-of-the-art ANN search methods, such as FAISS and HNSW.

W4.2. Scalability Concerns: Although the authors claim they have improved the efficiency issue, the paper doesn't delve deep into the scalability of the proposed method, especially when dealing with extremely large datasets.

W5. Poor Presentation: This paper seems to be written in a hurry way. For example, there exist many "???" and empty "()." The two Algorithms are isolated from the main paper, and no explanation is provided. In the experiments, they claim to use five datasets, but I failed to find the results on CIFAR10.

### Questions
Regarding W1:

Q1: Could you provide the time complexity of finding the OSCs and discuss any alternative way to accelerate this process?

Regarding W2:

Q2: Can the authors provide a rigorous definition of "distortion" as used in the context of this paper? How is it measured, and what makes it a suitable metric for evaluating the quality of the mapping?

Q3: Given the emphasis on minimizing distortion with OSC, could the authors provide comparisons with other methods in terms of this metric? This would offer a clearer perspective on the proposed method's advantages.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose OSC (Optimal Spherical Codes) as a technique for generating sparse representations that can be used for locality sensitive hashing. The code vectors are generated by an optimization procedure that maximizes the minimum angle between any pair of code points. They show that their approach outperforms FlyLSH and random projections LSH. 

While their technique is very similar to the approach proposed by Andoni et al., the main advantage seems to be the flexibility in choosing the embedding dimension.

### Strengths
There could be some (computational) advantage to the flexibility in using different embedding dimensions (though they have not explored this in the paper).

### Weaknesses
1) On the datasets they consider, there is not a marked improvement over the baselines.
2) It seems that they are performing an optimization procedure for computing the code vectors, but not actually using any data - it seems like if you are going through the trouble to optimize the code vectors then it might help to actually use the data (so that the code vectors chosen can reflect the actual underlying data distribution).

### Questions
It would be useful to understand if the flexiblity in the choice of embedding dimension actually produces interesting tradeoffs (recall vs computation, etc). If not what do you see as the main advantage over Andoni et al?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
