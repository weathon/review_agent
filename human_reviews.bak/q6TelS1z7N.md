# Information-Theoretic Active Correlation Clustering

- Decision: Reject
- Scores: 6, 8, 3, 6

## Abstract
We study correlation clustering where the pairwise similarities are not known in advance. For this purpose, we employ active learning to query pairwise similarities in a cost-efficient way. We propose a number of effective information-theoretic acquisition functions based on entropy and information gain. We extensively investigate the performance of our methods in different settings and demonstrate their superior performance compared to the alternatives.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an active correlation clustering method based on information theory. It applies the mean-field approximation to approximate the Gibbs distribution and proposes four acquisition functions for active correlation clustering based on the approximation.

### Strengths
Strengths:
1. The idea of information-theoretic active correlation clustering together with the mean-field approximation is interesting. 
2. The experimental results are good.
3. The presentation and organization are good.

### Weaknesses
Weaknesses and Questions:
1. The setting where we can only access to noisy similarities of some pairs is interesting. However, it would be better to introduce some application scenarios in real world, and apply at least one of the real-world data sets in the experiments instead of synthetically adding some noises to the ground truth, because the oracles used in the experiments may not follow the distribution of real-world noises. 

2. The mean-field approximation method seems similar to the widely-used variational inference. What are the differences between the mean-field approximation and variational inference?

3. In many data sets, the paper only uses a subset. Why not use the whole data set? Cannot the proposed method handle large-scale data sets?

4. In some data sets of Figures 1 and 2, the results seem incomplete. For example, in Figure 1(g) and 1(h), the results of EIG-O and EIG-P are missing. Why?

### Questions
Please See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscript presents information-theoretic acquisition functions for active correlation clustering, where pairwise similarities are queried from an oracle to construct clusters. The authors introduce four acquisition functions: Entropy, EIG-O, EIG-P, and JEIG. mean-field approximations are adopted to make computations tractable. Through extensive experiments across multiple datasets and noisy oracles of various types, the proposed approaches show superior performance compared to existing methods.

### Strengths
The authors proposed novel information-theoretic acquisition functions to active correlation clustering. Mean-field approximations are derived for these functions.

The paper presents comprehensive empirical validation across diverse datasets. The authors discuss how noisy oracles with different types impact the performance of the model. 

The authors utilize clear mathematical notation and describe the algorithms in detail. The comprehensive appendices provide details in the implementation and in the experiments.

### Weaknesses
This manuscript proposes a correlation clustering method that operates solely on pairwise similarities, without utilizing sample features. A significant limitation is the method's extensive labeling requirements. As shown in the experiments, for datasets with no more than 1,000 samples, thousands of pairwise labels are needed to achieve acceptable performance. This high labeling burden makes the method impractical for real-world applications, where collecting pairwise labels is typically more resource-intensive than collecting instance-level labels. Given these practical constraints, the paper's impact is likely to be limited despite its contributions.

### Questions
How does the cluster imbalance impact the performance of the proposed method?

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
2

### Summary
The paper addresses an interesting problem: mapping data points into distinct clusters with high accuracy by querying the similarity between them a limited number of times. The objective function, as outlined in Equations (1) and (2), aims to minimize clustering disagreement. However, the rationale behind the objective function is somewhat unclear, as it appears that clustering disagreement may also be influenced by the scale of the data points.

The main theoretical guarantee presented is a local minimum in Theorem 3.1, though it is unclear how this guarantee compares to those in existing literature. Additionally, it would be helpful to understand the authors' considerations regarding the number of clusters, as how to infer the number of clusters from the structure of the similarity matrix graph.

Notably, this approach diverges from the usual practice of providing a guarantee for the distance between ground truth and predicted cluster centers. Achieving the guarantee in Theorem 3.1 involves multiple steps of relaxation by introducing concepts such as information gain, entropy, and mean-field approximation, which raises questions about the accuracy of the theorem.

The technical details in Algorithms 1 and 2 are not discussed. Algorithm 1 (on page 4) queries similarity through an oracle, producing a similarity matrix with both positive and negative values. A positive value indicates that a data pair belongs to the same class, while a negative one implies otherwise. This differs from the standard approach of thresholding positive similarity values, so I am curious about how this algorithm would perform in real-world applications. Additionally, the "budget," a critical parameter, is not discussed. In Algorithm 2 (on page 5), the criteria for defining the convergence of Q in Step 2 remain unclear.

The experimental results are a positive aspect of a theoretical paper.

### Strengths
The paper addresses an interesting problem: mapping data points into distinct clusters with high accuracy by querying the similarity between them a limited number of times. 

The experimental results are a positive aspect of a theoretical paper.

### Weaknesses
It is not clear if the objective function makes sense. The objective function, as outlined in Equations (1) and (2), aims to minimize clustering disagreement. However, the rationale behind the objective function is somewhat unclear, as it appears that clustering disagreement may also be influenced by the scale of the data points.


The technical details in Algorithms 1 and 2 are not discussed. For example, the "budget" in Algorithm 1 (on page 4)  a critical parameter, is not discussed.  In Algorithm 2 (on page 5), the criteria for defining the convergence of Q in Step 2 remain unclear.

### Questions
The objective function, as outlined in Equations (1) and (2), aims to minimize clustering disagreement. However, the rationale behind the objective function is somewhat unclear, as it appears that clustering disagreement may also be influenced by the scale of the data points.

The main theoretical guarantee presented is a local minimum in Theorem 3.1, though it is unclear how this guarantee compares to those in existing literature. 

It would be helpful to understand the authors' considerations regarding the number of clusters, as how to infer the number of clusters from the structure of the similarity matrix graph.

How do authors define "budget" in Algorithm 1? In Algorithm 2 (on page 5), the criteria for defining the convergence of Q in Step 2 remain unclear.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a new approach to correlation clustering with a limited budget for similarity queries between pair objects. The main contribution is the design of new techniques (the so-called acquisition functions) for selecting object pairs whose similarity should be computed by a noisy oracle and which can then be used as input to a correlation clustering algorithm.

### Strengths
- The considered problem is important and has practical applications.
- An interesting approach for the selection of object pairs whose similarity should be computed.
- Good experimental results.

### Weaknesses
- The approach is just a heuristic, there are no approximation guarantees, not even an attempt to discuss the quality of approximation from a theoretic point of view.
- The writing can be improved. For example, Theorem 1 is not central to the presented approach, it is rather about an efficient approximation of Gibbs sampling. I think the presentation should first focus on the 4 acquisition functions and then explain how to approximate Gibbs sampling.
- I couldn't understand why only noisy oracles are considered, see my question below.

### Questions
I don't understand why only noisy oracles are considered. I would like to know how the approach performs if the oracle returns the precise pair similarity. This is a realistic and practical use case and I think it should be considered. Would we be then able to obtain some approximation guarantees, maybe after assuming some properties of the acquisition function?

### Soundness
4

### Presentation
3

### Contribution
2
