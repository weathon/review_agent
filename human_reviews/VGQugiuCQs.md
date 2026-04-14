# Fair Clustering in the Sliding Window Model

- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6

## Abstract
We study streaming algorithms for proportionally fair clustering, a notion originally suggested by Chierichetti et al. (2017), in the sliding window model. We show that although there exist efficient streaming algorithms in the insertion-only model, surprisingly no algorithm can achieve finite ratio without violating the fairness constraint in sliding window. Hence, the problem of fair clustering is a rare separation between the insertion-only streaming model and the sliding window model. On the other hand, we show that if the fairness constraint is relaxed by a multiplicative $(1+\varepsilon)$ factor, there exists a $(1 + \varepsilon)$-approximate sliding window algorithm that uses $\text{poly}(k\varepsilon^{-1}\log n)$ space. This achieves essentially the best parameters (up to degree in the polynomial) provided the aforementioned lower bound. We also implement a number of empirical evaluations on real datasets to complement our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the problem of fair clustering in the sliding window model. First, it presents the inapproximability of the problem. Then, it proposes a new coreset-based approximation algorithm for the problem by relaxing the fairness constraint. Finally, some experimental results are provided to show that the proposed algorithm is better than trivial uniform sampling.

### Strengths
S1. This paper addresses an open theoretical problem - whether the proportional fairness constraint and the sliding window model are compatible in sublinear space - by providing the inapproximability result.

S2. The proposed algorithm is the first known one with an approximation guarantee for fair clustering in the sliding window model.

S3. The theoretical results are thorough and solid.

S3. The paper is generally well-written and organized.

### Weaknesses
W1. Although the theoretical part of this paper is sound and solid, the experimental part is highly insufficient.
- At least, two additional types of baselines should be compared: (1) the algorithms for sliding-window clustering without fairness such as those in [Borassi et al., 2020; Epasto et al., 2022;  Woodruff et al., 2023], which indicates "the price of fairness", and the algorithms for offline and insert-only streaming with fairness such as those in [Chierichetti et al., 2017; Schmidt et al., 2018; Huang et al., 2019], which provides lower bounds of clustering costs and presents the challenge of the sliding-window model. Uniform sampling is a too-weak baseline for coreset-based clustering.
- How about the performance of the proposed algorithms on datasets with higher dimensions (e.g., [Census](https://archive.ics.uci.edu/dataset/116/us+census+data+1990) and synthetic data in about 10-100 dimensions)?
- Details about experimental setup and implementation (such as dataset preprocessing and code availability) are concise.
- The efficiency results (e.g., running time and coreset size) and scalability (e.g., performance w.r.t. k and d) are not provided. For example, the clustering cost, runtime, and coreset size w.r.t. different values of k and d can be presented in figures or tables.
- According to the theoretical results, it seems that the proposed algorithms can work for multiple $l>2$ attribute groups. However, the experiments only use a binary attribute for each dataset.

W2. Minor presentation problems. Please double-check the paper carefully. Just one example is presented here.
- In the abstract, "*we show that if the fairness constraint by a multiplicative $\varepsilon$ factor, ...*" missing "is relaxed"?

### Questions
See the questions listed in W1.

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
2

### Summary
In the paper, the authors consider solving the problem of achieving (1+\epsilon) multiplicative approximate fair clustering in the fixed-size sliding window model. The author proved that there exists a sublinear space algorithm to solve the ((1-\epsilon)\alpha, (1+\epsilon)\beta) clustering problem. The core solution builds on top of online corset algorithms. At a high level, it first constructs a set of sketches to approximate the cluster, then decomposes the clusters into rings, and lastly union the uniform samples from the rings.

### Strengths
The paper studies an important problem with praticial impacts.

It not only presents novel theoretical insights but also implemented the algorithm. The evaluation showcased that the proposed algorithm consistently achieves better results compared to a uniform sampling baseline.

### Weaknesses
It would be nice to introduce a bit more on the (Augemented) Meyerson Sketches.

Please add more discussions and implications of multiplicative/additive approximation error on applications..

### Questions
See weakness.

### Soundness
4

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
3

### Summary
This paper studies the fair clustering problem within the sliding window model, where the goal is to maintain approximation for the most recdent data points in a stream with minimum space compelxity used. This paper first identifies a unique separation in clustering performance between the insertion-only and sliding window models, specifically for fair clustering, where no algorithm can achieve a bounded approximation ratio under strict fairness constraints without requiring linear space. To address this issue, ths paper then introduces an approximation algorithm that achieves a $(1 + \epsilon)$-approximation for fair clustering by allowing slight relaxation in fairness constraints, using sublinear space in $poly(k, \epsilon^{-1}, log n)$. Theoretical contributions include a strong lower bound proof, supported by communication complexity techniques, establishing the necessity of linear space for strict fairness in the sliding window model. Complementing these theoretical findings, this paper presents empirical results on real-world datasets, demonstrating that their approach significantly improves clustering cost and stability over a uniform sampling baseline, supporting the algorithm’s practical effectiveness in dynamic, fairness-sensitive applications.

### Strengths
1. Novelty and Insight: The paper identifies a crucial separation between fair clustering in different streaming models, offering new theoretical insights into the limitations of fair clustering in the sliding window model without any fairness violations.

2. Theoretical Contributions: The lower bound result is strong and well-supported by communication complexity arguments, which can have positive impact on the theoretical understanding of fair clustering.

3. Approximation Guarantee: The proposed approximation algorithm provides a $(1+\epsilon)$-approximation with sublinear space, which is a valuable contribution in the context of fair clustering and sliding windows.

4. Clear Presentation of Methods: The algorithms, particularly the coreset construction, are explained with clarity, and the theoretical proofs, while complex, are well-structured.

### Weaknesses
1. The space complexities used for the proposed method are dependent on the aspect ratio of the given clustering instances (an $O(\log\Delta)$ term, where $\Delta$ is the aspect ratio). Although, the aspect ratio can usually be assumed to be bounded by a polynomial function of the data size, it can be arbitrarily large in the worst case. 

2. The empirical comparison primarily involves a uniform sampling baseline. A more comprehensive evaluation with other clustering or local search methods would strengthen the experimental section and provide a clearer picture of the proposed method’s effectiveness.

### Questions
1. What impact does the window size $W$ have on performance? Since sliding windows dynamically retain recent data, it would be insightful to see more analysis on how varying $W$ affects the clustering quality and computational efficiency of the proposed algorithm.

2. Is there any methods that can remove the dependence of aspect ratio on the space complexity? How to deal with the case if the aspect ratio of the given clustering instance is large. In previous work, it was pointed out that the log function of aspect ratio can be linearly dependent on the data size $n$ [1]. Additionally, the method in [1] provides an efficient way for reducing the aspect ratio of any arbitrary clustering instance to bounded $poly(n, d)$ in static setting.

[1] Draganov A, Saulpic D, Schwiegelshohn C. Settling Time vs. Accuracy Tradeoffs for Clustering Big Data[J]. Proceedings of the ACM on Management of Data, 2024, 2(3): 1-25.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the fair clustering problem in the sliding window model, where the goal is to maintain an approximation for clustering the most recent data points while satisfying proportional fairness constraints. Specifically, given a $k$-clustering instance in the form of a data stream, the goal of this paper is to achieve a $(1+\epsilon)$-approximation to the optimal fair clustering for the dataset defined by the sliding window with minimum space complexity. However, ensuring fairness under strict space constraints in a data stream poses significant challenges. To tackle the challenges in sliding window model, this paper proposes an online assignment-preserving coreset construction method. The proposed method first reduces clustering in the sliding window model to an online coreset construction via a standard merge-and-reduce technique. Then, this paper introduces an algorithm for assignment-preserving coreset construction that processes each window as a suffix of the input stream at a particular time $t$. Finally, by constructing the coreset in reverse order across all time steps from $1$ to $t$, the resulting prefix of this online coreset serves as a valid coreset for the sliding window, achieving $(1+\epsilon)$-approximation for the fair clustering problem in sliding window model. This paper shows that if the fairness constraint is allowed to be violated by a multiplicative factor, there exists a $(1+\epsilon)$-approximate sliding window algorithm that uses only $poly(k\epsilon^{−1}log n)$ space. Empirical evaluations on real-world datasets further validates the effectiveness of the proposed framework, complementing the theoretical results.

### Strengths
The strengths of this paper can be summarized as follows.

1. The theoretical results of the paper are solid.

2. This paper establishes lower bounds for the fair clustering in sliding window model.

3. The proposed method achieves near-optimal clustering performances with  $(1+\epsilon)$-approximation on clustering quality guarantees while the space complexity nearly matches the lower bound provided.

4. The proposed method uses sublinear space in sliding window model, which can be used for handling large, dynamic datasets that require efficient memory usage.

### Weaknesses
1. The proposed algorithm uses a multiplicative relaxation rather than an additive violation for fairness constraints, which is slightly different from previous fair clustering algorithms.

2. Although the theoretical results nearly match the lower bound, the space complexity still depends on the aspect ratio of the given clustering instances. 

3. The paper lacks a sufficient number of comparison algorithms, making the experimental results less convincing. Additionally, the parameter choices and values for $\alpha$ and $\beta$ are not specified, limiting the reproducibility and clarity of the experimental parts.

### Questions
Q1: The proposed algorithm violates the fairness constraint by a multiplicative factor, which is slightly different from previous fair clustering algorithms with additive violations. Does multiplicative loss lead to better approximation ratios than previous algorithms with additive loss? What happens when only additive violation is allowed for group fairness constraints in the sliding window model, as achieving an approximation ratio of $1+\epsilon$ for group fair clustering is challenging.

Q2: As mentioned in the paper, the prefix property of online assignment-preserving coresets can tolerate a $1±\epsilon$ relative error in the weights. Does this property make the proposed algorithm easier to implement compared to the algorithm in [1]?

Q3: The comparison algorithms used this paper are limited, which makes the numerical experiments not convincing enough. The authors should add more sliding window algorithms, fair algorithms or heuristic algorithms as comparisons to make the experimental parts better.

Q4: How to determine the value of parameter $k$? Does different choices of $k$ influence the experimental results of the proposed algorithms? What are the choices of the values for $\alpha$ and $\beta$.

[1] Woodruff, David, Peilin Zhong, and Samson Zhou. "Near-Optimal k-Clustering in the Sliding Window Model." Advances in Neural Information Processing Systems 36 (2024).

### Soundness
3

### Presentation
2

### Contribution
4
