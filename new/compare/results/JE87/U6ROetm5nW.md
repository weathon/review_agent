# Review

## Summary
The paper studies the problem of kernel density estimation (KDE). KDE is defined as follows: Given a set of points $X$ in Euclidean space and a kernel $K(p, q)$, the goal is to build a data structure that approximates the following quantity: $\frac{1}{|X|} \sum_{p \in X} K(p, q)$. The paper provides a data structure that approximates the KDE within a multiplicative factor of $(1 \pm \epsilon)$ and achieves a trade-off between space and query time. The main idea is to reduce the KDE problem to a version of the approximate nearest neighbor (ANN) problem. The paper utilizes the asymmetric LSH construction of Andoni et al. to design a data structure for the ANN problem, which in turn leads to a data structure for the KDE problem.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The paper provides a data structure for the KDE problem that offers improved query time and space trade-offs compared to existing methods. The use of asymmetric LSH allows for flexible trade-offs between space and query time, which is a novel approach in the context of KDE.

2. The paper introduces a general framework for achieving trade-offs between query time and space for KDE, which is the first such trade-off for KDE.

## Weaknesses
1. The paper's technical contribution is limited. The main result of the paper follows from combining the reduction of KDE to ANN from [Charikar et al. 2020] with the time-space trade-offs for ANN provided by Andoni et al. 2017. While the paper does provide some generalization of these results, the technical novelty is limited.

2. The paper is poorly written. The main body of the paper is hard to follow, and many of the proofs in the appendix are incomplete. For example, Lemma 31 is stated without a proof, and the proof of Lemma 32 refers to terms and equations that are not present in the statement of the lemma. This makes it difficult to verify the correctness of the results.

## Questions
1. Can you provide a complete proof of Lemma 31? This would help in verifying the correctness of the main result of the paper.

2. Can you provide a more detailed comparison with the previous work of [Charikar et al. 2020]? Specifically, it would be helpful to understand how the query time and space trade-offs in this paper compare to those in [Charikar et al. 2020].

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4