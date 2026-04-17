# Review

## Summary
This paper proposes a new graph learning paradigm, named forest-based graph learning (FGL). The key idea is to reinterpret message passing on a graph as transportation over spanning trees, which naturally facilitates long-range knowledge aggregation. Several trees can capture complementary topological pathways. The authors also provide theoretical analysis to show that as edge-homophily estimates improve, the induced distribution biases towards higher-homophily trees, which enables generating a high-quality forest by refining a homophily estimator. Experiments show that FGL achieves comparable results against SOTA counterparts on semi-supervised node classification tasks while remaining efficient.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
S1. The paper is well-written and easy to follow.

S2. The idea of using a forest of trees to model a graph is interesting and novel.

S3. Theoretical analysis is provided to show that as edge-homophily estimates improve, the induced distribution biases towards higher-homophily trees, which enables generating a high-quality forest by refining a homophily estimator.

S4. The proposed FGL has comparable or better accuracy than SOTA methods, and has lower running time.

## Weaknesses
W1. It is not clear why using a forest of trees to model a graph can achieve better accuracy and lower running time. It is suggested to add more discussions and intuitions.

W2. It is not clear why the proposed tree aggregator can achieve linear time complexity. It is suggested to add more details and discussions.

W3. The proposed method has several hyperparameters, e.g., the hyperparameters in Eq. (9) and (10). It is not clear how to set them in practice. Are they sensitive?

W4. The experiments only focus on semi-supervised node classification. It is not clear whether the proposed method can also achieve good performance on other tasks, e.g., supervised node classification, graph classification, link prediction.

## Questions
Please see the weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4