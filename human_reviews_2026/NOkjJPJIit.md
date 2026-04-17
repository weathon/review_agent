# Active Learning for Decision Trees with Provable Guarantees

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
This paper advances the theoretical understanding of active learning label complexity for decision trees as binary classifiers. We make two main contributions. First, we provide the first analysis of the **disagreement coefficient** for decision trees—a key parameter governing active learning label complexity. Our analysis holds under two natural assumptions required for achieving polylogarithmic label complexity: (i) each root-to-leaf path queries distinct feature dimensions, and (ii) the input data has a regular, grid-like structure. We show these assumptions are essential, as relaxing them leads to polynomial label complexity. Second, we present the first general active learning algorithm for binary classification that achieves a **multiplicative error guarantee**, producing a $(1+\epsilon)$-approximate classifier. By combining these results, we design an active learning algorithm for decision trees that uses only a **polylogarithmic number of label queries** in the dataset size, under the stated assumptions. Finally, we establish a label complexity lower bound, showing our algorithm’s dependence on the error tolerance $\epsilon$ is close to optimal.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper considers the problem of active learning on decision trees. There are two main contributions of the paper. In the first contribution, the authors derive new bounds on the disagreement coefficient of the class of decision trees with a fixed depth and disjoint decision parameters at each node. This allows us to use existing algorithms and guarantees for active learning of decision trees that use the disagreement coefficient of the hypothesis class. In the second contribution, the authors propose a new algorithm for learning the optimal classifier upto a multiplicative error, as opposed to additive error, which is the common metric in learning theory. The authors provide proofs of their claims and combine the two contributions to obtain new algorithms with theoretical guarantees on sample complexity for active learning on decision trees.

### Strengths
I think the contribution of disagreement coefficient of decision trees is an interesting and useful contribution that can help us better understand learning using decision trees. Moreover, this will allow us to better contextualize decision trees among the other widely studied hypothesis classes for learning.

The multiplicative error bounds are also interesting, especially for the cases when the error of the true hypothesis is very small (or zero).

### Weaknesses
The paper does not have any glaring weakness but I do have some questions (see next section).

### Questions
I have several questions:

- I am still a bit confused about the setup. So in your setting, the learner has access to a fixed dataset $S$ consisting of $n$ unlabeled data points.  And whenever the learner wants the label of some point, it can query an oracle that returns a label $y \sim \mathbb{P}(y|x)$, based on some distribution?

- What does curly R in the algorithms denote? (This is likely somewhat related to the previous question)

- Theorem A.1. only holds when the distribution $\mathbb{P}(Y|X)$ is sub-Gaussian. In learning theory literature, this is more commonly referred to as the Massart noise condition. In this scenario $|\Pr(y|x) - 1/2| > \eta$ for some $\eta > 0$ and all $x, y$. This is something that should be clarified both in your upper and lower bound. There are also less benign noise conditions (Tsybakov conditions) where the dependence on $\varepsilon$ will get worse.

- The terms $\mathrm{err}(h)$ and $\widehat{\mathrm{err}}(h)$ have not been explicitly defined.

- How is the proposed algorithm for stump classifier different from a classical binary search (with repeated queries for noise)?

- In typical scenarios in active learning, it is assumed that the samples come from underlying distribution. In your setting, you have sort of fixed the distribution to the empirical dataset. While this might be fine, but does it not automatically restrict the number of hypotheses your hypothesis class? Does this not make your problem simpler in some sense?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the classic pool-based active learning problem, where given a pool of unlabeled examples, one wants to design an adaptive algorithm that makes as few label queries over the pool as possible. This paper focuses on active learning for decision trees, a broadly studied class. Specifically, this paper contains two main results.
1. For structured decision trees and structured datasets, this paper develops the disagreement coefficients of the class of decision trees, using the framework developed by prior works, which gives a label complexity upper bound for active learning decision trees.
2. This paper generalizes the classic active learning framework, which considers additive error, to multiplicative error, and develops an algorithm for the upper bound.

These results are mathematically nontrivial, and the paper is written clearly.

### Strengths
Decision tree is an important and broadly studied class and the theory of active learning for decision trees has not been well understood yet. This paper makes progress in this direction by considering the disagreement coefficient framework proposed by Hanneke. For structured decision trees and structured distributions (uniform distribution over grid point), this paper gives an upper bound on the disagreement coefficient, which gives an upper bound on the label complexity of the problem. Furthermore, this paper also propose an algorithm that aims to achieve a multiplicative error instead of additive error, which is broadly considered in the literature. Overall, these results seem technically non-trivial and the proofs are complete.

### Weaknesses
Although technically solid, I am not sure if the contribution is significant enough. 

1. For the disagreement coefficient of the decision trees, this paper places structural assumptions on both the type of decision trees and the datasets. This looks a bit too strong. In particular, if the marginal distribution is structured, then the disagreement coefficient does not characterize the min-max label complexity for the problem (for example, a halfspace has disagreement coefficient $\sqrt{d}$ under uniform distribution over the sphere, which leads to a label complexity $d^{3/2}$ using Hanneke's framework, but clearly there are algorithms with label complexity $O(d)$). This makes the motivation of bounding the disagreement coefficient a bit confusing, especially given that the bound is exponentially large and grows with respect to the size of the grid.

2. The motivation for considering multiplicative error is not very clear to me. Specifically, the label complexity upper bound for considering a multiplicative error depends on the size of the dataset $n$. In the realizable setting, a hypothesis with $(1+\epsilon)$ multiplicative error means it has $0$ prediction error over the distribution $D$, which statistically needs a dataset with unbounded size to learn, which makes the label complexity bound meaningless.

### Questions
1. This paper considers learning over a discrete dataset, which is a bit different from the classic generalization setting, where one has a dataset sampled from a distribution and one considers the generalization error over the distribution. This difference looks important, since the distribution considered in this paper is uniform over an integer grid and the disagreement coefficient depends on the size of the grid. What if we consider learning a decision tree over the uniform distribution over a box? Does this blow up the disagreement coefficient?

2. The above difference also makes learning upto a multiplicative error a bit confusing. Since a dataset is usually generated from a distribution, in the low-noise rate setting, it seems to me if we consider the generalization error, then learning to a $(1+\epsilon)opt$ error is not realistic, as it needs an unbounded number of samples to achieve this. Can you comment on this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the problem of actively learning decision trees and the problem of establishing multiplicative error bounds for general binary classification in active learning. The authors provide the first rigorous analysis of the *disagreement coefficient* for decision trees, a parameter that controls the label complexity in active learning. They bound this coefficient as $O(\ln^d(n))$, under some assumptions: (i) the input data must possess a regular, grid-like structure, and (ii) each root-to-leaf path in the decision tree must query distinct feature dimensions. The paper proves these assumptions are necessary; relaxing them leads to significantly higher (polynomial) label complexity. The authors introduce the first general active learning algorithm for binary classification that achieves a *multiplicative* error guarantee. This algorithm produces a $(1 + \epsilon)$-approximate classifier, meaning its error is at most $(1 + \epsilon)$ times the error of the optimal classifier. This framework offers stronger accuracy control than traditional additive error models. By combining these results, the authors design an active learning algorithm for decision trees that requires only a polylogarithmic number of label queries, provided the structural and data distribution assumptions hold.

### Strengths
- The results are interesting and make progress on important problems in learning theory.

- The techniques are interesting and non-trivial.

- Overall I found the paper well-written.

### Weaknesses
- The bounds are a bit unsatisfactory.

- The paper might be a bit hard to follow for some members of the ICLR community who are less on the theory side.

### Questions
- Is dependence on the VC dimension in Theorem 1.2 necessary?

- Is exponential dependence on $d$ in Corollary 1.3 necessary?

- The algorithm boxes are a bit notation heavy.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates active learning for decision trees, providing theoretical bounds on the disagreement coefficient and proposing an analysis based on multiplicative error guarantees.

### Strengths
- The problem addressed is inherently interesting: active learning for non-linear, structured models like decision trees is a challenging and important direction.  

- The choice to use multiplicative (rather than additive) bounds is conceptually appealing, as it aligns with realizable-case analyses and could, in principle, yield sharper guarantees.

### Weaknesses
- **Limited theoretical novelty and impact**  
The theoretical contributions are incremental. The derived bounds on the disagreement coefficient $\theta = O((\ln n)^d)$ are relatively weak and pessimistic: $(\ln n)^d$ can grow very quickly even for moderately large depth, making the guarantee practically meaningless in realistic settings. Moreover, no *lower bounds* are provided, leaving it unclear whether the obtained rates are at all tight or informative.

- **Strong and unrealistic assumptions**  
The analysis critically depends on highly restrictive assumptions: discrete and regular input domains, unique features per root-to-leaf path, and bounded tree depth. Such conditions are far from how decision trees are actually used in practice (e.g., CART, random forests, or gradient-boosted trees reuse features and operate over continuous domains). As a result, the results have very limited applicability.

- **Intractable use of the disagreement region**  
The proposed Algorithm 2 relies on having access to disagreement regions. While this is standard in theoretical active learning, for decision trees the version space is exponentially large, and computing or even approximating $\text{DIS}$ is computationally hard. The paper does not propose any practical surrogate or heuristic for this step, making the algorithm purely theoretical and not implementable in realistic settings.

- **Lack of empirical validation**  
While pure theoretical papers are completely acceptable, here the derived results are rather fragile (strong assumptions and exponential dependencies). On the other hand, no convincing experiments or simulations are provided to demonstrate the practical relevance of the theoretical claims. The “empirical study” mentioned in the appendix is neither summarized nor discussed in the main text. Without empirical evidence, it is hard to assess whether the proposed insights translate into meaningful performance gains.

- **Venue suitability**  
This paper would be more suitable for a theoretical conference, where the focus on formal guarantees would be appreciated. However, even in that context, the work would likely be borderline due to the limited novelty, overly strong assumptions, and the lack of accompanying lower bounds.

While the topic is of potential interest, the present paper does not substantially advance the theory. The derived guarantees are fragile, the assumptions unrealistic, and the theoretical improvements marginal. Combined with the lack of lower bounds and the poor presentation quality, the work currently falls below the bar for a strong theoretical contribution.

### Questions
See the weaknesses section.

How do you define the calligraphic R appearing in Algorithm 1 and 2?

### Soundness
3

### Presentation
2

### Contribution
2
