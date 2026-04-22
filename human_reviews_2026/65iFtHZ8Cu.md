# Discounted Online Convex Optimization: Uniform Regret Across a Continuous Interval

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Reflecting the greater significance of recent history over the distant past in non-stationary environments, $\lambda$-discounted regret has been introduced in online convex optimization (OCO) to gracefully forget past data as new information arrives. When the discount factor $\lambda$ is given, online gradient descent with an appropriate step size achieves an $O(1/\sqrt{1-\lambda})$ discounted regret. However, the value of $\lambda$ is often not predetermined in real-world scenarios. This gives rise to a significant \emph{open question}: is it possible to develop a discounted algorithm that adapts to an unknown discount factor. In this paper, we affirmatively answer this question by providing a novel analysis to demonstrate that smoothed OGD (SOGD)  achieves a uniform $O(\sqrt{\log T/1-\lambda})$ discounted regret, holding for all values of $\lambda$ across a continuous interval simultaneously. The basic idea is to maintain multiple OGD instances to handle different discount factors, and aggregate their outputs sequentially by an online prediction algorithm named as Discounted-Normal-Predictor (DNP). Our analysis reveals that DNP can combine the decisions of two experts, even when they operate on discounted regret with different discount factors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies Discounted Online Convex Optimization (OCO) to minimize the λ-discounted regret. The
authors introduce a Smoothed OGD (SOGD) algorithm to achieve uniform discounted regret $O(\sqrt{\log(T)/(1-\lambda)})$
across a continuous interval of discount factors without prior knowledge of $\lambda$. They also provide a novel analysis of DNP-cu under discounted payoff settings, showing that it can combine experts operating under different discount factors.

### Strengths
1. The application of DNP-cu to combine experts with different discount factors is novel and the analysis of DNP-cu under discounted payoffs (Theorem 2) is a key contribution.

2. The paper provides a uniform regret bound across a continuous interval of $\lambda$ with detailed and well-organized proofs in the appendix.

3. The paper is well-structured with clear motivation, problem setup and technical exposition. The presentation of figures and algorithms help illustrate the main ideas.

4. The authors provide clear statement of problem setting and experimental setup, supporting the reproducibility of both the theoretical and empirical findings.

### Weaknesses
1. The experiments conducted in the paper only compare the performance of SOGD with typical OGD. It would be valuable to compare with other adaptive or meta-learning baselines.

2. Computational Practicality: The theoretical bounds of this paper depend on several constants and assumptions, which may lead to limited practicality and expensive computational cost in high-dimensional or large-scale settings.

### Questions
1. Can the problem be extended to settings with strongly convex or exp-concave losses? What regret bounds could be achieved?

2. How sensitive is SOGD to the choice of $Z$ and $\tau$? Are there practical guidelines for setting these parameters in real applications?

3. How does SOGD compare empirically and theoretically with state-of-the-art algorithms with strong adaptive regret when evaluated under discounted regret?

### Soundness
2

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
4

### Summary
The authors consider the setting of online convex optimization with $ \lambda $-discounted regret. Prior work assumes known $ \lambda $ and shows online gradient descent achieves $O ( (1-\lambda)^{-\frac12} ) $ discounted regret. In this work the authors consider the setting of \emph{unknown} $ \lambda $ and show that a carefully designed aggregation of online gradient descent instances can achieve $ O ( \log T (1-\lambda)^{-1} ) $ discounted regret for any $ \lambda $ in a defined continuous interval.

### Strengths
* The paper studies an important open problem: that of achieving discounted regret bounds with an unknown $ \lambda $. The paper does a decent job of motivating that in some settings $ \lambda $ is truly unknown and is not just a tunable hyperparameter.

* In contrast to typical aggregation mechanism in meta algorithms with multiple instances of an online gradient descent or experts algorithm, this work relies on the less well-known mechanism of discounted normal predictor with conservative updates. The authors show that this aggregator, works used in a particular order of $ \lambda $.

* The authors also provide some empirical results that reflect their theoretical findings.

* The authors rely on existing algorithms (e.g., aggregation with DNP-cu was already used in Zhang et al. (2022)). However, they prove an important property of DNP-cu: that it can aggregate across different discount factors.

### Weaknesses
* There is a related line of work on online convex optimization with unbounded memory (Kumar, Dean, Kleinberg, NeurIPS 2023). One special case is $\rho$-discounted infinite memory, where the loss in each round depends on the entire history of decisions and each past decision is weighted by a geometric factor of $\rho$. This paper does not discuss similarities and differences from this line of work.

* Algorithm 3 is hard to read - I had to keep jumping around to look at the algorithm and at the equations that it references. It would be easier for the reader if you wrote the equations inline.

* Since the algorithm is the same in Zhang et al. (2022) but you prove an additional property of DNP-cu that allows it to be used in the $\lambda$-discounted setting, can you discuss a bit more what are the differences between this work and Zhang et al.?

### Questions
* Please see the weaknesses section about lack of discussion on online convex optimization with unbounded memory. Could you discuss when using one model over the other is preferable? Can their techniques be extended to $\lambda$-discounted regret? Can your techniques be used to derive results for their model?

* Can you expand a bit more on why the ordering is crucial when aggregating the difference OGD instances?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies discounted online convex optimization and demonstrates that it is possible to achieve regret results for unknown bounded discount factors. It achieves this by analyzing and utilizing the existing algorithm DNP.

### Strengths
Originality:
The paper creatively applies the DNP algorithm to the discounted optimization problem and remove the need to know the discount factor.

Quality:
The submission seems technically correct. Experiments are a plus. The algorithms and the experimental settings appear reproducible.

Clarity:
The submission is clear in general.

Significance:
Theoretical novel findings in the form of discounted regret results for unknown discount factors are obtained.

### Weaknesses
I am leaning towards rejection. Below are the reasons.

Utilizing DNP-cu seems to help in arriving to a clean result, but it seems any combiner could have worked. I am not sure if the issue of different discounted performance measures as explained in Figure 1 is as great as advertised. Since the combined $\lambda$ values have a difference of $1/T$, the regret redundancy propagating due to mismatches seems to be finite at each combination node, similar to DNP-cu.

Aside from that, exponentially growing step-sizes and iterative combination of experts (with intermediate experts in the mix) are established methods in the literature.

### Questions
Questions:

Page 6 Line 301: is the effective window size the sum of the discount coefficients?

Page 6 Line 305: for $\lambda$, $\tau$ is related to the lower-bound, while the footnote is talking about the upper-bound. Is there a mistake here?

Page 8 Line 409: why $Z=1/T$, why not smaller?


Suggestions:

Give more substance to the need to use DNP as opposed to another mixture of experts algorithm.

### Soundness
3

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
The paper addresses online convex optimization (OCO) with discounted regret, where recent losses are weighted more heavily than distant ones via a discount factor $\lambda$. Existing papers assume the discounting factor is known, whereas this manuscript considers the unknown $\lambda$ case. The main idea is to run many instances of SOGD with a specific lambda, and then run a meta selection algorithm to aggregate the outcome.

### Strengths
Addresses a clear open problem in discounted OCO, by providing a discounted regret bound for uniform range of lambda. The learner does not know lambda a priori. 

The technical challenges of the problem were clearly presented. The prior aggregation framework needed to operate under a uniform performance metric, whereas in this paper each expert has a different metric. 

Provide a step by step derivation on the motivation behind the algorithm design which is insightful.

 Provides rigorous theoretical analysis and complete proofs. Simulations are provided as well.

### Weaknesses
-	I encourage the authors to provide **more concrete** motivations on the relevance of the problem of unknown lambda, why it is practically relevant other than the mere theoretic interest. 
-	As mentioned in the intro, part of the motivation is that the user’s preference might change, indicating a time-varying lambda. Can the paper be extended to the time-varying lambda case? i.e. the learner has some feedback signal that is indicative of lambda, and can adapt itself to optimize the regret with time-varying lambda?

### Questions
-	(17) (18) are not too obvious, providing more details will help. 

-	Compare theorem 3 to theorem 1, the dependence on lambda is worse. I wonder whether the bound is tight or if there are potential ways to further improve it.

### Soundness
3

### Presentation
3

### Contribution
3
