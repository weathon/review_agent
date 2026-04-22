# Personalized Prediction By Learning Halfspace Reference Classes Under Well-Behaved Distribution

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
In machine learning applications, predictive models are trained to serve future queries across the entire data distribution. Real-world data often demands excessively complex models to achieve competitive performance, however, sacrificing interpretability. Hence, the growing deployment of machine learning models in high-stakes applications, such as healthcare, motivates the search for methods for accurate and explainable predictions. This work proposes a Personalized Prediction scheme, where an easy-to-interpret predictor is learned per query. In particular, we wish to produce a "sparse linear" classifier with competitive performance specifically on some sub-population that includes the query point. The goal of this work is to study the PAC-learnability of this prediction model for sub-populations represented by "halfspaces" in a label-agnostic setting. We first give a distribution-specific PAC-learning algorithm for learning reference classes for personalized prediction. By leveraging both the reference-class learning algorithm and a list learner of sparse linear representations, we prove the first upper bound, $O(\mathrm{opt}^{1/4} )$, for personalized prediction with sparse linear classifiers and homogeneous halfspace subsets. We also evaluate our algorithms on a variety of standard benchmark data sets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies a new personalized prediction problem where the objective is defined by some PAC-like inequality. Specifically, the hypothesis class is considered to be the class of sparse linear functions and the "personalization" constraint is modeled by halfspaces. The paper draws a connection with the reference class learning problem and builds algorithm for the personalized prediction problem based on some improvement of algorithms therein.

### Strengths
The paper is very technical but well-written. In my opinion, the contribution of the paper lies in two-fold:

- The formulation of the personalized prediction problem. The paper proposes a new formulation, and it has not been studied in the literature. 
- The formulation draws connection with the reference class learning problem. The paper twists Huang and Juba (2025) and improves the algorithm and adapts it to the context of personalized prediction problem.

### Weaknesses
For the context of writing this review, there is a clear stream of papers working on the topic of this paper: Diakonikolas et al., 2020b;c; 2021; 2022; 2024; Juba and Li (2020); Huang and Juba (2025). I, as the reviewer, am not familiar with this stream of work; I have moderate exposure to learning theory and write papers using learning theory. So my judgment could be, on the one hand, biased by the limited scope of my own background, but on the other hand, it could be representative of the general ML people who do research at the intersection of theory and methodology.

In this light, my main concerns are:
- The studied formulation is too niche and doesn't provide much practical insight. Personalized prediction is nevertheless an important problem, but the algorithms and analyses in this paper don't provide much insight for practical usage of these algorithms. 
- The theoretical contribution along the existing literature above is unclear from my reading of the paper. I understand that the problem studied is new and can't be resolved by directly applying the existing algorithms. But the question is whether the new treatment and analysis in this paper is generalizable to other theory problems, say, to be used for other problems to make it possible/improve the analysis therein. 

To this end, I feel the paper is more suitable for COLT, where the technical contribution can be better accessed.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an algorithm for personalized prediction for more interpretable learning by focusing on a subset of feature points. Its algorithm combines a robust list learner for the class of sparse linear classifiers and a reference class learner. Specifically, the learning is based on projected gradient descent with an analysis that each step of the update will improve the correlation between the candidate classifier and the assumed ground truth that has small prediction error with respect to the underlying distribution. It is shown that the algorithm runs in polynomial time with respect to parameters $d, 1/\epsilon, 1/\delta$.

### Strengths
The paper uses (mostly) existing techniques and builds an algorithm to solve the personalized prediction problem. The algorithm works under the more general distributional assumptions, i.e. well-behaved marginal distribution that resembles uniform, Gaussian, and many log-concave distributions, by scarifying a small amount of accuracy. From the technical perspective, it improves upon the existing algorithmic guarantees over that of Huang and Juba (2025) by showing the existence of the candidate point $x$ in predicted set $S$.

From a broader perspective, the paper builds a learner for a more interpretable model, using sparse linear classifiers. It maintains the accuracy at the same time by personalizing on a specific query $x$. The theory is sound and the algorithms runs in polynomial time with empirical study on benchmark data sets.

### Weaknesses
Comparing to its prior work, e.g. Huang and Juba (2025), Diakonikolas et al. (2022), the contribution seems incremental. Besides a slight different setting, i.e. query $x' \in S$ and well-behaved distribution, the algorithm and analysis resume those were already in prior works. Hence, it is more like a new application of the existing techniques in a personalized learning setting. I wonder how broad the applications of the proposed algorithm can be.

### Questions
See weaknesses. What is the essential contribution for the proposed algorithms in personalized learning setting, besides interpretability? What is the key difference between personalized learners and simple invoking the traditional PAC learners for sparse linear classifiers? Some examples on specific $x$ where traditional PAC learner would fail yet the personalized learner can accurately and efficiently predict it, would help much more.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
this paper propose a personalized prediction framework for sparse binary classification, where instead of learning one universal model, the approach learns a dedicated classifier for each query by focusing on a relevant subpopulation that best represents the query. The classifier is also designed to be sparse and interpretable. The key idea is that focusing on such subpopulations, characterized by homogeneous halfspaces, makes learning and interpretation easier, since simpler representations can capture local structures within a subset of the data more effectively than a single global model. The authors also provide algorithms with theoretical guarantee and conduct numerical experiments.

### Strengths
1. The paper is well written and clearly structured, making both the motivation and technical developments easy to follow. The exposition of definitions, algorithms, and proofs is systematic and readable.

2. The paper provides algorithms generating useful results with theoretical guarantees, which is a non-trivial extension of existing conditional learning results to the personalized prediction setting.

### Weaknesses
The numerical section is a major weakness of the paper. Even after carefully reading the appendix, it remains difficult to understand what the experiments are actually demonstrating. This lack of clarity significantly undermines the overall quality of the paper, especially given that the theoretical section is otherwise well written.

1. From a conceptual standpoint, it is unclear how the numerical experiments connect to the theoretical results. Simply stating that “the results are lower” is too vague to be meaningful. The authors should explain why a lower value supports the theorem, and how the intuition from the theoretical analysis is reflected in the empirical behavior.

2. The authors also fail to properly explain Table 2, which appears to be the central empirical result. The statement “* indicates statistically significant improvement with 95% confidence (over SPARSE for PERS, and over PERS for the other baselines)” is confusing because the metric and experimental setup are never formally defined. Moreover, in the HYPO dataset, XGB (0.142*) and PERS (0.379*) differ by a large margin (nearly a factor of two) yet both are marked as significant. Without a clear description of the evaluation protocol, metrics, and comparison criteria, the empirical results are difficult to interpret and fail to substantiate the theoretical claims.

I recommend that the authors improve the numerical section in the appendix.

### Questions
The authors should also provide intuition on why the chosen UCI datasets are appropriate for supporting the theoretical results (provide reference if already validated in other works). Since the theory relies on assumptions such as well-behaved distributions and sparse feature relevance, it would be helpful to include illustrations or diagnostic plots showing whether the datasets approximately satisfy these assumptions (e.g., marginal distributions, sparsity patterns, or subspace structure). 

Additionally, what is the experimental setup? Readers need information about the parameter settings, number of trials, and criteria for statistical significance to properly evaluate the experiments.

### Soundness
3

### Presentation
3

### Contribution
2
