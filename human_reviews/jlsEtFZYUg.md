# Theoretical Limitations of Ensembles in the Age of Overparameterization

- Decision: Reject
- Scores: 3, 5, 5

## Abstract
Classic tree-based ensembles generalize better than any single decision tree. In contrast, recent empirical studies find that modern ensembles of (overparameterized) neural networks may not provide any inherent generalization advantage over single but larger neural networks. This paper clarifies how modern overparameterized ensembles differ from their classic underparameterized counterparts, using ensembles of random feature (RF) regressors as a basis for developing theory. In contrast to the underparameterized regime, where ensembling typically induces regularization and increases generalization, we prove that infinite ensembles of overparameterized RF regressors become pointwise equivalent to (single) infinite-width RF regressors. This equivalence, which is exact for ridgeless models and approximate for small ridge penalties, implies that overparameterized ensembles and single large models exhibit nearly identical generalization. As a consequence, we can characterize the predictive variance amongst ensemble members, and demonstrate that it quantifies the expected effects of increasing capacity rather than capturing any conventional notion of uncertainty. Our results challenge common assumptions about the advantages of ensembles in overparameterized settings, prompting a reconsideration of how well intuitions from underparameterized ensembles transfer to deep ensembles and the overparameterized regime.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper provided an asymptotic theoretical analysis for model ensemble in over-parameterization regime.

### Strengths
1. The story is clear, and the paper is well-organized.
2. The figures verify the theoretical results, and make the statement persuasive.

### Weaknesses
1. The result in Theorem 1 (question 1) has been proposed in a work titled "On the Benefits of Over-parameterization for Out-of-Distribution Generalization"(see [1]), which has been posted on March 2024. They have analysed such phenomenon under the same setting. Please add the citation, as well as comparing the difference between your result and the result in this work.
2. Assumption 2 might related to the distribution on data x, please characterise it with respect to x, to make the assumption more clear.

[1] Hao Y, Lin Y, Zou D, et al. On the Benefits of Over-parameterization for Out-of-Distribution Generalization[J]. arXiv preprint arXiv:2403.17592, 2024.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explores the generalization and uncertainty quantification aspects of overparameterized ensembles, particularly in neural networks. It examines how modern overparameterized ensembles, such as those constructed from neural networks, differ from traditional, underparameterized ensembles. Using random feature (RF) regressors, the paper theoretically demonstrates that overparameterized ensembles tend to be equivalent to a single large model in terms of generalization and predictive variance. Key contributions include proving that, under certain conditions, overparameterized ensembles provide no inherent generalization advantage over a single large model, and showing that predictive variance in overparameterized ensembles does not equate to traditional uncertainty measures.

### Strengths
- Builds on existing theoretical framework to address an important question of whether there is any advantage of training ensembles vs a bigger model.
- The assumptions are clearly stated and the proofs of the theorems are complete.
- The notation and setup are clearly explained

### Weaknesses
(See Questions for details)
- The paper does not explore empirical implications of the theoretical results 
- The results are preliminary in nature - focus mostly on asymptotic regime and corresponding finite-regime rates are not provided
- Very few examples to illustrate the assumptions and the actual implications of the theorems.

### Questions
- Assumption 1.2 (a.s. Positive definite): Can you provide some examples of features distributions for which this assumption is true. For the ReLU and Leaky-ReLU approximations, can you provide the precise rate at which the approximations yield the positive definite matrix. For example, in footnote (2), what is the dependence on $\alpha$ in the condition number of the matrix \sum_i w_i w_i^\top?
- Theorem 1: Could you please explain where in the proof of this theorem would you need the assumption that D >> N? What is the rate of convergence in terms of number of elements in the ensembles and in terms of the number of features of the single model?
- Theorem 1: What are the main challenges in obtaining a finite dimensional version of this theorem? 
- Theorem 1: Can you please instantiate the rates of the theorem for a few different classes of feature vectors which satisfy Assumption 1?
- What classes of features and distributions satisfy assumption 2?
- Theorem 2: what is the exact characterization of the Lipschitz constant \lambda? Can you provide a few examples for this constant for different distributions/features?
- What are the implications of these results for actual deep networks? How well do these results transfer empirically to the deep learning regime? It would be useful to provide some experiments with small real-world datasets like Cifar-10 to study the implications of these results.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the effect of ensembling in random features (RF) models. The basic question in this area is whether ensembling provides any advantages over a single larger model, such as implicit regularization or better generalization. Building on the work of Jacot et al. (2020), the authors demonstrate that in the regression setting, an infinite ensemble average of ridgeless RF models is equivalent to a single infinitely-wide ridgeless RF model. Notably, this result does not rely on the gaussian universality assumption common in the literature. The authors then discuss the positive ridge case, showing that the same pointwise result holds up to an error proportional to lambda. As a consequence, they point out that popular ensemble-based uncertainty measures capture deviations from an infinite-width limit, not the true response, and therefore must be interpreted carefully.

### Strengths
The paper is clearly written and provides a good overview of recent papers in the area. The strongest result in the paper is Theorem 1. This theorem establishes equivalence results under Assumption 1, which assumes subexponential distributions on the whitened random features and allows for dependence. This is weaker than the gaussian process assumption required by Jacot et al (2020). The key technical result is Lemma 1, which leverages concentration inequalities for subexponential random variables to handle an error term in the pointwise analysis of Theorem 1. The experiments are clear and support Theorem 1 and its corollaries.

### Weaknesses
One concern is that Theorem 1 does not offer substantially new insights into ensembling compared to Jacot et al. (2020), which also showed the equivalence of ridgeless ensembles and a ridgeless infinite-width limit, albeit under stronger assumptions.
Also, I believe the paper could be strengthened if it provided more theoretical analysis in the positive ridge setting. The authors prove that the error is Lipschitz in the regularization $\lambda$ in Theorem 2. However, in Jacot et al. (2020), the authors characterize the regularization $\tilde \lambda$ of the ensemble and show that $\tilde \lambda$ is strictly greater than $\lambda$. If the results in Theorem 2 are sharpened, can this implicit regularization be recovered?
Finally, I believe the discussion of uncertainty quantification in Section 3.2 could use further development. In particular, can the results in Theorem 1 or Theorem 2 be used to demonstrate a lack of robustness to distribution shifts, as discussed in Abe et al. (2022)?

### Questions
Can Theorem 1 be generalized to account for feature learning? For instance, does the result hold true if 1 step of gradient descent is performed on the first layer weights? Is it possible for the analysis in Theorem 2 to be refined so as to place bounds on the implicit regularization $\tilde \lambda$ of the ensembled model? Also, can the discussion in Section 3.2 be used to evaluate robustness of the ensemble to distribution shifts?

### Soundness
3

### Presentation
3

### Contribution
2
