# Convergence of Bayesian Bilevel Optimization

- Decision: Accept (spotlight)
- Scores: 8, 6, 6, 6

## Abstract
This paper presents the first theoretical guarantee for Bayesian bilevel optimization (BBO) that we term for the prevalent bilevel framework combining Bayesian optimization at the outer level to tune hyperparameters, and the inner-level stochastic gradient descent (SGD) for training the model. We prove sublinear regret bounds suggesting simultaneous convergence of the inner-level model parameters and outer-level hyperparameters to optimal configurations for generalization capability. A pivotal, technical novelty in the proofs is modeling the excess risk of the SGD-trained parameters as evaluation noise during Bayesian optimization. Our theory implies the inner unit horizon, defined as the number of SGD iterations, shapes the convergence behavior of BBO. This suggests practical guidance on configuring the inner unit horizon to enhance training efficiency and model performance.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provides a proof for the convergence of the Bayesian Bi-level optimization (BBO). By modeling the excess risk of the SGD-trained parameters, a regret bound is established for BBO with EI function, which bridges the analytical frameworks of Bayesian optimization and Bi-level optimization. Moreover, the authors introduce adaptable balancing coefficients to give a sublinear regret bound for BBO the UCB acquisition function.

### Strengths
1. The paper is theoretically solid. Useful regret bounds are provided and a convergence framework for BBO is established.

2. The paper is well-organized. The motivation, technique, proof schemes, and results are clearly stated.

3. Some tricks presented in the paper are interesting. For example, the adaptation of balancing coefficients could be a useful technique in other Bayesian applications.

### Weaknesses
1. The regularity assumptions are not intuitive. It would be better if the authors provided some real applications and models satisfying these assumptions.

2. Some assumptions are restrictive from the view of optimization, like the Lipschitz continuity and smoothness conditions in Theorem 1. Only a few classes of functions *simultaneously* satisfy them on $\mathrm{R}^d$.

### Questions
What is the fundamental difficulty of establishing convergence of BBO compared with other Bi-level algorithms?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
This paper introduces the initial theoretical assurance for Bayesian bilevel optimization (BBO). It is proved sublinear regret bounds suggest simultaneous convergence of the inner-level model parameters and outer-level hyperparameters to optimal configurations for generalization capability.

### Strengths
1. This work conducts lots of theoretical analysis Bayesian bilevel optimization (BBO). Specifically, a novel theoretical analysis of convergence guarantees for generalization performance within a BBO framework is provided.

2. A regret bound for BBO using the EI function is discussed in this work.

3. A significant advancement in this research lies in the conceptualization of SGD excess risk as a form of noise within the framework of Bayesian optimization. This approach allows for the adjustment of noise assumptions to better match real-world scenarios and greatly simplifies convergence analysis.

### Weaknesses
I can't find any experimental results in this work. I understand this work puts more attention on the theoretical analysis in Bayesian bilevel optimization (BBO). However, the authors should also conduct experiments to substantiate the theoretical analysis.

### Questions
My concerns are about the experimental results.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper focuses on Bayesian bilevel optimization (BBO), which combines outer-level Bayesian opti- mization for hyperparameter tuning with inner-level stochastic gradient descent (SGD) for model parameter optimization. The paper proves sublinear regret bounds for BBO using expected improvement (EI) and upper confidence bound (UCB) acquisitions. This provides theoretical assurance that BBO enables simul- taneous convergence of parameters and hyperparameters. For EI, the optimal number of SGD iterations is shown to be N ≍ T 2, balancing training and tuning. This achieves regret savings compared to previous works. For UCB, sublinear regret is proven even with fewer SGD iterations, showing UCB is more robust to SGD noise. The UCB balancing coeﬀicients are adapted based on the SGD/Bayesian iteration ratio. The analysis bridges the gap between Bayesian and bilevel optimization frameworks by modeling SGD excess risk, which enables adapting convergence guarantees to the BBO setting.

### Strengths
(1) The paper provides a new theoretical analysis bridging the frameworks of Bayesian optimization and bilevel optimization by modeling the excess risk of SGD-trained parameters as noise to tackle challenges in convergence guarantees for BBO generalization performance.

(2) Based on a noise assumption better suited to practical situations, the authors derive sublinear regret bounds for Bayesian bilevel optimization using the expected improvement function, which is better than previous work.

(3) By introducing adaptable balancing coeﬀicients $\beta_t$ for the UCB acquisition function, the paper establishes a sublinear regret bound for BBO with UCB that holds with fewer SGD steps, enhancing inner unit horizon flexibility and overcoming limitations of rapidly increasing coeﬀicients from previous analyses.

### Weaknesses
(1) The current paper is primarily theoretical with a lack of numerical experiments on actual data, which limits the persuasiveness. Experiments using real-world hyperparameter tuning tasks could offer tangible evidence of the convergence behavior and help assess how well the assumptions fit such scenarios.

(2) This paper focuses solely on Gaussian process priors for the Bayesian optimization portion, but the choice of prior may significantly impact the convergence guarantees. The current analysis leverages nice properties of GP priors and posters but may not directly extend to other priors that require different proof techniques, which could limit wider applicability.

(3) Bayesian optimization is adopted for hyperparameter tuning at the outer layer, so the algorithm in this paper may require extensive sampling and integration to estimate the posterior distribution, making it computationally demanding and diﬀicult to apply to high-dimensional complex problems.

### Questions
(1) How do the convergence guarantees extend to deep neural network training? Are there any unique challenges posed by DNNs?

(2) For the inner-level SGD, will using SVRG or introducing acceleration techniques lead to better corresponding results compared to standard SGD?

(3) Do assumptions such as the bounded RKHS norm of the objective function correspond cleanly to properties of other priors?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This is paper presents the first convergence analysis of Bayesian bilevel optimization where the outer level is hyperparameter tuning and the inner level is SGD. The key results are sublinear regret bounds showing the convergence behaviors of both outer and inner optimization problems. The key technical novelty is modeling the excess risk of SGD training as the noise of the outer Bayesian optimization. This paper doesn’t have experiments.

### Strengths
1. First convergence analysis of BBO is important, which is the main contribution of this paper.
2. I appreciate the innovation that modeling the excess risk of inner level SGD-trained parameters as the primary noise source of outer-level BO. It makes great sense in this problem setting.
3. I like “practical insights” sections, which are helpful.
4. The whole paper is well written and easy to follow except some notation problems mentioned below.

### Weaknesses
1. Motivation of BBO is not very clear. No detail is shown in “significant promise in engineering applications” in 2nd paragraph of Introduction.
2. Convexity assumption in Definition 1 is strong. How can you assume the loss function is convex given potentially non-convex objective function? I want to learn more justification from the author.
3. L is taken as both loss function and Lipschitz constant, which introduces some confusion.
4. Upper bound in Theorem 1 is too vague, only showing dependence on N. How does it depend on other terms?

### Questions
1. Why is modeling the noise as a martingale difference a key limitation? Why does this approach not align with hyperparameter optimization? See fourth line of page 2.
2. In third line of Section 3.2, you assume function L has a uniquely determined value for each \lambda. In my understanding, it is needed otherwise some \theta rather than \theta* may lead to lower value given some \lambda and it would be hard to define regret. However, do you have more justification on this assumption especially in practical scenarios?
3. What’s $\varphi(N)$ in Theorem 1?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
