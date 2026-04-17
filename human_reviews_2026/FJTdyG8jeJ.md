# A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Neural network-based optimal transport (OT) is a recent and fruitful direction in the generative modeling community. It finds its applications in various fields such as domain translation, image super-resolution, computational biology and others. Among the existing OT approaches, of considerable interest are adversarial minimax solvers based on semi-dual formulations of OT problems. While promising, these methods lack theoretical investigation from a statistical learning perspective. Our work fills this gap by establishing upper bounds on the generalization error of an approximate OT map recovered by the minimax quadratic OT solver. Importantly, the bounds we derive depend solely on some standard statistical and mathematical properties of the considered functional classes (neural nets). While our analysis focuses on the quadratic OT, we believe that similar bounds could be derived for general OT case, paving the promising direction for future research. Our experimental illustrations are available online https://github.com/milenagazdieva/StatOT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors study a particular semi-dual formulation of the quadratic Optimal Transport (OT) problem and derive statistical guarantee bounds for recovering the true OT map when using a solver that optimizes this formulation. More precisely, they adopt a generative modeling framework and assume that the input and target distributions are absolutely continuous probability measures given through samples. They consider the case where the solver relies on two neural networks optimized using the semi-dual objective, which is approximated via Monte Carlo estimation based on the empirical distributions of the measures. The authors provide an upper bound on the L2 distance between the true OT map and the best transport map that could be obtained by the considered solver. To achieve this, they first decompose this gap, referred to as the generalization error in the paper, into two terms: the estimation error, which accounts for the discretization of the semi-dual formulation, and the approximation error, which arises because the optimization is performed over a restricted class of neural networks rather than the entire space of functions. They bound the first term using the Rademacher complexity of the function class and show that the second term can be made arbitrarily small by choosing an appropriate class of neural networks.

### Strengths
The theoretical results presented in this paper are interesting: the authors derive an upper bound on the number of samples required to approximate the true OT map using the proposed solver. Section 4, which focuses on deriving this upper bound, is well organized and clearly written, which permits to easily follow the main steps of the derivation.

### Weaknesses
- I am a bit perplexed by the experimental setup. In Section 5.1, the authors assume that, because they use the same neural network architecture as the one corresponding to the ground-truth OT map, there is no gap in the approximation error, and thus the observed generalization error corresponds solely to the estimation error. This appears to be a strong assumption, as it implies that the optimization process finds the global minimum of the loss defined in Equation (8) and does not get trapped in local minima when training the ICNN.

- Regarding the writing, Sections 2 and 3 could be improved. The theoretical background on OT with general cost functions is not necessary to understand the paper’s contributions. The authors should focus exclusively on the quadratic cost in Section 2. Similarly, the paragraph “Continuous OT solvers for quadratic cost” does not seem essential for understanding the main contributions.

- Finally, the writing should be polished and carefully revised for clarity and precision (see, for example, lines 352–353, 405–406, and 410–411).

### Questions
- Does the class of neural networks described in Proposition 4.4 represent any $\beta$-strongly convex function that is totally bounded with respect to the Lipschitz norm?

- What does assuming that the optimal potential $\phi^*$ is $\beta$-strongly convex imply about the considered distributions $p$ and $q$ ? 

- In Section 5.1, you assume that, because you use the same neural network architecture as the one corresponding to the ground-truth OT map, there is no gap in the approximation error, and that the observed generalization error is, in fact, the estimation error. This seems to be a strong assumption, as it implies that the optimization process reaches the global minimum of the loss defined in Equation (8) and does not get trapped in local minima when optimizing the ICNN (a similar comment applies to Section 5.2). Could you comment on this? Do you observe that, for different initializations of the ICNN, the optimization consistently recovers approximately the same transport map? Or, did you try initializing the neural network with several random seeds and retaining the best result?

- Your experiment in Section 5.1 seems ideal, as it suggests that the bound you derive is tight in dimension 4—is that correct? I am not a specialist in statistical bounds, but is it common to verify experimentally that such bounds are tight as you did? Could you elaborate on this?

- During the experiments, did you constrain the ICNN as described in Proposition 4.4? Do you observe that imposing these constraints helps the optimization in practice?

- Finally, there is a small typo: it should be $\varepsilon / \beta$ in Equation (20), although this does not affect the result.

### Soundness
2

### Presentation
2

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
The paper provides a theoretical characterization of the generalization error for optimal transport mappings parameterized by input-convex neural networks, focusing on a class of continuous Wasserstein-2 distance solvers. The analysis decomposes the overall error into two components, estimation error and approximation error, each of which is independently derived and subsequently combined to yield the final generalization bound. The theoretical findings are supported by experimental validation on a simple, low-dimensional synthetic dataset, which confirm the consistency of the results with the proposed theoretical framework.

### Strengths
The paper makes an original theoretical contribution by deriving generalization error bounds for quadratic optimal transport (OT) solvers parameterized by neural networks, particularly within the class of input-convex models. This work provides valuable insights into the learnability and error behavior of neural OT mappings and offers practical guidance for their use in learning-based transport problems. The theoretical framework is rigorous, decomposing the overall generalization error into estimation and approximation components with clear mathematical justification.

The paper is clearly written, logically structured, and easy to follow, even when presenting technically complex material. Although the experiments are limited to simple synthetic datasets, they effectively validate the theoretical results. The significance of this work lies in establishing foundational learnability guarantees for neural OT solvers, which bridge the gap between theoretical understanding and practical implementation in modern OT-based learning frameworks.

### Weaknesses
A key limitation of the paper is its exclusive reliance on low-dimensional synthetic datasets for empirical validation. While the theoretical analysis is sound, the absence of experiments on higher-dimensional or real-world datasets limits the assessment of the framework’s practical relevance and robustness. Extending the experiments to more complex domains would provide stronger empirical support for the theoretical claims and demonstrate the scalability of the proposed bounds.

Additionally, the paper assumes \beta-strong convexity, a condition that is often difficult to guarantee in neural network parameterizations. While the authors acknowledge this assumption as restrictive, a deeper discussion of its implications and potential violations would be valuable.

Some missing references, e.g. [proof ref.] in multiple theorems and propositions.

### Questions
In Figure 3, the slope appears to increase as the dimensionality rises. Could the authors provide further insights into the underlying reasons for this behavior? Additionally, as the dimensionality continues to grow, do the authors expect the slope to remain below or equal to 0.5?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies the approximation capabilities of a certain class of optimal transport problems for quadratic cost.  In particular, it focuses on the estimation of the *generalization* error: this stem from several factors, the approximation error (due to the dual potential being parametrized with a neural network) and the estimation error (stemming from finite sample).

This family of problems (and their solutions) is well studied, but the case of semi-dual solvers is less studied. Here, the semi-dual formulation allows to parametrize a potential using a convex class of functions, the ICNN. 

The experimental part of the paper study a particular implementation of the approach using Input Convex Neural Network (ICNN).

I did not check the proofs in appendix

### Strengths
The paper is well written, very clear, and accessible to people with familiarity in optimal transportation, learning theory and statistical consistency. Overall, the strategy to tackle the problem is well explained.  

The related work is well done, especially on semi-dual formulations, which are a getting popular, making the contribution timely.  

The methodology is standard, but the result is new. Only a subset of results actually depend on the approximation property of certain neural networks classes. Others are a classical decomposition of the error between generalization and approximation.

We can appreciate the diversity of tools and techniques used.

### Weaknesses
### Missing related work

The recent work of Nietert and Goldfeld is relevant and could be added to the related work.

Nietert, S. and Goldfeld, Z., *Estimation of Stochastic Optimal Transport Maps*. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

### Somewhat unconvincing experiments

My enthusiasm for the paper is somewhat tempered by the numerical results. 

**In Figure 3**, the logarithmic scale hides many phenomena. For example, I'd be curious to see what the experimental error looks like with the naive barycenter translation estimator: $T(x)=x+(\mathbb{E}_Q[x]-\mathbb{E}_P[x])$.

I have similar concerns with Figure 4 (see questions).  

Overall, since some experiments do not depend on neural networks, and since ICNN are notably hard to train, it could be good to plot results of other (possibly non-neural) parametrizations.

### Questions
### Approximation power as function of depth

**In Figure 4, Dim 2**, I spot a counter-intuitive behaviors. I see that the approximation error does not diminish with $\max H_{\phi}$. Do you know why? This somewhat contradicts Theorem 4.3.  

Can you clarify what is the **x-axis**: why does it cover the 0-1 range? 

Finally, we sometimes see the error accumulating on special values (e.g Dim 2 maximum width, or Dim 4 small width). Can you double check that the network is not degenerating toward a trivial solution?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper develops a statistical learning framework for the proposed method, stating finite sample guarantees and excess risk bounds for the estimator. It highlights how regularity and beta-strong convexity assumptions drive stability and convergence rates.
The analysis connects optimization error and estimation error, with explicit dependence on sample size and ambient dimension. The results are positioned relative to statistical optimal transport, where dimension often controls rates.

### Strengths
Theoretical framing is clear, separating approximation, optimization, and estimation errors. The proofs use standard tools from empirical process theory and stability.

### Weaknesses
The beta strong convexity condition is assumed rather than derived from the model or data-generating process. Please either prove it under verifiable conditions or provide a constructive check that practitioners can apply to certify it.

### Questions
1. Under what concrete distributional conditions on the source and target does beta strong convexity hold? 
2. The estimation error exhibits explicit dimensional dependence that resembles known lower bounds in statistical optimal transport. Can the authors clarify whether structures such as low intrinsic dimension, manifold support, or spectral decay can reduce the exponent, and if so, how this would change the bounds and constants?

### Soundness
3

### Presentation
2

### Contribution
2
