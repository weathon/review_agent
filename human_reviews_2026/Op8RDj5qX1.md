# Optimizing optimizers for fast gradient-based learning

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
We lay the theoretical foundation for automating optimizer design in gradient-based learning. Based on the greedy principle, we formulate the problem of designing optimizers as maximizing the instantaneous decrease in loss. By treating an optimizer as a function that translates loss gradient signals into parameter motions, the problem reduces to a family of convex optimization problems over the space of optimizers. Solving these problems under various constraints not only recovers a wide range of popular optimizers as closed-form solutions, but also produces the optimal hyperparameters of these optimizers with respect to the problems at hand. This enables a systematic approach to design optimizers and tune their hyperparameters according to the gradient statistics that are collected during the training process. Furthermore, this optimization of optimization can be performed dynamically during training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies preconditioner-based optimizers, and develops a general class of problems from which preconditioners for many standard machine learning optimizers can be derived as solutions. The problems are posed in terms of finding a preconditioner that maximizes a power budget subject to constraint over a class of PSD matrices. The class of problems can be extended by allowing convolutions over time in order to facilitate the formulation of preconditioners that are computed using EMAs (e.g., use of variance estimators such as in Adam), as well as momentum. The authors present a list of problem formulations from which each standard optimizer is derived.

### Strengths
- The proposed class of optimization problems allows for a very wide range of optimizers to be derived. Derivations are presented for many optimizers that have not been explored from this view before.
- The authors present a novel way to derive momentum from the extension to the framework presented in Section 3.

### Weaknesses
- The main result, which is the unifying view of a preconditioner as the solution of this kind of constrained minimization problem, is not new. See AdaReg [1].
- Missing related work on AdaReg [1], Linear Minimization Oracles [2] which unify "step-as-a-minimizer" optimizers just like this paper unifies "preconditioner-as-a-minimizer" optimizers, and older Quasi-Newton methods like BFGS [3] which try to adaptively estimate the preconditioner $Q$.
- The authors propose a new method of tuning hyperparameters for this class of optimizers by collecting gradient covariances (line 366), but this is completely infeasible on all but tiny low dimensional problems due to how much memory it would require, and would be an empty suggestion. Even the gradients themselves take a substantial amount of memory to store and communicate; there is no way the gradient covariance, which needs the square of that amount of memory, would fit. Not only that, but the gradient covariance will also converge incredibly slowly since it would need to be estimated as a large sum over rank-one components.
- While the paper presents a theory to connect existing optimizers, there is not demonstration of how the theory is intended be useful. For example, no new optimizers were derived from this theory or tested. The authors make no statements about the properties or convergence rates of existing optimizers or new optimizers that emerge as a result from this framework; only results to prove that the known optimizers indeed emerge as derived from the framework.
- While the theory may give rise to a new optimizer, and the optimal $Q$ and gradient may mathematically exist, difficulty in computing the parameter update (without materializing $Q$ due to memory constraints) may easily block the resulting optimizer from ever becoming feasible to implement for many choices of $Q$ that the user might pick, unless this problem is addressed somehow.

[1] Gupta, V., Koren, T., Singer, Y. (2017). A Unified Approach to Adaptive Regularization in Online and Stochastic Optimization. arXiv preprint arXiv:1706.06569.

[2] Garber, D., & Wolf, N. (2021, July). Frank-Wolfe with a nearest extreme point oracle. In Conference on Learning Theory (pp. 2103-2132). PMLR.

[3] Fletcher, R. (1988). Practical Methods of Optimization.

### Questions
Is there any case where this theory can comes into use, where pre-existing work does not suffice?

### Soundness
4

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
This paper proposes a unifying theoretical framework that treats the design of optimization algorithms as a constrained maximization of instantaneous loss reduction. By formulating the update rule as the optimal solution to a convex problem over a “budget set” of positive semidefinite operators, the authors show that a wide range of optimizers can all be derived as special cases. The overall idea is elegant and offers a fresh geometric perspective on optimization design, but the paper lacks substantial empirical evidence to demonstrate practical effectiveness.

### Strengths
- The paper formulates optimizer design as a convex optimization problem that maximizes the instantaneous decrease of the training loss. This simple but powerful view connects many existing algorithms under a single mathematical framework and provides clear geometric intuition.
- Modeling momentum and EMA-style updates as single-pole linear filters and showing their optimality under extended dynamic budgets is technically sound and conceptually appealing.
- If empirically validated, the framework could unify theoretical understanding and provide a foundation for automatic optimizer design across architectures and modalities.

### Weaknesses
- The paper presents only small-scale toy examples. There are no experiments on standard deep-learning benchmarks such as CIFAR, ImageNet, or language models. As a result, it is difficult to assess whether the proposed “optimal” updates translate to practical gains in convergence or generalization.
- The framework relies on estimating gradient covariance and cross-moment statistics, which can be unstable or expensive in large-scale settings. The paper does not discuss how these quantities are maintained efficiently or how noise affects the theoretical guarantees.

### Questions
- How does the proposed “instantaneous loss reduction” objective correlate with the final validation or test loss in large-scale training? Have you observed cases where it leads to over-aggressive or unstable updates?
- Can you provide quantitative experiments on at least one deep-learning benchmark, comparing your analytic optimizer against AdamW, Shampoo, or K-FAC under equal training budgets?
- How sensitive is the method to errors in these statistical estimates? Does the Lipschitz stability analysis in the appendix extend to stochastic updates with mini-batches?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors provide a framework to obtain the update rules of commonly used optimizers as the result of an optimization problem interpreted as greedily minimizing the loss in one step. They also obtain rules for hyperparameter choices.

### Strengths
The paper presentation is clear, except for the minor typos.

### Weaknesses
The authors, in their words, "reverse-engineer commonly used optimizers" by defining the update rule as the result of an optimization problem. The authors point out that this optimization problem gives an unbounded result and proceed to add different constraints to it, and pointing out that many optimizers in use arise as the result from this optimization problem, where each optimizer correspond to a specific constraint. 

One can always represent a (stateless or not) algebraic rule as the solution of an optimization problem (e.g. in other domain: a function value can be written as the Fenchel dual of its Fenchel dual, which is the result from an optimization problem) and given that this paper only cares about obtaining these optimizers without providing any way to obtain other effective algorithms or select among them, and it does not provide experiments either on the effectiveness on, for instance, their insights of hyperparameter tuning, it seems to me that this only provides a mapping one to one between algebraic rules and the solution of optimization problems, which is a weak result for ICLR. 

The authors suggest, in Proposition 7, to design optimizers that are optimized to decrease the validation loss . This seems to be essentially equivalent to using the validation set as training set and although the authors make a brief comment in passing about the potentially controversy of this, the discussion should be expanded, since as it is right now I am just inclined to think that the authors are simply not using any validation set and using such data as training data.  


A couple of minor typos: 

line 174 Eveidence
line 247 the followings -> the following
line 269 obtaind

### Questions
Suggestions: 
It would be good that the authors add an extended explanation for their choice of a positive semidefinite operator Q as opposed to any other choice.

Drori and Teboulle https://arxiv.org/abs/1206.3209 developed the widely used PEP framework (https://github.com/PerformanceEstimation/PEPit) that phrases the problem of finding the best optimization algorithm for a problem in an algorithmic class, into solving semidefinite programs. This literature is relevant and should be discussed.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors present a unified framework for designing optimizers that are “optimal” in the sense of maximizing the instantaneous reduction in loss. They develop this framework and show that it can both recover common optimizers as well as find closed-form solutions for the optimal hyperparameters of these optimizers. The framework is developed both with and without memory for the optimizer (e.g. momentum variables).

### Strengths
* The general approach is interesting, with potential applications to find better optimizers than the ones which are currently popular.
* The theoretical aspects are very thoroughly analyzed, including a very wide array of common optimizers in the new framework.

### Weaknesses
1. Formulating optimizer design as seeking to maximize the instantaneous decrease in loss seems like a greedy choice. A more natural definition would be to minimize the loss at the end of training, which is the common definition of optimization tasks as it is. Even though this point is addressed in the paper, I think it brings into question the motivation for this work. Can we really expect real-world benefits from this approach? Instead of analyzing so many theoretical aspects, it would have been better to take the next step and show that this approach can be extended to the more natural “minimize converged loss”.


2. Even though the generality of the framework is a good thing, it still seems a bit contrived. There’s no good a-priori choice of optimizer budget, so being able to reverse-engineer existing optimizers doesn’t seem like a surprising result. By shaping the budget I can make every optimizer seem “optimal”, even so the optimality doesn’t really make any sense. This makes me think that the new framework is too broad to be useful as it is.
Even calling the budget a “budget” is confusing, since there’s no real “cost” being spent here - it’s constraining the search space of the problem. The term optimizer family is more appropriate (also used in the paper).


3. No real-world problem was shown to benefit from this approach experimentally.

4. The paper is VERY dense with results, with no room for any of the proofs. I would have expected that at least from the reverse-engineering we could glean some insights into the choice of optimizer family, but the fact this was omitted strengthens my assertion that the “optimizer family” is too broad to be a useful concept. The density of results makes the paper hard to follow. Almost every paragraph introduces a new concept. The appendices are pretty much a required read, which shouldn’t be the case.

### Questions
1. When working on a new optimization problem, what suggestions do you have for choosing the optimizer family?
2. I would suggest making at least a few experiments with real-world problems, such as training even a small open-source LLM, comparing convergence rate and achieved performance. At least demonstrate you can find the optimal hyper parameters without expensive hyper opt search, and better yet - show you can choose an optimizer family that leads to faster convergence than e.g. Adam.

### Soundness
2

### Presentation
3

### Contribution
2
