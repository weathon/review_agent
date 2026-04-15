# Exact Mean Square Linear Stability Analysis for SGD

- Decision: Reject
- Scores: 6, 6, 5

## Abstract
The dynamical stability of optimization methods at the vicinity of minima of the loss has recently attracted significant attention. For gradient descent (GD), stable convergence is possible only to minima that are sufficiently flat w.r.t. the step size, and those have been linked with favorable properties of the trained model. However, while the stability threshold of GD is well-known, to date, no explicit expression has been derived for the exact threshold of stochastic GD (SGD). In this paper, we derive such a closed-form expression. Specifically, we provide an explicit condition on the step size that is both necessary and sufficient for the stability of SGD in the mean square sense. Our analysis sheds light on the precise role of the batch size $B$. Particularly, we show that the stability threshold is a monotonically non-decreasing function of the batch size, which means that reducing the batch size can only decrease stability. Furthermore, we show that SGD's stability threshold is equivalent to that of a process which takes in each iteration a full batch gradient step w.p. $1-p$, and a single sample gradient step w.p. $p$, where $p \approx 1/B $. This indicates that even with moderate batch sizes, SGD's stability threshold is very close to that of GD's. Finally, we prove simple necessary conditions for stability, which depend on the batch size, and are easier to compute than the precise threshold. We demonstrate our theoretical findings through experiments on the MNIST dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This manuscript provides a precise condition for the mean square stability of SGD around a minimum, considering both interpolation and non-interpolation cases. The authors also discuss the implications for the influence of batch size and provide numerical experiments to support their theoretical findings.

### Strengths
- The precise condition presented, particularly the explanation provided in Proposition 2, is interesting and greatly intuitive. Proposition 2 is especially well-explained and appreciated by the reviewer.

- The discussion on how batch size affects stability and the finding that the maximum eigenvalue of the Hessian is close to that of GD are intriguing. The experimental validation, although limited to small nets for fitting MNIST, is solid.

- The experiments also suggest that SGD **may** operate on the edge of stability. This interesting observation deserves more investigations.

### Weaknesses
- The extension to non-interpolation minima is uninteresting and does not offer any new insights, as far as the reviewer can tell. Additionally, the definition of regular minima is peculiar, as it requires all sample Hessian matrices to be SPD. It is unclear why such a strange assumption is relevant.

- The experiments are limited to a very smal-scale setup: one-hidden-layer nets + subset of MNIST.

### Questions
None

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper finds the explicit expression of the linear-stability threshold learning rate for SGD. This paper improves the existing result by deriving an explicit expression of the threshold learning rate from the implicit one. The new expression motivates linking the stability of the finite batch size SGD to a process of mixed online SGD and GD. Simpler bounds for the threshold learning rate are also found. Numerical experiments using MNIST confirm the analytical results.

### Strengths
This paper finds an explicit expression of the threshold learning rate and simplifies its computation by turning an optimization problem into an eigen-value problem, which I believe is a nice improvement over the existing results.  Also, this paper provides the insight that, in terms of linear stability, the SGD could be viewed as a mixture of GD steps and mini-batch SGD steps. The experiment results are well presented with highly informative figures.

### Weaknesses
Although explicit expression is always welcomed, the sufficient and necessary condition for linear stability involving learning rate, Hessian, and batch size already exists in previous works. So, this paper studies a well-understood problem. The writing could be improved. Inserting one subsection between theorem 3 and its proof may not be the best way of presenting them. Also, the notation $\theta^\parallel$, $\theta^\perp$, and $vec$ are not introduced in the main text.

### Questions
* (I didn’t check all the mathematical details so please excuse me if what I ask is already written in the paper.) I don’t see immediately why ${\bf C}$ is guaranteed to be PSD. Is there any relevant lemma in the paper? Or could the authors provide some intuition.
* It appears to me that getting relation (83) requires that ${\bf u^T Du}$ is positive. Is ${\bf D}$ also PSD?
* The authors mentioned analyzing the stability of SGD from a probabilistic point of view. I wonder if the quantitative results in the two papers can be compared.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present a stability threshold on the stochastic gradient descent. The proposed threshold is a monotonically non-decreasing function of the batch size. It differs from the existing thresholds since it is a closed-form formulation rather than another optimization problem.

### Strengths
- The problem that the authors consider is broader. Hessian is still assumed to be positive semi-definite. However, the individual gradients are allowed to be arbitrary if the mean of the gradients over all samples goes to zero.

- Both interpolating and non interpolating (regular) minima are discussed.

### Weaknesses
- Figure 2 could be drawn by using different seeds for SGD and could be represented by an error bar. It is not obvious how would these results change with a different initialization.

- The conclusion that authors mentioned in Figure 2 is not very obvious, how would the authors explain the fluctuation, how is the evaluation done when  the authors are claiming "optimized bound coincides"? "We see that for small batch sizes B = 1 and B = 2, the optimized bound (24) coincides with 2/η, confirming that SGD converged at the edge of stability".

Minor:

The paper writing needs further polishing. 

-There are some repetitions. The rates of the single sample gradient step and full batch gradient step are mentioned in the abstract, beginning of page 2, in the exact same words.

- The word "dynamics" is used quite frequently before mentioning what authors refer to as dynamics before mentioning the analysis of SGD's dynamics.

### Questions
- Can authors elaborate more on the mean-square sense?

A few questions are mentioned in the weaknesses section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
