# Safeguarded Stochastic Polyak Step-Sizes for Non-smooth Optimization: Robust Performance Without Small (Sub)Gradients

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
The stochastic Polyak step size (SPS) has proven to be a promising choice for stochastic gradient descent (SGD), delivering competitive performance relative to state-of-the-art methods on smooth convex and non-convex optimization problems, including deep neural network training. However, extensions of this approach to non-smooth settings remain in their early stages, often relying on interpolation assumptions or requiring knowledge of the optimal solution. In this work, we propose a novel SPS variant — Safeguarded SPS (SPS$_{safe}$) — for the stochastic subgradient method, and provide rigorous convergence guarantees for non-smooth convex optimization with no need for strong assumptions. We further incorporate momentum into the update rule, yielding equally tight theoretical results. Comprehensive experiments on convex benchmarks and deep neural networks corroborate our theory: the proposed step size accelerates convergence, reduces variance, and consistently outperforms existing adaptive baselines. Finally, in the context of deep neural network training, our method demonstrates robust performance by addressing the vanishing gradient problem.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel Polyak-type step size, namely Safeguarded SPS. It modifies the function minimum value in the previous Polyak-type steps to a lower bound \(\ell_i^*\), eliminating the need for the interpolation assumption while avoiding reliance on oracle information. Additionally, it clips the gradient norm information to ensure the smoothness of the step size, which can effectively address the vanishing gradient problem in neural network training.

### Strengths
1. It provides a comprehensive review of the existing methods with Polyak-type step sizes analyzed in the convex non-smooth stochastic setting. A table is used to compare these methods, including SPS, SPS*, IMA, and the two methods proposed in this paper. Notably, the proposed methods can maintain a favorable convergence rate without requiring the interpolation condition or prior knowledge of \(f_i(x^*)\). 

2. In the deep neural network (DNN) training section, key experimental data (especially the proportion of iterations using the Polyak step) and figures are presented to compare the proposed methods with \(SPS_{max}\) and \(Smooth SPS_{max}\). These data and figures are clearly explained, intuitively demonstrating the advantage of the proposed methods: their step sizes remain adaptive throughout the iteration process and never degenerate into constant values.

### Weaknesses
1. The experimental section only conducts experiments on Support Vector Machines (SVM), Phase Retrieval, and Deep Neural Networks (DNNs). More cases could be tested to verify whether favorable convergence results can be achieved. In the DNN training experiments, only ResNet-20/32 models and the CIFAR-10/100 datasets are used; experiments on different models and more complex datasets could be attempted. Moreover, the reported test accuracy (<90%) on CIFAR-10 appears lower than typical results for ResNet-20 with Adam (around 91–92%), indicating that further hyperparameter tuning or training optimization might be needed.

2. There seems to be no robust strategy for selecting \(\ell_i^*\) and \(M\). For tasks like neural network training, the \(\ell_i^*\) of the loss function may be easy to obtain, but are there any selection methods for other types of problems? The selection of \(\ell_i^*\) affects convergence results: if it is chosen too small, \(\sigma\) will become too large, thereby degrading convergence performance. Moreover, \(\ell_i^*\) may not be easily selectable in more complex problems. Additionally, in the sensitivity analysis of \(M\), experiments are only conducted for \(M = \{1, 10, 100\}\). 

3. The proofs of the theorems are mostly standard, and the convexity assumption is used in the proofs. 

4. There is a typo in the definition of \(f_i^*\) in Line 108: \(x\) is incorrectly written as \(x^*\).

### Questions
1. In the theorem proof section, the proofs are only provided for convex problems. However, the loss functions in practical neural network training are often highly non-convex. What difficulties are there in extending the convergence conclusions to non-convex problems?

2. Poor selection of \(\ell_i^*\) and \(M\) may lead to very poor convergence results, and they may be difficult to select in complex problems. How can this issue be addressed?

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
This paper introduces Safeguarded Stochastic Polyak Step Size (SPSsafe) for stochastic subgradient methods designed to handle non-smooth convex optimization problems. Unlike prior SPS variants for non-smooth convex problems, SPSsafe does not require knowledge of the optimal solution or rely on interpolation assumptions. The paper establishes convergence guarantees and extends their approach by integrating momentum, maintaining the theoretical properties. The paper experiments on both convex optimization benchmarks and deep neural network training.

### Strengths
The paper is well motivated, building on the growing interest in SPS-type step-size rules, which have recently proven highly effective for stochastic optimization and deep learning. It makes a good contribution by extending this family of methods to the non-smooth setting. To the best of my understanding, this is the first SPS variant that offers provable convergence for non-smooth problems, or at least a practical one that does so without requiring knowledge of the optimal value. 

There is a good comparison to the existing SPS step-size rules, in this way, I find the comparison good and useful to understand the contribution.

The work is also extensive, in the sense that it establishes convergence rate for both SGD and SGD with momentum. 

The numerical results show clear benefits of the proposed approach.

### Weaknesses
The theoretical results could be better contrasted with results for general SGD. What is not clear, if there are some theoretical benefits of SPS-type step-size scheduling, compared to other step-size choices. I don’t think so, the O(1/sqrt(t)) convergence is standard for SGD for a problem with functions of this  structure. I don’t know if there are improvements for the scalar factor, or what. The contribution is still interesting, because there are practical benefits, but it would be good if this could be explained better.

The methodology in the experiments could be strengthened. The reported results are based on single runs, which makes it difficult to assess the statistical significance or robustness of the improvements. Including multiple runs with confidence intervals or standard deviations would give a clearer picture of performance variability and help substantiate the claimed advantages. More detailed analysis of sensitivity to hyperparameters and comparisons under consistent stopping criteria would also make the experimental evaluation more convincing.

### Questions
From my understanding, SPS step-sizes are particularly applicable because they are more robust to parameter tuning, e.g., like initial step-size. It would be nice to do some analysis on this compare to other algorithms.

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
3

### Summary
The paper presents a new optimization technique called Safeguarded Stochastic Polyak Step-size ($SPS_{safe}$), which solves essential problems in non-smooth optimization for machine learning applications. The previous Polyak-type methods proved unusable because they needed data interpolation and unachievable "oracle" knowledge about the best solution. The main advancement of $SPS_{safe}$ introduces a basic protection system that stops the step-size from growing excessively when gradient values approach zero through the addition of a denominator floor value. The method provides the first Polyak-style optimization technique that proves convergence without needing unrealistic assumptions. The research develops a momentum-based algorithm from this concept and proves its connection to the popular gradient clipping heuristic. The proposed method demonstrates its effectiveness through multiple experiments on convex problems and deep neural networks, which show it achieves faster convergence and lower variance while outperforming standard adaptive optimizers by solving vanishing gradient problems.

### Strengths
1. A novel step-size for SSM without strong assumptions. The paper presents $SPS_{safe}$ as the first Polyak-type step-size for SSM, which proves convergence in stochastic convex non-smooth environments without needing interpolation conditions or oracle access to $f_{i}(x^{*})$. The algorithm reaches a solution neighborhood with a convergence rate of $\mathcal{O}(1/\sqrt{T})$.
2. The research conducts an extensive evaluation of the protection system's advantages. The paper demonstrates through formal algebraic methods that $SPS_{safe}$ operates identically to clipped SSM with an adaptive learning rate, thus establishing the first theoretical convergence proof for this method in stochastic environments.
3. Extension to momentum with rigorous guarantees. The safeguarding concept extends to the IMA framework, which results in $IMA-SPS_{safe}$. The method delivers the first adaptive momentum technique for non-smooth optimization, which proves convergence through rigorous analysis without needing oracle access, while supporting both Cesaro average and last iterate convergence.

### Weaknesses
1. The system performs image classification using ResNet-20/32 models on CIFAR data and synthetic convex tasks, but lacks training on large NLP datasets with transformers and multiple architecture types.
2. The paper establishes formal convergence results only for functions that are both convex and Lipschitz continuous. The method shows excellent performance on deep neural networks with non-convex structures, but lacks theoretical justification for this success.
3. The process of choosing $M$ value requires a delicate decision because increasing $M$ reduces the final error range, but it can decrease the speed of convergence at the beginning. The paper fails to study or offer methods to determine the optimal point for balancing this trade-off.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new stochastic Polyak stepsize for stochastic non-smooth optimization, along with convergence rates under different assumptions.

### Strengths
The novel stepsize appears to slightly relax previous restrictions, though it comes at a new cost of needing to know $\ell\_i^\*$.

### Weaknesses
The most apparent drawback of this work is that its convergence is limited to a neighborhood the size of which is determined by the quality of estimation for $\ell\_i^\*$. The authors do not adequately address the issue of estimating $\ell\_i^\*$; see questions. Additionally, the statement their "safeguarded momentum rule removes this oracle dependence while preserving the same rate" is incorrect. The same rate is preserved only when the oracle dependence is the same as before i.e. $f\_i(x^\*) = \ell\_i^\*$. Otherwise, the rate has an extra $\sigma^2$ term, which makes it different.

The work also notes it "removes the need for any oracle information," that the approach is "fully adaptive, i.e, need no additional problem knowledge," and that it converges "without extra information." These statements are in conflict with the fact that $\ell\_i^\*$ are required---and $\ell\_i^\*$ must be such that $\ell\_i^\* \leq f\_i^\*$---which constitutes both "extra information" and "additional problem knowledge".

There is also the misleading statement that "using the safeguarded step size SPSsafe, we achieve the same $O(T^{−1/2})$ convergence to a neighborhood of the solution." You do not achieve the same $O(T^{−1/2})$ convergence as in previous works. You achieve $O(T^{−1/2})$ convergence to a neighborhood of the solution. These are not the same, and the authors should clarify this point throughout the paper. Similarly misleading statements are:

- "it needs neither the interpolation condition nor oracle access to the values $f\_i(x^\*)$ and still achieves the rate $O(T^{−1/2})$."
- "Our results are the first Polyak-type algorithm that converges to convex and non-smooth settings without extra assumptions"
- "providing convergence rate $O(T^{−1/2})$ for stochastic, convex–Lipschitz objectives, the first Polyak rule to do so without assuming interpolation or the knowledge of $f\_i(x^\*)$."

In all of these cases, proper qualification of converging to a neighborhood is required.

### Questions
- The authors note that "many modern machine learning models satisfy this condition" for interpolation, where $\ell\_i^\* = f\_i^\*$ implies $\sigma^2 = 0$. But this case was already handled in Loizou et al. (2021). Can the authors provide an example that does **not** satisfy the interpolation condition, and also 1) give values $\ell\_i^\*$ guaranteed to be lower bounds for $f\_i$ (with proof) and 2) give an upper bound on $\sigma^2$?

- The paper claims their "proposed analysis provides the first convergence guarantees for an adaptive momentum method (through the equivalence of IMA and SSM with heavy ball momentum) that does not require any strong assumption." What about the fact that the stepsize depends on $\ell\_i^\*$? Is this not a strong assumption? It also claims to "remove the need for any oracle information." What about $\ell\_i^\*$?

- Is it possible to explain, and substantiate, the extent of their contributions beyond generalizing Theorem C.1 in Loizou et al. (2021) to the non-interpolated setting, using the proof techniques from the rest of Loizou et al. (2021) (which Loizou et al. observe, in Appendix C.1, that "one can easily obtain" "using the proof techniques from the rest of" their paper)?

- What hyperparameter tuning was done for Adam baseline? What was the choice of stepsize?

### Soundness
2

### Presentation
1

### Contribution
1
