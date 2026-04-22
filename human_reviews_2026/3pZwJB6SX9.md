# Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 2, 2, 6

## Abstract
Oblique decision trees combine the transparency of trees with the power of multivariate decision boundaries—but learning high-quality oblique splits is NP-hard, and practical methods still rely on slow search or theory-free heuristics.
We present the Hinge Regression Tree (HRT), which reframes each split as a non-linear least-squares problem over two linear predictors whose max/min envelope induces ReLU-like expressive power.
The resulting alternating fitting procedure is exactly equivalent to a damped Newton (Gauss–Newton) method within fixed partitions.
We analyze this node-level optimization and, for a backtracking line-search variant, prove that the local objective decreases monotonically and converges; in practice, both fixed and adaptive damping yield fast, stable convergence and can be combined with optional ridge regularization.
We further prove that HRT’s model class is a universal approximator with an explicit $O(\delta^2)$ approximation rate, and show on synthetic and real-world benchmarks that it matches or outperforms single-tree baselines with more compact structures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces hinge regression tree (HRT), a new method for learning oblique regression tree splits. The key idea is to formulate each split as a nonlinear least-squares problem over two linear predictors, yielding an alternating fitting procedure that is mathematically equivalent to a damped Newton method within fixed partitions. HRT demonstrates competitive performance in benchmarking experiments compared to standard baselines. Overall, the proposed method appears promising. My comments are as follows:

- Section 3: Consider adding a short subsection before Section 3.1 that briefly recaps oblique decision tree methods. The current text is related but does not explicitly state that this approach is an instance of oblique decision tree regression.

- Line 154: I believe this represents a linear decision boundary, rather than a hinge-based one. 

- Line 191: The optimization behavior accounting for partition changes is not analyzed, leaving a disconnect between the theoretical guarantee and the practical performance. The empirical results show that the algorithm converges on real data, which is reassuring; however, I would still view the algorithm as heuristic, and suggest avoiding terms such as “rigorously” (line 49) or “solid theoretical foundation” (line 50).

- The computational efficiency of the proposed algorithm has not been comprehensively evaluated.

### Strengths
See summary above.

### Weaknesses
See summary above.

### Questions
See summary above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes the Hinge Regression Tree (HRT), an algorithm for building oblique regression trees by jointly learning two linear predictors at each split. Each node models its output as a hinge function, leading to a piecewise linear regression surface. Training alternates between (i) fitting the two linear models with ordinary least squares (optionally ridge-regularized) and (ii) reassigning data to the branch that yields the higher prediction. The authors describe this alternating fitting as a damped Newton method within fixed partitions and show that, with small damping factors, it converges stably in practice. They also restate a known universal-approximation result for piecewise linear models and present experiments on synthetic and real regression datasets, comparing HRT to CART, TAO, DGT, DTSemNet, and XGBoost.

### Strengths
1. The node training procedure (alternate OLS fits for two linear predictors under a hinge) is simple, transparent, and easy to re-implement. The paper provides explicit pseudocode and a complete build procedure, which supports reproducibility.
2. Compared to the baseline methods, HRT enjoys competitive performance.
3. The paper is mostly clear and easy to understand.

### Weaknesses
1. The contribution is overstated. Within fixed partitions, the update equals a Gauss–Newton/OLS step (standard), which is not difficult to prove. However, the hard part in the decision tree method is partition switching. Unfortunately, there is no guarantee of monotone decrease or convergence of the alternating fit-reassign procedure. I am not convinced that this part should be left to future work, as this guarantee is far more interesting and important than the updates in each subspace to show the validity of the proposed method.

2. The discussion of the damping effect is insufficient. For example, the results improvement for $\mu=0.01$ (damping) and $\mu=1.00$ (no damping) is only marginal in Tables 4 and 5. If using $\mu=1.00$ (OLS) and the standard fallback algorithm can provide competitive results, then the advantage of the damping term, which is a contribution of the paper, is unclear, and the reformulation into Gauss-Newton update seems to be unnecessary.

3. More benchmarks could be given to further show the effectiveness of the methods and examine the alternating fit-reassign procedure. In particular, high-dimensional datasets such as Communities & Crime (UCI), BlogFeedback (UCI), and baselines such as LightGBM can be included.

4. Some of the improvements of HRT over other methods are marginal (within 1%). Statistical significance tests should be given to potentially make many differences statistically distinguishable.

5. Although the authors discussed the complexity, there is no comparison of the computational time, especially how much time is sacrificed for using damped optimization. This makes the efficiency of the proposed method in practice hard to assess. 

6. The technical contribution of the theory is limited. The approximation of piecewise linear functions to any $C^2$ target in many similar cases is known, e.g., Breiman (1993), and ReLU approximation Barron (1993). The sample complexity of Oblique trees is also established in Cattaneo et al. (2024). Therefore, the approximation result is not surprising.

Barron (1993): Universal approximation bounds for superpositions of a sigmoidal function, IEEE Transactions on Information Theory

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes to learn an oblique decision tree with linear leaf predictors using the traditional greedy recursive partitioning approach but with the variation on the splitting procedure. The splitting procedure uses the hyperplanes at both leaves to define the splitting hyperplane: $\mathbf{\theta_1} - \mathbf{\theta_2}$. The learning algorithm uses the current split to define Newton updates, but it is not clear how it relates to finding the global optimum. Experiments are performed on synthetic and real datasets showing improved accuracy of the proposed method.

### Strengths
Oblique decision trees are an important model class which has not been as widely studied. This paper proposes one approach in learning this relatively unexplored model class.

### Weaknesses
* Objective function is defined only for a splitting criterion. It is not clear what is the global objective being optimized.

* During splitting, the internal decision node hyperplane is defined by its two linear leaf weights. Ideally, these sets of parameters (decision split hyperplane and its two children linear weights) must be independent. It is not clear why this way of coupling is used.

* It is not clear whether the proposed algorithm optimizes eq. 1. No theoretical guarantees of convergence or optimality is shown.

* The comparison experiments on real world data are not apples-to-apples. The baselines of CART and TAO, for example, use constant leaves, while the proposed approach uses linear leaves. Piecewise constant models are in general not suitable for regression problems.

### Questions
No questions.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel splitting method for oblique regression trees. It uses two linear predictors and the splitting depends on which predictor is larger or smaller. This splitting problem is solved via an alternating fitting procedure which is equivalent to a damped Newton method within fixed partitions. This paper proves that the oblique regression tree with such a splitting mechanism is a universal approximator.

### Strengths
1. This paper is well written with clear content organization, method description, and mathematical notation.
2. The splitting mechanism of oblique regression tree is novel, i.e. the combination of two linear predictors and hinge function.
3. This paper proves that the proposed method is a piece-wise linear model class which is a universal approximator. Thus its expressive power is underpinned by theoretical foundation as well as experimental results.

### Weaknesses
1. Some implementation details are missing, such as how to initialize two linear predictors.
2. The formal global convergence proof is not provided. I understand this is challenging. But I think the authors can try some weaker conclusions. For example, under what conditions is the loss function monotonically decreasing?
3. It is better to add experiments for binary classification tasks.

### Questions
On average, how many iterations are needed to split a node?

### Soundness
3

### Presentation
3

### Contribution
3
