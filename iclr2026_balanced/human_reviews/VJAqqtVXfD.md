## Human Reviewer 1

### Summary
This paper introduces a method to solve the k-Hyperplane Clustering problem in the 2-norm ($k-HC_2$) to global optimality using spatial branch-and-bound (SBB). The approach strengthens the classical mixed-integer quadratically constrained quadratic programming (MI-QCQP) formulation by incorporating constraints from polyhedral norm variants $(p = 1, \infty)$. The authors show that including ∞-norm constraints allows the SBB method to obtain a nonzero lower bound in O(nk) nodes, compared to $\Omega(2^{k(n−1)})$ nodes for the baseline. Experiments on synthetic datasets report speedups of 8–41 times, improving the number of instances solved to global optimality.

### Strengths
1. Theoretical analysis proves that adding polyhedral norm constraints reduces the number of SBB nodes required to obtain a nonzero lower bound from exponential to polynomial.
2. The method is evaluated on two synthetic testbeds (Low-dim and High-dim) with statistical tests (Wilcoxon signed-rank) confirming significant speedups.
3. Clear MI-QCQP and MILP formulations are provided for the polyhedral norm constraints.
4. The multi-norm relaxation framework is general and may be applied to other problems with nonconvex norm constraints.

### Weaknesses
1. Experiments are limited to small-scale instances (up to m = 30, n = 5, k = 5) and use synthetically generated data with a specific noise model.
2. The theoretical advantage of $\infty$-norm constraints assumes branching occurs first on the binary variables of the polyhedral norm formulation; performance under default branching strategies is not tested.
3. The analysis focuses only on the number of nodes to the first nonzero bound, not the total tree size or gap closure.
4. No comparison with specialized heuristics or state-of-the-art approximate methods is provided.
5. Validation is limited to synthetic data; real-world applicability is not assessed.
6. The method requires solving complex MI-QCQPs, which may not scale to very large instances.
7. There is also no analysis of the trade-off between solution quality and computation time for practical use.
8. The paper does not provide guidance on selecting which polyhedral norm constraints to include for best performance.

### Questions
1. How does the method scale with the number of data points m, given that the assignment problem may become the bottleneck?
2. Can the authors provide wall-clock time profiles and total node counts (not just medians) to better characterize tree size and convergence?
3. How robust are the speedups under solver-default branching rules and parameters?
4. Why does the multi-norm formulation perform worse than the individual $\infty$-norm formulation in some cases?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
The authors give a spatial branch-and-bound (SBB) method to solve the k-hyperplane clustering problem (k-HC2), which minimizes the sum of squared Euclidean distances from points to their nearest hyperplane. By updating the MI-QCQP formulation with constraints from alternative p-norms (especially polyhedral norms like p = 1, $\infty$), they show that these additions improve SBB efficiency, reducing the number of nodes needed for nonzero lower bounds from exponential to linear under mild assumptions. Empirical results confirm speedups on the two listed datasets.

### Strengths
1. The k-Hyperplane Clustering problem (k-HC2) is a fundamental problem in classical machine learning. Addressing the k-HC2 problem using the a MI-QCQP formulation with constraints would be of interest to the community, particularly in the context of subspace clustering.

2. The main contribution of this work lies in using a multi-norm approach to estimate the lower bound for the objective value of k-HC_{2,1}, as compared to using a single polyhedral norm. In this way, the authors enable earlier pruning in the SBB procedure. Furthermore, under specific assumptions (as shown in Proposition 4), the authors provide theoretical guarantees for the use of two scaled polyhedral-norm constraints in the k-HC(2,1) formulation.

3. The proposed algorithm shows empirical speedups on datasets.

### Weaknesses
1. The paper does not discuss prior research in subspace clustering, which also aims to identify k subspaces that minimize the sum of squared Euclidean distances between each data point and its closest subspace [1][2][3]. 

2. The idea of using polyhedral norms and the SBB methods for k-HC2 problem has already been explored in several prior works (e.g., Dhyani and Liberti (2008), Amaldi & Coniglio (2013)). While this paper introduces a modification by incorporating multiple norm constraints($\ell_1$-norm and $\ell_{\infty}$-norm) for approximation purposes, the contribution may be viewed as incremental.

3. The problem remains computationally challenging for large-scale instances, and the paper does not clearly specify the time complexity of the proposed algorithm.

Reference:

[1]. Rademacher L, Vempala S, Wang G. Matrix approximation and projective clustering via iterative sampling[C]. Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm, 2006, Pages 1117 - 1126.

[2]. Sohler, Christian, and David P. Woodruff. Strong coresets for k-median and subspace approximation: Goodbye dimension. 2018 IEEE 59th Annual Symposium on Foundations of Computer Science. IEEE, 2018.
 
[3]. Eiben, Eduard, et al. EPTAS for k-means clustering of affine subspaces. Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms, 2021.

### Questions
1. How does the proposed method compare to existing subspace clustering approaches [2-3] in terms of formulation and performance?

2. What is the time complexity of the proposed algorithm, and how does it scale with data size?

3. What are the practical applications of the k-HC² algorithm, and in which domains does it offer clear advantages?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper focuses on improving spatial branch-and-bound (SBB) algorithms for optimally solving the 2-norm k-hyperplane clustering $k$-$HC_2$ problem, which seeks to choose k hyperplanes in a way that minimize the squared norm distance between each point and its nearest hyperplane.

The paper first considers a generalized objective $k$-$HC_{(p,c)}$ that considers the $p$-norm and a scaling constant $c$ in the constraints, and proves that optimal solution for  $k$-$HC_{(q,c')}$ for a careful choice of $c$ provides an optimal solution for $k$-$HC_{(p,c)}$. These bounds can be used specifically for the purpose of finding problems that approximate $k$-$HC_{(2,2)}$ = $k$-$HC_{2}$. The paper then focuses especially on approximations that are obtained for $k$-$HC_{2}$ in terms of $k$-$HC_{(\infty,1)}$ and $k$-$HC_{(1,1/\sqrt{n})}$. They then show how this leads to a strengthened formulation for  $k$-$HC_{2}$, which has a faster solution using SBB techniques. In particular, Section 4 proves that only a linear number of nodes in the branch-and-bound is needed to get a non-zero global lower bound for the new formulation, whereas an exponential number of nodes are needed to get a non-zero global lower bound for a basic formulation.

The paper shows in numerical experiments that the new approach leads to faster solve times in practice.

### Strengths
The topic of the paper is interesting and well-motivated from previous research.

The mathematical contributions of the paper is non-trivial; the technical contribution of the paper is quite high. 

Even though the paper is proving a large number of non-trivial technical results, the presentation is very good and explains the main ideas and components (and how they fit together) very well. The approach makes sense at a high level, and the paper does a good job presenting enough technical detail in the main text while knowing what to push to the appendix.

The technical approach leads to concrete improvements in numerical experiments.

### Weaknesses
There were a couple places where the technical details were a little unclear and could have been explained a bit better:

Lemma 2 seems written somewhat informally and I do not follow what it is saying. In particular, it states that imposing a certain inequality "coincides with accounting for each point to hyperplane distance as ...". I don't understand precise meaning of the wording "coincides with accounting for", and hence I do not follow the meaning or significance of this lemma.

It's also not that clear how Lemma 2 is used in the main results. It's clear how Lemma 3 and Theorem 2 together produce the useful bound in corollary 2, but what role does Lemma 2 play in all this? 

Figures 1 and 2 could use a bit more explanation, e.g., in the caption. At first something seemed backwards, because the text explained how the feasible region of a (q,c) problem with $q = 1, \infty$ and $c = 1, 1/\sqrt{n}$ contains the feasible region for $p = 2, c = 1$, but this appeared opposite to the pictures where a circle contained a red diamond and green square. It did not take me long to realize that this is because the feasible region is everything that lies *outside* these colored shapes, but it would have made things a lot clearer a lot quicker to note this explicitly, rather than just stating these are illustrations of certain feasible regions. Also, in the captions, shouldn't you write 2 instead of n in $\mathbb{R}^n$?

This is minor, but there are also some typos to fix:

* lemma 1: coincide Also ---> add missing period
* line 072: "d_p intrinsically" --> missing the word "is"?
* line 055: "nonzero global lower bounds is" --> bound is or bounds are

### Questions
Can you clarify the meaning of Lemma 2 and it's purpose?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3