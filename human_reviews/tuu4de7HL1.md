# Improving Convergence Guarantees of Random Subspace Second-order Algorithm for Nonconvex Optimization

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6

## Abstract
In recent years, random subspace methods have been actively studied for large-dimensional nonconvex problems. Recent subspace methods have improved theoretical guarantees such as iteration complexity and local convergence rate while reducing computational costs by deriving descent directions in randomly selected low-dimensional subspaces. This paper proposes the Random Subspace Homogenized Trust Region (RSHTR) method with the best theoretical guarantees among random subspace algorithms for nonconvex optimization. RSHTR achieves an $\varepsilon$-approximate first-order stationary point in $O(\varepsilon^{-3/2})$ iterations, converging locally at a linear rate. Furthermore, under rank-deficient conditions, RSHTR satisfies $\varepsilon$-approximate second-order necessary conditions in $O(\varepsilon^{-3/2})$ iterations and exhibits a local quadratic convergence. Experiments on real-world datasets verify the benefits of RSHTR.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper offers an extension of current random subspace methods to  the Homogenized Trust Region method of Zhang et al. At each iteration, a random Gaussian matrix is sampled, and they use it to project the gradient and hessian. A trust region problem is then solved in the reduced space by an eigendecomposition. The theoretical analysis proves convergence to an $\epsilon$-FOSP in complexity $O(\epsilon^{-3/2})$, under the assumption of a Lipschitz gradient and Lipschitz Hessian. Under the further assumption of a low-rank Hessian at the output point, the authors prove convergence to an $\epsilon$-SOSP in complexity $O(\epsilon^{-3/2})$. Finally, under strict convexity, the authors prove local linear convergence, and under the assumption of a low-dimensional Hessian, the authors get quadratic convergence.

### Strengths
The main strengths of this work are reduced complexity, both in memory and computational time.  The high-probability bounds developed also point to a clear choice of dimension in $s\asymp \log n$. This is to make the probabilities vanishingly small in the dimension. The reduced trust region problem reduces to an eigenvalue problem, just as in past work by Zhang et al 2022, which can be solved efficiently in the lower dimension using standard methods. Overall, this represents an important addition to the nonconvex optimization literature.

### Weaknesses
There are a few weaknesses to this paper. First, the paper should be made more clear. For example, in theorems 3.3 and 3.4, it is unclear what “sufficiently large $k$” means. If the authors could provide statements that make this more precise, that would make the statements of the theorems stronger.

Some things are introduced without proper motivation or definition. For example, what problem does 2.2.1 solve? I know that it somehow comes from the past work and solves a trust region problem, but the paper would be improved with further discussion to motivate the problem. 

In the section discussing overall complexity, the statements include an epsilon that references the later theorem on convergence to an $\epsilon$-FOSP. However, this theorem only applies under certain assumptions -- the authors should make clear in this section that they are examining complexity under these same assumptions.

Another weakness is that the theoretical innovations are not entirely clear. The authors should comment on the theoretical innovations within the text since proofs are pushed to the appendix. Is the theory just a mashup of Zhang et al. 2022 and the JL lemma, or is something more intricate involved?

The assumption of low-rankness in the SOSP and quadratic convergence results seem quite restrictive. While it may be necessary for the quadratic convergence, is Assumption 2 necessary, or is there some weaker condition that could lead to quadratic convergence?  A discussion of the proof techniques would help here.

### Questions
- Choosing $s= \Omega(n)$ seems to be the recommended dimensional choice in practice. If one ignores the high-probability bounds, could one show a convergence guarantee in expectation for smaller subspace dimensions under the various assumptions?
- I am a little confused about the probability in the theorem statements of linear and quadratic convergence. Is it that you get a linear/quadratic decrease for a certain iteration with this probability, or is it that this holds uniformly over iterations with this probability? I suspect it's the former, but I want it to be clear.

### Soundness
3

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
2

### Summary
This paper studies algorithms for solving unconstrained non-convex optimization problems of the form $\min_{x\in \mathbb{R}^n}f(x)$ for a non-convex function $f:\mathbb{R}^n\rightarrow \mathbb{R}$, in particular algorithms for finding an $\epsilon$-approximate first order stationary point. The paper considers algorithms that first pick a random small dimensional subspace of $n$-dimensional space and search for the best update direction in that subspace. The main idea of the paper is to develop a random subspace trust region method, and the paper is the first to give a complexity analysis for this class of methods. They show that their algorithm converges to an $\epsilon$-approximate first order stationary point with a global convergence rate of $\epsilon^{-3/2}$ iterations as well as local quadratic convergence under the assumption that $f$ is strongly convex. The main idea behind their work to make the analysis of random subspace trust region methods feasible is the minimum eigenvalue formulation based analysis of the trust region subproblem previous work of Zhang et al (this is a trust region based algorithm without random projections). This approach makes it feasible to analyze the effect of random projections in trust region based methods.

### Strengths
The main strength of the paper is present the first theoretical analysis of random subspace based trust region methods for non-convex optimization problems. The guarantees also match the best known guarantees for random projection based methods for non-convex optimization.

### Weaknesses
One minor weakness of the paper is that in the experiments section, the comparison of the proposed algorithm is not done with popular heuristic optimization algorithms such as adagrad or adam for experiments on deep neural networks.

### Questions
I would be interested in knowing the performance of the proposed algorithms with popular heuristics for training deep neural networks such as adagrad or adam.

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
2

### Summary
This paper is primarily inspired by the HSODM method developed by Zhang et al. (2022) and extends their ideas to the domain of random subspace methods. The authors  propose a new approach called the Random Subspace Homogenized Trust Region (RSHTR) method. They establish global convergence guarantees under specific conditions and provide theoretical proofs for convergence to a second-order stationary point. Additionally, the paper demonstrates local linear and superlinear convergence under certain assumptions. Experiments conducted on real-world datasets confirm that RSHTR offers significant computational advantages and improved convergence properties over previous algorithms.

**Reference**: Chuwen Zhang, Dongdong Ge, Chang He, Bo Jiang, Yuntian Jiang, Chenyu Xue, and Yinyu Ye. A homogenous second-order descent method for nonconvex optimization. arXiv preprint arXiv:2211.08212, 2022.

### Strengths
- The paper is well-structured and provides a clear presentation of the proposed algorithm. It establishes a strong theoretical foundation, presenting the first theoretical results for random subspace methods that demonstrate quadratic convergence properties for certain classes of functions.
- The experimental setup is appropriate, with sufficient comparisons made against existing methods. The results clearly illustrate both the practical convergence and favorable computational complexity of the proposed algorithm.

### Weaknesses
- Despite the logical layout, the paper contains numerous typos and punctuation errors, which negatively affect readability and the overall presentation. 
- The paper mentions several constants (e.g., $C, c,\bar{C}, \bar{c}$) but does not provide detailed information regarding them throughout the text and appendix. It would be beneficial to clarify what parameters these constants are associated with, as they are likely not arbitrary values. 
- While the theoretical contributions are substantial, the authors note that it is also possible to provide theoretical guarantees by adopting a standard line search strategy in RSHTR to compute $\eta_k$.
- To enhance the practical significance of the theoretical guarantees, it would be valuable to include additional experiments, such as classification accuracy tasks, to further validate the algorithm's effectiveness. Nonetheless, the current experimental results are already quite strong. Additionally, Figure 5 in Appendix F is missing a title and legend.

### Questions
1. In line 43 and line 168, the notation "subspace $P_{k}^{\top}\mathbb{R}^{s}$" appears. Could the authors please confirm whether this should be corrected to "subspace $P_{k}\mathbb{R}^{n}$" and ensure that the correct notation is used consistently throughout the paper?
2. In Table 1,  could the authors provide a legend or explanation for the symbols "1" and "2" in the Local column? This would help clarify the meaning for readers and improve understanding of the information presented.
3. In line 170, the normal distribution $N(0,1/s)$ is used, with variance chosen as $1/s$? Could the authors provide additional context or a reference to explain this choice of variance, as it would assist readers who may be less familiar with random subspace methods?
4. For equation (2.2.1), the reference to the "homogenization trick" could be expanded. Could the authors either provide a citation or briefly explain how this trick is derived such as Zhang et al. (2022)?
5. In line 184, the authors state that "unlike HSODM, RSHTR allows for other eigensolvers beyond the Lanczos method." However, the experimental section does not specify which algorithm was used for solving. Could the authors clarify this point?
6. In line 255, I understand that HSODM requires calculating $n+1$ Hessian vector products, resulting in a computational complexity of $O(n^2)$, while RSHTR requires $s + 1$ Hessian vector products. However, if RSHTR does not need to store the Hessian, why does HSODM require storing it?
7. In lines 318 and 360, could the authors clarify the relationship between the gradient norm and the relative function values? This would help in understanding why the complexities are expressed as $(n/s)^{3/4}$ and $(n/s)^{3}$.
8. In Algorithm 2, I noticed a difference in the stopping criterion compared to HSODM in Zhang et al. (2022). Could the authors elaborate on the equivalence of the stopping criteria used in both algorithms?
9. In line 1480 and below, the notation $\sqrt{\bar{H}}$ is not defined. Could the authors provide clarification on this notation?

**Reference**: Chuwen Zhang, Dongdong Ge, Chang He, Bo Jiang, Yuntian Jiang, Chenyu Xue, and Yinyu Ye. A homogenous second-order descent method for nonconvex optimization. arXiv preprint arXiv:2211.08212, 2022.

The article contains several typographical and punctuation errors that require careful review by the authors. For example:

- Line 28: "as follows. " should be "as follows:".
- Line 110: "Local" should be "local".
- Lemma 3.4, lines 728, 736, 741, Lemma B.3, line 1345, and so on: unctuation errors are present. I recommend that the authors conduct a thorough review of the entire paper. 
- Line 372 and 417:  "global_mode" should be "False".
- Line 1525: $\||A_3+A_4\||$ should be $\||A_3-A_4\||$.

### Soundness
3

### Presentation
2

### Contribution
3
