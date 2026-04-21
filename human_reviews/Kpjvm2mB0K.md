# Streaming Algorithms For $\ell_p$ Flows and $\ell_p$ Regression

- Avg Score: 8.00
- Decision: Accept (Spotlight)
- Scores: 8, 8, 8

## Abstract
We initiate the study of one-pass streaming algorithms for underdetermined $\ell_p$ linear regression problems of the form
  $$
      \min_{\mathbf A\mathbf x = \mathbf b} \lVert\mathbf x\rVert_p \,, \qquad 
      \text{where } \mathbf A \in \mathbb R^{n \times d} \text{ with } n \ll d \,,
  $$
  which generalizes basis pursuit ($p = 1$) and least squares solutions to
  underdetermined linear systems ($p = 2$). We study the column-arrival
  streaming model, in which the columns of $\mathbf A$ are presented one by one in a
  stream. When $\mathbf A$ is the incidence matrix of a graph, this corresponds to an
  edge insertion graph stream, and the regression problem captures $\ell_p$
  flows which includes transshipment ($p = 1$), electrical flows ($p = 2$), and
  max flow ($p = \infty$) on undirected graphs as special cases. Our goal is to
  design algorithms which use space much less than the entire stream, which has
  a length of $d$.

  For the task of estimating the cost of the $\ell_p$ regression problem for
  $p\in[2,\infty]$, we show a streaming algorithm which constructs a sparse
  instance supported on $\tilde O(\varepsilon^{-2}n)$ columns of $\mathbf A$
  which approximates the cost up to a $(1\pm\varepsilon)$ factor, which
  corresponds to $\tilde O(\varepsilon^{-2}n^2)$ bits of space in general and
  an $\tilde O(\varepsilon^{-2}n)$ space semi-streaming algorithm for
  constructing $\ell_p$ flow sparsifiers on graphs. This extends to $p\in(1,
  2)$ with $\tilde O(\varepsilon^{2}n^{q/2})$ columns, where $q$ is the H\"older
  conjugate exponent of $p$. For $p = 2$, we show that $\Omega(n^2)$ bits of
  space are required in general even for outputting a constant factor
  solution. For $p = 1$, we show that the cost cannot be estimated even to an
  $o(\sqrt n)$ factor in $\mathrm{poly}(n)$ space.

  On the other hand, if we are interested in outputting a solution $\mathbf
  x$, then we show that $(1+\varepsilon)$-approximations require $\Omega(d)$
  space for $p > 1$, and in general, $\kappa$-approximations require
  $\tilde\Omega(d/\kappa^{2q})$ space for $p > 1$. We complement these lower
  bounds with the first sublinear space upper bounds for this problem, showing
  that we can output a $\kappa$-approximation using space only
  $\mathrm{poly}(n) \cdot \tilde O(d/\kappa^q)$ for $p > 1$, as well as a
  $\sqrt n$-approximation using $\mathrm{poly}(n, \log d)$ space for $p = 1$.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The $\ell_p$ regression $\min_{\mathbf{Ax} = \mathbf{b}} ||\mathbf{x}||_p$ is a fundamental problem in machine learning, data science, and numerical linear algebra. When $p = 2$, it is the classical linear regression problem and has a closed-form solution $\mathbf{x}^* = \mathbf{A}^\top (\mathbf{A}\mathbf{A}^\top)^{-1} \mathbf{b}$. The general $\ell_p$ regression has been well-studied, in particular, when $\mathbf{A}$ is the vertex-edge incidence matrix, it is the $\ell_p$-norm flow problem, which corresponds to transshipment, Laplacian solver, and maximum flow if $p = 1, 2$, and $\infty$ respectively. This paper considers the scenario that the matrix $\mathbf{A} \in \mathbb{R}^{n \times d}$ has $n \ll d$, which leads to the linear system $\mathbf{Ax} = \mathbf{b}$ being underdetermined. In this case, it is impractical to store all the $d$ columns of $\mathbf{A}$, which motivates the usage of the column-arrival streaming model. In addition, there are two kinds of solutions to $\ell_p$ regression: (1) $\textit{cost estimation}$ that approximates the optimal $\ell_p$ norm; (2) $\textit{vector-valued}$ that approximates the optimal solution $\mathbf{x}^*$. For the first kind, this paper achieves space complexity $\widetilde{O}(\epsilon^{-2} n^{\ 1+max\\{1, p/2(p-1)\\}})$, which is $\widetilde{O}(\epsilon^{-2} n^2)$ when $p > 2$, and also gives lower bounds for $p = 0, 1, 2$. For the second kind, this paper also provides lower bounds and upper bounds for different values of $p$.

### Strengths
(1) This paper researches the general $\ell_p$ regression problem under the column-arrival streaming model, while the existing work only considers the special case, like $p = 2$ or $\mathbf{A}$ is the vertex-edge incidence matrix.  
(2) For general $p$, this paper establishes both lower and upper bounds on space complexity for the two kinds of solutions: cost estimation and vector-valued problems. These new results strengthen the paper's contributions.  
(3) This paper has an extensive set of references and a comprehensive summary of the related work.

### Weaknesses
The $p$-norm flow problem defined in (1) (Line 61) is not exact. The correct one should be $\min_{\mathbf{Ax} = \mathbf{b}} ||\text{diag}(\mathbf{w}) \cdot \mathbf{x}||_p$. To be specific, when $p = 1$ and $\mathbf{w}$ is the cost vector, then it is a transshipment problem; when $p = 2$ and $\mathbf{w} = \mathbf{r}^{1/2}$, where $\mathbf{r}$ is the resistance vector, then it is reduced to a Laplacian solver; when $p = \infty$ and $\mathbf{w} = \mathbf{c}^{-1}$, where $\mathbf{c}$ is the capacity vector, then it is the maximum flow problem.      

By this formulation, it is the same as formulation (2). As a special case, when $\mathbf{A}$ is a vertex-edge incidence matrix in the graph streaming problem, there is some existing work (Line 91-97) on transshipment, electrical flow, maximum flow, etc. Could you compare this work with their results?

### Questions
(1) In Theorem 1, Line 153, should "$s$ rows" be "$s$ columns"? At Line 164, should the second $\widetilde{O}(\epsilon^{-2} n)$ be $\widetilde{O}(\epsilon^{-2} n^2)$? Please verify that.     
(2) In Theorem 1, when $p \to 1$, like $p = 1.1$, $q = p/(p-1)$ would be a large constant. In this case, could you discuss the behavior of the proposed algorithm? If it is indeed a non-trivial issue, could you give a brief analysis or comment in this paper on $p \to 1$?

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
5

### Summary
The paper initiates the study of $p$-norm regression in the one-pass streaming setting. They consider the underdetermined setting $\min_{Ax=b} \|x\|_p$ where $A$ has size $n\times d$, $n<<d$, receives $d$ column updates. They consider $p\in [1,\infty)$. When we consider instances graphs, these updates correspond to edge insertions. 

The first result is an algorithm for “p-norm flow sparsifier” in $O(n^2)$ space which is a general version of graph sparsifiers which minimize the $\ell_p$-norm. This algorithm follows from using the online lewis weight sampling algorithm of Woodruff and Yasuda’22 on the dual problem. The other algorithmic result deals with giving a tradeoff between the size of the sparsifier and objective error when the goal is to maintain the entire solution vector x. The first algorithm only maintains the objective value.

The paper also presents several information theoretic lower bounds which match these upper bounds. They also give some extra lower bounds for the case of $p=2$.

### Strengths
The paper studies an important problem and makes a significant amount of progress in the new direction of streaming algorithms for regression. They also leave some good open questions in the paper. The techniques are also quite simple and use previous works quite well. The main technical core is the lower bound part, which is of independent interest.

### Weaknesses
It seems like there are too many ideas and previous results used in the paper. Would be useful to have some more preliminaries in the appendix or somewhere.

### Questions
1. Can the authors add a comparison with what is known in online algorithms in the corresponding settings?
2. The tables in the paper, that give the lower and upper bounds for different settings, can these be split in a different way so that its easier to contrast the lower and upper bounds for all settings? This would also make clear what settings are still unknown.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper gives upper bounds and lower bounds for streaming underdetermined $\ell_p$ linear regression in the column arrival model. The results apply both to the value of the best fit, and to the actual solution. There are various results, depending on the value of $p$ and the approximation guarantees. The arguments use duality and sparsification.

### Strengths
The paper explores the full range of values of $p$.

### Weaknesses
The methods rely substantially on past work.

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
3
