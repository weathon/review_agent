## Human Reviewer 1

### Summary
The paper presents OBC to tackle nonsmooth composite optimization with orthogonality constraints. OBCD updates multiple rows of the matrix variable while globally solving a small nonsmooth optimization problem. The authors demonstrate that OBCD converges to block-k stationary points, providing stronger optimality guarantees than traditional critical points, and is the first greedy descent method in this domain to exhibit monotonicity. They establish strong limit-point convergence under the Kurdyka-Lojasiewicz inequality, solve subproblems with breakpoint searching techniques, and provide greedy strategies to improve computational efficiency. Extensive experiments validate OBCD's superior performance in both accuracy and efficiency compared to existing methods.

### Strengths
1. This paper proposes a very interesting row-wise BCD method for composite optimization over orthogonality constraints, which enables low-cost update.
2. The authors show that their algorithm converges to a BSk-point, which is sharper than usual critical points. This is a great point.
3. The proposed algorithm outperforms other existing methods in various problems.

### Weaknesses
1. The paper is not well-written. The organization requires a thorough revising to make this paper more readable.
2. The Asm-iii in Line 49 is restrictive. As the problem can be hard to solve. This can hinder the practical meaning of the theoretical guarantee. The authors propose to use the majorization surrogate and minimize the majorization surrogate only in k=2. Did the authors considered solving subproblems inexactly?

### Questions
1. In Figure 1, the objective function values of other methods seem unchanged. Could the authors explain the reason and elaborate more about the performance of other methods? Besides, sharper notions of stationarity do not necessarily lead to lower objective function values. It can be better to present some average results based on multiple tests.
2. Could the authors add some discussion on the possible parallel implementation of OBCD? Or explain possible difficulties of parallel implementation. Because this aspect would greatly strengthen the practical popularity of OBCD.
3. From Proposition 4.6, the KL holds locally. To apply the KL property, it is required that the iterates stay in the local region. However, the randomness in the algorithm can make the iterates fail to stay in the local region. Could the authors explain how to overcome this difficulty?
4. Could the authors revise the presentation and flow of this paper to make it more readable?
5. Could the authors add some discussion on how this BCD framework can be generalized to other manifold?
7. There are also other papers that tackle nonsmooth manifold opt using BCD (e.g., https://arxiv.org/pdf/2409.01770). Could the authors add some related references?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper is concerned with solving a composite optimization problem, whose objective is the sum of a smooth and non-smooth function, over the set of $n\times r$ orthogonal matrices, called the Stiefel manifold. The main innovation of the proposed work is to consider a form of **block-coordinate update**, in which only some, $k\leq n$, rows are updated. 

Having a method that updates only $k$-rows would have great computationally practical advanteges: 1) the gradient of the function does not have to be evaluated on all of the $nr$ entries, 2) only $kr$ entries have to be updated, 3) data can be distributed based on the updated blocks. Moreover, if the update consists of only orthogonal rotation by a $k\times k$ on the selected columns, the update automatically belongs to the feasible set.

The main contributions of the paper are:
* the concept of $k$-row update,
* strong limit point convergence of the method under KL-inequality
* greedy strategies for selecting the $k$ indices which are to be updated

### Strengths
I find the concept of updating only $k$-selected rows in the composite optimization over the Stiefel manifold to be an important and useful problem. Having good algorithms in this setting allows for all kind of additional applications, including distributional and privacy aware settings. The idea of the algorithm is straightforward to understand. The authors also focus on $k=2$ case when the iteration subproblems have closed form solution and connect these with classical notions of Givens rotations and Jacobi reflections.

### Weaknesses
While I enjoyed reading the paper, there are several points that are concerning to me.

**Convoluted presentation of some results**
At some points in the paper, there are basic linear algebra concepts presented in an unusual convoluted way. For example the start of Section 2, lines 136 to 186 basically say, that if we have an update $X^+ = V_B X$, where the matrix $V_B$ is an identity apart from a diagonal block indexed by $B$ where it has an orthogonal $k\times k$ matrix, then the next iterate will satisfy $(X^+)^\top X^+ = X^\top V_B^\top V_B X = X^\top X$ and that the distance will depend on the magnitude of the rotation $V_B$. However, since the authors write the update in an unusual form in eq. (4), this fact is, in my opinion, less apparent, than by using orthogonality of $V_B$.

**Inaccuracies**
Similarly, there is a discussion of using not only Givens rotations, stating the advantage of using Jacobi reflections since it will generate whole Stiefel manifold. This is also not entirely accurate I think. Givens rotations generate the whole SO(n) which is sufficient in solving PCA since there is an inherent ambiguity in the solution up to $\pm1$ sign anyway.

**Other minor typos/writing fixes**
* Another difficulty I had was understanding the notation, for example: $[\cdot]_BB$ appearing for the first time in line 199, becomes slightly clear only later in line 224. Overall, I would also suggest to bring more focus on the computational advantages of the formulation.
* The notation of $\mathcal{M}$ in line 187 should be Stiefel manifold I think.

### Questions
* What is the impact of different strategies for choosing $B_k$?
* How is the problem in (10) solved when $Q$ is not diagonal?
* In the proof of the convergence Theorem 4.2, in line 1495, how come you can upper bound the inequality with the stationary point of the algorithm?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
The authors study a the problem of minimizing a particular class nonconvex nonsmooth function $f$ over the space of n x r orthogonal matrices (a Stiefel manifold $St(n,r)$ ). Three standard approaches for such problems are as follows: (1) Use gradient descent-type updates in the ambient Euclidean space but use projection onto $St(n,r)$;  (2) Make gradient descent update within the tangent space and retract onto the manifold, see [2]; and (3) Iteratively construct a majorizing surrogate $\hat{f}$ of the objective $f$ and minimize it within the manifold, see [1,3]. The last approach has been particularly successful for optimization problems on Stiefel manifolds, which is the topic of the current paper, see [1]. Due to the non-convex nature of both the objective and the constraint set, researchers typically aim at obtaining first-order convergence guarantees (asymptotically or non-asymptotically with sub-linear rate), suitably defined. Convergence rate can be improved by imposing additional assumptions such as the KL property on the objective. 

The authors use an approach under the umbrella of block majorization-minimization, which progressively minimizes a majorizing surrogate of the block-restricted objective. There proposed method essentially reduces working with high-dimensional (n) orthonormal vectors to only k-dimensional (k fixed, but k=2 seems the only effective case) orthonormal columns, by subsampling k rows at random or according to a pre-determined schedule. The authors derive non-asymptotic iteration complexity to obtain approximate first-order optimal points and obtain improved convergence rate under the additional KL condition. 

There are many existing works on block majorization-minimization in both the Euclidean and the Riemannian settings for nonconvex constrained problems, so the idea itself does not seem very novel. However, there are some interesting ideas and observations that the authors make in this work specifically for Stiefel manifold problems (I will say more on them in Strengths) that can be quite valuable for high-dimensional problems on Stiefel manifolds. But, unfortunately, the paper's presentation is very poor and it cannot be considered for a publication in top ML conferences such as ICLR. The authors fail to give a cohesive narrative of their work, arguments and evidences are fragmented across the main text and the appendix, numerous typos and undefined quantities, and so on (I will say more in the Weaknesses). I think the paper has a valuable contribution, but it needs a significant rewriting from ground-up. I will need to see the whole revised paper so that these issues are successfully addressed, which definitely goes beyond that is expected to be done during the rebuttal. Therefore, overall I recommend a rejection of the paper. 


[1] Breloy, Kumar, Sun, and Palomar, "Majorization-Minimization on the Stiefel Manifold With Application to Robust Sparse PCA", IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 69, 2021

[2] Yuchen Li, Laura Balzano, Deanna Needell, Hanbaek Lyu, “Convergence and Complexity Guarantee for Inexact First-order Riemannian Optimization Algorithms.” ICML 2024 

[3] Yuchen Li, Laura Balzano, Deanna Needell, Hanbaek Lyu, “Convergence and complexity of block majorization-minimization for constrained block-Riemannian optimization” arXiv:2312.10330 (2023)

### Strengths
1. Algorithm 

The most novel part of their algorithm is that, instead of updating the entire orthogonal matrix $X\in \mathbb{R}^{n\times r}$ (here $n\gg 1$), they update only two chosen rows of $X$ by left-multiplying a suitable $2 x 2$ orthonormal matrix $V\in St(2,2)$ to those rows. The action of a large orthogonal matrix restricted on two rows does not need to be orthonormal. However, the authors make this simplifying assumption and shows that this does not loose any generality as far as first-order convergence is concerned. I think this is a nice observation. Practically, this brings down the per-iteration computational complexity to $O(n)$, which could be much less than the cost of computing the Riemannian gradient of some loss functions. The authors give an extensive details on how to solve the sub-problem involving $St(2,2)$ to make this approach executable. This algorithm could be quite practical for large-scale problems on Stiefel manifolds. 

2. Optimality measure 

The authors introduce a new measure of first-order optimality that is in general stronger than the usual (Riemannian) stationary points (block-k stationary points). Roughly speaking, the sub-problem should not be improved from the current parameter by an arbitrary adjustment using the 2 x 2 frame. This measure comes directly from the sub-problem of minimizing a majorizing surrogate over $St(2,2)$, so it depends on the particular surrogate $\mathcal{K}$ chosen. This is a somewhat unpleansent feature of their proposed optimality measure, which should in principle depend only on the information about the objective function at the given parameter (e.g., Riemannian subgradient norm of the objective). Nonetheless, the authors provide some examples (in Appendix C.1) that the number of critical points can be strictly larger than the number of their block-k stationary points. 

3. Iteration complexity 

The authors show that an epsilon-relaxation of their block-k stationarity decays is obtained after $O(\eps^{-1})$ iterations, matching  standard iteration complexity results in nonconvex optimization literature. 

4. Improved iteration complexity under KL condition: 

The authors show finite-length property (in Thm. 4.8)  under additional assumption. 

5. Extensive computation and examples for the problems on Stiefel manifold. 

The authors provide nice lemmas that are based on extensive computations using properties of Stiefel manifold along with numerical examples.

### Weaknesses
1. Presentation: 

The paper does not read very well. This is partly because of the style of the paper writing, which puts most of texts within proposition, lemma, remarks and there are not much narratives that connects them. There are numerous places that quantities are not properly defined. Important equations are written in line within text, which makes it extremely hard to read. Space constraints granted, there are still well-written, easy-to-read theoretical papers in top ML conferences. The current manuscript looks like it is a math paper crammed into 10 pages without much effort to make it readable.

(1) In the very first paragraph, the authors state assumptions in text. This is extremely hard to read and almost impossible to refer back when reading further into the paper. 

(2) 

2. Assumptions: 

In (2), the authors assume that the objective $f$ admits a quadratic surrogate of very specific form. This looks like some kind of restricted smoothness property but with the inner product depending on an additional kernel $$H$$. The authors should discuss if this is a reasonable assumption (e.g., if $f$ is $L$-smooth in the Euclidean or Riemannian sense, is it satisfied?) Otherwise, it is unclear if the paper's framework applies beyond a handful of objective functions (essentially $|| X - B||_{F}^{2}$ plus some nonsmooth regularization terms) and this makes the paper's scope very narrow. 

3. Optimality measure: 

While the optimality measure based on the authors' notion of block-k stationary points is interesting, it is hard to assess how it compares with the usual optimality measure based on (sub)gradient norms or the ones that directly relaxes the variational inequality defining stationary points. Also, another complication is that this measure involves the chosen surrogate as part of the definition, so it is not clear that if one has small BS_k measure then one has small subgradient norm or other standard stationarity measure. 

Furthermore, the authors does not show asymptotically this proposed measure goes to zero. If this result is given, then using their hierarchy between optimality measures, one can deduce that any limit point of the algorithm is a BS_k-point, which is stationary point with additional potentially desirable properties. 

4. Analysis assumes exact solver for the sub-problems: 

While in Remark 2.5 the authors claim that "For general k and h(·), the subproblem may not be solved globally, but a critical point can still be reached. Although strong optimality may be lost, a critical point (discussed later) for the final configuration X∞ remains achievable." This claim is never justified in the paper. First, even with exact computation of sub-problems, they do not show that the iterates asymptotically converge to the stationary points. Second, to support their claim, the authors extend their analysis for Thm. 4.2 with the first point here under the assumption that the sub-problems are inexactly solved. Justifying this is in fact is a non-trivial problem. See [1] for such a result for block inexact Riemannian coordinate descent. 

[1] Yuchen Li, Laura Balzano, Deanna Needell, Hanbaek Lyu, “Convergence and Complexity Guarantee for Inexact First-order Riemannian Optimization Algorithms.” ICML 2024 
 
5. Analysis under KL:

Thm. 4.8 states that essentially the iterates converge after a finite number of iterations under the additional KL assumption. The authors do not specify the the exponent $\sigma\in [0,1)$ in the desingularization function $\varphi$. This cannot be true for all range of $\sigma$ and it should only hold when $\sigma=0$. See Thm. 2.9 in [1]. 

[1] Xu, Yangyang, and Wotao Yin. "A block coordinate descent method for regularized multiconvex optimization with applications to nonnegative tensor factorization and completion." SIAM Journal on imaging sciences 6.3 (2013): 1758-1789.

5. Minor comments: 

(1) Below (2), the norm on $H$: is it the spectral norm? 
(2) L47: $\lVert X \rVert_{p}$ is not coordinate-wise separable. 
(3) L88: The authors imply that subgradient descent with dimihsing stepwise is a limitation. If the stepsizes diminish at a rate not too fast (e.g., $O(n^{-1/2})$), it still give an optimal iteration complexity. Please specify. 
(4) The authors are recommended to cite the following references and discuss their relevance/differences. 

[1] Breloy, Kumar, Sun, and Palomar, "Majorization-Minimization on the Stiefel Manifold With Application to Robust Sparse PCA", IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 69, 2021
[2] Yuchen Li, Laura Balzano, Deanna Needell, Hanbaek Lyu, “Convergence and Complexity Guarantee for Inexact First-order Riemannian Optimization Algorithms.” ICML 2024 
[3] Yuchen Li, Laura Balzano, Deanna Needell, Hanbaek Lyu, “Convergence and complexity of block majorization-minimization for constrained block-Riemannian optimization” arXiv:2312.10330 (2023)

(5) Summary: The authors state that there are not much known results about block marjoziation-minimization or similar type of algorithms for nonconvex nonsmooth problems. This is simply not true. Please revise this assessment.
(6) In Lemma 2.1, the authors define $X^{+}$ and then specify $X$. They should first say "For any $X$, define $X^{+}=$.." 
(7) In Algorithm 1, $\mathcal{K}$ is not defined nor referred.  
(8) In eq. (6), remove comma. 
(9) The construction of quadratic surrogate in (9) resembles very much the discussion in Breloy et a. [1]. The authors should discuss the relation. 
(10) L215: The authors claim that the so-constructed quadratic surrogate can be exactly minimized over $St(k,k)$. But this is the case only if the nonsmooth part is zero (by SVD) or if $k=2$. This should be clearly stated. 
(11) Table 1: The numbers are too small to read. There is no point of adding this kind of table since it is unreadable anyways. 
(12) Appendix C.3: The authors compared computational complexity of computing the Riemannian gradient directly v.s. incrementally updating the gradient using their k-row approach.

### Questions
N/A

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
3