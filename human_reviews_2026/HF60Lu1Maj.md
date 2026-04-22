# Deep Learning for Subspace Regression

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 8

## Abstract
It is often possible to perform reduced order modelling by specifying linear subspace which accurately captures the dynamics of the system. This approach becomes especially appealing when linear subspace explicitly depends on parameters of the problem. A practical way to apply such a scheme is to compute subspaces for a selected set of parameters in the computationally demanding offline stage and in the online stage approximate subspace for unknown parameters by interpolation. For realistic problems the space of parameters is high dimensional, which renders classical interpolation strategies infeasible or unreliable. We propose to relax the interpolation problem to regression, introduce several loss functions suitable for subspace data, and use a neural network as an approximation to high-dimensional target function. To further simplify a learning problem we introduce redundancy: in place of predicting subspace of a given dimension we predict larger subspace. We show theoretically that this strategy decreases the complexity of the mapping for elliptic eigenproblems with constant coefficients and makes the mapping smoother for general smooth function on the Grassmann manifold. Empirical results also show that accuracy significantly improves when larger-than-required subspaces are predicted. With the set of numerical illustrations we demonstrate that subspace regression can be useful for a range of tasks including parametric eigenproblems, deflation techniques, relaxation methods, optimal control and solution of parametric partial differential equations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The problem and a method of subspace regression (using deep learning) are proposed. The problem is to predict a subspace (i.e., an element of a Grassmann manifold) from some inputs. The authors propose loss functions, use cases of neural net-based architectures, and a way to facilitate training by regressing subspaces with larger dimensionality. The numerical examples also support the utility of the proposed framework.

### Strengths
The motivation is clearly explained with concrete examples, the proposed framework is technically sound, and the numerical examples are convincing to show the approach's effectiveness.

I assume the problem setting itself is something new, but as I am not an expert in the area, I cannot fully assess the novelty in the particular context of subspace regression.

### Weaknesses
I basically don't see major issues.

Although the paper is carefully written, a kind of schematic drawing of (an instance of) the problem and the method will be highly appreciated to facilitate smoother understanding.

### Questions
What are the most relevant studies, particularly in terms of the problem setting, i.e., regression to subspaces? As the paper does not have a distinct Related Work section, it is a little difficult to follow the exact technical context.

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
3

### Summary
This work proposes a reliable approach for subspace regression in ROM problems. To address the difficulty of recovering the exact number of subspace bases, the authors introduce a relaxation named subspace embedding that learns a higher-dimensional subspace instead, intended to sufficiently encompass the target lower-dimensional subspace and improve both accuracy and robustness at the expense of some gap between the dimensions of optimal and learned subspaces. This new formulation is evaluated in several applications and compared to different baselines to highlight its pros and cons.

### Strengths
1. Well-structured paper with clear problem definition, rigorous theoretical analysis, thorough experiments and meticulous evaluation. 
2. The subspace embedding method is novel and seems easily generalizable to different ROM problems with good robustness. 
3. Different implementations of the formulation are given in Theorem 1. 
3. Both pros and cons of this method are discussed in detail, the explanations are insightful and intuitive. 
4. Several nice illustrations helping readers appreciate the theoretical analysis in the appendix.

### Weaknesses
Overall, this is a solid paper without significant flaws. Just a few minor ones. I'm willing to increase the score if they are addressed and my questions are answered. 

1. Seems like a $\psi(x)$ next to $U(x)$ is missing in (3).
2. I think a clear definition of "relative error" is needed. See Question 1 below.
3. Use \citep and \citet properly.
4. Some sentences could be streamlined and a few typos can be fixed.

### Questions
1. How is "relative error" defined? $W_\theta$ and $V$ should have different ranks ($r$ and $k$), right? 
   * Maybe related to the Frobenius norm in Theorem 2?
   * Is the definition of "relative error" the same for different models in Figure 2?
2. Will subspace embedding perform worse as the ambient dimension $n$ increases? I assume that as $n$ increases, the low-dim linear subspaces will be more likely orthogonal with one another and make subspace embedding trickier. Is this intuition reasonable?
   * In other words, what would Figure 1 approximately look like if $n$ is the third dimension apart from $r$ and error?
3. How impactful is the gap $r-k$ to different ROM problems when the predicted larger subspace is used? 
   * Any postprocess for trimming the predicted subspace to reduce its rank to $k$?
4. How trainable or reusable is a subspace regressor for a collection of problems of varying dimension $n$ or domain $\Omega$? 
   * For instance, FEA is widely used to solve PDEs over domains of complicated geometries, and its approximate solution often takes the form on Line 107. How applicable is your method considering that the number of $\phi_i$ often changes due to either the varying nonlinearity in the solution field or the varying $\Omega$?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies a subspace regression problem that aims to learn a mapping from a parameter set to a manifold of subspaces. One central challenge is that the target of the regression is itself a *set of spaces*, rather than points in a Euclidean domain. The main contributions lie in designing appropriate loss functions and embedding techniques that allow these spaces to be compared, optimized over, and represented effectively within a learning pipeline.

### Strengths
The problem is challenging, and the numerical results appear promising.

### Weaknesses
There is no comparison with classical non–learning methods. For example, in the eigenvalue problem, when approximating the smallest eigenvalue, conventional numerical approaches are already highly efficient and accurate. Without such baselines, it is difficult to assess whether the proposed learning framework offers any practical advantage.

The presentation is weak, making the article difficult to read. Beyond apparent typos (see below), the referee identifies the following concerns:

   1. Above Eq. (1), both matrices and parametric models are denoted by $W$, which is confusing.

   2. Above Eq. (1), in the general setting, it would help to specify $V(r)$ explicitly for each example discussed later.

   3. In the approximate eigenspaces example, $U$ and $E$ should be defined. On line 113, the object described is not a subspace.

   4. On line 141, $V$ should be a matrix rather than a subspace based on context. If $V$ were a subspace, $S(V)$ would be ill-defined.

   5. In Theorem 3, the mappings $F$ and $G$ should be indexed as $F_k$ and $G_k$, since they are $k$-dependent. The quantities $\#_F(k, D)$ and $\#_G(k, D)$ are not defined.

   6. The sentence “As the main neural network architecture we used FFNO Tran et al. (2021), a descendant of FNO Li et al. (2020), and performed extensive grid searches for all experiments” is unclear; its relevance and meaning should be clarified.


Numerous grammar issues:

   1. Line 107: “to approximates” → “to approximate”
   2. Line 112: “is solver repeatedly” → unclear phrasing
   3. Line 206: “we allowed” → revise for tense consistency
   4. Line 261: “Theorem 3 suggest” → “Theorem 3 suggests”
   5. Line 304: missing punctuation

   …and many others throughout the manuscript.

### Questions
There are several technically unclear points in the presentation:

1. Line 209: the meaning of “interpolation” is unclear and should be defined precisely in this context.

2. Theorem 3: it is not explained why the functions $F_k$ and $G_k$ are piecewise constant, nor why they take distinct values. A justification or reference is needed.

3. The description of the **SUBSPACE EMBEDDING** technique is very unclear. How does it work in practice? Does it imply that the learnable data consist of higher-dimensional subspaces, and if so, how are they embedded and compared?

4. In the tables of numerical results, the meaning of the reported percentages is not explained. Do they represent accuracy, relative error, or improvement over a baseline?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Subspace Regression, a learning framework in which the target is a linear subspace rather than a vector or a scalar. It arises naturally in reduced-order modeling (ROM), eigenspace approximation, deflation methods, and parametric PDEs. It formulates the subspace regression on the Grassmann manifold $Gr(k, n)$, proposes two new loss functions invariant under right $GL(k, V)$ actions, and proposes the subspace embedding technique, which predicts a higher-dimensional subspace that contains the target, theoretically shown to smooth the mapping and reduce complexity. I mainly focused on checking the proofs of the theorems and the examples, which look very rigorous to me.

### Strengths
1. Formally consider the subspace regression problem with differential geometry. 
2. The list of example problems (Sec. 2.2) to solve with subspace regression is very valuable and illustrative, IMHO. 
3. I personally like the writing quite a bit, which is concise and right to the point; not much redundancy in the sentences. 
4. Theorem 1–3 rigorously justify the loss functions and the subspace embedding principle using differential geometry and asymptotic complexity arguments.
5. Comparisons with classical interpolation, FNO, DeepONet, and POD variants show consistent advantages, especially in high-dimensional parameter spaces.

### Weaknesses
1. Tiny suggestion: for QR factorization, it is more precise to say $A = Q_A R_A$, where $Q_A\in \mathbb{R}^{n\times k}$ since for a tall matrix $A$, you can factorize it into a unitary $Q \in \mathbb{R}^{n\times n}$.
2. Another small suggestion, in Thm. 1, for $L_1$, if $A = B \cdot G$, for some $G\in GL(k, V)$, then $L_1(A, B) = 0$, which explains the $p$ term in the loss, which might look a bit weird at first look. 
3. Yet another thing: for the parametrization of $V$, you wrote $V(r) \colon \mathbb{R}^p \rightarrow Gr(k, n)$ (which I understand), which, however, confused me at the first glance: I thought it mean for each $r$, $V(r)$ is a map from $\mathbb{R}^p$ to $Gr(k, n)$. Better to write $V \colon \mathbb{R}^p \rightarrow Gr(k, n)$.
4. Thm. 2 shows that larger subspaces help with reducing the magnitude of the tangent vectors, but it doesn’t quantify why it helps with optimizing the loss function, e.g., I think Lipschitz analysis of the tangent vectors might help. 
5. Many results are reported as single-run metrics without confidence intervals or error bars.

### Questions
1. Just a tiny suggestion: Sec. 2.1, I suggest using [W] to denote the equivalence class. 
2. Line 88, why the parameteric model maps to Gr(r, n), r \geq k, which is more expressive than needed?
3. In the Schrödinger eigenproblem, I do not really get why it is a subspace regression problem. Do you wish to learn the coefficients or the basis $\phi_i$? Also, in line 113, what is $f$? $\alpha_i$ should be complex-valued.
4. How is the embedding dimension $r > k$ chosen in practice? 
5.  How sensitive is the training to the choice of r.v. in the stochastic loss $L_2$?
6. For high-dimensional PDEs, does the cost of orthogonalization or QR decomposition outweigh the benefits of manifold losses?

### Soundness
4

### Presentation
4

### Contribution
3
