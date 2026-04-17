# The Multi-Block DC Function Class: Theory, Algorithms, and Applications

- Decision: Reject
- Scores: 6, 4, 8, 4

## Abstract
We present the Multi-Block DC (BDC) class, a broad class of structured nonconvex functions that admit a DC (“difference-of-convex”) decomposition across parameter blocks. This block structure not only subsumes the usual DC programming, it turns out to be provably more powerful. Specifically, we demonstrate how standard models (e.g., polynomials and tensor factorization) must have DC decompositions of exponential size, while their BDC formulation is polynomial. This separation in complexity also underscores another key aspect: unlike DC formulations, obtaining BDC formulations for problems is vastly easier and constructive. We illustrate this aspect by presenting explicit BDC formulations for modern tasks such as deep ReLU networks, a result with no known equivalent in the DC class. Moreover, we complement the theory by developing algorithms with non-asymptotic convergence theory, including both batch and stochastic settings, and demonstrate the broad applicability of our method through several applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Multi-Block DC (BDC) function class, a generalization of difference-of-convex (DC) programming to functions that admit DC decompositions per parameter block. The authors provide explicit, constructive BDC formulations for practical problems, including deep ReLU networks (with extensions to MSE regression and cross-entropy classification losses). The paper further develops the Block DC algorithms, including proximal and stochastic versions, with non-asymptotic convergence guarantees. Empirical performance is also illustrated through experiments.

### Strengths
1. The BDC class extends DC programming to multi-block settings, addressing practical challenges in finding global DC decompositions. The BDC definition is simple yet powerful. 

2. The paper provides explicit tools and examples for formulating problems as BDC, making it accessible for applications in machine learning (e.g., tensor decomposition, neural network training).

3. The proposed BDCA algorithms, including stochastic extensions, come with rigorous non-asymptotic convergence rates, which are solid and clearly stated.

### Weaknesses
1. While the abstract mentions "several experiments", the paper focuses heavily on theory and algorithms. The current results compare mainly to vanilla SGD and on modest datasets. Full evaluation (e.g., benchmarks against baselines like ADAM on real datasets) is not enough, potentially weakening claims of practical superiority.

2. For the deep ReLU BDC formulation, how does it handle common extensions like batch normalization, dropout, or other activations (e.g., GELU)? Is there a way to extend the constructive approach to these?

3. The paper should also talk about computation and memory overhead of the proposed methods versus standard training.

### Questions
See the Weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors study a class of (multi-)block DC programs that is **provably** broader than the classical class of DC programs. Specifically, given an objective function $f(\theta)$ with $\theta \in \mathbb{R}^d$, it is said to be a multi-block DC function if there exists a partition of the $d$ coordinates into $n$ blocks such that $f$ admits a DC decomposition with respect to each block of coordinates when the remaining blocks are fixed.

The main motivation stems from the observation that identifying a block DC structure is often much easier than finding a full DC structure, especially in problems with coupled variables. The authors further demonstrate that for monomials, an exponential number of atoms is required to represent them as DC functions, whereas only a polynomial number of atoms suffices under the block DC formulation.

Exploiting this block DC structure, the authors propose a block DC algorithm, which combines the principles of DCA with randomized block selection. Under the assumption that the first DC components (in each block) are $L$-smooth (or generalized $L$-smooth), the squared gradient norm of the objective function---or its expected value in the stochastic case---converges to zero at a rate of $\mathcal{O}(1/K)$, matching the known rate for both stochastic and deterministic DCA.

### Strengths
- The proposed framework is quite general and practically relevant. In many applications, it is indeed much easier to identify a block DC structure than a full DC structure, particularly when the variables are coupled. The presentation of the block DC formulation for neural networks with ReLU activation is also neat compared to existing approaches that rely on full-form DC decompositions for such composite functions. **P.S.** Please also do take a look at the paper *Cui, Y., He, Z., & Pang, J. S. (2020). Multicomposite nonconvex optimization for training deep neural networks. SIAM Journal on Optimization, 30(2), 1693-1723.* that gives explicit DC/MM structure for ReLU-activated neural networks of arbitrary depth.

- The observation that DC decompositions of monomials exhibit exponential complexity, in contrast to the polynomial complexity of BDC, is insightful and provides solid motivation for the paper.

- The paper is generally well-written.

### Weaknesses
- The novelty of the paper appears quite limited. The core algorithmic design was previously introduced by [Pham et al., 2022] for DC functions (albeit not for BDC), and this prior work is not cited. Given this, the main contribution of the present paper likely lies in the non-asymptotic convergence analysis (for a larger class of BDC), which provides a rate comparable to that of standard (stochastic) DCA. If this is indeed the key advancement, the authors should explicitly emphasize this analytical contribution and highlight the technical contribution to obtain this results.

- Another major concern is that the experiments do not show the advantages of the proposed method. The BDCA's performance is actually worse than simple SGD in all cases. Also, the classification accuracies of 90% for MNIST and 50% for CIFAR10 are very far from SOTA results. This makes me wonder if the method really works for complicated tasks.

**REFERENCE**

Pham, V. T., Luu, H. P. H., & Le Thi, H. A. (2022). A block coordinate DCA approach for large-scale kernel SVM. In International Conference on Computational Collective Intelligence (pp. 334-347). Cham: Springer International Publishing.

### Questions
- The definition of the gradient norm $\mathcal{G}$ seems invalid to me. The notion of $\partial f$ is ill-posed for DC functions, what kind of $\partial$ is it?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces multi-block DC (BDC) functions, a special class of nonconvex functions. Each function is split into blocks of variables, and within each block, it can be written as a convex function minus another convex function. This allows some functions, such as monomials or deep ReLU networks, to be represented more compactly than with standard DC decomposition.

The authors show how to decompose deep ReLU networks into BDC form, develop algorithms to minimize BDC functions (including a stochastic block method), and prove convergence guarantees. They also provide numerical experiments to illustrate the practicality of the decomposition for analyzing neural networks.

### Strengths
The paper is conceptually interesting and clearly demonstrates an advantage of the BDC formulation over classical DC decomposition, particularly in settings such as the training of deep ReLU networks. The authors provide convergence guarantees for their proposed algorithms, which adds theoretical rigor to the work. In addition, the paper validates its contributions with numerical experiments on relevant examples, showing that the proposed methods are practical.

### Weaknesses
The encoding size of the BDC decomposition for deep ReLU networks is not clearly discussed, and it seems that there could be an exponential blow-up with the number of layers. It would be helpful if the authors could clarify this point. Furthermore, the theoretical convergence guarantees assume $L_i$-smoothness of the functions, which does not hold for ReLU networks. A discussion of this limitation and its implications for practical use would strengthen the paper.

### Questions
- Can the authors provide an estimate of the encoding size or complexity of the BDC decomposition for deep ReLU networks, and discuss - In the paper https://arxiv.org/pdf/2411.03006, Proposition 4.2 presents a more efficient way to decompose a ReLU function into a difference of two convex functions (using maxout, but this should be adaptable to ReLU). Can this approach be used to obtain a smaller BDC decomposition of the training problem?
- How do the assumptions in the convergence proofs (e.g., $L_i$-smoothness) affect the applicability of the algorithms to ReLU networks?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Multi-Block DC (BDC) class—a broad new family of structured nonconvex functions.

The authors show that this block-wise structure is more general and powerful than the classical DC class.

They provide explicit, constructive BDC decompositions for modern machine learning models, including deep ReLU networks.

They also establish non-asymptotic convergence guarantees under batch, stochastic, and generalized smoothness settings.

The method is shown to be applicable to tasks such as sparse dictionary learning, multitask feature learning, and neural network training.

### Strengths
1. This paper shows that deep ReLU networks can be reformulated as a non-smooth DC function, as presented in Equation (3.2).

2. This paper considers a suite of algorithms—batch, stochastic, and proximal—with detailed non-asymptotic convergence analyses under various conditions.

### Weaknesses
1. The authors reformulate ReLU neural networks as equivalent DC optimization problems and solve them using proximal DC algorithms. However, the motivation behind this reformulation and the choice of the proximal DC approach is not clearly justified. It is also unclear what advantages these methods offer over simpler and widely used algorithms such as SGD.

2. The proximal DC algorithm and its theoretical analysis lack novelty. Many of the results do not appear particularly new. The algorithm discussed in Section 4 is a well-established method—essentially a form of coordinate descent—that has been extensively used for various nonconvex optimization problems (including those listed in Section 5). Hence, both the algorithm and the analysis feel incremental.

3. While the paper presents numerical experiments, the empirical section is rather limited compared to the theoretical development. The experiments mainly serve as proof-of-concept demonstrations rather than comprehensive validations against state-of-the-art methods.

4. The proposed algorithms introduce considerable complexity, as each block update involves solving a convex subproblem. The computational cost of these inner-loop optimizations—particularly for complex blocks such as neural network layers—is not thoroughly discussed or compared with simpler alternatives like SGD.

5. Sections 3 and 5 feel somewhat disconnected; the applications are presented in a fragmented manner rather than being integrated into a cohesive narrative.

6. It remains unclear why the ReLU network is assumed to satisfy the (L_i)-smoothness condition.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
2
