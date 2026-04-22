# Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Bilevel optimization is a hierarchical framework where an upper-level optimization problem is constrained by a lower-level problem, commonly used in machine learning applications such as hyperparameter optimization. Existing bilevel optimization methods typically assume strong convexity or Polyak-Łojasiewicz (PL) conditions for the lower-level function to establish non-asymptotic convergence to a solution with a small hypergradient. However, these assumptions may not hold in practice, and recent work (Chen et al. 2024) has shown that bilevel optimization is inherently intractable for general convex lower-level functions with the goal of finding small hypergradients.

In this paper, we identify a tractable class of bilevel optimization problems that interpolates between lower-level strong convexity and general convexity via lower-level uniform convexity. For uniformly convex lower-level functions with exponent $p\geq 2$, we establish a novel implicit differentiation theorem characterizing the hyperobjective's smoothness property. Building on this, we design a new stochastic algorithm, termed UniBiO, with provable convergence guarantees, based on an oracle that provides stochastic gradient and Hessian-vector product information for the bilevel problems. Our algorithm achieves $\widetilde{O}(\epsilon^{-5p+6})$ oracle complexity bound for finding $\epsilon$-stationary points. Notably, our complexity bounds match the optimal rates in terms of the $\epsilon$ dependency for strongly convex lower-level functions ($p=2$), up to logarithmic factors. Our theoretical findings are validated through experiments on synthetic tasks and data hyper-cleaning, demonstrating the effectiveness of our proposed algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new tractable problem class for bilevel optimization: lower-level uniform convexity (LLUC), and proposes an algorithm that achieves the $\tilde{\mathcal{O}}(\epsilon^{-5p+6})$ oracle complexity bound for finding an $\epsilon$-stationary point when the lower-level function is uniformly convex with an exponent $p \ge 2$. The result in this paper recovers the known near-optimal rate when $p=2$.

The proposed algorithm consists of two loops: the inner loop applies Epoch-SGD (tailored to uniformly convex problems), and the outer loop applies normalized SGD with momentum to tackle nonconvex optimization under the generalized smooth assumption.

### Strengths
I believe the main advantage of this paper lies in its originality, as it presents new categories of problems and studies the solution algorithms under these settings. 

Moreover, the proposed algorithms can recover the (near)-optimal complexity for known settings ($p=2$).

The Holder continuity of the lower-level optimal solution mapping (Lemma 4.2) and the implicit function theorem under the LLUC assumption (Lemma 4.3) presented in this paper are of independent interest and may also be helpful for other problems.

### Weaknesses
Although I appreciate the effort to introduce new problem classes for bilevel optimization in this paper, the assumptions of the introduced class do not look very clean:

1. In the last item of  Assumption 3.2,  it is assumed that $\lambda_{\min} ( \frac{{\rm d} \nabla_y g(x,y)}{{\rm d} [y]^{\circ (p-1)}} ) \ge \mu >0$. This assumption seems to be derived from the proof; can the authors provide practical examples that satisfy these conditions? For instance, does the application in Section 3.2 meet this assumption?

 2. The second item of Assumption 3.4 looks strange. While the other items all assume bounded variance, the second part assumes that the noise follows a sub-Gaussian distribution. Could this be a typo?

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies a tractable class of bilevel optimization problems that interpolates between lower-level strong convexity and general convexity via lower-level uniform convexity (LLUC). For uniformly convex lower-level functions with exponent $p \ge 2$, the authors establish an implicit differentiation theorem characterizing the smoothness of the hyperobjective. Building on this, they propose a stochastic algorithm, termed UniBiO, with provable convergence guarantees, based on an oracle that provides stochastic gradient and Hessian-vector product information. The algorithm achieves an oracle complexity bound of $\tilde{O}(\epsilon^{-5p+6})$ for finding $\epsilon$-stationary points.

### Strengths
1. This paper introduces the concept of lower-level uniform convexity (LLUC) to bilevel optimization, which is new.

2. For uniformly convex lower-level functions with exponent $p \ge 2$, the authors establish an implicit differentiation theorem that characterizes the smoothness of the hyperobjective.

3. The authors design a stochastic algorithm, termed UniBiO, with provable convergence guarantees, and their algorithm achieves $\tilde{O}(\epsilon^{-5p+6})$ oracle complexity bound for finding $\epsilon$-stationary points.

### Weaknesses
1. My primary concerns relate to the assumptions and presentation of the paper:

1-1. In Assumption 3.2(v), the notation $d[y]^{o~p-1}$ is not clearly given. Although the authors refer to Theorem 3.1, this theorem does not appear in the text. The authors also refer to Theorem A.1, this theorem also does not appear in the text (may be Definition A.1). Such omissions are critical, as this definition is important in the paper.

Moreover, similar questions exist in this paper, e.g., in line 209, it should be Assumption 3.4, rather than Theorem 3.4.

1-2. Some assumptions appear strong. In Assumption 3.2(v), the generalized Jacobian is assumed to satisfy $\lambda_{\min}\left(\frac{d\nabla g(x,y)}{d[y]^{o\~p-1}}\right) > 0$. Does this imply that $\frac{d\nabla g(x,y)}{d[y]^{o\~p-1}}$ is invertible at $y^{o\~p-1}$? If so, Theorem 4.1 may simply review standard results for hypergradients [Ghadimi \& Wang, 2018], rather than offering new insights.

1-3. The definitions of $F$ and $G$ are not given in Assumption 3.4. Moreover, the paper focuses on stochastic methods for bilevel optimization, but there is insufficient discussion of stochastic settings.

1-4. This paper is not well-written, which will make the reader confused. For example, in Sections 4.1 and 5.3, the authors introduce the proof sketches of some theorems. However, these statements can be moved to the appendix. Conversely, some important elements, such as the stochastic approximate hypergradient $\hat{\nabla}f(x_t, y_t;\bar{\xi}_t)$ in Algorithm 2, are only introduced in the appendix and should be moved to the main text.

### Questions
1. What is the difference between $[y]^{o\~p-1}$ and $y^{o\~p-1}$?

2. In Assumption 3.3(iv), is the constant $\Delta_{\Phi}$ known? If not, would it be more appropriate to assume that $\Phi$ is lower-bounded?


3. In Lemma 4.3, would it be much better to analyze the error between the stochastic approximate hypergradient $\hat{\nabla}f(x_t, y_t;\bar{\xi}_t)$ and $\nabla\Phi(x_t)$, rather than $\widehat{\nabla}\Phi(x_t)$? Since the former is used in Algorithm 2 and $\widehat{\nabla}\Phi(x_t)$ is just an auxiliary quantity.

4. In Theorem 5.1, the definitions of $I$ and $Q$ are not provided, and the detailed form of $\delta$ is omitted. Moreover, standard convergence results for stochastic bilevel optimization [Ghadimi \& Wang, 2018; Ji et al., 2021; Chen et al., 2021] typically state $\frac{1}{T}\sum_{i=1}^T \mathbb{E}\\|\nabla\Phi(x_t)\\| \le \epsilon$ in expectation, not with high probability. Can the authors clarify why high probability and expectation are used simultaneously in their result?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of bilevel optimization when the lower-level problem is not strongly convex. The authors identify a tractable problem class characterized by uniform convexity in the lower-level function. They establish a new implicit differentiation theorem for this class and propose a stochastic algorithm, UniBio, with a convergence guarantee of $\widetilde{O}(\epsilon^{-5p+6})$ for finding an $\epsilon$-stationary point. Notably, this rate matches the optimal complexity for the strongly convex case ($p=2$). The theoretical findings are supported by experiments on synthetic and data hyper-cleaning tasks.

### Strengths
1. The paper identifies a novel and tractable class of bilevel problems with uniformly convex lower-level functions, providing a crucial pathway between strong convexity and general convexity.

2. The presentation is clear and well structured.

### Weaknesses
The oracle complexity $\widetilde{O}(\epsilon^{-5p+6})$ becomes prohibitively high for large $p$, creating a significant gap between theoretical tractability and practical efficiency for near-general convex problems.

### Questions
1. In Algorithm 2 (line 8), a momentum term is utilized in the iterative update. What is the precise role of this term in the convergence analysis? Would its removal deteriorate the final oracle complexity?

2. As noted in the weaknesses, the practical efficiency for larger values of the uniform convexity exponent `p` remains a concern. Could the authors include additional experiments, perhaps on a synthetic task, to explicitly investigate and demonstrate how the value of `p` influences the convergence speed and runtime of the proposed UniBio algorithm?

3. The analysis heavily relies on the uniform convexity of the lower-level problem. Is there a potential extension of these results under a generalized Polyak-Lojasiewicz (PL) condition for the lower-level function? Could a similar tractable class and convergence guarantee be established under such a condition?

4. A minor suggestion: please check some typos, such as “theoremthat” on page2; “$T_{k}$” in line 7 in Alg1 on page7.

### Soundness
3

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
This paper introduces the concept of lower-level uniform convexity (LLUC) as an intermediate class between lower-level strong convexity (LLSC) and general convexity in bilevel optimization. Under LLUC (parameterized by exponent $p \geq 2$), the authors (i) develop an implicit differentiation theorem characterizing the smoothness of the hyperobjective (showing that the hypergradient exists and is Hölder continuous with order depending on $p$), and (ii) propose **UniBiO**, a stochastic algorithm using normalized momentum for the upper level and a multistage Epoch-SGD with a shrinking-ball strategy for the lower level. They prove an oracle complexity of $\tilde{O}(\varepsilon^{-5p + 6})$ to find an $\varepsilon$-stationary point, which reduces to the known optimal complexity when $p = 2$. Experiments on synthetic data and a data hypercleaning task are provided. The work aims to formalize a tractable intermediate class between LLSC and general convexity and provide the first theoretical and algorithmic results for this class.

### Strengths
1. The paper addresses an important and timely problem by extending bilevel optimization analysis beyond the conventional lower-level strong convexity (LLSC) assumption.
2. The introduction of lower-level uniform convexity (LLUC) as an intermediate class is a novel theoretical contribution.
3. A new implicit differentiation theorem is derived for the LLUC setting.

### Weaknesses
1. The practical motivation for LLUC is not sufficiently justified; the provided examples (e.g., $\ell_p$-regression) appear contrived and do not reflect modern, complex bilevel learning tasks.
2. The theoretical framework relies on multiple technical and non-standard assumptions to establish convergence.
3. The proposed UniBiO algorithm appears structurally similar to existing methods (e.g., BO-REP), with limited algorithmic innovation.

### Questions
1. Can the authors provide a compelling, real-world machine learning example where the lower-level problem is uniformly convex with $p > 2$ but not strongly convex ($p = 2$)?
2. What is the core algorithmic innovation in UniBiO compared to existing bilevel optimization frameworks such as BO-REP?

### Soundness
2

### Presentation
2

### Contribution
2
