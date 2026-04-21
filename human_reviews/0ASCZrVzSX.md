# Blessing of Dimensionality for Approximating Sobolev Classes on Manifolds

- Avg Score: 5.33
- Decision: Reject
- Scores: 5, 5, 6

## Abstract
The manifold hypothesis says that natural high-dimensional data lie on or around a low-dimensional manifold. The recent success of statistical and learning-based methods in very high dimensions empirically supports this hypothesis, suggesting that typical worst-case analysis does not provide practical guarantees. A natural step for analysis is thus to assume the manifold hypothesis and derive bounds that are independent of any ambient dimensions that the data may be embedded in. Theoretical implications in this direction have recently been explored in terms of generalization of ReLU networks and convergence of Langevin methods. In this work, we consider optimal uniform approximations with functions of finite statistical complexity. While upper bounds on uniform approximation exist in the literature in terms of ReLU network approximation, we consider the opposite: lower bounds to quantify the fundamental difficulty of approximation on manifolds. In particular, we demonstrate that the statistical complexity required to approximate a class of bounded Sobolev functions on a compact manifold is bounded from below, and moreover that this bound is dependent only on the intrinsic properties of the manifold, such as curvature, volume, and injectivity radius.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper is focused on the manifold assumption in machine learning. The goal is to further shed light on how the intrinsic dimension of the data manifold enters into notions of complexity in function approximation. In particular, the authors prove lower bounds on the complexity of approximating Sobolev space functions on manifolds, and show that the lower bounds, which are essentially 1/n^(1/d), depend only on the intrinsic dimension d of the manifold, and not the ambient dimension of the space the manifold lies in. The authors use a notion of pseudodimension that is an extension of VC-dimension and measure complexity by the nonlinear n-width.

### Strengths
The question of why deep neural networks work well on extremely high dimensional data is an important problem and the manifold hypothesis may be a good way to explain this. Work on this problem is important and valuable in machine learning. The problem of lower bounds on complexity is not studied as often as upper bounds. The results appear to be new and non-trivial.

### Weaknesses
It's not clear to me how applicable these results are in practice. Even when data satisfies the manifold assumption, the intrinsic dimension d may be quite large. It is not clear how large the authors think d is in practice, and how large a d would make these results applicable vs vacuous. For MNIST, for example, it's often given that d is between 7 and 14, depending on the digit. One can assume d is much larger for more challenging problems, maybe 20-40? In this case, the error bound 1/n^(1/d) is vacuous, unless the number of data points n is astronomically large (e.g., if d=20 we need 10^(20) data points!).

### Questions
I don't understand why the main result, Theorem 3.1, is measuring nonlinear n-width between the Sobolev space W^{1,p} and the Lebesgue space L^q. It's really unclear to me why this implies anything about the difficulty in approximating Sobolev space functions. I'd like to see this clarified.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper concerns a lower bound for a nonlinear width (involving the pseudo-dimension, a generalized version of the VC dimension) of Sobolev classes over smooth manifolds of dimension $d$. The authors claim that while the manifold can be embedded in a higher dimensional space with dimesion $D \gg d$ the width of the Sobolev class has a lower bound that depends only on the dimension $d$ of the manifold.

### Strengths
For a technical paper, the presentation is approachable and is self-contained (modulo some typos and missing definitions). The paper extends the lower bound proved in (Maiorov and Ratsaby, 1999) to manifolds.

### Weaknesses
- Not only is the result of Theorem 1 independent of ambient dimension $D$, the ambient dimension does not appear _anywhere_ in the estimates. This is somewhat odd because the abstract mentions the manifold hypothesis which concerns both $D$ and $d$. In some similar approximation bounds, there is typically some dependence on $D$. The authors should address this.

- In a similar vein, the authors do not present a connection between the sample complexity mentioned in Proposition 2.3 and the main Theorem 1, as far as I can see. The assumed connection is that, due to this property of classes with finite pseudo-dimension $\mathcal{H}_n$, the Sobolev class can also be estimated with the sample complexity given in (2), once the approximating class $\mathcal{H}_n$ is determined, it can be estimated with this sample complexity. This connection should be made somewhere.

- The main structure of the proof is almost identical to (Maiorov and Ratsaby, 1999), except the construction of the $L^1$-separated set of functions, due to the domain being a manifold. There are a few questions about the extended lower bound, which I ask below in the "questions" section.

### Questions
- The Theorem 1 lower bound (12) has a dependence on $p$, unlike in the Euclidean case (Maiorov and Ratsaby, 1999). Is there a plausible reason for this, i.e. is $p$ there for an inherent reason? 

- Why is Theorem 1 restricted to the case $K < 0$? What is different about the positive curvature case?

- line 535: it is mentioned that the cutoff functions with bounds on higher derivatives is difficult to construct, but I am having trouble seeing why this should be so. Can the authors explain further?

- line 758: The authors say "By maximality, balls of radius $2\epsilon$ at the $p_i$ cover $M$" but why is this true? The manifold can potentially be very narrow.

- line 691: What is the notation $\mathcal{A}(z, 16 / \sqrt{\text{length}(z)})$?

- Should the empirical risk in (56) be scaled by the sample size?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the complexity of Sobolev function class on Riemannian manifolds. Specifically, the paper derives lower bound of the approximation error of a Sobolev ball by a smaller class with complexity bounded as pseudo-dimension. By constructing explicitly functions of bounded Sobolev norm that are separated in $L^1$, the paper connects the packing number of the manifold with a hard-to-learn subclass in the Sobolev ball, thus forcing a larger error/width. The main theorem claims a lower bound that only depends on intrinsic quantities of the manifold.

### Strengths
The paper has a natural motivation, and the concluded rate seems matching that of classical Euclidean case. The presentation is lucid, and the proof sketch and extended discussion is well written. Overall this paper is a solid contribution on the topic of manifold learning.

### Weaknesses
1. The result is not too surprising on the 1,p-Sobolev class, and considering that higher Sobolev space can even be an RKHS [1], one would expect major improvement on the rate. Also using volume comparison to control the packing number is rather standard, and one might further ask if the same technique is applicable to metric measure spaces or RCD spaces, though I understand this technicality may not be particularly befitting of this venue.


2. Typos: 
Definition 2.7 is defining packing number not covering number, and also metric entropy is not commonly defined this way. This version of metric entropy is exactly the same as packing number, hence (7) is not needed. (The key proposition C.1 is correct.) $CPf_a$ out side the balls on line 375 is not necessarily 0.

[1] De Vito, Ernesto, Nicole Mücke, and Lorenzo Rosasco. "Reproducing kernel Hilbert spaces on manifolds: Sobolev and diffusion spaces." Analysis and Applications 19.03 (2021): 363-396.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
