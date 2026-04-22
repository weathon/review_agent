# CONVERGENCE OF OPTIMIZERS IMPLIES EIGENVALUES FILTERING AT EQUILIBRIUM

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Ample empirical evidence in deep neural network training suggests that a variety of optimizers tend to find nearly global optima. In this article, we adopt the reversed perspective that convergence to an arbitrary point is assumed rather than proven, focusing on the consequences of this assumption. From this viewpoint, in line with recent advances on the edge-of-stability phenomenon, we argue that different optimizers effectively act as eigenvalue filters determined by their hyperparameters. Specifically, the standard gradient descent method inherently avoids the sharpest minima, whereas Sharpness-Aware Minimization (SAM) algorithms go even further by actively favoring wider basins. Inspired by these insights, we propose two novel algorithms that exhibit enhanced eigenvalue filtering, effectively promoting wider minima.Our theoretical analysis leverages a generalized Hadamard–Perron theorem and applies to general semialgebraic $C^2$ functions, without requiring additional non-degeneracy conditions or global Lipschitz bound assumptions. We support our conclusions with numerical experiments on feed-forward neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this work the authors develop a general framework for analyzing the properties of the loss landscape after convergence. Their methodology uses tools from analysis/dynamical systems theory to derive the geometric properties of any converged points. Their framework recapitulates known results for GD and some momentum variants, as well as some SAM variants.

The authors propose some SAM variants with different properties at convergence according to their methods, and provide some experimental evidence that these variants have stronger regularization properties.

### Strengths
The paper provides a very clean general setup for analyzing the loss landscape properties at convergence. In addition all the main theoretical results are cleanly presented and generally reference the previous works well.

### Weaknesses
The main weakness of the work is that it doesn't give any actionable lessons to the community working on sharpness regularization. The method called Hessian SAM in the paper is known in the literature (e.g. as Penalty SAM here [1]). The 2-step SAM has no benefits for a large computational cost. The general idea that SAM modifies the edge of stability (and therefore converged eigenvalues) is already known. While having a more precise theoretical characterization is nice, many of these results come quite naturally from Taylor expansion of losses around stationary points.

The experiments use minibatching, which can strongly affect the curvature dynamics, particularly as it can become unclear if networks actually reach convergence in this regime. In addition, some of the experimental settings use ReLU, which has to be treated very carefully with what the authors call Hessian SAM (Penalty SAM in the reference) [1].



[1] https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee3ce0121939f42098cdefd3ea025bf1-Abstract-Conference.html

### Questions
How does proposition 3.7 compare to the edge of stability bound in [1], e.g. Equation 16? there appears to be a potential discrepancy of a factor of $\lambda$ in the term linear in $\rho$.

Hessian USAM has been studied elsewhere, particularly in [2] where it is called "Penalty SAM". How do the results of [2] relate to the results in the paper?

[1] https://proceedings.mlr.press/v202/agarwala23a

[2] https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee3ce0121939f42098cdefd3ea025bf1-Abstract-Conference.html

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a unified theoretical perspective to analyze the convergence of various optimization algorithms. Building on this perspective, the authors introduce two new optimizers, USAM2 and HUSAM, and provide convergence results. Experimental validation is conducted on MNIST and CIFAR10 using standard architectures.

### Strengths
1. The idea of viewing the convergence of different optimizers through a unified lens is interesting and conceptually valuable.  

2. The theoretical development for the proposed USAM2 and Hessian-USAM algorithms is well-presented.

### Weaknesses
1. Aside from the authors’ own methods (USAM2 and HUSAM), the theoretical results for other optimizers seem largely covered by prior work, so the novelty is limited.  

2. Experimental evaluation is weak. MNIST is too small to provide convincing evidence, and the performance gains on MNIST of USAM2 and HUSAM, especially in accuracy, are very marginal.  

3. The WideResNet-16-8 on CIFAR10 setting is outdated. In this setting, USAM2 even underperforms USAM. Although HSAM shows a more noticeable improvement, it requires second-order Hessian information. It is questionable whether such a method is practical or valuable for large-scale neural networks.  

4. The empirical section lacks experiments on modern benchmarks and larger-scale models where the claimed contributions would matter.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper provides an interesting new perspective on the properties that the convergence point should satisfy under the assumption that the algorithm converges. The authors present a general statement based on the Stable Manifold Theorem, showing that the corresponding spectral radius is at most 1. They further apply their theory to several different algorithms, including gd, usam, and momentum variants, and demonstrate that different algorithms essentially select minima with different degrees of flatness.

### Strengths
This paper establishes an interesting and novel theoretical framework that studies a “dual” version of stability analysis, in which the algorithm is assumed to converge and the focus is on the properties of the initialization and the solution. The theory is built upon a generalized Hadamard–Perron stable manifold theorem. In contrast to stability analyses that require local diffeomorphism assumptions, this work relies on much milder conditions, such as the requirement of only semi-algebraic functions, and remains valid for large step sizes. The authors apply their theory to various practical algorithms, demonstrating different eigenvalue filtering effects and providing experimental results that validate their theoretical claims. Overall, the arguments presented in this paper are sound and well supported.

### Weaknesses
Although I enjoyed reading Section 2 of this paper, there are several shortcomings. The authors repeatedly claim that their theory is built upon weaker assumptions; however, their framework requires an additional assumption of algorithmic convergence, which is itself non-trivial and implicitly imposes constraints. For example, while Theorem 2.1 is stated to hold for large (even unbounded) step sizes, excessively large step sizes would clearly lead to divergence.

More importantly, to the best of my knowledge, the analyses for different algorithms in this paper do not appear to yield new results. For instance, Proposition 3.1 does not differ in essence from the well-known stability condition, and Proposition 3.4 does not go beyond the findings of Zhou et al. (2025), who analyzed the stability of USAM in detail. Although the theoretical perspective is interesting, it seems to offer limited new insights to the machine learning community.

### Questions
The class of semi-algebraic functions does not include common deep learning activation functions such as sigmoid and tanh. Although the authors claim that their results can be extended to all smooth losses, it remains unclear why this was not done, which may give the impression that the work is somewhat incomplete.

The authors mention the connection to EOS several times, but the discussion is too brief. Would it be possible to conduct a deeper analysis to provide new insights, or include some related experimental evidence?

Since the results presented in this paper are for the full-batch setting, why are stochastic variants still used in the experiments?

Minor: The definition on Semi-algebraic sets and functions should be moved to the beginning of Section 2 to aid understanding.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates the following optimization question: assuming an algorithm converges, what properties does its limiting point satisfy? The authors show that at any convergence point, the Jacobian of the update map has a spectral radius $\le 1$, which imposes eigenvalue filters on the Hessian that depend on the hyperparameters. They then apply this general result to several examples, including Gradient Descent, Heavy Ball, Nesterov’s method, and Unnormalized Sharpness-Aware Minimization (USAM). For USAM, the filter can admit negative eigenvalues, suggesting possible convergence to saddle points. Motivated by this observation, the authors propose Two-step USAM and Hessian USAM, whose filters exclude negative curvature, thus avoiding strict saddles. Small-scale experiments (MLPs on MNIST/Fashion-MNIST and WRN on CIFAR-10) show smaller top Hessian eigenvalues at convergence and similar performance.

### Strengths
-	Understanding the implicit bias or structure of the minima that optimization algorithms converge to is an interesting research problem.
-	The paper’s main contribution is to characterize the geometry of the minima to which the algorithm can converge (assuming convergence). The assumption is quite general and covers many relevant settings.
-	Overall, the paper is well written and easy to follow.

### Weaknesses
-	The proposed algorithms, Two-step USAM and Hessian USAM, appear to require more computation or backpropagation per iteration than SAM. Since SAM updates are already relatively expensive, the proposed methods may be slower for large-scale applications.
-	The experiments conducted on small-scale datasets are not sufficient to support the empirical claims, especially it is not clear to see the difference between the performance. Larger-scale experiments would better validate the results.

### Questions
-	Is there any analysis or discussion of the additional complexity of the proposed algorithms compared to SAM?
-	Can the results be extended to the stochastic setting or to adaptive methods such as Adam?
-	If additional structure of the problem is known (e.g., certain smoothness assumptions on the objective function), could the results be strengthened to hold for all parameters instead of almost surely?
-	Is there a robustness version of the results? For $x$ near the limiting point, is the Jacobian’s spectral radius bounded by $1+\delta$ for some small $\delta$?
-	It would be helpful to include more details about the experimental setup in the appendix. For example, how is the performance of different algorithms compared fairly. Are they run for the same number of epochs or the same number of backpropagations (since the proposed algorithm seems to require more backpropagations per iteration)?

### Soundness
3

### Presentation
3

### Contribution
2
