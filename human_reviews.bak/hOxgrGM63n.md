# Langevin Monte Carlo for strongly log-concave distributions: Randomized midpoint revisited

- Decision: Accept (poster)
- Scores: 6, 5, 8, 6

## Abstract
We revisit the problem of sampling from a target distribution that has a smooth strongly log-concave density everywhere in $\mathbb{R}^p$. In this context, if no additional density information is available, the randomized midpoint discretization for the kinetic Langevin diffusion is known to be the most scalable method in high dimensions with large condition numbers. Our main result is a nonasymptotic and easy to compute upper bound on the $W_2$-error of this method. To provide a more thorough explanation of our method for establishing the computable upper bound, we conduct an analysis of the midpoint discretization for the vanilla Langevin process. This analysis helps to clarify the underlying principles and provides valuable insights that we use to establish an improved upper bound for the kinetic Langevin process with the midpoint discretization. Furthermore, by applying these techniques we establish new guarantees for the kinetic Langevin process with Euler discretization, which have a better dependence on the condition number than existing upper bounds

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors considered Langevin Monte Carlo method in the problem of sampling from a target distribution that has a smooth strongly log-concave density. Specifically, the authors developed a novel proof technique that led to nonasymptotic $W_2$ error bounds for the randomized midpoint method for the Langevin Monte Carlo (RLMC) and the randomized midpoint method for the kinetic Langevin Monte Carlo (RKLMC). The upper bounds are competitive with the best available results for LMC and are free from a term that’s linearly dependent on the sample size. The authors also provided a nonasymptotic $W_2$ error bounds for the kinetic Langevin Monte Carlo (KLMC) algorithm that has an improved dependence on the condition number. Numerical experiments were conducted.

### Strengths
The paper is well-organized and clearly written. The authors did a good job discussing backgrounds and intuitions. The proposed error bounds improved upon existing results, and the novel proof technique itself has the potential to be used to re-examine other existing analyses. The paper includes sufficient comparison with and reference to related results/literature. The authors also addressed several limitations.

### Weaknesses
A major part of this paper’s contribution is removing the (square root of) $mnh$ for the error bounds of RLMC and RKLMC, yet I’m not clear on how important this is. The removal of this term, claimed by the authors, is an important step toward extending these results to potentials that are not strongly convex. Similarly, in the comments for the result for RKLMC, the authors claimed that not requiring the algorithm to be initialized at the minimizer of the potential is important for extending the method to non-convex potentials. However, as the authors pointed out in the discussion, strong convexity seems to be an essential assumption for these results. Therefore, I’m a bit unclear on the significance of this paper’s contribution.

### Questions
Please see the weakness part. In particular, how would the extension to non-convex or non-strongly convex potentials depend on the authors’ proposed methods?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper is an in-depth study on the randomized Langevin algorithms, particularly emphasizing the Randomized Midpoint technique in LMC. It is claimed that the bounds are superior to those previously established in the literature.

### Strengths
The paper consider widely-used randomized midpoint discretizations, offering notable enhancements over current standards, particularly in terms of condition number dependency and the stability of bounds. A key advantage of these bounds is their explicit numerical constants. The Randomized Midpoint is important  to achieve optimal convergence rates in first-order algorithms. The potential of extending this technique to other domains also presents an intriguing avenue for exploration.

### Weaknesses
While the paper presents some intriguing results, most of them primarily offers incremental advancements in the field. It applies the Randomized Midpoint technique to lessen discretization errors, which in turn marginally enhances the rate of convergence. However, the mathematical methods employed are not particularly interesting or surprising.

The empirical evidence looks weak to justify the criticality of the Randomized Midpoint in standard LMC.

### Questions
Can you provide any empirical evidence that the current algorithm improves in real world?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers the randomized midpoint discretization for the kinetic Langevin diffusion for sampling from a target distribution with smooth and strongly log-concave density. A non-asymptotic upper bound on the W_2 error of this discretization is obtained. A bound on Euler discretization for the kinetic Langevin process is also obtained.

### Strengths
The paper provides strong error bounds on randomized midpoint discretization for the kinetic Langevin diffusion for sampling from a target distribution with smooth and strongly log-concave density. The bounds substantially improve upon earlier results, have explicit constant and transparent reliance on the initialization, and don't require starting at the minimizer of the potential. The proof technique is novel and is based on summation by part.

### Weaknesses
It would be nice if some simulations in higher dimensions p could be included in Section 5.

### Questions
In the second paragraph after (7), "a close" might be "close"?

Would the qualitative behavior of numerical experiments change when the dimension p increases?

It seems that two notations d and p are used for dimension in Section 5.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper revisits the problem of sampling from strongly log-concave continuous distributions in high dimensions. A classical sampling algorithm for this task is the “Langevin Monte Carlo”  MCMC method which uses Langevin diffusion, a stochastic differential equation (SDE) that models the motion of a particle in a fluid. This paper provides an improved analysis of the randomized discretization scheme called “randomized midpoint method” for the aforementioned SDE. The results rely on the assumption that the magnitude of the eigenvalues of the potential function (i.e logarithm of the distribution’s density function) is both upper and lower bounded, while the ratio $\kappa$ between these upper and lower bounds appears in the bounds presented. Specifically, faster convergence of the randomized midpoint method for Langevin diffusion is shown for sufficiently small ratio $\kappa$. Similarly, the authors analyze the randomized midpoint method for the kinetic Langevin Monte Carlo method to obtain improved bounds, although with some slightly stronger conditions and with the advantage of not having to find a minimizer of the potential function for initialization.

### Strengths
This is a well written paper that provides improved results on existing algorithms for Langevin sampling, which can be potentially applicable in a wide range of problems.

### Weaknesses
Some of the results are potentially still not optimal as the authors also suggest. It would also be nice to see how the improved bounds can be applied to some more concrete problems even for theoretical results. 


Minor comments:
-Page 2, line 8: “strongly”->”strong”
-Page 2, line 22: “at”->”a”
-Page 2, “notation” paragraph, line 4: “semi-definite positive”->”positive semi-definite”
-Page 6, line 5: “designatex”->”designate”

### Questions
Do you an example application of the improved analysis do get better results for a specific problem?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair
