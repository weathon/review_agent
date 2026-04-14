# Nonasymptotic Analysis of Stochastic Gradient Descent with the Richardson–Romberg Extrapolation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We address the problem of solving strongly convex and smooth minimization problems using stochastic gradient descent (SGD) algorithm with a constant step size. Previous works  suggested to combine the Polyak-Ruppert averaging procedure with the Richardson-Romberg extrapolation to reduce the asymptotic bias of SGD at the expense of a mild increase of the variance. We significantly extend previous results by providing an  expansion of the mean-squared error of the resulting estimator with respect to the number of iterations $n$. We show that the root mean-squared error can be decomposed into the sum of two terms: a leading one of order $\mathcal{O}(n^{-1/2})$ with  explicit dependence on a minimax-optimal asymptotic covariance matrix, and a second-order term of order $\mathcal{O}(n^{-3/4})$, where the power $3/4$ is best known. We also extend this result to the higher-order moment bounds. Our analysis relies on the properties of the SGD iterates viewed as a time-homogeneous Markov chain. In particular, we establish that this chain is geometrically ergodic with respect to a suitably defined weighted Wasserstein semimetric.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper contains an analysis of one-pass SGD on stochastic objective functions:

$$\min\_{\theta \in \mathbb{R}^d} f(\theta), \quad \nabla f(\theta)=\mathrm{E}\_{\xi \sim \mathbb{P}_{\xi}}[\nabla F(\theta, \xi)],$$

This is run with constant step-size $\gamma \asymp 1/\sqrt{n}$ with $n$ the number of samples,  and it is shown that $p$-th moments satisfy a bound

$$\mathrm{E}^{1 / p}\left[\left\|\bar{\theta}\_n^{(R R)}-\theta^{\star}\right\|^p\right] \leq \frac{C p^{1 / 2} \sqrt{\operatorname{Tr} \Sigma_{\infty}}}{n^{1 / 2}}+\frac{C(f, d, p)}{n^{3 / 4}}+\mathcal{R}\left(\left\|\theta_0-\theta^{\star}\right\|, n, p\right),$$

Here 
* $\bar{\theta}\_n^{(R R)}$ is the ``Richardson-Romburg'' estimator for reducing lower order terms (specifically achieving the $n^{3/4}$ correction.
* $\mathcal{R}$ is an error term that controls the dependence on initial condition.

This improves the correction term in existing work from $n^{1/2+\delta}$ to $n^{3/4}$.

### Strengths
This improves an existing bound in various ways, both in the sense of
(1) achieving moment bounds and
(2) improving the rate of the correction term from $n^{1/2+\delta}$ to $n^{3/4}$, which is claimed as optimal.

It does it using the method of the `Richardson-Romberg' extrapolation.  

For more common Polyak-Ruppert averaging, it also achieves the same rate provided the minimization is strongly convex.

### Weaknesses
1) This paper performs a detailed mathematical analysis of a correction term to the behavior of SGD on fixed problems.  It is hard to imagine important practical conclusions of this -- we are talking about improving subleading correction terms in an _asymptotic_ setting when the number of samples overwhelms all the important problem-dependent constants in a stochastic optimization context.  And yes, I use the term _asymptotic_ intentionally: the $n$ in this problem has to overwhelm constants which are _quite clearly_ not effective, and only once that happens will the subleading terms will have improved dependence.   The authors have _not_ attempted to demonstrate that there are settings where this theorem is experimentally realizable, nor honestly, do I think it would be reasonable to expect them to do so (if they can show their 3/4-power in an experiment, I'd retract this and happily ``eat crow'').

2) The 3/4 power seems to be the essential point of this story.  It is asserted at multiple places that the 3/4 is optimal, but there is no clear construction.  There is a citation to another paper (Li et al), which considers a different algorithm.  If the example from that other paper works, it should be either displayed or precisely cited here (it was not possible to find it in a quick perusal).

### Questions
1) As stated: can the authors show that they can realize this n^{3/4} correction term in an experiment, (the setup should be such that I can reproduce it and, the 3/4 fit cleanly on a log-log plot)?  

2) The optimality of the 3/4 is asserted multiple places, but it is nowhere displayed.  What construction demonstrates the non-improvability of the 3/4?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors provide convergence analysis of both the Polyak-Ruppert averaged SGD (PR-SGD) and the Richardson-Romberg extrapolated SGD (RR-SGD) by a two-term expansion of the quantity $ \mathbb E^{1/2}[ || \hat \theta - \theta^* ||^2  ]$, a quantitative refinement of asymptotic normality. They show that RR-SGD achieves the same rate as PR-SGD which is known to be optimal.

### Strengths
- The paper is well-motivated and well-written. In particular, the last-iterate bound for RR-SGD, despite some issues raised below, is novel.

### Weaknesses
See Questions

### Questions
1. The extensive use of $\lesssim$ is rather confusing and potentially undermining in the paper. For example, in Corollaries 4 and 7, are there multiplicative constants suppressed on the leading $n^{-1/2} \sqrt{ \operatorname{tr} \Sigma }$ terms? 

* If there are: (1) these corollaries do not match in sharpness the existing results, and provide little insight into how RR-SGD compares to PR-SGD as the latter is favorable exactly because of its Cramer-Rao-type optimality in the leading term. 
(2) is it possible to spell out and compare the explicit leading terms?

* If there are not, I wonder why $ \lesssim$ instead of $\le$ is used.

Given how the notation $\lesssim$ is defined in Line 118 I'm afraid the former is the case, right?

Either way, the authors might want to say a few words on what terms/constants are being suppressed at least for presentation purposes.


----


2. Along the same line, we know that (considering decreasing step sizes for the moment --- you might be able to work back and forth between these two regimes) with PR-SGD the asymptotic normal distribution one can prove for
$\sqrt{t} (\widehat{\theta}_{t}^{\mathsf{avg}} - \theta^*)$
has a variance that matches the Cramer-Rao lower bound, 
which happens for any step sizes $ \gamma_t \propto t^{-w}$ where $w\in[1/2, 1]$.  Whereas for SGD without averaging, the asymptotic normality takes the form of 

$t^{w - 1/2} ( \widehat{\theta}_{t} - \theta^* ) $ where $w$ is the decay rate of step sizes defined above. Even in the $ w= 1 $ case, one gets a variance that's strictly larger than Cramer-Rao (hence than that of the iterate average). When $ w<1 $ this is a strictly slower rate of convergence. 

Long story short, PR averaging accelerates the optimization via a "removal of the step size effect". Is it the case that RR has a similar effect without averaging?

### Soundness
4

### Presentation
3

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
The paper considers the minimization of an objective function through unbiased estimates of its gradients. It brings tools from the theory of Markov chains to the study of stochastic gradient descent, showing that, under suitable assumptions, the Richardson-Romberg extrapolation achieves the optimal non-asymptotic risk bound (proved by Li et al. 2022 for the Root-SGD algorithm). Extensions to the p-th moment bound are also provided.

### Strengths
The paper interprets SGD updates with constant step sizes as an homogenous Markov chain, and uses a trick from numerical analysis to show that the difference of the average processes obtained for step sizes $\gamma$ and $2\gamma$ (defined in eq. (37)) will converge to a point that is closer to the optimum (as shown by Dieuleveut et al. 2018).

### Weaknesses
The paper could use some more polishing in its presentation:
* The geometric ergodicity of the sequence of updates is a crucial aspect, yet is not defined in the preliminaries. It would also be valuable to add an intuitive explanations for how this fact would be used to derive the risk bounds. 
* $D_{{\rm last}, 2}$ used eq. (19) in line 222 is not defined.
* In line 480, the beginning of the sentence should be "Directions ..."
* The authors should also make explicit the fact that the number of samples, $n$, needs to be known a priori to optimize the step size $\gamma$.

### Questions
Could the authors provide a geometric interpretation for the definition of $c(\theta, \theta')$, and the ensuing geometric ergodicity of the SGD iterates with respect to $W_c$? Also, are the authors aware of other contexts were such non-standard distance-like functions have been used, which could be cited in the paper?

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
2

### Summary
The paper investigates the convergence of stochastic gradient descent (SGD) for strongly convex and smooth problems. Concretely, using the mean squared distance as the performance metric, the author proves that the Polyak-Ruppert averaging procedure with the Richardson-Romberg extrapolation technique can achieve the optimal convergence rate for both the leading and higher-order terms.

### Strengths
1. The authors provide interesting results on SGD with Richardson-Romberg extrapolation.
2. I also appreciate the authors' honest discussion of their stronger assumption compared to the previous works, e.g., the higher-order smoothness. 
3. Moreover, the writing is easy to follow.

### Weaknesses
Since I am not very familiar with related works of the Richardson-Romberg extrapolation, the author could feel free to point out anything wrong in my following question.

1. Could the authors provide more discussion on C1? Especially, when can (20) hold?

2. The paper studies the convergence in expectation, is it possible to establish the high-probability convergence result with the same rate for both leading and higher-order terms?

3. One point I found is that when the stochastic problem degenerates into the deterministic problem, the current rate cannot recover the well-known linear convergence result. Could the authors make some statement on this point?

4. Another thing I am interested in is that the leading term $O(n^{-1/2})$ seems not to be infected by the stepsize (even if without the Richardson-Romberg extrapolation), which seems very different from the existing analysis of SGD for the same problem. Could the authors elaborate more on this fact?

### Questions
See **Weaknesses** part.

### Soundness
3

### Presentation
3

### Contribution
3
