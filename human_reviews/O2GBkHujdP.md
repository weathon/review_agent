# Independently-Normalized SGD for Generalized-Smooth Nonconvex Optimization

- Avg Score: 4.25
- Decision: Reject
- Scores: 3, 3, 5, 6

## Abstract
Recent studies have shown that many nonconvex machine learning problems meet a so-called generalized-smooth condition that extends beyond traditional smooth nonconvex optimization. However, the existing algorithms designed for generalized-smooth nonconvex optimization encounter significant limitations in both their design and convergence analysis.
In this work, we first study deterministic generalized-smooth nonconvex optimization and analyze the convergence of normalized gradient descent under the generalized Polyak-Lojasiewicz condition. Our results provide a comprehensive understanding of the interplay between gradient normalization and function geometry. Then, for stochastic generalized-smooth nonconvex optimization, we propose an independently-normalized stochastic gradient descent algorithm, which leverages independent sampling, gradient normalization, and clipping to achieve an $\mathcal{O}(\epsilon^{-4})$ sample complexity under relaxed assumptions. Experiments demonstrate the fast convergence of our algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this work the authors study the convergence of Normalized Gradient Descent and Normalized Stochastic Gradient Descent under generalized smoothness assumption and PL-condition.

### Strengths
This is a theoretical work on optimization. Assumptions are presented, theorem statements are written. The Appendix contains the proofs of the theorems.

### Weaknesses
I think that the contribution of this paper is not good enough for the standards of the ICLR conference.

NGD was already proposed and analyzed in the nonconvex setting in [1]. Extending their analysis to the PL-case does not seem as a strong contribution to me.

The second question in the paper is: "Can we develop a novel algorithm tailored for stochastic generalized-smooth optimization that
achieves fast convergence in practice while providing convergence guarantee under relaxed assumptions?"

There are papers with the analysis of NSGD with momentum [2], Adam [3,4], AdaGrad-Norm/AdaGrad [5,6]. All are tailored for the stochastic optimization in the generalized smooth setting and seem more advanced. All have convergence guarantees. Is the newly proposed I-NSGD faster in practice than any of them? As far as I can see, I-NSGD is compared to SGD, Clipped-SGD, NSGD in the experimental section only. The original paper [7], which introduced the generalized smoothness assumption, explains that this assumption is based on empirical findings from language model training and image classification tasks. Does I-NSGD perform well on them?

There is a recent paper [8], which contains a similar to NSGD method with smoothed clipping (Algorithm 1: (L0,L1)-GD). You mention this work in your paper. I can see that the authors develop the adaptive versions of the algorithm, the accelerated version. How does your theory compares to theirs? As far as I can tell from their contribution section, they improve upon many previous results.

[1] Ziyi Chen , Yi Zhou , Yingbin Liang , and Zhaosong Lu, Generalized-Smooth Nonconvex Optimization is As Efficient As Smooth Nonconvex Optimization.

[2] Hübler, F., Yang, J., Li, X., and He, N., Parameter-Agnostic Optimization under Relaxed Smoothness.

[3] Wang, B., Zhang, Y., Zhang, H., Meng, Q., Ma, Z.-M., Liu, T.-Y., and Chen, W., Provable adaptivity in adam.

[4] Li, H., Rakhlin, A., and Jadbabaie, A., Convergence of adam under relaxed assumptions.

[5] Faw, M., Rout, L., Caramanis, C., and Shakkottai, S., Beyond uniform smoothness: A stopped analysis of adaptive sgd.

[6] Wang, B., Zhang, H., Ma, Z., and Chen, W., Convergence of adagrad for non-convex objectives: Simple proofs and relaxed assumptions.

[7] Jingzhao Zhang, Tianxing He, Suvrit Sra, Ali Jadbabaie, Why gradient clipping accelerates training: A theoretical justification for adaptivity

[8] Eduard Gorbunov, Nazarii Tupitsa, Sayantan Choudhury, Alen Aliev, Peter Richtárik, Samuel Horváth, Martin Takáč, Methods for Convex (L0,L1)-Smooth Optimization: Clipping, Acceleration, and Adaptivity

### Questions
There is a standard sanity check procedure in optimization. The field is connected to physics (take the heavy-ball method of Polyak, for instance). And different quantities in physics have different units of measure. Take, e.g., the mechanical work: $W=|F|\cdot |s| \cdot cos\theta.$ We get that 1 Joule is 1 Newton multiplied by 1 meter.

I will try to clarify my next concern with the example from physics.

Let us consider the one-dimensional problem of minimization of the coordinate $x(t)$ of the material point, where $t$ is a time variable. Suppose $x(t)$ is some $(L_0,L_1)$-smooth twice-differentiable function. In the work [7] (see above) the $(L_0,L_1)$-smoothness reduces to the condition $|\nabla^2 x(t)|\le L_0 + L_1|\nabla x(t)|$ in this case. Let us observe that $|\nabla^2 x(t)|$ is the absolute value of the acceleration and is measured in m/s^2. Therefore, $L_0$ has the same unit of measure, m/s^2. In turn, $|\nabla x(t)|$ is the absolute value of the speed and is measured in $m/s.$ Therefore, $L_1$ has the unit of measure $s^{-1}.$

In Theorem 1, when choosing $\gamma,$ the constants $L_0$ and $L_1$ are added to each other. How two quantities with different units of measure can be added to each other? This happens over the whole paper. In Theorem 2 when choosing $\gamma$ in the minimum you compare $\frac{1}{4L_0}$ to $\frac{1}{4L_1\sqrt{T}};$ when you define $\Lambda$ you add them again.

In the original work [7] the stepsize is chosen as either $\frac{1}{L_0+L_1\|\nabla f(x_t)\|}$ or $\frac{1}{L_0+L_1M},$ where $M$ is an upper bound on the gradient, which has the same unit of measure. In the work [8] as well.

The constants $L_0$ and $L_1$ are not added to each other as in your paper. They do not have this problem with the units of measure. Therefore, there might be technical flaws in your proofs.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper analyzes stochastic normalized SGD with an extra sampling under generalized smoothness assumption and obtains known complexity rate. For the deterministic case paper additionally analyzes normalized SGD under generalized Polyak-Łojasiewicz condition.

### Strengths
- the convergence of normalized gradient descent under the generalized Polyak-Łojasiewicz (PŁ) condition
- generalized variance assumption

### Weaknesses
- only symmetric case, which is much easier to analyze; [Hubler etal, 2024] (mentioned by authors) obtains the same rate with non-symmetric, more general smoothness setup
- proposed I-NSGD requires additional sampling, while does not improve the known complexity rate. [Hubler etal, 2024] does not need additional sampling and has the same rate.
- having the above, [Hubler etal, 2024] is not experimentally compared

### Questions
- I would like to hear authors comment on the result with Polyak-Łojasiewicz condition. Despite I am not aware of such a result and it is clearly a contribution then, its value seems limited. I would like authors to explain in details why this result is important. I would also like an explanation of why NSGD is considered under this setup and not SGD for example. Does NSGD remarkably benefit from Polyak-Łojasiewicz setup?
- But my main concern is contribution of proposed I-NSGD. The most important factor is the focus of the paper on only symmetric generalized smoothness. Non-symmetric smoothness holds for a problem formulation appearing in Distributionally Robust Optimization (Jin et al., 2021). Chen et al. (2023) also show that exponential function satisfies non-symmetric case. Moreover, the rate is the same despite the less general assumption.
So, I want authors comment on why they think the case of less general smoothness overweights more general variance assumption. Can authors argue that the more general variance assumption is not compatible with a kind of [Hubler etal, 2024] analysis? Why haven't authors included [Hubler etal, 2024]'s NSGD with momentum to experimental comparison? It is highly related method to the best of my understanding.
By now, neither experiments nor theory tells me that I-NSGD is preferable in any scenario.

References:
Jikai Jin, Bohang Zhang, Haiyang Wang, and Liwei Wang. Non-convex distributionally robust optimization: Non-asymptotic analysis. Advances in Neural Information Processing Systems, 34: 2771–2782, 2021
Ziyi Chen, Yi Zhou, Yingbin Liang, and Zhaosong Lu. Generalized-smooth nonconvex optimization is as efficient as smooth nonconvex optimization. In International Conference on Machine Learning, pp. 5396–5427. PMLR, 2023.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper examines non-convex optimization under the well-known generalized smoothness condition and an extended (PL) condition. Under these relaxed assumptions, they establish the convergence rate of the Normalized Gradient Descent (NGD) method under both deterministic and stochastic regimes. In particular, for the stochastic setting, they address the limitations of NGD by using independent samples to compute the direction and normalization factor of the stochastic gradient. This method, referred to as Independently-Normalized SGD (I-NSGD), provably converges in a rate of $O(\epsilon^{-4})$.

### Strengths
Overall, this is a solid and well-written paper. The proofs have a good presentation and appear to be sound based on my reading. The rates are novel under the generalized PL condition. In particular, the idea of using different samples for the direction and scale of the normalized stochastic gradient estimator seems to be an interesting trick and might find its application in similar situations.

### Weaknesses
Nevertheless, the major concern comes from the novelty and the contribution perspective. This paper is clearly an extension of [Chen et al., 2023] and establishes new rates under the generalized PL condition. While some rates are indeed novel, no results really outperform the known results in comparable settings. Moreover, the PL condition seems to greatly restrict the universality of these rates: although PL does not imply convexity, it allows the objective to have favorable geometric properties that lead to fast rates, sometimes even comparable to the convex or strongly-convex setting. This makes the analysis under PL to be less general and less interesting.

A more serious concern is about the stochastic regime. The authors argue the analysis of I-NSGD outperforms [Zhang et al., 2021] by using an affine noise bound, as in Assumption 4. Nevertheless, similar assumptions have already been explored in other prior works like [1], [2], and [3], where convergence rates of $O(\epsilon^{-4})$ are established for adaptive gradient methods when the same general smooth assumption is satisfied. This reduces the contribution of this work, as the above results already achieve the optimal rate under similar conditions. A more thorough discussion of the connection to the literature would help the readers have a more complete understanding of the whole picture.

As a result, it seems that the current paper does not bring a lot of novel insights to the optimization community. I tend to reject this paper unless they are able to provide a more convincing argument to defend the broader impact of this work.

References

[1] Bohan Wang, Huishuai Zhang, Zhiming Ma, and Wei Chen. Convergence of AdaGrad for non-convex objectives: simple proofs and relaxed assumptions. COLT 2023.

[2] Matthew Faw, Litu Rout, Constantine Caramanis, and Sanjay Shakkottai. Beyond uniform smoothness: a stopped
analysis of adaptive SGD. COLT 2023.

[3] Yusu Hong, Junhong Lin. Revisiting Convergence of AdaGrad with Relaxed AssumptionsRevisiting Convergence of AdaGrad with Relaxed Assumptions.

### Questions
There is no further questions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the problem of nonconvex optimization under the $(L_0, L_1)$ smoothness condition (also known as generalized smoothness) and specifically considers two problems: (a) adapting to the generalized Polyak-Łojasiewicz (PŁ) condition, a generalization of strong convexity, when it exists, and (b) solving the stochastic problem, with or without the PŁ condition. For the former problem, the authors show that the normalized gradient descent method can achieve different (faster) rates depending on the PŁ condition, recovering linear convergence in the classical case ($L_1 = 0$ in generalized smoothness and $\rho = 2$ in the generalized PŁ condition). For the stochastic problem, the authors consider a variant of normalized gradient descent where the normalization is done with a different, independently sampled point. Theorem 2 shows that this method can achieve the $1/\epsilon^4$ convergence rate (to achieve error $\epsilon$) without the need for large ($\epsilon$-dependent) batch sizes or assuming that the stochastic gradient noise norm is bounded almost surely. Finally, experiments show the proposed algorithm performs well compared to the baselines (SGD, Normalized SGD, SGD with clipping).

### Strengths
- The paper's idea is pretty nice, I'm quite surprised independent normalization has not been considered in the literature before-- it's such a simple and straightforward solution to the problem of correlated samples in normalization. That said, it might be the case that it works worse in practice than normalized SGD (see the weaknesses section), but we know that normalized SGD may be nonconvergent in general. Independent samples have been used before in the estimation of the smoothness constant in ordinary (smooth) optimization [1], but not in generalized smoothness optimization as far as I can see.
- The analysis removes the requirement for large batch sizes or almost surely bounded gradient noise present in prior work, although it still requires very specific choices of (constant) batch sizes.
- The paper is written quite clearly, and is easy to read. 

[1] Malitsky and Mischenko, Adaptive Gradient Descent without Descent, arXiv 1910.09529 (2020).

### Weaknesses
- (Unrealistic requirement on the batch size) The batch sizes used, while constant, depend directly on the constants $\tau_1$ and $\tau_2$ in Assumption 4. These constants are in general not knowable, how are we to choose a batch size that depends on them when we can't know them? Can Theorem 2 be generalized to hold for any arbitrary batch size instead? For example, it seems that the rate explodes if $\tau_1 = 0$, which should be an easier case.
- (Unfair experimental evaluation) It seems that for I-NSGD, you tune the $\beta$ parameter while for NSGD you do not? For a fair comparison, it should be tuned as well. The same applies to gradient clipping: the text states you did tune the learning rate for the algorithm, but not the clipping constant. That should be tuned.
- (Limited technical novelty) The technical novelty in this paper is really quite limited, save for using independent normalization all the tools used are quite direct and similar to prior work.

Overall, despite the drawbacks, I lean towards recommending acceptance for the paper

### Questions
- Please address my concerns in the weaknesses sections.
- Why is the generalized PŁ condition interesting to study? I know you do give an example in lines 172-175 but how relevant is this in practice?

### Soundness
3

### Presentation
3

### Contribution
3
