# Complexity Lower Bounds of Adaptive Gradient Algorithms for Non-convex Stochastic Optimization under Relaxed Smoothness

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 5

## Abstract
Recent results in non-convex stochastic optimization demonstrate the convergence of popular adaptive algorithms (e.g., AdaGrad) under the $(L_0, L_1)$-smoothness condition, but the rate of convergence is a higher-order polynomial in terms of problem parameters like the smoothness constants. The complexity guaranteed by such algorithms to find an $\epsilon$-stationary point may be significantly larger than the optimal complexity of $\Theta \left( \Delta L \sigma^2 \epsilon^{-4} \right)$ achieved by SGD in the $L$-smooth setting, where $\Delta$ is the initial optimality gap, $\sigma^2$ is the variance of stochastic gradient. However, it is currently not known whether these higher-order dependencies can be tightened. To answer this question, we investigate complexity lower bounds for several adaptive optimization algorithms in the $(L_0, L_1)$-smooth setting, with a focus on the dependence in terms of problem parameters $\Delta, L_0, L_1$. We provide complexity bounds for three variations of AdaGrad, which show at least a quadratic dependence on problem parameters $\Delta, L_0, L_1$. Notably, we show that the decorrelated variant of AdaGrad-Norm requires at least $\Omega \left( \Delta^2 L_1^2 \sigma^2 \epsilon^{-4} \right)$ stochastic gradient queries to find an $\epsilon$-stationary point. We also provide a lower bound for SGD with a broad class of adaptive stepsizes. Our results show that, for certain adaptive algorithms, the $(L_0, L_1)$-smooth setting is fundamentally more difficult than the standard smooth setting, in terms of the initial optimality gap and the smoothness constants.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the complexity of adaptive optimization algorithms in non-convex stochastic settings under $(L_0,L_1)$-smoothness assumption. The study addresses an open question regarding whether the complexity of these adaptive algorithms, which typically have higher-order dependencies on problem parameters, can be tightened.

The authors establish complexity lower bounds for different variants of AdaGrad, focusing on the dependency on parameters $Δ$, $L_0$, and $L_1$. They provide:
- A complexity lower bound for the Decorrelated AdaGrad-Norm variant.
- A lower bound for Decorrelated AdaGrad under specific hyperparameter conditions.
- A lower bound for SGD with a broad class of adaptive stepsizes.

The findings of the paper show that, for certain adaptive algorithms, the $(L_0, L_1)$-smooth setting is fundamentally more difficult than the standard smooth setting.

### Strengths
- The paper is clearly structured, with each theorem building on the previous results to form a coherent narrative.

- The theoretical contributions are solid, with clear mathematical rigor and well-constructed complexity bounds. By investigating multiple AdaGrad variations, the paper thoroughly explores different adaptive step-size strategies under non-standard conditions.

- The results significantly contribute to the understanding of optimization in non-convex and non-smooth settings, highlighting limitations of popular adaptive methods when applied to challenging optimization landscapes.

### Weaknesses
N/A

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Recent advances in theory and optimization of transformers and particularly adaptive algorithms have shed light on a new type of assumption called $(L_0, L_1)$-smoothness. This paper proposes novel non-convex lower bound instances satisfying the $(L_0, L_1)$-smoothness condition for various types of adaptive algorithms including (decorrelated) AdaGrad and AdaGradNorm. The results show an iteration complexity lower bound of $\Omega(\Delta^2 L_1^2 \epsilon^{-4})$ (naive, omitting $\sigma, \gamma$) to achieve stationarity in expectation, which shows a gap with previous results on $L$-smooth functions. The paper also provides a new lower bound for a class of single-step adaptive SGD algorithms in terms of an iteration complexity lower bound of $\Omega(\Delta L_0 \epsilon^{-4})$ (naive) to achieve stationarity with high probability.

### Strengths
- The paper suggests novel lower bounds through diligent combinations of previous lower bound techniques.
- The overall writing is clear and easy to follow (except for only a few parts and bits of typos), and the proofs in the appendices are also organized well.
- The paper relevantly compares results with previous literature, provides rich discussion points, and is transparent about limitations.

### Weaknesses
- The lower bounds in Theorem 1 require high dimensions like $d = \Omega(\Delta^2 L_1^2 \sigma^2 \epsilon^{-4})$ which is quite large and implies that, for a fixed $d$, the proposed lower bound can only cover up to a finite range of $T$.
- As pointed out in the paper, the results (especially Theorems 1-3) are for specific algorithms rather than a class of algorithms. Although it seems that it will be quite difficult, it could be interesting to see if the lower bound results can extend to a *class* of Adam-type algorithms (Chen et al., 2019).
- This is a minor thing, but I think Section 6 and Theorem 4 could be organized better. It is hard to get what the new constants exactly are without reading Appendix C.1, especially the constants involving $\zeta$. IMHO, it would have been a bit easier to understand if the details about the constants (including the random walk parts) were all in the former part of Section 6. (I reckon that it was written this way just for now because of the page limit.)

### Questions
- In Theorem 1, the proof uses a 1-dimensional instance for the large $\eta$ case and a large dimensional instance for the small $\eta$ case. The $T = \Theta(\epsilon^{-4})$ condition for the latter case results in a dimension of $d = \Omega(\epsilon^{-4})$ as stated in W1. On the other hand, in Theorems 2 and 3, the proof is quite the opposite: it uses a $d \ge T$ dimensional instance for the large $\eta$ case which does NOT have requirements like $T = \Theta(\epsilon^{-4})$, and a 1-dimensional instance for the small $\eta$ case. Why can’t we use a similar approach to DAN to avoid high dimension requirements, or what are the technical difficulties in doing so?
- In Theorem 3, the lower bound holds when $\gamma \le \sigma$, but the rates themselves do not contain $\gamma, \sigma$ anywhere, which is probably because they cancel out somewhere if we follow the same procedure as the proof of Theorem 2. Then what can we say about the case of $\gamma > \sigma$ for now? By definition of AdaGrad (without decorrelation), I reckon that we must choose a not-too-small $\gamma$ so that the first step of the algorithm doesn’t ‘explode’, and I’m just curious about what behavior we could see or expect from this lower bound instance if we send the noise to zero for a fixed $\gamma > 0$. (The ‘better noise-dependency’ argument of AdaGrad over DAG is also true only when $\gamma \le \sigma$.)
- Why exactly does Theorem 4 derive iteration complexity lower bounds in terms of high probability (instead of expectation)? Are they just completely different results, or is this sort of a stronger result than getting lower bounds in terms of expectation (from the same instance)? (I think the two types of bounds are fundamentally different in general, but I’m asking this because the approach towards getting the lower bound instance with $\| \nabla f(\boldsymbol{x}_t) \| \ge \epsilon$ seems to share quite a similar spirit with those of Theorems 1-3.)
- Light Question: Is it possible to obtain other lower bound results analogous to Theorem 4 if we focus on the (stronger) bounded-noise assumption?
- Light Question: Can you compare Theorem 4 with the upper bounds in Section 7 of Gorbunov et al. (2023) which also has a few results assuming the finite-sum oracle setting? For SGD-PS (Algorithm 6, Gorbunov et al. (2023)), is this algorithm incomparable with the lower bound because it uses the values $f_i(\boldsymbol{x}^{\star})$ in the step sizes?
- **Small Typos.** In p.17, I think we should remove the subscript $t$, as in $f(\boldsymbol{x}) = \epsilon \langle \boldsymbol{x}, \textbf{e}_1 \rangle + \sum\_{i=2}^T h_i ( \langle \boldsymbol{x}, \textbf{e}_i \rangle ) $.
    
    In p.18, $j(\boldsymbol{x}) = i+2$ $\Rightarrow$ $j(\boldsymbol{x}_i) = i+2$.
    
    In p.21, $D_1 \le \dots \le \Delta/6 + \Delta/6 + \Delta/6 = \Delta/2$, and some constants after this part should be slightly modified according to this.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the complexity **lower bounds** for certain adaptive gradient (descent) algorithms for non-convex optimization problems under relaxed smoothness assumptions, i.e., the objective function $f$ is continuously differentiable and $(L_0, L_1)$-smooth as stated in assumption 1. More specifically, the results are in Table 1 (line 77-84). From 4 lower bounds derived for AdagGad variants, the paper concluded that the Adagrad-type algorithms cannot converge under relaxed smoothness **without** a higher-order polynomial complexity that depends on the problem parameters $\nabla, L_0, \sigma, L_1, \epsilon$.

### Strengths
- The presentation of this paper is well-structured. For example, it effectively presents the related complexity upper bounds in Table 1 and provides clear proof intuitions along with key takeaways for each theorem.
- The results of this paper indicate that for decorrelated AdaGrad-Norm, the algorithm cannot achieve the optimal complexity attainable under a stronger smoothness assumption. This finding suggests a meaningful performance gap between $L$-smooth and $(L_0, L_1)$ smoothness conditions.

### Weaknesses
- As noted in the limitations, decorrelated AdaGrad is not commonly used in practice; therefore, proving a lower bound for decorrelated AdaGrad may lack significant technical novelty.

### Questions
- Assumption 2 appears to be rather strong. Are there existing references that support the validity of these assumptions?

### Soundness
2

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
2

### Summary
This paper investigates the theoretical lower bounds for finding $\epsilon$-stationary points using adaptive gradient algorithms under $(L_0,L_1)$-smoothness. With the assumption of bounded noise, the authors prove that the complexities of AdaGrad and its variants have a greater dependency on the problem parameters $\Delta$, $L_0$, $L_1$ compared to their $L$-smooth counterparts.

### Strengths
In my opinion, these results reveal the potential difficulty of adaptive methods under relaxed smoothness and add to the current understanding within the community.

### Weaknesses
The primary weakness of this paper is that the analysis is heavily relies on update policies similar to AdaGrad. Although the authors do investigate the behavior of adaptive SGD, they assume the noise has a less common affine property, and their results do not extend to popular variants of Adam.

### Questions
**Typos and minor suggestions:**

Line 81: the theorem for AdaGrad should be "Theorem 3".

Line 287: Table 1 provides the result of AdaGrad-Norm under affine noise, which is not equivalent to Equation 5.

Line 746: the logic of Equation 7 is not easy to comprehend at first glance, as it appears to tighten the condition for $\ell_{t}\geq 4 m_{t+1}$. I suggest that the authors should explicitly point out this relationship.

Line 774: "$\log x\leq 1+x$" -> "$\log(1+x)\leq x$".

Line 797: missing explanation for step (i).

Line 908: "$\mathbf{x}_t$" -> "$\mathbf{x}$".

Line 927-929: "$L$" -> "$L_0$".

Line 958-959: missing \\rangle for $\langle\mathbf{x}_i,\mathbf{e}_j\rangle$.

Line 1012: "$t\geq 1$".

Line 1134: "Lemma 1" -> "Lemma 2".



**Questions:**

* We note that Assumption 1 is generally less rigid than the typical assumption proposed by Zhang et al. (2019). Specifically, Assumption 1 aligns with the $\mathcal{L}^*_{\mathrm{asym}}$ function class introduced by Chen et al. (2023). Nonetheless, the lower bounds established in this paper should still hold under the typical assumption.
* Since the constructive proof of Lemma 2 is rather complicated, I encourage the authors to provide necessary remarks following the presentation of the results. Specifically, we are interested in how the higher dependency on $\Delta$ and $L_1$ emerges in the dominant term $\mathcal{O}(\Delta^2L_1^2\sigma^2\epsilon^{-4})$. The same applies to Lemma 4.
* In Line 484, the authors state that "the lower bound goes to $0$ when $\sigma_2\rightarrow 1$". However, the dominant term $\mathcal{O}(\Delta L_0\sigma^2\epsilon^{-4})$ does not approach zero, as $\sigma_1>0$.
* I encourage the authors to discuss the results of Li et al. (2023). They provide analyses for Adam under relaxed assumptions, which covers the case of $(L_0,L_1)$-smoothness.



References:

Jingzhao Zhang, Tianxing He, Suvrit Sra, Ali Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity. In International Conference on Learning Representations, 2019.

Ziyi Chen, Yi Zhou, Yingbin Liang, Zhaosong Lu. Generalized-Smooth Nonconvex Optimization is As Efficient As Smooth Nonconvex Optimization. In Proceedings of the 40th International Conference on Machine Learning, 2023.

Haochuan Li, Alexander Rakhlin, Ali Jadbabaie. Convergence of Adam Under Relaxed Assumptions. In In Advances in Neural Information Processing Systems 35, 2023.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper analyzes complexity lower bounds for AdaGrad variants under $(L_0, L_1)$-smoothness conditions. The investigation includes Decorrelated AdaGrad (in both norm and coordinate-wise versions), as well as AdaGrad and SGD with current step-based adaptive stepsizes. A key finding is that optimization under $(L_0, L_1)$-smoothness is fundamentally more challenging than standard $L$-smoothness, as evidenced by lower bounds showing quadratic dependence on constants rather than the linear dependence on $L$ and $\Delta$ typically seen in $L$-smooth settings.

### Strengths
- The paper is well-written, easy to follow, and well-organized.
- The problem it investigates is interesting and important to the study of adaptive algorithms in non-convex settings, given that the discussion of constant dependence in non-convex and $(L_0, L_1)$-smoothness settings is still not rich.
- The results are novel and provide some theoretical insights into adaptive gradient algorithms.

### Weaknesses
The reviewer has two major concerns about this paper:

- The definition of $(L_0, L_1)$-smoothness (Assumption 1) differs from most existing work in this topic ($(L_0, L_1)$-smoothness). Assumption 1 says the inequality holds for any $x$ and $y$. Essentially, for any $y$, we can set $x$ to be a stationary point and, without loss of generality, assume this point is the origin. This implies that $\\|\nabla f(y)\\| \leq L_0 \\|y\\|$, which means the growth of $\\nabla f(y)$ is linear. I guess this is $L$-smoothness.
- Another major concern is the restriction of hyper-parameter selection in the lower bounds. In Theorems 1 and 3, parameter $\gamma$ is required to be small. In Theorem 2, $\\gamma$ appears in the lower bound. While it’s understandable that a small $\gamma$ may be practical, shouldn’t lower bounds typically be related to the best algorithm within a family of algorithms? Parameter tuning should be allowed if one want to explore the limits of an algorithm family. In this case, I’m not sure how insightful these results are, particularly Theorem 2.

Other concerns are relatively minor:


- As mentioned in the limitation section, the lower bound for AdaGrad feels weak because it doesn’t depend on $\\sigma$, especially the $1/\\epsilon^4$ term. Lower bounds for the original AdaGrad should be more interesting than the decorrelated ones, which are not commonly used in practice but are primarily for analytical convenience. However, the difference in lower bounds would also be interesting if there are fundamental differences between decorrelated and the original versions, as mentioned in the paper, “so that the update size of AdaGrad is not as sensitive to noise as the decorrelated counterpart.” However, from the existing upper bounds, it doesn’t seem so.
- The lower bound for Single-step Adaptive SGD is also weak due to the affine-noise assumption. Therefore, we cannot compare it to existing rich upper bounds for algorithms like gradient clipping, which usually assume bounded noise or bounded variance.

### Questions
Could the author clarify the points in Weakness? If there are misunderstandings in these comments, I would like to increase my score.

Update: After reading the author’s response, my first major concern was addressed. However, the second concern persists. The lower bound in Theorem 2 may be vacant when $\\gamma$ is allowed to be tuned. I have adjusted my score accordingly.

### Soundness
3

### Presentation
4

### Contribution
2
