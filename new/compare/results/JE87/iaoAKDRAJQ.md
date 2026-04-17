# Review

## Summary
This paper studies the convergence of adaptive optimization algorithms such as Adam and its variants under a general non-Euclidean geometry. In particular, the authors extend the adaptive smoothness condition from the convex setting to the nonconvex setting and show that it precisely characterizes the convergence of adaptive optimizers. Moreover, the authors show that the adaptive smoothness enables acceleration of adaptive optimizers with Nesterov momentum in the convex setting, which is unattainable under the standard smoothness condition for certain non-Euclidean geometry. The authors also develop an analogous result for stochastic optimization by introducing adaptive gradient variance, which parallels adaptive smoothness and leads to dimension-free convergence guarantees that cannot be achieved under the standard gradient variance for certain non-Euclidean geometry.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
The paper is very well-written and easy to follow. The motivation is clear and the theoretical results are solid. In particular, the extension of adaptive smoothness from convex setting to nonconvex setting is very interesting and the obtained results are novel. The technical contribution is solid.

## Weaknesses
I do not have any major concern about this paper. I only have a few minor comments which are provided below.

1. In the abstract, the authors mentioned that "We further develop an analogous comparison for stochastic optimization by introducing adaptive gradient variance, which parallels adaptive smoothness and leads to dimension-free convergence guarantees that cannot be achieved under standard gradient variance for certain non-Euclidean geometry." It would be better if the authors could provide more explanations regarding the "certain non-Euclidean geometry".

2. In Definition 2.4, it would be better if the authors could provide more explanations regarding the second equivalence. In particular, the authors may want to explain the relationship between the trace of $H$ and the operator $\preceq$.

3. In Section 2.2, the authors mentioned that "This term is introduced as H-smoothness in Xie et al. (2025b). We rename it to highlight this notion adapts to the structure of H, in contrast to the standard smoothness." It would be better if the authors could provide more explanations regarding the "adapts to the structure of H".

4. In Section 3.1, the authors mentioned that "Algorithm 1 recovers several standard optimizers by specifying H as follows:". It would be better if the authors could provide more explanations regarding the connection between Algorithm 1 and the standard optimizers.

5. In Section 3.3, the authors mentioned that "A central difficulty in our analysis is the extension from diagonal preconditioners to a general preconditioner set H. In the diagonal case, the proof basically decomposes to entry-wise analyses, and scalar telescoping readily yields the desired bounds. However, for general H, noncommutativity prevents such simplification, and bounding the second-order terms requires handling delicate matrix inequalities." It would be better if the authors could provide more explanations regarding the "decomposition to entry-wise analyses".

6. In Section 4, the authors mentioned that "At a high level, these two angles share the same underlying mechanism: Under non-Euclidean geometry, averaging might not be effective in reducing the norm." It would be better if the authors could provide more explanations regarding the "might not be effective in reducing the norm".

7. In Section 4.1, the authors mentioned that "This adaptive variance is inspired by the noise assumption in Kovalev (2025a), both capturing the overall variation of gradient noise in the geometry induced by H." It would be better if the authors could provide more explanations regarding the "overall variation of gradient noise".

8. In Section 4.3, the authors mentioned that "In particular, the dimension-dependent factor $\rho = \sup_x \frac{\|x\|_{H,*}}{\|x\|_2}$ captures the mismatch between $\| \cdot \|_{H,*}$ and $\| \cdot \|_2$." It would be better if the authors could provide more explanations regarding the "mismatch between $\| \cdot \|_{H,*}$ and $\| \cdot \|_2$".

9. In Theorem 4.7, the authors mentioned that "Theorem 4.7 also shows two kinds of lower bound we can achieve on signGD with momentum". It would be better if the authors could provide more explanations regarding the "two kinds of lower bound".

## Questions
Please see the Weaknesses section above.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4