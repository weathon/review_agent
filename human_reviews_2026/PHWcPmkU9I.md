# Convergence of Clipped-SGD for Convex $(L_0,L_1)$-Smooth Optimization with Heavy-Tailed Noise

- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Gradient clipping is a widely used technique in Machine Learning and Deep Learning (DL), known for its effectiveness in mitigating the impact of heavy-tailed noise, which frequently arises in the training of large language models. Additionally, first-order methods with clipping, such as \algname{Clip-SGD}, exhibit stronger convergence guarantees than \algname{SGD} under the $(L_0,L_1)$-smoothness assumption, a property observed in many DL tasks. However, the high-probability convergence of \algname{Clip-SGD} under both assumptions -- heavy-tailed noise and $(L_0,L_1)$-smoothness -- has not been fully addressed in the literature. In this paper, we bridge this critical gap by establishing the first high-probability convergence bounds for \algname{Clip-SGD} applied to convex $(L_0,L_1)$-smooth optimization with heavy-tailed noise. Our analysis extends prior results by recovering known bounds for the deterministic case and the stochastic setting with $L_1 = 0$ as special cases. Notably, our rates avoid exponentially large factors and do not rely on restrictive sub-Gaussian noise assumptions, significantly broadening the applicability of gradient clipping.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors study Clip‑SGD for convex objectives that satisfy $\left(L_0, L_1\right)$-smoothness under heavy‑tailed gradient noise. The authors aim to answer the question of how to choose the clipping level so that it simultaneously respects the geometry induced by $\left(L_0, L_1\right)$ and controls heavy tails (which typically demand growing thresholds).

### Strengths
1. High-probability convergence guarantees for Clip-SGD under both $\left(L_0, L_1\right)$-smoothness and heavy-tailed noise fills a gap, assuming that this is the first work to do so as is claimed.

2. The authors analyze standard Clip-SGD, which is both a strength and a weakness. As a strength, it is closer to what is done in practice, as a "first step" toward more complex analyses.

### Weaknesses
1. There is no empirical section, and the paper would possibly benefit greatly from even synthetic experiments that support their theory. There are no experiments to illustrate behavior versus unclipped SGD or double‑sample methods (even toy convex problems would be useful).

2. Could the authors discuss the primary technical obstacles in extending this high-probability analysis to the non-convex setting (that is, without assumption 1)?

### Questions
Please see weaknesses.

As a small suggestion, one work also may be worth looking into in the literature review is the paper [1], which also heavily studies clipping in the presence of heavy-tailed noise, though in a different setting than this paper. 

[1] Lee et al., Efficient Distributed Optimization under Heavy-Tailed Noise. ICML, 2025.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies clipped stochastic gradient methods for convex $(L_0,L_1)$-smooth objectives under heavy-tailed noise with finite $\alpha$-th moments $(\alpha\in(1,2])$. It presents high-probability guarantees using a single-sample clipping scheme and claims improved comparisons to prior analyses (e.g., avoiding certain exponential dependences) together with brief illustrative experiments.

### Strengths
- The presentation connects $(L_0,L_1)$-smoothness with heavy-tailed noise in a single framework and recovers several special cases.

- Technically careful: the proofs are self-contained and the algorithmic template (standard clipping) is simple to implement.

- The organization is very clear.

### Weaknesses
- Problem novelty is weak. Both \emph{heavy-tailed} robustness for SGD (with clipping/truncation) and the \emph{$(L_0,L_1)$-smoothness} framework have been extensively studied; the paper largely resembles a \emph{combination} of two well-trodden threads (``A + B''), rather than introducing a new core idea or methodology.
- Topic saturation and maturity. Techniques used (clipping-based potential arguments, tail-sensitive concentration) are standard in this area; the contribution reads as a consolidation within known toolkits rather than a conceptual advance.
- Practical guidance is limited: the guarantees hinge on several constants and iteration thresholds (with explicit $\delta$-dependence), yet the paper does not delineate when its schedules outperform light-tailed baselines, nor provide actionable tuning rules when $\alpha$ and $L_1$ are unknown.

### Questions
While technically careful, the paper addresses a non-novel combination: $(L_0,L_1)$-smoothness and heavy-tailed robustness via clipping have each been widely covered, and the present work effectively composes these mature lines without a fresh idea that shifts the frontier. 

Analyzing the advantages of Adam over SGD may be more interesting.

### Soundness
3

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
3

### Summary
This paper investigates the convergence of Clipped-SGD for convex optimization under $(L_{0},L_{1})$-smoothness and heavy-tailed noise where the gradient estimate has a bounded $\alpha$-th central moment for $\alpha \in (1, 2]$.

### Strengths
The authors identify a conflict in prior work: to handle $(L_{0},L_{1})$-smoothness, the clipping threshold $\lambda$ is typically set to a fixed constant, whereas to handle heavy-tailed noise, $\lambda$ needs to grow with the number of iterations $K$.
The main contribution of this paper is to bridge this gap, providing a high-probability convergence bound for Clipped-SGD under both conditions simultaneously with an unified clipping threshold strategy. This result also successfully avoids the exponential dependence on $L_{1}R_{0}$ that appeared in previous work.

The paper is in general well written, and the comparisons with related works are clear to me, especially the one with (Gaash et al. 2025), which make the technical contribution of the paper more clearer and interesting.

### Weaknesses
- Dependence on  $1/\delta$: To establish the high-probability bound, Theorem 1 (case 2) requires the total number of iterations $K = \Omega(\frac{(L_{1}R_{0})^{2+\alpha}}{\delta})$. This polynomial dependence on $1/\delta$ is not standard comparing with  $\log(1/\delta)$ in Theorem 1 (case 1).  Could the authors comment on whether it might be possible to use more advanced probabilistic tools to improve the dependency to $\log(1/\delta)$?

- As noted by the authors in the final section, the paper starts with the convex case, which is reasonable, and it would be also interesting to consider the non-convex case.

### Questions
- Could the authors comment on whether it might be possible to use more advanced probabilistic tools to improve the dependency to $\log(1/\delta)$ for Theorem 1 (case 2)?

- Does the analysis could be also applied to the more general noise models where $\sigma^2 = A(f(x) - f^*) + B \|\nabla f(x)\|^2 + C$ (Yu et al. 2025)?

- As the authors mentioned a lot on ``while in the presence of heavy-tailed noise, the threshold is often required to grow with the total number of iterations to ensure stability and convergence,'' it would be also good to add the threshold parameter in Table 1.

- Missing a “max” in the convergence rate of Thm 1(case 1)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies stochastic optimization for convex and $(L_0, L_1)$ smooth functions under heavy-tailed gradient noise. Clipped-SGD is analyzed and the first high-probability bound is shown for Clipped-SGD for this problem class.

### Strengths
1. The high-probability bound of Clipped-SGD is derived for the first time for the considered problem. 
2. The presented bounds recover the current best result when $L_1  = 0$.

### Weaknesses
1. The problem class and algorithm are both motivated by some deep learning in particular attention models, but the theoretical results are only presented for convex functions. It would be great if non-convex case can be studied. 
2. The considered problem class (for more general nonconvex functions) has been studied in [1] and optimal in-expectation rate using normalized SGD has been derived. It would be helpful if this work can be compared with. 
3. No numerical experiments presented. 


[1] Liu, Zijian, and Zhengyuan Zhou. "Nonconvex Stochastic Optimization under Heavy-Tailed Noises: Optimal Convergence without Gradient Clipping." The Thirteenth International Conference on Learning Representations.

### Questions
1. Given that existing works have studied high-probability convergence for clipped-SGD for convex smooth functions, it would be helpful if the challenges in dealing with additional $(L_0, L_1)$ smoothness can be highlighted.

### Soundness
4

### Presentation
4

### Contribution
3
