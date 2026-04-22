# Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Optimization under heavy-tailed noise has become popular recently, since it better fits many modern machine learning tasks, as captured by empirical observations. Concretely, instead of a finite second moment on gradient noise, a bounded $\mathfrak{p}$-th moment where $\mathfrak{p}\in(1,2]$ has been recognized to be more realistic (say being upper bounded by $\sigma_{\mathfrak{l}}^{\mathfrak{p}}$ for some $\sigma_{\mathfrak{l}}\geq0$). A simple yet effective operation, gradient clipping, is known to handle this new challenge successfully. Specifically, Clipped Stochastic Gradient Descent (Clipped SGD) guarantees a high-probability rate $\mathcal{O}(\sigma_{\mathfrak{l}}\ln(1/\delta)T^{\frac{1}{\mathfrak{p}}-1})$ (resp. $\mathcal{O}(\sigma_{\mathfrak{l}}^{2}\ln^{2}(1/\delta)T^{\frac{2}{\mathfrak{p}}-2})$) for nonsmooth convex (resp. strongly convex) problems, where $\delta\in(0,1]$ is the failure probability and $T\in\mathbb{N}$ is the time horizon. In this work, we provide a refined analysis for Clipped SGD and offer two faster rates, $\mathcal{O}(\sigma_{\mathfrak{l}}d_{\mathrm{eff}}^{-\frac{1}{2\mathfrak{p}}}\ln^{1-\frac{1}{\mathfrak{p}}}(1/\delta)T^{\frac{1}{\mathfrak{p}}-1})$ and $\mathcal{O}(\sigma_{\mathfrak{l}}^{2}d_{\mathrm{eff}}^{-\frac{1}{\mathfrak{p}}}\ln^{2-\frac{2}{\mathfrak{p}}}(1/\delta)T^{\frac{2}{\mathfrak{p}}-2})$, than the aforementioned best results, where $d_{\mathrm{eff}}\geq1$ is a quantity we call the generalized effective dimension. Our analysis improves upon the existing approach in two respects: better utilization of Freedman's inequality and finer bounds for clipping error under heavy-tailed noise. In addition, we extend the refined analysis to convergence in expectation and obtain new rates that break the known lower bounds. Lastly, to complement the study, we establish new lower bounds for both high-probability and in-expectation convergence. Notably, the in-expectation lower bounds match our new upper bounds, indicating the optimality of our refined analysis for convergence in expectation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a refined analysis of the Clipped SGD method under both convex and strongly convex regimes. Through a more careful application of Freedman’s inequality and an enhanced clipping lemma providing tighter bounds on bias and variance, the authors derive improved convergence rates compared to existing results.

### Strengths
1. **Extensive theoretical results.** The authors provide a fairly comprehensive analysis of various cases. They also emphasize that the discrepancy between the obtained upper bounds and the existing lower bounds is natural, since the classes of problems do not coincide in terms of the stochastic oracle model.

2. **Improvement of the result in terms of $\ln(T)$.** The authors applied the concentration inequality in a tighter form (to me more precise, Freedman's inequality), which allowed them to eliminate the arising factor $\ln(T)$ compared to Das et al. (2024).

3. **Great clipping lemma.** The authors propose a refined clipping lemma that yields improved bounds for bias and variance. Importantly, this lemma (see Theorem 5) does not rely on fact that the norm of the true gradient should be less than $\frac{\tau}{2}$ (this condition is required in many papers, e.g. [1], [2]), which marks a significant methodological advantage.

---

**References**

[1] High-Probability Bounds for Stochastic Optimization and Variational Inequalities: the Case of Unbounded Variance. Sadiev et al. (2023)

[2] High Probability Convergence of Clipped-SGD Under Heavy-tailed Noise. Nguyen et al. (2023)

### Weaknesses
### **Minor weaknesses**

1. **Failure of SGD.** To be fair, the inability of SGD to cope with heavy-tailed noise deserves a clearer explanation. While it is true that, in the absence of a finite second moment for the stochastic gradient error norm, the classical analysis paradigm breaks down, there exists [3] showing that SGD still converges in expectation under condition $p \in (1, 2]$. Nevertheless, it is provable that SGD, and even Adam, fail to adequately cope with this type of stochasticity when high-probability convergence is considered. These results were established in [4] and [5] for SGD and Adam, respectively. Accordingly, work [4] should be cited to substantiate this point.

2. **Clipped SGD or Proximal Clipped SGD.** It would be more appropriate to refer to the method considered by the authors as Proximal Clipped SGD, since the notation introduced in the paper may confuse readers, and this terminology is formally more accurate.

### **Major weaknesses**

1. **Assumptions.** A key limitation concerns the assumptions adopted. In particular, the set $\mathbb{X} \subseteq \mathbb{R}^d$ on which the optimization problem is defined is assumed to be nonempty, closed, and convex. Therefore, $\mathbb{X} = \mathbb{R}^d$ can be considered. At the same time, in the paper, the function $f$ is assumed to be both convex and Lipschitz with constant $G$. Regrettably, this assumption severely restricts the class of admissible functions, as it effectively admits **very limited** examples of $f$ -- linear functions or $|| \cdot ||$ (or other examples of convex functions with unformly bounded subgradient). This automatically leads to the need to impose additional constraints on the set $\mathbb{X}$.

2. **Fighting for $\ln(T)$ elimination.** In fact, one can consider an additional assumption of **boundedness** of the set, that is, taking its closedness into account, the optimization can be considered over a compact set. From the perspective of the class of functions considered, this class is quite broad. However, in this case, striving to eliminate $\ln(T)$ becomes entirely meaningless. In existing analyses, such as in work [6], the factor $\ln\left(\frac{T}{\delta}\right)$ arises from the problem formulation and the assumptions made. The analysis in this work, as well as in other similar studies, is constructed so that the optimization problem is solved over the entire $\mathbb{R}^d$, while the assumptions are made on a compact subset $\mathbb{X}$. Consequently, during the convergence analysis, it is necessary to use induction to ensure that points conducted by optimization method do not leave the domain where assumptions hold. Based on this type of analysis, the probability estimate deteriorates by $\frac{\delta}{T}$ at each step (which is precisely why  $\ln\left(\frac{T}{\delta}\right)$ appears). However, if the problem is formulated on a compact set, no inductive steps are required, and thus the bound with factor $\ln\left(\frac{1}{\delta}\right)$ instead of $\ln\left(\frac{T}{\delta}\right)$ can be achieved.

Emphasizing the above, the main issue lies in the assumptions made, since in the current formulation, without a bounded set, the class of considered functions $f$ is very limited. If boundedness of the set is added, $\ln(T)$ can be eliminated based on existing analyses. Therefore, in my opinion, the main contribution of this work at present is a **more precise treatment of the heavy-tailed stochasticity via Assumption 4**.

Despite the concerns mentioned, I am open to reassess the paper according to further explanations or revisions by authors.

---

**References**

[3] Can SGD Handle Heavy-Tailed Noise? Fatkhullin et al. (2025)

[4]  High-Probability Bounds for Stochastic Optimization and Variational Inequalities: the Case of Unbounded Variance. Sadiev et al. (2023)

[5] Clipping Improves Adam-Norm and AdaGrad-Norm when the Noise Is Heavy-Tailed. Chezhegov et al. (2024)

[6] High Probability Complexity Bounds for Non-Smooth Stochastic Optimization with Heavy-Tailed Noise. Gorbunov et al. (2024)

### Questions
1. Can convergence in the strongly convex setting be proved for the last iterate (for the term $f(x) - f^*$)? What are the technical challenges involved in performing such an analysis (if I am not mistaken, the current results are derived for a specially weighted point)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the clipped stochastic gradient descent under tighter assumptions on heavy-tailed noise (Assumption 4 in the manuscript). They analyze the proximal version, assuming that the objective function is convex and $G$-Lipschitz. By better utilizing the Freedman's inequality, they obtain better convergence rates for high-probability convergence, replacing the $\log\frac{\log T}{\delta}$ term with $\log \frac{1}{\delta}$. For in-expectation convergence, utilizing Assumption 4 results in an improvement by a factor of $\Theta\left(d_{\text{eff}}^\frac{p-2}{2p}\right)$.

### Strengths
1)Improving convergence rates in both high-probability and in-expectation analyses.

2)Considering tighter assumption on heavy-tailed noise and introducing a generalized effective dimension $d_{\text{eff}}$.

3)Conducting a tighter analysis using martingale inequalities.

### Weaknesses
1)Bounded gradients are quite a restrictive assumption for Clipped SGD. 

2)Following the previous point, the lower bound on iteration depends on $G$. Therefore, with exploding gradients we have to perform significantly more iterations.

3)The paper lacks a performance comparison of Clipped SGD with $\sigma_s\neq\sigma_l$, as well as their evaluation on the different datasets.

### Questions
1)What were the main difficulties in adapting the previous analysis in [1]?

2)How can the analysis be extended to more realistic assumptions -- $L$-smooth or $(L_0,L_1)$-smooth?

3)Can the existing analysis be extended for nonconvex case? 

[1] Das, A., Nagaraj, D., Pal, S., Suggala, A., & Varshney, P. (2024). Near-optimal streaming heavy-tailed statistical estimation with clipped SGD. Advances in Neural Information Processing Systems, 37, 8834-8900.

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
The authors consider the problem of heavy-tailed composite optimization with a convex or strongly convex regularizer. They provide refined high-probability rates of convergence for proximal clipped stochastic gradient descent in both convex and strongly convex scenarios. The improvements are based on two main ingredients. The first one is a tighter bound on the variance term in Freedman's inequality. The second one is a better bound on clipping error following from Lemma 1 (in particular, from the refined bounds on $\|\| \mathbb E_{t - 1} [d_t^u (d_t^u)^\top]\|\|$ and $\|\|d_t^b\|\|$). The authors also prove in-expectation bounds (Theorems 3 and 4) with essentially the same rates of convergence.

### Strengths
The rates of convergence in Theorems 1 and 2 are tighter compared to [Das et al., NeurIPS 2024], [Gorbunov et al., JOTA 2024], [Liu, Zhou, arXiv:2303.12277, 2023]. The present paper also removes some artefacts. For instance, in [Das et al., NeurIPS 2024] the upper bound blows up when $\sigma_s$ tends to zero.

### Weaknesses
1. The upper bound on $\mathbb E_{t - 1} X_t^2 = \mathbb E_{t - 1} \langle d_t^u, y_t\rangle^2$ is quite standard. The fact that $\mathbb E_{t - 1} X_t^2 \leq \|\| \mathbb E_{t - 1} [ d_t^u (d_t^u)^\top ] \|\|$ has been used in statistics. I am a bit surprised that it was overlooked in optimization.

2. According to Section 5, the main novelty of the paper is based on the refined bounds on $\|\| \mathbb E_{t - 1} [d_t^u (d_t^u)^\top]\|\|$ and $\|\|d_t^b\|\|$ stated in Lemma 1. Their proofs take a few lines each. For this reason, the contribution looks somewhat incremental.

### Questions
Is it possible to choose the clipping threshold independently of $\delta$? If you could do so, then the results of Theorem 3 and Theorem 4 would easily follow from the high-probability upper bounds.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies (strongly) convex non-smooth stochastic optimization under heavy-tailed gradient noise. Clipped-SGD is analyzed, and two refined rates are derived in both high-probability and in-expectation sense.

### Strengths
1. This paper provides a careful analysis that better utilizes Freedman’s inequality that leads to refined high-probability and in-expectation bounds. 
2. This work generalizes the effective dimension into heavy-tailed robust optimization.

### Weaknesses
1. The boundedness of gradients is a strong assumption given that the motivation of heavy-tailed noise is typically training neural networks.
2. The improvements of the bound is mostly in terms of constant, and it is not clear whether this is a significant improvement. 
3. No experiments provided. Some simple numerical examples to validate theory can also be helpful.

### Questions
1. The improvement depends on the magnitude of $d_{eff}$. It would help to discuss how the noise structure change $d_{eff}$. 
2. The breaking of lower bounds seem to be dependent on the restrictions on noise family. If that's the case, can you discuss more on this point why lower bounds can be improved? 
3. Can this refined analysis be applied on other nonlinearities such as normalization and sign?

### Soundness
3

### Presentation
4

### Contribution
3
