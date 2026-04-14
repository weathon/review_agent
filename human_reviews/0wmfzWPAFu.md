# Methods for Convex $(L_0,L_1)$-Smooth Optimization: Clipping, Acceleration, and Adaptivity

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 5

## Abstract
Due to the non-smoothness of optimization problems in Machine Learning, generalized smoothness assumptions have been gaining a lot of attention in recent years. One of the most popular assumptions of this type is $(L_0,L_1)$-smoothness (Zhang et al., 2020). In this paper, we focus on the class of (strongly) convex $(L_0,L_1)$-smooth functions and derive new convergence guarantees for several existing methods. In particular, we derive improved convergence rates for Gradient Descent with (Smoothed) Gradient Clipping and for Gradient Descent with Polyak Stepsizes. In contrast to the existing results, our rates do not rely on the standard smoothness assumption and do not suffer from the exponential dependency on the initial distance to the solution. We also extend these results to the stochastic case under the over-parameterization assumption, propose a new accelerated method for convex  $(L_0,L_1)$-smooth optimization, and derive new convergence rates for Adaptive Gradient Descent (Malitsky and Mishchenko, 2020).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper present a study for gradient method for solving optimization problems involving (L0,L1)-smooth objective function, which was first introduced in (Zhang et al, 2020b). The analysis in the current paper is devoted fully for the convex and strongly convex case, where the convergence analysis of several methods is proopsed, both in the deterministic and stochastic setting. However, there are some major issues needed to be addressed.

### Strengths
The paper is well-writen, and thus easy to follow. The paper has some signification theoretical contributions for the existing algorithms. There are also new algorithms being adapated from the classical smoothness to the new (L0,L1)-smoothness.

### Weaknesses
1. The analysis of the current paper is mostly based on Assymetric and Symmetric (L0,L1) smoothness, which can be hold for functions that are not twice differentiable. This is good since this class cover the original class of function in (Zhang et al, 2020b), where twice differentiability is a must. However, which meaningful classes of functions satisfy Assymetric/Symmetric (L0,L1) smoothness while not C2? I found that all the examples presented are C2, and thus it seems that the use of  Assymetric and Symmetric (L0,L1) smoothness is not neccesary, which reduces the importance of the current paper much. Note that examples for L-smooth functions that are not C2 are diverse, so it is reasonable to use the gradient Lipschitz condition instead of the stronger one, bounded Hessian. I do not think it is the case for (L0,L1) smoothness.

2. Now assume that the function is C2. Lemma 2.1 shows that (L0,L1) smoothness implies (not equivalent) equation (7). In comparison with Lemma 2.2 (Vankov et al., 2024) below, their equation (2.2) seems to be a shaper inequality and in fact is a equivalent condition. Based on this, I suspect that the results obtained by the paper under review is not as tight as claimed by the authors, especially when compared with (Vankov et al., 2024).  

3. The convergence theory of algorithms for (L0,L1)-smooth function does not explain why they are better than standard gradient descent. For example, can the authors explain why standard GD performs worse than designated algorithms for (L0,L1)-smooth function in solving logistic regression?

Reference.
D Vankov, A Rodomanov, A Nedich, L Sankar, SU Stich, Optimizing (L0,L1) - Smooth Functions by Gradient Methods, https://arxiv.org/abs/2410.10800

### Questions
See weakness

### Soundness
2

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
5

### Summary
This paper focuses on analyzing optimization methods for convex problems under $(L_0,L_1)$-smoothness settings. It provides improved convergence rates for the Gradient Descent (GD) method with Gradient Clipping and the Gradient Descent method with Polyak Stepsizes. It introduces a new accelerated method based on the Similar Triangles Method and provides new convergence rates for the Adaptive Gradient Descent Method. Finally, it extends the analysis to the stochastic case in overparametrized settings.

### Strengths
1. The analysis in this work does not rely on the L-smooth assumption, which was required in previous works. In this way, the work proves a convergence rate for the Gradient Descent (GD) method with Gradient Clipping and the Gradient Descent method with Polyak Stepsizes, with the dominant part having a smaller constant depending on $L_0$.

2. This work proposes a new variant of the Similar Triangles Method that accelerates GD.

3. This work provides a faster convergence rate for the Adaptive Gradient Descent Method in $(L_0,L_1)$-smoothness settings compared to the locally L-smooth setting.

### Weaknesses
1. A new acceleration of GD is proposed. It would be supportive to add more remarks to highlight the theoretical merits of this acceleration compared with the STM in (Gasnikov & Nesterov, 2016). Additionally, it would be more supportive to numerically compare its performance with STM in (Gasnikov & Nesterov, 2016).

2. Example 1.3 considers a logistic function with L2 regularization. However, $f(x)$ is not related to L2. It would be better to specify where the L2 regularization is.

3. The discussion after Theorem 7.1 (line 521) claims that the probability must be smaller than $\frac{8nL_1^2\|x^0-x^*\|^2}{\eta\nu(N+1)}$. It would be clearer to explain why the probability should be smaller than this value.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper takes a closer look at $(L_0,L_1)$-smoothness for the setting of convex optimization. There, it derives more fine-grained convergence rate guarantees than existing work, while discussing extensions to accelerated, stochastic, and certain adaptive settings.

### Strengths
**UPDATE:** Given the authors willingness (mentioned in their final comment) to make the paper's claims more transparent and to perform the pedagogic and scholarly improvements, I have raised my score.

---
The paper makes a technical contribution to the convergence analysis of gradient descent(like) methods, under the general $(L_0,L_1)$ smoothness for differentiable functions. In the reviewer's opinion, the main contribution is a tighter bound on Clip-GD under the generalized smoothness assumption (e.g., The Polyak stepsize is chosen to minimize the upper bound in (59), so the result for GD-PS is not hard to get after having proved (45) for Clip-GD, while STM and AdGD have exponential terms in their upper bounds, and hence only of marginal interest).

Strengths: This article fills a gap in the analysis of convex L0-L1 problems that are differentiable, and claims to do that without having to resort to small stepsizes as in the recent work of Li et al. The work is reasonably clearly written, and is easy to follow. Some aspects that could do with more discussion are the two-phase nature of GD (with the $>, < L_0/L_1$), and what that may mean when compared with other work (does the same happen in the nonconvex case for instance? how does it relate to bounds from the work of Li et al for instance?)

### Weaknesses
*  From a quick skim of the literature, it seems that Zhang et al noted in passing that twice differentiability could be dropped at the expense of some more analysis, but the full details of the differentiable case were worked out in subsequent work. This aspect is not clear from how the current work cites related work, and should be fixed.

* While the related work section is overall fairly good, a more precise statement about the results of Chen et al is needed, especially because that work introduced some of the key technical tools too, and studied the nonconvex case; in particular, it would be worth noting what happens if one trivially takes their nonconvex results, and tries to adapt them to the convex case (by boosting stationarity to function suboptimality using the current assumptions). Also, their slightly more general $\alpha$-version of Assumption 3 could be noted.

* Section 5 on acceleration could be deferred to the appendix or noted in passing, and more discussion given to the exponential terms in the bound, which are tantamount to saying that essentially no practical acceleration happens, even though the technical result itself is interesting to note. Similarly, the bounds arising in Section 6 should be discussed a bit more, because due to the central assumption of the paper, ultimately the pessimistic exponential terms in D arise.

* The authors's motivation for studying AdGD should be as a result further enhanced: it seems inherently unsuitable to use under the generalized smoothness assumptions since AdGD does not utilize clipping.  And the result for the stochastic case (Section 7) requires a common minimizer for all the stochastic components (Assumption 4). Although such an assumption is also used in some works, this assumption is not that weak, and renders the results less applicable.

### Questions
* Can a version of the method that is somewhat more agnostic of the knowledge of L0 and L1 be designed, since right now this knowledge is quite necessary for selecting the correct step sizes?

* Ultimately, a high-level takeaway of the work (which is likely also a takeaway that follows from the Chen paper if one could quickly bridge their stationarity to function suboptimality here) seems to be that the results are ultimately of a *local nature*. What I mean by that is that due to the unavoidable dependence of the type $\exp(\|x-y\|)$, the terms involving that in the bound will be large, unless $\|x-y\|$ is sufficiently small, and hence, in spirit the results can be viewed as showing that the GD methods studied in the paper are good but only locally. This aspect is not a criticism of the paper, but just a comment on how one can typically interpret bounds that involve exponentials.

* Do the results really offer a significant improvement over Li et al. 2024a (_Convex and Non-convex Optimization Under Generalized Smoothness_). The results provided by Li et al. 2024a for GD and NAG also do not rely on the assumption of L-smoothness, and their acceleration results for NAG  **do not** have a dependence on exp(L1). The authors claim that the advantage of their results is that they do not require dependencies on $\Vert \nabla f(x_0) \Vert$ and $f(x_0) - f^*$, as in Li et al. 2024a, and they argue that these quantities could potentially be exponentially large in the worst case (Line 408-420)---But in the reviewer's opinion, assuming these initial quantities to be constants is not a very strong assumption, whereas in comparison, the authors' acceleration result depends on exp(L1), which seems to be less favorable.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper focuses on convex $(L_0, L_1)$-smooth optimization and answers important open questions in this domain. In particular, new convergence rates are derived for the gradient method with smoothed clipping and Polyak stepsizes, improving existing results. The best-known convergence rate is derived for $(L_0, L_1)$-STM. Also, new results proved for AdGD and a stochastic case under the additional assumption on a shared minimizer. The statements in the paper are clear, and compared with existing results in the literature, the proofs are correct.

### Strengths
1. Paper improved existing convergence results for gradient methods with (smoothed) clipping and Polyak stepsizes.
2. New convergence results for AdGD under $(L_0, L_1)$-smooth assumption.
3. Proposed $(L_0, L_1)$-STM recovers the best-know convergence rate for accelerated methods without additional knowledge on $R_0$ and $f(x^0) - f^*$.

### Weaknesses
1. The paper is missing the conclusion and experiments sections. However, the experiments are provided in the appendix.
2. The assumptions for stochastic problem is restrictive; however, the derived results are new and better than previous ones.

### Questions
- In the last part of the equation (15), should it be N+1 - T instead of N+1?
- Why is it so crucial to choose $\gamma= 1/4$? In Theorem 6.1, to achieve a rate better than in (25)? Can it be relaxed?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper analyzes the iteration complexities of several algorithms targeted at convex $(L_0,L_1)$-smooth optimization. In comparison to previous works, the authors focus on a more refined analysis, including the elimination of dependence on the smoothness parameter $L$ (although it is not the dominant term) for variants of Gradient Descent, as well as improvements on the naive adaption of Adaptive Gradient Descent for generalized smoothness.

### Strengths
This paper covers existing methods for convex $(L_0,L_1)$-smooth optimization, specifically Gradient Descent with Smoothed Clipping and Polyak Stepsizes. It also provides analyses of Similar Triangles Methods and Adaptive Gradient Descent under the setting of generalized smoothness. The results give an overview of existing and adapted methods.

### Weaknesses
My major concern is about the significance of the results. To be specific, for the first two variants of Gradient Descent, this paper only improves the non-dominant $\mathcal{O}(\sqrt{1/\varepsilon})$ term of iteration complexity, which is inconsequential for small $\varepsilon$, i.e., when finding a solution with good quality. For the adapted versions of Similar Triangles Method and Adaptive Gradient Descent, the iteration complexity is in the form of $\mathcal{O}(\sqrt{L_0L_1A\exp(L_1A)A^2/\varepsilon^2})$, where $A$ equals either $R_0$ or $D$, which generally aligns with a specification of Li et al. (2024). Thus, the overall contribution in terms of novel theoretical guarantees appears quite limited to me at this stage.

### Questions
**Typos and minor suggestions:**

Line 203: Since you cite the conference version rather than the arXiv version of the paper, please refer to it as "Proposition 1".

Line 383-387 / Inequality (20-21): I suggest that the authors avoid breaking the entire inequality into two labels, which can cause ambiguity as seen in the second step of (73). Using an underbrace might be a better option.

Line 419: "$\varepsilon$-solution".

Line 1095: a multiplier of $\exp(\eta)$ is missing in the last term.

**Questions:**

* As I mentioned in the Weaknesses section, are there any practical examples (either theoretical or experimental) that can justify the importance of the *improved* complexities?
* For Theorem 3.1, it appears that you integrate the analyses from Li et al. (2024) and Koloskova et al. (2023), substituting the normalization term $\ell(G)$ in $\eta$ into something related to $\|\nabla f(x^k)\|$, so that the sequence enjoys the monotonic properties and can be analyzed in two cases. Could you elaborate on the intuition behind this approach?
* You mention new technical results for $(L_0,L_1)$-smooth functions in Section 1.3. Could you specify these results for reference? 



Haochuan Li, Jian Qian, Yi Tian, Alexander Rakhlin, and Ali Jadbabaie. Convex and non-convex optimization under generalized smoothness. In Advances in Neural Information Processing Systems 36, 2024.

Anastasia Koloskova, Hadrien Hendrikx, and Sebastian U Stich. Revisiting gradient clipping: Stochastic bias and tight convergence guarantees. In Proceedings of the 40th International Conference on Machine Learning, 2023.

### Soundness
3

### Presentation
3

### Contribution
1
