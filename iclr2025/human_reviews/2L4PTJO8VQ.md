## Human Reviewer 1

### Summary
The paper considers the problem of first-order stochastic optimization where the learner can only access a stochastic gradient whose expectation is only guaranteed to be correlated with (but not equal to) the true gradient at the queried point.  The authors consider three different settings commonly encountered in machine learning problems where the learner can only access biased gradients. For each of the three settings, they propose a new algorithm and provide its analysis.

### Strengths
I like the paper overall. It uses simple but interesting additions to existing strategies for optimization to design the algorithm with biased gradient estimates which often yield optimal performance. I think the results are sufficiently novel and interesting and improve upon the best known results so far.

### Weaknesses
I don't see any glaring weakness in the paper but I have some questions listed in the next section.

### Questions
1. Can the authors explain the intuition behind the update in line 6 in algorithm 2? It seems like you just want to consider the update in the orthogonal direction but I can't quite understand why? Is it simply to reduce the norm of the update (and not considering the update along $x\_t$ helps) or is there something fundamental that is going on there?

2. Does strong-convexity help for algorithm 2 and 3? In other words, if we are given the additional information of strong convexity, how much does that help improve the error? Particularly for alg 2, if it does not help then what exactly from the analysis in Demidovich et al, 2023 does not work out in this case?

3. I am not sure if this will work, but might be worth a try: If I understand correctly, the term $f(x_t) - f(x^{\star})$ in line 413 cancels out with the term $-\frac{\eta\_t \alpha \| g\_t \| }{3}$ in line 416. I think with an appropriate change of constants, we can retain a fraction of the negative term in line 416 and carry forward it to the equation in line 421. Now, if $\|g\_t\| \leq \frac{C}{\sqrt{t}}$ for some $C > 0$, then we are at a point with small gradient. Otherwise, it will cancel out the $\frac{1}{\sqrt{B_t}}$ term in the equation in line 421 with $B_t = t + 1 + k$. This might help you achieve optimal error rate. Of course this needs to be checked but it might be helpful to address the sub-optimality gap.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper studies stochastic convex optimization where the stochastic gradient oracle is biased but correlated with the true gradient. The proposed algorithms achieve the following performances: for Lipschitz, convex objectives and slowly varying bias, the rate is O(N^{-1/2}); for Lipschitz, smooth convex objectives and general correlated stochastic gradient oracle, the rate is O(N^{-1/3}). The results are applied to problems with hidden convexity, which achieves a rate O(N^{-1/3}).

### Strengths
The paper is overall well written, with clearly presented setups, algorithms, and performance. In addition, the correlated stochastic oracle studied, as pointed out on page 2, might have broad applications.

### Weaknesses
-- The paper is closely related to the stochastic optimization literature. Although the authors have cited many relevant works, the exact known results are missing in this paper. It might help the readers better appreciate the significance of the results by providing more details and/or comparisons with existing setups and known upper/lower bounds on the convergence rate.

-- The assumptions of each theorem are stated at the beginning of each corresponding section (informally). It might be better to present them more formally, either as Asssumption 1/2/3, or stated directly in the theorems. 

-- In terms of significance, it is unclear how tight the bounds are. Would it be possible to derive some lower bounds from known results for other related problems? This would greatly help the readers appreciate the significance of the results. In addition, it seems that the analysis is relatively standard. Could the authors provide more comparisons with existing proofs for stochastic convex optimization, or related problems/setups?

### Questions
-- All the convergence results are presented in expectation. I’m wondering how hard it is to obtain ``with high probability’’ performance guarantee? 

-- In line 113, ``note that this is equivalent to the condition that …”, this seems to require that f is differentiable. Otherwise, the gradient of f may not exist, and instead the bound holds for all subgradients.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper studies oracle-based optimization in three settings. First, when the oracle returns gradients that are misaligned with the true gradients in a specific manner: the expectation of the returned gradient is positively correlated with the true gradient (in terms of the inner product). Second, for more specific applications, they strengthen this assumption, and require that the lower bound not just be nonzero, but that it is at least the square of the true gradient. Third, for their setting of hidden convexity, they use the standard unbiased estimator assumption. 

The paper provides improved rates of convergence under all three settings.

### Strengths
I think it's a well-motivated paper with a coherent set of results. I like that the analysis in Section 3, though, simple, is complete and step-by-step. The authors are also quite honest about differences with prior work, though they could do a better job explaining why they do better than existing work in similar assumptions (see Questions).

### Weaknesses
My only complaint is that the paper's introduction suggests that the only assumption made is that the inner product of the true gradient with the expected gradient provided by the oracle is positive: however, this seems to hold only in Section 3. In Section 4, this is strengthened to say that the lower bound is the squared norm of the true gradient (an assumption same as Beznosikov et al), and in Section 5, it's further strengthened to be simply an unbiased estimator. Is my understanding accurate? If so, the "misaligned" description used throughout the introduction applies only to Section 3, and for the other results, there already exists standard terminology for those assumptions, so assigning them a new name wouldn't be the right thing to do. 

My recommendation to the authors is to please clarify all the assumptions (for each of the different settings studied) in the introduction, so as to avoid any confusion. 

Further, it would be useful to have a better understanding of what specific difference in the analyses in Section 4 lead to the improved rates as compared to existing work under this assumption (see Questions).

### Questions
1. Does the analysis in Section 3 have any connection with the analysis seen when using self-concordant barriers? The assumption that two matrices $A_t$ and $A_{t+1}$ do not change much is quite similar to saying that two successive Hessians do not (which is essentially what self-concordance captures). If the authors believe there could be a connection to this, it would be useful to add that to the paper and add pointers to the literature on interior-point methods, where this notion is used; if not, then it would still help to clarify why it differs. 

2. The assumption in Section 4 that the inner product of the true gradient and expected gradient (from the oracle) is lower bounded by the square of the true gradient norm is identical to that in Beznosikov et al (as the authors themselves note). Can the authors explain what exactly they do differently to improve the $\epsilon^{-4}$ rate to $\epsilon^{-3}$? Could they point to a specific step in their proof where they use this inner product assumption in a better manner?

3. There was a recent paper https://arxiv.org/abs/2304.08596 by Shu, Ramachandran, and Wang, which also talks about hidden convexity. I think it would be useful to cite the paper if the way the phrase "hidden convexity" is used is the same. If not, it would be helpful to clarify the differences.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper focuses on the case where the stochastic oracle produces feedback which is not necessarily unbiased. More precisely, it introduces the notion of misaligned stochastic gradients in order to capture the lack unbiasedness in several practical scenarios. To that end, the authors test their theoretical machinery for the optimization problems with hidden convexity (also studied in Sakos 2024 and references therein) and provide an algorithmic method which exhibit $\mathcal{O}(\varepsilon^{-3})$ iteration complexity.

### Strengths
The paper is very well written and easy to follow. Moreover, the mathematical analysis is sound and clear as far I have checked. The idea of misaligned stochastic vectors is quite intuitive and as far as my knowledge goes it paves the way for dealing with a useful practical methodology for structured biased stochastic gradients.

### Weaknesses
Concerning this paper I have two main concerns/questions:

1. The almost sure boundedness of the biased gradients seems to be a quite restrictive statistical assumption. As far as my knowledge this type of assumption is usually used in methods which are run with adagrad-type step-sizes (see for example Levy 2017). Thus, my question is two-fold: Does this statistical assumption hold in practice and secondly do the authors believe that it is an artefact of the analysis or the method in order to overcome it ?

2. The paper lacks a numerical comparison with other methods which consider biased gradients like the Stich 2020 paper. My question concerns the fact that the compression scheme presented in the said paper seems to cover the case of an "relative bias" (an analogy to Polyak's relative noise) in the sense that the bias vanishes when we approach a solution. To that end, some simple calculations may show that under this condition the second assumption in oracle & assumptions may be recovered. So, I think that a more thorough discussion is needed.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2