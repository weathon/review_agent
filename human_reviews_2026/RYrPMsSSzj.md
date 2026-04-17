# Improved Stochastic Optimization of LogSumExp

- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
The LogSumExp function, also known as the free energy, plays a central role in many important optimization problems, including entropy-regularized optimal transport and distributionally robust optimization (DRO). It is also the dual to the Kullback-Leibler (KL) divergence, which is widely used in machine learning. In practice, when the number of exponential terms inside the logarithm is large or infinite, optimization becomes challenging since computing the gradient requires differentiating every term. Previous approaches that replace the full sum with a small batch introduce significant bias. We propose a novel approximation to LogSumExp that can be efficiently optimized using stochastic gradient methods. This approximation is rooted in a sound modification of the KL divergence in the dual, resulting in a new $f$-divergence called the *safe KL divergence*. The accuracy of the approximation is controlled by a tunable parameter and can be made arbitrarily small. Like the LogSumExp, our approximation preserves convexity. Moreover, when applied to an $L$-smooth function bounded from below, the smoothness constant of the resulting objective scales linearly with $L$. Experiments in DRO and continuous optimal transport demonstrate the advantages of our approach over state-of-the-art baselines and the effective treatment of numerical issues associated with the standard LogSumExp and KL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a stochastic optimization framework for objectives involving the LogSumExp function (or log-partition functional), proposing a new “Safe KL” divergence formulation that yields a rescaled SoftPlus approximation of LogSumExp. The authors claim this relaxation preserves convexity, smoothness, and enables efficient stochastic gradient methods applicable to entropy-regularized optimal transport (EOT) and distributionally robust optimization (DRO). The theoretical analysis shows convergence properties and connections to Conditional Value-at-Risk (CVaR). Empirical demonstrations include low-dimensional EOT and DRO on the MNIST and California Housing datasets.

The theoretical formulation is sound and mathematically elegant, but the novelty and experimental validation are limited. The approximation technique and results overlap with prior SoftPlus-based LogSumExp relaxations, and the numerical demonstrations are too small-scale to convincingly support the claimed generality.

### Strengths
(1)The theoretical reformulation through an alternative f-divergence is clear and well motivated.
(2)The approach preserves convexity and smoothness, with rigorous proofs of approximation bounds.
(3)Applications to EOT and DRO are reasonable and illustrate numerical stability over the standard LogSumExp formulation with larger stepping.

### Weaknesses
(1)The proposed rescaled SoftPlus approximation is not entirely novel. Similar ideas have appeared—for instance, in “Safety Alignment Should Be Made More Than Just a Few Tokens Deep” (arXiv:2406.05946), where the authors also analyze the effect of the scaling parameter \rho. 
(2) The paper lacks comparison with other established techniques for optimizing or bounding LogSumExp, such as those discussed in 
- Convex Maximization via Adjustable Robust Optimization (https://optimization-online.org/wp-content/uploads/2020/07/7881.pdf)
- LSEMINK: A Modified Newton-Krylov Method for Log-Sum-Exp Minimization” (Kan et al. 2023) deals with optimization of LogSumExp via Newton-Krylov methods (https://arxiv.org/abs/2307.04871)
- Accurate Computation of the Log-Sum-Exp and Softmax Functions (https://arxiv.org/abs/1909.03469)
- The broader review “What is the log-sum-exp function?” (Higham, 2021 https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/)
(3) The experimental design is too simple for a top ML conference. For instance, the “semi-continuous” EOT test uses 1D feature inputs, which is insufficient to demonstrate scalability or real-world relevance. Stronger baselines (e.g., multi-dimensional or high-resolution OT problems) should be included.
(4) The empirical section does not explore how the rescaling parameter \rho affects performance and accuracy, especially in the MNIST experiments (Section 3.3). Sensitivity or ablation tests are needed to justify the parameter’s influence on convergence and bias.

### Questions
(1) The manuscript claims novelty for the “rescaled SoftPlus approximation” of the LogSumExp (LSE) function. Please clarify precisely how your method differs in formulation, analysis, or practical behavior from existing works, for instance listed above.
(2) Can you include additional experiments and comparison with another optimization methods of LSE, such as piecewise-linear approximations, Newton-type solvers, or surrogate bounds, and provide wall-clock/iteration numbers for a range of dimensions and scaling parameters?
(3) Could you provide a sensitivity/ablation study of \rho in both training stage and final objective quality, along with guidance on how to choose \rho in practice?
(4) How does your method scale to higher feature dimensions (e.g., dim > 10, or dim > 100), larger term-counts in LogSumExp, and different \rho values?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper considers the minimization of the functional $f \mapsto \log \int e^f d\mu$, whose specific forms appear in many applications, such as computing softmax probabilities and maximum-likelihood estimation for exponential families. The functional  can be viewed as LogSumExp, whose computation is known to suffer from numerical-stability issues, which the proposed approximation can mitigate substantially as shown emprically in the paper.

### Strengths
The paper studies a fundamental question: the optimization of LogSumExp. The contribution is clear in both theory and the provided empirical results, which demonstrate improvements across various tasks. As far as I can follow, I did not find inconsistencies or technical flaws, and thus the contribution appears significant.

### Weaknesses
Quantitive summary of the results of the experimental section is missing, especially in Section 3.1. Is it Figure 2? This figure is not referenced anywhere in the text. Due to the unclear presentation, it is difficult to obtain an accurate view of how well the proposed approach performs overall compared to the baseline.

Writing is not always smooth. Some examples:

“First, the decision variable φ(or a parameter θdefining φ) often has large or infinite dimension... The first challenge is usually addressed by the use of first-order methods, especially stochastic gradient descent (SGD), with cheap iterations.” What this means? Is a reader supposed to follow? 

Lines 55 and 57: incosistent usage of dashes

### Questions
How to pick the parameter $\rho$ controlling the accuracy of the approximation?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper,  the authors propose a new strategy of approximating the LogSumExp function which is based on the usage of a modified KL divergence (safe KL) in the dual formulation. This approximation is applied in relevant tasks, including the computation of entropy-regularized optimal transport (EOT) problem. It is shown that the approximation helps to alleviate the overflow issue in these applications.

### Strengths
The idea of the paper is rather interesting and supported by the conducted theoretical analysis. The authors show several use-cases of the proposed approximation.

### Weaknesses
My major concerns correspond to the overall contribution of the proposed approximation of LogSumExp and its practical significance. While the derived theoretical results show some nice properties of the derived approximation, its experimental validation has some white spots. To begin with, in section 3.1, the authors tested their approximation by introducing it in the method for entropic optimal transport (Genevay et al., 2016) which is rather old while being classic. For example, this method does not allow for the direct computation of EOT plans which is a more important problem than the computation of EOT cost itself. In recent years, many other methods for computing EOT plans between the continuous distributions have been derived, see, e.g., (Seguy et al., 2018; Daniels et al., 2021, Gushchin et al., 2023, Korotin et al., 2023). The paper would benefit from the testing of your approximation in these more recent approaches.

Besides, while in remark 3.1, the authors note that for small value of parameter $\epsilon$  the method of (Genevay et al., 2016) exhibits the overflow issue, the conducted experiment reveals the behaviour of this approach and its modified version (with the proposed regularization) only for one parameter $\epsilon$. Here I kindly suggest the authors perform an ablation study of this parameter (for bigger and smaller values) to show how it affects the performance of the method with and without regularization. Similarly, I kindly suggest the authors perform an ablation study on the parameters $\lambda$ and $\beta$ appearing in denominator of exponent in the distributionally robust optimization problems – balanced (16) and unbalanced (20) –  in experiments from sections 3.2, 3.3.

Moreover, there are several aspects of the paper which could be written more clearly. First, In equation (2), the authors introduce an alternative form of the functional $F(\phi,\mu)$ using the Gibbs variational principle. I kindly suggest including reference to one of the papers where this principle is introduced. Many formulas should be written in a more rigorous manner, i.e, integrals should include the spaces over which they are taken and sup/inf should be taken over some optimization variable (see Eq. (2)). The typos are summarized below. 

**Overall**, I have doubts regarding the contribution of the approximation of the LogSumExp function derived in the paper. While the theoretical results support this approximation, its usage should offer considerable benefits over the usage of the initial function. To prove it, some additional experimental evaluations with the recent entropy-regularized OT methods and ablation study on the parameters should be considered.

### Questions
- Could you perform an ablation study of the parameters $\epsilon$ in section 3.1 and parameters $\lambda$ and $\beta$ in section 3.2 and 3.3, respectively?
- Could you test your approximation with some of the more recent EOT methods?

*Typos:*

- in Eq. (2): ‘sup’ -> ‘sup_{\nu}’
- in line 108: minimum -> maximum?
- in Eq. (2), (5), (11)-(13) and almost everywhere in the text: \int -> \int_{X}
- how do you calculate the ‘optimality gap’ visualized in Fig. 2?

**References.** 

Seguy, V., Damodaran, B. B., Flamary, R., Courty, N., Rolet, A., & Blondel, M. (2018, April). Large-Scale Optimal Transport and Mapping Estimation. In ICLR 2018-International Conference on Learning Representations (pp. 1-15).

Gushchin, N., Kolesov, A., Korotin, A., Vetrov, D. P., & Burnaev, E. (2023). Entropic neural optimal transport via diffusion processes. Advances in Neural Information Processing Systems, 36, 75517-75544.

Daniels, M., Maunu, T. & Hand, P. Score-based generative neural networks for large-scale optimal transport. Advances in neural information processing systems, 34:12955–12965, 2021.

Korotin, A., Gushchin, N., & Burnaev, E. Light Schrödinger Bridge (2024). In The Twelfth International Conference on Learning Representations.

### Soundness
2

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
4

### Summary
The paper introduces a novel and mathematically sound approximation for the LogSumExp function, termed "safe KL divergence," which is designed for efficient and stable stochastic optimization. The method preserves desirable properties like convexity, and its approximation error is explicitly controlled by a tunable parameter. The empirical results in continuous optimal transport and distributionally robust optimization are strong, demonstrating clear practical benefits in terms of convergence speed and numerical stability over the **chosen** baselines. However, the paper's central claim regarding its novelty is significantly overstated due to a major omission of relevant literature on existing unbiased estimators for the log-partition function, which directly contradicts the assertion that "no cheap unbiased stochastic gradient have been proposed."

### Strengths
1. A key strength is that the bias introduced by the approximation is not arbitrary. The paper provides clear theoretical bounds showing the error is of order O(ρ), allowing practitioners to explicitly trade off accuracy for computational stability (Proposition 2.4 and Corollary 2.5).
2. The approximation preserves essential properties like convexity and smoothness, which are crucial for guaranteeing the performance of gradient-based optimizers.
3. The resulting gradient estimator is simple to implement and does not require complex machinery like control variates or sophisticated sampling schemes, making it an attractive and easy-to-adopt tool.

### Weaknesses
1. The paper does not do justice to prior work on approximating the log partition function. As of now, the prior work is largely incomplete. Few examples:
https://arxiv.org/pdf/2004.00353
https://arxiv.org/pdf/1703.05160
https://arxiv.org/pdf/1306.4032
https://projecteuclid.org/journals/statistical-science/volume-30/issue-4/On-Russian-Roulette-Estimates-for-Bayesian-Inference-with-Doubly-Intractable/10.1214/15-STS523.full
https://arxiv.org/pdf/1703.07370
https://arxiv.org/abs/1901.10517
2. Please add a proof for Lemma 2.7 into the Appendix.
3. Empirical baselines are weak (see item 1 above).

### Questions
1. Could the authors compare with existing unbiased estimators like REBAR or those based on Russian Roulette truncation?
2. Any other baselines Authors would like to add?
3. Why fρ(t) is strongly convex (lines 218, 219)?

### Soundness
3

### Presentation
3

### Contribution
2
