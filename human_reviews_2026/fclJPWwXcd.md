# Sample Complexity of CVaR Based Risk Sensitive Policy Learning

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
The conventional offline bandit policy learning literature aims to find a policy that performs well in terms of the average policy effect (APE) on the population, i.e. the **social welfare**. However, in many settings, including healthcare and public policies, the decision-maker also concerns about the **risk** of implementing certain policy. The optimal policy that maximizes social welfare could have a risk of negative effect on some percentage of the worst-affected population, hence not the ideal policy. In this paper, we investigate risk sensitive offline policy learning and its sample complexity, with conditional value at risk (CVaR) of covariate-conditional average policy effect (CAPE) as the risk measure. 
To this end, we first provide a doubly-robust estimator for the CVaR of CAPE, and show that the this estimator enjoys asymptotic normality even if the nuisance parameters suffer a slower-than-$n^{-\frac{1}{2}}$ estimation rate ($n$ being the sample size).
We then propose a risk sensitive learning algorithm that finds the policy maximizing the weighted sum of APE and CVaR of CAPE, within a given policy class $\Pi$. 
We show that the sample complexity of the proposed algorithm is of the order $O(\kappa(\Pi)n^{-\frac{1}{2}})$, where $\kappa(\Pi)$ is the entropy integral of $\Pi$ under the Hamming distance. The proposed methods are evaluated empirically, demonstrating that by sacrificing not much of the social welfare, our methodology improves the outcome of the worst-affected minority population.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses offline policy optimization focusing on the Conditional Value at Risk (CVaR) of the Covariate-Conditional Average Policy Effect (CAPE) as the objective, rather than the traditional Average Policy Effect (APE). The authors first propose a doubly-robust estimator for CVaR-CAPE, which naturally extends the existing doubly-robust estimator for APE. Building upon this, they adapt the offline policy learning framework from APE maximization to handle the CVaR objective, introducing the $\lambda$-RSL algorithm that optimizes a weighted sum of APE and CVaR-CAPE. Finally, the paper provides a regret bound for the learned policy, leveraging uniform convergence results derived for their CVaR estimator.

### Strengths
This work represents a natural and meaningful extension of the offline multi-action policy learning framework established by Zhou et al. (Operations Research, 2023) to incorporate risk sensitivity via the CVaR objective. The methodology effectively integrates insights from Kallus (Management Science, 2023) regarding risk estimation. This integration addresses the important practical need for risk-aware decision-making in policy learning contexts and is well-positioned within the current literature.

### Weaknesses
While technically sound, the paper's primary contribution lies in extending existing frameworks (APE optimization and CVaR estimation) to the CVaR-based policy learning setting. The core novelty arises from addressing the technical details introduced by the CVaR term in the objective function.

Several concerns arise regarding this extension:

1. **Uniformity of Regularity Conditions**: Lemma E.1, which bounds the estimation error, includes a density term $F_{\mu_\pi(X)}'$. While the convergence rate analysis for the VaR estimate $\hat\beta_\pi$ (Lemma 3.4) might implicitly handle the dependence on this term for a fixed $\pi$, the regret analysis (Theorem 4.3 11) requires uniform convergence bounds over the entire policy class $\Pi$. This necessitates that terms like $F_{\mu_\pi(X)}'$ (evaluated near the $\alpha$-quantile) remain uniformly bounded across all $\pi \in \Pi$. If $F_{\mu_\pi(X)}'$ could become arbitrarily large for some policies, this could potentially undermine the uniform bound and thus the validity of the regret analysis. The paper does not explicitly address or assume this uniform boundedness.

2. **Compatibility of Assumptions**: There appears to be a potential tension between Assumption 3.3 (Regularity of Quantile) and the $L_\infty$ convergence assumption ($||\hat\mu_{\pi} - \mu_{\pi}||{L_\infty} = o_p(n^{-1/4})$) within Assumption 3.2. Assumption 3.3, especially the focus on continuous differentiability of the CDF, seems most natural when the covariate space $\mathcal{X}$ is rich and continuous. However, achieving uniform ($L_\infty$) convergence rates for estimators like $\hat{\mu}_\pi$ over such rich spaces can be challenging, often requiring strong smoothness assumptions. The paper could benefit from clarifying the conditions under which both assumptions simultaneously hold for typical policy classes and estimators.

3. **Optimization Formulation**: Appendix C discusses the algorithm's convergence by formulating the problem as a bilevel optimization. While technically correct due to the definition of CVaR involving $\sup_\beta$, this formulation might be misleading. It's well-known that alternating optimization methods applied to general bilevel problems do not guarantee convergence to the true optimum. It might be clearer to view the problem as a joint optimization over $(\pi, \beta)$, where the optimal $\beta*$ satisfies the inner problem's optimality condition (i.e., $\beta*$ is the VaR for the optimal $\pi^*$). Framing it this way highlights that the alternating scheme used in $\lambda$-RSL is a heuristic for solving the joint problem, whose convergence properties (beyond stationary points for smoothed versions) remain an open question addressed only empirically in the paper.

### Questions
1. Regarding Theorem 4.3: Does Assumption 3.3 (Regularity of Quantile) need to hold uniformly for all policies $\pi \in \Pi$ for the regret analysis to be valid? If so, is this a reasonable assumption for common policy classes?

2. Lemma E.1 contains the density term $F_{\mu_\pi(X)}'$, but this term does not explicitly appear in the final regret bound of Theorem 4.3. How is this term handled or absorbed during the derivation of the final bound? Does this rely on an implicit assumption of uniform boundedness?

3. What is the complexity measure $\kappa(\Pi)$ (Hamming entropy integral) for the softmax policy class examined in the numerical experiments?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, the authors propose a framework for learning decision-making policies that maximize population-level social welfare while minimizing negative impacts on high-risk subpopulations. The authors specifically adopt the conditional value at risk (CVaR) of covariate-conditional average policy effect (CAPE) as a sub-population risk measure. The authors propose a doubly-robust estimator for the CVaR of the CAPE and establish its asymptotic normality. They then devise a policy learning algorithm that minimizes a weighted combination of the CVaR of the CAPE an the average policy effect, and illustrate that its sample complexity is on the same order as the baseline CAIPWL approach. The authors validate the proposed approach via experiments on synthetic and real-world data.

### Strengths
Policy learning is an important and timely problem. The authors identify a key issue in standard policy evaluation and learning frameworks — that some individuals may fare worse under policy changes, despite marginal population  gains. The proposed approach of learning a weighted combination of the CVaR and APE objective is well-motivated, and the authors report empirical evidence demonstrating promising performance of the proposed approach. Further, theoretical claims are supported by appropriate analysis and proofs.

### Weaknesses
At its core, the motivating problem and theoretical framework are both sound and interesting. However, in its current form, there are several issues with the work: 

## Empirical Validation: 

My chief concern surrounds the empirical validation of the work. While the current results present some supporting evidence of the findings, in my view, they are do not provide conclusive evidence of the efficacy of the proposed approach.

First, the decision to parameterize the learned policy via a softmax policy class is in tension with the high-stakes problem setting. In settings where we want to minimize CVaR of CAPE, there often concerns that undue harm can be caused to decision subjects by a policy, and these harms should be avoided. In such cases, a stochastic softmax policy class introduces stocastic action selection that may be poorly motivated. I invite the authors to justify this choice in the rebuttal / next version of the draft. 

Second, while the authors present empirical evidence that the proposed approach improves CVaR under small alpha (e.g., 0.01, 0.05), the gains under the proposed approach appear to attenuate quickly as alpha increases. Could the authors present results covering a larger range of alpha values -- e.g., 0.4 or 0.5? As currently presented, gains appear to be marginal when alpha > 0.1.

More broadly, it would be helpful to present an analysis that jointly varies alpha and lambda so that the reader can see the impact of each term in the objective as a function of alpha (fixing the sample size is fine for this analysis). It would also be helpful to see how varying lambda impacts both the CVaR and the APE in parallel -- e.g., through a contour plot that show a Pareto tradeoff or similar. The goal of this analysis should be to provide further evidence for the style of claim that is made via lines 456-462. While I the discussion in lines 456-462 is both interesting and strong, I view the scope of these results as limited given that it is fixed to $\alpha=0.01$, and that results appear to weaken for larger values of alpha. 

## Presentation: 

The current presentation of the work has several limitations which make it difficult to follow in some places. While only the first bullet impacts my score, I encourage the authors to revise the draft for polish and clarity:

- The current presentation of empirical findings makes it challenging to parse the main results of the paper. In particular, it is difficult to compare results with the long scientific notation (e.g., 3.9492e-2 +- 1e-3). Additionally, while the authors note that Table 4 reports confidence intervals for Figure 4, it is important to also include them in this plot so the reader can understand statistical uncertainty in these main results. 
- Typo: The counterfactual IPE of "any given any policy" => "any given policy" (line 66)
- Typo: Under slow parameter estimation rates (line 177). Does this refer to slow nuisance function estimation rates? 
- Non-standard inline reference style (244-248)
- The current conclusion is one short sentence. I encourage the authors to expand this to a short section to give the work a natural conclusion. 
- Minor: I was initially confused by use of Z in e.q. 1 given the footnote directly below defining Z as a tuple of random variables. Consider using a different R.V. letter here for clarity. 
- Presentation of 3.1 is currently dense, with several assumptions, lemmas, and results presented in quick succession with limited surrounding discussion. I encourage the authors to strengthen the narrative flow of this subsection. 

## Related Work: 

Coverage of related work is generally good, but some relevant work is missed. E.g., 

Policy Learning with Asymmetric Counterfactual Utilities, Eli Ben-Michael, Kosuke Imai, Zhichao Jiang, https://arxiv.org/abs/2206.10479

### Questions
- For assumption 2.1, is a consistency and/or STUVA assumption also required? Such assumptions typically take the form $Y = Y(a)$, for all $a \in A$.

- The debiased estimator presented on line 206 relies on several non-linear components, including an infimum and indicator function. The typical approach used to derive debiased estimators involves subtracting off the efficient influence function for debiasing - a step that requires pathwise differentiability which would be violated by these non-linear terms. How is this addressed in the framework? 

- Is cross-fitting or some additional proof strategy required to address the re-use of data twice - once for value function estimation and once for policy learning?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Standard policy learning focuses on minimizing the learning the optimal policy that minimizes the average/expected policy value. Since we cannot identify the CVaR of the outcome under a policy, the authors consider learning a policy that minimizes the CVaR of the conditional mean outcome under a policy.

### Strengths
-The paper is clearly written and easy to understand. 
-The problem of robust policy learning part of a broad and growing literature on robust and distributionally robust policy learning.

### Weaknesses
A critical limitation of this paper is that it references Kallus et al, 2023 to claim that the following is true:
$ \text{CVaR}(Y(\pi(X)) \leq \text{CVaR}(E[Y(\pi(X)) \mid X])$.
However, I believe this inequality should actually be reversed, i.e. $\text{CVaR}(Y(\pi(X)) \geq \text{CVaR}(E[Y(\pi(X)) \mid X])$ (can see this by applying Jensen's inequality to the dual representation of the CVaR). Furthermore, it is important to note that the result from Kallus et al, 2023 in fact is claiming something different-- that
$ \text{CVaR}(Y_i(1) - Y_i(0)) \leq \text{CVaR}[ E[Y(1) \mid X] - E[Y(0) \mid X]]$. 
Unfortunately, this is a critical limitation of the paper, as the authors use the first inequality to argue that $\text{CVaR}(E[Y(\pi(X)) \mid X])$ is an upper bound on $\text{CVaR}[Y(\pi(X))]$ as motivation for minimizing for minimizing $\text{CVaR}(E[Y(\pi(X)) \mid X])$. It is not clear to what extent $\text{CVaR}(E[Y(\pi(X)) \mid X])$ is a useful policy objective.

In addition, the paper is closely related to similar papers in the robust policy learning literature that also focus on learning a policy that minimizes a CVaR objective. A major weakness of this paper is that it does not cite (or discuss the contribution) in light of these very relevant works:
Mo, W., Z. Qi, and Y. Liu (2021). Learning optimal distributionally robust individualized treatment rules. Journal of the American Statistical Association 116 (534),
659–674.
Qi, Z., J.-S. Pang, and Y. Liu (2022). On robustness of individualized decision rules. Journal of the American Statistical Association 0 (0), 1–15.
In particular, Qi et al, 2022

### Questions
N/A

### Soundness
1

### Presentation
3

### Contribution
2
