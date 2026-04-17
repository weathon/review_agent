# When Aggregation Fails: From PAC-Bayes Theory to Practical Selection for Conformal Prediction

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
We identify and characterize a fundamental incompatibility between PAC-Bayes theory and conformal prediction: while PAC-Bayes minimizes average risk through posterior aggregation, conformal prediction's efficiency depends on quantile behavior. We prove that this \emph{average-quantile divergence} phenomenon causes standard PAC-Bayes aggregation to systematically select suboptimal models for conformal prediction, with linear aggregation methods unable to preserve quantile optimality and efficiency losses proportional to both posterior entropy and score heterogeneity. To address this limitation, we develop PAC-Bayes Informed Selection (PBIS), which uses quantile-aware posteriors for model selection rather than aggregation. We establish PAC-Bayes bounds for quantile functionals requiring novel techniques to handle their non-differentiable nature, and prove that PBIS achieves selection consistency with $O(\sqrt{T \log |\Theta|})$ regret in online settings. Empirical validation across 27 datasets demonstrates that PBIS achieves the narrowest prediction intervals among nine conformal methods while maintaining valid coverage, with 7.3\% average improvement in high-divergence scenarios versus 2.1\% in low-divergence ones compared to standard PAC-Bayes aggregation. The method maintains computational efficiency comparable to split conformal while being 82$\times$ faster than CQR. In online settings with distribution shifts, PBIS uniquely maintains valid coverage across gradual, sudden, and recurring shifts where competing adaptive methods fail. Our theoretical and empirical results establish that selection-based approaches fundamentally outperform aggregation for conformal prediction by avoiding the mathematical incompatibility between average risk and quantile optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The PAC Bayes framework for model selection is based on empirical risk minimization over a hypothesis class with a prior distribution placed over the parameterized hypothesis class. While such model selection is natural in some settings, this paper argues that it is at odds with the goal of optimal model selection with respect to maximal efficiency after conformalization. This is because PAC Bayes is predicated on average model performance, whereas conformalized model behavior is dictated by tail behavior. Towards this end, the paper presents a characterization of the circumstances in which such a behavior arises and demonstrates how the common strategy of model aggregation can give rise to precisely this scenario. The authors, in turn, suggest performing model selection in place of aggregation, specifically selecting the model with optimal efficiency, encouraging exploration with a tunable parameter $\lambda$. They then demonstrate several results of this proposed framework in experiments, demonstrating the coverage gap convergence, the improved predictive efficiency over alternate aggregation strategies, and validation of the claimed source of predictive inefficiency arising from naive aggregation.

### Strengths
The paper is very clearly presented, with each step of the proposed methodology both well-motivated and well-described. The paper makes claims and theoretically justifies each one, regarding the source of predictive inefficiencies and a clear proposal on how to address this issue. The method is novel and thoroughly justified in its theoretical analyses. The proofs also seem mostly sound, although there appear to be some (minor) bugs I came across in some of them (presented in the weaknesses section below). The empirical validation is also very thoroughly done, with a comprehensive set of benchmarks tested and a good collection of baselines that were compared against. Each claim is also separately justified in the empirical studies, outside of just the predictive efficiency claim, including the coverage convergence analysis and validation of the source of the aggregation inefficiency.

### Weaknesses
While the storyline and presentation is well done, there are a couple of technical bugs I seem to have come across while going through the paper. These seem fairly minor to the overall flow of the paper, but they, I believe, do require correction.

**Theorem 2**: I am unsure what this “entropy-variance inequality” refers to. It appears that the inequality as framed as 

$$ Var_{w}(q_i) \ge H(w) Var(\mathcal{Q}) \delta_{\min} $$

Is not true, which we can see in the following counterexample:

$$ (q_1, q_2) = (0, 1) $$
$$ (w_1, w_2) = (1-\epsilon, \epsilon) $$
$$ \overline{q} = (1-\epsilon) (0) + (\epsilon) (1) =\epsilon$$

$$ Var_w(q_i) = \sum_i w_i (q_i - \overline{q})^2$$
$$ = (1-\epsilon) (\epsilon)^2 + \epsilon (1-\epsilon)^2$$
$$ = \epsilon[ (\epsilon-\epsilon^2) + (1-2\epsilon + \epsilon^2) ]
= \epsilon(1-\epsilon)$$

$$ Var(\mathcal{Q}) = Var({0, 1}) = (1-1/2)^2 = 1/4$$

$$ \delta_{\min} = 1$$

$$ H(w) = -(\epsilon \log(\epsilon) + (1-\epsilon) \log(1-\epsilon))$$

For $\epsilon$ sufficiently small, we have that $\epsilon(1-\epsilon)\approx \epsilon$ and $1-\epsilon\approx 1$, meaning $(1-\epsilon) \log(1-\epsilon) \approx 0$. Thus, $H(w) \approx -\epsilon \log(\epsilon)$. This means, for small $\epsilon, we have

$$ H(w) Var(w) \delta_{\min} \approx -\epsilon \log(\epsilon)/4  $$

Thus, the claim of this statement is that

$$ \epsilon \ge\epsilon(-\log(\epsilon)/4) $$

Which is clearly untrue for any $\epsilon$ such that $-\log(\epsilon)/4 > 1$, i.e. for any $\epsilon < 10^{-4}$.

**Theorem 7**: In part b, the events $V_i < K/2$ sums over the events $Y_i\in C_k$. However, these $C_k$ will clearly be dependent on one another, since they all come from the same underlying dataset (training on slightly different folds). So, while the Chebyshev bound seems valid, I do not see how the bound relying on Hoeffding follows, as Hoeffding requires independence of the summed variables.

**Theorem 6**: (More of a nitpick than an actual error) The expression $| \widehat{Q} _{1-\alpha}(s _{\theta}) - Q _{1-\alpha}[s _{\theta}]| \le \frac{\tau}{2}\, Q _{1-\alpha}[s _{\theta^*}]$ is claimed to be “required”; however, this is actually just a sufficient condition for the proof and is not “required.”

### Questions
1. I believe this is implicitly handled by the proofs, but how is the issue of multiple testing handled to ensure coverage validity for the model selection from Phase 1 of PBIS?

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies a fundamental mismatch between PAC-Bayes aggregation and conformal prediction.It formalizes an average–quantile divergence and shows standard PAC-Bayes can favor quantile-suboptimal models. To address the problem, the authors propose PBIS: a quantile-aware PAC-Bayes posterior for model selection, with theoretical guarantees and an online variant. On 27 datasets and under distribution shifts, PBIS maintains valid coverage and runs fast, with largest gains in high-divergence settings.

### Strengths
1. Clear identification and formalization of the “average–quantile divergence” between PAC-Bayes aggregation and conformal prediction. Theorems 1–3 articulate why linear aggregation and standard PAC-Bayes objectives are misaligned.
 2. The authors demonstrate the practical utility of their methods with an extensive series of experiments.

### Weaknesses
1. The results in table 2 show that PBIS yields no significant improvement over traditional PAC-Bayes-CP. Besides, Coverage should be as high as possible; 0.898 should not be bolded.

 2. The experiments in the online adaptive Performence part lack comparisons with more recent baselines, such as [1] and [2]. The validiy of PBIS in this scenerio requires further consideration. See questions.

 [1]: Xu C, Xie Y. Sequential predictive conformal inference for time series[C]//International Conference on Machine Learning. PMLR, 2023

 [2]: Wu J, Hu D, Bao Y, et al. Error-quantified Conformal Inference for Time Series[C]//The Thirteenth International Conference on Learning Representations.

### Questions
1. In online settings with distribution shifts, the magnitude of scores and $(1-\alpha)$-quantile of scores seem less informative because the coverage of CP is not guaranteed. Instead, a more appropriate way to consider selection and aggregation may be to leverage quantile loss and replace the empirical risk in line 169 with sum of quantile loss. How do the authors comment about this? The discussions with the aggregation method in [1] and [2] should be included.    

[1]: Gibbs I, Candès E J. Conformal inference for online prediction with arbitrary distribution shifts[J]. Journal of Machine Learning Research, 2024.

[2]: Hajihashemi E, Shen Y. Multi-model ensemble conformal prediction in dynamic environments[J]. Advances in Neural Information Processing Systems, 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates a fundamental incompatibility between PAC-Bayes aggregation and conformal prediction. The authors identify the average–quantile divergence phenomenon—a mismatch between average-risk minimization (PAC-Bayes) and quantile-based efficiency (conformal prediction). They prove that any linear aggregation method fails to preserve quantile optimality and propose PAC-Bayes Informed Selection, a quantile-aware selection framework. PBIS achieves theoretical guarantees such as selection consistency and PAC-Bayes bounds for quantile functionals.

### Strengths
- Theorems are rigorously stated, covering impossibility results, new PAC-Bayes bounds for quantile functionals, and finite-sample guarantees.

- PBIS provides a simple yet useful modification.

- The experiments are extensive.

### Weaknesses
- The literature review is rather limited and could be expanded to better situate the paper within existing work.
- The presentation is at times difficult to follow, and additional intuition or explanations—particularly around the main theorems—would greatly enhance readability.
- The average–quantile mismatch is somewhat expected, as quantiles and expectations inherently capture different aspects of a distribution.
- The theoretical analysis assumes a finite and discrete model space; it would be valuable to discuss how the results extend or behave when $|\Theta|$is infinite.

### Questions
Could PBIS be extended to continuous posterior distributions or infinite hypothesis spaces?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper highlights a mismatch between PAC-Bayes aggregation and conformal prediction. The authors prove that linear aggregation can be quantile-inefficient, and motivates a solution, PAC-Bayes Informed Selection (PBIS), that uses a quantile-aware posterior to select a single model for conformal calibration instead of averaging. The authors derive finite-sample PAC-Bayes bounds for quantile functionals and give online selection guarantees, then show across many datasets (and under distribution shift) that PBIS attains valid coverage with narrower intervals and competitive runtime relative to standard baselines.

### Strengths
First of all, I think the authors the authors did a good job in the actual writing (and also formatting) of the paper and the corresponding supplementary material. Unfortunately this is not always the case these days, therefore already very good to see. 

Apart from this (while I am not entirely sure about its motivation, see later section of my review), the authors do indeed very nicely formalize, what they call the average–quantile divergence, which explains why aggregation can fail and motivates a quantile-aware alternative. I have to admit, I am not convinced about each of the steps in the proofs, but see my later comments, happy to get corrected, or confirmed.

An other substantial strength might be the following (at least my interpretation): The results give practitioners a principled reason to prefer selection over aggregation when model scores are heterogeneous, and the broad empirical sweep (e.g. distribution-shift scenarios) suggests real impact on how ensembles are built for calibrated uncertainty.

### Weaknesses
Frankly speaking, I have a hard time understanding the motivation of the paper itself. I am very happy to further discuss this with the authors, and change my mind. In particular, I have the following dilemma. Why should (PAC)-Bayes care about conformal prediction and vice versa? It would be great, if the authors could provide some (more) literature which works in the intersection of Bayesian and frequentist inference. 

Apart from this, let me elaborate on some (technical) things that caught my eye. As a disclaimer: I am by no means expert when it comes to PAC-Bayes, while I am quite confident about conformal prediction and its theoretical foundations; so please elaborate if there is at any point a misunderstanding from my side.  

Some comments for the theoretical parts (in particular proofs in the supplementary material):

>In the proof of Theorem 2, the CDF of the aggregated score is first defined for the sum $\bar{s} = \sum_i w_i s_{\theta_i}$ via $F_{\bar{s}}(t) = P(\sum_i w_i s_{\theta_i}\le t)$ and then, a few lines later, the argument switches to properties of mixtures $\bar{F} = \sum_i w_i F_i$. Those are not the same operation. The CDF of a weighted sum is a convolution, not a convex combination of the marginal CDFs. The subsequent Taylor argument is then carried out on $\bar{F}=\sum w_i F_i$, i.e., on the mixture, not on the sum originally defined. Why is then the chain from (19) to (21) – (26) still valid? (this argument is also used in the Proof of Theorem 5).

>The supplement claims "by Jensen’s inequality for quantiles (which are convex functionals in the Wasserstein metric)" and concludes
$Q_{1-\alpha}[s_\rho] \leq E_\theta[Q_{1-\alpha}[s_\theta]]$. The random object is $s_\rho = E_\theta[s_\theta]$, not a Wasserstein barycenter of distributions, hence why should convexity in Wasserstein apply to this averaging operation? 

>The proof expands $F_i(\bar q)$ around $q_i$ and then aggregates the linear terms to conclude an excess quantile $\approx Var_w(q_i)/(2\bar f(\bar q))$. This assumes differentiability and positive density at the quantile for all $I$, and that the object being expanded matches the earlier definition. Neither the smoothness, positivity assumptions nor independence or error control terms are stated, and the target remains the mixture expression. 

>The proof ends with $Var_w (q_i) \geq H(w) Var(Q) \delta_{min}$, but the precise definition of $H(w)$ or $\delta_{min}$ is not given here, and the relation mixes a variance of quantiles with a variance of an unspecified distribution $\mathcal{Q}$ over models/parameters, producing a dimensionally unclear bound that drives the claim.

Further, the authors claim that PBIS satisfies DKW while PAC-Bayes marginally violates it and attribute this to the Rademacher complexity introduced by convex combinations. DKW is a distribution-free, single-sample inequality for the empirical CDF of i.i.d. draws. In particular, it does not depend on whether a predictor is an average or a selected model, and comparing a standard deviation of an error to a DKW upper bound on the sup deviation feels like apples-to-oranges. Thus, in my opinion, the violation reported is therefore not evidence against DKW, it only reflects a mismatch of quantities.

Right now, I have put my score as marginally below acceptance, but I am happy to adjust, since I think the paper and its supplementary material is actually quite nicely written (modulo the things I mention), and the authors obviously have spend some thought on the topic.

### Questions
I implicitly formulated some questions in earlier parts of the review, but I will list more questions that I had while reading the paper.

Are PBIS decisions invariant to monotone re-scalings of the nonconformity score?

In the streaming scenarios, how often do you update the posterior versus the calibration threshold, and do you recommend recalibrating every update or only when a drift test triggers?

### Soundness
2

### Presentation
3

### Contribution
2
