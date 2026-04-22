# Testing Most Influential Sets

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 2, 4, 6

## Abstract
Small influential data subsets can dramatically impact model conclusions, with a few data points overturning key findings. While recent work identifies these most influential sets, there is no formal way to tell when maximum influence is excessive rather than expected under natural random sampling variation. We address this gap by developing a principled framework for most influential sets. Focusing on linear least-squares, we derive a convenient exact influence formula and identify the extreme value distributions of maximal influence – the heavy-tailed Fréchet for constant-size sets and heavy-tailed data, and the well-behaved Gumbel for growing sets or light tails. This allows us to conduct rigorous hypothesis tests for excessive influence. We demonstrate through applications across economics, biology, and machine learning benchmarks, resolving contested findings and replacing ad-hoc heuristics with rigorous inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper develops a statistical framework to distinguish natural sampling variation from excessive influence in most influential subsets. The authors derive exact expressions for subset influence in univariate linear regression and establish that the maximal influence follows asymptotic extreme-value distributions (Fréchet or Gumbel). Building on this, they propose a hypothesis testing procedure based on MLE (to estimate the EVD parameters) and an adaptive greedy algorithm from prior work (to estimate maximum influence). Empirical studies on economics, biology, and classic ML datasets illustrate the method’s interpretability and robustness benefits.

### Strengths
- Neat theoretical framework, which elegantly links subset influence to extreme-value asymptotics

- Comprehensive empirical evaluations across diverse domains

### Weaknesses
- The theory focuses exclusively on 1D OLS, which is too restrictive. Based on my understanding, it is very difficult (if not impossible) to generalize Proposition 1 to high-dimensional OLS. All empirical studies also concentrate on tabular data and linear models, further limiting the paper’s applicability to modern ML settings.

- ML practitioners typically treat the training dataset as fixed rather than as a random sample from an underlying distribution. The authors should better motivate why testing statistical significance rather than simply detecting influence is meaningful in this context.

- The paper would benefit from clearer exposition of extreme-value theory and stronger intuition for the Gumbel vs. Fréchet distinction, which are currently introduced with limited background or explanation.

In summary, while the work’s statistical rigor is admirable, its theoretical contribution does not quite meet the bar for a top-tier ML conference, and its relevance to representation learning is limited. I encourage the authors to further develop and generalize the theory and consider submitting to a statistics or econometrics venue, where inferential rigor is the primary focus.

### Questions
I don't have further questions at this point, but would be willing to reconsider my rating if the authors can provide a clear and technically plausible pathway for extending their theoretical framework beyond the 1D OLS setting.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work introduces a rigorous hypothesis testing framework for influential sets on the coefficient in a univariate regression. It identifies the asymptotic distribution for the maximum influence of any such set for both fixed size and varying sized sets. These insights are then used to test whether the influence of a set is significant, meaning its influence is beyond that expected from natural variations in data. This framework is validated and applied in six case studies.

### Strengths
- The paper is generally well written, with clean and easy to follow figures.
- This work is original, presenting the first hypothesis testing framework for influential observations (to my knowledge).
- This work is well motivated. The problem of rigorous testing for influential observations is challenging, but has ramifications for a wide range of practical applications. This range is highlighted well by the variety of case studies in this paper. In particular, the question of how terrain ruggedness influences GDP is interesting. Broadly, the domains chosen for these case studies are well selected.
- The presented statistical theory is rigorous and validated through simulation studies.

### Weaknesses
- The presented framework only applies to univariate regression with positive coefficients. This restriction excludes many practical use cases, including simple extensions like multivariate regression. As such, the machine learning case studies presented in this work seem somewhat contrived, as machine learning practitioners tend to approach these datasets using multivariate regression.
- It seems like there is a risk of extreme multiple hypothesis testing problem here. For example, on the Boston housing dataset, six observations were selected as the most influential set. If these were selected after considering, say, the 1, 2, 3, 4, and 5 most suspicious observations, six hypothesis tests have already been performed (or at the very least, six effect sizes measured). In practice, it seems very tempting (and easy) to consider many potential influential sets, ultimately invalidating the conclusion or making the p-values large if a correction is applied.

### Questions
- In the equation before (1), should the first sum be over n \in [N] rather than not in S?
- It is stated that "we assume a univariate model with a positive coefficient". However, the machine learning case studies include cases with negative coefficients. Does this violate the theory, or am I missing something?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies maximally influential data subsets in linear regression and proposes a statistical testing framework to determine when identified influential sets reflect excessive rather than natural sampling variability. The authors derive asymptotic extreme value distributions (Fréchet vs. Gumbel) for the maximum influence statistic under different scaling regimes of subset size, provide computation formulas for exact influence, and apply the approach to simulated and real datasets across multiple domains.

### Strengths
1. This paper tackles an important question on understanding the influential data subsets on the statistical estimator.
2. The presentation is easy to understand, with theoretical contributions.

### Weaknesses
1. The motivation for p-values in influential subset testing is not clearly justified. The main distinction from Broderick et al., (2021) is that this paper added a statistical significance test, but the paper does not convincingly demonstrate why this hypothesis-testing perspective materially improves decision-making in practice. For example, all the real data applications in this paper can be similarly done wth Broderick et al. (2020). I would suggest that the authors provide why the proposed approach yields better conclusions by using these p-values compared to existing influence quantification approaches.

2. The proposed approach is limited only to linear regression, while Broderick et al. (2021) can be applied to all M-estimation problems. Extensions to GLMs or modern models are only speculated about, yet the paper argues broadly about interpretability and fairness across ML. Can this approach be applied to all M-estimation problems?

Reference: 
Broderick, T., Giordano, R., & Meager, R. (2020). An automatic finite-sample robustness metric: when can dropping a little data make a big difference?. arXiv preprint arXiv:2011.14999.

### Questions
1. See weaknesses 1 and 2
2. Would the hypothesis test disagree meaningfully with simpler leverage or leave-one-out metrics or Broderick et al. (2021)?
3. What are the real-world applications that this approach can benefit? I understand that people can use it to find the influential sets, but what should be the next steps?

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
3

### Summary
This paper examines the most influential sets problem for linear regression model from the hypothesis testing point of view, deriving the limiting distribution of the influence of most influential sets.

### Strengths
1. Novel theoretical results: I'm not aware of related literatures that address the limiting distribution of the maximum influence for linear regression model.
2. Conceptual clarity and motivation: The identified gap is indeed an important research question to address.

### Weaknesses
1. Practical guidance: Since the theory focuses on asymptotic behavior, it is unclear how many samples are needed to render the theory applicable.
2. Presentation: The clarity of the paper can be improved by carefully restructuring the sections. For instance, it seems like the presentation before Section 3.2 largely follows [1]. However, some parts are not necessarily used, for instance, the influence function. While I won't say this is to the extent of "plagiarism", however, I do think some careful adaptation to the current paper will be more appropriate.
3. Limited scope: This is my main concern. While I do think this paper studies an important research question, however, the scope and the result are limited. With the fact that the empirical experiments are also focusing on simple datasets and without significant empirical findings, I do not think I can suggest acceptance to the conference. This is only my opinion, and I'll leave the final judgment to the AC.


[1] Yuzheng Hu, Pingbang Hu, Han Zhao, and Jiaqi W. Ma. Most influential subset selection: Challenges, promises, and beyond.

### Questions
1. Typo: Line 118 states that $\epsilon=0$ recovers $\hat{\theta}$, which is think is not, as the left-hand side is summing over $n\notin \mathbb{S}$. Similarly, the claim for $\epsilon=-N^{-1}$ does not recover $\hat{\theta}_{-\mathbb{S}}$, but rather $\epsilon=0$.
2. Typo: Equation (1): It is *fine* to define influence as what it is, but it is far from correct to write $\hat{\theta}(\epsilon; \mathbb{S}) \approx \mathcal{I}(\mathbb{S})$. The correct interpretation is that, $\\hat{\\theta}\_{-\\emptyset} - \\hat{\\theta}\_{- \\mathbb{S}} \\approx \\epsilon \\mathcal{I} (\\mathbb{S})$.
3. Line 113 should reference the influence function paper by Koh & Liang, 2017.
4. To justify the contribution, I think it'll be beneficial to provide some critical (real-world/conceptual) examples that require hypothesis testing for the most influential sets problem.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the problem of assess the statistical significance of the most influential sets in OLS and proposes the first testing framework by deriving extreme-value limits for the maximum subset influence and turning them into calibrated p-values.

The proposed method is applied to several existing datasets in economics, biology, and machine learning benchmarks, demonstrating that some datasets indeed contain outliers with excessive influence while the influence of most influential subsets in some other datasets are not significantly excessive than natural sampling variation.

### Strengths
- The problem of testing most influential subsets is novel and well-motivated.
- The proposed framework is theoretically sound.
- The proposed approach may find broad applications in scientific domains that relies on OLS for data analysis.

### Weaknesses
- The proposed framework is limited to OLS.
- The block-maxima MLE used in the proposed approach suffers from a bias.
- The theory is only applicable to the low dim regime.

Minor: this work may be a better fit to statistics or economic venues than machine learning ones.

### Questions
How does a failure to identify the most influential subset impact the estimated p-value?

### Soundness
3

### Presentation
3

### Contribution
3
