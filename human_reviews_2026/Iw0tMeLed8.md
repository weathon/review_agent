# Revisiting Active Sequential Prediction-Powered Mean Estimation

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
In this work, we revisit the problem of active sequential prediction-powered mean estimation, where at each round one must decide the query probability of the ground-truth label upon observing the covariates of a sample. Furthermore, if the label is not queried, the prediction from a machine learning model is used instead. Prior work proposed an elegant scheme that determines the query probability by combining an uncertainty-based suggestion with a constant probability that encodes a soft constraint on the query probability.  We explored different values of the mixing parameter and observed an intriguing empirical pattern: the smallest confidence width tends to occur when the weight on the constant probability is close to one, thereby reducing the influence of the uncertainty-based component. Motivated by this observation, we develop a non-asymptotic analysis of the estimator and establish a data-dependent bound on its confidence interval. Our analysis further suggests that when a no-regret learning approach is used to determine the query probability and control this bound, the query probability converges to the constraint of the max value of the query probability when it is chosen obliviously to the current covariates. We also conduct simulations that corroborate these theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper revisits the problem of active sequential prediction-powered mean estimation, where a learner estimates the mean of an unknown label distribution under a limited labeling budget by adaptively deciding whether to query true labels or rely on model predictions. Building on prior work (Zrnic & Candès, 2024), the authors make an intriguing empirical observation that confidence intervals are often narrowest when the query policy is nearly uniform, ignoring model uncertainty. Motivated by this, they develop a non-asymptotic analysis providing time-uniform, data-dependent confidence bounds for the estimator and reformulate the label-querying process as an online convex optimization problem solved via the Follow-the-Regularized-Leader (FTRL) algorithm. Their theoretical results show that the FTRL policy converges to the maximal allowable query rate and that a uniform querying strategy is asymptotically optimal. Experiments across real and synthetic datasets confirm that this simple strategy matches or outperforms uncertainty-based methods in confidence width and coverage, revealing that uncertainty-aware sampling may offer limited benefits for sequential mean estimation.

### Strengths
The paper is original in revisiting active sequential mean estimation through a new theoretical lens, uncovering an unexpected empirical pattern that challenges prior assumptions about the value of model uncertainty. Its non-asymptotic analysis fills a key theoretical gap left by earlier asymptotic-only results, providing rigorous, time-uniform, and data-dependent confidence bounds that enhance understanding of estimator behavior. The introduction of a no-regret online learning formulation using FTRL to govern the query policy is conceptually elegant and technically sound, connecting active inference with online convex optimization theory. The experimental validation is thorough and consistent with the theoretical findings, strengthening the overall credibility of the claims. The paper is also clearly written and well-organized, effectively bridging empirical observation, theoretical reasoning, and practical implications—making it a significant and high-quality contribution to the fields of active statistical inference and prediction-powered estimation.

### Weaknesses
While the paper is theoretically well-grounded, several aspects could be improved to strengthen its impact. First, the novelty is somewhat incremental, as the core setup largely extends the framework of Zrnic & Candès (2024) by introducing a non-asymptotic analysis and reformulating the query policy using standard online learning tools (FTRL). The contribution could be made more compelling by deeper theoretical insight into why uncertainty-based sampling fails in practice, beyond the observed convergence of the no-regret policy. Second, the experimental scope is limited, the datasets used (text politeness, wine review, post-election survey, and one synthetic set) are small-scale and do not test generalization to high-dimensional or noisy real-world domains where uncertainty estimation might matter more. Adding experiments in such settings or sensitivity analyses for different noise structures would better validate the theoretical claims. Third, the discussion of assumptions and limitations is brief; for example, the oracle approximation and boundedness assumptions may not hold in practical scenarios, and their empirical impact is not fully analyzed. Finally, connections to related frameworks such as active learning or adaptive importance sampling[1][2] could be discussed more thoroughly to position the work within the broader literature. Addressing these points would make the contribution more robust and its insights more broadly applicable.

[1] Pareto smoothed importance sampling, jmlr 2025

[2] Active Advantage-Aligned Online Reinforcement Learning with Offline Data, arxiv 2025

### Questions
see weakness

### Soundness
3

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
3

### Summary
The paper revisits the problem of active sequential prediction-powered mean estimation — estimating the mean of a label $y$
$y$ when each observation’s label may or may not be queried. If the label is not queried (to save cost), a machine-learning model’s prediction is used instead and there is finite budget $T_b \ll T$ of queries allowed. Here are the follewing finding.

1) Based on Freedman's theorem (a version of Bernstein concentration inequality) the author provide non-asymptotic confidence intervals
and data-dependent confidence bound valid at any step.

2) Reinterpret query-probability selection as a no-regret online learning problem (via the Follow-the-Regularized-Leader algorithm).

3) The discovery  that under sequential settings, uniform querying (without conditioning on covariates) may be both theoretically justified and empirically optimal.

### Strengths
1. Rigorous theoretical analysis based on martingale difference concentration inequality.

2. New connection with no-regret online learning problem which suggest practical algorithm

3. Clear empirical insights that uniform querying performs comparably (or better) than uncertainty-based querying which may simulate further work.  Since this optimal strategy is also very simple and easy to implement this is a significant finding.

4. Very well written paper combing theory, algorithmic development and experimental findings.

Overall the paper looks very strong to the reviewer (who is not an expert in this field, especially online learning).

### Weaknesses
No obvious one.

### Questions
Have you considered extending your framework to other risk functionals (e.g., variance or quantile estimation) beyond the mean, and would the same uniform-querying phenomenon persist?

### Soundness
4

### Presentation
3

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
This paper revisits the active statistical inference methods proposed by Zrnic & Candes (2024)--where that previous paper presents methods for prediction powered inference (Angelopoulos et al 2023a) under active data selection--and the current paper claims the following contributions: (1) Empirically exploring the effect of different values of the mixture weight (ie, a mixture weight on the active data selection policy) in Zrnic & Candes (2024) and finding that the sharpest intervals tended to occur with more weight on the constant probability; (2) they present non-asymptotic analysis of the mean estimator with a data-dependent bound; (3) analysis on convergence when a no-regret learning approach is used; (4) simulation experiments to support the theoretical results.

### Strengths
This paper studies an important problem of how to do valid statistical inference under active data collection, building on prior work of active statistical inference (Zrnic & Candes 2024). The observations about the role of the mixture parameter, the non-asymptotic analysis, and the analysis on convergence seem like they would be of interest to the community. The writing is overall clear and the analysis seems sound. Overall I enjoyed reading the paper.

### Weaknesses
**Adding acknowledgement & discussion of prior non-asymptotic version of Zrnic & Candes (2024):** While the paper does have an overall fairly thorough discussion of related work and its relationship to the prior work of Zrnic & Candes (2024), there is one key aspect of its discussion that seems like it needs to be updated, to acknowledge that Zrnic & Candes (2024) do have some consideration of non-asymptotic versions of their method, which is the main claimed contribution of the current paper. That is, at the top of Page 2 the current paper states “The non-asymptotic analysis of the estimator was missing in the prior work [Zrnic & Candes (2024)]”; however, Appendix C in Zrnic & Candes (2024) is titled “Non-asymptotic results” and it both explains how its main results could extend directly to non-asymptotic analogs, as well as presents some experimental results of non-asymptotic variants. I understand that Zrnic & Candes (2024) do not comprehensively or explicitly write out the formal analysis for the non-asymptotic variants, but given that this is the core claimed contribution of the current paper, I think that the current paper needs to be revised to acknowledge this prior consideration.

### Questions
Can the authors explain how their non-asymptotic analysis and results relate to Appendix C in Zrnic & Candes (2024)? I think that such discussion should also be added to the paper.

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
2

### Summary
This paper revisits the active inference procedure proposed by Zrnic & Candes, 2024 [ZC24], where the goal is to better estimate the mean of a target variable with a query policy that is a mixing of uniform sampling and uncertainty-based sampling. This investigation is motivated by the empirical observation that uniform sampling yields tighter or comparable confidence intervals to uncertainty-based active sampling. A non-asymptotic upper bound on the L1 error of the active sequential prediction-powered mean estimation is provided, showing a rate of $O(1/\sqrt{t})$ at large sample sizes $t$. Then the paper demonstrates that a query policy independent of the current covariate and designed in a non-regret fashion converges to uniform sampling. This policy leads to similar confidence interval widths to the method proposed by Zrnic & Candes, 2024, in several real data experiments, where they both perform significantly better than uniform sampling.

### Strengths
- The analysis is motivated by an intriguing empirical observation that the active inference method [ZC24] performs comparably to uniform sampling in terms of the confidence interval width.

- It is an interesting conclusion that a query polity independent of the current covariate can produce similar confidence intervals to the active inference method [ZC24].

### Weaknesses
- There seem to be some contradictions in the empirical observations (See Questions).

- The motivation for looking at the non-regret approach is not sufficiently clear.

### Questions
- Does the method [ZC24] with mixing $\lambda=1$ correspond to the uniform sampling baseline in (6)? If so, why does the purple curve in Figure 1 differ significantly from the green one in Figure 4, while they both illustrate the results of uniform sampling on the post-election survey data set. Moreover, according to Figure 1, the method [ZC24] with a mixing policy of uniform and uncertainty-based samplings gives wider or comparable confidence intervals w.r.t. the uniform sampling baseline, which is the motivation of this article. However in Figures 2-4, the method [ZC24] yields consistently narrower confidence intervals than uniform sampling. Isn't there a contradiction?

- How are the hyperparameters $\gamma,\tau,\beta$ chosen for the non-regret method FTRL in the experiments of Section 6? Same question for the mixing parameter $\lambda$ for the method [ZC24].

### Soundness
2

### Presentation
2

### Contribution
3
