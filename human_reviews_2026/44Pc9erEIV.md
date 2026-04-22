# Efficient Difference-in-Differences Estimation when Outcomes are Missing at Random

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
The Difference-in-Differences (DiD) method is a fundamental tool for causal inference, yet its application is often complicated by missing data. Although recent work has developed robust DiD estimators for complex settings like staggered treatment adoption, these methods typically assume complete data and fail to address the critical challenge of outcomes that are missing at random (MAR) -- a common problem that invalidates standard estimators. We develop a rigorous framework, rooted in semiparametric theory, for identifying and efficiently estimating the Average Treatment Effect on the Treated (ATT) when either pre- or post-treatment (or both) outcomes are missing at random. We first establish nonparametric identification of the ATT under two minimal sets of sufficient conditions. For each, we derive the semiparametric efficiency bound, which provides a formal benchmark for asymptotic optimality. We then propose novel estimators that are asymptotically efficient, achieving this theoretical bound. A key feature of our estimators is their multiple robustness, which ensures consistency even if some nuisance function models are misspecified. We validate the properties of our estimators and showcase their broad applicability through an extensive simulation study.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies two-group, two-period Difference-in-Differences when pretreatment outcomes are Missing-At-Random (MAR). It targets the Average Treatment Effect on the Treated. The authors give two identification strategies: (i) MAR given $(X,A)$ and (ii) MAR given 
$(X,A,Y)$. For each, they derive the observed-data efficient influence function and the semiparametric efficiency bound. They then construct cross-fitted estimators that achieve the bounds and show “multiple robustness”: consistency holds if certain sets of nuisance models are correct. A simulation study illustrates bias/RMSE patterns across correctly specified vs misspecified nuisances.

### Strengths
Originality
- Treats missing pre- or post- outcomes in the canonical $2 \times 2$ DiD with covariates, and does so in a semiparametric framework with explicit EIFs. Prior DiD papers often assume full outcomes or focus on staggered timing without MAR-aware efficiency analysis. Framing MAR both as post-treatment outcome-independent (given $X, A$ ) and post-treatment outcome-dependent (given $X, Y_1, A$ ) is a clear conceptual step.
- The "nested regression" component $\eta_0(x, 0)=\mathbb{E}\left[\mu_0\left(x, Y_1, 0\right) \mid X=x, A=0\right]$ and its stability analysis give a concrete recipe to learn the extra layer needed under outcome-dependent MAR. This is a neat link to modern DR-Learner style ideas.

Quality
- Identification statements are precise, and the EIFs are derived with clear bookkeeping of nuisance components $(\mu, \pi, \gamma, \eta)$. The efficiency-loss decompositions versus the fully-observed benchmark are informative.
- The cross-fitting scheme and high-level conditions for asymptotic normality align with current best practice and are stated cleanly.

Clarity
- The paper lays out the two MAR regimes, writes explicit plug-in equations for estimators, and provides a concise "multiple-robustness" table. This helps readers see exactly which nuisance pieces must be correct for consistency.
- The simulations mirror the theory with clear toggles between correctly specified and misspecified nuisances. 

Significance
- DiD with missing outcomes is common in applied work. A semi-parametrically efficient, multiply robust procedure is valuable for practice, especially when complete-case strategies are biased. The results can plug into modern ML pipelines through cross-fitting.

### Weaknesses
1. Positioning vs related work (novelty claims).

The paper cites DiD with staggered adoption and some recent missing-data DiD work, but the empirical reader will ask: how do your estimators compare to (a) complete-case DiD (DiD with missing items dropped), (b) simple inverse-probability weighting for $R_0$, (c) regression imputation baselines, and (d) standard DR DiD with fully observed outcomes?

So, could the authors add a benchmark section with these baselines, including plots of bias, RMSE, and coverage? It would be better include a brief analytic comparison to DR DiD under full data to highlight where efficiency is lost and recovered.

2. Inference and coverage.

The simulations emphasize bias and RMSE but not confidence interval coverage or interval length. Since the paper claims efficiency and gives EIFs, empirical coverage is central.

Could the authors report empirical coverage and average CI width across scenarios, including small-sample behavior and the impact of fold choice $J$? The authors can show whether the sandwich variance based on the estimated EIF is reliable.


3. Plausibility of MAR assumptions.

Outcome-dependent MAR given $Y_1$ is subtle: it conditions on a post-treatment variable. I was wondering which examples in the Introduction (job training programs in labor economics, prior test scores in education policy, and EHR in health research) would arguably satisfy this assumption.

4. Nested regression learning details.

The stability definition and "oracle" rates are high-level. Readers will want concrete guidance: what models to fit for $\mu_0\left(x, y_1, a\right)$, how to construct pseudo-outcomes with augmentation, and how to tune them.

Can the authors add an algorithm box for learning $\eta_0$ under both the regression and conditional-density approaches? They can also give default choices (e.g., gradient-boosted trees for $\mu$, logistic for $\gamma$, random-forest regression for $\eta$ ), with practical notes on hyperparameter tuning, cross-fitting folds, and clipping for $\hat{\gamma}$. 

5. External validity and real data.

The paper would benefit from at least one real application to show end-to-end feasibility and the size of efficiency gains in practice.

Can the authors add a compact empirical example (even in the appendix)? They can report the estimated ATT with the proposed method and the baselines, Cl widths, and a short discussion of assumption plausibility in that setting.

### Questions
1. On outcome-dependent MAR:

In practice, when would you recommend modeling $\operatorname{Pr}\left(R_0=1 \mid X, Y_1, A\right)$ rather than $\operatorname{Pr}\left(R_0=\right. 1 \mid X, A)$ ? Can you provide guidance or a rule-of-thumb diagnostic for deciding whether including $Y_1$ is likely to help or to add variance? Any empirical check that conditioning on $Y_1$ does not induce harmful instability?


2. Efficiency gains in practice:

Can you report the magnitude of efficiency gains (variance reduction) relative to simple IPW or regression imputation in your simulations and, ideally, one real dataset? A table with bound estimates, plug-in variances, and empirical variances would help.


3. Coverage under misspecification:

When "multiple robustness" conditions fail, how badly do Cls mis-cover? Please provide coverage heatmaps across nuisance-misspecification cells to show the method's breaking points.


4. Learning $\eta_0$ :

For the augmented pseudo-outcome strategy, do you recommend trimming on $\hat{\gamma}$? If so, what thresholds and what is the implied target estimand after trimming? How sensitive are results to this choice?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors tackle a very common but under-studied problem: doing DiD when some pre- or post-treatment outcomes are missing at random (MAR). They give (i) nonparametric identification of the ATT under two MAR regimes, (ii) the semiparametric efficiency bounds, and (iii) efficient, multiply-robust estimators that attain those bounds with cross-fitting. One estimator applies when missingness is MAR given covariates and treatment; the other allows missingness that also depends on the other period’s outcome (handled via a “nested regression”).

### Strengths
Advantage: the paper is theoretically grounded, with very rigorous statements for assumptions and theoretical results. Based on my experience with semiparametric theory and double ML, the results look correct to me (such as identification, multiple robustness, and construction of confidence intervals). Combining the DR-learner theory to build the nested regression is also pretty interesting, which might motivate a series of other problems, say surrogate variable approaches. Overall, I appreciate the authors's theoretical contribution in this problem.

### Weaknesses
Weakness: 
1. the paper presents two sets of assumptions for MAR, one involving Y1 and the other not. These two set of assumptions are not overlapping in general, so in practice how to decide which regime that the real data is in and which estimator to use? Practically this will be confusing. Moreover, assumption 2.4 feels hard to interpret too, and the authors did not provide enough explanation around it. How could controling for some future outcome makes past outcome and its missness status independent? Similar question exists when the authors try to generalize the results to allow both pre- and post-treatment outcome to be missing. How to interpret an outcome dependent missingness assumption in such cases? 

2. the paper did not provide any real-world experiments, thus making it hard to judge the real motivation and infer under what scenario the assumptions would hold, related to my point 1. I would appreciate some real-world connections of the assumptions and methods.

3. the paper also lacks a proposal of parallel trend checking in the presence of missingness. In practice, how should people check whether parallel trend holds under the two assumption regimes? 

4. the proposed methods involve many nuisance parameters, especially under Assumption 2.4, which might render the final estimator very unstable in finite sample. For example, based on Figure 1, the left top panel shows when outcome model is misspecified but propensity and missing probability are correct, the estimator gives some non-negligible bias, while in theory there shouldn't be as ensured by the double robustness property. Moreover, such stability issue will propagate to variance estimation, yet I did not show any results that demonstrates how well inference works. So I believe the evaluation of the methods is very limited.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper develops a semiparametric framework for efficient Difference-in-Differences (DiD) estimation when outcomes are missing at random (MAR)—a key challenge that undermines standard estimators. It establishes nonparametric identification of the Average Treatment Effect on the Treated (ATT) under two minimal MAR conditions and derives the corresponding semiparametric efficiency bounds. Building on these results, the authors propose asymptotically efficient estimators that achieve the efficiency bound and possess a multiple-robustness property, remaining consistent even with misspecified nuisance models. Extensive simulation studies confirm their theoretical properties and demonstrate the framework’s broad applicability to empirical research with incomplete panel data.

### Strengths
1.  This paper is among the first to provide a rigorous, efficient, and robust framework for MAR in DiD.
2. The proposed method is supported by rigorous theoretical analysis.
3. Experimental results demonstrate that the proposed approach.

### Weaknesses
1. The paper does not compare its estimators with existing approaches, making it unclear how much improvement it offers in efficiency or bias reduction.
2. The study relies entirely on Monte Carlo simulations without testing the method on real-world datasets, leaving its practical utility and challenges in empirical applications unexamined.
3. The paper focuses exclusively on outcomes missing at random (MAR) and does not consider missing not at random (MNAR), which can occur in practice.

### Questions
1. For applied researchers, what practical advice would you give for estimating the nested regression? For example, do linear models typically satisfy the stability condition (Definition 3.2), or are non-linear models (e.g., gradient-boosted trees) necessary?
2. Although the paper derives asymptotic normality for the proposed estimators, it does not provide simulation results illustrating practical inference.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper examines the application of the Difference-in-Differences (DiD) when the outcomes are missing at random. Applying arguments in semiparametric theory,  the paper provide an efficient estimator. Simulation studies are conducted to validate the estimator's performance.

### Strengths
The authors' arguments are grounded in established theoretical results within the semiparametric literature. This might be regarded as a strength, while the arguments are now relatively conventional and cannot be recognized as a novel scholarly contribution.

### Weaknesses
1. 
If they argue it as a "common problem", the authors should genuinely justify the "missing at random" assumption, given that numerous instances of missing observations do not inherently conform to randomness. 
While some researchers might contend that a sufficient set of conditional variables can validate this assumption, such an argument remains deeply problematic. Indeed, the assertion that these variables are adequate undermines the fundamental rationale for experimental data, a premise that empirical research consistently refutes.



2. 
If the argument is substantially simplified, the paper essentially employs standard semiparametric-efficiency argument twice: one addressing missing potential outcomes within conventional causal inference, and the second addressing missing observations.
Consequently, the theoretical contributions might be regarded as marginal at best. 

3. 
Assumption A 2.1 (a) requires the statistical independence for parallel conditional trends. While I noticed that the authors describe it just for the sake of notational consistency, the explanation is unclear. The authors should assume that the standard assumption if it is enough and I cannot understand "notational consistency" here.

### Questions
Q1. It would be desirable for the paper to demonstrate the justifiability of its assumptions through compelling real-life examples.



Q2. To enhance the method's realism, additional conditioning variables are necessary in simulation. Moreover, the paper's limitation to the most rudimentary two-period, two-group scenario is problematic, as such a simplistic framework rarely reflects real-world applications. While the authors may argue that extensions are theoretically possible, this assertion is not necessarily valid, particularly given the  missing-at-random mechanism.

### Soundness
2

### Presentation
2

### Contribution
1
