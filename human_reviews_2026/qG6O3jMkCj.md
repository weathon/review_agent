# SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 4

## Abstract
Estimating heterogeneous treatment effects (HTEs) from right-censored survival data is critical in high-stakes applications such as precision medicine and individualized policy-making. Yet, the survival analysis setting poses unique challenges for HTE estimation due to censoring, unobserved counterfactuals, and complex identification assumptions. Despite recent advances, from causal survival forests to survival meta-learners and outcome imputation approaches, evaluation practices remain fragmented and inconsistent. We introduce SurvHTE‐Bench, the first comprehensive benchmark for HTE estimation with censored outcomes. The benchmark spans (i) a modular suite of synthetic datasets with known ground truth, systematically varying causal assumptions and survival dynamics, (ii) semi-synthetic datasets that pair real-world covariates with simulated treatments and outcomes, and (iii) real-world datasets from a twin study (with known ground truth) and from an HIV clinical trial. Across synthetic, semi-synthetic, and real-world settings, we provide the first rigorous comparison of survival HTE methods under diverse conditions and realistic assumption violations. SurvHTE‐Bench establishes a foundation for fair, reproducible, and extensible evaluation of causal survival methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper is a benchmark comparing many methods for time-dependent conditional average treatment estimation, which is key in survival analysis.

### Strengths
This paper tackles the important evaluation of the plural methods existing for HTE with survival outcomes. The benchmark evaluates methods under a grid of relevant synthetic scenarios in a structured way, and on real-world datasets.

### Weaknesses
While the paper presents a valuable and ambitious benchmarking effort, I found it somewhat difficult to read due to repetition, particularly concerning the repeated description of the three families of survival HTE estimators. Streamlining these sections could make the paper more concise and readable.

A major omission in the related work is the lack of reference to existing benchmarks for survival ATE, such as [1]. This recent contribution on causal inference for survival outcomes and meta-learning estimators is highly relevant to the present study. Including and situating the current benchmark within that context would strengthen the paper’s positioning.

In Table 1, the RCT/Observational distinction is presented as an assumption being held or violated, whereas it is in fact a study design choice rather than an assumption. I would recommend removing that column or clearly separating design factors from causal assumptions.

Regarding the simulation scenarios, the rationale for comparing RCTs with 50% versus 5% treated participants is not entirely clear. Since randomization guarantees ignorability regardless of treatment imbalance, it is not obvious what this variation adds conceptually beyond a variance stress test. Similarly, it is surprising that there is no RCT scenario with non-ignorable censoring, given that informative censoring can occur even in randomized studies.

There is also a typo at line 227, where “2,500 points” is reported twice.

Conceptually, the description of informative censoring is somewhat unclear. The paper defines A4 as conditional independence between censoring and event time given covariates and treatment, but it remains ambiguous how this assumption is violated across scenarios (e.g., whether censoring depends on T directly, on unobserved confounders, or both). In the causal configurations labeled A–D, several aspects appear to change simultaneously, making it difficult to isolate the effect of each violation. Similarly, in Figure 2, it is confusing that the paper refers to “Scenario C” while ignorable and non-ignorable censoring seem to coexist.

Figure 1 is not very easy to read; in addition to being small, it may not fully respect the ICLR template in terms of spacing between the plot and caption. Moreover, I am not convinced that evaluating estimators’ average performance across all scenarios is particularly informative. It would be much more insightful to emphasize “which method works best under which conditions.” In particular, comparing IPCW-based estimators outside of the informative censoring (InfC) settings does not seem meaningful, since those estimators are specifically designed for that violation.

From a results standpoint, Figure 2 shows relatively small differences in RMSE across estimators, suggesting that performance gaps may not be statistically significant—perhaps because ten repetitions are insufficient for stable estimates.

Overall, I find that a great deal of work was put into this paper, but more attention is needed to the presentation and analysis of results, so that the article can better answer questions such as “which method works best for a given scenario?” Knowing which method performs best on average is less informative, since some assumptions are testable and practitioners often operate under known or partially known conditions.

Finally, while the benchmark is well implemented, it remains somewhat descriptive. The paper would benefit from a stronger analytical component—focusing less on global rankings and more on interpretable insights about which methods are most reliable under specific types of assumption violations. This would make the contribution more actionable for practitioners.

Also, I do not really see a large difference in RMSEs in Figure 2 across estimators—perhaps ten repeats is not sufficient for clear differentiation.

Finally, I am willing to increase my grade if the authors answer my concerns, especially concerning the "when does each method works best" issue.

[1] Voinot, Charlotte, et al. “Causal survival analysis, Estimation of the Average Treatment Effect (ATE): Practical Recommendations.” arXiv preprint arXiv:2501.05836 (2025).

### Questions
Why is there no RCT scenario with non-ignorable (informative) censoring? Since informative censoring can occur even in randomized studies, such a configuration would seem important for completeness.

Would it be possible to present the results more clearly by scenario, explicitly showing which methods perform best under each assumption violation (e.g., lack of ignorability, positivity, or informative censoring)? This would make the benchmark’s insights much easier to interpret.

What is the motivation for including IPCW-based estimators in settings where censoring is already ignorable? Their inclusion in those cases seems less informative, as IPCW adjustments are designed specifically for informative censoring scenarios.

How do you interpret the strong performance of DeepSurv-based meta-learners compared to Double-ML in high-censoring settings?

Could you explain why Double-ML performs relatively poorly on the Twins dataset, despite ranking highly in synthetic experiments? Does this reflect differences in sample size, feature distribution, or unmeasured confounding?

Could you include more references to existing papers that explicitly state which methods they use for survival CATE estimation? This would help situate your benchmark within the current methodological landscape.

Would it be possible to provide clearer results by scenario, highlighting which methods work best under which assumption violations?

In Figure 1, DeepSurv-based methods achieve overall strongest RMSE performance across many scenarios, likely due to their overparameterization and resulting robustness to model misspecification. How do these methods perform in simpler, well-specified settings such as the Cox proportional hazards scenario? Do they still outperform other approaches when the underlying model assumptions hold?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a comprehensive benchmark designed to evaluate methods estimating heterogeneous treatment effects (HTEs) from (right-censored) survival data. Estimating HTEs in survival analysis is vital for applications such as precision medicine but remains challenging due to censoring, unobserved counterfactuals, and complex modeling assumptions. This paper addresses these challenges by providing a unified categorization of survival HTE methods into three families—outcome imputation, direct-survival CATE models, and survival meta-learners—along with modular implementations of their variants. It also includes synthetic datasets with known ground-truth HTEs that systematically vary causal assumptions (e.g., randomization, confounding, positivity, informative censoring) and survival dynamics (e.g., Cox, AFT, Poisson models with varying censoring rates). Additionally, it offers multiple semi-synthetic datasets and two real-world datasets.

### Strengths
-	Addresses a Critical Need: The paper tackles a significant gap in the causal inference and survival analysis literature -- the lack of standardized evaluation practices for HTE estimation with survival data. 
-	The benchmark incorporates multiple data types (synthetic, semi-synthetic, real). The synthetic data explores a wide range of crucial factors: different causal assumption violations (i.e., confounding, positivity, informative censoring) and commonly used survival/censoring distributions (Cox, AFT, Poisson).

### Weaknesses
Major comments:

-	Limited synthetic data generation processes: The synthetic datasets are restricted to three survival time generation mechanisms—Cox, AFT, and Poisson—which may not capture more complex, non-parametric scenarios. For instance, SurvITE defines discrete hazards and samples time-to-event or censoring outcomes based on those hazards. Also, the validity and representativeness of such synthetic data remain unclear. In particular, how treatment effects, selection biases, and covariate shifts influence the generated time-to-event outcomes is not well articulated, as the datasets are primarily described in terms of their underlying hazard functions.
- Lack of actionable insights from the benchmark: The benchmark results do not provide sufficient insight into how model design decisions should be made based on the findings. As a result, the analysis feels largely descriptive rather than being guided by principled intuitions or hypotheses about model behavior. The paper could better highlight how specific model characteristics or data properties influence performance, offering clearer guidance for model selection and application.
-	The paper introduces a category of "Direct-Survival CATE Models", but, deep survival models that directly estimate the conditional average treatment effect (CATE) (such as SurvITE) on survival outcomes are not included. 
-	Although the paper claims to provide a unified framework for handling various survival models, it does not address models that directly predict time-to-event outcomes (e.g., Chafuwa et al., 2021). It is unclear how such models would be integrated into the proposed benchmark.
-	Estimates of hazard or survival functions can vary substantially depending on the chosen time horizon, yet the benchmark does not clearly specify or analyze how this dependency affects the evaluation results.

Minor comments:

-	Visualizations (e.g., conditional hazard and survival function plots) for synthetic datasets under different causal configurations would help clarify and validate the underlying data generation processes.

### Questions
-	If ground-truth hazard functions are available, it would be more meaningful to evaluate HTE performance with respect to the hazard functions themselves. Since many of the evaluated methods are capable of directly estimating conditional hazards from data, it is unclear why these performance results are not reported in the paper.
-	Regarding Weakness 2, it remains unclear how the benchmark results can be practically leveraged when applying the methods to new datasets or models.

### Soundness
3

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
4

### Summary
This work introduces a benchmark for evaluating estimators of the CATE in right-censored survival settings. The benchmark considers several violations or near violations of the assumptions that these methods commonly rely on (overlap, ignorability, etc.). The authors then evaluate a variety of existing learners for survival functions, which they classify into 3 categories: outcome imputation, direct methods, and meta-learners.

### Strengths
To my knowledge, this is the first large-scale benchmark for CATE estimation in the context of right-censored outcomes.

The benchmark covers a wide mix of datasets, including semi-synthetic and real ones.

Reports a comprehensive set of metrics.

Provides reproducible code.

### Weaknesses
The provided README in the anonymized GitHub doesn't make it clear how to add new learners to the benchmark. For this benchmark to be widely adopted, this should be described (and made as simple as possible).

Minor point, but TMLE can't generally be used to estimate the CATE. The parameter isn't smooth enough for TMLE to be used to estimate it. An exception to this occurs when measuring effect modification with respect to a discrete summary of the baseline covariates (e.g., Stitelman et al., 2010), but that doesn't seem to be your main case of interest.

### Questions
The proposed methods seem to require the set of covariates you're conditioning on in the CATE to be the same as the ones to be used to satisfy the ignorability assumptions. This is somewhat restrictive, as it means that using this CATE in the future requires measuring all of X, which may be infeasible. For non-survival CATE estimation, people get around this by instead considering the CATE conditional on a user-specified subvector V of X. The identifiability assumptions remain unchanged, and many methods (e.g., DR-learner) also work in these cases. Can your benchmark be used in these settings as well?

Would it be easy to extend your benchmark to other types of coarsened data - e.g., left-truncated data?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a benchmark for evaluating models which estimate a heterogeneous treatment effect from survival data. The benchmark consists of 40 synthetic, 6 semi-synthetic, and 2 real datasets. The synthetic and semi-synthetic datasets include ground truth HTEs which realistically violate standard assumptions in causal inference to different degrees. One of the two real datasets, Twins, is conducted so that one twin receives treatment and one does not; as close to a ground truth as can be obtained in reality. The other dataset is from an HIV clinical trial. The authors also implement 52 methods for survival HTE estimation and compare their efficacy on the benchmark. They find that there is no single dominant method, with the best method frequently depending on which causal inference assumptions are violated. Based on their observations, they provide guidelines suggesting which method or class of methods to use in a given scenario.

### Strengths
The paper is clearly written and easy to follow.

The proposed benchmark is thorough and does an excellent job of covering many different scenarios of practical relevance for HTE estimation with survival data. The motivation for the construction of the different synthetic datasets (in particular, tying each one to a combination of the "causal configuration" and survival analysis setup) makes it clear exactly what is being tested by each synthetic dataset. The semi-synthetic and real datasets are also constructed sensibly. The semi-synthetic data tests a method's ability to cope with more realistic covariate distributions (as the fully synthetic datasets only use uniformly distributed covariates) while still maintaining access to a ground truth CATE. For the real data, the Twins dataset seems to be the closest possible thing to having a ground truth measurement with real data. Modulating the censoring pattern on the HIV clinical trial dataset is an effective method for testing the robustness of each method to censoring on real data without a ground truth. I found Fig. 4 related to this dataset to be particularly interesting, as it shows that *none* of the existing methods have completely desirable behavior in the case of increasing censoring on real-world data. The CATE either changes drastically under increased censoring, or (as in the case of the CSF) the CATE estimate is simply very insensitive to the covariates.

In addition to providing a consolidated benchmark for HTE estimation with survival data--already useful in its own right to standardize the evaluation of these methods--the authors make good use of their benchmark to derive practical recommendations for the strenghts and weaknesses of existing methods. Access to these results as well as the collected implementations of a large number of relevant baselines should be of great use to other researchers working in this area.

### Weaknesses
While the types of CI assumption violations considered are extensive, the ground truth CATEs used for the synthetic data generation (discussed in Appendix A.3) are somewhat limited. In particular, the functional form of the ground truth consists of mostly linear components, with a couple of instances of quadratic terms, square root terms, or threshold discontinuities. Of particular note is that there are no interaction terms between covariates. Examining the effect of more severe nonlinearities, as well as interactions between the covariates, would strengthen the baselines. There are also a large number of presumably hand-picked constants used to define the ground truth. Some justification for how these particular constants were chosen, or else results showing that the conclusions of the paper are robust to the particular choice of constants, would also help greatly. I believe these are the most pressing issues and I would happily raise my score for the paper if they were addressed.

The CATE is defined to be the difference of some function of the survival time with and without the binary treatment. This is a reasonable design choice which is common in causal inference settings. However, in practical settings such as the analysis of clinical trials, treatment is often measured as a change in the underlying hazard function, and it is not immediately clear if this quantity can be represented in the form of equation (1).

Lastly, as the authors acknowledge, the evaluation is limited to binary, static treatments and static covariates, with time-varying or more complicated treatment variables left for future work. While this is a limitation, I agree with the authors that the scope they consider is already of great value.

Typos:
1. Lines 82-83: $T_i$ and $\widetilde{T}_i$ are reversed.

### Questions
How were the coefficients chosen for the synthetic data generation? How robust are your conclusions to variations in these coefficients?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose survHTE-Bench, a benchmark for estimating heterogeneous treatment effects (HTE) with right-censored survival outcomes. The paper summaries 52 estimators across three families, and evaluate them on 40 synthetic datasets, 6 semi-synthetic datasets, 2 real datasets. Results are reported through Borda count rankings, RMSE metrics.

### Strengths
1. The paper is well written with a clear scope. The benchmark topic is important and interesting in the study of treatment effect on health science datasets.

2. The paper considers a large set of HTE estimators, the evaluations are done on multiple settings. 

3. The paper includes a very complete reproducibilities resources.

### Weaknesses
1. A complete benchmark on HTE with survival datasets is surely needed in the community. There are some benchmark papers on HTE (not on survival setting), including [Crabbé, J., et al. 2022], [Shimoni, Y., et al. 2018], [Kapkiç, A. et al. 2024] and others. However, extending the benchmark on HTE from complete datasets to right-censored datasets can be a weak improvement. There can be overlaps among those benchmarks. I believe a comprehensive benchmark named by survHTE-Bench should include methods and datasets with all types of censoring.

Crabbé, J., Curth, A., Bica, I., & Van Der Schaar, M. (2022). Benchmarking heterogeneous treatment effect models through the lens of interpretability. Advances in Neural Information Processing Systems, 35, 12295-12309.

Shimoni, Y., Yanover, C., Karavani, E., & Goldschmnidt, Y. (2018). Benchmarking framework for performance-evaluation of causal inference analysis. arXiv preprint arXiv:1802.05046.

Kapkiç, A., Mandal, P., Wan, S., Sheth, P., Gorantla, A., Choi, Y., ... & Candan, K. S. (2024, October). Introducing causalbench: A flexible benchmark framework for causal analysis and machine learning. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (pp. 5220-5224).

2. Synthetic data design: The data generation process is too simple and not realistic, especially in health science. For example, all five covariates are **independently** generated from Uniform(0,1); treatment assignment is simple and have weak dependence on covariates. The benefits of using simulation studies is we can include all possible settings with ground truth, therefore, the data generation should include more settings, such as low/high dimensional, unbalanced cases, correlated features, causal related features, various tail behaviors, non-smooth hazards, treatment policy with feedbacks and so on.

3. Semi-synthetic data design: I have the similar comments as in weakness 2. The treatment assignment is independent and Bernoulli(0.5). Censoring times are independently drawn from Poisson($\lambda_c$), $\lambda_c$ is constant.  The potential outcomes are Poisson with means linearly depends either on $(S,X_{36})$ or $X_{36}$. Semi-synthetic datasets are combinations of real data with reasonable synthetic data, the simple setting on treatment assignment and censoring generation can be non-realistic. 

4. Synthetic data design: The violation severity is binary. Authors acknowledge violations are binary (present or absent). But realistic applications require graded severities (e.g., overlap quantified by min propensity, MNAR censoring strength). It would be great if you can provide dose response grids of violation magnitude.

5. I think Twins “ground truth” for survival is debatable. The Twins setting is typically used for counterfactual outcomes like birthweight; “ground truth” for survival times under treatment and control is not fully inherent and is partly constructed (randomized treatment assignment and censoring). The authors should justify why observed twin outcomes constitute true counterfactual survival outcomes and clarify how censoring was constructed (84.8% is extremely high).

6. The benchmark considers multiple assumptions and considers the cases when the assumptions hold or not (binary). However, there are cases when certain assumption is not simply satisfied or not satisfied. In application, we also want sensitivity analysis, for example, if those assumptions are a little bit wrong, how much would the conclusions change? There are methods for such sensitivity analysis, for example, Rosenbaum’s $\Gamma& for unmeasured cofounding.

7. Methods: The benchmark includes three families of models. This is good. However, there are models that are closely related, but are not included. For example, TMLE, IV-based survival HTE, Dynamic treatment cases, Bayesian survival HTE. Here I list several references.

Stitelman, O. M., Wester, C. W., De Gruttola, V., & van der Laan, M. J. (2011). Targeted maximum likelihood estimation of effect modification parameters in survival analysis. The international journal of biostatistics, 7(1), 19.

Tchetgen, E. J. T., Walter, S., Vansteelandt, S., Martinussen, T., & Glymour, M. (2015). Instrumental variable estimation in a survival context. Epidemiology, 26(3), 402-410.

Cho, H., Holloway, S. T., Couper, D. J., & Kosorok, M. R. (2023). Multi-stage optimal dynamic treatment regimes for survival outcomes with dependent censoring. Biometrika, 110(2), 395-410.

Chen, X., Harhay, M. O., Tong, G., & Li, F. (2024). A Bayesian machine learning approach for estimating heterogeneous survivor causal effects: applications to a critical care trial. The annals of applied statistics, 18(1), 350.

8. Borda count ranking averages hide effect sizes and are sensitive to the method set and dataset composition. It would be better to add uncertainty quantification.

9. For a benchmark paper, the evaluation of using only RMSE is restricted. Quantile based treatment effect or other rank based treatment effect would be robust in certain cases. For  survival data, fixed time survival probability difference can be meaningful in clinic study. These alternative considerations would improve the novelty of the benchmark.

### Questions
1. The benchmark defines CATE in terms of RMST up to a time horizon $h>0$ time horizon $h$. If I am correct, $h$ is the maximum observed time per dataset. If this is the case, $h$ varies across datasets (and splits for some methods), so different runs are literally evaluating different estimates. When censoring is heavy or tails differ, this changes both identifiability and difficulty in a way unrelated to the method, it can confound any cross dataset ranking. Then, there should be a sensitivity report with respect to $h$ and include additional estimates such as median survival. Please clarify this.

2.	The benchmark mentions serval references within the direct-survival CATE models in section 2. but  why you only include and evaluate causal survival forest in this family in your experiments? Am I missing the results of other models, for example SurvITE in [Alicia C., et al. 2021 NeurIPS]?

### Soundness
2

### Presentation
3

### Contribution
2
