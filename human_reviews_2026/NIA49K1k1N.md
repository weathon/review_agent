# Assumption-lean inference on treatment effect distributions

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
In many fields, including healthcare, marketing, and online platform design, A/B tests are used to evaluate new treatments and make launch decisions based on average treatment effect (ATE) estimates. But this workflow can overlook distributional risks, such as a large fraction of individuals affected negatively by the treatment. In this paper, we propose a novel method for inference on the treatment effect distribution. Prior work in this setting has estimated partial identification bounds, known as Makarov bounds, on the cumulative distribution function of the treatment effect by making restrictive assumptions on the outcome distribution. In contrast, our method guarantees accurate estimation and valid asymptotic inference of the Makarov bounds for any outcome distribution. Our main technical contributions are to develop smoothed surrogates for the Makarov bounds, derive semiparametrically efficient estimators of these surrogates, and propose a procedure for optimal selection of the smoothing parameters. We show empirically on synthetic and semi-synthetic datasets that, by not relying on the assumptions made by other methods, our estimators achieve a better bias-variance trade-off and lower mean-squared error. Finally, we deploy our method on real A/B test data from a large social media platform, and show how estimates of the treatment effect distribution can inform decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
na

### Strengths
pros:
This paper makes a clear and original contribution to causal inference by proposing an assumption-lean framework for inferring the distribution of treatment effects. Unlike prior methods that rely on restrictive margin or smoothness assumptions, the authors introduce smoothed Makarov bounds that allow valid semiparametric inference even when standard assumptions fail. The method combines theoretical rigor—through efficiency theory and bias control—with strong empirical validation on synthetic, semi-synthetic, and real-world A/B test data. Its ability to uncover heterogeneous and potentially harmful treatment effects, even when the average treatment effect is positive, highlights significant practical value for risk-aware decision-making in experiments.

con:
The main limitations lie in computational complexity and scope. The proposed smoothing-based estimators require intensive numerical integration and careful tuning of smoothing parameters, which may limit scalability in high-dimensional or large-scale settings. In addition, the framework currently applies only to binary treatments and assumes bounded outcomes, leaving open challenges for extending it to multi-valued or continuous treatments and heavy-tailed outcomes. 
Despite these constraints, the paper's methodological innovation and practical relevance make it a strong step toward more robust and distributional approaches to causal inference.

### Weaknesses
na

### Questions
na

### Soundness
2

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
3

### Summary
This paper focuses on the problem of estimating the full distribution of treatment effects rather than just the average treatment effect (ATE). The authors focus on estimating the so-called Makarov bounds, which define the sharp limits on the possible treatment-effect distribution given only the observed marginal outcome distributions.

Estimating these bounds is challenging because they involve non-smooth operations and previous methods either rely on strong “margin” assumptions (which are often violated in practice), or use plug-in estimators that lack valid inference guarantees. The paper proposes an assumption-lean approach that smooths the non-differentiable components of the Makarov bounds using differentiable approximations. This smoothing enables the derivation of efficient influence functions and the construction of debiased, asymptotically normal estimators. The authors also provide an explicit bound on the bias introduced by smoothing and adjust their confidence intervals accordingly. They propose two data-driven procedures for selecting the smoothing parameters: one minimizing an empirical MSE bound and another based on a Lepski-type adaptive rule. Empirically, the method outperforms plug-in and “envelope” baselines on synthetic, semi-synthetic, and real A/B test data, especially when the margin assumption fails. The results show improved bias–variance trade-offs and more reliable inference.

### Strengths
1. The paper targets inference on treatment-effect distributions rather than just the mean effect. This is an important and underexplored direction for causal inference and A/B testing.

2.Their method removes the need for the restrictive margin assumption used in prior work, which is often violated in realistic settings where treatment effects are constant or nearly constant. This makes the inference procedure more robust and broadly applicable. 

3.The use of smooth surrogates (log-sum-exp and softplus) to approximate the non-differentiable Makarov bounds is conceptually elegant. It may have impact on other statistical topics too. Moreover, for smoothing parameters, it introduces two adaptive, data-driven methods to select them: (i) minimizing an empirical MSE upper bound and (ii) a Lepski-type adaptive selection rule. The experiments also showed their method's better performance.

### Weaknesses
1. Issues on widened and potentially conservative confidence intervals. Because the method adds an explicit smoothing-bias correction term to guarantee valid coverage, the resulting confidence intervals can become conservative and potentially wider than necessary. In practice, this may dilute the practical utility of the inference especially when the true bias is small.

2. A follow up question to 1 is: this is actually the cost (from smoothing) of this paper's approach by translating the difficulty in avoiding the restrictive margin assumption to the difficulty of balancing the bias and variance. Hence, although the paper provided two data-driven ways for picking the smoothing parameters, there are no theoretical guarantees on these choices. This makes the issue not totally solved (simulation or real data results could be from an optimal search of these smoothing parameters rather than from a solid guideline). Also, data-splitting approach is also used and this will reduce the data efficiency. This problem then hinges on further investigation both theoretically and empirically. So far, it is only heuristic.

### Questions
Please see Weakness and the following:

1. The bias bound $b(t_1,t_2)$ is derived under compact outcome support and finite Lebesgue measure. How sensitive are your guarantees to this assumption, and can it be relaxed for unbounded or categorical outcomes?

2. The framework assumes no interference between units. Could it be adapted to clustered or networked data, where spillovers exist? 

3. Due to this increased confidence intervals by smoothing, do you have empirical evidence or calibration plots showing how often they over-cover relative to the nominal level? To be precise, any evaluations on whether this bias-correction term $b(t_1,t_2)$ makes confidence intervals conservative?

### Soundness
3

### Presentation
3

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
The article under review proposes an efficient method for estimating bounds on the distribution of treatment effects. The approach combines two existing ideas: debiased estimators and smoothing techniques. This combination yields element-wise semiparametrically efficient estimators while simultaneously addressing the smoothing bias introduced by the technique.

### Strengths
The article relies on recent techniques from the literature, and the analysis appears sound.

### Weaknesses
1. **Bounds** 

- The paper's abstract, which argues that existing workflows "overlook distributional risks," appears to be overstated. There is a considerable literature on partial identification for treatment effect distributions that directly addresses this. The more pressing challenge, which the paper does not discuss, is the practical utility of these existing methods. Often, the identified bounds—and particularly their confidence bands—are too wide to provide meaningful guidance, thus limiting their impact. This essential context is missing from the paper's framing.


- This omission becomes more concerning in light of the paper's own results. The authors are critical of existing work for its "non-standard inference methods," yet the figures presenting their own estimations omit the corresponding confidence bands.
To show the practical utility of the proposed method and maintain consistency with its own critiques, reporting these confidence bands is essential. Hiding this information prevents a full and fair evaluation of the method's practical contribution.

2. **Theoretical Contributions** 

- *Concerns Regarding Novelty and Development*: 
While Table 1 summarizes the paper's stated contributions, the core techniques, such as debiasing and smoothing methods, are established tools from the existing literature. This reliance on established methods raises concerns about the paper's marginal contribution and novelty. Moreover, the analysis appears underdeveloped in key areas, as detailed below.

- *On the Definition of "Efficiency"*: 
The paper's claim of an "efficient" estimator appears to hold only in an element-wise sense (i.e., for each plug-in estimator). However, the primary quantity of interest is the **interval estimator**. The paper does not demonstrate that efficiency of the individual endpoints implies efficiency for the interval itself. This is a critical distinction that needs to be rigorously addressed.

- *Omission of Uniform Inference*:
Furthermore, when discussing distributions, inference should not be restricted to point-wise estimation. The more appropriate and relevant analysis would consider a **uniform bound** that holds over the entire distribution. This approach has been extensively investigated in the literature, and it is unclear why the authors did not conduct such an analysis. This omission is a significant gap, as it avoids the standard method for this class of problem.

3. **Simulation and Synthetic Data Studies**

- The table reports only the Mean Squared Error (MSE). Given that the proposed smoothing method intentionally introduces bias, bias must be reported separately. Presenting only MSE hides the trade-off at the heart of the technique.

- The MSE of each interval endpoint is of limited relevance. The primary object of interest is the interval itself. The authors should report metrics appropriate for interval estimation, such as average interval length and empirical coverage probability, rather than treating the problem as one of point-estimation.

### Questions
- Questions 1-3: Please see Weaknesses 1-3. 

- Question 4. 
The paper highlights the lack of "asymptotic normality" in existing methods as a significant drawback. This criticism, however, appears overstated. Many well-established estimators in statistics feature non-standard asymptotic distributions, and this alone does not invalidate them. This critique is particularly questionable given that the proposed method introduces its own set of practical complications, namely smoothing bias and the necessity of selecting tuning parameters. Is a more fair discussion of these respective trade-offs helpful?

- Question 5. 
The paper’s asymptotic analysis lacks clarity, particularly regarding the convergence rate and the precise role of the smoothing bias in the large-sample results.
While Corollary 4.3 provides a formula for a confidence interval, the main text lacks the explicit weak convergence results necessary to formally justify this corollary.
Furthermore, the proposed confidence interval depends on numerous nuisance parameters and functions. The paper fails to demonstrate that the interval remains theoretically valid when these nuisance components are estimated via a "plug-in" approach. This is a critical omission, as the validity of such a procedure is not self-evident. This concern is compounded by the vague specification of the paper's underlying assumptions, which makes it impossible to verify the method's theoretical soundness.
Where can I found those information in the paper?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper develops a new way to estimate Makarov bounds on treatment effect distributions by replacing the non-smooth max/min operators with smooth log-sum-exp (LSE) approximations. This smoothing enables valid inference even when traditional semiparametric estimators fail due to non-unique extrema (margin violations).

### Strengths
1. The idea of using log-sum-exp smoothing to recover differentiability is straightforward and simple. 
2. The paper provides a principled and theoretically grounded framework for valid semiparametric inference under margin violations.

Overall, the work is technically sound, and well supported by strong theoretical guarantees and empirical validation.

### Weaknesses
_1. Weak litearture review._
The paper omits a substantial body of work on Quantile Treatment Effects (QTE), which  serves as a core approach to distributional causal inference. This omission weakens the positioning of the proposed Makarov-based framework within the broader context of distributional effect estimation. I recommend to discuss prior QTE literature, and clearly articulate why the Makarov-based approach is advantageous when the joint distribution of potential outcomes is unidentified, whereas QTE only captures marginal contrasts.

_2. Unclear motivation and significance._
The paper does not sufficiently justify why margin violations pose a critical practical problem. While the theory is correct, the empirical motivation could more clearly demonstrate concrete failure cases of existing methods under margin violations, ideally with a real-world example rather than synthetic illustrations. Without such evidence, the significance of the proposed smoothing appears unclear.

_3. Incremental contribution_
The main methodological idea (replacing non-smooth max/min operators with log-sum-exp smoothing) is well-known in optimization and statistical theory. The paper mainly applies this existing trick to the Makarov bounds without offering fundamentally new theoretical insight or stronger guarantees beyond standard smoothing arguments.

_4. Presentation_
The table is too packed and the fonts are too small. I believe this is a violation of the font-size regulation ("do not change font sizes")

_5. Title and scope._ 
The title “Assumption-Lean Inference on Treatment Effect Distributions” is overly broad relative to the actual methodological contribution. The paper’s method still depends on very strong assumptions (e.g., ignorability condition, discrete treatments, overlap, etc.) so the “assumption-lean” claim seems overstated. A more concrete title explicitly reflecting the smoothing-based inference for Makarov bounds convey the scope and focus.

### Questions
Q1. How tight or sharp the smooth approximation of the Makarov bound?

### Soundness
3

### Presentation
2

### Contribution
1
