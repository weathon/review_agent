# Privacy-Aware Data Integration for Enhanced Quantile Inference under Heterogeneity

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Quantile estimation and inference play essential roles in diverse scientific and industrial applications, and their accuracy can often be enhanced by integrating auxiliary data from multiple sites. However, developing efficient aggregation methods for quantile inference under potential privacy constraints, particularly with heterogeneous datasets, remains challenging. To address these issues, we propose a systematic framework for quantile estimation and inference under potential local differential privacy (LDP). The key idea is to construct weighted estimators by adaptively aggregating quantile estimates from target and source sites. The adaptive weights are determined by minimizing the asymptotic variance, incorporating an additional $\ell_2$ penalty to account for parameter shift. A parallel stochastic gradient descent algorithm under LDP constraints is developed for weight estimation and valid inference. Additionally, we introduce a conservative weighted estimator to ensure robust inference across diverse heterogeneous scenarios. Rigorous theoretical analysis establishes the consistency, normality, and effectiveness of the proposed methods. Extensive numerical studies corroborate our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies quantile estimation and inference under heterogeneous datasets, where each dataset contains a different number of data points generated through a locally differentially private (LDP) mechanism applied to underlying true data. The goal is to estimate the true quantile for a target site by leveraging not only its own dataset but also information from other sites. To achieve this, the authors propose a weighted estimator that aggregates individual quantile estimates obtained by applying parallel SGD on each dataset. The weight vector is determined by minimizing a loss function that accouns for both the asymptotic variance and the heterogeneity of true quantiles. The authors rigorously establish theoretical guarantees of the proposed estimator, such as consistency and normality. Extensive numerical experiments further demonstrate the empirical advantages of the proposed method over existing approaches.

### Strengths
1. The authors develop a general and systematic framework for privacy-aware quantile estimation and inference via data integration
1. The analysis of statistical properties of the proposed quantile estimator is thorough and technically deep. 
2. The numerical experiments convincingly demonstrate the improved performance.

### Weaknesses
1. The introduction section (including the motivating example in Figure 1) does not appear well aligned with the formal model and may confuse readers regarding the problem formulation and practical scope. The introduction gives the impression that the setting involves (i) a federated learning framework with data generated via a LDP mechanism and (ii) the use of only source sites' sensitive data to estimate the target site's quantile. However, the actual model is not a federated learning setup and the target site's data are also sensitive. Overall, I believe that the paper would benefit from a more consistent presentation of the problem setup in the introduction.
2. The current discussion focuses on the proposed estimator's statistical properties. Although such discussion and analyses are important, given the paper's emphasis on privacy, more efforts could be put into examining the impact of privacy constraints. For example, the presentation of several theorems (e.g. Theorem 4.1, 4.2, 4.3) could be improved by explicitly incorporating the randomness level $r_k$ of the LDP mechanisms.
3. The paper's technical contributions could become easier to digest for readers if more intuitive explanations can be provided.
4. The privacy guarantee in Proposition 4.1 requires further justification. While each local randomizer meets LDP, the interactive nature of SGD may need composition theory to properly track cumulative privacy loss. The authors refer to [1] to support the privacy claim; however, as far as I can tell, the cited work only establishes the LDP guarantee for their local randomizer, not for the overall algorithm. Therefore, a detailed proof or a proper reference is needed for the claimed LDP guarantees of both PSGD and Algorithm A.2.


[1] Yi Liu, Qirui Hu, Lei Ding, and Linglong Kong. Online local differential private quantile inference via self-normalization. In International Conference on Machine Learning, pp. 21698–21714. PMLR, 2023.

### Questions
1. Could the authors provide an intuitive explanation for why the estimator is designed to be a weighted one that incorporates quantile estimates from source sites rather than solely relying on the target site's? This design and the corresponding results appear to be counterintuitive to me. Since the data generating processes at different sites are assumed to follow different distributions (see line 116, $\mathcal{P}_k,\forall k$), it is unclear why using these potentially irrelevant source sites' data could be beneficial.
2. It might be helpful to explicitly write out $r_k$ in the statement of your main theoretical results, e.g., Theorem 4.1 - 4.4. In particular, for theorem 4.3, it would be interesting to see how the reduced variance is related to $\{r_k\}_{k=0}^N$? Establishing the relationship could help reveal how the level of randomness introduced by LDP mechanisms affects the estimator’s efficiency.
3. In the numerical experiments, can you also show empirical evidence of the claimed asymptotic normality? This would strengthen the connection between the theoretical results and empirical findings.
4. Regarding the setup of how private data is generated. Why did you choose to perturb the binary indicator function, rather than assuming that true data points are perturbed by a LDP mechanism and then shared with the target site? The latter situation seems more practical for your motivating example. Will this choice significantly affect your analysis? (I guess yes, as the binary indicator function in gradients would be more complex.)

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
The authors consider the problem of data integration/transfer learning for quantile estimation under local DP.  The authors recognized that in the Average Stochastic Gradient Descent algorithm, there is an update involving binary terms that lends itself well to incorporating local DP via a randomized response approach.  They apply this approach per dataset and then pool estimates together to produce a final estimate.  The pooling leverages a weighting scheme to downweight data that don't aid in the original estimation task.

### Strengths
Combines several active areas: quantile estimation, privacy, and transfer learning/data integration.  Extensive mathematical results including asymptotic results for the estimators.  Interesting weight scheme for combining estimates.

### Weaknesses
Overall, the paper seems strong mathematically (except a few small points), but the motivation for the paper seems a bit weak.  Data integration/transfer learning is very prominent with complex models, but not for a single quantile.  Furthermore, the role of LDP in the analysis seems like a bit of an afterthought as in the end it seems to increase your variance a bit, but the much bigger focus is on the bias between datasets.  However, LDP will inflate the uncertainty of estimates dramatically, which can be missed when treating the epsilon/r as fixed in the asymptotic analysis.

### Questions
- Minor, but 3.1 is a property, not a definition (i.e. that's not the definition of a quantile).
- Thm 4.1 and Assumption 4 don't make sense to me.  There are still sample sizes on the RHS.  What are the limits being taken with respect to?
- Early on they don't really state what the problem is.  What are they trying to estimate?  Especially if there is "drift".  Based on later discussions, it's \theta_0 based on the definition of the bias term.  But I don't see this in the problem statement.
- What role does r_k play in the asymptotic?  Accuracy of the estimates is a function of both r_k and the sample size.  I think in modern DP papers it isn't reasonable to ignore it's contribution by treating it as fixed.
- This problem seems weakly motivated (data integration to estimate a simple quantile), but maybe the LDP requirement can help motivate it since it will add a lot of uncertainty?  
- Sample sizes are quite large in the empirical work.
- When does LDP add so much noise that you are better off just working with no privacy and the target data?  This is discussed in Thm 4.3, but there is no mention of the role of privacy.

### Soundness
3

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
3

### Summary
This paper introduces a privacy aware multi site framework for quantile estimation under local differential privacy and distributional shifts, building a variance and bias penalized weighted estimator with a conservative variant for borderline bias and estimating variances via parallel SGD chains that comply with local differential privacy. The method achieves consistency, asymptotic normality, and efficiency gains over single site estimation when at least one source is informative, and experiments on Normal and Cauchy simulations and a real wage dataset show improved coverage and shorter intervals, with limitations including reliance on multiple SGD chains and a fixed number of sites.

### Strengths
The paper tackles a timely and practically relevant problem of multi-site quantile estimation under local differential privacy, and provides an implementable weighting framework with parallel-SGD-based variance estimation. Theoretical results establish consistency, asymptotic normality, and efficiency gains over single-site estimation, and experiments on Normal and Cauchy simulations and a real wage dataset demonstrate improved coverage and shorter intervals. The presentation is clear, with explicit weight formulas and transparent privacy parameters, indicating strong practical impact for cross institutional analytics.

### Weaknesses
1. Novelty is weak; the method mainly recombines known ingredients including local differential privacy, weighted aggregation, and parallel SGD without a distinct new idea; please identify one concrete, verifiable contribution that prior distributed or LDP quantile methods do not achieve.

2. The privacy analysis is under-specified; please provide a clear epsilon and delta privacy guarantee.

3. Sensitivity is unreported; please report coverage and interval length as a function of the privacy budget.

4. The intermediate bias regime with bias ≈ N^{-1/2} may jeopardize validity; please add a finite-sample coverage bound for this case.

5. Experiments cover Gaussian and Cauchy simulations plus a single real-world dataset, with few strong privacy-preserving baselines; please add stronger baselines.

6. Theoretical guarantees assume a diverging number of PSGD chains and a fixed number of sites; please clarify behavior when the number of sites grows.

### Questions
see the detail in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies locally differential private (LDP) quantile estimation in the presence of auxiliary data sources that may have different privacy requirements and different data distributions. It provides a method for LDP quantile estimation at each data source as well as a method for aggregating these estimates to augment that of the target data source. In addition to a (straightforward) privacy analysis, it proves several asymptotic utility guarantees about its algorithm and presents experiments on synthetic and real data.

### Strengths
The problem setup is plausible, and I didn't find existing work on it. The proposed approach is reasonable and, to my reading, doesn't rely on any particularly strong assumptions, and is an algorithm I can actually imagine being run in the real world. This is not so common in DP.

### Weaknesses
*Clarity*: One weakness of this paper is that it lacks a concise explanation of its algorithm in the main body, so a reader has to piece the algorithm together over the course of many paragraphs of text. I suggest that the authors think about a clearly signposted, coherent, and self-contained presentation of the algorithm that can go in the main body -- a diagram, pseudocode, a high-level sketch, something. This would be a better use of space than Figure 1 or 2, IMO. As is, the paper tries to convey a narrative about the algorithm's construction as a sequence of ideas, which is maybe how the research project was developed, but is IMO not the clearest way to describe the finished product.

*Experiments*: I didn't get much info from the experiments. I suggest dropping ADP(0) ($\lambda = 0$ seems to be a qualitatively different algorithm than what the paper is interested in) and ADP(cv) (cross-validation, unless accounted for in the DP guarantee, feels unrealistic for this kind of application) and focusing on ADP(1) and maybe ADP(cons). This would also make the plots in Figure 3 easier to parse. Also, unless I missed it, the final LDP guarantee is not actually specified in the plots, though at response rate 0.5 I think it ends up being a reasonable value. The experiments also seem to be missing a suggested interpretation -- the paper just describes the experiment setup, provides a plot, and moves on to the conclusion. It looks like there might be a 1-2 order of magnitude improvement in the MSE, but I'm not sure how useful that is -- are we just making an already good enough estimate even better?

I would also suggest moving the real dataset experiments into the main body and providing a similar discussion, though I was confused by its presentation in Table A.3 -- are any of the provided metrics actually error? It looks like the experiment just records the values returned by different approaches, but says nothing about how good they are.

*Incremental?*: My understanding is that the overall algorithm is to run several instances of locally randomized ASGD on disjoint partitions of the data at each site (which it calls PSGD), use the end results to obtain quantile and variance estimates, and then aggregate these estimates by weighting according to bias and variance. As the paper notes, locally randomized ASGD (or a single data source has previously appeared in Liu et al. 2023, and PSGD appeared in Zhu-Li-Wang 2024 (and is anyway an instance of the classic subsample-and-aggregate algorithm). The paper's contribution is the weighted aggregation, which is logical but not so novel.

### Questions
See "Weaknesses". Overall, I think the paper makes a decent but limited contribution that is muddied by unclear presentation. A significantly revised version of the paper could be acceptable at ICLR or a similar venue, but I can't champion the current version, and I think even a significantly polished version would be a ~weak accept.

### Soundness
3

### Presentation
2

### Contribution
2
