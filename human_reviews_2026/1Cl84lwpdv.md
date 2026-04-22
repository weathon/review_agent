# Random-projection ensemble dimension reduction

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
We introduce a new, flexible, and theoretically justified framework for dimension reduction in high-dimensional regression, based on an ensemble of random projections. Specifically, we consider disjoint groups of independent random projections, retain the best projection in each group according to the empirical regression performance on the projected covariates, and then aggregate the selected projections via singular value decomposition. The singular values quantify the relative importance of corresponding projection directions and guide the dimension selection process. We investigate various aspects of our framework, including the choice of projection distribution and the number of projections used. Our theoretical results show that the expected estimation error decreases as the number of groups of projections increases. Finally, we demonstrate that our proposal consistently matches or outperforms state-of-the-art methods through extensive numerical studies on simulated and real data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Random-Projection Ensemble Dimension Reduction (RPE), a flexible sufficient dimension reduction (SDR) framework. It samples many low-dimensional random projections, selects the best in each group by held-out regression performance, and aggregates selected projections via the average of outer products followed by SVD; the singular values also drive a data-dependent estimate of the target dimension (Algorithm 2) and a “double pass” refinement (Algorithm 3) when further compression is desired. The authors give a finite-sample bound that decays as L^(-1/2)in the number of groups (Theorem 1), extensive simulations (nine models), and real-data results claiming competitive or superior performance versus SIR, pHd, MAVE, DR, gKDR, and drMARS.

### Strengths
1. The method cleanly combines random projections, held-out selection, and SVD-based aggregation, making it plug-and-play with different projection families and base regressors while yielding a stable subspace estimator (with an optional double-pass refinement and a built-in dimension-selection procedure). 
2. The finite-sample analysis isolates an “infinite-simulation” term and a sampling error that shrinks as L^(-1/2), clarifying how performance improves with the number of projection groups.
3. Extensive simulations spanning multiple p,n settings and diverse baselines, plus real-data evaluations, show competitive or superior accuracy; the algorithms and defaults are presented clearly enough to reproduce results.

### Weaknesses
1. The Contributions section is overly long and diffuses the main claims. The current text mixes method mechanics into the contributions. The Contributions should state only the new ideas.
2. The paper discusses how to choose L,M, and d, but provides no guidance for n_1 (the within-group training split used for projection selection).
3. Some arguments in the proof are confusing.

### Questions
1. Related work is missing several recent SDR directions that fit your setting. Such as BC (Huang, H. et.al. (2024)), KSDR (Liu, B. et.al. (2024)) and neural networks based SDR (Xu, S. et.al. (2025)).
2. The introduction foregrounds the high-dimensional p>n regime, but the main text only highlights a p<n case (e.g., n=200,p=50). Please surface at least one representative p>n setting (e.g., n=50,p=100) in the main paper—moving the results from the appendix if needed. Otherwise, the empirical narrative does not match the paper’s stated scope.
3. In the proof of Theorem 1, you state that {P_(l,*) }_(l=1)^Lare independent, but marginally they are not: each P_(l,*) is a data-dependent function of the same sample. If the intended claim is conditional independence given the observed dataset D, please state and use it explicitly. Moreover, all concentration steps should be written with P(⋅|D)and E[⋅|D]. 
4. The paper does not report the projection Frobenius distance d_F in unknown-d_0 settings; please include this results. 
5. Please add a sensitivity study varying n_1/n∈{0.5,0.67,0.8}, since n_1 trades off selection variance versus validation reliability. 

Reference:
1. Huang, H. H., Yu, F., & Zhang, T. (2024). Robust sufficient dimension reduction via α-distance covariance. Journal of Nonparametric Statistics, 1-16.
2. Liu, B., & Xue, L. (2024). Sparse kernel sufficient dimension reduction. Journal of Nonparametric Statistics, 1-24.
3. Xu, S., & Yu, Z. (2025, April). Neural Networks Perform Sufficient Dimension Reduction. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 20, pp. 21806-21814).

### Soundness
3

### Presentation
3

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
The main goal of this paper is to accurately estimate the central mean subspace $S(A_0).$ To achieve that, paper develops random-projection ensemble dimension reduction (RPEDR) framework for high-dimensional regression problems. RPEDR first generates independent random projections and selects the optimal projection based on mean-squared error in each group. After averaging outer products of selected projection matrices to get $\hat{\Pi}$, final estimated dimension reduction directions and relative importance of each direction are obtained from SVD of $\hat{\Pi}$. The paper then derives the expected estimation error
bound of proposed method to show the effects of multiple parameters. Finally, extensive empirical results present the effectiveness of new method.

### Strengths
1. The proposed method is parallelizable, making it applicable to distributed systems.  
2. The paper not just shows empirical results but also provides theoretical bound for estimation error.

### Weaknesses
1. The superiority of proposed method is not so promising. According to experiment results in Table 2-7, drMARS sometimes shows better performances and needs less runtime. On the other hand, proposed RPE2 and RPE are not always optimal and consume much more runtime. This weakens the contribution of this work.
2. This framework has multiple parameters, including L, M and d, and each needs special tuning, which makes this framework less practical.
3. The computation cost of proposed method may bring concerns because of the long runtime.
4. Considering the main goal of this work is dimension reduction, the reviews about this topic is not clear enough to show existing methods and their relationships with proposed method.
5. Lack of comparison of theoretical bounds. Since paper derives the error bound, it's necessary to compare with existing bounds.

### Questions
As the main assumption in this work, the existence and uniqueness of the CMS, a more detailed description about it is needed, like the exact conditions about this assumption.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Random Projection Ensemble Dimension Reduction (RPEDR). The method projects data onto multiple low-dimensional random subspaces, evaluates each projection within its group using a base regression model, selects the best projection in each group, and aggregates the winners via SVD. This procedure adaptively identifies informative directions, yielding stable estimates at modest computational cost. The authors also study design choices, projection distributions and the number of projections, and provide theory showing that the expected estimation error decreases monotonically with the number of projection groups.

### Strengths
* Proposes a novel, flexible, and theoretically principled framework for dimension reduction in high-dimensional regression via random-projection ensembles.
* Adaptively selects informative projection directions, delivering more stable estimates.
* Provides theoretical guarantees showing that the expected estimation error decreases monotonically as the number of projection groups increases.

### Weaknesses
I am not deeply familiar with the SDR literature; my expertise lies more in random-projection methods for computational efficiency. Consequently, I cannot fully determine whether the comparisons with existing SDR approaches are sufficiently comprehensive or representative of the current state of the art. The following comments are therefore offered as provisional observations:

* Theorem 1 relies on the assumption $L = \infty$, which limits its practical value for guiding the selection of $L$ in real applications. Could similar theoretical guarantees be derived for finite $L$?

* The introduction frames the problem as high-dimensional ($n<p$), yet Appendices analyzes parameter choices primarily for $n>p$. This inconsistency weakens empirical support for the stated high-dimensional motivation. Additional experiments or discussion for the $n<p$ regime would improve credibility.

* Algorithm 3 is described as an enhanced version of Algorithm 1, but it is not clear whether Theorem 1, or a corresponding variant, extends to this setting. Clarifying the theoretical validity of the two-stage procedure would strengthen the methodological contribution.

* Algorithm 3 fixes $L = 200$ without justification. Is this choice empirically optimal, or merely heuristic? How sensitive is performance to $L$ across different $n, p, d$? 

* While the paper repeatedly emphasizes the flexibility and efficiency of the proposed method, it lacks an explicit characterization of computational complexity, 
The authors are encouraged to clearly specify the time and memory complexity under typical parameter settings and provide a comparison with established methods like MAVE and drMARS to better illustrate the computational advantages and trade-offs of the proposed approach.

* Although Appendix D mentions real-data experiments, the main text presents only simulation results. It is recommended to include at least one illustrative real-data application n the main paper to enhance the work’s practical relevance and overall impact.

### Questions
1. In Algorithm 2 (line 6), how is the threshold $T_{\ell} > 1/2$ chosen? Could alternative thresholds (other than $1/2$) be used?

2. In Algorithm 3, $L = 200$ is used while many experiments set $n = 200$, which may suggest nontrivial computational cost. Can a smaller $L$ be employed without materially degrading performance? If so, is there guidance or a sensitivity analysis for selecting $L$ as a function of (n,p,d)?

3. Beyond Gaussian and Cauchy projections, are other projection families, e.g., sub-Gaussian or Hadamard-structured transforms, compatible with the framework?

4. I recommend adding a  framework diagram to illustrate the workflow and the relationships among Algorithms 1–3.

### Soundness
3

### Presentation
3

### Contribution
3
