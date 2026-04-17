# We Have It Covered: A Resampling-based Method for Uplift Model Comparison

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Uplift models play a critical role in modern marketing applications to help understand the incremental benefits of interventions and identify optimal targeting strategies. A variety of techniques exist for building uplift models, and it is essential to understand the model differences in the context of intended applications. The uplift curve is a widely adopted tool for assessing uplift model performance on the selection universe when observations are available for the entire population. However, when it is uneconomical or infeasible to select the entire population or a large simple random sample, it becomes difficult or even impossible to estimate the uplift curve without appropriate sampling design. To the best of our knowledge, no prior work has addressed how to construct uplift curve estimates on the entire population along with uncertainty quantifications, using only a population subsample. We propose a two-step sampling procedure and a resampling-based approach to compare uplift models with uncertainty quantification, examine the proposed method empirically through simulations and real data applications, and conclude with a discussion.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors propose a resampling-based approach for quantifying the inference uncertainty in uplift curve. They claim the statistical difficulty for such uncertainty quantification, due to the impossibility of completely random sampling from population, as the x-axis of uplift curve determines how the sub-population is sampled, resulting in difference between sub- and entire populations. 

Overall, I feel that the problem setup is important. However, due to the clarity issues and the lack of related work, the paper itself is very hard to follow. Moreover, the experiments lack naive baselines, such as the interval estimators under the i.i.d. assumptions, and ablation studies for comparing with e.g., naive Boostrapping. Thus, it is unclear how the proposed approach is effective in practice.

### Strengths
- Problem setup is important: Uncertainty quantification for uplift models is crucial for reliable decision making.
- The authors illustrate several examples to claim the statistical difficulty for this problem.

### Weaknesses
(A) No baseline

Due to the lack of baselines, it is impossible how greatly the proposed sampling approach is effective. Although the authors say “However, this i.i.d. assumption may not hold, which invalidates the asymptotic confidence interval obtained above.”, this invalidity is unclear, as there is no empirical comparison in the main paper. 

- Is it impossible just to compare their method with such naive baselines? Coverage and length can be evaluated for such naive estimators, right?

(B) No related work section

The paper is very hard to follow at the beginning of Abstract and Introduction. 

- First paragraph of Section 1: Although the authors suddenly split the two research streams (causal “impact” of interventions vs. HTE estimation), the difference seems unclear. Do they refer to Pearl’s framework vs. Rubin’s framework? 

- Second paragraph of Section 1: Why these three categories? There are so many conditional average treatment effect (CATE; HTE in the authors’ wording) estimation methods. Also unclear what each category means. E.g., transform original outcome to new outcome, what does this mean? Are you referring to some representation-learning-based HTE estimation? Unclear.

In particular, I found the problem setup related to post-selection inference in statistics. I believe that some work addresses the statistical inference problem after top-K selection. Why don’t the authors cite literature on this? Is it unrelated?

(C) No Ablation Study

It seems that there is no ablation study for comparing the proposed sampling procedure with some naive one (e.g., just an ordinary Bootstrapping). Without ablation studies, readers cannot evaluate the effectiveness of proposed approach.


(D) No run time comparison

In general, resampling approaches are known to be computationally demanding. However, there is no computational complexity analysis in theory or no run time evaluation, making it hard to understand the efficiency of proposed approach.

### Questions
See the above questions in (A), …, (D).

### Soundness
2

### Presentation
1

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
The authors propose a two-step sampling design and nested bootstrap procedure to estimate the uplift curve for the whole population with UQ using a population subsample. The method first draws a random subset of the population and then adds a sample of top-ranked individuals to ensure coverage of high-uplift segments. Using inverse-probability weighting, they estimate the theoretical uplift curve from this subsample, and apply a nested bootstrap for uncertainty quantification.

### Strengths
- the paper addresses a relevant and challenging topic of valuating uplift models on the full population using only a subsample
- the two-step sampling and weighting idea is simple and intuitive, easy to implement
- the nested bootstrap provides a practical way to compute confidence intervals for uplift curves
- the authors test the method on both synthetic and real-world data

### Weaknesses
- I think the paper writing is not very easy to follow and could be organized more clearly in my opinion. For instance, I would have appreciated a more intuitive introduction of the nested bootstrap 
- the authors assume that the treatment assignment $T$ is independent of $X$, which is plausible in randomized experiments. This should be better emphasized as limits the applicability of the setting.
- I am not sure whether this is a sufficient contribution. I still have some doubts, see below.

Minor: 
- Plots 3,4 are hard not visually nice and hard to read. The text should be bigger and I would recommend putting more time into the layout improvement

### Questions
- this nested bootstrap procedure is very computationally heavy, can you elaborate on the computational cost of the algorithm? I think this should be added to the paper
- empirically, how robust is the method if treatment assignment is not fully randomized, for instance if there is some form of confounding?
- can the authors elaborate on how the variance of the estimator scales with the inclusion probabilities and sample size, and whether any finite-sample hold?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors propose a method to construct uplift curve estimates for the entire population using a population subsample. This approach primarily relies on combining data randomly sampled from the entire population with the subsamples and utilizing Inverse Probability Weighting (IPW) to eliminate selection bias between the subsamples and the entire population. The authors conduct experiments on both synthetic and real-world datasets to validate the effectiveness of the proposed method.

### Strengths
1. The paper is well-organized and easy to follow.
2. The authors provide a diverse set of scenarios in the simulation study, and the extensive experimental results show the robustness of the proposed method.
3. The use of real-world datasets adds significant credibility and practical relevance to the proposed method.
4. The experimental details are thoroughly presented, which should make the results easy to replicate.

### Weaknesses
1. The authors' understanding of the uplift model evaluation in industrial applications seems to deviate from practical realities. The authors claim that the dataset used for uplift model evaluation (the "subsample of the entire population") needs to be firstly sorted by the model scores and then top-x% samples selected for treatment-control division. Such sample selection process introduces selection bias between the subsample and the entire population, and thus they propose a method to address this issue. However, in practical industrial applications, it is more common to directly apply random treatment-control assignment to the entire subsample through A/B tests, rather than using the top-x% “sub-subsamples”. The uplift curve is then evaluated over the entire subsample by ranking based on the model scores [1]. Therefore, it is questionable whether the selection bias between the subsample and the entire population, as the primary challenge to address in the paper, is a common issue in real-world applications.

2. The authors overlook a significant body of work that addresses the selection bias between subsamples and the entire population [2]. The approach the authors use is closely related to the commonly applied inverse probability weighting (IPW) methods in this area, where they explicitly define the inclusion probability rather than estimating it via a model, as is typically done in related works. While the idea is good, the lack of citations to relevant literature is a concern and should be addressed.

3. The experiments only consider the scenarios with only three models to be evaluated, but in real-world applications, there are often many more models as candidates. It remains unclear whether the algorithm's performance and efficiency would hold up under a larger set of models. The authors should consider adding experiments with more models to validate the scalability and robustness of their method.

[1] Zhang, Weijia, Jiuyong Li, and Lin Liu. "A unified survey of treatment effect heterogeneity modelling and uplift modelling." ACM Computing Surveys (CSUR) 54.8 (2021): 1-36.

[2] Colnet, Bénédicte, et al. "Causal inference methods for combining randomized trials and observational studies: a review." Statistical science 39.1 (2024): 165-191.

### Questions
Please see Weaknesses. My primary concern lies in Weaknesses 1 and 2. I would appreciate it if the authors could provide further clarification on the specific scenarios where the problem studied in this paper would apply.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to address the gap of estimating full-population uplift curves using only subsamples.A two-step sampling scheme and a nested bootstrap approach are proposed. It is validated by both simulations and real datasets (Criteo, MegaFon), where the proposed method successfully identifies meaningful model differences, which provides a reliable tool for uplift model comparison.

### Strengths
1. It is the first to propose a framework for estimating full-population uplift curves using subsamples, with explicit uncertainty quantification.
2. Rigorous simulation and real-data experiments confirm the method’s validity (coverage, low bias) and ability to identify meaningful model differences.

### Weaknesses
The major weakness is the presentation, especially the problem formulation part. The contents seem to be a combination of several issues without clear connection, making it hard to follow.

### Questions
The Qini coefficient and uplift curve are only suitable for the cases of single action (T \in {0,1}) and binary response (Y \in {0,1}), is it possible to extend the proposed approach for more general settings, e.g. multiple actions and response in real numbers?

### Soundness
3

### Presentation
2

### Contribution
3
