# Impute-MACFM: Imputation based on Mask-Aware Flow Matching

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Tabular data are central to many applications, especially longitudinal data in healthcare, where missing values are common, undermining model fidelity and reliability. Prior imputation methods either impose restrictive assumptions or struggle with complex cross-feature structure, while recent generative approaches suffer from instability and costly inference. We propose Impute-MACFM, a mask-aware conditional flow matching framework for tabular imputation that addresses missingness mechanisms, missing completely at random, missing at random, and missing not at random. Its mask-aware objective builds trajectories only on missing entries while constraining predicted velocity to remain near zero on observed entries, using flexible nonlinear schedules. Impute-MACFM combines: (i) stability penalties on observed positions, (ii) consistency regularization enforcing local invariance, and (iii) time-decayed noise injection for numeric features. Inference uses constraint-preserving ordinary differential equation integration with per-step projection to fix observed values, optionally aggregating multiple trajectories for robustness. Across diverse benchmarks, Impute-MACFM achieves state-of-the-art results while delivering more robust, efficient, and higher-quality imputation than competing approaches, establishing flow matching as a promising direction for tabular missing-data problems, including longitudinal data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new framework for tabular imputation by integrating a mask-aware conditional flow-matching framework. MVI is indeed an important problem in many application areas. Since this is a classic and traditional research question, in my opinion, a newly proposed method should be considered significant only when it has a significant contribution compared to classic methods, and when the trade-off between computational cost, the drawbacks of black-box models, and improvements in MVI is sufficient. Therefore, my comments will focus on how the paper justifies that the proposed method is really significant.

### Strengths
1. The baselines are sufficient, and the evaluation of MVI methods is reasonable (checking three different missingness mechanisms), covering a diverse set of datasets and conditions.

2. Overall, the writing is clear and easy to follow.

### Weaknesses
The paper's evaluation lacks persuasive statistical evidence to support its claims of superiority. The stated protocol of using only ten random masks is an insufficient sample size for making robust, generalizable claims. Furthermore, the main figures and tables omit variance measures, making it impossible for a reader to determine if the Impute-MACFM's performance gains are statistically significant or merely a result of random chance. A sufficient evaluation would require more replications, proper statistical tests, and simulations across varied settings (like changing sample size or data dimension). Ideally, the evaluation should also be extended beyond imputation error (data recovery) to assess the impact on downstream tasks.

Minor issues:

1. The figures are somewhat unclear due to small font size and formatting limitations in the ICLR template.
For instance, in Figure 2, the performance of the proposed method on the Bean dataset is not clearly presented—the color label appears to be incorrect.

2. For future revisions, please check whether these can be improved. In addition, Figure 1 appears to have rendering issues when viewed in MacBook Preview; please verify and correct this in future versions.

### Questions
1. Could the authors provide additional details to address the statistical concerns raised in the "Weaknesses" section? 


2. For the general motivation of the paper, it is important to convince the audience that modern generative models have significant advantages that traditional methods cannot achieve. It would be helpful if the authors could provide more existing evidence from the literature to justify this point.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Impute-MACFM, a missing data imputation approach based on generalized flow-matching. It is featured by stability penalties on observed positions, consistency regularization enforcing local invariance, and time-decayed noise injection for numeric features. Experiments are conducted on public datasets to showcase the validity.

### Strengths
This paper investigates flow matching, which is a prevalent and powerful generative model in recent studies.

This paper investigates imputation performance over different datasets and missing patterns (MAR, MNAR, MCAR).

This paper add efficiency for comparison which is an important aspect in real-world application.

### Weaknesses
The related works need to be carefully polished and improved in terms of coverage and structure. For example, in section 2, there is only one subsection: 2.1. The listed 5 bullet points are mutually isolated. In each bullet point, the provided information, especially information density is limited (for example, only 3 diffusion-based methods are reviewed in lines 110-120).

The definition in lines 167-182 is a little hard to follow. It would be beneficial to add a toy example or a diagram figure for it.

The proposed method claimed that a primary methodology contribution is to modify FM by constraining velocity to remain near zero on observed entries, which could be more like an engineer trick instead of research innovation. 

It is unclear whether the constraint-enforcing algorithm could be further simplified, for instance, through a straightforward masking strategy applied at each step. Additionally, the proposed predictor–corrector paradigm appears to be highly influenced by classical convex optimization methods, such as Mehrotra’s predictor–corrector algorithm. Therefore, it is necessary for the authors to provide proper references to relevant literature and clearly justify how their approach differs from or extends these established methods.



Experiments. There are a lot of aspects for this paper to improve in terms of experiments. A fraction of them are listed as the following bullet points for the convenient reference of authors.
- Standard error is missing in experimental results which makes it infeasible to ensure the significance of the results (Tables 3-14). Moreover, statistical test would be necessary in these Tables to showcase the statistical significance.
- The imputation error in the tables seems to be large. Authors should clarify how their results are consistent with respect to prior baselines: whether the baseline performance is consistent to their officially reported values? If not, it would be of great importance to explain what makes the inconsistency.
- The missing ratio seems to be 0.3. It would be necessary to investigate performance given different missing ratios.
- Some performance is not reported. For example, in Table 6, on the Bean dataset, the performance of mean is not reported but that of Median is reported. What makes Median work but Mean fail?
- The performance of MICE is inferior to Mean. It would be a bit strange since MICE with reasonable hyper-parameter configs would be competitive. The performance of MIWAE is also a little strange.
- In table 1, imputation time is reported per dataset. It would be necessary to jointly compare  the efficiency and accuracy. The baselines to compare should cover representative non-generative models.
- Presenting ablation study in tabular format with statistical test and standard error would be more concise to support the claims.
- Authors claim three contributions in abstract: (i) stability penalties on observed positions, (ii) consistency regularization enforcing local invariance, and (iii) time-decayed noise injection for numeric features. But in ablation study, three different aspects are examined: (i) trajectory trials, (ii) Mask-aware conditioning, (iii) ODE steps.

A reproducible codebase is not provided, which is essential for assessing the reproducibility of this study. This concern is amplified by the omission of hyperparameter details for both the baselines and the proposed method across the datasets, as well as the lack of clarification on whether the reproduced baseline performances align with their officially reported, well-established results. Therefore, it is necessary for the authors to release the implementation of both the baselines and the proposed method, and to provide comprehensive details on hyperparameter selection and tuning procedures across all datasets.

### Questions
Please see the weakness window.

### Soundness
2

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
4

### Summary
This paper proposes Impute-MACFM, a flow matching method that works on tabular data with missing values. It achieves SOTA-comparable performances with only pretty simple backbone architecture.

### Strengths
The paper is well-written and logically organized. The motivation for applying flow-matching to tabular imputation is clear, and the presentation of notation and algorithmic steps is clean. The introduction does a good job summarizing challenges in tabular imputation and positioning flow matching as an efficient alternative to diffusion models. The method description (mask partitioning, schedule-consistent velocity field, and constraint-preserving ODE solver) is clearly explained and mathematically consistent.

### Weaknesses
- I don't think this paper contains theoretical/technical novelty. It applies flow matching on tabular data, but nothing proposed besides some tricks from the authors. While the paper claims a “mathematically principled velocity field formulation,” the derivations (e.g., Eq. 2–3) are a direct restatement of standard flow-matching objectives with masks added as multiplicative indicators. No formal proof or novel property is introduced beyond existing formulations in Albergo et al. (2023) and Lipman et al. (2023). The “schedule-consistent velocity” is essentially a straightforward application of the chain rule. Thus, the contribution feels incremental rather than conceptually new.
- Empirical results are poorly presented. Authors chose to visualize in bar graphs which is extremely hard to see, even if they have table in appendix. Maybe this is because their empirical performance is not great. Bold numbers (usually used to indicate best performance) in the table is wrong, this method seems not great empirically. (which totally makes sense)

### Questions
- Novelty clarification: What is the exact theoretical advancement beyond standard conditional flow matching? Is there any property (e.g., convergence guarantee, bias reduction) that holds only under the proposed mask-aware formulation?
- Evaluation scope: Why are only MAE/RMSE reported? Would predictive downstream accuracy or uncertainty calibration tell a different story?

### Soundness
2

### Presentation
3

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
This paper proposes Impute-MACFM, a mask-aware conditional flow matching framework for tabular data imputation to address different missing mechanisms, including MCAR, MAR, and MNAR. The proposed method contains three components: (i) stability penalties on observed positions, (ii) consistency regularization enforcing local invariance, and (iii) time-decayed noise injection for numeric features. Empirical evaluations are performed using eight public tabular datasets and three private NIH datasets.

### Strengths
- Overall, utilizing flow matching for tabular data imputation is interesting and original.
- The schedule-consistent velocity field and the constraint-preserving ODE solver, which constrain the learning of FMs in some particular dimensions, are interesting domain-specific adaptations for tabular data. They look technically sound and original to me.
- The proposed methods, as well as the training objectives, are described in detail, enhancing the reproducibility.
- The paper utilizes 11 datasets (8 of which are public) to do empirical evaluations, and promising results are observed on many datasets. 
- The efficiency of the proposed method is empirically validated where the ODE steps K=10 achieves similar imputation performance (RMSE) as K=50, while being around 7x faster, the efficiency is important in many domains where fast or even real-time imputation is desired.

### Weaknesses
- The paper argues its abstract explicitly that addressing MAR and MNAR is among its objectives, but I do not quite follow how MNAR is addressed in the proposed model. It seems to me that the handling of different missing mechanisms relies mainly on random masking. The proposed method does not model $p(\text{missing mask}|X)$; it is unclear to me how the proposed approach methodologically addresses the MNAR problem.
- Fig. 2 and Fig. 3 are difficult to read due to the dense organization and the color schemes. I suggest that the authors consider other ways to present the results instead of using this bar graph. Besides, a few methods are missing on a few datasets from the figures, for example, HyperImpute and EM on the News dataset.
- There is no quantification of the uncertainty for the experimental results to allow assessments on model stability.
- The downstream utility of the proposed imputation method is not measured. Since the datasets used have annotations spanning tasks of classification and regression, it is interesting and important to understand if the improvement of imputation from the proposed model could be translated into better downstream task performances and to what extent.

### Questions
1. How does the proposed method address MNAR?
2. Why are some methods missing from Fig. 2 and Fig. 3?
3. Is the model training stable? Are there any uncertainty measures, such as error bars or standard deviation, for the results?
4. Does the proposed method improve downstream prediction tasks?

Overall, I think this paper is interesting and makes some meaningful contributions to the field. I am willing to raise my scores if the weaknesses and questions above are completely addressed.

### Soundness
3

### Presentation
2

### Contribution
2
