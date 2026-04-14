# Conformalized Survival Analysis for General Right-Censored Data

- Decision: Accept (Poster)
- Scores: 3, 8, 8, 3

## Abstract
We develop a framework to quantify predictive uncertainty in survival analysis, providing a reliable lower predictive bound (LPB) for the true, unknown patient survival time. Recently, conformal prediction has been used to construct such valid LPBs for *type-I right-censored data*, with the guarantee that the bound holds with high probability. Crucially, under the type-I setting, the censoring time is observed for all data points. As such, informative LPBs can be constructed by framing the calibration as an estimation task with covariate shift, relying on the conditionally independent censoring assumption. This paper expands the conformal toolbox for survival analysis, with the goal of handling the ubiquitous *general right-censored setting*, in which either the censoring or survival time is observed, but not both. The key challenge here is that the calibration cannot be directly formulated as a covariate shift problem anymore. Yet, we show how to construct LPBs with distribution-free finite-sample guarantees, under the same assumptions as conformal approaches for type-I censored data. Experiments demonstrate the informativeness and validity of our methods in simulated settings and showcase their practical utility using several real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a reliable lower predictive bound (LPB) for the survival time under the general right-censored data based on the work in Gui et al. (2024). The proposed LPB is a PCA-type with distribution-free finite-sample guarantee.

### Strengths
1.	The LPB is doubly robust by theorem.
2.	The paper extends the result in Gui et al. (2024) to the general right-censoring setting.

### Weaknesses
1.	I don't see a strong motivation for using the general right-censoring setting, which appears to be quite similar to the Type-I right-censoring setting. Other than two citations (which only mention the definition), the authors do not provide a clear explanation of why the general right-censoring setting is timely or important in real-world data.
2.	The entire framework builds on Gui et al. (2024), except for the miscoverage estimators. The proposed focus calibration and fused calibration use only part of the information in the calibration set, which may lead to a waste of information. Additionally, I have concerns about the sufficiency of the proposed calibrations in cases of a large imbalance between censored and uncensored data in the calibration set (e.g., when the censoring rate is high). Therefore, there are doubts regarding the novelty of the proposed method.
3.	Many parts of the paper are derived from Gui et al. (2024), including the main idea, the theoretical results, the simulations, and the proofs. Several sections are not clearly explained in this paper, requiring a read of Gui et al. (2024) to understand the context. For example, why is a PAC-type LPB considered instead of a marginally calibrated LPB? How does the PAC-type LPB feature in your method? As a result, phrase like "Notably, ..." in line 162-163 is vague and confusing.
4.	Section 3.2 presents a list of results without a clear interpretation of the underlying ideas. The nested notation further complicates the readability of the section.
5.	In your experiments, there is no comparison between your method and other methods, except the naïve LPB. The experimental section is mainly a collection of results without sufficient interpretations. For instance, the focused and fused LPBs are not consistently better than the naïve LPB in settings 4 and 6. In the real data experiment, the focused and fused LPB are not significantly better than the naïve LPB.
6.	The data used in your experiments does not reflect a general right-censoring setting.
7.	Some notations in the proof are clunky and confusing. For example, in the proof of Lemma B.1, $q=q(X)$ in your $\mathbb{P}_x(\cdot)$ 

is deterministic? In Section B.2, the form 
$\mathbb{E}[\mathbb{P}(\tilde{T}<\hat{q}_{\tau}(X)|X=x,e=1)]$ 

does not make sense, 
as $\hat{q}_{\tau}(X)$ is not random given $X=x$.

8. Some references are not in correct format. For example, the arxiv references should be in arxiv format.

### Questions
1.	How do you handle the extreme $\hat{\omega}$?
2.	How does the estimator $\hat{s}_{\tau}(x)$ affect your method?
3.	The fused and focused LPB may not as tight as the LPB in Gui et al. (2024), then how loss are they? Can you quantify?
4.	In what settings, the fused and focused LPB are not better than the naïve LPB?
5.	Why the authors not include the average LPB in the experiments?
6.	What is the coverage result of the real data experiment? Can you run experiments on other real data set? For example, the data used in Candes et al. (2023) and Gui et al. (2024).


Emmanuel Cande`s, Lihua Lei, and Zhimei Ren. Conformalized survival analysis. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(1):24–45, 2023.

Yu Gui, Rohan Hore, Zhimei Ren, and Rina Foygel Barber. Conformalized survival analysis with adaptive cut-offs. Biometrika, 111(2):459–477, 2024.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The manuscript is well-structured, offering solid theoretical support and a thorough experimental evaluation. The authors present a framework to quantify predictive uncertainty in survival analysis with general right-censoring, specifically addressing cases where the censoring time is not directly observed. The main contribution lies in adapting conformal prediction methods to this broader right-censoring context, providing a lower predictive bound (LPB) for patient survival times. Unlike previous approaches that assume known censoring times, this work extends the methodology to general right-censored data, where either the censoring or survival time is observed, but not both.

### Strengths
- The paper extends conformal prediction methods to general right-censored data, addressing a significant gap where censoring times are unknown.
- The framework is backed by strong theoretical support, providing distribution-free, finite-sample guarantees for lower predictive bounds.
- The authors provide both simulated and real-world data experiments, showcasing the method’s validity and practical effectiveness.

### Weaknesses
- In Section 4.1, the proposed methods appear to underperform compared to the naive method in Setting 6 regarding the lower predictive bound (LPB). Could the authors provide additional insights into why the naive method performs exceptionally well in this specific setting?

### Questions
- This is a general question. Why do researchers tend to prioritize the lower predictive bound rather than the upper predictive bound for survival time? Is there any existing work on the upper predictive bound for survival time?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper extends on Gui et al.'s paper (Conformalized survival analysis with adaptive cut-offs), to construct valid and reliable lower predictive bound for survival analysis in the more general right-censoring settings, where either censoring time or event time can be observed but not both. The paper provides distribution-free finite-sample guarantees under some necessary conditions, and provides some experiments to demonstrate the effectiveness.

### Strengths
This paper is well-written and a pleasure to read. It tackles the important topic of achieving calibrated (probably approximately correct) predictions in survival analysis with the right censoring settings, an area where previous research has not provided methods with finite-sample distribution-free guarantees. The authors try to fill this gap by utilizing the concept of covariate shift. The paper is well-motivated and introduces three methods to address the problem. Starting with the simplest approach, it gradually highlights the limitations of each method and proposes solutions to overcome them. The authors provide finite-sample guarantees for the fused method, and the results on synthetic datasets support the theoretical findings.

### Weaknesses
Despite all the merits I mentioned, I have concerns regarding the experimental results. The experiments are limited to oversimplified synthetic datasets and a single small real-world dataset, without comparisons to existing methods. This raises questions about the practical effectiveness and generalizability of the proposed approach.

Below are my major concerns with the paper:
1. **Lack of Comparison with Existing Methods:** 
The paper does not compare the proposed method with any existing approaches. Although it is the first to provide finite-sample guarantees, this does not inherently mean it outperforms methods without such guarantees. Previous studies, such as Cresswell et al. [1], have shown that methods lacking finite-sample guarantees can perform equally well or better in practice. It is crucial to include comparative analyses with methods like those proposed by Qin et al., Meixide et al., and Qi et al. [2] to compare the empirical performance of the proposed method.
2. **Simplistic Synthetic Datasets:** The main claim relies solely on 6 synthetic datasets with simplistic settings where survival and censoring times follow lognormal and exponential distributions. These settings are not reflective of real-world scenarios. For example, event and censor distributions can be non-parametric and censoring can stem from various sources (e.g., a combination of loss to follow-up, competing risks, and administrative censoring). I recommend designing more complex synthetic datasets that better mimic the complexities encountered in practice.
3. **Inadequate Evaluation on Real-World Data:**
The evaluation of the real-world dataset is insufficient. The TCGA dataset used is small (n=1044), and the chosen evaluation metrics do not adequately support the paper's main findings. There are established metrics for assessing coverage and calibration in censored datasets, such as the distribution calibration metric discussed in Haider et al. [3]. The distribution calibration is proved to be asymptotically unbiased under the conditional independent assumption (Assumption 1.1 in this paper). Although it is not perfect (as it doesn't have a finite sample guarantee), incorporating such metrics would strengthen the validity and reliability of the experimental results.
4. **Sensitivity to High Censoring Rates:** 
The proposed method employs inverse probability weighting, which is known to have high variance and be sensitive to the number of uncensored subjects, especially in datasets with high censoring rates. The paper should report the censoring rates in both synthetic and real datasets and assess the method's performance across varying levels of censoring. Additionally, testing on larger datasets would provide more robust evidence of the method's practical applicability.
5. **Challenges with Double Robustness in Practice:** While the double robustness property is theoretically appealing, achieving either condition in practical settings is challenging. The paper should discuss the practical implications of this limitation and explore potential strategies to mitigate its impact.

---

Below are my additional thoughts on the lower predictive bound (LPB) and marginal coverage/calibration:

Achieving a Probably Approximately Correct (PAC) lower predictive bound is important, but focusing solely on calibration or coverage is insufficient. Simple methods like the Kaplan-Meier estimator can provide calibrated predictions without the need for complex modeling. For example, a PAC-type LPB at 90% can be contained by finding the time when the KM curve drops below 90%.  A more important question is: how does the proposed method affect (or possibly decrease) other performance metrics by achieving a better calibration/coverage? Some important metrics include ranking accuracy (Concordance index), probabilistic accuracy (Brier score), and time-to-event prediction accuracy (mean absolute/squared error).

The requirement of a separate calibration set reduces the number of samples available for training, which could negatively impact predictive performance. Qi et al. [2] proposed a conformal-based method that addresses this issue by enhancing calibration without sacrificing other metrics. Discussing such alternatives would provide valuable context and highlight potential limitations for the current manuscript for improvement.

Moreover, providing only the LPB may not offer sufficient information for real-world applications compared to generating survival or hazard functions (which are the standard prediction outcomes for survival analysis models). These functions provide a comprehensive view of the time-to-event distribution, which is crucial for decision-making in clinical and other applied settings. LPB is a degenerated output that can be obtained from the survival function while losing a lot of other information.


[1] Cresswell et al., Conformal Prediction Sets Improve Human Decision Making. ICML 2024

[2] Qi et al., Conformalized Survival Distributions: A Generic Post-Process to Increase Calibration. ICML 2024

[3] Haider et al., Effective Ways to Build and Evaluate Individual Survival Distributions. JMLR 2020


---
**Update after rebuttal**
Thank you for your comprehensive rebuttal and the additional experimental results on complex synthetic datasets, real-world datasets, and supplementary baselines. Your responses have effectively addressed my concerns. I find the current draft to be a strong paper and am now more inclined toward acceptance. Accordingly, I am revising my rating to 8.

### Questions
Lines 294-295 state "inaccurate estimation of $\hat{s}_\tau$ does not impact the validity of the calibration procedure", what do you mean by that exactly? Theorem 3.1 states that the calibration is achieved with *accurate weights*, which seems to contradict the first statement.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces an extended conformal survival analysis framework specifically designed to handle general right-censored data, aiming to provide a reliable lower predictive bound (LPB) for patient survival times. The main contribution of the paper is the development of a novel, doubly robust calibration method that constructs valid LPBs with finite-sample guarantees, even under uncertain censoring conditions.

### Strengths
The paper proposes a doubly robust calibration method capable of constructing valid lower predictive bounds (LPBs) for general right-censored data, addressing a gap in survival analysis. The proposed method provides finite-sample guarantees, which are particularly important for ensuring model reliability in high-stakes survival analysis applications.

### Weaknesses
Although the paper introduces a doubly robust calibration method, it fundamentally relies on existing conformal prediction methods (e.g., the work by Gui et al., 2024). The real innovation lies in extending these techniques to the general right-censored data setting, representing a relatively incremental contribution. The comparisons in the paper are primarily against a relatively simple "naive" calibration method rather than other state-of-the-art survival analysis methods.

### Questions
1. In the "focused calibration" and "fused calibration" methods, the estimation of weights w(x) is crucial. How much impact does error in weight estimation have on the prediction results? Are there established standards for evaluating the accuracy of weight estimation, or strategies to reduce its sensitivity?

2. In the experiments, the proposed method is compared with the "naive method." Are there other commonly used methods for handling censored data that could serve as benchmarks? For example, traditional weighted Cox regression models or other survival analysis methods.

3. What is the computational complexity of the proposed "fused calibration" method when handling large-scale datasets in practice? How does its computational efficiency compare to other methods?

4. The proposed method relies on the estimation of weights and conditional quantiles; however, the article does not include a sensitivity analysis of these critical hyperparameters. How do the hyperparameters impact model performance? Is it possible to provide a robust hyperparameter selection strategy?

5. The effectiveness of doubly robust calibration depends on the accuracy of conditional quantile or weight estimation. Could explicit theoretical bounds be provided to describe the calibration method’s performance under inaccurate estimations? It is recommended to clearly specify how such biases impact the final coverage.

### Soundness
2

### Presentation
2

### Contribution
1
