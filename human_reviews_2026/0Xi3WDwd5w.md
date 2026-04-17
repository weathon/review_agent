# Overlap-weighted orthogonal meta-learner for treatment effect estimation over time

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Estimating heterogeneous treatment effects (HTEs) in time-varying settings is particularly challenging, as the probability of observing certain treatment sequences decreases exponentially with longer prediction horizons. Thus, the observed data contain little support for many plausible treatment sequences, which creates severe overlap problems. Existing meta-learners for the time-varying setting typically assume adequate treatment overlap, and thus suffer from exploding estimation variance when the overlap is low. To address this problem, we introduce a novel overlap-weighted orthogonal WO meta-learner for estimating HTEs that targets regions in the observed data with high probability of receiving the interventional treatment sequences. This offers a fully data-driven approach through which our WO-learner can counteract instabilities as in existing meta-learners and thus obtain more reliable HTE estimates. Methodologically, we develop a novel Neyman-orthogonal population risk function that minimizes the overlap-weighted oracle risk. We show that our WO-learner has the favorable property of Neyman-orthogonality, meaning that it is robust against misspecification in the nuisance functions. Further, our WO-learner is fully model-agnostic and can be applied to any machine learning model. Through extensive experiments with both transformer and LSTM backbones, we demonstrate the benefits of our novel WO-learner.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper tackles the challenge of estimating HTEs in time-varying settings, where treatment sequence overlap decreases with longer horizons. The authors propose a Weighted Orthogonal (WO) meta-learner, which introduces an overlap-weighted, Neyman-orthogonal population risk function to focus learning on regions with sufficient treatment overlap. This design stabilizes estimation variance and enhances robustness to nuisance model misspecification. The method is model-agnostic and demonstrated with transformer and LSTM backbones, showing improved reliability and accuracy over existing time-varying meta-learners.

### Strengths
* **Problem Significance:** The paper addresses a highly relevant and challenging problem at the intersection of causal inference and time-series analysis. Estimating HTEs over time is crucial for personalized medicine, and the issue of exponentially decreasing overlap is a core, practical barrier that the paper directly confronts.
* **Methodological Novelty:** The proposed WO-learner is methodologically novel and sound. The key idea of designing a population risk function that explicitly minimizes an *overlap-weighted* oracle risk, while simultaneously enforcing Neyman-orthogonality, is the primary technical contribution.
* **Theoretical Guarantees:** The method is supported by strong theoretical guarantees. The proof that the weighted population risk is Neyman-orthogonal with respect to all nuisance functions (including the newly defined weight functions) is a non-trivial and important result.
* **Experimental Design:** The paper presents a broad set of experiments, including four synthetic datasets designed to isolate specific challenges (low overlap, complex nuisance functions, small sample size) and a semi-synthetic experiment based on real-world MIMIC-III covariates.

### Weaknesses
* **Marginal Absolute Improvement:** While the paper reports large *relative* improvements the *absolute* improvements in RMSE are often in the $10^{-2}$ magnitude. The practical significance of such a small absolute gain is questionable, especially given the significant added complexity of estimating the new WO pseudo-outcomes and weight functions.
* **Flawed Real-World Validation:** The validation on real-world data is fundamentally flawed. The authors state this is a "real-world outcome estimation" task, which is a standard *prediction* (supervised learning) task, not a *causal inference* (HTE estimation) task. Demonstrating that the WO-learner is "not worse" than baselines on a predictive task provides **no evidence** of its efficacy for its stated purpose (HTE estimation). This experiment does not support the paper's core claims and raises questions about the method's applicability.
* **Omission of Key Baselines:** The paper omits a crucial and common baseline: a standard DR learner or IPW learner with **propensity score truncation (clipping)**. This is a very common heuristic to handle low-overlap regimes. It is unclear if the WO-learner's complexity is justified over this much simpler, strong baseline.
* **Lack of Intuition for Key Formulas:** The derivation of the key components, particularly the WO pseudo-outcomes $\xi_{t}^{\circ}$ and the weighting term $\rho_{t}^{\circ}$ in Definition 4.2, lacks intuition. They appear to be complex formulas derived to make the orthogonality proof work, rather than being motivated by a clear, first-principles-based explanation. This makes the method difficult to understand and build upon.
* **Unclear Practicality of New Nuisance Function:** The new weight function $\omega_{j}^{\overline{a}}$ (Eq 11) is itself a complex expectation of products of future propensity scores. The practical challenges and stability of estimating this new, non-trivial nuisance function (especially for long horizons $\tau$) are not sufficiently discussed.
* **Missing Discussion of Failure Cases:** The paper does not adequately discuss the potential failure cases or limitations of the WO-learner. When would this method *not* be preferred? How does it behave in very high-dimensional or extremely sparse data settings? A "Limitations" section is missing.
* **Overall Impression and Presentation:** Overall, while the problem is valid and the method is technically correct, the contribution feels incremental. The solution is an orthodox application of established causal inference principles (weighting, orthogonality) to the time-series setting, but the practical gains appear marginal, and the real-world validation is not fit-for-purpose. The work lacks a "surprising" element. Furthermore, the paper's presentation is overly dense, and it notably lacks a "Conclusion" section, which weakens the paper's summary and impact.

### Questions
**CATE vs. CAPO Weighting:** Why is the overlap weight for CATE (Eq 10) defined as the *product* of the individual propensity weights ($\omega^{\overline{a}}\omega^{\overline{b}}$)? Why not estimate the CAPO for each arm separately (using the CAPO propensity weight $\omega^{\overline{a}}$) and then simply subtract them? What is the theoretical motivation for this specific product-based weighting for CATE?

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
This paper proposes a new framework, Overlap-Weighted Orthogonal Learning (OWL), for estimating heterogeneous treatment effects in time varying settings where covariate overlap between treatment groups decreases exponentially with time horizon. The method incorporates the a propensity/overlap weights that up-weights samples with a higher probability to having the treatment sequence of interest. Moreover, the proposed estimator is Neyman orthogonal which makes the estimator insensitive to nuisance estimation error.

### Strengths
- Clear motivation and the proposed estimator addresses the problem of insufficient overlap. 
- The estimator is flexible and is robust to nuisance errors as well. 
- Performed comprehensive experiments that addresses different settings. Results demonstrates the effectiveness of the proposed method at addressing the different issues discussed in the paper.

### Weaknesses
- While the paper showed that the proposed population risk is Neyman orthogonal, papers on mete-learners usually also provide error analysis and show that the error terms from the nuisance functions are higher order, which is not presented in the paper. 
- The main innovation is to address the problem of limited overlap, so it would be nice to have some theorems that showcase how this estimator have better behavior (e.g. variance) in those regimes.

### Questions
- Do we expect the performance of the proposed estimator to deteriorate more that the DR-learner when propensity is hard to learn, as the weights also rely on the propensity estimates?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes an over-weighted meta-learner to estimate HTE in a very challenging time-varying scenario with a treatment sequence scenario. It also develops a new risk esimation to reduce the effects of first order bias in the estimation. The paper provides detailed theorectical analysis. The empirical results show that the proposed method achieve the best performance compared with other adjustment method.

### Strengths
- Compared with other meta-learners in this field, the study proposes a novel method to address the challenges of other learners in the low overlap scenario, which is common in the time varying treatment assignment strategy. 
- The overlap weighted orthogonal 
- The study has conducted both theorectical and empirical analysis to demonstrate the advantages of their proposed method.
- The results demonstrate dramatical improvement of the performance in Table 2. And the performance is uniformly the best over all the sample sizes.

### Weaknesses
- Can you please revise the writing of the problem setup? Maybe provide some examples to illustrate what the treatment looks like. Based on my understanding of your problem, the two sequences should be a list of 1 or 0, correct? i.e. $a = [1, 0, 0, 1], b = [0, 1, 1, 1]$
- One very relevant study as mentioned in the paper is the IVW method. As stated in the IVW paper, the framework is a composition of IVW and DR learner [1]. The paper provides very brief description in line 116-125. I wonder here is the IVW refers to the same model as the IVW-DR learner in the original paper or just the IVW only.  
- Another concern about the time-varying treatment strategy is that, the proposed method only considers confounders happen before $t$. But in the real-world setting, the propensity scores are usually depend on what happened during $t:t+\tau$. I know it makes the problem much harder, but it is very realistic in healthcare domain. 

[1] Frauen, D., Hess, K., & Feuerriegel, S. Model-agnostic meta-learners for estimating heterogeneous treatment effects over time. In The Thirteenth International Conference on Learning Representations.

### Questions
- In table 4, why the IVW performance get dramatically incresed when the prediction horizon increases? In the paper [1], the performance looks normal when $\tau=2,3$.

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
3

### Summary
The paper proposes an overlap-weighted orthogonal (WO) meta-learner to estimate time-varying heterogeneous treatment effects (HTEs) under low-overlap conditions. The framework aims to stabilize inverse-propensity-based estimators by re-weighting high-overlap regions.

### Strengths
- The paper proposes a model-agnostic framework for adjusting time-varying confounders.
- The paper provides a theoretical foundation for orthogonality.
- An implementation code is available for review.

### Weaknesses
- The claim that weighting by overlap improves estimator stability is weakly justified. By emphasizing high-overlap regions, the learner avoids extreme inverse-propensity weights but can ignore potentially outcome-informative regions. Is there a trade-off between propensity and outcome predictions?
- The motivation behind the study problem (treatment effect estimation over time) needs to be further clarified, particularly in relation to its real-world applicability. It is not evident whether the problem is testable, whether the authors have empirically validated it in real applications, or whether the underlying assumptions commonly hold in practical settings.
- The experimental validation is primarily based on synthetic data. Even in the semi-synthetic MIMIC-III experiment, both treatments and outcomes are simulated, so there is no demonstration of applicability to real observational data. The study does not evaluate whether estimating CATE over time provides practical benefits or is actually necessary or useful in real-world contexts. The study shows performance under a controlled scenario rather than demonstrating practical utility.
- The authors explicitly note that “factual outcome prediction is not the task our method is tailored for,” yet give no alternative validation criterion for real-world use. How would it be verified in practice? The authors should add a discussion and limitations on this point.
- The baselines are narrow, limited to RA, IPW, DR, and IVW  from [1]. The paper omits several relevant baselines for CATE over time, e.g., G-Net [2] and G-Transformer [3].

[1] Frauen, D., Hess, K., & Feuerriegel, S. Model-agnostic meta-learners for estimating heterogeneous treatment effects over time. In The Thirteenth International Conference on Learning Representations.

[2] Li, R., Shahn, Z., Li, J., Lu, M., Chakraborty, P., Sow, D., ... & Lehman, L. W. H. (2020). G-Net: a deep learning approach to G-computation for counterfactual outcome prediction under dynamic treatment regimes. arXiv preprint arXiv:2003.10551.

[3] Hess, Konstantin, Dennis Frauen, Valentyn Melnychuk, and Stefan Feuerriegel. "G-Transformer for Conditional Average Potential Outcome Estimation over Time." CoRR (2024).

### Questions
- The paper claims to estimate HTE but reports only CATE in the experiments. There is no discussion of heterogeneity across subgroups or covariate-dependent variation in effects. It may mislead readers about what “HTE” means in this context.
- Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
