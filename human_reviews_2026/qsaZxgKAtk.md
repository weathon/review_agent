# Missing Pattern Recognized Diffusion Imputation Model for Missing Not at Random

- Decision: Reject
- Scores: 6, 6, 6, 6, 4

## Abstract
Missing data frequently arises across diverse domains, including time-series and image domains. In the real world, missing occurrences often depend on the unobservable values themselves, which are referred to as Missing Not at Random (MNAR). To address this, numerous generative models have been proposed, with diffusion models in particular demonstrating strong capabilities in out-of-sample imputation. However, most existing diffusion-based imputation approaches overlook the MNAR setting and instead rely on restrictive assumptions about the missing process, thereby limiting their applicability to practical scenarios. In this work, we introduce the Missing Pattern Recognized Diffusion Imputation Model (PRDIM), a novel framework that explicitly captures the missing pattern and precisely imputes unobserved values. PRDIM iteratively maximizes the likelihood of the joint distribution for observed values and missing mask under an Expectation-Maximization (EM) algorithm. In this sense, we first employ a pattern recognizer, which approximates the underlying missing pattern and provides guidance during every inference toward more plausible imputations with respect to the missing information. In various experimental settings, we demonstrate that PRDIM achieves the state-of-the-art performance compared to previous diffusion imputation approaches under MNAR setting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To address the limitations of existing diffusion-based imputation models that overlook MNAR (Missing Not At Random) scenarios and rely on strong assumptions about missing patterns, this paper proposes PRDIM. The method introduces a pattern recognizer capable of identifying missing patterns in the data and employs the Expectation-Maximization (EM) algorithm to iteratively maximize the likelihood of the joint distribution of observed values and the missing mask. Extensive experiments on multiple datasets are conducted to verify the effectiveness of the proposed model.

### Strengths
S1: This paper conducts extensive experiments, covering multiple baselines, diverse datasets, and various imputation scenarios.

S2: This paper uses clear notations, solid theoretical derivations, and well-crafted figures to effectively illustrate its ideas.

### Weaknesses
W1: The experiments are conducted by simulating MNAR missing patterns on complete real-world datasets, rather than using datasets that contain naturally occurring missing values.

W2: As the proposed method belongs to the class of diffusion-based models, it lacks evaluation metrics such as CRPS to assess the quality of imputation results.

### Questions
Q1: Could the authors explain why missing patterns in real-world data are considered to be more consistent with the MNAR assumption? Is it possible to conduct experiments on datasets with naturally occurring missing values to demonstrate that the simulated MNAR patterns resemble natural ones? Alternatively, could the naturally missing patterns be used to artificially simulate missing values for evaluation?

Q2: This paper focuses on the MNAR scenario and notes in Table 5 that the advantage of PRDIM diminishes under the MCAR setting. Could the authors further explore the model’s performance under the MAR scenario? In addition, it would be helpful to clarify in which types of missingness patterns or data distributions the proposed pattern recognizer is applicable, and under what circumstances it may not be effective. Please further discuss the limitations of PRDIM in such contexts.

Q3: In Table 1, how is the imputation error computed for the original missing entries, and how is the ground truth obtained? In addition, could the authors clarify how the out-of-sample and in-sample imputation settings are specifically configured in the experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the application of diffusion model-based imputation methods under the missing not at random (MNAR) scenario. To address the challenges posed by MNAR, the authors propose the Pattern Recognized Diffusion Imputation Model (PRDIM), which incorporates a pattern recognizer to explicitly model the missingness pattern and mitigate the bias associated with MNAR. The authors further demonstrate that guidance from this pattern recognizer provides additional, approximate information for imputing missing values. Extensive experiments are conducted to validate the effectiveness of the proposed approach.

### Strengths
1. The topic of missing data imputation under the MNAR setting, combined with the use of diffusion models, is both timely and of significant interest to the ICLR community.
2. Theoretical derivations presented in the paper are rigorous and well-formulated.
3. The experimental results are comprehensive and adequately support the theoretical claims.

### Weaknesses
1. The notation should be revised for consistency and clarity. For example, $\mathcal{L}\_{diff}$ should be written as $\mathcal{L}\_{\text{diff}}$.
2. The conditions under which the assumptions in Proposition 3.1 hold should be described in greater detail. In particular, could the authors analyze the condition $q(X_{1:T}\vert X_0,M) = q(X_{1:T}\vert X_0)$ throughout the noisy process of the diffusion models?
3. According to reference [1], the diffusion process promotes diversity, which may not be ideal for imputation tasks. How does the proposed approach address this issue? Furthermore, reference [2] utilizes the median value for imputation; how does the proposed method estimate the imputed values?
4. Why did the authors choose DDPM-based diffusion models for their approach? What would be the implications of using VP-SDE-based diffusion processes, as discussed in reference [3]? Additionally, while the authors cite Theorem 1 from reference [3], it is proven in the context of the VP-SDE framework. Is the proposed method compatible with the DDPM-based diffusion models used in this work?
5. Since standard deviations for each model are reported, it would be beneficial to conduct paired-sample $t$-tests to more robustly support the theoretical claims.
6. For MNAR scenarios, reference [4] introduces MIRACLE, which post-adjusts imputed values. Could the authors compare their approach with MIRACLE?
7. The comparisons primarily focus on RMSE MAE, and MRE. Could the authors also consider metrics that evaluate distributional similarity, such as the Wasserstein distance?

---
References:  
[1]. Self-Supervision Improves Diffusion Models for Tabular Data Imputation  
[2]. CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation  
[3]. Diffputer: Empowering diffusion models for missing data imputation.  
[4]. MIRACLE: Causally-Aware Imputation via Learning Missing Data Mechanisms

### Questions
Please refer to the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes PRDIM (Pattern Recognized Diffusion Imputation Model), a diffusion-based imputation framework designed to handle the Missing Not At Random (MNAR) problem. PRDIM integrates an Expectation-Maximization (EM) framework with a pattern recognizer and provides gradient-based guidance during the diffusion denoising process. Experiments show that PRDIM
achieves the state-of-the-art performance compared to previous diffusion imputation approaches under the MNAR setting.

### Strengths
1. The paper focuses on the Missing Not At Random (MNAR) setting, which reflects real-world missing mechanisms but has been largely neglected in prior diffusion-based imputation studies.

2. The method jointly models the data distribution and the missing pattern within an EM-based diffusion framework, offering a coherent and theoretically grounded solution to MNAR imputation.

3. Extensive experiments across four benchmark datasets show that the proposed method achieves consistent performance improvements over strong baselines, validating the effectiveness of the approach.

### Weaknesses
1.  Figure 1 lacks sufficient guidance on how to interpret the plots. The authors should provide a more detailed explanation of the axes, the meaning of color codings, and the overall message being conveyed by the figure.

2. From my understanding, Phase 1 mainly serves as an initialization for the diffusion model before the EM iterations begin. An ablation study examining the necessity and impact of this pretraining phase would strengthen the paper.

3. The derivation of equation 15 from proposition 3.2 seems not clear. Specifcially,  where does the $\frac{1-\alpha_t}{\sqrt{\alpha_t}}$ come from?

4. Since both the proposed method and DiffPuter are based on the EM framework, it would be better to include a more thorough discussion on a head-to-head comparison of design choices, etc. My understanding is that DiffPuter's diffusion model directly learns the joint distribution of X_mis and X_obs, while this work's diffusion model learns the conditional distribution p(X_mis|X_obs) like CSDI. The authors should clarify the rationale and potential advantages of this design choice.

### Questions
1. Can the author give more explanation on what Soft and hard EM are?

2. Why modify DiffPuter into a conditional diffusion framework instead of using the original form?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper focuses on addressing the Missing Not at Random (MNAR) problem, a common issue in missing data across time-series and image domains where missing occurrences depend on unobservable values. It points out that while existing diffusion models excel in out-of-sample imputation, most overlook the MNAR setting and rely on restrictive assumptions about the missing process, limiting practical applicability. To solve this, the paper proposes the Missing Pattern Recognized Diffusion Imputation Model (PRDIM), a novel framework that explicitly captures missing patterns and accurately imputes unobserved values. PRDIM leverages the Expectation-Maximization (EM) algorithm to iteratively maximize the likelihood of the joint distribution of observed values and missing masks, and incorporates a pattern recognizer to approximate the underlying missing pattern and guide inference for more plausible imputations. Experimental results show that under the MNAR setting, PRDIM outperforms previous diffusion-based imputation approaches.

### Strengths
The paper directly addresses the understudied challenge of Missing Not at Random (MNAR) imputation, a prevalent but overlooked setting in real-world data where missingness depends on unobserved values. By contrast, most existing diffusion-based imputation methods rely on restrictive assumptions that fail to reflect practical scenarios—this focus on MNAR fills a meaningful niche in the literature.
The paper demonstrates state-of-the-art (SOTA) performance of PRDIM against existing diffusion-based imputation methods under MNAR settings. This empirical rigor is critical: it validates that the model’s theoretical focus on MNAR translates to tangible improvements in imputation quality, making the work credible for researchers and practitioners working with incomplete data.

### Weaknesses
MNAR encompasses diverse subtypes, but the paper does not clarify which MNAR scenarios PRDIM excels at. For example, it does not test PRDIM on data with varying degrees of MNAR severity or different missingness triggers—limiting conclusions about its applicability beyond the specific experimental settings studied.
While the pattern recognizer is framed as a core component of PRDIM, the paper provides insufficient detail on its internal design  and how it specifically learns to model MNAR mechanisms. Without ablation studies or qualitative examples of its outputs, readers cannot fully assess whether this component is driving the model’s SOTA performance—or if it is redundant with other parts of the framework.

### Questions
Did you test PRDIM on these distinct MNAR subtypes, and if so, could you share results on which subtypes PRDIM performs best/worst? If not, do you have hypotheses about PRDIM’s limitations when facing MNAR mechanisms different from those in your current experiments?
Could you clarify whether the MNAR mechanisms in your test data align with common real-world scenarios?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PRDIM (Missing Pattern Recognized Diffusion Imputation Model), a diffusion-based framework for imputing missing values under the Missing Not at Random (MNAR) setting. Unlike most existing imputation models that assume missingness is random or only dependent on observed values, PRDIM explicitly models the missing pattern using a pattern recognizer within an Expectation-Maximization (EM) framework.

### Strengths
1. The paper targets the Missing Not at Random (MNAR) scenario, which is more realistic yet underexplored in diffusion-based imputation research, addressing a clear research gap.

2. The proposed PRDIM model a pattern recognizer within an EM-based diffusion framework, providing a principled way to model and leverage the missing pattern during inference.

### Weaknesses
1. One of the strong motivations of the paper is that missing patterns, such as Missing Not at Random (MNAR), are more common in real-world scenarios. However, the data used in the subsequent imputation tasks are artificially split from complete datasets such as ETT, PEMS, and Stock, and do not actually use real-world missing data.

2. The proposed model is intended to be a time-series imputation model, but the validation is done on static image data from Fashion MNIST. It is recommended to use a time-series dataset with temporal variations, such as Moving-MNIST, for validation.

3. PRDIM is an imputation model for time-series data. For time-series datasets, there is a significant issue of temporal shift, i.e., the distribution of data in the training set differs greatly from that in the test set. However, PRDIM does not provide a specific solution to address this issue.

4. The terms "out-of-sample imputation tasks" and "in-sample imputation tasks" are not clearly defined in the main text.

5. PRDIM's performance heavily relies on the accuracy of the pattern recognizer. If the missing patterns are estimated inaccurately, it could affect the final imputation accuracy. This could be especially problematic for time-series datasets with significant distribution shifts, as it may lead to overfitting on the training data.

6. Although the pattern recognizer provides an estimate of the missing pattern, there is a lack of visual interpretability and analysis of how the generated imputations are guided by the pattern. This was one of the motivations of the paper, but it was not well validated in the experimental section.

7. The diffusion backbone and missing pattern recognizer require joint optimization, which may lead to high training costs. The paper lacks an analysis of the model's efficiency.

### Questions
1. Is the ETT dataset using ETTH1, ETTH2, or ETTM1, ETTM2?

2. In addition, there are many more complete and standardized forecasting datasets, such as Traffic or Electricity. Why use such a small ETT dataset?

### Soundness
3

### Presentation
2

### Contribution
2
