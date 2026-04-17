# Learning Survival Distributions with Individually Calibrated Asymmetric Laplace Distribution

- Decision: Accept (Poster)
- Scores: 2, 8, 8, 6

## Abstract
Survival analysis plays a critical role in modeling time-to-event outcomes across various domains. 
Although recent advances have focused on improving _predictive accuracy_ and _concordance_, fine-grained _calibration_ remains comparatively underexplored. 
In this paper, we propose a survival modeling framework based on the Individually Calibrated Asymmetric Laplace Distribution (ICALD), which unifies _parametric_ and _nonparametric_ approaches based on the ALD.
We begin by revisiting the probabilistic foundation of the widely used _pinball_ loss in _quantile regression_ and its reparameterization as the _asymmetry form_ of the ALD. 
This reparameterization enables a principled shift to _parametric_ modeling while preserving the flexibility of _nonparametric_ methods.
Furthermore, we show theoretically that ICALD, with the _quantile regression_ loss is probably approximately individually calibrated.
Then we design an extended ICALD framework that supports both _pre-calibration_ and _post-calibration_ strategies. 
Extensive experiments on 14 synthetic and 7 real-world datasets demonstrate that our method achieves competitive performance in terms of _predictive accuracy_, _concordance_, and _calibration_, while outperforming 12 existing baselines including recent _pre-calibration_  and _post-calibration_ methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a survival analysis framework that augments the most recent asymmetric Laplace distribution-based model (Sheng & Henao, 2025) with calibration-oriented loss functions. The proposed model is demonstrated to have a profitable property of monotonic PAIC (probably approximately individually calibrated). Experiments on 14 synthetic and 7 real datasets show that it achieves substantial improvements in various accuracy metrics against conventional models.

### Strengths
- The idea of the paper is clearly presented, and the paper is easy to read.
- The validity of the proposed model was evaluated on a lot of (14 synthetic and 5 real-world) data sets.

### Weaknesses
- This paper largely follows the most related work of the asymmetric Laplace distribution (ALD) survival model (Sheng & Henao, 2025) in data sets, baselines, and protocol, and shows that tuning a calibration-oriented term/module in the objective further improves predictive performance. This is a reasonable but unsurprising outcome given ALD’s already strong empirical performance. The central idea of adding a calibration-encouraging term/module to the objective function has been established in prior works (e.g., X-CAL (Goldstein et al., 2020)). Thus, the contribution of this paper feels incremental from a technical viewpoint.
- Although Section 3.1 highlights that the pinball loss can be interpreted as the ALD’s negative log-likelihood in quantile form, it is unclear to me what practical advantage this brings here (e.g., training stability, improved optimization landscapes). If the authors insist that the pinball loss can be evaluated analytically, then this is not a non-trivial idea: the same holds for simple parametric distributions beyond ALD. I may be missing/misunderstanding some important aspect here, so if this property plays an essential role in substantiating this paper’s contribution, please explain it clearly.
- The ablation study in Table 1 is interesting, but the comparison of L_{ALD+Cal} vs. L_{X-CAL} does not seem to be appropriate. For example, we can consider ALD+X-CAL within the contribution of (Goldstein et al., 2020). 

Minor comments:
- The Related Work section, explicitly discussing how this paper relates to ALD (Sheng & Henao, 2025) and X-CAL (Goldstein et al., 2020), should appear in the main text.

### Questions
- I could not find a clear description of the better/worse/same evaluation procedure. Did you split each dataset into multiple folds for training and testing and then evaluate performance in a cross-validation manner?
- How was the regularization parameter in X-CAL (\gamma in the original paper) selected or tuned? Please specify the procedure.

### Soundness
2

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
The paper "Learning Survival Distributions with Individually Calibrated Asymmetric Laplace Distribution" introduces a novel framework called ICALD, which integrates parametric and nonparametric approaches using the Asymmetric Laplace Distribution (ALD) to enhance individual calibration in survival analysis. The authors provide a comprehensive methodology, theoretical foundation, and empirical validation of their approach.

### Strengths
- The integration of parametric and nonparametric methods through ICALD is a significant advancement in survival analysis. Further, the emphasis on individual-level calibration addresses an important gap in existing survival models, particularly relevant for high-stakes applications like patient prognosis.
- The paper establishes the model's Probably Approximately Individually Calibrated (PAIC) and Monotonically PAIC (MPAIC) properties, adding robustness to their claims.The claims were further backed by extensive experiments across 21 datasets demonstrate competitive performance against 12 baselines, with strong results in predictive accuracy, concordance, and calibration.
- The authors provide strategies to address overfitting and asynchronous convergence during pre-calibration, enhancing the model's practicality.

### Weaknesses
Some of the challenged with the paper are as below:
- The benefits of ICALD appear contingent on dataset characteristics, particularly for highly skewed distributions like LogNorm. This raises concerns about generalizability across diverse data types.
- The experimental results are based on a specific selection of datasets, and validation against an independent, diverse benchmark could improve confidence in the model's broader applicability.
- The paper does not extensively discuss scalability, which is crucial for large datasets or high-dimensional features in real-world applications.
- While the authors propose solutions for capturing extreme quantiles and distribution mismatch in tails, it's unclear if these methods are sufficient for extremely heavy-tailed distributions.  The experiments primarily focus on medical datasets, limiting broader applicability to other domains unless further validation is conducted.
- The methods described assume specific types of censoring, which may not be robust to all missing data scenarios. Further, while the paper discusses outlier robustness, it doesn't provide extensive testing, which could limit performance in real-world scenarios with outliers.

### Questions
Some questions for the authors
- Have you considered the challenges with computational complexity? The need to sample 2000 quantile percentages during training may increase computational complexity, especially for high-dimensional models
- While detailed implementation is provided, the absence of open-source code limits reproducibility and integration into existing workflows. Are there plans to open-source the experiments?
- The model's complexity may hinder interpretability compared to simpler methods, a concern for applications requiring factor-level insights. Have you explored intepretability criteria either through static or dynamic visual analysis
-  Have you explored the model's ability to maintain calibration and accuracy over extended periods? InIsights into how errors propagate through the model and affect predictions at different quantile levels would enhance understanding, particularly for individual calibration.

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
4

### Summary
his paper tackles the critical, and often overlooked, problem of fine-grained calibration in survival analysis models. The authors propose a novel framework, the Individually Calibrated Asymmetric Laplace Distribution (ICALD), which skillfully unifies parametric and nonparametric approaches built upon the Asymmetric Laplace Distribution (ALD).

The core contribution is a new training objective that combines the standard ALD negative log-likelihood (for global distribution shape) with a specific calibration loss (either a quantile regression loss, L_Cqr, or a direct CDF-based loss, L_Cal). This joint loss encourages the model to be "Probably Approximately Individually Calibrated" (PAIC), a property the authors provide theoretical guarantees for (Theorem 1).

The ICALD framework is notably flexible, supporting both:

Pre-calibration: Jointly training the model with the combined loss from scratch.

Post-calibration: Using a lightweight adapter module to calibrate the outputs of a pre-trained base model, which is a more efficient approach.

The authors conduct an extensive empirical evaluation on 14 synthetic and 7 real-world datasets, comparing ICALD against 12 diverse baselines (including other parametric, non-parametric, and calibration methods). The results show that their method, particularly the post-calibration variant with the L_Cal loss, achieves state-of-the-art performance across metrics for predictive accuracy, concordance, and, most significantly, all levels of calibration (average, group, and individual).

### Strengths
Addresses an Important Problem: The paper focuses on individual-level calibration, which is far more stringent than average calibration and is essential for high-stakes, personalized applications (e.g., clinical decision support), where the reliability of a prediction for a single individual is paramount.

Strong Theoretical Foundation: This is not just an empirical "trick." The authors provide formal proofs (Theorem 1) demonstrating that their joint loss objectives result in models that are Probably Approximately Individually Calibrated (PAIC). This theoretical grounding is a significant strength.

Comprehensive and Rigorous Evaluation: The experimental validation is exceptionally thorough. The use of 21 datasets, 12 baselines, and a full suite of metrics covering accuracy (MAE, IBS), concordance (C-Index), and all three levels of calibration (ECE, Wasserstein distance) makes the empirical claims highly credible.

Flexible and Practical Framework: The authors provide two ways to use their method. The post-calibration approach (Section 3.4) is particularly valuable as it allows for the calibration of existing models with a simple, lightweight adapter, making it practical to implement and deploy.

Clear Empirical Gains: The results presented in Tables 1, 2, and 3 show a clear and consistent advantage for ICALD over all baselines, especially in its primary goal: improving calibration without sacrificing (and often while improving) accuracy and concordance.

### Weaknesses
The paper is very strong, and the authors are commendable for including a detailed "Limitations" section and several case studies in the appendix that proactively address potential issues. The primary weaknesses are:

Training Instability in Pre-Calibration: The authors note that the pre-calibration model can suffer from "asynchronous convergence" between the NLL and calibration losses, requiring a "warm-up" strategy (Appendix C.4). They admit this heuristic does not fully solve the mismatch, suggesting the joint optimization can be brittle.

Overfitting Potential: The method's effectiveness relies on sampling many quantile percentages, which can necessitate prolonged training. The authors report (Section 5, Appendix C.4) that this can lead to overfitting, especially on heavily skewed datasets. This requires careful use of early stopping.

Sensitivity of the L_Cqr Loss: One of the two proposed calibration losses, L_ALD+Cqr (based on pinball loss), is shown to be highly sensitive to censored data (Appendix C.4, Table 9). While the authors correctly identify the L_ALD+Cal variant as the superior and more robust choice, this does highlight a fragility in the quantile-regression-based formulation.

### Questions
could it be used as a general-purpose calibrator for other survival models (e.g., DeepSurv, RSF, or DeepHit)?
Clarity on L_Cal for Censored Data: s applied to all data points, including censored ones. For a censored observation at time y, the true event time is unknown (T > y). How does the lossrovide a reliable calibration signal in this case, especially for quantiles q that may be far from the true (but unknown) event time? A more intuitive explanation of why this is robust to censoring would be helpful.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes a method to learn survival distributions based on calibration of the asymmetric Laplace distribution. The ICALD model includes a ALD module as a backbone and an adapter module that refines the predictions further. The authors show that their ICALD model is probably approximately individually calibrated. They also propose pre and post calibration methods and via concentration show that as the number of quantile samples increases, the model's individual calibration performance improves. 

Note: This paper is far beyond my reviewing expertise and I have brought this up to the AC. I will defer to other reviewers on acceptance decisions.

### Strengths
This paper is well written and is somewhat accessible to an audience outside the area. The experiments seem to be thorough enough with sufficient abalations.

### Weaknesses
Disclaimer: this is outside my expertise and are issues that I am bringing up from a layperson's perspective. In terms of assessment, I will defer to other reviewers.

- This paper appears to target a fairly niche audience in the ML community. The neural network architecture (to me) seems to be fairly simple, utilizing the standard reparameterization trick. This is not surprising since ALDs involve only a handful of parameters; part of the authors' contributions is to include a mixture of ALD (line 238).
- Why was the distribution over $q$ the uniform distribution (line 242)? Was this prior chose for convenience or some other reason?

### Questions
- Table 3 reports the proposed approach against a handful of baselines based on whether performance was better, worse or the same. This looks to me to be a little strange, why did the authors not simply report the average of whatever metric was chosen, as opposed to a qualitative approach like this? 
- The same goes for the following results, why were "wins" reported instead?

### Soundness
3

### Presentation
3

### Contribution
2
