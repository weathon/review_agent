# Flexible Transfer Learning in Deep Cox Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 2, 6, 6

## Abstract
Prognosis prediction is an important topic in survival analysis. Historically, research aimed at predicting survival outcomes has largely been confined to individual datasets. These datasets often have limitations, such as rare event rates, small sample sizes, high dimensionality, and low signal-to-noise ratios.  To overcome these limitations, integrated survival analysis and transfer learning has been proposed to improve prediction accuracy by incorporating external prediction models into the analysis of newly collected data. However, traditional integrated approaches, such as the integrated Cox proportional hazards model, often face limitations in prognostic prediction capabilities due to their dependence on the linearity and proportional hazards assumptions. In reality, the relationship between event times and risk factors can be intricate, often involving non-linear effects, influences that vary over time, and interactions. To effectively capture the complexities of integrated time-to-event data, it is essential to employ computationally efficient deep learning techniques.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a deep learning framework that integrates external risk models with limited internal time-to-event data for improved prognostic prediction through Kullback-Leibler (KL) based transfer learning. The target internal cohort satisfies a flexible  proportional hazards model, and some external risk scores are available from an external model that may not be the same as the internal model. The disparity between the internal and external ranking metric is measured by the Kullback-Leibler (K-L) divergence. Simulation studies and a real-world application on a prostate cancer dataset demonstrate improved prediction accuracy and robustness compared to baseline Cox and deep survival models. 

Overall, this is an interesting work with practical relevance.

### Strengths
1. The proposed method addresses a real challenge in biomedical research when the target dataset is too small for a deep neural network approach.

2. The introduction of a generalized KL divergence penalty within a deep neural network survival model represents a significant advancement. 

3.  The empirical success of the approach underscores its significance.  

4. The paper is clearly structured and includes proofs and implementation details.

### Weaknesses
1. The success of the framework hinges on the relevance and fidelity of external information; poor or misaligned sources may limit improvement.

2. The exploration of domain shift is limited. While some heterogeneity scenarios are tested, more systematic evaluation of extreme domain discrepancies would strengthen claims of robustness.

3. The inclusion of deep learning components may obscure the interpretability advantages typically associated with Cox models.

### Questions
1. The proposed internal model may be unidentifiable if the form of the risk function r and the parameter beta are both unknown. 

2. How sensitive is the performance  of NNCoxKL to the scaling or monotonic transformation of external risk scores?

3. The choice of the penalty η requires careful cross-validation, which may be computationally intensive and unstable in small datasets. Could η-selection be guided by an information criterion rather than cross-validation to improve efficiency?

4. How does the model handle multiple external sources simultaneously?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces NNCoxKL, a deep transfer learning framework designed to integrate external risk models with internal time-to-event data for survival analysis. The approach extends the Cox proportional hazards model by introducing a generalized Kullback–Leibler (KL) divergence penalty, which aligns internal and external risk scores and facilitates knowledge transfer from external risk assessment tools. The framework employs neural networks to capture non-linear relationships among covariates and is evaluated on both synthetic simulations and one real-world dataset, demonstrating performance gains through the incorporation of external risk information.

### Strengths
Leveraging external tools to enhance risk prediction or improve generalization is indeed an important problem, especially in survival analysis where sample sizes are often limited and external models typically encode knowledge from larger, more diverse populations.

### Weaknesses
-	The applicability of the proposed approach appears quite narrow. The method assumes that the internal dataset is relatively small and that the external tool provides informative signals about the same samples based on a subset of the internal covariates. Such a setting may not commonly occur in practice, which likely explains why the authors were only able to demonstrate results on a single real-world dataset that meets these criteria.
-	The paper also lacks a solid theoretical foundation or generalization analysis that could clarify when or why the generalized KL regularization improves transfer performance. In addition, the study does not examine the method’s potential failure modes under substantial domain shift between internal and external data.
-	The exposition around the use of KL divergence is unnecessarily lengthy. Since KL divergence is a well-established concept in machine learning, its detailed reintroduction offers limited value. The core contribution—interpreting ranking as a probabilistic measure and applying KL regularization to align internal and external risk predictions—is relatively straightforward and could be presented more concisely.
-	Finally, the implementation of the ranking-based loss (i.e., the Cox partial likelihood) raises practical concerns. This quantity should ideally be computed over the entire risk set, yet the paper does not sufficiently discuss the implications of using mini-batch training for the deep model $𝑟(𝑍_𝑖, \beta)$. Moreover, the computational cost is likely to be substantial, as the ranking must be evaluated across all event times, making the proposed approach potentially inefficient for large-scale applications.

### Questions
-	The baseline should include a variant that uses the external tool’s risk scores as an additional input covariate, since this alone may already capture much of the information derived from the broader external population.
-	Following Weakness 4, what is the impact of mini-batch training on the accumulated generalized KL divergence term $D(\tilde{r} || r)$?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
NNCoxKL, a generalized KL-based transfer learning framework, flexibly integrates external risk info with internal time-to-event data. It outperforms traditional Cox models and non-transfer deep models in small/moderate datasets. Real-world prostate cancer data and simulations confirm it boosts C-index and reduces loss, validating its value for prognostic prediction.

### Strengths
• Works for probabilistic/non-probabilistic external data (e.g., risk groups) and homogeneous/heterogeneous settings.
• Non-linear modeling: Uses DNN to avoid linearity constraints of classic Cox models.
• Overfitting mitigation: Transfer learning stabilizes performance vs. data-limited NNCox.

### Weaknesses
The method assumes that clients share overlapping or similar feature spaces, which may not hold in highly heterogeneous cross-domain settings.
Performance depends heavily on hyperparameters in the meta-graph (e.g., similarity thresholds, edge weights), but their tuning process is not clearly described.

### Questions
Tuning parameter η needs cross-validation, is there any better way to optimize the tuning?

### Soundness
3

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
This paper presents NNCoxKL, a transfer learning framework that augments a flexible neural Cox model with external prognostic information using a generalized Kullback Leiber divergence constructed over risk set rankings. The core idea is to align internal risk scores produced by a neural network with external scores or even coarse risk groupings by minimizing a divergence between Plackett Luce style distributions at each event time, while retaining a partial likelihood structure so standard deep learning optimization remains applicable. A tuning parameter controls how strongly the external signal enters the penalized objective, and a proof shows the objective reduces to a familiar form that weights event indicators by an externally induced pseudo event rate. The paper supports the method with simulations that vary nonlinearity, censoring, data size, and domain shift, and with an application that integrates STAR CAP risk groups into a MUSIC prostate cancer cohort, reporting gains in C index and loss on held out data. The authors also discuss implementation choices such as network architecture, regularization, AdamW, early stopping, and cross validation for selecting the integration weight, and provide additional experiments on public survival datasets.


Overall this is a promising and practically relevant contribution that bridges a real gap between clinical risk tools and modern survival learning. With stronger analysis of invariance and negative transfer, broader calibration focused evaluation, and more comprehensive baselines and ablations, the paper would be significantly stronger and more actionable for practitioners.

### Strengths
The work is well motivated by the small sample challenge in survival prediction and by practical constraints that often limit external information to model scores or clinical groupings rather than individual level data. Framing integration as a divergence between ranking distributions is elegant and versatile, letting the method use heterogeneous external sources without requiring probability calibrated survival outputs. The derivation that preserves a Cox style training objective is a strong practical contribution because it allows the method to drop into existing neural survival toolchains with minor changes. The empirical study is thoughtful, covering both linear and nonlinear data generating processes, different external model qualities, and explicit domain shift, and it shows consistent discrimination gains and improved optimization behavior with reduced overfitting sensitivity. The real data example with prostate cancer is compelling because it demonstrates how to convert a points based staging system into usable external signal for a modern survival learner. The paper is clearly written, situates itself in the literature on KL based integration and neural survival models, and attends to reproducibility details.

### Weaknesses
There are also weaknesses that limit the paper’s current impact and clarity. Despite the flexible network for covariate effects, the method still inherits a proportional hazards assumption for the baseline, which can be restrictive in settings with strong time varying effects. The approach is sensitive to the scale and even monotone transformation of the external score, and the proposed one step rescaling via a univariate Cox fit is ad hoc and may not be robust across cohorts with heavy shift; a principled invariance or calibration procedure would strengthen the method. The reliance on cross validated selection of the integration weight introduces variance in small internal samples, and no guidance is given for safe defaults or information criteria based selection. The theory focuses on the objective rewrite but offers no guarantees about consistency, oracle properties under correct ranking, or bounds on negative transfer when the external model is poor or misaligned; even a simple analysis under misspecification would be helpful. The evaluation emphasizes discrimination and partial likelihood loss but gives little attention to calibration, which is critical for clinical adoption, and does not report competing risks, time dependent AUC, or D calibration. Comparisons omit recent neural survival baselines that integrate external knowledge through stacking or representation learning, and ablations on architecture depth, dropout, and the role of the external score as an explicit input versus only via the penalty would clarify where the gains come from. Finally, practical aspects such as computational cost, convergence stability, and handling of ties are only briefly mentioned, and the combination of multiple external sources is deferred to future work despite being a natural and common scenario.

---

Weaknesses itemized (for rebuttal and discussion)

1. Inherits a proportional hazards structure from the Cox framework, which can be restrictive when effects vary strongly over time. 
2. The divergence is not invariant to rescaling of external scores and the proposed remedy is an ad hoc rescaling via a univariate Cox fit, with unclear robustness across domains. 
3. Relies on cross-validated selection of the integration weight, which may introduce variance in small samples and lacks clear default guidance. 
4. Provides limited theoretical guarantees beyond the objective rewrite, with no explicit bounds on negative transfer under misaligned external signals.
5. Evaluation emphasis appears to be on discrimination and loss, with less attention to absolute risk calibration and competing-risk settings that matter for clinical adoption.
6. Practical guidance on combining multiple external sources, convergence stability, and tie handling is brief relative to likely practitioner needs.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3
