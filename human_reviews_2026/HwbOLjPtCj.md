# CONTRASTIVE TIME SERIES FORECASTING WITH ANOMALIES

- Decision: Reject
- Scores: 4, 2, 8, 2

## Abstract
Time-series forecasting predicts future values from past data. In real-world settings, some anomalous events have lasting effects and influence the forecast, while others are short-lived and should be ignored. Standard forecasting models fail to make this distinction, often either overreacting to noise or missing persistent shifts. We propose **Co-TSFA** (Contrastive Time-Series Forecasting with Anomalies), a regularization framework that learns when to ignore anomalies and when to respond. Co-TSFA generates input-only and input–output augmentations to model forecast-irrelevant and forecast-relevant anomalies, and introduces a latent–output alignment loss that ties representation changes to forecast changes. This encourages invariance to irrelevant perturbations while preserving sensitivity to meaningful distributional shifts. Experiments on the Traffic and Electricity benchmarks, as well as on a real-world cash-demand dataset, demonstrate that Co-TSFA improves performance under anomalous conditions while maintaining accuracy on normal data. The implementation of Co-TSFA will be released publicly upon acceptance of the paper.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Co-TSFA, a contrastive regularization framework for time-series forecasting under anomalies. By generating input-only and input–output augmentations during training and enforcing latent–output alignment, Co-TSFA learns to suppress forecast-irrelevant perturbations while adapting to forecast-relevant shifts. Experiments on Traffic, Electricity, and a real-world cash-demand dataset show that Co-TSFA improves forecasting accuracy under anomalous conditions without sacrificing performance on clean data, outperforming existing robust and adaptive methods.

### Strengths
**S1:** The paper identifies and formalizes an important gap in time-series forecasting: the need to distinguish and handle both forecast-relevant and forecast-irrelevant anomalies at test time. 

**S2:** The tables are well-organized and clearly formatted, making it easy to compare results across datasets and baselines.

**S3:** Mathematical symbols, variables, and equations are consistently defined and formatted throughout the paper, aiding readability.

### Weaknesses
**W1:** All experiments introduce anomalies artificially using curves designed by the authors. Specifically, a particular “anomaly shape” is created, and Co-TSFA is then shown to handle these anomalies most effectively. As noted in the captions of each table (“we sample anomalies from the anomaly function…”), the anomalies are not drawn from the original datasets but are inserted into clean training/testing sets at varying intensities and positions. This raises concerns that the method is primarily evaluated on anomalies tailored to its design, which limits the real-world validity and general applicability of the results.

**W2:** In Figure 1, the anomalies presented resemble trends rather than true anomalies. Existing models should be able to handle such trends effectively. Indeed, as shown in Table 1, the improvement of Co-TSFA over baseline methods is relatively modest, which suggests that the advantage of the proposed method may be limited for these types of “anomalies.”

**W3:** The experiments in Table 1 do not include datasets used in other experiments, such as Traffic and Electricity, nor widely used benchmarks (e.g., ETT, Weather). This makes it difficult to assess Co-TSFA’s performance across different types of time series and diverse application domains.

**W4:** The loss function relies heavily on a softmax-normalized dot-product similarity between both latent representations and outputs. However, it remains unclear how these similarities are computed for high-dimensional outputs (Equations 3–4): are they temporal, aggregated across channels, or computed in some latent space? This aspect should be explicitly clarified or justified.

**W5:** Table 3 is not cited in the text, which should be addressed to ensure clarity and proper referencing.

**W6:** Several referenced models (e.g., TimesNet, FedFormer) should cite their formally published versions rather than preprints, and AutoFormer appears to be used twice (lines 562–566). The manuscript should verify all citations. Please check all references, remove duplicates, and make sure their details are correct.

### Questions
see W1-W6

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work focuses on time series forecasting under distribution shift and introduces a contrastive-based framework to learn to ignore unnecessary anomalies (or noise) in the input and output sequences. This is achieved through an end-to-end forecasting framework that incorporates an alignment loss between the latent representations and the output sequences, thereby encoding invariance to meaningless perturbations. Experiments consider several common benchmark datasets in time series forecasting, along with various setups, including the presence of anomalies in both the input and output series.

### Strengths
The strong points of the paper are the following:
- **Clarity:** The paper is easy to follow, and the main ideas are simply explained.
- **Originality:** Although robustness has been a long-term issue in time series forecasting, focusing on anomaly types is a more recent research approach (RobustTSF was introduced in 2024), and definitely merging this with invariance learning is an interesting modeling aspect.
- **Quality:** Different experimental setups are considered with respect to the presence of anomalies in the input and output series.

### Weaknesses
The weaknesses of the paper are the following:
1. **Poor positioning against related works (Clarity, Originality):** The authors do not sufficiently present related work in contrastive learning, e.g., recent soft contrastive learning for time series (Lee et al., 2023) (please see how extensive the discussion in the relevant paper is). It is hard to understand how the proposed method differs from existing methods in the formulation of the loss and the selection of augmentation for the contrasting views. With the present explanations, the architectural design seems rather heuristic and unconventional compared to recent approaches (for instance, amplitude transformations are considered hard, and contrast sensitivity is also sensitive in the selection of pseudo labels). Similarly, in invariance learning, recent works emphasize hard-coded invariances to offset shift and trend in time series analysis, to handle distribution shifts (Germain et al., 2025), while recent works also quantify statistically the lookback-horizon distribution shift for accurate forecasting (Fan et al., 2023).
2. **Poor formulation of anomaly types and distribution shift characterization (Quality):** In RobustTSF (Cheng et al., 2024) and in the distribution-shift paper (Fan et al., 2023), shifts between input-output sequences and the formulation of anomalies are mathematically proven and motivated with clear experimental arguments. In this work, it is unclear how shift and anomaly types apply to real-world scenarios and extend the related works, which should be proven mathematically or experimentally.
3. **Few baseline comparisons (Significance):** Other contrastive-based methods should be considered in the forecasting setup (e.g., method in (Lee et al., 2023) is applied to both forecasting and anomaly detection), as well as alternative approaches for handling distribution shift (see methods and baselines in (Germain et al., 2025; Fan et al., 2023). Additionally, the considered datasets are limited in number and represent a small subset of the tiem series forecasting benchmark.
4. **Lack of reproducible and statistically-significant results (Significance):** No code for reproducing the results is available as part of the supplementary material. Additionally, the standard deviations for independent runs should be provided, as the results are very close to the baselines in several cases (alternatively, statistical significance could be demonstrated). 

References:
- Lee, Seunghan, Taeyoung Park, and Kibok Lee. "Soft contrastive learning for time series." arXiv preprint arXiv:2312.16424 (2023).
- Germain, Thibaut, Chrysoula Kosma, and Laurent Oudre. "Time Series Representations with Hard-Coded Invariances." Forty-second International Conference on Machine Learning.
- Fan, Wei, et al. "Dish-ts: a general paradigm for alleviating distribution shift in time series forecasting." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 6. 2023.

### Questions
1. Can the authors justify the contributions of their proposed methodology against the relevant literature, showcasing example where prominent methods fail? Discussion should cover both the contrastive and the distribution-shift modelization.
2. Can the authors prove the significance of their contributions from a methodological and an experimental side? Consider adding baselines, illustrative examples, and clear formulations.
3. Statistical significance for results and additional datasets is also an important consideration.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Co-TSFA, a novel method designed to address robust time series forecasting problems. Specifically, the authors consider both point-wise and continuous anomalies in the time series domain, as well as scenarios where anomalies may exist in the test set. In terms of methodology, the authors first augment the original time series and then leverage contrastive learning to penalize discrepancies between the similarity of latent representations and the similarity of their corresponding outputs. Experiments conducted on multiple datasets demonstrate that the proposed approach achieves strong performance compared to other methods.

### Strengths
1. Compared to RobustTSF, which focuses on point-wise anomalies and clean-test settings, Co-TSFA generalizes to both point-wise and continuous-wise anomalies and considers both test-clean and test-noisy settings.

2. The contrastive regularization framework that enforces latent–output alignment is novel and interesting.

3. The experiments are thorough and demonstrate consistent improvements across different anomaly settings (point-wise, continuous-wise, test-clean, and test-noisy).

### Weaknesses
1. The alignment loss shares a similar concept with [R1] in the context of learning with noisy labels. The authors are encouraged to discuss the differences between the two approaches.

2. The use of augmentations may increase training costs. It would be beneficial to include a training time analysis compared to other methods.

3. The stability of Co-TSFA requires further investigation. For example, it should be examined whether MAE/MSE suddenly increases at certain epochs during training, and whether the best and final MAE/MSE values are consistent, as discussed in [R2].

[R1] Co-learning: Learning from Noisy Labels with Self-supervision

[R2] RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies

### Questions
See **Weaknesses**

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes Co-TSFA, a model-agnostic framework for robust time-series forecasting. Co-TSFA introduces a contrastive regularization strategy that aligns similarities between latent representations and forecast outputs under both clean and augmented conditions. By generating input-only and input–output augmentations to simulate different types of anomalies, Co-TSFA learns to remain invariant to irrelevant perturbations while adapting to meaningful distributional shifts. Experimental results demonstrate that Co-TSFA consistently improves forecasting accuracy across various backbone models on datasets with synthetic anomalies, and outperforms the existing robust baseline, RobustTSF.

### Strengths
The proposed method is conceptually straightforward and easy to follow, and its overall technical soundness appears solid. It employs a targeted form of data augmentation to enhance robustness against anomalies and, more generally, to mitigate distribution shifts in time-series forecasting.

The experimental evaluation covers multiple types of perturbations (as shown in Table 3), demonstrating the effectiveness and adaptability of Co-TSFA across diverse forecasting scenarios.

### Weaknesses
1. The overall evaluations of this study are not sufficiently strong to support the claims.

(1) The experiments mainly rely on the ECL and Traffic datasets, which are known to be relatively stable with limited irregular or non-stationary patterns. However, the key motivation of this work is to address forecasting under “unclean” or anomalous conditions. Evaluations on more irregular datasets, such as those exhibiting significant trend shifts, sparsity, or frequent spikes, would provide stronger and more convincing evidence of the method’s robustness. Currently, the synthetic perturbations added to otherwise stable datasets may not fully capture the challenges of real-world anomaly scenarios.

(2) Although the framework claims to be model-agnostic, the experiments only cover a few older forecasting backbones. It would strengthen the paper to include results on more advanced and recent forecasting architectures such as TimeMixer, iTransformer, or PAttn. This would better demonstrate that the observed gains indeed come from the design of Co-TSFA, rather than from limitations of the underlying forecasting model.

(3). The baseline set is limited to RobustTSF, which mainly focuses on robustness against pointwise anomalies. Since the proposed method can also be interpreted as improving generalization under distribution shifts, it would be more convincing to include comparisons with standard normalization or adaptation techniques such as InstanceNorm or RevIN. In practice, even simple combinations like Informer + InstanceNorm can give notable gains, and including such baselines would make the comparison more complete and informative.

2. The evaluation results do not clearly validate the key claims. For example, in Table 1, Co-TSFA shows relatively larger improvements on the CashDemand dataset in the clean setup than under perturbed conditions, which seems inconsistent with the stated goal of enhancing robustness to perturbations. Moreover, no statistical tests (e.g., t-test or p-value analysis) are provided to confirm whether the observed improvements are statistically significant, for example, bigger improvements on the noisy setup compared to the clean setup.

3. Several entries in Table 1 (6 out of 27) show exactly identical results between the base and Co-TSFA, which seems strange given stochastic training. It would be better to clarify possible intutions behind or reporting standard deviations over multiple seeds.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
2

### Contribution
2
