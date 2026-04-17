# The Forecast After the Forecast: A Post-Processing Shift in Time Series

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Time series forecasting has long been dominated by advances in model architecture, with recent progress driven by deep learning and hybrid statistical techniques. However, as forecasting models approach diminishing returns in accuracy, a critical yet underexplored opportunity emerges: the strategic use of post-processing. In this paper, we address the last-mile gap in time-series forecasting, which is to improve accuracy and uncertainty without retraining or modifying a deployed backbone. We propose $\delta$-Adapter, a lightweight, architecture-agnostic way to boost deployed time series forecasters without retraining. $\delta$-Adapter learns tiny, bounded modules at two interfaces: input nudging (soft edits to covariates) and output residual correction. We provide local descent guarantees, $O(\delta)$ drift bounds, and compositional stability for combined adapters.
Meanwhile, it can act as a feature selector by learning a sparse, horizon-aware mask over inputs to select important features, thereby improving interpretability.
In addition, it can also be used as a distribution calibrator to measure uncertainty. Thus, we introduce a Quantile Calibrator and a Conformal Corrector that together deliver calibrated, personalized intervals with finite-sample coverage.  
Our experiments across diverse backbones and datasets show that $\delta$-Adapter improves accuracy and calibration with negligible compute and no interface changes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes **δ-Adapter**, a lightweight and architecture-agnostic framework that improves frozen time-series forecasting models through post-processing rather than retraining. It introduces two modules: **input nudging**, which softly edits covariates before inference, and **output residual correction**, which refines predictions after inference. The method provides theoretical guarantees of local descent, bounded drift, and compositional stability. Additionally, the authors design a **feature-selector adapter** that identifies influential input features and **quantile/conformal calibrators** that enhance uncertainty estimation. Experiments across multiple forecasting backbones and datasets demonstrate that δ-Adapter consistently improves accuracy and calibration with minimal computational cost.

### Strengths
The paper’s main strength lies in its original and practical reformulation of time-series improvement as a post-processing problem rather than a model-design or retraining task. This shift in perspective is both creative and highly relevant to real-world forecasting systems, where retraining large backbones is often infeasible. The proposed δ-Adapter is simple yet elegant, introducing bounded input and output modules that deliver measurable gains without altering the base model. Theoretical sections are rigorous, offering provable guarantees of local descent, drift stability, and compositional safety, which is an uncommon level of formalism for this kind of lightweight method. The paper is also clear and well-structured, with intuitive explanations of each component and thoughtful use of figures to convey mechanisms. Empirically, the approach demonstrates consistent and transferable improvements across architectures and datasets with negligible overhead, highlighting its significance and applicability to a broad range of forecasters. Overall, the work stands out for its conceptual clarity, strong theoretical underpinnings, and practical impact on efficient and reliable time-series forecasting.

### Weaknesses
While the paper presents a strong theoretical foundation and an appealing practical idea, several areas could be improved to strengthen its impact and clarity.

First, the experimental evaluation, though broad in dataset coverage, lacks comparative depth. The study primarily benchmarks δ-Adapter against frozen and fine-tuned baselines, but it omits comparisons to parameter-efficient adaptation methods such as LoRA, adapters, or residual fine-tuning strategies that have been explored in related forecasting and NLP contexts. Including such baselines would better position δ-Adapter within the broader adaptation literature and clarify its relative advantages.

Second, while the theoretical analysis is thorough, the empirical link to these guarantees remains weak. The paper could provide visual or quantitative evidence of stability, such as loss landscapes or δ–performance trade-offs, to demonstrate how the theory manifests in practice.

Third, the related work section should be expanded to discuss recent advances in adaptation and test-time adaptation (TTA) for time-series forecasting, which directly address similar motivations to δ-Adapter. In particular, works such as Kim et al. (AAAI 2025) [1], Medeiros et al. (arXiv 2025) [2], Grover & Etemad (ICML 2025 Workshop) [3], and Lee et al. (ICML 2025) [4] explore gradient-based, parameter-efficient, and online adaptation mechanisms to handle non-stationarity and distribution shifts. Adding a dedicated subsection contrasting δ-Adapter with these methods would clearly position it as a post-hoc alternative and highlight its novelty within the adaptation landscape.

Overall, the paper would benefit from stronger empirical grounding, richer comparative analysis, and a more comprehensive connection to existing adaptation research to fully demonstrate the breadth and impact of its contribution.

References:

[1] Kim, H., Kim, S., Mok, J., & Yoon, S. (2025, April). Battling the non-stationarity in time series forecasting via test-time adaptation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 17, pp. 17868-17876).

[2] Medeiros, H. R., Sharifi-Noghabi, H., Oliveira, G. L., & Irandoust, S. (2025). Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting. arXiv preprint arXiv:2506.23424.

[3] Grover, Shivam, and Ali Etemad. "Shift-Aware Test Time Adaptation and Benchmarking for Time-Series Forecasting." Second Workshop on Test-Time Adaptation: Putting Updates to the Test! at ICML 2025. 2025.

[4] Lee, Thomas L., et al. "Lightweight Online Adaption for Time Series Foundation Model Forecasts." Forty-second International Conference on Machine Learning.

### Questions
1. **Comparison to Test-Time Adaptation (TTA) Methods**.
   How does δ-Adapter conceptually and empirically differ from recent TTA approaches that also adapt frozen forecasters at inference time (e.g., Kim et al., AAAI 2025; Medeiros et al., 2025; Grover & Etemad, ICML 2025)? Since these methods share similar goals such as robustness to non-stationarity and post-hoc improvement, please clarify what specific challenges δ-Adapter addresses that are not tackled by gradient-based or parameter-efficient TTA methods.

2. **Choice and Sensitivity of δ**.
   The theoretical analysis emphasizes δ as a small trust-region parameter controlling stability. How sensitive are results to δ in practice, and how should it be chosen for new datasets or models? Could adaptive δ-scheduling be beneficial?

3. **Theoretical–Empirical Link**.
   The paper provides strong theoretical guarantees for stability and local descent, but the experiments do not directly validate these properties. Could the authors include empirical evidence such as δ vs. loss curves or visualization of stability bounds to support the theoretical claims?

4. **Comparison to Lightweight Adaptation Baselines**.
   Have the authors considered comparing δ-Adapter with parameter-efficient fine-tuning strategies such as LoRA or standard adapter layers, which also introduce small modules on frozen backbones? This would help quantify how δ-Adapter performs relative to existing low-overhead adaptation techniques.

5. **Practical Deployment Considerations**.
   Since δ-Adapter is advertised as a post-hoc enhancement for deployed models, what is the typical training cost, latency overhead, and implementation effort in a real-world deployment scenario? Including such details would reinforce the paper’s practical relevance.

These clarifications would significantly help assess the robustness, novelty, and real-world applicability of δ-Adapter and could strengthen the case for acceptance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an adapter-based finetuning method for pretrained forecasting models. The paper proposes input and output-based adapters in two forms: additive and multiplicative. The authors provide theoretical results for the stability and improvement under certain conditions with respect to the base model.  Finally, the authors expose how to utilize their adapters as calibration methods for models that only offer fixed-point predictions.

### Strengths
- Overall, I think this is a good paper. The presentation is clear and the theory is solid as far as I can tell.
- The presented experiments consistently show the improvement of input and output adapters. 
- Benchmarks look reasonable within popular datasets from the time series literature.

### Weaknesses
- Some of the models are relatively old and would not be considered SOTA at the current time (e.g. Autoformer). I would suggest including models like PatchTST, Non-stationary Transformer, TimesNet or TimeMixer.
- Additive vs multiplicative: 
- Reports metrics in Table 2 are  averaged across lenghts. I understand the difficulty of reporting metrics across many horizons, but in my opinion, this makes it harder to interpret the effect over prediction lengths, over which errors may differ significantly.

**Typos and minor comments**
- L132: "Obviously". This can be dropped :) 
- Table 2 Exange (also in the appendix table). caption: "veraged"
- L64: Despite adapter is simplicity

### Questions
- L461-463: Could the authors elaborate more on why this type of input/output adapter is desirable over a LoRA style adapter?
- How does additive vs multiplicative updates compare empirically? It seems the experiments mostly focus on additive adapters.
- Why not report results for Ada X+y adapters together (Tables 1 and 2)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces sigma-Adapter, a lightweight post-processing framework for improving time series forecasting without retraining the backbone model. The approach works by learning small bounded corrections at two interfaces: (1) input nudging (soft edits to covariates) and (2) output residual correction. The authors provide theoretical guarantees for local descent and stability and demonstrate feature selection capabilities through learnable masks, and introduce two distributional correctors (Quantile and Conformal calibrators). The experiments across multiple datasets and backbone models show consistent improvements in both accuracy and calibration.

### Strengths
The main strengths of the paper include,
1. The post-processing approach is a well-motivated solution for real-world deployments where retraining large models is costly. The ability to improve frozen forecasters is important and useful problem to tackle. 
2. The paper provides rigorous theoretical analysis for several algorithms: Local descent guarantees (Theorems 2 & 3), Compositional stability for combined adapters (Proposition 3.2) etc.
3. The authors compared against diverse backbones (DistPred, iTransformer, FourierGNN, FreTS, Autoformer) across several datasets (ETT, Traffic, Weather, Electricity, Exchange). Both pre-trained and SOTA models show consistent improvements in both accuracy and calibration.
4. The authors conducted clear ablation studies to 1. Compare the performance of sigma-adapter variants. 2. Compare the effectiveness of input adapters and output calibrators across several datasets.

### Weaknesses
The main weakness of the paper include,

1. The proposed Input/output adapters approach is conceptually similar to existing adapter methods in NLP. The theoretical results (gradient descent, Lipschitz stability) are relatively standard. The main contribution does not appear to be novel as the claims in the paper and looks like an application of the NLP concept to time series.
2. Some important ablation studies such as the impact of adapter architecture (MLP depth, width etc), effect of different sigma values across datasets are missing. 
3. The comparison with other post-processing method (e.g., calibration techniques) and adapter approaches are not presented; The comparison with lightweight fine-tuning methods (e.g., LoRA for time series) across multiple pre-trained models is missing; The LoRA comparison in Figure 8(c) is limited to one model (Sundial) and shows high variance but needs more investigation. 
4. Discussion on fine-tuning/adaptation time across different adapter approaches and comparison against wider range of pretrained models might be helpful to strengthen this paper.

### Questions
1. Can you compare against other adapter approaches available in time series literature and also compare with adapters used in NLP literature?
2. Can you have thorough discussion on the comparison with LoRA method and investigation on high variance?
3. Can you compare against few other pre-trained models like TabPFN-TS to validate the claims made are generalizable across pre-trained models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces δ-Adapter, a lightweight and model-agnostic post-processing framework to improve deployed time series forecasting models without retraining them. The method learns small, bounded adjustments at two interfaces: "input nudging" to refine covariates and "output residual correction" to adjust predictions. The framework is extended into three applications: a feature selector for interpretability, a quantile calibrator, and a conformal calibrator for reliable uncertainty quantification. Extensive experiments on various models and datasets validate that δ-Adapter effectively boosts prediction accuracy and calibration.

---

### Strengths
1.  **Practicality and Novelty**: The paper addresses the critical real-world problem of updating deployed models in a computationally cheap manner. The δ-Adapter is a practical "plug-and-play" solution compared to costly alternatives like full retraining or fine-tuning.

2.  **Methodological Completeness**: The framework is versatile, being applicable to different backbone forecasting model. It is also comprehensive, addressing crucial aspects like interpretability (feature selection) and reliable uncertainty estimation, which are often overlooked.

3.  **Theoretical and Empirical Support**: The proposed method is well-grounded in theory, with stability and performance guarantees. These theoretical claims are also backed by extensive experiments across multiple models and standard benchmarks.

### Weaknesses
1.  **Limited Comparison with Alternative Post-Processing Methods:** While the paper positions δ-Adapter as a post-processing technique, the primary comparisons are against fine-tuning or continue-training the backbone model. However, there is a significant body of work on post-processing and test-time adaptation for time series forecasting designed to handle concept or distribution shifts. This includes methods for both batch-training settings (e.g., [SOLID](https://arxiv.org/abs/2310.14838)、[TAFAS](https://arxiv.org/abs/2501.04970)) and online settings (like FSNet and OneNet, mentioned in the paper's related works). An experimental comparison against these direct alternatives is necessary to properly situate the δ-Adapter's advantages; without it, the claims of superiority are not fully convincing.
2.  **Insufficient Motivation:** The motivation presented is not fully developed. The paper's opening question—"Can we keep the strong forecaster intact and learn only a tiny, post-hoc module... without heavy retraining?"—frames the problem but doesn't sufficiently justify why a post-hoc module is the required solution over other existing approaches. The discussion on how alternative methods tackle this challenge is not sufficient enough. Additionally, the descriptions of existing problems like "condition drift" and the need for a "low-complexity structure" are quite brief, which weakens the overall problem statement.
3.  **Insufficient Detail on Training Combined Adapters (Ada-X+Y):** The experiments (e.g., Figure 2 and Table 6) consistently show that the combination of input and output adapters (Ada-X+Y) yields the best results. However, the paper lacks a clear description of how these two modules are trained together. For instance, are they optimized jointly with a combined loss function or trained sequentially. A more detailed explanation of the co-training strategy can better understand their interaction and potential interference.
4.  **Lack of Analysis of Adapter Scale and Computational Cost:** The paper claims that the δ-Adapter is lightweight and has "negligible compute." However, it only mentions that the adapter is implemented as a shallow MLP without providing crucial details like its specific hyperparameters or total parameter count. Furthermore, there is no quantitative analysis of the additional computational overhead. To substantiate these claims, the paper is hoped to report metrics such as the percentage increase in wall-clock training time, the number of additional parameters, or the added inference latency compared to the original backbone model.

### Questions
1.  **The choice of hyperparameter δ**: In the experiments, δ is set to 0.1 for most datasets but 0.01 for the ETT datasets. Was this value selected empirically, or was there a systematic tuning process (e.g., grid search on a validation set)? Does the optimal value of δ correlate with properties of the backbone model (e.g., complexity) or the dataset (e.g., noise level, degree of concept drift)?

2.  **The online learning setup**: In Figure 2 and Table 6, you demonstrate superior performance under an online training setting. Could you please provide more details on this online experimental setup? Specifically, how is data streamed to the adapter (e.g., sample-by-sample, mini-batches), and how does this setup simulate a real-world scenario where new data arrives continuously?

3.  **The choice between the Quantile Calibrator (QC) and Conformal Calibrator (CC)**: The paper proposes two effective uncertainty calibrators, QC and CC. Is there recommendations for us on when to choose one over the other, when faced a new real-world dataset?

### Soundness
2

### Presentation
3

### Contribution
2
