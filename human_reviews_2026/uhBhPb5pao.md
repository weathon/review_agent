# Pattern-Guided Diffusion Models

- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Diffusion models have shown promise in forecasting future data from multivariate time series. However, few existing methods account for recurring structures, or patterns, that appear within the data. We present Pattern-Guided Diffusion Models (PGDM), which leverage inherent patterns within temporal data for forecasting future time steps. PGDM first extracts patterns using archetypal analysis and estimates the most likely next pattern in the sequence. By guiding predictions with this pattern estimate, PGDM makes more realistic predictions that fit within the set of known patterns. We additionally introduce a novel uncertainty quantification technique based on archetypal analysis, and we dynamically scale the guidance level based on the pattern estimate uncertainty. We apply our method to two well-motivated forecasting applications, predicting visual field measurements and motion capture frames.  On both, we show that pattern guidance improves PGDM’s performance (MAE / CRPS) by up to 40.67% / 56.26% and 14.12% /
14.10%, respectively. PGDM also outperforms baselines by up to 65.58% / 84.83% and 93.64% / 92.55%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Pattern-Guided Diffusion Models (PGDM), a framework that enhances diffusion-based time-series forecasting by explicitly leveraging recurring patterns inherent in temporal data. The authors first apply Archetypal Analysis (AA) to extract interpretable “archetype” patterns from training data and represent each data as a convex combination of these archetypes. A lightweight neural network then predicts future pattern coefficients, which guide a diffusion model to generate realistic future sequences. To dynamically control how much the model relies on pattern guidance, the paper proposes Archetypal Analysis Uncertainty Quantification (AAUQ), which measures geometric distance from the training distribution and adjusts guidance strength accordingly.

### Strengths
(1) The proposed model enhances diffusion-based time-series forecasting by explicitly leveraging recurring patterns inherent in temporal data, where such "recurring patterns" is common in real-world datasets.

(2) The solution is well-motivated and the proposed model is technically sound although no novel components is proposed compared to previous work.

(3) Evaluation on two very different tasks—medical vision fields and motion capture—demonstrates the model’s generality and consistent performance gains over multiple baselines.

### Weaknesses
(1) The need to first learn the pattern estimator (via AA and the guidance network) before training the diffusion model introduces potential error propagation and optimization complexity. Is there any way to conduct such process in an end-to-end manner?

(2) Intuitively, the input of the proposed model and other diffusion-based model in the same, i.e., the time series data, the only difference is the proposed model add addtional ``predicted pattern'' from the raw time series. In that sense, no additional information is introduced for the proposed model. And for other diffusion-based models, we can consider them conducting the "pattern prediction" implicitly. When there is enough data, other diffusion-based models should be able to learn a good pattern prediction implicitly in end-to-end manner. In that sense, other diffusion-based models should performance no different compared to the proposed model. Can the author explain more on why the proposed model get better performance?

### Questions
please see the weakness above.

### Soundness
3

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
This paper proposes Pattern-Guided Diffusion Models (PGDM), a time-series forecasting approach that combines Archetypal Analysis (AA) with diffusion models. The method first projects input sequences into an archetype space, uses a lightweight neural network to predict future archetype coefficients, and injects them as conditional guidance during diffusion. The paper also introduces an uncertainty metric AAUQ to dynamically adjust the guidance strength. Experiments are conducted on visual field prediction and human motion prediction.

### Strengths
1. The idea of incorporating archetypal analysis into diffusion-based forecasting is conceptually interesting and adds a degree of interpretability.
2. The paper is clearly written and well-structured; the method is presented coherently with theoretical support and illustrative figures.

### Weaknesses
1. **Insufficient experimental design.** The experimental scope is small (only two datasets in relatively narrow domains); the baselines are dated and omit stronger modern models (e.g., recent diffusion- or Transformer-based forecasters); the reason for selecting the two variants $PGDM_{MAE}$ and $PGDM_{GDE}$ are unclear; and there is a lack of comprehensive ablations to establish the contribution of key components (archetype space, AAUQ weighting, etc.).
2. **Limited analysis and discussion.** The narrative largely describes *what* was done and *what* the results are, with limited discussion of *why* these design choices are appropriate and *why* the observed results occur.
3. **Limited methodological novelty.** In essence, the method adds a predictive module as a conditional signal within an existing diffusion framework. Since using external predictive signals with classifier-free guidance is already known to be effective, the contribution feels more stylistic than fundamentally novel. The paper should provide deeper justification and empirical evidence, explaining why this specific form of “pattern guidance” is principled and how it outperforms standard conditional diffusion in practice.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Pattern-Guided Diffusion Models (PGDM) for time-series forecasting. The approach first applies archetypal analysis  to construct a low-dimensional pattern space from the training data. A predictor forecasts future pattern contributions, which condition a diffusion model via classifier-free guidance. A data-driven uncertainty score (AAUQ) adaptively modulates the guidance scale, and a lightweight pattern-mixing step at inference further refines samples. The method is evaluated on visual-field progression (UWHVF) and human motion and it consistently improves MAE over strong baselines while providing informative ablations of the guidance weight. The presentation is clear, the theoretical rationale is coherent, and the empirical evidence substantiates the central claims regarding accuracy and stability.

### Strengths
1 The work targets recurring temporal structure and argues that guiding diffusion in a low-dimensional pattern space improves efficiency and interpretability, with a succinct, easy-to-follow pipeline.

2 The training and inference procedures are explicit; the guidance mechanism is formalized with equations; and AAUQ provides a principled, data-dependent way to scale guidance rather than relying on a fixed heuristic.

3 Bounds and geometric arguments link the uncertainty proxy to expected guidance behavior, offering credibility beyond purely empirical results.

4 Across two distinct domains, PGDM reduces MAE and variance; ablations reveal a stable operating range and diminishing returns as the guidance scale increases.

### Weaknesses
1 The authors argue that AAUQ explicitly modulates guidance based on uncertainty but this paper does not report probabilistic metrics (e.g., CRPS, NLL), coverage (Prediction Interval Coverage Probability, PICP), or calibration diagnostics (e.g., reliability diagrams). 

2  The attribution of gains seem ambiguous. I suggest an abalation study for the AA representation or the dynamic guidance against standard alternatives under the same backbone and compute budget,.

3 I think  it would be better to add  parameter counts and basic timing to demonstrate contextualize efficiency. It seems that parameter counts are not reported; hardware and memory are only briefly noted (single 42-GB GPU; <1 GB per model). Training/throughput and sampling latency are also unspecified.

### Questions
See weaknesses.

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
This paper proposes Patten-Guided Diffusion Models (PGDM), a framework for the diffusion-based forecasting of data sequences that form inherent patterns. PGDM introduces the use of Archetypal Analysis (AA) and learn the generative model with the condition of archetypal patterns. The authors further design a dynamic guidance scale based on the distance between historical samples and the archetypal patterns. Experiments on visual field and human motion prediction tasks demonstrate the effectiveness of PGDM.

### Strengths
- The introduction of Archetypal Analysis for capturing inherent patterns is interesting.
- The method is described clearly and is easy to implement.
- The dynamic guidance scale is a thoughtful addition that enhances model performance.

### Weaknesses
- The application scope is somewhat narrow and the baseline comparisons could be more extensive.
- The experiment setup is somewhat confusing.
- The dynamic guidance scale is insufficiently effective.

### Questions
1. **The scope of application, and baseline methods.** In my opinion, PGDM is build on a prior that data sequences have inherent patterns. However, this assumption may not hold for all types of data sequences. Could the authors discuss the potential limitations of PGDM when applied to data sequences that do not exhibit clear patterns? Furthermore, the baseline methods TimeGrad and CSDI are general-purpose diffusion models for time series forecasting. Could the authors consider including more specialized baseline methods that are specifically designed for pattern-based forecasting to provide a more comprehensive evaluation of PGDM's performance? Additionally, how can PGDM be adapted or extended to handle data sequences that lack inherent patterns?

2. **Experiment setup of MAE/GDE.** The authors describe in Sec.5 that *"The first model ... lowest validation mean absolute error... The secon model ...  highest capacity"*. However, I feel confused about the setup of proposing different model selection criteria. Could the authors clarify why MAE-based and GDE-based models should be simultaneously discussed. Furthermore, the GDE criterion is not clearly described. Could the authors provide more details?

3. **Effectiveness of dynamic guidance scale.** The dynamic guidance scale is designed to adjust the influence of archetypal patterns based on the distance between historical samples and the archetypal patterns. However, in Human the motion prediction task, as demonstrated in Table 1, the improvement of $w>0$ is relatively small. How can the authors explain the limited effectiveness of the dynamic guidance scale in this context? Are there specific scenarios or types of data where the dynamic guidance scale is more beneficial?

4. **Clarity of writing.** While the overall structure of the paper is logical, certain sections could benefit from clearer explanations. For instance, the authors could first provide a detailed high-level overview of data types with inherent patterns (such as Figure 3). This would help readers better understand the motivation behind PGDM before delving into the technical details. Could the authors consider revising these sections to enhance clarity and accessibility for a broader audience?

5. **Novelty of proposed method.** The use of Archetypal Analysis (AA) to capture inherent patterns in data sequences is an interesting approach. However, AA itself is not a novel technique, and I believe it is natural to apply AA in this context. Could the authors elaborate on the specific contributions of PGDM that distinguish it from existing methods that utilize AA or similar techniques? How does PGDM advance the state-of-the-art in diffusion-based forecasting beyond the application of AA?

### Soundness
3

### Presentation
1

### Contribution
1
