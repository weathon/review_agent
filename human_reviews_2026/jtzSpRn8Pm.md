# TCBench: A Benchmark for Tropical Cyclone Track and Intensity Forecasting at the Global Scale

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
TCBench is a benchmark for evaluating global, short to medium-range (1–5 days) forecasts of tropical cyclone (TC) track and intensity. To allow a fair and model-agnostic comparison, TCBench builds on the IBTrACS observational dataset and formulates TC forecasting as predicting the time evolution of an existing tropical system conditioned on its initial position and intensity. TCBench includes state-of-the-art physical (TIGGE) and neural weather models (AIFS, Pangu-Weather, FourCastNet v2, GenCast). If not readily available, baseline tracks are consistently derived from model outputs using the TempestExtremes library. For evaluation, TCBench provides deterministic and probabilistic storm-following metrics. On 2023 test cases, neural weather models skillfully forecast TC tracks, while skillful intensity forecasts require additional steps such as post-processing. Designed for accessibility, TCBench helps AI practitioners tackle domain-relevant TC challenges and equips tropical meteorologists with data-driven tools and workflows to improve prediction and TC process understanding. By lowering barriers to reproducible, process-aware evaluation of extreme events, TCBench aims to democratize data-driven TC forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TCBench, a comprehensive benchmark designed for tropical cyclone forecasting. It provides standardized datasets and evaluation protocols to facilitate fair comparison and advancement in cyclone prediction research. The benchmark emphasizes reproducibility and realistic forecasting scenarios to encourage the development of more accurate and reliable models.

### Strengths
- This study presents a variety of benchmark tasks and experimental protocols for tropical cyclone prediction, encompassing data preprocessing, visualization tools, and evaluation metrics. In particular, it highlights the challenge of Rapid Intensification, pointing out the limitations of existing data-driven approaches in effectively capturing this phenomenon.

- In contrast to previous data-driven methods that have mainly focused on reducing errors in track prediction, this study emphasizes the importance of accurately forecasting intensity, which directly determines the potential damage tropical cyclones can cause to urban areas.

### Weaknesses
- In Line 144, the term “real-time-available data” is used, but ERA5 is a reanalysis dataset, which means it is not available in real time. Therefore, it seems that real-time prediction would not be possible through TCBench.

- This study proposes a benchmark framework that introduces various tasks and conducts experiments using baseline models. Since it aims to cover a wide range of aspects, there is still room for further experiments to demonstrate the utility of the benchmark. As shown in Figure 2(b), tropical cyclones of varying intensities occurred across different regions and seasons over six years, yet there is no detailed analysis of these cases. This work should include experiments analyzing the performance of each model across different seasons and scenarios, identifying where and when each model performs well or poorly.

### Questions
- As stated in Line 032 — “Tropical cyclones (TCs), also called ‘hurricanes’ or ‘typhoons’ depending on the basin” — the terminology differs depending on the region of occurrence. However, there appear to be no experiments analyzing the performance differences or prediction characteristics among these different types of tropical cyclones. Such an analysis could enhance the reliability and comprehensiveness of the benchmark dataset.
- Models such as typhoon trajectory prediction models presented in [1] and [2] could also serve as valuable baselines for comparison. Please provide the experimental results comparing with these models.
- The prediction of Rapid Intensification (RI) is formulated as a binary classification task, but it would be helpful to explain how this formulation contributes to practical disaster prevention and emergency response, particularly from a humanitarian or operational preparedness perspective.
- In Figure 4(c) of [3] (Pangu-Weather), the lead time of 120 hours corresponds to an error of approximately 200 km, whereas Figure 3(a) of TCBench shows that the track error of Pangu-Weather is significantly higher. Is this discrepancy due to different evaluation metrics between DPE and [3], or because different test datasets were used? Please provide the experimental results using consistent evaluation metrics and test datasets with those in [3].

[1] Huang, Cheng, et al. "MGTCF: multi-generator tropical cyclone forecasting with heterogeneous meteorological data." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 4. 2023.

[2] Park, Young-Jae, et al. "Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach Without Reanalysis Data." The Twelfth International Conference on Learning Representations. 2023. 

[3] Bi, Kaifeng, et al. "Accurate medium-range global weather forecasting with 3D neural networks." Nature 619.7970 (2023): 533-538.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
TCBench​ is a benchmark for evaluating 1–5 day global tropical cyclone (TC) track and intensity forecasts. It uses ​IBTrACS​ observations and treats forecasting as predicting storm evolution from given initial conditions, covering both physical (TIGGE) and neural models (Pangu-Weather, etc.). While neural models perform well on track prediction, intensity forecasts need post-processing. Designed for accessibility, it provides evaluation metrics and tools to help AI and meteorology researchers improve TC prediction and understanding.

### Strengths
1.TCBench establishes a standardized and relatively fair evaluation pipeline. It uses IBTrACS as the "ground truth," converting all data into a unified format based on IBTrACS identifiers for consistency. For models that do not provide readily available tracks, it employs the unified TempestExtremes library with consistent parameters to derive tracks from raw model outputs. When a model fails to forecast a storm, TCBench does not simply ignore the sample but fills it using the persistence baseline to prevent models from selectively forecasting only easier-to-predict TCs.

2.TCBench features a comprehensive and multi-dimensional evaluation framework. It assesses both deterministic (DPE, RMSE, MAE) and probabilistic (CRPS) aspects of model predictions, providing a holistic view of model accuracy and reliability. It specifically designs evaluation metrics like CSI and PSS for Rapid Intensification (RI) events, addressing the gap of systematic RI evaluation in traditional assessments.

3.TCBench implements strict data inclusion criteria and a clear data split. Storms are partitioned by year into training (2017-2020), validation (2021-2022), and test (2023) sets, preventing data leakage and ensuring the consistency and validity of the evaluation.

4.TCBench integrates diverse data sources, provides multiple baseline models, and offers full toolchain support. It incorporates IBTrACS, ERA5, physics-based models (e.g., GEFS, IFS), and neural weather models (e.g., FourCastNetv2, Pangu-Weather, AIFS, GenCast). It provides various baseline models (e.g., MT-LB, Pangu, FourCastNet) and post-processing models (MLR, ANN, UNet). It also offers detailed experimental guidelines, with open-sourced code for reproducibility, and is designed for extensibility.

5.TCBench demonstrates strong experimental results and broad application potential. Regarding track forecasting, all models show practical utility, with some neural models exhibiting leading deterministic performance. For intensity forecasting, errors are smaller at longer lead times; post-processing mitigates the weakness of raw neural models, while physics-based models show robust baseline performance. Notably, only the post-processed models demonstrate any skill in forecasting RI. Furthermore, TCBench shows application potential in predicting key scalars for wind field reconstruction, nowcasting short-term precipitation and TC winds, developing wind downscaling models, predicting global TC activity across timescales, and identifying key environmental predictors for TC intensity.

### Weaknesses
1.TCBench relies on the IBTrACS observational dataset as the ground truth. While IBTrACS is the most complete and authoritative global TC archive currently available, it has limitations:
(1) Its quality varies by basin (e.g., lower reliability in the South Indian Ocean), inconsistencies exist in the initial track points determined by different agencies, and it lacks rigorous cross-validation against other data sources (e.g., regional satellite observations, ground radar data). Therefore, its absolute accuracy cannot be guaranteed, and using it as the definitive benchmark represents an idealization.
(2) For capturing RI events, the evaluation is confined to 24-hour windows, while IBTrACS data is at 6-hour intervals. Consequently, the precise timing and peak intensity of RI events might not be fully resolved, posing challenges for evaluating RI forecasts. As indicated by the experimental results, the assessment of extreme event forecasting capability remains insufficient.

2.When model-specific TC track information is unavailable, TCBench uses the TempestExtremes library to derive tracks from raw data. TempestExtremes identifies all candidate systems meeting its criteria, including short-lived, non-TC pressure systems. These must then be filtered by spatiotemporally matching them to known TCs in IBTrACS. This process introduces several issues:
(1) The algorithm detects numerous short-lived features (spurious tracks), increasing computational complexity and overhead.
(2) Applying the same unified TempestExtremes methodology to all model outputs may not optimally capture the TC center for each individual model.
(3) The matching process between TempestExtremes output and IBTrACS is complicated by significant disparities in model "coverage" (Figure 7) of the IBTrACS records. A realistically forecasted TC track that differs slightly from the IBTrACS record might be incorrectly filtered out as a "spurious track," thereby inflating the perceived model error.

3.In TCBench, using raw AI model outputs directly for TC intensity prediction often yields poor results (as shown in Figure 3). Consequently, post-processing models are introduced to refine the intensity forecasts from the raw neural weather models. This process itself has potential drawbacks:
(1) It may accumulate and amplify errors from the upstream AI model.
(2) As a purely data-driven optimization method, it can potentially produce results that are physically inconsistent from a meteorological perspective.
(3) The post-processing focuses solely on correcting intensity forecasts and does not address errors in track predictions.

### Questions
(1) Incorporate ablation experiments. The paper proposes numerous norms and innovations for improving the TC forecasting evaluation system, such as the unified TempestExtremes library for track derivation, the dedicated evaluation framework for Rapid Intensification (RI) events, and the strict partitioning of training, validation, and test sets by year. Conducting ablation studies using these specific features as model variants would effectively demonstrate the performance differences under various configurations. This would more robustly validate the critical importance of these design choices within the TCBench framework.

(2) In Section 3.2 (Data Inclusion Criteria), seven requirements are listed. However, the definitions of some criteria are overly vague. For instance:
- Criterion (e) does not specify which variables are included under "environmental fields".
- Criterion (g) lacks a precise definition and methodology for calculating "resolution" (which may vary across datasets).
It is recommended to add a "Detailed Specification Table for Data Inclusion Criteria" in an appendix to clarify these points explicitly.

(3) Integrate references to meteorological literature to strengthen analytical arguments and explanations. For example:
- When defining the forecasting tasks for TC track and intensity, cite relevant meteorological studies to elaborate on the driving factors of TC motion and the physical mechanisms governing intensity change.
- When analyzing RI prediction results, introduce literature discussing the inherent complexity and challenges in forecasting RI due to its multi-faceted causal factors.

(4) Provide a comprehensive workflow diagram illustrating the entire TCBench pipeline. A visual representation would greatly enhance the intuitive understanding of the process, from data ingestion and standardization to track processing, model evaluation, and benchmark generation.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes TCBench, a benchmark for global tropical cyclone and intensity forecasting. TCBench uses IBTrACS observational data as the ground truth, and formulates TC forecasting as predicting the temporal evolution of both the location (latitude and longitude) and the intensity (maximum sustained wind speed). TCBench supports benchmarking both numerical models and data-driven methods like AIFS, Pangu-Weather, FourCastNetv2, and GenCast. TCBench provides deterministic and probabilistic storm-following metrics.

### Strengths
- The paper is original to the best of my knowledge.
- The paper is significant. It proposes a meaningful step forward for the field of data-driven weather forecasting, where most existing benchmarks evaluate the overall accuracy of methods while ignoring the equally important aspect of predicting extreme events like cyclones.
- The benchmark uses a standard data format and consistent evaluation pipelines, which ensures fairness and reproducibility.
- The benchmark provides post-processing steps to parse neural model outputs and supports both deterministic and probabilistic models.

### Weaknesses
- One major weakness is that the benchmark only considers forecasting tracks and the intensity of an existing cyclone, not an upcoming one. However, this is still a valid setting and has practical relevance.

### Questions
- How do we finetune a deep learning model with the provided cyclone data in TCBench?
- Is it possible to extend the benchmark to consider forecasting future/upcoming cyclones?

### Soundness
3

### Presentation
3

### Contribution
3
