# HURST: Learning Heterogeneity-Adaptive Urban Foundation Models for Spatiotemporal Prediction via Self-Partitional Mixture-of-Spatial-Experts

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Urban foundation models (UFMs) are pre-trained spatiotemporal (ST) prediction models with the ability to generalize to different tasks. Such models have the potential to transform urban intelligence by reducing domain-specific models and generalizing to tasks with limited data. However, building effective UFMs is a challenging task due the existence of spatial heterogeneity in ST data, i.e., data distribution and relationship between attributes vary over space. Existing UFMs lack sufficient consideration of this important issue and thus have unsatisfactory performance over spatially heterogeneous urban settings. To address this limitation, this paper proposes  $\textbf{HURST}$, a $\underline{\textbf{H}}$eterogeneity-Adaptive $\underline{\textbf{UR}}$ban Foundation Model for $\underline{\textbf{S}}$patio-$\underline{\textbf{T}}$emporal Prediction, that is capable of capturing the spatial pattern of heterogeneity underlying the urban setting to enhance the UFM's performance. HURST presents two key technical innovations: (1) a self-partitional Mixture-of-Spatial-Experts (MoSE) network that automatically learns to stratify urban areas into partitions, where region-specific expert networks are trained in a hierarchical manner, and (2) an error-guided adaptive spatio-temporal masking strategy that dynamically adjusts masking patterns based on region-specific training feedback. A prompt-tuning strategy is also designed to facilitate the above innovations. Comprehensive experiments on over ten datasets from three urban areas of varying sizes show that HURST achieves up to 46.9\% performance gain over SOTA baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces HURST, a heterogeneity-adaptive urban foundation model for spatiotemporal prediction. It captures spatial heterogeneity through a self-partitioning mixture-of-spatial-experts (MoSE) network that stratifies urban areas into partitions. An error-guided adaptive spatiotemporal masking strategy further refines learning by dynamically adjusting masking patterns based on region-specific training feedback. Experiments on ten datasets show that HURST achieves up to a 46.9% performance improvement over state-of-the-art baselines.

### Strengths
1. The use of MoSE to automatically stratify urban areas into partitions for addressing data heterogeneity is novel. The case study demonstrates that HURST effectively partitions urban areas.

2. The error-guided adaptive masking strategy helps the model focus on regions with higher reconstruction errors, which enhances performance under strong spatial heterogeneity.

3. The proposed method achieves strong performance across all-for-one, few-shot, and zero-shot prediction tasks on ten datasets spanning three geographic regions, with up to a 46.9% improvement over state-of-the-art baselines.

### Weaknesses
1. The key techniques are not clearly described, which may cause confusion. For instance, does the method partition urban areas based solely on target prediction data, such as traffic, accident, or crime records? Or does it also incorporate additional inputs like POI information to guide the partitioning process?

2. The statement “each expert has its own assigned region” is ambiguous. Does this imply a one-to-one mapping between regions and experts? How many experts are used at most in the experiments? Additionally, since using multiple experts may introduce computational overhead, it would be helpful to include a runtime comparison with baseline models to assess the trade-offs of the proposed method.

3. The experimental setup lacks clarity. What is the look-back window length? Which datasets are used for pre-training in each experiment? Given that datasets have different spatial dimensions, how is pre-training conducted on all data with inconsistent H×W?
The experimental setup for the zero-shot and few-shot settings is unclear. What are the exact configurations used in these experiments?

4. The experimental analysis is insufficient.

5. The method does not compare against several important baselines, including spatial embedding models like STAEformer [1], mixture-of-experts architectures such as TESTAM, and context-aware networks like DeepSTN+ that incorporate POI information.

6. Although the paper states that the relevant code has been deposited, the provided anonymized repository contains only a README file, and no code appears to be included.

[1] Liu, Hangchen, et al. "Spatio-temporal adaptive embedding makes vanilla transformer sota for traffic forecasting." Proceedings of the 32nd ACM international conference on information and knowledge management. 2023.

[2] Lee, Hyunwook, and Sungahn Ko. "TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts." The Twelfth International Conference on Learning Representations.

[3] Lin, Ziqian, et al. "Deepstn+: Context-aware spatial-temporal neural network for crowd flow prediction in metropolis." Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.

### Questions
1. The key techniques are not clearly described, which may cause confusion. For instance, does the method partition urban areas based solely on target prediction data, such as traffic, accident, or crime records? Or does it also incorporate additional inputs like POI information to guide the partitioning process?

2. The statement “each expert has its own assigned region” is ambiguous. Does this imply a one-to-one mapping between regions and experts? How many experts are used at most in the experiments? Additionally, since using multiple experts may introduce computational overhead, it would be helpful to include a runtime comparison with baseline models to assess the trade-offs of the proposed method.

3. The experimental setup lacks clarity. What is the look-back window length? Which datasets are used for pre-training in each experiment? Given that datasets have different spatial dimensions, how is pre-training conducted on all data with inconsistent H×W?
The experimental setup for the zero-shot and few-shot settings is unclear. What are the exact configurations used in these experiments?

4. The experimental analysis is insufficient.

5. The method does not compare against several important baselines, including spatial embedding models like STAEformer, mixture-of-experts architectures such as TESTAM, and context-aware networks like DeepSTN+ that incorporate POI information.

6. Although the paper states that the relevant code has been deposited, the provided anonymized repository contains only a README file, and no code appears to be included.

### Soundness
2

### Presentation
2

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
This paper addresses the challenge of spatial heterogeneity in building Urban Foundation Models (UFMs) for spatiotemporal (ST) prediction. To overcome heterogeneity, the authors propose HURST (Heterogeneity-Adaptive URban Foundation Model for Spatio-Temporal Prediction), a framework that adaptively learns spatial partitions and expert models, which integrates two key components: (1) a Self-Partitional Mixture-of-Spatial-Experts (MoSE) and (2) an Error-Guided Adaptive Spatiotemporal Masking strategy. Experiments on ten datasets from New York City, Chicago, and Iowa demonstrate that HURST achieves up to 46.9% improvement in prediction accuracy over state-of-the-art baselines while maintaining scalability and interpretability.

### Strengths
The paper is well written and presents a Heterogeneity-Adaptive Urban Foundation Model (HURST) that effectively addresses spatial heterogeneity in spatiotemporal prediction. The authors conduct comprehensive experiments on ten large-scale urban datasets from New York City, Chicago, and Iowa, covering diverse urban scenarios such as traffic, mobility, and crime. The experimental results demonstrate that HURST consistently outperforms state-of-the-art baselines by up to 46.9% in MSE and 45.2% in MAE, while also achieving strong zero-shot and few-shot generalization. Overall, the paper provides robust evidence for the model’s effectiveness, scalability, and generalizability in heterogeneous urban prediction tasks.

### Weaknesses
The article(line 64-73) briefly mentions but does not deeply discuss existing solutions to spatiotemporal heterogeneity. The paper does not provide an in-depth comparison of how these methods explicitly handle spatial or temporal heterogeneity, nor does it analyze their limitations quantitatively or conceptually.

### Questions
1. Why did the challenge of spatiotemporal heterogeneity (line 53) only address spatial heterogeneity?
2. Why choose to use a linear layer instead of static expert settings to compare MoE and MoSE in the w/o MoSE ablation study?

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
5

### Summary
This paper proposes a model called HURST, an Urban Foundation Model (UFM) designed to enhance the generalization capability of spatiotemporal prediction tasks. The authors argue that existing UFMs perform poorly when facing spatial heterogeneity, and therefore introduce two core innovations: the Self-Partitional Mixture-of-Spatial-Experts (MoSE), which adaptively partitions urban areas into semantically distinct regions and trains region-specific expert networks; and the Error-Guided Spatiotemporal Masking strategy, which dynamically adjusts masking patterns during pre-training based on reconstruction errors to better learn heterogeneous regions. In addition, a prompt-tuning mechanism is employed to facilitate effective knowledge transfer across tasks. Experiments conducted on ten real-world datasets from New York City, Chicago, and the state of Iowa demonstrate that HURST significantly outperforms existing state-of-the-art models, achieving up to a 46.9% improvement in prediction accuracy.

### Strengths
The two key innovations to address the challenge of spatial heterogeneity are reasonable.
First, the MoSE module adaptively partitions urban areas based on learned spatial heterogeneity patterns and trains specialized expert networks for each partition, effectively capturing region-specific dynamics.
Second, the Error-Guided Spatiotemporal Masking strategy dynamically adjusts masking patterns during the pre-training stage according to reconstruction errors, enabling the model to focus on heterogeneous or hard-to-learn regions.

### Weaknesses
1. The paper lacks the latest pre-trained models as baselines, such as UrbanGPT. 

2. The recent urban foundation models, such as UrbanDIT, already support multiple tasks like forecasting and imputation. Since HURST only supports forecasting, calling it a "foundation model" seems a bit of a stretch. 

3. The experimental setup, which uses one historical frame to predict only the next single future frame, is not a standard task in time-series literature (e.g., 12-step-ahead for 12 steps). Predicting just one frame cannot reveal whether the model has captured periodic spatio-temporal patterns. Moreover, the authors do not specify the temporal duration of one frame (seconds? half an hour?).

4. There are figure and formatting errors in the manuscript, such as the masking portion in Figure 1 and tables that are too long.

### Questions
1. While this paper introduces an error-guided masking strategy built upon random masking, a key benefit of masking in foundation-model training is to enable downstream transfer to diverse urban time-series tasks, e.g., imputation, or arbitrary cities. The proposed error-guided approach may cause the model to over-focus on certain time periods or locations, potentially limiting its ability to transfer to new tasks or cities.

2. The paper presents a study on hyper-parameters such as embedding dimension; however, the results suggest the model is almost insensitive to any of them, with performance hardly changes regardless of the settings (at least in Figure 7). For example, increasing the number of experts from two to eight yields virtually no gain, implying that the MoE architecture itself contributes little. The drop observed when MoE is removed may simply stem from the resulting reduction in total parameters. For HURST, are there other, more influential hyper-parameters that truly govern performance?

3. This paper borrows several design elements from UniST; nevertheless, with the same Transformer blocks and prompt network, HURST ends up with fewer parameters than UniST even after incorporating the MoSE module. It is rather counter-intuitive.

### Soundness
2

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
5

### Summary
This paper presents HURST, a heterogeneity-adaptive urban foundation model (UFM) designed for spatiotemporal prediction tasks. The key motivation stems from the challenge of spatial heterogeneity in urban data — where correlations and distributions vary across space and time — which existing UFMs fail to model effectively. Together with a prompt-tuning module for downstream adaptation, HURST aims to produce robust, generalizable spatiotemporal representations. Comprehensive experiments on ten datasets across three urban regions (New York, Chicago, Iowa) demonstrate substantial gains—up to 46.9% improvement in MSE over state-of-the-art baselines such as UniST and PromptST—across one-for-all, zero-shot, and few-shot prediction tasks.

### Strengths
1. Building foundation models for urban spatiotemporal prediction is an important and emerging direction.
2. The paper is easy to follow and well structured.
3. The paper conducts detailed experiments, including one-for-all, zero-shot, and few-shot evaluations.

### Weaknesses
1. Spatial heterogeneity is a well-studied problem. While the motivation is clear, the problem of modeling spatial heterogeneity has been extensively investigated in prior work. Thus, the novelty of the problem statement itself is somewhat limited.
2. The proposed self-partitional MoSE and error-guided masking are interesting combinations, but both ideas are similar to existing approaches in spatiotemporal MoE. The paper could be strengthened by a deeper discussion of how HURST differs fundamentally from recent adaptive MoE methods such as ST-MoE, HiMoE, or CP-MoE.
3. Although experiments cover three urban areas, all datasets are within similar domains (urban mobility, traffic, and service data) and from U.S. cities. This limited diversity makes it difficult to conclude whether HURST can generalize to other urban contexts.

### Questions
See in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
