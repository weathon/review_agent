# TiMi: Empowering Time Series Transformers with Multimodal Mixture of Experts

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Multimodal time series forecasting has garnered significant attention for its potential to provide more robust and accurate predictions than traditional single-modality models by leveraging rich information inherent in other modalities. However, due to fundamental challenges in modality alignment, existing methods often struggle to effectively incorporate multimodal data into predictions, particularly textual information that has a causal influence on time series fluctuations, such as emergency reports and policy announcements.
In this paper, we reflect on the role of textual information in numerical forecasting and propose **Ti**me series transformers with Multimodal **Mi**xture-of-Experts, **TiMi**, to unleash the causal reasoning capabilities of LLMs. Concretely, TiMi utilizes language models to generate inferences on future developments, which then serve as guidance for time series forecasting.
To seamlessly integrate both exogenous factors and time series into predictions, we introduce a Multimodal Mixture-of-Experts (MMoE) module as a lightweight plug-in to empower Transformer-based time series models for multimodal forecasting, eliminating the need for explicit representation-level alignment. Experimentally, our proposed TiMi demonstrates consistent state-of-the-art performance on sixteen real-world multimodal forecasting benchmarks, outperforming advanced unimodal and multimodal baselines while offering strong adaptability and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes TiMi, a time-series–centric Transformer enhanced with a Multimodal Mixture-of-Experts (MMoE) that injects causal guidance from text. A frozen LLM first extracts structured inferences about future developments (trends/periodicity/shocks) from exogenous text. Then they are routed via a Text-informed MoE (TMoE), while historical series form a global representation that routes a Series-aware MoE (SMoE). This design avoids explicit representation-level alignment and aims to guide prediction instead of fusing features. Across multiple real-world multimodal forecasting benchmarks, TiMi reports consistent SOTA over unimodal and multimodal baselines, with claims of adaptability and interpretability.

### Strengths
1. Clear problem reframing and architecture: The paper articulates why text–series semantic misalignment makes standard early/late fusion suboptimal and instead uses text as guidance through MoE routing (TMoE + SMoE), which is a clean, modular idea that fits common TS Transformers.

2. Good experiment results: The proposed TiMi method consistently achieves superior forecasting performance on multiple datasets than baselines, with notable error reductions.

### Weaknesses
1. The proposed TiMi presumes that available text causally informs future series. However, when text is noisy, off-topic, or adversarial, TMoE routing might misguide the forecaster. Besides, the proposed method does not thoroughly stress-test this with ablation on text quality, noise level, or contradictory narratives. 

2. While TiMi leverages LLM causal reasoning, the experiments don’t include causal identification/diagnostics (e.g., interventions, counterfactual text deletion, or do-calculus-style tests). Instead, current case studies are suggestive but do not disentangle correlation from causation.

### Questions
See weaknesses

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
The authors propose Time-series Transformers with multimodal Mixture of experts (TiMi) for multimodal time series forecasting. By incorporating a Multimodal Mixture-of-Experts as a plug-in module into Transformer-based time series forecaster, it enables the seamless integration of structured extracted knowledge into context-based prediction, rather than performing vague feature-level fusion. The authors demonstrated the effectiveness of their algorithm through comparative experiments with unimodal and multimodal baselines.

### Strengths
The model architecture demonstrates a certain degree of innovation. Leveraging the concept of selective routing of MoE, it introduces TMoE and SMoE modules, which enable adaptive representation learning of historical time series and extracted future textual guidance, respectively.

### Weaknesses
1. With the continuous advancement of research in multimodal time series forecasting, many new SOTA models have emerged in the past year. The selected baselines do not cover the latest SOTA models. It is recommended to include more recent SOTA models for comparison, particularly with a greater focus on multimodal time-series forecasting models.  

2. There is a lack of error bar analysis.  

3. The explanation for Figure 6 is not sufficiently clear and should be revised.

### Questions
Using large language  models to extract information may, on one hand, involve potential data leakage issues, and on the other hand, result in extracted information that is irrelevant or misleading to the time series data. 

Regarding the data leakage problem, how can we ensure that no information leakage exists, and how do we evaluate whether the model's performance improvement stems from its architecture or from leaked future information? 

As for irrelevant or misleading extracted information, part of the issue arises from the inherent noise in textual data itself, while another part stems from the hallucination problems inherent in large language models. I am curious about how the TiMi framework addresses these two aspects respectively?

### Soundness
3

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
4

### Summary
TiMi tries to solve the problem of aligning textual data with time-series for forecasting tasks. They propose a guidance based approach where the textual embeddings generated from LLMs are used to guide some of the MoE experts of the transformer backbone whereas other experts are guided by the backbones' temporal embeddings. This approach is claimed to be more efficient than early or late fusion approached of past works and enables better performance on wide range of domains. The ablations show the importance of mixed multimodal MoE approach, with case studies showing examples of textual information guiding the forecasts.

### Strengths
1. The methodology is well motivated is non-trivial
2. helps solves an important problem in time+text forecasting
3. Results are promising across wide range of benchmarks from past datasets

### Weaknesses
1. Can the models adapt to varying degree of textual information? In many applications text data is sparse and not as well aligned as those used in say Time-MMD which the paper uses. Does the model handle such situations by adapting to use more of temporal MoEs when necessary?
2. How does the model compare against other integrated solutions such as using both time and text inputs as part of single embedding representations (like OpenTSLM https://arxiv.org/abs/2510.02410) ?
3. How does the performance degrade with increase in horizon lenght? How are different MoEs used at longer horizons?

### Questions
See Weaknesses

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
3

### Summary
The paper is about a multimodal model to process text and numerical data for time series forecasting. Time series transformers with multimodal MoE (TiMi) uses language models toto guide the time series forecasting.  To integrate exogenous text + numerical data, the authors introduce a TMoE (text MoE) and SMoE (series aware MoE), in order to circumvent text-series representation alignment. Experiments are performed on the Time-MMD and Time-IMM datasets (16 datasets total). The model is evaluated against unimodal and multimodal approaches and achieves superior performance compared to baselines.

### Strengths
1. The proposed approach embeds an MoE modules for both the series and the text instead of explicitly aligning both modalities. The LLM then guides the prediction of the primary time-series branch. To the best of my knowledge this is novel. The design is well-motivated for semantically misaligned series–text pairs common in practice.

2. The modular approach allows authors to plug in the MMoE into other transformer based methods (PatchTST, autoformer etc, table 2). In all cases, the MMoE improves predictive performance which clearly shows the advantage of the methods.

3. The ablation on LLMs also seems to suggest that stronger/larger LLMs improve the model performance, which make the method general.  

4. The Mann-Kendall trend test to detect monotonic trends adds a layer of interpretability to the model.

### Weaknesses
1. The work misses providing finer details of the exogenous text data (how it is obtained or generated), and how the LLM is prompted for guiding the time-series prediction. The “causal knowledge” claim hinges on how text is curated, and aggregated; beyond average pooling of LLM embeddings, key preprocessing choices and robustness to noisy/off-topic text are not thoroughly stress-tested. 

2. Given many works in the (multimodal) TSF community and experimental settings and implementations often differing between papers, I believe submission of code (ideally for both the method and baselines) would strengthen the paper. Currently I see a single python file submitted for the supplementary material containing just the model class. But this misses details on how data processing was performed and how experiments were run on baselines etc.

### Questions
- How sensitive is the TMoE routing to noisy or irrelevant text?
- L446 says figure 4.3 but I am unable to find it.

### Soundness
3

### Presentation
3

### Contribution
3
