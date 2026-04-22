# Multi-Scale Hypergraph Meets LLMs: Aligning Large Language Models for Time Series Analysis

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Recently, there has been great success in leveraging pre-trained large language models (LLMs) for time series analysis. The core idea lies in effectively aligning the modality between natural language and time series. However, the multi-scale structures of natural language and time series have not been fully considered, resulting in insufficient utilization of LLMs capabilities. To this end, we propose MSH-LLM, a Multi-Scale Hypergraph method that aligns Large Language Models for time series analysis. Specifically, a hyperedging mechanism is designed to enhance the multi-scale semantic information of time series semantic space. Then, a cross-modality alignment (CMA) module is introduced to align the modality between natural language and time series at different scales. In addition, a mixture of prompts (MoP) mechanism is introduced to provide contextual information and enhance the ability of LLMs to understand the multi-scale temporal patterns of time series. Experimental results on 27 real-world datasets across 5 different applications demonstrate that MSH-LLM achieves the state-of-the-art results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MindTS, a multimodal time-series anomaly detection framework that leverages fine-grained alignment between time-series segments and textual signals from two sources: endogenous (template-guided descriptions generated per segment) and exogenous (external background texts). The method first aligns time–text features via cross-view attention and contrastive objectives, then performs a content condensation step (information-bottleneck-style masking) to reduce textual redundancy, and finally conducts cross-modal reconstruction with masked time-series to compute anomaly scores. Experiments over six real-world datasets and 17 baselines show consistently strong or SOTA performance, with ablations validating the necessity and ordering of the key modules.

### Strengths
1. The design of each modular ( alignment → condensation → cross-modal reconstruction ) is well-motivated and confirmed by ablation studies.
2. The empirical coverage is broad: 6 datasets, 17 baselines, plus multimodalized baselines,  with consistent gains and careful sensitivity studies.
3. It provides insightful guidance on the coordinated use of endogenous and exogenous texts in alignment with time-series signals, and the proposed method for assessing and condensing informative content is compelling.
4. The exposition is well-structured and logically coherent, with rigorous reasoning throughout.

### Weaknesses
1. Limited depth in optimizing multimodal alignment. Multimodal alignment somtimes trained as a standalone objective (e.g., BLIP2), whereas here it is treated as an auxiliary task, which may constrain the depth of alignment achieved. Although Figure 3 reports preliminary component ablations, a more systematic study would substantiate the alignment claims.
2. Insufficient ablation on endogenous prompt design. The construction of endogenous prompts appears to substantially affect performance in time-series contexts (cf. TimeLLM, HiTime). Some controlled ablations over prompt templates, statistical descriptors, and temporal granularity would clarify robustness and inform design choices.

### Questions
1. It is better to train the multimodal alignment as a standalone objective (rather than only as an auxiliary loss) and report a dedicated set of results—ideally comparing standalone vs. auxiliary alignment settings.
2. It is better to provide a ablation study on the design of endogenous prompts (e.g., template variants, statistical descriptors, temporal granularity), to quantify their impact on the final performance.
3. The framework appears extensible to broader multimodal time-series tasks (e.g., classification and forecasting). It is better to evaluate on a few forecasting datasets, or more narrowly, on forecasting-based anomaly detection (early-warning) to assess transferability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a Multi-Scale Hypergraph method that aligns large language models for time series analysis. A hyperedging mechanism is designed to enhance the multi-scale semantic information of the time series semantic space, while a mixture of prompts mechanism is introduced to provide contextual information. Extensive experimental results on real-world datasets demonstrate the superior performance of the proposed method.

### Strengths
- The paper introduces a multi-scale hypergraph framework to align LLMs for time series analysis. The proposed hyperedging mechanism and mixture of prompts strategy demonstrate better performance beyond existing single-scale or prompt-based methods.
- This paper is well-written and easy to follow.
- The framework is technically solid and systematically evaluated on 27 real-world datasets covering forecasting, classification, few-shot, and zero-shot learning.

### Weaknesses
- The provided code files could not be opened successfully, which raises concerns about the reliability of the reported experimental results.

- The overall framework appears to combine elements from Time-LLM [1] and hypergraph-based multi-scale modeling. As a result, the methodological novelty may be limited unless the authors can clearly articulate the unique contributions beyond this integration.

- The Cross-Modality Alignment (CMA) module used for aligning text and time-series modalities has already been proposed in prior work [2]. Since this paper does not cite that reference, it is unclear whether the presented CMA design introduces any substantive technical advancement.

[1] Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. ICLR 2024.

[2] TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment. AAAI 2025, 39(18): 18780–18788.

### Questions
- Could the authors clarify how their proposed hyperedging mechanism differs from prior hypergraph-based models such as MSHyper or Ada-MSHyper in terms of functionality?

- Since the CMA module is conceptually similar to that in TimeCMA [2], could the authors explain what improvements are brought by its integration in this work?

- Each file in the code repository link cannot be opened successfully. Could the authors ensure reproducibility by providing a working link or additional implementation details?

- The paper claims novelty in combining multi-scale structures and LLMs. Could the authors discuss whether this combination provides theoretical advantages beyond empirical gains, and how generalizable it is to other LLM backbones?

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
To address the modality alignment between natural language and time series in LLM-based time series analysis, this paper focuses on multi-scale structures by proposing a novel multi-scale hypergraph method, MSH-LLM. This method integrates multi-scale extraction, hyperedging, cross-modality alignment and mixture of prompts mechanisms to enhance the multi-scale semantic information in LLM-based analysis. Experimental results demonstrate the effectiveness of MSH-LLM.

### Strengths
1. This paper introduces a hyperedging mechanism to extract group-wise information from multi-scale temporal features, enhancing multi-scale semantic information of time series semantic space as a novel contribution.

2. This paper designs a cross-modality alignment module to perform multi-scale alignment and obtain richer representations, while the mixture of prompts mechanism enhances the reasoning ability of LLMs.

3. This paper conducts comprehensive experiments on 27 real-world datasets across 5 different applications, demonstrating the effectiveness of MSH-LLM.

### Weaknesses
1. In Sec 4.1, the word token embeddings U are transformed into U1 through linear mapping. Is the linear mapping learnable or a semantic-distance-based mapping? Will each text prototype possesses explicit meaning? 

2. As the pre-trained LLM is freezed during training according to Figure 1, I am confused whether all the proposed modules will be trained simultaneously. If so, what are the training objectives? 

3. Experimental results in Table 1 and 2 show minor improvements between MSH-LLM and the runner-up methods. As the base LLM is the early LLaMa-7B, it is hard to judge whether the performance gap come from the LLM backbones or the method itself.

### Questions
See weakness part for details.

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
The paper addresses the problem of multi-scale semantic misalignment between large language models (LLMs) and time series data by proposing MSH-LLM, a cross-modal alignment framework based on multi-scale hypergraphs. The core innovation lies in the hyperedge mechanism, which enhances the multi-scale semantic representation of time series, together with a hybrid prompting strategy that activates the temporal reasoning capability.

### Strengths
1. The paper insightfully identifies the multi-scale structural misalignment between natural language and time as a fundamental bottleneck for applying LLMs to time-series analysis.

2. By introducing learnable hyperedges and a sparsification strategy, the proposed method not only mitigates the noise sensitivity of traditional patch-based approaches but also captures implicit group-level interactions within time series data.

### Weaknesses
1. The paper does not justify the node–hyperedge similarity in the hyperedging mechanism (Eqs. 2–3), nor does it analyze the theoretical relation between the number of hyperedges M^s and the time-series dimensionality D.

2. The work does not report inference latency on ETTh1 with H=720, so it is unclear whether the method is suitable for real-time scenarios such as power-load dispatch. Memory consumption is also missing; GPU usage on high-dimensional data is not reported, limiting an assessment of deployability.

3. Within the capability-enhancing prompts, “logical reasoning appears to overlap with 'time-series reasoning, but there is no ablation to test redundancy.

### Questions
1. Please include training-time trajectories of the hyperedge embeddings E_hyper^s. An analysis that maps dominant embedding directions to known multi-scale patterns in the data, for example, daily and weekly cycles in ETT, could substantially improve the interpretability of the hyperedging mechanism.

2.  Please evaluate the robustness of MSH-LLM under extreme conditions. Specifically, include results on ETTh1 subsets with 10% to 20% injected anomalies, ultra-long forecasting with H=1008, and Electricity subsets with 5% missing data. A comprehensive discussion of how the model behaves across these challenging scenarios—and particularly where performance begins to degrade—would provide valuable insight into the robustness boundaries of MSH-LLM.

3.  Please provide inference latency and memory usage on an NVIDIA A100. For example, report single-pass inference time on ETTh1 with H=720and the peak GPU memory under batch size =32. These measurements are important to assess real-time feasibility.

4.  Please run targeted ablations that remove only the “logical reasoning” prompt or only the “time-series reasoning” prompt, and report the resulting performance changes. This would help determine whether the two components are redundant and if they can be merged to simplify the prompt design.

### Soundness
3

### Presentation
2

### Contribution
2
