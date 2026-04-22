# Semantic-Enhanced Time-Series Forecasting via Large Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Time series forecasting plays a significant role in finance, energy, meteorology, and IoT applications. Recent studies have leveraged the generalization capabilities of large language models (LLMs) to adapt to time series forecasting, achieving promising performance. However, existing studies focus on token-level modal alignment, instead of bridging the intrinsic modality gap between linguistic knowledge structures and time series data patterns, greatly limiting the semantic representation. To address this issue, we propose a novel Semantic-Enhanced LLM (SE-LLM) that explores the inherent periodicity and anomalous characteristics of time series to embed into the semantic space to enhance the token embedding. This process enhances the interpretability of tokens for LLMs, thereby activating the potential of LLMs for temporal sequence analysis. Moreover, existing Transformer-based LLMs excel at capturing long-range dependencies but are weak at modeling short-term anomalies in time-series data. Hence, we propose a plugin module embedded within self-attention that models long-term and short-term dependencies to effectively adapt LLMs to time-series analysis. Our approach freezes the LLM and reduces the sequence dimensionality of tokens, greatly reducing computational consumption. Experiments demonstrate the superiority performance of our SE-LLM against the state-of-the-art (SOTA) methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Key contributions include:

1. Enhances token embeddings by infusing temporal patterns (periodicity, anomalies) into LLM semantic spaces.

2. A plugin for self-attention that models short/long-term dependencies, adapting frozen LLMs to time-series forecasting.

3. Freezes LLM parameters and reduces token sequence dimensionality, lowering computational costs.

4. Outperforms SOTA on benchmarks (ETTh1, Traffic, M4) in long/short-term and zero-shot forecasting.

### Strengths
1. Novel integration of temporal dynamics (anomalies, periodicity) into LLM semantic spaces via TSCC, addressing a critical modality gap.
2. Time-Adapter innovatively combines LSTM with low-rank projections to capture multi-scale dependencies, unlike prior token-level alignment methods.
3. TSCC’s AM-VAE for anomaly modeling and cross-correlation filtering is technically sound.
4. Well-structured with clear figures and pseudocode.

### Weaknesses
1. Experiments limited to smaller LLMs. Larger models (e.g., Llama-7B) are untested due to computational constraints, raising questions about generalizability.

2. AM-VAE’s reconstruction loss (Algorithm 1) lacks quantitative comparison to other anomaly detection methods

3. Missing comparisons to hybrid methods (e.g., LLM + GNNs) in Table 1

4. No discussion on latency or memory overhead for edge/IoT applications, despite claims of efficiency.

### Questions
1. Could the authors estimate the performance trade-offs for larger LLMs (e.g., Llama-7B) without full training?

2. Why choose VAE over other generative models (e.g., Diffusion Models) for anomaly simulation in TSCC?

3. Have you explored comparing TSCC with graph-based temporal models (e.g., MTGNN) to capture spatial dependencies?

4. How does TSCC handle datasets with non-stationary distributions (e.g., financial time-series)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper works on time series forecasting (TSF) and presents an LLM-based TSF framework SE-LLM to address the modality gap between language and time series. It mainly includes:  (1) TSCC which infuses temporal patterns (e.g., periodicity, anomalies) into semantic embeddings, and (2) Time-Adapter that can capture both long-term and short-term dependencies in time series. The model freezes the LLM backbone to preserve generalizability and reduces computational cost via sequence dimensionality reduction. Experiments on long-term, short-term, and zero-shot forecasting tasks demonstrate performance improvements over state-of-the-art methods, with ablation studies validating the effectiveness of the proposed modules.

### Strengths
1. It's reasonable to study the modality gap between language and time series, and integrate temporal patterns to bridge the gap.
2. Time-Adapter’s dual LSTM paths for long/short-term dependency modeling can complement Transformers’ weaknesses in temporal dynamics.
3. Extensive experiments have been conducted to validate the effectiveness of modules.

### Weaknesses
1. While TSCC and Time-Adapter address known limitations, their individual components (cross attention, VAE for anomaly modeling, adapter-based fine-tuning) are not fundamentally new. The paper fails to sufficiently articulate how their combination offers a unique contribution beyond incremental improvements over existing modular designs.
2. In Table 1, the performance gain over the best baseline is minor in almost all cases.
3. The choice of lightweight LLMs (GPT2, Qwen2.5-0.5B) limits generalizability. No experiments on larger LLMs (e.g., Llama-7B) raise questions about scalability, even with the claim of prioritizing efficiency.
4. Zero-shot forecasting results lack analysis of why the model generalizes across datasets.
5. The impact of different temporal patterns (e.g., seasonal vs. irregular) on performance is unexplored.
6. The exclusion of Time-Adapter from some short-term/zero-shot experiments due to computational constraints undermines the framework’s completeness.
7. The description of the TSCC module’s channel dependency enhancement and the Time-Adapter’s integration into multi-head attention is vague. Key formulas (e.g., Eq 8) lack sufficient explanation.
8. (Minor) The natbib package is not correctly used.
9. (Minor) There are some typos, e.g., (1) in Figure 2(b), "JS with Temporal-Semantc" -> "JS with Temporal-Semantic", (2) in Table 1, 4, 6, 12, 14, "Mertics" -> "Metrics".
10. (Minor) In Table 7, some column names are missing.

### Questions
N.A.

### Soundness
2

### Presentation
3

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
This paper presents a framework that enhances the performance of using LLM for time series forecasting by bridging the modality gaps between time series numerical data and language word embeddings. Specifically, it designs two modules: Temporal-semantic cross-correlation, which aligns temporal embedding with LLM's semantic space using cross-attention and anomaly modeling; and the Time-Adapter, which is a plug-in module that augments the LLM attention mechanism through dual LSTM pathways. The evaluation on multiple datasets and in various settings demonstrates that the SE-LLM consistently outperforms baselines and ablation studies show the effectiveness of design modules.

### Strengths
1. The designed TSCC module addresses the modality gaps between temporal embeddings and the semantic space of LLMs by employing cross-attention and a VAE-based anomaly decomposition. It presents a more interpretable semantic representation for time series forecasting.

2. The low-rank style of using dual LSTM is suitable for freezing LLMs for time series forecasting. It compensates for the Transformer’s weaknesses in handling local and global temporal dependencies while maintaining parameter efficiency.

3. The paper evaluates SE-LLM across a wide range of forecasting horizons, datasets, and various settings, showing the robustness of the designed model.

### Weaknesses
1. The pipeline presents several interdependent components such as cros-attention, AM-VAE, top-k correlation filtering and gating fusion, which make the pipeline highly complex. The implementation details are missing.

2. Missing qualitative analysis showing how the semantic space affects LLM token representations for time series forecasting. Whether the model truly benefits from the semantic space is questioned, which is a claim of this paper. 

3. The designed component introduces significant model complexity; thus, it is important to compare the number of parameters or states training requirements.

### Questions
See weakness

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
The paper addresses adapting large language models (LLMs) to time series forecasting without fully fine-tuning the LLM backbone, targeting long-term trends, short-term dynamics, cross-channel dependencies, and anomalous behaviors. It proposes SE-LLM, which combines a Temporal-Semantic Cross-Correlation (TSCC) module that aligns time series embeddings with the LLM semantic space, separates anomalous and de-anomalized semantics, and performs gated multi-channel fusion, with a Time-Adapter module that injects lightweight temporal inductive bias into the attention key and value paths via parallel long- and short-horizon branches. The contribution is a modular architecture that augments a mostly frozen LLM through cross-modal temporal–semantic alignment and structured temporal adapters instead of full fine-tuning. The method is evaluated on long-horizon multivariate forecasting benchmarks, short-term forecasting benchmarks at different frequencies, and cross-domain transfer settings, reporting reductions in standard forecasting metrics (MSE, MAE, SMAPE, MASE, OWA) relative to recent LLM-based and transformer-style baselines, supported by ablations isolating TSCC and Time-Adapter.

### Strengths
1. The paper proposes a structured framework (SE-LLM) intended to bridge the mismatch between raw temporal dynamics and the semantic token space of a largely frozen LLM by explicitly aligning time series signals with language representations and injecting temporal inductive bias through specialized modules.
2. The architecture separates into TSCC and Time-Adapter. TSCC performs temporal–semantic cross-alignment, anomaly and non-anomalous semantic disentanglement, and gated multi-channel fusion, while Time-Adapter augments attention key and value paths using lightweight temporal branches for long-term and short-term dynamics instead of full backbone fine-tuning.

### Weaknesses
1 The paper attributes interpretability to anomaly and non-anomalous semantic separation and channel-gated fusion, but it does not present qualitative, per-sequence visual evidence (e.g., highlighted time steps or channels) that would allow readers to verify that the model is focusing on meaningful temporal events rather than incidental fluctuations.  
2 Reported efficiency comparisons (e.g., training and inference cost versus error) do not specify the hardware environment, batch size, sequence lengths, or caching assumptions, which limits the reproducibility and interpretability of the efficiency and lightweight adaptation claims.  
3 Some core components are only partially specified at the technical level. For example, the anomaly modeling and semantic disentanglement step (AM-VAE within TSCC) is described conceptually, but the precise probabilistic objective (e.g., reconstruction terms and regularization) is not fully detailed, so it is difficult to assess how the model enforces the claimed anomalous versus de-anomalous semantic separation.  
4. The presentation quality is below typical conference standards, including missing or underspecified captions (Figure 1 does not explain subfigure (c), the efficiency plot lacks runtime conditions, and Table 6 does not clearly describe the comparison protocol).

### Questions
1. Can you provide statistical variation (for example, multiple random seeds, confidence intervals, or significance tests) for the main quantitative comparisons to recent LLM-based and transformer-style baselines in the primary result tables?  
2. Can you include qualitative analyses or failure cases that visualize which time points and channels TSCC marks as anomalous or high-impact, and how this correlates with the model's predictions, in order to substantiate the interpretability claims?  
3. Can you precisely describe the runtime and efficiency measurement setup (hardware, batch size, input length, prediction horizon, caching policy) used to generate the reported training and inference cost comparisons, so that the claimed efficiency of the proposed Time-Adapter can be reproduced?

### Soundness
3

### Presentation
2

### Contribution
2
