# Deep Heterogeneity: A New Paradigm for  Time Series Forecasting

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Time series forecasting frequently employs signal decomposition to disentangle complex series into simpler, more structured components. However, a prevalent limitation within this paradigm is the principle of Shallow Heterogeneity, where structurally homogeneous processing blocks are indiscriminately applied to signal components with fundamentally different properties. This misalignment between a model's fixed inductive biases and the heterogeneous nature of the decomposed signals imposes a significant bottleneck on forecasting accuracy. To address this challenge, we propose a novel framework centered on the principle of Deep Heterogeneity, instantiated in our model, HeteroMixer . Our model first decomposes the series using a wavelet transform and then introduces a deeply heterogeneous architecture, featuring a powerful, dual-domain Trend-Seasonality Extractor (TSE) specifically for the vital low-frequency trend, alongside lightweight expert modules for high-frequency details. Furthermore, we replace static signal reconstruction with a novel Hybrid Prediction Framework, where an attention-based Multi-scale Fusion Transformer (MFT) learns to adaptively predict a corrective residual to a stable, classical baseline. Extensive experiments on seven challenging benchmarks demonstrate that HeteroMixer  establishes a new state-of-the-art, significantly outperforming existing methods. Our work validates the thesis that deeply aligning specialized, heterogeneous architectural biases with the intrinsic properties of signal components is a crucial strategy for advancing forecasting accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a limitation in current decomposition-based time series forecasting models, termed "Shallow Heterogeneity," where structurally homogeneous processing blocks are applied to decomposed signal components with different properties. To address this, the authors propose a "Deep Heterogeneity" framework instantiated in their model, HeteroMixer. The model uses a Discrete Wavelet Transform (DWT) for decomposition, processes low-frequency components with a powerful, dual-domain Trend-Seasonality Extractor (TSE), and handles high-frequency components with lightweight Stochastic Component Modelers (SCMs). A novel Hybrid Prediction Framework replaces static reconstruction: a classical inverse DWT provides a stable baseline forecast, while a Multi-scale Fusion Transformer (MFT) learns a corrective residual. Extensive experiments on seven benchmarks show state-of-the-art performance, which is further validated through ablation studies and model analysis.

### Strengths
1. This paper makes a valuable conceptual contribution by clearly identifying and formalizing "Shallow Heterogeneity" as a bottleneck in existing models. The proposed principle of "Deep Heterogeneity" is a compelling and well-motivated research direction.

2. This paper is well-written and readable with a clear structure.

### Weaknesses
1. While the framing is novel, the core technical components are incremental combinations of existing ideas. The use of wavelet decomposition, dedicated modules for different frequencies, and residual learning from a classical baseline are all established concepts. The TSE (FFT + Mixer) and MFT (Transformer) are powerful but not fundamentally novel architectures. 

2. This paper does not ablate the choice of DWT against other decomposition methods (e.g., MA, MODWT, or learnable decompositions). It is therefore difficult to disentangle how much of the performance gain comes from the "Deep Heterogeneity" principle versus the specific choice of using a multi-resolution wavelet analysis. The claim of a general paradigm is weakened by this tight coupling to a single decomposition technique.

3. While the paper cites very recent works from 2025 (e.g., T3Time, AMD, WaveTS-M), the empirical comparisons in Table 2 are limited to models from 2024 and earlier. A convincing SOTA claim requires direct comparison with the strongest and most recent contemporaries, which is not provided. This makes it difficult to assess the true current competitiveness of the proposed method.

### Questions
1. The "Deep Heterogeneity" principle is well-argued, but its instantiation relies heavily on existing building blocks (Wavelet, FFT, Mixer, Transformer). Could the authors clarify what they believe is the primary novel technical contribution of this work that distinguishes it from a well-engineered combination of established techniques?

2. How critical is the specific use of DWT? Have the authors experimented with other decomposition methods (e.g., MA, MODWT, or a learnable decomposition layer)? If the wavelet is replaced, does the "Deep Heterogeneity" framework still provide significant gains, or is the performance heavily dependent on the decomposition method itself?

3.  What is the computational overhead of HeteroMixer compared to baselines like PatchTST, iTransformer, or DLinear? Please provide data on training/inference time, FLOPs, and parameter counts. Could the performance gains be attributed primarily to increased model capacity?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper challenges decomposition-based LTSF for enforcing “Shallow Heterogeneity,” where heterogeneous signal components are processed with homogeneous blocks. It proposes a paradigm of Deep Heterogeneity instantiated by HeteroMixer: a wavelet-based disentanglement routes low-frequency trends to a dual-domain Trend-Seasonality Extractor (TSE) and high-frequency details to lightweight Stochastic Component Modelers (SCM). A Multi-scale Fusion Transformer (MFT) performs adaptive fusion, embedded in a hybrid prediction scheme that reuses inverse transform as a stability baseline while MFT learns a corrective residual ($y = y_{baseline} + y_{correction}$). Experiments on several time series datasets report new SOTA.

### Strengths
1. The idea of Shallow Heterogeneity—highlighting that existing decomposition-based models use homogeneous blocks to model fundamentally different components—is novel.
2. The Related Work section is comprehensive, and demonstrates a thorough understanding of prior research in time series forecasting.

### Weaknesses
1. The paper’s core claim—arguing that homogeneous blocks fail to model heterogeneous components—is not convincingly validated. There is no ablation where both decomposed components ($X_A$ and $X_D$) are modeled using only TSE or only SCM. If such homogeneous configurations outperform the proposed heterogeneous setup, the main argument would be undermined.
2. The motivation for Adaptive Fusion is unclear. Since wavelet and inverse wavelet transforms are lossless, the necessity of augmenting the inverse output is questionable; the combination feels somewhat ad hoc.
3. Experimental results appear inconsistent. Some reported metrics do not match the original papers’ numbers (such as PDF), and the input sequence length is unspecified, making replication difficult.
4. The writing quality is poor and suggests the paper was rushed:
    - Frequently misuse `\cite` instead of `\citep`, and fail to insert spaces before citations.
    - Table 2 mislabels bold (best) and underlined (second-best) results, which is a careless error.
    - The font size in Figures 1 and 2 is too small to read when printed.

### Questions
See **Weakness**.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes HeteroMixer, a novel time-series forecasting framework designed to overcome the “Shallow Heterogeneity” problem in signal decomposition. It employs a wavelet-based decomposition with specialized modules for trend and seasonal components, and an attention-based transformer for adaptive residual correction.

### Strengths
1. The paper identifies and analyzes the key limitations of existing time-series forecasting methods.

2. A novel combination of modules is proposed to address these limitations.

### Weaknesses
1. Compared with existing time-series forecasting models, the proposed method has a more complex structure; however, the paper lacks a comparison of computational efficiency.

2. The related work section points out the limitations of the FEDformer method, yet the experiments do not include a comparison with it. Additional experiments are needed to demonstrate the claimed advantages.

3. Although the paper emphasizes the importance of time-series decomposition, it does not compare the proposed approach with existing or recent decomposition-based forecasting methods.

4. The methodology section lacks a professional, symbolic formulation—there is only one equation in the entire paper. A systematic and formalized mathematical description should be provided to improve readability and academic rigor.

5. While the experimental section includes ablation and model analyses, the results are presented on only one dataset, which raises concerns about their reliability. More experimental results are needed for validation.

6. Compared with the baselines, the paper lacks visualization analyses, including overall forecasting visualizations and visualizations of ablation modules.

7. The paper lacks completeness in several aspects, such as problem definition, descriptions of baseline methods, experimental settings, and training function specifications.

### Questions
1. A comparison of runtime efficiency against the baselines on all datasets should be provided, including mean runtime and mean memory consumption.

2. Comparisons with FEDformer and other decomposition-based forecasting methods must be included.

3. Additional ablation studies and model analyses across more datasets are required.

4. The authors should present a concrete case study to demonstrate the method’s practical utility.

5. Since the method relies on specialized modules, the authors should discuss whether this design sacrifices generality, and provide an analysis of the method’s strengths and limitations.

### Soundness
2

### Presentation
2

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
The paper identifies "Shallow Heterogeneity" as a key limitation in current decomposition-based time series forecasting models, where structurally homogeneous blocks are applied to intrinsically different signal components (e.g., smooth trends vs. volatile details). To address this, the authors propose HeteroMixer, a model embodying "Deep Heterogeneity". The model uses a Discrete Wavelet Transform (DWT) for decomposition and routes components to specialized modules: a dual-domain Trend-Seasonality Extractor (TSE) for low-frequency components, and lightweight Stochastic Component Modelers (SCM) for high-frequency details. Furthermore, it introduces a Hybrid Prediction Framework that combines a stable baseline (via inverse DWT) with a corrective residual learned by a Multi-scale Fusion Transformer (MFT).

### Strengths
Novel Conceptual Framework: The formalization of "Shallow Heterogeneity" vs. "Deep Heterogeneity" is a compelling theoretical contribution that addresses a genuine bottleneck in existing decomposition architectures.

Principled Architecture: The design of HeteroMixer is well-motivated. Using a heavy, dual-domain (time+frequency) model for the trend (TSE) while using lightweight MLP-Mixers for stochastic details (SCM) logically aligns architectural inductive biases with signal properties.

Strong Empirical Results: The model achieves state-of-the-art performance across 7 standard benchmarks (including Electricity and Traffic), outperforming strong recent baselines like iTransformer and PatchTST.

Thorough Validation: The ablation studies are exceptionally sound. They not only validate the removal of components (Table 3) but also provide quantitative analyses verifying that the MFT indeed learns a residual highly correlated with the baseline's error and that the TSE/SCM learn fundamentally different representations (Table 6).

### Weaknesses
Architectural Complexity: The proposed framework is significantly more complex than many recent SOTA models (like DLinear or TimesNet). It involves DWT, FFT (within TSE), Mixers, and Transformers (MFT) all in one pipeline. This raises concerns about implementation difficulty and hyperparameter sensitivity.

Computational Overhead: While the authors acknowledge that the dual-stream hybrid framework increases computational complexity, there is no quantitative data (e.g., training time, inference latency, FLOPS) comparing HeteroMixer to simpler baselines to evaluate if the performance gains justify the added cost.

Reliance on Fixed Decomposition: The model currently relies on a fixed wavelet decomposition. While effective, a fully learnable decomposition might offer greater adaptability to diverse datasets.

### Questions
Can the authors provide a quantitative comparison of computational resources (training time/memory and inference latency) required for HeteroMixer versus key baselines like PatchTST or iTransformer?

How sensitive is the model's performance to the hyperparameters of the initial Wavelet Decomposition (e.g., choice of wavelet function, decomposition level)?

In the Hybrid Prediction Framework, the baseline is derived via iDWT. Did the authors experiment with using a simple learnable linear summation as the baseline instead of the fixed iDWT, to see if the MFT correction is still as effective?

### Soundness
3

### Presentation
3

### Contribution
3
