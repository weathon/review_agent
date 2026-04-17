# PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Long-term time series forecasting (LTSF) plays a crucial role in fields such as energy management, finance, and traffic prediction. Transformer-based models have adopted patch-based strategies to capture long-range dependencies, but accurately modeling shape similarities across patches and variables remains challenging due to scale differences. 
To address this, we introduce patch-mean decoupling (PMD), which separates the trend and residual shape information by subtracting the mean of each patch, preserving the original structure and ensuring that the attention mechanism captures true shape similarities. 
Futhermore, to more effectively model long-range dependencies and capture cross-variable relationships, we propose Trend Restoration Attention (TRA) and Proximal Variable Attention (PVA). The former module reintegrates the decoupled trend from PMD while calculating attention output. And the latter focuses cross-variable attention on the most relevant, recent time segments to avoid overfitting on outdated correlations. Combining these components, we propose PMDformer, a model designed to effectively capture shape similarity in long-term forecasting scenarios. Extensive experiments indicate that PMDformer outperforms existing state-of-the-art methods in stability and accuracy across multiple LTSF benchmarks. The code is available at https://github.com/aohu1105/PMDformer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes PMDFormer, a novel Transformer-based model for Long-term Time Series Forecasting (LTSF). The key insight is that the attention mechanism in existing patch-based models is often biased by the scale (amplitude) of patches, hindering its ability to capture true shape similarities. To address this, the authors introduce three core components: 1) Patch-Mean Decoupling (PMD), which centers patches by subtracting their mean to separate trend from shape; 2) Proximal Variable Attention (PVA), which focuses cross-variable modeling only on the most recent patch to avoid noisy historical correlations; and 3) Trend Restoration Attention (TRA), which reintegrates the global trend back into the attention mechanism. Extensive experiments on eight benchmarks show that PMDFormer achieves state-of-the-art performance, outperforming recent strong baselines. Ablation studies and theoretical analysis validate the design of each component.

### Strengths
1. Novel and well-motivated core idea. The identification of "scale bias" in patch-based attention is insightful. 

2. The model demonstrates compelling state-of-the-art performance, outperforming a wide range of recent and strong baselines across eight standard benchmarks.

3. The paper provides thorough ablation studies and a theoretical analysis that convincingly justify the contribution of each proposed module and the underlying motivation.

### Weaknesses
1. The paper positions PMDFormer as effectively modeling cross-variable dependencies, a domain where many previous "Variable-Dependent" (VD) models have struggled. However, the PVA module is applied only to the most recent patch. While the results are excellent, this design choice essentially limits cross-variable modeling to a very short, recent context. The paper could more explicitly discuss the implications of this: is the success of PMDFormer evidence that long-range cross-variable dependencies are generally not useful for LTSF, or that they are too noisy to model effectively? A comparison of PVA's performance when applied to the last $K$ patches (beyond just the ablation on $k$ in the sensitivity analysis) could have deepened this analysis.

2. While the efficiency of PVA is mentioned, a more formal and overall complexity analysis of the full PMDFormer architecture compared to other leading models (e.g., PatchTST, iTransformer, TimeBase) is missing. Given the use of a Transformer encoder in the TRA module and the separate PVA module, a discussion of the total parameter count and FLOPs would be beneficial for a complete picture.

3. The figure references "Figure X" in the Patch Size sensitivity analysis (Page 9), which appears to be a placeholder. This should be corrected to the appropriate figure number (likely 4b).

### Questions
1. The PVA module operates on the embedded tokens after PMD. Given that PMD centers the patches, does this mean PVA is exclusively modeling the shape similarities of variables at the most recent time segment, completely independent of their absolute levels? Could there be scenarios where the absolute values (or scales) of variables in the proximal patch are also critical for prediction?

2. The TRA module reintegrates the trend via a simple broadcast addition to the Value projection (Eq. 8). Was there an exploration of more complex fusion mechanisms (e.g., gating, concatenation followed by a linear layer)? Why was additive reintegration chosen as the most effective method?

3. For large multivariate datasets (e.g., Traffic with 862 variables), can the authors provide detailed training/inference time and memory usage comparisons between PMDformer and baselines (e.g., iTransformer, TimeBase)? How does PMDformer’s efficiency scale with the number of variables (C) or input sequence length (L), and is there potential to further optimize the TRA module’s computation (e.g., via parameter sharing)?

4. The paper finds moderate patch sizes (24–72) are optimal, but what guidelines would the authors recommend for selecting patch size based on dataset properties (e.g., sampling interval, length of input sequence, seasonality period)? For example, should a dataset with 10-minute sampling intervals (e.g., Weather) use a smaller patch size than one with 1-hour intervals (e.g., ECL)?

### Soundness
3

### Presentation
3

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
This paper proposes PMDformer, targeting the core issue that shape matching is often dominated by scale. It adopts Patch-Mean Decoupling (PMD), which removes only the mean of each patch, applies Proximal Variable Attention (PVA) to perform cross-variable attention on the nearest patch to the forecasting window, and uses Trend Restoration Attention (TRA) to inject trend information back into the Value branch without disturbing the Q/K shape alignment, thereby achieving a unified design that separates and then fuses “shape” and “trend.” Across eight datasets and multiple forecasting horizons, PMDformer shows stable improvements over strong baselines, and ablation studies validate the necessity of the three modules and their ordering.

### Strengths
S1. The problem is clearly defined, and each module is well-motivated by the design goals.

S2. The paper includes a formal derivation of the “mean-dominance” condition and multi-module ablations; the experimental design is fairly complete, and PMDformer performs well.

S3. The presentation is clear and the code structure is easy to follow.

### Weaknesses
W1. PMD is conceptually close to RevIN (reversible instance normalization that removes mean/variance per series) and to decomposition-based approaches such as DLinear and Autoformer (alleviating scale/distribution shift or decoupling trend and seasonality). The paper could strengthen the analysis to emphasize the differences.

W2. Restricting cross-variable modeling to only the nearest patch helps suppress noise and overfitting, but may lose interpretable dependencies when long-lag cross-variable effects exist.

W3. The similarity of patch-level trends depends heavily on the patch segmentation scheme, yet the paper does not analyze this sensitivity.

### Questions
Q1. After injecting trend into the Value branch, could residual pathways indirectly affect the attention output and thus interfere with shape alignment?

Q2. Beyond a fixed k=1, can a gated/learned scheduler adaptively determine the size of the proximal window or whether to include cross-variable attention over earlier patches?

Q3. How do you rule out—or control for—the confounding effects introduced by the patch segmentation choice?

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
4

### Summary
This work proposes a Transformer model named PMDformer for Long-Term Time Series Forecasting. The core idea is to decouple the trend and shape information of each time patch via Patch-Mean Decoupling (PMD), thereby preserving the original amplitude structure and avoiding the distortion of shape information caused by traditional normalization methods. Furthermore, the model introduces the Proximal Variable Attention (PVA) and Trend Restoration Attention (TRA) modules, which are designed to capture the most relevant short-term dependencies among variables and restore global trend information, respectively.

### Strengths
1. The ideas of this work are easy to understand. The description and presentation are clear.
2. This work conducts ablation experiments for the three core modules—PMD, PVA, and TRA—on multiple datasets, validating the effectiveness of each module.

### Weaknesses
1. The foundational premise of the PMD module is not fully convincing. According to Figure 1, the original patches (P1, P2), which have more similar means, receive a lower attention score than (P1, P3). This observation appears to contradict the authors' claim that "the scale differences initially obscure true shape similarity", as the patches with similar scales (P1, P2) are not assigned higher attention. This inconsistency raises questions about the necessity and motivation of the proposed decoupling.
2. In the TRA module, the operation of adding μ to V is not sufficiently motivated. Directly adding the raw, unprojected trend mean μ to the projected value V forcibly mixes vectors from disparate spaces—the original data space and the projected feature space.
3. The "double addition" of the trend in the architecture appears redundant. The trend term μ is added in the TRA module and again at the final projection layer. This design is not well-principled and risks over-emphasizing the trend component, potentially distorting the learned representations.

### Questions
See Weaknesses.

### Soundness
2

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
4

### Summary
The paper proposes PMDformer, a Transformer-based model for long-term time series forecasting that preserves cross-patch and cross-variable shape similarities despite scale differences. It introduces three components: Patch-Mean Decoupling (PMD) to separate trend from residual shape by subtracting each patch’s mean; Proximal Variable Attention (PVA) to emphasize recent, relevant cross-variable relationships and mitigate overfitting to outdated correlations; and Trend Restoration Attention (TRA) to reintegrate global trend into the attention mechanism without distorting shape. The authors claim PMDformer achieves superior stability and accuracy over state-of-the-art baselines across multiple LTSF benchmarks.

### Strengths
The paper is well written and easy to follow.

The proposed method demonstrates superior performance compared to several existing baselines.

### Weaknesses
The paper’s motivation is unclear: there is no well-defined objective or problem statement, and the connection between the proposed method and the problem it aims to solve is not clearly articulated.

There is little theoretical or empirical justification for the design of the proposed PMDFormer; the choice of each component appears arbitrary and based on empirical intuition, with no systematic evaluation.

Efficiency is not discussed: the paper lacks a complexity analysis (parameter count, runtime, and memory usage) relative to the baselines.

The LLM usage statement is not in the original manuscript.

### Questions
please see weakness

### Soundness
2

### Presentation
2

### Contribution
2
