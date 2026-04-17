# Adaptive Graph Convolutional Network with Attention Fusion for Multivariate Time Series Forecasting with Variable Missing

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Multivariate time series forecasting (MTSF) plays a vital role in diverse applications such as traffic prediction, weather research, and energy management. However, missing subset variable forecastinh has emerged as a critical challenge in MTSF due to factors such as sensor failures and maintenance. Variable incompleteness severely hinders the ability of forecasting models to capture intrinsic inter-variable relationships. Existing approaches either suffer from severe error accumulation, lack flexible mechanisms for handling missing data, or overly rely on local spatiotemporal correlations. To address these limitations, we propose VMPredictor, a novel end-to-end framework that effectively models spatiotemporal dependencies among incomplete variables for accurate forecasting. VMPredictor incorporates two key components: (1) the Adaptive Missing Filling and Enhancement Layer , which introduces learnable embeddings to adaptively fill missing positions and dynamically refine incomplete representations during training; and (2) the Spatiotemporal Dependency Mining Layer, built upon a Dynamic Graph Convolution Layer-Normalized Gated Recurrent Unit, where dynamic graph convolution adaptively reconstructs spatial correlations and replaces all fully connected layers in GRU to capture synchronized spatiotemporal dependencies. Together, these innovations endow VMPredictor with robust missing-data handling and precise spatiotemporal relation modeling. Extensive experiments on five real-world datasets under varying missing rates demonstrate the superiority of our approach over existing baselines. Codes can be found at https://anonymous.4open.science/r/Code-A216/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes VMPredictor, a novel end-to-end framework for multivariate time series forecasting (MTSF) under variable-missing scenarios. Unlike conventional two-stage imputation–forecasting methods, VMPredictor directly learns to model incomplete variables through adaptive representation learning. The framework integrates: (1) Adaptive Missing Filling and Enhancement Layer (AMFE Layer). (2) Embedding Layer: injects temporal and spatial embeddings. (3) Spatiotemporal Dependency Mining Layer (STDMLayer. (4) Multi-Head Temporal Self-Attention Layer (MHTSA): captures global temporal context for final prediction.

Comprehensive experiments on five real-world datasets (PEMS04/08, METR-LA, PEMS-BAY, China AQI) show that VMPredictor consistently outperforms 10+ SOTA baselines, especially at high missing rates (75%–90%).

### Strengths
1. The introduction of learnable missing embeddings allows dynamic representation of incomplete data, effectively reducing bias from fixed-value imputation. This design significantly improves robustness against missing patterns.
2. The combination of dynamic graph convolution and layer-normalized GRU enables simultaneous learning of temporal dynamics and spatial correlations. This achieves more precise inter-variable modeling compared to static GCN or vanilla GRU baselines.
3. The multi-head temporal self-attention layer captures long-range dependencies, while the SE-based enhancement layer adaptively reweights important channels and timestamps, leading to balanced local-global feature fusion.
4. The proposed method demonstrates consistent SOTA performance across diverse domains (traffic and air quality) and under severe missing conditions. Ablation and sensitivity analyses further confirm the effectiveness and interpretability of the design.

### Weaknesses
1. The introduction is somewhat disjointed. The authors could provide a deeper analysis of the core challenges in multivariate time series forecasting with variable missing data before introducing how the proposed method addresses these challenges.
2. The core technical innovation seems to lie mainly in the embedding layer, while the rest of the framework is similar to mainstream STGNNs. The authors are encouraged to further emphasize how the proposed embedding layer specifically mitigates existing problems, such as error accumulation in current methods.
3. The ablation experiment can consider directly removing all the additional embeddings. This will further demonstrate the effectiveness of the core technical contribution of this paper and its significance in this task.
4. It would be beneficial to conduct additional experiments in more widely used missing-data scenarios [1] to further verify the robustness of VMPredictor.
5. There are some formatting errors, such as the “Table ??” on page 13, line 684, and page 17, line 888.

[1] Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling

### Questions
1. I’m curious whether the proposed model can be transferred to random missing or block missing tasks.
2. Is the superior performance of the proposed method mainly attributed to the model architecture or to the specifically designed embedding layer?
3. If the proposed embedding method is transferred to other backbones, will it be able to improve their performance?

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
The paper proposes **FedTRL**, a federated learning framework designed to train **time series foundation models (TSFMs)** under **bi-level heterogeneity** — both inter-domain and intra-domain differences across clients.
It introduces a **dual-level optimization** strategy combining adversarial local regularization and domain-aware global aggregation to achieve domain-invariant and temporally coherent representations.
Extensive experiments across in-domain, full-shot, and zero-shot forecasting tasks show that FedTRL achieves **state-of-the-art performance**, outperforming both centralized and existing federated baselines.

### Strengths
The paper proposes **FedTRL**, a federated learning framework designed to train **time series foundation models (TSFMs)** under **bi-level heterogeneity** — both inter-domain and intra-domain differences across clients.
It introduces a **dual-level optimization** strategy combining adversarial local regularization and domain-aware global aggregation to achieve domain-invariant and temporally coherent representations.
Extensive experiments across in-domain, full-shot, and zero-shot forecasting tasks show that FedTRL achieves **state-of-the-art performance**, outperforming both centralized and existing federated baselines.

### Weaknesses
The paper proposes **FedTRL**, a federated learning framework designed to train **time series foundation models (TSFMs)** under **bi-level heterogeneity** — both inter-domain and intra-domain differences across clients.
It introduces a **dual-level optimization** strategy combining adversarial local regularization and domain-aware global aggregation to achieve domain-invariant and temporally coherent representations.
Extensive experiments across in-domain, full-shot, and zero-shot forecasting tasks show that FedTRL achieves **state-of-the-art performance**, outperforming both centralized and existing federated baselines.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper attempts to resolve the challenge of variable incompleteness in time series forecasting by proposing VMPredictor, a model that effectively captures spatiotemporal dependencies among incomplete variables to improve forecasting accuracy. Specifically, VMPredictor incorporates two key modules: the Adaptive Missing Filling and Enhancement Layer, which imputes and refines incomplete variables, and the Spatiotemporal Dependency Mining Layer, which captures both intra- and inter-series dependencies. Extensive experiments on five benchmark datasets demonstrate that the proposed model achieves state-of-the-art performance.

### Strengths
- This paper proposes a one-stage framework for time series forecasting with variable missing.
- It introduces a learnable embedding to alleviate learning bias caused by fixed fill-in values.
- Experimental results demonstrate that the proposed model achieves superior performance, even at high missing rates.

### Weaknesses
- This paper aims to address the problem of variable missing in time series forecasting. However, the proposed method lacks specific mechanisms or designs that explicitly target this issue, leading to a misalignment between the stated motivation and the model design. The author should clarify the rationale for introducing the squeeze-excitation module and the Dynamic Graph Convolution Layer-Normalized Gated Recurrent Unit, and explain how these components contribute to handling variable missing problems. Moreover, the proposed model primarily integrates existing modules, and its overall novelty appears limited. The author should more clearly highlight the unique contributions or theoretical insights beyond this integration.

-  Several arguments require clarification. For example, how does the learnable embedding $E_X \in \mathbb{R}^{T\times N\times d}$ alleviate parameter bias? How is $A_s$ defined for datasets that lack a predefined graph structure? In Eq. 14, the definitions of $\alpha$, $\beta$, and $\gamma$ are unclear. In line 271, the author states that 'where $W_v$, $W_n$, $b_v$, and $b_n$ are all trainable parameters', but these variables are not introduced or defined earlier in the paper. 

- The hyperparameter settings require clarification. Specifically, the values of $d$, $d_s$, and $K$ vary across different datasets. Could the authors explain the rationale behind these choices and provide guidance on how to select appropriate values for new datasets?

### Questions
- There are numerous grammatical and stylistic errors in this paper. For example, the forecasting window size is inconsistently denoted as $F$, $H$, and $T$ in lines 157, 161, 315, and 316.  In Eq. 4, subscript notations are used inconsistently (1:T and 1:t). In addition, there are several typographical and capitalization errors, such as forecastinh (line 16), ': ,' (line 353), ', ,' (line 761), and 'Table ??' (lines 685 and 889). These issues significantly affect the readability and professionalism of the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose VMPredictor, an end-to-end framework that addresses key challenges in missing subset variable forecasting, including severe error accumulation, the lack of flexible mechanisms for handling missing data, and the overreliance on local spatiotemporal correlations in existing methods.

### Strengths
S1. The adaptive missing-data imputation and enhancement layer introduces learnable embeddings to adaptively fill missing positions and dynamically refine incomplete representations during training.

S2. The spatiotemporal dependency mining layer is built upon a dynamic graph convolutional gated recurrent unit, where dynamic graph convolution adaptively reconstructs spatial correlations and replaces all fully connected layers in the GRU to capture synchronized spatiotemporal dependencies.

### Weaknesses
W1. The definition and role of $E_p$ in Equation (11) are unclear. The author should clarify its purpose, explain how it relates to $E_a$, and elaborate on their connections with $E_{day}$ and $E_{week}$. 

W2. There are several typographical errors, such as “Table ??” in Line 889 and “forecastinh” in Line 015.

W3. The author randomly masks a fixed proportion of data; therefore, results from multiple random seeds should be reported to demonstrate the stability of the proposed method. Furthermore, sufficient implementation details should be provided to facilitate code reproducibility.

W4. The author does not specify the source of the “China AQI” dataset. While it appears to refer to air quality index data, which is dynamically computed based on multiple pollutant concentrations, the author should clarify which specific pollutant was used to ensure consistency with the characteristics of other datasets.

### Questions
See W1-4.

### Soundness
3

### Presentation
2

### Contribution
3
