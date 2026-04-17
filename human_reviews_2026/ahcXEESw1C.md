# HEPHAESTUS: Hierarchical Periodic Heterogeneous Adaptive Spatio-Temporal Unified System for Traffic Forecasting

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Accurate traffic forecasting requires modeling complex spatio-temporal dynamics characterized by multi-scale temporal patterns, periodic dependencies, and spatial heterogeneity. While recent advances in spatio-temporal graph neural networks (STGNNs) have improved predictive performance, they often rely on fixed architectures that lack adaptivity to input-driven variations in temporal granularity and spatial connectivity. In this work, we propose \textbf{HEPHASTUS}, a novel framework that unifies adaptive multi-scale temporal modeling, explicit periodicity-aware attention, and dynamic spatial heterogeneity learning within a lightweight, scalable architecture. Our approach introduces three key components: (i) an Adaptive Multi-Scale Mixture of Experts (AMS-MoE) that deeply integrates multi-scale modeling with expert routing, using a dynamic router to automatically assign input sequences of different time scales to specialized experts, each expert focuses on temporal feature extraction at a specific scale (e.g., local fluctuations or long-term trends), with scale weights adaptively adjusted according to the time-varying input characteristics, enabling collaborative capture of global dependencies and local details; (ii) a Periodic Temporal Attention (PTA) mechanism that explicitly captures daily and weekly patterns via parameterized period matrices; and (iii) a Heterogeneous Spatial Attention (HSA) module that balances global structure and local specificity through node embeddings and a learnable pattern library, with low parametric cost. Experiments on six real-world traffic datasets, METR-LA, PEMS-BAY, PEMS03 and others, show that the proposed method achieves state-of-the-art performance across MAE, RMSE, and MAPE, with consistent gains over existing baselines. Ablation studies confirm the necessity of each design choice. Our results highlight the importance of adaptive, structured modeling in capturing the intrinsic dynamics of urban traffic.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Proposes HEPHAESTUS for traffic prediction with adaptive multi-scale patch selection, periodic temporal attention, and heterogeneous spatial attention, achieving SOTA by addressing multi-scale dynamics, periodicity, and spatial heterogeneity.

### Strengths
- Novel modules (adaptive multi-scale router, PTA, HSA) target traffic’s spatio-temporal challenges (multi-scale, periodicity, spatial heterogeneity) with tailored designs.
- Adaptive multi-scale mechanism dynamically adapts to traffic’s time-varying granularities, enhancing flexibility.
- Clear architectural innovation in integrating spatio-temporal attention with multi-scale adaptation.

### Weaknesses
- Lacks thorough efficiency analysis (e.g., inference time, parameter count) vs. lightweight baselines, critical for real-world deployment.
- Each patch_size triggers independent network processing, leading to potential high computational overhead without explicit efficiency justification.
- Dataset diversity is limited to traffic; generalizability to other spatio-temporal domains (e.g., meteorology) is unvalidated.

### Questions
- What’s the inference speed and computational cost when handling multiple patch sizes, and how does it compare to single-scale models?
- Can the adaptive patch selection be optimized to reduce redundant computations for real-time traffic scenarios?
- Is there empirical evidence that multi-patch_size design is indispensable, or would a subset suffice for similar performance?

### Soundness
3

### Presentation
3

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
The paper introduces a novel framework HEPHAESTUS for spatio-temporal traffic forecasting that integrates adaptive multi-scale temporal modeling, explicit periodicity-aware attention, and dynamic spatial heterogeneity learning. HEPHAESTUS addresses the limitations of traditional traffic forecasting models by introducing an Adaptive Multi-Scale Mixture of Experts for dynamic input-driven scale selection, a Periodic Temporal Attention mechanism to explicitly capture daily and weekly traffic patterns, and a Heterogeneous Spatial Attention module for effectively modeling both global and node-specific spatial dependencies. Extensive experiments on multiple real-world traffic datasets demonstrate that HEPHAESTUS outperforms existing baselines, achieving state-of-the-art performance in traffic forecasting tasks.

### Strengths
S1. HEPHAESTUS achieves state-of-the-art performance across multiple traffic forecasting benchmarks. This consistent performance across different datasets and tasks highlights the effectiveness and robustness of the proposed framework.

S2. The paper conducts detailed ablation studies, demonstrating the importance of each component and validating the necessity of adaptive, input-dependent modeling. These studies provide strong evidence for the design choices made in HEPHAESTUS and showcase how its key components work synergistically to improve performance.

### Weaknesses
W1. While the proposed HEPHAESTUS framework is innovative in its integration of periodic temporal attention, heterogeneous spatial attention, and multi-scale routing, the individual components are not entirely novel. These mechanisms have been explored in previous works, such as GMAN, STAEFormer, and PathFormer. The novelty of HEPHAESTUS primarily comes from combining these existing methods into a unified framework, but it doesn't introduce radically new techniques or concepts. Therefore, while the combination of these elements is effective, it may not be considered highly novel, especially for researchers already familiar with the mechanisms used in GMAN, STAEFormer, and PathFormer.

W2. The paper lacks comparisons with some recent relevant baselines like iTransformer [1], STWave [2], and PDFormer [3], which have shown strong performance in spatio-temporal forecasting tasks. By not including these baselines, the paper misses an opportunity to position its work more comprehensively within the current state-of-the-art in traffic forecasting.

W3. Although the paper demonstrates strong performance across various datasets, there is a notable lack of efficiency experiments. Moreover, the experiments conducted in the paper are primarily on small to medium-scale datasets, and there is no large-scale testing on datasets like LargeST [4].

[1] iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

[2] When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks

[3] PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction

[4] LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets three long-standing challenges in traffic forecasting: (i) multi-scale temporal patterns, (ii) strong daily/weekly periodicity, and (iii) spatial heterogeneity. The authors propose a unified architecture combining: (a) Moving-Patch tokenization, (b) a temporal router, (c) periodic time-aware attention with learnable day/week embeddings used as queries, and (d) heterogeneous spatial attention that fuses a shared linear value with a node-specific low-rank value synthesized from a pattern library. Experiments on standard traffic datasets report SOTA-level results and ablations indicate each module contributes to gains.

### Strengths
1.The paper is well-structured and self-contained: the methodological pipeline is clearly laid out, and the experimental section is thorough and comprehensive. 

2.The proposed pattern-library (PL) decomposition offers a principled balance between expressivity and efficiency.

3.Evaluates on common traffic benchmarks, which helps situate results in the existing literature.

### Weaknesses
1.The paper concatenates multi-scale Xpatch with the original H and linearly projects to Xh(Eq.5), but does not justify why concatenation is required. Please provide ablations comparing:(a)Xpatch only,(b)H only,and(c) concatenation + different projection choices, with accuracy and routing stability.

2.Why is the stride fixed to S=1? Does denser overlapping actually help under matched compute? Please report a grid over strides (e.g., 1/2/4) including both accuracy and compute/latency.

3.The abstract claims a lightweight method, but but lacks a comparison with baseline time complexity.

4.The motivation cites “spatial heterogeneity,” yet the text only mentions a “spatial indistinguishability problem” (around line 272) without an operational definition or diagnostics.

5.In the main diagram, the arrows/dataflow and the direction of the “Weight” signal are not self-evident. Please revise the figure with numbered steps and tensor shapes to make the routing → experts → aggregation path unambiguous.

6.The paper combines adaptive multiscale routing, periodic attention, and heterogeneous spatial modeling, but why this combination is necessary and how it improves over established designs (e.g., PatchTST, Pathformer) is not demonstrated under matched budgets.

7.Code is “to be released upon acceptance,” which limits verifiability.

8.Equation (19) appears to be missing a comma; please fix the typographical error.

### Questions
Please make revisions with reference to the weaknesses section.

### Soundness
3

### Presentation
2

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
This paper proposes HEPHAESTUS, a unified, lightweight spatio-temporal framework for traffic forecasting with adaptivity to input-driven variations in temporal granularity and spatial connectivity. HEPHASTUS is composed of three components: (i) AMS-MoE for input-adaptive multi-scale temporal modeling via routed experts and Top-K sparse gating; (ii) PTA for explicit daily/weekly periodic embeddings into temporal attention; (iii) HSA for modeling global structure and node-specific heterogeneity with a low-rank pattern dictionary. The authors conduct extensive experiments on six datasets and demonstrate state-of-the-art MAE/RMSE/MAPE, with ablation study, parameter sensitivity analysis, and case studies. In summary, HEPHASTUS addresses the important issue in traffic forecasting for adaptivity to input-driven variations in temporal granularity and spatial connectivity, elaborately combine the three components, and the experimental results show the effectiveness of HEPHASTUS. However, this paper could be improved by strengthening theoretical foundations, discussion and verification of computational complexity and scalability, criterion for selecting hyper-parameters, generalization to more complex spatio-temporal patterns, and further experiments for missing values and holidays effects.

### Strengths
- HEPHAESTUS is a well-ground design, which elaborately separate and integrate multiscale-time (AMS-MoE), periodicity (PTA), and heterogeneous space (HSA). 
- Although each component in HEPHAESTUS is already proposed one, the authors integrate them for addressing the issue of spatio-temporal modeling, adaptivity to input-driven variations in temporal granularity and spatial connectivity. 
- In AMS-MoE, Top-K gated experts equip HEPHAESTUS with input-dependent temporal granularity. 
- Low-rank node-specific transforms via a pattern dictionary reduce parameters while retaining locality.
- The extensive experimental results show superiority with ablation study, sensitivity analysis, and case studies.

### Weaknesses
- There is limited theoretical foundation. For example, online/adaptive behavior (e.g., Shai Shalev-Shwartz 2012) and scale-routing idetifiability (e.g., experts specialization in frequency bands) are not addressed. 
- Although the parameter sensitivity analysis is extensively conducted, there is no or limited discussion and experimental verification on computational complexity and scalability. For example, computational complexity and scalability of M (the number of experts), K (the number of selected experts), and r ( rank dimension) is not enough discussed for large N (the number of spatial nodes) and H (past time steps). 
- Although the parameter sensitivity analysis is conducted, there is limited criterion for selecting M, K, and r. 
- The imbalance of the experts in AMS-MoE and remedy for it, such as design of loss function, should be further discussed and demonstrated (c.f., Shazeer et al. 2017, Lepikhin et al. 2021, Fedus et al. 2021, Lewis et al. 2021, DeepSeek-AI 2024).
- PTA focuses on fixed daily/weekly patterns. It does not generalize to more complex spatio-temporal patterns, such as drifts and multi-periodicity beyond daily/weekly patterns. Some previous studies address this problem with learnable seasonal/trend components. 
- Experiments are limited. For example, there is no or limited experiments for missing values and holidays effects. 

[Shai Shalev-Shwartz 2012] Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends® in Machine Learning, 4(2), pp.107-194, 2012. 

[Shazeer et al. 2017] Noam Shazeer et al. Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. ICLR, 2017. 

[Lepikhin et al. 2021] Dmitry Lepikhin et al. GShard: scaling giant models with conditional computation. ICLR, 2021. 

[Fedus et al. 2021] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: scaling to trillion parameter models with simple and efficient sparsity. JMLR, 23, pp.1-29, 2021

[Lewis et al. 2021] Mike Lewis et al. BASE layers: simplifying training of large, sparse models. ICML. 2021. 

[DeepSeek-AI 2024] DeepSeek-AI. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437. 2024.

### Questions
Please answer the points listed in the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
