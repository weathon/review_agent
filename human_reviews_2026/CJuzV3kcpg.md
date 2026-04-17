# PQ-Net: Periodic Quantum Networks for Multivariate Time Series Forecasting

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Multivariate time series forecasting (MTSF) requires capturing both periodic structures and cross-channel dependencies from complex temporal signals. To address this challenge, we propose Periodic Quantum Networks (PQ-Net), a quantum--classical hybrid forecasting architecture that integrates a learnable temporal query mechanism for cycle alignment and a channel aggregation module for modeling inter-channel correlations. PQ-Net preserves permutation equivariance across variables while jointly representing frequency-domain and cross-channel information in a principled manner. At the core of PQ-Net lies the Data Re-uploading Quantum Circuit (DRQC), whose representational capacity we theoretically analyze. We show that DRQC are mathematically equivalent to truncated Fourier series, enabling natural encoding of periodic patterns, while quantum entanglement provides a means to capture inter-variable dependencies. This interpretation establishes DRQC as a rigorous and interpretable foundation for periodic modeling within PQ-Net. Extensive experiments on twelve real-world datasets demonstrate that PQ-Net consistently achieves state-of-the-art forecasting accuracy over strong classical and quantum baselines, and preliminary hardware results further validate its practicality on real quantum devices.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a multivariate time series forecasting algorithm that can perform interactive modeling based on both time and channel dimensions.

### Strengths
PQ-Net effectively integrates periodic structure modeling and multivariate dependency learning through a theoretically grounded quantum-Fourier formulation, providing interpretable and compact periodic representations.

### Weaknesses
1. The motivation mainly targets the periodicity of multivariate time series, but what about other non-periodic characteristics such as trends, non-stationary shifts, or transient dependencies?

2. Quantum entanglement essentially acts as a **channel-mixing matrix**, functionally similar to an MLP-Mixer or Channel-Mixer layer, lacking true novelty in channel-level feature extraction.

3. The experiments do not clearly demonstrate whether quantum entanglement yields superior cross-channel representation compared with conventional channel-mixing mechanisms.

### Questions
See Weaknesses.

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
3

### Summary
To tackle the challenge of capturing periodic structures and cross-variable dependencies in multivariate time-series forecasting, this work proposes PQ-Net, a quantum framework built on Data Reuploading Quantum Circuits that encode periodicity and model variable interactions via quantum entanglement. Theoretical analysis and experiments seem to have verified the effectiveness of its performance。

### Strengths
1. This work is an interdisciplinary approach with unique insights.

2. The experiment was relatively thorough.

### Weaknesses
1. In the multivariate time series prediction, I have doubts about the conclusion of "their heavy reliance on self-attention makes them vulnerable to noise".  As one of the research motivations of this paper, this view needs to be analyzed in detail. Can the author verify this insight through experiments?

2. I believe this work is theoretical , and the author's insights are very unique, bringing about interdisciplinary thinking. However, this manuscript demonstrates a fusion of technical reports and formula derivations in its presentation, failing to bring new insights to the this filed.

3. The improvement in performance is gradual, so it is hard to admit that the author's interdisciplinary insights can bring about substantial enhancements.

### Questions
1. The model architecture of this work is composed of a series of modules, i.e., IN, LPV, DRQC. I want to know if there is a logical coupling relationship among them. In the current manuscript, they appear to be in a state of separation.

2. Where can the ability of IN to alleviate distribution shift mentioned in Fig 1 be reflected?

### Soundness
2

### Presentation
2

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
This paper introduces PQ-Net, a periodic quantum network integrating explicit periodic-structure modeling with expressive cross-variable dependency learning for multivariate time series forecasting.

### Strengths
1. Theoretical analysis in this paper demonstrates that the architecture of Data-Reuploading Quantum Circuits (DRQC) can be rigorously expressed as a truncated Fourier series.  
2. This paper utilizes learnable periodic vectors to provide phase-aligned periodic priors and employs stackable DRQC blocks to capture spectral structure while modeling inter-variable entanglement.  
3. Extensive experiments on 12 real-world multivariate datasets demonstrate the superior performance of PQ-Net.

### Weaknesses
1. Many important claims in the Introduction lack the necessary, strong citations or experimental evidence. For example, the statements “They lack a unified mechanism that can simultaneously represent periodic structure while capturing rich cross-variable interactions” and “yet their heavy reliance on self-attention makes them vulnerable to noise” are presented without adequate support.  
2. The paper’s motivation is unclear. The authors argue that existing approaches lack a unified mechanism that can simultaneously represent periodic structure while capturing rich cross-variable interactions, yet they cite only CycleNet. In fact, there is a substantial body of work on time–frequency–based periodic modeling (e.g., Peri-MidFormer [1], DEPTS [2]), and numerous methods already address cross-variable interactions (e.g., FourierGNN [3]). Consequently, the necessity and novelty of “simultaneously” modeling periodic structure and cross-variable interactions are insufficiently justified. Additionally, the advantages of introducing Quantum Networks are not clearly articulated.  
3. It is recommended to include a relevant time series pattern modeling baseline such as TimeMixer++ [4] to enable a more comprehensive evaluation.

[1] Peri-midFormer: Periodic Pyramid Transformer for Time Series Analysis. NeurIPS, 2024.  
[2] DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting. ICLR, 2022.  
[3] FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective. NeurIPS, 2023.  
[4] TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis. ICLR, 2025.

### Questions
pls refer to weakness

### Soundness
2

### Presentation
2

### Contribution
2
