# KANomaly: Fourier-KAN-based Multi-Scale Patch Mixer for Multivariate Time Series Anomaly Detection

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Multivariate Time Series Anomaly Detection (MTSAD) is crucial for system stability in domains such as industrial monitoring, yet rare events, nonlinear dependencies, and limited labels necessitate unsupervised methods. However, existing approaches struggle to model subtle anomalies and detect diverse patterns, as they rarely integrate explicit frequency-domain representations and rely on fixed-scale analysis. To address these limitations, we propose KANomaly, a novel model inspired by Fourier-based Kolmogorov–Arnold Networks (KANs) for MTSAD. The model incorporates three key innovations: (i) Fourier basis functions embedded within the KAN architecture to capture subtle periodic and spectral anomalies; (ii) a coarse-to-fine multi-scale patching strategy that enhances detection of both point and pattern anomalies; and (iii) a Fourier-KAN Mixer that aggregates information across channel, patch, and temporal dimensions to model complex local and global interdependencies. Extensive experiments demonstrate that KANomaly consistently outperforms state-of-the-art models on multiple real-world datasets, validating the effectiveness of each component.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
To address the limitations of existing MTSAD methods, such as their implicit learning of temporal and channel dependencies, insufficient utilization of frequency domain information, and challenges in capturing a wide range of short- and long-term anomalies, the authors propose Kanomaly. Kanomaly replaces the spline network in KAN with Fourier series and incorporates multi-scale patching and blending techniques, achieving state-of-the-art detection performance.

### Strengths
1. Clear and well-designed figures that enable readers to quickly grasp the motivation and methodological details.  
2. Comprehensive equations that clearly convey the computational aspects of the model.

### Weaknesses
1. All key modules have been previously explored, such as Fourier Transform in KAN (KAN-AD) [1], Patching (PatchTST) [2], and Patch Mix (TimeMixer) [3], but they lack substantial innovation.
2. The data in the visual analysis is synthesized by TODS, not from the real data set, and the data synthesized by TODS is a single indicator, which does not match the MTS problem that the method in this article wants to solve. In addition, the basic shape of the synthetic data is a sine function, which is too easy for contemporary models. (Detailed in Questions) 
3. There is a misunderstanding of some related work, and it is ignored as a baseline method. (Detailed in Questions)

### Questions
1. The authors claim in section 2.2 that Kanomaly is the first KAN-based method for MTSAD. However, several prior attempts have used KAN to address MTSAD [4] [5]. Additionally, even the KAN-AD [1] method referenced by the authors includes MTSAD benchmarks. These approaches should be considered as part of the baseline.

2. In Section 5.2, a visualization analysis is presented (detailed in C.1). However, this visualization is not based on actual datasets (such as SMD), but rather uses synthetic data. The training set of TODS consists of regular sine waves (as indicated in line 8 of Listing 1), which are too simple for deep models. This simplicity undermines the argument that Kanomaly can address a wide range of anomalies. Can Kanomaly handle a variety of anomalies on real datasets?

3. It is worth noting that the dataset synthesized by TODS is based on UTS, which indicates that Kanomaly is capable of handling univariate tasks. From this perspective, what distinguishes MTS from UTS in terms of their unique characteristics? Furthermore, how does Kanomaly perform when applied to univariate datasets?



[1] https://dl.acm.org/doi/10.1145/3746709.3746752

[2] https://iclr.cc/virtual/2023/poster/10876

[3] https://iclr.cc/virtual/2024/poster/19347 

[4] https://link.springer.com/article/10.1007/s10489-025-06650-8

[5] https://icml.cc/virtual/2025/poster/45584

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes KANomaly, an unsupervised approach for multivariate time series anomaly detection. The method is built on a Kolmogorov–Arnold Network (KAN) backbone in which learnable Fourier basis functions are incorporated to capture frequency-domain structure. The model further employs a coarse-to-fine multi-scale patching strategy to handle anomalies at both point and pattern levels, and uses a mixer-style architecture to integrate information across temporal, channel, and patch dimensions. The authors evaluate the approach on five commonly used MTSAD benchmarks and report improvements over prior methods, along with ablation studies and qualitative analyses to support the contribution.

### Strengths
## Strengths
- Architectural Innovation：Rather than simply applying KAN to multivariate time series anomaly detection, the paper tailors the architecture to characteristics of different anomaly types and temporal structures. The use of multi-scale patching and mixer-based aggregation seems well-motivated.
- Clarity of Presentation and Visualization: The manuscript is clearly written and easy to read. The figures are detailed and informative, effectively supporting readers’ understanding of the architectural components.

### Weaknesses
## Weaknesses
- Limited Baseline Comparison: The related work section discusses KAN-AD, which also applies KAN to anomaly detection and demonstrates performance on multivariate settings (as reported in Table 6 and Section 4.7 of the KAN-AD paper). Including KAN-AD under its MTSAD configuration in the main experiments would strengthen the empirical claims, especially regarding the contributions of the multi-scale patching strategy and the mixer-style architecture. In addition, the experimental benchmark selection omits some widely used industrial datasets such as WADI, which would provide a more comprehensive evaluation in higher-dimensional real-world scenarios.
- Insufficient Analysis of Results: The performance gains vary considerably across datasets in Table 1. For instance, the improvements on SMD are notably smaller compared to other datasets, yet the paper does not discuss possible reasons. This variability could highlight limitations or conditions under which the method is more or less effective, and a deeper discussion would be valuable.
- Incomplete Ablation Studies: Since the coarse-to-fine multi-scale patching mechanism is a core contribution, additional ablations focused on patch size would be beneficial. For example, evaluating smaller patch sizes (potentially better for point anomalies) and larger ones (potentially better for pattern anomalies) would provide clearer evidence for the design choices.
- Interpretability Claims Remain Underdeveloped: The method leverages KANs’ basis-function parameterization, suggesting potential interpretability benefits. This paper does not provide qualitative or visual analysis to show what insights the learned basis functions reveal about anomaly patterns across channels.
- Lack of Clear Computational Cost and Parameter Analysis: Although efficiency comparisons appear in Figure 3, there is no direct computational cost / parameter-size metrics reported in the ablation models. This makes it difficult to assess the efficiency-performance trade-off of the proposed architecture.

### Questions
## Questions
There are relatively few KAN-based approaches for anomaly detection, and I find the architectural extensions in this work meaningful. I would appreciate more comprehensive experimental support on following two aspects to further strengthen the contribution and clarify the model’s advantages.
- Since the core contribution builds on the KAN architecture by introducing the multi-scale patching strategy and mixer-style design, could the authors provide experimental comparison with KAN-AD under the multivariate anomaly detection setting?  Such a comparison would help clarify how much of the observed performance gains stem from the proposed architectural modifications, beyond the benefits inherent to the KAN backbone itself.
- Given that KAN models offer basis-function parameterization and channel-wise functional decomposition, could the authors provide either qualitative interpretability visualizations and/or quantitative analysis of computational cost (e.g., parameter counts, FLOPs, inference latency)? This would both deepen understanding of the advantages of KAN-based models in anomaly detection and help assess the practicality of the proposed design in real-world deployment.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes KANomaly, a novel unsupervised multivariate time series anomaly detection model that integrates Fourier basis functions into Kolmogorov-Arnold Networks (KANs) to capture frequency-domain anomalies. The model also introduces a coarse-to-fine multi-scale patching strategy and a Fourier-KAN Mixer to capture dependencies across channel, patch, and temporal dimensions. Extensive experiments on five real-world datasets demonstrate that KANomaly outperforms 17 state-of-the-art baselines, with ablation studies confirming the contribution of each component.

### Strengths
Combines Fourier analysis with KAN in a novel way for Multivariate Time Series Anomaly Detection, a direction not previously explored. The approach elegantly and rationally combines the intuitive physical interpretation of Fourier analysis—namely, its ability to capture periodicity and spectral anomalies—with the universal function-approximation power of Kolmogorov–Arnold Networks.

Table 2’s ablation systematically swaps Fourier-KAN for MLP/Chebyshev/Vanilla-KAN and reverses the coarse-to-fine order, convincingly proving the current design optimal.

### Weaknesses
The paper’s key performance comparisons rely heavily on precision, recall, and F1 scores obtained after point-adjustment, a protocol that “forgives” detection delays and fragmented predictions: as long as at least one point inside an anomalous event window is flagged, the entire event is counted as correctly detected.

The comparison with cutting-edge work is insufficient. A recent method, CATCH, is highly relevant in core idea: it also performs fine-grained, patch/band-level processing of frequency-domain information and highlights inter-channel spectral differences, overlapping significantly with KANomaly. Moreover, KANomaly does not demonstrate superior performance over CATCH.

### Questions
The paper relies primarily on the F1 score computed under the point-adjustment strategy—a metric widely criticized for drastically inflating real-world performance. Why do the authors choose this as their central evidence?

The learnable sine/cosine bases in Fourier-KAN, the multi-scale patching strategy, and the tri-dimensional mixer undoubtedly introduce substantial computational overhead; yet the paper only offers a single-dataset, coarse comparison of training time and parameter count in Figure 3 (right).

Is there any theoretical justification for choosing Fourier bases over other basis functions in the context of anomaly detection?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes KANomaly, an unsupervised model for multivariate time series anomaly detection that integrates KANs with a coarse-to-fine multi-scale patching strategy and a Fourier-KAN Mixer module. The core idea is to explicitly model frequency-domain characteristics—such as periodicity shifts and spectral disturbances—through learnable Fourier basis functions within the KAN framework, which is theoretically grounded in the Kolmogorov–Arnold representation theorem. The multi-scale patching enables simultaneous detection of both short-term point anomalies and long-term pattern anomalies, while the mixer module captures complex interdependencies across channel, patch, and temporal dimensions. The authors evaluate KANomaly on five real-world benchmarks, reporting improvements over current methods.

### Strengths
•	The integration of Fourier basis functions into KAN for MTSAD is a meaningful extension of recent KAN-based approaches, particularly as most prior KAN applications focus on forecasting rather than anomaly detection.  
•	Comprehensive empirical validation: The paper includes extensive experiments across five standard datasets, ablation studies, efficiency analysis, and evaluation under multiple metric paradigms (point-adjusted, range-based, and affiliation-based), which strengthens the credibility of the claims.  
•	Clear methodological pipeline: The coarse-to-fine multi-scale patching strategy combined with dimension-wise mixing (channel → patch → temporal) is well-motivated and implemented with attention to reconstruction fidelity and anomaly localization.

### Weaknesses
- The paper asserts that “existing approaches struggle to model subtle anomalies and detect diverse patterns” because they “rarely integrate explicit frequency-domain representations,” but this claim is not rigorously supported. In fact, several recent works explicitly incorporate frequency-domain modeling for time series tasks, including anomaly detection. The authors acknowledge some of these in Related Work but fail to clearly differentiate KANomaly’s functional integration of frequency information from these predecessors. The motivation appears overstated.

- The criticism that current models “struggle to detect subtle anomalies” remains qualitative and lacks concrete examples or failure cases from existing SOTA models. For instance, the paper could have included a comparative case study (e.g., on a synthetic or real anomaly where spectral features are decisive) showing that Transformer- or reconstruction-based baselines miss anomalies that KANomaly captures—thereby grounding the claimed advantage in observable behavior rather than architectural assumptions.

- While the use of Fourier-KAN is central, the ablation study shows that replacing Fourier-KAN with an MLP causes performance drops—but it does not rule out whether a well-designed frequency-aware MLP (e.g., inspired by FITS or FreTS) could achieve similar gains. The contribution would be stronger if the authors clarified whether the performance gain stems from the KAN structure itself (learnable univariate functions per edge) or simply from explicit Fourier feature learning, which could potentially be implemented in other frameworks.

### Questions
1. Could the authors clarify what constitutes “explicit functional integration” in their framework, and how KANomaly’s use of Fourier basis functions within KAN provides a qualitatively different capability compared to these prior frequency-aware models?  

2. Could the authors provide further analysis or ablation (e.g., Fourier features + MLP) to isolate the contribution of the KAN structure versus the frequency representation?

### Soundness
2

### Presentation
2

### Contribution
2
