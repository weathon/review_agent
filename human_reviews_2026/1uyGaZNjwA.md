# Graph-driven Autonomous Adaptation for Multi-stream Concept Drift

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Multi-stream concept drift introduces substantial challenges beyond traditional single-stream scenarios, as inter-stream dependencies produce complex, evolving dynamics that place increasingly stringent demands on real-world forecasting tasks. Existing adaptation methods typically address drift in isolation, overlooking spatio-temporal correlations—not only between streams but also among drift events—and failing to enable synchronized adaptation across large-scale data streams. Furthermore, current graph-based approaches often rely on static, pre-defined embeddings, limiting adaptability in highly dynamic environments. To address these limitations, we propose GAMAD, a graph-driven autonomous adaptation framework for multi-stream concept drift that integrates spatio-temporal graph construction with dynamic and predictive adaptation mechanisms for multi-stream forecasting. GAMAD dynamically constructs correlation graphs from historical distributional statistics and subgraph structures, eliminating dependence on pre-defined topology and enabling generalizable representations. For online multi-stream evolution, it performs noise-tolerant windowing for accurate node-level drift detection, and then expands from drift-centric nodes to localized subgraphs based on current multi-stream correlations. To further enhance forecasting generalization, we employ a hierarchical topological matching strategy to retrieve and reuse previously observed drift patterns, enabling more predictable adaptation to inherently unpredictable drifts. Extensive experiments on three large-scale real-world datasets demonstrate that GAMAD consistently outperforms state-of-the-art baselines in forecasting performance. We also show applicability to recommendation scenarios, where continuous adaptation to evolving user preferences is essential. We release code at: https://anonymous.4open.science/r/GAMAD-6AAB.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles the problem of forecasting in multi-stream environments where data distributions change over time (concept drift) and streams are correlated—think traffic sensors or weather stations that influence each other. The proposed GAMAD framework builds dynamic correlation graphs, detects drift at individual stream level, and reuses past drift patterns to adapt more effectively.

### Strengths
1. The paper addresses a real challenge in multi-stream environments where data distributions change over time (concept drift) and streams are correlated. 
2. GAMAD proposes a complete solution that: constructs dynamic correlation graphs without requiring pre-defined topology; detects drift at the node level rather than globally; reuses previously observed drift patterns to improve adaptation
3. Tests on real-world datasets show reasonable performance improvement.

### Weaknesses
1. The proposed GAMAD are quite complex and involes many components, making it diffuicult to understand which parts actually work, and there is limited ablation studies on this.
2. The drift detection method are actually quite simple, mean and variance change might not necessary indicating drift. Why not use more sophisticated statistical tests?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces GAMAD, a framework for adaptive learning in multi-stream environments where data distributions evolve over time. The key idea is to dynamically construct and update a correlation graph that captures relationships across streams, detect concept drift at the node level via adaptive error-thresholding, and react locally by expanding subgraphs around drifted nodes while reusing historical drift patterns for faster adaptation. By combining online graph updates, localized model adjustment, and memory of past drift behaviors, GAMAD aims to maintain accurate forecasting under irregular, asynchronous, and previously unseen drifts in complex real-world settings such as traffic and weather systems.

### Strengths
- Provided localized, adaptive drift handling by detecting node-level changes and focusing updates only where necessary, avoiding expensive global retraining.
- The proposed method learns and updates inter-stream correlations dynamically, removing reliance on fixed topologies and allowing adaptation to evolving relationships. 
- Reuse past drift patterns to accelerate recovery and handle recurring or similar drifts more effectively

### Weaknesses
- Relies on prediction error to signal drift, making detection reactive and susceptible to noise or delayed response.
- Performance depends on window and threshold settings, which may require tuning across environments.
- Maintaining per-node buffers and dynamic subgraphs may increase computational cost in large real-time systems.
- The method emphasizes dynamic inter-stream correlation changes, but the evaluation does not deeply analyze performance under systematically varied correlation patterns or correlation breakdown scenarios.
- The paper could better isolate contributions from node-level detection, dynamic subgraph expansion, and pattern reuse to show how each affects drift handling.

### Questions
See Weaknesses

### Soundness
3

### Presentation
4

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
The paper proposes GAMAD, a graph-driven autonomous adaptation framework for multi-stream concept drift. It dynamically builds a spatio-temporal correlation graph from historical statistics and subgraph structures, detects node-level drifts with adaptive thresholds, expands localized drift subgraphs in real time, and reuses matched historical drift patterns via hierarchical topological matching to improve forecasting under evolving dynamics. Experiments on three large-scale datasets show consistent gains over state-of-the-art baselines.

### Strengths
S1. The paper addresses the problem of adaptive concept drift in multi-stream settings, which is of practical importance and has clear real-world application demands. The research motivation is well grounded.

S2. The manuscript is clearly written, and the roles of different components are well aligned with the overall framework. The method section is generally readable and appears reproducible.

S3. The experiments are conducted on multiple real-world multi-stream datasets and include systematic comparisons with various existing methods. The results demonstrate consistent performance improvements across different scenarios.

### Weaknesses
W1. Our main concern regarding originality is that multi-stream concept drift adaptation has been widely studied, with prior work already exploring dynamic graphs and localized adaptation. The proposed framework largely integrates existing components rather than introducing a fundamentally new mechanism. The contribution is therefore more engineering-oriented than conceptual.

W2. The paper does not adequately demonstrate that the proposed components are individually necessary. There is no clear ablation across the graph-level, subgraph-level, and node-level representations to validate the claimed hierarchical modeling benefits. The paper employs a fusion-before-diffusion strategy, but does not compare it against alternative designs. Further, the framework lacks ablation on the hierarchical drift pattern matching module and the local subgraph expansion mechanism, both of which are highlighted as key contributions.

W3. The framework relies on multi-stage graph and subgraph sampling with adjacency-matrix-based operations, which may introduce substantial computational cost. The paper lacks complexity analysis and runtime comparisons, making scalability to large graphs unclear.

W4. The dynamic graph construction relies on historical distribution statistics and subgraph structure extraction. However, in scenarios where inter-stream correlations change abruptly at high frequency, the constructed dynamic graph may lag behind the real underlying dynamics, leading to delayed or suboptimal adaptive responses. 

W5. The node-level drift detection mechanism is based on temporal window statistics and adaptive thresholding, which implicitly assumes local distribution stability. In high-noise or adversarial disturbance settings, this approach may result in false positives or delayed detection. The paper currently lacks noise robustness analysis.

### Questions
Please see the above-mentioned weaknesses W1-W5.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses concept drift in multi-stream time-series environments, where multiple correlated data streams evolve over time. The authors propose GAMAD, a graph-driven adaptive framework that dynamically constructs hierarchical correlation graphs and detects local drifts at the node level. When drift occurs, the method identifies affected subgraphs and reuses similar historical drift patterns to enhance online adaptation. A diffusion-based graph recurrent model (DCGRU) is used for forecasting, and both the model and graph are updated through an online learning mechanism. Experiments on real-world traffic and weather datasets show that GAMAD significantly outperforms state-of-the-art baselines.

### Strengths
1. While most existing methods assume a single temporal stream, the proposed framework explicitly models multi-stream environments where concept drift can propagate through correlated nodes. This makes the method significantly more realistic for applications such as traffic and weather forecasting.

2. Existing spatio-temporal forecasting models typically rely on fixed adjacency matrices (e.g., distance-based or learned once offline). In contrast, the proposed Global-Subgraph-Node hierarchical graph construction dynamically updates correlations using Gumbel sampling and differentiable subset sampling, enabling the model to capture time-varying relationships that reflect evolving drift patterns.

3. The proposed approach detects drift and identifies recurring drift patterns by matching current subgraphs with historical ones and reusing them for sample augmentation, resulting in more predictable and generalizable adaptation.

4. The proposed method demonstrates consistent and substantial gains over strong state-of-the-art baselines on real-world datasets, highlighting the significance of the underlying problem setting and the considerable impact achieved by effectively addressing concept drift in multi-stream environments.

### Weaknesses
1. The proposed multi-stream correlation graph construction computes adjacency matrices at the global, subgraph, and node levels. However, it remains unclear which of these components primarily contributes to the performance gains. An ablation study isolating each adjacency matrix would significantly strengthen the empirical validation.

2. The graph-driven online adaptation module introduces several hyper parameters (e.g., β, h, K, q), yet the paper does not discuss the sensitivity of the method to these choices. It would be helpful if the authors could analyze the robustness of the method with respect to these hyper parameters and provide guidance on how to set them in practical scenarios.

3. The method requires online updates, which could impose additional computational overhead. However, the paper does not provide any analysis or discussion regarding the computational cost of the proposed approach. A runtime comparison or a complexity analysis would clarify whether the method is suitable for real-time or large-scale applications. Moreover, the scalability of the approach as the graph size increases should also be discussed.

4. The improvements over CGLM appear relatively small, making it difficult to assess the practical significance of the gains. The authors should include statistical significance tests or confidence intervals to demonstrate that the observed improvements are meaningful and not due to random variation.

### Questions
Please add explanations regarding the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
