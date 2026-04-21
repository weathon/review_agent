# GraphSTAGE: Channel-Preserving Graph Neural Networks for Time Series Forecasting

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
Recent advancements in multivariate time series forecasting (MTSF) have increasingly focused on the core challenge of learning dependencies within sequences, specifically intra-series (temporal), inter-series (spatial), and cross-series dependencies. While extracting multiple types of dependencies can theoretically enhance the richness of learned correlations, it also increases computational complexity and may introduce additional noise. The trade-off between the variety of dependencies extracted and the potential interference has not yet been fully explored. To address this challenge, we propose GraphSTAGE, a purely graph neural network (GNN)-based model that decouples the learning of intra-series and inter-series dependencies. GraphSTAGE features a minimal architecture with a specially designed embedding and patching layer, along with the STAGE (Spatial-Temporal Aggregation Graph Encoder) blocks. Unlike channel-mixing approaches, GraphSTAGE is a channel-preserving method that maintains the shape of the input data throughout training, thereby avoiding the interference and noise typically caused by channel blending. Extensive experiments conducted on 13 real-world datasets demonstrate that our model achieves performance comparable to or surpassing state-of-the-art methods. Moreover, comparative experiments between our channel-preserving framework and channel-mixing designs show that excessive dependency extraction and channel blending can introduce noise and interference. As a purely GNN-based model, GraphSTAGE generates learnable graphs in both temporal and spatial dimensions, enabling the visualization of data periodicity and node correlations to enhance model interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents GraphSTAGE, a novel GNN-based model for multivariate time series forecasting (MTSF). By separating the learning of temporal (intra-series) and spatial (inter-series) dependencies while preserving the original channel structures, GraphSTAGE minimizes noise and computational overhead caused by channel mixing. It incorporates the Spatial-Temporal Aggregation Graph Encoder (STAGE), which enhances interpretability by visualizing temporal and spatial patterns. Experiments across 13 real-world datasets demonstrate that GraphSTAGE outperforms state-of-the-art models by up to 20.8%.

### Strengths
Overall, good empirical result: the proposed model outperforms state-of-the-art models by up to 20.8%. The experiments in this paper focus on evaluating the proposed GRAPHSTAGE model for multivariate time series forecasting (MTSF). Here’s a summary of the experiments and results:

Experimental Setup
* Datasets: The model is tested on 13 real-world datasets, including ETT, ECL, Exchange, Traffic, Weather, Solar-Energy, and PEMS (with subsets PEMS03, PEMS04, PEMS07, and PEMS08). These datasets represent diverse MTSF tasks, covering domains like energy, weather, and traffic forecasting. See Section 4.1 in Page 6.

* Baselines: Seven well-known models, including iTransformer, PatchTST, Crossformer, DLinear, RLinear, SCINet, and TimesNet, are used for comparison​. See Section 4.1 in Page 6.

Main Results
* Overall Performance: GRAPHSTAGE demonstrates up to a 20.8% improvement in performance and achieves first place in 22 out of 30 comparisons across all datasets. It achieves lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) compared to state-of-the-art (SOTA) models, particularly excelling in datasets like ECL, ETT, Weather, Solar-Energy, and PEMS​. See Section 4.2 in Pages 6 and 7.

* Model Efficiency: Compared to models like Crossformer, GRAPHSTAGE reduces memory usage by 47.0% and training time by 60.9%, while achieving a 36.5% improvement in predictive accuracy. See Page 8, Lines from 378 to 385.

Ablation Studies
* Correlation Learning Mechanism: Ablations reveal that removing or replacing the Inter-GrAG and Intra-GrAG modules degrades performance, confirming the importance of the decoupled spatial-temporal extraction in GRAPHSTAGE​. See Section 4.3 and Table 2 in Page 8.

* Embedding & Patching Mechanism: Ablations also show that removing patching or adaptive embedding reduces accuracy, indicating the critical role of these components in effective forecasting. See Section 4.3 and Table 3 in Page 8.

### Weaknesses
Incremental idea & Lack of novelty: I think the core idea of decoupling the learning of intra-series and inter-series instead of channel mixing is incremental and not significant.

* By decoupling intra-series and inter-series dependencies, GRAPHSTAGE enables more efficient modeling without the interference and noise associated with channel blending. See Section 3.2, Lines from 228 to 242 in Page 5.

* GRAPHSTAGE’s decoupled architecture reduces computational overhead and enhances model interpretability by generating separate learnable graphs for temporal and spatial dimensions​. See Section 3.2, Lines from 243 to 266 in Page 5.

### Questions
Please address the concern regarding novelty that I have mentioned in weaknesses!

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This study focuses on graph structure learning for time series forecasting based on modeling inter-series and intra-series dependencies for complex multivariate time series. To enable efficient modeling of dependencies with controlled interference between variables and reduced noise, the authors propose GraphStage architecture, a GNN-based approach that, contrary to previous approaches, is channel-preserving. To achieve this, the GraphStage model is built upon patching and embedding layers, followed by decoupled inter-series and intra-series correlation graph structure modules that produce spatial and temporal graphs fine-tuned with the task. Finally, GraphStage is evaluated for time series forecasting against recent state-of-the-art architectures, showing competitive performance. Several ablation studies and visualizations of the learnable correlation provide additional insights into the potential of the proposed method.

### Strengths
The authors tackle a long-standing problem in time series forecasting, which is inferring a graph capturing evolving dependencies in terms of multiple variables. The main strengths of this work and the proposed method are summarized as follows:
- The authors combine embedding and patching mechanisms recently proposed for time series forecasting with the challenging task of graph structure learning, enabling channel independence.
- Correlations are captured in terms of a graph structure separately for the spatial and temporal dimensions, which remains a relatively novel design choice.
- Experimental results show that the proposed graph-based method outcompetes, in several cases, recent baselines in time series forecasting. Additionally, the visualizations of the learned temporal correlations are very interesting qualitative aspects of the presented study.

### Weaknesses
1. **Poor Experimental Evaluation and Unclear Performance Significance:** (W1) Chosen baseline methods for comparisons are limited to non-graph-based models. In contrast, several methods leveraging GNNs for time series forecasting can be found in the literature [1]. (W2) Additionally, the performance improvements achieved by the proposed method are, in most cases, very small compared to the best competitor (for instance, the biggest difference achieved by GraphStage from its best competitor in Table 1 is for Solar-Energy: from MSE equal to 0.233 to MSE 0.192, yet GraphStage is outcompeted in terms of MAE). (W3) It is unclear if the authors have performed multiple runs with random seeds to capture the model’s performance variability. The lack of standard deviations makes it impossible to assess whether the results' difference is statistically significant.
2. **Limited Related Works Presented:** (W4) Graph structure learning for time series forecasting is a long-term studied problem in the literature, entirely before the pure graph paradigm (Yi et al., 2024). Several methods that capture dependencies in the spatial and temporal domain jointly (Shang et al., 2021; Wu et al., 2020) or separately (Kipf et al., 2018; Xu et al., 2023) have been proposed, but are rather not mentioned in the paper and not considered in the experiments. (W5) Similarly the graph learning module (softmax on dot products between pairs) is a standard choice in relevant works (Wu et al., 2020), yet not correctly cited in this work.
3. **Lack of Clarity in the Methodological Presentation and Inflated Claims:** (W6) Section 3 is not very easy to follow since it has a great amount of text, making it hard to understand how the two graph learning blocks interact and how the final output is derived and optimized. (W7) Some claims are slightly exaggerated, e.g., that the proposed components “effectively capture intra-series and inter-series dependencies, respectively, while generating interpretable correlation graphs”. Additionally, performance improvements are, in the average case, around 6% (and no improvement for four datasets), thus the statement “with improvements up to 20.8%” is slightly misleading.

[1] Yi, K., Zhang, Q., Fan, W., He, H., Hu, L., Wang, P., ... & Niu, Z. (2024). FourierGNN: Rethinking multivariate time series forecasting from a pure graph perspective. Advances in Neural Information Processing Systems, 36.

[2] Shang, C., Chen, J., & Bi, J. (2021). Discrete graph structure learning for forecasting multiple time series. arXiv preprint arXiv:2101.06861.

[3] Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). Neural relational inference for interacting systems. In International conference on machine learning (pp. 2688-2697). PMLR.

[4] Xu, N., Kosma, C., & Vazirgiannis, M. (2023, November). TimeGNN: Temporal Dynamic Graph Learning for Time Series Forecasting. In International Conference on Complex Networks and Their Applications (pp. 87-99). Cham: Springer Nature Switzerland.

[5] Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020, August). Connecting the dots: Multivariate time series forecasting with graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 753-763).

### Questions
1. Based on (W1-W3), the experimental evaluation could be enhanced with comparisons with relevant graph-based modules and better capture the variability of scores in comparisons with the baselines.
2. Based on (W4, W5), the related works section could be improved and include differences between design choices in the graph structure learning component, including the incorporation of sparsity, binary or continuous edges, and spatial or temporal dependencies, such that the technical contribution of the work is positioned in a more evident and fair way.
3. Based on (W6), some parts could be replaced by equations (for instance, how the combination of spatial and temporal graphs is achieved and how the final output is derived) or references since several components are similarly used to other studies, e.g., patching. Similarly, some details about the model could be moved from the appendix to the main text.
4. Based on (W7), it is unclear how interpretability is achieved. Since patching is performed first, followed by temporal graph learning and then spatial graph learning, seems a bit exaggerated that interpretability is achieved, e.g., what can someone tell from the spatial graph concerning inter-series dependencies?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces GRAPHSTAGE, a fully GNN-based method that captures intra-series and inter-series dependencies while maintaining the shape of the input data through a channel-preserving strategy. Extensive experiments conducted on 13 real-world datasets
demonstrate that GRAPHSTAGE achieves competitive performance.

### Strengths
1. The paper is well-written and easy to follow.

2. Extensive experiments  across 13 benchmark datasets have demonstrated that GRAPHSTAGE achieves good performance.

### Weaknesses
1. My primary concern is regarding the term "noise" (see line 16). The authors claim that previous methods introduce additional noise, but it remains unclear what this "noise" refers to. How is it defined, and what impact does it have on the effectiveness of the methods?

2. This paper appears to be an incremental contribution, with its components largely derived from previous works. Channel-preserving Strategy is inspired by iTransformer, and the fully GNN perspective is inspired by FourierGNN.

### Questions
please refer to the weaknesses part

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The proposed GRAPHSTAGE model innovatively employs a pure Graph Neural Network (GNN) structure to preserve the channel structure of the input data. The authors claim that this approach avoids the noise and interference introduced by channel mixing methods.

### Strengths
1. By introducing the Spatio-Temporal Aggregation Graph Encoder (STAGE) module, the authors effectively separated and captured internal and external dependencies in the time series, enhancing the model's interpretability and predictive performance.

2. The paper conducted extensive experiments on 13 real-world datasets, demonstrating superior performance in multivariate time series prediction tasks, with significant improvements in prediction accuracy and computational efficiency compared to existing methods.

### Weaknesses
1. I don't quite understand why a purely graph-based structure can introduce noise and interference through channel mixing methods.

2. I have doubts about the effectiveness of purely using GNNs for time series analysis. The performance of GNNs heavily depends on the rationality of the graph structure. However, in certain datasets or tasks, constructing a reasonable graph structure is itself a challenge. If the graph structure fails to accurately capture the dependencies in the data, even the most advanced GNNs may struggle to achieve optimal performance. Additionally, the representational capacity of graphs may be limited in high-noise or irregular data, leading to a decline in the model's predictive capabilities. I hope the authors can provide a more comprehensive explanation.

3. FourierGNN also performs time series prediction, so why didn't the authors conduct a comprehensive comparison?

### Questions
see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
