# Fine-Grained Urban Traffic Forecasting on Metropolis-Scale Road Networks

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Traffic forecasting on road networks is a complex task of significant practical importance that has recently attracted considerable attention from the machine learning community, with spatiotemporal graph neural networks (GNNs) becoming the most popular approach. The proper evaluation of traffic forecasting methods requires realistic datasets, but current publicly available benchmarks have significant drawbacks, including the absence of information about road connectivity for road graph construction, limited information about road properties, and a relatively small number of road segments that falls short of real-world applications. Further, current datasets mostly contain information about intercity highways with sparsely located sensors, while city road networks arguably present a more challenging forecasting task due to much denser roads and more complex urban traffic patterns. In this work, we provide a more complete, realistic, and challenging benchmark for traffic forecasting by releasing datasets representing the road networks of two major cities, with the largest containing almost 100,000 road segments (more than a 10-fold increase relative to existing datasets). Our datasets contain rich road features and provide fine-grained data about both traffic volume and traffic speed, allowing for building more holistic traffic forecasting systems. We show that most current implementations of neural spatiotemporal models for traffic forecasting have problems scaling to datasets of our size. To overcome this issue, we propose an alternative approach to neural traffic forecasting that uses a GNN without a dedicated module for temporal sequence processing, thus achieving much better scalability, while also demonstrating stronger forecasting performance. We hope our datasets and modeling insights will serve as a valuable resource for research in traffic forecasting and, more generally, urban computing and smart city development.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces two large-scale, fine-grained urban traffic forecasting datasets (city-traffic-M with 53k road segments and city-traffic-L with 94k segments) derived from GPS traces in two major (anonymized) cities. These datasets provide real road graph connectivity, rich static road features, and dual targets (5-min granularity traffic speed and volume) over ~4 months. They starkly contrast prior benchmarks (e.g., METR-LA: 207 sparse highway sensors; LargeST: 8.6k), which lack urban density, true graphs, and volume data. The authors benchmark popular spatiotemporal GNNs (DCRNN, GRUGCN, STGCN, GWN), showing most fail to scale (memory/time blow up on 80GB GPU). They propose a scalable alternative: per-node linear flattening of the lookback window into a single embedding, followed by a lightweight GNN (mean or transformer-attention aggregation, with skips/LN/MLPs). This "time-then-graph" design has O(nd) memory (n=nodes, d=dim), enabling longer horizons cheaply. It outperforms baselines in accuracy while training faster.

### Strengths
1- Large-scale dataset: The released dataset is openly available, metropolis-scale urban traffic datasets, 10-500x larger than priors (e.g., METR-LA: 207; LargeST: 8.6k) with true directed road graphs (adjacency via traffic rules), 26 rich static features (length, speed limits, surface quality, endpoints, transit lanes), and dual 5-min GPS targets (speed + volume, Jul-Nov 2024; realistic 5-25% speed missingness). Urban-dense (Figs. 1-2) vs. priors' sparse highways; first speed+volume (Table 1).

2- Extensive and Reproducible Experiments: Experiments compare against major baselines (DCRNN, STGCN, GRUGCN, GWN) under strict GPU constraints. Ablation on lookback window length, scalability tests, and runtime benchmarks are provided. Clear documentation of training environment (single A100 GPU, full-batch training) and public code/data links.

### Weaknesses
1 - Experimental Limitations: i) Metrics are narrow: Only MAE, no MAPE/RMSE/MSE (std for traffic); ii) Horizons missing: Fixed prediction length (1-step?), no 15/30/60min (critical for routing); iii) Baselines dated: Miss recent (Chronos/TimesFM-GNN; MTGNN; STNorm); no multi-task (joint speed+vol); no extrinsics (events/holidays). No inference latency; CPU/FLOPs; zero-shot roads.

2- Narrow Evaluation Scope: All experiments are conducted on the two proposed datasets only. The model is not evaluated on existing public datasets (e.g., METR-LA, PeMS-BAY), which limits direct performance comparison and external validity. No cross-city transfer or domain generalization experiments are shown, despite mentioning this as a future direction.

3 - Limited Long-Term Dynamics: The dataset covers only four months, restricting the evaluation of seasonal and long-term forecasting. The authors acknowledge this but do not discuss how short data duration affects model generalization to yearly cycles.

4 - Research Scope: most importantly, in my opinion, providing the research community a large time series traffic dataset is a good contribution; nevertheless, I wonder if this research focus is suitable for the ICLR conference, which mainly focuses on learning representation. Please emphasize the main contribution of this work for better clarification.

### Questions
Please see the comments in Weaknesses Section.

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
3

### Summary
This paper addresses limitations in existing traffic forecasting benchmarks by introducing two large-scale datasets, city-traffic-M (53,530 road segments) and city-traffic-L (94,009 road segments), representing detailed urban road networks with actual connectivity, rich road features, and simultaneous traffic volume/speed measurements. The authors demonstrate that most existing spatiotemporal GNN models struggle with scalability on these datasets and propose an efficient alternative approach that uses a linear layer to encode temporal information followed by a GNN, achieving better performance and significantly reduced training times compared to established baselines.

### Strengths
- Provides realistic urban road networks with actual connectivity instead of heuristic sensor-based graphs, enabling more authentic evaluation of spatial models 

- Systematically evaluates computational limitations of existing models on large-scale data, revealing that only 4 of 8 considered models could run on city-traffic-M with 80GB VRAM 

- Achieves state-of-the-art results on both datasets and prediction tasks, with GNN-TrfAttn outperforming all baselines in most configurations.

### Weaknesses
- The core technical approach adapts existing "time-then-graph" paradigms with minimal innovation beyond application to traffic forecasting (see Sec. 4.1)
- The temporal encoding strategy using a simple linear layer has been explored in recent time series literature ([Zeng et al., 2023]; [Das et al., 2023]), though the adaptation to graph settings is less common
- No ablation studies examine the contribution of different components (skip connections, normalization, MLP blocks) to the overall performance (see Sec. 4.1)


- Despite including 26 road attributes, the experiments do not systematically evaluate how these features impact forecasting performance or which are most valuable (see Sec. 3; Sec. 4)
- No experiments explore feature selection or importance analysis, missing opportunity to provide insights about critical urban traffic factors (see Table 1; Appendix A)
- The proposed models use all features without justification or analysis of their individual contributions (see Sec. 4.1; Appendix D)

- The complexity analysis in Section 4.1 provides Big-O notation but lacks explicit mathematical formulations of the proposed models' operations
- The description of the GNN architectures is somewhat vague, mentioning "mean aggregation" and "transformer-like multihead attention" without precise mathematical definitions (see Sec. 4.1)
- No equations specify how the temporal encoding layer transforms the lookback window into node representations, leaving implementation details ambiguous


- Only four established baselines are evaluated, missing recent attention-based architectures that might offer different scalability-performance tradeoffs (see Sec. 4.1)
- The comparison uses fixed hyperparameters from LargeST repository rather than optimized configurations for each model on the new datasets (see Appendix D)
- No analysis of why specific models (STGCN) fail to scale while others succeed, beyond general complexity arguments (see Table 4; Sec. 4.2)

### Questions
**Suggestions for Improvement**
- Conduct ablation studies to quantify the contribution of each architectural component (skip connections, normalization, attention mechanisms) to performance and scalability
- Compare against a wider range of temporal encoding strategies beyond simple linear projection, such as MLP encoders or frequency-domain approaches
- Explore hybrid approaches that balance the efficiency of the proposed method with the expressive power of more complex temporal models

- Systematically evaluate the impact of different road attributes through feature ablation studies and importance analysis
- Design experiments that specifically test the models' ability to leverage different feature types (categorical vs. numerical, structural vs. regulatory)
- Include analysis of which features are most predictive for different traffic conditions (congestion vs. free flow, urban vs. peripheral roads)


- Provide explicit mathematical formulations for both GNN-Mean and GNN-TrfAttn architectures, including aggregation functions and update rules
- Include equations specifying the temporal encoding operation and how it integrates with the GNN components
- Formalize the complexity analysis with concrete examples showing how parameters scale with dataset size and architecture choices


- Include additional baselines, particularly attention-based models like ASTGCN or GMAN, even if they require sampling or approximation for large graphs
- Conduct hyperparameter optimization for all models to ensure fair comparison rather than using fixed configurations from previous work
- Perform deeper analysis of failure cases, examining specific computational bottlenecks and memory usage patterns across different architectures

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
This paper introduces Fine-Grained Urban Traffic Forecasting on Metropolis-Scale Road Networks, addressing a long-standing limitation of existing traffic forecasting benchmarks such as METR-LA, PEMS-BAY, and LargeST. The authors identify that these datasets are too small, lack true road connectivity information, and primarily represent intercity highways rather than dense urban environments. To overcome these gaps, the authors construct two large-scale city-traffic datasets—city-traffic-M and city-traffic-L—with ≈50K and ≈94K road segments respectively. Each segment has 26 static attributes and fine-grained 5-minute resolution traffic measurements (speed and volume) derived from GPS traces. The datasets capture actual road-level connectivity, making them the most realistic open urban traffic benchmarks to date. Empirically, the paper benchmarks several representative graph-based spatiotemporal models and demonstrates their poor scalability to these large networks. Motivated by this, the authors propose a simple yet efficient GNN framework that discards explicit sequence modules (e.g., RNN or temporal convolutions) and instead linearly encodes the temporal window into a single embedding per node before applying GNN layers.

### Strengths
1. The introduction of city-scale urban traffic benchmarks is a substantial and timely contribution. 
2. The paper proposes a computationally elegant re-formulation of temporal modeling for graph time series.
3. The benchmarking section is thorough and replicable, covering baselines from naïve heuristics to state-of-the-art graph models.

### Weaknesses
1. Limited methodological novelty in modeling. The proposed model, while effective, essentially adapts known ideas from recent efficient time-series methods (e.g., N-BEATS-like linear temporal encoders) to graph data.
2. The datasets span only four months, precluding evaluation of seasonal or annual trends. While understandable for an initial release, this limits long-term forecasting and transfer learning studies.
3. The study would benefit from including RMSE and MAPE, as these are common metrics in traffic forecasting and provide complementary perspectives on error characteristics.
4. Given the dataset’s richness (26 attributes), it would be informative to evaluate the marginal benefit of these features—do they meaningfully improve forecasting over using only historical dynamics?

### Questions
See in weaknesses.

### Soundness
2

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
This paper introduces a novel benchmark for urban traffic forecasting, providing detailed datasets and proposing an efficient model to address the challenges of large-scale traffic prediction. The contributions are as follows:
1. Existing benchmarks for evaluating traffic forecasting methods have limitations, including sparse sensor data, lack of detailed road connectivity, and limited coverage of urban areas. This paper addresses these limitations by providing comprehensive datasets for two major cities and proposing an efficient GNN-based model for traffic forecasting.
2. The authors introduce two new datasets, city-traffic-M and city-traffic-L, representing detailed road networks of two major cities. These datasets contain almost 100,000 road segments, significantly more than existing datasets. They provide rich road features, including speed limits, road types, and traffic volume and speed data. The datasets are constructed using GPS measurements, offering fine-grained temporal data and actual road connectivity, which is a major improvement over heuristic-based graph construction in existing datasets.

### Strengths
#### Originality
The authors introduce two new datasets, city-traffic-M and city-traffic-L, which represent a significant advancement over existing benchmarks. These datasets are the first to provide detailed, fine-grained traffic data for large-scale urban road networks, including nearly 100,000 road segments. This is a substantial increase compared to existing datasets, which typically contain only a few hundred to a few thousand segments. The datasets also include rich road features and actual road connectivity, addressing critical limitations of previous benchmarks.

#### Quality
The quality of the research presented in the paper is high. The authors have meticulously collected and processed the data, ensuring that it is comprehensive and representative of real-world urban traffic conditions. The datasets include a wide range of static and dynamic features, providing a rich source of information for model training and evaluation.

#### Clarity
The paper is well-written and easy to follow, making it accessible to both experts and non-experts in the field.

#### Significance
The significance of the paper lies in its potential to advance the field of urban traffic forecasting and related areas. The new datasets provide a valuable resource for researchers working on traffic forecasting, urban computing, and smart city applications. The datasets' fine-grained nature and rich features enable more realistic and comprehensive studies, potentially leading to more accurate and robust forecasting models.

### Weaknesses
1. The datasets are derived from only two major cities, which may limit the generalizability of the findings to other urban environments. Different cities can have unique traffic patterns, road structures, and regulatory frameworks that might not be captured by these datasets.
2. The datasets cover only a four-month period (July 1st to November 1st, 2024). This limited temporal coverage might not be sufficient to capture long-term trends and seasonal variations in traffic patterns, which are crucial for developing models that can handle annual cycles.
3. While the authors compare their proposed models against several established baselines, the evaluation is limited to mean absolute error (MAE) as the primary metric. Other metrics, such as mean squared error (MSE) or root mean squared error (RMSE), could provide additional insights into the models' performance, especially in terms of handling outliers and larger errors.
4. The paper mentions the potential for cross-city generalization but does not provide any experimental results or analysis to support this claim. Understanding how well models trained on one city can generalize to another is crucial for developing universally applicable solutions.

### Questions
1. Could the authors provide more details on the selection criteria for the two cities included in the datasets? Specifically, what characteristics of these cities make them representative of broader urban environments?
2. Given the limited temporal coverage of the datasets (four months), how do the authors plan to address the potential limitations in capturing long-term trends and seasonal variations?
3. Why did the authors choose MAE as the primary evaluation metric? Could they provide insights into the potential benefits of including additional metrics like MSE, RMSE, or R²?
4. Could the authors provide more details on the computational complexity and resource requirements during inference for the proposed models? How do they plan to address the efficiency of models during real-time prediction?
5. Could the authors provide more details on their plans to evaluate the cross-city generalization capabilities of the proposed models? Are there any preliminary results or insights they can share?
6. How do the authors plan to handle missing data in the datasets, particularly for traffic speed? Are there any specific methods or techniques they are considering?
7. How do the authors plan to address the practical deployment of the proposed models in real-world traffic monitoring systems? Are there any specific challenges or considerations they foresee?
8. Could the authors provide a more detailed comparative analysis of their datasets with existing benchmarks? Specifically, how do the proposed datasets address the limitations of existing benchmarks in terms of urban traffic forecasting?

### Soundness
3

### Presentation
3

### Contribution
3
