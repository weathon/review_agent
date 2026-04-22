# AutoMixer: A Lightweight and Scalable Industrial 5.0 Safety Assurance Model with Multi-Scale Adaptive Dual-Attention

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 8, 2

## Abstract
With the rapid growth of intelligent transportation and industrial automation, traffic safety management and industrial system safety generate vast amounts of spatio-temporal data. These data offer rich temporal and spatial patterns for analysis but pose significant challenges, including dynamic traffic patterns, high-dimensional sensor data, and complex anomalies in industrial systems. Traditional methods struggle to capture nonlinear accident patterns, handle noisy sensor data, or model intricate multi-variable interactions, especially in real-time scenarios. Although deep learning and large-scale models have improved the accuracy of accident prediction and anomaly detection, their reliance on complex spatial operations and large parameter sizes creates computational bottlenecks, limiting scalability in large-scale and real-time safety applications. Therefore, we propose AutoMixer, a lightweight and scalable model that avoids explicit spatial modeling. It uses a dual cross-attention module to identify coupled trend and periodic features in multi-resolution spatio-temporal data. Extensive experiments demonstrate that AutoMixer consistently outperforms state-of-the-art baselines, achieving 7% higher detection accuracy while effectively handling large-scale node distributions and high-frequency data. AutoMixer provides a practical and deployable solution for real-time accident detection and industrial system safety analysis, enhancing computational efficiency and applicability in resource-constrained environments, thus optimizing performance for large-scale traffic and industrial safety tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents **AutoMixer**, a lightweight time-series forecasting model designed for Industrial 5.0 safety and traffic anomaly detection. The model applies frequency-domain decomposition (DFT-based) to capture multi-scale temporal features and uses a dual cross-attention mechanism to integrate trend and periodic components without explicit spatial modeling. The authors claim state-of-the-art performance on multiple datasets such as ETT, Electricity, and PEMS-BAY.

### Strengths
1.   Lightweight design with efficient inference suitable for real-time industrial applications.\
2.   Ablation studies are reasonably complete.
3.   Stable performance on short-term forecasting tasks.
4.   Implementation simplicity facilitates reproducibility.

### Weaknesses
1.   Limited novelty: The proposed combination of frequency decomposition and attention mechanisms is not conceptually new.

2.   Weak long-horizon performance: The model’s accuracy degrades significantly for long forecasting horizons (e.g., output=720), contradicting the claimed “long-term scalability.”
3.   Marginal overall improvement: Across most datasets, AutoMixer’s improvements over baselines such as FEDformer and TimeMixer are small and often within the margin of experimental variance. In some long-term cases, it even performs worse. This weakens the empirical significance of the claimed contribution.
4.   Presentation flaw: Main comparative results are absent from the main text, violating transparency norms.
5.   Lack of empirical validation for spatial modeling removal: No evidence that omitting spatial graphs maintains robustness in truly spatially correlated datasets.
6.   Lack of evidence for removing spatial modeling: The paper does not test whether omitting explicit spatial graphs maintains robustness in spatially correlated datasets such as traffic networks.

### Questions
1.   What is the fundamental difference between AutoMixer and TimeMixer/FEDformer in terms of mechanism?
2.   Since AutoMixer eliminates spatial graphs, have the authors evaluated its performance on datasets with strong spatial topology (e.g., traffic networks)?
3.   The results show a noticeable drop in performance for output=720.Have the authors analyzed why AutoMixer fails to maintain long-horizon stability?
4.   The paper frequently emphasizes scalability and lightweight design. Could the authors provide FLOPs, parameter count, and inference latency comparisons with TimeMixer or FEDformer?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents AutoMixer, a lightweight and scalable model for spatial-temporal modelling.   
It consists of four components: adaptive frequency-domain decomposition, dynamic coupled feature weighting, a dual-attention mechanism for spatiotemporal analysis, and a multi-resolution dynamic coupling module. 

Experiments are conducted on time-series and spatial-temporal modelling tasks such as energy, weather, traffic, etc.   
Several baseline methods such as TimeMixer, PatchTST, AutoFormer, and FedFormer are adopted for comparison.   
Ablation studies of module and parameter analysis are conducted. Model size, training time, inference time, and GPU memory are also compared and reported.

### Strengths
## Strengths
- This paper proposes AutoMixer, with four core modules: Adaptive Frequency-Domain Decomposition, dynamic coupled feature weighting, dual-attention mechanism, and multi-resolution dynamic coupling. 
- Experiments are conducted on several datasets: Ett, Electricity, Weather, Metr-LA, PEMS-BAY, Traffic, and PEMS03/08. 
> Baseline methods include TimeMixer 2024, Informer 2021, AutoFormer 2021, FedFormer 2022, PyraFormer 2022, DLinear 2022, and PatchTST 2023.   
> Ablation study, parameter analysis, complexity and runtime analysis are reported.

### Weaknesses
## Weaknesses
- **Lack of Novelty** The method is a simple combination of mature modules/techniques in the area of time-series analysis and spatiotemporal modelling. 
> E.g., Frequency-Domain Decomposition has been proposed in FedFormer.   
> Dynamic coupled feature weighting is a basic and simple re-weighting using NN.   
> Dual-attention mechanism modifies self-attention, which has been extensively studied in existing literature such as AutoFormer, Informer.   
> Multi-resolution processing is also a common technique in spatial-temporal modelling, e.g., PatchTST. 
- **Out-of-Date Comparison** The baseline models are out-of-date: TimeMixer 2024, Informer 2021, AutoFormer 2021, FedFormer 2022, PyraFormer 2022, DLinear 2022, and PatchTST 2023. Among them, only TimeMixer is a 2024 paper, others are not new and recent SOTA methods. 
> Most recent methods such as LLMs are not discussed and compared, e.g., Time-LLM, Uni-ST [R2]. 


[R1] Jin, Ming, et al. "Time-llm: Time series forecasting by reprogramming large language models." arXiv preprint arXiv:2310.01728 (2023).

[R2] Yuan, Yuan, et al. "Unist: A prompt-empowered universal model for urban spatio-temporal prediction." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2024.

### Questions
In table 3, why the results of Informer, ... , DLinear are quite large (e.g., 64608), while TimeMixer is only 41.1397?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents an ML model designed to predict accidents and detect problems in traffic systems and industrial facilities (like factories and power plants). Its problem statement is as follows: Modern transportation systems and factories generate massive amounts of sensor data. Traditional analysis methods struggle because the data is complex and noisy, real-time analysis is needed but computationally expensive and existing AI models are too slow or inaccurate for practical use. Innovations in Automixer are stated as: (a) smarter pattern recognition - It breaks down data into different frequency components to identify both regular patterns and trends (b) Multi-scale analysis at different time scales simultaneously, (c) its dual-attention mechanism and (d) No need for explicit spatial modeling of the data. Results claimed as outperforming existing methods by roughly 4% in detection accuracy when tested on traffic accident prediction (using data from road sensors) and industrial equipment monitoring for failures in power systems and factories, even in the presence of 
short-term historical data (100 data points),  incomplete sensor readings, and at scale.

### Strengths
This is a well-written paper that looks at the standard problem of predicting accident potential on roads and defect potential in manufacturing operations. The techniques employed seemed sound. Results were good (a 4% improvement). Ablation studies were done (good).

### Weaknesses
Lines of the graphs in Figures 2 and 3 need to have more contrast in visibility. Also the use of colors will be a barrier for readers with vision issues.

Computational resource usage of the various techniques were presented. So that the claim of real time response is supported sugges adding graphs on computational times.

### Questions
How did the presented techniques compare with the baselines with respect to computational time?

### Soundness
3

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
5

### Summary
This paper explores accident prediction and anomaly detection with an attention-based framework. It leverages a dual cross-attention head to model both spatial and temporal dependency. The authors aim to address the high computational cost of large-scale models. Experiments have been conducted on real datasets by comparing with transformer-based models.

### Strengths
1. The paper pointed out the weaknesses of the transformer-based model for traffic prediction.
2. Both accuracy and efficiency have been reported.
3. Figure 1 is a good example to explain the complicated framework.

### Weaknesses
1. The paper states that "their reliance on complex spatial
operations and large parameter sizes creates computational bottlenecks" but have not provided any supportive empirical data about the bottlenecks. In fact, according to Table 4 and Figure 3, the proposed method is even worse and at the same level as the baseline. It is confusing how the proposed method addresses "Lightweight and Scalable".

2. The framework is not novel. It basically modifies the transformer framework. Such a change is incremental.

3. According to Table 3, the MSE of Informeron London 64608, where the MSE of the proposed method is 40. For some columns, the result is NA. It is unclear why NA is there, and it looks like the baselines have not been used correctly. 

4. The paper only compares with transformer-based models. It is well-known transformer is heavy. Since the motivation is the "cost", the author should compare with light models, such as graph-based models.

5. The experiment quality is low. The maximum number of nodes is only 300. It is not large enough to prove the scalability. Varying the number of nodes has not been conducted. The error variance has not been reported. There is no visualization on a real map for the prediction.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
1
