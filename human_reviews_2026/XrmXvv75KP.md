# SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
Pre-trained models exhibit strong generalization to various downstream tasks. However, given the numerous models available in the model hub, identifying the most suitable one by individually fine-tuning is time-consuming. In this paper, we propose \textbf{SwiftTS}, a swift selection framework for time series pre-trained models. To avoid expensive forward propagation through all candidates, SwiftTS adopts a learning-guided approach that leverages historical dataset-model performance pairs across diverse horizons to predict model performance on unseen datasets. It employs a lightweight dual-encoder architecture that embeds time series and candidate models with rich characteristics, computing patchwise compatibility scores between data and model embeddings for efficient selection. To further enhance the generalization across datasets and horizons, we introduce a horizon-adaptive expert composition module that dynamically adjusts expert weights, and the transferable cross-task learning with cross-dataset and cross-horizon task sampling to enhance out-of-distribution (OOD) robustness. Extensive experiments on 14 downstream datasets and 8 pre-trained models demonstrate that SwiftTS achieves state-of-the-art performance in time series pre-trained model selection. The code and datasets are available at \href{}{https://github.com/decisionintelligence/SwiftTS}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SwiftTS, a learning-guided framework for rapid selection of time series pre-trained models using a multi-task meta-learning strategy. The approach employs a dual-encoder (temporal-aware data encoder and knowledge-infused model encoder) to encapsulate both dataset and model features, leveraging patch-wise cross-attention to compute compatibility scores for model selection across various horizons. To improve generalization across datasets and forecasting horizons, the method incorporates horizon-adaptive expert composition as well as a cross-task meta-learning protocol. SwiftTS is evaluated on 14 real-world time series datasets and 8 pre-trained models, demonstrating strong results against a wide range of feature-analytic and learning-based baselines.

### Strengths
1.	The idea of selecting more suitable pretrained models for different downstream tasks and datasets is both interesting and valuable, addressing a practically important yet underexplored challenge in time-series foundation modeling.

2.	The dual-encoder architecture is well-motivated for heterogeneous time series model pools and directly addresses the issue of costly, inconsistent feature extraction in prior work. The use of a patch-wise attention mechanism reflects a careful design choice that captures local temporal structures relevant for model-dataset compatibility.

3.	SwiftTS introduces a horizon-adaptive expert composition module that flexibly and effectively addresses horizon-specific variability, as showcased in both the framework diagram and associated experimental tables.

### Weaknesses
1.	The paper emphasizes that SwiftTS avoids costly forward passes through all candidate models. Yet, the functional embedding module still requires each candidate model to be evaluated (albeit offline) on synthetic inputs such as Gaussian noise. This operation remains linearly proportional to the number of candidate models and does not scale well to continuously evolving model pools. The efficiency claim is therefore only partially valid and should be quantified more carefully.
2.	Does the sampling strategy used in the Temporal-Aware Data Encoder introduce randomness that leads to different results across runs? If so, how is this variance controlled or mitigated during training and evaluation?
3.	What exactly does the topological structure represent? Specifically, what are the semantic meanings of the nodes and edges in this context?
4.	The results only report the alignment metric between the estimated and true rankings, but not the actual forecasting performance of different selection methods. Without these results, it is difficult to assess the real effectiveness of the proposed selection strategy.
5.	What are the forecasting results of the pretrained models on these datasets without applying the selection process? Reporting these would provide a clear baseline for evaluating the benefit of the proposed selection mechanism.
6.	There exist many other meta-learning-based methods for time-series forecasting, such as AutoForecast [1], which should be included for comparison in the experimental section.
7.	Could the authors clarify the rationale and empirical justification for using Gaussian noise to generate the functional embeddings of candidate models? Would using real or synthetic time-series inputs affect the informativeness or stability of these embeddings?
8.	How sensitive are the framework’s predictions to the number of candidate models in the hub? Does performance degrade with larger, more heterogeneous model pools, and what are the observed scaling behaviors?
9.	Could the authors provide more clarity about the construction of meta-training tasks, especially the sampling policy for forecasting horizons and dataset divisions?
10.	Could the authors clarify how they handle cases where meta-training tasks share overlapping datasets or closely related forecasting horizons, which may cause distribution leakage or task redundancy?

[1] Abdallah M, Rossi R, Mahadik K, et al. Autoforecast: Automatic time-series forecasting model selection[C]//Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022: 5-14.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

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
The paper presents SwiftTS, a learning-based framework for selecting pre-trained time series models efficiently. It uses a dual-encoder architecture (data encoder + model encoder) and multi-task meta-learning to predict model–dataset compatibility without exhaustive fine-tuning. Experiments on 14 datasets and 8 models show strong gains over existing methods.

### Strengths
- Addresses an important and under-explored problem in time series foundation model selection.
- Well-designed method combining meta, topological, and functional model embeddings.
- Extensive experiments with clear, consistent improvements.

### Weaknesses
- The design choices for the data and model encoders appear somewhat heuristic and lack sufficient justification. For example, why does the model encoder capture domain information while the data encoder does not? The paper could be strengthened by clarifying the design rationale of these encoders.  
- The meta-learner is trained on a relatively small pool (14 datasets × 8 models); a data-efficiency analysis or a discussion explaining why this scale suffices to learn reliable dataset–model correlations would improve credibility.  
- The functional embedding is obtained by probing models with Gaussian noise, which seems heuristic. Random noise may not reflect how models respond to real-world temporal structures; a justification for using Gaussian inputs would be helpful.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SwiftTS, a framework for fast and scalable selection of pre-trained time series models via multi-task meta-learning. Rather than fine-tuning each candidate model or relying on computationally expensive feature extraction for each selection, SwiftTS uses a lightweight dual-encoder to embed datasets and models, computes patchwise compatibility via cross-attention, and applies a horizon-adaptive mixture-of-experts approach. Further, the method leverages transferable cross-task meta-learning across datasets and forecasting horizons to enhance out-of-distribution robustness. The framework is evaluated on 14 real-world datasets and 8 pre-trained model families, showing state-of-the-art ranking accuracy and efficiency across a variety of horizons and domains.

### Strengths
1. The proposed dual-encoder architecture is well-conceived and technically sound. One encoder incorporates temporal awareness through the use of patching and attention mechanisms, while the other enables knowledge injection by integrating architectural metadata, graph-based topological structures, and functional embeddings derived from model behavior. This design is particularly well-justified for highly diverse model repositories.

2. The manuscript presents extensive experimental results and visualizations, which provide strong and multifaceted evidence supporting the effectiveness of the proposed model.

3. The application of meta-learning to address the heterogeneity of time-series domains and various pretrained models is highly appropriate and demonstrates solid methodological reasoning.

### Weaknesses
Although the paper shows runtime savings over fine-tuning, there's insufficient discussion of the the practical scaling beyond a fixed model zoo. For example, how does graph2vec embedding scale with hundreds or thousands of models with complex DAGs? Is there a resource bottleneck for functional embedding inference as the number of candidate models grows? The scalability arguments are more empirical than architectural; a more detailed analysis would be valuable.

### Questions
1. How is the meta information embedded and utilized? In particular, how are the five types of meta information combined within the model?

2.SwiftTS performs selection among multiple pretrained models. Could it support the addition or removal of models without retraining SwiftTS? This consideration may have implications for the scalability of the approach.

### Soundness
3

### Presentation
3

### Contribution
3
