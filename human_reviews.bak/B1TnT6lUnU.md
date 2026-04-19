# Structural Knowledge Informed Continual Learning for Multivariate Time Series Forecasting

- Decision: Reject
- Scores: 3, 5, 3, 3, 8

## Abstract
Recent studies in multivariate time series (MTS) forecasting reveal that explicitly modeling the hidden dependencies among different time series can yield promising forecasting performance and reliable explanations. However, modeling variable dependencies remains underexplored when MTS is continuously accumulated under different regimes (stages). Due to the potential distribution and dependency disparities, the underlying model may encounter the catastrophic forgetting problem, i.e., it is challenging to memorize and infer different types of variable dependencies across different regimes while maintaining forecasting performance.
To address this issue, we propose a novel Structural Knowledge Informed Continual Learning (SKI-CL) framework to perform MTS forecasting within a continual learning paradigm, which leverages structural knowledge to steer the forecasting model toward identifying and adapting to different regimes, and selects representative MTS samples from each regime for memory replay.
Specifically, we develop a forecasting model based on graph structure learning, where a consistency regularization scheme is imposed between the learned variable dependencies and the structural knowledge (e.g., physical constraints, domain knowledge, feature similarity, which provides regime characterization) while optimizing the forecasting objective over the MTS data. As such, MTS representations learned in each regime are associated with distinct structural knowledge, which helps the model memorize a variety of conceivable scenarios and results in accurate forecasts in the continual learning context.
Meanwhile, we develop a representation-matching memory replay scheme that maximizes the temporal coverage of MTS data to efficiently preserve the underlying temporal dynamics and dependency structures of each regime. 
Thorough empirical studies on synthetic and real-world benchmarks validate SKI-CL's efficacy and advantages over the state-of-the-art for continual MTS forecasting tasks. SKI-CL can also infer faithful dependency structures that closely align to structural knowledge in the test stage.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a continuous multivariate time series prediction framework based on structural knowledge， which utilizes structural knowledge to enhance the prediction of multivariate time series in a continuous learning environment. The proposed framework incorporates a deep prediction model that combines a graph learner to capture variable dependencies and a regularization scheme to ensure the consistency between the learned variable dependencies and the structural knowledge. The authors tackle the challenge of modeling variable dependencies across different regimes while maintaining the prediction performance. Experimental results on several real datasets are presented, demonstrating the effectiveness of the proposed framework in improving the prediction performance and maintaining consistency with the structural knowledge.

### Strengths
1. This paper propose a new freamwork which is aimed at knowledge transfer learning.  

2. The proposed model can well use the knowledge of former tasks. 

3. The paper is well written.

### Weaknesses
1. The complexity of the model increases so much relative to the performance gain that I don't see the need for such a complex design.

2. The model is not novel enough, as far as I know, the graph structure, the memory module, these are not new concepts. 

3. Models are not so essential to the development of the field.

### Questions
The OFA is published in ICML2022, the more recently model should be added as baselines. 


Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency
enhanced decomposed transformer for long-term series forecasting. In International Conference
on Machine Learning, pp. 27268–27286. PMLR, 2022.

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
This paper introduces the Structural Knowledge Informed Continual Learning (SKI-CL) framework for multivariate time series forecasting, which addresses the challenge of catastrophic forgetting when modeling variable dependencies across different regimes. SKI-CL leverages structural knowledge to guide the model in identifying and adapting to regime-specific patterns, and employs a representation-matching memory replay scheme to preserve temporal dynamics and dependency structures. The framework's efficacy is validated through experiments on synthetic and real-world datasets, demonstrating its superiority over state-of-the-art methods in continual MTS forecasting and dependency structure inference.

### Strengths
1. **The motivation is very meaningful.**

   As recent research [1] has pointed out, distribution drift of time series (including dependency structures) may be the core bottleneck in the forecasting process. Therefore, I believe the authors are attempting to conduct a very significant study.

2. **The writing is good and easy to follow.**

### Weaknesses
1. **I am concerned whether the change in dependency structures is indeed the core bottleneck in real-world scenarios.**
- 1.1 The authors should consider using analyses based on real data rather than just the schematic diagram in Figure 1.  
- 1.2 Is this issue the core bottleneck of the dataset chosen by the authors? Are other more commonly used datasets, such as METR-LA and PEMS04, also applicable to this method?  
- 1.3 In the current manuscript, it seems that the effectiveness of modeling the change in dependency structures can only be validated through experimental results. Thus, the authors need to compare against a broader range of stronger baseline methods, such as latent (but static) graph models, dynamic (but predefined graph-based) models (e.g., DGCRN [2]), and non-graph models (e.g., STID[3], STNorm[4], STEAformer[5]). The baselines currently chosen by the authors are not strong enough. For example, TCN is a conventional temporal model, while PatchTST, DLinear, TimesNet, and iTransformer are long-sequence forecasting models that are not specifically designed for spatiotemporal prediction and do not explicitly model the dependency graph between sequences. Additionally, the code for GTS contains unintentional errors that significantly impact its performance compared to the original paper.

2. **Lack of sufficient insights.**
- 2.1 What are the core challenges and solutions in modeling the dynamic changes of dependency structures for time series forecasting? Currently, I cannot clearly see the connection between the challenges and the proposed techniques.
- 2.2 What is the distinction between dynamically changing dependency structures and dynamic graph learning?  

---

[1] BasicTS: Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis. TKDE 2024.

[2] DGCRN: Dynamic graph convolutional recurrent network for traffic prediction: Benchmark and solution. TKDD 2023.

[3] STID: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting. CIKM 2022.

[4] STNorm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting. SIGKDD 2022.

[5] STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting. CIKM 2023.

### Questions
See Weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose a novel Structural Knowledge Informed Continual Learning (SKI-CL) framework with a graph-based forecaster and a novel representation-matching memory replay scheme, which can perform MTS forecasting and infer dependency structures accurately under the continual learning setting. Experiments demonstrate the superiority of the proposed framework in continual MTS forecasting and dependency structure inference.

### Strengths
1.The paper proposes an interesting method by combining dynamic graph learning with a novel representation-matching memory replay scheme for MTS forecasting and dependency structures inference under continual learning setting.

2.The organization of this paper is clear.

### Weaknesses
1.The paper just combines continual learning with multi-variate time series forecasting, which can be seen as an incremental work. The authors should identify the research topic and main contribution, instead of adapting continual learning to multi-variate time series forecasting.

2.The design of structural knowledge-informed graph learning model lacks innovation. The parameterized graph learning is similar to many works, e.g., AGCRN [1]. The authors could further clarify how their graph learning method differs significantly from existing works.

3.The analysis of representation-matching memory replay scheme is uncompleted:

(1)The authors should clarify how their representation-matching memory replay scheme differs with other experience-replay methods.

(2)The analysis of efficiency about the scheme is not well discussed in the paper. It would be beneficial to explain the scheme's efficiency from both theoretical and experimental perspectives. 

(3)The visualization of selected samples in the scheme should be included to demonstrate the effectiveness of the model.

4.In section 3.4, the authors could explain the inference process more clearly.

5.The paper has some weaknesses in the experiments, which are not convincing enough:

(1)Since one of the main contributions is developing a graph-based forecaster, some recent graph-based time series forecasting models should be mentioned and compared, e.g., CrossGNN [2] and MSGNet [3]. In addition, the continual learning methods applied to forecasting methods are old and some latest methods could be compared.

(2)Different datasets have different methods to construct regimes, e.g., by year, state, activity, and adjacency, authors could further investigate the effect of different construction methods on the performance of SKI-CL. Some construction methods, e.g., by state and activity, are not reasonable and deviate from the intention of continual learning. In addition, the paper misses details regarding the train-test data splits.

6.From a reader's perspective, the authors should enhance presentation to avoid misunderstanding. For example, for Fig. 6, the horizontal coordinate and vertical coordinate of heat map should start from 1. For Table. 1, what do the bolded and underlined results mean? For Equation 3, what does the $n_K$ mean?

7.Strong recommendation to make the code publicly available.

[1] Bai, L., Yao, L., Li, C., Wang, X., & Wang, C. 2020. Adaptive graph convolutional recurrent network for traffic forecasting. Advances in Neural Information Processing Systems, 33, 17804-17815.

[2] Huang, Q., Shen, L., Zhang, R., Ding, S., Wang, B., Zhou, Z., & Wang, Y. 2023. CrossGNN: Confronting noisy multivariate time series via cross interaction refinement. Advances in Neural Information Processing Systems, 36, 46885-46902.

[3] Cai, W., Liang, Y., Liu, X., Feng, J., & Wu, Y. (2024, March). MSGNet: Learning multi-scale inter-series correlations for multivariate time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, 38, 11141-11149.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Existing MTS models suffer from the forgetting problem in the continuous learning scenario, which makes it difficult to remember the variable dependencies from each regime. To solve this problem, the authors propose a structural knowledge informed continuous learning framework to infer the dependency structure between different regimes. In this framework, the authors propose to use a graph structure to explicitly represent the dependencies between regimes, and introduce a regularization to facilitate continuous learning. In addition, in order to alleviate the forgetting problem, the authors propose a new memory replay method, which effectively preserves the temporal dynamics and dependency structure of historical regimes by selecting MTS data samples that satisfy the maximum temporal coverage. Finally, the authors verify the superior performance of the proposed method through a large number of experiments.

### Strengths
1. Modeling dependencies between states in a graph is novel.
2. When describing the method, a large number of model diagrams and structure diagrams are used to help readers understand the method.
3. The authors compared a large number of baselines, and the experimental workload was large.

### Weaknesses
1. On the one hand, the authors introduce a graph structure to model dependencies. On the other hand, the authors propose a new replay method to solve the forgetting problem in continuous learning. In my opinion, these are two points, but it is inappropriate for the author to mix the two points together in the abstract and introduction.
2. This paper did not conduct ablation experiments and did not demonstrate the significance of each part of the method.
3. The performance on the four datasets is not much better than the baselines.

### Questions
1. How well does the model resist forgetting after many new regimes coming
2. How to understand "We emphasize that we don't intend to use structural knowledge as a ground truth", but isn't structural knowledge still used as a label in the loss of $L_G$?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Structural Knowledge Informed Continual Learning (SKI-CL), a framework for multivariate time series (MTS) forecasting that addresses challenges posed by variable dependencies across different regimes. By leveraging structural knowledge (such as physical constraints and domain knowledge), SKI-CL aims to mitigate catastrophic forgetting and improve model adaptation to shifting data distributions. The framework utilizes dynamic graph learning with consistency regularization, aligning learned dependencies with structural knowledge for better regime recognition. A representation-matching memory replay scheme is introduced to retain essential temporal dynamics from each regime. Experiments on synthetic and real-world datasets show superior forecasting accuracy and reliable dependency inference compared to state-of-the-art methods.

### Strengths
1. This paper is well written. The notations are clear.

2. It provides up-to-date literature on MTS techniques with regards to regime shift. It underscores the potential of graph-based learning, paving ways for deep regime awareness in MTS.

3. Among many lines of work addressing the regime discovery in MTS, graph-based learning has been well explored. However, this paper provides a systematic approach to tackle the regime shift in a rationale and reasonable way.

4. The experiments are convincing and supports the arguments in the merits of the proposed SKI-CL framework. Especially, Figure 4, 5 and Figure 7,8,9,10 high light the differentiation of the proposed framework from competing methods well.

### Weaknesses
1. Overall, the technical documentation is comprehensive. It would be clearer if an algorithmic procedure is provided to give a high-level reference of how different components orchestrates in Figure 2 and 3. 
2. While the traceback of nodes is valid and transparent in the inference, in domain applications, calibrated regimes matter to model owners because it helps interpret the results in the format of narratives. The proposed learning and inference of graph structure contain Coverage Maximization and Representation Matching Selection which are unsupervised, therefore not yet assembled for regime calibration tasks.

### Questions
1. Benchmarking regimes in multivariate time series forecasting is a foundation problem in MTS research, traditional econometrics method, e,g., Markov regime switch model that can be solved by EM algorithm can serve as a baseline model for regime discovery. Would that be something that can help bring the diverging methodologies to the same ground for a fair and reasonable competition, instead of the checking the numerical metrics?

2. The message of Table 5 is to compare the number of baseline model parameters. Therefore, would it be better if the sorting is number of baseline model parameters instead of chronological order?

### Soundness
2

### Presentation
4

### Contribution
3
