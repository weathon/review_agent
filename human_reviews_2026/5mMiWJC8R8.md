# DSTN: Early Spatio-Temporal Forecasting with Dynamic Propagation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
In most spatio-temporal prediction tasks, the timeliness of predictions is more critical than their accuracy. For instance, in tasks such as crime prediction, traffic congestion forecasting, and wildfire early warning, waiting longer to gather additional information may improve prediction accuracy, but it does not provide enough preparation time for subsequent actions, rendering the precise predictions valueless. Therefore, balancing between prediction timeliness and accuracy is essential for such tasks. In this paper, we propose an adaptive early spatio-temporal prediction model with a dynamic propagation matrix (DSTN), which captures causal relationships between nodes to enhance prediction timeliness while maintaining accuracy. Our model makes the following contributions: (1) Exploiting the similar long-term patterns of node signals for early prediction. (2) Proposing the concept of Asynchronous Spatio-temporal Causal Frame Pair to effectively capture the spatio-temporal causal relationships between different nodes. (3) Constructing a dynamic propagation matrix to filter out irrelevant information for early prediction. Experimental results on four large-scale real-world datasets demonstrate that the performance of our proposed DSTN model generally outperforms all baselines. The source code is available at https://anonymous.4open.science/r/DSTN-DB49.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DSTN (Dynamic Spatio-Temporal Network), an adaptive early spatio-temporal prediction model that balances prediction timeliness and accuracy. The core contribution lies in constructing a dynamic propagation matrix by combining frequency-domain similarity (capturing long-term periodic patterns) and asynchronous spatio-temporal causal relationships (capturing short-term dynamic causality). A reinforcement learning-based controller determines the optimal prediction time for each node. The authors evaluate DSTN on four real-world datasets (METR-LA, PEMS08, EMS, NYPD) and claim superior performance over baselines in early prediction scenarios.

### Strengths
**S1. Important and well-motivated problem**: Early spatio-temporal prediction is practically valuable for time-sensitive applications (crime prevention, traffic management). The paper clearly articulates the motivation.

**S2. Novel conceptual framing**: The "Asynchronous Spatio-temporal Causal Frame Pair" concept, offers an interesting perspective on time-delayed causality that differs from sliding window approaches.

**S3. Comprehensive experiments**: The paper tests on four datasets with multiple baselines and includes ablation studies, demonstrating effort in empirical evaluation.

### Weaknesses
**W1. Weak theoretical foundation:** Phase-based delay estimation (Eq. 4) assumes **strict periodicity**, which fails for non-stationary signals with multiple frequency components and time-varying patterns. And there’s No analysis of estimation error propagation to downstream causal matrix computation.

**W2. Marginal and inconsistent performance gains:** RMSE on METR-LA/PEMS08 is not SOTA. Improvements over STEMO/ESTGCN are incremental. 

**W3. Unjustified design choices**: There’s no justification about the method. For example, why construct dynamic propagation matrix by simply adding $A^F$ and ASCM with different scales(Eq. 8)?

**W4. Oversimplified controller**: Single-layer FC network may be insufficient for complex stopping decisions. There’s no proof on the greedy strategy. And RL reward functions are standard.

**W5. Questionable "large-scale" claim**: The authors claimed that they evaluated on four “large-scale dataset”. But to the best of my knowledge, LargeST[1] has 8,600+ nodes and 500,000+ frames, which is much larger than the dataset used in the manuscript.

**W6. Lack of Efficiency Study and Complexity Discussion.**

[1] Liu X, Xia Y, Liang Y, et al. Largest: A benchmark dataset for large-scale traffic forecasting[J]. Advances in Neural Information Processing Systems, 2023, 36: 75354-75371.

### Questions
Q1. How does the method handle nodes with multiple dominant frequencies or no clear dominant frequency? Does it select the strongest frequency or use all frequencies?

Q2. In Eq. (8), $A^F$ and ASCM have different dimensions (semantically and possibly numerically). Why is simple addition justified? This implies equal contributions, which seems arbitrary. Additionally, ASCM varies with $t$, shouldn't temporal decay be applied to earlier timesteps? Otherwise, t=0 and t=11 contribute equally to prediction at t=12, which is unreasonable.

Q3. Causality validation: Can you provide evidence that DTW-based similarity truly captures causality (not just correlation)? For example, synthetic experiments with known ground-truth causal graphs?

Q4. Computational cost: What is the runtime complexity of computing A_D at each timestep? How does it compare to baselines?

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
This paper proposes an adaptive early spatio-temporal forecasting model designed to deliver reliable predictions before the full observation window is available. It leverages long-term pattern similarity to support early forecasting, introduces an Asynchronous Spatio-temporal Causal Frame Pair mechanism to estimate pairwise time delays through causal relationships, and constructs a dynamic propagation matrix to filter out irrelevant signals. Experiments on four datasets demonstrate consistent improvements over baseline methods.

### Strengths
1. The paper's study on timeliness is valuable and has clear applications to crime prediction and traffic congestion forecasting. 

2. Modeling inter-node time delays using dominant frequency and phase shift is also an interesting and well-motivated approach.

3. Using a reinforcement-learning controller to determine whether each node has reached its optimal prediction time helps avoid misleading signals.

### Weaknesses
1. The time delay is estimated, but only one example is provided to show that the estimated value is close to the actual delay. A more thorough analysis of time-delay estimation accuracy would strengthen the paper. Since time delays likely vary across node pairs, it would be helpful to report accuracy metrics across multiple pairs and examine how estimation errors affect prediction performance or propagate through the model.

2. The experimental setup is not clearly described. Since the optimal prediction time may differ across nodes, the controller’s decision process for determining this per-node timing is unclear. It would be helpful to clarify the state how the model handles per-node heterogeneity.

3. The target prediction horizon is not clearly stated. The paper mentions an observation window of t=12, but it is unclear what the corresponding prediction window is. Additionally, the average stopping time per dataset is not reported, which is important for understanding the model’s early prediction behavior.

4. The data‑horizon notation is confusing. The use of 100%, 75%, 50%, and 25% would be clearer if mapped explicitly to observation lengths, such as t=12,9,6,3, along with an explanation of how these settings reflect early‑prediction behavior. Although the experiments show that the model performs well with limited observations, it remains unclear how the method determines the optimal prediction time under a fixed observation length.

5. Although the paper motivates crime and congestion forecasting, there are no experiments on crime or traffic-congestion datasets, which weakens the application claims.

6. The paper lacks case studies or visualization showing that the method truly filters unnecessary signals and adapts prediction times effectively.

### Questions
1. The time delay is estimated, but only one example is provided to show that the estimated value is close to the actual delay. A more thorough analysis of time-delay estimation accuracy would strengthen the paper. Since time delays likely vary across node pairs, it would be helpful to report accuracy metrics across multiple pairs and examine how estimation errors affect prediction performance or propagate through the model.

2. The experimental setup is not clearly described. Since the optimal prediction time may differ across nodes, the controller’s decision process for determining this per-node timing is unclear. It would be helpful to clarify the state how the model handles per-node heterogeneity.

3. The target prediction horizon is not clearly stated. The paper mentions an observation window of t=12, but it is unclear what the corresponding prediction window is. Additionally, the average stopping time per dataset is not reported, which is important for understanding the model’s early prediction behavior.

4. The data‑horizon notation is confusing. The use of 100%, 75%, 50%, and 25% would be clearer if mapped explicitly to observation lengths, such as t=12,9,6,3, along with an explanation of how these settings reflect early‑prediction behavior. Although the experiments show that the model performs well with limited observations, it remains unclear how the method determines the optimal prediction time under a fixed observation length.

5. Although the paper motivates crime and congestion forecasting, there are no experiments on crime or traffic-congestion datasets, which weakens the application claims.

6. The paper lacks case studies or visualization showing that the method truly filters unnecessary signals and adapts prediction times effectively.

### Soundness
2

### Presentation
2

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
This paper proposes an adaptive early spatio-temporal prediction model with a Dynamic Spreading Transmission Network (DSTN), aimed at addressing the trade-off between timeliness and accuracy in spatio-temporal forecasting tasks. DSTN captures causal dependencies between nodes by constructing a dynamic propagation matrix, identifies long-term patterns using frequency-domain similarity, and captures short-term causal relationships through asynchronous spatio-temporal causal frame pairs. Additionally, a reinforcement learning-based controller is employed to adaptively determine the optimal prediction time for each node. Experiments demonstrate that DSTN outperforms existing baseline methods on four large-scale real-world datasets.

### Strengths
1. The study of early spatio-temporal prediction is both interesting and practically meaningful.
2. The proposed model achieves superior performance compared to baseline models in early prediction tasks.

### Weaknesses
1. The paper lacks comparisons with a wide range of SOTA spatio-temporal baselines. Most of the general ST baselines used for comparison were proposed before 2020, which weakens the persuasiveness of the experimental results.
2. The early prediction task seems to be a subset of long-term forecasting, e.g., predicting data from 12:00 to 18:00 based on data from 00:00 to 06:00 is included in predicting from 06:00 to 18:00. Has the author explored or experimented with this dimension?
3. The paper equates a small DTW (Dynamic Time Warping) distance with strong causality, but correlation does not necessarily imply causation.

### Questions
Please refer to Weaknesses

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
5

### Summary
This paper aims to balance between prediction timeliness and accuracy is essential for spatio-temporal forecasting. The adaptive early spatio-temporal prediction model with a dynamic propagation matrix is proposed, which captures causal relationships between nodes to enhance prediction timeliness while maintaining accuracy. Experimental results on four real-world datasets demonstrate that the performance of the proposed model generally outperforms all baselines.

### Strengths
- This paper presents an adaptive early spatio-temporal prediction model with a dynamic propagation matrix, which is a trade-off solution between the timeliness and accuracy of spatio-temporal predictions.

- The proposed method is simple yet effective according to the experimental results.

### Weaknesses
- There is no evidence that the long-term characteristic is ignored in existing studies [1].

- Many recent works focused on the causality of spatio-temporal data [2, 3].

- The baseline methods used in the evaluation are relatively outdated. It would be better to compare the latest state-of-the-art methods, such as HimNet [4] and STPGNN [5].

[1] Foundation models for spatio-temporal data science: A tutorial and survey[C]//Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 2025: 6063-6073.

[2] Nuwadynamics: Discovering and updating in causal spatio-temporal modeling[C]//The Twelfth International Conference on Learning Representations. 2024.

[3] Ma J, Cui Z, Wang B, et al. Causal learning meet covariates: Empowering lightweight and effective nationwide air quality forecasting. IJCAI 2025.

[4] Dong Z, Jiang R, Gao H, et al. Heterogeneity-informed meta-parameter learning for spatiotemporal time series forecasting[C]//Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining. 2024: 631-641.

[5] Spatio-temporal pivotal graph neural networks for traffic flow forecasting[C]//Proceedings of the AAAI conference on artificial intelligence. 2024, 38(8): 8627-8635.

### Questions
1. You mention that existing studies overlook long-term temporal dependencies in the Introduction section. Could you please provide empirical evidence to support this observation? How does your method explicitly capture long-term dynamics compared to prior models?

2. Several recent works have investigated causal spatio-temporal modeling. Could you elaborate on how your model differs from existing causal-based frameworks?

3. The baseline methods in the current experiments appear relatively dated. Is it possible to compare with more recent state-of-the-art models?

4. Could you provide insight into the computational complexity of the proposed method?

### Soundness
2

### Presentation
3

### Contribution
2
