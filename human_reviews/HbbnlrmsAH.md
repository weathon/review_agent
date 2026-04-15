# Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts

- Decision: Reject
- Scores: 3, 6, 5, 8

## Abstract
Time series foundation models have demonstrated impressive performance as zero-shot forecasters, i.e. tackling a wide variety of downstream forecasting tasks without explicit task-specific training. However, achieving effectively unified training on time series remains an open challenge. Existing approaches introduce some level of model specialization to account for the highly heterogeneous nature of time series data. For instance, Moirai pursues unified training by employing multiple input/output projection layers, each tailored to handle time series at a specific frequency. Similarly, TimesFM maintains a frequency embedding dictionary for this purpose. We identify two major drawbacks to this human-imposed frequency-level model specialization: (1) Frequency is not a reliable indicator of the underlying patterns in time series. For example, time series with different frequencies can display similar patterns, while those with the same frequency may exhibit varied patterns. (2) Non-stationarity is an inherent property of real-world time series, leading to varied distributions even within a short context window of a single time series. Frequency-level specialization is too coarse-grained to capture this level of diversity. To address these limitations, this paper introduces Moirai-MoE, using a single input/output projection layer while delegating the modeling of diverse time series patterns to the sparse mixture of experts (MoE) within Transformers. With these designs, Moirai-MoE reduces reliance on human-defined heuristics and enables automatic token-level specialization. Extensive experiments on 39 datasets demonstrate the superiority of Moirai-MoE over existing foundation models in both in-distribution and zero-shot scenarios. Furthermore, this study conducts comprehensive model analyses to explore the inner workings of time series MoE foundation models and provides valuable insights for future research.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces Moirai-MoE, a time series model that incorporates sparse Mixture of Experts (MoE) for forecasting applications. The authors adapt an existing Transformer-based forecasting model, namely Moirai, by replacing its feed-forward networks (FFNs) with well established MoE layers. Evaluations in both in-distribution and zero-shot settings indicate that the use of MoE layers is beneficial across forecasting applications.

### Strengths
1. The paper is well structured.
2. The authors conduct extensive forecasting experiments, including 29 in-distribution and 10 zero-shot applications, to evaluate their method.

### Weaknesses
1. Time series models utilising Mixture of Experts (MoE) have been well studied since the 1990s [1] up until now [2,3,4]. As the work under review combines well-established MoE approaches with existing time series models for a specific task, i.e. forecasting, it offers a limited technical contribution to the field of time series analysis. 

2. The related work section is incomplete, as the authors fail to discuss existing time series model that utilise MoE. Previous works stated above [1,2,3,4] and further works should be included by the authors to provide a representative overview of the current literature. 

3. The methodology section of the paper is poorly elaborated, e.g.:
   - None of the equations (1) to (7) state the dimension of the input and the output variables. 
   - While it may refer to a single Transformer layer, the variable $l$ is introduced in equation (1) without further explanation.  
   - In equation (5), the load balancing loss is multiplied by $M$ without further explanation, the indicator function $\mathbb{1}$ is introduced without further explanation, and the argmax operation is performed without defining over which set. 
   - It is unclear whether the load balancing loss $L_{load}$ is defined for each layer $l \in L$. 
   - It is unclear how the load balancing loss $L_{load}$ is combined with the training objective $L_{pred}$ provided in equation (7). 
   - It is unclear how the cluster centroids $C \in \mathbb{R}^{M \times D}$ are derived from the self-attention output representations $\tilde{x}^{l-1}$.

4. The experiments should include existing time series models that use MoE, e.g. [3,4], to enable a fair comparison of the proposed MoE approach. Furthermore, the results should be reported over multiple seeds to ensure robustness. Well established metrics such as the mean squared error (MSE) or mean absolute error (MAE) should be reported to enable fair comparison with methods that are not included as baselines. 

5. The authors do not elaborate on the limitations of their work. 

6. The authors do not provide code to support reproducibility.

[1] Zeevi et al. "Time series prediction using mixtures of experts." NeurIPS (1996). 

[2] Yuksel et al. "Twenty years of mixture of experts." IEEE transactions on neural networks and learning systems (2012).

[3] Ni et al. "Mixture-of-Linear-Experts for Long-term Time Series Forecasting." AISTATS (2024).

[4] Shi et al. "Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts." arXiv (2024).

### Questions
1. How does the choice of pre-trained model influence the resulting cluster centroids $C \in \mathbb{R}^{M \times D}$ and, consequently, the downstream performance? To this end, have the authors investigated any other pre-trained model besides Moirai?

2. In Table 1, rather than presenting the number of activated paramters, it would be more insightful to show how the downstream performance of Moirai-MoE evolves during training, e.g. with increasing FLOPs and GPU hours, compared to Moirai. Does Moirai-MoE match or even outperform Moirai with less compute? If so, at what point during training do the advantages of the MoE layers become evident?

3. The trends depicted in Figure 4 indicate that it might be worth increasing the training steps ($>$ 100k) and the number of experts ($>$ 32). To this end, could the authors extend their analysis to see whether downstream performance can be further improved?

4. In Figure 5, it seems that experts of the Covid Daily Deaths are a subset of the experts allocated for the Traffic Hourly data. Could the authors explain whether this is based on data similarity or any other phenomenon?

5. In Figure 6, the expert allocation appears very similar across frequencies and is notably sparse, suggesting that some experts might be redundant. How does this align with Figure 4, which indicates that downstream performance improves as the number of experts increases?

6. Furthermore, is it necessary to replace all FFNs with MoE layers? Existing work on MoE in natural language processing [5] and computer vision [6] show that replacing only certain layers has little impact on downstream performance, while saving computation time due to less communication overhead. 

7. Have the authors analysed what the experts learn? Do some experts focus on local patterns while others focus on global patterns? Do some experts analyse low-frequency components while others analyse high-frequency components? The authors attempt to explain expert allocation with reference to Figure 7, however, the figure is poorly structured and does not support clear conclusions.

8. The authors provide an efficiency analysis focused on the inference cost of predicting a single token, i.e. a forecasting horizon of 1 token. While this does not allow for a fair evaluation of real-world applications, the authors might rather investigate how inference cost scales with an increasing forecasting horizon.

9. Regarding Figure 2 and Table 3, do the authors plan to provide full results for a fair comparison?

[5] Lepikhin et al. "GShard: Scaling giant models with conditional computation and automatic sharding." ICLR (2021).

[6] Riquelme et al. "Scaling vision with sparse mixture of experts." NeurIPS (2021).

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
5

### Summary
In response to the variability of temporal patterns in pre-trained time series datasets, this paper proposes the MOIRAI-MOE. It leverages token embeddings from existing time series pre-trained models to derive cluster centers, resulting in a more effective gating function. Extensive experiments show that MOIRAI-MOE achieves enhanced predictive accuracy and overall superior performance.

### Strengths
- The article proposes the application of MOE to time series foundational models, achieving token-level specialization to enable the model to handle temporal data with significant heterogeneity in temporal patterns.
- Through extensive experiments, the paper demonstrates the effectiveness of this method, achieving impressive results with high inference efficiency.
- The paper provides insights through experimental analysis, such as how heterogeneous experts reflect the periodic pattern information of the time series data.

### Weaknesses
- **Methodology Limitations:** First, MOE is a well-known technique in LLMs, so its direct application results in limited novelty for the article. Furthermore, the gating functions used in the MOE depend on the temporal embeddings from pre-trained models, which constrains the effectiveness of this approach.
- **Experimental Setup:**
    - The article lacks a detailed description of the experimental setup, including input and output lengths, which significantly impact the results. Additionally, it does not use established evaluation metrics (e.g., MAE, MSE) that correspond to previous mainstream studies. This omission makes it difficult for reviewers to intuitively compare the results with earlier time series prediction works, thereby reducing the credibility of the model's effectiveness.
    - Furthermore, the paper does not include recent significant works in time series foundational models, such as Timer[1], Moment[2], TTM[3], and Time-MOE[4].
- In Appendix Section B.1, the ablation study on patch size reveals that it significantly affects the model's performance. This raises the question of whether multi-patch projection is indeed an effective strategy, suggesting that MOE may not adequately address this issue, which contradicts the article's motivation.

**References**

[1] Liu, Y., Zhang, H., Li, C., Huang, X., Wang, J., & Long, M. (2024). Timer: Transformers for time series analysis at scale. arXiv preprint arXiv:2402.02368.

[2] Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). Moment: A family of open time-series foundation models. arXiv preprint arXiv:2402.03885.

[3] Ekambaram, V., Jati, A., Nguyen, N. H., Dayama, P., Reddy, C., Gifford, W. M., & Kalagnanam, J. (2024). TTMs: Fast Multi-level Tiny Time Mixers for Improved Zero-shot and Few-shot Forecasting of Multivariate Time Series. arXiv preprint arXiv:2401.03955.

[4] Shi, X., Wang, S., Nie, Y., Li, D., Ye, Z., Wen, Q., & Jin, M. (2024). Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. arXiv preprint arXiv:2409.16040.

### Questions
- Could the authors provide detailed experimental setups, including the input and prediction lengths for each model across the datasets, as well as a comparison table with previous mainstream time series prediction works? Specifically, the prediction lengths should be (96, 192, 336, 720) to facilitate intuitive comparisons for reviewers.
- The authors need to include recent works on time series foundational models to better substantiate the effectiveness of your model.
- Could the authors explain why patch size still significantly affects the model, and whether multi-patch projection remains necessary under this phenomenon?
- Could the authors provide insight into whether the token clusters generated by different time series foundational models affect the performance of MOIRAI-MOE?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces MOIRAI-MOE, a time series foundation model that overcomes limitations of frequency-based specialization by using a sparse Mixture of Experts (MoE) architecture within Transformers, allowing token-level, data-driven specialization. A novel gating function based on cluster centroids improves expert allocation accuracy, reducing reliance on human heuristics. MOIRAI-MOE outperforms existing models across the 39 testing datasets, including its predecessor MOIRAI. It generalizes effectively across varied time series patterns, with shallow layers capturing short-term variability and deeper layers learning more stable, abstract patterns.

### Strengths
1. MOIRAI-MOE’s mixture-of-experts design for time series reduces reliance on frequency-based specialization by using sparse experts for flexible, data-driven adaptation.

2. This work proposed an automatic, token-level specialization that allows the model to handle diverse time series patterns, enhancing flexibility and robustness.

3. This work introduced a gating function based on pre-trained cluster centroids that can potentially improve expert selection accuracy.

### Weaknesses
Writing:

1. The token clustering mechanism used as a gating function, intended as a key differentiator from existing work, is only briefly described in about ten lines. Please consider spending more text on the advantages of the proposed gating function.

Experimentation: 

2. Section 4.3 does not sufficiently demonstrate the irreplaceability of token clusters in the gating function.

3. In Figure 6, from layer 4 onward, different tokens tend to route to the same experts (three experts in layer 4 and two in layer 6), suggesting that the model may have more experts than necessary for practical use.

4. If MOIRAI-MOE is indeed a promising architecture, it should demonstrate lower MSE as its parameter count scales up. However, Table 2 lacks sufficient evidence for this, as it only includes the MoE-Base model with 86M/935M parameters.

Reproducibility: 

5. The team has not open-sourced the code for evaluation, impacting the reproducibility of this work.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Moirai-MoE, a method for unified time series forecasting, which addresses challenges of the human-imposed frequency-level model specialization for time series foundation model. It utilizes a single input/output projection layer while delegating the modeling of diverse time series patterns to the sparse mixture of experts (MoE) within Transformers, which enables automatic token-level specialization of time series foundation model.

### Strengths
1. The manuscript demonstrates a high level of completeness.
2. It provides an effective large-scale framework for advancing large models in time series analysis.
3. The empirical evaluation is comprehensive and promising results are shown.

### Weaknesses
1. From the table2, the CRPS is used for probabilistic forecasting of time series models. The paper also does not include some probabilistic metrics, such as QICE and PICP, which could provide a more comprehensive understanding of the model's probabilistic accuracy. Including metrics like QICE and PICP [1,2] would provide a more comprehensive understanding of the model's accuracy in distribution estimation.
2. In Equation 6, the author mentions using the output of Moirai as cluster centroids applied in Moirai-MoE. Since Moirai itself is a foundational time series model based on a specific frequency, wouldn’t this approach introduce Moirai's frequency preferences into Moirai-MoE? Following this logic, wouldn’t it deviate from the objective stated in Moirai-MoE to avoid human-imposed frequency-level specialization?

[1] Han, X., Zheng, H., & Zhou, M. (2022). Card: Classification and regression diffusion models. Advances in Neural Information Processing Systems, 35, 18100-18115.

[2] Li, Y., Chen, W., Hu, X., Chen, B., & Zhou, M. (2023, October). Transformer-Modulated Diffusion Models for Probabilistic Multivariate Time Series Forecasting. In The Twelfth International Conference on Learning Representations.

### Questions
Please refer to Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
