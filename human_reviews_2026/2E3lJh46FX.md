# Robust Time Series Forecasting via Basis-Aligned Sampling in Decycled Residual Space

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Time series forecasting is crucial in domains such as finance, energy, and traffic, yet real-world data are often contaminated by anomalies and noise.
In this work, we first identify a fundamental limitation of existing approaches—their excessive reliance on specific input points, particularly the most recent observation—which makes them highly susceptible to point-wise perturbations and undermines prediction reliability.
To further address this challenge, we propose RESAM, a novel approach for robust time series forecasting that effectively mitigates the impact of point-wise perturbations while maintaining high overall forecasting accuracy. 
RESAM utilizes a basis-aligned randomized sampling strategy to comprehensively exploit the global context and achieve a unified representation for irregularly sampled sequences.
Moreover, RESAM employs a learnable periodicity extraction module with a two-stage training protocol to enhance the accuracy and robustness of both periodicity and residual learning.
Comprehensive evaluations on eight benchmark datasets show that RESAM achieves competitive forecasting accuracy and significantly surpasses state-of-the-art models in robustness to point-wise perturbations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the issue that recent observations in time series have a significant impact on forecasting performance. To mitigate this, the paper proposes a new robust time series forecasting method, RESAM, aimed at improving prediction reliability. Specifically, RESAM uses a basis-aligned random sampling strategy to explore global context information while achieving a unified representation for irregularly sampled sequences. Additionally, RESAM incorporates a learnable periodicity extraction module to capture periodic patterns. Evaluations on multiple datasets show that it outperforms baseline models in terms of performance.

### Strengths
1. The paper clearly identifies the issue of the last points.
2. The architecture design is simple and efficient.

### Weaknesses
1. The experimental baseline models are not cutting-edge enough. It is recommended to include more recent, high-performing baseline methods. Additionally, the effectiveness of the method is not sufficiently demonstrated in the main experiments.
2. The paper seems more like an engineering-driven modular approach for performance improvement rather than an in-depth exploration of the issue, which significantly weakens its contribution.
3. The motivation behind the paper seems illogical. It is reasonable that the last points in a time series are important and that learning such biases is valid. It is also intuitive that perturbing these points would lead to a significant drop in performance.
4. The related work section is too broad and lacks adequate discussion of relevant research. In my knowledge, the importance of more recent observations is a widely accepted phenomenon, especially in fields like finance where "market efficiency" often dictates that recent data hold higher significance.

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
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method that leverages a basis-aligned randomized sampling strategy and a learnable periodicity extraction module with a two-stage training protocol to forecast time series robustly.

### Strengths
1. Easy to follow and understand.
2. Well-motivated

### Weaknesses
1. There is a lack of a sufficient literature review. In the introduction part, the authors should discuss on the existing works and the reason why they cannot solve the challenges.
2. The baselines are not comprehensive enough to claim a SOTA performance. More baselines, such as iTransformer, Time-LLM, and GPT4TS, should be compared.
3. There is no experiment on short-term forecasting, limiting the application scope.
4. The overall performance increase is not significant and robust. It seems that only on ETTm1 and ETTm2, the proposed method shows clear advantages. Besides, on robustness evaluation, RESAM is worse than Crossformer and TimesNet according to the MSE increase.

### Questions
See Weakness.

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
This paper proposes a randomized sampling strategy to exploit global context to mitigate the impact of point-wise perturbations for time series forecasting. Experiments on 8 datasets show the robustness to point-wise perturbations.

### Strengths
1. This motivation is easy to follow.
2. Visualization and showcases.
3. Code are provided for reproducibility.
4. Randomized sampling strategy to exploit global context to mitigate the impact of point-wise perturbations is very interesting.

### Weaknesses
1. The claimed contributions and novelty are exaggerated. There are existing works on anomalies or perturbations on time series forecasting, such as [1-2]. There are also many works on basis / decycle or decomposition, such as [3-4]. And the proposed Learnable Periodicity Extraction module seems to be from [3], or I cannot find how to learn periodic cycle $W$ from the presentation in Section 3.2.

[1] RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies. ICLR 2024.

[2] Weakly Guided Adaptation for Robust Time Series Forecasting. Proc. VLDB Endow. 17(4): 766-779 (2023).

[3] Revitalizing multivariate time series forecasting: Learnable decomposition with inter-series dependencies and intra-series variations modeling. ICML 2024.

[4] CycleNet Enhancing time series forecasting through modeling periodic patterns, Lin et al., NIPS 2024.

2. Key experimental comparison for baselines [1-2] is missing. The author should also compare state-of-the-art baselines such as [3].

3. How to evaluate the improvements by identifying the real perturbations in original datasets? Or the improvements are just gained by randomized sampling strategy which enhance generalization ability?

### Questions
see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents RESAM, a method for robust time series forecasting designed to mitigate vulnerability to point-wise perturbations, particularly on the most recent input points. The approach combines Basis-Aligned Randomized Sampling (BARS) to represent sampled time points via trigonometric basis functions and a Learnable Periodicity Extraction (LPE) module that learns the periodic component independently before sampling. A two-stage training procedure further stabilizes dual-space learning. Extensive experiments across eight benchmarks demonstrate improved perturbation robustness and competitive forecasting accuracy relative to prominent baselines.

### Strengths
1. The paper convincingly identifies and empirically demonstrates a core vulnerability in modern time series forecasting architectures: an over-reliance on recent input points, as shown in Figure 1 and Figure 3. This motivates the necessity for robustness-oriented model design.
2. A comprehensive set of ablation studies (Figure 8) shows that both LPE and the BARS sampling improve robustness and accuracy, offering deeper insight into the contribution of each component.
3. The paper is clearly and lucidly written, making the methodology easy to follow. The authors' provision of source code further enhances the credibility and reproducibility of their work.

### Weaknesses
1. On Novelty and Contribution: The Learnable Periodicity Extraction (LPE) module, a key component of the proposed framework, appears to be heavily inspired by or substantially similar to prior work, particularly CycleNet. This significant overlap raises concerns about the novelty and the incremental contribution of this paper. While integrating existing ideas is valid, the reliance on this established architecture may limit the perceived originality of the overall work.
2. In Stage 1 of training, the authors replace the basis-aligned sampling module and MLP backbone with a simple linear layer. However, the ablation study does not include a baseline that only uses this Stage 1 architecture for final prediction (i.e., without proceeding to Stage 2). 
3. On Overall Performance: From a practical standpoint, the empirical results suggest that RESAM does not consistently achieve state-of-the-art performance across all benchmarks. Even on the datasets where RESAM performs favorably, the margin of improvement over existing baselines is often limited. This raises questions about the practical significance of the proposed method in standard, noise-free forecasting scenarios and whether the added complexity is justified by the marginal gains.
4. On the Perturbation Model and Scope of Conclusions: The paper's core motivation hinges on the finding that existing models are more sensitive to a last-point perturbation than a random-point one. However, the realism of this specific perturbation model is a significant concern; it is difficult to identify practical, real-world scenarios where noise or corruption would systematically affect only the single most recent data point. Furthermore, the proposed methodology, especially the basis-aligned sampling, seems inherently capable of addressing other non-point-wise disturbances, such as block-wise corruption or multi-point noise. The authors' failure to discuss or evaluate their model against these more general and arguably more realistic perturbation types restricts the applicability and scope of their conclusions. This leaves the model's robustness in a broader range of practical scenarios underexplored.
5. My concerns about the limited generalizability of the conclusions are further amplified by the appendix, where the sensitivity analysis for the perturbation ratio α is conducted exclusively on the ETT series of datasets. To strengthen the paper's claims, I recommend that the authors extend this important ablation study to a more diverse range of datasets.

### Questions
1. **On the Necessity of Stage 2 Training**: Fundamentally, fitting basis coefficients is a linear task. It is therefore plausible that replacing the basis-fitting module with a linear layer, as done in Stage 1, could theoretically achieve comparable results. My primary concern is whether the Stage 2 training genuinely provides a significant performance improvement over the foundation established in Stage 1. To address this, could the authors supplement the ablation study by reporting the performance of the Stage 1 model (trained and evaluated on its own) across the various datasets? This would clarify the true value added by Stage 2.
2. **On the Complexity of the Two-Stage Protocol**: The introduction of a linear or MLP layer in Stage 1, which is later replaced by the full RESAM architecture in Stage 2, appears to significantly increase the model's overall parameter count and training complexity. Is this added complexity truly necessary? Could the authors consider a simpler approach for Stage 1, where only a periodic pattern matrix is trained, and the forecast is generated by simply extrapolating this learned cycle? Such a method would likely be more computationally efficient and would help isolate the benefits of the periodicity learning itself.
3. **On the Realism of the Last-Point Perturbation Model**: The paper demonstrates that RESAM is effective at mitigating last-point perturbations, but its efficacy against other point-wise disturbances seems limited by design. Could the authors comment on the practical relevance of this specific last-point perturbation model? In which real-world scenarios does this type of isolated, terminal-point corruption actually occur? Justifying the focus on this particular failure mode is crucial for the paper's practical impact.
4. **On the Generalizability to Other Perturbation Types**: Could RESAM be extrapolated to handle other common disturbance patterns, such as block-wise corruption, random multi-point noise, or missing value imputation? The current analysis is narrowly focused on a single type of perturbation. An evaluation of RESAM's performance against these more varied and arguably more common types of data corruption would provide a much more comprehensive assessment of its robustness and practical utility.

I believe this paper has potential, but the concerns raised above are significant. Should the authors provide a convincing response that effectively addresses these weaknesses and clarifies the open questions, I would be happy to reconsider my evaluation and increase my score accordingly.

### Soundness
3

### Presentation
3

### Contribution
2
