# S4M: S4 for multivariate time series forecasting with Missing values

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 5

## Abstract
Multivariate time series data play a pivotal role in a wide range of real-world applications, such as finance, healthcare, and meteorology, where accurate forecasting is critical for informed decision-making and proactive interventions. However, the presence of block missing data introduces significant challenges, often compromising the performance of predictive models. Traditional two-step approaches, which first impute missing values and then perform forecasting, are prone to error accumulation, particularly in complex multivariate settings characterized by high missing ratios and intricate dependency structures. In this work, we introduce S4M, an end-to-end time series forecasting framework that seamlessly integrates missing data handling into the Structured State Space Sequence (S4) model architecture. Unlike conventional methods that treat imputation as a separate preprocessing step, S4M leverages the latent space of S4 models to directly recognize and represent missing data patterns, thereby more effectively capturing the underlying temporal and multivariate dependencies. Our framework comprises two key components: the Adaptive Temporal Prototype Mapper (ATPM) and the Missing-Aware Dual Stream S4 (MDS-S4). The ATPM employs a prototype bank to derive robust and informative representations from historical data patterns, while the MDS-S4 processes these representations alongside missingness masks as dual input streams to enable accurate forecasting. Through extensive empirical evaluations on diverse real-world datasets, we demonstrate that S4M consistently achieves state-of-the-art performance. These results underscore the efficacy of our integrated approach in handling missing data, showcasing its robustness and superiority over traditional imputation-based methods. Our findings highlight the potential of S4M to advance reliable time series forecasting in practical applications, offering a promising direction for future research and deployment. Code is available at https://github.com/WINTERWEEL/S4M.git.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper explicitly models missing patterns within the Structured State Space Sequence (S4) architecture, developing two key modules: the Adaptive Temporal Prototype Mapper and the Missing-Aware Dual Stream. The experiments demonstrate that S4M achieves state-of-the-art performance.

### Strengths
1. This paper explores the integration of State Space Models with missing patterns for long-term time series forecasting.

2. The proposed adaptive temporal prototype mapper and missing aware dual stream S4 modules effectively capture rich historical patterns and learn robust representations.

3. Experimental results illustrate that the proposed model achieves state-of-the-art performance in handling missing data.

### Weaknesses
1. The authors should further explain the motivation for introducing $\mathbf{\bar{E}}E_m(m_t;\theta_m)$ to the SSM model in Equation 4.

2. This paper lacks a discussion between the S4M and existing methods designed for handling missing values, which diminishes the significance of the proposed model.

3. The settings of hyperparameters $K_1$, $K_2$, $\tau_1$, and $\tau_2$ are emprical. The authors should provide guidance on how to set these hyperparameters across different datasets with varying characteristics.

### Questions
1. The proposed model struggles when the input length is shorter than the output length. In addition, it is general to fix the lookback length and adapt to various prediction lengths [1]. Thus, it would strengthen this paper to add experimental results with various horizons.

2. Baselines such as Transformer and Autoformer are designed for complete data. The authors should clarify how these models can be adapted for partially observed data. Besides, the experimental analysis should include state-of-the-art baselines specifically designed for long-term forecasting tasks, such as iTransformer [1], CARD [2], and Crossformer [3].

[1]  Liu Y, Hu T, Zhang H, et al. itransformer: Inverted transformers are effective for time series forecasting[C]. The eleventh international conference on learning representations. 2023.

[2]  Wang X, Zhou T, Wen Q, et al. CARD: Channel aligned robust blend transformer for time series forecasting[C]. The Twelfth International Conference on Learning Representations. 2024.

[3]  Zhang Y, Yan J. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting[C]. The eleventh international conference on learning representations. 2023.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose S4M. S4M is an adaption of S4 which can handle missing values by:
- Using prototype clusters, look-back information and an encoder to find representations also for time-points where values are missing
- By explicitly also incorporating the masking matrix M into the S4-Layers.
 
 They evaluate S4M on the standard data for regular time-series forecasting when some data is hidden and show
 that i) S4M is competitive or often outperforms other S4 and Missing-Value approaches.

### Strengths
+ The idea of the prototype bank is very compelling and thoughtful, I like it a lot.
+ The presentation is very good. The paper is written in a manner which makes it comfortable to follow.
  Especially having an algorithm for each of the crucial parts helped me a lot.
+ The results are not looking like fundamental break-throughs, but they are very promising for such a novel approach and there are a lot of ablations studies/hyperparameter experiments.

### Weaknesses
The two main weaknesses I identified:

- My largest critique point is, that the authors are not comparing at all with recent results from the "Irregular Sampled Time-Series wit Missing Values" Literature. There is a plentitude of recent works solving irregular time-series forecasting in an end-to-end manner via ODEs, modelling latent dynamics or graph modelling.[1-5]. Furthermore, these papers provide a set of standard datasets for time-series forecasting with missing values, thus no need to synthetically make the normal regular datasets irregular.

- There are a lot of standard methods missing, where one could do simply linear interpolation to use them in the experiments on which S4 is tested. For example, this work is not referring to important forecasting works like PatchTST or iTransformer at all. The results of S4M in Table 1, are way worse then the results of PatchTST (see Table at https://github.com/yuqinie98/PatchTST?tab=readme-ov-file) that it may be the case that PatchTST with 0.06 missing values indeed outperforms S4M.

[1] De Brouwer, E., Simm, J., Arany, A., & Moreau, Y. (2019). GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series. Advances in neural information processing systems, 32.

[2] Yalavarthi, Vijaya Krishna, et al. "GraFITi: Graphs for Forecasting Irregularly Sampled Time Series." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.

[3] Schirmer, Mona, et al. "Modeling irregular time series with continuous recurrent units." International conference on machine learning. PMLR, 2022.

[4] Biloš, Marin, et al. "Neural flows: Efficient alternative to neural ODEs." Advances in neural information processing systems 34 (2021): 21325-21337.

[5] Klötergens, Christian, et al. "Functional Latent Dynamics for Irregularly Sampled Time Series Forecasting." Joint European Conference on Machine Learning and Knowledge Discovery in Databases.

## My Current Rating
I do really like the idea and think that it has potential, even beyond S4 models for irregular time-series forecasting. However, for a top conference like ICLR, the amount of missing comparison to important related work is too high for recommending acceptance.

### Questions
Additionally to the critique points mentioned above, I have the following comments/question:

- Have you tested S4M when there are no missing values at all? I would be curious whether your prototype bank is also useful if no values are missing.
- Table 3: Do I understand correctly, that you are comparing: Prototyping Bank + Masking (i.e. having m_t in (4)) against only having the prototype bank? I would also like to see Only Masking, i.e. having the mask m_t in (4) but no prototype bank, i.e. replacing o_t with X_t. Is your prototype-bank really needed for irregular time-series forecasting?

### Soundness
2

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
The paper introduces S4M, an extension of the S4 framework to multivariate time series forecasting with missing values. It combines a prototype-based representation learning module (ATPM) with a dual-stream S4 architecture (MDS-S4) to handle missing values directly rather than through preprocessing. The method is evaluated on four datasets under various missing data scenarios.

### Strengths
The paper addresses a practical and relevant problem. Real-world data often contains missing values (or mis-recorded values) which makes developing principled methods to handle them in forecasting models a worthwhile endeavor. The paper is well written and relatively easy to follow. The empirical evaluation does employ a set of strong baselines for comparison. The use of a "prototype bank" for backfilling is new to this reviewer and represents an interesting practical way to tackle missing values.

### Weaknesses
The paper introduces a few new key components, notably the prototype bank and the MDS-S4 architecture which, while well described, are only subjected to limited analysis and theoretical justification. For instance, complexity analysis is missing and ablation studies are partial. Some architectural choices seem arbitrary. The datasets chosen in the empirical evaluation (Traffic, Electricity, ETTh1, Weather) are all fairly simple datasets. Given that there are many more publicly available time-series evaluation datasets, I would like to see a more comprehensive evaluation.

### Questions
What is the computational complexity of ATPM vs traditional approaches?
Can you provide theoretical justification for the prototype bank design?
How sensitive is performance to prototype bank initialization?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents S4M, an innovative end-to-end framework consisting of the Adaptive Temporal Prototype Mapper (ATPM) and the Missing-Aware Dual Stream S4 (MDS-S4) for multivariate time series forecasting that addresses the challenge of missing data.

### Strengths
1. The proposed S4M model is innovative, integrating missing data handling within the model architecture.

### Weaknesses
1. Although the results are promising, the authors did not provide their source code which results in very low reproducibility;
2. Computational efficiency comparison is missing;

### Questions
1. What are the computational costs and scalability of the S4M model, especially when dealing with large-scale multivariate time series data with high missing ratios? How does it compare to the baseline models in terms of training and inference time?
2. How does the dual stream processing impact the model's ability to capture temporal dependencies?
3. Can the authors confirm whether they implemented these baseline methods using official code or leveraged existing unified Python libraries, such as the Time-Series-Library [1] or PyPOTS [2]? It's important to note that data processing varies significantly among different imputation algorithms. Utilizing unified interfaces could help ensure that the experimental comparisons are conducted fairly. 

### References
[1] https://github.com/thuml/Time-Series-Library

[2] Wenjie Du. PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series. In KDD MiLeTS Workshop, 2023. https://github.com/WenjieDu/PyPOTS

### Soundness
2

### Presentation
3

### Contribution
2
