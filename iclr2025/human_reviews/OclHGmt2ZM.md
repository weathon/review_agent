## Human Reviewer 1

### Summary
The paper titled "CMamba: Channel Correlation Enhanced State Space Model for Multivariate Time Series Prediction" introduces a new model called CMamba, which builds upon the existing Mamba state space model and addresses the key limitation of capturing cross channel dependencies that are crucial for multivariate time series prediction (MTSF). The CMamba architecture integrates two main components: (1) the M-Mamba module for modeling temporal dependencies, and (2) the Global Data Dependency MLP (GDD-MLP) for capturing cross channel correlations. In addition, a channel mixing strategy was introduced to improve generalization ability and alleviate overfitting. The author validated their method through extensive experiments on seven benchmark datasets, demonstrating that CMamba outperforms state-of-the-art models in long-term prediction tasks.

### Strengths
Originality: M-Mamba is used for time-dependent analysis, while GDD-MLP is used for cross channel correlation, which creatively enhances the state space model and solves the limitations of existing MTSF methods.
Quality: This article presents a set of well executed experiments that validate the effectiveness of CMamba in multiple datasets and prediction tasks. The consistent and statistically significant results demonstrate the robustness of the proposed model.
Clarity: The model architecture and experimental setup are clearly described and accompanied by useful charts (as shown in Figure 3). Tables and charts are well organized for easy interpretation of results.
Meaning: CMamba solves an important problem in time series prediction - modeling cross channel dependencies - which has implications for many real-world applications such as traffic management and energy consumption prediction.

### Weaknesses
Long term dependency exploration: Although the GDD-MLP module has shown promising results in mid-term time series prediction, its effectiveness in handling very long-term dependencies (such as prediction ranges exceeding 720) has not been fully explored. Additional experiments on long-term prediction tasks can further validate the robustness of the model.
Generalization in different fields: This article mainly focuses on standard datasets such as electricity and transportation, but it is beneficial to test CMamba's generalization ability on other types of time series data (such as financial or biological datasets), which may have different time and cross channel characteristics.

### Questions
Has the model been tested on datasets outside the field explored in this article, such as financial or biological time series? Understanding the generalization level of CMamba to different types of multivariate time series would be valuable.
Can the author provide more insights into the computational efficiency of CMamba compared to Transformer based models such as iTransformer? A more detailed analysis of computational trade-offs may be helpful for practitioners considering large-scale deployment of the model.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The article presents two main contributions: M-Mamba and GDD-MLP+Channel Mixup. The former is a further exploration of the application of Mamba in time-series tasks, but in my opinion, there is a certain degree of controversy. The latter is a good solution to the problem of time-series channel dependence, with sufficient argumentation and experiments. The overall performance of the model is good, but it mainly comes from the latter contribution.

### Strengths
The overall writing logic of the article is clear, and the drawings are relatively clear. The results are presented in detail.
I think the combination of GDD-MLP+Channel mixup is a good contribution point. Moreover, applying it to other frameworks further proves its effectiveness, and an analysis of its efficiency is also conducted. However, on the contrary, as can be seen from Table 3, M-Mamba has no advantage compared to the state-of-the-art.

### Weaknesses
1. Line 53-55, There have been many studies on the application of past state space models in temporal dependency. Including: S4, SpaceTime, and the direct application of Mamba itself in temporal tasks has long existed, including the temporal dependency based on patches same with your M-mamba. I think it is worthy of research to consider cross-channel on the basis of temporal for mamba-based models. However, the problem raised here is not very accurate and the theme is not very clear.
2. Line 61-64, MLP is a multi-layer perceptron. In the Transformer, FFN is a fully connected mapping layer. It is somewhat inappropriate to say that it is part of the Transformer. I don't know what is the meaning of expressing FFN as a part of Attention? Because in iTransformer, the overall method of Attention is applied to channel dependence. The function of FFN in the attention mechanism has no direct connection with the dependence between channels. There is no relevant content in iTransformer. 
3. Line 84-91 This paragraph in the introduction does not meet the basic requirements of paper writing very well. The experimental results proving the effectiveness of the method should be placed in the experimental part.
4. Line 132-140, you also mentioned S4, and later in Table 12 also compared the effect with the mamba-based model. Still the above problem, there exist studies on SSM in temporal. It should still highlight that the model considers cross-channel on the basis of temporal, rather than generally saying that there is insufficient research on temporal.
5. For M-mamba (removing convolution and using FI), i am not sure that it is a general and effective improvement. I think there is controversy as a contribution. Mainly because the ablation experiment in section 5.2 is only carried out on one dataset, Weather, and the performance difference is not significant. In addition to the experiment, there is a lack of necessary explanations. For example, In Time-SSM Table 2 gives a detailed comparison with the patch Mamba (Mamba4TS).

### Questions
Line 64-68, The view put forward is not reasonable in my opinion. Why should MLP be an ideal structure? There is no basis. Can you provide relevant paper citation basis? Has there been any work in the past using MLP to practice between variables or similar work?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
5

---

## Human Reviewer 3

### Summary
In this work, the authors proposed model, CMamba, incorporates a modified Mamba (M-Mamba) module for temporal dependencies modeling, a global data dependent MLP (GDD-MLP) to effectively capture cross-channel dependencies, and a Channel Mix-up mechanism to mitigate over-fitting.

### Strengths
1.  The authors tailor the vanilla Mamba module for multivariate time series forecasting. A modified Mamba (M-Mamba) module is proposed for better cross-time dependencies modeling.
 2. The authors enable Mamba to capture multivariate correlations using the proposed global data dependent MLP (GDD-MLP) and Channel Mix-up. Coupled with the M-Mamba module, the proposed CMamba captures both cross-time and cross-channel dependencies to learn better representations for multivariate time series forecasting.

### Weaknesses
1. The description of the contribution seems to be incremental innovation, and the modeling of dependencies across time variables is the purpose, rather than a clear design motivation. In addition, different exogenous time series have positive or negative effects on the target series. How should this dependence be measured?
2. It can be seen from the experimental results in Table 1 that the performance improvement is weak. For example, on the ETTm1 dataset, the authors' method improved the MSE metrics by about 2% compared to the SOTA method (i.e., TimeMixer). If the authors claim that the cross-time dependence should produce significant performance improvement, the current experimental results are contrary to the intuition. After all, the dependencies between multiple time variables in electric time series data are much clearer.
3. Especially in the ablation experiments of Table 2 with standard Mamba, we can find that the performance improvement is close to the test error. In terms of the MSE metric, the standard Mamba is 0.240 and the proposed method is 0.237, which is only 1.25% improvement.
4. The reviewer is also concerned about the extent to which such modifications impose computational overhead on the several datasets. Especially the GDD-MLP module and CMamba-encoder module, is it because these complex components increase the complexity of the proposed prediction model and thus improve the prediction effect? Therefore, authors are expected to provide training and inference time on multiple time series datasets.

### Questions
Please see weaknesses!

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
CMamba is a new state-space model for multivariate time series prediction. It combines the Mamba model with global data-dependent MLP and channel mixing strategies to enhance capture across time series and cross-channel dependencies. Experiments on seven real-world data sets showed that CMamba achieved significant improvements in predictive performance.

### Strengths
The main advantage of this paper is that uses a Mamba-base model and channel-independent and mixed strategies to improve the performance of time series prediction.At the same time, the article is well written and clearly expressed

### Weaknesses
CMamba model mainly focuses on the prediction of endogenous variables of multivariable time series but does not involve the influence of exogenous variables. On large-scale channel data sets, such as weather and power data, the model has limited improvement in the MAE index, especially on traffic data sets. This may be due to the strong periodicity of traffic data covering up the physical connection between channels. Therefore, the inclusion of external variables and the use of prior knowledge of the relationships between channels may further improve the prediction accuracy.

### Questions
1. What are the specific advantages of the Mamba-based model compared to the MLP-based model?
2. Some of the baseline models MAE/MSE appear to differ from those reported in the original article(ModernTCN)

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3