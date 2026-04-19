# Transformers versus LSTMs for electronic trading

- Decision: Reject
- Scores: 3, 3, 3, 3, 5, 3

## Abstract
The rapid advancement of artificial intelligence has seen widespread application of long short-term memory (LSTM), a type of recurrent neural network (RNN), in time series forecasting. Despite the success of Transformers in natural language processing (NLP), which prompted interest in their efficacy for time series prediction, their application in financial time series forecasting is less explored compared to the dominant LSTM models. This study investigates whether Transformer-based models can outperform LSTMs in financial time series forecasting. It involves a comparative analysis of various LSTM-based and Transformer-based models on multiple financial prediction tasks using high-frequency limit order book data. A novel LSTM-based model named DLSTM is introduced alongside a newly designed Transformer-based model tailored for financial predictions. The findings indicate that Transformer-based models exhibit only a marginal advantage in predicting absolute price sequences, whereas LSTM-based models demonstrate superior and more consistent performance in predicting differential sequences such as price differences and movements.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This research examines the performance differences between Transformer-based models and LSTMs across three cryptocurrency limit order book data prediction tasks. It also introduces DLSTM, a LSTM-based model, and a Transformer-based model redesigned for financial forecasting.

### Strengths
(1) This research presents findings that compare the performance of two types of models.
(2) It successfully highlights the weaknesses in current measurement metrics.
(3) Interesting task definition.

### Weaknesses
1.Baseline Selection Rationale: The paper does not clearly explain why specific Transformer and LSTM variants, such as Autoformer and FEDformer, were chosen in the comparison. It remains unclear if these variants have unique advantages for financial time series forecasting. Providing additional theoretical support or rationale for model selection would enhance the scientific basis of this choice.

2.Data Risk: The study only tests on a single asset (BTC-USDT), lacking a broader dataset. This limited scope may mean the model’s performance does not generalize well to other financial data. Testing on a single asset is insufficient to comprehensively assess the model’s generalizability.

3.Lack of Experimental Details: The paper lacks adequate details on the experimental setup, especially regarding hyperparameter settings and baseline model architectures. This omission makes replication challenging and affects the reliability of the results. Sufficient information is not provided to ensure a fair comparison among baseline models.

4.Unclear Result Interpretation: The paper does not adequately explain the significant differences in performance between experiments with and without transaction costs. Lacking theoretical support or data analysis, it's hard for me to understand the causes behind these variations under different settings.

5.Limited Community Contribution: Time series decomposition, used in this study, appears to be a common approach, closely resembling classical time series decomposition methods. It is unclear how this study provides any specific advantage over the standard decomposition methods.

6.Although the paper points out shortcomings in MSE and MAE metrics, it fails to propose a robust method to address these deficiencies.

7.Some capitalization inconsistencies, eg. in line 034  Self-attention mechanism.

### Questions
1.Given the limited dataset used and the lack of detailed experimental information (settings of baselines), I am very concerned about the reliability of this paper's conclusions. How would you address or demonstrate the robustness of your findings under these limitations?

2.How do you explain the significant differences in experimental results with and without transaction costs? What factors contribute to this discrepancy?

3.What are the specific advantages of your time series decomposition method compared to other decomposition approaches, and why do these advantages arise?

4.Other questions can refer to the weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this paper, the authors conduct a comparative analysis of various LSTM-based and Transformer-based models for multiple financial prediction tasks using high-frequency limit order book data. They introduce a novel LSTM-based model called DLSTM and a newly designed Transformer-based model specifically tailored for financial predictions. Their results reveal that Transformer-based models offer a slight advantage in predicting absolute price sequences. However, LSTM-based models show superior and more consistent performance in predicting differential sequences, such as price differences and movements.

### Strengths
The structure and logic of the paper is well organized. 

The experimental setup, description, and analysis are clearly stated with sufficient detail.

### Weaknesses
1. The authors compare Transformers and LSTMs, concluding that LSTMs have advantages in multiple electronic trading tasks. However, the selection of Transformer-based models is limited to earlier studies (prior to 2023) and does not include recent state-of-the-art (SOTA) works, such as those mentioned in references [1], [2], and [3]. Notably, Liu et al. [2] claim significant improvements on similar tasks. Excluding these recent studies makes it premature to conclude that Transformer-based models underperform compared to LSTMs. Additionally, there is insufficient evidence to assert that the authors' proposed DLSTM model is the optimal choice for this application. Could you please include comparisons with some of these SOTA results to more robustly justify the conclusion?

[1] Garza, A., Challu, C., & Mergenthaler-Canseco, M. (2023). TimeGPT-1. arXiv preprint arXiv:2310.03589.

[2] Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2023). itransformer: Inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625.

[3] Das, A., Kong, W., Sen, R., & Zhou, Y. (2023). A decoder-only foundation model for time-series forecasting. arXiv preprint arXiv:2310.10688.

2. The authors' conclusion lacks novelty and largely aligns with the findings and conclusions of Zeng et al. [4] It appears to apply established approaches and conclusions to domain-specific practices. While retaining empirical relevance, the study does not offer methodological breakthroughs.

[4] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023, June). Are transformers effective for time series forecasting?. In Proceedings of the AAAI conference on artificial intelligence (Vol. 37, No. 9, pp. 11121-11128).

3. The experimental setup could be made more representative by incorporating additional metrics such as Mean Absolute Scaled Error and Relative Mean Absolute Error.

### Questions
Please refer to questions to be addressed, the weaknesses section.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper compares the performance of LSTM and Transformer models in financial time series forecasting (limit order book data). They compared with FEDformer, Autoformer, Informer, Reformer, Transformer and LSTM. The main results show that Transformer has a slight advantage in predicting absolute price series, but the LSTM model performs more consistently and accurately in the prediction of price changes and price movements. In addition, the paper introduces DLSTM inspired by DLinear and Autoformer.

### Strengths
Even relatively simple LSTM models perform well in financial time series forecasting tasks, compared with transformer-based model.

### Weaknesses
1. The writing quality of the paper is low, especially the format of literature citation is not uniform and some of the citations are not standardized in formatting and arrangement. 
2. The experimental setup lacks comparison with the frameworks and standards widely used in the current research field and fails to demonstrate the advantages of the selected model. For example, the authors failed to cite and use the latest limit order book (LOB) benchmark frameworks, such as LOBFrame (https://github.com/FinancialComputingUCL/LOBFrame) and LOBCAST (https://arxiv.org/abs/2308.01915), both of which are open source frameworks currently widely used for Limit Order Book Forecasting. In addition, the authors did not include some of the latest Transformer based models (e.g., iTransformer and PatchTST), which have demonstrated advantages in terms of performance and efficiency in time series forecasting. Comparing these latest models would make the experimental results more convincing and practical.
3. The experimental data used in this paper is limit order book data from three cryptocurrencies, which, although suitable for high-frequency forecasting tests, is not representative of the financial market, and the volatility and noise characteristics of the cryptocurrency market are quite different from those of the traditional financial market. Data from LOBSTER (https://lobsterdata.com/) are more common and widely used in the literature currently.

### Questions
If possible, include LOB data from the LOBSTER dataset, to increase the generalizability of the experiment. If possible, include latest transformer based model (e.g. iTransformer, PatchTST). Recommend to use benchmarking frameworks such as LOBFrame or LOBCAST  in the experimental design to ensure that the results can be more comparable to existing studies. A more detailed discussion of the specific differences and advantages of DLSTM over other temporal decomposition methods (e.g., DLinear) could be added. could also include some ablation studies. include code for reproducibility.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper explores the use of Transformer and LSTM-based models for financial time series forecasting tasks using high-frequency limit order book (LOB) data. A new LSTM-based model, DLSTM, is proposed alongside a modified Transformer architecture tailored for financial predictions. The study compares these models across three tasks: mid-price prediction, mid-price difference prediction, and mid-price movement prediction. Results suggest that Transformer-based models offer only marginal improvements in specific tasks, while LSTM models, particularly DLSTM, are more reliable in predicting mid-price differences and movements.

### Strengths
Relevant Application: The use of LSTM and Transformer models for financial predictions on LOB data is timely and relevant given the growing interest in high-frequency trading and predictive models in finance.

Comparative Scope: The study covers multiple models and tasks, providing a broad comparison between LSTM- and Transformer-based architectures on real-world financial data.

### Weaknesses
Unconvincing Novelty: The paper lacks substantial novelty. The DLSTM model is essentially a combination of existing methods, such as time series decomposition and LSTM layers, without a clear innovation. Similarly, the Transformer modifications are incremental and do not provide a compelling improvement. As a result, the contributions seem incremental and insufficiently distinct from existing work in financial time series forecasting.

Interpretability Issues: The added complexity of Transformer-based models raises interpretability concerns, especially given the unclear benefit over simpler LSTM-based models. Without a more interpretable mechanism or explanation for its performance gains, the model’s added complexity appears unnecessary.

Insufficient Performance Gain for Complexity: The study demonstrates only marginal improvements from the proposed Transformer modifications over traditional LSTMs, particularly in mid-price prediction. Despite the significant computational complexity introduced by Transformer-based models, the improvements are minimal and do not convincingly justify their adoption for practical trading applications.

### Questions
What specific modifications were made to the Transformer architecture to adapt it to financial prediction tasks?

Can the authors elaborate on the metrics used to evaluate the models' performance? What criteria were significant in determining the practical utility of the models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
5348_Transformers_versus_LSTMs
pdf
LT
Here is a new paper needs to be reviewed. Summary*
Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.
*

Summary

This paper conducts a comparative study between LSTM-based and Transformer-based models for financial time series forecasting, specifically in the context of electronic trading using high-frequency limit order book (LOB) data. The authors investigate the performance of these models across three prediction tasks: mid-price prediction, mid-price difference prediction, and mid-price movement prediction.

For the mid-price prediction task, the study finds that Transformer-based models like FEDformer and Autoformer achieve lower prediction errors than LSTM-based models. However, the authors note that the practical utility of these results for high-frequency trading is limited due to insufficient prediction quality.

In the mid-price difference prediction task, LSTM-based models demonstrate superior performance and robustness compared to Transformer-based models. The canonical LSTM achieves the highest R^2 of around 11.5% within about 10 prediction steps, while state-of-the-art Transformer models struggle to effectively process difference sequences.

The paper's main contribution lies in the mid-price movement prediction task, where the authors introduce a novel LSTM-based model called DLSTM. This model integrates LSTM with a time series decomposition approach inspired by the Autoformer architecture. DLSTM significantly outperforms all other models in classification metrics and proves its effectiveness in trading simulations, particularly when transaction costs are considered.

Additionally, the authors adapt the architecture of existing Transformer-based models to better suit the demands of the movement prediction task. They incorporate both past and projected mid-price data, followed by a linear layer and softmax activation, to determine price movements.

Overall, the study highlights that while Transformer-based models may excel in certain aspects of mid-price prediction, LSTM-based models, particularly the proposed DLSTM, demonstrate consistent superiority and practicality in financial time series prediction for electronic trading.

### Strengths
Originality: The study offers a novel perspective on the application of LSTM-based and Transformer-based models in financial time series forecasting, specifically in the context of electronic trading using high-frequency LOB data. The authors introduce a new LSTM-based model, DLSTM, which creatively combines LSTM with a time series decomposition approach inspired by the Autoformer architecture. This innovative integration of existing ideas allows DLSTM to outperform other models in the mid-price movement prediction task.
Quality: The paper demonstrates a high level of quality in its experimental design and analysis. The authors conduct a comprehensive comparative study across three prediction tasks (mid-price prediction, mid-price difference prediction, and mid-price movement prediction), using a diverse range of LSTM-based and Transformer-based models. The experiments are well-structured, and the results are thoroughly analyzed, providing valuable insights into the performance of different models in each task.
Clarity: The paper is well-written and easy to follow. The authors provide clear explanations of the problem formulation, the proposed DLSTM model, and the experimental setup. The use of tables and figures enhances the clarity of the results, making it easy for readers to compare the performance of different models across various metrics and prediction horizons.
Significance: The findings of this study have significant implications for the application of deep learning models in financial time series forecasting, particularly in the context of electronic trading. The authors demonstrate that while Transformer-based models may excel in certain aspects of mid-price prediction, LSTM-based models, especially the proposed DLSTM, exhibit superior and more consistent performance in tasks such as mid-price difference prediction and mid-price movement prediction. The incorporation of trading simulations with and without transaction costs further highlights the practical significance of the proposed DLSTM model for real-world trading scenarios.

Moreover, the paper's adaptation of existing Transformer-based models' architecture to better suit the demands of the movement prediction task showcases the potential for further improvements in this domain. By incorporating both past and projected mid-price data, followed by a linear layer and softmax activation, the authors demonstrate a creative approach to enhancing the performance of Transformer-based models in financial time series forecasting.
In summary, the paper's originality, quality, clarity, and significance make it a valuable contribution to the field of financial time series forecasting using deep learning models, offering new insights and directions for future research in this domain.

### Weaknesses
While the paper presents valuable insights and contributions, there are a few areas that could be improved or require further clarification:

Limited dataset diversity: The experiments in this study are conducted using LOB data from a single cryptocurrency pair (BTC-USDT or ETH-USDT) on one exchange (Binance). To demonstrate the generalizability of the proposed DLSTM model and the comparative analysis between LSTM-based and Transformer-based models, it would be beneficial to include a wider range of financial instruments, such as stocks, forex, or other cryptocurrencies, as well as data from multiple exchanges. This would strengthen the paper's conclusions and provide a more comprehensive assessment of the models' performance across diverse financial time series.
Lack of ablation studies: While the paper introduces the novel DLSTM model, which integrates LSTM with a time series decomposition approach, there is a lack of ablation studies to investigate the individual contributions of each component. For example, the authors could compare the performance of DLSTM with and without the time series decomposition to assess the impact of this specific modification. Additionally, a more detailed analysis of the adapted Transformer-based models' architecture for the movement prediction task would provide valuable insights into the effectiveness of the proposed changes.
Limited discussion on model interpretability: Interpretability is a crucial aspect of financial time series forecasting models, especially in the context of electronic trading, where understanding the factors driving the model's predictions is essential for risk management and decision-making. The paper could benefit from a more in-depth discussion on the interpretability of the proposed DLSTM model and the adapted Transformer-based models, as well as a comparison with the interpretability of other LSTM-based and Transformer-based models.
Hyperparameter tuning and model selection: The paper does not provide a detailed description of the hyperparameter tuning process and model selection criteria for the various models used in the experiments. It is essential to discuss the approach used for hyperparameter optimization, such as grid search, random search, or Bayesian optimization, and the specific hyperparameters tuned for each model. Additionally, the authors could provide more information on the model selection process, such as the use of validation sets or cross-validation techniques.
Robustness to market conditions: The experiments in this study are conducted using LOB data from a specific time period (e.g., July 2022). To demonstrate the robustness of the proposed DLSTM model and the comparative analysis between LSTM-based and Transformer-based models, it would be valuable to evaluate the models' performance under different market conditions, such as periods of high volatility, market crashes, or significant news events. This would provide a more comprehensive assessment of the models' ability to generalize and adapt to various market scenarios.

Addressing these weaknesses would further strengthen the paper's contributions and provide a more comprehensive and robust analysis of the proposed DLSTM model and the comparative study between LSTM-based and Transformer-based models in financial time series forecasting for electronic trading.

### Questions
Dataset diversity and generalizability: Can you provide more insights into the choice of using only Binance LOB data for a single cryptocurrency pair in your experiments? How do you expect the proposed DLSTM model and the comparative analysis between LSTM-based and Transformer-based models to perform on a wider range of financial instruments, such as stocks, forex, or other cryptocurrencies, as well as data from multiple exchanges? Providing results on more diverse datasets could strengthen the claims of generalizability and robustness of the findings.
Ablation studies and component contributions: Can you conduct ablation studies to investigate the individual contributions of the time series decomposition approach in the proposed DLSTM model? It would be helpful to compare the performance of DLSTM with and without this specific modification to assess its impact on the model's effectiveness. Additionally, can you provide a more detailed analysis of the adapted Transformer-based models' architecture for the movement prediction task, highlighting the importance of each proposed change?
Model interpretability: Can you elaborate on the interpretability of the proposed DLSTM model and the adapted Transformer-based models? How do these models compare with other LSTM-based and Transformer-based models in terms of interpretability? Providing insights into the factors driving the models' predictions and their relative importance could be valuable for understanding the models' decision-making process and enhancing trust in their applications for electronic trading.
Hyperparameter tuning and model selection: Can you provide more details on the hyperparameter tuning process and model selection criteria used for the various models in your experiments? Specifically, what approach was used for hyperparameter optimization (e.g., grid search, random search, Bayesian optimization), and which hyperparameters were tuned for each model? Additionally, how were the validation sets or cross-validation techniques employed in the model selection process?
Robustness to market conditions: Have you considered evaluating the performance of the proposed DLSTM model and the comparative analysis between LSTM-based and Transformer-based models under different market conditions, such as periods of high volatility, market crashes, or significant news events? Demonstrating the models' ability to generalize and adapt to various market scenarios could provide a more comprehensive assessment of their robustness and practical applicability in electronic trading.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This research compares the effectiveness of Transformer and LSTM architectures in financial forecasting. The study examines both model types using high-frequency trading data and introduces DLSTM and a finance-specific Transformer. Results show that Transformers only slightly outperform in absolute price predictions, while LSTMs showing more reliable performance overall.

### Strengths
1. The paper addresses a relevant and significant question by comparing LSTM and Transformer models in financial time series forecasting.
2. The experimental setup is extensive and provides substantial data.

### Weaknesses
1. The paper lacks code and detailed implementation information for both the Transformer and LSTM models, which limits reproducibility.
2. The novelty of the proposed approach is limited. While the authors introduce a DLSTM model to improve performance, the idea of decomposition was previously explored in models like DLinear [1], diminishing the originality of the contribution. Beyond the comparative analysis, additional innovation is also limited.
3. The decomposition strategy appears to be applied only to the LSTM model. For a fair comparison, a decomposition approach for the Transformer model should also be included. In Table 3, DLSTM significantly outperforms LSTM, which suggests that a decomposed Transformer might also show improved results.
4. The paper does not include several state-of-the-art (SOTA) Transformer-based models, such as PatchTST [2], Crossformer [3], and iTransformer [4], in the comparison, which limits the comprehensiveness of the analysis.
5. The statement "Transformer-based models exhibit only a marginal advantage in predicting absolute price sequences, whereas LSTM-based models demonstrate superior and more consistent performance in predicting differential sequences such as price differences and movements" requires further investigation. A deeper analysis into the underlying causes of this observed difference is missing, which weakens the interpretability of the results.

[1] Zeng, Ailing, et al. "Are transformers effective for time series forecasting?." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 9. 2023.

[2] Nie, Yuqi, et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." The Eleventh International Conference on Learning Representations.

[3] Zhang, Yunhao, and Junchi Yan. "Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting." The eleventh international conference on learning representations. 2023.

[4] Liu, Yong, et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." The Twelfth International Conference on Learning Representations.

### Questions
1. What are the fundamental architectural characteristics that make LSTM models more effective for differential sequences compared to Transformers?
2. Can you provide deeper analysis to support the generalizability of your findings of LSTM vs Transformer?
3. How does financial time series forecasting different from other time series forecasting (like weather, traffic, etc.?)
4. To address the remaining limitations identified in *Weaknesses*: a) Could you provide detailed model implementations and hyperparameter configurations? b) How would decomposition techniques benefit Transformer architectures? c) Please include comparisons with state-of-the-art models

### Soundness
2

### Presentation
2

### Contribution
1
