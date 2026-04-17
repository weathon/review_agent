# Review

## Summary
The paper introduces a new paradigm in time series forecasting called Influence-Aware Time Series Forecasting (IATSF), which addresses the performance plateau observed in recent years, where even large foundation models struggle to outperform simple linear baselines. The authors argue that this stagnation is not due to model architecture but rather a flawed assumption that time series models are self-stimulating, meaning they predict future values based solely on historical data, ignoring external influences that drive real-world systems. Through a control-theoretic analysis, the paper formally proves that this self-stimulation assumption imposes a mathematical barrier on forecasting accuracy. To overcome this barrier, the authors propose IATSF, which reframes time series forecasting as dynamic system modeling rather than correlation-based inference. They introduce a leak-free, temporally synced benchmark that incorporates textual influences to capture qualitative or uncertain dynamics missed by traditional variables. Additionally, they develop FIATS (Forecaster for Influence-Aware Time Series), a lightweight model designed to interpret these influences and adjust its sensitivity to both textual signals and historical data in a channel-specific manner. The paper's main contributions include a control-theoretic analysis revealing forecasting barriers caused by the self-stimulation assumption, the introduction of the IATSF paradigm for influence-aware time series modeling, and the implementation of the Temporal-Synced IATSF benchmark and the FIATS model, which demonstrates the effectiveness of influence-aware forecasting.

## Soundness
3

## Presentation
2

## Contribution
2

## Strengths
- The paper introduces a new paradigm in time series forecasting called Influence-Aware Time Series Forecasting (IATSF), which addresses the performance plateau observed in recent years. This is a significant contribution to the field as it offers a fresh perspective on how to improve forecasting accuracy.
- The authors provide a control-theoretic analysis that reveals the limitations of the self-stimulation assumption in time series forecasting. This analysis is rigorous and provides a solid foundation for the proposed IATSF paradigm.
- The paper introduces a leak-free, temporally synced benchmark that incorporates textual influences to capture qualitative or uncertain dynamics missed by traditional variables. This benchmark is a valuable resource for the research community and can facilitate further development of influence-aware forecasting models.

## Weaknesses
- The paper does not provide a detailed comparison with existing methods that incorporate external influences or textual data into time series forecasting. This makes it difficult to assess the relative performance of the proposed FIATS model compared to other state-of-the-art methods.
- The paper does not provide a detailed analysis of the computational complexity of the proposed FIATS model. This information would be valuable for assessing the practical feasibility of the model, particularly for large-scale applications.
- The paper does not provide a detailed analysis of the sensitivity of the FIATS model to various hyperparameters. This information would be valuable for understanding how to optimally tune the model for different forecasting tasks.

## Questions
- How does the FIATS model compare to other state-of-the-art methods that incorporate external influences or textual data into time series forecasting? Can you provide a detailed comparison of the performance of FIATS against these methods on the proposed benchmark datasets?
- What are the limitations of the FIATS model? Are there any scenarios or types of time series where FIATS may not perform well? Can you provide a detailed analysis of the limitations of FIATS and potential directions for future research to address these limitations?
- How does the computational complexity of the FIATS model compare to other state-of-the-art methods? Can you provide a detailed analysis of the computational complexity of FIATS and discuss its practical feasibility for large-scale applications?
- How sensitive is the FIATS model to various hyperparameters, such as the number of layers, the dimensionality of the embeddings, and the learning rate? Can you provide a detailed analysis of the sensitivity of FIATS to these hyperparameters and discuss how to optimally tune the model for different forecasting tasks?
- How does the FIATS model handle missing or unreliable influence data? Can you provide a detailed analysis of the robustness of FIATS to missing or unreliable influence data and discuss potential strategies for dealing with such situations?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4