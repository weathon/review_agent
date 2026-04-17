# Mixed Channel Dependency Diffusion Model with Retrieval Guidance for Time Series Forecasting

- Decision: Reject
- Scores: 4, 4, 4, 6, 6, 2

## Abstract
Recent advancements in deep learning techniques have improved the performance of time series forecasting, especially with state-of-the-art generative models.
Despite making progress in capturing conditional time-series patterns with uncertainty, existing time series generative models face reliability and computational challenges in long-term forecasting, especially when the number of variate is large. 
Moreover, the maximum likelihood objective of generative modeling is prone to underestimation for low-density region of the data manifold, therefore leading to sub-optimal conditional sampling quality.
In this paper, we propose a mixed channel dependency diffusion model with retrieval guidance (MiDDiR) to address these challenges. 
In MiDDiR, we employ a novel mixed channel dependency method on time series diffusion model, encoding historical time series in a channel-dependent manner to obtain informative historical representation while denoising in a channel-independent manner to decrease modeling complexity.
During inference, we retrieve similar history occurrence for explicitly tilting the score estimation as retrieval guidance to enhance forecasting quality.
Extensive experiments demonstrate the effectiveness of \rgdiff, outperforming baselines in a variety of real-world time series datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces a promising diffusion-based generative framework that enhances multivariate time series forecasting by addressing challenges of high-dimensional dependency and data sparsity. 
The model employs a mixed channel dependency mechanism, where temporal features are encoded with inter-variable awareness while denoising operates independently across channels, balancing expressiveness and scalability. 
In addition, a retrieval guidance module retrieves historically similar patterns and integrates them into the diffusion sampling process, effectively improving prediction accuracy in low-density or unseen regions. 
Experiments on standard benchmarks (ETT, Electricity, Traffic, and Weather) show that MiDDiR achieves state-of-the-art performance in both deterministic and probabilistic forecasting metrics, outperforming recent diffusion-based models such as NsDiff and TMDM, while offering better parameter efficiency and uncertainty calibration.

### Strengths
S1.
MiDDiR introduces a mixed channel dependency diffusion mechanism that combines the strengths of both approaches—encoding inter-channel relationships while performing channel-independent denoising to improve scalability and stability. 

S2.
The retrieval-guided diffusion process leverages stored historical patterns to guide sampling in low-density regions of the data manifold.
This idea is novel to the time series domain and represents a creative and transferable paradigm shift, bridging memory-based reasoning with generative diffusion modeling.

S3.
The experimental design is robust: MiDDiR is evaluated on multiple widely used long-horizon benchmarks (ETT, Electricity, Traffic, Weather), using both deterministic metrics (MAE, MSE) and probabilistic scores (CRPS, QICE) to demonstrate consistent improvements.

### Weaknesses
W1.
The etrieval guidance might bias generation toward seen patterns rather than truly generalizable dynamics, especially when the data distribution shifts.

W2.
The visualization in Figure 4 suggests varying dependency patterns across datasets, but the analysis stops short of explaining why such differences occur or how they affect model generalization.

W3.
The paper empirically demonstrates the superiority of diffusion-based forecasting but does not explain why the proposed channel-dependent encoding and independent denoising yield better uncertainty calibration or robustness. The theoretical link between these design choices and the underlying data distribution modeling is not clear.

W4.
The experiments rely primarily on standard benchmarks (ETT, Electricity, Traffic, Weather). While these are widely used, they may not sufficiently test the retrieval mechanism’s robustness under different spatiotemporal scales or non-stationarity. 
Moreover, the performance on highly dynamic datasets (e.g., finance, healthcare) remains untested.

W5.
The model is evaluated using probabilistic metrics (CRPS, QICE), but there is no analysis of how well the predicted uncertainty intervals reflect true variability. 
This omission makes it hard to assess whether MiDDiR provides reliable probabilistic forecasts or merely sharper—but miscalibrated—distributions.

### Questions
Q1.
Could you clarify how the mixed channel dependency encoder is trained to balance between inter-channel and intra-channel representations? 
For instance, is there an explicit regularization term or architectural design that controls this mixture, or is it purely learned through backpropagation?

Q2.
The paper motivates channel-independent denoising as a way to improve efficiency and reduce parameter coupling, but could the authors provide a theoretical or empirical justification for why independence at the denoising stage does not degrade the learned dependency structure?

Q3.
How does MiDDiR handle distribution shift or noisy retrievals (e.g., when retrieved samples differ significantly from the target input)? Is there a threshold or adaptive weighting mechanism to prevent harmful guidance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MiDDiR, a diffusion model for multivariate time series forecasting. Its core ideas are a "mixed channel dependency" strategy that uses a channel-dependent encoder and a channel-independent denoiser, and a "retrieval guidance" mechanism that leverages similar historical patterns during inference to tilt the sampling process. The authors demonstrate state-of-the-art performance on several benchmarks compared to existing probabilistic and deterministic models.

### Strengths
The proposed "mixed channel dependency" is a thoughtful and well-motivated design. It cleverly balances the need to capture informative inter-channel dependencies in the historical context (via the encoder) with the computational benefits of channel-independent generation (via the denoiser). This approach directly tackles the known issue of maximum likelihood training underestimating low-density regions.

### Weaknesses
1. Insufficient theoretical explanation and details for the mixed channel dependency mechanism. Although the authors propose the "channel-dependent encoding + channel-independent denoising" architecture, they do not deeply explain the rationality of the connection between the two phases. For example, how is the cross-channel correlation information generated by channel-dependent encoding effectively utilized in channel-independent denoising? If the denoising phase processes each channel completely independently, is there information loss in the cross-channel information captured during the encoding phase? Additionally, details of the channel-dependent encoder design (e.g., basis for selecting the number of attention heads and layers) are not clearly stated.

2. Lack of justification for key design choices in retrieval guidance. The retrieval phase adopts a strategy of "cosine similarity + Top-K weighted average," but the logic for selecting the K value is not explained (only mentioning "selected via the validation set" without providing sensitivity analysis for different K values); meanwhile, the reason for choosing L2 distance as the energy function is not justified, nor is it compared with other distance metrics (e.g., Manhattan distance, Dynamic Time Warping (DTW)—a commonly used metric for time series data), making it impossible to verify the optimality of the current choice.

3. Missing comparison with non-diffusion probabilistic models.

4. Unverified the Performance in longer forecasting scenarios (e.g., 720 steps).

### Questions
1. Through what specific means is the cross-channel correlation information captured by the channel-dependent encoder injected into the channel-independent denoising phase?

2. The retrieval database is built based on the training set, and the impact of training set distribution shifts on retrieval effectiveness (e.g., whether retrieval guidance fails when the test set contains new patterns not present in the training set) is not discussed.

### Soundness
2

### Presentation
3

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
The paper proposes MiDDiR, short for mixed channel dependency diffusion model, for time series forecasting. The paper is motivated by the idea of improving long-horizon forecasts for diffusion-style generative models. MiDDiR has a channel-dependent encoding MLP block which mixes information across different variates in a multivariate time series. This encoding is passed on to a Diffusion Transformer (DiT)-based architecture for denoising and sampling. MiDDiR also allows a mechanism to potentially improve the forecast samples by using the training corpus for additional guidance. This is achieved by selecting related time series in the embedding space and modifying the sampling score function with a weighted guidance term. Experiments on 7 benchmark datasets show that MiDDiR improves on probabilistic forecasting compared to other baselines.

### Strengths
- The most interesting aspect about this paper is the idea of retrieval guidance. It could potentially be helpful in specific forecasting scenarios such as cold-start forecasting. However, this specific dimension has not been explored. 
- The paper is generally easy to follow.

### Weaknesses
- Upon reading the introduction, it is not immediately obvious what the key motivating factor of this work is. As per my understanding, the main motivation is to improve the performance of diffusion-based generative models on long-horizon forecasting. Although there have been several works on the problem of long-horizon forecasting and the long-term forecasting benchmark, it holds limited value in practice. 
- "To ensure fair comparison, we 218 evaluate with a look back window of 168 steps and target window in 96, 192, 336 for all datasets." Restricting context length is not ensuring "fair comparison". Some models work better with longer context lengths and practically context length is not restricted and longer contexts are used whenever possible. A fair comparison would be to either provide longer contexts for model that work well with them or experiment with different context lengths and report the best results for each model.
- The evaluation benchmark used in this paper is often criticized for its limitations. 4 of the 7 datasets are essentially the same dataset (ETTh1, ETTh2, ETTm1, ETTm2). Please refer to the talk (and paper) from C. Bergmeir [1, 2] where he discusses the limitation of this benchmark and current evaluation practices. A recent position paper [3] also conducted a comprehensive evaluation of models on this benchmark showing that there's no obvious winner. Authors should consider using better benchmarks to demonstrate the effectiveness of their method. See, for example,
    - Chronos Benchmark II: This benchmark includes 27 datasets (42, if you include Benchmark I) providing a comprehensive coverage over domains, frequencies and other properties [4].
    - GIFT-Eval: This benchmark includes 90+ tasks across multiple datasets and domains. Please refer to https://github.com/SalesforceAIResearch/gift-eval.
    - The Monash Benchmark: https://forecastingdata.org/

[1] https://neurips.cc/virtual/2024/workshop/84712#collapse108471               
[2] Hewamalage, Hansika, Klaus Ackermann, and Christoph Bergmeir. "Forecast evaluation for data scientists: common pitfalls and best practices." Data Mining and Knowledge Discovery 37.2 (2023): 788-832.                 
[3] Brigato, Lorenzo, et al. "Position: There are no Champions in Long-Term Time Series Forecasting." arXiv preprint arXiv:2502.14045 (2025).                      
[4] Ansari, Abdul Fatir, et al. "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).

### Questions
I previously reviewed this paper for NeurIPS. The results reported for MiDDiR have improved considerably compared to the NeurIPS submission. Can the authors clarify what lead to this improvement?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MiDDiR for multivariate time series forecasting. The key innovation is a "mixed channel dependency" strategy that encodes historical data in a channel-dependent manner while performing channel-independent denoising to balance modeling expressiveness and computational efficiency. Additionally, the paper introduces retrieval guidance that tilts the diffusion sampling process using similar historical patterns from the training set. Extensive experiments on multiple real-world datasets demonstrate that MiDDiR outperforms existing baselines in both probabilistic and point forecasting tasks.

### Strengths
1.The proposed "mixed channel dependency" strategy leverages channel-dependent encoding to capture inter-channel correlations while maintaining a channel-independent generation approach to reduce computational complexity, representing a well-balanced trade-off for high-dimensional multivariate forecasting.

2.The proposed retrieval-guided sampling method enhances generation quality by leveraging historical patterns, specifically addressing the low-density region problem.

3.The method shows consistent improvements across multiple datasets, achieving substantial gains over baselines

### Weaknesses
1.The Retrieval Guidance mechanism drastically increases computational overhead and latency by requiring an expensive Top-K search and calculation at every diffusion step for every channel. The paper fails to provide any quantitative analysis of this critical increase in inference time, making the model's practical utility for real-time long-term forecasting questionable.

2.The retrieval system risks overfitting to the training data and lacks robustness. Over-reliance on the database can be induced by a high guidance strength (λ), while for novel or low-density data patterns, retrieved historical samples may be irrelevant, noisy, or actively misguide the diffusion process, thus degrading prediction quality.

3.The model relies on simple cosine similarity to measure feature similarity during retrieval, a geometric matching approach that may not fully capture complex temporal characteristics such as dynamic patterns, seasonality, or phase alignment in time series data, thereby raising concerns about the robustness of the retrieval process.

4.The paper does not discuss how to efficiently and dynamically update or manage this large-scale retrieval database in a practical online forecasting setting. The lack of consideration for dynamic data environments limits its feasibility for long-term deployment in the real world.

### Questions
1.Please quantify the inference time overhead introduced by the retrieval guidance mechanism and discuss potential strategies to reduce this latency.

2.Please include a robustness analysis under different guidance strengths (λ), especially for novel or low-density patterns where retrieval may be unreliable.

3.Consider evaluating more temporally-aware similarity metrics (e.g., DTW or phase-aware measures) to better capture complex time-series dynamics.

4.Please propose a feasible strategy for dynamically updating the retrieval database in evolving data environments to support long-term deployment.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a diffusion model-based generative model to forecast time series data, with two key features: (1) a channel dependent encoder which models the dependencies between the input channels, and (2) retrieval guidance, to guide the model predictions to capture the low density regions of the forecasting dataset. The authors compare their proposed method on widely-used forecasting datasets, against a few time series forecasting baselines.

### Strengths
To the best of my knowledge, the proposed method is novel, and on the compared datasets, against the compared baselines, the proposed methods seems to perform well.

### Weaknesses
I believe this is a good paper, however, I have a few recommendations, which I believe would significantly improve the paper in its current forms: 

1. **More baselines**. I believe comparing the proposed model against state-of-the-art time series foundation models (e.g., Chronos-1/2, TimesFM, Tirex, etc.), some of which provide distributional forecasts, will make the results more convincing. 

2. **Better datasets**. The datasets used (e.g., ETT, Exchange, Weather) are known to have limited diversity and issues. A few new benchmarks such as GIFT-Eval and fev-bench were released to partially address some of these gaps. I would highly recommend that the authors supplement their results with findings from these benchmarks. Moreover, one of the key features of the proposed study is modeling inter-channel dependencies. However, prior work such as [1] and Chronos-2 have also reported that multivariate modeling yields little or no benefit on these public datasets, which weakens the empirical claims around MiDDiR's improvements.

###
1. Żukowska, Nina, et al. "Towards long-context time series foundation models." arXiv preprint arXiv:2409.13530 (2024).
Goswami, Mononito, et al. "Moment: A family of open time-series foundation models." arXiv preprint arXiv:2402.03885 (2024).

### Questions
I do not have any specific questions for the authors.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel mixed channel dependency method on time series diffusion model, encoding historical time series in a channel-dependent manner to obtain informative historical representation while denoising in a channel-independent manner to decrease modeling complexity. During inference, we retrieve similar history occurrence for explicitly tilting the score estimation as retrieval guidance to enhance forecasting quality.

### Strengths
The writing logic is clear, and each major part of the module is clearly described.

The idea of using retrieved samples to improve forecasting is also making sense to the time series forecasting tasks.

### Weaknesses
The simulations are not persuasive enough. The Fig. 4's attention map does not reveal many insights of the forecasting task (and there is no units of the color map?).

There are not enough illustrated examples how the retrieved samples look like, and how they would help the forecasting task.

The motivation behind using diffusion model for time series forecasting is not well justified.

Why the Equation (11) is applied to evaluate the similarity score? Has the authors considered other similarity measure?

The complexity of sample retrieval is not reported.

### Questions
Please see weakness above.

### Soundness
2

### Presentation
3

### Contribution
2
