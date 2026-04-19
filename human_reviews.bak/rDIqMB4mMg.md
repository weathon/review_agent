# PostRainBench: A Comprehensive Benchmark and A New Model for Precipitation Forecasting

- Decision: Reject
- Scores: 5, 6, 3, 5

## Abstract
Accurate precipitation forecasting is a vital challenge of both scientific and societal importance. Data-driven approaches have emerged as a widely used solution for addressing this challenge. However, solely relying on data-driven approaches has limitations in modeling the underlying physics, making accurate predictions difficult. Coupling AI-based post-processing techniques with traditional Numerical Weather Prediction (NWP) methods offers a more effective solution for improving forecasting accuracy. Despite previous post-processing efforts, accurately predicting heavy rainfall remains challenging due to the imbalanced precipitation data across locations and complex relationships between multiple meteorological variables. To address these limitations, we introduce the PostRainBench, a comprehensive multi-variable NWP post-processing benchmark consisting of three datasets for NWP post-processing-based precipitation forecasting. We propose CAMT, a simple yet effective Channel Attention Enhanced Multi-task Learning framework with a specially designed weighted loss function. Its flexible design allows for easy plug-and-play integration with various backbones. Extensive experimental results on the proposed benchmark show that our method outperforms state-of-the-art methods by 6.3\%, 4.7\%, and 26.8\% in rain CSI on the three datasets respectively. Most notably, our model is the first deep learning-based method to outperform traditional Numerical Weather Prediction (NWP) approaches in extreme precipitation conditions. It shows improvements of 15.6\%, 17.4\%, and 31.8\% over NWP predictions in heavy rain CSI on respective datasets. These results highlight the potential impact of our model in reducing the severe consequences of extreme weather events.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Precipitation forecasting holds significant scientific and societal value. While data-driven techniques are increasingly popular in addressing forecasting challenges, they struggle with accurately representing the physics involved. Marrying AI post-processing with established Numerical Weather Prediction (NWP) methods can bolster forecast accuracy. Yet, predicting heavy rainfall remains a hurdle, given uneven precipitation data and intricate meteorological variable interplays. This study presents the PostRainBench, a robust NWP post-processing benchmark comprising three datasets for enhanced precipitation forecasting. We introduce CAMT, a Channel Attention Enhanced Multi-task Learning framework, integrated with a tailored weighted loss function. This framework seamlessly merges with various backbones. Tests on our benchmark indicate that CAMT surpasses existing methods by notable margins, especially under extreme rainfall conditions. Significantly, our model is the pioneering deep learning tool that outdoes traditional NWP methods in predicting intense rain, showcasing its immense promise in mitigating extreme weather implications.

### Strengths
The introduction of the CAMT framework stands out as a noteworthy achievement. Its ability to integrate smoothly with different backbones offers versatility, while its significant performance improvements in extreme precipitation predictions highlight its potential in addressing some of the most challenging aspects of weather forecasting.

### Weaknesses
1. The authors claim in the paper that they are the first to surpass NWP using a deep learning approach. However, to my knowledge, several studies have already ventured into this area, as evidenced by references [1-3]. I suggest the authors review the accuracy of this statement and make appropriate amendments.

2. In my perspective, the proposed model in this paper seems to lack innovation, appearing to be a combination of the swinTransformer and unet. This raises concerns about the reproducibility of the experimental results. Could the authors provide more details or rationale to support their design choices?

3. The selected baselines in the paper don't appear comprehensive. For instance, recent significant works in the relevant field, such as OpenSTL[4] and Fourcastnet[5], are not considered. I would recommend the authors to include these for comparison.


[1] Zhang, Yuchen, et al. "Skilful nowcasting of extreme precipitation with NowcastNet." *Nature* 619, no. 7970 (2023): 526-532.

[2] Bi, Kaifeng, et al. "Pangu-weather: A 3d high-resolution model for fast and accurate global weather forecast." *arXiv preprint arXiv:2211.02556* (2022).

[3] Chen, Kang, et al. "FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead." *arXiv preprint arXiv:2304.02948* (2023).

[4] Tan, Cheng, et al. "OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning." *arXiv preprint arXiv:2306.11249* (2023).

[5] Pathak, Jaideep, et al. "Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators." *arXiv preprint arXiv:2202.11214* (2022).

### Questions
See Weaknesses

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Tree datasets are grouped to provide an evaluation dataset for precipitation forecasting. A method is proposed for precipitation forecasting as a weighted combination of three different algorithms with an without learning. Comparative experiments are proposed and discussed as well as an ablation study. The datasets is announce to be released after publication.

### Strengths
The dataset is useful and may help researchers in their work on the subject of precipitation forecasting. The obtained results in precipitation forecasting with the combination of three algorithms is significant. The comparison with other methods seems well performed.

### Weaknesses
The proposed algorithm is not particularly surprising.

### Questions
Is there a part of the datasets newly provided by the authors ?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a unique deep learning-based post-processing technique designed for Numerical Weather Prediction (NWP) methods, focusing specifically on the task of precipitation forecasting. The authors curate three distinct datasets from Korea, Germany, and China for evaluation purposes. The architecture of the proposed model is tripartite: it consists of a Channel Attention Model (CAM) to handle the high dimensionality in NWP variables, a Swin-Unet backbone, and a multi-task learning loss function that incorporates both classification and regression. Noteworthy are the contributions of the CAM and the multi-task learning loss. Comparative evaluations demonstrate that the proposed model surpasses traditional methods in performance.

### Strengths
1. The model delivers a marked performance improvement across all three test datasets.

2. The paper investigates an intriguing application of machine learning, namely, precipitation prediction.

### Weaknesses
1. The model lacks substantial innovation; both the Channel Attention Model (CAM) and the multi-task learning loss function appear to be straightforward engineering optimizations rather than novel contributions.

2. Despite acknowledging the presence of significant data imbalance, the authors do not incorporate any mechanisms within the model to address this issue.

3. The experimental section could benefit from an in-depth discussion analyzing how the model's performance varies with different lead times.

### Questions
Could the authors elaborate on the weaknesses in the model, particularly in relation to handling imbalanced data distributions?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper constructs a comprehensive benchmark in the field of weather forecasting and proposes a baseline model to evaluate the proposed benchmark. To be specific, to facilitate the development of precipitation prediction, the authors propose to compile 3 existing precipitation datasets into a new benchmark. Meanwhile, they propose a baseline model with the designs of channel attention module, multi-task learning, weighted loss and etc.. They validate the proposed method on three datasets, and observe significant improvements over previous methods in different scenarios.

### Strengths
	This paper is well written and easy following.
	This paper provides all-round and detailed performance comparisons with different kinds of classical precipitation methods on these three datasets. Those comparisons will become valuable for the community if the authors release their codebases and benchmarks in the future.
	The proposed method achieves significant performance improvements over those classical precipitation methods, showing the superiority of their method.

### Weaknesses
1. It is suggested that the authors could show some qualitative results to demonstrate the superiority of their method. It is interesting to see how the proposed channel attention and weighted loss work, which plays a key role in this paper.
2. It is suggested that the authors could discuss some limitations of their method in this paper, and open some potential possibilities with their benchmark in this paper. I do not see many application scenarios shown in this paper, which makes this paper sound relatively restricted.
3. The lead time in this paper is fixed as 3h. In practice, the length of lead time has significant influence on precipitation results. It would be better if the authors could exploit the lead time in their method, to show the robustness of their method.
4. For a benchmark paper, the reproduce ability is very important, especially for the climate science domain. Thus, I urge the authors to present full data and code for the reproduce ability checking.

### Questions
please check the weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
