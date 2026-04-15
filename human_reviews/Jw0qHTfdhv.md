# Learning to Generate Predictor for Long-Term Time Series Forecasting

- Decision: Reject
- Scores: 6, 5, 5, 3

## Abstract
Long-term time series forecasting (LTSF) is a significant challenge in machine learning with numerous real-world applications. Although transformer architecture have shown promising performance in the LTSF task, recent research suggests that they are not suitable for time series forecasting due to their permutation invariant characteristic, and proposes a simple linear predictor which outperforms all existing transformer architectures. However, the linear predictor is inflexible and cannot reflect the characteristics of the time series for prediction due to its simple architecture. In this paper, we introduce a novel Learning to Generate Predictor (LGPred) framework, which generates a linear predictor adaptively to the given input time series by leveraging time series decomposition. LGPred obtains representations from the decomposed time series and generates a predictor suitable for the given time series from these representations.
Our extensive experiments demonstrate that LGPred achieves state-of-the-art performance for both multivariate and univariate forecasting tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new framework called Learning to Generate Predictor (LGPred) for long-term time series forecasting. The key idea is to generate a linear predictor dynamically tailored to the input time series, to overcome limitations of fixed linear predictors. LGPred uses time series decomposition into trend and seasonality components. Separate representation modules extract features from each component. A predictor generator uses the extracted features to generate the weights and biases of a linear predictor suited to the input series. A template predictor with bottleneck architecture is used to incorporate common forecasting knowledge and reduce computation cost. Experiments show state-of-the-art performance on 6 benchmark datasets covering disease, economics, energy, traffic and weather domains.

### Strengths
(1) Novel idea of generating parameters of predictor based on input series, enabling adaptation to each series.

(2) Bottleneck template predictor shares knowledge among different time series and reduces computational cost.

(3) Well-written paper and easy to understand.

### Weaknesses
(1) The proposed method includes multiple modules and each of them have their own hyper-parameters to tune. Extensive hyper-parameter tuning was not favorable. Also, I was not clear why a bottleneck architecture was used for template predictor and intuition behind this design was not discussed. 

(2) Time series decomposition is a standard technique already used by some prior works.

(3) Experimental results do not include mean value and standard deviations. It will be good to know if the proposed method is sensitive to initialization. 

(4) Results are mainly conducted on time series datasets with simple patterns. What if the patterns are complicated and hard to capture for linear predictors?  More experimental results or at least discussion of limitations are needed. Time series can vary significantly in terms of distribution and it is good to know when an algorithm can perform good/bad.

### Questions
See my comments in Weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a model (LGPred) which learns the predictor for each sample in the long-term time series forecasting tasks. The LGPred generates a part of weights and bias for the projection from the input to the output, based on representations learnt from the trend and seasonality of each sample. Experiments on several benchmark datasets are conducted to evaluate the effectiveness of LGPred.

### Strengths
The proposed LGPred can generate dynamic predictors for different samples, which is novel for the long-term time series prediction tasks. The paper is well-written in general and easy to understand.

### Weaknesses
1. Some parts in the Preliminary and Method sections are not clear, e.g.,

-It is not clear why time series forecasting with T>48 is considered as LTSF problem, are there any reasons or references?

-Why change the number of channels in the trend block?

-The dilated temporal convolutional network should be introduced, in case some readers do not have related background Knowledge.

2. The comparison with baselines may be unfair and experiments are insufficient.

-I think it is unfair to compare with PatchTST/64 which uses lookback window length 512 only. As shown in the Figure 2 of the PatchTST paper, the performance is changed with different lookback windows. It is better to choose the best results from different lookback windows for PatchTST for a fair comparison. In addition, even based on the current results shown in the Table 1, the proposed LGPred cannot beat PatchTST/64. 

-It is better to add the results of PatchTST in Table 3 due to its superiority.

-There is no complexity analysis between the proposal and baselines.

-It is better to provide some experiment results of using RNN and transformer for the trend component.

### Questions
Same to the Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a learning-to-generate-predictor model, LGPred, for long-term forecasting. In particular, LGPred consists of two parts, a weights generator, and a feature extraction, and then uses a bilinear-type structure to merge them. Moreover, the seasonality trend decomposition is used in the weights generator. Numerical results on 9 datasets are reported.

### Strengths
The usage of bilinear structure seems new in the time series forecasting domain.

### Weaknesses
1. The term *Learning to Generate Predictor* is a little bit overstated from my perspective. My first impression would be a meta-learning model is considered. However, after reading the paper and codes. It seems just a usage of bilinear-type layer for me. I appreciate applying the bilinear layer since it seems not being used in recent forecasting literature. But *Learning to Generate Predictor* may not be the best term to summarize the model novelty for me. If the authors still prefer using *Learning to Generate Predictor*, it would be better to add more discussion to clearly state the difference from the meta-learning type model.

2. The statement "*LGPred is the first attempt at adaptively generating a predictor reflecting the characteristics of each time series.*" seems also a little bit overclaimed. For example, in DeepAR, the network will first generate the $\mu$/$\sigma$ or $\mu$/$\alpha$ for Gaussian distribution or negative binomial (NB) distribution respectively. During the inference stage, the forecasting point will be sampled from the Gaussian/NB distribution. In this case, the Gaussian/NB distribution can be viewed as the *Predictor*, and the parameters in the predictor are learned with a network. 

3. The test data loader sets `drop_last = True`. In this case, the last several test samples are ignored, which will impact the accuracy of results in Table 1- Table 3. It would be better if the authors could fix it.

4. It seems that the random control experiments are not conducted. 
 

Reference:

DeepAR: Salinas, David, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." International Journal of Forecasting 36, no. 3 (2020): 1181-1191.

### Questions
1. The main results in Table 1 - Table 3 are from the model after hyperparameter searching. I'm wondering if the authors can provide a sensitive analysis of the parameter choices to further highlight the robustness of the proposed model.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes the Learning to Generate Predictor (LGPred) framework, a novel approach to enhancing linear time series forecasting models. LGPred adaptively generates a linear predictor tailored to the specific characteristics of a given time series by time series decomposition. This allows the model to discern and adapt to each time series' unique trend and seasonality components. Experimental evidence presented in the paper indicates that LGPred consistently delivers top-tier performance on various benchmarks.

### Strengths
1. The proposed method provides clear motivation for its designs.
2. Empirical results showcase commendable performance, effectively outperforming many preceding methodologies.

### Weaknesses
1. The paper seems to omit discussions on contemporary related works. Notably, the structure of the proposed trend representation module is almost the same as TSMixer [1]. Furthermore, TiDE [2] has previously delved into refining linear models specifically for time series forecasting. Given the architectural similarities between these MLP-based models, a more in-depth comparison and differentiation would enhance clarity.
2. The absence of comprehensive ablation studies leaves the intrinsic value of each component in the proposed method ambiguous. For instance, the ablation analysis in [1] revealed that simpler stacked linear models (i.e., TMix-Only) could rival the performance of the presented methodology. This raises questions regarding the neccesity of LGPred's individual components.
3. The delineation of the dimensions for the linear and fully-connected layers remains ambiguous. For multivariate time series data, these layers could be applied across either time or feature dimensions, as depicted in Figure 2. Unfortunately, the descriptions on the predictor generator and template predictor (page 4) do not explain the dimensional characteristics of these layers adequately.

[1] Chen, Si-An, et al. "TSMixer: An All-MLP Architecture for Time Series Forecasting." Transactions on Machine Learning Research. 2023

[2] Das, Abhimanyu, et al. "Long-term Forecasting with TiDE: Time-series Dense Encoder." Transactions on Machine Learning Research. 2023

### Questions
1. Does the proposed architecture incorporate any non-linear activation functions? It's worth noting that certain linear modules, such as $b_{gen}$, might be redundant given that the concatenation of multiple linear layers essentially functions as a single linear layer.
2. Considering the insights from recent works ([1], [2]) highlighting the potential inadequacy of LTSF benchmarks in reflecting models' capability in handling cross-variate correlations, can the LGPred framework be generalized to tackle more intricate datasets like M5 or Favorita, as explored in [1] and [2]?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
