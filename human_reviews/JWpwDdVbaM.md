# ARM: Refining Multivariate Forecasting with Adaptive Temporal-Contextual Learning

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Long-term time series forecasting (LTSF) is important for various domains but is confronted by challenges in handling the complex temporal-contextual relationships. As multivariate input models underperforming some recent univariate counterparts, we posit that the issue lies in the inefficiency of existing multivariate LTSF Transformers to model series-wise relationships: the characteristic differences between series are often captured incorrectly. To address this, we introduce ARM: a multivariate temporal-contextual adaptive learning method, which is an enhanced architecture specifically designed for multivariate LTSF modelling. ARM employs Adaptive Univariate Effect Learning (**A**UEL), Random Dropping (**R**D) training strategy, and Multi-kernel Local Smoothing (**M**KLS), to better handle individual series temporal patterns and correctly learn inter-series dependencies. ARM demonstrates superior performance on multiple benchmarks without significantly increasing computational costs compared to vanilla Transformer, thereby advancing the state-of-the-art in LTSF. ARM is also generally applicable to other LTSF architecture beyond vanilla Transformer.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces ARM : a multivariate temporal-contextual adaptive learning method for long-term time series forecasting, which is an enhanced architecture specifically designed for multivariate LTSF modelling. ARM consists 3 modules: Adaptive Univariate Effect Learning (AUEL), Random Dropping (RD) training strategy, and Multi-kernel Local Smoothing (MKLS).  AUEL is for adaptively estimate mean and variance and capture temporal pattern for each series; RD is for robust learning and avoid overfitting when series are interdependent; Multi-kernel Local Smoothing (MKLS) for capturing various temporal dependency among series.  ARM demonstrates superior performance on multiple benchmarks without significantly increasing computational costs compared to vanilla Transformer.

### Strengths
Originality: this paper is fairly novel. It attempts to address previous inferior performance of transformer models compared to a simple feed forward neural network DLinear(Zeng et al. 2022) in long term forecasting task. It proposes unique insights on temporal pattern learning of the output sequence distribution and interdependency among univariate series.

Quality: The quality of the paper is high. The proposal of the three core modules are well motivated. AUEL is for adaptively estimate mean and variance and capture temporal pattern for each series; RD is for robust learning and avoid overfitting when series are similar; Multi-kernel Local Smoothing (MKLS) for capturing dependency among series. The authors have performed experiments on 10 benchmarks which are diverse enough to contain different aspect of multivariate time series. The authors also incorporate different ablations by incorporate one or two modules of ARM. 

Clarity: the overall clarity of presentation is good. It is motivated by the drawbacks of current sota models and address issues one by one by proposing the modules of ARM.

Significance: the paper can be important to time series learning community.

### Weaknesses
I think quality and clarity can be further improved. 

Quality: while the empirical results (Table 1 and 2) show superior performance of the proposed model ARM, it will be better to also demonstrate how significant these results are. The authors claim the proposed approach will not significantly increase computational costs. But such discussion/analysis is not included in the paper. 

Clarity: I found MKLS block in section 3.3 difficult to follow due to notations. The authors introduced many (subscripts and subscripts of X for example); it would be nice to refer to somewhere. 
Table 2 contains large amount of results which are replicate of Table 1. Maybe the authors can eliminate some and find a way to better present them? 
Figure 3 & 4. Due to large amount of information in Figures 3&4, when introducing the components such as MKLS block, I think it will be great to introduce pointers to these components in the figures.   For example \tilde{X}^{*i} need pointers to point to in Figure 3.

### Questions
1.	How does MoE better capture temporal patterns compared to autoregressive and exponential smoothing models? 
2.	What is inverse processing stage? 
3.	The paper mentions causal dependency/ relationship among different series. Could the authors explain in what sense the notion of causality is embedded? 
4.	I am confused about the how Multi-kernel local smoothing helps capture temporal dependencies among different series due to notations. For example, What is X_j. Could the authors elaborate more? 
5.	The paper demonstrates superior empirical performance for forecasting task.  It is hard to see how significant the improvement is compared to other recent models such as PatchTST and DLinear. Do the authors have some insights on the sensitivity of hyperparameters in ARM module in the tuning/training process since many learnable hyperparameters are introduced. 
6.	What are the complexity/runtime compute/memory to incorporate these modules in the vanilla model? How does that compare to PatchTST and DLinear?  
7.	A few discrepancies regarding Table 1. I found in the original PatchTST paper on traffic datasets Table 3: for MAE,  0.249 for L_P = 96, 0.256 for L_P = 192, different from 0.239 and 0.246 in this paper. Could the authors double check if they are consistent?

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The manuscript tackles the multivariate time series forecasting problem, and proposes a solution called ARM.
The proposed method consists of 3 modules, and can be employed to many existing Transformer based time series forecasting models.
Empirical evaluation show that ARM, as well as its 3 modules individually, can improve the forecasting accuracy of the base Transformer model.

### Strengths
The idea of applying the 3 modules to existing Transformer models for time series forecasting is novel, to the best of my knowledge. 

The text of the manuscript is clear and not difficult to understand.

The proposed model has the potential to contribute to the time series modelling community.

### Weaknesses
As mentioned in the Strength part, the text of the manuscript is clear and easy to follow.
However, the figures and tables in the manuscript are very hard to read.
 - Font size of the figures and tables are very small
 - The captions are super long and have a very small font size, I assume this violates the submission guideline.
 - The figures are arranged in an order which is different from how the text refers. Readers have to jump back and forth to find the corresponding (sub-)figure.

### Questions
It is not clear to me how MoE is operated from the manuscript.

Why Vanilla+ARM performs better than applying ARM to other Transformer based models?

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes an enhanced architecture for multivariate time series forecasting using Transformers. The proposed ARM approach incorporates three innovations - Adaptive Univariate Effect Learning (AUEL), Random Dropping (RD), and Multi-kernel Local Smoothing (MKLS). AUEL component introduces learnable exponential moving average instead of classic autoregressive and exponential smoothing for initiating prediction part into the encoder. Random Dropping is almost like an ensemble model which models selected subsets of time series, with aim to reduce spurious patterns among the timeseries. And MKLS which uses one-dimensional convolutional kernels and a channel-wise attention to encapsulate local information. The approach is evaluated on multiple datasets against several competitor approaches.

### Strengths
Consistently overperforms multiple strong competitor approaches on several datasets.
Ablation study is assessing effects of each of the three presented components.

### Weaknesses
Some of the performances in the Table one are reported from their respective papers, so I am wondering if it is likely to assure the same experimental setup. 
Certain statements are presented without appropriate evidence to support the claim. For example 'we introduce ARM, a methodology designed for correctly training multivariate LTSF models.', which is strong claim, and moreover implies that other approaches are 'incorrectly training'.

### Questions
N/A

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
