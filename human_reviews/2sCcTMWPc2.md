# TimelyGPT: Recurrent Convolutional Transformer for Long Time-series Representation

- Avg Score: 5.50
- Decision: Reject
- Scores: 5, 6, 6, 5

## Abstract
Pre-trained models (PTMs) have gained prominence in Natural Language Processing and Computer Vision domains. When it comes to time-series PTMs, their development has been limited. Previous research on time-series transformers has mainly been devoted to small-scale tasks, yet these models have not consistently outperformed traditional models. Additionally, the performance of these transformers on large-scale data remains unexplored. These findings raise doubts about Transformer's capabilities to scale up and capture temporal dependencies. In this study, we re-examine time-series transformers and identify the shortcomings of prior studies. Drawing from these insights, we then introduce a pioneering architecture called Timely Generative Pre-trained Transformer (TimelyGPT). This architecture integrates recurrent attention and temporal convolution modules to effectively capture global-local temporal dependencies in long sequences. The relative position embedding with time decay can effectively deal with trend and periodic patterns from time-series. Our experiments show that TimelyGPT excels in modeling continuously monitored biosignal as well as irregularly-sampled time-series data commonly observed in longitudinal electronic health records. This breakthrough suggests a priority shift in time-series deep learning research, moving from small-scale modeling from scratch to large-scale pre-training.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes TimelyGPT, a new transformer architecture for long-time series representation and modeling. For the embedding, xPOS embedding segments long sequences and uses exponential decay based on relative distances to enable extrapolation. For the representation, recurrent attention retention mechanism models global context by integrating short and long-term memory; convolution modules applied on inputs (tokenizer) and hidden states capture local interactions; and ecoder-only autoregressive design allows generating representations for future unseen timesteps. The authors demonstrate the effectiveness of TimelyGPT in modeling both continuously monitored biosignal data and irregularly-sampled time-series data, which are common in longitudinal electronic health records. They advocate for a shift in time-series deep learning research from small-scale modeling from scratch to large-scale pre-training.

### Strengths
Originality:

The integration of relative position modeling, recurrent attention, and convolutions for transformers is novel and creative.

Applying these architectures specifically for long-time series modeling is an original contribution.

The idea of decoding future representations and extrapolation is innovative.

Significance:

The work moves transformer research for time series modeling in an important new direction.

The extrapolation capabilities and integrated components can positively impact many applications.

The ability to handle long irregular sequences has high relevance for domains like healthcare.

Clarity:

The paper is clearly structured with the techniques, experiments, results, and analyses presented in a logical flow.

The writing clearly explains the intuition and technical details of the methods.

Tables and figures help summarize key quantitative results.

### Weaknesses
Analysis of the computational overhead and scaling behavior compared to other transformers could be expanded.

Only one large-scale time series dataset (Sleep-EDF) was used for pre-training experiments. More diverse pre-training data could help. 

For the classification tasks, comparison to a broader range of datasets and architectures like RNNs would be useful. For the forecasting, what's the performance for the ETT, weather and electricity dataset used in the PatchTST paper.

Theoretical analysis or frameworks to relate the different components to time series properties is limited.

Hyperparameter sensitivity analysis could provide more insights into optimal configurations.

Evaluating true few-shot generalization with limited fine-tuning data could better demonstrate transfer learning benefits.

The focus is on a decoder-only architecture. Encoder-decoder and encoder-only architectures could also be analyzed.

Societal impacts regarding potential misuse of forecasting need to be considered.

### Questions
While this paper explores an intriguing aspect of time-series analysis—specifically, modeling irregular time-series—it is not without its shortcomings, particularly in the design of experiments meant to objectively situate this work within its field. Please refer to the 'weaknesses' section for a detailed list of concerns raised by the reviewer. The reviewer would be pleased to revise their score upwards provided these issues are adequately addressed.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper first revisits time series transformers and identifies the shortcomings of previous studies. Based on the findings, this paper introduces a Timely Generative Pre-trained Transformer (TimelyGPT), which combines recurrent attention and temporal convolution modules to capture global and local temporal dependencies in long sequences. TimelyGPT is comprised of three major components: xPOS embeddings, retention module for irregularly sampled time series and a convolution model for local interaction.

### Strengths
1. This paper introduces an interesting TimelyGPT for long time series representation learning. A major advance is that TimelyGPT has a superior performance on ultra-long time series forecasting.

2. The proposed architecture seems technical sound. xPOS and retention module are capable of modeling long time series and the convolution module could effectively capture local information.

3. Experimental results on several larger-scale datasets show that the proposed TimelyGPT could outperform SOTA baselines.

### Weaknesses
1. Most of the empirical insights (section 2) for using Transformers for time series presented by this paper are kind of repeating the findings of previous works [1][2].

2. In the ablation studies, this paper only presents the results of classification and the results for forecasting tasks are not provided. 

3. The model architecture is a little bit incremental. xPOS embeddings and retention module are simple extensions of previous works.

[1] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting?, 2022. 

[2] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation, 2022.

### Questions
1. How do you calculate the dataset size?

2. How does TimelyGPT perform if pre-training is removed?

3. For ultra-long-term forecasting (section 4.2), 

3.a. How many parameters do baselines have? 

3.b. Are the baselines pre-trained? How do you pre-train baselines?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a Transformer approach for time series modelling. The main components of the proposed architecture are relative positional embeddings that extract both trend and periodic patterns, recurrent attention with time decay and convolutional module that captures local temporal information.

### Strengths
The paper is well written and easy to follow. Some of the aspects of the proposed model appear novel although it is hard to pinpoint exactly which parts are novel as most components such as for example xPOS are based on previous work. Strong empirical performance particularly on long range prediction with large data (e.g. Sleep-EDF). Detailed ablation study and empirical evaluation demonstrating the benefits of each added component and the impact of data and model size.

### Weaknesses
Authors emphasise the focus on pre-training. However, I could not find results ablating the benefits of pre-training and whether TimelyGPT is particularly well suited for it. Some experimental settings are odd where models are pre-trained on different datasets for classification vs regression. Task specific pre-training deviates from the foundational model paradigm, why was that done?

Figure 3.a results look almost too good to be true, 720 vs 6K prediction window size is a huge difference and one would expect at least some performance decay. At what prediction window size do you start seeing performance decay and what in TimelyGPT makes it so well suited for such long range prediction?

Some components like the temporal convolution module have very specific sets of operators. How sensitive are the results to these choices? I think the paper would also benefit from clearly highlighting novel components vs novel application of previous work.

### Questions
Please see the weaknesses section.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces TimelyGPT, a pre-trained model designed specifically for time series. This model integrates recurrent attention and temporal convolution modules to capture dependencies in long sequences. The recurrent attention leverages the time decay mechanism to handle continuous and irregular observations. Besides, a relative position embedding is introduced to the transformer to help extract dependencies.

### Strengths
The summary provided in Table 1, which compares related baselines in terms of the number of parameters and dataset size, is informative and useful for gaining a quick understanding of the landscape of existing approaches.

### Weaknesses
- As mentioned in Section 1, it is suggested that *"challenges observed in small-scale time series might stem from overfitting limited data rather than inherent flaws in transformers"*. This statement is a bit confusing as the title suggests that the model can handle long time series. However, it seems that the goal is to address the overfitting to limited data, which implies that only short-time series are available. This appears to be contradictory. Please clarify what problems in time series applications are mainly addressed here. 

- Table 1 can be further improved. For example, it is unclear what is meant by 'data size.' Is it the number of time points or the number of windows? And what are the numbers of the proposed model? 

- The writing could be further improved. The formulas in section 3.2 are difficult to follow, and it would be helpful to introduce each formula one by one in a logical manner.

- The proposed TimelyGPT refers to the Timely Generative *Pre-trained* Transformer. However, details of the *pretraining* and fine-tuning processes are missing. Specifically, what datasets and how many epochs were used for pretraining? What fine-tuning strategies were used for downstream tasks? 

- There are many related works (e.g., TS2Vec, TimesNet) on time series self-supervised representation learning, but there is a lack of systematic investigation, discussion, and comparison. 

- Regarding the prediction results in Figure 3, it should be noted that the dataset is not commonly used for forecasting. 
Additionally, a too-long prediction horizon may not make sense as real-world prediction scenarios are usually dynamic and unpredictable.

- It is recommended to summarize the prediction results in a table rather than an overly simplified figure. This makes it easier for other works to follow.

- Visualization results are encouraged to be included.

- The use of a relative position embedding is not a new idea and it has been studied in different communities, e.g., transformer variants and time series applications. For example, 
> [ref] STTRE: A Spatio-Temporal Transformer with Relative Embeddings for Multivariate Time Series Forecasting  
[ref] Improve transformer models with better relative position embeddings

- If the proposed recurrent attention can handle irregular time series well, it would be beneficial to compare it with popular irregular time series methods. 
> [ref] Multi-time attention networks for irregularly sampled time series

- The source code is incomplete.

### Questions
Please see the Weaknesses.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
