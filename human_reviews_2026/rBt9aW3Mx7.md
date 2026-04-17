# Complexity- and Statistics-Guided Anomaly Detection in Time Series Foundation Models

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 4

## Abstract
This paper introduces a methodology for anomaly detection in time series using Time Series Foundation Models (TFMs). While TFMs have achieved strong success in forecasting, their role in anomaly detection remains underexplored. We identify two key challenges when applying TFMs to reconstruction-based anomaly detection and propose solutions. 

The first challenge is overgeneralization, where TFMs reconstruct both normal and abnormal data with similar accuracy, masking true anomalies. We find that this effect often occurs in data with strong low-frequency components. To address it, we propose a complexity metric, $\alpha$, that reflects how difficult the data is for TFMs and design a Complexity-Aware Ensemble (CAE) that adaptively balances TFMs with a statistical model. 

The second challenge is overstationarization, caused by instance normalization layers that improve forecasting accuracy but remove essential statistical features such as mean and variance, which are critical for anomaly detection. We resolve this by reintroducing these features into the reconstruction process without retraining the TFMs. 

Experiments on 23 univariate and 17 multivariate benchmark datasets demonstrate that our method significantly outperforms both deep learning and statistical baselines. Furthermore, we show that our complexity-based metric, $\alpha$, provides a theoretical foundation for improved anomaly detection, and we briefly explore prediction-based anomaly detection using TFMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address the issues of overgeneralization and overstationarity in Time Series Foundation Models (TSFMs). To mitigate overgeneralization, it introduces an adaptive weighting mechanism that combines anomaly scores from TSFMs and traditional statistical methods. To alleviate overstationarity, the model integrates mean and variance information into the encoder’s input representation.

### Strengths
1. The paper tackles the overgeneralization issue, which is a very important yet insufficiently addressed problem in anomaly detection using Time Series Foundation Models (TSFMs).
2. The introduction is well-written, clearly presenting the motivation, the research problem, and the methodological approach, making it easy for readers to follow the authors’ reasoning.
3. The experimental results convincingly demonstrate that TSFMs indeed suffer from the overgeneralization problem, providing strong empirical support for the paper’s premise.

### Weaknesses
1. The paper’s notation system is somewhat confusing, and the methodological presentation could be improved. For example, in lines 153–156, several symbols with different subscripts (e.g., P with varying indices) appear without clear definitions, making the section difficult to follow. Moreover, key background concepts such as scale energy and detail energy should be introduced in a dedicated Preliminaries section to help readers better understand the core methodological design.

2. While the paper points out that TSFMs suffer from overstationarization, it lacks further discussion or empirical evidence on how normalization operations affect anomaly detection accuracy.

3. The experimental setup is not fully convincing:
3.1. Since TSFMs exhibit more severe overgeneralization than time-series models specifically designed for anomaly detection, it is essential to compare the proposed CAE method not only with widely used TSFMs (e.g., One-Fits-All, Moment) but also with task-specific anomaly detection models. The current experiments do not sufficiently demonstrate that CAE outperforms existing anomaly detection methods.
3.2. To substantiate the claim that CAE alleviates overgeneralization, the paper should also compare the recall values of different methods (TSFMs and dedicated anomaly detection models such as Anomaly Transformer and DCDetector) under the condition of achieving their respective best F1 scores.

4. It is doubtful whether mainly combining the mean and variance with the input representation in the encoder and optimizing it by reconstruction error can solve the problem of overstationarity. The experiments on this question are also unconvincing as Moment-stat only makes a marginal improvement compared with Moment (FT).

### Questions
1. Could the authors provide both theoretical and experimental analyses to clarify how normalization operations affect anomaly detection performance?

2. I suggest comparing the proposed CAE with TSFMs such as One-Fits-All and Moment, as well as anomaly detection models like Anomaly Transformer and DCDetector, in terms of VUS-PR to more comprehensively evaluate its effectiveness.

3. I recommend adding a comparison of recall values under the condition of achieving the best F1 score for each method, to better demonstrate that the proposed approach effectively mitigates the overgeneralization problem.

4. Could the authors elaborate on the underlying principle of how incorporating mean and variance into the encoder input can effectively address the overstationarization problem?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors propose an anomaly detection method built on time-series foundation models (TFMs). The authors found that when using TFMs (pre-trained for forecasting) in a reconstruction-based anomaly detection setting, the model may overgeneralize. Also, the use of instance normalization layers removes statistical features that are actually useful for anomaly detection. To address these issues, the authors propose a complexity metric  that measures how difficult a segment is for the TFM. Also, a method to reintroduce lost statistical features into the reconstruction process without retraining the TFM is introduced. Comprehensive experiments are conducted on 23 univariate benchmark datasets

### Strengths
- The use of foundation models (TFMs) in time‐series is growing, but the paper identifies important pitfalls when using them for anomaly detection
- The introduction of the complexity metric α and the ensemble approach is novel
- The authors run experiments on 23 univariate time-series datasets

### Weaknesses
- The experiments are only on univariate time series. Many real-world anomaly detection tasks are multivariate. The method’s applicability to multivariate TFMs is unclear. The paper does not sufficiently discuss how the method scales to multivariate or high-dimensional data.
- While the idea of a “complexity” metric is intuitive, the exact definition and properties of α need to be clearly explained. What does “complexity” measure exactly?
- How sensitive are results to the choice of α. 
- Stronger empirical evidence regarding the claim that  "instance normalization (and other normalization) removes statistical features useful for anomaly detection" is needed.
- Reintroducing statistics is consistent across models and domains?
- If the underlying TFM is trained for forecasting, reconstruction may not be the most natural anomaly detection mechanism. Predictio-based anomaly detection might be better in some cases.

### Questions
- How are anomalies labelled? Are there multiple anomaly types per dataset? 
- Are anomalies artificially injected or real events?
- Does the complexity metric α help provide insight and is that interpretably presented?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework that leverages complexity- and statistics-guided ensembling (CAE) to improve the anomaly detection capability of TFMs. The approach aims to mitigate the overgeneralization and overstationarization issues commonly observed in pretrained TFMs by integrating a statistical anomaly detector with foundation-model representations.

### Strengths
* The paper is among the first to systematically investigate the use of time-series foundation models for anomaly detection tasks.
* The proposed complexity- and statistics-guided ensemble framework is well-motivated.

### Weaknesses
* Some claims of effectiveness, particularly regarding the superiority of TFMs within the CAE framework, are insufficiently supported by experiments.
* The evaluation scope is limited to univariate cases and reconstruction-based TFMs.

Please find the detailed comments in the following section.

### Questions
* The ensemble strategy indeed improves performance, but the claim that this improvement demonstrates the effectiveness of TFMs is not fully convincing. It is important to support the argument by comparing against ensembles of two purely statistical methods, thereby isolating the specific contribution of TFMs.
* The paper mentions overgeneralization and overstationarization issues, which also affect forecasting-based TFMs. Why are the proposed methodologies not extended or tested in this setting?
* The discussion on the multivariate case is incomplete. Could the CAE framework be generalized to such case?
* The difference between CAE and CAE-per-data should be more clearly articulated. What causes CAE-per-data to underperform, and what insights does this provide about model generalization?
* How is the ensemble factor w determined? Have the authors conducted a sensitivity analysis to assess its stability across datasets?
* Including more visual examples or case studies would improve readability and help contextualize the motivation behind the method.

### Soundness
2

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
The paper introduces methods which enhance the performance of time series foundation models (TFMs) in two different ways. The first is a direct adjustment on the networks so that global timeseries statistical information (mean, variance) which is lost in normalization steps is re-introduced in the end. The second method, called compexity-aware ensemble (CAE), ensembles the TFM with a statistical methods in an adaptive way to improve performance.

After a short description of the task, the CAE method and the statistical augmenting method applied on the RevIN layer are introduced. The TFM used is MOMENT, which was along the high scoring NN-based methods in the TSB-AD-U benchmark. When statistially enhanced, it is called MOMENT-Stat.

In the experiments section, the effect of CAE is demonstrated on different types of synthetic series. Then MOMENT-stat is compared with the original versions of MOMENT and shown to achieve slightly better scores. Afterwards, MOMENT-stat is fixed as the TFM used and CAE is applied on different statistical methods and shown to consistently increase their performance. An ablation is also present comparing different fixed values of alpha with CAE which results to the best performance. Additionally, alternative methods to weight the ensemble based on entropy are compared with CAE, again yielding lower performance. Finally, on an additional section some forecasting TFMs are briefly studied. It is observed that their forecasting performance is correlated with their anomaly detection performance and finally they are compared with each other and with MOMENT on multi-variate time series.

### Strengths
- The paper is quite readable and the method and goals are clear.

- Descent datasets and evaluation methods where used, something often missing in works on timeseries anomaly detection

- The paper touches different topics on TFMs and their usage in anomaly detection.

### Weaknesses
- It is not very clear from the paper that the CAE method is much better than simply ensembling statistical models with MOMENT. On table 3, one can see that the scores when using the value alpha = 0.5 are relatively close to the scores of CAE. To be convinced that CAE is preferable to simple ensembling one would expect a thorough analysis on how the values of alpha vary across datasets, where they peak and how close to this peak CAE is, compared to constant values of alpha. 

- The relation of alpha to the ensembling weight w is not adequately explained. It is only mentioned that it is an increasing function. It is not clear if it is the same for all datasets or also adaptive or how it is defined. This also raises some more questions about the difference CAE with simple ensembling, as no information is present on e.g. how ensembles with fixed weights w (e.g. w = 0.5) would perform. It could be that those are close to the columns of table 3, but it is not clear. Finally, this makes it also difficult to judge which mixture of MOMENT with the statistical models is tendentially better.

- Though MOMENT-Stat performs slightly better than MOMENT as a standalone, MOMENT with CAE is never compared with MOMENT-Stat with CAE. One could imagine that datasets dominated by low frequency anomaly signal would tendentially also rely on high level statistics, so the enhancement of the TFM might not provide any significant advantage when ensembled. 

- The paper is lacking some focus. The CAE seems to be the central topic, but at the same time the statistical enhancement is introduced and near the end there is a quick study of forecasting TFMs. We feel it would have been better to have focused on CAE and try to support it more thoroughly, e.g. provide cleaner and more thorough definitions, add more details, like the relation of alpha with w, also study simple ensembles and properly compare the to CAE, further study how CAE is behaving on different datasets and picking its weighting.

### Questions
- (More a remark) Section 3.1 is quite messy. A lot of symbols and operations are used without being introduced and details are missing. It would be great to revise and improve it. The usage of intervals and their complements to filter series is nicely used though.

### Soundness
2

### Presentation
3

### Contribution
2
