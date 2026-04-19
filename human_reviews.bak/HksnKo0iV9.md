# TimeBase: The Power of Minimalism in  Long-term Time Series Forecasting

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
Long-term time series forecasting (LTSF) has traditionally relied on models with large parameters to capture extended temporal dependencies. However, time series data, unlike high-dimensional images or text, often exhibit strong periodicity and  low-rank structures, especially in long forecasting horizons. This characteristic can lead many models  focusing on redundant patterns, resulting in inefficient use of computational resources. In this paper, we introduce TimeBase, an ultra-lightweight network with fewer than 0.4$k$ parameters, designed to harness the power of minimalism in LTSF. TimeBase extracts core periodic features by leveraging full-rank typical period representations under orthogonality constraints, enabling accurate prediction of future cycles. Extensive experiments on real-world datasets demonstrate that TimeBase not only achieves minimalism in both model size and computational cost, reducing MACs by 35x and parameter counts by over 1000 times compared to standard linear models, but also wins state-of-the-art forecasting performance, ranking Top1-Top5 in all 28 prediction settings. Additionally, TimeBase can also serve as a very effective plug-and-play tool for patch-based forecasting methods, enabling extreme complexity reduction  without compromising prediction accuracy. Code is available at \url{https://anonymous.4open.science/r/TimeBase-fixbug}.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This article presents a lightweight model requiring only 0.39k parameters, providing a simple periodic-based method for time-series forecasting and good paper writing.

### Strengths
The model exhibits rapid training and inference speeds, and its lightweight nature allows for scalability to edge devices.

### Weaknesses
The article has several problems with novelty as well as experimentation.
1) The a priori assumptions about the period of the data in the paper are too strong, and the frequency of data collection and the period characteristics of the data are not equivalent, in the paper, the authors set P=24 for every hour of data collection. in addition, the paper needs to explain in more detail that in the absence of a priori knowledge, the Fourier series is applied to extract the period, and exactly how to achieve this, if the data is realized in the full time series, then there is data leak, if the data is computed once for each input len length, then it is likely that different values of the period P will be obtained.

2) The experiments did not use the same input length for different models to compare, in the past research, the mainstream experimental comparison of the input length of the time series was set to 96, the article did not provide the reason why such a change was made, and directly set the model inputs to 720 and compare the final prediction performance with other models within the respective paper's input setting parameters.

3) The model's input data and model parameter experiments were conducted with input sequence lengths ranging from 720 to 6480, which is clearly outside the range of parameter settings of previous SOTA models. Such an experimental approach is not representative and is independent of the input sequence lengths used in the main experiment. Input lengths ranging from 96 to 720 are recommended for comparison with mainstream models.

4) For the parameter sensitivity experiments, the article only experiments on input length, and the common batch size (bs), learning rate (lr), and hidden size (H) should also be reported, which is a very basic and necessary experiment.

5) The 336 inputs reported in the experimental appendix do not match the 720 inputs stated in the paper.
There is a discrepancy between the thesis and the code in that the actual early stops in the code are set to 100 rounds, which contradicts the description in the thesis of no more than 30 epochs and 5 rounds of early stops. The experimental design as well as the experimental results are very confusing.

### Questions
1) Extension of the Model to Non-Periodic Data Sets:
   - How can the proposed model be adapted or extended to accommodate data sets that exhibit insufficient periodicity in time?

2) Selection of Periodic Interval (P) Value in Periodic Data Sets:
   - In the absence of prior knowledge or conditions, what methodologies can be employed to determine an appropriate periodic interval (P) value for a given periodic data set?

3) Consistency of Results with Varying Input Lengths:
   - Under the experimental setups utilized in other mainstream studies, specifically when the input length is set to 96, would it be feasible to derive results consistent with those presented in this paper?

4) Rationale Behind Experimental Design Choices:
   - What is the basis for designing the experiment in such a way that only 10 models were selected for comparison, with the performance of the top 5 models being evaluated using an input length of 720, while other models adhered to the settings outlined in their respective papers?

These questions aim to clarify the methodological considerations and experimental design choices made in the study, highlighting potential areas for further exploration and adaptation.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The authors propose a very small and efficient model for long-term time-series forecasting by utilizing the periodicity in many real-world time series. They show that their minimal model (TimeBase) outperforms many competing models.

### Strengths
Long-term time-series forecasting is an important problem and how to solve it efficiently has many real-world implications.

### Weaknesses
I am not impressed by the theoretical results in the paper. The theorems and lemmas seem quite obvious and can be easilly conveyed with plain English.

### Questions
1. What about time series without a clear periodicity (e.g., chaotic systems)?
2. I would love to see tests on some synthetic data, for example, time series with both fast and slow oscillations (e.g., |cos(t)|sin(100t)).
3. For Fig. 5, what about R<6?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This manuscript introduces a lightweight framework designed to generate predictions for MTS data at the segment level using just two linear layers. These segments are composed of specific time steps that represent periods within the MTS data.

### Strengths
This manuscript focuses on improving efficiency in MTS forecasting. The proposed framework comprises only two linear layers, yet it achieves superior efficiency compared to recent state-of-the-art methods. The manuscript provides extensive theoretical analysis to demonstrate the effectiveness of the proposed method.

### Weaknesses
Efficiency improvement is crucial for TimeBase; it requires users to have a solid estimate of the dataset's seasonalities. If these are not known, an FFT is utilized to determine them. Please include the computational cost of the FFT in the efficiency analysis, or provide an explanation for its omission. Including this information would offer a more comprehensive understanding of the model's overall efficiency.

Figure 7 indicates that the "Basis Orthogonal Restriction" has a minimal impact on accuracy as measured by MSE. Are there any benefits not captured by this metric? Including an analysis that discusses the potential advantages of "Basis Orthogonal Restriction" could enhance the contributions of TimeBase.

The accuracy of TimeBase, while adequate, is modest compared to state-of-the-art methods. TimeBase's primary contribution lies in its efficiency. However, according to Table 3, improvements in memory usage and epoch time are modest compared to those achieved by Dlinear and iTransformer. Please provide a detailed analysis discussing the advantages of TimeBase. Additionally, to ensure a fair comparison, could you set the look-back window length across all methods the same?

### Questions
(a) Please consider improve the quality of some figures. 

(b) Could you explain the effectiveness of orthogonal constraint? Figure 7 shows the performance improvements are trivial for all horizons of four datasets

(c) Could you include Dlinear in Figure 4?

(d) Section 3 describes the workflow for univariate time series data. Including an explanation of how to apply TimeBase to MTS data would clarify and enhance the workflow description.

(e) Does TimeBase support multiple seasonality? Such as Traffic could show daily and weekly periods.

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
5

### Summary
This submission presents TimeBase, an ultra-lightweight network for long-term time series forecasting (LTSF). Traditional LTSF models often use a large number of parameters to capture temporal dependencies. However, the authors argue that time series data usually has periodic and low-rank structures and doesn't require so many parameters. TimeBase extracts core periodic features by leveraging full-rank typical period representations under orthogonality constraints to achieve accurate predictions. Experimental results show that TimeBase achieves minimalism in both model size and computational cost.

### Strengths
1. The authors propose a new lightweight time series forecasting model, especially based on the decomposed of basis and period

### Weaknesses
1. The proposed network is extremely light, but the significant difference between MLP and MLP-mixer work could be discussed.
2. The design of orthogonal constraints lakes direct motivation. Although the authors claim the orthogonal term could enhance the capturing of essential temporal patterns. However, the evaluation of the experiments (Figure 6) indicates that it may not be a successful component from $\lambda = 0$.
3. section  4, namely theoretical analysis, is not a serious mathematical proof. The overall process is more like a derivation than a complete proof. The primary support for Lemma 1 and Lemma 2 is finished by observation and assumption.
4. For a lightweight model, much further prediction length must be analyzed to demonstrate its real effects on LSTF.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
