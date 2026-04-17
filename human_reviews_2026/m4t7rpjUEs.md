# DCTS: Fusing Discrete and Continuous Information for Time Series Forecasting

- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
In time series analysis, data is usually treated as a set of continuous values. Conventional methods do all the computations, from inputs to outputs, in continuous form. While this continuous representations are highly expressive, it can also pay too much attention to fine-grained details. This risks introducing noise and overlooking critical information. In contrast, discrete representations can assign a single code to each temporal pattern. In this way, key patterns underlain in the data are extracted more effectively, while some informative details are probably filtered out along with noises. In order to combine the advantages of both, we propose fusing discrete and continuous information for time series forecasting (DCTS), that incorporates both continuous and discrete approaches, it fuses the expressive power of continuous encoding and pattern-abstracting ability of discrete encoding. It uses a codebook learned via vector quantization to extract discrete encoding from the time series and then fuses it with the continuous encoding. In doing so, the model can benefit from the strengths of both continuous and discrete representations. Additionally, we use multiple codebooks to encode the time series. A single code can hardly cover the entire feature space of a time series. In contrast, multiple discrete values can be combined, exponentially expanding the encoding space and achieving much stronger expressive power. We evaluated our proposed method on multiple real-world datasets and achieved the best performance compared to the baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a time series forecasting method that combines discrete encodings with continuous encodings. The contributions of this paper are as follows:
1. Proposes fusing discrete and continuous information for time series forecasting (DCTS)
2. Attempts to replace single codebook with multiple codebooks to encode time series.
The proposed method outperforms baseline methods on multiple real-world datasets.

### Strengths
Originality: This is the first work trying to combine the advantages of both continuous and discrete encodings in time series forecasting. Though the idea is kind of straight, this paper still deserves originality.
Quality: Overall the solution is clearly motivated and reasonably implemented, and corresponding evaluations are comprehensive.
Clarity: The paper is easy to read, the results are easy to understand.
Significance: This study integrates continuous and discrete encoding characteristics, achieving improved performance compared to current purely continuous-encoding time series prediction models.

### Weaknesses
1. The performance comparison could be strengthened by including a more contemporary set of baseline models. The selection appears somewhat conservative, with only one model (SOFTS) published after 2024. While the ablation studies effectively demonstrate the benefit of combining continuous and discrete encodings over either alone, the primary goal remains advancing time series forecasting performance. Therefore, comparisons with other recent open-source works (e.g., CycleNet) would be more compelling. Furthermore, to more directly validate the superiority of the proposed hybrid approach, a comparison against a strong discrete-based baseline—such as by adapting SDFormer to the time series forecasting task—would be highly informative.
2. The manuscript contained a number of grammatical issues and non-idiomatic expressions that hindered understanding. I strongly recommend thorough language polishing to ensure the writing's clarity matches its technical quality.

If my above concerns are resolved, I would consider increasing my rating.

### Questions
My questions are listed above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces an innovative time series forecasting model, DCTS (Discrete and Continuous Time Series Fusion), which combines discrete encoding and continuous encoding to enhance both pattern abstraction and numerical expression abilities.
Traditional time series models typically focus solely on continuous features, ignoring the underlying discrete patterns in the data (such as workday/weekend or peak/off-peak characteristics). This can lead to overfitting to noise or neglecting key patterns. DCTS addresses this issue by introducing vector quantization (VQ) into continuous models, leveraging a multi-codebook structure to generate multi-dimensional discrete codes, significantly enhancing the feature space’s expressiveness. The model consists of Discrete Module and Continuous Module. Experiments on multiple real-world datasets, including ETT, Weather, Traffic, Electricity, and Solar, show that DCTS outperforms state-of-the-art models (e.g., SOFTS, PatchTST, iTransformer) in terms of both MSE and MAE. Ablation studies confirm the complementarity of the two modules, while contrastive experiments highlight the advantages of the multi-codebook mechanism for capturing complex time series patterns.

### Strengths
1. DCTS is the first model to integrate discrete vector quantization with continuous time series modeling, utilizing multi-codebooks to expand the discrete encoding space. 
2.  DCTS improved MSE by 2.7% and MAE by 1.5% compared to state-of-the-art models, demonstrating its effectiveness and generalization ability
3. The proposed dual-module structure (Discrete + Continuous) clearly separates the responsibilities: the discrete module captures global patterns, while the continuous module focuses on fine-grained changes.

### Weaknesses
1. Except for SOFT, all compared baselines are from 2023 or earlier, lacking comparisons with the most recent works.
2. The performance improvement achieved by the model is minimal and remains lower than that of some more advanced baselines[1,2].
3. There is no analysis of key hyperparameters. In addition, the input length is fixed at 96, but many of the compared baselines (such as PatchTST and DLinear) are generally not suitable for this setting. Overall, the experimental section is incomplete and insufficient.
4. The core technology mainly consists of a combination of existing methods. Moreover, Vector Quantization has already been explored in prior studies [3,4]. Therefore, the technical innovation of this paper is limited.
5. Since the authors emphasize the advantages of using a codebook, the following comparative experiments should be conducted: replace the codebook used in the proposed model with traditional patch or adaptive patch techniques [5], or alternatively, use simple temporal and variable encoding methods [6].
6. There is a lack of visualization for the codebook, making it impossible for reviewers to determine whether issues such as dead codes exist.
7. Since the authors emphasize that the proposed method can help the model recognize certain pattern characteristics of data across different time periods, and the example used is traffic data, corresponding experiments should be conducted to validate this motivation. As shown in paper [7], can the method accurately predict sudden events in datasets such as PEMS or METR-LA?
8. In summary, I believe that the experimental results presented in the current version are insufficient. The proposed method is neither solid nor innovative. I recommend that the authors conduct a comprehensive revision, particularly by adding experiments that validate the motivation and by including comparisons with more advanced baselines.

[1]  TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables

[2] Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling

[3] Sparse-vq transformer: An ffn-free framework with vector quantization for enhanced time series forecasting

[4] Fusionsf: Fuse heterogeneous modalities in a vector quantized framework for robust solar power forecasting

[5] Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective

[6] Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting

[7] Hutformer: Hierarchical u-net transformer for long-term traffic forecasting

### Questions
See weakness

### Soundness
2

### Presentation
2

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
This paper advocates emphasizing discrete representations for time series, extracting discrete features via Vector Quantization (VQ), and introducing a multi-codebook (multi-vocabulary) mechanism. Through comprehensive experiments, it demonstrates the importance of fusing discrete and continuous representations for time series forecasting.

### Strengths
1. The manuscript proposes a method based on Vector Quantization (VQ) method that fuses discrete and continuous features, thereby improving its performance.
2. DCTS achieves state-of-the-art results by extensive experiments on many real-world datasets, demonstrating the effectiveness and validity.

### Weaknesses
1. One of the main innovations—using a multi-codebook for representation to enhance the representation ability. Existing work on representation learning has thoroughly explored this approach, such as RVQ [1] and Group-Residual VQ [2].
2. Regarding the claimed advantages of discrete encoding over continuous encoding in the introduction: in practice, periodic encodings (e.g., the sinusoidal positional encodings commonly used in Transformer models) already ensure that Monday follows Sunday, avoiding boundary issues. Moreover, many models assign an individual embedding to each time step for time encoding, which is also unordered and does not induce large artificial distances between adjacent days.
3. The experiment is somewhat weak. It lacks the results of the latest baselines of time series forecasting in 2025. Besides, in contrastive analysis, the manuscript omits hyperparameter sensitivity analyses, especially the impact of the number of codebooks. Meanwhile, the contrast results for add VQ can be more abundant, including the results of other baselines and datasets. 
4. In the ablation study, removing the entire CI or CD module makes it unclear whether the encoder–decoder and frequency-domain components are driving the performance improvement. To convincingly show the value of the discrete and continuous parts, a core ablation should construct the fused representation X using only the discrete or only the continuous representation, while keeping all other network components within the CI and CD modules unchanged.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes the DCTS model to address limitations of traditional time series forecasting methods relying solely on continuous or discrete representations. Continuous methods are noisy and prone to numerical errors, while single-codebook discrete methods fail to cover full feature spaces. DCTS integrates a multi-codebook vector quantization-based Discrete Module for pattern extraction and a Continuous Module (with Fourier Transform and low-pass filtering) for noise reduction and information fusion. Experiments on 8 real datasets (e.g., ETTm1, Traffic) show it outperforms 8 baselines (e.g., iTransformer), with ablations confirming dual-module necessity and multi-codebooks’ superiority over single ones.

### Strengths
1. **Fuses dual representations for balanced performance**: DCTS combines continuous encoding’s expressive power (capturing fine-grained details) and discrete encoding’s pattern-abstracting ability (extracting key temporal patterns), avoiding pure continuous methods’ noise sensitivity and pure discrete methods’ detail loss
2. **Expands discrete space via multi-codebooks**: Unlike single-codebook vector quantization, it uses multiple independent codebooks to exponentially expand the encoding space (like combining "letters" into "words"), covering complex time series features without increasing codebook/vector size
3. **Reduces noise in continuous processing**: The Continuous Module uses Fourier Transform and low-pass filtering to remove high-frequency noise, focusing on critical trends and stabilizing predictions 
4. **Validated effectiveness and generalizability**: It outperforms 8 baselines (e.g., iTransformer) on 8 real datasets (energy, traffic, etc.), and its Discrete Module can enhance other continuous models (e.g., 3.5% MSE gain for PatchTST)

### Weaknesses
1. DCTS relies on Euclidean distance for similarity measurement in vector quantization. When time series samples are phase-shifted, this distance metric tends to increase, leading to incorrect discrete code assignment and thus affecting the accuracy of discrete pattern extraction.
2. Although DTW is mentioned as a potential solution to handle phase-shifted samples by focusing on time series shape similarity, DCTS has not addressed the trade-off between DTW's effectiveness and its much lower computational efficiency compared to Euclidean distance.
3. While DCTS is validated on 8 real-world datasets, these datasets do not cover extremely complex time series scenarios (e.g., high-dimensional data with severe missing values or sudden abnormal fluctuations), leaving its adaptability in such scenarios unproven.

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
