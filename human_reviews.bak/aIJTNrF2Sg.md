# Frequency-Conditioned Diffusion Models for Time Series Generation

- Decision: Reject
- Scores: 5, 3, 5, 5

## Abstract
Time series data, commonly used in fields like climate studies, finance, and healthcare, usually faces challenges such as missing data and privacy concerns. Recently, diffusion models have emerged as effective tools for generating high-quality data, but applying them to time series is still difficult, especially for capturing long-range dependencies and complex information. In this paper, we introduce a new diffusion model that uses frequency domain information to improve time series data generation. In particular, we apply Fourier analysis to adaptively separate low-frequency global trends from high-frequency details, which helps the model better understand important patterns during the denoising process. Finally, our approach uses a specialized frequency encoder to integrate this information, enhancing the model's ability to capture both global and local features. Through exhaustive experiments on various public datasets, our model shows an impressive performance in generating time series data for diverse tasks like forecasting and imputation, outperforming existing methods in accuracy and flexibility.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a new diffusion model that uses frequency domain information to improve time series data generation, and the proposed model outperforms some advanced methods in accuracy and flexibility.

### Strengths
1. The paper's introduction of frequency domain information into the time series diffusion model exhibits a certain degree of innovation.

2.The method proposed in the paper achieves superior performance on series generation tasks.

3.The paper is understandable in most parts.

### Weaknesses
1.Despite the existence of some related research, I remain uncertain about the practical applications of unconditional time series generation. I think that newly generated time series data is far less valuable than text or images produced by other generative tasks. Therefore, I place greater importance on conditional generation tasks such as forecasting and imputation. However, the experiments related to forecasting or imputation in the paper are not sufficiently comprehensive. The paper should evaluate and compare more advanced forecasting or imputation models, such as iTransformer and TimesNet, on a broader range of datasets.

2.The description of the method is not very clear. It is suggested that the authors use a pseudocode format to describe the training and inference processes of the proposed algorithm. 

3.The paper states that “in the inference step, frequency information is directly extracted from the dataset and utilized”. Could this lead to information leakage?

The paper includes some statements that appear rather definitive without providing detailed explanations or citing sources. For instance: 1) those approaches make it nearly impossible to capture all time-related information, particularly in long-range sequences, making it a highly challenging problem；2) generative models struggle to effectively model the temporal-spatial-spectral information inherent in multivariate time series data, particularly during the denoising process, where capturing this complex information proves to be challenging.

### Questions
1.During the reverse process of the training phase, we restore the noisy series XT back to the original series X0. The frequency domain information of the noisy series XT is obviously of little value. Could the authors clarify how the proposed method specifically applies frequency domain information? Similarly, during the inference stage, extracting frequency domain information from the noisy series would be unhelpful, whereas extracting it from the original series would raise concerns about information leakage.

2.How does the forecasting performance of the model compare to the SOTA models, such as iTransformer and PatchTST?

3.How does the training and inference efficiency of the proposed method compare to other methods, such as Diffusion-TS?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a frequency-conditioned diffusion model to conduct time series generation tasks including forecasting and imputation. The author believes that previous methods did not fully utilize  frequency domain information, leading to poor modeling of low-frequency signals such as trends. As a result, the proposed method utilize FFT to extract frequency domain information and models low-frequency and high-frequency information separately during denoising. The authors claim that the proposed method achieves state-of-the-art performances under multiple evaluation settings.

### Strengths
1. The network design for joint utilization of time and frequency domain information is reasonable. The authors uses a cross attention structure for frequency information modeling, which is scalable and reasonable.

2. The energy-based adaptive frequency selection strategy is novel. The authors determine the division of low-frequency and high-frequency parts by spectral energy. This principle is transferable across different data sets and holds certain value for subsequent research based on frequency domain information.

### Weaknesses
1. The settings of most experiments are not well explained. For example, in the generation task, how is the noise sampling conducted? Is there a reasonable control to ensure sampling consistency across different methods? 

2.The authors did not provide detailed descriptions of how the proposed method performs conditional generation such as forecasting and permutation. The formulation in Sec. 4 only describes unconditional generation.

### Questions
**Information leakage**: The frequency information used by denoising model is extracted from the original sample 
$x_0$ rather than the noisy $x_t$ (as seen in line #241 of diffusion.py), which undoubtedly cause information leakage.  Consequently, The experimental results presented in  this article are meaningless.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a diffusion model for generating time series data. The model improves upon existing methods by incorporating frequency domain information, specifically separating low-frequency global trends from high-frequency details. This separation allows the model to better capture important patterns during the denoising process. The model then uses a specialized frequency encoder to integrate this information, improving the model's ability to capture both global and local features. The paper evaluates the model's performance on several public datasets and compares it to existing models, demonstrating its effectiveness in generating time series data for diverse tasks such as forecasting and imputation. The paper is well-structured and easy to understand.

### Strengths
(1) The proposed generation diffusion model operates in the time domain while concurrently leveraging frequency domain information as a priori knowledge, allowing to capture and represent the data's characteristics by combining information from both domains during the diffusion learning process.
(2) The adaptive dissects of frequency information based on the power spectrum enables the model to represent the temporal aspects embedded within the data, leading to the synthesis of high-quality data that more accurately reflects the trends and patterns present within the dataset.
(3) Experiments across multiple datasets showcase the model's versatility across diverse time series generation challenges where it achieves superior performance in long-range data generation, imputation and forecasting tasks.

### Weaknesses
(1) While incorporating frequency domain information and a Transformer-based decoder can enhance model performance, it also introduces additional complexity and computational overhead compared to simpler approaches. A thorough comparative analysis with baseline models is necessary to assess the trade-off between performance gains and increased computational cost.
(2) A more detailed explanation of the frequency encoder's architecture would enhance the paper's clarity. While the use of a 1D convolutional encoder is noted, a deeper discussion on the reasons for selecting this specific architecture and its suitability for capturing frequency-domain information would be valuable.
(3) While the use of a threshold parameter (\gamma) to separate low and high-frequency components based on spectral density is a reasonable approach, a more detailed analysis is needed to assess the sensitivity of the model's performance to variations in this parameter.

### Questions
(1) The authors assert that exclusive reliance on frequency domain modeling is suboptimal due to the loss of temporal information. However, this claim is contradicted by Diffusion-TS, which effectively leverages Fourier basis functions to capture the crucial seasonal component of time series. Furthermore, Carabbe et al. [Time Series Diffusion in the Frequency Domain. (ICML 2024)] empirically demonstrate that frequency diffusion models exhibit superior performance in capturing the training distribution compared to their time-domain counterparts. This is attributed to the inherent localization of time series data in the frequency domain, rendering it more amenable to modeling in this space. To strengthen the core argument of this work, a more detailed elaboration on this point is essential.

(2) The majority of baseline models employed in the comparative analysis are time-domain based, limiting the scope of direct comparison with purely frequency-domain diffusion models. Additionally, the ablation study lacks a variant that exclusively operates in the frequency domain. This would help to show if combining both ways is really better than using each one separately.

(3) In the inference phase, how do the randomly initialized noise samples acquire the necessary frequency information to generate diverse and realistic samples? Moreover, how can we ensure that the generation process covers the entire spectrum of frequencies present in the target data distribution?

(4) In Section 4.1, the authors posit that their analysis reveals a concentration of information primarily in the first frequency component, with subsequent frequencies exhibiting a more even distribution of lower density. However, this claim lacks empirical validation. To substantiate this assertion, the authors should conduct experiments with varying values of γ to demonstrate the impact on model performance. 

(5) How does the utilization of a Transformer decoder contribute to the model's performance? Could a simpler encoder-only architecture suffice to reduce computational complexity?

(6) In Table 1, what is the reference point for the "original" values? While the proposed model exhibits comparable performance to Diffusion-TS across most datasets, it demonstrates a significant improvement on the fMRI dataset. Is there any explanation to this?

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
This paper introduces a novel diffusion model for time series generation that combines both time-domain and frequency-domain information. The model leverages frequency as prior knowledge, allowing it to capture both global and local patterns more effectively during the diffusion process. A key contribution is the introduction of a signal energy-based adaptive frequency selection module, which separates low and high frequency components based on spectral density to better represent temporal features. While the experiments conducted on several public datasets indicate promising results in tasks such as long-range data generation, imputation, and forecasting, the experimental validation could benefit from further solidification across more diverse datasets and conditions.

### Strengths
The paper presents a novel approach by integrating frequency-domain information as prior knowledge into a diffusion model for time series generation. This is an innovative idea that addresses limitations in capturing both global trends and local details within time series data. A key strength lies in the signal energy-based adaptive frequency selection module, which intelligently separates low-frequency components for global trends and high-frequency components for semantic details, based on spectral density. This thoughtful and systematic design enhances the model's ability to effectively balance time-domain and frequency-domain information during the denoising process, making the approach both conceptually strong and technically well-crafted. The careful decomposition and integration of frequency components reflect a deep understanding of the challenges in time series generation, providing a promising direction for future research in this field.

### Weaknesses
1.	Unclear Problem Definition:
The paper does not clearly define the inputs and outputs for each task, such as prediction and anomaly detection, which makes it difficult for readers to understand the exact setup and application of the model across different tasks. For instance, the specific requirements and challenges for each task are not sufficiently explained, leaving readers unclear on how the proposed approach addresses them.
2.	Choice of Datasets in Generation and Prediction Tasks:
In the generation task, the selection of datasets such as Sines and MuJoCo is problematic, as some evaluation metrics on these datasets yield extremely low values (e.g., 0.001), which may undermine the reliability and interpretability of the results. Additionally, for the prediction task, only two datasets are used, which is too limited to adequately assess the model’s predictive capabilities across different settings.
3.	Outdated Baselines:
Some of the baselines used in the main experiments, such as TimeGAN and TimeVAE, are relatively outdated and may not represent the current state of the field. These older models are less competitive compared to more recent approaches, which diminishes the persuasive power of the experimental comparisons. Including stronger, more recent baselines could significantly enhance the validity and rigor of the results.
4.	Ablation Study
Although the paper performs a thorough ablation study, some components appear to have minimal impact on performance (as shown in Table 3). For example, removing certain components (e.g., high-frequency or adaptive frequency modules) does not substantially degrade the results, raising questions about their necessity. A more targeted ablation analysis focusing on the essential components would make the findings more concise and impactful, avoiding the impression of redundant complexity.

### Questions
1.	Could you provide a clearer definition of the tasks, specifying the input and output for each (e.g., anomaly detection, prediction)?
The current description of tasks lacks clarity on the exact inputs and outputs, making it difficult to understand the model's specific application to each task. For instance, a more explicit definition of input-output pairs for prediction and anomaly detection would be helpful.
2.	Could you comment on the choice of datasets for the generation and prediction tasks?
In the generation task, certain datasets like Sines and MuJoCo yield extremely low evaluation metric values (e.g., 0.001), which may undermine the reliability of the results. For the prediction task, only two datasets were used, which may not be sufficient to comprehensively evaluate the model’s predictive capabilities. Expanding on the dataset choices would improve confidence in the findings.
3.	Why were certain outdated baselines chosen for the main experiments, and could you provide comparisons with more recent methods?
Some baselines, such as TimeGAN and TimeVAE, are relatively outdated and may not provide a rigorous benchmark. Including more recent models could provide a stronger validation of the proposed method’s performance and strengthen the comparisons.
4.	In the ablation study, could you elaborate on the necessity of each component tested?
Certain ablation components, such as “w/o high frequency” and “w/o adaptive,” appear to have minimal impact on the results. A more detailed justification for including these components, or focusing only on the critical ones, would make the ablation study more concise and impactful.

### Soundness
2

### Presentation
2

### Contribution
2
