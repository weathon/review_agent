# SRT: Super-Resolution for Time Series via Disentangled Rectified Flow

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Fine-grained time series data with high temporal resolution is critical for accurate analytics across a wide range of applications. However, the acquisition of such data is often limited by cost and feasibility. This problem can be tackled by reconstructing high-resolution signals from low-resolution inputs based on specific priors, known as super-resolution. While extensively studied in computer vision, directly transferring image super-resolution techniques to time series is not trivial. To address this challenge at a fundamental level, we propose **S**uper-**R**esolution for **T**ime series (SRT), a novel framework that reconstructs temporal patterns lost in low-resolution inputs via disentangled rectified flow. SRT decomposes the input into trend and seasonal components, aligns them to the target resolution using an implicit neural representation, and leverages a novel cross-resolution attention mechanism to guide the generation of high-resolution details. We further introduce SRT-large, a scaled-up version with extensive pretraining, which enables strong zero-shot super-resolution capability. Extensive experiments on nine public datasets demonstrate that SRT and SRT-large consistently outperform existing methods across multiple scale factors, showing both robust performance and the effectiveness of each component in our architecture.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript introduces a framework for time series super-resolution that leverages a decomposition-based approach combined with a rectified flow mechanism. The authors propose an implicit time function to align multi-resolution features along a shared temporal axis, and incorporate a velocity predictor with multi-resolution fusion to generate high-fidelity high-resolution time series. Extensive experiments are conducted across multiple datasets, demonstrating the method’s capability to achieve superior performance compared to several baseline models.

### Strengths
1.	The authors address an under-explored problem in signal processing i.e., time series super-resolution, by proposing a comprehensive framework that explicitly incorporates prior knowledge of the data’s structure.
2.	The idea of decomposing the time series into periodic and trend components and generating these separately is technology sound and aligns with classical signal analysis principles.
3.	Extensive experimental validation across diverse datasets and scales indicates the robustness of the proposed approach. Moreover, the ablation studies effectively highlight the importance of each component.
4.	Also, extending the model to a zero-shot setting via large-scale pretraining shows promising potential for practical deployments.

### Weaknesses
1.	As far as I know, rectified flow has been widely used, the key innovation of your model in time series SR filled remains unclear. 
2.	The evaluation is primarily limited to controlled datasets, the model’s performance on in-the-wild images with diverse and challenging conditions remains underexplored.
3.	The computational complexity and inference speed are relatively high, which may hinder practical application, please analyze and compare the model complexity and resource consumption.
4.	There is a lack of comparison with some recent state-of-the-art SR methods, particularly those based on diffusion architectures. Besides, the ablation needs more elaboration, especially the hyperparameters setting and visualizations.

### Questions
Please refer to the weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes SRT, a novel disentangled rectified flow framework for time series super-resolution (TSSR), along with its large-scale pre-trained variant SRT-large. The work makes a solid conceptual and methodological contribution by adapting rectified flow modeling, which was originally developed for image generation, to temporal data through a structured decomposition and alignment pipeline. The proposed framework demonstrates state-of-the-art results across nine datasets

### Strengths
1. Consistent improvements have been achieved over the baselines (SRDiff, IDM, FTS-Diff, FlowTS).

2. The ablation experiments are extensive and well-designed, quantifying the contributions of each module.

3. The zero-shot generalization capability of the SRT-large model is remarkable, echoing the scaling trends of foundational models in language and vision.

### Weaknesses
1. There are cases where formulas in the experiments lack numbering. Some formulas are complex and difficult to understand, though their principles can be roughly grasped by comparing them with the code. For instance, Formula 3, which lacks a number, could have a more detailed introduction of Vs. The variables c, l, and theta in Formula 3 and Formula 4 should also be described more thoroughly.

2. Although it is claimed that decomposition enhances interpretability, no visualizations or attribution analyses (such as how trend and seasonal components affect the final reconstruction) are presented.

3. While the empirical performance is excellent, the theoretical reasoning behind using rectified flow for continuous temporal domains (as opposed to standard diffusion) could be further elaborated upon. A brief discussion on the temporal smoothness prior induced by rectified flow would strengthen the argument.

4. There is no discussion on its training efficiency, memory usage.

### Questions
1. Does the research in this paper belong to the migration of studies from the field of image super-resolution?

2. It is recommended to conduct visualizations of the trend and seasonality components from the decomposition to highlight the interpretability advantages of the untangling design. 

3. Could annotations be added to the content in the appendix, as it is difficult to establish connections with the main text?

4. Does the improvement in model performance rely on the decomposition assumption? In the text, you mention "low-resolution" and "high-resolution." Does the kernel size of Avgpool have an impact on this process?

5. The method presented in this paper is mostly described in the text as a combination of existing methods. Please point out the differences between it and the existing methods.

6. It is physically unreasonable to infer fine-grained components solely through average pooling without distinguishing between SSR and ASR modules in super-resolution tasks. Please provide an explanation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents Super-Resolution for Time Series (SRT), a method that reconstructs high-resolution time series from low-resolution data. SRT decomposes the input into trend and seasonal components, using rectified flow and cross-resolution attention to generate details. An extended version, SRT-large, offers zero-shot capabilities with large-scale pretraining. Experiments show that SRT outperforms existing methods in accuracy and robustness across multiple datasets and tasks.

### Strengths
The paper presents a fresh approach to time series super-resolution, combining time series decomposition, implicit time functions, and rectified flow. The proposed model is carefully designed, with a clear explanation of its architecture, including the innovative cross-resolution attention mechanism.

### Weaknesses
While SRT outperforms other methods on the tested datasets, it’s unclear how well the model generalizes to other, unseen datasets. Future experiments on more diverse datasets might be necessary to confirm whether the high performance is consistent across varied domains. Including more methods, especially more recent advancements or other domain-specific models, could provide a clearer picture of SRT's performance in a broader context. The paper includes a quantitative comparison but doesn’t showcase qualitative results

### Questions
1.	Could the authors include visual examples of the reconstructed high-resolution time series from both the SRT model and the baseline methods?
2.	The Implicit Time Function (ITF) seems crucial to the model's performance. How does it perform in other types of time series tasks, like forecasting or anomaly detection? Would it be effective for multi-modal time series data?
3.	Due to the complex generative mechanisms introduced by the SRT model (such as the implicit time function and velocity predictor), is the computational complexity during the inference stage reasonable? It would be helpful to include a comparison of computational complexity or inference time.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces SRT, a novel framework for time series super-resolution (TSSR), which aims to reconstruct high-resolution signals from low-resolution inputs. The authors formally distinguish two TSSR subtypes: Sampled Super-Resolution (SSR) and the more challenging Aggregated Super-Resolution (ASR). The core of SRT involves disentangling the target high-resolution details into trend and seasonal components, generating them in parallel via separate rectified flows. Key innovations include an Implicit Time Function (ITF) for continuous temporal alignment and a Cross-Resolution Attention (CRA) mechanism within a decoder-only velocity predictor to fuse multi-resolution information. The authors also present SRT-large, a scaled-up, pre-trained model that demonstrates strong zero-shot TSSR capability. Extensive experiments on nine datasets show that SRT outperforms adapted baselines from image SR and time series generation.

### Strengths
- **Originality:** The work makes several original contributions. It is among the first to formally define and tackle the distinct problems of SSR and ASR in time series. The proposed architecture is a creative and non-trivial synthesis of time series decomposition, implicit neural representations, and modern generative models (rectified flow), moving beyond simple adaptations of image-based methods.

- **Significance:** The ability to perform high-fidelity TSSR, especially for the ill-posed ASR task and in a zero-shot setting, is highly significant for many real-world applications where high-resolution data collection is constrained. The proposed benchmark and clear problem definitions provide a solid foundation for future research.

- **Quality and Clarity:** The paper is well-structured and the method is clearly explained. The experimental design is thorough, evaluating on nine diverse datasets and using both point-wise (MSE) and structural (DTW) metrics. The comprehensive ablation studies and component analysis convincingly validate the design choices. The introduction of a large-scale pre-trained model for zero-shot super-resolution is a notable contribution.

### Weaknesses
- **Computational Efficiency:** While the rectified flow enables faster sampling than diffusion models, the paper does not discuss the overall training or inference cost of SRT and SRT-large. The computational burden of the two-stage training (PD and then reverse diffusion) and the large-scale pre-training for SRT-large could be a practical limitation. A comparison of inference time with baselines would be informative.

- **Limitation of Decomposition Priors:** The method heavily relies on the assumption that time series can be effectively disentangled into trend and seasonal components. While this holds for many periodic signals, its performance on time series with irregular, non-stationary, or abrupt transient patterns is less clear and could be a limitation.

- **Baseline Adaptations:** The baselines are primarily adapted from other domains (image SR) or tasks (imputation/generation). While this is understandable given the nascent stage of TSSR, it would strengthen the paper to include comparisons with a broader range of time-series-specific interpolation or reconstruction techniques, even if simpler, to better contextualize the performance gain.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
