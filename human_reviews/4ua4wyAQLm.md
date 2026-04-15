# Local Patterns Generalize Better for Novel Anomalies

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
Video anomaly detection (VAD) aims to identify novel actions or events which are unseen during training. Existing mainstream VAD techniques typically focus on the global patterns with redundant details and struggle to generalize to unseen samples. In this paper, we propose a framework that identifies the local patterns which generalize to novel samples and models the dynamics of local patterns. The capability of extracting spatial local patterns is achieved through a two-stage process involving image-text alignment and cross-modality attention. Generalizable representations are built by focusing on semantically relevant components which can be recombined to capture the essence of novel anomalies, reducing unnecessary visual data variances. To enhance local patterns with temporal clues, we propose a State Machine Module (SMM) that utilizes earlier high-resolution textual tokens to guide the generation of precise captions for subsequent low-resolution observations. Furthermore, temporal motion estimation complements spatial local patterns to detect anomalies characterized by novel spatial distributions or distinctive dynamics. Extensive experiments on popular benchmark datasets demonstrate the achievement of state-of-the-art performance. Code is available at https://github.com/AllenYLJiang/Local-Patterns-Generalize-Better/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This article is about video anomaly detection (VAD). This paper proposes a framework for recognizing local patterns, which can be generalized to new samples and dynamic modeling of local patterns. This paper proposes image-text alignment and cross-modal attention. Generalizable representations are built by focusing on textual information features that filter out unnecessary differences in visual data. In addition, time motion estimation complements spatial local models to detect anomalies characterized by new spatial distributions or unique dynamics. A large number of experiments have verified the effectiveness.

### Strengths
1. The two-stage training method of this paper is reasonable. And allow for more fine-grained local features.
2. This paper gives a lot of visualizations to make it easier to understand the specific content.
3. The paper has achieved good performance, and the ablation experiment is given.

### Weaknesses
1. Not giving motivation for each part of the method. In my opinion, a good paper should give a specific reason and then introduce the method.
2. The efficiency of the model is worth discussing. You have proposed a lot of model modules. How much more reasoning time will they add to the network?

### Questions
My understanding of this field is unprofessional. So I will further follow the opinions of other reviewers.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel framework for video anomaly detection (VAD) that prioritizes identifying local patterns over conventional global patterns. The authors contend that local patterns generalize better to novel anomalies that were not encountered during training. Their proposed approach follows a two-stage process involving image-text alignment and cross-modality attention to efficiently capture and model local patterns. Additionally, the framework includes a State Machine Module (SMM) to integrate temporal dynamics, enabling enhanced anomaly detection by leveraging both spatial and temporal cues. Experimental results show that this approach achieves state-of-the-art performance on well-established benchmark datasets.

### Strengths
1. The proposed framework is well-structured, with thoughtfully implemented methods.  
2. Experimental results confirm that the approach achieves state-of-the-art performance on established benchmark datasets for video anomaly detection.  
3. The method focuses on fine-grained anomaly features and employs text-image alignment to effectively capture local patterns.  
4. It incorporates Long-range Memory technology, specifically HiPPO, into video anomaly detection.

### Weaknesses
1. The core of the proposed SMM module comes from works like HiPPO, so it seems that the proposed SMM is directly applying these modules to the VAD task. 
2. Both the Image-Text Alignment Module and Cross-Modality Attention Module are based on pre-existing techniques, which limits the methodological innovation.
3. Is the observed performance improvement attributed to the additional large vision-language models, such as Qwen-VL and BLIP2? The comparison may not be entirely fair. It would be beneficial if the authors could provide evidence or experimental results to clarify whether these powerful external models are the primary contributors to the performance gains.
4. The motivation for the work lacks clarity. How do the image-text alignment and cross-modality attention modules achieve “identification of local patterns that are consistent across domains and generalize well”? Additionally, how do they contribute to “generalizing model representations to novel anomalies”?
5. Certain claims may require further validation, such as the statement: “the complementary relation between visual and textual features remains underexplored.”
6. The paper lacks runtime and efficiency analysis. The code introduction is incomplete, and several experimental details are missing, such as the specific version and scale of Qwen-VL used.

### Questions
1. The Global and Local Pattern representations in Figure 1 are hand-drawn, which limits their reliability. Are there any real feature visualization images available instead? Using actual visualizations could better illustrate the motivation and effectiveness of the proposed method, particularly in showing whether it yields more distinguishable local patterns. Figure 1 alone does not provide enough information to convey the method’s motivation and impact.
2. Utilizing Qwen for cropping bounding box regions based on prompts could significantly impact efficiency.
3. Is the introduction of the Qwen-Chat model the primary source of performance improvement? My concern is that the proposed method incorporates numerous external models, and it remains unclear whether these additions are the main contributors to the observed performance gains.
4. Could smaller models be used to replace these large multimodal models? If so, would this result in a significant decrease in performance?
5. Could you provide statistical results on runtime and efficiency? Does the proposed method have a significant impact on operational efficiency?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel framework for video anomaly detection (VAD) that focuses on identifying local patterns to better generalize to unseen anomalies.  The framework employs a two-stage process: first, it uses image-text alignment to locate local patterns that are consistent across visual data variances; second, it refines these patterns using cross-modality attention.  To further enhance the model, the authors introduce temporal clues through a State Machine Module (SMM) and temporal motion estimation.

### Strengths
The proposed two-stage framework for identifying local patterns is novel and well-motivated. The use of image-text alignment and cross-modality attention is interesting and potentially useful.

### Weaknesses
1) The paper lacks a clear discussion of the computational complexity of the proposed framework. Given the use of large language models (LLMs) and other complex modules, it is important to address the efficiency of the approach.
2) What's the role of State Machine Module (SMM) in temporal sentence generation, there need more detailed explanation of the SMM and its role.
3) How does the proposed method handle situations with significant occlusions or viewpoint changes, which are common in real-world surveillance videos?
4) The two-stage process for extracting spatial local patterns using image-text alignment and cross-modality attention is not explained in enough detail. The paper lacks a clear, step-by-step explanation of how these complex processes work.

### Questions
The paper mentions limitations related to the reliance on VLM-based object detectors. How can this limitation be addressed in future work?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel framework for video anomaly detection, aiming to improve generalization for detecting new, unseen anomalies by focusing on local patterns rather than global event patterns. Traditional video anomaly detection (VAD) methods often struggle with unseen anomalies, as they primarily analyze global patterns. This framework utilizes image-text alignment and cross-modality attention to identify and refine local patterns while enhancing them with temporal information. Core components include the Image-Text Alignment Module (ITAM), Cross-Modality Attention Module (CMAM), and State Machine Module (SMM). The proposed approach demonstrates superior performance on several benchmark datasets, suggesting it can generalize better to novel anomalies.

### Strengths
* By using image-text alignment and cross-modality attention, this method successfully extracts local patterns that remain consistent across varying visual data, enhancing its ability to detect novel anomalies.

* The State Machine Module (SMM) and motion estimation integrate temporal clues, effectively strengthening the detection capabilities by including sequential information for more accurate anomaly detection.

*By combining visual and textual features in identifying local patterns, the model benefits from enhanced robustness and accuracy across different visual domains.

### Weaknesses
* The method relies on the detection effect of visual-linguistic modeling (VLM), whereas multi-object image processing may ignore contextual information and affect performance. The authors need to provide more analysis on the ablation of the foundational models.

* The need for multiple layers of modules (e.g., ITAM, CMAM, SMM) to work jointly results in a complex training process that consumes more time and resources. Please provide a comparison of the spatio-temporal complexity analysis with previous methods to demonstrate the practical effectiveness of the method.

* In low-resolution scenes, the generated text description loses detail information, which affects the anomaly detection effect.

### Questions
* How to further improve the generalization of local patterns without relying on visual-linguistic models?

* How does the method ensure adaptability to low-resolution videos in different datasets and real-world application scenarios?

### Soundness
3

### Presentation
4

### Contribution
3
