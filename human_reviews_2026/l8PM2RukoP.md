# Kaleido: Open-Sourced Multi-Subject Reference Video Generation Model

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
We present Kaleido, a subject-to-video (S2V) generation framework, which aims to synthesize subject-consistent videos conditioned on multiple reference images of target subjects. Despite recent progress in S2V generation models, existing approaches remain inadequate at maintaining multi-subject consistency and at handling background disentanglement, often resulting in lower reference fidelity and semantic drift under multi-image conditioning. These shortcomings can be attributed to several factors. Primarily, the training dataset suffers from a lack of diversity and high-quality samples, as well as cross-paired data, i.e., paired samples whose components originate from different instances. In addition, the current mechanism for integrating multiple reference images is suboptimal, potentially resulting in the confusion of multiple subjects. To overcome these limitations, we propose a dedicated data construction pipeline, incorporating low-quality sample filtering and diverse data synthesis, to produce consistency-preserving training data. Moreover, we introduce Reference Rotary Positional Encoding (R-RoPE) to process reference images, enabling stable and precise multi-image integration. Extensive experiments across numerous benchmarks demonstrate that Kaleido significantly outperforms previous methods in consistency, fidelity, and generalization, marking an advance in S2V generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Kaleido, a multi-subject reference video generation model， aiming to address the shortcomings of existing S2V models in maintaining multi-subject consistency and background disentanglement. Its core innovations include: constructing a dedicated training data pipeline with low-quality sample filtering and cross-paired data synthesis, and proposing Reference Rotary Positional Encoding (R-RoPE) to achieve stable and accurate multi-image integration. Experiments show that Kaleido significantly outperforms existing open-source models in key metrics such as subject consistency  and background disentanglement and is comparable to closed-source models

### Strengths
- R-RoPE is an effective strategy:  Isolating reference images only in the temporal  dimension is indeed unreasonable. This approach is prone to confusing the model between reference image information and the original video generation sequence during video generation, as the model may misinterpret reference images as consecutive frames in the video. Therefore, it is necessary to additionally introduce distinctions in the width (W) and height (H) spatial dimensions.

### Weaknesses
1. Lack of innovation in the data pipeline: In related methods such as Conceptmaste and  Phantom-data [1,2], in-depth explorations have been conducted on cross-paired strategies for synthetic data and real data. 
2. Lack of experimental validation for data pipeline-related methods: The paper only provides an overall comparison of experimental results, it lacks targeted validation for individual key components of the pipeline.

[1] ConceptMaster: Multi-Concept Video Customization on Diffusion Transformer Models Without Test-Time Tuning
[2] Phantom-Data : Towards a General Subject-Consistent Video Generation Dataset

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper propose Kaleido, a fully open-sourced S2V generation model. It includes a scalable data pipeline to collect training data and a lightweighted conditioning schemes that apply R-Rope. It achieves SOTA result among open-sourced S2V models.

### Strengths
- The paper is well-written and easy to understand.
- The proposed data collection pipeline takes into account the cross-paired images, that can solve the background leakage problems during training.
- The proposed R-RoPE is simple but effective to disentangle denoised image from condition.
- The model achieves SOTA results, and it is fully open sourced and faciliate the community.

### Weaknesses
- This proposed framework concatenates tokens but not token-channels, which may make the inference slow. 
- The paper does not discuss why they did not use channel-wise concatenation, which is efficient and widely adopted.
- For R-RoPE, why the t-dim of RoPE for refernces images is not shift-T?
- The paper lacks novelty and is mostly engineer work, but it should be fine.

### Questions
see weaknesses

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
4

### Summary
This paper presents Kaleido, an open-source framework for subject-to-video (S2V) generation that focuses on maintaining multi-subject consistency and background disentanglement. The authors propose (1) a comprehensive data construction pipeline with cross-paired data, filtering, and augmentation; and (2) a novel Reference Rotary Positional Encoding (R-RoPE) for integrating multiple reference images. Experiments show that Kaleido achieves state-of-the-art results on both general video quality metrics and S2V-specific metrics, approaching the performance of closed-source systems like Kling and Vidu.

### Strengths
1.The proposed large-scale, cross-paired data construction process is well-designed and will be valuable for the community.
2. Comprehensive experiments: Evaluation covers humans, objects, and multi-subject settings, with both quantitative and user studies.

### Weaknesses
1. The architectural novelty is limited. The model mainly relies on simple concatenation for conditioning; R-RoPE, while useful, is a modest modification. Besides, its design is mostly empirical without deeper analysis.
2. The validation of proposed dataset is missing. It lacks quantitative evidence for dataset diversity and annotation accuracy, as well as the comparision with previous dataset.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an open-source subject-to-video (S2V) generation framework that creates subject-consistent videos from multiple reference images and text prompts. Built upon the Wan 2.1 T2V-14B base model and fine-tuned for multi-reference input, it achieves near–closed-source performance in subject fidelity, background disentanglement, and video quality.

### Strengths
1. Introduce a pipeline to enhance subject and scene diversity, improve overall data fidelity, and ensure clear separation of subjects from irrelevant components.
2. A reference-based position encoding to emphasize the references, leading to better results.

### Weaknesses
1. The paper use CLIP as evaluation metrics. However, CLIP is not finegrained enough for Subject consistency. I suggest using face recognition metrics for human faces.

### Questions
1. How does the artifacts produced by image editing methods like Flux affects the generated video? For example, Flux redux reposes the human, which could introduce subject inconsistencies. 
2. How many subjects can be inserted to the video at the same time? What is the limiting factor?

### Soundness
3

### Presentation
3

### Contribution
3
