# OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 2

## Abstract
Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings.
We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, improves over Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens — a 6× reduction compared to Qwen2.5-Omni’s 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces OmniVinci, an open-source omni-modal LLM. The core contributions are twofold: (1) A new model architecture featuring three specific mechanisms (OmniAlignNet, TEG, and CRTE) to improve semantic and temporal alignment between vision and audio. (2) A novel data curation pipeline, the "Omni-Modal Data Engine," designed to mitigate "modality-specific hallucination" by using an LLM to correct and fuse single-modality captions. The authors demonstrate that OmniVinci achieves SOTA performance, notably outperforming Qwen2.5-Omni with 6x fewer training tokens.

### Strengths
- Novel Alignment System: The paper introduces a novel system for aligning information across text, audio, and video. The proposed architectural modules (OmniAlignNet, TEG, CRTE) are well-motivated and supported by extensive ablation studies (e.g., Table 1).
- Innovative Data Curation: The authors build an "Omni-Modal Data Engine"  to address the common and difficult problem of "modality-specific hallucination." They provide valuable insights (in Figure 4) into the limitations of single-modality captioning .
- State-of-the-Art Performance: The model delivers SOTA results on numerous industry benchmarks for audio, video, and omni-modal tasks among models of similar scale (including Qwen2.5-Omni). Significant improvements are shown on key tests like DailyOmni (+19.05) and Video-MME (+3.9).
- High Training Efficiency: The model achieves this strong performance while being highly efficient, using only 0.2T training tokens compared to Qwen2.5-Omni's 1.2T

### Weaknesses
- Limited Gains from RL Post-Training: The performance improvement from the GRPO reinforcement learning (RL) post-training appears relatively modest. As shown in Table 8, the score improvements on Worldsense, Dailyomni, and Omnibench are all less than 1 percentage point. Given the complexity and cost of RL training, does this minor gain suggest a bottleneck in the current method or data?

### Questions
- On the Combination of Loss Functions: The OmniAlignNet module introduces a contrastive loss, $L_{o-align}$ (Eq 1)20, while the LLM backbone uses a standard generative cross-entropy loss (implied in Figure 2). How are these (and potentially other) losses combined during the omni-modal joint training phase? Are they simply summed, or are they weighted (e.g., $L_{total} = \alpha L_{LLM} + \beta L_{o-align}$)? If weighted, how were these hyperparameters determined?
- Regarding Modality Conflict at Inference: The data engine is designed to resolve modality-specific hallucination during training . How does the final OmniVinci model handle new, explicit modality conflicts at inference time? For example, if the model is fed a video showing a dog but the audio narration says, "this is a cat," how does the model prioritize or fuse these contradictory signals?
- Clarification on 6x Training Efficiency: The 6x reduction in training tokens (0.2T vs. 1.2T)  is a very impressive efficiency claim. To fully contextualize this: (a) Does the 0.2T token count include the computational cost of running the "Omni-Modal Data Engine" to generate the 24M samples? (b) What was the total computational cost (e.g., total GPU-hours) and wall-clock time required for the omni-modal joint training phase, compared to the baseline?

### Soundness
3

### Presentation
4

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
This paper introduces OmniVinci, an omni-modal Large Language Model (LLM) designed for comprehensive cross-modal understanding by jointly processing vision, audio, and language. The authors present notable architectural contributions, including OmniAlignNet for vision-audio alignment, Temporal Embedding Grouping (TEG) for structured temporal representation of modality tokens, and Constrained Rotary Time Embedding (CRTE) for encoding absolute temporal information. They also curated and synthesized a diverse, large-scale dataset comprising 24 million conversations spanning both single- and multi-modal scenarios. Commendably, the work includes demonstrations of practical applications in real-world settings.

### Strengths
- The work proposes the innovative multi-modal integration architecture, OmniAlignNet, which aligns image and video dimensions within a unified feature space. The introduction of TEG and CRTE further enhances modality feature learning, substantially boosting the model's overall omni-modal understanding performance.

- The work construct a substantial dataset of 24 million samples and implemented multi-modal reasoning augmentation. The workflow illustrat in Figure 4 provides a clear mechanism for handling modality-specific hallucinations and generating high-quality cross-modal supervision.

- The paper includes a relatively comprehensive set of evaluations and ablation studies, effectively demonstrating the model's capabilities. Furthermore, the work showcases initial deployment and application potential in real-world scenarios.

### Weaknesses
- The study employs a paradigm where single modalities (image, audio) are trained separately before a unified modal alignment is performed. The paper lacks a performance comparison of modality-specific tasks before and after the cross-modal unified alignment. Reporting the respective performances on pure Image and Audio tasks before and after this alignment stage would significantly help validate the effectiveness of the proposed method.

- The image data significantly outweighs the audio data during pre-training. I wonder what the ratio of image-to-audio modality tokens is during the cross-modal alignment phase. Moreover, if the compression ratio for both the vision and audio encoders is identical during alignment,  how to eliminate potential bias resulting from the inherent token quantity imbalance between these modalities.

- Some reported model results are not reflective of the current state-of-the-art. It would be beneficial to use more recent, cutting-edge model results for comparison (e.g., updating InternVL-2 to InternVL-3 and Qwen2-vl to Qwen3-vl) to ensure the novelty claims are appropriately contextualized against the strongest contemporaries.

### Questions
same as weakness

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
2

### Summary
This paper introduces a suite of modules to enhance video–audio multimodal understanding, including OmniAlignNet, Temporal Embedding Grouping (TEG), and Constrained Rotary Time Embedding (CRTE).
OmniAlignNet is proposed to align video and audio latent representations. TEG and CRTE are designed to improve the temporal alignment of video and audio features, thereby facilitating model learning.
Ablation studies validate the effectiveness of each component. The proposed method outperforms Qwen2.5-Omni on several video and audio understanding benchmarks.

### Strengths
- Experiments on dataset engines highlight the importance of fully exploiting cross-modal information in sounding videos, which benefits both clean data construction and model performance. These findings offer meaningful insights for future research.
- The proposed modules improve model performance from two key perspectives—video–audio semantic alignment and temporal alignment—both of which are well-motivated and demonstrated to be effective for video and audio understanding tasks.

### Weaknesses
- Although the authors claim the model is omni-modal, the work primarily focuses on video and audio, leading to degraded performance on the image modality. Furthermore, results for text-to-text tasks are not reported. Combined with the claim that the model uses far more tokens than Qwen2.5-Omni, it raises concerns that the proposed approach may neglect text and image modalities.
- TEG involves interleaving video and audio tokens when feeding the LLM, which is similar to the approach used in Qwen2.5-Omni. Likewise, contrastive learning is a common practice for aligning video and audio semantic features.
- Comparisons with Qwen2.5-Omni in terms of audio generation latency, training cost, and total parameter count are not reported.

### Questions
Since the 24M multimodal model is part of the contributions, is there a plan to open-source it?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces OmniVinci with three techniques including i) OmniAlignNet to align vision and audio in a video ii) Temporal embedding grouping for capturing the temporal alignment between vision and audio and iii) Constrained rotary time embedding for adding temporal information into vision-audio embeddings. The experiments show improvement on various multimodal understanding, audio understanding and vision understanding tasks over prior work.

### Strengths
The paper tackles an important goal of building open-source, omni-modal LLM by incorporating OmniAlignNet, temporal embedding grouping and constrained rotary time embedding. The paper curated a large-scale dataset spanning audio, video and image domains and shows improvements to multimodal understanding, audio understanding and vision understanding tasks over prior work, which would be of interest to the community. Overall, the proposed method is simple and paper is easy to read and well-written.

### Weaknesses
* The distinction of OmniAlignNet module, use of position encoding from the current video-audio alignment common in existing work [1,2,3] is unclear. The paper lacks a discussion with these works making its positioning among them unclear.
* Similarly there exists many studies [4,5] that have incorporated the temporal sequence in multiple ways, which the paper lacks a comparison or distinction with.
* The paper argues the need of Omnimodal data engine and gives an example of where both audio and video are required. But as shown in many prior multimodal studies [6,7,8], there exists many datasets where one modality suffices and thus explicitly enforcing interactions is suboptimal and often leads to unnecessary correlations. The paper lacks any discussion in this aspect as well. 
* The paper highlights modality-specific training in section 3.1 by using data for each modality but it is unclear how this is incorporated in the omni-modal joint training and more details need to be provided on the separation of the modality-specific and omni-modal training. 
* The claims of the paper are not well supported by the empirical results. For example, i) While OmniVinci only improves the performance on Dailyomni in Table 3, its worse than almost all models on Omnibench with upto 10% worse than Qwen. The performance on Worldsense is also not convincing without confidence intervals. Similar conclusions hold for Image benchmarks in Table 7 and speech recognition benchmarks in Table 5, where OmniVinci obtains worse performance across baselines. 

Overall, in the current state I recommend recommend rejection due to the lack of discussion with prior work in multiple aspects and claims not being supported by empirical results.

The following can improve the paper further:
* A common trend for multimodal models is the lack of temporal reasoning. It would be useful to see the performance of the proposed method on cases [9,10] which are explicitly designed to evaluate the same. 
* The font size and presentation for most results is extremely small, which makes it challenging to interpret the results meaningfully.
* The position of Table 5 and Table 6 can be switched.

References:   
[1] Cheng et al. MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis.     
[2] Kim et al. Deep Visual Forced Alignment: Learning to Align Transcription with Talking Face Video.    
[3] Guo et al. Aligned Better, Listen Better for Audio-Visual Large Language Models.  
[4] Zerveas et al.  A transformer-based framework for multivariate time series representation learning.    
[5] Eldele et al. TSLANet: Rethinking Transformers for Time Series Representation Learning.  
[6] Liang et al. Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework.  
[7] Madaan et al. Jointly Modeling Inter- & Intra-Modality Dependencies for Multi-modal Learning.   
[8] Wang et al. An Information Criterion for Controlled Disentanglement of Multimodal Data.  
[9] Shangguan et al. TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models.   
[10] Video SimpleQA: Towards Factuality Evaluation in Large Video Language Models.

### Questions
Please refer to my comments above.

### Soundness
2

### Presentation
2

### Contribution
2
