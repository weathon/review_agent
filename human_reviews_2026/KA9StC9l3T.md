# It Hears, It Sees too: Multi-Modal LLM for Depression Detection By Integrating Visual Understanding into Audio Language Models

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Depression is one of the most prevalent mental health disorders globally. In recent years, multi-modal data, such as speech, video, and transcripts, has been increasingly used to develop AI-assisted depression assessment systems. Large language models have further advanced this field due to their strong language understanding and generalization capabilities. However, conventional LLMs remain text-centric and cannot process the rich non-verbal cues found in audio and visual modalities, which are critical components in mental health evaluation. While multi-modal LLMs offer a promising direction, few are tailored for psychological applications. In this study, we propose a novel multi-modal LLM framework for depression detection. Our approach augments an audio language model with visual understanding and aligns audio-visual features at the timestamp level. This fine-grained alignment improves modeling of temporal dynamics across modalities while reducing the need for extensive training data and computational resources. Experiments on the DAIC-WoZ dataset demonstrate that our model outperforms both single-modality approaches and previous multi-modal methods. Moreover, the proposed framework can be extended to incorporate additional physiological signals, paving the way for broader clinical applications beyond mental health.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a multi-modal large language model (MLLM) framework for depression detection by integrating visual understanding into an audio language model. The authors propose timestamp-level alignment of audio and visual features to improve temporal synchronization and demonstrate the model’s performance on the DAIC-WOZ dataset. While the idea of enhancing depression detection using MLLMs is timely and relevant, the current contribution appears incremental compared to existing LLM-based approaches, and several methodological and presentation issues limit the paper’s impact.

### Strengths
- The topic is important and highly relevant to both AI and mental health research.

- The integration of audio and visual modalities into an LLM for depression detection sounds reasonable.

- The experimental results show performance improvement on the DAIC-WOZ dataset, demonstrating the method’s potential.

### Weaknesses
Limited novelty: The contribution beyond existing MLLM-based depression detection frameworks appears marginal. The model primarily combines known components (audio LLM + visual encoder + timestamp alignment) without introducing fundamentally new mechanisms or theoretical insights.

Single dataset evaluation: The method is only evaluated on the DAIC-WOZ dataset, which limits generalizability and makes it difficult to assess robustness. Other publicly available datasets such as DVlog could strengthen the validation.

- The method section lacks sufficient mathematical formulation. Key components such as the fusion mechanism, timestamp-level alignment, and loss functions should be accompanied by equations and clear variable definitions.

- Figures are difficult to interpret due to missing notations and insufficient correspondence to text descriptions. For example, Figure 1 is never referenced in the main text and lacks a detailed caption that explains the entire training pipeline.

- The model’s performance can be highly sensitive to prompt instructions. However, the paper does not provide justification or comparative analysis of different prompting strategies during instruction tuning.

- Section 4.3.3 reveals that removing the interviewer’s utterances significantly improves performance, yet the paper does not explore how the model could automatically distinguish between interviewer and participant—an important step for practical deployment.

### Questions
- How does your framework substantially differ from recent MLLMs used for depression detection? What conceptual or algorithmic advance distinguishes it?

- Have you considered evaluating the model on other datasets (e.g., DVlog or E-DAIC) to confirm robustness and generalization?

- Can you include explicit equations that describe how the model fuses and aligns multimodal inputs, including notation that matches the figures?

- What prompt template did you use for instruction tuning, and how sensitive is the model’s performance to different prompting styles?

- Can the proposed model be extended to automatically identify the target speaker (participant) to ensure consistent detection across multi-speaker conversations?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a multimodal LLM-based framework for depression detection tasks that utilizes audio and visual knowledge.

### Strengths
**1.** The overall task and motivation behind the work are clearly defined.

**2.** The visuals are clear and informative.

**3.** The experiment setup and rationales behind each evaluation are clearly stated. I also appreciate the sub-subsections under Section 4.3.

### Weaknesses
**1.** A major concern regarding this work is its novelty. The multimodal approach to depression detection is not a new concept, as various encoders have been used for feature extraction across different modalities. Additionally, many previous studies have been conducted on the DAIC-WOZ dataset with similar frameworks.

**2.** One suggestion for improvement is to conduct more experiments using other multimodal datasets for depression, particularly those representing diverse languages and demographics. Furthermore, incorporating additional metrics, especially for regression tasks rather than solely for classification, would be beneficial.

**3.** Overall, I think the paper is well-written. But the level of novelty and the extent of the work presented may not meet the standards required for a paper at this conference.

### Questions
N/A.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a multi-modal large language model for depression detection. The framework integrates an audio language model with a visual encoder, using timestamp-level alignment to fuse audio and visual data streams. This method is presented as a way to capture fine-grained temporal dynamics relevant to mental state assessment.

### Strengths
The approach of augmenting an existing audio language model is a practical method for developing a multi-modal system. The timestamp-level alignment of audio and visual information is a logical design for processing behavioral data. The sequential three-stage training process, which involves self-supervised pretraining of the visual encoder, cross-modal alignment, and parameter-efficient fine-tuning, is a structured way to integrate a new modality.

### Weaknesses
The model relies on pre-extracted visual features from the dataset, not raw video input. This limits the evaluation of the visual component to the quality of these specific features and does not demonstrate an ability to learn from unprocessed video. The data augmentation technique removes interviewer audio and video, which, while focusing on participant data, discards conversational context that might influence participant behavior. The evaluation is conducted on a single dataset, DAIC-WoZ, which, while a standard benchmark, has a limited number of participants.

### Questions
How might the system's performance change if the visual encoder was trained on raw video frames rather than pre-extracted features?

What are the implications of removing interviewer data from the input streams, and were alternative methods for handling the interviewer's conversational context considered?

Was the effect of an instruction prompt that explicitly references the visual modality investigated during the fine-tuning stage?

### Soundness
3

### Presentation
3

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
This paper presents a multi-modal large language model (MLLM) for automated depression detection that integrates visual understanding into an audio language model (ALM). The authors build upon the Qwen2-Audio model and propose a framework that aligns audio and visual features at the timestamp level, enabling fine-grained temporal fusion. The training process involves three stages: (1) self-supervised visual pretraining, (2) utterance-level audio-visual alignment, and (3) multi-modal instruction tuning using parameter-efficient fine-tuning (PEFT/LoRA). Experiments on the DAIC-WoZ dataset demonstrate superior performance compared to single-modality and previous multi-modal baselines, achieving a relatively good performance in this specific task.

### Strengths
+ Good empirical results. The model achieves consistent gains across modalities, outperforming both single-modality and previous multi-modal methods for this specific task.

+ Parameter efficiency. By leveraging LoRA and QLoRA, the authors effectively reduce computational cost while maintaining performance, making the approach feasible for research and clinical use.

### Weaknesses
- In Figures 1 & 2, a wave is used to denote the video input, which is really inappropriate.

- Unclear Design. The model fuses audio + visual embeddings by simple element-wise addition. Was any normalization or linear projection used to match feature scales? Did the authors compare addition vs concatenation or MLP fusion, or other fusion methods?

- Another concern is the design choice of merging the tokens from the two modalities before feeding them into the LLM, rather than inputting each modality separately and allowing the LLM to perform cross-modal reasoning internally. It would be helpful if the authors could clarify the motivation and potential advantages of this early fusion strategy.

- In Equation 2, it is strongly recommended to use a letter rather than “Sim” to denote the similarity matrix. Moreover, is it a mistake that tau was used, rather than the Greek letter? An ablation study on this parameter is also missing.

- In Figure 3, it is said that “Only the upper layers of the visual encoder receive gradient updates”. This is something that could be indicated in Figure 3. Moreover, what do the red arrows on the low levels mean?

- The main issue with this paper lies in its lack of true innovation. The authors attempt to demonstrate novelty by narrowing the research scope to an extremely specific task, depression detection, rather than by introducing fundamentally new techniques. However, multi-modal fusion of audio, video, and text has already been extensively explored in prior works such as NExT-GPT, Macaw-LLM, and AnyGPT. The paper deliberately avoids comparing with these models simply because they were not evaluated on this particular task, but in fact, such comparisons are feasible and necessary to substantiate the claimed contributions. Furthermore, even within this specific subdomain, the paper misses several recent approaches, such as MultiDepNet (WACV 2025), which are directly relevant to depression detection.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
1
