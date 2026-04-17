# Beyond Words: Multimodal LLM Knows When to Speak

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
While large language model (LLM)-based chatbots have demonstrated strong capabilities in generating coherent and contextually relevant responses, they often struggle with understanding when to speak, particularly in delivering brief, timely reactions during ongoing conversations. This limitation arises largely from their reliance on inputs of limited modality, lacking the rich contextual cues in real-world human dialogue. In this work, we focus on real-time prediction of response types, with an emphasis on short, reactive utterances that depend on subtle, multimodal signals across video, speech, and text. To support this, we introduce a new multimodal dataset constructed from real-world conversational videos, containing temporally aligned visual, auditory, and textual streams. This dataset enables fine-grained modeling of response timing in dyadic interactions. Building on this dataset, we propose MM-When2Speak, a multimodal LLM-based model that adaptively integrates visual, auditory, and textual context to predict when a response should occur, and what type of response is appropriate. Experiments show that MM-When2Speak outperforms state-of-the-art LLM-based baselines, achieving up to a 4× improvement in response timing accuracy over leading commercial LLMs. These results underscore the importance of multimodal inputs for producing timely, natural, and engaging conversational AI. We will release code and dataset to support further research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues a critical limitation of large language model (LLM)-based chatbots: their inability to accurately determine "when to speak" (especially for brief, reactive utterances) in real-world conversations, which stems from overreliance on single-modal inputs (lacking visual and auditory cues). To solve this, the research proposes a multimodal framework and corresponding dataset, with key contributions as follows: 1) A Multimodal Conversational Dataset. 2) A Multimodal LLM for “when to speak” Prediction. The experimental results show that adding modalities consistently improves accuracy, and the MM-When2Speak outperforms baselines.

### Strengths
1. The paper addresses a key limitation of existing LLM-based chatbots—their inability to accurately determine "when to speak" (especially for brief, real-time backchannel reactions) in natural conversations—rather than just "what to say." This focus fills a critical gap: while most research prioritizes response content coherence, the timing and appropriateness of responses are equally vital for human-like interaction. By tackling this underexplored challenge, the work directly enhances the practicality of conversational AI in real-world scenarios.
2. The paper constructs a high-quality, multimodal dataset with fine-grained annotations. A major strength lies in the creation of a curated, multimodal dataset that addresses the scarcity of resources integrating visual, auditory, and textual cues for response timing prediction.
3. The paper proposes a well-designed multimodal model with adaptive fusion. The model surpasses baselines on the "when to speak" prediction task.

### Weaknesses
1. The dataset, while carefully curated, has notable limitations in scope that may restrict the generalizability of the findings. It is sourced exclusively from public YouTube videos, focusing on only two conversational scenarios: virtual Zoom meetings and broadcast news interviews (e.g., CNN interviews) . This narrow scenario coverage fails to include other common real-world dialogue contexts.
2. The dataset is limited to interactions between two participants , with no exploration of multi-party conversations.
3. The dataset is relatively small compared to large-scale multimodal datasets in related fields.
4. The paper simplifies the conversational dynamics. The paper’s framing of "when to speak" as a dense classification task using fixed 10-second sliding windows (with 0.5-second strides)  oversimplifies the complexity of natural conversational dynamics. Human dialogue is highly variable in pace.

### Questions
1. Do you recognise the limitations of the dataset? Do you have plans to further improve the dataset?
2. MM-When2Speak uses a fixed 10-second sliding window for prediction, but human dialogue often has variable turn lengths (e.g., 2-second backchannels vs. 20-second explanations). Have you experimented with adaptive window sizes (e.g., windows that shrink/grow based on speech pause duration) or dynamic stride lengths? 
3. For the three-modal (Video+Speech+Text) baseline. Why did you not include other well-established multimodal LLMs like Video-ChatGPT (Maaz et al., 2024)?

### Soundness
3

### Presentation
3

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
This paper addresses real-time conversational response planning. The authors build a multimodal model that predicts when a system should speak, stay silent, or give short listener reactions. They compile a new dataset of natural two-party conversations, and train a sliding-window classifier that takes speech, video, and transcript as input. Experiments show that the proposed model outperforms several multimodal and audio-visual LLM baselines.

### Strengths
1. The problem is practical and underexplored, most dialogue systems focus on what to say, but rarely model when they should respond or remain silent.

2. The dataset is a useful contribution: multimodal, time-aligned, and supported by high-quality human-verified annotations.

3. The method is straightforward and computationally lightweight, making real-time deployment feasible.

4. Experimental results indicate clear improvements over strong multimodal LLM baselines across multiple settings.

### Weaknesses
1. One concern is the practical usefulness of the proposed model in real applications. Since it only outputs a response type rather than the actual content, it seems more like an auxiliary module that provides a triggering signal for another system to generate the verbal response. The paper does not discuss how this module integrates with a full conversational pipeline.

2. The evaluation does not report latency or real-time efficiency. Given that the model is intended for online interaction, end-to-end latency is critical, and it also relates to the concern above about practicality.

3. The model is trained and tested primarily on the authors' own dataset, which mainly consists of two-party, news/interview-style conversations. It remains unclear how well it would generalize to casual, noisy, or multi-speaker settings.

4. The pre-training stage is mentioned but not clearly described. It is not obvious what exact objective is optimized, how multimodal alignment is learned, and to what extent this pre-training contributes to downstream performance.

### Questions
1. How would this module be integrated with a downstream response generation system in a real conversational agent?

2. What is the actual runtime latency under streaming input? Can the model maintain real-time performance in longer conversations?

3. Do the authors have any evidence of generalization beyond the curated dataset, such as informal chats, noisy environments, or multi-speaker scenarios?

4. Can the authors provide more details about the pre-training stage?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper tackles “when should an LLM speak, and with what short reaction?” in dyadic video conversations. It builds two datasets (Short-Clips and Full-Videos) from real conversational videos with aligned video, audio, and text, and proposes MM-When2Speak, a multimodal LLM-based classifier that fuses the three modalities and uses a sliding window for online inference. Experiments against strong text-only and multimodal LLM baselines (e.g., GPT-4o, Gemini-1.5, VITA-1.5) report notable gains, including up to 4× improvement in response-timing accuracy over text-only LLMs.

### Strengths
Targeting the underexplored yet vital problem of “when to speak” (vs. “what to say”) in human–AI interaction, the paper reports consistent gains across datasets, modalities, and strong baselines with informative ablations, adopts sliding-window inference for online deployability, and documents transparent data construction and labeling (including confusion matrices and class-wise recall).

### Weaknesses
1. The pipeline figure is ambiguous—window length/stride and label–time alignment are unspecified, the transcript appears stretched across multiple audio windows, and it’s unclear whether outputs are per-window predictions or post-processed events.
2. Lacks qualitative case analyses and a breakdown of reaction/query diversity (per-class distribution, long-tail behavior, representative successes/failures).
3. No end-to-end latency or memory profiling in realistic settings; please quantify per 10-s clip on commodity CPU/GPU and compare against lighter baselines to demonstrate real-time feasibility.
4. The architecture is a fairly straightforward tri-modal fusion atop an LLM; motivate why this design is preferable to stronger temporal/event-boundary models (e.g., causal streaming, CTC-style detectors) and provide deeper modeling insights.
5. Auto-labeling with limited human checks needs stronger evidence—report inter-annotator agreement (e.g., Cohen’s κ/α) by class and timing tolerance, plus failure analyses for nuanced reactions (e.g., *pondering* vs *surprise*).
6. Comparisons to GPT-4o/Gemini may be prompt- and seed-sensitive; include prompt ablations, few-shot variants, and variance across runs/seeds.

### Questions
1. How do different window lengths/strides affect early vs. late backchannel detection?
2. What fraction of the dataset received human verification?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors address the challenge that existing conversation agents struggle to determine "when to speak" during interactions. They attribute this limitation to the dependence on single modality inputs and propose a multimodal approach as a solution. To this end, they collect publicly available dyadic conversation videos from YouTube to construct a dataset, where each video segment is annotated with one of nine types indicating whether the agent should speak, remain silent, or give a brief reaction. Using this dataset, they train MM-When2Speak model that integrates video, audio, and textual modalities through a self-attention fusion mechanism, and compare its performance against several baseline models.

### Strengths
1. The paper introduces a new task that predicts when a conversation model should speak by jointly leveraging text, audio, and video modalities.
2. The authors build a new dataset by collecting real dyadic conversation videos from YouTube and annotating each segment with nine response types (seven short reaction categories, full response, and silence). This dataset is specifically designed to support fine-grained modeling of response timing in natural conversations.
3. The proposed MM-When2Speak model processes video, audio, and text inputs simultaneously and employs a self-attention fusion mechanism to integrate multimodal cues effectively. Experimental results show that this multimodal approach achieves substantially better performance than models using single or fewer modalities.

### Weaknesses
1. Although the collected dyadic conversation dataset is carefully curated and of high quality, its overall scale is rather limited. In addition, the data are confined to dyadic interactions, leaving more complex multi-party conversations unexplored. Even if the model is trained on dyadic exchanges, evaluating it on multi-party dialogue (small sample like full-video split) could provide meaningful insights into its generalization capability.
2. The model addresses only when to speak, not how to speak. Because it is fine-tuned solely for response type classification, it can determine the timing of speech but not the linguistic or expressive content of the response. This separation means that another model would be required to handle "how to speak" making the overall system design somewhat inefficient and limited in scope.
3. The paper's motivation focuses primarily on modality limitations. However, one might argue that the issue arises not only from missing modalities but also from the fact that current LLMs and VLMs have not been trained on such response timing tasks (when to speak). While the authors conduct the experiments in a zero-shot setting, it would be informative to fine-tune LLMs on textual transcripts and compare the results. Additional ablation studies would strengthen the motivation and make the paper's claims more robust and convincing.
4. The length of each video segment is fixed at 10 seconds. Although the paper mentions a sliding-window approach to move across the segment, the rationale behind choosing this specific window length (10s) is unclear. I am also curious about how the model's performance would change if the window size were varied. Overall, the justification for this design choice seems insufficient.

### Questions
1. I suggest integrating the supplementary material as Appendix within the paper and explicitly referencing it in the main text wherever relevant. This could improve the paper's readability and flow.
2. Could you provide more detailed statistics about the collected video dataset? For example, what is the average video length? Including such quantitative details would help readers better understand about the dataset.
3. Since the dataset was constructed from YouTube videos, were copyright and ethical issues taken into account during data collection? (I haven't flagged this for ethics review yet, but depending on the authors' clarification, an ethics review may be required)
4. Why was the model architecture limited to a classification task? Would enabling the model to also generate responses make it more realistic and practically useful? I wonder if extending the model to include generation might introduce challenges or potential performance degradation.
5. As I understand, the human verification study was conducted to validate the labels generated by GPT, right? It would be better to include more information about the annotators, such as their background, and annotation procedures to enhance transparency.

### Soundness
2

### Presentation
3

### Contribution
2
