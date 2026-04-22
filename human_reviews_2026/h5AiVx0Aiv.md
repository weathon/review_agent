# DrVoice: Parallel Speech-Text Voice Conversation Model via Dual-Resolution Speech Representations

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Recent studies on end-to-end (E2E) speech generation with large language models (LLMs) have attracted significant community attention, with multiple works extending text-based LLMs to generate discrete speech tokens. Existing E2E approaches primarily fall into two categories: (1) Methods that generate discrete speech tokens independently without incorporating them into the LLM’s autoregressive process, resulting in text generation being unaware of concurrent speech synthesis. (2) Models that generate interleaved or parallel speech-text tokens through joint autoregressive modeling, enabling mutual modality awareness during generation. This paper presents DrVoice, a parallel speech-text voice conversation model based on joint autoregressive modeling, featuring dual-resolution speech representations. Notably, while current methods utilize mainly 12.5Hz input audio representation, our proposed dual-resolution mechanism reduces the input frequency for the LLM to 5Hz, significantly reducing computational cost and alleviating the frequency discrepancy between speech and text tokens and in turn better exploiting LLMs’ capabilities. Experimental results demonstrate that DrVoice-7B establishes new state-of-the-art (SOTA) on prominent speech benchmarks including OpenAudioBench, VoiceBench, UltraEval-Audio and Big Bench Audio, making it a leading open-source speech foundation model in ∼7B models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the use of dual speech representations (both continuous and discrete representations) and their grouping (to reduce the token rate and reduce the computation) as input along with the text tokens to an LLM and then at the output first a text head produces the text tokens which helps guide the speech head that produces the ungrouped sequence of speech tokens (referred to as Chain of Modalities). The output speech tokens are then converted into mel features via a flow matching model and a HiFiGAN vocoder is used to convert the mel features into the wave files. Hence, the system can take both speech and text as input and can produce output in both text and speech format (or a combination of the two). Based on the input output types, the paper describes 7 types of modes for the model which can be switched with a predetermined prompt. Another training detail that the paper emphasizes is that the importance of linearly mixing the base LLM's weights with that of the multimodal tuned model. Experiments on various speech+text benchmarks competitive performance to the several other multimodal LLMs including Qwen2.5-Omni  and Kimi-Audio. In addition, the paper claims to achieve the SOTA numbers on the Big Bench Audio tasks among the LLMs tested. In terms of generated speech quality, UTMOS score suggest that it is high quality audio even though the WER evaluation indicates some potential issues with the generated speech.

### Strengths
originality
* Use of both continuous and discrete representations of audio simultaneously is relatively less investigated in multimodal LLMs. Chain of modality and model weight mixing are presented as novelties which are somewhat novel. Especially, model averaging by linear combination is not particularly new but its application to LLMs during training might be an experimental design novelty.

quality
* The experimental results show competitive performance and the paper claims the SOTA results on the Big Bench Audio speech-to-text and speech-to-speech tasks.

clarity
* Mostly clearly written. However, some details are left to the Appendices.

significance: 
* Successful implementation of multimodal LLMs is a relatively popular topic nowadays and this paper would be relevant to that community.

### Weaknesses
* The novelty is limited but sufficient (see the comments in the strengths section) Most of the individual components are well known (audio encoders, audio decoders, the LLM backbone, etc.) 

* The main text skipped many details and provided those details in the Appendices. Which makes the reviewer questioning whether the paper needed more number of pages and a conference setting is not suitable for this paper.

* Even though one of the major claims of the paper is the computational cost reduction by grouping of audio tokens to reduce the token rate, the presented experiments do not provide the details of that improvement other than the token rate. What would be the computational savings in terms of FLOPs of RTF (whenever applicable)?

### Questions
1. Even though one of the major claims of the paper is the computational cost reduction by grouping of audio tokens to reduce the token rate, the presented experiments do not provide the details of that improvement other than the token rate. What would be the computational savings in terms of FLOPs of RTF (whenever applicable)? 

2. Although the appendices contain some details, the main text seems to be glossing over the details. Do the authors think that the manuscripts would require a longer paper? 

3. Based on the results in Table 2, the proposed DrVoice performs similarly to the other baselines in the table for all benchmarks except the Big Bench tasks. That is the relative differences are rather small between models. But for the Big Bench tasks we see a very large improvement (about 20% as compared to the Kimi Audio (the 2nd best)). Do the authors have any explanation of why DrVoice performs this well on these two tasks? Please comment on that.

### Soundness
3

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
4

### Summary
The author proposed a 7B speech text voice conversion model, DrVoice, that leverages joint autoregressive modeling and dual speech representations. 

The paper contributed on both model architectures and the training strategies. The model introduces a mechanism to alleviates the temporal resolution mismatch between text and speech tokens and reduce the computational cost. Specifically, it applied several grouping/ungrouping techniques to map 25 Hz audio tokens to 5Hz speech tokens and combined it with text tokens. Also, it uses a speech refined head(SRH) to preserve the intermediate semantic and acoustic representation during token generation. On the training side, two training strategies are applied: chain-of-modality(CoM) mix training for generating structured thoughts being used as language model text prompt, and a two-stage training method called Core-Cocktail Training for refining model's performance during training with large learning rate.

The model was trained on 100k hours of audio-tex paried data and evaluated with comprehensive benchmark metrics comparing to multiple baselines including previous SOTA. The author claimed their model achieved state of the art on OpenAudioBench and Big Bench Audio benchmarks, and performance comparable to SOTA on VoiceBench and UltraEval-audio benchmarks.

### Strengths
The model extended existing parallel joint speech-text model by the new methods below:
1. The author contributed an idea of reducing the temporal resolution of extracted audio representations from 25Hz to 5Hz to match the 3Hz text representation using speech token grouping. This benefits to reduce the computational costs and alleviate the frequency discrepancy between different representations. Leveraging the fine-grained acoustic information during generative scenarios is intuitive. The evaluation shows the speech quality is not compromised. 
2. The author proposed to use both CoM-mixing and Core-Cocktail Training for model performance during training phase. The chain-of-modality mixing training allows the model to summarize intermediate textual transcriptions before feeding into LLM. The Core-Cocktail Training contributes to model training with large learning rate while preserving performance.
3. The SRH is proved useful for speech generation in the ablation study.

### Weaknesses
1. Although the grouping mechanism helps with reducing the temporal resolution and alleviate two representations, the ungrouping and the Speech Refined Head is making the model architecture more complicated and may worsen the model's speed due to its auto-regressive nature.
2. In the benchmark results, the model doesn't beat all the baselines in OpenAudioBench. The author claimed that the average score is the highest, but it is also true that it is not showing comparable perforamance on some tasks such as Reasoning QA(57.92 vs 63.76 Qwen2.5). There are not enough evidence showing that DrVoice is a new SOTA on openAudioBench. 
3. Similarly, for VoiceBench and UltraEval-Audio, DrVoice only shows best perforamance on 1 or 2 tasks. While Kimi-Audio as a former SOTA still dominates on 5 out of 8 tasks. It could be better to discuss the performance comparison more.

### Questions
1. Does the ungrouping(linear + split) and SRH cause any increased computational cost? How does it compare to the effect of reducing temporal resolution?
2. From the ablation study, the ungrouping and SRH contributes more to S2M and T2M, while it doesn't help much with T2T. Is SRH making redundancy for T2T tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DrVoice, a parallel speech-text voice conversation model built upon a large language model (LLM). The core contribution is a novel Dual-Resolution Speech Representation (DRSR) mechanism designed to address the temporal resolution mismatch between speech and text tokens in end-to-end (E2E) spoken dialogue systems. Specifically, the model groups higher-frequency (25Hz) input speech tokens into lower-frequency (5Hz) representations before feeding them to the LLM, reducing computational cost and aligning the speech processing rate more closely with that of text. The authors also propose two training strategies: a CoM-Mixing strategy to train the model on various interaction patterns (e.g., text-only, speech+text), and a Core-Cocktail training strategy to mitigate catastrophic forgetting of the base LLM's capabilities. Experimental results show that DrVoice-7B achieves good performance on OpenAudioBench and Big Bench Audio, and is competitive on other benchmarks like VoiceBench and UltraEval-Audio, establishing it as a strong open-source model in its size class.

### Strengths
1. The central idea of using dual-resolution speech representations is well-motivated and addresses a key, practical problem in joint speech-text modeling: the significant discrepancy in token rates between speech and text. The proposed grouping/ungrouping mechanism is an elegant solution that directly tackles this issue, leading to significant computational efficiency gains (processing at 5Hz within the LLM) without sacrificing output quality, thanks to the Speech Refined Head.

2. The paper presents a comprehensive set of experiments on multiple challenging benchmarks. DrVoice achieves SOTA results on two benchmarks (OpenAudioBench, Big Bench Audio) and demonstrates performance on par with other leading models on VoiceBench and UltraEval-Audio. This strong and balanced performance across tasks involving both understanding and generation solidifies the effectiveness of the proposed architecture.

3. The reduction of the LLM's operating frame rate from the common 12.5Hz or 25Hz to 5Hz is a major practical advantage. As shown in the ablation studies (Figure 2 in appendix), this leads to a nearly 50% reduction in GPU hours for training. This makes training and deploying such powerful spoken language models more feasible and is a valuable contribution to the field.

4. The proposed training strategies, CoM-Mixing and Core-Cocktail, are thoughtful additions. CoM-Mixing provides a structured curriculum for learning complex, multi-stage reasoning (think-then-speak), while the Core-Cocktail method offers a pragmatic approach to balance new learning with knowledge retention from the base LLM. The ablation studies effectively demonstrate the positive impact of these techniques.

### Weaknesses
1. The paper notes that DrVoice's ASR-WER (11.2) is higher than that of Qwen2.5-Omni (3.48), suggesting weaker text-speech alignment. The authors hypothesize this is because Qwen2.5-Omni feeds text directly into its "Talker" module, while DrVoice only uses hidden states. This is a crucial architectural trade-off. While the proposed solution (adding text as input to SRH) is mentioned as future work, the current limitation is significant. High ASR-WER can indicate issues with intelligibility or word-level alignment, which is a critical aspect of a voice conversation model.

2. A large portion of the training data (3B assistant tokens) is synthesized using CosyVoice. While the paper uses WER filtering, training on synthetic data can introduce biases from the teacher model and may not fully capture the diversity and nuances of real human speech. The paper could benefit from a discussion on the potential limitations of this approach and how it might affect the model's performance on in-the-wild, real-world conversations.

3. While the dual-resolution mechanism is a clever and effective engineering contribution, the overall architectural design of DrVoice bears a strong resemblance to existing models, particularly the Qwen-Omni series. The core "Thinker-Talker" paradigm, where a central LLM ("Thinker") generates intermediate representations that are then passed to a separate speech synthesis module ("Talker"), is conceptually very similar to DrVoice's structure of a shared LLM layer feeding into a dedicated Speech Refined Head (SRH).

### Questions
1. Could you clarify the training process for the Speech Refined Head (SRH)? Is the SRH jointly pre-trained with the LLM backbone, allowing gradients from the SRH loss to propagate back to the LLM? Or is the SRH trained separately, merely using the frozen LLM's hidden states as input? Have you explored both strategies, and what is their comparative impact on performance?

2. In the ablation study presented in Table 4, you analyze the impact of removing the Continuous Speech Encoder (CSE) and the Speech Refined Head (SRH). Could you please specify the alternative mechanisms used in these ablated models? For the "w/o. CSE" setting, how are the raw speech inputs processed and fed into the LLM? For the "w/o. SRH" setting, how are the speech output tokens generated from the shared LLM layer's hidden states? Is a simpler projection layer used instead?

3. Table 3 shows that DrVoice has a relatively high ASR-WER of 11.2, which is significantly worse than top-performing models like Qwen2.5-Omni (3.48). This suggests a potential weakness in speech-text alignment or intelligibility. How do you interpret this result, and what specific architectural or training factors do you believe contribute to this performance gap?

### Soundness
3

### Presentation
3

### Contribution
2
