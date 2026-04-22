# From Text to Talk: Audio-Language Model Needs Non-Autoregressive Joint Training

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6, 4

## Abstract
Recent advances in large language models (LLMs) have attracted significant interest in extending their capabilities to multimodal scenarios, particularly for speech-to-speech (S2S) conversational systems. However, existing multimodal models handling interleaved audio and text rely on autoregressive (AR) methods, overlooking that text depends on target-target relations whereas audio depends mainly on source-target relations. In this work, we propose Text-to-Talk (TtT), a unified audio-text framework that integrates AR text generation with non-autoregressive (NAR) audio diffusion in a single Transformer. By leveraging the any-order AR property of absorbing discrete diffusion, our approach provides a unified training objective for text and audio. To support this hybrid generation paradigm, we design a modality-aware attention mechanism that enforces causal decoding for text while allowing bidirectional modeling within audio spans, and further introduce three training strategies that reduce train-test discrepancies. During inference, TtT employs block-wise diffusion to synthesize audio in parallel while flexibly handling variable-length outputs. Comprehensive experiments on audio question answering (Audio-QA), automatic speech recognition (ASR), automated audio caption (AAC) and S2S benchmarks show that TtT consistently surpasses strong AR and NAR baselines, with additional ablation and training-strategy analyses confirming the contribution of each component.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Prevailing SLM adopt an interleaved text-audio sequence for next-token prediction. (e.g. GLM4-Voice [1], Kimi-Audio [2]). In the mean time, CV research shows promising results on adopt AR for text prediction and NAR for image prediction, such as Transfusion [3] and Show-O [4], compared to modeling both text and image tokens with the same AR objective. This paper thus research the possibility to replace the AR prediction on the interleaved audio token spans with NAR and bidirectional discrete diffusion. The main argument is simple: "AR for text+NAR for audio" > "AR for text and audio" by a large margin.

1. GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot
2. Kimi-Audio Technical Report
3. Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model
4. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation

### Strengths
1. Given their experiments, AR+NAR significantly outperforms AR.
2. The ablation study on the train/test mismatch is clear and effective.

### Weaknesses
## Major Weakness
1. The paper only compares its models to self-established baselines. However, it is unclear how the trained model performs relative to existing SLM systems. It is not necessary to outperform all existing systems, as the training resources differ, but such comparisons would still help readers understand whether the baseline AR model is completely failing or serves as a reasonable baseline. Comparisons with models such as GLM4-Voice, Kimi-Audio, and LLaMA-Omni are required. If the AR baseline significantly lags behind these models, the conclusions may not generalize and could undermine the significance of the claimed contributions (e.g., conclusions drawn from small models may not hold for large models).
2. The paper should provide a demo page for readers to understand the quality of the generated speech compared to the pure AR approach.

## Minor Weakness
1. No human evaluation on the generated speech (both semantic coherence and speech quality)
2. The novelty is minor, as the CV domain already has extensive work demonstrating that a single Transformer can benefit from both AR and NAR training. Extending this idea to speech thus seems natural.

### Questions
- Stochastic Span Truncation (SST) is unclear and seems wrong. I understand the motivation of such design, but it does not seem correct to me. If you randomly crop the last audio span (while the complete text sentence is already generated), does it mean that the generation of the final audio span does not adhere to the previously generated text sentence?

**Note**

If all my concerns and questions are resolved I will consider raising the score to 6.

### Soundness
2

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
This paper proposes to combine AR and NAR modeling in SpeechLLMs, based on the motivation that text token generation has a strong dependence on previously generated text token, while speech token generation is strongly dependent on text tokens. To make the model work, they also proposed 3 training tasks/schemes to mitigate training-inference discrepancy. Experiments show the effectiveness of modeling text token with AR method and audio tokens with NAR method.

### Strengths
Novelty. Autoregressive modeling for text tokens and non-autoregressive modeling for audio tokens all in the same Transformer model is a novel methodology contribution. The proposed training tasks make sense and are empirically verified to work well with the model.

### Weaknesses
1. the motivation is confusing, in particular, "target-target" and "source-target" needs more rigorous explanation - source-target language is used usually in seq2seq scenario such as machine translation, while for autoregressive LLMs, source and target are the same sequence.

2. The input format of AudioLLMs are not settled yet, for example there is Moshi's time-aligned interleaving and there is GLM's alignment-free fixed-length interleaving. I think the paper used the later, but it would make it easier to follow if the data format is stated clearly in early section (e.g. section 2)

3. experiment part is a bit weak because there is no comparison with SotA models on audio benchmarks.

### Questions
1. what is the text in AAC, SEC, ASC?

2. does any-order AR means the latency of the model is higher? can a user talk to the model in real time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the limitation of existing multimodal audio-text models that adopt unified autoregressive (AR) training while ignoring inherent dependency differences between text (target-target causal dependence) and audio (semantic reliance on source text). It proposes the Text-to-Talk (TtT) framework, which integrates AR text generation and non-autoregressive (NAR) audio diffusion within a single Transformer, leveraging absorbing discrete diffusion for a unified training objective, a modality-aware attention mechanism, and three training strategies to mitigate train-test discrepancies. Experiments show TtT outperforms AR/NAR variants of Qwen2.5 on Audio-QA and ASR tasks, but key shortcomings remain: no open-source code/Demo, insufficient comparison with field SOTA models, and lack of subjective evaluations for audio generation.

### Strengths
1. **Logically adapts to modal essence**: Precisely captures the inherent dependency differences between text (target-target causal dependence) and audio (semantic dependence on source text), with theoretically sound design of the unified training objective based on absorbing discrete diffusion, filling the gap in modeling "modal dependency asymmetry".

2. **Targeted architecture design**: The modality-aware attention mechanism and unified modeling within a single Transformer avoid error propagation and modal separation in traditional cascaded models, well-suited for the practical scenario of "interleaved text-audio generation" in speech conversations.

3. **Consistent internal experiments**: Ablation experiments validate the effectiveness of the three training strategies and architectural components, while scaling experiments (1.5B vs 3B) show a performance improvement trend, demonstrating logical internal experimental design.

### Weaknesses
1. **Lack of verifiability**: No interactive Demo or immediate open-source code/data, only a "subsequent open-sourcing" statement—hinders reproducibility and audio effect verification, undermining research credibility.

2. **Insufficient baselines**: Only compares Qwen2.5’s AR/NAR variants, excluding other recent field SOTA models—fails to prove industry positioning or highlight hybrid architecture innovation.

3. **Missing subjective evaluations**: Relies solely on quantitative metrics (WER, LLM scoring), ignoring MOS-like standard subjective indicators—cannot assess actual audio usability.

4. **Inadequate hyperparameter/deployment analysis**: No ablation for block-wise diffusion hyperparameters, no exploration of complex scenarios (long conversations, low-resource languages)—unclear deployment potential.

5. **Limited novelty**: AR-NAR+LLM-diffusion ideas are not original; fails to clarify core differences from existing works or prove adaptation irreplaceability.

### Questions
1. What's the specific open-source timeline? Can this paper provide a temporary repo or interactive Demo (e.g., Hugging Face Spaces) during rebuttal to verify reproducibility and audio quality through a display loop?

2. Can this paper add subjective evaluations (e.g., 1-5 MOS scoring, blind listening) to assess audio usability using a display loop?

3. What are TtT's core differences from existing AR-NAR works (e.g., CosyVoice)? Can this paper prove its advantages with experiments utilizing a display loop?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Text-to-Talk (TtT), a unified audio-text generation framework that combines autoregressive (AR) text generation with non-autoregressive (NAR) audio diffusion within a single Transformer backbone (Qwen2.5-Base). The model introduces a unified AR–NAR training objective based on absorbing discrete diffusion, and a modality-aware attention mechanism that enforces causal decoding for text and bidirectional context modeling for audio.

### Strengths
**1. Unified AR–NAR Training Framework**

The paper introduces an elegant and theoretically grounded method that unifies AR text generation and NAR audio diffusion in a single Transformer. This hybrid objective bridges two fundamentally different generation paradigms under a coherent formulation.

**2. Modality-Aware Architectural Design**

The proposed modality-aware attention effectively reflects the distinct properties of text and audio—maintaining causal decoding for text while allowing bidirectional modeling for audio spans. This design shows a strong understanding of each modality.

**3. Well-Designed Training Objective and Inference Gap Mitigation**

The authors incorporate a carefully structured training objective and propose several strategies (e.g., BANOM, PPM, SST) to reduce the training–inference discrepancy. These design choices demonstrate practical awareness of the challenges in aligning AR and NAR generation and help improve stability and sample quality during inference.

### Weaknesses
**1. Lack of Comparison with Strong Speech-Language Baselines**

The paper does not include comparisons with existing speech-language foundation models such as SALMONN or SpeechGPT, which already perform Audio-QA and ASR tasks. Without such baselines, it remains unclear whether the proposed approach truly outperforms existing speech-language models or if the observed improvements are merely relative to a text-only baseline.

**2. Limited Evaluation Scope**

The evaluation focuses only on Audio-QA and ASR, whereas prior Speech LLMs (e.g., SALMONN, SpeechGPT) cover a broader set of speech reasoning and generative tasks such as speech captioning, translation, and dialogue.

### Questions
**1. Regarding Lack of Comparison with Strong Baselines**

Could the authors provide results comparing TtT with existing speech-language models such as SALMONN or SpeechGPT to verify whether the proposed approach truly surpasses these baselines? (For audioQA or ASR)

**2. Regarding Evaluation Scope and Applicability**

Do the authors plan to evaluate their model on a broader range of speech-language tasks (e.g., speech captioning, translation, or dialogue) to test whether the proposed framework can generalize beyond Audio-QA and ASR?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Text-to-Talk (TtT), a multimodal large language model (MLLM) that unifies autoregressive (AR) text generation with non-autoregressive (NAR) audio synthesis via discrete diffusion within a single Transformer architecture. The core motivation is the identified fundamental asymmetry between text and audio modalities: text generation relies on strong target-target (causal) dependencies, while audio generation is primarily driven by source-target dependencies. To address this, the authors propose a hybrid training objective, using a standard AR cross-entropy loss for text spans and an any-order AR loss (equivalent to absorbing discrete diffusion) for audio spans. The paper provides a theoretical justification, showing that the combined training objective is an upper bound on the desired joint distribution. Furthermore, three specialized training strategies—BANOM, PPM, and SST—are introduced to mitigate train-test discrepancies inherent in this hybrid paradigm. Extensive experiments on Audio-QA and ASR tasks demonstrate that TtT significantly outperforms strong AR-only and NAR-only baselines, validating the effectiveness of the proposed approach.

### Strengths
1. The paper identifies a clear and compelling problem—the mismatch between a uniform AR objective and the distinct dependency structures of text and audio. The proposed hybrid AR-NAR framework is a principled and innovative solution to this problem.
2. The work is not merely empirical. It provides a solid theoretical grounding by framing the model within a partial-order factorization and proving that the practical training objective serves as a tight upper bound on the negative log-likelihood of the theoretical joint distribution. This strengthens the validity of the approach.
3. The three proposed strategies (BANOM, PPM, SST) are effective in addressing specific train-test mismatch issues. The ablation study clearly demonstrates their individual and critical contributions to the final performance, with SST showing a particularly dramatic impact on conversational ability.

### Weaknesses
1. Mismatch Between Training and Evaluation Tasks: The model was trained on a diverse set of tasks, including Automated Audio Captioning (AAC) using datasets like Clotho-v2 and MACS. However, the evaluation completely omits any assessment of this capability. It is unclear whether the model has effectively learned to understand and describe general, non-speech audio, which is a significant limitation given its training objective. 
2. Lack of Audio Quality Metrics: The Audio-QA evaluation, while innovative, is limited to semantic correctness. There is no objective or subjective evaluation of the audio quality (e.g., MOS - Mean Opinion Score) or prosody/naturalness of the generated speech, which is a crucial aspect of a speech-generation model.
3. Absence of Standard Conversational Benchmarks: The evaluation is conducted on custom-held splits of generic QA datasets. The model is not benchmarked against established, standardized benchmarks for conversational AI (e.g., URO-Bench[1]), making it difficult to compare its dialogue capabilities directly with other state-of-the-art spoken dialogue systems.
4. Limited Comparative Analysis: The experimental results only compare TtT against its own backbone (Qwen2.5) trained with AR or NAR objectives. There is no comparison with other recent, open-source, end-to-end audio-language models such as Mimi-Omni[2], SLAM-Omni[3], Moshi[4], GLM-4-Voice[5], or VITA-Audio[6]. This lack of external benchmarking makes it challenging to gauge the true competitive standing of the proposed method.
5. Clarity on "Source" Context: The term "source-target" for audio is sometimes ambiguous. While it's clear the primary source is the corresponding text, the model can also condition on previously generated audio tokens within the same span (as allowed by the any-order AR). The paper could more precisely delineate the relative importance of the textual context versus the intra-span audio context.

[1] URO-Bench: Towards Comprehensive Evaluation for End-to-End Spoken Dialogue Models

[2] Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming

[3] SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training

[4] Moshi: a speech-text foundation model for real-time dialogue

[5] GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot

[6] VITA-Audio: Fast Interleaved Cross-Modal Token Generation for Efficient Large Speech-Language Model

### Questions
See weakness. The main concerns lie in evaluation about tasks(training with AAC but not evaluated), benchmark(without common benchmarks),  and comparison (without comparing with other models).

### Soundness
3

### Presentation
3

### Contribution
3
