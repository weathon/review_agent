# Vox-Infinity: Benchmarking the Limits of Long-Context Spoken Language Models

- Decision: Reject
- Scores: 6, 4, 8, 4

## Abstract
Long-context reasoning remains a fundamental challenge for large language models, as excessively long inputs often lead to the forgetting of salient information. This issue is even more pronounced in the speech domain, where audio, as a low-compression modality, requires significantly more embeddings than text to preserve both semantic content and acoustic cues. To address this, we introduce \textbf{Vox-Infinity}, the first benchmark specifically designed to evaluate long-context understanding in spoken language models. Vox-Infinity systematically extends audio history along two dimensions: turns and duration. It covers a diverse range of representative scenarios, including dialogues with varying structural depth and semantic complexity. Crucially, it provides explicit answer provenance annotations and organizes samples based on the context length required to resolve each query, enabling precise and length-aware evaluation of model performance. Furthermore, we present the first comprehensive study of history modeling strategies in this setting, analyzing how models balance the trade-off between preserving long-range semantics and retaining recent acoustic signals. Cases and datasets are available at \url{https://vox-infinity.github.io}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a benchmark for evaluating long-context understanding in spoken language models. It scales dialogue history by turn count and duration, includes answer provenance annotations, and covers diverse scenarios. The study compares text-, audio-, and hybrid-history strategies, highlighting trade-offs and proposing Modality-Aware Positional Editing (MoPE) for efficient context modeling.

### Strengths
1. The paper is highly original in framing long-context understanding as a core challenge in spoken language modeling, an underexplored domain compared to text-based benchmarks. 

2. The paper demonstrates strong methodological quality through detailed dataset construction, human validation, and systematic model benchmarking. I

3. The paper is well-organized, easy to follow, and supported by clear figures, tables, and examples.

### Weaknesses
1.  Evaluation is limited to a few models; broader inclusion of emerging audio-LLMs could better validate Vox-Infinity’s scalability and generality.

2. The benchmark focuses mainly on QA tasks; adding generative or reasoning evaluations would strengthen coverage of long-context language understanding.

3. MoPE’s algorithmic details lack ablation on parameter sensitivity; reporting computational cost variations would clarify its efficiency and reproducibility.

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a new benchmark, Vox-Infinity, benchmarking long context for spoken language models. It mainly compares on three models: GLM-4 Voice, Mimo-Audio and Qwen2.5-Omni. The evaluation reveals that text-based history modeling remains more stable as context length increases, while audio-based history preserves crucial acoustic cues but degrades rapidly with longer contexts. The austhors also propose a new approach (MoPE) to handle this problem.

### Strengths
- First benchmark specifically for evaluating long-context spoken dialogue.
- 154 hours of audio covering four scenarios.
- Propose Modality-Aware Positional Editing, a way to modify positional encodings.

### Weaknesses
- Only three models evaluated (GLM-4-Voice, Qwen2.5-Omni, Mimo-Audio) while many other spoken dialogue systems exist.
- Section 4.2 dedicates significant space to a relatively incremental position encoding method, not within the scope of a benchmark paper.
- Relying solely on GPT-4o for accuracy assessment without human evaluation baseline or inter-rater agreement is problematic. 
- Table 2 is dense with too many columns making trends hard to identify.
- While latency figures are provided, the paper lacks a discussion on memory requirements.

### Questions
- Given the paper's reliance on GPT-4o for evaluation , have the authors conducted any human validation studies or inter-rater agreement checks to confirm the reliability of these automated judgments?
- Regarding the "Conversational Dialogues" subset, could the authors specify which Text-to-Speech (TTS) model was used for audio synthesis and what methods were employed to manage speaker identity?
- The paper provides valuable latency metrics, but could the authors also report the memory requirements (e.g., VRAM) for running the benchmark? This information is critical for assessing the total computational cost.
- Could the authors elaborate on the decision to dedicate a significant portion of the analysis (Section 4.2) to the new MoPE method? 
- The evaluation is limited to three models. Consider adding AudioFlamingo3, Qwen-2 Audio, Step-Audio, UltraVox, GPT-Realtime etc.
- Would the authors consider providing an aggregate performance metric or reorganizing Table 2? The current dense format makes it difficult to quickly identify performance trends.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the very first benchmark specifically designed to evaluate the limits of long-context understanding in spoken language models. The benchmark systematically scales context length across two dimensions: dialogue turns and turn duration, and includes four diverse scenarios: Ultra Multi-turn Dialogue, Conversational Dialogue, Personal Monologue, and Beyond-Semantic Dialogue. The paper also provides a comparison of text-only, audio-only, and hybrid history modeling strategies, demonstrating key trade-offs in semantic retention versus acoustic preservation.

### Strengths
1.	The motivation behind the paper is strong, addressing the need for a dedicated, rigorous benchmark for long-context spoken language models.
2.	The paper itself is clearly written.
3.	The analyses in the paper are well-designed.

### Weaknesses
1.	Limited numbers of LALMs, as only 3 open-source models are provided.

### Questions
1.	Why, in Table 1, Length Distribution for Conversational Dialogue presents 0-2min and 2-4min, but in Table 2, numbers for 4-8min and 8min+ are also provided?
2.	What does “Value” in Figure 3(a) indicate?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a new benchmark for analyzing spoken language models (SLMs) on long-context tasks. Existing long-context benchmarks typically cover either short scenarios (30 seconds – 1 minute) or extremely long ones (around 1 hour). However, these settings are either too short or too long to reflect realistic conversation lengths. This work fills the gap by introducing benchmark clips ranging from less than 2 minutes to 8 minutes.

The authors design four dialogue scenarios and categorize the testing clips into four types:
(a) Ultra Multi-turn Dialogue – frequent back-and-forth exchanges involving requirement verification and detail clarification.
(b) Conversational Dialogue – extended, casual interactions emphasizing sustained conversation and advice-giving.
(c) Personal Monologue – single-speaker, long-form speech such as lectures or presentations.
(d) Beyond-Semantics Dialogue – conversations enriched with acoustic information (e.g., environmental sounds, sound effects, or music) beyond pure semantics.

The study shows that existing SLMs perform well when using text-based context histories, with little degradation from short to long contexts. However, when audio is used as the context, performance degrades significantly in long-context scenarios. Interestingly, the authors also demonstrate that audio information is essential, as audio-based context modeling outperforms text-based modeling in the Beyond-Semantics Dialogue category.

Finally, they propose a hybrid context modeling approach that uses text for distant history and audio for recent history. However, the effectiveness of this hybrid method is not well established by the presented experiments.

### Strengths
- A clear gap in evaluating long-context modeling in SLMs
- Include the "Beyond-Semantics Dialogue" category that gauge the essentialness of audio modeling.

### Weaknesses
### Review Comment

- Given that this is a speech benchmark, I would expect most of the testing scenarios to require the model to rely on **paralinguistic** or **acoustic information** in the context history. However, only one category currently addresses the importance of acoustic information, and it focuses on **sounds and music** rather than other rich aspects of speech beyond verbal content (e.g., **speaker identity**, **emotion**, **tone**). This design makes the benchmark easily solvable using **pure text-based context modeling**. As shown by the experiments in the paper, none of the context modeling methods involving audio significantly outperform pure text modeling, including the proposed hybrid approach. Audio-based methods merely catch up with text-based modeling, but at a significantly higher **latency cost**. Ideally, this benchmark should advocate for audio-based context modeling, yet the results tell the **opposite story**.

- The effectiveness of the proposed **hybrid modeling** is also questionable. The paper only reports results on the *“Ultra Multi-Turn Dialogue”* category, where acoustic signals are largely unnecessary. What about the *“Beyond-Semantics Dialogue”* category? How does the hybrid method compare to the pure audio-based method in that case? Since the hybrid method only models acoustics for recent history, it is very likely to outperform the pure audio-based method in *“Ultra Multi-Turn Dialogue”*, but underperform it in *“Beyond-Semantics Dialogue”*. If this is the case, the contribution of the proposed hybrid method becomes **unclear**.

---

### Summary of Weaknesses

1. The benchmark does **not fully consider the rich acoustic information** in speech to design tasks that require the SLM to *“listen to”* the context rather than merely *“look at”* it.  
2. The paper does **not propose an effective context modeling method** that surpasses text-based approaches by additionally modeling context audio. This limitation may stem from weaknesses in either the **benchmark design** or the **method design**.

### Questions
No

### Soundness
2

### Presentation
3

### Contribution
2
