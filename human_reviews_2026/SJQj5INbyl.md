# AVI-Bench: Toward Human-like Audio-Visual Intelligence of Omni-MLLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Recent breakthroughs in Omni-Multimodal Large Language Models (Omni-MLLMs), such as GPT-4o, have showcased remarkable progress in integrating visual and audio modalities with language, bringing us closer to human-like audio-visual intelligence. However, a critical gap remains: the lack of systematic benchmarks to rigorously evaluate these models’ audio-visual capabilities. Existing evaluations are often fragmented, focusing on isolated tasks and overlooking the multifaceted nature of audio-visual intelligence.
To address this, we introduce \textbf{AVI-Bench}, a cognitively inspired benchmark designed to assess Omni-MLLMs across three stages: perception, understanding, and reasoning. Each stage comprises cross-modal tasks that require simultaneous interpretation of visual and audio inputs, enabling fine-grained diagnostics of model strengths and weaknesses.
To further explore models’ robustness against unfamiliar sensory inputs, we propose \textbf{AVI-Bench-PriSe}, an extension targeting the ``primitive sensation'' of Omni-MLLMs on unfamiliar-domain audio-visual inputs with low-semantic stimuli, thereby probing their generalization beyond commonly used general-domain training data.
Through comprehensive experiments on both open- and closed-source models, AVI-Bench reveals critical limitations and bottlenecks of current Omni-MLLMs. Building on these insights, we present a four-level taxonomy for classifying the audio-visual intelligence.
Our work provides the community with a principled evaluation framework that not only benchmarks performance but also guides future development toward more robust, adaptive, and human-aligned audio-visual intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents AVI-Bench, a benchmark to evaluate Omni-MLLMs across omni-modal perception, understanding, and reasoning ability. To explore the Omni-MLLM's generalization towards unfamiliar sensory inputs, they further propose AVI-Bench-PriSe, an extension of AVI-Bench that focuses on unfamiliar-domain audio-visual inputs. AVI-Bench contains over 5000 samples and was evaluated on more than 10 open-source and closed-source models, which showed that perception and understanding might be bottlenecks for reasoning, and on unfamiliar input, there is still a large room for improvement.

### Strengths
1. **Comprehensive tasks.** AVI-Bench incorporate 13 task types across categories such as understanding, perception, and reasoning, allowing for a thorough evaluation of Omni-MLLM’s understanding across different aspects.

2. **Well-grounded tasks.** Most tasks in AVI-Bench require comprehension across more than one modality, enabling a comprehensive assessment of multimodal collaboration capability for Omni MLLMs. In particular, it can also evaluate the ability to understand multi-segment audio and multiple images.

3. **4-level Classification Scheme** This paper proposes a 4-level classification scheme, which is reasonable and insightful, provides good metrics for evaluating Omni-MLLM.

### Weaknesses
1. **Discussion with related work.** There is some related work on evaluating Omni-MLLM which are not compared and discussed with. The author should compare AVI-Bench with these works and show the main differences.  
[1] Hong, Jack, et al. "Worldsense: Evaluating real-world omnimodal understanding for multimodal llms." arXiv preprint arXiv:2502.04326 (2025).  
[2] Daily-Omni: Zhou, Ziwei, Rui Wang, and Zuxuan Wu. "Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities." arXiv preprint arXiv:2505.17862 (2025).  
[3] Video-Holmes: Cheng, Junhao, et al. "Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning?." arXiv preprint arXiv:2505.21374 (2025).

### Questions
1. **Adding more models.** Suggest adding more open-source Omni-MLLMs in this benchmark in Table 3 for a more comprehensive evaluation.  
[1] Mini-Omni2: Xie, Zhifei, and Changqiao Wu. "Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities." arXiv preprint arXiv:2410.11190 (2024).  
[2] Lyra: Zhong, Zhisheng, et al. "Lyra: An efficient and speech-centric framework for omni-cognition." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.  
[3] MiniCPM-o 2.6: Yuan Yao, et al. "Minicpm-o 2.6: A gpt4o level mllm for vision, speech, and multimodal live streaming on your phone."

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces AVI-Bench, a cognitively inspired benchmark designed to assess Omni-MLLMs across perception, understanding, and reasoning capabilities. The benchmark contains over 5K question–answer pairs covering 14 different tasks, grouped into perception, understanding, reasoning, and primitive sensation. To further approximate each Omni-MLLM’s performance in terms of human-like intelligence, the authors propose four levels of metrics to characterize audio-visual intelligence: task-adaptive intelligence, modality-adaptive intelligence, stage-adaptive intelligence, and domain-adaptive intelligence.

### Strengths
- Designing the benchmark in a cognitively inspired manner is an interesting direction for understanding the behavior of Omni-MLLMs from a human-like perspective.

### Weaknesses
- The contribution in constructing the dataset appears limited. As mentioned in Section A.6.1, most of the tasks are directly taken from existing benchmarks and only reformatted. The tasks newly created by the authors also overlap at a high level with existing ones. In addition, ASQA was previously proposed in AV-Odyssey Bench: Can Your Multimodal LLMs Really Understand Audio-Visual Information? (Arxiv, 2024).
While combining existing tasks from different benchmarks can be meaningful, the current scale of AVI-Bench, as shown in Table 1, is relatively small—insufficient to fully support claims of modeling human cognitive processes. What is the advantage of using subsets of existing benchmarks within AVI-Bench, compared to using the full datasets from each benchmark to evaluate Omni-MLLMs?

- In L330, the authors mention that PandaGPT-7B and PandaGPT-13B perform poorly in perception but well in understanding, which limits reasoning. However, perception should inherently affect understanding, as accurate understanding depends on accurate perception. Therefore, the categorization into four different tasks (perception, understanding, reasoning) may be somewhat entangled.

- Regarding Equation 1, each task uses a different evaluation metric with distinct meanings and scales. Is it appropriate to simply average all quantitative results across tasks? Were any normalization, weighting, or other methods applied to ensure consistency across tasks?

- Equation 2 also raises concerns. A and V represent performance on audio-only and visual-only dominant tasks, respectively, but why does A+V equal zero? Are A and V averages over tasks for each modality? Moreover, since the number of tasks and metric scales vary, is it valid to average them all? Additionally, from a cognitive perspective, the order should be perception → understanding → reasoning. Perception and understanding should not be treated as equivalent in priority.

### Questions
### Questions and Suggested Experiments

- Providing recommendations or experiments that could improve model performance in terms of human-like intelligence would improve this work.

### Minor Questions and Suggestions

- In Figure 1, for the visual-reference audio retrieval task, why is the correct answer “no” instead of selecting one option?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents AVI-Bench, a cognitively inspired benchmark for evaluating audio-visual intelligence in Omni-Multimodal LLMs such as GPT-4o and Gemini. It assesses models across three stages: Perception, Understanding, and Reasoning, as well as a Primitive Sensation stage for unfamiliar low-semantic inputs. The authors also evaluate various open-sourced and commercial models, revealing weaknesses in audio reasoning and cross-modal grounding.

### Strengths
- Comprehensive and cognitively grounded design: Mirrors human cognition and enables fine-grained diagnosis of model abilities.
- Extensive experimental coverage: Evaluates 28 models  (both open-sourced and commercial) across diverse tasks, giving strong empirical credibility.
- Insightful taxonomy and analysis: The four-level framework offers a clear lens to interpret model progress toward human-like intelligence.

Therefore, I believe it is a useful benchmark for audio-visual llms, even though similar benchmarks have been preoposed before.

### Weaknesses
- Considering similar benchmarks have been proposed, the main contribution is just an advanced benchmark for omni-MLLMs, not a pioneer in this field.
- Although the proposed AVI-Bench can evaluate the omni models under different modalities, it still cannot serve as a replacement of single modality evaluation, such as MMMU for image understanding.

### Questions
na

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
This paper presents a novel benchmark for evaluating Omni-MLLMs, encompassing 14 diverse tasks including classification, grounding, matching, retrieval, QA, and captioning, among others. The benchmark offers a more comprehensive evaluation framework, structured around Perception, Understanding, Reasoning, and Primitive Sensation. Notably, the paper reports several interesting experimental findings, such as the observed discrepancy between audio and visual intelligence. In addition, it introduces innovative metrics like the modality-adaptive score, which is designed to assess such modality-specific discrepancies.

### Strengths
1. The experimental findings are interesting, particularly observation 3, which could inform future research aimed at improving audio support.
2. The four-level metrics introduced in Section 5 are interesting, providing a novel framework for evaluating models’ abilities in terms of balanced modality and stage performance.
3. The experiments are comprehensive and solid.

### Weaknesses
Lack of novelty: The task types in AVI-Bench are largely already covered by existing audio-visual benchmarks. Moreover, AVI-Bench omits several important audio-visual tasks, such as speech-speaker matching and conversation-action-camera interleaved captioning, which are critical for real-world applications of audio-visual models.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3
