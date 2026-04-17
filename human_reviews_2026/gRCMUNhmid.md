# Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Recent Multimodal Large Language Models (MLLMs) achieve promising performance on visual and audio benchmarks independently. However, the ability of these models to process cross-modal information synchronously remains largely unexplored. In this paper, we introduce: 1) Daily-Omni, an Audio-Visual Questioning and Answering benchmark comprising 684 videos of daily life scenarios from diverse sources, rich in both audio and visual information, and featuring 1197 multiple-choice QA pairs across 6 major tasks; 2)Daily-Omni QA Generation Pipeline, which includes automatic annotation, QA generation and QA optimization, significantly improves efficiency for human evaluation and scalability of the benchmark; 3) Daily-Omni-Agent, a training-free agent utilizing open-source Visual Language Model (VLM), Audio Language Model (ALM) and Automatic Speech Recognition (ASR) model to establish a baseline for this benchmark. The results show that current MLLMs still struggle significantly with tasks requiring audio-visual integration, but combining VLMs and ALMs with simple temporal alignment techniques can achieve substantially better performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Daily-Omni, a new audio-visual question answering (AVQA) benchmark comprising 684 real-world daily-life videos and 1,197 multiple-choice QA pairs spanning six diverse tasks, from audio-visual event alignment to complex cross-modal reasoning. The authors also propose a scalable Daily-Omni QA Generation Pipeline with five automated modules, enabling efficient annotation with high quality (30% acceptance rate) and minimal human effort (~30 hours for one annotator). To establish a strong open-source baseline, they present the Daily-Omni Agent, which integrates off-the-shelf visual language models, audio language models, and automatic speech recognition systems without fine-tuning. Experiments show this agent achieves state-of-the-art performance among open-source methods, while also revealing that current multimodal large language models (MLLMs) still struggle with tasks requiring deep audio-visual temporal integration. The work highlights the effectiveness of modular, alignment-based approaches and provides a valuable resource for advancing multimodal reasoning research.

### Strengths
- This paper is well written and easy to follow.
- The authors devise a daily-omni QA generation pipeline to curate an audio-visual question answering dataset, providing a new AVQA dataset to the research community.

### Weaknesses
- The paper omits several relevant large models in the audio-visual question answering (AVQA) domain, such as VITA [1]. VITA is a multimodal large language model specifically designed for AVQA tasks. The authors could strengthen their empirical evaluation by including VITA in their benchmark comparison to better contextualize the performance of their proposed approach.

- The related work section overlooks several important AVQA datasets, including FortisAVQA [2] and MUSIC-AVQA-R [3]. These datasets enable more comprehensive and fine-grained evaluations. For instance, FortisAVQA [2] supports generative AVQA evaluation rather than being limited to multiple-choice questions. Given the rapid progress in large language models, generative answering capabilities, as demonstrated in works like [2] are increasingly aligned with real-world application requirements. In contrast, multiple-choice QA may not fully capture the reasoning and generation abilities of modern multimodal large language models (MLLMs).

- The novelty of the Daily-Omni Agent pipeline appears limited. It reads more like an engineering integration of existing components rather than a methodological contribution. The paper does not clearly articulate what conceptual or technical innovation this framework introduces beyond modular composition.

- There are a few minor typos. For example, in Section 1, the second line contains the phrase "of and interactions," which seems grammatically incorrect and should be revised.

References:  
[1] https://github.com/VITA-MLLM/VITA  
[2] FortisAVQA and MAVEN: A Benchmark Dataset and Debiasing Framework for Robust Multimodal Reasoning  
[3] Look, Listen, and Answer: Overcoming Biases for Audio-Visual Question Answering

### Questions
- In Section 3.2, what is the sampling interval used to segment the videos? Specifically, for a 30-second clip, how are the 10 segments selected? Are they sampled at fixed intervals, or is another strategy (e.g., random or content-aware sampling) employed?

- Also in Section 3.2, the paper states that a single annotator can complete quality filtering for all 1,197 QA pairs within 30 hours. Can the authors clarify how this single-annotator process ensures consistent and reliable dataset quality? Was inter-annotator agreement measured or pilot testing conducted to validate annotation reliability?

- I observe that the Daily-Omni Agent workflow closely mirrors the pipeline used for dataset curation (e.g., leveraging ASR, VLM, and ALM with temporal alignment). Does this imply that the agent’s strong performance on the Daily-Omni benchmark is partly due to the alignment between the data generation process and the agent’s design? In other words, could the benchmark be inadvertently biased toward this specific modular architecture?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the ability of Multimodal Large Language Models (MLLMs) to perform synchronized audio-visual reasoning. It introduces the Daily-Omni benchmark with 684 daily-life videos and 1,197 QA pairs across six tasks, the Daily-Omni QA Generation Pipeline for scalable and efficient QA creation, and the Daily-Omni Agent, a training-free model combining VLMs, ALMs, and ASR through temporal alignment. Experiments show that existing MLLMs struggle with audio-visual integration, while simple temporal alignment notably enhances multimodal reasoning performance.

### Strengths
1. Compared with WorldSense, this paper introduces a new benchmark dataset, an automated QA generation pipeline, and a training-free agent. Although the improvements are incremental, they hold practical significance within the field of multimodal research.
2. The proposed Daily-Omni benchmark covers diverse daily-life scenarios from multiple sources, including music, speech, and various environmental sound events. It also provides a complementary QA generation pipeline, offering a useful tool for future data creation and extension.
3. The paper evaluates multiple MLLMs on the Daily-Omni benchmark, demonstrating its challenging nature, while the proposed Daily-Omni Agent achieves state-of-the-art performance among open-source methods.

### Weaknesses
1. The paper presents a multi-stage data annotation and QA construction process based on Gemini 2.0 Flash and Deepseek-R1, which, while complete, functions more as a systematic workflow connected primarily through prompt engineering.

(1) Although each 30-second video is divided into three 10-second segments and each 60-second video into three 20-second segments, the entire process still relies heavily on Gemini 2.0 Flash’s interpretation of these clips. The method for aligning visual and audio events simply involves feeding the full audio-visual segment into Gemini 2.0 Flash and asking the model to identify corresponding visual events for each audio cue, essentially constituting a straightforward application of existing models.

(2) The QA construction stage relies on simple prompt-based generation using Deepseek-R1, which appears overly direct and lacks methodological depth.

(3) While the resulting Daily-Omni QA is presented as an automated data generation pipeline, the final human filtering process retains only about 30% of the generated samples, indicating that the automatically produced content remains inconsistent in quality and still requires substantial human effort.

2. The implementation of the Daily-Omni Agent primarily builds on existing model capabilities in a relatively straightforward and procedural manner. After dividing the video and audio streams into three equal temporal segments, the agent applies pre-trained models for event alignment and audio-visual understanding. While the authors note that WorldSense lacks explicit training strategies or architectural guidance for improving model performance, the proposed approach similarly does not provide concrete solutions to this issue.

3. I think that the state-of-the-art performance of the Daily-Omni Agent largely stems from its extensive use of multiple tools and models, such as Qwen2.5-VL and Qwen2.5-14B-Instruct, through repeated and combined applications. In contrast, other models in the comparison are evaluated directly on the benchmark without similar multi-tool integration or repeated inference.

### Questions
Is it sufficient to systematically evaluate a model’s performance across different temporal scales using only 30-second and 60-second clips? In fact, the difference between 30 and 60 seconds remains within the same order of magnitude, which limits the benchmark’s ability to assess model behavior over truly varying temporal durations ranging from seconds to hours.

### Soundness
3

### Presentation
2

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
This paper proposed Daily-Omni, a new AVQA benchmark designed to evaluate the synchronous, cross-modal reasoning capabilities of MLLMs. The proposed scalable, MLLM-assisted pipeline allows for a substantial reduction in data construction costs. Additionally, the authors propose Daily-Omni-Agent, an agent-based MLLM that achieves superior performance over open-source MLLMs on the proposed benchmark. Further comparative analysis reflects that existing audio-visual models still have significant room for improvement in audio-visual cross-modal understanding. By comparing the performance of the omni-model, visual-only models, and text models, it is demonstrated that Daily-Omni indeed requires joint audio-visual reasoning capabilities.

### Strengths
- The paper is well-written and easy to understand, the designed pipeline is reasonably structured, and the performance achieved by the proposed Daily-Omni-Agent reflects the current limitations of existing MLLMs.
- The comparative analysis across different models validates the importance of audio for video comprehension, thereby underscoring the necessity of developing an appropriate benchmark.

### Weaknesses
- I noticed that an earlier work, AVUT [1], s highly relevant to the research presented in this paper. The authors should provide a comparative analysis in the manuscript to clearly distinguish the contributions of this work from the prior study.
- This benchmark's reliance on data drawn solely from existing datasets is problematic for two reasons. 1) It potentially undermines the validity of the test, as the data may have been seen or is out of its original context. 2) It bypasses the critical, foundational challenge of raw data sourcing and filtering. The quality of a benchmark is heavily dependent on its raw data, and an ideal data construction pipeline should feature an automatic or semi-automatic mechanism for this curation.

[1] Yang et al., "Audio-centric Video Understanding Benchmark without Text Shortcut", EMNLP 2025.

### Questions
I am curious whether the authors plan to extend the evaluation to an open-ended format. If so, do they have any insights regarding the benchmark design and scoring mechanism for such a setup? (This question does not affect my assessment of the paper.)

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
4

### Summary
The submitted manuscript addresses the limitation of current multimodal large language models in synchronously processing cross-modal information from both visual and audio streams. To tackle this issue, the authors introduce a new benchmark dataset named Daily-Omni for Audio-Visual Questioning and Answering benchmark, along with a Daily-Omni QA Generation framework and a Daily-Omni-Agent model. The proposed system integrates open-source vision-language models, audio-language models, and ASR technology, enabling temporal-aware reasoning without the need for additional training. Overall, the work demonstrates a notable degree of novelty.

### Strengths
1. The manuscript clearly identifies the problem and presents a well-defined motivation.
2. The proposed Daily-Omni benchmark contributes to advancing research in audio-visual reasoning.
3. The Daily-Omni QA Generation framework shows strong potential for extensibility and future development.
4. The writing is clear and well-organized, making the paper easy to read and understand.

### Weaknesses
1.It is recommended to discuss the differences between this work and related studies such as *MMAU*, *AURA*, and *OmniVideoBench*.

2. The generalization experiments for Daily-Omni are insufficient. It remains unclear whether the model demonstrates a truly generalizable multimodal reasoning ability or if its effectiveness is limited to the constructed dataset.

3. The validity of the results may largely stem from the inherent capabilities of the large models used. How are the spatio-temporal associations between audio and visual elements verified?

4. Section 4.3 requires a more detailed explanation of aspects such as event pair matching, threshold selection, and grounding determination.

5. Considering that MUSIC-AVQA is one of the established benchmarks for audio-visual reasoning, why not conduct a more in-depth exploration on this dataset?

6. It is strongly recommended to use vector graphics for figures to improve visual quality and readability.

### Questions
My main questions are reflected in the Weaknesses Section.

### Soundness
3

### Presentation
3

### Contribution
3
