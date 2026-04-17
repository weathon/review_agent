# EQUALS: An Audio-Visual LLM with One-Stage Question-Guided Alignment and Flexible Fusion

- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Audio-Visual Question Answering (AVQA) has emerged as a crucial task for multimodal reasoning in human-computer interaction, requiring models to align and interpret visual and auditory signals conditioned on natural language questions. Despite recent progress, three key challenges remain: (1) difficulty in locating question-relevant segments within lengthy and redundant video streams, (2) suboptimal audio-visual alignment due to the decoupling between pretraining and task-specific supervision, and (3) insufficient flexibility in fusion strategies across diverse tasks. To address these issues, we propose EQUALS (onE-stage Question gUided Alignment and fLexible fuSion), a unified end-to-end AVQA framework. EQUALS integrates compression, alignment, and fusion within a single stage. Specifically, we interleave optimal transport-based loss modules before and after the question-guided pooling module to achieve fine-grained semantic alignment. To enhance adaptability in fusion, we introduce FlexFuseMoE, a mixture-of-experts module that supports early, mid, and late fusion via flexible expert routing. Experiments on MUSIC-AVQA and its challenging variant FortisAVQA demonstrate that EQUALS achieves new state-of-the-art results with interpretability. Our findings highlight the importance of jointly modeling alignment and fusion under explicit question guidance, offering a flexible and scalable solution for audio-visual understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces EQUALS, a unified single-stage audio-visual language model developed for the Audio-Visual Question Answering (AVQA) task.
The study focuses on addressing three key challenges in this area. First, video sequences often contain numerous segments that are irrelevant to the given question, making it difficult to extract and process meaningful information efficiently. Second, the separation between pretraining and task-specific learning limits the alignment between audio and visual modalities. Third, most existing fusion mechanisms lack the flexibility to adapt to different task conditions.

### Strengths
This paper demonstrates impressive performance improvements on both the MUSIC-AVQA and Fortis-AVQA datasets. While MUSIC-AVQA is known to have an imbalanced answer distribution, EQUALS maintains strong performance even on Fortis-AVQA, which mitigates such bias by dividing answers into head and tail categories based on their frequency. This indicates that the model does not rely solely on frequently occurring answers but instead exhibits a balanced generalization ability across diverse question types. Furthermore, the model achieves state-of-the-art accuracy overall, demonstrating robustness and scalability in the field of audio-visual question answering.

### Weaknesses
The overall experimental design and evaluation methodology are systematic and effectively support the model’s strong performance. Evaluating closed-source models such as GPT-4o and Qwen2.5-VL in a zero-shot setting is entirely reasonable. However, since these models do not directly target the MUSIC-AVQA task, including comparisons with other large multimodal models that explicitly address this benchmark would have made the study even more convincing.

For instance, while MAVEN is an appropriate and meaningful baseline, it remains at the preprint stage and appears to report slightly lower performance than QA-TIGER, which does not utilize large language models. Additionally, Meerkat: Audio-Visual Large Language Model for Grounding in Space and Time and CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios also tackle the MUSIC-AVQA task to some extent. Although these models adopt multi-task settings, including them for comparison or brief discussion would help clarify the position and contribution of EQUALS more clearly.

Finally, while EQUALS demonstrates excellent performance on the single MUSIC-AVQA task, it would be interesting to see future extensions toward multi-task or more general audio-visual reasoning settings, as many recent studies are moving in that direction. Such an expansion could further enhance the overall impact of this line of research

### Questions
Overall, this is an excellent and well-executed study. As far as I know, Meerkat: Audio-Visual Large Language Model for Grounding in Space and Time is also an AVQA model that employs Optimal Transport for cross-modal alignment, sharing a similar research direction with EQUALS. However, it seems that a direct performance comparison with Meerkat was not included in the paper. I am curious whether conducting such an experiment was infeasible. To my understanding, Meerkat reports higher performance than MAVEN, so including or briefly discussing this model could further clarify and strengthen the contribution of EQUALS within the current landscape of AVQA research.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces EQUALS, an audio-visual large language model for answering questions about videos with sound. The model is built as a single-stage, question-guided pipeline. It (1) aligns the question with audio and visual tokens using an optimal transport loss, (2) uses Question-Guided Pooling to keep only the audio and video regions relevant to the question, (3) aligns the compressed audio and video again, and (4) fuses them with FlexFuseMoE, a mixture-of-experts that adapts its fusion strategy based on the question. The final fused representation is fed to a language model to produce the answer.

### Strengths
1. Question-Guided Pooling focuses on the relevant time-frequency and spatial-temporal regions instead of passing the full video and audio, improving both efficiency and reasoning.

2. The use of optimal transport to align question with audio and question with video before pooling, and audio with video after pooling, ties supervision directly to what the question asks, instead of doing generic cross-modal pretraining first.

3. FlexFuseMoE routes features through different fusion experts (audio-heavy, cross-modal, etc.) depending on the question, which helps on challenging audio-visual reasoning and tail cases.

### Weaknesses
1. The paper does not clearly explain how FlexFuseMoE decides which expert or fusion style to use for a given question at inference time in human-interpretable terms.

2. The model applies optimal transport both before and after pooling, but the paper does not isolate which stage matters most. This makes it hard to judge which parts are essential.

3. The paper presentation needs to be improved. Some parts are hard to be understand by reviewers.

### Questions
1. Authors apply optimal transport between the question and each modality before pooling, and then optimal transport between audio and video after pooling. Can authors clarify the intended functional role of each stage?

2. FlexFuseMoE is described as enabling early, mid, and late-style fusion through different experts and routing. Can authors define, in plain terms, what behavior each expert class is intended to capture? Also, is expert identity fixed once trained, or is it purely emergent from training without any intended semantic role?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the task of Audio-Visual Question Answering (AVQA). Motivated by the issues of (1) difficulty in locating question-relevant segments, (2) suboptimal audio-visual alignment and (3) insufficient flexibility in fusion strategies, this paper introduces EQUALS, a unified end-to-end AVQA framework. Specifically, the interleave optimal transport-based loss modules before and after the question-guided pooling module helps achieve fine-grained semantic alignment. To enhance adaptability in fusion, the paper introduces FlexFuseMoE, a mixture-of-experts module that supports early, mid, and late fusion via flexible expert routing. Extensive experiments validate the superiority of the proposed method.

### Strengths
1. This paper is well-motivated and easy to follow.

2. The proposed method outperforms prior methods on two AVQA benchmarks, i.e., MUSIC-AVQA and FortisAVQA.

### Weaknesses
- The novelty of this paper is limited. For example, the pooling layer to obtain global semantics, and the basic design of Text-Audio/Video Alignment. These points are basic and common designs in the multimodal community, I am afraid the method could not bring some insights to the community.
- Paper writing and presentation needs to be further improved. For example, figures and tables should be placed on top of each page. Line 319 might be a new subsection. Related works should be placed in the main manuscript.
- The evaluation benchmarks are limited. It mainly contains two AVQA datasets only in the music domain.
- Important baselines are missing. Some previous SOTA methods, including Qwen2.5-Omni, Kimi-Audio, VideoSALMONN should be included.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The submitted manuscript addresses key challenges in AVQA task, including temporal redundancy, alignment difficulty, and insufficient cross-task fusion strategies. To tackle these issues, it proposes the EQUALS framework, which integrates compression, alignment, and fusion within a single stage. The proposed method has been validated across multiple datasets, demonstrating its effectiveness. Overall, the core idea of the paper shows a certain degree of novelty.

### Strengths
1. The problem statement is clear, and the motivation is well-defined.
2. The proposed EQUALS framework exhibits novelty and has been thoroughly validated on several datasets.
3. The paper is clearly written and easy to follow.

### Weaknesses
1. It remains unclear whether the performance gains in the QA task primarily stem from the inherent capabilities of large models rather than modeling of intrinsic audiovisual relationships.
2. The use of questions to localize key temporal segments in MUSIC-AVQA has already been explored in prior works, such as *PSTP-Net*, *TSPM* et., al. The authors are encouraged to discuss the distinctions between this work and those approaches.
3. Under the absence of spatial supervision signals, how are the visual and audio modalities aligned spatially? This point requires clarification.
4. The comparative experiments could be strengthened by including more recent audiovisual QA methods, as the current comparisons are too limited.
5. Minor writing suggestions include avoiding widowed words at the end of paragraphs and adding citations for the compared methods in tables.

### Questions
My main questions are reflected in the *Weaknesses Section*.

Additionally, although unrelated to the review decision, I am curious about how this manuscript differs from the version submitted to *AAAI 2026*.

### Soundness
3

### Presentation
3

### Contribution
3
