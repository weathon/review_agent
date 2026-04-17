# Query-Guided Spatial–Temporal–Frequency Interaction for Music Audio–Visual Question Answering

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Audio–Visual Question Answering (AVQA) is a challenging multimodal task that requires jointly reasoning over audio, visual, and textual information in a given video to answer natural language questions. Inspired by recent advances in Video QA, many existing AVQA approaches primarily focus on visual information processing, leveraging pre-trained models to extract object-level and motion-level representations. However, in those methods, the audio input is primarily treated as complementary to video analysis, and the textual question information contributes minimally to audio–visual understanding, as it is typically integrated only in the final stages of reasoning. To address these limitations, we propose a novel Query-guided Spatial–Temporal–Frequency (QSTar) interaction method, which effectively incorporates question-guided clues and exploits the distinctive frequency-domain characteristics of audio signals, alongside spatial and temporal perception, to enhance audio–visual understanding. Furthermore, we introduce a Query Context Reasoning (QCR) block inspired by prompting, which guides the model to focus more precisely on semantically relevant audio and visual features. Extensive experiments conducted on two AVQA benchmarks demonstrate the effectiveness of our proposed method, achieving significant performance improvements over existing Audio QA, Visual QA, Video QA, and AVQA approaches. The code is released under https://github.com/lik1996/QSTar.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the audio-visual question answering task, this paper proposes a Query-guided Spatial-Temporal-Frequency (QSTar) interaction method. QStar primarily consists of the query-guided multimodal correlation module, spatial-temporal-frequency interaction module, and query context reasoning block. Experiments conducted on the MUSIC-AVQA and AVQA datasets verify the effectiveness of the proposed method.

### Strengths
- The discussion on the audio frequency and query context is good.
- The proposed method achieves new state-of-the-art performance on the relevant datasets.
- In general, the proposed method is described clearly.

### Weaknesses
- Throughout the method section, the proposed network heavily relies on prevalent self-attention and cross-attention mechanisms. The query-guided multimodal correlation module is similar to the prior method, APL. Prior work, like TSPM, etc, has already explored the utilization of question modality for early fusion. Overall, the proposed method still falls into the convention of multimodal fusion, not providing sufficient advances or improvements.

- Although the introduction of frequency makes sounds, it incorporates an additional AST backbone for feature extraction, which may make the proposed framework more complex. An analysis of the efficiency is required to justify this.

- The paper lacks a discussion or comparison of MLLMs. For the studied audio-visual question answering problem, the current omni multimodal large language models, such as Qwen2.5-Omni, Ming-Omni, and video-SALMONN2, can be used. What performance can be achieved by such omni models?

### Questions
- In the Introduction, the paper highlights the advantages of frequency in music scenarios, which may be suitable for the MUSIC-AVQA dataset. But the AVQA dataset contains more diverse scenarios. Would the frequency be less effective in AVQA, or would the writing in the introduction make it inconsistent for different datasets (scenarios)? 
- The paper highlights the frequency by employing the AST backbone to extract features. Similar to VGGish, AST is also a general model for audio feature extraction. Why not directly use the identical AST in the temporal audio extraction? 
- The query-context reasoning block utilizes the 'context-related keywords' (Line 307). How these context-related keywords are obtained?

- Figure 2 can be improved. For example, the abbreviation of several modules should be added at the end of the corresponding full module names.

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
3

### Summary
The paper introduces QSTar, a novel method for Audio-Visual Question Answering (AVQA) with a strong focus on complex musical scenes. The core contribution is a multi-stage, query-guided architecture that injects textual guidance throughout the pipeline: early with a Query-Guided Multimodal Correlation (QGMC) module, in the middle via a Spatial-Temporal-Frequency Interaction (STFI) module, and finally with a Query Context Reasoning (QCR) block that uses task-aware prompts. The model explicitly enhances audio processing by incorporating frequency-domain analysis to capture timbral characteristics crucial for instrument identification. The method achieves new state-of-the-art performance on the MUSIC-AVQA benchmark, reporting an average accuracy of 78.98%. A smaller-scale evaluation on the general AVQA dataset further demonstrates the model's robustness.

### Strengths
- Strong Empirical Performance: The method achieves state-of-the-art results on MUSIC-AVQA, with consistent improvements across Audio, Visual, and Audio-Visual question types.
- Thorough Ablation Studies: The paper provides extensive ablations that validate the contribution of each key module (QGMC, STI, TFI, QCR) and design choice, strengthened by further controls in the supplementary material.
- The Query Context Reasoning (QCR) block implements a lightweight and reproducible prompting mechanism. It uses a fixed set of task-relevant keywords (e.g., instrument type, duration, location) derived from the dataset's question types, which are encoded and used to guide the final fusion of audio-visual features.

### Weaknesses
At the top: I have reviewed an earlier version of this paper. The current submission remains largely unchanged, except for minor language polishing. All major methodological and empirical concerns I previously raised remain unaddressed.
- First of all, the motivation: The paper makes broad claims that prior work treats audio as secondary and uses text late, really? How the author provide some results/examples to demonstrate it? Reader could NOT buy the motivation just by plain sentences. To summarize, the motivation does not substantiate these claims with targeted analysis or experiments on specific baselines.
- The author say AST is superior to VGGish for capturing timbral information, why? this claim is untested.
- I think the dataset scope is too narrow: Strong focus on MUSIC-AVQA; AVQA appears only as a small table; no Pano-AVQA. Limits generality claims beyond music scenes.
- My another concern drop into 'Prompting fairness': The fixed keyword prompts mirror the dataset’s question taxonomy. In the wild, question types are not given and may differ (long-tail intents, non-music scenes). The models' performance might hinges on dataset priors rather than generalizable reasoning.
- Also, given the recent progress of MLLMs (e.g., GPT-4o, Phi series) in vision-language tasks, I suggest that the authors either: (1) include a comparison between QSTar and one or more MLLM baselines, or (2) clearly explain why such comparison is not applicable and discuss the corresponding limitations.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes QSTar, a Query-guided Spatial–Temporal–Frequency Interaction framework for Audio–Visual Question Answering task, with a focus on music-related datasets. The key idea is to enhance multimodal reasoning by introducing frequency-domain analysis alongside spatial and temporal interactions. The method introduces three main components: Query-Guided Multimodal Correlation (QGMC) module for early query-conditioned feature alignment; Spatial–Temporal–Frequency Interaction (STFI) module for fine-grained multimodal fusion; and Query Context Reasoning (QCR) block inspired by prompt-based context modeling.
Experiments on MUSIC-AVQA show that QSTar achieves competitive results.

### Strengths
- It is interesting and meaningful to introduce frequency-level interaction in AVQA tasks. This idea is natural given the acoustic characteristics of music-related videos and provides a new perspective besides conventional spatial–temporal fusion.
- The overall framework is well structured, clearly integrating linguistic guidance across multiple stages of the model.
- The implementation details and experimental setup are well described, and the ablation studies comprehensively verify the contribution of individual modules.

### Weaknesses
- The experimental analysis is rather shallow. Most evaluations focus only on accuracy numbers. More in-depth discussion on how frequency-level cues contribute to specific question types or modalities would strengthen the claim. For example, what kinds of questions benefit most from frequency reasoning and why?
- The improvements over recent strong baselines (e.g., QA-TIGER) are relatively modest—about 1–2% overall—and mainly appear on Audio QA and temporal AVQA subsets. There are no gains in Visual QA. This raises questions about the generality of the proposed design beyond audio-dominant scenarios.
- The ablation results show only slight differences between “with QCR” and “without QCR,” suggesting that the reasoning block contributes limited additional value.
- The paper would benefit from more qualitative or case-level studies, such as visualizing which frequency bands or audio/video segments are attended to under different query types. This would make the contribution of “frequency-level interaction” more convincing.

### Questions
Please check the above section.

### Soundness
3

### Presentation
2

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
The submitted manuscript addresses the problem that previous studies in AVQA task often overlooked the importance of the audio modality. To tackle this issue, the authors propose a Query-guided Spatial-Temporal-Frequency framework (QSTar), which effectively integrates question-guided cues with the distinctive frequency-domain characteristics and spatio-temporal perception of audio signals to enhance audio-visual understanding. Experiments conducted on relevant datasets demonstrate the effectiveness of the proposed method. Overall, the framework exhibits a clear novelty.

### Strengths
1. The problem addressed in the manuscript is clearly defined and well-motivated. By focusing on frequency-domain information from the audio perspective, the work presents a distinctive approach that draws attention.
2. The proposed QSTar framework shows a degree of originality and has been extensively validated on multiple datasets, confirming its effectiveness.
3. The writing is clear and well-structured, making the paper easy to read and understand.

### Weaknesses
1. Figure 1 effectively illustrates how frequency-domain cues assist in detecting instrument activities that purely spatial or temporal reasoning may miss. However, can similar improvements be observed for fine-grained visual understanding?
2. How does the model handle off-screen audio sources, such as when the sound of an instrument exists in the audio but the corresponding instrument does not appear in the video?
3. Considering the limited spatial supervision for audio-visual correspondence, how does the method effectively associate audio and visual cues, especially for rare instruments such as the suona or erhu?
4. Although the motivation for QCR originates from the analysis of the MUSIC-AVQA dataset, the paper does not discuss in depth the specific prompt forms or their performance compared with more dynamic or generative alternatives, raising some concerns about the scalability of the approach.
5. Some writing suggestions include avoiding widowed words at the end of paragraphs and adding references for the comparative methods listed in the tables.

### Questions
My main questions are reflected in the Weaknesses Section.

### Soundness
3

### Presentation
3

### Contribution
3
