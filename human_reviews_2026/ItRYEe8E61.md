# OmniVideoBench: Towards Audio-Visual Understanding Evaluation for Omni MLLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Recent advances in multimodal large language models (MLLMs) have demonstrated substantial potential in video understanding. However, existing benchmarks fail to comprehensively evaluate synergistic reasoning capabilities across audio and visual modalities, often neglecting either one of the modalities or integrating them in a logically inconsistent manner. To bridge this gap, we introduce OmniVideoBench, a large-scale and rigorously designed benchmark dedicated to assessing synergistic audio-visual understanding, with a strong emphasis on modality complementarity and logical consistency. Specifically, OmniVideoBench comprises 1000 high-quality question-answer(QA) pairs, each annotated with step-by-step reasoning traces, derived from 628 diverse videos ranging from several seconds to 30 minutes, and manually verified to guarantee complete correctness and uniqueness. Moreover, OmniVideoBench encompasses 13 carefully designed question types, covering temporal reasoning, spatial localization, counting, causal inference, summarization, and beyond, thereby capturing the essential challenges of video understanding. Evaluation of multiple MLLMs on OmniVideoBench reveals a pronounced gap between model performance and human reasoning, with open-source models lagging significantly behind their closed-source counterparts, underscoring the inherent difficulty of genuine audio-visual reasoning. We will release OmniVideoBench to foster the development of MLLMs with stronger and more generalizable reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors propose OmniVideoBench, a benchmark for audio–visual understanding in videos. It consists of 1000 QA pairs from 628 videos ranging from several seconds to 30 minutes, within 13 question types, covering temporal reasoning, spatial localization, counting, causal inference, summarization, etc. They also evaluate multiple MLLMs on OmniVideoBench to performance investigation.

### Strengths
* Novelty

Currently,  the video understanding community mainly focuses on understanding and reasoning from the visual modality. It ignores the fact that video is the combination of visual and audio clues. This OmniVideoBench provides a bench for comprehensively evaluating reasoning capabilities across both modalities. Hence, it somehow shows the value of this bench for multimodal video understanding.

* Clarity

The paper is well-written with good structure. Hence, the clarity is basically good.

* Significance

This paper focuses on evaluating audio–visual understanding capacity of MLLMs, which is an important and practical problem for video understanding. Hence, the significance is basically OK for video research community.

### Weaknesses
* Question Type 

The authors choose 13 types for QA pairs. Please further explain why to choose these types. Is it sufficient to evaluate audio–visual understanding in videos? 

* Method Insight

It woule be more interesting to investigate or indicate how to design MLLMs to boost the tasks in this benchmark.

* Small Size

The authors collected only 628 original videos. The small number of videos would restrict the generalization of this benchmark.

### Questions
Please see the weakness section.

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
3

### Summary
This work proposes OmniVideoBench, a new benchmark designed to address key limitations in existing video datasets—namely, the lack of systematic evaluation of audio–visual co-reasoning and inconsistencies in logical task composition. The benchmark is constructed from 628 real-world videos ranging from a few seconds to around 30 minutes, covering three types of audio (speech, sound, and music). It includes 1,000 multiple-choice questions spanning 8 major categories, 68 subcategories, and 13 task types, each accompanied by human-annotated step-by-step reasoning chains.

Experimental results show that current models struggle significantly on this benchmark. While the best model, Gemini-2.5-Pro, only achieves 58.90% accuracy, and most open-source models perform near random. Besides, Long video understanding remains a major challenge for most models. Notably, performance drops sharply for music-dominated audio even for the strongest models. 

Overall, OmniVideoBench combines long temporal structure with audio–visual complementarity, providing a more realistic and comprehensive testbed for advancing multimodal video reasoning research.

### Strengths
* Broad Coverage and Task Diversity。 The dataset spans 8 high-level categories, 68 subcategories, and 13 task types, with video durations ranging from 4 to 1955 seconds. It also explicitly includes an Ultralong category for videos longer than 10 minutes.
* Step-by-Step Human-Annotated Reasoning Chains. Each question is accompanied by a human-labeled step-by-step reasoning chain, in terms of modality, evidence, and inference triples.
* Data Quality Control. The authors take multiple steps to ensure data quality, including a three-stage filtering pipeline and the exclusion of videos with large-scale on-screen subtitles that might leak answers or bias model predictions.
* Comprehensive Evaluation. The benchmark is evaluated using a wide range of models, including both open-source and close-source of varying scales.

### Weaknesses
* Limited Per-Task Coverage. The dataset contains 1,000 questions spread across 13 distinct task types, resulting in fewer than 100 examples per task on average. This limited coverage may constrain the robustness of task-specific evaluation and generalization analysis.
* Unbalanced Audio Category Distribution. The distribution of audio types is heavily skewed—Speech accounts for over three-quarters of the dataset, while Music constitutes only 9.1%. This imbalance may bias models and limit insights into performance under underrepresented audio conditions.
* Lack of Human Baseline. Although the conclusion emphasizes a large gap between human and model performance, no human baseline is reported in the experiments. 
* Dataset unavailable. The authors did not provide the dataset link in the paper.

### Questions
1. What is the core distinction between this work and WorldSense that is mentioned in Table 1?
Table 1 suggests that the two benchmarks share similar characteristics across several dimensions—including modality coverage, domain diversity, video type, audio type, and answer format. A more explicit comparison would help clarify the novel contributions of OmniVideoBench.
2. Qwen2.5-Omni performs better on WorldSense than on OmniVideoBench. Does this imply that OmniVideoBench presents a more difficult challenge?
Beyond video length, are there other factors or task-level distinctions that contribute to the increased difficulty of OmniVideoBench? Providing such an analysis would help justify the benchmark’s added value.
3. What is the significance of emphasizing long videos?
Table 1 shows that SOTA model performance is comparable between the 5–10 minute and 10+ minute video subsets. Additionally, WorldSense already includes videos of up to 10 minutes. Could the authors clarify what unique challenges OmniVideoBench introduces with longer videos?

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
4

### Summary
This paper introduces OmniVideoBench, a large-scale benchmark designed to evaluate the collaborative audiovisual reasoning capabilities of multimodal large language models. The benchmark comprises 1,000 manually annotated high-quality question-answer pairs (QA) across 628 videos, each featuring explicit step-by-step reasoning chains indicating modalities and evidence. OmniVideoBench spans 8 primary video genres, 68 subcategories, and 13 distinct task types (e.g., temporal, spatial, causal reasoning), structured to comprehensively evaluate modal complementarity and logical consistency. The paper evaluates both open-source and proprietary MLLMs on OmniVideoBench, revealing that model performance lags significantly behind human capabilities, particularly on tasks requiring genuine multimodal integration.

### Strengths
1. The paper ensures the richness and coverage of the dataset, comprising 1,000 distinct QA pairs that span a wide range of real-world scenarios, video durations, and audio types. It also contains 13 different task types covering diverse reasoning skills.
2. The annotation protocol ensures that all questions require true audio-visual integration and stepwise reasoning, with multi-layered filtering to weed out unimodal or bias-prone items.

### Weaknesses
1. The paper does not provide statistical results comparing it with other relevant datasets, which fails to highlight the contribution of the proposed dataset.
2 From Table 1 and Figure 3, it is evident that the vast majority of QA pairs relate to Speech (76.2%) versus Sound (14.7%) and Music (9.1%), creating considerable class imbalance.
3 The benchmark's positioning relative to several directly analogous or recently proposed audio-visual (AV) benchmarks is incomplete. Recent works, such as AVHBench (Kim et al., 2025) and DAVE (Radevski et al., 2024), are not cited or discussed, nor are specialized audio-visual QA datasets, including AVQA (Yang et al., 2022) and MusicAVQA (Li et al., 2022).
4. While human annotation is used in construction, the paper does not report human baseline accuracy or response variability for the main test set.
5. While Figure 1 gives some specific sample breakdowns, the paper lacks a deeper set of qualitative analyses of successful versus failure cases, especially for (a) long video cases and (b) music understanding tasks.

### Questions
1. Can the authors provide quantitative statistics on the efficacy of the automated filtering steps in Section 2.4? Specifically, what is the rejection or retention rate at each filtering stage, and what percentage of QA pairs end up truly requiring both modalities?
2. Please clarify how "semantic units" ($S_i$) are operationalized for the semantic distance metric. Is this manual phrase decomposition, or is some NLP toolchain applied? This is crucial to evaluating distractor design reproducibility.
3. Will human benchmark results (e.g., accuracy, agreement rates) on OmniVideoBench be reported?

### Soundness
3

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
The paper presents OmniVideoBench, a large-scale, carefully curated benchmark evaluating synergistic audio-visual reasoning in MLLMs. OmniVideoBench consists of 1,000 manually verified QA pairs with explicit step-by-step reasoning traces, based on 628 diverse long-form videos across 8 major categories and 68 subcategories. The benchmark emphasizes logical consistency, modality complementarity, and covers 13 question types relevant to real-world video understanding. The authors systematically filter and annotate the data to ensure questions demand audio-visual integration. Baseline experiments on both open- and closed-source MLLMs reveal a substantial gap to human-level reasoning, particularly regarding music, long videos, and abstract audio understanding.

### Strengths
1. The benchmark features 1,000 high-quality, manually verified QA examples, each annotated with step-by-step reasoning chains with human proofreading. This explicit annotation supports analysis of both model answers and reasoning processes.
2. Long Videos span 8 major categories and 68 subcategories, ensuring comprehensive coverage and evaluation of a wide range of real-world scenarios.
3. The authors conducted extensive and comprehensive evaluations and experiments.

### Weaknesses
1. Missing Comparisons with Key Recent Audio-Visual Benchmarks: Several highly relevant, recently released benchmarks should be compared and discussed [1,2,3]. 
2. It will be great to see the error analysis on the properties of the questions/items themselves—e.g., what makes some reasoning chains or audio-visual interactions particularly difficult?
3. More qualitative examples and error cases could be included in the appendix to provide deeper insights into model behavior and help readers better understand the paper.








[1] Sung-Bin, Kim, et al. "Avhbench: A cross-modal hallucination benchmark for audio-visual large language models." arXiv preprint arXiv:2410.18325 (2024).
[2]Chowdhury, Sanjoy, et al. "Avtrustbench: Assessing and enhancing reliability and robustness in audio-visual llms." arXiv preprint arXiv:2501.02135 (2025).
[3[ Sakshi, S., et al. "Mmau: A massive multi-task audio understanding and reasoning benchmark." arXiv preprint arXiv:2410.19168 (2024).

### Questions
Please check the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
