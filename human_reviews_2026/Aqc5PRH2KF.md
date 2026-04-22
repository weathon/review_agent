# ChineseVideoBench: Benchmarking Multi-modal Large Models for Chinese Video Question Answering

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
This paper introduces ChineseVideoBench, a pioneering benchmark specifically designed for evaluating Multimodal Large Language Models (MLLMs) in Chinese Video Question Answering. The growing demand for sophisticated video analysis capabilities highlights the critical need for comprehensive, culturally-aware evaluation frameworks. ChineseVideoBench addresses this gap by providing a robust dataset and tailored evaluation metrics, enabling rigorous assessment of state-of-the-art MLLMs on complex Chinese video content. Specifically, ChineseVideoBench comprises 8 main classes and 12 sub-classes, encompassing tasks that demand both deep video understanding and nuanced Chinese linguistic and cultural awareness. Our empirical evaluations reveal that ChineseVideoBench presents a significant challenge to current MLLMs. Among the models assessed, Gemini 2.5 Pro achieves the highest performance with an overall score of 77.9%, while Intern-VL-38B emerges as the most competitive open-source model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents ChineseVideoBench, a large-scale benchmark specifically designed to evaluate Multimodal Large Language Models (MLLMs) on Chinese Video Question Answering (VideoQA). The authors argue that existing benchmarks are predominantly English-centric, making it difficult to assess MLLM performance in non-English, culturally specific contexts such as Chinese videos, which contain unique linguistic and cultural cues.

A hierarchical structure of 8 main task categories (world knowledge, topic recognition, scene understanding, character recognition, temporal localization, object perception, action recognition, and logical reasoning) and 12 sub-tasks across 1,625 CC0-licensed videos and 6,507 manually annotated multiple-choice QA pairs.

The paper concludes that ChineseVideoBench exposes a performance gap between English-trained models and those trained with Chinese data, emphasizing the need for culturally aware multimodal evaluation.

### Strengths
### **Hierarchical and diagnostic task design**
The benchmark’s structure (8 categories, 12 sub-tasks) allows for fine-grained analysis of model capabilities, from perception-level (object, action) to reasoning-level (logic, world knowledge).

### **Comprehensive and fair evaluation**
The experiments evaluate over a dozen state-of-the-art MLLMs across closed and open domains, providing a rich comparative baseline. Tables 2 and 3 and Figures 1 and 4 clearly present detailed quantitative comparisons.

### **Insightful cultural analysis**
The paper demonstrates that models trained mainly on English data fail to generalize to Chinese contexts, highlighting the importance of multilingual and culturally grounded datasets for MLLMs.

### Weaknesses
### **Insufficient cross-lingual or ablation studies**

Although the authors highlight the gap between English- and Chinese-trained models, they do not include detailed ablation or transfer experiments to analyze why certain models fail or how fine-tuning on Chinese data affects performance. The benchmark provided is still monolingual.

### **Temporal reasoning analysis remains qualitative**
While the paper identifies temporal localization (TL) as a major bottleneck, the discussion is mostly descriptive. Quantitative or visualization-based error analysis (e.g., frame sampling vs. accuracy) would strengthen the conclusions.

### **Lack of Analysis**
While this paper share some distribution patterns of provided benchmark, they don't provide any further analysis. For example, Figure5 show how the video durations distribute, but this doesn't introduce any further insights, how long video/short video performs differently?

### Questions
Have you tested whether English-trained MLLMs can improve on ChineseVideoBench through few-shot or instruction-tuning using a small amount of Chinese video data? Explain more on why MLLMs perform bad on Chinese video QAs, and how hard it is to solve to provide a stronger motivation.

For other questions, I have put it on the weakness part. I would love to raise my score if my questions are solved.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ChineseVideoBench, a human-annotated benchmark for evaluating Multimodal LLMs on Chinese VideoQA. The dataset comprises 1,625 real-world Chinese videos across 11 domains and 6,507 multiple-choice QA pairs organized through a hierarchical task structure covering 8 categories and 12 sub-tasks.

Comprehensive evaluation of leading MLLMs reveals significant gaps: while top models like Gemini 2.5 Pro achieve 77.9% accuracy, this substantially trails human performance (94.8%). Models with Chinese training data show advantages, but all systems struggle with fine-grained spatiotemporal reasoning. The benchmark effectively exposes current limitations in culturally-grounded video understanding and provides valuable diagnostic insights for future model development.

### Strengths
This paper makes valuable contributions through its novel ChineseVideoBench dataset and rigorous evaluation. The benchmark features carefully curated videos and high-quality human annotations across diverse domains. A key strength is the comprehensive evaluation of leading MLLMs, revealing significant performance gaps between models and human capability, particularly in temporal understanding. The work provides important insights into Chinese video understanding and offers a solid foundation for future research.

### Weaknesses
1. The dataset scale (1,625 videos, 6,507 QA pairs) is modest compared to major English benchmarks, and task distribution is uneven - some categories like "World Knowledge" contain under 100 questions, potentially affecting evaluation reliability despite claims of balance.

2. The benchmark's scope is constrained by design: audio tracks are removed and only multiple-choice format is supported. While this simplifies evaluation, it limits real-world applicability and excludes generative QA formats increasingly important for modern LLMs.

3. Crucial details about dataset release and licensing are absent, raising concerns about reproducibility. The evaluation methodology also lacks discussion of potential prompt bias, particularly for non-Chinese optimized models, and reports only accuracy without statistical significance measures.

These limitations in dataset scale, task balance, modality coverage, and evaluation design represent opportunities for future improvements to enhance the benchmark's utility.

### Questions
1. Will the ChineseVideoBench dataset and evaluation code be released?
Public availability is a key aspect of benchmark papers. The paper does not mention whether the dataset or evaluation scripts will be released, under what license, or when. This significantly affects reproducibility and community adoption.

2. How were the multiple-choice distractors constructed and balanced?
While the annotation process is described, it is unclear whether distractors were manually crafted to be semantically close to the correct answer or randomly chosen. Could the authors provide more detail or examples on distractor design to assess question difficulty?

3. Was any inter-annotator agreement measured?
Since the dataset is entirely human-annotated, some measure of consistency—such as agreement scores in the validation phase—would be helpful to quantify annotation quality.

4. Can the authors comment on how prompt design may bias model performance?
A single fixed prompt was used for all models, including those with different training languages or instruction formats. Might this disadvantage certain open-source models not optimized for Chinese inputs? Have alternative prompt formats (e.g., simplified/standardized instructions) been tested?

5. Why are some task categories underrepresented in the dataset?
For example, World Knowledge only has 73 QA pairs, compared to over 1,000 for other categories. Is this due to difficulty in annotation or inherent scarcity in source videos? How does this affect per-task benchmarking reliability?

6. Are the sampled frames sufficient for evaluating fine-grained temporal tasks?
The paper mentions fixed frame counts (e.g., 32 or fewer) for evaluation. Could this sampling resolution cause key temporal cues to be missed in tasks like action or posture recognition? Would adaptive or dense sampling improve performance in these areas?

7. Have the authors considered incorporating audio or multimodal input in the future?
Since Chinese video content often contains spoken or musical cues, excluding audio limits real-world applicability. Do the authors have plans to extend the benchmark to audio-visual QA?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper ChineseVideoBench: Benchmarking Multimodal Large Models for Chinese Video Question Answering presents a new benchmark to evaluate multimodal large language models (MLLMs) on Chinese video understanding tasks. It addresses the lack of culturally and linguistically relevant datasets for non-English scenarios. The dataset contains 1,625 CC0-licensed Chinese videos and 6,507 human-annotated multiple-choice QA pairs, covering 8 main tasks and 12 sub-tasks such as world knowledge, scene understanding, temporal localization, and logical reasoning.

### Strengths
The paper introduces the first large-scale benchmark for Chinese VideoQA, covering 1,625 CC0-licensed videos and 6,507 manually annotated QA pairs
The benchmark exposes systematic failure patterns in temporal localization and fine-grained spatiotemporal grounding.

### Weaknesses
The paper does not report inter-annotator consistency, distractor calibration, or difficulty-level validation, leaving the annotation quality claims insufficiently quantified.

The paper emphasizes long-video evaluation but does not provide convincing evidence that its videos require long-term temporal reasoning; average durations appear modest, weakening this claim

### Questions
Why don't the authors test the videoQA without removing the audio, as an ablation?

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
The paper introduces first large-scale fully human-annotated benchmark for Chinese video question answering. The benchmark contains 6507 MCQs from 11 domains organized into eight tasks. The dataset is annotated in three stages by using three groups of human annotators. First group generated questions and answers from collected videos, the second group independently reviewed and corrected the generated questions, and the third group answered the questions after watching videos serving as human performance check. Several multimodal LLMs including open and closed source models are evaluated against the developed benchmark.

### Strengths
- It is a useful resource for a non-English language to evaluate multimodal LLMs.
- In the annotation process, human annotators have been used to annotate, verify and validate the whole dataset that ensures an unbiased and potentially correct benchmark. 
- The developed benchmark covers sufficient number of tasks and underlying sub-categories showing diversity of topics. 
- The paper presents an evaluation of several MLLMs including open and closed source models in comparison with human performance.

### Weaknesses
- Although, the annotation process has been done by using human annotators, but the paper does not show understanding and knowledge of annotators by using any quantitative measures, e.g., inter-annotator agreement (IAA).
- Error analysis is not presented for LLM evaluation. For example, which type of tasks or questions are difficult for the models to answer.
- The videos were collected from CC0-licensed platform, so it is very much possible that the LLMs evaluated in the paper have already seen the videos in their training sets resulting in higher (biased) accuracy.

### Questions
- How annotation guidelines were prepared and how were the annotators trained?
- Is the developed benchmark unseen to the LLMs which have been evaluated?

### Soundness
2

### Presentation
2

### Contribution
2
