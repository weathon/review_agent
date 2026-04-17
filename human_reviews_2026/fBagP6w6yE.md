# From Natural Alignment to Conditional Controllability in Multimodal Dialogue

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
The recent advancement of Artificial Intelligence Generated Content (AIGC) has led to significant strides in modeling human interaction, particularly in the context of multimodal dialogue. 
While current methods impressively generate realistic dialogue in isolated modalities like speech or vision, challenges remain in controllable Multimodal Dialogue Generation (MDG). 
This paper focuses on the natural alignment between speech, vision, and text in human interaction, aiming for expressive dialogue generation through multimodal conditional control. 
To address the insufficient richness and diversity of dialogue expressiveness in existing datasets, we introduce a novel multimodal dialogue annotation pipeline to curate dialogues from movies and TV series with fine-grained annotations in interactional characteristics.
The resulting MM-Dia dataset (360+ hours, 54,700 dialogues) facilitates explicitly controlled MDG, specifically through style-controllable dialogue speech synthesis. 
In parallel, MM-Dia-Bench (309 highly expressive dialogues with visible single-/dual-speaker scenes) serves as a rigorous testbed for implicit cross-modal MDG control, evaluating audio-visual style consistency across modalities. 
Extensive experiments demonstrate that training on MM-Dia significantly enhances fine-grained controllability, while benchmarks on MM-Dia-Bench reveal limitations in current frameworks to replicate the nuanced expressiveness of human interaction. 
These findings provides new insights and challenges for multimodal conditional dialogue generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses critical limitations in multimodal dialogue generation—specifically the overemphasis on content transmission over style controllability, scarcity of high-quality datasets, and lack of benchmarks for cross-modal consistency. It focuses on achieving expressive, controllable multimodal dialogue through natural alignment of speech, vision, and text, while constructing a large-scale dataset and systematic benchmarks to advance the field.

### Strengths
The paper introduces MM-DIA, a dataset curated from 700+ hours of movies/TV series (200+ films, 9 shows) with 360.26 hours of dialogue, 54,700 clips, and 449,138 turns. It features fine-grained annotations across modalities. The paper’s flagship contribution—the MM-DIA dataset—is the first to center on "multimodal dialogue expressiveness". To evaluate implicit cross-modal style consistency (a long-overlooked gap), the paper builds MM-DIA-BENCH—a balanced benchmark of 309 dual-speaker dialogues (1.69 hours, 1,851 turns) with guaranteed speaker visibility. Experiments show MM-DIA significantly enhances style controllability.

### Weaknesses
The paper claims MM-DIA and its findings support "a wide range of applications in human–computer interaction, social computing, and film-making" but exclusively uses cinematic data (movies/TV series) for dataset construction and experiments. This creates a critical gap: it is unclear if the proposed framework (annotations, tasks, model insights) generalizes to non-scripted, real-world multimodal dialogue—arguably the most impactful use case for HCI and social computing.

### Questions
You claim MM-DIA supports "broad applications in HCI and social computing" but exclusively use cinematic data (movies/TV series) for training and testing. Given that movie dialogue is scripted and emotionally exaggerated (e.g., MM-DIA’s average emotion intensity score of 6.76/10 via Gemini; )—a stark contrast to casual real-world interactions—have you tested if models fine-tuned on MM-DIA (e.g., Higgs-Audio-V2-SFT) retain style controllability on real-world multimodal datasets?

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
The paper introduces MM-DIA, a large-scale, richly annotated multimodal dialogue dataset from movies and TV series, and MM-DIA-BENCH, a benchmark for evaluating cross-modal conditional generation. Experiments show MM-DIA improves style-controllable dialogue generation

### Strengths
1. This is the first dataset to focus on dialogue expressiveness across multiple modalities. The benchmark (MM-DIA-BENCH) fills a gap for evaluating cross-modal style consistency, which is underexplored in prior work.
2. The paper provides a unified framework for MDG, with well-defined tasks and evaluation metrics. Experiments are thorough, with both objective and subjective metrics.

### Weaknesses
1. The paper’s main contribution is dataset and benchmark creation; the modeling advances are limited to fine-tuning existing architectures and adapter modules for controllability. No novel end-to-end model for multimodal dialogue generation is proposed or evaluated.
2. The paper is too dense and at times it is difficult to follow, especially in the technical details of the pipeline and annotation process.
3. The dataset is sourced primarily from movies and TV series, which may limit the diversity and generalizability to real-world, spontaneous dialogues

### Questions
1. How do you envision MM-DIA supporting research on more spontaneous, real-world dialogues? 

2. How does your approach handle long-range dependencies, such as multi-turn conversations or scenes with complex speaker dynamics?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the limitations in scale, expressiveness, and benchmarking of existing datasets for multimodal dialogue generation by proposing a novel data curation and annotation pipeline, resulting in the large-scale and expressive multimodal dialogue dataset MM-DIA. The authors further introduce a unified framework for Multimodal Dialogue Generation (MDG) and define three representative downstream tasks, including style-controllable speech synthesis, vision-conditioned speech synthesis, and speech-driven video generation. Through systematic benchmarks and experiments, the paper demonstrates the effectiveness of the new dataset in enhancing dialogue style controllability and cross-modal consistency, while revealing the shortcomings of current methods in expressiveness and multimodal alignment. This work provides valuable resources and new challenges for future research in conditional multimodal dialogue generation.

### Strengths
1. The unified framework for multimodal dialogue generation proposed in this paper is highly practical and extensible, providing systematic task definitions and benchmarking foundations that will facilitate further advances in the field.
2. The experimental section is comprehensive, covering various downstream tasks and systematically benchmarking the dataset and methods for style controllability and cross-modal consistency, with convincing results.
3. The paper is well-structured and clearly articulated, progressing logically from problem motivation, dataset construction, method design, to experimental evaluation, making it easy for readers to follow and understand the research.

### Weaknesses
1. Although the dataset is large and expressive, it is mainly sourced from movies and TV series, which may differ from real-life conversations and affect the generalizability of models to real-world scenarios.
2. The evaluation methodology in the paper is insufficient for assessing the generalization ability of the trained models. In Section 5.1, the authors mention an out-of-domain dataset, but do not clearly specify whether it comes from different data sources or demonstrate its differences. The model’s performance in real-world scenarios, as well as the potential degradation of its original capabilities after SFT (fine-tuning), require further consideration and analysis.
3. There are some typographical errors in the paper, such as the table on page 782 not being cited.

### Questions
1. In the evaluation, the authors use Gemini as a judge. Has the accuracy of using large models as judges been tested, and has the model’s performance been compared with human evaluation?
2. In this paper, the authors achieve "From Natural Alignment to Conditional Controllability" from a data perspective. From a methodological standpoint, do you think further improvements at the model level could help achieve this goal?

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
This paper introduces MM-DIA, the first large-scale and highly expressive multimodal dialogue dataset designed for Multimodal Dialogue Generation (MDG). In addition, the authors present MM-DIA-BENCH, a dual-speaker benchmark specifically developed for evaluating cross-modal conditional generation.
Experiments show that training on MM-DIA significantly enhances controllable dialogue generation, while evaluations on MM-DIA-BENCH reveal notable limitations of current models in achieving consistent multimodal style alignment.

### Strengths
* This paper is well-motivated, and the proposed dataset paves the way for future research on style controllability of multi-modal dialogue generation.

* Strong experimental validation with ablations on both controllability and user satisfaction metrics.

* This paper is clearly structured and easy to follow.

* Although the dataset creation heavily relies on models, the authors try to demonstrate that the proposed pipeline achieves human-level quality in annotation consistency and reliability

### Weaknesses
* The data creation and evaluation partly rely on GPT-based scoring, which could cause an upper bound of future research.

### Questions
Pls refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
