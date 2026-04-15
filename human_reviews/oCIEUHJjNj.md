# OMCAT: Omni Context Aware Transformer

- Decision: Reject
- Scores: 5, 6, 5, 8

## Abstract
Large Language Models (LLMs) have made significant strides in text generation and comprehension, with recent advancements extending into multimodal LLMs that integrate visual and audio inputs. However, these models continue to struggle with fine-grained, cross-modal temporal understanding, particularly when correlating events across audio and video streams. We address these challenges with two key contributions: a new dataset and model, called OCTAV and OMCAT respectively. OCTAV (Omni Context and Temporal Audio Video) is a novel dataset designed to capture event transitions across audio and video. Second, OMCAT (Omni Context Aware Transformer) is a powerful model that leverages RoTE (Rotary Time Embeddings), an innovative extension of RoPE, to enhance temporal grounding and computational efficiency in time-anchored tasks. Through a robust three-stage training pipeline—feature alignment, instruction tuning, and OCTAV-specific training—OMCAT excels in cross-modal temporal understanding. Our model demonstrates state-of-the-art performance on Audio-Visual Question Answering (AVQA) tasks and the OCTAV benchmark, showcasing significant gains in temporal reasoning and cross-modal alignment, as validated through comprehensive experiments and ablation studies. Our dataset and code will be made publicly available. The link to our demo page is https://om-cat.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces OMCAT, a model aimed at advancing cross-modal temporal understanding by leveraging both audio and visual inputs. To support this objective, the authors present OCTAV, a dataset specifically designed to enable detailed event correlation between audio and video streams. The model employs a three-stage training approach and incorporates Rotary Time Embeddings (RoTE) to enhance temporal grounding. Experimental results showcase the model's effectiveness on Audio-Visual Question Answering (AVQA) tasks, as well as its adequate performance on the newly introduced OCTAV benchmark.

### Strengths
The authors address a well-defined gap in existing multimodal models—fine-grained temporal reasoning between audio and video inputs. They contribute both a new dataset (OCTAV) and a novel model (OMCAT), advancing the field significantly.

OCTAV fills the need for datasets that handle fine-grained temporal and cross-modal alignment. The synthetic generation strategy for question-answer pairs is innovative and provides a valuable resource for future research.

The experimental evaluation is thorough, covering multiple benchmarks and including both quantitative and qualitative analyses. The performance gains on AVQA and temporal grounding tasks are compelling evidence of the model's efficacy.

### Weaknesses
The use of GPT-4 for evaluation introduces an element of subjectivity, especially in grading responses on a 0-5 scale. Although this approach is common, the absence of standardized quantitative metrics could complicate comparisons with future studies. Additionally, human evaluation is recommended to achieve a more reliable and robust assessment.

Given that the dataset is primarily constructed using GPT, I am concerned about its quality control. It would be beneficial to incorporate human annotation or validation to ensure greater accuracy and reliability.

While the paper compares OMCAT to several state-of-the-art models, it would be beneficial to include more baselines from diverse modalities to fully illustrate its strengths and weaknesses across different scenarios.

### Questions
The paper relies heavily on GPT-4 for scoring responses with a subjective grading scale. How consistent and reliable is this approach across different response types? Did the authors test the robustness of the GPT-4 evaluation process, such as through inter-rater reliability tests or additional scoring methods?

Can the authors elaborate on the quality control measures applied to ensure the reliability and accuracy of the synthetically generated OCTAV dataset? Given that synthetic data may sometimes lack the subtle variations of real-world data, how did the authors verify that OCTAV’s question-answer pairs and temporal annotations closely reflect realistic audio-visual interactions? Additionally, are there any known limitations in OCTAV’s data quality that might impact OMCAT’s generalizability?

### Soundness
2

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
4

### Summary
This article presents a new dataset and model aimed at addressing the issues of fine-grained audiovisual alignment and cross-modal temporal alignment. The dataset, OCTAV, is capable of capturing event transitions between audio and video. The model, OMCAT, enhances temporal grounding and computational efficiency through RoTE and RoPE, allowing OMCAT to perform well in cross-modal temporal understanding.

### Strengths
1. Clear writing with a logical structure.
2. Innovative ideas that incorporate the latest research findings or technologies, presenting a unique perspective.
3. Reproducible results by providing detailed experimental methods or steps, allowing other researchers to replicate the experiments.

### Weaknesses
1. The experimental results are insufficient, and there is no comparison regarding the understanding ability of audio.
2. How can the quality of the generated dataset be determined?
3. In the appendix, the video understanding experiments, such as MSRVTT-QA, MSVD-QA, and ActivityNet-QA, show average performance.
4. The comparative experiments in Table 8 are not entirely fair. The data used for OMCAT with only LP, WC, and V is 430.7k less than the total dataset, so the performance of the first row being worse than the second row does not necessarily prove that the improvement is due to the characteristics of the 430.7k data rather than the difference in data volume.
5. The dataset consists of artificially added audio clips that are not aligned with the visuals and are unrelated to the overall video. However, there are no annotations regarding audiovisual misalignment in the dataset. Wouldn't this affect the model's understanding ability?
6. There are no ablation studies conducted to evaluate the effects of the proposed dataset.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a new method for audio-visual question-answering. Concretely, it proposes a new method for generating a synthetic audio-vision instruction-tuning dataset, OCTAV. The method, OMCAT, is a multi-modal large language model employing staged training. The innovation is a new way of interleaving tokens and time embedding. The experiments compare the proposed method with a few recent MLLMs.

### Strengths
- The authors' construction of OCTAV to capture the transition between audio and video is interesting and seems effective. A lot of methods in literature create videos that have time-aligned audio and video data that correspond to the same event. Here, the authors insert arbitrary audio into the video and pose new problems for training.

### Weaknesses
- The method innovation is somehow mediocre. The RoTE over RoPE seems to be a small but effective improvement. However, given the recent MLLM progress, the staged training approach seems standard. Interleaving tokens is effective, but others are also employing this technique. 

- The evaluation doesn’t fully support the claims that OMCAT has achieved significant gains in temporal reasoning and cross-modal alignment. First, a lot of entries in Table 5 are missing. This is quite unfortunate to draw a well-rounded comparison. Second, among them, AVSD and zero-shot Music-AVQA have relatively more methods benchmarked. On AVSD, a few methods outperform OMCAT by a 3-4% margin. On zero-shot Music-AVQA, OMCAT outperforms the second-best by a 1.6% margin. 

- Some claims were exaggerated. For example, “OCTAV dataset addresses all the above mentioned limitations and provides a comprehensive benchmark for interwoven and fine-grained audio-visual understanding.” It is not a very convincing comparison to other datasets since OCTAV doesn’t collect new videos or new annotations. Later, the method training still needs to be performed on a range of audio-video datasets besides OCTAV. The combined training also shows that L84-85 in the Introduction is exaggerated, saying, "Despite this artificial setup, our experiments show that a model trained with this data performs well in naturally occurring video and audio pairs."

### Questions
Please see comments in the Weaknesses section. Also: 
- L155-156, there are typos and much confusion. What is the unit for m? What is the rationale behind "ensuring that the videos are not too far apart and their length is not too long?"

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper targets solving fine-grained, cross-modal temporal understanding, particularly when correlating events across audio and video streams. To better tackle this task, they introduce the Omni Context Aware Transformer(OMCAT), together with a dataset Omni Context and Temporal Audio Video(OCTAV). OMCAT integrates time embedding and is proposed to improve temporal localization and computational efficiency through RoTE (Rotary Time Embeddings).OCTAV is designed to capture the transformation of audio and video events, helping the model to better understand the temporal order of events and cross-modal associations. Comprehensive experiments and ablation studies are provided to demonstrate that OMCAT significantly improves temporal reasoning and cross-modal alignment.

### Strengths
The paper addresses a clear gap in cross-modal temporal understanding, providing new opportunities for research in integrating audio and video modalities and giving related solutions and datasets.

The approach is soundness:
1. The use of RoTE (Rotary Time Embeddings) as an extension of RoPE to enhance temporal grounding is a creative advancement that could inspire further research in temporal embeddings.
2. The multi-training-stage ensures thorough model development and evaluation

The experiment comparison is comprehensive. experiments and ablation studies provide strong evidence for the model’s state-of-the-art performance on Audio-Visual Question Answering (AVQA) tasks.

### Weaknesses
1. The synthetic nature of the OCTAV dataset may limit the performance of the model in real-world applications.

2. The technical details of methods such as RoTE are not explained in detail, which may affect understanding and reproducibility.

3. RoTE is claimed to have a better ability to capture actual elapsed between frames while lacking more comparison of how fine-grand it can do.

4. The dataset is comprehensive but including also the evaluation dataset. This raises the concern of proposed methods.

### Questions
You mentioned the ability to process music. Could you show some result of out-of-distribution results in the event-intensive data like movie trailers? 

Audio tracks often overlap. Have you considered the relevant design?

Video and music or video often have strong coupling or delay. Have you considered how to find the time node of multi-modal events? For example, due to broadcasting issues, the time node of the ball landing and the time node of the sound in a tennis video may not be synchronized. So I ask which mode should be prioritized when the tennis ball lands?

### Soundness
3

### Presentation
4

### Contribution
3
