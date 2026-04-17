# VidText: Towards Comprehensive Evaluation for Video Text Understanding

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Visual texts embedded in videos carry rich semantic information, which is crucial for both holistic video understanding and fine-grained reasoning about local human actions. However, existing video understanding benchmarks largely overlook textual information, while OCR-specific benchmarks are constrained to static images, limiting their ability to capture the interaction between text and dynamic visual contexts. To address this gap, we propose VidText, a new benchmark designed for comprehensive and in-depth evaluation of video text understanding. VidText offers the following key features: 1) It covers a wide range of real-world scenarios and supports multilingual content, encompassing diverse settings where video text naturally appears. 2) It introduces a hierarchical evaluation framework with video-level, clip-level, and instance-level tasks, enabling assessment of both global summarization and local retrieval capabilities. 3) The benchmark also introduces a set of paired tasks for  perception and reasoning, ranging from visual text perception to cross-modal reasoning between textual and visual information. Extensive experiments on 18 state-of-the-art  Large Multimodal Models (LMMs) reveal that current models struggle across most tasks, with significant room for improvement. Further analysis validates the effectiveness of VidText—spanning joint video-text and multimodal reasoning, multi-granularity task structure, and temporal modeling. It also reveals substantial effects from model-intrinsic factors (input resolution, OCR capability) and external factors (auxiliary context and video-text-centric Chain-of-Thought strategies). We hope VidText will fill the current gap in video understanding benchmarks and serve as a foundation for future research on multimodal reasoning with video text in dynamic environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces VidText, a new benchmark designed to comprehensively evaluate video text understanding in Large Multimodal Models (LMMs). The authors identify a significant gap in existing video understanding benchmarks. The paper evaluates 18 LMMs on VidText. The study also validates the benchmark's design through ablations, showing that performance drops when key components are masked.

### Strengths
*   **Comprehensive Benchmark Design:** The hierarchical (video/clip/instance) and paired (perception/reasoning) task structure is a major strength, enabling a much more nuanced evaluation than previous benchmarks.
*   **Extensive Empirical Analysis:** The evaluation of 18 models provides a valuable snapshot of the field's capabilities and limitations. The ablation studies effectively validate the benchmark's design choices.

### Weaknesses
*   **Outdated Model Comparison:** A significant weakness is the omission of the very latest flagship models (e.g., Gemini 2.5 Pro, GLM-4V). This quickly diminishes the paper's relevance and the persuasiveness of its conclusions about the current state-of-the-art.
*   **Limited Discussion on Data Biases:** While the dataset is diverse, there is no discussion of potential biases in the video sources (e.g., geographic or cultural biases from YouTube) or the annotation process that might affect model generalization.

### Questions
*   **Question on Human Evaluation:** Please provide a detailed description of the human evaluation protocol for the 89.5% baseline. What were their backgrounds, and what was the inter-annotator agreement? **Suggestion:** Strengthen this section with rigorous methodology to make the human benchmark a reliable point of comparison.
*   **Suggestion on Generalization:** Perform an analysis or discuss potential biases within the VidText dataset itself. Could the performance gaps be partly due to specific types of videos or text that models struggle with? This would add depth to the analysis of model shortcomings.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes VidText, a video-text understanding benchmark to evaluate MLLM's capabilities in tackling multimodal tasks. The proposed benchmark divides the tasks into video-level, clip-level, and instance-level and includes both perception and reasoning tasks. The authors evaluate 18 frontier MLLMs on the established benchmark.

### Strengths
1. The proposed benchmark is human-annotated with a double human check. I believe the manually labeled benchmark benefits the community and could positively guide the development of MLLMs.

2. The OCR-related tasks are interesting.

3. The authors evaluate several frontier methods to show their capabilities in tackling video understanding tasks.

### Weaknesses
1. I do not think the proposed benchmark evaluates any new aspect in comparison to existing video understanding benchmarks. The proposed benchmark includes 8 question types, all of them has been involved in existing video understanding benchmarks such as Video-MME and MVBench. The average video length is 108.2 seconds, so it is not a long video understanding benchmark. Though the authors claim that the proposed benchmark supports open-ended evaluation, the so-called open-ended protocol can only be applied to a small portion of the taxonomy (OCR and grounding), and the evaluation method is also rule-based and closed-ended. which is quite similar to the MCQ. Demos for the reasoning tasks also look similar to the perception tasks, and I cannot find any crucial reasoning factors to get the final answers. Overall, the proposed benchmark is not highlighted, and I suggest the authors provide additional analysis to clarify what new aspect it can evaluate and what it actually brings to the community.

2. The evaluation is not comprehensive. The authors evaluate Gemini 1.5 and GPT-4o, yet we can now access Gemini 2.5 and GPT-5. The total number of evaluated models is also relatively small, and many frontier MLLMs are not involved. The experimental analysis is not exhaustive. I cannot find too much valuable insight after evaluating these models on the proposed benchmark. The authors only tell the readers some obvious conclusions, such as adding input resolution or model size could enhance the model performance (Table 3), and adding CoT or audio modality helps the video understanding (Table 4). What are the key factors for holistic and local perception/reasoning? What is the main barrier for current frontier MLLMs to catch up to human performance (more than 40% on the proposed benchmark)? I believe these analyses are more important for readers.

### Questions
1. How many annotators are used for the annotation? Is there any training for these annotators? 

2. How to control the quality of the CoT? 

3. Appendix F claims the average length is 108.2 seconds, yet line 193 claims the minimum threshold duration is 3 minutes, which seems to be in conflict.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces VidText, a new benchmark designed for the comprehensive evaluation of video text understanding in LLM. The authors argue that existing video benchmarks largely ignore textual information, while static image OCR benchmarks cannot capture the dynamic interaction between text and video context. VidText addresses this gap by offering several key features with a set of eight paired perception and reasoning tasks, such as Holistic OCR and Holistic Reasoning, or Text Localization and Temporal Causal Reasoning. The authors conduct an extensive evaluation of 18 state-of-the-art proprietary and open-source LMMs, revealing that even the best models, like Gemini 1.5 Pro, achieve an average score of only 46.8%, far below the human baseline of 89.5%.

### Strengths
- Fills a Critical Research Gap: The paper convincingly argues for and fills an important, underexplored niche in multimodal evaluation. Understanding text embedded in dynamic scenes is crucial for holistic video comprehension, and VidText is the first benchmark to address this systematically.
- Comprehensive and Well-Designed Benchmark: The benchmark's design is a major strength. The multi-granularity structure (instance, clip, video) tests a wide range of capabilities, and the paired perception-reasoning tasks provide a deep, analytical framework for understanding model failures.

### Weaknesses
- Reliance on Multiple-Choice for Reasoning: For standardization, most reasoning tasks are formulated as multiple-choice questions. This is a practical choice but may not fully capture the nuanced reasoning failures or generative capabilities of models. An analysis of open-ended responses, even on a small subset, could provide complementary insights.
- Limited Evaluation of Recent SOTA Models: The benchmark omits newer iterations like Gemini 2.5 Pro and GPT-5. Since these latest models may have advanced video-text understanding capabilities, their absence weakens the benchmark’s reflection of the current state of model performance.

### Questions
Regarding the evaluation of reasoning tasks, have you explored methods to automatically score open-ended, generative answers? Given the rapid progress in LLM-based evaluators, this seems like a promising direction to move beyond the constraints of multiple-choice formats. What challenges do you foresee in applying such methods to your tasks?

### Soundness
3

### Presentation
3

### Contribution
3
