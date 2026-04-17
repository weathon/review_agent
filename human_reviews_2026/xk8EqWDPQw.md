# MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
The sequential structure of videos poses a challenge to the ability of multimodal large language models (MLLMs) to locate multi-frame evidence and conduct multimodal reasoning. However, existing video benchmarks mainly focus on understanding tasks, which only require models to match frames mentioned in the question (hereafter referred to as "question frame'') and perceive a few adjacent frames. To address this gap, we propose **MMR-V: A Benchmark for Multimodal Deep Reasoning in Videos**. The benchmark is characterized by the following features. **(1) Long-range, multi-frame reasoning**: Models are required to infer and analyze evidence frames that may be far from the question frame. **(2) Beyond perception**: Questions cannot be answered through direct perception alone but require reasoning over hidden information. **(3) Reliability**: All tasks are manually annotated, referencing extensive real-world user understanding to align with common perceptions. **(4) Confusability**: Carefully designed distractor annotation strategies to reduce model shortcuts. MMR-V consists of 317 videos and 1,257 tasks. Our experiments reveal that current models still struggle with multi-modal reasoning; even the best-performing model, Gemini-2.5-pro, achieves only 64.3% accuracy. Additionally, current reasoning enhancement strategies (Chain-of-Thought and scaling test-time compute) bring limited gains. Error analysis indicates that the CoT demanded for multi-modal reasoning differs from it in textual reasoning, which partly explains the limited performance gains. We hope that MMR-V can inspire further research into enhancing multi-modal reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MMR-V, a new benchmark designed to evaluate the deep multimodal reasoning capabilities of large language models in the context of videos. The authors argue that existing video benchmarks primarily test perceptual understanding over short temporal windows, failing to assess more complex reasoning. MMR-V addresses this gap by featuring tasks that require long-range, multi-frame reasoning to infer information that is not directly visible. A key contribution is the categorization of tasks into "implicit reasoning" (understanding subtext, metaphors, themes) and "explicit reasoning" (logical deduction from perceivable evidence). The benchmark consists of 317 videos and 1,257 manually annotated multiple-choice questions with carefully designed distractors to challenge current models. The experimental results, covering a wide range of state-of-the-art models, demonstrate that even top-performing models like Gemini-2.5-pro struggle significantly, revealing a substantial gap between model and human performance and highlighting the limitations of current reasoning-enhancement techniques like CoT.

### Strengths
1、The paper introduces a benchmark that addresses a critical gap in video understanding evaluation. It introduces a novel task structure focused on deep, multi-frame reasoning. This requires models to synthesize information across long temporal spans and infer hidden context, a much-needed step up in complexity that better reflects real-world reasoning challenges.
2、The quality and rigor of the benchmark construction are a major strength. The reliance on manual annotation, guided by real-world user consensus, ensures high-quality, nuanced data that is difficult to create automatically. Furthermore, the innovative strategy of using model-generated errors to create plausible and challenging distractors is a key feature that effectively tests the robustness of models and mitigates the risk of them relying on superficial cues.
3、The paper provides a clear, comprehensive, and valuable evaluation of a wide range of state-of-the-art models. This extensive testing not only establishes a strong and reliable baseline for future research but also offers insightful analysis into the current limitations of MLLMs in video reasoning. The results clearly highlight the significant performance gap between models and humans, underscoring the benchmark's importance and difficulty.

### Weaknesses
(1) The benchmark’s 317 videos primarily cover six categories but underrepresent real-world genres like mystery, puzzle-solving, or gaming (acknowledged in Section B as a limitation). Additionally, most videos are in English, with only a small proportion in other languages, restricting MMR-V’s applicability to multilingual video reasoning tasks.
(2) While the paper notes that models perform better on implicit tasks (+7.9% average gain) due to dispersed visual cues, it does not explore why explicit tasks are harder. For example, whether it is due to fewer evidence frames or stricter logical requirements. This limits insights into how to improve model performance on specific reasoning types.
(3) While the paper tests frame count impact (Figure 4) for Gemini-2.0-Flash, it does not report efficiency metrics (e.g., inference time, memory usage) for models processing long videos. This is relevant for real-world deployment, as long-video reasoning often requires balancing accuracy and efficiency.

### Questions
1、The paper acknowledges that video categories like mystery and gaming are underrepresented. Could you elaborate on why these categories were excluded (e.g., difficulty in annotating reasoning tasks, lack of popular videos) and whether there are plans to expand MMR-V to include them in future updates?
2、For implicit reasoning tasks, the paper attributes models’ better performance to "abundant visual cues dispersed throughout the video." Have you explored whether models’ pre-trained world knowledge (e.g., understanding that "room number 7 symbolizes good luck") also contributes to this performance gap?
3、The human experiment uses 200 tasks (100 correct/100 incorrect for GPT-4o). Were these tasks balanced across implicit/explicit reasoning and video categories? If not, could this imbalance affect the measured human-model gap?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces MMR-V, a new benchmark designed to evaluate the deep multimodal reasoning capabilities of large language models in videos. MMR-V requires models to locate and analyze evidence across multiple, long-range frames to answer questions that involve hidden information. Through extensive testing on 21 different models, the study reveals that even the best-performing model, Gemini-2.5-pro, only achieves 64.3% accuracy, indicating a significant gap between current AI capabilities and complex video reasoning.

### Strengths
1.	It is an interesting and novel step to divide the reasoning tasks to be implicit and explicit. Evaluating the model’s capability to combine the precepted information with previously learned world knowledge to understand the metaphors is important. 
2.	The entire benchmark is human-labeled, and several approaches are conducted from the video collection to data annotation to guarantee the overall quality to make the benchmark more reliable.
3.	Experiments are comprehensive, the authors evaluate enough frontier open-source and proprietary models, and compare with human performance. The authors also conducted a series of ablation studies regarding the input frames and modalities.

### Weaknesses
1.	The questions designed for reasoning are not as deep as the authors claim. First, all question-answer pairs are single-step and do not require reasoning chains. This makes the proposed benchmark less challenging compared with some widely adopted text or image reasoning benchmarks, which include some mathematical or scientific problems that require multi-step reasoning. Besides, for many reasoning question types, such as personal reflection, video naming, and meta-emotion, it is more likely to be visual perception tasks. What reasoning capabilities are needed for these tasks? Readers may think the authors overclaim some perception problems as “deep reasoning” tasks.
2.	The authors have detailed the generation process of the wrong answers, yet the description of the human annotation process is vague and unclear. Are the annotators specifically trained for this labeling task? Is there any human review process for the human annotations? How to guarantee the overall quality of the human labeling process? More clarifications are needed.
3.	The literature review only includes some video perception benchmarks, but more recent video reasoning benchmarks are not involved and compared, such as VideoEspresso, VRBench, VideoReasonBench, CG-Bench, and MMVU. I believe some discussions are needed to highlight the contribution.

### Questions
1.	Why do the authors only select the multiple-choice question as the evaluation protocol? Are open-ended questions essential for the reasoning tasks?
2.	For the annotation process, are there any time interval limitations for the questions? The checklist requires long distance, how long is it, and how to guarantee it? 
3.	For the implicit reasoning question type, how can to ensure that all annotators can obtain all world knowledge to understand the metaphors in the videos? 
4.	Is there the possibility that the discrepancy in the annotators’ cultural background leads to different reasoning results?

### Soundness
2

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
In this work, the authors propose MMR-V, a benchmark for multimodal reasoning in videos. MMR-V consists of 317 videos and 1,257 tasks, for  long-range, multi-frame reasoning in the real world. The authors also investigate the current MLLMs models on this benchmark.

### Strengths
*Clarity

The paper is well-written with good structure. Hence, the clarity is basically good.

*Significance

This paper focuses on evaluating video reasoning capacity of MLLMs, which is an important and practical problem for video understanding. Hence, the significance is basically OK for video research community.

### Weaknesses
* Reference

1) The recent work [VRBench: A Benchmark for Multi-Step Reasoning in Long Narrative Videos, ICCV 2025] proposes the similar topic for video understanding. Please clarify the key difference.
2) Small suggestion: It would be clearer to include a table to show key statistics difference between this bench and the existing ones such as Video-MME,  LongVideoBench, LVBench, Video-MMMU, MMVU, etc.

* Method Insight

1) It woule be more interesting to investigate or indicate how to design MLLMs to tackle the tasks in this benchmark.
2) Actually, there exists some agentic mechanism for long video reasoning such as VideoAgent, VideoTree, etc. These methods are suitable for reasoning via tool usage with multi-round clue discovery. Hence, it would be great to include these works for evaluating the benchmark. 

* Small Size

The authors collected only 317 original videos. The small number of videos would restrict the generalization of this benchmark.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel video reasoning benchmark, MMR-V, to evaluate multimodal deep reasoning in videos. Unlike existing video understanding benchmarks, MMR-V emphasizes multi-frame reasoning, implicit and explicit reasoning. Extensive experiments show the limitation of current MLLMs in handling complex reasoning tasks.

### Strengths
1. The benchmark is carefully curated with human annotation, with evidenced distractors generated by GPT.

2. The authors evaluated a wide range of models (including GPT-5, Gemini, Claude, and open-source alternatives) and provide detailed error analysis, scaling trends, and modality impact (e.g. audio).

### Weaknesses
1. The distinction between “implicit” and “explicit” is not always clear-cut in practice. For example, in detective films, it might require both implicit and explicit clues to determine the criminals. 

2. Comparsion with existing video reasoning benchmarks (e.g. VRBench) could be further discussed.

[1] VRBench: A Benchmark for Multi-Step Reasoning in Long Narrative Videos. ICCV 25.

3. To further understand MLLMs reasoning ability, it is encouraged to include the comparison based on either different video lengths (e.g. <60s, 60s-300s, etc), or different clue lengths (e.g. 1-frame, 5-10 frames).

### Questions
1. In L452, it is not very clear what GPT-4.1 is labeling, can you explain in detail?

### Soundness
4

### Presentation
4

### Contribution
3
