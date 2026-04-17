# Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically ref-
erencing visual regions, just like human “thinking with images”. However, no
benchmark exists to evaluate these capabilities holistically. To bridge this gap, we
propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic
benchmark built on three principles: (1) focused visual perception of subtle targets
in complex scenes, (2) traceable evidence via bounding box evaluation, and (3)
second-order reasoning to test object interactions and spatial hierarchies beyond
simple object localization. Prioritizing images with dense objects, we initially
sample 1K high-quality images from SA-1B, and incorporate eight LMM experts
to manually annotate questions, candidate options, and answers for each image.
After three stages of quality control, TreeBench consists of 405 challenging vi-
sual question-answering pairs, even the most advanced models struggle with this
benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only
54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual
Grounded Reasoning), a training paradigm to supervise localization and reasoning
jointly with reinforcement learning, enabling accurate localizations and explainable
reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench
(+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is
key to advancing vision-grounded reasoning. The code and data will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to assess and improve the grounding reasoning of Vision-Language Models. To this end, it initially introduces a small, high-difficulty benchmark targeting detailed visual queries. Furthermore, the authors present a two-stage training method, which is demonstrated to boost the performance of Qwen2.5-VL-7B on their self-constructed benchmark.

### Strengths
1. The proposed benchmark fills a gap in existing open-source benchmarks by specifically testing a model's ability to attend to fine-grained details
2. The proposed method significantly improves the performance of Qwen2.5-VL-7B on their self-constructed benchmark

### Weaknesses
1. Regarding the ground truth for the intermediate reasoning: 1) How was this data obtained or generated? 2) What measures were taken to ensure its correctness and accuracy? 3) What was the underlying method and rationale for its construction?
	2. The authors have collected a 37k-sample dataset for RL, but it is unclear how the contributions of the data and the proposed method are disentangled. The paper would be strengthened by an ablation study that isolates the impact of each component. To provide a clearer picture of the method's efficacy, I suggest including a baseline that applies conventional RL algorithms to the same 37k dataset. This would help clarify whether the performance gains stem from the novel method itself or the curated data.
	3. The paper would benefit from a clearer justification for the evaluation dimensions presented in Section 3. Could the authors elaborate on the rationale behind their selection and explain the method used to ensure their comprehensiveness? Furthermore, providing data distribution across these different dimensions would be crucial.
	4. The paper only provides results on a self-constructed benchmark that focuses on small regions. Provide results on public benchmarks. Besides, does this specialized training approach compromise the model's generalization ability?
	5. Does focusing on a single small point per image create bias? This approach may allow the model to succeed without understanding the global context of the image.

### Questions
Please refer to the issues detailed in the Weakness part.

### Soundness
3

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
5

### Summary
This paper tackles the challenge of making vision language models (VLMs) not only answer visual questions correctly but also show where in the image their reasoning comes from. Current VLMs like Qwen2.5-VL or GPT-4V often produce correct answers without verifiable visual grounding, leading to untraceable or hallucinated reasoning.

To address this, the authors propose TreeVGR, a two-stage training framework built on Qwen2.5-VL-7B, and introduce TreeBench, a new benchmark for evaluating traceable visual grounded reasoning.

TreeBench contains manually verified image–question–answer triplets with annotated bounding boxes marking the visual evidence, covering ten sub-tasks spanning perception (e.g., color, attribute, OCR) and reasoning (e.g., ordering, contact, spatial containment).

TreeVGR first performs supervised fine-tuning to teach the model to produce structured reasoning with bounding boxes, then applies reinforcement learning with a reward that combines answer correctness, reasoning format, and bounding-box IoU.

The resulting TreeVGR-7B achieves nice gains over the base Qwen2.5-VL-7B on TreeBench (+13.4 points) and shows moderate improvements on other multimodal reasoning benchmarks such as V*Bench and MME-RealWorld. The authors claim that TreeVGR enables more transparent and verifiable visual reasoning by linking model predictions to explicit image evidence.

### Strengths
1. TreeBench is carefully constructed and manually verified for correctness and visual traceability, providing one of the first benchmarks that explicitly links reasoning answers to bounding-box evidence.

2. The paper identifies a gap in current multimodal research: the lack of verifiable, evidence-grounded reasoning, and frames the need for “traceable visual reasoning” in a straightforward way.

3. The two-stage TreeVGR pipeline (supervised fine-tuning followed by RL with evidence-based rewards) is simple, reproducible, and effectively demonstrates that incorporating bounding-box supervision can improve visual reasoning performance.

4. The experiments cover diverse perception and reasoning sub-tasks, results are consistently reported, and the writing is clear, making both the dataset and the approach easy to understand and potentially useful for future follow-up work.

### Weaknesses
1. TreeBench, though high quality, relies heavily on manual verification of image–question–evidence triplets, making it difficult to scale to larger or more diverse data. The process is partly automated but still human-dependent, which restricts reproducibility and extensibility.

2. The paper equates correct reasoning with the ability to predict accurate bounding boxes, but provides no empirical evidence that models failing to output boxes are not attending to the correct regions internally. This makes the “traceability” assumption more procedural than cognitive, and potentially misleading.

3. The model is trained and tested on tasks with nearly identical structures and output formats (question + bounding-box evidence). As a result, the large reported gains likely reflect adaptation to the curated data and benchmark design rather than a general improvement in reasoning ability.

4. The RL stage yields only minor improvements over supervised fine-tuning and is insufficiently analysed. The paper does not demonstrate that reinforcement learning meaningfully enhances reasoning depth or grounding beyond improving output formatting.

5. The study evaluates only on reasoning-style benchmarks and does not test whether TreeVGR retains the broader multimodal skills of the base Qwen2.5-VL model. This leaves open the possibility of overfitting or catastrophic forgetting of non-reasoning capabilities.

6. Although the method claims to enhance traceable reasoning, the paper never validates whether the predicted evidence regions align with the model’s internal attention patterns or decision process, limiting the interpretability claims it makes.

### Questions
1. How do the authors envision scaling TreeBench to larger or more diverse datasets without compromising annotation quality or requiring extensive human effort, since this is one of the reasons the authors claim their dataset to be superior?

2. The paper assumes that correct reasoning must manifest through accurate bounding-box prediction, yet models may still attend to the correct region internally without explicitly outputting coordinates. Have the authors analysed attention maps or other internal signals to verify that bounding-box accuracy genuinely reflects visual grounding?

3. Since both the training data and TreeBench share nearly identical QA structures and output formats, to what extent do the reported gains represent task adaptation rather than generalizable reasoning improvements? Have the authors tested transfer to unseen reasoning styles or datasets?

4. The RL component seems to add only marginal improvements. Can the authors clarify what qualitative or behavioural differences emerge after RL fine-tuning compared to supervised fine-tuning alone, and whether these differences justify the added complexity?

5. Given that TreeVGR fine-tunes the full Qwen2.5-VL model, have the authors evaluated whether general tasks such as captioning or VQA are preserved, or does the model overfit to the traceable reasoning format?

6. The paper claims that the method enhances “traceable reasoning,” but without analyzing the alignment between predicted evidence and model attention. Can the authors provide evidence—such as attention heatmaps or token-level visualization—that the model truly focuses on the localized regions it predicts or what was the before vs after effect of their method on the base model?

### Soundness
2

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
This paper presents TreeBench, a benchmark for evaluating traceable visual grounded reasoning, and TreeVGR, a reinforcement learning framework that jointly supervises reasoning and localization using a dual IoU reward. TreeBench includes 405 expert-curated VQA samples with bounding box annotations to assess perception, reasoning, and evidence traceability. TreeVGR improves visual reasoning through a two-stage training process with supervised initialization followed by reinforcement learning, achieving notable improvements across multiple benchmarks and contributing to more explainable multimodal reasoning.

### Strengths
- Studying visual grounding in the reasoning process is very meaningful because most models may ignore intermediate results in decision-making and fail to learn the true causal relationships.
- A new benchmark has been constructed, which allows researchers to consider a wider range of factors.

### Weaknesses
- Although the paper emphasizes achieving traceability through bounding boxes, the "intermediate interpretability" of the inference chain is still weak if it relies solely on box localization measure (mIoU).
- Although the paper includes extensive comparisons with open-source multimodal models such as LLaVA-OneVision and Qwen2.5-VL, these models are not specifically designed for reasoning or reinforcement learning–based visual grounded reasoning. As a result, the experimental comparison may not fully demonstrate TreeVGR’s advantage over other reasoning-oriented approaches.
- In the *Reinforcement Learning with Traceable Evidence* stage, the paper assumes that reasoning chains should explicitly include bounding boxes to ensure traceability. However, the necessity of this design is not fully justified. Reasoning transparency could also be achieved through implicit or attention-based grounding without inserting explicit box tokens. The paper lacks an analysis or ablation to clarify whether explicit box supervision is essential for reasoning quality.
- It is recommended to discuss work related to the interpretability of visual grounding [1].

[1] Interpreting Object-level Foundation Models via Visual Precision Search. CVPR 2025.

### Questions
Please see the weaknesses

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
3

### Summary
This paper aims to enhance the "thinking with images" capability of large multimodal models (LMMs). To this end, we introduce a benchmark named TreeBench and a training pipeline called TreeVGR. Specifically, TreeBench comprises 406 visual question-answer pairs, each accompanied by trace evidence that serves as a verifiable grounding instance. The proposed TreeVGR method extends the reinforcement learning algorithm GRPO to incorporate relevant visual instances (e.g., bounding boxes) as a form of Chain-of-Thought. Extensive experiments on TreeBench and the V* benchmark demonstrate the effectiveness of TreeVGR.

### Strengths
1. The TreeBench Benchmark: They construct a novel VQA benchmark wherein each question is explicitly linked to a groundable instance that serves as traceable evidence, ensuring verifiability.

2. The TreeVGR Method: They propose a GRPO-based training pipeline designed to enhance the groundedness of LMMs in VQA. The TreeVGR method guides the model to identify and utilize relevant evidential instances as a Chain-of-Thought.

### Weaknesses
1. The paper does not explicitly measure the quality of the groundable evidence in TreeBench. Given the semi-automatic annotation process (Lines 91-101), how is the correctness of this evidence guaranteed? A compelling way to validate the importance of the evidence would be to observe if masking the critical instances (e.g., the bounding boxes) leads to a significant drop in VQA performance.

2. The overall reward $R=R_{acc} + R_{iou} + R_{format}$ combines terms from different scales. Were these reward components normalized to a common range to prevent any single term from disproportionately dominating the optimization?

### Questions
1. How is the causal relationship between the evidence and the final answer validated? A critical test would be to see if the model's performance drops significantly when the key evidence instances are masked.

2. Should the individual reward components ($R_{acc}, R_{iou}, R_{format}$) be normalized to the same scale? If not, there is a risk that the term with the largest magnitude could dominate the entire training process.

### Soundness
3

### Presentation
3

### Contribution
2
