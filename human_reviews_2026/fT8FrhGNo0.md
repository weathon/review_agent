# ODI-Bench: Can MLLMs Understand Immersive Omnidirectional Environments?

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Omnidirectional images (ODIs) provide full 360$^{\circ} \times$ 180$^{\circ}$ view which are widely adopted in VR, AR and embodied intelligence applications. While multi-modal large language models (MLLMs) have demonstrated remarkable performance on conventional 2D image and video understanding benchmarks, their ability to comprehend the immersive environments captured by ODIs remains largely unexplored. To address this gap, we first present ODI-Bench, a novel comprehensive benchmark specifically designed for omnidirectional image understanding. ODI-Bench contains 2,000 high-quality omnidirectional images and over 4,000 manually annotated question-answering (QA) pairs across 10 fine-grained tasks, covering both general-level and spatial-level ODI understanding. Extensive experiments are conducted to benchmark 20 representative MLLMs, including proprietary and open-source models, under both close-ended and open-ended settings. Experimental results reveal that current MLLMs still struggle to capture the immersive context provided by ODIs. To this end, we further introduce Omni-CoT, a training-free method which significantly enhances MLLMs’ comprehension ability in the omnidirectional environment through chain-of-thought reasoning across both textual information and visual cues. Both the benchmark and the code will be released upon the publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ODI-Bench, a 360°×180° omnidirectional-image benchmark for VR/AR/embodied settings: 2,000 ODIs, 4,000+ human-annotated QA pairs, 10 fine-grained tasks covering general semantics and spatial reasoning, with both close- and open-ended evaluation. Benchmarking 20 proprietary/open-source MLLMs shows they underperform on ODIs despite strong 2D results, indicating poor immersive/spatial understanding. The authors propose OmniCoT, a training-free chain-of-thought procedure that fuses textual and visual cues, yielding substantial accuracy gains across ODI tasks. Benchmark and code will be released upon publication to catalyze research in panoramic scene understanding.

### Strengths
1. The topic is timely and relevant.
2. The manuscript is clearly written and well structured.
3. The proposed method Omni-COT delivers consistent, meaningful improvements.

### Weaknesses
1. In Table 1, the authors name the proposed benchmark "360Bench." For consistency with the main paper, I recommend renaming it to "ODI-Bench."

2. ODI-Bench focuses on 360-degree views, which the authors claim benefits AR/VR. I’m not fully convinced. Human field of view is roughly ~180 degrees; in AR/VR, users still turn their heads/bodies to see other views. Is it necessary to require models to reason directly on a single 360-degree image?

3. The ODI-Bench images are 360-degree panoramas re-projected to 2D. A concern is that most current models were not trained on such projections and may treat them as "wrapped images," causing train–inference mismatch and errors.

4. As a human evaluator, my answer in Figure 4 would also be "No." The image seems confusing for both VLMs and people. Without seeing Appendix Figure 1 (which reveals that the far left and far right edges are adjacent—i.e., the "back" region), I would likely misinterpret it.

5. When benchmarking, consider giving VLMs minimal necessary priors, e.g., "This is a 360-degree panoramic (pano) view image," to reduce avoidable misunderstandings. I have asked GPT-4o for the Appendix Fig. 6 question: "This is a 360 degree pano view image. Standing under the shelter facing the railway tracks, where is the train in relation to me? A. Behind;B. Right;C. Left;D. Front" The model could correctly solve this question.

6. I’m curious whether performance improves if the panorama is split into multi-view images or converted into a short continuous video and then fed to the VLM, followed by inference on the benchmark questions (not viewpoint guiding—just direct inference).

### Questions
See weaknesses.

Besides, authors claim the benchmark are high-resolution images. However, in recent paper [1, 2], the researchers discussed that in most general scenarios, even simple resizing can achieve strong performance, do not need such high resolution image. How do the authors view this issue? I look forward to some discussion on this point.


[1] VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning

[2] Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper makes two main contributions:

1.	ODI-Bench, an omnidirectional image question-answering (QA) benchmark for multimodal large language models (MLLMs), consisting of 2,000 omnidirectional images and over 4,000 manually annotated QA pairs.
2.	Omni-CoT, a training-free method designed to enhance MLLMs’ comprehension ability on omnidirectional image QA tasks.

The authors demonstrate that both open-source and proprietary MLLMs still struggle with reasoning and understanding in omnidirectional settings. The proposed Omni-CoT method improves performance by cropping and wrapping ODI images from multiple viewpoints before feeding them into the models. Experiments show that this approach consistently enhances MLLM performance across different architectures.

### Strengths
* The paper is clearly written and well structured, making it easy to follow.
* The dataset is carefully designed and systematically organized.
* The evaluation of MLLMs is comprehensive and covers a wide range of models.
* The related work section provides a thorough and insightful overview of prior research.
* The proposed Omni-CoT method is simple yet effective, demonstrating consistent improvements across various models.

### Weaknesses
* Limited technical novelty of Omni-CoT: While Omni-CoT is effective, its core idea mainly involves viewpoint decomposition and prompt-based aggregation, which may be seen as a straightforward extension of existing multi-view prompting techniques.
    * Clarification: In lines 396–402, the authors discuss the drawbacks of directly splitting ODIs and feeding them into the model. Could the authors clarify how this baseline differs from Omni-CoT? Is Omni-CoT’s improvement primarily due to its CoT reasoning structure, or due to the view cropping itself? An ablation that isolates these effects would significantly strengthen the paper.
* Dataset scale and reliability: With only 2,000 images and ~4,000 QA pairs, the benchmark is relatively small for evaluating large-scale models. The results may be statistically unstable
    * I would suggest reporting mean ± standard deviation over multiple runs or random seeds to quantify evaluation noise and ensure reproducibility.
* Broader impact and future directions: The paper could benefit from a brief discussion on how ODI-Bench might be used for training, not just evaluation — for instance, as a pretraining or fine-tuning resource for spatial reasoning in immersive environments.

### Questions
* Line 355: Please clarify what is meant by “absolute directions”. Does it refer to directions with respect to a global (earth-fixed) frame, or relative to the ego’s orientation in the scene?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the gap in evaluating the ability of Multi-modal Large Language Models (MLLMs) to understand Omnidirectional Images (ODIs) and constructs ODI-Bench, the first comprehensive benchmark for this task. The benchmark comprises 2,000 high-quality omnidirectional images and over 4,000 manually annotated question-answering (QA) pairs, covering 10 fine-grained tasks. It supports both close-ended and open-ended evaluations, enabling a thorough assessment of MLLMs’ general-level and spatial-level understanding of ODIs. Experiments on 20 representative MLLMs reveal significant shortcomings in current models’ ability to comprehend immersive ODI environments. To tackle this, the paper proposes Omni-CoT, a training-free framework that enhances MLLMs’ ODI understanding through step-by-step reasoning—including viewpoint-guided answering, crop cue grounding and refinement, and response refinement—with its effectiveness validated across multiple models.

### Strengths
1. Benchmark Construction Fills a Critical Domain Gap: ODI-Bench addresses key flaws of existing ODI benchmarks (e.g., low resolution, limited scene diversity, constrained question domains) by providing high-resolution images, covering both indoor and outdoor scenes, and designing diverse tasks. It adopts a hybrid annotation approach (automated pipeline + human verification) to ensure data quality, serving as a unified, reliable benchmark for evaluating MLLMs’ ODI understanding and promoting standardized research in this field.

2. Comprehensive Evaluation Dimensions and Rigorous Experimental Design: For the first time, the paper employs both close-ended (multiple-choice/yes-no) and open-ended evaluation settings. This dual design not only assesses models’ recognition accuracy under constrained options but also measures their generative reasoning ability in unconstrained scenarios. Experiments cover 20 MLLMs of varying types (proprietary/open-source) and parameter scales, with additional baselines (Blind GPT-4o, random choice) for comparison. In-depth result analysis effectively reveals the challenges MLLMs face in ODI understanding.

3. Innovative and Practical Training-Free Enhancement Framework: Omni-CoT targets MLLMs’ insufficient comprehension of immersive ODI environments by introducing a human-like step-by-step chain-of-thought strategy. It guides models to interpret ODI scenes via compact textual prompts (instead of additional image inputs) and refines reasoning using crop cues, avoiding the high resource consumption of training-based methods. The framework demonstrates strong versatility, achieving performance improvements on both proprietary and open-source models.

### Weaknesses
1. Stepwise Ablation of Omni-CoT’s Reasoning Stages Is Insufficient: Existing experiments validate the overall effectiveness of Omni-CoT but fail to disassemble and analyze the individual contributions of its three core steps (viewpoint-guided answering, crop cue grounding and refinement, response refinement). For example, it remains unclear how much each step independently improves performance on spatial-level tasks, or whether crop refinement (a key sub-step) effectively filters out irrelevant cues. Supplementing stepwise ablation experiments will help clarify the role of each component and strengthen the framework’s interpretability.

2. Evaluation of Reasoning Efficiency Is Lacking: Omni-CoT enhances performance through multi-step reasoning but does not report the increase in inference time compared to direct answering. It is recommended to add quantitative analysis of inference efficiency—such as comparing Omni-CoT with direct answering and Zero-shot CoT in terms of average reasoning time per sample—to balance performance gains against time costs.

### Questions
See weaknesses "Evaluation of Reasoning Efficiency Is Lacking".

### Soundness
3

### Presentation
3

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
This paper introduces ODI-Bench, a benchmark designed to evaluate the spatial and reasoning capabilities of multimodal large language models (MLLMs) in immersive omnidirectional environments. The benchmark covers 10 fine-grained tasks across 2,000 images with over 4,200 QA pairs. The authors further propose Omni-CoT, a training-free chain-of-thought prompting framework that decomposes reasoning into multiple stages. Experiments on a wide range of MLLMs are performed.

### Strengths
1.	The proposed benchmark for ODI is timely.
2.	The paper is generally easy to follow and polished.
3.	The results are promising with the proposed Omni-CoT.

### Weaknesses
1.	The dataset scale is somewhat limited. Could the diversity of ODI-Bench cover the real-world scenes? 
2.	The benchmark relies heavily on automatic template-based question synthesis, which may restrict linguistic diversity and introduce annotation bias.
3.	Could the authors provide ablation studies to show the effects of viewpoint, crop, and refinement stages? It is suggested to provide more hyperparameter ablation to provide more insights.
4.	The comparison focuses only on MLLMs. Could the authors compare with the method that first reconstructs 3D, followed by evaluation by 3D-aware LLM methods?
5.	The authors should clarify the data licenses. 
6.	Figure 1 is confusing, especially the upper right figure.

### Questions
The questions are listed above.

### Soundness
3

### Presentation
3

### Contribution
2
