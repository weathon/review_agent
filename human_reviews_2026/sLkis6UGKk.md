# DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Recent advances in multimodal language models (MLLMs) have achieved remarkable progress in vision-language reasoning, especially with the emergence of “thinking with images,” which integrates explicit visual steps into the reasoning process. While this paradigm strengthens image-based reasoning, a significant challenge remains: models may arrive at correct answers by relying on irrelevant or spurious regions, driven by prior knowledge or dataset biases. Even when the answer is correct, flawed reasoning indicates that the model has not truly understood the image, highlighting the critical importance of reasoning fidelity in multimodal tasks. To address this issue, we propose DeFacto, a counterfactual reasoning framework that jointly enforces accurate answering and faithful reasoning. A key component of our approach is the design of three complementary training paradigms: (i) positive, (ii) counterfactual, and (iii) random-masking. To enable these paradigms, we develop a pipeline that automatically localizes question-relevant evidence and constructs positive, counterfactual, and random variants, resulting in a dataset of about 100k images. Building on this framework, we train multimodal language models with GRPO-based reinforcement learning, where we design three complementary rewards to guide the model toward accurate answering and evidence-grounded reasoning. Experiments on diverse benchmarks demonstrate that DeFacto substantially improves both answer accuracy and reasoning faithfulness, establishing a stronger foundation for interpretable multimodal reasoning. The code and datasets will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a counterfactual reasoning framework aimed at improving the faithfulness and interpretability of multimodal language models. It ensures that models not only generate correct answers but also ground their reasoning on the relevant visual evidence. The proposed approach leverages Counterfactual GRPO training with three complementary paradigms (positive, counterfactual, and random-masking) to jointly enforce answer accuracy and reasoning consistency. In addition, the authors construct a large-scale counterfactual dataset using open-vocabulary detection and OCR-guided masking to automatically identify question-relevant regions. Extensive experiments across diverse benchmarks demonstrate significant improvements in both answer accuracy and reasoning faithfulness compared to strong baselines.

### Strengths
Pros:
1. The paper proposes a counterfactual training framework that leverages GRPO-based reinforcement learning to jointly optimize both answer correctness and region-level reasoning faithfulness.
2. The authors introduce an automated counterfactual dataset construction pipeline that effectively generates visual grounding cues (bounding boxes) to support multimodal large language model (MLLM) reasoning.
3. The experiments are comprehensive and well-structured, covering a wide range of benchmarks (e.g., VQAv2, OKVQA, GQA, DocVQA, TextVQA). The results demonstrate consistent and substantial improvements in both accuracy and reasoning faithfulness over strong baselines.
4. The paper is clearly written and well-organized, with structured explanations and illustrative figures that make the approach easy to understand and follow.

### Weaknesses
Cons:
1. The automatic evidence localization pipeline relies heavily on open-vocabulary detectors and OCR, which may introduce noise or misalignment when identifying evidence regions. In essence, the framework distills knowledge from these components, making it difficult for the model’s visual grounding capability to surpass that of the underlying detectors.
2. Accurately detecting bounding boxes in figure-type images (e.g., those in Figures 5, 11, and 17) is more challenging than in natural images, such as when distinguishing a specific polyline among multiple overlapping ones. This limitation likely contributes to the smaller performance gains observed in Table 2 compared to Table 1.

### Questions
Questions:
1. Could the authors provide quantitative metrics or human validation to assess how accurately the automatic pipeline identifies question-relevant evidence regions (R⁺)? For example, including human evaluation results on a small subset (e.g., 1,000 images) would help validate the reliability of the evidence localization process. Besides, show some failure examples in Appendix could also help us better understanding the datasets.
2. If the open-vocabulary detectors were used directly as external tools to validate all bounding boxes, and the positive filtered bounding boxes were then provided to the original Qwen2.5-VL-7B model in prompt, would the model achieve performance comparable to the GRPO-finetuned results?
3. The paper mentions a 100k-image dataset constructed from multiple benchmarks. Could the authors elaborate on the domain distribution (e.g., natural images, documents, charts) and discuss how domain balance may influence performance generalization?

### Soundness
3

### Presentation
4

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
“Thinking with images” can lead to correct answers even when the reasoning process is irrelevant. The paper proposes a framework called DeFacto, a counterfactual reasoning framework that aligns reasoning trajectories with supporting evidence. The key idea is to jointly optimize the model with GRPO, not only for accuracy and format reward, but also through (1) positive supervision, (2) counterfactual abstention, and (3) random masking. To enable this optimization, a new counterfactual dataset is automatically constructed. After counterfactual training, the models perform significantly better compared with the based models.

### Strengths
1. The proposed approach is simple yet effective, and the paper is easy to follow.

2. A new counterfactual dataset is introduced, which could benefit the research community.

3. The improvement of the base model after training with the counterfactual dataset is significant.

### Weaknesses
1. The paper lacks comparisons with several recent works listed in [1].

2. The correctness of the constructed dataset depends on multiple upstream models.

3. The benchmarks presented in the main paper are not recent. Although some recent benchmarks are included in the appendix, the selected models differ from those listed in the main paper.

Reference:
[1] Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers

### Questions
What is the model’s performance on more recent benchmarks?

How does it compare with the recent models mentioned in [1]?

Reference:
[1] Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers

### Soundness
2

### Presentation
2

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
This paper introduces ​​DeFacto​​, a novel counterfactual reasoning framework designed to enhance the evidence-groundedness and faithfulness of multimodal large language models (MLLMs). The core innovation addresses a critical weakness in existing "thinking with images" paradigms: models often arrive at correct answers by relying on spurious or irrelevant image regions, indicating a disconnect between their reasoning and the actual visual evidence. To solve this, DeFacto employs three complementary training paradigms, ​​positive, counterfactual, and random-masking​​, which are automatically constructed using a pipeline that localizes question-relevant evidence. The models are then trained using GRPO-based reinforcement learning with a composite reward function that jointly enforces ​​answer correctness, format consistency, and region selection coherence​​.

### Strengths
- The work introduces an innovative approach that effectively addresses a key limitation of previous methods in the field.
- It demonstrates significant and consistent improvements over state-of-the-art baselines across multiple benchmark datasets.
- The proposed framework is efficient and has the potential for broad application in real-world scenarios.

### Weaknesses
- DeFacto's data construction process relies heavily on components such as region proposal networks and open vocabulary detectors. The performance limitations of these upstream models (such as missed detections and false detections) will directly introduce noise, affecting the quality of counterfactual samples, thereby limiting its performance ceiling.
- Lack of the necessary experiments on the suitable task MSTI [a], which includes the object detection, entity recognition and answer generation.
- Lack of the necessary case studies and error analysis to intuitively illustrate the success and failure modes of the proposed DeFacto work.

[a] Chen, Zixin, et al. "CofiPara: A coarse-to-fine paradigm for multimodal sarcasm target identification with large multimodal models." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

### Questions
How to evaluate the quality of the <think> part in terms of explainability or helpfulness or coherence?

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
5

### Summary
This paper targets the problem that existing MLLMs often reason based on irrelevant visual regions. It identifies two failure types: Mislocalized Failure (attending to wrong regions) and Spurious Correctness (producing correct answers from unfaithful evidence). To address this, DeFacto introduces a GRPO-based training framework with counterfactual and random-masking paradigms, encouraging evidence-grounded reasoning and abstention when evidence is missing. The method improves accuracy on several VQA benchmarks.

### Strengths
1. The motivation is fundamental and important. Making the model’s reasoning truly grounded in visual evidence is a key step toward trustworthy multimodal reasoning, and this topic is still relatively underexplored.
2. The design of the region selection coherence reward and answer correctness reward is conceptually sound and practically reasonable.
3. The overall training idea is clear, and the results show noticeable quantitative improvements on several benchmarks.

### Weaknesses
[Major Weakness]
The main concern is that the paper does not directly verify whether DeFacto actually achieves its goal which is to enable the model’s reasoning to be faithful to the image.
1. Although Table 3 includes an ablation between DeFacto and GRPO, this only compares answer accuracy. It does not measure whether the reasoning process itself becomes more visually faithful. As a result, there should be quantitative results  showing whether the thinking process indeed correctly utilizes the predicted visual information (e.g., selected regions or bounding boxes) to answer after applying DeFacto.
2. The paper also does not analyze how often other methods (like GRIT or Visual-SR1) fail due to unfaithful reasoning or succeed despite unfaithful reasoning. Showing these ratios before and after using DeFacto would strongly support the claimed motivation.

[Minor Weakness]
1. It would be beneficial to evaluate DeFacto on additional model backbones (e.g., InternVL3.5) to assess its generalizability across architectures.
2. The paper lacks qualitative comparisons on the reasoning processes between DeFacto and GRPO.

In short, while the method shows numerical improvement, it remains unclear whether the model actually “thinks with the right regions”. These analyses are essential to confirm the motivation of this work, and my overall rating will depend heavily on how the authors address the points in major weakness.

### Questions
1. In Table 1, the term “CF reward” is not clearly defined. Based on my understanding, CF reward refers to your added components (the region selection coherence and answer correctness rewards) combined with the original GRPO training. Please confirm if this interpretation is correct.

### Soundness
3

### Presentation
2

### Contribution
4
