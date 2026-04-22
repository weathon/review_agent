# MC-LLaVA: Multi-Concept Personalized Vision-Language Model

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Current vision-language models (VLMs) show exceptional abilities across diverse tasks, such as visual question answering. To enhance user experience, recent studies investigate VLM personalization to understand user-provided concepts. However, they mainly focus on single-concept personalization, neglecting the existence and interplay of multiple concepts, which limits real-world applicability. This paper proposes the first multi-concept personalization paradigm, MC-LLaVA. Specifically, MC-LLaVA employs a multi-concept instruction tuning strategy, effectively integrating multiple concepts in a single training step. To reduce the costs related to joint training, we propose a personalized textual prompt that uses visual token information to initialize concept tokens. Additionally, we introduce a personalized visual prompt during inference, aggregating location confidence maps for enhanced recognition and grounding capabilities. To advance multi-concept personalization research, we further contribute a high-quality instruction tuning dataset. We carefully collect images with multiple characters and objects from movies and manually generate question-answer samples for multi-concept scenarios, featuring superior diversity. Comprehensive qualitative and quantitative experiments demonstrate that MC-LLaVA can achieve impressive multi-concept personalized responses, paving the way for VLMs to become better user-specific assistants. The code and dataset will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes MC-LLaVA, a method for multi-concept Large Vision-Language Model personalization. The contribution of this paper includes:
1. A dataset for the task of multi-concept LVLM personalization.
2. A method to address the problem of multi-concept LVLM personalization, including joint training to eliminate the high-quality negative samples, and visual prompting to enable LVLM personalization for multiple concepts.

The dataset contribution is significant to the field of Multi-concept LVLM personalization, since this field lacks a large dataset for training and evaluation, as the current datasets are quite simple and mostly are focused on single-concept personalization.

### Strengths
1. The dataset contribution of this paper is crucial. There are not many available datasets for LVLM personalization, especially in terms of the multi-concept setting. Most of the dataset from YoLLaVA and MyVLM focuses on a single concept of LVLM personalization, and most images from that dataset only contain a single identity.
2. The experiments indicate the superior performance of the proposed methods compared with the Baselines, including YoLLaVA, MyVLM, and RAP-MLLM.
3. The joint training mechanism can reduce the need for high-quality negative samples, which is essential, since the YoLLaVA paper requires a huge dataset (Laion) to retrieve the high-quality negative samples.

### Weaknesses
There are several weaknesses in this paper:

1. Lack of novelty: The whole framework is based solely on YoLLaVA for personalization, including the system prompt design and test-time training for each personalized concept.
2. The mark indicator is the one that enables multi-concept recognition; however, it is not new in terms of LVLM personalization. The Personalization Toolkits [1] paper has already used this idea to mark the personalized concept. The paper should discuss this.
3. Inefficient method: This method is inefficient; it requires both test-time fine-tuning and a memory bank to store the personalized CLIP features for multi-concept personalization, especially in the case of the continual setting, while new concepts arrive at test time. To me, it looks like we are both using test-time fine-tuning and also using a RAG mechanism to address the LVLM personalization. The test-time fine-tuning takes a lot of time, showing this method is less efficient compared to the RAG-based method, such as Personalization Toolkits [1] and RAP-MLLM [2]. Also, this paper lacks a comparison with the LVLM models [3], which accept multiple images as input, and can accept multiple images of the same concept as inputs.
4. Dataset: How can we ensure the quality of the dataset for the QA task? If there are multiple IDs in the same image, is there any chance that the LVLM can have a correct answer on the QA task, but it will be wrong in the recognition task with the same concept? From what I see in the dataset construction, we are only counting the correct answer, similar to the YoLLaVA paper.
And in addition, it is hard to describe the location of a concept in the photo for the grounding task, especially in the case of multiple objects in the image. In computer vision, the position of an object should be indicated by either a segmentation mask or a bounding box.
It is better for the grounding task to be compared with grounded-capability LVLM [4].

Reference:

[1] Seifi, Soroush, et al. "Personalization Toolkit: Training Free Personalization of Large Vision Language Models." arXiv preprint arXiv:2502.02452 (2025).

[2] Hao, Haoran, et al. "RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[3] https://huggingface.co/docs/transformers/en/model_doc/llava_next

[4] Zhang, Tao, et al. "Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding." Advances in neural information processing systems 37 (2024): 71737-71767.

### Questions
Based on the weaknesses, there are several follow-up questions:
1. What is the fine-tuning time of MC-LLaVA? How do we choose the concept to be the negative samples when we want to train new concepts?
2. What are the results compared with the Personalized Toolkit [1] and some LVLMs that accept multiple images as input [2]?
3. If there are multiple IDs in the same image, is there any chance that the LVLM can have a correct answer on the QA task, but it will be wrong in the recognition task with the same concept? 
4. What is the difference between the S and M versions of YoLLaVA baselines?
5. Is the comparison with GPT-4o P fair? Because the GPT-4o only receives the text prompt as the knowledge of the concept, and MC-LLaVA trains on the visual appearance of the concept, thus making the comparison of MC-LLaVA with the LVLM model that receives multiple images as input less efficient.
6. What is the performance of MC-LLaVA on multi-turn conversation?

[1] Seifi, Soroush, et al. "Personalization Toolkit: Training Free Personalization of Large Vision Language Models." arXiv preprint arXiv:2502.02452 (2025).

[2] https://huggingface.co/docs/transformers/en/model_doc/llava_next

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
3

### Summary
This paper investigates multi-concept personalization for Vision-Language Models (VLMs), advancing beyond existing single-concept approaches. To achieve this goal, the authors propose a multi-concept instruction tuning strategy that integrates: Personalized textual prompts (initialized with visual tokens for faster training convergence) and Personalized visual prompts (encoding spatial location information). Additionally, the authors introduce a new multi-concept instruction tuning dataset. Experiments are conducted on their proposed dataset and two benchmarks (Yo’LLaVA and MyVLM), evaluating four tasks: recognition, visual grounding, VQA, and captioning.

### Strengths
1. The writing is concise and the method is sound.
2. Significant performance improvements are demonstrated across all three datasets for both single- and multi-concept personalization.

### Weaknesses
1. Primary Concern: The core contribution of the paper lies in extending single-concept personalization to multi-concept personalization. The proposed method is technically sound, as it guides each textual prompt with its corresponding visual concept. However, the paper primarily addresses the increasing number of concepts while lacking depth in tackling critical challenges inherent to multi-concept personalization, such as concept disambiguation—for instance, how to differentiate between visually similar objects (e.g., distinguishing between two nearly identical objects).

2. Missing important ablation studies on hyperparameters: 1) K-means clustering: Why is k-means used to generate personalized textual prompts? How is the optimal k (number of clusters) determined? Is k fixed across all concepts, or does it adapt to the variance in visual appearance? 2) Number of images per concept (e.g., minimum required for robust personalization).

### Questions
1. Must all textual concept prompts be provided at inference? If so, how does this scale with increasing concepts?
2. How is location information integrated into the VLM? 
3. Clarify the role of "Mark number 1" (e.g., is it a placeholder for a mask image or a learned token?).

### Soundness
4

### Presentation
4

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
This paper presents a vision-language model designed for multi-concept personalization. Unlike prior single-concept approaches, the proposed MC-LLaVA jointly learns multiple user-defined concepts through multi-concept instruction tuning, visual-based token initialization via k-means, and personalized visual prompts for grounding. Moreover, the authors also build a new multi-concept instruction dataset.

### Strengths
1. The paper addresses an important limitation of existing personalized VLMs by focusing on multi-concept personalization, a realistic setting that significantly broadens the applicability of personalized vision-language models.

2. The experimental results and ablation studies showing the effectiveness of the proposed method.

### Weaknesses
1.  The paper lacks analyses on key factors such as the number of concepts per image, backbone dependence, and the choice of (k) in k-means initialization, leaving the robustness and sensitivity of the method insufficiently explored.

2. It is unclear how the confidence thresholds τ and γ are determined or tuned. It also lacks ablation studies or criteria to justify their choice, making the robustness of the method with respect to these parameters uncertain.

3. While useful, the proposed MC-LLaVA seems a straightforward engineering combination of existing ideas (e.g., multi-concept joint training and visual prompt aggregation) for multi-concept setting.

4. Although several visual examples are shown, the paper does not include failure cases or discuss limitations such as concept confusion, hallucination, or overfitting to synthetic identifiers.

### Questions
The QA pairs and captions of the proposed dataset are primarily produced via GPT-4o, with limited human verification. This raises concerns about potential annotation noise, circular evaluation bias, and whether GPT-generated labels are truly 'ground truth'.

### Soundness
3

### Presentation
3

### Contribution
3
