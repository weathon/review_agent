# Catching the Details: Self-Distilled RoI Predictors for Fine-Grained MLLM Perception

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Multimodal Large Language Models (MLLMs) require high-resolution visual information to perform fine-grained perception, yet processing entire high-resolution images is computationally prohibitive. 
While recent methods leverage a Region-of-Interest (RoI) mechanism to focus on salient areas, they typically present a difficult trade-off: training-based approaches depend on large-scale annotated datasets, while training-free methods that utilize the model's internal attention are computationally inefficient and less accurate, requiring either multi-pass prefill stages or reliance on the slow auto-regressive decoding process.
In this paper, we propose an efficient, annotation-free Self-Distilled Region Proposal Network (SD-RPN) that resolves this trade-off. The SD-RPN is built around a pipeline that transforms the noisy attention maps from the MLLM's middle layers into high-quality pseudo-RoI labels by explicitly denoising the signal and resolving ambiguity. We use these labels to train a lightweight Region Proposal Network (RPN) that learns a more precise localization. This RPN is also highly efficient, predicting the RoI in a single forward pass using features from the MLLM's middle layers, decoupling RoI identification from the auto-regressive generation and avoiding costly multi-pass operations.
To validate our approach, we integrate the framework into the LLaVA-1.5 architecture. Despite being trained on only a few (e.g. 10K) question-answer pairs, our method demonstrates exceptional data efficiency and generalization, achieving over a 10\% absolute accuracy improvement on unseen benchmarks, including TextVQA, DocVQA, and V-Star. Our work presents a practical and scalable solution for enhancing the fine-grained perception of MLLMs without requiring costly supervision or full model fine-tuning. Code is available at https://github.com/YuHengsss/SD-RPN .

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
**Catching the Details: Self-Distilled ROI Predictors for Fine-Grained MLLM Perception** proposes a Self-Distilled ROI Predictor (SD-RPN) to address the high-res image perception problems. Unlike previous sft-based or training-free ROI-based methods, this paper pioneered using the filtered middle layer attention map as supervision signal, and trained several middle layers to serve as the SD-RPN. The overall idea is clear, elegant, and generalizable to other encoder+llm type of VLMs. The effectiveness of the proposed method is demonstrated via experiment, with solid and comprehensive growth across benchmarks.

### Strengths
1. The idea of Self-Distilled ROI Predictor is valuable, as it fully leverages the contextualization power of a VLM model, without relying on human-annotated data.
2. The training is not computation-efficient, and demonstrates consistent growth when data quantity is increased (Tab. 4b). This shows the potential of scaling up of this method.
3. The writing and illustration of the paper are excellent. The paper has expressive and clear figures, solid algorithms and detailed explanation for methods.

### Weaknesses
1. The experiments are limited to ROI methods and do not compare against other SoTA methods for high resolution. If authors could add some work such as Monkey, Token-Packer, Mini-Gemini, Honey-bee, LLaVA-NeXt, which have competitive numbers on the selected benchmarks such as DocVQA, the paper would be more solid.
2. The design of taking middle layers is not ablated. I have read Appendix A. While intuitively the middle layers should be a balance of visual details and global semantics, it should be proven, since using the middle layers can be much more expensive in computation compared with shallow layers.
3. (minor) The 1.6x slowdown does not support the claim of "light weight". Maybe try rephrasing the claim to be clearer.
4. (minor) The hyperparameters, like B and Sbg, Sfg thresholds, are not ablated.

References:
[1] Li, Z.; Yang, B.; Liu, Q.; Ma, Z.; Zhang, S.; Yang, J.; Sun, Y.; Liu, Y.; and Bai, X. 2023d. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607

[2] Li, W.; Yuan, Y.; Liu, J.; Tang, D.; Wang, S.; Zhu, J.; and Zhang, L. 2024a. Tokenpacker: Efficient visual projector for multimodal llm. arXiv preprint arXiv:2407.02392

[3] Li, Y.; Zhang, Y.; Wang, C.; Zhong, Z.; Chen, Y.; Chu, R.; Liu, S.; and Jia, J. 2024b. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814

[4] Cha, J.; Kang, W.; Mun, J.; and Roh, B. 2024. Honeybee: Locality-enhanced projector for multimodal llm. In IEEE CVPR, 13817–13827

[5] Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024a. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge.

### Questions
1. Have authors considered ablating shallow layers? If not, is there a more convincing explanation?
2. Why are DocVQA and ChartQA significantly lower than the token compression methods? 

I have a positive attitude towards this paper's contributions. I would maintain the high score if authors can address my concerns.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel annotation-free framework termed the Self-Distilled Region Proposal Network (SD-RPN), designed to address the challenge that MLLMs face in comprehending fine-grained text or objects within high-resolution images. It circumvents the limitations of existing approaches: training-based methods that rely on large-scale annotated datasets, and training-free techniques that suffer from computational inefficiency and lower accuracy. The proposed approach transforms the inherently noisy attention maps within MLLMs into high-quality pseudo-RoI labels, which are subsequently used to train a lightweight RPN. Furthermore, the weights from middle layers of the MLLM are used to initialize the RPN, enhancing its performance. Empirical results demonstrate the high effectiveness of the method, achieving substantial absolute accuracy gains (over 10%) on benchmarks such as TextVQA and DocVQA when integrated into existing MLLMs.

### Strengths
1. This paper introduces a self-distillation framework for training a lightweight RPN, resolving the difficult trade-off between training-based methods, which require costly human annotations, and training-free methods, which are often inefficient and inaccurate.
2. The pseudo-label generation pipeline proposed is clear and feasible, as it involves (i) identifying and removing "sink tokens" and (ii) adopting a selective classification strategy that labels only high-confidence foreground/background regions.
3. The paper validates the effectiveness of the proposed approach through a wide range of benchmarks.

### Weaknesses
1. This paper introduces thresholds in stages such as pseudo-label generation and RoI prediction. However, the authors do not justify the rationale behind these specific threshold values, nor do they explore how variations in these thresholds might affect model performance.
2. The paper seems to lack an in-depth analysis regarding the sources of "noise" during pseudo-label generation. For instance, it does not adequately address the types of noise involved, how such noise is generated, or what kinds of images or tasks are prone to producing noisy labels.
3. The qualitative examples presented in Figure 5 effectively showcase the successes of SD-RPN. However, a comprehensive evaluation should also include an analysis of its failure modes. The paper would be strengthened by showing examples where SD-RPN fails and diagnosing the cause.

### Questions
1. How were the specific threshold values (e.g., 0.2 and 0.1 for foreground/background definition) determined?
2. Which types of images or questions are more prone to introducing noise in the attention maps?

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
This paper proposes a method to enhance the visual comprehension capability of a multimodal large language model (MLLM) by training a region-proposal network, which is initialized from a subset of the MLLM's parameters. The training data for the region proposals are derived from the attention maps of intermediate layers in the MLLM. To obtain more precise and less noisy region-of-interest (ROI) targets, the authors introduce several techniques, such as removing sink tokens and dynamically assigning labels to instances.

With the trained region-proposal network, the MLLM can perform two-stage inference: in the first stage, the model predicts ROIs using the proposal network; in the second stage, it processes a high-resolution version of the selected ROI to enable more detailed visual information extraction. The authors conduct extensive experiments to validate the effectiveness of the proposed method, and provide ablation studies to assess the contribution of each design component.

### Strengths
* The overall presentation of the paper is strong. The manuscript is well-written, logically structured, and easy to follow.

* Extensive experiments are conducted to demonstrate the superiority of the proposed method across multiple MLLM benchmarks.

* The method achieves notable performance improvements, particularly on OCR-related tasks, highlighting its strengths in fine-grained visual understanding.

* The ablation studies and visualizations provide clear evidence of the effectiveness of the key components in the proposed approach.

### Weaknesses
* I question the necessity of extracting ROIs from attention maps, as they can be noisy and spatially imprecise. Given the availability of large-scale object detection datasets (e.g., OpenImages) and powerful MLLMs capable of generating captions, it may be feasible to construct a dataset with high-quality region-text annotations. It would be helpful for the authors to discuss whether their approach offers any advantages—such as cost, scalability, or task-specific alignment—over using such manually or model-annotated datasets.

* The authors propose several strategies for utilizing the predicted bounding boxes. However, a concern arises: if only the cropped ROI is used for answering the current question, could this lead to a loss of global visual context? While the global scene may appear irrelevant to the immediate query, it might provide valuable information for subsequent questions in a multi-turn dialogue. In such cases, does the proposed two-stage inference risk discarding useful contextual cues, potentially harming performance over extended conversations?

* The RPN is designed to predict ROIs based on the provided textual context. However, using a portion of an MLLM with billions of parameters for this purpose seems computationally heavy. Since the goal resembles that of open-set grounding models, it would be insightful to compare against lightweight alternatives such as fine-tuned Grounding DINO or YOLO-World. Could such models achieve comparable ROI localization performance with significantly lower computational overhead?

* The standalone performance of the proposed RPN remains unclear. There are no dedicated benchmarks or quantitative metrics reported to evaluate its accuracy in ROI prediction. Could the authors clarify how they validated the quality of the generated proposals during development? For instance, were human evaluations or proxy metrics used?

* I noticed that some visualization results are provided in the supplementary material. Including a few representative examples in the main paper—such as attention maps, predicted ROIs, and corresponding model outputs—would greatly enhance the reader’s understanding of the method’s working mechanism and effectiveness.

* I am also interested in potential failure cases. Are there instances where the baseline MLLM answers correctly but the proposed method fails? Analyzing such cases could shed light on the limitations of the current approach and guide future improvements.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a Self-Distilled Region Proposal Network (SD-RPN). The pipeline first converts the response-to-image attention from the MLLM into high-quality pseudo RoI labels through two key steps: removing sink tokens and assigning labels. Then, the RPN is initialized using the corresponding layers of the MLLM. It collects hidden-state sequences from the second-to-last layer and uses the corresponding token sequences to predict an RoI map. Finally, the RPN is trained with self-distillation, constrained by these pseudo RoI labels. Trained on only a few question-answer pairs, the method achieves great accuracy and efficiency improvement on unseen benchmarks.

### Strengths
1. Annotation-free RoI distillation mechanism that elegantly leverages internal attention for supervision. 
2. Improvement on accuracy and efficiency with comprehensive experimental validation across diverse benchmarks. 
3. Clear motivation and insight, especially the analysis of noisy attention and the role of sink tokens.

### Weaknesses
1. The pseudo-label generation method relies on fixed thresholds (e.g., $τ_{fg}$=0.2) to distinguish foreground, background, and ignored regions. These thresholds directly affect the quality of the pseudo labels, but the paper lacks experimental validation with different parameter settings across multiple datasets.
2. Lack of experiments on very recent MLLMs (e.g., Qwen2.5-VL).
3. Efficiency improvement can be further quantified in wall-clock latency and GPU cost.

### Questions
1. SD-RPN was tested on LLaVA and DeepSeek-VL. How does it transfer to other architectures such as Qwen2.5-VL?
2. What is the performance of SD-RPN if pseudo-labels were replaced by ground-truth bounding boxes? Is it the upper bound of SD-RPN?
3. What is the performance of SD-RPN compared to thinking with image methods?
4. Why was BCE chosen over other robust alternatives (e.g., focal loss or IoU-based loss) given the heavy class imbalance between foreground and background tokens?

### Soundness
2

### Presentation
3

### Contribution
3
