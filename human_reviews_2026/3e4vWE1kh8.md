# Textual Steering Vectors Can Improve Visual Understanding in Multimodal Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Steering methods have emerged as effective tools for guiding large language models’ behavior, yet multimodal large language models (MLLMs) lack comparable techniques due to recency and architectural diversity. Inspired by this gap, we demonstrate that steering vectors derived solely from text-only LLM backbones can effectively guide their multimodal counterparts, revealing a novel cross-modal transfer that enables reuse of existing interpretability tools. Using community-standard methods—Sparse Autoencoders (SAE), Mean Shift, and Linear Probing—we systematically validate this transfer effect across diverse MLLM architectures and visual reasoning tasks. Text-derived steering consistently enhances multimodal performance, with mean shift achieving up to +7.3% improvement in spatial relationship accuracy and +3.3% in counting accuracy on CV-Bench, and exhibits strong generalization to out-of-distribution datasets. These results highlight textual steering vectors as a powerful, efficient mechanism for enhancing grounding in MLLMs with minimal additional data collection and computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Textual steering vectors enable cross-modal transfer of interpretability tools, providing a lightweight, training-free mechanism to enhance visual reasoning in MLLMs.

### Strengths
The paper demonstrates a cross-modal transfer phenomenon that allows reusing textual interpretability tools to improve visual reasoning in multimodal models without any additional fine-tuning.

### Weaknesses
1. Limited Scope of Related Work on Cross-Modal Steering

The related work section is critically underdeveloped, failing to contextualize this work within the emerging and diverse landscape of cross-modal and multimodal steering. Specifically, the paper omits discussion of methods that directly learn steering from multimodal pairs, or approaches like the concurrent "LLaVA Steering: Visual Instruction Tuning with 500x Fewer Parameters through Modality Linear Representation-Steering" which explores highly parameter-efficient steering, or "Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs," which provides a token-selective, attribution-guided alternative. The claim of architectural agnosticism and novelty is significantly weakened by this failure to rigorously compare and contrast against existing or very recent, conceptually similar cross-modal interventions.

2. Lack of Mechanistic Insight for Cross-Modal Transfer 

The paper's key finding is a "novel cross-modal transfer" of textual steering vectors, yet the underlying mechanism is not explored beyond vague references to "preserved semantics" and "shared semantics." There is no deep analysis or experimental evidence to explain why a vector optimized to represent a concept in a purely linguistic context (e.g., in a text-only LLM backbone) so effectively manipulates the representation of a visual concept (e.g., spatial relationship in an image token) in the MLLM. The current evidence merely demonstrates correlation of effect, not causation or deep mechanistic understanding, leaving the central claim of the paper underspecified and unproven.

3. Instability and Quality Dependence of Steering Vector Extraction 

The authors explicitly state that a "primary limitation" is the "reliance on the quality of extracted steering vectors" and that the vectors from SAE and Linear Probing "can be of poor quality and fail to adequately represent the target concepts," leading to "variable steering performance across different layers and models." This concession undermines two-thirds of the presented methodologies. The consistent superiority of MeanShift suggests the other two methods fail the core test of interpretability and effectiveness, rendering them more of a negative result rather than a robust cross-validation of the transfer effect. The current approach is therefore highly dependent on one specific, and less interpretable, extraction technique.

4. The Superior Generalization of Steering is Overstated and Task-Dependent

While the MeanShift method shows strong out-of-distribution (OOD) performance on highly focused, synthetic datasets like CLEVR (where it is significantly better than LoRA), its generalization advantage diminishes drastically on complex, real-world tasks. As the results in Table 5 show, the average improvement on real-world tasks like VQAv2, COCO Captions, and ChartQA is marginal (often $<+1.0\%$) and in many cases, non-significant or even negative. The claim of "superior generalization capabilities of steering" must be severely qualified: it is an effect primarily observed on datasets designed to isolate the exact visual concept being steered, not a general robustness property for complex, integrated multimodal reasoning.

5. Empirical Efficacy is Limited by Model Size and Prompting Baseline Weakness

The effectiveness of the steering intervention is inversely correlated with model size; the 3B model consistently shows a much larger gain than the 10B model, suggesting the approach is mainly effective for smaller, more malleable models. Furthermore, the prompting baseline is explicitly noted to "barely steer" the MLLMs, contrasting sharply with its efficacy in text-only domains. This weak baseline is a critical issue: the improvements observed may simply be a function of overcoming the inherent difficulty MLLMs have in following any fine-grained visual reasoning instructions, rather than proof of a uniquely powerful steering mechanism. The true margin over a stronger, more optimized baseline is unknown.

### Questions
See Above.

### Soundness
1

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
This paper proposes a novel approach to steer Multimodal Large Language Models (MLLMs) by using "steering vectors" derived from their text-only LLM backbones. The central finding is that semantic representations from the text-only LLM remain effective for intervention even after multimodal fine-tuning, allowing for lightweight, inference-time activation additions to enhance specific visual capabilities like spatial reasoning and counting. The authors demonstrate that this method can achieve better out-of-distribution generalization compared to LoRA fine-tuning. The paper suffers from significant limitations in its experimental setup, evaluation scope, and practical utility, which collectively undermine the generality and impact of its conclusions.

### Strengths
1. The key contribution is demonstrating that representation steering from the text domain can be transferred to the multimodal domain to improve visual task performance.
2. The proposed method is a "plug-and-play" technique that does not require any model retraining or fine-tuning.

### Weaknesses
1. The evaluation is confined to PaliGemma2 and Idefics3-8B. It fails to include a representative set of more recent, powerful, and widely-used open-source MLLMs, such as the LLaVA series, the Qwen-VL family, or InternVL series. This narrow selection makes it difficult to ascertain whether the findings are a generalizable phenomenon or an artifact of the specific architectures tested. Without broader validation, the claims of general applicability are unsubstantiated.
2. The paper's evaluation is heavily concentrated on a few sub-tasks from CV-Bench (spatial relations, counting) and closely related OOD datasets. These are diagnostic, often simplified tasks designed to probe isolated skills. The method's utility is largely confined to simple, isolated cognitive skills and does not translate effectively to the complex, compositional reasoning required by general-purpose multimodal applications.
3. The method requires a user to first define a specific capability to improve (e.g., counting, color recognition), then extract a corresponding steering vector, and finally perform an expensive grid search to find the optimal intervention layer and strength for each model-task pair. This workflow is cumbersome and does not scale to open-ended, complex scenarios where the required "cognitive skills" cannot be easily enumerated and calibrated in advance.

### Questions
Sea weakness.

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
3

### Summary
This paper presents an interesting insight: steering vectors derived from LLM backbones can effectively guide their multimodal counterparts to enhance visual understanding. The authors systematically validate the effectiveness of various steering vector extraction methods across different MLLM architectures and visual reasoning tasks. Experimental results demonstrate that textual steering vectors significantly improve MLLMs' performance on tasks such as spatial relation reasoning and object counting.

### Strengths
1. The paper presents a novel insight by demonstrating the transferability of textual representations from LLMs to MLLMs, revealing an effective steering mechanism.
2. The method achieves strong empirical performance, consistently improving results on spatial relation and counting tasks across multiple MLLM architectures.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The evaluation is limited to visual reasoning tasks (e.g., counting, spatial relations) without exploring more complex multimodal scenarios such as OCR.
2. The experiments are restricted to a small set of MLLM backbones; validation on more recent and advanced models (e.g., Qwen-VL, InternVL families) would further strengthen the generality of the conclusions.

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
3
