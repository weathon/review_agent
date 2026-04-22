# LINK: Learning Instance-level Knowledge from Vision-Language Models for Human-Object Interaction Detection

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Human-Object Interaction (HOI) detection with vision-language models (VLMs) has progressed rapidly, yet a trade-off persists between specialization and generalization. 
Two major challenges remain: (1) the sparsity of supervision, which hampers effective transfer of foundation models to HOI tasks, and (2) the absence of a generalizable architecture that can excel in both fully supervised and zero-shot scenarios. 
To address these issues, we propose \textbf{LINK}, \textbf{L}earning \textbf{IN}stance-level \textbf{K}nowledge from VLMs. 
First, we introduce a HOI detection framework equipped with a Human-Object Geometrical Encoder and a VLM Linking Decoder. 
By decoupling from detector-specific features, our design ensures plug-and-play compatibility with arbitrary object detectors and consistent adaptability across diverse settings. 
Building on this foundation, we develop a Progressive Learning Strategy under a teacher-student paradigm, which delivers dense supervision over all potential human-object pairs. 
By contrasting subtle spatial and semantic differences between positive and negative instances, the model learns robust and transferable HOI representations. 
LINK sets new state-of-the-art on SWiG-HOI, HICO-DET, and V-COCO across zero-shot, fully supervised, and open-vocabulary settings, with strong generalization to unseen interaction categories.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LINK, a method for Human-Object Interaction (HOI) detection that addresses the specialization-generalization trade-off. Its core is a decoupled architecture with a geometrical encoder and a vision-language model (VLM) linking the decoder, enabling plug-and-play use with any object detector. To overcome sparse supervision, it introduces a progressive learning strategy using self-distillation, which provides dense, instance-level guidance across all human-object pairs. The method achieves new state-of-the-art results across fully-supervised, zero-shot, and few-shot benchmarks, demonstrating strong generalization and open-vocabulary capability.

### Strengths
- This paper investigates the critical challenges in HOI detection: the sparsity of supervision.
- The experimental results demonstrate the effectiveness of the proposed methods.
- The evaluation covers various settings, including zero-shot, few-shot, and fully-supervised HOI detection.

### Weaknesses
- The proposed teacher-student paradigm has an inherent circular dependency. The teacher model, which is intended to mitigate annotation sparsity by providing dense supervision, is itself trained solely on the original sparse annotations. This bootstrapping approach may limit the upper bound of knowledge that can be distilled to the student, as the teacher's guidance is fundamentally constrained and potentially biased by the initial sparse data. 
- The teacher-student paradigm lacks significant novelty. Knowledge distillation is a well-established and generic technique in deep learning. While its application to the HOI domain is sensible and effective, the paper does not specifically optimize the distillation for HOI problem, but rather adapts an existing general-purpose framework (using logit, feature, and query-level losses) to a new problem.
- The architectural generality is not a distinct advantage over existing paradigms. The paper positions the lack of a generalizable architecture as a key challenge. However, numerous prior VLM-based methods (e.g., those relying on CLIP embedding matching for relation prediction) are inherently designed to operate in both fully-supervised and zero-shot settings by leveraging the same pre-trained semantic space. Therefore, the goal of creating a generalizable architecture is not novel, and the paper's claim rests primarily on its specific implementation (the LINK framework) achieving superior performance, rather than on identifying a new problem.
- Missing explanation of annotation in the Table. In Table 1, the subscript on the LINK is not explained, which causes confusion.

### Questions
Please kindly refer to the weakness section.

### Soundness
2

### Presentation
2

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
To address the sparsity of supervision and the absence of a generalizable architecture, this paper propose a HOI detection framework equipped with a Human-Object Geometrical Encoder and a VLM Linking Decoder, and develop a Progressive Learning Strategy under a teacher-student paradigm to deliver dense supervision. Extensive experiments demonstrate state-of-the-art results in both zero-shot and fully supervised settings.

### Strengths
1. The paper provides a well-articulated motivation, perceptively identifying two main challenges in HOI: (1) the sparsity of supervision, and (2) the absence of a generalizable architecture capable of achieving strong performance in both fully supervised and zero-shot scenarios. In response, the paper propose a novel learning strategy and a method to address the limitations.

2. The proposed Human-Object Geometrical Encoder and VLM Linking Decoder comprehensively account for both spatial and semantic associations between HO pairs, maintaining consistent performance across diverse settings. The learning learning strategy, where the student model is jointly supervised by explicit ground-truth annotations and guidance from a pre-trained teacher, effectively enhances its generalization capability.

3. Comprehensive experiments compare the proposed model with baselines across zero-shot, few-shot, fully supervised, and open-vocabulary scenarios. The results demonstrate that LINK excels in both specialization and generalization.

### Weaknesses
1. Risk of Error Accumulation in Model Architecture: inaccurate detector outputs (e.g., incomplete bounding boxes, redundant detections) can lead to erroneous geometric relationship, erroneous position encoding, and imprecise ROI feature extraction, thereby adversely impacting subsequent inference steps. The paper lacks experiments or visualizations to substantiate the robustness of the proposed model under such conditions.

2. In the application, detectors are fine-tuned on target dataset rather than functioning as a training-free plug-and-play module. It is recommended to explore stronger open-vocabulary detectors, such as Yolo-World, Grounding-DINO.

3. While knowledge distillation can enhance generalization ability, the teacher and student models in this work share identical architectures and capacities. Such a distillation setup may not yield substantial improvements and could potentially degenerate into a simple continued training process.

4. Insufficient Baseline Comparison. Sseveral state-of-the-art baselines (e.g., BC-HOI, UniHOI, LAIN) are not included for evaluation.

5. The corner-mark annotations in Tab.1 left unexplained.

### Questions
See the limitations and cons above.

### Soundness
3

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
This paper proposes a unified framework for HOI detection. It introduces two key components, a Geometrical Encoder, which models spatial relationships between humans and objects, and a VLM Linking Decoder, which bridges global vision-language features with instance-level HOI reasoning. In addition, the paper presents a Progressive Learning Strategy based on a teacher-student paradigm, delivering dense supervision to all human-object pairs to alleviate sparse annotations. The experimental study is comprehensive, demonstrating consistent improvements across fully supervised, zero-shot, few-shot, and open-vocabulary settings on multiple benchmarks.

### Strengths
1. A well-motivated design balancing generalization to unseen HOIs and HOI-specific reasoning. The proposed Geometrical Encoder and Linking Decoder jointly enhance HOI reasoning while preserving model generality across diverse foundation models.
2. The Progressive Learning strategy provides dense supervision via teacher-student distillation, enabling the model to learn from all human-object pairs and effectively alleviate sparse-annotation issues.
3. The experiments are comprehensive and achieve SOTA performance consistently.

### Weaknesses
1. The motivation for introducing the Geometrical Encoder is not clearly justified. The paper claims it enhances spatial awareness, yet it lacks quantitative or cited evidence showing insufficient spatial understanding in existing VLMs like CLIP. 
2. In Table 6, the baseline (A1) used for ablation is under-specified. It is unclear what feature representation or decoder architecture it employs. When evaluating the proposed Geometrical Encoder and VLM Linking Decoder, the paper does not clarify whether the baseline uses the original spatial encoding or a standard decoder. This ambiguity makes it difficult to assess the exact improvement source.
3. The teacher model in the progressive learning strategy uses a stronger ViT-L backbone to supervise the ViT-B student, introducing extra knowledge capacity beyond dense supervision. A comparison using a same-scale teacher (e.g., ViT-B to ViT-B) is necessary to ensure the fair comparison.

### Questions
Please refer to the weaknesses. No other questions.

### Soundness
3

### Presentation
2

### Contribution
3
