# MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Pre-trained vision language models have shown remarkable performance on visual recognition tasks, but they typically assume the availability of complete multimodal inputs during both training and inference. In real-world scenarios, however, modalities may be missing due to privacy constraints, collection difficulties, or resource limitations. While previous approaches have addressed this challenge using prompt learning techniques, they fail to capture the cross-modal relationships necessary for effective multimodal visual recognition and suffer from inevitable computational overhead. In this paper, we introduce MoRA, a parameter-efficient fine-tuning method that explicitly models cross-modal interactions while maintaining modality-specific adaptations. MoRA introduces modality-common parameters between text and vision encoders, enabling bidirectional knowledge transfer. Additionally, combined with the modality-specific parameters, MoRA allows the backbone model to maintain inter-modality interaction and enable intra-modality flexibility. Extensive experiments on standard benchmarks demonstrate that MoRA achieves an average performance improvement in missing-modality scenarios by 5.24% and uses only 25.90% of the inference time compared to the SOTA method while requiring only 0.11% of trainable parameters compared to full fine-tuning. The code is available at https://github.com/Tree-Shu-Zhao/MoRA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MoRA (Missing Modality Low-Rank Adaptation), a parameter-efficient fine-tuning method for vision-language models (VLMs) under missing modality scenarios.

MoRA addresses this by introducing two types of parameters:
* Modality-specific parameters for independent adaptation of each modality.
* Shared cross-modal parameters that facilitate bidirectional knowledge transfer via Gram matrices in a low-rank space.

This allows MoRA to maintain inter-modality alignment while preserving intra-modality flexibility, without introducing inference overhead.

### Strengths
Novel contribution: The use of Gram matrices for cross-modal low-rank adaptation is original and mathematically elegant, avoiding dimension mismatch between modalities.

Comprehensive evaluation: The paper evaluates across multiple datasets, missing ratios (50–90%), and cross-scenario generalization settings.

Strong empirical results: MoRA outperforms baselines in accuracy.

Efficiency: Achieves high performance with minimal trainable parameters and no inference overhead, a key improvement over prompt-based tuning.

Insightful analyses: Includes embedding-space, eigenvalue, and ablation studies supporting claims of inter-modal alignment and modality-specific adaptability.

Extensible framework: Demonstrates that MoRA generalizes beyond classification to retrieval tasks (CIRR→COCO).

### Weaknesses
Evaluation diversity: All experiments use CLIP ViT-B/16 as the backbone; results on other architectures are only briefly summarized (Fig. 5). 

Limited modality scope: Experiments are restricted to image–text tasks. The paper claims generality but does not test on other modality pairs.

Hyperparameters were likely tuned on the test set and differ from dataset to dataset. Hyperparameters should be either selected on a validation set or on one dataset and shared across all datasets.

Experiments are restricted to image-text tasks, despite claiming generality, the method is not tested on audio–visual or video–text datasets.

### Questions
Could the Gram matrix mechanism be extended to more than two modalities (e.g., vision, text, and audio)?

How well does MoRa perform when it uses the same hyperparameters for all datasets (for example, the one for Food101)

How sensitive is MoRA on the rank?

Does MoRA’s shared Gram-space adaptation still yield improvements when applied to unaligned encoders?

### Soundness
4

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
This paper addresses the missing modality problem in visual recognition tasks and proposes missing modality low-rank adaptation (MoRA), a parameter-efficient fine-tuning approach that explicitly models cross-modal interactions while maintaining modality-specific adaptations. Specifically, the MoRA introduces shared cross-modal parameters based on the Gram Matrix to enable bidirectional knowledge transfer, along with modality-specific adapting parameters to preserve the unique characteristics of each modality. Extensive experiments and ablation studies are conducted to demonstrate the efficacy of the proposed method.

### Strengths
+ The paper is technically sound, well-motivated, and tackles the missing modality issue, which is an important and practical challenge in multimodal learning.
+ The proposed MoRA method is conceptually interesting and carefully designed, as it explicitly models cross-modal interactions while retaining modality-specific adaptations.
+ The proposed MoRA brings impressive improvements and consistently outperforms other SOTA methods across various missing-modality scenarios. 
+ Extensive ablation studies are conducted to validate the contribution of each component and showcase the efficiency of MoRA in both training and inference time and computational cost.

### Weaknesses
- Scalability to more modalities (e.g., beyond two) is a potential concern. While MoRA appears feasible for dual-modality tasks such as image–text or audio–text learning, its current design may not scale efficiently to scenarios involving multiple modalities (e.g., 4 or more), where cross-modal adaptations could grow exponentially in cost and complexity.
- The generalizability across multimodal architectures is not clearly demonstrated. The proposed MoRA is primarily evaluated on a CLIP-like architecture, which has a separate encoder for each modality. However, for architectures such as ViLT or InstructBLIP, where early fusion occurs and modality interactions are embedded within transformer attention layers, it remains unclear whether the proposed adaptation strategy would still be directly applicable or equally effective.

### Questions
Please address my concerns in the weaknesses section.

### Soundness
4

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
4

### Summary
This paper addresses performance degradation in Vision-Language Models (VLMs) when a modality is missing, a scenario where existing prompt-based solutions suffer from high inference latency. The authors propose MoRA (Missing Modality Low-Rank Adaptation), a new Parameter-Efficient Fine-Tuning (PEFT) method that combines modality-specific adapters with novel shared low-rank parameters to model cross-modal interactions. Experiments on three benchmarks show MoRA significantly outperforms prior SOTA methods like DCP in missing-modality scenarios while being substantially more efficient at inference time.

### Strengths
- The core mechanism of MoRA is clever. Using Gram matrices to facilitate shared, low-rank adaptation between two frozen encoders of different dimensions is a technical solution to the dimension-mismatch problem.
- The primary advantage of MoRA over its main competitors (MMP, DCP) is its efficiency. Because MoRA is a LoRA-based method, all its adapter weights can be merged into the backbone model post-training.

### Weaknesses
- The central premise of the paper—that missing modalities are a common, critical "real-world scenario"—is weakly motivated. The paper justifies this by vaguely citing "privacy constraints, collection difficulties, or resource limitations" but provides no concrete, compelling examples.
- In most practical VLM applications (VQA, captioning, retrieval), the user provides the modalities. The experimental scenarios, such as classifying a "hateful meme" with the text missing, or identifying food from a recipe with the image missing, feel highly contrived and niche.
- The proposed solution is architecturally specific to dual-encoder models like CLIP. It is not clear how this Gram-matrix-based cross-encoder sharing would be applied to modern, single-stack, fused MLLMs (e.g., LLaVA-style models), which are becoming the standard.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MoRA (Missing Modality Low-Rank Adaptation), an innovative approach for parameter-efficient fine-tuning (PEFT) that tackles the issue of absent modalities in pre-trained Vision-Language Models (VLMs) during both training and inference. The proposed approach introduces modality specific and shared parameters with minimal computational overhead. The proposed methodology demonstrates superior performance compared to baseline models across multiple benchmarks involving missing modalities.

### Strengths
* Inclusion of low rank updates for modality specific and shared parameters. 
* Inference time efficiency in terms of minimal computational overhead.
* Superior performance across multiple benchmarks involving diverse missing modality scenarios.
* Strong cross-scenario generalization in cases when training time missing-modality patterns differ significantly from the testing patterns.

### Weaknesses
* The proposed low rank update approach is restricted to the dual encoder models i.e. CLIP or SLIP. The applicability of the proposed approach should be explored for single stream models that relies on alignment of visual representations with LLM inputs.
* The current set of analysis is restricted to image-text datasets with significant text modality dominance. Tasks involving other modality combinations ( audio-visual inputs ) can be considered e.g. audio-visual action recognition.
* Situations involving partial modality corruptions are not considered. Examples include image corruptions by black patches and text corruption by random word dropouts or redaction by [MASK] tokens.
* Detailed analysis regarding the layer placement of MoRA updates w.r.t. individual datasets (MM-IMDb and Hateful-Memes) are not mentioned.

### Questions
* What is the sensitivity of proposed method MoRA to the initialization of the matrices Sv and St ?
* Further, have the authors explored any constraints on the shared parameters e.g. effect of orthonormal columns etc ?
* Have the authors considered a scaling factor between the modality specific and shared modality based updates (Eq 2 and 4) ?

### Soundness
3

### Presentation
3

### Contribution
2
