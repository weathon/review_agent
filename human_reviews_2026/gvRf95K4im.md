# K-Prism: A Knowledge-Guided and Prompt Integrated Universal Medical Image Segmentation Model

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Medical image segmentation is fundamental to clinical decision-making, yet existing models remain fragmented. They are usually trained on single knowledge sources and specific to individual tasks, modalities, or organs. This fragmentation contrasts sharply with clinical practice, where experts seamlessly integrate diverse knowledge: anatomical priors from training, exemplar-based reasoning from reference cases, and iterative refinement through real-time interaction. We present $\textbf{K-Prism}$, a unified segmentation framework that mirrors this clinical flexibility by systematically integrating three knowledge paradigms: (i) $\textit{semantic priors}$ learned from annotated datasets, (ii) $\textit{in-context knowledge}$ from few-shot reference examples, and (iii) $\textit{interactive feedback}$ from user inputs like clicks or scribbles. Our key insight is that these heterogeneous knowledge sources can be encoded into a dual-prompt representation: 1-D sparse prompts defining $\textit{what}$ to segment and 2-D dense prompts indicating $\textit{where}$ to attend, which are then dynamically routed through a Mixture-of-Experts (MoE) decoder. This design enables flexible switching between paradigms and joint training across diverse tasks without architectural modifications. Comprehensive experiments on 18 public datasets spanning diverse modalities (CT, MRI, X-ray, pathology, ultrasound, etc.) demonstrate that K-Prism achieves state-of-the-art performance across semantic, in-context, and interactive segmentation settings. Code is available at https://github.com/bangwayne/K-Prism.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work propose K-Prism, which is a unified framework for medical image segmentation that innovatively integrates three knowledge paradigms into a single model: semantic priors (Mode-1), in-context exemplars (Mode-2), and interactive feedback (Mode-3). Its core technique encodes heterogeneous knowledge into a unified dual-prompt representation (1D sparse prompts for "what" and 2D dense prompts for "where") , which is then processed by a Mixture-of-Experts (MoE) decoder with dynamic routing. Experiments on 18 public, multi-modal datasets demonstrate SOTA performance across all three segmentation settings.

### Strengths
- Significant Problem: Directly addresses the long-standing problem of model fragmentation in medical segmentation.

- Novel Architecture: The proposed "dual-prompt" and MoE decoder design is elegant, successfully unifying three heterogeneous knowledge sources.

- Comprehensive Experiments: Extensively validated on 18 public datasets covering diverse modalities.

- SOTA Performance: Outperforms existing state-of-the-art methods in all three tracks: semantic, in-context, and interactive segmentation.

### Weaknesses
- 2D Limitation: The model currently processes 3D medical data (like CT/MRI) based on 2D slices, which loses 3D contextual information.
- In-context Bottleneck: For complex multi-organ segmentation tasks, the in-context mode (Mode-2) performs noticeably worse than the semantic mode (Mode-1).
- Computational Cost: The MoE architecture, while effective, introduces a higher parameter count and computational overhead, potentially hindering real-time deployment.

### Questions
- 3D Extension: What are the main challenges in extending this framework to 3D? Specifically, can the dual-prompt and MoE mechanisms be efficiently scaled computationally?

- In-context Bottleneck: Could using k-shot (k>1) references in Mode-2 help alleviate the performance drop observed in multi-organ tasks?

- MoE Expert Specialization: What specific functions have the MoE experts specialized in? (e.g., are some experts dedicated to positive clicks and others to negative clicks?)

- Reference Quality: How sensitive is the performance of Mode-2 (in-context) to the quality of the reference exemplar (e.g., inaccurate mask) or domain shift (e.g., different scanner)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes K-Prism, a unified medical image segmentation framework that integrates three knowledge sources—semantic priors, few-shot in-context examples, and interactive user feedback. Using a dual-prompt and Mixture-of-Experts design, K-Prism flexibly adapts across tasks and modalities. Tested on 18 public datasets, it achieves state-of-the-art performance in semantic, in-context, and interactive segmentation.

### Strengths
1. Unified design integrating semantic, in-context, and interactive segmentation.
2. Consistent performance gain across diverse datasets and modalities.

### Weaknesses
The novelty of the paper is somewhat unclear. It appears to mainly combine existing ideas from interactive segmentation models (e.g., MedSAM) and in-context segmentation models (e.g., UniverSeg). It is not fully evident what new conceptual or methodological insight the proposed framework contributes beyond this integration. The authors should clarify the unique innovation or theoretical advancement that distinguishes K-Prism from prior works.

The three proposed “knowledge paradigms” — semantic priors, in-context examples, and interactive feedback — are somewhat loosely defined and may not convincingly qualify as distinct forms of knowledge. They are more simply like working mode instead of knoledge.The authors should better justify why these modes are conceptualized as “knowledge sources”.

The motivation for adopting a Mixture-of-Experts (MoE) design is insufficient. In medical segmentation, computational efficiency is typically not a critical bottleneck, so replacing a dense model with MoE seems unnecessary and possibly superficial.

### Questions
1. Authors should consider adding MedSAM for comparison
2. As shown in Table 2 and 3, the improvement is limited

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors propose K-Prism, a unified segmentation framework capable of performing (1) standard segmentation on in-class labels, (2) in-context learning–based segmentation using reference exemplars, and (3) interactive segmentation through user feedback. This is achieved by learning a 1-D sparse prompt that encodes what to segment and a 2-D dense prompt that encodes where to attend, which are jointly processed through a Mixture-of-Experts decoder for dynamic, task-aware feature routing. Experiments conducted on 18 public datasets spanning CT, MRI, X-ray, pathology, and ultrasound demonstrate that K-Prism achieves state-of-the-art performance across all three segmentation paradigms, with generalization to unseen classes and external datasets.

### Strengths
1. K-Prism unifies semantic, in-context, and interactive segmentation within a single framework through a novel design that leverages 1-D and 2-D prompts to encode complementary information about what and where to segment. 
2. The use of a Mixture-of-Experts decoder is well-motivated, enabling task-aware specialization and effective fusion of different prompt types while maintaining a shared representation space.
3. The paper is well-written and easy to follow, with clear methodological explanations and thorough experimental validation that make the work accessible and reproducible.

### Weaknesses
1. While K-Prism performs well on in-distribution and cross-dataset tests, its accuracy drops notably for unseen anatomical classes, suggesting limited generalization beyond trained label types.
2. The paper does not compare against nnU-Net trained under equivalent conditions—either as a single generalist model trained on all datasets or as individual task-specific models. Such comparisons are important to validate that the proposed unification truly offers an advantage over established supervised baselines.

### Questions
1. The model shows noticeably lower accuracy on unseen anatomical classes compared to in-distribution results. Could the authors provide an analysis explaining why this gap occurs?
2. To better contextualize the proposed framework, can the authors include experiments comparing K-Prism to nnU-Net trained (a) jointly on all datasets as a generalist model, and (b) separately on each dataset as task-specific specialists? 
3. The paper mentions that all competing methods are trained on the same dataset, but it is unclear how interactive baselines (e.g., SAM2, MultiverSeg) were handled. Were these models trained or fine-tuned from scratch, or initialized from existing pretrained weights?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents K-Prism, a unified framework for medical image segmentation that systematically integrates three clinically relevant knowledge paradigms: (i) semantic priors learned from annotated datasets, (ii) in-context knowledge from few-shot reference examples, and (iii) interactive feedback from user inputs.
The key design lies in a dual-prompt representation—1-D sparse prompts encoding “what to segment” and 2-D dense prompts encoding “where to attend”—together with a Mixture-of-Experts (MoE) decoder for dynamic, task-aware routing.
K-Prism supports three operational modes (semantic, in-context, interactive) within a single architecture and achieves state-of-the-art results across 18 datasets spanning CT, MRI, X-ray, ultrasound, pathology, and fundus imaging. The model demonstrates strong generalization to external and unseen-class datasets and enables efficient refinement through user interaction.

### Strengths
- The paper addresses a clear and practical gap: the fragmentation of current medical segmentation models across different knowledge paradigms.
- The dual-prompt + MoE design is conceptually elegant and technically sound, enabling unified training and inference across tasks without architectural modifications.
- Experiments are comprehensive and convincing, covering 18 public datasets, multiple modalities, and both in-distribution and cross-dataset evaluations.
- K-Prism achieves consistent SOTA performance across semantic, few-shot, and interactive settings, with particularly impressive efficiency in interactive segmentation (lowest NoC90 and NoC95).
- Ablation and analysis are thorough, confirming the contribution of each component and the dynamic specialization of experts.
- The work has practical clinical potential, as it can streamline deployment and annotation pipelines.

### Weaknesses
- All experiments are conducted on 2D slices, while many clinical workflows require 3D volumetric segmentation.
- The method relies on the quality and availability of reference examples for in-context mode, which may constrain performance in data-scarce or noisy scenarios.
- Discussion on model interpretability or failure cases is relatively limited.

### Questions
- Could the framework be extended to 3D volumetric segmentation without major architectural changes?
- How significant is the computational cost of the MoE decoder compared to standard transformer-based decoders (e.g., in inference latency)?
- Is there any strategy to automatically balance or select the optimal knowledge source (semantic / in-context / interactive) during inference, especially when user feedback is unavailable?

### Soundness
3

### Presentation
3

### Contribution
3
