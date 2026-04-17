# JUDO: A Juxtaposed Domain-Oriented Multimodal Reasoner for Industrial Anomaly QA

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Industrial anomaly detection has been significantly advanced by Large Multimodal Models (LMMs), enabling diverse human instructions beyond detection, particularly through visually grounded reasoning for better image understanding. However, LMMs lack domain-specific knowledge, which limits their ability to generate accurate responses in complex industrial scenarios. In this work, we present JUDO, Juxtaposed Domain-Oriented Multimodal Reasoner, a framework that efficiently incorporates domain knowledge and context in visual and textual reasoning. Through visual reasoning, our model segments the defect region by juxtaposing query images with normal images as visual domain context, enabling a fine-grained visual comparative inspection. Furthermore, we inject domain knowledge through supervised fine-tuning (SFT) to enhance context understanding and subsequently guide domain reasoning through reinforcement learning (GRPO) with tailored rewards, opting for a domain-oriented reasoning process. Experimental results demonstrate that JUDO achieves superior performance on the MMAD benchmark, surpassing models such as Qwen2.5-VL-7B and GPT-4o. These results highlight the importance of enhancing domain knowledge and context for effective reasoning in anomaly understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on multi-modal anomaly reasoning via large-scale multi-modal language model. Specifically, it considers how to make the multi-modal language model be more aware of the domain-specific anomaly. To achieve this, it proposes a three-stage model training/post-training pipeline by considering the anomaly discrimination of normal and abnormal cases, the supervised instruction tuning and reinforcement-learning-based post training. The proposed method acgieves state of the art perfroance of anomaly reasoning on the MMAD benchmark.

### Strengths
- The proposed method is well-motivated, seeking to enhance domain-specific anomaly understanding by injecting fine-grained anomaly knowledge throughout the training stages of a large multi-modal language model.

- The proposed three-stage training pipeline, which integrates both visual and linguistic anomaly data, is logically sound and effectively develops strong anomaly reasoning capabilities.

- The reward engineering implemented in the reinforcement-based post-training stage is well-designed and appropriate for the task.

- The paper is well-written and easy to follow, presenting a clear and logical flow of ideas.

### Weaknesses
- The methodological contribution appears limited, as the proposed framework primarily involves a sequential application of standard techniques (instruction-based fine-tuning and reinforcement learning-based post-training). While effective, this approach offers fewer novel insights into the fundamental challenges of injecting anomaly knowledge into multi-modal language models.

- The proposed method demonstrates a significant performance degradation in anomaly discrimination compared to the Qwen-2.5 baseline. The paper offers several hypotheses for this outcome in the analysis section, but these are not experimentally validated, which undermines their credibility. A more rigorous analysis, such as an ablation study, is needed to clarify which training stage introduces the apparent trade-off between anomaly detection and anomaly reasoning capabilities.

- The manuscript lacks a detailed discussion of the training data division and statistics, which are critical for evaluating the effectiveness of the post-training stage. Furthermore, the paper's central claim—that injecting domain knowledge during training is superior to doing so at test-time—is not substantiated with experimental evidence. A direct comparison is needed to validate this assertion.

- The comparison to general-purpose LLMs could be strengthened. While the inclusion of GPT-4o is a good start, this baseline is rapidly becoming dated. To provide a more current and comprehensive understanding of the performance landscape, it would be beneficial to include comparisons with more recent state-of-the-art commercial models (e.g., Claude 3, Gemini 1.5).

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes JUDO, a multimodal reasoning framework for industrial anomaly understanding. The method integrates visual comparative segmentation, domain knowledge injection via supervised QA fine-tuning, and domain-oriented GRPO alignment to improve both defect localization and explanation quality.

### Strengths
1. The problem is well-motivated, addressing the domain knowledge gap that limits existing VLMs in industrial anomaly analysis. 
2. The three-stage training pipeline is clear, systematic, and practically meaningful. The incorporation of juxtaposed segmentation grounding enhances fine-grained visual reasoning, while domain QA fine-tuning contributes to more reliable textual analysis. 
3. The paper is clearly structured and the experimental results are presented in a well-organized manner.

### Weaknesses
1. The proposed framework is mainly an integration of existing components—comparative segmentation, supervised domain knowledge tuning, and GRPO-based post-training—and thus reads more as pipeline engineering rather than introducing a fundamentally new reasoning mechanism or architectural innovation. The methodological novelty therefore appears limited.
2. The model depends on GPT-4o–generated reasoning chains as pseudo ground truth, yet the paper does not evaluate whether these rationales are reliable, consistent, or susceptible to hallucination. This creates a conceptual gap, since the paper also claims that general LMMs (including GPT models) lack domain reasoning capability.
3. The paper attributes the performance degradation in anomaly discrimination to catastrophic forgetting but provides no empirical analysis or mitigation. Without verification or intervention, this explanation remains speculative and raises concerns about the robustness of the multi-stage training pipeline.

### Questions
1. My interpretation is that the method essentially fine-tunes an existing VLM on domain-specific multimodal data, achieving better performance due to domain alignment. Could the authors clarify what they consider the core novelty beyond a well-designed training pipeline?
2. The paper argues that GPT-based multimodal models lack sufficient industrial anomaly reasoning capability. However, GPT-4o is still used as the source of "golden reasoning" in the training pipeline. How do the authors justify the reliability and consistency of GPT-generated reasoning as ground truth in a domain where GPT is claimed to be insufficient? More specifically, how do you ensure the reasoning produced is correct and not hallucinated?

If I have misunderstood the intended contribution or positioning, I would appreciate clarification from the authors.

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
This paper presents JUDO, a Juxtaposed Domain-Oriented Multimodal Reasoner, to address the current weaknesses in industrial anomaly detection using LMMs (Large Multimodal Models). While the GRPO-based LMMs are optimized for tasks grounded in general knowledge, pinpointing the defects in industrial setups requires superior domain knowledge and visual context through a three-stage progressive framework. The framework uses juxtaposed segmentation learning, supervised fine-tuning (SFT) that injects domain-specific knowledge, and reinforcement learning (GRPO) with tailored multi-component rewards for domain-aware reasoning, resulting in an average accuracy of 80.73% and improvement in explainability through domain-aligned, visually grounded reasoning.

### Strengths
1. The three-stage learning framework proposed in the paper is logical and helps advance the integration of domain knowledge into multimodal reasoning. The three-stage progressive learning pipeline (SegJux, DomInj, GRPOdom) effectively combines visual segmentation, textual domain knowledge, and reinforcement learning. This hierarchical design is both conceptually strong and technically sound, marking a meaningful step beyond conventional GRPO-based post-training methods.

2. The visual grounding via segmentation and the domain-aligned textual reasoning make outputs transparent and practical for industrial use. By segmenting the input images into semantically meaningful regions (e.g., defect areas, product components, the model anchors its subsequent textual explanations to specific visual evidence. This ensures that every reasoning step is traceable to an observable region within the image, reducing errors and improving trustworthiness.

3. This paper addresses a significant industrial challenge of creating an automated, explainable anomaly detection system. The findings have potential applicability across various industry segments like manufacturing, logistics, etc.

4. State-of-the-art average on MMAD versus both general LMMs and AnomalyR1; ablations demonstrate additive benefits of each stage.

### Weaknesses
1. Compared with other general-purpose models, JUDO lags in binary anomaly detection tasks. JUDO’s score (64.51%) in simple anomaly-vs-normal classification tasks is lower than that of general-purpose VLMs such as Kimi-VL (72.93%). Although the paper claims this can be attributed to the advanced vision encoder of Kimi-VL, this could also suggest that the model’s complex reasoning optimization may not translate well to simpler binary tasks.

2. Reasoning mode not systematically evaluated: The paper argues that free-form CoT can hurt MMAD accuracy, thus opting for a constrained `<seg>/<think>/<answer>`  format with reward shaping. However, the trade-offs among no reasoning, free CoT, and structured reasoning are not empirically studied. A controlled comparison would clarify why structured reasoning helps and when CoT hurts, strengthening claims.

3. The paper does acknowledge that sequential training across three stages can cause performance degradation in earlier-learned tasks. Although addressed to some extent, this issue remains a known weakness of multi-phase fine-tuning and may affect scalability to more diverse industrial domains.

### Questions
1. Could the authors provide an ablation comparing different reasoning styles – (a) direct answer (no reasoning), (b) free-form Chain-of-Thought, and the proposed structured `<seg>/<think>/<answer>` reasoning – to quantify the trade-offs between interpretability and task accuracy?

2. JUDO underperforms compared to general-purpose models in binary anomaly discrimination. Could the authors clarify whether it is due to catastrophic forgetting, visual encoder limitations, or reinforcement objectives prioritizing reasoning quality over detection precision?

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
3

### Summary
1. The paper presents JUDO (Juxtaposed Domain-Oriented Multi-modal Reasoner), a framework designed to enhance industrial anomaly detection by integrating domain-specific knowledge and contextual reasoning into large multimodal models (LMMs).
2. JUDO performs visual comparative reasoning by juxtaposing defect and normal images for fine-grained inspection, and injects domain expertise through supervised fine-tuning followed by reinforcement learning (GRPO) with tailored rewards.
3. Experiments on the MMAD benchmark show that JUDO outperforms strong baselines such as Qwen2.5-VL-7B and GPT-4o, highlighting the value of domain-oriented reasoning for anomaly understanding.

### Strengths
1. Novel Multistage Training Framework:
The paper introduces a well-structured two-stage pipeline combining supervised fine-tuning (SFT) and reinforcement learning (GRPO), effectively aligning domain-specific knowledge with multimodal reasoning objectives.

2. Conceptual Contribution in Knowledge Integration:
It provides a meaningful conceptual advancement in how to inject out-of-domain or domain-specific knowledge into large multimodal language models, enabling more accurate and interpretable reasoning for industrial anomaly understanding.

### Weaknesses
1. Clarity and Presentation:
The paper’s presentation is somewhat difficult to follow, particularly in the implementation and dataset construction sections.
For example, in Lines 216–224, it would be helpful to include a concrete example comparing a QA pair derived directly from the domain snippet versus one generated for general inspection knowledge, to clearly illustrate their differences.

2. Incomplete Baseline Comparison:
The experimental section lacks evaluation against recent state-of-the-art vision–language models (VLMs) such as Gemini 2.5 Pro, which would provide a stronger and more convincing benchmark for JUDO’s effectiveness.

### Questions
1. Dataset Scale and Composition:
How many QA pairs were generated for the “general inspection knowledge” subset? It would be useful to know the relative proportion of these conceptual QA pairs compared to those derived directly from domain snippets in MMAD.

2. Use of External Knowledge in Closed-Source Models:
Given that open-source VLMs still lag slightly behind but demonstrate strong reasoning capabilities, could the proposed “general inspection knowledge” QA corpus be used as external prompt knowledge for closed-source models such as GPT-4o, GPT-5, or Gemini 2.5 Pro?
How would this prompt-based knowledge injection compare to the performance of the proposed fine-tuned JUDO model in terms of reasoning accuracy and interpretability?

### Soundness
3

### Presentation
2

### Contribution
3
