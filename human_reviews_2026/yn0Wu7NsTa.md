# ReactID: Synchronizing Realistic Actions and Identity in Personalized Video Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Personalized video generation faces a fundamental trade-off between identity consistency and action realism: overly rigid identity preservation often leads to unnatural motion, while emphasis on action dynamics can compromise subject fidelity. This tension stems from three interrelated challenges: imprecise subject-video alignment, unstable training due to varying sample difficulties, and inadequate modeling of fine-grained actions. To address this, we propose ReactID, a comprehensive framework that harmonizes identity accuracy and motion naturalness through coordinated advances in data, training, and action modeling. First, we construct ReactID-Data, a large-scale dataset annotated with a high-precision pipeline combining vision-based entity label extraction, MLLM-based subject detection, and post-verification to ensure reliable subject-video correspondence. Second, we analyze learning difficulty along dimensions such as subject size, appearance similarity, and sampling strategy, and devise a progressive training curriculum that evolves from easy to hard samples, ensuring stable convergence while avoiding identity overfitting and copy-paste artifacts. Third, ReactID introduces a novel timeline-based conditioning mechanism that supplements monolithic text prompts with structured multi-action sequences. Each sub-action is annotated with precise timestamps and descriptions, and integrated into the diffusion model via two novel components: subject-aware cross-attention module to bind sub-action to the specific subject of interest and temporally-adaptive RoPE to embed the rescaled temporal coordinates invariant to action duration. Experiments show that ReactID achieves state-of-the-art performance in both identity preservation and action realism, effectively balancing the two objectives.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents ReactID, a comprehensive framework designed to address the fundamental trade-off between identity consistency and action realism in personalized video generation. The authors identify three core challenges—imprecise subject-video alignment, unstable training dynamics, and inadequate modeling of fine-grained actions—and propose a holistic solution. The main contributions are threefold: (1) the creation of ReactID-Data, a large-scale, high-precision dataset with detailed temporal annotations; (2) a novel Difficulty-Aware Curriculum Learning strategy to mitigate the "copy-paste" artifact and ensure stable convergence; and (3) a structured, timeline-based conditioning mechanism with two novel components (Subject-Aware Cross-Attention and Temporally-Adaptive RoPE) for fine-grained action control. The paper demonstrates through extensive experiments that ReactID achieves state-of-the-art performance, effectively balancing identity preservation with natural motion generation. It is a solid piece of work with significant practical contributions to the field.

### Strengths
1.  **Clear Motivation and Problem Formulation:** The paper begins with a clear and insightful analysis of the core challenges in personalized video generation. It methodically breaks down the problem into three specific bottlenecks. The proposed methods directly and logically address each of these points, creating a coherent and well-motivated narrative.

2.  **High-Quality Data Pipeline and Dataset:** The construction of the ReactID-Data dataset is a major strength. The data processing pipeline is thorough, robust, and well-reasoned, combining vision-based entity extraction, MLLM-based detection, and post-verification. The decision to annotate videos with structured action timelines and fine-grained textual descriptions is particularly valuable, addressing a critical need for temporally precise data in the field.

3.  **Effective Curriculum Learning Strategy:** The proposed Difficulty-Aware Curriculum Learning is an elegant and well-thought-out solution to the common "copy-paste" problem, where models overfit to easy samples at the expense of motion realism. By formalizing sample difficulty along three intuitive axes (subject size, appearance similarity, and sampling strategy) and progressively introducing harder samples, the method effectively guides the model toward a more generalized solution. This claim is convincingly supported by a thorough ablation study (Table 5), which demonstrates the value of each component and the full curriculum.

4.  **Novel and Controllable Timeline Conditioning:** The timeline-based conditioning mechanism is a significant step forward for controllable video generation. Decomposing monolithic prompts into timestamped sub-actions gives users granular control over complex action sequences. The architectural innovations—Subject-Aware Cross-Attention to bind actions to specific subjects and Temporally-Adaptive RoPE to handle variable action durations—are technically sound and well-justified.

5.  **State-of-the-Art Performance:** The experimental results are comprehensive, demonstrating superior performance against numerous baselines across multiple benchmarks (OpenS2V-Eval and the newly proposed ReactID-Eval-SEQ). The qualitative results (Figure 1, Figure 4) further underscore the model's ability to generate fluid, natural movements while faithfully preserving the subject's identity, a feat many prior methods struggle with.

### Weaknesses
1.  **Lack of Ablation on Data Annotation Quality:** While the paper shows that a model trained on ReactID-Data outperforms one trained on OpenS2V-5M (Table 6), it lacks a direct ablation study on the impact of the fine-grained timeline and textual annotations. It would be beneficial to see how the ReactID model itself performs when trained on the same video data but with coarser captions or no timeline promtps, which would more directly quantify the benefit of the expensive and detailed annotation pipeline.

### Questions
None

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
1

### Summary
This paper addresses a challenge in personalized video generation: balancing identity consistency (preserving a subject’s unique features across frames) and action realism (generating natural, dynamic motions). The authors identify three root causes of this trade-off—imprecise subject-video alignment, unstable training due to varying sample difficulty, and coarse-grained action modeling—and propose ReactID, a holistic framework to mitigate them.

### Strengths
1. The problem of identity-action balance is central to personalized video generation, and ReactID’s solutions (dataset, curriculum, temporal modeling) advance the field beyond incremental improvements. The dataset release also fosters collaboration.
2. ReactID’s holistic approach (data + training + modeling) is innovative. For example, the subject-aware cross-attention’s label binding mechanism (assigning unique labels to subjects) solves a longstanding issue of action-subject misalignment in multi-subject scenarios.
3. Ablation studies validate each component.

### Weaknesses
1. The paper does not report training/inference time compared to baselines. For real-world use (e.g., edge devices), efficiency is critical—readers cannot assess if ReactID’s performance gains come at the cost of slower runtime.
2. ReactID uses LLMs to convert single prompts to timelines, but does not evaluate how LLM errors (e.g., incorrect action order, wrong timestamps) affect generation quality. This is a practical limitation, as LLMs are not perfect.

### Questions
1. How does ReactID perform on scenarios with 3+ subjects (e.g., a family of 4 performing distinct actions)? Does the label binding mechanism in subject-aware cross-attention scale efficiently, or does it introduce misalignment/overhead with more subjects?
2. Can you provide training time per step (on a standard GPU, e.g., A100) and inference time per 5-second video, compared to baselines ? This would help readers assess ReactID’s practicality.
3. ReactID focuses on 5-second videos—can it generate longer videos (e.g., 30 seconds) while maintaining identity-action consistency? If not, what are the main barriers (e.g., temporal drift)?

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
This paper targets personalized video generation and argues that current methods face a persistent tension between identity fidelity and realistic action. The authors propose ReactID, a framework that couples a new data pipeline, a difficulty-aware curriculum, and a timeline-based conditioning mechanism. ReactID-Data is built with entity extraction for living and non-living subjects, MLLM-assisted subject detection and verification, and subject masks for faces, aiming to deliver precise subject–video correspondences. The training strategy scores sample difficulty using subject size, appearance similarity, and sampling strategy, and then increases a difficulty threshold over time. The model introduces a subject-aware cross-attention with label binding and a temporally-adaptive RoPE that rescales time within sub-actions so temporal bias aligns with action boundaries. Experiments on OpenS2V-Eval and a new ReactID-Eval-SEQ show consistent gains on aggregated metrics and ablations indicate each component contributes. The paper also describes an LLM planner that converts a single prompt into a timeline at inference.

### Strengths
1. The paper is well scoped around balancing identity fidelity and action realism, and it articulates the root causes spanning data noise, training instability, and coarse action modeling.

2. The data pipeline is thoughtfully designed, separating living and non-living entities, grounding with MLLMs, and adding face-centric masks, which is important for identity preservation.

3. The timeline-based conditioning is technically interesting: subject-aware cross-attention with label binding plus temporally-adaptive RoPE to normalize sub-action durations targets a known failure mode at action boundaries.

### Weaknesses
1. The timeline annotations are produced automatically by a VLM at scale, yet the paper does not quantify timestamp accuracy or action-boundary noise, which could systematically bias training and evaluation for sequence control. A small human audit or agreement study would help.

2. Subject masks are supervised by SAM-style masks. Failure modes such as strong occlusion, rapid camera motion, or multi-subject overlap are not analyzed, although these are common in personalized scenarios.

3. Curriculum details are under-specified. The schedule for the difficulty threshold, the mixing ratio of intra-clip and inter-clip references over time, and the sensitivity to the λ weights are not provided, limiting reproducibility beyond the high-level recipe.

### Questions
1. How accurate are the VLM-derived timelines, both in absolute timestamps and boundary placement between sub-actions. Please report a small human study with boundary F1 and inter-annotator agreement.

2. Please clarify the compute budget, number of GPUs, and wall-clock time for the reported 10k-step training, and whether the method remains effective when scaled up or down.

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
5

### Summary
This paper tackles the central trade-off in personalized video generation between identity consistency and action realism. The authors propose ReactID, a framework that coordinates advances in data curation, training strategy, and action conditioning to balance identity fidelity and motion naturalness. They introduce ReactID-Data, a large-scale dataset built via a high-precision pipeline. To stabilize learning, they analyze difficulty along axes such as subject size, appearance similarity, and sampling, and adopt a progressive curriculum from easy to hard to mitigate identity overfitting and copy-paste artifacts. For action modeling, they propose a timeline-based conditioning scheme that augments text prompts with structured multi-action sequences annotated with timestamps, integrated through two components: a subject-aware cross-attention module and a temporally-adaptive RoPE mechanism. Experiments report state-of-the-art performance on both identity preservation and action realism, suggesting the method effectively balances these competing objectives.

### Strengths
1. The manuscript is clearly written and easy to follow.
2. The motivation is well defined, pinpointing three key challenges in video customization: inaccurate identity preservation, unstable convergence, and compromised action naturalness.
3. The work is solid, introducing a new dataset and method components, including Subject-Aware Cross-Attention and Temporally-Adaptive RoPE.
4. Extensive experiments convincingly demonstrate the effectiveness of the approach.

### Weaknesses
1. Will the dataset, code, and model weights be released? Open-sourcing would significantly benefit community research and reproducibility.
2. How well does the method generalize to multi-subject customization beyond two subjects (e.g., three or more)?
3. For multi-subject scenarios, can the approach handle more complex, independent action timelines per subject (e.g., multiple subjects performing distinct actions concurrently)?
4. The results shown in the paper are about human subjects. Can the method support customizing animals and general objects?
5. Please consider citing closely related work:
    - CustomCrafter: Customized Video Generation with Preserving Motion and Concept Composition Abilities
    - ReVersion: Diffusion-Based Relation Inversion from Images
    - DreamRelation: Relation-Centric Video Customization

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
