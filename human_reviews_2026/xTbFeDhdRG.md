# RAR: Reversing Visual Attention Re-Sinking for  Unlocking Potential in Multimodal Large Language Models

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Multimodal Large Language Models (MLLMs) have achieved remarkable success in vision-language tasks, yet they frequently exhibit suboptimal output layers, where intermediate decoder layers outperform the final ones, signaling underutilized model capacity. In this work, we delve into the root causes and attribute this issue to the Visual Attention Re-sinking phenomenon, precipitated by attention gradient sparsity driven by textual supervision dominance. This degradation causes attention heads to evolve into sink heads that prioritize low-semantic backgrounds, thereby disrupting modality fusion, neglecting visual information, and biasing outputs toward textual priors, ultimately impairing model performance. To mitigate this, we introduce a parameter-free Sink Attention Dynamic Sparsification (SADS) framework that dynamically identifies and retains all vision heads(concentrating visual attention on semantically relevant regions) while sparsifying sink heads, preserving essential global context through a shared head. Integrated into diverse MLLMs, our framework yields substantial performance gains across 20 benchmarks spanning five task categories (visual grounding, general VQA, OCR-related VQA, vision-centric tasks, and visual hallucination mitigation) surpassing supervised fine-tuning while boosting inference speed by 10.3\%. This approach offers a novel avenue for maximizing MLLMs capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents RAR: a parameter-free method to address the problem of re-sinks in the final layers of a vision-language model, that cause the model to underperform due to undesired attentions. Unhealthy attention heads are removed, which improves the performance.

### Strengths
The observations about the problems in the final layers, the degradation of performance because of that, the illustrations of spurious visual attention, are all interesting. The finding that visual attention is bi-modal and separates the vision heads and sink heads, is interesting too. The performance gains are structural but marginal. The proposed method is effective and simple, which makes the approach feasible for a broader audience.

### Weaknesses
The illustrations are sparse (e.g. Figures 3c, 4a and 8a) and sufficient to convey the main idea, but they are redundant and simplistic, as a consequence a better understanding of the separation into visual and attention heads is not provided. More importantly, the distinction between useful and unhealthy sink heads is not illustrated, and details are lacking to understand this better. 

The whole idea of the paper is to remove the relatively small subset of those unhealthy sink heads. A better understanding of that subset, which ones they are, why they hurt the performance; this is fundamental for a paper that builds on those principles.

### Questions
Which subset of the attention heads is unhealthy and why is that, why does that hurt the performance of the last layers? Can you provide insights with some illustrations, and with some statistics, e.g. are they mostly on particular areas in the image, or semantic regions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focus on the problem that the intermediate decoder layers outperform the final ones in MLLMs, and attribute this issue to the Visual Attention Re-sinking phenomenon. Then, a parameter-free Sink Attention Dynamic Sparsification framework is proposed. The proposed SADS achieves superior effectiveness and inference efficiency on several benchmarks.

### Strengths
1. It deeply identifies text-only supervision as the cause of suboptimal MLLM output layers.
2. The SADS framework effectively addresses key issues to optimize output layers.
3. Comprehensive experiments across 20 benchmarks validate its superiority.

### Weaknesses
1. From Figure 2 and Table 1, it can be observed that the performance degradation caused by the re-sinking phenomenon is limited. It can also be seen from the experiments that the improvement brought by the proposed method is limited.
2. To demonstrate its effectiveness and  generalizability, more models of different sizes (7B, 13B, ...) should be tested.
3. An ablation experiment on the proportion of sink heads should be added; additionally, is it reasonable to use a fixed proportion?

### Questions
Refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the root cause of the visual attention resinking phenomenon, and shows that this phenomenon is the reason why output layers of MLLM yield worse performance than middle layers, a frequently observed phenomenon in existing work. The paper finds that the visual attention resinking phenomena is caused by attention gradient sparsity, which makes the gradient distribution over attention heads sparse hence causing sink tokens to concentrate in sink heads. Based on these findings, the paper proposes a "Sink Attention Dynamic Sparsification" strategy, which selects vision heads per layer and forces model to focus on vision signals. Experimental results show improvements over baselines.

### Strengths
The paper developed a systematic approach to diagnose a commonly observed but not well-understood phenomenon. The analysis is logical, insightful and convincing, revealing an overlooked aspect in MLLM training and inference.

### Weaknesses
The proposed approach can be viewed as a patch on existing models rather than a solution to the root cause of the resinking problem. It would strengthen the paper if it also proposed ways to address the issue more fundamentally (for example, through improved training objectives).

### Questions
- It seems the gradient sparsity problem gets worse as the training steps increases. If this is the case, I wonder whether the MLLM performance (from the last output layer) would be better at an earlier checkpoint?
- In Fig 5a, how is the gradient sparsity precisely defined? 
- The framework retains the top 25% of sink heads, is this ratio empirically optimized, and how sensitive is the model performance to this threshold?
- The tested models used are up to only 7B parameters. Would the same sparsification principle hold in larger models? Due to increased capacity of larger models, it's conceivable that the resinking problem would be alleviated.
- Would it be possible to address the gradient sparsity problem with better training objectives (e.g. adding regularization terms to upweight vision heads), which address the problem at a more fundamental level?

### Soundness
3

### Presentation
3

### Contribution
3
