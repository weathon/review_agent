# Discrepancy-Aware Knowledge Distillation for Large Language Models

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Knowledge Distillation (KD) is a key technique for enhancing the capabilities of student models by transferring knowledge from powerful teachers. In Large Language Models (LLMs), however, the effectiveness of this transfer is fundamentally limited by distributional mismatch. The generic data used for distillation often fails to reflect the specialized distribution underpinning core expertise of the teacher. This gap hinders the acquisition of the teacher's most valuable capabilities. The challenge is fundamental because the ideal corrective method, importance weighting, is intractable without access to the unknown target density.
We propose Discrepancy Aware Knowledge Distillation (DAKD), a framework that re-frames this problem. Instead of estimating the unknown distribution, DAKD approximates the ideal importance weights by measuring the predictive discrepancy between the full teacher and a pre-trained-only base teacher, which serves as a distributional probe. The DAKD framework is "discrepancy aware" in a dual sense. It leverages the teacher-base divergence for distributional correction while using the teacher-student divergence for adaptive learning focus. This re-weighting is applied across multiple granularities, from the sequence and position down to the vocabulary level. Extensive experiments show that DAKD substantially outperforms state-of-the-art methods, enabling student models to more effectively inherit the nuanced capabilities of more powerful teachers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose DAKD, which essentially is a re-weighting of the standard KL divergence. Specifically, the author calculate the weights by considering at three levels: sequence, position, and vocabulary. The "importance" scores are calculated based on both the discrepancy between a finetuned teacher and a student, as well as between a finetuned teacher and pretrained teacher.

### Strengths
1. The proposed method show improvements over existing distillation methods.
2. The authors experimented with both instruction tuning and reasoning.

### Weaknesses
1. Higher performance could be due to hyper-parameter efforts. The authors combine the two discrepancies to produce these weights, which involves a alpha mixing hyper-parameter. As Table 3 shows, performance at 0 is pretty good, and as alpha goes to 1, half the values of alpha is better and the other half being worse.
2. Additional cost. The proposed method uses extra compute, because it requires an additional pass on the pretrained teacher model, which could be hard to justify if improvement is not significant.

### Questions
The paper states that " Ideally, the student would be optimized under the teacher’s true data distribution, p⋆... However, p⋆ is often inaccessible".

Why is this? We have the teacher's distribution and sampling from the teacher is a standard technique.

### Soundness
3

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
This paper addresses the suboptimal performance of knowledge distillation when applied to limited (and potentially suboptimal) datasets. Drawing on a similar rationale as importance sampling, the authors propose using the KL divergence between the instruction-tuned teacher model and its base version to quantify the importance of different samples/tokens/vocabulary items. Extensive empirical analysis validates the effectiveness of the proposed method.

### Strengths
- The proposed method appears intuitively sound and effectively addresses the key research problem of performing knowledge distillation on limited (and potentially suboptimal) datasets.

- The paper is well-structured and easily accessible to readers.

- Experimental results demonstrate superior performance compared to baseline methods, with the effectiveness of different components being systematically validated.

### Weaknesses
- **Potential Efficiency and Scaling Concerns**  
The proposed method requires running inference with both the instruction-tuned teacher model and the base model across the entire training dataset to compute importance weights. This additional computational overhead may be significant, particularly considering that similar computational resources could be allocated to expanding the dataset size - which might naturally address the core issue of limited and suboptimal distillation data as raised in the introduction.

- **Insufficient Experimental Validation.**  
The distinction between the instructed model and base model largely originates from the RLHF process designed to align the base model's outputs with human preferences (as acknowledged in the introduction). However, the experimental framework predominantly examines enhancements in reasoning capabilities, despite base models having already undergone extensive exposure to mathematical and reasoning-specific data during pretraining. This approach appears somewhat misaligned with the stated motivation. To more convincingly demonstrate the method's significance, we suggest supplementing the evaluation with experiments conducted under preference alignment settings.

- **Missing Baselines**  
The study demonstrates effectiveness primarily using KL divergence. Given the rich variety of divergence measures in distillation literature (e.g., α-β-divergence [1], f-divergence [2]), it would be insightful to examine whether the observed improvements persist across different divergence formulations, thus providing a more thorough understanding of the method's robustness.

[1] ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence. ICML 2025

[2] f-Divergence Minimization for Sequence-Level Knowledge Distillation. ACL 2023.

### Questions
See Weaknesses.

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
4

### Summary
Motivation of the paper: To address the problem of distribution mismatch between the high-quality knowledge in the teacher model and the knowledge learned by the student model during distillation, especially in scenarios where access to the teacher model's high-quality aligned data distribution is not available.
Method: The paper proposes a "Difference-Aware Knowledge Distillation" (DAKD) framework. Its key innovation lies in introducing a pre-trained "base version" of the teacher model as a distribution probe to measure the relevance of data to the core knowledge of the teacher.
Experiments: The paper systematically validates the effectiveness of the DAKD method in surpassing state-of-the-art approaches across various scenarios, as well as the rationality and synergy of its multi-level, dual-signal design mechanism, through comprehensive performance comparisons, data efficiency analysis, model scale evaluation, and detailed ablation studies.

### Strengths
1. This paper presents a clear and well-justified motivation, focusing on the important challenge of effective knowledge distillation from large to small models. 
2. The paper presents a simple and intuitive method, and Section 4.1 provides some theoretical justification for the proposed approach.

### Weaknesses
1. The method may has certain limitations, such as the issue of cross-tokenization in real-world scenarios. and the method may be highly dependent on having access to both the base and SFT models simultaneously.
2. The method yields limited performance gains when the knowledge sources are the same.
3. The paper presents a very comprehensive quantitative analysis, It will be better that could provide a more intuitive demonstration of the method by some qualitative examples.

### Questions
1. If an exact base model cannot be found, or we have only to different model(Qwen/LLaMA), can the DAKD method still work? If yes, what is the performance like?
2. Regarding the hyperparameter λ that controls the "difficulty" of the learning "curriculum". A fixed λ may not be optimal. Have the authors considered annealing or adaptive strategies? For example, using a small λ in the early stages of training (resulting in a smoother weight distribution and encouraging broader exploration by the student), and gradually increasing λ as training progresses (leading to sharper weights and focusing the student on harder examples). Would this lead to more stable or better performance?

### Soundness
2

### Presentation
3

### Contribution
2
