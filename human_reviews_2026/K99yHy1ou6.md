# POMP: A Theoretical Approach to Mitigate Forgetting in Finetuning Multi-Modal Models

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Catastrophic forgetting is a major challenge when adapting pretrained models to new tasks in multi-modal contrastive learning (MMCL). We provide a theoretical analysis of finetuning by introducing a *contrastive target matrix* that reformulates the linearized contrastive objective as a matrix least-squares problem. This formulation yields closed-form solutions for direct finetuning, weight-space regularization, and self-distillation, providing a geometric interpretation of how each strategy manages pretrained knowledge. Our analysis reveals that self-distillation preserves knowledge in the subspace orthogonal to the finetuning data while forming a convex combination of the pretrained and new solutions within the task subspace. We extend this analysis to a dynamic self-distillation framework with a weighted moving average (WMA) teacher. We prove that, unlike standard Exponential Moving Average (EMA) teachers which eventually collapse onto the student, the WMA teacher maintains a persistent, non-vanishing regularizing force throughout training by integrating the full optimization trajectory. These theoretical insights motivate our method, **POMP** (Preserve-Orthogonal-Mix-Parallel), which operationalizes this framework. POMP uses a composite distillation loss guided by the WMA teacher to achieve state-of-the-art out-of-distribution robustness and calibration when finetuning CLIP.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper reformulate finetuning as a matrix least-squares problem using a contrastive target matrix, enabling a geometric interpretation of how different strategies (direct finetuning, L2 regularization, self-distillation) affect pretrained knowledge. This paper introduces POMP which maintains a persistent regularizing force and bias-free convergence. Empirical results show that POMP achieves state-of-the-art robustness, calibration, and OOD generalization across multiple architectures, offering both theoretical insight and practical improvement in finetuning large multimodal models.

### Strengths
1. The paper presents a novel theoretical reformulation of multimodal finetuning as a matrix least-squares problem, introducing a contrastive target matrix that provides an insightful geometric interpretation of catastrophic forgetting.
2. The POMP method effectively translates these theoretical findings into practice, combining contrastive learning with dynamic self-distillation via a Weighted Moving Average (WMA) teacher.
3. Experiments on both synthetic and large-scale multimodal datasets demonstrate strong empirical support, showing reduced forgetting, improved OOD robustness, and better calibration.

### Weaknesses
1. The theoretical analysis is based on a linearized model, and its applicability to deep non-linear networks remains is doubtful.
2. The selection and sensitivity of the hyperparameter $\lambda_{SD}$ are insufficiently discussed, leaving ambiguity about practical tuning and stability.

### Questions
1. The theoretical analysis appears to be derived under a linearized model assumption. Could the authors clarify how these results generalize to deep, non-linear multimodal architectures such as CLIP?
2. The paper introduces $\lambda_{SD}$ as a key coefficient balancing contrastive finetuning and self-distillation. Could the authors provide more details on how this parameter is selected, its sensitivity to different datasets, and whether there are guidelines or heuristics for practical tuning?
3. Since CaRot also addresses forgetting via regularization and distillation mechanisms, could the authors elaborate more clearly on how POMP differs conceptually and algorithmically from CaRot?

### Soundness
3

### Presentation
2

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
- The paper is well written and has all the necessary details and the execution is solid. I need some clarification on the contribution here, because, once you get to equation one, the rest seems to be trivial analysis and the experimental results also seems to be marginal.  What novel insight is being proposed in this paper, how are these insights useful?

### Strengths
- Theoretical analysis seems sound
- Good Experimental analysis as well.

### Weaknesses
- **Novelty and contribution**: The paper starts with a motivation that a linearized transformation of the CLIP is available from the two other papers and then, the setup analyzes the impact of different fine tuning strategies. Due to this, the contribution becomes unclear, because, once you linearize the system, it is trivial to converge to the idea that the convex combination of the weights is the right way to balance between forgetting and generalization. As the linearization process depends on the idea that there exists a locally compact set with a one minima, over which the linearization is done. So, I am unclear on what is new on that front, if the linearization is pulled from another set of papers.
- The notion of EMA and WMA and the idea that EMA collapsed still seems obvious because of the presence of an exponential function that converges because of exponentially decaying past values.
- With all that, the improvement is always less than 2-4 % as in table 3 and 4 with no standard deviation for any understanding of how the method varies

### Questions
The main question, i have is with respect to the novelty of the paper as discussed in the weakness section.

### Soundness
4

### Presentation
4

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
This paper addresses catastrophic forgetting in multi-modal contrastive learning (MMCL) when finetuning pretrained models. The authors reformulate the linearized contrastive objective as a matrix least-squares problem, enabling closed-form analysis of finetuning, regularization, and self-distillation. They show that self-distillation preserves pretrained knowledge in the orthogonal subspace while blending new and old tasks in the shared subspace. Extending this, they propose a weighted moving average (WMA) teacher that maintains a persistent regularization effect, unlike standard EMA. Building on these insights, they introduce POMP (Preserve-Orthogonal-Mix-Parallel), which achieves state-of-the-art robustness and calibration in finetuning CLIP.

### Strengths
1. The proposed method is novel and intuitive.
2. The paper is well-written with a clear structure and detailed explanations.

### Weaknesses
1. The experiments are mainly taken on image tasks and ViT models, could you extend to larger model and other tasks, such as LLM?
2. If you prefer to claim the method as a ''theoretical approach'', maybe it is better to present the theoretical benefits of this method in mitigating forgetting, especially in over-parameterization regime.
3. The structure of the proposed method is not clear, maybe you could provide an algorithm framework in section 4.
4. Could you provide an analysis or discussion on the efficiency of your method, comparing with other methods?

### Questions
Please see weakness.

### Soundness
2

### Presentation
2

### Contribution
2
