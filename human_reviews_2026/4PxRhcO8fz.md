# Simple yet Effective Semi-supervised Knowledge Distillation from Vision-Language Models via Dual-Head Optimization

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 2, 8

## Abstract
Semi-supervised learning (SSL) has emerged as a practical solution for addressing data scarcity challenges by leveraging unlabeled data. Recently, vision-language models (VLMs), pre-trained on massive image-text pairs, have demonstrated remarkable zero-/few-shot performance that often surpasses SSL approaches due to their exceptional generalization capabilities. This gap motivates us to question: how can we effectively harness the powerful generalization capabilities of VLMs into task-specific models? Knowledge distillation (KD) offers a natural framework for transferring VLM capabilities, but we identify that it suffers from gradient conflicts between supervised and distillation losses. To address this challenge, we propose Dual-Head Optimization (DHO), which introduces dual prediction heads for each distinct signal. We observe that DHO resolves gradient conflicts, enabling improved feature learning compared to single-head KD baselines, with practical benefits of minimal computational overhead and test-time hyperparameter tuning without retraining. Extensive experiments across 15 datasets show that DHO consistently outperforms KD baselines, often outperforming teacher models with smaller student models. DHO also achieves new state-of-the-art performance on both in-distribution ImageNet semi-supervised learning and out-of-distribution generalization across ImageNet variants. We will publicly release our code and model checkpoints to facilitate future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses how to transfer the strong generalization ability of vision-language models (VLMs) to task-specific semi-supervised learning (SSL). The authors identify gradient conflicts between supervised and distillation losses as a key limitation of existing knowledge distillation (KD) approaches. To overcome this, they propose Dual-Head Optimization (DHO), which uses two separate prediction heads to decouple these gradients. DHO allows flexible post-training adjustment and achieves state-of-the-art results on 15 datasets, outperforming KD baselines and sometimes even the teacher models.

### Strengths
1. Clear and easy-to-follow writing: The paper is well organized, with a logical flow from motivation to methodology and results. The problem is clearly stated, and the proposed solution is introduced in a concise and coherent manner, making the paper easy to read and understand.

2. Comprehensive theoretical analysis: The authors provide solid theoretical reasoning for the proposed Dual-Head Optimization framework. The analysis of gradient conflicts between supervised and distillation losses is thorough and well-motivated, clearly explaining why the dual-head design can mitigate this issue and improve feature learning.

3. Extensive experimental validation: The paper conducts experiments covering both in-distribution and out-of-distribution settings. The results demonstrate significant improvements over existing baselines.

### Weaknesses
1. The proposed Dual-Head Optimization framework appears overly simple and provides limited new insights for the research community. The idea of using different heads for different losses is not novel and has already been explored in prior works such as which author mentioned SSKD and DHKD. Although the authors claim that "they do not target distillation from foundation models or combine predictions at inference", this distinction seems marginal and insufficient to establish substantial novelty.

2. The DHO-F variant leverages a stronger teacher model compared to baselines, which raises concerns about the fairness of the comparisons.

3. As described in Lines 316–318, the Alpha and Beta parameters must be manually tuned for different tasks, which reduces the practical applicability of the method. Furthermore, the hyperparameter search shown in Figure 9 makes the comparisons less fair, since improper choices of these parameters can significantly degrade performance, even falling below the baseline in some cases.

### Questions
The experiments indicate that training with the KD loss alone already provides a very strong baseline. This raises the question of whether a simple two-stage procedure: first training with KD, then fine-tuning with an established SSL method, could achieve comparable or even better results. The paper would benefit from an ablation or comparison along these lines to clarify whether the proposed DHO framework truly provides additional advantages beyond existing sequential approaches.

### Soundness
3

### Presentation
4

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
This paper proposes DHO (Dual-Head Optimization), to improve the effectiveness of knowledge transfer from large Vision-Language Models (VLMs) to smaller student models in semi-supervised settings. The core idea is to mitigate gradient conflicts between the supervised loss (on labeled data) and the distillation loss (from the teacher VLM) by using two separate classification heads for each objective. The distilled model indeed achieve achieves state-of-the-art results on multiple benchmarks, primarily for visual recognition, and sometimes even surpassing the teacher model. However, the assumption used for method design seems too strong and more clarification is needed.

### Strengths
- Clear paper writing and in-depth analysis on the gradients
- Promising performance and consistent gain via extensive experiments.

### Weaknesses
- The claim of gradient conflicts is still empirical. While the derivations and visualizations are helpful, a more formal theoretical analysis is still needed. Also, why you consider cosine similarity as the metric to determine the conflicts. How about dot-product, or just the sign value of gradients themselves.
- The method seems to be too simple. Just splitting two objectives into two heads? Then, how can you ensure the gradients that reaches the deepest layer of the shared backbone is always free of conflict. 
- Inconsistent comparison in Table 1: 1) Why use two different network architecture in “training from scratch” and “self-supervision”. 2) For the self-supervision, you still assume that you have enough of the training data, right? Regardless of whether they are labeled or not.
- Experiment assumption: For “training from scratch”, the assumption is not strong but performance is still below than the teacher model baseline. For “self-supervision”, despite better performance, what is the concrete application scenario that you can perform self-supervision first and then perform you DHO.
- The compared methods are too limited, while many of them are pretty old.

### Questions
Please see my comments in weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper identifies a gradient conflicts issue for semi-supervised knowledge distillation for VLM models, and it proposes Dual-Head Optimization(DHO), a simple yet effiective method, to address the issue. Different from traditional single head Knowledge Distillation(SHO), DHO uses two independent prediction heads, one for supervised learning on labeled dataset, and one for Knowledge Distillation. Their method does not increase the computing resources too much. DHO jointly leverages labeled samples and probabilistic output of the teacher model, showing the gradient conflicts are mitigated empirically. Besides that the author also proposed a Dual-Head Interpolation(a linear combination) method to tune the two dual head output weights at inference time. 
The method has been proved effective under the semi-supervised KD setting, and shows SOTA performance on over 15 datasets and sometimes even outperforms the zero-shot teacher models on image classification tasks. 
The author provided comprehensive math analysi to explain the gradient conflicts, though they still mainly rely on empirical findings to justify their conclusions. They also showed good math proves for the Dual-Head interpolation method to emulate SHO hyperparameter tuning without re-training.
The writing is in a good shape and easy to follow. Overall, this paper contributed a good insights on problem framing and simple yet effective solution under VLM semi-supervised KD setting.

### Strengths
1. The paper presents a clear and intuitive solution to address gradient conflicts in semi-supervised KD. It is intuitive but effective to use separate head for supervised and distillation. It is also not significantly increasing the overhead(<10% parameter size increase, and negligible FLOPs change) 
2. The paper contributed a Dual Head interpolation method that supports post-training hyperparameter tuning. It also provides the prove process that it can be used for emulation of SHO hyperparameter tuning.
3. The paper provided theoretical reasoning with empirical evidence to clearly illustrate gradient conflicts in semi-supervised KD setting.
4. Effectiveness is well proved on over 15 datasets and achieved SOTA accuracy.

### Weaknesses
The paper provided a theoretical discussion of the gradient conflicts, but the discussion heavily rely on empirical findings. Also the theoretical discussion of why DHO resolves the gradient conflicts are also based on empirical results. It would be better to provide more insights on the reasoning.

The paper mainly limited the discussion and experiments on the VLM semi-supervised KD on classification tasks. It could be better to explore detection or segmentation tasks for the next step.

### Questions
Is the gradient conflicts only observed in this scenario or a common issue for general tasks?

### Soundness
4

### Presentation
4

### Contribution
3
