# RestoRect: Degraded Image Restoration via Latent Rectified Flow & Feature Distillation

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Current approaches for restoration of degraded images face a critical trade-off: high-performance models are too slow for practical use, while fast models produce poor results. Knowledge distillation transfers teacher knowledge to students, but existing static feature matching methods cannot capture how modern transformer architectures dynamically generate features. We propose 'RestoRect', a novel Latent Rectified Flow Feature Distillation method for restoring degraded images. We apply rectified flow to reformulate feature distillation as a generative process where students learn to synthesize teacher-quality features through learnable trajectories in latent space. Our framework combines Retinex theory for physics-based decomposition with learnable anisotropic diffusion constraints, and trigonometric color space polarization. We introduce a Feature Layer Extraction loss for robust knowledge transfer between different network architectures through cross-normalized transformer feature alignment with percentile-based outlier detection. RestoRect achieves better training stability, and faster convergence and inference while preserving restoration quality. We demonstrate superior results across 15 image restoration datasets, covering 4 tasks, on 8 metrics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes RestoRect, a novel framework that reformulates knowledge distillation as a generative rectified flow process. Instead of static feature matching, the student learns to synthesize teacher features through dynamic latent trajectories. The method also introduces FLEX loss (cross-normalized feature alignment) and integrates physical priors (Retinex, diffusion, HVI color). Experiments on 15 datasets show consistent gains over state-of-the-art models with fewer inference steps and better perceptual quality.

### Strengths
Novel concept: The paper introduces a fresh perspective by reframing knowledge distillation as a generative rectified flow process. This idea is original, conceptually sound, and well-motivated by recent advances in flow-based modeling.

Strong empirical results: RestoRect consistently outperforms prior approaches across multiple datasets and metrics, while requiring significantly fewer inference steps — demonstrating both effectiveness and efficiency.

### Weaknesses
Limited theoretical grounding: The connection between rectified flow and knowledge distillation is mostly intuitive; a more formal theoretical treatment would strengthen the paper.

Ablation depth: While ablations are provided, they do not fully disentangle the effects of each component (Rectified Flow, FLEX loss, Retinex prior).

Efficiency analysis: The paper lacks concrete comparisons in terms of FLOPs, parameter counts, or runtime benchmarks to substantiate efficiency claims.

### Questions
see the weaknesses.

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
3

### Summary
This paper focuses on image enhancement, particularly tasks involving color or lighting restoration, such as low-light enhancement. It uses heterogeneous knowledge distillation to improve the efficiency of generative restoration models. However, the paper lacks a thorough comparison of efficiency metrics (e.g., inference time and FLOPs). Furthermore, most of the compared methods are not generative restoration methods, which raises questions about the reasonableness and fairness of the experimental comparisons.

### Strengths
This paper presents a variety of experiments, which fully demonstrate the effectiveness of the proposed method, although its efficiency remains questionable.

### Weaknesses
I have some questions about the technical details and novelty of this paper.

1. Spatial Channel Layer Normalization (SCLN) uses a flattened spatial-channel dimension to calculate the mean and variance during normalization. Does this lead to significant computational overhead? Current LayerNorm has a well-developed fused CUDA kernel; would changing its computational logic affect the acceleration on CUDA? Furthermore, ablation experiments with SCLN seem to lack numerical metrics. I hope to see a significant improvement in model performance in terms of fidelity or aesthetics after using SCLN.
2. This paper introduces the HVI color space as an auxiliary constraint. What is the fundamental difference between this and HVI-CIDNet? I also noticed the use of LPIPS-VGG for training the teacher model. Would the absence of this constraint lead to a performance decrease?
3. The core of the FLEX proposed in this paper is to utilize cross-norm to address feature distribution mismatch, which seems to highly overlap with [1].
4. Is the distillation method proposed in this paper superior to other heterogeneous distillation algorithms [2]?
5. Use \citep{} instead of \cite{}.

[1] CrossNorm and SelfNorm for Generalization Under Distribution Shifts. ICCV 2021.

[2] Heterogeneous Knowledge Distillation using Information Flow Modeling. CVPR 2020.

### Questions
Many of the tasks listed in the paper are related to lighting. Could some re-lighting models be used to restore lighting and thus solve these image enhancement tasks?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors proposed a teacher student framework for the task of non-blind image restoration with paired ground truths for training. Task specific restoration networks are trained for different restoration objectives. The authors claimed SOTA results in almost all tasks and datasets. The authors claimed novelty in using latent rectified flows for knowledge transfer and novel normalization and a new feature loss.

### Strengths
The authors put in significant effort to test their proposed framework over several datasets and restoration tasks. However, much is left desired in presentation quality.

### Weaknesses
- Writing and presentation is poor and unclear in general. Some pointers and suggestions are here to help the authors improve the article for future submission. The article is difficult to read and hard to follow.   
- Major issue: *Retinex is NOT physics based\!* It is a study of the human, biological, color vision system. It is OK to draw inspiration from retinex, by using this hypothesised code as presented in Land 1977, but this should not be called physics based. This should be referred to as biologically inspired.  
- Do not simply list a huge bunch of slightly related papers in the related works section. Instead, selectively choose key background material, and briefly cover the key ideas that are of note to the proposed method. For example, in the work, the use of retinex inspired information coding seems to be an important part of the architecture design, explain what it is and what it’s hypothesized effect is in the proposed architecture. Is the use new? Or are there prior works that also use this biologically inspired coding scheme? This also sets the stage for subsequent ablation studies, showcasing the importance of ideas, either completely novel, novel use/application, or novel interpretation.   
- Figure 2 is extremely unclear. I’m unable to understand what each component is and how they relate to each other. There’s a rather involved training workflow as I understood from repeated re-reading, this can be presented in a clear manner as a flow char of sorts. Reused components in each stage can also be marked out and referred to.   
- There are far too many acronyms this makes reading the results extremely difficult  
- Make better use of table and figure captions\! They should have more substantial description in them, at least include a key insight from the data in the table or the images shows. For example: Fig 4\. Qualitative results on for low-light enhancement on 4 datasets. Why are there boxes in the images? What do you want the readers to pay attention to in these boxes?

### Questions
* Since this is a teacher-student framework, how well does the teacher perform?  
* For each task, is the teacher network the same?  
* How is the teacher network selected? Is the teacher network a well performing network from a previous work?  
* How is the student network selected? And again, is the design based on any previous work?  
* Is the same architecture used for all restoration tasks?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies degraded image restoration through knowledge distillation framed as a generative rectified-flow process from teacher to student. A teacher transformer is trained with physics-inspired auxiliaries (e.g., Retinex, anisotropic diffusion, polarized HVI color) and attention stabilizers (SCLN, QK-norm). The student learns velocity fields along straight-line feature paths and uses a FLEX loss combining cross-normalization, percentile masking, and resolution weighting. The claimed benefits are few-step inference and competitive quality across multiple restoration tasks.

### Strengths
1. A cohesive rectified-flow perspective on feature-level distillation that is implementable with simple Euler updates and clear path parameterization.
2. A concrete, reproducible teacher recipe that mixes classical priors with modern transformers and documents key loss weights and stabilization tricks.
3. A two-stage student training pipeline that is logically structured and easy to follow.
4. The FLEX loss addresses teacher–student distribution mismatch in a targeted way via student-statistic normalization and percentile outlier handling.
5. The few-step inference angle is practical for deployment scenarios where latency is constrained.

### Weaknesses
1. The teacher pipeline appears to use ground-truth signals during pretraining or feature extraction, risking supervision leakage and motivating an LQ-only teacher-target variant.
2. Student-side ablations are shallow because the effects of rectified flow, FLEX, trajectory consistency, and step-size scheduling are not disentangled and there is no sensitivity to percentile, SNR, or step count.
3. Positioning against modern transformer-aware KD remains largely narrative without controlled head-to-head experiments on attention, token, or logit distribution alignment.
4. Evidence across datasets and metrics is uneven, with limited discussion of fidelity–perceptual trade-offs and missing analysis of clear failure cases.
5. The proposed method needs to be tested on other image restoration tasks (such as super-resolution and denoising).

### Questions
1. Are any KD targets derived from inputs containing ground truth (e.g., concatenated [LQ, GT] or GT-based Retinex)? If so, what is the performance when targets are computed from LQ-only inputs?
2. Can you add per-component ablations: rectified flow on/off, FLEX on/off, and within FLEX: cross-normalization only vs. +percentile vs. +resolution weighting, with sensitivity to the percentile threshold and SNR/step-size settings?
3. How does the method compare to attention-/token-level transformer KD under identical student backbones and training budgets?
4. Can you expand results across all stated restoration categories with both fidelity (PSNR/SSIM) and perceptual (LPIPS/CLIP-IQA/FID) metrics, and discuss cases where metrics disagree?
5. Will you release code, training/evaluation scripts, and the exact list of selected inference steps and hyperparameters to ensure full reproducibility?

### Soundness
4

### Presentation
4

### Contribution
3
