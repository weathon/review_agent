# Pose-RFT: Aligning MLLMs for 3D Pose Generation via Hybrid Action Reinforcement Fine-Tuning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Generating 3D human poses from multimodal inputs such as text or images requires models to capture both rich semantic and spatial correspondences. While pose-specific multimodal large language models (MLLMs) have shown promise, their supervised fine-tuning (SFT) paradigm struggles to resolve the task's inherent ambiguity. Its reliance on objectives like SMPL parameter regression creates a critical alignment gap, compromising the model's ability to achieve the required semantic and spatial fidelity. To close the gap, we propose Pose-RFT, a framework that shifts the learning paradigm from supervised imitation to reward-driven reinforcement fine-tuning (RFT). We address the core technical challenge of this task: a 
hybrid action space requiring joint optimization of discrete language and continuous pose outputs. To this end, we introduce HyGRPO, a hybrid reinforcement learning algorithm that enables stable optimization by performing group-wise reward normalization over sampled responses. Pose-RFT incorporates task-specific reward functions to guide optimization towards spatial alignment in image-to-pose generation and semantic consistency in text-to-pose generation.
Extensive experiments on multiple pose generation benchmarks demonstrate that Pose-RFT significantly improves performance over existing pose-specific MLLMs, validating the effectiveness of our approach in closing the alignment gap for 3D pose generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the alignment gap in text/image-to-pose generation, where the one-to-many nature of 3D poses makes supervised fine-tuning (SFT) suboptimal. The authors propose Pose-RFT, framing the task as hybrid-action reinforcement learning and introducing HyGRPO for joint optimization over discrete (text) and continuous (pose) spaces. HyGRPO normalizes group-wise rewards and decomposes the advantage into discrete (language) and continuous (pose) components. Four task-specific reward functions are designed: spatial, semantic, format, and text-similarity.

### Strengths
*Originality:* The motivation is clear and well-grounded. Applying reinforcement fine-tuning to 3D pose generation is novel and well-justified given the one-to-many mapping nature.

*Quality:* Extensive experiments across multiple datasets (3DPW, H36M, PoseScript) demonstrate SOTA performance against MLLM baselines. Ablations are comprehensive.

*Clarity:* Writing and figures are clear and intuitive.

*Significance:* Introducing RFT into multimodal 3D pose generation is timely and impactful for the community.

### Weaknesses
1. **Lack of ablation on reward components.**
   Although the paper emphasizes the importance of the four reward functions, it provides no analysis isolating their individual contributions. Section 4.3 only examines Pose-aware Encoder, distribution modeling, and RFT as a whole. Hence, it remains unclear which reward(s) primarily drive the improvements—semantic alignment, spatial accuracy, or the mere presence of RL signal.

2. **Reward–evaluation entanglement.**
    Some reward terms use the same embedding models as the evaluation metrics (e.g., BGE-M3, PoseScript retriever), creating potential bias or “reward hacking.” Cross-evaluator validation would strengthen claims of generality.

3. **Computational overhead.**
   HyGRPO requires sampling multiple candidates and performing group normalization. While the appendix mentions resource cost, there are no efficiency statistics or scaling curves quantifying the trade-off.

### Questions
1. Have the authors conducted reward ablations? If not, please clarify which components are most influential.
2. Could you evaluate under a different embedding/retrieval model to test generalization?
3. How do key hyperparameters (group size G, continuous sample count V) affect performance, efficiency, and stability?

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
3

### Summary
This paper aims to address the alignment gap problem in current 3D human pose generation methods. The authors point out that existing methods primarily rely on Supervised Fine-Tuning, learning a deterministic mapping from input to a single ground truth pose through regression and other methods. However, 3D pose generation tasks (whether text-to-pose or image-to-pose) inherently possess a "one-to-many" ambiguity, which the SFT paradigm struggles to handle. 

To address this issue, the paper proposes the Pose-RFT framework, using reinforcement learning fine-tuning. The core technical challenge lies in handling a hybrid action space, requiring simultaneous optimization of discrete text outputs and continuous 3D pose parameters. To this end, the authors design a novel online reinforcement learning algorithm called **HyGRPO** (Hybrid Action Space Group Relative Policy Optimization). This algorithm achieves stable optimization of the hybrid policy by grouping and normalizing the rewards of a sampled set of candidate outputs.

Furthermore, the paper designs a set of task-specific reward functions to guide spatial alignment for image-to-pose tasks and semantic consistency for text-to-pose tasks, respectively. Extensive experiments on multiple 3D pose generation benchmarks demonstrate that Pose-RFT achieves significant performance improvements over existing pose-specific MLLMs.

### Strengths
- In the context of MLLM, the authors define pattern collapse in generative models as "alignment gap”. Furthermore, the authors demonstrate why a shift from supervised imitation to a reward-driven optimization paradigm is necessary.  
- The authors demonstrate that RL can not only be used to align a model's internal understanding, but also to directly guide the model to generate complex, structured non-textual data.  
- The authors propose the HyGRPO algorithm to provide independent and targeted feedback signals for different output heads of the model, which is a simple and logically consistent approach.

### Weaknesses
- Pose-RFT achieved PA-MPJPE values of 44.5 mm and 85.9 mm on Human 3.6M and 3DPW, respectively. The paper mentions a "performance gap" compared to traditional expert models, but quickly shifts its focus to its state-of-the-art (SOTA) performance on the RPE task. However, this performance gap is not insignificant, but rather quite substantial. Current SOTA expert models already have PA-MPJPE values far below this value on Human 3.6M, for example, 29.1 in ([CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Learning_Analytical_Posterior_Probability_for_Human_Mesh_Recovery_CVPR_2023_paper.html)) and 29.4 in ([Springer link](https://link.springer.com/chapter/10.1007/978-3-031-72640-8_27)). This may be due to differences in assessment protocols, which the authors need to clarify, but it undoubtedly underestimates the true gap. The authors should consider repositioning the paper's contribution as "bringing new interactive and inference capabilities to pose estimation," rather than merely competing on traditional metrics.  

- In the text-to-pose task, the authors used a pre-trained text-pose retrieval model to define the semantic alignment reward. However, in the evaluation phase, they also used the same retrieval model (derived from the PoseScript paper) to calculate the Recall@K metric. This design leads to a methodological loop: the RFT process directly optimizes the model to generate poses that score high on the similarity metric of the retrieval model, and then uses the same metric to evaluate its performance. This can lead to overfitting to the evaluation metric, where the generated poses may not appear more reasonable or accurate to human observers, but are simply better at matching the feature space of the evaluation tool. This is a serious limitation that undermines the credibility of the results for the text-to-pose generation task.  

- HyGRPO appears to be an adaptation of the GRPO algorithm in a hybrid action space. While this application is novel, its core algorithmic components likely have stronger precedents in the broader hybrid RL literature than the paper acknowledges. The paper should more clearly compare its algorithm with these existing works to clarify its exact novelty.  

- The paper proposes four independent reward functions, but does not discuss the weight balance and potential conflicts among them.

### Questions
1. On standard benchmarks such as Human3.6M and 3DPW, your approach exhibits a significant performance gap compared to domain expert models. Could you elaborate on the reasons for this gap and articulate it as a fundamental trade-off between using a general-purpose MLLM architecture and a specialized architecture?

2. Evaluation of text-to-gesture tasks relies on the same retrieval model that provides the semantic reward signal, which can lead to a "test-taking" problem. How do you view this potential methodological limitation? Have you considered employing alternative evaluation strategies?

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
3

### Summary
Pose-RFT introduces a RL fine-tuning framework for MLLM-based 3D pose generation. This framework, HyGRPO, utilizes a mixed optimization strategy to optimize both discrete tokens (from the LLM) and continuous motion (SMPL params).

### Strengths
Enabling LLMs to estimate / generate poses is a highly relevant task and utilizing a RL-based framework to optimize this is a reasonable strategy.

### Weaknesses
My major concern is that one of the main premises of this work, that Pose-RFT is required for better semantic and spatial alignment of HMR, is misleading. The statement that the “reliance on objectives like SMPL parameter regression creates a … alignment gap, … [to] … achieve the required semantic and spatial fidelity” (l016-019) is inaccurate or unclear.
Arguably, the simplified assumptions about the context (3d world, pose, camera) are the reason for “poor” performance HMR. On one hand, generative models like Score-HMR [1] have been utilized successfully and the authors should discuss this line of work. On the other hand, more accurate modeling of the context, i.e. like has been done in CameraHMR [2] solves the problem and simply egressing the SMPL parameters actually outperforms all other methods.

One of the main contributions of the paper is HyGRPO which is an optimization algorithm to optimize both discrete and continuous outputs. This is necessary as the authors note themselves, because MLLMs operate in discrete space but poses are in continuous space. However, there is a body of work that successfully learns discrete tokenization for human poses, for example TokenHMR. I wonder if the authors could leverage the TokenHMR tokenizer instead of having to jointly optimize discrete and cont’ signals - at least as a baseline.

The authors should also evaluate on EMDB [3], which provides more accurate gt poses.


[1] ScoreHMR: Score-Guided Diffusion for 3D Human Recovery; CVPR 2024

[2] CameraHMR: Aligning People with Perspective; 3DV 2025

[3] EMDB: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild; ICCV 2023

### Questions
How well does this method do on videos? It would be super interesting to see how stable the model predictions would be in a temporal setting.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Pose-RFT, a reinforcement fine-tuning (RFT) framework for 3D human pose generation from multimodal inputs such as text and images. This work aims to overcome a fundamental limitation in existing pose-specific MLLMs, which is the semantic–spatial alignment gap that arises from their reliance on supervised fine-tuning (SFT) with regression-based objectives.  
Instead of directly regressing SMPL parameters via supervised learning, Pose-RFT reformulates the task as a hybrid action space reinforcement learning problem, where the model simultaneously generates discrete language tokens and continuous 3D pose parameters. To achieve stable and efficient optimization in this mixed action space, they introduce HyGRPO, a hybrid reinforcement learning algorithm that applies group-wise reward normalization across sampled responses. Furthermore, the framework incorporates four task-specific reward functions targeting spatial accuracy (for image-to-pose), semantic alignment (for text-to-pose), output format correctness, and embedding-level consistency.
Experiments are conducted on 3DPW, Human3.6M, and RPE for evaluating 3D HPS; on PoseScript for test-topose generation, demonstrating promising performance gains compared with prior SFT-based pose-specific MLLMs, validating the effectiveness of the reinforcement fine-tuning paradigm in achieving both semantic and spatial fidelity in 3D pose generation.

### Strengths
1. The paper makes a clear paradigm shift from supervised regression to reward-based reinforcement optimization for multimodal 3D pose generation, which is a timely direction. 
2. They adopt HyGRPO to addresses the challenge of jointly optimizing discrete and continuous outputs, which seems to be a reasonable and effective solution. .  
3. The integration of task-specific rewards provides a well-structured mechanism to guide policy learning toward both geometric and semantic alignment. 
4. Experimental results show solid quantitative improvements over multiple baselines, reinforcing the proposed method’s effectiveness and generality. The ablation studies of HyGRPO show its positive effects on the performance.

### Weaknesses
1. Limited qualitative results. Only a few images in Figure 5 and Figure 6 presented in this paper. Providing more visualizations would help readers gain a more intuitive understanding of the performance of the proposed model. 
2. Reward sensitivity. Since multiple reward terms are combined, it would be helpful to include a discussion or visualization of how different reward weights affect learning dynamics. 
3. Computational overhead. Reinforcement fine-tuning can be computationally expensive; providing efficiency comparisons or scalability analysis would strengthen the practical relevance.

### Questions
1. How efficient are the model's inference and training (compared to the supervised training)? This information helps readers understand its advantages and disadvantages.
2. How well the model understands complex textual expressions—is this one of the factors limiting the model's performance?

### Soundness
3

### Presentation
3

### Contribution
3
