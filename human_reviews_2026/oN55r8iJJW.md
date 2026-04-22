# SAM-Veteran: An MLLM-Based Human-like SAM Agent for Reasoning Segmentation

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Significant progress has been made in reasoning segmentation by combining multi-modal large language models (MLLMs) with the Segment Anything Model (SAM): the former excel in reasoning and vision–language alignment, while the latter offers powerful pixel-level understanding. However, current paradigms fall short in exploiting SAM’s strengths, especially the ability to support iterative mask refinement by interactive segmentation, a process that human users can naturally perform. To bridge this gap, we introduce  **SAM-Veteran**, an experienced mask-aware SAM agent capable of emulating human interaction with SAM via a reasoning-driven segmentation workflow that integrates (i) generating bounding boxes given image–query pairs for SAM input, (ii) proposing refinement points based on SAM-generated masks, and (iii) adaptively terminating the process. Aiming for this goal, we propose a multi-task reinforcement learning framework based on Group Relative Policy Optimization (GRPO), which enhances the MLLM’s abilities in textual grounding and mask comprehension. Furthermore, we introduce a dynamic sampling strategy tailored for generating both boxes and points to stabilize training. Extensive experiments across diverse datasets show that SAM-Veteran achieves human-like interaction with SAM and establishes new state-of-the-art performance on both in-domain and out-of-domain benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces SAM-Veteran, a reasoning-driven segmentation framework that enables human-like interaction with the Segment Anything Model (SAM). The method integrates a multi-modal large language model (MLLM) with SAM to perform iterative segmentation through three key steps: generating bounding boxes from image–query pairs, proposing refinement points based on SAM-generated masks, and adaptively terminating the process. To train the SAM-Veteran, diverse reward functions are utilized. Extensive experiments demonstrate that SAM-Veteran achieves state-of-the-art performance across both in-domain and out-of-domain datasets, effectively leveraging SAM’s interactive segmentation strengths.

### Strengths
- The paper presents the first unified framework that integrates bounding box generation, iterative mask refinement, and adaptive termination into a single reasoning-driven segmentation process.
- It introduces a well-designed multi-task reinforcement learning framework to effectively train the SAM-Veteran workflow, enhancing both textual grounding and mask comprehension.
- The proposed method achieves state-of-the-art performance on both in-domain and out-of-domain datasets, demonstrating strong generalization and robustness.

### Weaknesses
- Recently, in the reasoning segmentation task, several datasets beyond ReasonSeg have been proposed. (MUSE [1] for multi-target cases and MMR [2] for part-level reasoning). It would be interesting to see whether SAM-Veteran demonstrates good generalization performance across these diverse reasoning segmentation scenarios.

[1] Ren, Zhongwei, et al. "Pixellm: Pixel reasoning with large multimodal model." CVPR  2024.

[2] Jang, Donggon, et al. "Mmr: A large-scale benchmark dataset for multi-target and multi-granularity reasoning segmentation." ICLR 2025.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
4

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
This paper proposes SAM-Veteran, an MLLM-based agent that mimics human interaction with Segment Anything Model, enabling a complete “box generation to iterative refinement to adaptive termination” reasoning segmentation workflow. Built on a multi-task RL framework using Group Relative Policy Optimization, the method enhances textual grounding and mask comprehension, achieving new state-of-the-art performance and strong cross-domain generalisation.

### Strengths
1. Comprehensive experiments. The authors conduct extensive evaluations across multiple in-domain and out-of-domain datasets, providing solid empirical evidence for their claims.

2. Clear visualisations. The workflow and results are illustrated with intuitive and well-structured figures, making it easy to understand the model’s behaviour.

3. Well-organised and easy to follow. The paper is logically structured, with each component of the method introduced and justified clearly.

4. Strong motivation and validation. The motivation is well grounded, and the proposed design is effectively validated through both quantitative results and qualitative analysis.

### Weaknesses
1. Missing related works. Some relevant literature, such as recent works on task-generic promptable segmentation[1][2] and Grounding SAM[3], is not discussed or compared in the related work section. This weakens the contextual positioning of the contribution.

2. Insufficient hyperparameter analysis. Many components involve manually set hyperparameters (e.g., the threshold of R_{iou}^B=
0.4), but the paper does not explain how these values were chosen or their sensitivity.

3. Potential training complexity and stability issues . The framework integrates multiple tasks simultaneously, which may lead to excessive complexity. The paper does not provide sufficient analysis on training stability or convergence behaviour.

[1] Hu, Jian, et al. "Relax image-specific prompt requirement in sam: A single generic prompt for segmenting camouflaged objects." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 11. 2024.

[2] Tang, Lv, et al. "Chain of visual perception: Harnessing multimodal large language models for zero-shot camouflaged object detection." Proceedings of the 32nd ACM international conference on multimedia. 2024.

[3] Ren, Tianhe, et al. "Grounded sam: Assembling open-world models for diverse visual tasks." arXiv preprint arXiv:2401.14159 (2024).

### Questions
1. Training stability. Given the multi-task RL design, how stable is training in practice? Are there signs of task interference or convergence issues?

2. Generalisation & scalability. How well does the method scale with more refinement steps or larger models, and can it generalise to broader segmentation scenarios beyond the tested datasets?

3. Hyperparameters & ablation. Many hyperparameters are manually set, but their selection process and sensitivity are unclear. Could the authors provide ablation or sensitivity analyses to justify these choices?

### Soundness
3

### Presentation
3

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
This paper presents SAM-Veteran as an MLLM-based solution for reasoning segmentation, of which the key feature is to emulate the human SAM users to refine the mask iteratively. Instead of relying on supervised fine-tuning of the MLLM, SAM-Veteran is built on top of the current RL-based training paradigms. The paper proposes to divide the iterative reasoning segmentation task into three sub-tasks and design reward models to help train the MLLM on each of those sub-tasks. The proposed method shows strong results on several reasoning segmentation benchmarks compared to prior SFT and RL-based methods.

### Strengths
The paper is well-written, well-organized, and easy to follow. In general, the proposed method shows noticeable improvement over the prior RL-based methods and SFT-based methods, and is closer to actual human interaction with the SAM-like segmentation models. The paper includes a thorough experiment for analyzing the contribution of each design choice, quantitative results, and qualitative visualizations to clearly show the strength and potential failing cases of the proposed method.

### Weaknesses
The motivation for having an auxiliary task is clear; however, the improvement is not significant, as shown in Table 4. Meanwhile, this means that there is room for improvement in task 2. A better reward design or pipeline for mask comprehension is needed. This weakness, as the paper points out, is linked to the limited ability of MLLM to understand images with masks, and adding a specific type of color mask on the visual input directly will inevitably compromise the valuable information contained in the target region.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SAM-Veteran, a multi-modal large language model (MLLM)-based agent designed to emulate human-like interaction with the Segment Anything Model (SAM) for reasoning segmentation. The key contributions include:  

- A **multi-task reinforcement learning (RL) framework** that trains the MLLM to generate bounding boxes, refine masks iteratively via points, and adaptively terminate the process.  
- A **dynamic sampling strategy** to stabilize training by diversifying box/point generation.  
- State-of-the-art performance on in-domain (RefCOCO) and out-of-domain (ReasonSeg) benchmarks, demonstrating improved generalization over existing supervised fine-tuning (SFT) and RL-based methods.

### Strengths
**Originality**: The work creatively combines MLLMs with SAM’s interactive segmentation capabilities, addressing a gap in prior RL-based methods that neglect iterative refinement. The integration of three RL tasks (grounding, mask comprehension, and auxiliary) to mimic human-SAM interaction is novel.  
**Quality**: The experiments are thorough, including ablation studies on reward design, multi-task training, and dynamic sampling. The comparison with SFT and RL baselines (e.g., Seg-Zero, SAM-R1) validates the framework’s effectiveness.  
**Clarity**: The paper is well-structured, with clear task formulations (e.g., MDP definitions) and visualizations of the workflow. The appendices provide implementation details, prompts, and failure case analysis.  
**Significance**: The work advances MLLM-based segmentation by enabling human-like SAM usage, which could inspire future research on interactive vision-language systems. The code and configuration details (Figure 6) enhance reproducibility.

### Weaknesses
**Limited comparison with SegAgent**: While SegAgent (Zhu et al., 2025b) is mentioned, the paper does not quantitatively compare SAM-Veteran against it, despite both addressing iterative refinement. This omission weakens the claim of novelty.  

**Inference time:** The paper does not provide inference time comparisons with baseline methods, which is critical for evaluating practical deployment feasibility. Additionally, detailed training resource requirements (e.g., GPU hours, memory consumption) are not explicitly reported, limiting reproducibility assessments and computational cost analysis for researchers with constrained resources.

**SAM dependency**: The framework heavily relies on SAM for mask generation and reward computation. The impact of SAM’s inherent limitations (e.g., failure modes on fine-grained objects) on SAM-Veteran’s performance is not discussed.  
**User study absence**: While the paper emphasizes "human-like" behavior, no user study evaluates whether the termination policy or refinement steps align with human preferences.

### Questions
**Q1**: How does SAM-Veteran compare to SegAgent in terms of refinement steps and termination accuracy? The authors should include a direct comparison to clarify their method’s advantages.  
**Q2**: Could the inference time and training requirement be provided? This would potentially enhance the feasibility of real-world deployment.

**Q3**: The paper states that SAM-Veteran avoids "catastrophic forgetting" seen in SFT methods. Is there empirical evidence (e.g., performance on general MLLM benchmarks) to support this claim?

### Soundness
3

### Presentation
3

### Contribution
3
