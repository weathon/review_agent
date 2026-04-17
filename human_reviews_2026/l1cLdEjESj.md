# Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction–Reasoning Synergy

- Decision: Accept (Oral)
- Scores: 6, 6, 8

## Abstract
Recent developments in Multimodal Large Language Models (MLLMs) have significantly improved Vision–Language (VL) reasoning in 2D domains. However, extending these capabilities to 3D scene understanding remains a major challenge. Existing 3D Multimodal Large Language Models (3D-MLLMs) often depend on 3D data inputs, which limits scalability and generalization. To address this limitation, we propose Vid-LLM, a video-based 3D-MLLM that directly processes video inputs without requiring external 3D data, making it practical for real-world deployment. In our method, the geometric prior are directly used to improve the performance of the sceen perception. To integrate the geometric cues into the MLLM compactly, we design a Cross-Task Adapter (CTA) module to align the 3D geometric priors with the vision-language representations. To ensure geometric consistency and integrity, we introduce a Metric Depth Model that recovers real-scale geometry from the reconstruction outputs. Finally, the model is fine-tuned with a two-stage distillation optimization strategy, realizing fast convergence and stabilizes training. Extensive experiments across diverse benchmarks verified the effectiveness of our method on 3D Question Answering, 3D Dense Captioning and  3D Visual Grounding tasks,  demonstrating the superior multi-task capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Vid-LLM, a video-based Multimodal Large Language Model for 3D scene understanding that works directly from video, removing the need for external 3D data. The method leverages geometric priors and incorporates them efficiently using a Cross-Task Adapter (CTA) to align 3D geometry with vision-language features. A Metric Depth Model ensures accurate real-scale geometry, and a two-stage distillation strategy improves training. Experiments show Vid-LLM excels at 3D Question Answering, Dense Captioning, and Visual Grounding, outperforming existing methods in multi-task 3D reasoning.

### Strengths
1. The integration of 3D geometric information into 3D LLMs is well-motivated in this paper.

2. The paper is clearly written and easy to follow.

3. The experiments effectively demonstrate the improvements brought by the proposed modules.

4. It is encouraging to see that the proposed method achieves competitive performance on both pose estimation and depth estimation tasks.

### Weaknesses
1. Integrating 3D foundation models into 3D MLLMs is not a novel concept. Recent works such as VG-LLM [1] and 3DRS [2], which also utilize VGGT to enhance the 3D-awareness of 3D MLLMs, are not cited or compared in this paper. Including a discussion or direct comparison with these approaches would strengthen the related work section and better contextualize the contributions of this work.

2. In Tables 7 and 8, it is unclear how the proposed model's performance compares to that of the teacher model, VGGT. Providing this comparison would improve the clarity and completeness of the evaluation.

References:

[1] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors. NeurIPS 2025.

[2] MLLMs Need 3D-Aware Representation Supervision for Scene Understanding. NeurIPS 2025.

### Questions
1. Please incorporate comparison with recent works.

2. Please add performance comparison with VGGT.

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
This submission proposes Vid-LLM, a video-based 3D multimodal large language model that performs both 3D reconstruction and vision-language reasoning directly from monocular video inputs. The main contributions of this submission are

1. Cross-Task Adapter (CTA) that aligns geometric priors with vision-language features, enabling joint geometry–semantic interaction.
2. Metric Depth Model (MD) that restores real-scale 3D geometry with bin-based depth prediction and adaptive scale alignment.
3. Two-Stage Training Strategy with dual-teacher distillation and joint optimization to improve convergence and stability.

The authors conduct extensive evaluations on 3D-QA, dense captioning, and grounding benchmarks showing consistent performance gains over 3D- and video-based baselines.

### Strengths
- The paper is well written and structured. Overall it easy to follow the technical narrative and experiments.
- The related work section provides a comprehensive overview of 3D multimodal large language models and video-based reasoning approaches.
- The overall design, including the Cross-Task Adapter (CTA), Metric Depth (MD) module, and the two-stage training strategy, is well motivated and technically sound. The integration of geometric and semantic cues is conceptually coherent and justified.
- The ablation studies are thorough, demonstrating the necessity of each proposed component to the final performance.
- The experimental results show consistent and significant improvements across multiple 3D vision-language reasoning benchmarks.

### Weaknesses
- Missing details and clarifications:
    - In Figure 2, the predicted 3D reconstruction results are used for generating 3D position embeddings. It would be helpful to clarify whether gradients are propagated through this path when jointly training the reconstruction and reasoning modules.
    - In Section 3.4, Equation (6), the notation $\text{Norm}(\cdot)$ appears ambiguous. Please clarify whether it denotes feature normalization (e.g., L2-normalization) or the 2-norm operation itself. An explicit definition would avoid confusion.
    - In Section 3.4, Equation (6), what is the dimensionality of $T_{\text{tea}}^{\text{lang}}$? Is the feature-level loss$L_{\text{feat}}^{\text{lang}}$ averaged over N samples similar to $L_{\text{feat}}^{\text{geo}}$? Clarifying this would help in understanding the implementation consistency.
- Simplification of the bridge token mechanism:
    
    The current Cross-Task Adapter introduces bridge tokens for cross-modal interaction. Would a simpler alternative—such as applying self-attention directly on the concatenation of $ T_{\text{geom}} $ and $ T_{\text{lang}} $—achieve similar performance with lower complexity? Some empirical or ablation analysis could strengthen the design justification.
    
- Qualitative analysis:
    
    More qualitative results and failure case discussions would be valuable to understand when the model fails. The authors are suggested to include them in the camera-ready version.

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Vid-LLM, a compact multimodal model that unifies 3D reconstruction and vision-language reasoning within a single framework. I really like this idea -- combining reconstruction with VL reasoning not only bridges geometry and semantics but also achieves faster inference through a more efficient architecture. The work is conceptually elegant and technically well-motivated.

### Strengths
- The model design is well thought out, especially the Cross-Task Adapter (CTA) that enables effective geometry–semantics interaction.

- The overall training pipeline, including two-stage distillation and joint optimization, is clear and coherent.

- The experimental setup is solid - reconstruction and 3D-VL baselines are comprehensive, and the ablation studies are extensive enough to verify the core hypothesis and support the narrative.

- Results demonstrate consistent gains in both reasoning and reconstruction efficiency, validating the proposed synergy between the two tasks.

### Weaknesses
The main baseline (VGGT + LLaVA-3D) appears to involve two independently fine-tuned components that are only concatenated during inference, rather than being jointly optimized end-to-end. This makes the comparison to the proposed method somewhat unfair, as Vid-LLM benefits from full end-to-end training while the baselines do not. It would strengthen the paper to clarify this distinction and, if possible, include a jointly optimized baseline to better isolate the benefit of the Cross-Task Adapter.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
