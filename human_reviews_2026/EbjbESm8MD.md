# Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Visual Prompt Tuning (VPT) has proven effective for parameter-efficient adaptation of pre-trained vision models to downstream tasks by inserting task-specific learnable prompt tokens. Despite its empirical success, a comprehensive theoretical understanding of VPT remains an active area of research. Building on the recently established connection between Mixture of Experts (MoE) and prompt-based methods, wherein each attention head can be conceptualized as a composition of multiple MoE models, we reinterpret VPT as the introduction of new *prompt experts* into these MoE structures. We identify a key limitation in existing VPT frameworks: the *restricted functional expressiveness* of prompt experts, which remain static and thus limited in their adaptability. To address this, we propose **Visual Adaptive Prompt Tuning (VAPT)**, a novel method that endows prompt experts with enhanced expressiveness while preserving parameter efficiency. Empirical evaluations on VTAB-1K and FGVC demonstrate that VAPT achieves *substantial performance improvements*, surpassing fully fine-tuned baselines by **7.34%** and **1.04%**, respectively. Moreover, VAPT consistently outperforms VPT while *requiring fewer additional parameters*. Furthermore, our theoretical analysis indicates that VAPT achieves optimal sample efficiency. Collectively, these results underscore the theoretical grounding and empirical advantages of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper reinterprets VPT as the introduction of new prompt experts into these MoE structures, solving the current limitation in existing VPT frameworks: the restricted functional expressiveness of prompt experts.

### Strengths
1. The experiment settings are sound with sufficient numbers of ablation studies in the Appendix. 
2. The paper is easy to follow, and the motivation for introducing MoE to prompt tuning is reasonable.

### Weaknesses
1. The baselines provided in this paper are not new. More recent prompt tuning and other PEFT methods [1-4] should be included for completeness. 

2. A critical problem of this paper is its novelty; [5] has proposed MoE prompt tuning as a manifold mapper, indicating that MoE design on prompt tuning can bring stronger expressivity. This work is highly related to the proposed research, although it has not been discussed. 

3. Inconsistent experimental report. Table 1 includes E2VPT, while Table 2 does not. Similar for GateVPT.

4. The claim on the low-data regime in the introduction does not have further showcases. Also, the noticeable performance gap might be an outline for VPT. The reason is, as shown in [6], VPT generally brings good few-shot performance, when the training repeats for several times (to avoid bad samples for the training). 

[1] Visual Fourier Prompt Tuning

[2] Visual instance-aware prompt tuning

[3] Apla: A simple adaptation method for vision transformers

[4] DS2VP: Dynamically-Selected Spatially Visual Prompting

[5] MEPT: Mixture of Expert Prompt Tuning as a Manifold Mapper

[6] Facing the Elephant in the Room: Visual Prompt Tuning or Full Finetuning?

### Questions
The major concern is the novelty and some claims in this paper. The most relevant paper on MoE prompt tuning is not discussed in this paper. Also, the paper sounds like the direct integration of MoE. Even without a similar approach, the novelty itself is questionable. The authors do not clearly separate their method from traditional MoE attempts.

Another problem is that some claims might be misleading, though the authors formulate the MoE and the proposed method's training objective; the core idea is intuitive and simple. I think some equations are unnecessary and further complex the understanding of the basic concept of this paper.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits Visual Prompt Tuning through the lens of Mixture of Experts and identifies a fundamental limitation: conventional prompt tokens are static and input-invariant, thus lacking expressive power. To address this, the authors propose Visual Adaptive Prompt Tuning, which generates input-dependent adaptive prompt experts using lightweight token-wise projectors, channel-wise convolutions, and a shared feature projector. This design enhances the functional expressiveness of prompts while maintaining parameter efficiency. Theoretically, VAPT achieves optimal sample efficiency under the MoE framework, and empirically, it outperforms both fully fine-tuned and VPT baselines across VTAB-1K and FGVC benchmarks.

### Strengths
1. The paper offers a clear theoretical reinterpretation of VPT through the lens of MoE, providing both conceptual insight and mathematical grounding for understanding prompt tuning behavior.

2. The proposed Visual Adaptive Prompt Tuning (VAPT) effectively enhances the expressiveness of prompt experts by introducing input-dependent adaptive prompts while maintaining parameter efficiency.

3. Overall writing is clear and easy to follow,

### Weaknesses
1. Conceptually, VAPT’s “input-adaptive prompt experts” is similar to prompt-pool-based approaches [R1. R2]. These methods also condition prompt selection or generation on input features. Especially, [R2] generates tokens based on visual prompts based on the input. If authors could provide comparison between proposed method and existing  prompt-pool-based approaches, it strengthens the novelty of  works.

[R1] Wang, Zifeng, et al. "Learning to prompt for continual learning." CVPR 2022.

[R2] Kim, Youngeun, et al. "Open-world dynamic prompt and continual visual representation learning." ECCV 2024.

2. The proposed approach introduces multiple small components (channel-wise conv, token-wise projector, shared MLP). While lightweight in current ViT-B/16 settings, their scalability to larger backbones (ViT-L/14, ViT-H/14) or higher-resolution inputs is not discussed. The added modules could potentially become computational bottlenecks or require additional tuning.

3. Although the paper argues that VAPT enhances functional expressiveness, there are no visual analyses (e.g., attention maps, learned prompt diversity, or feature attribution) to substantiate this claim. Qualitative results could help illustrate how adaptive prompts differ in behavior from static ones.

4. (optional) All experiments are performed on classification and segmentation benchmarks. It would strengthen the contribution to show that VAPT generalizes to non-classification visual tasks, such as detection or vision-language retrieval, especially since prompt-tuning is often used in multimodal settings.

### Questions
Please address questions in Weakness section. Thank you.

### Soundness
3

### Presentation
3

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
This paper reinterprets Visual Prompt Tuning (VPT) through the lens of Mixture of Experts (MoE) and identifies a key limitation: conventional VPT uses static, input-invariant prompt tokens, which restricts expressiveness. Building on this observation, the authors propose Visual Adaptive Prompt Tuning (VAPT)—a parameter-efficient extension that generates input-dependent prompts. The authors empirically demonstrate consistent gains over VPT and other PEFT methods across VTAB-1K and FGVC benchmarks.

### Strengths
- The main motivation of this paper can provide a mathematically grounded analysis

- The paper is easy to follow.

- The authors provide a variety of experiments, with results on FGVC, VTAB-1K, and supervised and self-supervised pretrained backbones, showing the robustness of the proposed method. In addition, ablation studies in the Appendix are very helpful to understand the proposed method.

### Weaknesses
- My major concern is the novelty.
For adaptive visual prompt tuning, there are many visual prompt tuning works (e.g., CVPT, CoCoOp, ViaPT, V2APT) already exploring visually adaptive or instance-aware prompts. Hence, the contribution in adaptivity itself is incremental rather than fundamentally new. In addition, MoE Interpretation is also heavily motivated by Le et al, who already framed attention and prompting under MoE theory. 


[CVPT] CVPT: Cross-Attention help Visual Prompt Tuning adapt visual task, NeurIPS 2024

[CoCoOp] Conditional Prompt Learning for Vision-Language Models, CVPR 2022

[ViaPT] Visual Instance-aware Prompt Tuning, MM 2025

[V2APT] Visual Variational Autoencoder Prompt Tuning, arXiv 2025


- The paper lacks sufficient comparison with other recent variants of Visual Prompt Tuning. There has been a surge of follow-up works for VPT, yet these are not adequately discussed or compared. Especially, input-dependent or adaptive prompting methods should be carefully compared and discussed.

- There are some fairness issues, where the Tuned/Total ratio differs substantially across methods. For instance, Gated VPT (in Table 2) uses only about 0.05 % of trainable parameters, while VAPT tunes 0.27 % – 0.28 % of the model. Such discrepancy can partly explain the performance gap, making the comparison less fair. A more rigorous evaluation should control for the number of trainable parameters.

### Questions
- Can the proposed visual adaptive prompt method be applied on top of recent VPT variants?

### Soundness
3

### Presentation
3

### Contribution
2
