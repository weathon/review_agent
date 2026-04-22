# FARTrack: Fast Autoregressive Visual Tracking with High Performance

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Inference speed and tracking performance are two critical evaluation metrics in the field of visual tracking. However, high-performance trackers often suffer from slow processing speeds, making them impractical for deployment on resource-constrained devices. To alleviate this issue, we propose $\textbf{FARTrack}$, a $\textbf{F}$ast $\textbf{A}$uto-$\textbf{R}$egressive $\textbf{T}$racking framework. Since autoregression emphasizes the temporal nature of the trajectory sequence, it can maintain high performance while achieving efficient execution across various devices. FARTrack introduces $\textbf{Task-Specific Self-Distillation}$ and $\textbf{Inter-frame Autoregressive Sparsification}$, designed from the perspectives of $\textbf{shallow-yet-accurate distillation}$ and $\textbf{redundant-to-essential token optimization}$, respectively. Task-Specific Self-Distillation achieves model compression by distilling task-specific tokens layer by layer, enhancing the model's inference speed while avoiding suboptimal manual teacher-student layer pairs assignments. Meanwhile, Inter-frame Autoregressive Sparsification sequentially condenses multiple templates, avoiding additional runtime overhead while learning a temporally-global optimal sparsification strategy. FARTrack demonstrates outstanding speed and competitive performance. It delivers an AO of 70.6\% on GOT-10k in real-time. Beyond, our fastest model achieves a speed of 343 FPS on the GPU and 121 FPS on the CPU. Source code is available at: https://github.com/MIV-XJTU/FARTrack.git

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper aims to make the Autoregressive Visual Tracking method more lightweight for deployment. It presents two key technical contributions: the first is self-distillation for multiple templates, and the second is token sparsification for the tokens being processed. The self-distillation module learns trajectory features layer by layer, enabling lightweight compression and avoiding the manual layer matching required in conventional cross-layer distillation. Meanwhile, the inter-frame sparsification module filters redundant background information from the template frames at the sequence level and propagates the sparse results in an autoregressive manner, thereby avoiding additional computational overhead. The experimental results are robust and comprehensive. The comparisons to the efficient trackers show that the method is effective and achieves the SOTA results in three tracking benchmarks. The author also presents many visualization results and ablation studies to show the effectiveness of the method.

### Strengths
1.FARTrack compresses the backbone via: 1) task-specific self-distillation method: each layer teaches the one below using the trajectory-token stream, 2) inter-frame autoregressive sparsification. 
2.The experimental results are promising, demonstrating strong speed and accuracy in the efficient trackers. 
3.The motivation is good. The autoregressive tracking methods need specific lightweight methods, especially considering the temporal domain. 
4.The method is straightforward and exceptionally easy to understand. It has the potential to be applied to more autocratic autoregressive model pipelines.

### Weaknesses
1.The writing could be improved. For example, the terminology used in the paper is not precise enough. The word "performance" tends to describe the efficiency of the model, and using "accuracy" would more accurately depict the tracking effect of the model.
2.Too few models are compared on the VastTrack benchmark, and more models should be added for comparison.
3.It is suggested to supplement the comparison of training costs between cross-layer distillation and task-specific self-distillation to more fully demonstrate the advantages of the proposed method.

### Questions
1.In Figure 2, why is the ratio of the model’s speed to its performance used to represent the circle’s size instead of FLOPs?
2.Is there any basis for the setting of the number of templates in Table 2?
3.What is the basis for the REMOVE_LAYERS settings [0, 3, 6, 9, 12] and [0, 2, 4, 6] in the Deep-to-Shallow distillation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a fast autoregressive visual tracking framework called FARTrack, aiming to achieve both high accuracy and real-time speed. Its main innovations include Task-Specific Self-Distillation and Inter-Frame Autoregressive Sparsification. In experiments, FARTrack achieved 70.6% AO performance on the GOT-10k dataset, reaching 135 FPS on GPU and maintaining 121 FPS on CPU. It significantly outperforms several state-of-the-art trackers, including AsymTrack, MixFormerV2, and HiT, in terms of speed and performance balance.

### Strengths
1.The paper is well written.

2.Improve the efficiency of tracker is a good direction in visual object tracking, which makes progress in this feild.

3.Use Inter-frame Autoregressive Sparsification sounds new.

### Weaknesses
1.Self-Distillation is not new. The concept has been proposed by diffrenet field.

2.The comparison methods are insufficient — most baselines are from earlier years, with no inclusion of 2024 works and only one paper from 2025. The paper should include more recent state-of-the-art methods for a fair and comprehensive comparison.

3.No MACs and Parameters analysis with sota methods.

### Questions
Can you provide more sota methods resutls? And compare the performance and complexity.

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
3

### Summary
The paper proposes FARTrack, a fast tracking framework combining Task-Specific Self-Distillation and Inter-frame Autoregressive Sparsification to improve inference speed while maintaining tracking performance. The method builds on ARTrack and achieves competitive results with 135 FPS on GPU (tiny) and 343 FPS (pico).

### Strengths
- **Strong empirical results** FARTracktiny achieves 70.6% AO on GOT-10k at 135 FPS (GPU), outperforming closest competitors while maintaining comparable speed. Multi-platform evaluation (GPU/CPU/NPU) demonstrates practical applicability.
- **Comprehensive ablation studies** Thorough analysis of distillation strategies, sparsification methods and design choices provide insights on the impact of distillation and other architectural choices. 
- **Efficient Distillation Strategy** Avoids manual layer assignment: The layer-by-layer self-distillation shows consistent improvements over cross-layer distillation

### Weaknesses
- **Incremental technical contribution** The core novelty is primarily an engineering combination of ARTrack with existing techniques (self-distillation and attention-based sparsification). While the application is competent and results are solid, the conceptual advance over applying standard acceleration techniques to ARTrack is limited. The paper would benefit from clearer articulation of what makes this combination non-trivial beyond implementation.
- **Missing baseline experimental comparisons** No comparison with shallow models trained from scratch (only distilled models), which would clarify whether distillation provides benefits beyond simply reducing layers.

### Questions
- What causes the LaSOT performance gap? Lower performance on long-term object tracking make sense but beyond "model capacity loss," can you provide deeper analysis (e.g., attention pattern changes, temporal modeling degradation) of why distillation specifically hurts long-term tracking more than short-term?
- How does a 10-layer or 6-layer model trained from scratch compare to your distilled versions? This would clarify whether distillation provides benefits beyond simply using fewer layers.
- How does the method generalize to other challenging scenarios like NFS (fast motion) or UAV123?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents FARTrack, a lightweight autoregressive visual tracking framework that combines task-specific self-distillation and inter-frame autoregressive sparsification to improve efficiency while maintaining accuracy. The method achieves a balance between speed and performance across different hardware platforms.

### Strengths
- The paper is well presented, with clear motivation, solid organization, and consistent writing quality.
- FARTrack achieves impressive speed–accuracy trade-offs across GPU, CPU, and NPU platforms.
- The design is simple and potentially applicable to other lightweight tracking pipelines.

### Weaknesses
- Compared with prior autoregressive tracking baselines or other light-weight trackers, the training set now includes VastTrack and LaSOT Ext, which significantly expand data diversity and size. Since VastTrack has been shown to enhance generalization and robustness, using it for training raises fairness concerns in comparing FARTrack with earlier methods that were trained on smaller datasets.
- The visualization in Figure 5 suggests that feature representations from layers 7–14 appear highly similar, indicating limited hierarchical diversity. This calls into question the necessity and effectiveness of performing distillation across all consecutive layers.
- Incorporating trajectory tokens inherently enhances temporal modeling and tracking robustness, which has already been verified in prior autoregressive trackers (e.g., ARTrack, ARTrackV2). The paper does not clearly disentangle this known benefit from the gains attributed to the proposed self-distillation or sparsification modules, thereby weakening the originality of the contribution.

### Questions
Pls see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
