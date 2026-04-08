## Human Reviewer 1

### Summary
This paper introduces EmbodiedMAE, a unified 3D multi-modal representation learning framework for robot manipulation. The authors first create DROID-3D, a large-scale, high-quality 3D dataset by enhancing the DROID dataset with temporally consistent RGB, depth, and point cloud data. They then propose a multi-modal Masked Autoencoder (MAE) architecture that uses stochastic masking and cross-modal fusion for pre-training. The model, which is initialized with DINOv2 weights and uses a knowledge distillation strategy, consistently outperforms state-of-the-art vision foundation models (VFMs) across 70 simulation tasks (LIBERO, MetaWorld) and 20 real-world tasks on two different robot platforms (SO100 and xArm). The core contribution is a robust VFM that effectively leverages 3D spatial information for precise embodied AI tasks.

### Strengths
- The paper is exceptionally well-written and clear. The methodology is easy to follow, from the data preparation for DROID-3D to the detailed explanation of the encoder, multi-modal decoder, and distillation process.
- It provides a robust and scalable method for effectively utilizing 3D inputs, which are crucial for precise manipulation but often degrade performance in prior VFM adaptation attempts.
- The DROID-3D dataset is a valuable public resource, and the demonstration of superior performance on low-cost (SO100) and high-performance (xArm) robots proves the model's practical utility and generalization power.

### Weaknesses
- The core contribution is a unified 3D model, yet the real-world results show the Point Cloud (PC) policies (EmbodiedMAE-PC) significantly underperform the RGB-only and RGBD variants on the xArm platform. This contradicts the paper's goal of effective 3D fusion. The paper attributes this to sensor noise, but this suggests the DP3 encoder or the PC representation itself is not robust enough. A more thorough analysis or ablation comparing different PC encoders is necessary to validate the PC pipeline choice.

### Questions
- The core results rely on the RDT diffusion policy. The ACT ablation (Table 2) is currently limited to the RGB-only setting. Please extend the ACT policy ablation to include the EmbodiedMAE-RGBD and EmbodiedMAE-PC variants on the LIBERO-Goal benchmark. This is crucial to demonstrate that the superior performance of the 3D features is independent of the RDT policy type.
- The paper notes processing the complete 76K trajectories of DROID (L. 138). Did the authors test the impact of using only a subset of DROID-3D (e.g., 1/2 or 1/4 size) on the final policy learning curve? This would quantify the direct value of the sheer scale of the DROID-3D dataset, which is currently only implied.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper introduces EmbodiedMAE, a unified 3D multi-modal masked autoencoder designed to improve sensory representation for robot manipulation by jointly learning from RGB, depth, and point cloud inputs. To overcome domain gaps and low-quality 3D data in existing embodied datasets, the authors build DROID-3D, augmenting the DROID dataset with high-fidelity metric depth maps and point clouds via ZED SDK processing. EmbodiedMAE uses stochastic modality masking and cross-modal fusion to learn spatially grounded representations, and a ViT-Giant model is pre-trained then distilled into smaller variants. Evaluated across 70 simulation tasks and 20 real-world robot manipulation tasks on two platforms, the method consistently outperforms state-of-the-art vision foundation models, scales effectively with model size, and proves especially strong when leveraging 3D sensing, demonstrating improved spatial perception and manipulation precision in both simulation and real-world settings.

### Strengths
- The paper is clearly written and easy to follow, with technical concepts presented in a coherent and accessible manner.
- The experimental setup is well-structured and comprehensive, demonstrating careful design and thorough evaluation across diverse benchmarks.
- The analysis in Section 3.2 presents intriguing and insightful experimental designs that effectively validate the model’s cross-modal learning capabilities.
- The work makes a meaningful contribution to the open-source community by releasing both the DROID-3D dataset and the EmbodiedMAE model implementation.
- The results compellingly demonstrate the benefits of developing a dedicated vision foundation model for embodied AI.

### Weaknesses
- The paper does not clearly justify the choice of MAE-based pre-training over alternative paradigms such as CLIP-style contrastive learning or DINO-style self-distillation. This decision is central to the method’s novelty, yet MAE is introduced abruptly (e.g., L48) without sufficient motivation or discussion of trade-offs. A deeper explanation of why MAE is particularly suitable for embodied 3D perception — and why contrastive or language-conditioned methods may be less effective — would significantly strengthen the narrative. In addition, ablations comparing MAE to CLIP-like or DINO-like pre-training objectives would provide stronger empirical evidence for this design choice.

- Considering the submission timeline, VGGT (CVPR 2025) and other emerging 3D transformer backbones are not included as baselines. While understandable, this omission makes it difficult to assess the competitiveness of EmbodiedMAE against the latest 3D perception architectures. A discussion or retrospective comparison to VGGT-style models would help contextualize performance relative to the rapidly evolving 3D VFM landscape.

### Questions
- L209, there appears to be a formatting issue with the punctuation — specifically, a period placed before the citation: ... . (He et al., 2022). Please revise to maintain standard citation formatting.

- The paper notes that the predicted RGB images appear smoothed due to the use of L2 reconstruction loss, yet the predicted depth images do not exhibit similar smoothing artifacts. Could the authors provide further clarification on this discrepancy? In particular, why does L2-loss cause noticeable smoothing for RGB reconstruction but not for depth prediction?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents EmbodiedMAE, a novel framework for learning unified 3D multi-modal representations tailored for robot manipulation. The authors address two critical issues in current embodied AI research: the domain gap between standard Vision Foundation Model (VFM) training data and robotic tasks, and the lack of effective architectures for incorporating 3D perception. Their solution involves creating DROID-3D, an enhanced version of the DROID dataset with high-quality depth maps and point clouds, and developing a multi-modal masked autoencoder that learns joint representations across RGB, depth, and point cloud modalities through stochastic masking and cross-modal fusion. Extensive evaluation across 70 simulation tasks and 20 real-world tasks demonstrates that EmbodiedMAE outperforms state-of-the-art VFMs in both training efficiency and final performance, while showing favorable scaling properties and enabling effective policy learning from 3D inputs.

### Strengths
1. This work provides an end-to-end solution addressing both data scarcity (via DROID-3D) and architectural limitations (via EmbodiedMAE), offering substantial value to the research community.

2. The use of ZED SDK for generating metric depth maps and point clouds represents a significant improvement over commonly used estimated depth methods, providing temporally consistent 3D data that is crucial for robotic manipulation.

3. The combination of modality-agnostic stochastic masking (using Dirichlet distribution) with explicit cross-modal fusion in the decoder is elegant and effectively encourages learning aligned representations across different modalities.

4. The experimental design is particularly strong, covering diverse simulation environments and real-world platforms, with comprehensive comparisons against relevant VFM baselines.

5. The distillation approach from Giant to smaller variants makes the technology accessible for practical applications while maintaining strong performance.

### Weaknesses
1. While the paper demonstrates strong performance metrics, it lacks discussion of inference latency for the different EmbodiedMAE variants. For real-world robotic deployment where real-time performance is critical, understanding the latency characteristics on target hardware would help assess practical applicability.

2. The observation that point-cloud-only policies underperform RGB-only inputs is noted but not thoroughly analyzed. The paper would benefit from exploring whether alternative point cloud processing techniques could improve PC-only performance.

3. The ablation study focuses mainly on masking ratios but provides limited analysis of the Dirichlet distribution parameter α. Understanding how different α values affect learning across various task types would offer deeper insights into optimal masking strategies for multi-modal representation learning.

### Questions
1. What is the inference latency of different EmbodiedMAE variants (particularly Base and Large) on common robotic hardware platforms? How does this compare to baseline models in terms of the latency-performance trade-off?

2. Given the underperformance of PC-only policies, did the authors experiment with alternative point cloud encoders or pre-processing techniques to improve robustness to sensor noise? What modifications might help bridge this performance gap?

3. How sensitive are the pre-training results to the Dirichlet parameter α? Have the authors experimented with different α values to understand its impact on cross-modal learning efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper introduces EmbodiedMAE, a novel framework addressing the challenge of 3D multi-modal representation learning for embodied AI. A key contribution is the introduction of the DROID-3D dataset, which extends the large-scale DROID robot manipulation dataset by incorporating depth maps generated from its stereo camera data via the ZED SDK. The core of the EmbodiedMAE method is a multi-modal Masked Autoencoder (MAE) pre-trained on RGB, depth, and point cloud data to learn a robust representation encoder. To create more efficient models, the authors distill knowledge from a ViT-giant model, producing small, base, and large variants of their encoder. For downstream evaluation, the framework is paired with a compact version of a Robotic Decision Transformer (RDT) as a policy network. Experiments conducted in both simulated and real-world robotic manipulation tasks demonstrate that EmbodiedMAE achieves superior performance, validating its effectiveness for embodied intelligence.

### Strengths
- The paper is well-written and clearly structured. The contributions are easy for the reader to understand and appreciate.

- The development of the DROID-3D dataset, which augments an existing large-scale robotics dataset with processed depth information, is a notable contribution. If made publicly available, this dataset will be a valuable asset to the embodied AI and robotics research communities.

- The proposed method is technically sound, addressing the important problem of multi-modal representation learning for embodied AI.

- The potential ability to perform depth2RGB, RGB2Depth, and Re-Color is really interesting.

- This work has a thorough experimental validation, which encompasses both simulation and real-world robotic manipulation tasks. This strengthens the paper’s conclusions, convincingly demonstrating the robustness and practical applicability of the method.

### Weaknesses
1. The reliance on the ZED SDK for depth estimation warrants further discussion. Depth data from stereo SDKs can often be noisy, sparse, or contain artifacts, which could significantly impact the quality of the learned representations. The paper would be strengthened by:

    - Clarifying if any specific preprocessing or filtering techniques were applied to the raw depth maps.

    - Discussing the potential impact of depth noise on the model's performance.

    - Exploring or at least acknowledging more robust solutions, such as hybrid methods that combine SDK output with modern AI-based depth completion or denoising networks.

2. The DROID dataset provides rich multi-view information (two third-person, one ego-centric) that appears to be under-leveraged. The manuscript is unclear on which view(s) were used for pre-training and how the point clouds were generated. This raises two key points:

    - It should be explicitly stated whether the point clouds are derived from a single view or fused from multiple views.

    - Leveraging multi-view consistency could be a powerful tool to denoise the depth data or generate more complete 3D point clouds. The current single-view (or ambiguously generated) approach may be a significant limitation.

3. The selection of FPS and a DP3-style encoder for point cloud processing is a potential weakness. These methods, to my knowledge, are outdated and poor in 3D vision. The authors should either justify this architectural choice or, ideally, compare it against more contemporary and powerful alternatives such as those based on sparse convolutions (e.g., Zhu et al., 2024) to demonstrate that the chosen backbone is not a performance bottleneck.

4. The experimental comparisons between EmbodiedMAE and baselines like SPA and DINOV2 are potentially confounded by mismatched model sizes (e.g., comparing a 'Giant' variant to a 'Large' variant). For a fair and convincing evaluation, the authors should:

    - Provide a direct comparison of performance with all methods normalized to the same model size (e.g., all at 'Base' or 'Large' scales).

    - Include a parameter count table for all compared encoder models in different tables.

    - Present a scaling analysis curve (performance vs. model size) for EmbodiedMAE to better characterize its efficiency and scalability.

5. The core motivation for simultaneously pre-training on three modalities (RGB, depth, and point cloud) is not fully substantiated. It is unclear if this tri-modal approach offers a true synergistic benefit. The paper would be much more compelling with rigorous ablation studies that isolate the contribution of each modality. For instance:
    
    - How does the full model compare to variants pre-trained on simpler combinations, such as RGB-only, Depth-only, or RGB+Depth?

    - If the downstream policy primarily consumes point clouds, is the point cloud branch of EmbodiedMAE superior to state-of-the-art encoders pre-trained exclusively on point cloud data? Without these ablations, the necessity of the proposed multi-modal complexity remains an open question.

[1] Haoyi Zhu, Yating Wang, Di Huang, Weicai Ye, Wanli Ouyang, and Tong He. Point cloud matters: Rethinking the impact of different observation spaces on robot learning. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, NIPS, 2024.

### Questions
See the weakness section

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5