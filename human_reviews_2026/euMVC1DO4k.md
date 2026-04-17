# Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Vision-language-action (VLA) models have recently shown strong potential in enabling robots to follow language instructions and execute precise actions. However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world. Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets. Alternative methods that estimate 3D cues from 2D images also suffer from the limited performance of depth estimators. We propose Spatial Forcing (SF), a simple yet effective alignment strategy that implicitly forces VLA models to develop spatial comprehension capabilities without relying on explicit 3D inputs or depth estimators. SF aligns intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. By enforcing alignment at intermediate layers, SF guides VLAs to encode richer spatial representations that enhance action precision. Extensive experiments in simulation and real-world environments demonstrate that SF achieves state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further accelerates training by up to 3.8× and improves data efficiency across diverse robotic tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To address the limitations of VLA models largely relying on 2D visual features and the difficulty of acquiring and representing explicit 3D inputs, this paper proposes Spatial Forcing (SF), a method that aligns the deep features of VLA models with target features embedded with spatial awareness, thereby enhancing the spatial reasoning capability of VLA models during task execution and improving action prediction accuracy. The target features containing 3D spatial information are provided by the VGGT model. 

The authors investigate the importance of 3D spatial perception for the success rate of VLA tasks and validate the effectiveness of the SF method in enhancing 2D-based VLA models through multiple simulators and real-world experiments. Additionally, various visualization techniques demonstrate that the VLA model’s features are focused on spatial information and object geometry representation.

### Strengths
1.	Analysis of 3D spatial perception limitations in existing VLA methods: The paper identifies that most current VLA approaches lack understanding of 3D spatial information and highlights the limitations of existing attempts to incorporate 3D cues into VLA models.
2.	Deep insights into LLM feature layers: The authors investigate the features from different layers of the LLM and determine appropriate layers for optimal feature alignment.
3.	Efficient feature alignment with SF: SF is a training-only feature alignment method that significantly improves data efficiency and model capability, particularly in real-world scenarios.

### Weaknesses
1.	While SF demonstrates impressive capability in obtaining spatially informed representations, aligning VGGT features with deep layers of the LLM (e.g., the 24th layer) might potentially compromise the 2D image-derived features acquired in the preceding layers, such as semantic understanding and generalization ability. The authors could provide experimental verification, for example, through real-world experiments testing scene generalization or tasks requiring strong semantic comprehension.
2.	In the real-world experiments, there is no baseline model that explicitly takes 3D inputs. Given that SF emphasizes comparison with 3D-aware models in both method and simulation experiments, including an explicit 3D-input baseline in more spatially complex real-world scenarios could more convincingly demonstrate the advantages of SF.

### Questions
In Table 2, the first row represents a model without target representation and without feature alignment, which uses OpenVLA-OFT as the base model. However, the reported performance in Table 2 does not match the results in Table 1, which may affect the validity of the ablation study. It would be helpful if the authors could explain the cause of this discrepancy or provide additional experimental details.

### Soundness
2

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
This submission employs intermediate-layer alignment to enhance spatial comprehension.Rather than utilizing explicit 3D inputs or a depth predictor in the VLA pipeline, this method only relies on VGGT during the training phase.Evaluations in both simulated and real-world environments demonstrate that the proposed method outperforms existing 3D VLAs.

### Strengths
Other 3D-aware VLA methods explicitly feed depth maps into the VLA model, whereas the proposed method does not require the input of depth maps during the inference phase—thus enabling faster training and inference. The method is simple and straightforward, and evaluations indeed show better performance compared to existing counterparts.

### Weaknesses
As Figure 5 illustrates, while the method with alignment improves the success rate more rapidly than the method without alignment, as the data volume and training iterations increase, the two methods achieve the same performance. This indicates that the proposed method does not remain valid as the data size grows. This raises a question: if the data volume or training steps continue to increase, could the method without alignment exhibit stronger performance instead?

### Questions
The main question concerns the effectiveness of the alignment. Given that data size and computational power will undoubtedly continue to grow in the future, will the proposed method still hold its effectiveness at that point?

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
This paper addresses a critical challenge in Vision-Language-Action (VLA) models: the lack of inherent 3D spatial understanding, which stems from their foundation in Vision-Language Models (VLMs) trained on 2D data. The authors argue that existing solutions, which rely on explicit 3D sensor inputs (e.g., depth, point clouds) or 2D-to-3D estimators, are fraught with practical issues like sensor noise, hardware heterogeneity, data availability, and sub-optimal performance. To overcome this, they propose Spatial Forcing (SF), a novel and simple training strategy. Instead of modifying the model's input, SF implicitly instills spatial awareness by aligning the VLA's intermediate visual representations with geometric features extracted from a powerful, pretrained 3D foundation model (specifically, VGGT). This alignment is achieved via an auxiliary cosine similarity loss during the fine-tuning process. The authors demonstrate through extensive experiments in both simulation and the real world that this implicit supervision method not only achieves state-of-the-art performance, surpassing both 2D and explicit 3D VLA baselines, but also dramatically accelerates training convergence (up to 3.8x) and improves data efficiency.

### Strengths
- The paper tackles a fundamental and highly significant problem in embodied AI. The proposed SF method offers an elegant and practical paradigm that sidesteps the many real-world challenges associated with collecting and using explicit 3D data, making it a potentially high-impact contribution for the robotics community. 

- The core idea of using representation alignment to implicitly distill 3D knowledge from a foundation model into a VLA is novel and elegant.

- The motivation is exceptionally well-established. The authors' use of a "depth probing" experiment is a simple yet powerful diagnostic tool that compellingly visualizes the problem—the lack of spatial information in standard VLA embeddings—and provides a clear justification for their approach.

- The experimental design is thorough, validating the method across different base models (OpenVLA-OFT, $\pi_0$), diverse simulation environments, and crucial real-world scenarios. The component-wise analysis is also well-executed, providing clear insights into the effects of different target representations and alignment layers.

### Weaknesses
- The success of Spatial Forcing is heavily contingent on the availability and quality of a powerful, pretrained 3D foundation model (VGGT in this case). This introduces a strong dependency, and the paper does not discuss the potential limitations if such a model is not available for a specific domain or embodiment, or how the performance of SF scales with the quality of this teacher model.

- While the method is inference-free in terms of overhead, it introduces a non-trivial computational cost during training. It requires running a forward pass through the large VGGT model for every training sample to generate the target representations. This additional overhead in terms of computation and VRAM is not quantified or discussed, which is an important practical consideration that affects the overall "efficiency" claim.

- The paper claims to outperform methods that use explicit 3D inputs (Table 1). While impressive, this comparison may not be entirely fair. SF is effectively distilling knowledge from a very powerful model (VGGT) that has already processed and structured 3D information. In contrast, methods using raw depth maps or point clouds must learn to interpret noisy, unstructured sensor data from scratch. The comparison is thus more akin to "distilled 3D knowledge" vs. "raw 3D data."

- The method uses the "latent representation" from VGGT as the supervision signal. This is somewhat of a black box. It would be beneficial to understand what specific geometric properties (e.g., relative depth, surface normals, object boundaries) are most dominant in these features and are being transferred to the VLA. An ablation using more interpretable outputs from VGGT (like its predicted depth map) as the target could provide valuable insight.

### Questions
The paper argues that the aligned representations still preserve their "original representational identity" based on the t-SNE visualization. Could you elaborate on this? How do you ensure that the "forcing" process doesn't cause the visual features to discard important non-spatial information (e.g., texture, color) that might be crucial for other aspects of the task?

### Soundness
3

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
The paper presents **Spatial Forcing (SF)**, a training-time regularizer that aligns intermediate visual embeddings of a VLA with geometry-rich representations from a pretrained 3D foundation model (VGGT). This auxiliary alignment loss injects privileged spatial priors into the model during fine-tuning, without requiring explicit 3D inputs or architectural modifications at inference time. Empirically, the authors evaluate SF on **LIBERO** and **RoboTwin 2.0** simulation benchmarks as well as several real-world manipulation tasks. SF consistently improves performance over both 2D and explicit-3D VLAs, achieving up to 3.8× faster convergence and demonstrating robustness to lighting, layout, and height variations in the physical setup.

### Strengths
> ### originality
>
> - Addresses the dependency of 3D VLAs on explicit geometric inputs such as depth maps or point clouds.
> - Introduces a simple yet effective co-training mechanism that transfers 3D priors through a pretrained foundation model.
>
> - The privileged-representation intuition is intuitive, and the depth-probing visualization improves interpretability.
>
> ### Quality
>
> - The method is compatible with diverse VLA backbones and requires little engineering effort.
> - The ablation studies on alignment layer and teacher model strengthen the empirical claims

> ### Clarity
>
> - Presents results on LIBERO, RoboTwin 2.0, and real-robot experiments for a comprehensive evaluation.
> - Figures and comparisons are well organized and easy to interpret.
>
> ### Significance
>
> - Bridging 2D vision-based manipulation with 3D perception is an important and rapidly growing research direction.
> - SF provides a practical baseline and may inspire further work on representation alignment and spatial grounding

### Weaknesses
- The experiment setting is a bit simple. For simulation the improvement is marginal, while the real world tasks do not demonstrate enough spatial generation in initial pose distribution 
- The related work part lacks formal explanation for why representation-level cosine alignment leads to stable 3D awareness.
- The supplemental video includes only few real world experiments, and the tasks are easy with jerky behavior. Please include both simulation and more real world experiments under different noise to prove robustness. 
- Currently, the alignment depth layer is chosen empirically, where authors could incorporate an adaptive strategy to improve robustness.

### Questions
- In Table 1,the statistics of SpatialVLA, GeoVLA and 3D-CAVLA are same as the origin paper shows, do authors rigorously train different VLAs under same initialization and parameter?
- In Table 2, VGGT without Position Embedding shows high success rates on two Libero benchmark, could you further analyze potential reason?
- In Figure 4, the results are shown as success rates, but there are no explicit percentage labels or variance/error indicators (e.g., standard deviation or confidence intervals). Could the authors clarify whether these bars represent mean performance across seeds or single-run outcomes, and provide variance information to assess statistical significance?
- Did you experiment with multi-layer or progressive alignment instead of a single-layer constraint?
- Can SF improve other vision-driven policies (e.g., diffusion policy or world-model-based RL)?
- Could feature alignment harm language following of base VLAs? Is there a trade-off between spatial and linguistic? Take pi0 experiment for example, can you ablation other task regression?

### Soundness
3

### Presentation
3

### Contribution
3
