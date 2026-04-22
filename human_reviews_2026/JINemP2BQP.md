# AsyncBEV: Cross-modal flow alignment in Asynchronous 3D Object Detection

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
In autonomous driving, multi-modal perception tasks like 3D object detection typically rely on well-synchronized sensors, both at training and inference. However, despite the use of hardware- or software-based synchronization algorithms, perfect synchrony is rarely guaranteed: Sensors may operate at different frequencies, and real-world factors such as network latency, hardware failures, or processing bottlenecks often introduce time offsets between sensors.  Such asynchrony degrades perception performance, especially for dynamic objects. To address this challenge, we propose AsyncBEV, a trainable, lightweight, and generic module to improve the robustness of 3D Bird's Eye View (BEV) object detection models against sensor asynchrony. Inspired by scene flow estimation,  AsyncBEV first estimates the 2D flow from the BEV features of two different sensor modalities, taking into account the known time offset between these sensor measurements.  The predicted feature flow is then used to warp and spatially align the feature maps, which we show can easily be integrated into different current BEV detector architectures (e.g., BEV grid-based and token-based).  Extensive experiments demonstrate AsyncBEV improves robustness against both small and large asynchrony between LiDAR or camera sensors in both the token-based CMT and grid-based UniBEV, especially for dynamic objects. We significantly outperform the ego motion compensated CMT and UniBEV baselines, notably by $16.6$ % and $11.9$ % NDS on dynamic objects in the worst-case scenario of a $0.5 s$ time offset. Code is available at \url{https://github.com/tudelft-iv/AsyncBEV}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AsyncBEV, a lightweight module designed to enhance the robustness of multi-modal 3D object detectors against sensor asynchrony by introducing a novel Δ-BEVFlow estimation task. The method predicts motion in BEV feature space and warps asynchronous features to a reference timestamp, demonstrating compatibility with both token-based and grid-based detectors. The writing is clear and well-structured, and the experimental evaluation on nuScenes shows significant improvements in handling asynchronous inputs, especially for dynamic objects.

### Strengths
1.The paper is clearly written and easy to follow.

2.It explores an important and underexplored problem in multi-modal perception.

3.The method is simple yet effective, with a lightweight and generic design.

4.Extensive experiments validate the effectiveness of the proposed approach under various asynchronous settings.

### Weaknesses
1.The validation is limited to the nuScenes dataset, lacking cross-dataset generalization.

2.The method does not address scenarios where multiple sensors are asynchronous at the same time, which may lead to increased computational complexity.

3.The Motion Compensation-based baseline is somewhat weak. The comparison to asynchronous fusion methods in related work, like StreamingFlow and TimeAlign, is missing.

### Questions
1.Have the authors considered evaluating AsyncBEV on other autonomous driving datasets, such as Waymo, to further demonstrate its generalization capability?

2.How would the method scale and perform when more than two sensors are asynchronous simultaneously, and would the current flow estimation strategy still be effective?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work presents a novel framework for improving the robustness of 3D object detection models in the presence of asynchronous sensors. The proposed framework consists of AsyncBEV, a lightweight and generic module that can be integrated into existing BEV object detection models for improved robustness against sensor asynchrony. AsyncBEV estimates the 2D flow in the BEV feature space and then warps the asynchronous sensor data to align with a reference frame. Extensive experiments on the nuScenes dataset with both grid-based and token-based BEV detectors show the effectiveness of the proposed approach in varying levels of asynchrony between LiDAR and camera sensors, especially for dynamic objects.

### Strengths
- Addressing robustness against sensor asynchrony is important from a practical viewpoint since autonomous vehicles often come equipped with multiple sensors.
- The paper is well-written and easy to follow. The description is detailed and the figures (Fig.2,3,5) are informative.
- AsyncBEV module is a generic and lightweight module that can be combined with both grid-based and token-based BEV detectors.
- The proposed module predicts the delta 2D scene flow in BEV space, conditioned on delta timesteps across multimodal BEV features.
- Experiments on nuScenes (Tab.1) show the benefits of AsyncBEV over egomotion compensation (EMC) on both token-based (CMT) and grid-based (UniBEV) approaches.
- Ablations in Fig.4, Tab.2 provide more insights into the capabilities of different components.

### Weaknesses
- In Sec.4.2, for the finetuning UniBEV variant, is the delta timestep offset (between reference and asynchronous sensor) also used as input? It'd be useful to have a finetuning baseline that also incorporates delta timestep offset. For example, LiDAR BEV can be augmented with the delta timestep (on a per-point basis) as an additional feature channel (a similar strategy was also used in the Fan et al. 2025 referenced paper). This would help understand if the delta flow formulation is indeed effective compared to simpler alternatives like finetuning with the delta timestep as an additional feature channel.
- The finetuning variant in Tab.2 should also be a baseline in Tab.1 (applied to both CMT and UniBEV) since it leads to extensive gains (as noted in Tab.2). This is relevant since CMT and UniBEV are not trained with any asynchronous data.
- Since EMC is quite widely used in the autonomous driving literature (as mentioned in the paper), it'd be useful to report the delta gains with respect to EMC variants in Tab.1. The gains of the proposed approach are still clear, this would better contextualize the benefits of AsyncBEV over the standard EMC approach.

### Questions
The paper is well written, and the claims are validated in the experiments. My main concern is regarding simpler alternatives as baselines to better contextualize the benefits of the delta flow formulation (more details in the weaknesses above):
- A finetuning variant for both CMT & UniBEV should be added to Tab.1 since these methods are not trained on asynchronous data.
- A finetuning variant where the delta timestep offset is used as additional input should also be considered.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a critical challenge in autonomous driving perception: sensor asynchrony (caused by mismatched sensor frequencies, network latency, or hardware bottlenecks). Most 3D object detectors rely on perfectly synchronized multi-modal data (LiDAR/camera), but asynchrony degrades performance—especially for dynamic objects, which are key to safety. To solve this, the authors propose AsyncBEV, a lightweight, generic module compatible with both token-based (e.g., CMT) and grid-based (e.g., UniBEV) detectors.
AsyncBEV introduces Δ-BEVFlow estimation, a novel task that predicts 2D flow in Bird’s Eye View (BEV) space between multi-modal features using known time offsets. Unlike Ego Motion Compensation (EMC, which only aligns static objects) or traditional scene flow methods (requiring point clouds and fixed offsets), Δ-BEVFlow explicitly models dynamic object motion. It offers two formulations: motion-based (direct flow regression) and velocity-based (predict velocity first, then scale by time)—the latter is adopted for better regularization.
Experiments on the nuScenes dataset show AsyncBEV significantly enhances robustness: in the worst-case 0.5s time offset, it outperforms EMC baselines by 16.6% (CMT) and 11.9% (UniBEV) in NDS for dynamic objects, with minimal computational overhead (marginal FPS loss).

### Strengths
**High Practical Relevance**

Sensor asynchrony is unavoidable in real-world autonomous driving, yet it is often overlooked in detector design. By targeting this gap, AsyncBEV directly improves the safety and reliability of perception systems—particularly for dynamic objects (e.g., pedestrians, moving vehicles), which are the primary cause of accidents. This makes the work valuable for both academic research and industrial deployment.

**Effective Δ-BEVFlow Design** 

Δ-BEVFlow addresses key limitations of prior methods: it operates on multi-modal BEV features (not raw point clouds) and supports variable time offsets, enabling flexible cross-modal alignment. The velocity-based formulation further strengthens the design: by decoupling velocity from time, it ensures flow approaches zero for near-synchronous sensors (avoiding unnecessary distortions) and simplifies learning, as validated by ablation studies (Table 2).

**Generality and Lightweight Integration**

AsyncBEV is architecture-agnostic: it adapts to token-based detectors (by adjusting token coordinates) and grid-based detectors (by generating grid look-up tables) with minimal modifications. It also introduces negligible computational overhead—experiments show only a 0.3–0.4 FPS drop (Table 1)—making it suitable for real-time autonomous driving pipelines.

**Strong Interpretability**

Explicit flow prediction allows intuitive visualization (Figures 5, 6), where predicted Δ-BEVFlow closely aligns with ground truth and directly corrects bounding box misalignments. Quantitative results (e.g., Figure 4’s flat performance curves for AsyncBEV) further confirm its robustness across varying time offsets, enhancing trust in the module’s mechanism.

### Weaknesses
**Limited Novelty Compared to Prior Asynchrony-Robust Work**

The core idea of using BEV flow for asynchrony compensation is not entirely new. For example, CoBEVFlow (Wei et al., NeurIPS 2023) already uses BEV flow to handle asynchronous collaborative perception, though it relies on object proposals and is more computationally heavy. Additionally, recent work like UniV2X (Yu et al., AAAI 2025) explores end-to-end autonomous driving with V2X cooperation, which also involves addressing multi-agent asynchrony. The paper acknowledges these works but does not sufficiently emphasize how Δ-BEVFlow advances beyond them—e.g., why feature-based flow (without proposals) offers better generalization, or how variable time offset handling outperforms alternatives. This weakens the claim of novelty.

**Simplified Asynchrony Assumptions**

The paper assumes only two sensors (one synchronous reference, one asynchronous), but real autonomous driving systems use 5–10 sensors (e.g., 6 cameras, 1 LiDAR, 5 radars in nuScenes). AsyncBEV cannot handle scenarios where 3+ sensors have overlapping offsets (e.g., LiDAR delayed by 0.2s, front camera by 0.1s). The authors mention extending to multi-sensor asynchrony as future work, but the current design lacks scalability to full-scale vehicle perception pipelines.

**Synchronous Performance Trade-Off**

Default AsyncBEV causes small but consistent performance drops in the synchronized case (0s offset): 0.4% NDS for CMT and 1.0% NDS for UniBEV (Table 1). The "frozen detector" variant (AsyncBEV-FD) avoids this drop but underperforms in asynchronous scenarios. The paper does not explore alternative training strategies (e.g., adaptive loss weighting) to resolve this trade-off—critical for deployment, as synchronized sensors are the most common real-world scenario.

### Questions
**Limited Novelty Compared to Prior Asynchrony-Robust Work**

The core idea of using BEV flow for asynchrony compensation is not entirely new. For example, CoBEVFlow (Wei et al., NeurIPS 2023) already uses BEV flow to handle asynchronous collaborative perception, though it relies on object proposals and is more computationally heavy. Additionally, recent work like UniV2X (Yu et al., AAAI 2025) explores end-to-end autonomous driving with V2X cooperation, which also involves addressing multi-agent asynchrony. The paper acknowledges these works but does not sufficiently emphasize how Δ-BEVFlow advances beyond them—e.g., why feature-based flow (without proposals) offers better generalization, or how variable time offset handling outperforms alternatives. This weakens the claim of novelty.

**Simplified Asynchrony Assumptions**

The paper assumes only two sensors (one synchronous reference, one asynchronous), but real autonomous driving systems use 5–10 sensors (e.g., 6 cameras, 1 LiDAR, 5 radars in nuScenes). AsyncBEV cannot handle scenarios where 3+ sensors have overlapping offsets (e.g., LiDAR delayed by 0.2s, front camera by 0.1s). The authors mention extending to multi-sensor asynchrony as future work, but the current design lacks scalability to full-scale vehicle perception pipelines.

**Synchronous Performance Trade-Off**

Default AsyncBEV causes small but consistent performance drops in the synchronized case (0s offset): 0.4% NDS for CMT and 1.0% NDS for UniBEV (Table 1). The "frozen detector" variant (AsyncBEV-FD) avoids this drop but underperforms in asynchronous scenarios. The paper does not explore alternative training strategies (e.g., adaptive loss weighting) to resolve this trade-off—critical for deployment, as synchronized sensors are the most common real-world scenario.

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
4

### Summary
This paper introduces AsyncBEV, a module designed to improve the robustness of LiDAR–camera 3D object detection under asynchrony. The proposed module predicts a Δ-BEVFlow, a BEV flow to spatially align asynchronous lidar features prior to fusion. The approach generalizes across both grid and token-based BEV detectors and improves performance over ego-motion compensation on nuScenes, particularly for dynamic objects and large temporal offsets (0.1–0.5 s). While technically sound and empirically validated, the work would benefit from stronger comparison against alternative approaches to help motivate the usefulness.

### Strengths
- Conceptually intuitive and interpretable formulation for feature alignment under temporal misalignment and dynamic motion.
- Generalizable design compatible with both grid and token BEV frameworks.
- Consistent improvements across offset magnitudes with negligible runtime overhead.

### Weaknesses
- Lacks discussion of real-world latency handling in AV stacks, where stale data (>100 ms) are typically discarded. Comparison to such baselines (e.g., camera-only inference or temporal propagation) would clarify practical value.
- Limited benchmarking against contemporary temporal alignment methods like StreamingFlow.
- Performance could be impacted by stochastic latency profiles, but no analysis is provided on sensitivity to time offset estimation error.

### Questions
- How does AsyncBEV compare to fallback strategies such as camera-only inference or simple temporal propagation when LiDAR is delayed or missing?
- What are the performance and latency trade-offs relative to recent streaming or time-aware fusion methods such as StreamingFlow?
- Could Δ-BEVFlow be trained to handle uncertain or estimated time offsets instead of relying on ground-truth Δt?

### Soundness
3

### Presentation
3

### Contribution
2
