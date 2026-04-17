# STARTrack:Learning Spatio-Temporal Representation Evolution for Target-Aware Tracking

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Efficient modeling of spatio-temporal representations in videos is crucial for achieving accurate object tracking. Existing popular one-stream tracking frameworks typically introduce memory mechanisms or specialized modules for temporal modeling. However, due to the gradual degradation of the initial template and unstable updates of target representations, their performance often deteriorates over time. To address this issue, we propose a simple yet effective video-level tracking framework, STARTrack, which realizes the temporal evolution of target and context representations through an iterative token propagation mechanism. Our framework takes as input the features of the search region along with two types of tokens that carry historical representations, and employs a visual encoder for joint modeling. This design enables target-aware perception and adaptively fuses current and historical representations. The proposed method explicitly avoids the sustained reliance on the initial template during long-term tracking, without introducing additional complex context inputs or motion modeling modules, thereby achieving faster inference. Furthermore, we develop a training strategy tailored to our framework. It enhances the semantic coherence of target representations over time via a representation consistency constraint, and for the first time explicitly incorporates occluded frames into the training process. This guides the tracker to learn context representations that are highly correlated with the spatio-temporal state of the target, thereby reducing the reliance on target appearance itself. Extensive experiments on standard benchmarks demonstrate that STARTrack achieves state-of-the-art performance, while maintaining a favorable balance between accuracy and efficiency. The code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
STARTRACK is a novel video-level object tracking framework that redefines visual tracking as a spatio-temporal representation evolution problem. Unlike traditional trackers that rely on static template matching or sparse temporal updates, STARTRACK introduces an iterative token propagation mechanism to dynamically model the evolution of both target and context representations across video frames.

Key components include:

Two types of tokens: target tokens and context tokens, which store and propagate spatio-temporal representations.

A dual-stream causal attention mechanism with negative attention guidance to ensure temporal consistency and avoid semantic entanglement.

A Frame-wise Information Gain Principle (FIGP) to ensure high-quality token updates.

Dense frame sampling and explicit inclusion of occluded frames during training to enhance robustness.

The method achieves state-of-the-art performance on multiple benchmarks (LaSOT, LaSOText, GOT-10K, TrackingNet, UAV123, TNL2K) while maintaining high inference speed.

### Strengths
1. State-of-the-Art Performance Across Diverse and Challenging Benchmarks
STARTRACK does not merely achieve top results on one or two benchmarks; it demonstrates generalized superiority across a wide spectrum of challenges, which is a strong indicator of its robustness.

LaSOT & LaSOText: Achieving 75.2% AUC on LaSOT and 53.2% on LaSOText is significant because these are large-scale, long-term benchmarks. LaSOText, in particular, with its focus on similar distractors and frequent occlusions, directly validates the core claim that STARTRACK excels at spatio-temporal modeling and discrimination beyond simple appearance matching.

GOT-10K: The high score of 78.5% AO under the strict one-shot protocol (training only on GOT-10K's training split) is a powerful testament to the model's generalization capability. This shows that the learned representation evolution strategy is not overfitted to specific object categories.

UAV123 & TNL2K: Superior performance on UAV123 (aerial perspective, small objects, fast motion) and TNL2K (diverse media including cartoons) proves the framework's adaptability to different domains and data sources, moving beyond conventional RGB video.

2. A Novel and Paradigm-Shifting Pipeline: From Matching to Evolution
The most profound contribution is the conceptual shift from a static matching paradigm to a dynamic evolution paradigm.

Beyond Template Degradation: Traditional trackers, even those with dynamic template updates, fundamentally perform matching. STARTRACK abandons this entirely. Its iterative token propagation mechanism allows the tracker to build a continuously updating "memory" of the target and its relationship with the environment. This explicitly mitigates the Achilles' heel of long-term tracking: the gradual irrelevance of the initial template.

Token as a Dynamic State Vector: The target and context tokens act as a compact, learned state vector that carries all necessary historical information. This is more elegant and potent than hand-crafted update strategies for multiple templates or complex cross-frame attention mechanisms, leading to a simpler yet more effective architecture.

3. A Holistic and Innovative Training Strategy

Dense Sampling with Occlusion: By using densely sampled sequences and, for the first time, explicitly including occluded frames in training, the model is forced to learn a crucial skill: reasoning without appearance. This trains the context tokens to capture the underlying spatial structure and motion patterns of the scene, enabling the tracker to hypothesize the target's location even when it is invisible.

Frame-wise Information Gain Principle (FIGP): This is a clever solution to a key problem in propagation-based models: ensuring that each update is beneficial. FIGP provides a self-supervised, internal consistency signal that actively prevents representation collapse and encourages the tokens to become progressively more informative, ensuring stable long-term performance.

### Weaknesses
1. Insufficient Depth in Related Work on Temporal Modeling
The paper's review of existing temporal methods is somewhat narrow, missing a discussion of several influential works that would provide a richer context for its contributions. PrDiMP (Probabilistic Regression and DIMP)，STMTrack (Spatio-Temporal Memory Trackers)，TCTrack (Temporal Context Trackers)，MeMOTR (Memory-Augmented MOT with Transformers)

2. Limited Analysis of Failure Modes and Robustness Boundaries
The paper convincingly demonstrates success but offers less insight into its limitations.

Extreme Deformation or Fast Motion: How does the token propagation mechanism cope when the target undergoes radical non-rigid deformation or moves so fast that its appearance changes drastically between frames? The dense sampling may help, but the upper limits are not explored.

Full Scene Changes: What happens when the camera cuts to a completely different scene (a common challenge in long-term TV show tracking)? The reliance on spatio-temporal context would likely break down, and it's unclear how the model would recover.

Initialization Sensitivity: While an ablation on target token initialization is provided, a deeper analysis of how sensitive the entire tracking process is to errors or noise in the initial bounding box is missing.

3. Practical Deployment Considerations

Computational Cost of Dense Sampling: While the inference FPS is high, the training cost of using densely sampled sequences is significantly higher than sparse sampling. The paper does not discuss the computational overhead of this training strategy.

Hyperparameter Sensitivity: The performance appears sensitive to hyperparameters like the token matrix size (Fig. 5) and sampling length (Fig. 6). This suggests that optimal deployment on a new dataset might require non-trivial tuning, potentially limiting its applicability.

### Questions
1. Insufficient Depth in Related Work on Temporal Modeling
2. Limited Analysis of Failure Modes and Robustness Boundaries
3. Practical Deployment Considerations

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work presents a simple yet effective video-level tracking framework termed STARTrack, which realizes the temporal evolution of target and context representations using an iterative token propagation mechanism. It takes two types of tokens that carry historical representations to update the static target templates over time for target state inference. The tracker has been evaluated on various public benchmarks.

### Strengths
1. The writing is clear and method is easy to follow. The introduction of two distinct types of tokens to decouple and model the target's appearance and environmental relationship seems reasonable.
2. The paper provides extensive quantitative results on the public tracking benchmarks, demonstrating the method's robustness, and efficiency.

### Weaknesses
1. The strategy of using dynamic template tokens to propagate temporal information is not very novel. Numerous prior works like HIPTrack, SPMTrack also employ token storage and temporal information propagation techniques to improve the performance. The proposed dual-stream token mechanism is an incremental combination of existing ideas rather than a groundbreaking new paradigm. 
2. The performance gain compared to SPMTrack in Table 1 is marginal.
3. The in-depth analyses of the two types of tokens have not been clearly provided.

### Questions
1. Could the authors provide more detail analyses or visualizations of the two types of tokens?
2. What’s the reason that the AUC drops significantly (from 73.8% to 44.7%) when the Frame-wise Information Gain Principle (FIGP) is removed?

### Soundness
2

### Presentation
2

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
STARTrack is a video-level visual object tracking framework that departs from static template-matching paradigms by explicitly modeling the spatio-temporal evolution of both target and context representations. The method uses an iterative token-propagation design: two kinds of learnable representation tokens (target tokens and context tokens) are propagated across frames and fused with current-frame visual features through a dual-stream causal attention mechanism. The model also incorporates negative attention guidance to encourage context tokens to focus on spatio-temporally relevant cues. To support stable temporal updates, STARTrack introduces a tailored training strategy centered on the Frame-wise Information Gain Principle (FIGP), dense sampling, and occlusion-aware training; losses include a temporal refinement term that enforces that current-frame predictions should improve (or not be worse than) those from previous tokens. Experiments on multiple benchmarks (LaSOT, GOT-10K, TrackingNet, TNL2K, UAV123, LaSOText) report state-of-the-art accuracy while maintaining competitive inference speed. Ablations validate the benefits of dual token types, dense sampling, FIGP, and occlusion inclusion during training.

### Strengths
- Iterative token-propagation for spatio-temporal representation evolution: STARTrack proposes propagating learned target and context representation tokens across frames and fusing them with current-frame features. This token-centric propagation explicitly models representation evolution over time and reduces dependence on a fixed initial template, addressing long-term drift.

- Dual-stream causal attention with negative attention guidance: The architecture separates modeling of target and context into parallel causal attention streams to avoid mutual interference, and introduces negative attention guidance to help context tokens converge faster and focus on discriminative spatio-temporal context relevant to the tracked object.

- Training innovations for temporal consistency and occlusion robustness: They formulate the Frame-wise Information Gain Principle and add a temporal refinement term to losses that enforces improvement (or non-degradation) of current predictions relative to previous tokens. They also densely sample frames and explicitly include occluded frames in training, helping the model learn context-driven localization when appearance cues are weak.

### Weaknesses
- Complexity and interpretability of token dynamics: While token propagation is powerful, the paper relies on many design choices (two token types, negative guidance sign flips, dual-stream masking/ordering, initialization strategies). These choices introduce complexity; ablation shows sensitivity (e.g., FIGP removal causes large drops), but the conceptual interpretability and theoretical understanding of token dynamics and stability over very long sequences remain limited.
- Dependence on many engineered training choices and hyperparameters: The method’s strong performance hinges on several specific training strategies (dense sampling, occlusion inclusion, FIGP temporal loss terms, representation initialization variants). This could make reproducibility or transfer to other datasets/domains sensitive to hyperparameters and dataset composition; the paper indicates some instability when modifying multi-frame variants.
- Limited analysis of failure cases and computational trade-offs in diverse settings: Although STARTrack reports good FPS and benchmark numbers, the paper contains limited discussion of failure modes (e.g., heavy crowding, severe distractors, extreme viewpoint/scale changes) and how the token mechanism behaves in such scenarios. Also, while the framework avoids explicit motion modules to keep inference fast, more explicit comparison of compute/memory cost vs. competing long-term trackers (especially under very long sequences or constrained hardware) would strengthen practical claims.

### Questions
SEE WEAKNESS

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a new method to tackle object tracking problem. The proposes method introduces a few new designs in the network architecture to enhance the performance of object tracking. It adopts ViT as visual encoder to embed the input images, together with historic tokens into embeddings. Secondly, it introduces extra query tokens that helps to decouple semantic features. It also adopts dual stream attention mechanism to learn discriminative features from targets and contexts. Extensive experiments show that the proposed method outperforms baseline methods on the benchmark quantitatively.

### Strengths
1. The quantitative results show that the proposed method performs the best compared to baseline methods.

### Weaknesses
1. What are $H_1, H_2, ... H_n$ in Eq.(2) ?   $H_i$ and $W_i$  are used to represent both image size and network parameters in a confusing way. 
2. What do symbols $q, k, v$ represent in Eq. (2)?  How do they differ from $Q, K, V$?
3. Where are the dual stream attention shown in the Figure 2?  It's difficult to understand how the dual stream work in the entire network architecture. 
4. Section 3.3.2 Why do the query tokens avoid semantic entanglement? How do they achieve this goal?
5. The experiments do not have any qualitative results.

### Questions
The main issue of this paper submission is the presentation and the development of the paper content.  The organization and presentation of this manuscript is poor such that it's hard to follow the logic flow. The mathematical symbols are not properly defined and explained progressively. The figures and diagrams are not properly referred from the text.  The authors are suggested to paraphrase and polish the manuscript text carefully.

### Soundness
1

### Presentation
1

### Contribution
2
