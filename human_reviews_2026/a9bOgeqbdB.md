# RAP: 3D Rasterization Augmented End-to-End Planning

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Imitation learning for end-to-end driving trains policies only on expert demonstrations. Once deployed in a closed loop, such policies lack recovery data: small mistakes cannot be corrected and quickly compound into failures. A promising direction is to generate alternative viewpoints and trajectories beyond the logged path. Prior work explores photorealistic digital twins via neural rendering or game engines, but these methods are prohibitively slow and costly, and thus mainly used for evaluation. In this work, we argue that photorealism is unnecessary for training end-to-end planners. What matters is semantic fidelity and scalability: driving depends on geometry and dynamics, not textures or lighting. Motivated by this, we propose 3D Rasterization, which replaces costly rendering with lightweight rasterization of annotated primitives, enabling augmentations such as counterfactual recovery maneuvers and cross-agent view synthesis. To transfer these synthetic views effectively to real-world deployment, we introduce a Raster-to-Real (R2R) feature-space alignment that bridges the sim-to-real gap at the representation level. Together, these components form the Rasterization Augmented Planning (RAP) pipeline, a scalable data augmentation framework for planning. RAP achieves state-of-the-art closed-loop robustness and long-tail generalization, ranking 1st on four major benchmarks: NAVSIM v1/v2, Waymo Open Dataset Vision-based E2E Driving, and Bench2Drive. Our results demonstrate that lightweight rasterization with feature alignment suffices to scale end-to-end training, offering a practical alternative to photorealistic rendering. Project page: https://alan-lanfeng.github.io/RAP/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical problem of brittleness in end-to-end autonomous driving models trained via imitation learning. The authors argue that this brittleness stems from the lack of "recovery" data, as models are only exposed to expert demonstrations. To overcome this, they propose Rasterization Augmented Planning (RAP), a scalable data augmentation framework.

The core contribution is a lightweight 3D rasterization pipeline that generates synthetic camera views from annotated geometric primitives (e.g., agent cuboids, lane polylines). This approach deliberately forgoes photorealism, arguing that semantic and geometric fidelity are sufficient for training robust driving planners, and that this method is far more scalable than computationally expensive alternatives like neural rendering or game engines.

This rasterization pipeline enables two key data augmentations: (1) **Recovery-oriented perturbations**, which simulate off-expert-path maneuvers to teach the policy how to recover from mistakes, and (2) **Cross-agent view synthesis**, which re-renders scenes from the perspective of other agents to dramatically increase the volume and diversity of training data.

To bridge the domain gap between the abstract rasterized views and real-world images, the paper introduces a Raster-to-Real (R2R) alignment module. R2R operates in the feature space, using a combination of spatial-level feature distillation (aligning real features to "cleaner" raster features) and global-level adversarial adaptation to enforce domain invariance.

The authors demonstrate the effectiveness of RAP through extensive experiments, achieving state-of-the-art results on four major benchmarks: NAVSIM v1/v2, the Waymo Open Dataset (WOD) Vision-based E2E Challenge, and Bench2Drive. The results show consistent improvements in closed-loop robustness and long-tail generalization, not only for their proposed model but also when RAP is applied to existing state-of-the-art planners.

### Strengths
1. **Pragmatic and Well-Motivated Core Idea:** The central thesis—that photorealism is not necessary for training robust E2E planners and that semantic/geometric fidelity is sufficient—is compelling and challenges a prevailing trend in the community. This provides a practical, scalable, and computationally efficient alternative to the prohibitively expensive photorealistic simulation pipelines (NeRFs, 3DGS, etc.). This insight is a significant conceptual contribution.
2. **Exceptional Experimental Validation:** The empirical evidence supporting the paper's claims is extensive and highly convincing.
    - **Breadth:** Achieving #1 ranking on four diverse and challenging benchmarks (NAVSIM, WOD, Bench2Drive) is a remarkable feat and strongly validates the method's effectiveness and generalizability.
    - **Rigor:** The evaluation covers both open-loop metrics (WOD) and, more importantly, pseudo-closed-loop (NAVSIM v2) and full closed-loop (Bench2Drive) simulations, directly addressing the core problem of closed-loop brittleness.
    - **Model-Agnosticism:** The authors demonstrate that RAP is not just a component of their specific model but a general framework that provides significant gains when applied to other SOTA models (e.g., `RAP-iPad`, `RAP-DiffusionDrive`). This greatly strengthens the paper's impact.
3. **Thorough and Insightful Ablation Studies:** The paper includes a comprehensive set of ablations that methodically validate the key design choices.
    - The study on rasterization design (Table 5) clarifies why specific choices like solid faces and depth decay are important.
    - The ablation on recovery perturbations (Table 6) directly links this augmentation to improved performance on the closed-loop-oriented NAVSIM v2 benchmark.
    - The analysis of the R2R alignment module (Fig. 5) clearly shows the benefit of both spatial and global alignment.
    - The scaling curve for cross-agent synthesis (Fig. 6) is particularly strong, demonstrating a clear log-scaling law that mirrors findings on real data scaling. This provides powerful evidence for the value of the generated synthetic data.
4. **High-Quality Presentation:** The paper is well-written, clearly structured, and easy to follow. The figures, especially the comparative illustration in Figure 1 and the system overview in Figure 2, are excellent and effectively communicate the core concepts.

### Weaknesses
1. **Oversimplification of the Static World:** The rasterization pipeline represents the world as a set of key primitives against a simple (often black) background. While the ablations show this is effective, it discards a vast amount of visual context from the static environment (buildings, foliage, non-annotated road signs, etc.). This simplification might be a vulnerability in complex urban scenes where such context is semantically important for driving decisions (e.g., navigating a complex, non-standard construction zone).
2. **Information Loss.** A primary weakness of the RAP framework is the inherent information loss resulting from its reliance on abstract 3D rasterization. By training the real-world encoder to align its features with a "clean" representation derived solely from annotated primitives (e.g., agent cuboids and lane lines), the model is implicitly taught to ignore any visual information not present in the labels. This poses a significant safety risk, as the system may fail to perceive critical, unannotated hazards such as road debris, potholes, temporary construction signs, or police officers directing traffic. Furthermore, this abstraction discards subtle but predictive visual cues, like wet road surfaces or holes in the road, which are crucial for nuanced, proactive driving. While this approach demonstrably improves robustness against common visual variations like weather and lighting, it does so at the cost of creating a potential "blindness" to novel or out-of-annotation long-tail events, a critical limitation for real-world deployment.
3. **RL Training.** A weakness is the paper's narrow focus on imitation learning, which fails to exploit the full potential of the proposed RAP framework. The authors have effectively built a fast and scalable driving simulator, an ideal environment for reinforcement learning (RL). By restricting their method to data augmentation for IL, they miss the opportunity to use RL to train an agent that could learn through active exploration, optimize for more complex rewards, and potentially discover policies superior to the original expert. This confinement to IL undersells the power of their core contribution, leaving its most transformative application as an RL training engine unexplored.

### Questions
1. Regarding the R2R alignment module: The spatial alignment loss (Eq. 2) updates the real features `F^r` to match the frozen raster features `F^s`, treating the latter as a "clean" supervision signal. Could you elaborate on the intuition behind this directional alignment? Did you experiment with a symmetric loss or alternative objectives where both feature distributions are jointly optimized?
2. Regarding the scaling laws (Fig. 6): The log-scaling behavior with cross-agent data is very promising. What do you theorize is the limiting factor for this scaling? Does the performance plateau come from the finite variety of agent behaviors in the log, or from the inherent domain gap that cannot be fully closed by R2R alignment?

### Soundness
4

### Presentation
4

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
This paper propose that 3D rasterization of annotated primitives are semantically similar to images, so it can be used as input for end-to-end driving. By doing so, multiple data augmentation tricks can be applied to end-to-end driving, without costly image rendering. A raster-to-real alignment is further proposed to align the two representations in feature spaces.

### Strengths
* For the input of end-to-end driving models, a 3D rasterization representation is proposed to be the substitute of raw images , which is interestingly to already have similar semantics in DINOv3 feature space.
* With this representation, two types of augmentation are applied in the training of end-to-end driving model, with the need for costly image-space rendering in previous methods.
* A raster-to-real alignment module is proposed to enforce feature consistency between rasterized and real inputs at both spatial and global levels.

### Weaknesses
* On NAVSIM and WOD benchmarks, the method achieves SOTA performance. However, on the real close-loop benchmark Bench2Drive, the performance gain is incremental compared with baseline, which is contrary to Recovery-oriented perturbations.

### Questions
* Any rules considered in Recovery-oriented perturbations, for example, not collide with nearby vehicles?
* Why use minADE as metric in ablation study?
* In Fig.5, why the performance of 100% real data is not the best?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an end-to-end driving framework that uses 3D rasterization instead of costly photorealistic rendering. By focusing on semantic and geometric fidelity and aligning synthetic with real data in feature space, RAP enables more flexible data augmentation and achieves robustness and generalization across multiple driving benchmarks.

### Strengths
- replaces expensive photorealistic rendering, offering a controllable way to generate diverse driving scenarios.
- provides experiments and ablations across multiple major benchmarks (NAVSIM, Waymo, Bench2Drive), consistently achieving strong results
- bridges synthetic and real data efficiently, reducing the sim-to-real gap without costly photorealistic rendering

### Weaknesses
- RAP remains within the imitation learning framework, so it still can display issues like causal confusion and lack of active policy improvement.
- the simplified rasterized scenes may miss fine-grained visual cues or rare real-world conditions that could matter for extreme edge-cases

### Questions
- do the authors plan to evaluate RAP outside of the IL setting? rl evaluations would demonstrate if the model can generalize under agent exploration
- can the authors provide additional examples showing reconstruction results on very OOD scenes? e.g. scenes with objects that have very little representation in the training dataset.

### Soundness
3

### Presentation
3

### Contribution
4
