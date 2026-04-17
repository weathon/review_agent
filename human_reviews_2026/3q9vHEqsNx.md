# FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
High-quality 3D world models are pivotal for embodied intelligence and Artificial General Intelligence (AGI), underpinning applications such as AR/VR content creation and robotic navigation. 
Despite the established strong imaginative priors, current video foundation models lack explicit 3D grounding capabilities, thus being limited in both spatial consistency and their utility for downstream 3D reasoning tasks. 
In this work, we present FantasyWorld, a geometry-enhanced framework that augments frozen video foundation models with a trainable geometric branch, enabling joint modeling of video latents and an implicit 3D field in a single forward pass. 
Our approach introduces cross-branch supervision, where geometry cues guide video generation and video priors regularize 3D prediction, thus yielding consistent and generalizable 3D-aware video representations. 
Notably, the resulting latents from the geometric branch can potentially serve as versatile representations for downstream 3D tasks such as novel view synthesis and navigation, without requiring per-scene optimization or fine-tuning. 
Extensive experiments show that FantasyWorld effectively bridges video imagination and 3D perception, outperforming recent geometry-consistent baselines in multi-view coherence and style consistency. 
Ablation studies further confirm that these gains stem from the unified backbone and cross-branch information exchange.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tries to make camera-controlled videos that stay consistent by giving the generator an internal sense of 3D. It keeps the front part of a standard video model fixed to lightly clean the input and mix in text, image, and camera hints. The main part has two paths: one draws the video frames, the other tracks the scene’s 3D shape. The two paths talk to each other, and a small module reads the 3D path to output depth, a set of 3D points, and the camera’s position and direction.

### Strengths
1) Clean unified video+3D design with a frozen VFM and a trainable geometry branch; avoids per-scene optimization

2) Produces camera-conditioned video + 3D signals in one forward pass, a practical interface for downstream tasks.

### Weaknesses
1) Several parts of the paper are unclear. The overall setup is unclear. What is the implicit 3D field here: tensor shapes, spatial/temporal resolution, and how it is applied at inference? The implicit 3D field definition is unclear here, and there is no diagram to support it. PCB conditioning terminology is also confusing; it's just a frozen pre-denoiser -- why is it conditioning? 

2) There is no ablation isolating (a) cross-branch exchange with the geometry branch kept, or (b) the effect of a unified backbone vs a non-unified alternative. The paper’s abstract line “ablation studies confirm gains stem from the unified backbone and cross-branch information exchange” is therefore only partially supported by the evidence shown (it demonstrates the benefit of having the 3D branch + adapters together, not each factor independently). The current “w/o 3D” removes both branch and adapters, so it can’t attribute gains to the unified backbone or to cross-branch exchange.

3) 3D geometry consistency is also not validated. One way to do this would be to run COLMAP on generated videos and report BA reprojection error, % registered frames, avg track length/inlier ratio, and pose ATE/RE (Absolute Trajectory Error and Reprojection Error) vs the commanded camera path. 

4) Missing key baselines. Add apples-to-apples comparisons to single-image→world and camera-controllable video models (Odin/360-1M[1], GenEx[2], VD3D[3]/V3D[4]) under shared camera paths/seeds; report PSNR/SSIM/LPIPS, pose ATE/RE, and identity/multi-view consistency.

Minor: 
1) I also noticed that the overall resolution of the resulting frames (336 × 592) is lower compared to compared baselines
2) The code provided for the reproducibility check is not available.  [The requested file is not found on the anonymous GitHub]


References:

[1] From an Image to a Scene: Learning to Imagine the World from a Million 360 Videos (NeurIPS 2024)

[2] GenEx: Generating an Explorable World (ICLR 2025)

[3] VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control (ICLR 2025)

[4] V3D: Video Diffusion Models are Effective 3D Generators [T-PAMI 2025]

### Questions
Please provide: 

1) Missing ablations: 
- a no-adapter ablation 
- a no-unification control (separate backbones, or freeze geometry branch outputs with no shared features) 

2) Comparisons with missing baselines/related work 

3) 3D geometry consistency evaluation

Minor: 

1) The first line of the abstract starts with "High-quality 3D world models are pivotal for embodied intelligence and Artificial General Intelligence (AGI)"  -- why?

### Soundness
3

### Presentation
2

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
This paper presents FantasyWorld, a unified framework that jointly models video generation and 3D geometry prediction within a single feed-forward pipeline through cross-branch supervision. By employing bidirectional cross-attention, the model enables mutual information flow between appearance and geometry, effectively merging them into one coherent representation. The authors conduct extensive quantitative and qualitative experiments across multiple datasets, demonstrating clear improvements in 3D consistency, photorealism, and scene coherence compared to baseline models. Ablation studies further confirm that both the geometry branch and the cross-branch coupling are essential for performance gains, highlighting the core innovation and technical contribution of the work.

### Strengths
1. Well-motivated and clearly written: The paper is well motivated and easy to follow. The problem formulation and technical pipeline are clearly described, making the overall logic coherent and accessible.
2. Intuitive and meaningful innovation: The proposed model design makes sense from the first intuition — the unified modeling of video generation and 3D geometry prediction is conceptually appealing and addresses an important gap between appearance synthesis and spatial reasoning.
3. Effective cross-attention mechanism: The introduced bidirectional cross-attention effectively facilitates mutual information merging between geometry and appearance branches.
4. Comprehensive validation: The experiments in the manuscript, together with the demonstration video, provide convincing evidence that supports the claimed conclusions. The results are consistent across datasets.
5. Code reproducibility and openness:The released codebase and structured training setup make the method reproductive, which is valuable for the research community and enhances the paper’s credibility.

### Weaknesses
1. Lack of theoretical justification for cross-branch supervision: Although cross-attention mechanism effectively fuses geometry and appearance features, the underlying mechanism is only demonstrated empirically. A more theoretical or analytical explanation of why this coupling improves stability and consistency would strengthen the contribution.
2. Limited temporal scalability: As illustrated in the modeling and discussed in the Section 5, the model is designed for short, fixed-length video generation and does not address long-term temporal consistency or camera trajectory drift, which limits its generalization to continuous video required by downstream applications such as data expansion in autonomous driving and robotics.
3. The qualitative results for 3D reconstruction are relatively limited. It would strengthen the paper if the authors could provide more visual examples, especially on out-of-domain or unseen scenarios. For instance, testing the model on stylistically distinct or challenging inputs such as artistic or highly textured scenes (e.g., Van Gogh’s paintings)  would better demonstrate the model’s generalization and robustness in geometry-aware reconstruction.

### Questions
1. I am curious about how the proposed framework can be used to generate explicit 3D structures, such as meshes or 3D Gaussian splattings (3DGS), from its implicit representation. Have the authors attempted to extract these forms of geometry using the predicted depth, point maps, or other intermediate signals? It would be helpful to see qualitative or quantitative comparisons of such reconstructions, as they directly reflect how well the learned implicit field captures structural consistency and could provide valuable insights for robotics applications.
2. How does the proposed model behave in dynamic or non-rigid environments (e.g., moving objects, people, or changing lighting)? It would be acceptable if artifacts appear, but I am interested in whether the framework is fundamentally capable of reasoning about such dynamics, since the model takes advantages from the video foundation model.
3. How does the method handle metric scale issues in the reconstructed geometry? Since scale ambiguity is a common problem in monocular video-based 3D reconstruction, does the proposed implicit 3D representation preserve or estimate absolute scale, and if not, how might this affect potential applications in embodied or robotic settings?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
FantasyWorld proposes a novel framework that unifies video generation and 3D geometric reasoning in a single feed-forward pass. The method augments a frozen video foundation model (Wan2.1) with a trainable geometric branch that predicts an implicit 3D field alongside video latents. Key innovations include: (i) Preconditioning Blocks (PCBs) that provide partially denoised features as input to the geometry branch; (ii) Integrated Reconstruction and Generation (IRG) blocks featuring asymmetric dual branches coupled via bidirectional cross-attention; and (iii) cross-branch supervision where geometry guides video synthesis and video priors regularize 3D prediction. The resulting representations support downstream tasks like novel view synthesis without per-scene optimization. Experiments on benchmarks such as WorldScore and RealEstate10K demonstrate improved multi-view coherence, style consistency, and geometric fidelity over recent baselines including Voyager, WonderWorld, and AETHER.

### Strengths
1. Originality: Introduces a novel asymmetric dual-branch architecture that enables mutual reinforcement between video generation and 3D perception in a single forward pass—distinct from prior decoupled or iterative approaches.

2. Quality: Rigorous experimental validation across multiple datasets, camera motions, and evaluation protocols; strong baselines; thorough ablations.

3. Clarity: Exceptionally clear exposition of both high-level intuition and low-level implementation (e.g., inverted DPT reassembly strategy based on diffusion depth semantics).

4. Significance: Addresses a core limitation in current video foundation models—their lack of explicit 3D grounding—and provides a practical solution with immediate applicability to navigation, simulation, and content creation.

### Weaknesses
1. Limited dynamic scene evaluation: All experiments focus on static or near-static scenes (WorldScore static split, RealEstate10K). It remains unclear how the method handles non-rigid motion, deformable objects, or dynamic lighting—key challenges in real-world embodied settings.
2. Computational cost: While avoiding per-scene optimization is a strength, the reported training uses 112 H20 GPUs for 144 hours in Stage 2. The inference cost (memory/time) relative to baselines is not discussed, which matters for downstream deployment.
3. Downstream task validation is indirect: The claim that geometric latents are “versatile for downstream tasks” is supported only via 3DGS reconstruction. Direct evaluation on tasks like pose estimation, navigation planning, or policy learning would strengthen this assertion.

### Questions
1. How would FantasyWorld handle videos with significant non-rigid motion (e.g., humans, flowing cloth)? Does the implicit 3D field assume scene rigidity, and if so, how might this be extended?

2. What is the latency and memory footprint of your full model vs. baselines during inference? Is the geometry branch lightweight enough for real-time applications?

3. Have you tested the geometric features directly in downstream embodied tasks (e.g., as input to a navigation policy)? If not, do you plan to release the features for community benchmarking?

4. Are there scenarios where cross-branch supervision leads to over-constrained generation (e.g., sacrificing visual diversity for geometric plausibility)? Can you share any observed trade-offs?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a framework for outputting 3D-consistent world models given a single image, text-prompt, and a camera trajectory. It addresses the problem of missing 3D consistency in world model generation from camera-guided video generation models. The proposed method uses video priors from a frozen Wan 2.1 model to generate a geometry-aware implicit 3D field using the proposed Integrated Reconstruction and Generation (IRG) Blocks that are based on bidirectional cross-attention. The model is trained on 180k video clips from real-world and synthetic data using a diffusion loss. The model is evaluated on the WorldScore static benchmark and demonstrates improvements over existing methods for consistency-based metrics.

### Strengths
The paper identifies a key problem with camera controlled video generation models and motivates the design of the proposed model with a coherent and well-justified design choice. The results are promising in terms of their main goal of improving consistency in the generated outputs. This is evident based on both qualitative and quantitative results. The ablation presented for the geometry branch support the claimed efficacy of the geometry branch to improve the consistency metrics. The robustness of the method to large camera motion is a strong result. The authors have clearly described the limitations of the method helping the reader to scope the contributions well. The details mentioned in the paper are sufficiently detailed for reproducing the method.

### Weaknesses
The paper motivated their world based on using it for AGI, which could require camera and object control. This is a major limitation of the method since the metrics are significantly lower than other methods such as WonderWorld. The evaluation is limited to static scene, even though world models in robotics setup would require handling dynamic scenes. The model is supervised using existing pre-trained models, which could be a limiting factor for generalization as the errors can propagate to the proposed trained model. Moreover, the paper claims that the latents can be useful for downstream tasks, but not much evidence was provided to support this claim.

Minor:
- Table 1 can have a common small and large sections instead of repeating the same information in each row.
- The captions for the figures (i.e. Figure 5) need more details to guide the reader for better understanding the results.

### Questions
- Is there a reason why the camera and object control is not significant? It might help to see a few examples in the paper to understand this better.

### Soundness
3

### Presentation
3

### Contribution
3
