# Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
This paper presents Omni-View, which extends the unified multimodal understanding and generation to 3D scenes based on multiview images, exploring the principle that ``generation facilitates understanding". Consisting of understanding model, texture module, and geometry module, Omni-View jointly models scene understanding, novel view synthesis, and geometry estimation, enabling synergistic interaction between 3D scene understanding and generation tasks. By design, it leverages the spatiotemporal modeling capabilities of its texture module responsible for appearance synthesis, alongside the explicit geometric constraints provided by its dedicated geometry module, thereby enriching the model's holistic understanding of 3D scenes. Trained with a two-stage strategy, Omni-View achieves a state-of-the-art score of 55.4 on the VSI-Bench benchmark, outperforming existing specialized 3D understanding models, while simultaneously delivering strong performance in both novel view synthesis and 3D scene generation. The code and pretrained models are open-sourced at https://github.com/AIDC-AI/Omni-View .

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Omni-View, a unified 3D scene understanding and generation model based on multi-view imagery, exploring the principle of "generation promotes understanding." The model, composed of an understanding module, a texture module, and a geometry module, jointly models scene understanding, novel view synthesis, and geometry estimation. Using a two-stage training strategy, Omni-View achieved a state-of-the-art score of 55.4 on the VSI-Bench, surpassing existing dedicated 3D understanding models while also performing well in novel view synthesis and scene generation tasks.

### Strengths
Unified modeling is novel: This is the first systematic exploration of the "generation-driven understanding" mechanism in 3D scenes, which is both inspiring and forward-looking.

Rational modular design: The generation module is split into texture and geometry components, modeling appearance and structure respectively, effectively improving understanding capabilities.

Effective training strategy: The two-stage training (unified training + generation fine-tuning) balances understanding and generation performance, and the D2S mechanism improves robustness.

Comprehensive experiments: The effectiveness of the method is verified on multiple 3D understanding, spatial reasoning, and generation tasks, with results significantly outperforming existing unified models.

No 3D input required: Relying solely on multi-view images, this improves the model's practicality and generalization capabilities.

### Weaknesses
Weak theoretical analysis: Although "generation promotes understanding" has been proposed, there is a lack of theoretical or interpretable analysis of its underlying mechanisms.

Generation quality still has room for improvement: Despite leading in PSNR/SSIM, inter-frame consistency under large viewpoint variations remains suboptimal (see Appendix visualization).

The geometry module relies on synthetic data: The depth map is synthesized by Voyager, which may limit the realism and accuracy of geometry predictions.

Limited long sequence generation capability: The model currently does not support long sequence scene generation, limiting its application in open-world scenarios.

There is still a gap compared to state-of-the-art dedicated models: In particular, in the 3D grounding task, there is still a significant gap compared to methods that rely on 3D input.Strengthen theoretical analysis or visual explanation of the "generation promotes understanding" mechanism;

Compare with state-of-the-art methods on more 3D grounding tasks and analyze the sources of the gap.

### Questions
Strengthen theoretical analysis or visual explanation of the "generation promotes understanding" mechanism;

Compare with state-of-the-art methods on more 3D grounding tasks and analyze the sources of the gap.

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
3

### Summary
This paper introduces Omni-View, a unified model for 3D scene understanding and generation that explicitly investigates the hypothesis that “generation facilitates understanding.” The contributions of the paper are:
1. Unified architecture integrating 3D scene understanding and generation, composed of three main components:
1.1 Understanding model (for spatial reasoning and QA)
1.2 Texture module (for novel view synthesis)
1.3 Geometry module (for depth and pose estimation)
2. Proposed a novel two-stage training strategy:
2.1 Jointly trains understanding and generation to encourage mutual benefits through geometry and spatiotemporal modeling.
2.2 Fine-tunes generation with RGB-Depth-Pose joint learning for better geometric consistency.
3. Empirical validation showing state-of-the-art (SOTA) performance on the VSI-Bench (score 55.4), outperforming both specialized and unified 3D models in reasoning tasks.
Overall, Omni-View demonstrates how generative modeling (novel view synthesis, geometry estimation) can enhance 3D reasoning, localization, and understanding—a conceptually elegant and empirically supported contribution.

### Strengths
1. Clear intuition and solid empirical validation.
The paper builds upon a clear and intuitive idea — that generation can facilitate understanding — and the overall logic is easy to follow. Quantitative results across multiple benchmarks convincingly demonstrate the benefits of the proposed design, especially in spatial reasoning and novel view synthesis.
2. Architectural innovation.
By decomposing the generation process into texture and geometry modules, the authors present a meaningful and modular architecture that captures both appearance and structure. This decomposition aligns well with human visual reasoning and can be viewed as an innovative contribution for the community.
3. Comprehensive ablation studies.
The ablation results thoroughly verify the contributions of the proposed contributions. These analyses effectively demonstrate that each component contributes to the final understanding and reasoning performance.

### Weaknesses
1. The qualitative results in the appendix are sparse, and there are no depth estimation visualizations or broader test cases. This makes it difficult to verify the model’s generalization and effectiveness beyond the reported metrics. For instance, the quality and consistency of metric-scale prediction from the geometry module remain uncertain — the reported results could be influenced by selective visualization or data bias, since the paper lacks convincing examples that demonstrate accurate geometric reasoning across varied real-world scenes.
2. The technical details provided for training, implementation, and comparison setups are relatively limited. Without clearer supplementary material (e.g., dataset statistics, architecture specifics, or convergence behavior), it is challenging to fully reproduce the reported results or assess robustness under different conditions.
3. The absence of released code or live demonstrations restricts the ability of other researchers to validate or extend this work. Although acceptable for review, the paper would be strengthened by open-sourcing its checkpoints or providing additional evaluation on long-range world generation and 3D visual grounding.

### Questions
1. Could the authors provide qualitative results of the geometry module’s depth estimation, camera pose estimation with metric-scale? Without any depth estimation results or broader test cases, it is difficult to assess whether the geometry module truly learns meaningful 3D structure rather than overfitting to training priors, without any generalizability.
2. How well does Omni-View generalize to unseen or real-world multi-view scenes, rather than the well captured ones in the appendix? Have the authors tested its performance to verify the robustness and effectiveness of the learned spatial reasoning?
3. Could the authors clarify key implementation and training details—such as dataset splits, optimizer configurations, training epochs, and computational cost—to ensure reproducibility? Including more specifics or releasing supplementary materials would improve transparency.

### Soundness
3

### Presentation
3

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
- This paper proposes Omni-View, a unified multimodal model that jointly performs 3D scene understanding, geometry estimation, and novel view synthesis from multiview images, based on the principle that generation facilitates understanding.

- This paper introduces a dual-path architecture consisting of a texture module (appearance generation) and a geometry module (depth/pose estimation), enabling bidirectional synergy between generative and understanding tasks.

- This paper uses a two-stage training strategy where joint training enhances understanding via generative signals, followed by refinement for high-quality 3D scene generation; achieves state-of-the-art performance on VSI-Bench and strong results in NVS and 3D Q&A.

### Strengths
- This paper demonstrates that generative 3D tasks (novel view synthesis, geometry estimation) can actively enhance 3D scene understanding, rather than being separate objectives.

- This paper has a unified architecture for 3D reasoning, with separate texture and geometry modules allow complementary learning of appearance and spatial structure, leading to better localization, spatial reasoning, and depth-aware Q&A.

- This paper outperforms specialized models in 3D understanding benchmarks while maintaining competitive NVS and scene generation performance, closing the gap between multimodal understanding and 3D generative models.

- A systematically organized evaluation and ablation study would strengthen the credibility of this paper.

### Weaknesses
- It would be beneficial to include a diagram that more precisely illustrates the functionality of each module and the architecture, compared to the current version.

- Additionally, visualizing 3D scene understanding / spatial reasoning / NVS from a single view as a video could also be an effective way to present the capabilities of the system.

- Is there a reason why you refer to Texture Module and Geometry Module in the equations (e.g., (eq. 1), (eq. 2)) without using italics? Also, I believe writing them as TextureModule and GeometryModule (without a space) would improve readability and be more suitable for mathematical notation.

### Questions
Mentioned in the weaknesses

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
The paper presents Omni-View, a unified model for 3D scene understanding and generation from multiview images that tests the “generation facilitates understanding” hypothesis. Built on Bagel, it splits generation into a texture module (flow matching with Plücker pose encoding, autoregressive NVS) and a geometry module (depth and camera pose via flow matching with cross-attention to understanding features). A two-stage training recipe first jointly trains understanding/texture/geometry with a dense-to-sparse curriculum, then fine-tunes generation with RGB-Depth-Pose joint learning. Omni-View achieves SOTA on VSI-Bench (55.4), improves QA/localization versus unified baselines without 3D inputs, and delivers strong NVS/scene generation results on Re10k.

### Strengths
- The paper presents a unified 3D understanding–generation framework that cleanly separates texture and geometry, a simple yet original design that operationalizes “generation facilitates understanding.”
- The two-stage recipe with dense-to-sparse curriculum and autoregressive NVS is well-motivated, technically sound, and shows careful loss design and gradient routing to benefit the understanding model.
- Writing is clear and structured, with concrete training details, datasets, metrics, and ablations that isolate the contribution of each module and training choice.
- The empirical significance is strong, with SOTA results on VSI-Bench and competitive 3D QA/localization, plus solid NVS and scene generation metrics, demonstrating broad impact across 3D reasoning and generation.

### Weaknesses
- Limited novelty relative to prior unified frameworks (Bagel, VILA-U, BLIP3o, Harmon)  
The core idea of leveraging generation to aid understanding has precedents in 2D unified models and recent 3D works that inject reconstruction priors (e.g., Ross3D; VG-LLM/Spatial-MLLM via VGGT features). The split into texture vs. geometry resembles established “appearance vs. structure” decouplings in 3D pipelines (e.g., ViewCrafter, Voyager). Clarify what is fundamentally new beyond integrating these pieces within Bagel, and compare to a “single-branch with multi-heads” backbone.

- Ambiguity in camera control and absolute metric grounding   
The paper reports strong perceptual metrics but acknowledges difficulty in precise camera control and absolute depth scale. Because the gains on VSI categories like Abs. Dist. hinge on metric grounding, add analyses: scale consistency across scenes, depth-scale calibration via known baselines, and camera-pose accuracy vs. ground truth under diverse motions.

- Dataset overlap and generalization concerns  
Though the authors state they avoid using understanding images for generation training, several datasets share scene domains with Re10k-like indoor content, risking leakage of priors. Please report cross-dataset generalization (e.g., ScanNet -> Replica, RealEstate10K -> ACID/CO3D subsets) to support robustness claims.

- Incomplete ablations on design choices and routing  
The geometry module conditions only on the last-layer texture latent and uses cross-attention to the understanding model. Test alternatives: multi-scale latents, earlier-layer features, and gating that controls gradient flow to avoid potential interference. Provide compute/latency breakdowns for stage 1 vs. stage 2, and show sensitivity to $\lambda_{geo}$, pose-query design, and Plücker vs. other pose encodings.

### Questions
- Clarify the novelty beyond architectural decoupling: In what ways is the texture/geometry split more than a clean engineering separation compared to prior “appearance vs. structure” decouplings (e.g., ViewCrafter, Voyager) and unified frameworks (BAGEL, VILA-U, BLIP3o, Harmon)? Could you provide a controlled comparison to a single-branch generator with two prediction heads (texture, geometry) at equal parameter count?
- The ablations show AR improves spatiotemporal reasoning. Can you report exposure-bias analyses at inference time, e.g., teacher-forcing vs. free-running rollouts? Does diffusion forcing mitigate compounding errors, and how does performance vary with rollout length (8/16/32 frames)? Have you tested scheduled sampling or token-level AR only on camera poses while keeping texture bidirectional?
- Provide qualitative and quantitative failure analyses: cases where geometry improves understanding but harms texture fidelity (and vice versa), per-category VSI error tied to pose/depth errors, and sensitivity to large viewpoint changes where you noted inconsistencies. Would integrating a small, explicit 3D proxy (e.g., Gaussians or sparse point clouds) at training but not inference close these gaps?

### Soundness
3

### Presentation
3

### Contribution
3
