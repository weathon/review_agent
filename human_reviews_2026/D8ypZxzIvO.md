# PhyCAGE: Physically Constrained Compositional 3D Asset Generation from a Single Image

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
We present PhyCAGE, the first approach for physically constrained compositional 3D asset generation from a single image. Given an input image, we first generate consistent multi-view images for components of the assets. These images are then fitted with 3D Gaussian Splatting representations. To ensure that the Gaussians representing objects are physically compatible with each other, we introduce a Physical Simulation-Enhanced Score Distillation Sampling (PSE-SDS) technique to further optimize the positions of the Gaussians. It is achieved by setting the gradient of the SDS loss as the initial velocity of the physical simulation, allowing the simulator to act as a physics-guided optimizer that progressively corrects the Gaussians' positions to a physically compatible state. Experimental results demonstrate that the proposed method can generate physically plausible compositional 3D assets given a single image.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes PhyCAGE, a novel framework for generating physically constrained compositional 3D assets from a single 2D image.
The system integrates multi-view image generation, 3D Gaussian Splatting (3DGS) reconstruction, and a Physical Simulation-Enhanced Score Distillation Sampling (PSE-SDS) process. Unlike prior SDS-based or compositional 3D methods, PhyCAGE introduces a differentiable physical simulation (via MPM) that uses the SDS loss gradient as the initial velocity, enabling physics-guided correction of inter-object penetrations. Experiments on ComboVerse and other benchmarks demonstrate that PhyCAGE improves physical plausibility, visual quality, and component disentanglement.

### Strengths
1. Novel integration of MPM-based physics into SDS optimization.
2. Physically plausible compositional reconstruction, effectively addressing object penetration and stability.
3. Quantitative improvements in PSNR, CLIP score, and penetration rate compared to strong baselines (Part123, ComboVerse).
4. Extensive ablation and runtime analysis, demonstrating consistent improvements.

### Weaknesses
1. Limited novelty in theoretical contribution.
The PSE-SDS approach combines existing elements (SDS, MPM) rather than introducing a fundamentally new learning paradigm. The innovation is mainly in integration and practical application.

2. Heuristic parameter dependence.
The method uses several empirically chosen constants (e.g., λ₁–λ₃, 500 simulation steps, timestep decay) without sensitivity analysis. Stability under different simulation conditions is not examined, which may limit reproducibility.

3. Scalability and computational cost.
The runtime table shows PhyCAGE is slower than Part123, approaching ComboVerse’s cost (≈734s). Given its “single-image” input goal, the computational burden may hinder deployment.

4. Assumption of accurate segmentation and inpainting.
The method relies heavily on Grounded-SAM and SyncDreamer outputs. Errors in segmentation or multi-view synthesis propagate to the physics simulation, potentially breaking physical consistency.

5. Scope limitations.
The current pipeline handles only two-object interactions and assumes static scenes. Extending to multi-object or dynamic scenes would require further methodological development.

### Questions
1. How sensitive is PSE-SDS to the accuracy of Grounded-SAM segmentation or the inpainting model?
2. What physical parameters (mass, stiffness) are assumed in the MPM simulation? Could they be learned automatically?
3. How would the method behave in dynamic or non-rigid interactions (e.g., cloth over soft body)?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method named PhyCAGE, which supports physically-aware compositional 3D generation from a single input image. The pipeline first employs Grounded-SAM to segment the input image and then uses an inpainting process to recover occluded regions. Subsequently, a multi-view 3D generation method is adopted to generate 3D assets for each segmented part. Finally, the authors introduce a physics simulation–enhanced Score Distillation Sampling (SDS) strategy to optimize the spatial arrangement of the parts, resulting in a physically consistent 3D object.

### Strengths
1. The paper introduces a physically-based Score Distillation Sampling (SDS) strategy to ensure the physical correctness of the generated 3D compositions, which is a novel direction that has been rarely explored in existing methods.

2. The writing is clear, concise, and easy to follow, making the paper accessible and understandable to readers.

### Weaknesses
1. The paper mainly focuses on SDS-based methods and lacks discussion or comparison with recent native 3D part generation approaches, such as PartCrafter [1], which also tackles part-based 3D generation.

2. The SDS optimization is time-consuming and unstable, often requiring manual intervention and filtering to obtain valid results. This raises concerns about the method’s efficiency and robustness.


[1] PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers

### Questions
Based on the identified weaknesses, could the authors provide comparisons with recent native 3D part generation methods, such as PartCrafter, to better contextualize their contributions? In addition, could the authors discuss the potential of incorporating physical constraints or simulations into native 3D generation frameworks, and how such integration might enhance compositional accuracy and physical plausibility?

### Soundness
3

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
5

### Summary
This work proposes a novel method for compositional 3D asset generation. It first produces consistent multi-view images for each object component, reconstructs them using 3D Gaussian Splatting, and then refines their geometry through a Physical Simulation-Enhanced Score Distillation Sampling (PSE-SDS) process. This technique treats the SDS loss gradient as the initial velocity in a physical simulation, ensuring that objects interact realistically and avoid interpenetration.

### Strengths
1. The integration of physical simulation with score distillation sampling is innovative, enhancing the realism of generated 3D assets. I think this idea is interesting. 
2. The topic is under-explored while important for applications in gaming, virtual reality, and digital content creation.

### Weaknesses
1. The foundation techniques, such as multi-view image generation and 3D Gaussian Splatting, are a bit outdated compared to the latest advancements in 3D generation. The authors should compare their method with more recent approaches in compositional 3D generation, such as PartCrafter and OmniPart. The author should justify their proposed physics-based, gaussian-splatting-based method has advantages over these recent methods. 
2. The proposed method focuses on two-part compositions, which limits its applicability to more complex multi-component objects. Although the authors propose an iterative approach for handling more than two parts, this is not thoroughly explored or validated in the experiments. The authors should provide more experiments on multi-part compositions to demonstrate the scalability of their method. What's the maximum number of parts the method can handle effectively? 
3. The SDS method is a optimization-based approach, which is generally slower than recent diffusion-based 3D generation methods. The authors should provide a comparison of generation speed and quality with diffusion-based methods to highlight the trade-offs involved in their approach.

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces PhyCAGE, a method to generate compositional 3D assets from a single image without physical penetration. Its core contribution is PSE-SDS, an optimization technique where the SDS loss gradient is used as an initial velocity for a physical simulator, and optimizing the center (position) per 3DGS to a physically plausible state to solve the penetration issue. It demonstrates improved physical plausibility compared to Part123 and ComboVerse baselines on 45 images-to-3D generation.

### Strengths
+ This paper addresses a significant research problem of generating physically plausible, compositional 3D assets.

+ The proposed PSE-SDS is overall a novel optimization technique which avoiding the need for a large-scale, paired 3D training dataset.

### Weaknesses
+ My main concern lies in the paper's core abstraction: treating 3DGS centers ($\mu$) as physical particles for an MPM simulator. 3DGS is a volumetric rendering representation lacking a defined surface. Optimizing Gaussian centers does not guarantee a non-penetrating state at the object's actual boundary. This conceptual gap is amplified by the negligible PSE-SDS loss weight ($\lambda_3 = 1e-5$), which is too negligible compared to image loss. Therefore, I remain my concern if it can really work to solve the "penetration" issue as claimed in the paper.

+ The method is architecturally limited to pairwise composition, requiring a new design for real-world, multi-component scenes. This pairwise limitation is exacerbated by a physically arbitrary optimization scheme. The method designates objects as rigid ($G_2$) or non-rigid ($G_1$) based on 2D occlusion (L307-311), not physical properties. In the "frog wearing a sweater" example, this means the method freezes the sweater and deforms the frog to fix collisions. This is physically backward, as a soft sweater should deform to fit a mostly-rigid frog, not the reverse.

+ The paper overlooks comparisons to recent part-aware generative models (e.g., SAMPart3D, PartGen[1]). These methods are trained to generate 3D objects as a collection of distinct parts, and can avoid penetration by design. Including such baselines would strengthen the evaluation.

[1] Chen, Minghao, et al. "Partgen: Part-level 3d generation and reconstruction with multi-view diffusion models." CVPR 2025.

+ The paper's comparison to Part123 is problematic. The authors dismiss a penetration rate comparison with Part123 as "not meaningful" because it produces a single mesh (L411-415). Yet, the proposed method uses Part123's NeuS output as the geometric target for its foreground object. This is a contradictory stance, as the paper's main goal is to reduce penetration, and even their "single mesh" can still achieve this goal. A more persuasive analysis of physical plausibility beyond this single metric is needed.

+ Minor concern: The geometry of the background object ($G_1$) is initialized by first "hallucinating" its occluded 2D appearance via inpainting, and then performing a 3D reconstruction of that hallucination. The physics optimization is only designed to resolve penetrations, not to correct fundamental geometric flaws (e.g., a malformed back) that may be introduced by this uncertain initialization pipeline.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
