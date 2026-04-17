# Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Videos inherently represent 2D projections of a dynamic 3D world. However, our analysis suggests that video diffusion models trained solely on raw video data often fail to capture meaningful geometric-aware structure in their learned representations. To bridge this gap between video diffusion models and the underlying 3D nature of the physical world, we propose Geometry Forcing, a simple yet effective method that encourages video diffusion models to internalize latent 3D representations. Our key insight is to guide the model’s intermediate representations toward geometry-aware structure by aligning them with features from a pretrained geometric foundation model. To this end, we introduce two complementary alignment objectives: Angular Alignment, which enforces directional consistency via cosine similarity, and Scale Alignment, which preserves scale-related information by regressing unnormalized geometric features from normalized diffusion representation. We evaluate Geometry Forcing on both camera view–conditioned and action-conditioned video generation tasks. Experimental results demonstrate that our method substantially improves visual quality and 3D consistency over the baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Geometry Forcing, a REPA-style feature alignment method to enhance the 3D consistency of existing video diffusion models (VDMs). Authors first observe that existing VDMs cannot readout consistent 3D representations (i.e., point maps) from their diffusion features. To address this, prior works either jointly predict the 3D modality, or leverage a structured 3D representation as guidance. Geometry Forcing instead proposes to align the internal features of the VDM with the representation of a 3D foundation model, VGGT. Fine-tuning pre-trained VDMs with such auxiliary loss significantly improves the temporal consistency of the generated videos, demonstrated by both quantitative and user study results.

### Strengths
- The idea is simple and intuitive. Adding a REPA loss with VGGT does not introduce any sophisticated architecture design, yet it is able to bake the 3D information into the model.
- The ablation study is very comprehensive, which clearly shows the effectiveness of each module. 
- The experimental results seem strong. Geometry Forcing outperforms baselines in most of the metrics.

### Weaknesses
1. As the author pointed out in the Limitation, VGGT is only trained on static scenes, and thus cannot be used to supervise VDM training on dynamic videos. Can authors discuss more on how to extend Geometry Forcing to dynamic videos as required by general text-to-video training?
2. Have the authors tried other 3D foundation models? For example, MonST3R [1] and CUT3R [2] as they can handle dynamic videos.
3. Have you tried training a video diffusion model from scratch? REPA shows that using representation alignment loss can greatly speed up convergence. I'm curious if applying both DINO and VGGT loss can further accelerate this. I understand that the computation cost is high, so showing some early training loss curve (with vs without Geometry Forcing loss) is enough (or describe it, if figures are not allowed in rebuttal).
4. I'd like to see a comparison with a baseline that uses explicit 3D memory, e.g., GEN3C [3] that uses reprojected point clouds as additional conditioning for the VDM. You do not need to apply all its components. I think you can do this:
- To generate frame `n`, run VGGT on frame `1, 2, ..., n-1` to reconstruct a point cloud of the scene, then render it to 2D images, and condition the VDM on it. This can also work in an autoregressive way.

[1] Zhang, Junyi, et al. "Monst3r: A simple approach for estimating geometry in the presence of motion." arXiv preprint arXiv:2410.03825 (2024).

[2] Wang, Qianqian, et al. "Continuous 3d perception model with persistent state." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[3] Ren, Xuanchi, et al. "Gen3c: 3d-informed world-consistent video generation with precise camera control." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
Besides the questions in Weaknesses, here are some minor questions:
1. Can you explain why VideoREPA achieves a significantly better RVE than Geometry Forcing, while being much worse in other metrics?
2. How do you implement VideoREPA? Is it doing REPA alignment loss with per-frame DINOv2 features? Then why is its result much worse than the "DINOv2 Only" entry in Tab.2?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Geometry Forcing (GF) that enhances the geometric consistency of video diffusion models by aligning their internal representations with those of a 3D foundation model (VGGT). GF introduces Angular Alignment and Scale Alignment. The method is integrated into standard autoregressive video diffusion training without requiring explicit 3D supervision. Experiments on RealEstate10K and Minecraft benchmarks show that GF improves Fréchet Video Distance (FVD), SSIM, and 3D consistency metrics (RPE/RVE) compared to state-of-the-art baselines such as DFoT, REPA, and VideoREPA. Ablation studies confirm the complementary role of Angular and Scale alignment, and user studies indicate perceptually improved scene consistency.

### Strengths
1. The paper is very well written, with clear figures illustrating motivation and results.

2. The paper proposes a new concept of “geometry forcing”, transferring geometric awareness into video diffusion models without requiring 3D ground-truth supervision. The dual-objective design (Angular and Scale alignment) is interesting, addressing optimization stability in cross-domain feature matching.

3. Experiments are extensive. Quantitative: Benchmarks include both 16- and 256-frame video generation, using perceptual and geometric metrics. Qualitative: Visual comparisons (360° rotations) convincingly show consistent viewpoint revisiting.

### Weaknesses
1. Scale of experiments is modest (16–256 frames, 256×256 resolution). The authors acknowledge this but it limits claims of scalability.

2. While the ablations cover alignment types and layer depths, computational cost (training overhead, memory footprint) is missing. Geometry alignment likely adds feature extraction and projection costs that may be nontrivial.

3. The method relies heavily on VGGT as a teacher. It’s unclear whether GF’s success depends on the specific 3D foundation model or generalizes to others (e.g., DUST3R, FLARE).

4. There is limited discussion of failure cases (e.g., ambiguous depth, reflective surfaces).

### Questions
1. Generality of the 3D teacher. How sensitive is GF to the choice of 3D foundation model? Would a weaker teacher (e.g., DUST3R) still yield benefits, or is VGGT’s strong geometry essential?

2. Does alignment at all layers or at multiple scales (spatial or temporal) offer further improvement, beyond the mid-level layer shown to be best?

3. What is the added training cost (e.g., % increase in FLOPs or wall-clock time) from Geometry Forcing, given the need to compute VGGT features?

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
This paper adapts the REPresentation Alignment (REPA) [1] into 3D domain. Specifically, it employs VGGT [2] as a 3D foundation model and aligns the intermediate features of a video diffusion model with features extracted from layers of VGGT. The authors propose two alignment objectives for this purpose: an angular alignment objective, which enforces cosine similarity between feature maps, and a scale alignment objective, which directly supervises the rescaled features of the diffusion model. Experiments demonstrates that combining these two objectives enhances the video diffusion model's 3D understanding, thereby improving geometric consistency in the generated results.

[1] Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think

[2] VGGT: Visual Geometry Grounded Transformer

### Strengths
* The core concept of aligning intermediate features of a generative model with those of a foundation model is a promising research direction that has shown success in other fields.
* The proposed method is simple yet effective. The experiments demonstrate improvements over the baseline models.
* The paper is well-written and easy to understand.

### Weaknesses
* My primary concern is the paper's limited novelty. The proposed method appears to be a straightforward adaptation of the 2D alignment technique from REPA. The core contribution seems to be replacing the DINO model (used in 2D REPA) with VGGT for the 3D case. As such, the work feels incremental and offers limited new conceptual insights.

* The method's applicability appears to be limited to static scenes, but this is not explicitly stated. The authors should clearly acknowledge that the current approach does not handle dynamic scenes or significant camera motion, which restricts its use to a narrow set of conditions. 

*  The performance of the VGGT feature extraction likely depends on the number of views used. I would encourage the authors to provide an ablation study demonstrating how model performance varies with different numbers of views adopted during training time.

### Questions
*  The matching process of VGGT seems computationally expensive, especially as the number of views increases, potentially making the training alignment prohibitively slow. To clarify this, could the authors report the concrete training time?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces Geometry Forcing, a method designed to help video diffusion models better capture the inherent 3D structure of real-world scenes. While standard video diffusion models trained on raw 2D video data often lack geometric understanding, Geometry Forcing addresses this by aligning the model’s internal representations with features from a geometric foundation model. It employs two key alignment strategies: Angular Alignment, which enforces directional consistency through cosine similarity, and Scale Alignment, which preserves scale information via feature regression. Experiments on camera view–conditioned and action-conditioned video generation show that Geometry Forcing significantly enhances both visual quality and 3D consistency, outperforming existing baselines.

### Strengths
1. The paper is clearly written and well-organized.
2. The core idea is simple yet effective — aligning the internal representations of video diffusion models with features from a geometric foundation model.
3. Experimental results demonstrate strong performance, showing notable improvements in geometric consistency and long-term temporal coherence compared to baseline methods.

### Weaknesses
The training objectives of the diffusion model and the VGGT differ fundamentally. The diffusion model is designed to learn noise or velocity in a progressive manner—its target lies in the intermediate denoising process rather than the final outcome. In contrast, VGGT is result-oriented, directly learning to predict the final geometry. Although the experimental results appear promising, theoretically the learning targets are not of the same nature and may even be somewhat conflicting. It remains unclear how the proposed alignment between these two objectives effectively works in practice. Could the authors provide more theoretical or empirical justification for this compatibility?
The motivations and formulations of Angular Alignment and Scale Alignment are insufficiently explained. The directional correspondence between the hidden states of the diffusion model and the geometric features, as well as the scale differences across models, are both vague. It would be helpful to clarify how these factors influence the final generation quality.

The base model description is also unclear. The paper mentions “a U-ViT backbone for video generation,” but does not specify which model this refers to or provide an appropriate citation.

Including explicit geometry-based video generation methods as comparison baselines would strengthen the evaluation and provide more persuasive evidence of the proposed method’s effectiveness.

Additionally, there are citation errors, such as the one noted around line 290.

Finally, the paper lacks qualitative ablation studies that isolate the effects of Angular Alignment and Scale Alignment. Without such analyses, it is difficult to assess the actual contributions of each component to the overall improvement.

### Questions
1.The diffusion model and VGGT have different learning objectives — one is progressive (learning noise or velocity), while the other is result-oriented. How can the proposed alignment between these fundamentally different targets work effectively?


2.The motivations and mechanisms of Angular Alignment and Scale Alignment are unclear. How exactly do these objectives influence the final video generation quality?


3.What specific model does the “U-ViT backbone for video generation” refer to? Could the authors provide more details or a proper citation?

4. Could the authors include explicit geometry-based video generation methods as additional baselines to make the comparison more convincing?


5.The paper lacks qualitative ablation studies showing the individual effects of Angular and Scale Alignment. Can the authors provide such qualitative analyses?

### Soundness
3

### Presentation
3

### Contribution
3
