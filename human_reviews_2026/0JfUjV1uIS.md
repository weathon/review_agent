# Geometric Image Editing via Effects-Sensitive In-Context Inpainting with Diffusion Transformers

- Decision: Accept (Poster)
- Scores: 4, 4, 8

## Abstract
Recent advances in diffusion models have significantly improved image editing. However, challenges persist in handling geometric transformations, such as translation, rotation, and scaling, particularly in complex scenes. Existing approaches suffer from two main limitations: (1) difficulty in achieving accurate geometric editing of object translation, rotation, and scaling; (2) inadequate modeling of intricate lighting and shadow effects, leading to unrealistic results. To address these issues, we propose GeoEdit, a framework that leverages in-context generation through a diffusion transformer module, which integrates geometric transformations for precise object edits. Moreover, we introduce Effects-Sensitive Attention, which enhances the modeling of intricate lighting and shadow effects for improved realism. To further support training, we construct RS-Objects, a large-scale geometric editing dataset containing over 120,000 high-quality image pairs, enabling the model to learn precise geometric editing while generating realistic lighting and shadows. Extensive experiments on public benchmarks demonstrate that GeoEdit consistently outperforms state-of-the-art methods in terms of visual quality, geometric accuracy, and realism.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GeoEdit, a diffusion-transformer–based framework for geometric image editing that can accurately perform translation, rotation, and scaling of objects within complex scenes while preserving realistic lighting and shadow effects.
The key innovations are:

* Geometric Transformation Module – utilizes 3D reconstruction (via Hunyuan3D) to apply parametric transformations with precise control.

* Effects-Sensitive Attention (ESA) – a soft attention modulation designed to better model lighting and shadow consistency.

* RS-Objects Dataset – a large-scale (120K samples) dataset combining rendered and synthetic data to support geometric editing training.

GeoEdit is built upon the FLUX.1-Fill DiT backbone and shows superior results on both 2D and 3D edit benchmarks, achieving better FID, DINOv2, and consistency metrics than prior works such as FreeFine, GeoDiffuser, and DiffusionHandles.

### Strengths
* Well-written and structured paper with clear motivation and consistent organization.

* Strong experimental performance on both quantitative and qualitative metrics across multiple tasks.

* Comprehensive dataset (RS-Objects) with rigorous construction and filtering criteria; could benefit the broader community.

* Practical application value in realistic scene editing and geometric transformation control.

### Weaknesses
* The proposed ESA module appears conceptually similar to a soft attention bias, and its correlation with lighting effects is not convincingly demonstrated.

* The method’s heavy reliance on external geometry models (e.g., Hunyuan-3D) for 3D reconstruction compromises its originality and self-containment. It also remains unclear how GeoEdit performs when alternative geometry backbones are employed.

* The model is built upon the Flux backbone, while several baselines are not, raising concerns about the fairness of comparisons.

* The potential overfitting to the RS-Objects synthetic dataset may further limit the model’s generalization to real-world scenarios.

* In some qualitative examples (e.g., Fig. 7), objects like bottles and razors do not fully align with the target sketch positions—they shift slightly, suggesting weak spatial control.

### Questions
* The technical novelty requires further justification

* How sensitive is ESA’s improvement to the choice of the scaling factor α in different lighting conditions?

* Can GeoEdit function with other external 3D priors models (other than Hunyuan-3D), or without external 3D priors?

* Can comparisons be made using the same Flux backbone?

* What are the model’s failure modes—e.g., in cluttered or occluded scenes?

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
4

### Summary
The paper proposes GeoEdit, a diffusion-transformer–based framework for geometric image editing—tasks involving object translation, rotation, and scaling while maintaining realism (e.g., lighting and shadow coherence).  GeoEdit outperforms prior works such as DragDiffusion, GeoDiffuser, and FreeFine on FID, DINOv2 distance, and geometric consistency metrics across both 2D and 3D editing tasks. Ablations show that ESA and dataset composition are critical for performance.

### Strengths
1. The paper presents a clearly structured geometric editing pipeline with well-defined, reproducible steps. Each transformation—translation, rotation, and scaling—is handled through explicit procedures. The detailed description of these steps provides strong methodological clarity and makes the approach readily reproducible.

2. The proposed RS-Objects dataset is thoughtfully designed to align with the objectives of geometric image editing. It employs a two-stage rendering-to-synthesis pipeline that produces roughly 120,000 image–mask pairs. The process combines high-quality Blender-rendered samples, mesh-based scene generation, large-scale LoRA-driven synthesis (pre-filtered from about 800,000 candidates), and a final human filtering stage to ensure coherence and illumination realism. This carefully engineered dataset directly supports the training objectives of GeoEdit and provides a valuable resource for future research on geometry-aware image editing.

### Weaknesses
1.	Theoretical Contribution and Clarity of ESA
Theorem 3.1 offers a limited theoretical contribution. Since the hard-modulated attention focuses exclusively on insertion tokens, its KL divergence from the ideal attention map is trivially infinite, and showing that the ESA variant achieves a smaller divergence is therefore not a particularly informative result. The theorem appears unnecessary in its current form. The central issue is not the inequality itself, but the rationale behind defining the similarity for edited tokens in ESA as $q_i k_j^\top / \sqrt{d} + \delta$. The paper should clarify the intuition for introducing this additive bias $\delta$, its relationship to the observed attention patterns, and the impact of the hyperparameter $\alpha$ that scales it. Without such discussion, the theoretical section feels more decorative than explanatory.
2.	Architectural Substitution and Fairness of Comparison
The implementation introduces an architectural change by replacing the T5 text encoder with SigLIP and fine-tuning the model via LoRA on a FLUX.1-Fill DiT backbone.
(a) The description “SigLIP image encoder for textual inputs” is ambiguous and requires clarification. For instance, Figure 2 does not make it clear how SigLIP features and mask references are integrated into the model architecture or how textual prompts, if any, are represented.
(b) Moreover, several baselines appear not to have been re-run on this modified backbone. Without controlling for differences in encoder strength and conditioning design, part of GeoEdit’s reported performance gain might arise from the stronger base model rather than from the proposed ESA or geometric modules. A discussion or ablation controlling for these factors would make the comparisons more convincing.

### Questions
see weakness

### Soundness
3

### Presentation
3

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
This paper proposes GeoEdit, a diffusion-transformer–based framework for geometric image editing.
Two key components are introduced: (1)Geometric Transformation module – performs translation, rotation, and scaling in a 3D-aware way via object reconstruction. (2)Effects-Sensitive Attention (ESA) – a soft guidance mechanism that modulates attention logits to preserve lighting and shadow realism, theoretically supported by a KL-divergence analysis.

The authors also construct RS-Objects, a dataset of 120k rendered + synthetic image pairs designed for geometric transformations and visual-effects learning.Extensive experiments on GeoBench show consistent quantitative and qualitative improvements over strong baselines such as FreeFine, GeoDiffuser, and Diffusion Handles.

### Strengths
1 Clear motivation and problem definition – focuses on geometric (translation / rotation / scaling) image editing, which remains under-explored compared to semantic or text-guided edits.

2 Comprehensive dataset pipeline – the RS-Objects dataset seems carefully designed (render + AIGC + human), addressing a genuine data gap.

3 Strong experiments – covers both 2D and 3D edits, reports seven metrics, and includes ablations and a user study

4 Readable writing and solid figures

### Weaknesses
1 Dataset authenticity & reproducibility. whether any real photographs with ground-truth geometric edits exist for validation. Public release status is unclear

2 Many baselines are test-time or training-free. The proposed method is training-based with a large custom dataset, so the comparison is not entirely apples-to-apples. how much overhead does ESA add versus standard DiT inpainting?

3 Limitations are not discussed, what is the thing that GeoEdit can not achieve? Discussing these aspects is crucial for guiding future research and fair benchmarking.

### Questions
The proposed ESA biases attention distribution statistically rather than modeling physical light transport. So, can GeoEdit guarantee physically correct shadow direction, reflection geometry, or color temperature consistency, especially when global illumination changes?

### Soundness
3

### Presentation
3

### Contribution
3
