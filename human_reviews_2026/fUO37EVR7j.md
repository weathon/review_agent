# RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
In this paper, we propose a 3D asset-referenced diffusion model for image generation, exploring how to integrate 3D assets into image diffusion models. Existing reference-based image generation methods leverage large-scale pretrained diffusion models and demonstrate strong capability in generating diverse images conditioned on a single reference image. However, these methods are limited to single-image references and cannot leverage 3D assets, constraining their practical versatility. To address this gap, we present a cross-domain diffusion model with dual-branch perception that leverages multi-view RGB images and point maps of 3D assets to jointly model their colors and canonical-space coordinates, achieving precise consistency between generated images and the 3D references. Our spatially aligned dual-branch generation architecture and domain-decoupled generation mechanism ensure the simultaneous generation of two spatially aligned but content-disentangled outputs, RGB images and point maps, linking 2D image attributes with 3D asset attributes. Experiments show that our approach effectively uses 3D assets as references to produce images consistent with the given assets, opening new possibilities for combining diffusion models with 3D content creation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents RefAny3D, a diffusion-based framework for image generation conditioned directly on 3D assets, enabling consistent alignment between generated images and 3D geometry.
Using multi-view renderings of the 3D asset along with their point maps as conditioning inputs, RefAny3D introduces a dual-branch diffusion architecture that jointly predicts RGB images and point maps, thereby establishing pixel-level correspondences between texture and geometry.
The model employs Shared Positional Encoding (SPE) for spatial correspondence between rgb image and point map, and combines Domain-specific LoRA with Text-agnostic Attention to decouple texture and geometry domains.
For training, the authors construct a pose-aligned dataset derived from Subjects200k using Grounding DINO, Hunyuan3D and Foundation Pose.
Experimental results demonstrate consistent improvements over existing baselines on CLIP, DINO, GIM metrics, and GPT-based evaluations, particularly in maintaining geometric and textural consistency.

### Strengths
1. Novelty lies in using 3D assets as direct conditioning for diffusion models, addressing a key gap in 2D reference-based generation by ensuring geometric consistency.
2. The dual-branch architecture is well-motivated and technically sound . Claims are strongly supported by comprehensive experiments and ablations.
3. The paper is clearly written, well-organized, and easy to follow. The proposed architecture is well-illustrated, and component functions are explicitly defined .
4. This work offers an effective and natural solution to the major challenge of spatial consistency in reference-based generation. The joint modeling of RGB and point maps significantly enhances 3D awareness, representing an important contribution to the field.

### Weaknesses
1. Missing "Render-and-Edit" Baseline: The introduction mentions a straightforward "render-and-edit" alternative, noting its foreground-background inconsistency issues . However, this intuitive baseline was not included in the experimental comparisons . Empirically demonstrating RefAny3D's superiority over this method would significantly strengthen the paper's claims.
2. Dataset Transparency and Reproducibility: While the data construction pipeline is detailed , key statistics about the final dataset (e.g., scale, category coverage) and its release status are omitted. Providing these details is crucial for transparency and enabling reproducibility.
3. Lack of Failure Analysis: The paper admits poor performance on non-rigid objects due to dataset limitations  but provides no qualitative examples or analysis. It is unclear if this is purely a data issue or an architectural limitation (e.g., rigid spatial alignment). Including such an analysis would make the work more complete.

### Questions
The paper sets the number of reference views to N = 8.
Have the authors tested other values (e.g., 4, 12, 16)?
How sensitive is the performance to the number and distribution of viewpoints?
Understanding this sensitivity would clarify the scalability and generalization of the approach.

### Soundness
3

### Presentation
4

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
This paper proposes RefAny3D, a 3D asset-referenced diffusion framework for image generation. The model jointly synthesizes RGB images and point maps through a spatially aligned dual-branch architecture to ensure geometry–texture consistency. It further introduces domain-specific LoRA and text-agnostic attention to decouple visual and structural domains. Experiments show clear improvements over 2D reference-based baselines in geometric fidelity and visual quality.

### Strengths
1. The paper is clearly structured and well-written, with smooth logical flow from motivation to methodology and results.
2. The work introduces a novel 3D asset-referenced diffusion framework that bridges the gap between 2D reference-based generation and 3D-aware synthesis.
3. The experimental section is comprehensive, including qualitative, quantitative, and ablation studies that convincingly demonstrate the superiority of the method.

### Weaknesses
1. Motivation. The paper employs 3D assets as conditioning inputs to ensure geometry–texture consistency; however, generating 2D images does not inherently require multi-view conditioning. The authors should clarify the motivation for introducing 3D asset-based conditioning in this context.
2. Task Definition. Given that the 3D asset is already available, there exist simpler approaches to achieve similar results—for example, rendering the desired viewpoint as a conditioning image and feeding it into an inpainting or editing model. The authors should justify why their proposed framework is necessary compared to these straightforward alternatives.
3. Controllability. The proposed method lacks explicit control over the camera viewpoint of the generated image, which seems inconsistent with the premise of conditioning on a 3D asset. The authors are encouraged to discuss this limitation and explain how viewpoint control could be incorporated.
4. Missing References. The Subject-Driven Generation section in the related work only covers a group of image-guided generation methods, where several related 3D-guided generation methods are missed, such as:
- Wu R, Liu R, Vondrick C, et al. Sin3dm: Learning a diffusion model from a single 3d textured shape
- Wu R, Zheng C. Learning to generate 3d shapes from a single example
- Wang Z, Wang T, Hancke G, et al. Themestation: Generating theme-aware 3d assets from few exemplars
- Wang Z, Wang T, He Z, et al. Phidias: A generative model for creating 3d content from text, image, and 3d conditions with reference-augmented diffusion

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies how to synthesize photorealistic images of a specific 3D object given the 3D object model, a straightforward and trivial problem given the current literature. The 3D object is first rendered to multi-view RGB images and point maps encoding the per-view geometry. Then, the stylized image is generated simply via conditioning on the rendered images and point maps.

### Strengths
Product-wise, this task is very useful for commercializing generative models for rendering different photos of specific object products, mostly like a diffusion model shader.

### Weaknesses
There is too little technical contribution in this paper. Although the pipeline works, it is a straightforward engineering pipeline. I believe this is very practical for industry and product applications, but far below the standard of an ICLR paper.

### Questions
I only have one simple question, why is this method novel?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces RefAny3D, a 3D-asset–referenced diffusion framework that jointly generates an RGB image and its point map conditioned on a rendered multi-view set of a reference 3D asset plus text. Core ideas include: (1) a spatially aligned dual-branch architecture with shared positional encodings so RGB and point-map tokens stay pixel-aligned; (2) Domain-specific LoRA (Domain-LoRA for point-map tokens and Reference-LoRA for appearance/reference tokens) and a text-agnostic attention design to reduce cross-domain interference; and (3) a new pose-aligned dataset constructed by extracting objects with GroundingDINO/SAM, generating meshes with Hunyuan3D, estimating 6D pose with FoundationPose, and rendering point maps aligned to the images. Experiments show improved geometric/texture consistency versus baselines (DreamBooth, IP-Adapter, DSD, OminiControl), with comprehensive ablations and qualitative/quantitative comparisons.

### Strengths
1. The pipeline that pairs real images with mesh assets (via Hunyuan3D + FoundationPose) to obtain aligned point maps is practically useful for research on 3D-aware image diffusion.

2. Domain-LoRA and Reference-LoRA help the network simultaneously generate point maps and RGB, improving stability and disentanglement.

3. The paper is easy to follow, with sound motivation and diagrams.

4. Ablations and comparisons are extensive; qualitative results show crisper textures and better geometric adherence than baselines.

### Weaknesses
1. Potential supervision noise from image-to-3D. The dataset relies on image-to-3D generators, which may not perfectly preserve reference fidelity, injecting bias into the training signal. The paper should quantify how frequently generator artifacts or pose errors degrade the learned 3D-conditioned diffusion, and propose mitigation.

2. It remains unclear how much of the gain comes from generating point maps versus simply conditioning on multi-view images; a rigorous comparison against a “no-point-map” generator (or a latent geometric proxy) would clarify necessity.

3. No user study. Given the goal (faithful, identity-preserving generation consistent with a 3D asset), a human preference study would strengthen claims about perceptual quality and faithfulness.

### Questions
More details of the dataset construction pipeline should be provided, such as data filtering and robust checking.

### Soundness
3

### Presentation
3

### Contribution
3
