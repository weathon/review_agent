# GarmentPainter: Efficient 3D Garment Texture Synthesis with Character-Guided Diffusion Model

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Generating high-fidelity, 3D-consistent garment textures remains a challenging problem due to the inherent complexities of garment structures and the stringent requirement for detailed, globally consistent texture synthesis. Existing approaches either rely on 2D-based diffusion models, which inherently struggle with 3D consistency, require expensive multi-step optimization or depend on strict spatial alignment between 2D reference images and 3D meshes, which limits their flexibility and scalability. In this work, we introduce GarmentPainter, a simple yet efficient framework for synthesizing high-quality, 3D-aware garment textures in UV space. Our method leverages a UV position map as the 3D structural guidance, ensuring texture consistency across the garment surface during texture generation. To enhance control and adaptability, we introduce a type selection module, enabling fine-grained texture generation for specific garment components based on a character reference image, without requiring alignment between the reference image and the 3D mesh. GarmentPainter efficiently integrates all guidance signals into the input of a diffusion model in a spatially aligned manner, without modifying the underlying UNet architecture. Extensive experiments demonstrate that GarmentPainter achieves state-of-the-art performance in terms of visual fidelity, 3D consistency, and computational efficiency, outperforming existing methods in both qualitative and quantitative evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents GarmentPainter, a framework for synthesizing garment textures in UV space. It leverages a UV position map as a 3D structural guide to ensure texture consistency across the garment surface. Experimental results show that GarmentPainter achieves state-of-the-art performance in terms of visual fidelity, 3D consistency, and computational efficiency.

However, the overall framework lacks substantial novelty, as employing a UV position map for 3D structural guidance is an established practice. In addition, the motivation for introducing the type selection module is not clearly justified.

### Strengths
1. An application-driven study that delivers convincing results.
2. Construct a high-quality garment dataset tailored for texture generation.

### Weaknesses
1. The paper shows limited novelty. Using a UV position map is not a new idea, and Paint3D also support direct generation in UV space.

2. The introduction of the type selection module is not well-motivated. Why not directly use the cloth type as a text prompt instead of introducing an additional type encoder? It would be helpful to report the performance when using cloth type as a textual condition. Moreover, the claim that the type selection module works without alignment between the reference image and the 3D mesh seems somewhat overstated.

3. The discussion of 2D virtual try-on in the related work section appears unnecessary, since the paper’s focus is on 3D texture generation.

4. The discussion of limitations is insufficient. How does the proposed method perform on fine or automatically generated UV maps, such as those produced by atlas-based methods?

5. Minor issue: There are a few typos in Figure 3.

### Questions
1. The paper shows limited novelty. Using a UV position map is not a new idea, and Paint3D also support direct generation in UV space.

2. The introduction of the type selection module is not well-motivated. Why not directly use the cloth type as a text prompt instead of introducing an additional type encoder? It would be helpful to report the performance when using cloth type as a textual condition. Moreover, the claim that the type selection module works without alignment between the reference image and the 3D mesh seems somewhat overstated.

3. The discussion of limitations is insufficient. How does the proposed method perform on fine or automatically generated UV maps, such as those produced by atlas-based methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes **GarmentPainter**, a diffusion-based framework for generating 3D garment textures directly in UV space. The method modifies Stable Diffusion 1.5 by removing text cross-attention and injecting multiple VAE-encoded latent modalities (reference image, UV position map, UV texture/mask) to guide generation. A small type-selection module is added to control generation across top/bottom/one-piece garments.

### Strengths
- **Data Contribution**: The authors curate a garment-specific dataset with UV maps, reference images, and mask/position data, which is valuable for this niche area of 3D garment texturing.
- **Structural Innovation on SD1.5**: The way the authors adapt SD1.5 — particularly replacing text cross-attention with multi-modal VAE latent conditioning — is a novel and neat architectural modification that simplifies conditioning without heavy architectural changes.
- **Experimental Soundness**: The ablation studies are well-designed and convincingly demonstrate the necessity of each component (UV position map, type selection), showing clear performance degradation when removed.

### Weaknesses
**Concerns on Generalization**
> *[Sec.3 L206-209]* “Ultimately, we curate a dataset comprising 7,579 clothing items, including 3,703 tops, 2,114 bottoms, and 1,762 one-piece garments.”
- Although the dataset creation is commendable, the total scale (~7.6k) appears small relative to the architectural modifications made to SD1.5 (multi-modal VAE inputs, removal of text cross-attention). It raises the question of whether such a limited dataset is sufficient to grant **true generalization** rather than overfitting.
- The paper does not specify the **train/validation/test split**, nor clarify whether evaluation is on the same data source or on external data. This omission further amplifies concerns regarding generalization.
- UV robustness is under-explored. All data are processed in Blender with a consistent UV unwrapping workflow. It remains unclear whether the method can generalize to **other UV layouts**, especially auto-generated or platform-specific UVs, which often introduce discontinuities.

**Limited Novelty**
- While the architectural attempt is appreciated, using a UV coordinate/position map to maintain spatial continuity is not new — both **Paint3D** and **TEXGen** adopt similar strategies.
- Furthermore, directly generating UV maps inherently struggles to guarantee 3D spatial consistency due to UV seam discontinuities and varying unwrapping conventions. This is why many recent works still rely on multi-view synthesis[1,2,3] for texture painting. That said, for **garment textures**, where UVs tend to be more regular, the approach is acceptable and practical.

**Relatively Outdated Foundation**
- From a generative model standpoint, the field has rapidly evolved beyond SD1.5 (e.g., SDXL, FLUX, Qwen-Image), with substantially improved image quality, resolution, and visual priors. Operating on SD1.5 limits output resolution to 512px and inherently lags behind current capabilities.
- The paper neither discusses nor compares against more recent texture-oriented or multi-view-consistent generation approaches such as: Mv-Adapter[1], FlexiTex[3], which would provide a more up-to-date benchmark of competitiveness.

--- 

**Reference**

[1] "Mv-adapter: Multi-view consistent image generation made easy." *in ICCV 2025*.

[2] "Mvpaint: Synchronized multi-view diffusion for painting anything 3d." *in CVPR 2025*.

[3] "FlexiTex: Enhancing Texture Generation via Visual Guidance." *in AAAI 2025*.

### Questions
See weakness

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
3

### Summary
The paper targets the intesting problem of 3D garment texture synthesis which is of great importance to the industry applications like games and animations. The main idea of the proposed GarmentPainter is to leverage the UV map to align 2D images to enable the 2D models to generate the 3D UV textures. To train the model, a new dataset is introduced with high-quality garments which is helpful for the garment texture generation problem. Experimental results validate the effectiveness and efficiency of the proposed algorithm.

### Strengths
* The problem of 3D garment generation is an important problem for industry applicatioons. 
* The proposed garment dataset with high-quality garments should be useful to the 3D community.
* The proposed algorithm obtain promising results with sufficient ablation studies.

### Weaknesses
* More implementation details should be provided to facilitate the reproduction of the paper. Or it would be better to provide the code for reproduction. 

* For the experimental results in Table 1, I would suggest to provide more comparisons against the papers published in the recent two years or in the year of 2025.

* In the experiments, the evaluation metric is based on FID and KID, which may not be consistent with human subjective evaluations. Thus, is it possible to provide a user study to verify the effectiveness of the proposed algorithm against the baselines. 

* The algorithm is based on the inpainting model trained from SD v1.5. As there are rapid developement of the text-to-image community, how about the performance of the proposed algorithm if better inpainting models, like flux-kontext, is utilized?

### Questions
Please mainly address the questions in the weakness section. More specifically, the questions related with the experiments should be well addressed.

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
3

### Summary
The paper proposes GarmentPainter, a method that generates garment textures directly in UV space from a reference person image. The approach encodes the reference image and a UV position map into latents, concatenates them (with a masked UV texture channel) as UNet inputs, and injects a garment-type embedding (top/bottom/one-piece) into the diffusion timestep embedding for better control. The authors also describe a dataset of ~7.6k garments with reference images, UVs, type labels, and position maps, and report strong speed and competitive FID/KID against several baselines.

### Strengths
1. Simple and practical design: Minimal modifications to a standard inpainting diffusion backbone (channel concatenation + type embedding) make the method easy to implement and deploy.

2. Fast inference: Reported end-to-end UV generation is notably fast (single forward path), which is attractive for production pipelines compared with multi-view/iterative methods.

3. Workflow alignment: Accepting a person-in-context reference image maps well to real authoring scenarios, reducing pre- and post-processing overhead.

4. Clear data description: The paper gives a concrete breakdown of categories (top/bottom/one-piece), rendering protocol (front/back), and labeling procedure, which improves readability and reuse.

### Weaknesses
1. Fairness & reproducibility: Different baselines appear to be run under different input protocols (e.g., prompts, masks, background handling, illumination). This can bias comparisons. A single unified evaluation protocol (resolution, masking, prompts, backgrounds/lighting) and a reproducible package would strengthen claims.

2. 3D consistency metrics are thin: The evaluation focuses on image-space metrics (e.g., FID/KID) and runtime. It lacks direct measures of UV seam continuity, cross-view consistency, and geometric adjacency color differences, which are crucial for textures that must look coherent on a mesh.

3. Generalization beyond the training domain: The dataset leans toward a specific visual domain. Claims of robustness to real photos, complex materials, heavy patterns, and challenging poses would be more convincing with systematic out-of-domain tests and a failure-mode analysis.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
