# Anime-Ready: Controllable 3D Anime Character Generation with Body-Aligned Component-Wise Garment Modeling

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
3D anime character generation has become increasingly important in digital entertainment, including animation production, virtual reality, gaming, and virtual influencers. Unlike realistic human modeling, anime-style characters require exaggerated proportions, stylized surface details, and artistically consistent garments, posing unique challenges for automated 3D generation. Previous approaches for 3D anime character generation often suffer from low mesh quality and blurry textures, and they typically do not provide corresponding skeletons, limiting their usability in animation. In this work, we present a novel framework for high-quality 3D anime character generation that overcomes these limitations by combining the expressive power of the Skinned Multi-Person Linear (SMPL) model with precise garment generation. Our approach extends the Anime-SMPL model to better capture the distinct features of anime characters, enabling unified skeleton generation and blendshape-based facial expression control. This results in fully animation-ready 3D characters with expressive faces, bodies, and garments. To complement the body model, we introduce a body-aligned component-wise garments generation pipeline (including hairstyles, upper garments, lower garments, and accessories), which models garments as structured components aligned with body geometry. Furthermore, our method produces high-quality skin and facial textures, as well as detailed garment textures, enhancing the visual fidelity of the generated characters. Experimental results demonstrate that our framework significantly outperforms baseline methods in terms of mesh quality, texture clarity, and garment-body alignment, making it suitable for a wide range of applications in anime content creation and interactive media.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a high-quality 3D anime character generation method. Overall, the proposed method is well-motivated, and the experimental results seems good.

### Strengths
•	The paper is well-written with a logical structure that makes the technical contributions easy to follow.
•	The proposed  framework is reasonable and well-justified. The experimental results demonstrate the effectiveness of the approach.
•	 The demo videos are excellent supplementary materials.

### Weaknesses
•	Recent works have explored video generation model-based avatar animation  capabilities. A more thorough comparison and discussion of the relationship between ANIME-READY and these methods would strengthen the paper. For example:
•	Animate anyone performs avatar animation via cross-attention and  video generation module. How between ANIME-READY compare to this strategy?
•	What are the trade-offs between the SDS-loss approach HumanNorm[2], video generation based approach Animate Anyone and the proposed method? Could you clarify when your method is preferable over other menetioned existing techniques?
•	Could you please provide more details about the private dataset?


1] Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation
[2] HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation

### Questions
I wonder can we use the Rodin to perform 3D avatar generation and then perform auto-rigging like Mixamo. What is the comparison with the proposed method?

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
3

### Summary
This paper proposes Anime-Ready, a framework for generating high-quality, controllable 3D anime characters from text or a single image. The key innovations include: (a) Anime-SMPL, a stylized extension of the SMPL body model that captures anime-specific proportions and provides unified UV maps for texture synthesis; (b) a Multi-Shape DiT with an MoE design for modular garment generation; (c) a body-aligned garment modeling scheme that encodes sampled body points to enforce spatial consistency and reduce interpenetration; (d) a component-wise high-resolution texture generation pipeline using multi-view diffusion and self-attention for disentangled texture synthesis.
Experiments show improvements in mesh and texture quality over baselines, also supported by a user study.

### Strengths
- The system design is very comprehensive, which integrates parametric body modeling, 3D diffusion, and texture synthesis, addressing multiple practical issues (alignment, rigging, texture bleeding).
- The novel parametric model is interesting and meaningful. It combines realistic rigging consistency with stylized geometry, enabling animation readiness.
- Applications such as retargeting and motion control demonstrate that the results are not merely visual but animation-ready.

### Weaknesses
- Quantitative evaluation is limited. The reliance on user studies instead of geometric or perceptual metrics (e.g., FID-3D, CLIP-score, interpenetration rate) makes comparisons less reproducible.
- The paper uses a private dataset of 20k characters, which may limit replicability. No evidence of generalization to unseen datasets is given.
- Though not considered as a reason of not accepting the paper, the training requires multiple A100s for ~10 days—resource cost is high for an ICLR contribution. Efficiency or inference speed is not discussed. Also, other components (e.g., MoE expert routing, diffusion conditioning choices) lack separate ablations.

### Questions
- How large is the parameter difference between Anime-SMPL and the original SMPL? Are blendshape parameters manually designed or learned?
- Does the body-aligned garment generation guarantee no interpenetration, or are post-processing steps still required?
- Could the authors release the Anime-SMPL model separately, even without the full dataset, to enable replication?
- The supplementary video basically shows frontal results. How robust is the canonical-pose generation to unusual camera angles or occluded limbs?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a unified framework for generating high-quality, animation-ready 3D anime characters. The authors introduce a parametric anime body model (Anime-SMPL) and design a component-wise garment generation pipeline. The resulting models are partially rigged, skinned, and fully textured. The method outperforms prior work in terms of mesh quality, texture clarity, and garment-body alignment.

**Contributions** 
- A novel parametric anime-style body model, Anime-SMPL, with full skeleton and blendshape rigging for animation.
- A component-wise garment modeling pipeline that separately generates hairstyles, upper garments, lower garments, and accessories, each aligned to the underlying body.
- A multi-view texture generation approach that operates per garment component to improve texture fidelity and modularity.
- Experimental results showing improvements over baselines in mesh quality, garment-body alignment, and texture realism.

### Strengths
- This paper introduces a parametric anime body model, Anime-SMPL, learned from 20,000 anime-style 3D models. This contribution is valuable for future research on animatable anime character modeling. 
- A novel pipeline is proposed for generating separate garments and the underlying body, making the outputs easy for editing and good for application. 
- The texture generation approach is also innovative: instead of generating multi-view images of the entire body at once, the model generates multi-view textures for individual garments separately.

### Weaknesses
- From my perspective, the output body is not fully skinned. Only the inner body is rigged and skinned, while body-hugging garments and accessories are bound to the inner body using nearest-neighbor (NN) skinning. The skirt is animated via physical simulation rather than skinning. 
- For tight-fitting garments, inheriting skinning weights via nearest-neighbor sampling introduces hard assignments, which can result in unnatural deformations—particularly in regions with complex articulation or discontinuous topology.
- The garments are decodered by the VAE decoder, which are independent to the inner body parametric model. This might lead to garment-body intersections or poor fit between the garments and the underlying body. 
- No ablation studies are provided to evaluate the effectiveness of the MoE (Mixture-of-Experts) structure.
- The paper provides very limited details on how the template Anime-SMPL model is generated, as well as its rigging parameters—including the blendshapes (expression, pose, and shape), joint regressor, and skinning weights.

### Questions
**Rigging and Animation**
- What is the dimensionality of the blendshape matrix used in SMPL-Anime? Does the model incorporate separate shape, expression, and pose blendshapes as in SMPL-X, or does it rely solely on shape blendshapes?
- The paper would benefit from more details on how the Anime-SMPL template is constructed. Specifically, what is the process to obtain rigging components (e.g., blendshapes and skinning weights) obtained? In SMPL, an important factor for learning accurate rigging parameters is that the scans are captured from minimally clothed or tight-clothing subjects. Do you remove garments or otherwise preprocess the 3D anime models to isolate the underlying body before estimating rigging parameters?
- How is the ground-truth $\boldsymbol{\beta}$ (shape) parameter estimated for each 3D anime model?  

**MoE-structured DiT Block**

It is reasonable to model the four components—hairstyles, upper garments, lower garments, and accessories—separately using a MoE design. However, it would be better to provide ablation results comparing models w/ and w/o the MoE structure. Intuitively, MoE is effective when different experts are activated for distinct inputs or tasks. For example, using upper garment latent tokens would trigger the corresponding expert for upper garments. In this paper, however, both the input image and the noisy garment tokens represent the full body and all garments. As a result, the experts (E1, E2, E3, E4) may process the input holistically rather than specializing, making it unclear how much benefit the MoE structure actually provides.

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
This paper presents Anime-Ready, a novel framework for generating high-quality, controllable, and animatable 3D anime characters from text or a single image. The core contributions are: 1) Anime-SMPL, a new parametric body model adapted from SMPL to better represent the unique and exaggerated proportions of anime characters. 2) A body-aligned, component-wise garment generation pipeline. This pipeline decomposes the character into a body, hairstyle, upper garment, lower garment, and accessories. It uses a novel MoE-based Multi-Shape DiT to generate each garment component's mesh. Experimental results show the proposed method produces siginificant better results than the baselines.

### Strengths
1. **High-Quality Results:** The method demonstrates a significant improvement over existing baselines (CharacterGen, StdGEN, Hunyuan3D 2.0) in user studies. The qualitative results, especially for fine-grained details, are visually impressive.
2. **Practical Applications:** The framework is designed for practical use. By generating a rigged (Anime-SMPL) body and separate garment components, it directly enables downstream tasks like garment retargeting.

### Weaknesses
1. **Anime-SMPL Template Details:** 
    - How were the ground-truth shape parameters for the 20,000 characters obtained? Line 235 mentions training a shape prediction network using MSE against "ground truth," but the origin of this ground truth is unclear.
    - The paper argues that Anime-SMPL is different from SMPL, but provides few details. A visualization of the unified body template's joint structure is needed to understand its topology.  (e.g. number of joints)
    - How does the unified Anime-SMPL model handle non-humanoid "accessories" like wings or tails? Are these part of the body model's joints? If so, how is the model "unified" when some characters have these features and others do not (or have multiples)?
    - Furthermore, a supplemental figure demonstrating the effect of varying the shape (beta) components would be helpful.
2. **Missing Ablation Studies:**
    - **MoE-structured DiT:** A central claim is that the MoE-structured DiT (L250) achieves "precise, component-aware generation" with "minimal parameter overhead." This claim requires ablation studies to be substantiated. I’m wondering how does this model compare against: (a) training separate, independent DiT models for each of the four components (hair, upper, lower, accessories); (b) a single DiT that does not use MoE, but is instead conditioned on a label token to differentiate the component to be generated.
    - **MVAdapter "Color Bleeding" (L308-310):** The paper mentions that an initial attempt using MVAdapter directly resulted in "color bleeding." A visual comparison showing this failure case should be provided.
3. **Reproducibility:** The entire method, and especially the core Anime-SMPL model, relies on a large, private dataset of 20,000 characters. The authors do not state whether this dataset or the pre-trained Anime-SMPL model will be released. Without access to either, the results are not reproducible, which is a significant drawback for the research community.

### Questions
1. How does the MoE approach ensure that the generated components (e.g., an upper and lower garment) are geometrically seamless? 
2. Hybrid Motion Control (L452): The hybrid approach for garment animation requires more detail. For simulation-driven garments: What physics simulation engine is used? How are the garment's physical parameters (e.g., mass, stiffness, friction) estimated from the image or text input? Are they simply hard-coded, and if so, how does this generalize?
3. The paper motivates Anime-SMPL by stating that anime skeletons differ from real human ones. However, the "Image-to-Image Synthesis" (pose canonicalization) stage uses OpenPose (L200) to estimate a skeleton for conditioning. Since OpenPose is trained on human poses, doesn't this introduce a contradiction?
4. In Fig 2, the input image for the character with a tail shows the tail is almost completely occluded. How is the model able to reconstruct it so accurately? Does this suggest potential overfitting to characters present in the training set?

### Soundness
3

### Presentation
3

### Contribution
3
