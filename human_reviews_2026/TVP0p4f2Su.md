# LiTo: Surface Light Field Tokenization

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
We propose a 3D latent representation that jointly models object geometry and view-dependent appearance. Most prior works focus on either reconstructing 3D geometry or predicting view-independent diffuse appearance, and thus struggle to capture realistic view-dependent effects. Our approach leverages that RGB-depth images provide samples of a surface light field. By encoding random subsamples of this surface light field into a compact set of latent vectors, our model learns to represent both geometry and appearance within a unified 3D latent space. This representation reproduces view-dependent effects such as specular highlights and Fresnel reflections under complex lighting. We further train a latent flow matching model on this representation to learn its distribution conditioned on a single input image, enabling the generation of 3D objects with appearances consistent with the lighting and materials in the input. Experiments show that our approach achieves higher visual quality and better input fidelity than existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a 3D latent representation that simultaneously models object geometry and view-dependent appearance. Unlike previous methods that separately reconstruct geometry or predict view-independent appearance, this approach encodes RGB-depth samples of a surface light field into compact latent vectors. The unified latent space effectively captures realistic visual phenomena, including specular highlights and Fresnel reflections, under complex lighting conditions.
The main contribution lies in introducing a unified 3D latent representation that jointly learns geometry and view-dependent appearance from surface light field samples. Additionally, the authors propose a latent flow matching model conditioned on a single image, enabling the generation of 3D objects with lighting- and material-consistent appearances. Experimental results demonstrate superior visual realism and fidelity compared to existing approaches.

### Strengths
Originality: It is the first work to unify 3D geometry and view-dependent appearance in a single framework by encoding surface light fields sampled from multi-view RGB-D images. It employs 3D Gaussians with 3rd-order spherical harmonics (SH) to reproduce view-dependent effects such as specular highlights and Fresnel reflections, which traditional methods fail to capture. Additionally, it introduces a k-NN-based 3D patchification and voxel-aware attention structure, overcoming the efficiency bottleneck in processing million-scale surface light field samples.

Quality: The experimental design is rigorous and validated on datasets including ObjaverseXL and Toys4k. In reconstruction tasks, it achieves a PSNR of 36.45±4.00 (surpassing TRELLIS by 3.27), while in generation tasks, it reaches an FID of 8.193 at the conditioning view (better than TRELLIS’s 12.84). The dual-decoder supervision (flow-matching geometry decoder and 3D Gaussian appearance decoder) balances geometric accuracy and appearance realism, producing reproducible results.

Clarity: The framework follows a clear sampling–encoding–decoding–generation pipeline. Core formulas (e.g., the flow-matching loss and the radiance loss) are clearly annotated, and key concepts (e.g., surface light field, 3D Gaussians) are explained consistently. The connections between symbol definitions, method descriptions, and experimental procedures are well aligned, ensuring easy comprehension.

Significance: It fills the gap of “separated geometry and appearance modeling” in 3D latent representations, providing an efficient framework for 3D content generation and virtual-real fusion. Its ability to avoid precomputed coarse geometry reduces data dependency, and the view alignment of single-image-to-3D generation promotes the application of downstream tasks (e.g., game asset creation, AR modeling).

### Weaknesses
-1. In the expression $I_{gt}=Render(scene,H,E)$ presented below Eq. 3, the term scene is used, but its definition or corresponding description is not provided. Could you clarify what it represents?

-2. The results presented in Table 2 indicate that LiTo still exhibits a performance gap compared to TripoSF and TRELLIS. I understand that this may be due to the differences in the types of input data. The authors could frame this as a deliberate design choice, prioritizing appearance over geometry; however, this trade-off is a key characteristic of the method that should be more clearly articulated.

### Questions
-1. In this work, the authors utilize an RGB-D dataset. I wonder whether using an RGB-only dataset augmented with depth predicted by a large model could achieve reconstruction results comparable to those obtained with an RGB-D dataset. In other words, how significant is the impact of accurate depth information on the reconstruction quality?

-2. The manuscript mentions that LiTo uses 150 views during training. It would be informative if the authors could provide a brief discussion or evaluation of the reconstruction results when using fewer input views. Such an analysis could help assess the robustness of the method to reduced view availability.

-3. The paper appears to address two aspects: jointly modeling object geometry and view-dependent appearance, and single-view 3D reconstruction. These seem to correspond to two distinct tasks. Could the authors clarify which of these is the primary focus of the work? My understanding is that single-view 3D reconstruction is more akin to a downstream application of the learned model, so a clear explanation would help resolve this point of confusion.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a 3D latent representation that jointly encodes both geometry and view-dependent color. The model takes surface locations, view directions, and colors as inputs, and encodes them into $k$ tokens of $d$ dimensions. A decoder then reconstructs geometry via predicted velocities and view-dependent Gaussian parameters. To assess the quality of the latent space, the authors train a generative model that synthesizes 3D latents from a single image. Experiments on multiple benchmarks demonstrate that the proposed representation achieves high fidelity, particularly in high-frequency details.

### Strengths
1. Comprehensive experiments. The experimental section is thorough, with well-designed ablation studies that clearly justify key architectural and training choices.
2. High reconstruction fidelity. The method achieves superior fidelity in input images, which is an important aspect of 3D generation quality.

### Weaknesses
- The view-dependent color entangles the representation with environment lighting. While this improves reconstruction fidelity, it limits the method’s applicability for tasks requiring relighting or lighting-invariant representations.

### Questions
1. How does the inference time compare to TRELLIS?

2. In Table 3, why does the method show worse performance on novel-view metrics (FID_dino and KID_dino)

3. Since this representation encodes view-dependent color, I’m curious how it performs on more challenging benchmarks such as Shiny Blender and NeRFactor, which involve complex lighting and reflectance conditions.

Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields

NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination

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
4

### Summary
The paper presents a novel autoencoder that learns a latent representation capturing the view-dependent appearance of 3D assets in addition to geometry. Besides reconstruction, the latents can be used to supervise a DiT model, and in turn can be used to generate 3D assets from a single image.

### Strengths
- The paper is well written and mostly easy to follow.
- Modeling of view-dependent appearance while retaining performance on geometry modeling is a novel and valuable contribution.
- Experiments and ablations are sufficient to evaluate architecture and performance. LiTo consistently achieves strong empirical results.

### Weaknesses
- Some architectural details are missing for reproducibility (see questions).

### Questions
Q1. Is the flow matching decoder only used to predict the coarse occupancy grid, which then guides the Gaussian decoder to predict the splat geometry (including xyz) and appearance? More details of the Gaussian decoder need to be provided (similar to Q3 below). Are the splat mean and covar predicted directly in world space, and are they scaled or clamped to lie within scene bounds?

Q2. What is the k used for selection of nearest neighbors for cross-attention/3D patchification?

Q3. For the generative model, what is the structure/dimensions of the learnable positional encoding? What is the patch size and architecture of the learnable patchifier? For reproducibility, authors should include a table of dimensions and structure for each module, instead of only reporting the number of parameters.

Q4. What are the inference-time memory and runtime costs for image to 3D generation?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose a 3D latent representation that jointly models object geometry and view-dependent appearance from RGB-D inputs, capturing both within a unified latent space. This approach reproduces realistic effects like specular highlights and Fresnel reflections and enables high-fidelity 3D generation consistent with the input image’s lighting and materials.

### Strengths
1. The writing is clear and easy to follow.

2. The proposed pipeline appears novel; however, the motivation could be made more convincing or better justified.

### Weaknesses
Unclear motivation:
The motivation behind the proposed method remains unclear. Existing approaches typically avoid incorporating view-direction information because doing so simplifies subsequent relighting tasks. If all lighting- and view-related information are modeled jointly, it becomes questionable how the proposed model can perform relighting and be naturally integrated into a scene without introducing inconsistent illumination or lighting variations. The authors should clarify the motivation and explain how their approach maintains relighting consistency under such conditions.

Lack of ablation studies:
The current version of the paper does not include any ablation studies in the main text. A detailed ablation analysis is essential to demonstrate the contribution of each component and validate the effectiveness of the proposed method. The absence of such studies significantly weakens the paper. The authors should provide comprehensive ablation results to substantiate their design choices; without them, the paper’s claims are difficult to evaluate.

### Questions
please see weakness

### Soundness
3

### Presentation
3

### Contribution
2
