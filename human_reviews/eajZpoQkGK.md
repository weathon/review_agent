# DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Recent advancements in 3D content generation from text or a single image struggle with limited high-quality 3D datasets and inconsistency from 2D multi-view generation. We introduce DiffSplat, a novel 3D generative framework that natively generates 3D Gaussian splats by taming large-scale text-to-image diffusion models. It differs from previous 3D generative models by effectively utilizing web-scale 2D priors while maintaining 3D consistency in a unified model. To bootstrap the training, a lightweight reconstruction model is proposed to instantly produce multi-view Gaussian splat grids for scalable dataset curation. In conjunction with the regular diffusion loss on these grids, a 3D rendering loss is introduced to facilitate 3D coherence across arbitrary views. The compatibility with image diffusion models enables seamless adaptions of numerous techniques for image generation to the 3D realm. Extensive experiments reveal the superiority of DiffSplat in text- and image-conditioned generation tasks and downstream applications. Thorough ablation studies validate the efficacy of each critical design choice and provide insights into the underlying mechanism.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes DiffGS, a novel 3D generative framework that leverages large-scale text-to-image diffusion models to directly generate 3D Gaussian splats.  It first introduces a lightweight, generalizable 3DGS network that regresses per-pixel Gaussians. Then, it fine-tunes existing VAEs to encode these per-pixel Gaussians into Gaussian latents. Finally, the large-scale text-to-image diffusion models are fine-tuned to predict the noise injected in Gaussian latents. Extensive experiments show that the proposed method achieves high-quality novel view synthesis results compared to baselines.

### Strengths
* The proposed method uses fewer parameters and achieves better novel view quality compared to baselines.

* A framework that generates 3DGS from noise is interesting.

* The paper is structured well, making it easy to understand and follow.

### Weaknesses
I think there is no critical weakness but there are some unclear components in this study:

1. The authors claim that DiffGS leverages large 2D diffusion priors and maintains 3D consistency through the 3D rendering loss. However, the reported results focus only on novel view synthesis and text similarity scores. It would be better to include evaluations of 3D consistency, similar to the COLMAP-based evaluation used in SyncDreamer [1] or the optical flow-based evaluation used in HarmonyView [2].

2. At the inference stage, how is the input camera pose obtained? Is it derived from the training dataset?

3. Although the Gaussian latents are already aligned in the 3D representation, is there still a need for Plücker ray embeddings? What would happen if ray embeddings were not used?

[1] Liu et al., Syncdreamer: Generating multiview-consistent images from a single-view image, ICLR 2024.

[2] Woo et al., Harmonyview: Harmonizing consistency and diversity in one-image-to-3d, CVPR 2024.

### Questions
See weakness

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
5

### Summary
This paper, DIFFGS, proposes a new 3D generative framework designed to improve 3D content generation, addressing challenges like limited high-quality datasets and inconsistencies from 2D multi-view generation. Unlike other models, DIFFGS harnesses large-scale text-to-image diffusion models, leveraging 2D priors while ensuring 3D consistency. It introduces a lightweight reconstruction model to create multi-view Gaussian grids, enabling scalable data curation. Alongside traditional diffusion loss, DIFFGS uses a 3D rendering loss to enhance 3D coherence across views. This framework is compatible with image diffusion models, allowing seamless integration of image-generation techniques into 3D applications. Experiments show DIFFGS excels in text- and image-conditioned tasks, with ablations confirming the importance of each component. Code and models will be publicly available.

### Strengths
1. This paper presents a novel paradigm of 3D generation, leveraging strong 2D pretrained T2I models. The proposed method first encodes the given object into multi-view splatter images, fine-tunes the 2D VAEs to the 12xHxW domain, then train the multi-view splatter images diffusion models with both diffusion loss and rendering loss. Both view-concat  and spatial-concat versions perform reasonably well.
2. The comparison experiment is sound, which included the most recent methods (LN3Diff, 3DTopia-XL) as the strong baselines, which greatly demonstrates the effectiveness of the proposed method.
3. The proposed method can be well integrated into existing 2D diffusion frameworks, e.g., introducing ControlNet into the 3D generation domain.
4. Overall, I like this paper very much and strongly lean to accept it as the ICLR main conference paper.

### Weaknesses
Whether representing the given 3D object with multi-view splatter images leads to inconsistency, like in LGM? Since the Gaussians have overlap between different regions, and some “ghosting artifacts” are very common especially in very thin objects (such as axes).

### Questions
1. When fine-tuning the 2D VAEs, have you fixed some parameters or you fine-tune the whole VAE model? Since if fine-tuning the whole 2D VAE, how to stay connected with the pre-trained diffusion latent space?
2. Does your stage-1 reconstruction model for data curation itself serve as a LGM/GRM-like model, since it consumes multi-view RGB-N-XYZ renderings into the pixel-aligned Gaussians.
3. Comparison with amortised SDS-based model (e.g., LATTE3D: Large-scale Amortized Text-To-Enhanced3D Synthesis). They also provide a way to leverage pre-trained 2D diffusion models for 3D generation.
4. One disadvantage of the proposed solution is a lack of 3D latent space, and the generation of interior regions of the 3D object. Is these issues solvable by augmenting the proposed framework?
5. Compared to training diffusion on the 3D latent space direction (e.g., 3DTopia and LN3Diff), what are the limitations of the proposed method?

### Soundness
4

### Presentation
4

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
The authors propose a new diffusion-based multi-view generative models using 3D Gaussian Splatting rather than directly on image space. They consist of latent space which compress the gaussian representation to apply Latent Diffusion Model (LDM). Both diffusion loss and rendering loss are combined for better multi-view consistent plausible outputs.

### Strengths
1. The authors propose a way to combine 3DGS with latent diffusion model for generating better 3D models.
2. The proposed 3D generation pipeline seems to be both plausible and novel in that it parametrizes Gaussians and adopts specific VAE architectures to apply LDM on multiple Gaussian sets for 3DGS.
3. The authors well demonstrate both quantitative and qualitative results compared to competitive methods in image-to-3D and text-to-3D tasks.
4. Ablation study shows how each component contributes to the performance including omitting 3D rendering loss and 2D diffusion loss.

### Weaknesses
1. The contribution of the paper is on synthesizing conditional 3D models (text, single image) without per-object optimization using SDS such as DreamFusion, MVDream, and Wonder3D. I think the paper lacks some recent competitive methods including ATT3D [1] and Latte3D [2] which also aims to synthesize 3D models without per-object optimization.

2. There is no comparison for synthesizing time during inference. As the required time to get a single object is important, it can be also a weakness for the paper.

[1] Lorraine, Jonathan, et al. "Att3d: Amortized text-to-3d object synthesis." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2] Xie, Kevin, et al. "Latte3d: Large-scale amortized text-to-enhanced3d synthesis." European Conference on Computer Vision. 2024.

### Questions
1. Please compare the proposed methods with more 3D multi-view generative models such as Att3D [1] or Latte3D [2].
2. Please report rendering time of both single 3D model and multi-view 2D images with the rendered 3D model.
3. It would be good if the authors compare the 3D model quality of the proposed method with per-single 3D optimization models like GaussianDreamer [3].

[1] Lorraine, Jonathan, et al. "Att3d: Amortized text-to-3d object synthesis." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2] Xie, Kevin, et al. "Latte3d: Large-scale amortized text-to-enhanced3d synthesis." European Conference on Computer Vision. 2024.

[3] Yi, Taoran, et al. "Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

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
This paper presents DiffGS, a new method to directly generate 3D Gaussians in a feed-forward manner. DiffGS is fine-tuned from image diffusion models with the 3D Gaussian latent. The experiments on both text and image condition 3D generation show that DiffGS can quickly generate reasonable 3D Gaussians with high-quality rendering quality.

### Strengths
1. DiffGS encodes the 3D Gaussian Splatting to latents and trains a diffusion model to generate the latents directly, which is efficient compared with the optimization-based methods. The 3D Gaussian Splatting also offers better rendering quality.
2. The design of fine-tuning from the image diffusion model is reasonable and effective.
3. The authors provide various experiments, including different conditions (text, image normal, and edge).
4. The presentation is clear and easy to follow.

### Weaknesses
1. The overall idea is not very interesting. DiffGS is another latent diffusion by changing the image vae to a 3D Gaussian vae. Only substituting the reconstruction module in LGM to a diffusion model lacks novelty. In fact, the 3D native diffusion model [1] has proved that we can generate very high-quality 3D objects by jointly representing the 3D shapes with geometry latents and PBR materials while the generated 3D Gaussians in DiffGS can only provide good renderings. The advantage of Gaussian Splatting representation is that it may contain higher-resolution information compared with implicit tokens in previous work. I am wondering if the authors could prove that the Gaussian-based latent diffusion can achieve more detailed generation. Besides, the poor geometry of 3D Gaussians is also a problem for applying this representation to 3D generation given that the 3D native diffusion has achieved such great progress.
2. It would be better if the authors could provide videos of the generated objects. It is hard to evaluate some 3D artifacts just by the current metrics (e.g., PSNR). For example, the Gaussians often show good rendering results on some views while suffering from obvious 3D inconsistency.  Some noisy Gaussians and artifacts can be easily observed in videos.
3. Applying a rendering loss combined with the diffusion loss is a common trick.

[1] CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets

### Questions
I think some video demos are recommended for a 3D generation paper. Video demos are important to visually understand the 3D consistency and some geometric artifacts.

### Soundness
2

### Presentation
3

### Contribution
3
