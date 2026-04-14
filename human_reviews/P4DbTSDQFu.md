# GaussianAnything: Interactive Point Cloud Flow Matching for 3D Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Recent advancements in diffusion models and large-scale datasets have revolutionized image and video generation, with increasing focus on 3D content generation. While existing methods show promise, they face challenges in input formats, latent space structures, and output representations. This paper introduces a novel 3D generation framework that addresses these issues, enabling scalable and high-quality 3D generation with an interactive Point Cloud-structured Latent space. Our approach utilizes a VAE with multi-view posed RGB-D-N renderings as input, features a unique latent space design that preserves 3D shape information, and incorporates a cascaded latent flow-based model for improved shape-texture disentanglement. The proposed method, GaussianAnything, supports multi-modal conditional 3D generation, allowing for point cloud, caption, and single-view image inputs. Experimental results demonstrate superior performance on various datasets, advancing the state-of-the-art in 3D content generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- This paper introduces a text- or image-to-3D generation framework using a VAE that processes multi-view RGB, normal, XYZ, and camera embeddings to construct a "Point-Cloud structured" Latent space, followed by a 3D DiT to decode 3D Gaussian parameters
- The proposed method employs cascaded geometric and texture latent diffusion models for effective shape-texture disentanglement, enhancing a "3D-aware editing" application.
- Experiments on text- and image-to-3D baselines consistently outperform existing methods.

### Strengths
- The paper proposes 3D-VAE and 3D-Diffusion models that are well-designed, flexibly supporting both Text- or Image-to-3D Generation and 3D Editing applications.
- Gaussian Anything improves "3D Gaussians Utilization", which reduces the rendering of white backgrounds, adaptively generates compact 3D Gaussians Splatting parameters and also benefits rendering quality.
- The paper is overall well-written, and the models (3D VAE and two-stage cascaded Diffusion model) are well-designed.
- It surpasses existing methods like 3DTopia, LN3Diff, and Shape-E in text-to-3D quantitative metrics and Lara/LGM in image-to-3D qualitative results.

### Weaknesses
- Qualitative metrics for Image-to-3D are **omitted**. Please add appropriate experiments in **Sec.** 5.1. Since the authors have rendered GSO datasets, GaussianAnything should be compared to LGM/Ln3Diff/Lara (refer to **Fig.** 3), and metrics such as PSNR, SSIM, LPIPS, or Image-CLIP Score may be reported.
- The claim that GaussionAnything can support point clouds or multi-view images as input lacks sufficient experimental proof (only VAE supports multi-view/point cloud as input). The authors should either abandon this claim or provide corresponding quantitative or qualitative results in **Sec.** 5.
- The use of flow matching for "velocity-prediction" in **Equations** 12 and 13, as opposed to direct "noise prediction", warrants further explanation in **Sec.** 4.2.


I would appreciate it if the authors could address my concerns by providing corresponding quantitative or qualitative results based on the **weaknesses** and **review feedback**.

### Questions
- Are there any insights during the proposed VAE training stage (see **Sec.** 4.1) regarding model design or loss functions?

### Soundness
3

### Presentation
3

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
This paper focuses on feed-forward 3D generation, introducing an elegant 3D latent diffusion framework. 

The process begins by training a point cloud-structured latent representation—unlike the tri-plane representation commonly used in LRM, 3DTopia, and LN3Diff—through a VAE, using multi-view images, depths, and normals as input. The decoder of the VAE consists of a DiT transformer to predict coarse Gaussians, and a few cascaded upsamplers to predict dense Gaussians. The VAE is trained with a combination of a rendering loss, geometry regularization, KL constraints as well as an adversarial loss.
Next, a cascaded latent diffusion model is trained using flow matching for disentangled geometry and texture generation, which enables interative 3D editing.
The proposed framework can handle various conditioning inputs, including point clouds, texts and images.

### Strengths
The paper is clearly presented, with a well-motivated and thoughtfully designed method. I appreciate the overall approach and its intuitive structure.

### Weaknesses
The generated 3D objects presented in the paper and video demos are relatively simple and lack high quality, which diminishes the overall effectiveness of the method.
There are no quantitative results on image-to-3D generation.

### Questions
What dataset is used for text-to-3D evaluation? It seems not the commonly used T^3 benchmark. Why are metrics like render-FID used in CLAY (Zhang etal. 2024) not used here?

Can the method generate 3D objects with more complex geometries and higher quality?

Can PBR modeling be incorporated into the proposed framework?

As mentioned in the paper, the intermediate Gaussian output naturally serves as K LoD, allowing for a balance between rendering speed and quality in various scenarios.  I would be interested in seeing visualizations of the Gaussians at different LoD learned by the decoder. Additionally, how long does it take to perform one upsampling?
    
I would also like to see a discussion on the limitations.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a unified 3D generation framework using a flow-matching network to sample pre-trained point cloud structured latent in a cascaded manner to achieve geometry-texture disentanglement. The paper uses multiview RGB-DN images as input to obtain point cloud structure latent vectors. These vectors are then used to generate Gaussian Surfels through a transformer decoder and further upsampled to obtain a dense Gaussian point cloud for rasterization. This stage is trained with a VAE objective to learn a latent distribution over the latent vectors. In the second stage, a flow-matching network is used to sample the point cloud structured latent vectors in a cascading manner. This achieves geometry and texture disentanglement and allows texture variation given sampled geometry. The proposed method generates visually impressive results compared to many baselines. It also contains a rich set of ablation studies.

### Strengths
1. The paper shows better generation results compared to competitive baselines such as GRM. The qualitative visuals also look promising in, for example, generating the thin structures in the first row of Fig. (3).  
2. The ablation study is comprehensive. It shows that the authors clearly put lots of efforts into validating the proposed system. 
3. The proposed system enables many applications including editing and generation tasks. This is further strengthened by its ability to disentangle geometry and texture.

### Weaknesses
1. Qualitative examples. In Fig. (3) why most of the input images are very dark? It seems that perhaps the other baselines wouldn't do well because of the particular input image choice. Also, the dark image makes it hard for me to judge if the proposed method is actually better than the baselines. For example, it's hard to see the color of the cactus on row 2. Also, why do all the baselines have different poses from each other? This also makes comparison difficult.  
2. For Fig. (6b), why editing on latent space better than directly editing on Gaussians based on this example? Both show disjoint geometry.
3. The generation results shown in the paper do not look very complex compared to the visuals in existing works such as GRM. It would be great if authors could provide further visual comparisons on more diverse prompt and caption inputs.

### Questions
1. Eq. (5) misses a parenthesis. 
2. The sentence from Ln 49 - 51 is difficult to understand. What do the authors mean by the set as the latent space? 
3.  Ln. 52-54. "Specifically, we project the un-structured features $\mathbf{z}_z$ onto the manifold of the input 3D shape through the cross attention layer:" The word "manifold" here is unnecessary and can lead to confusion since the queries are just fixed resolution sampled points, not the actual shape.

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
This manuscript introduces a point cloud-based latent diffusion model for 3D Gaussian splatting generation. The proposed pipeline consists of a point cloud VAE and a cascaded diffusion model. The VAE encodes multiview RGBDN into a downsampled features, and employs cross attention between points sampled from the 3D surface and the features to aggregate the information into the points, yielding the featured points as the set latent. A transformer decoder is adopted to upsample the latents into 2DGS representation. The DiT based diffusion model generates first the position of the points, and then the features conditioned on the positions. Experiments have demonstrated its performance on text/image-to-3D generation and 3D editing.

### Strengths
- First of all, I think it's quite impressive to me that a 3D-native diffusion model can be trained from scratch to achieve the presented results, although my knowledge may be outdated since I'm not an expert in point cloud-based 3D generation. Many SOTAs (e.g. 3DTopia) gain advantage using 2D diffusion priors from large pretrained models, so I'm impressed that the proposed model can compete against some of those method.

- Great writing overall. The network architectures are very clearly presented. 

- Geometry/texture disentanglement and the ability to edit the results in 3D add to the versatility of the proposed approach.

### Weaknesses
- I would disagree with the statements in L244-249. Why would a latent of shape $V\times (H/f)\times (W/f)\times C$ be difficult to generate for diffusion models? Isn't this what most multi-view image diffusion models are doing? My understanding is that, such high dimensional latents have more capacity and requires a larger model and a larger dataset to unleash its potential. Choosing a more compressed latent design is reasonable for this work since the model is trained from scratch, but it may not be the optimal choice when scaling up.

- Experiments are underwhelming in general, both quantitatively and qualitatively. 
  - Quantitatively, the only comparisons made are on text-to-3D with CLIP being the only metric. The details of the experiment setup, such as the number of objects tested and the source of the captions, are not specified at all. Image-to-3D and editing abilities are not evaluated quantitatively, so the real potential of its versatility is not well supported. 
  - Qualitatively, the few text-to-3D examples shown are not enough to demonstrate its generalization. The caption also appears very simple, unlike many previous approaches that adopt the more complex DreamFusion captions. Image-to-3D qualitative results are not satisfactory, given the missing divider of the first file sorter object and the slightly distorted shape of the final tower object. The competitors are also relatively weak. Many previous works (e.g. One2345++) can handle these simple objects easily. While it's understandable that many works are not open source, making fair comparison difficult, I find the performance of the proposed approach underwhelming in general.

- Using Dino features for image conditioning may introduce a major limitation for image-to-3D generation due to its large patch size and thus reduced resolution. Moreover, exact image-3D alignment gets more challenging in this case since the set latent doesn't really spatially align with any view. For these reasons, I'm doubtful about whether the model could produce faithful 3D results to the image. The first example in 
Fig. 3 shows that the generated file sorter has 4 dividers while the input has 5, indicating the flaw in image-to-3D generation.

- The conditioning mechanism is not really unique at all. Any diffusion model can use modulated normalization and cross attention for text/image conditioning. In L363, the claim that the LRM line of work can only handle image condition is definitely wrong (Instant3D for text-to-3D).

### Questions
- I find L183 confusing. Flow matching models typically adopt the velocity parameterization $v=\epsilon - x_0$. Is the network predicting $\epsilon$ or $v$?

### Soundness
3

### Presentation
3

### Contribution
2
