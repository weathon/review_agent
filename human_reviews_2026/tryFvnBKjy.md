# LHM++: An Efficient Large Human Reconstruction Model for Pose-free Images to 3D

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Reconstructing animatable 3D humans from casually captured images of articulated subjects without camera or pose information is highly practical but remains challenging due to view misalignment, occlusions, and the absence of structural priors. In this work, we present LHM++, an efficient large-scale human reconstruction model that generates high-quality, animatable 3D avatars within seconds from one or multiple pose-free images.  At its core is an Encoder–Decoder Point–Image Transformer architecture that progressively encodes and decodes 3D geometric point features to improve efficiency, while fusing hierarchical 3D point features with image features through multimodal attention.  The fused features are decoded into 3D Gaussian splats to recover detailed geometry and appearance. To further enhance visual fidelity, we introduce a lightweight 3D-aware neural animation renderer that refines the rendering quality of reconstructed avatars. Extensive experiments show that our method produces high-fidelity, animatable 3D humans without requiring camera or pose annotations. Our code and models will be released to the public.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents an advanced version of LHM that accepts multiple images and produces high-quality, animatable 3D avatars with a large feed-forward model. The authors propose an Encoder–Decoder Point–Image Transformer that fuses 3D points with image features to handle multiple images efficiently. The fused tokens are decoded into 3D Gaussian splats and rendered with a lightweight 3D-aware neural renderer for real-time animation.

### Strengths
- Paper is well-written and easy to understand.
- Extensive ablation studies and visualizations are provided in the main paper and supplementary materials.
- The proposed Encoder–Decoder Point–Image Transformer reduces inference time compared to LHM, especially as the number of images increases, while also improving performance.

### Weaknesses
- The main concern is that the network is designed for sparse views. Why is this design choice necessary? Although the authors extend LHM to multi-image input, the paper shows limited gains beyond 16 images (e.g., Table 2/5 at 64 views). This raises the question of whether the multi-image extension offers meaningful benefits at higher view counts. In particular, the paper notes that “the gains become marginal with an increasing number of views” (Line 431), which is counter-intuitive if more views should provide more information. Please clarify.

- The overall framework feels close to LHM: both use multimodal transformers to fuse 3D geometry with image features inside the network. The technical contribution beyond LHM on model design is not entirely clear from the framework description.

- In Model Design, the paper highlights LHM’s quadratic attention complexity O(N_points + N)^2, but in Point–Image Attention the complexity remains quadratic after token merging, while N reduced to N/r. This suggests limited improvement as N grows large. A more detailed complexity analysis would strengthen the efficiency claim.

- There is no visualization of avatar on canonical pose. It would be nice if the canonical pose visualization is included.

- Missing recent references (recommend adding and, if possible, comparing in Table 1):
    
    [1] Kocabas, Muhammed, et al. "Hugs: Human gaussian splats." CVPR 2024. → Monocular video based reconstruction
    
    [2] Shin, Jisu, et al. "Canonicalfusion: Generating drivable 3d human avatars from multiple images." ECCV 2024. → Monocular video based reconstruction
    
    [3] Liao, Tingting, et al. "High-fidelity clothed avatar reconstruction from a single image." CVPR 2023. → Single image based reconstruction

    [4] Moreau, Arthur, et al. "Human gaussian splatting: Real-time rendering of animatable avatars." CVPR 2024. → Multi-view video based reconstruction
    
    [5] Wang, Rong, et al. "FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images." CVPR 2025 → Few image based reconstruction

I will reconsider the score when all the concerns are handled well.

### Questions
- What is the main difference between LHM and this paper’s framework beyond the point–image attention details?

- Why doesn’t performance improve after 16 images? Is this due to training distribution, or difficulty handling deformations?

- What are the common failure cases of the framework (e.g., loose garments, extreme poses, or heavy occlusions)? A small failure case visualization would be informative.

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
The paper proposes LHM++, a network that reconstructs an animatable digital avatar in a feed-forward pass taking arbitrary number of images as inputs. It reduces the time cost of LHM by adopting token "merge" and "unmerge" operations. The token "merge" operation merges similar image tokens. By reducing the number of tokens, it speeds up the attention computations.

Another contribution is the neural renderer. Instead of rendering the predicted Gaussian directly, it renders the feature map in 2D and uses a DPT head to predict from the 2D feature map.

The modification in the number of tokens significantly speeds up the inference with more image inputs and reduces the memory cost. Meanwhile, the method outperforms the existing methods in terms of loose clothes animation due to the neural renderer.

### Strengths
* The paper demonstrates the ability to animate loose clothes and generalization, which is challenging in human rendering.
* With "merge" and "unmerge", the model runs much faster than LHM with a lower cost in memory.
* The paper is clearly written and highlights the contributions.

### Weaknesses
* The paper claims that in LHM, the time complexity of self-attention operations scales quadratically with the number of image tokens (and thus with the number of input images). Meanwhile, as the number of input images increases, image tokens begin to dominate the attention computation. Although the proposed “merge” and “unmerge” operations help reduce memory and computational overhead, the overall self-attention complexity remains quadratic with respect to the number of images. These operations only reduce the time cost by a constant factor.

* The proposed LHM++ is presented as an improvement over LHM; however, it is unclear where this improvement originates. The main contributions of the paper appear to be (1) the PIT block with the “token merge” mechanism and (2) the neural renderer. Since the “merge” and “unmerge” operations inevitably introduce information loss, they are more likely to degrade rather than enhance visual quality. Additionally, the paper’s ablation study in the appendix shows that the neural renderer only marginally improves PSNR and SSIM. I would appreciate it if the authors could clarify in more detail where the performance gain from LHM to LHM++ in Table 3 comes from.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

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
This paper presents LHM++: AN EFFICIENT LARGE HUMAN RECONSTRUCTION MODEL FOR POSE-FREE IMAGES TO 3D. Overall, the proposed method is well-motivated, and the experimental results seems good.

### Strengths
•	The paper is well-written with a logical structure that makes the technical contributions easy to follow.
•	The proposed  framework is reasonable and well-justified. The experimental results convincingly demonstrate the effectiveness of the approach across different scenarios.
•	 The demo videos are excellent supplementary materials.

### Weaknesses
Could you please give a discussion about the diffirence with 3D generation model, such like CLAY (Rodin). I wonder can we use the Rodin to perform 3D avatar generation and then perform auto-rigging such as Mixamo?

### Questions
please see the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LHM++, a feed-forward model to generate 3D human avatars from casually captured images. At the core of the method is a Encoder-Decoder Point-Image Transformer (PIT) module to fuse 3D and 2D features, which are decoded into 3D Gaussian parameters. The authors conducted experiments on different dataset to verify the effectiveness of the proposed method.

### Strengths
The paper is easy to follow.

Synthesizing 3D/4D humans from images is an interesting task with practical applications.

The method is technically sound by leveraging a multimodal transformer architecture to fuse 3D and 2D feature for 3D Gaussian generation.

### Weaknesses
Limited technical contribution. This paper is an extension for LHM, and the main difference is that LHM++ replaces the MBHT with PIT mode. However, both MBHT and PIT fuse 3D and 2D features for 3D Gaussian prediction. Does the LHM support multiple image processing by fusing multiple images using MBHT architecture? Why is the PIT required, and how does it outperform MBHT?

The paper proposes that the PIT architecture improves the results, whereas the results in Tab. 10 suggest that the number of 3D geometric points has a bigger impact on the results, i.e., for 40K points, LHM-0.7B even performs better. It’s not clear whether the PIT architecture or the number of query points improves the results.


The paper proposes DPT-head as the final renderer. In this case, why is the 3DGS representation required? Is it possible to just predict the SMPL offsets for LBS instead of the full 3DGS parameters?

### Questions
How does the method decouple the belongings (e.g., the bag in Fig. 2) and human clothing? 

Details about the implementation. Are the Gaussian rendering and neural rendering jointly trained? The Eq. 8 loss is not clear. Are both the RGB and perception loss applied for Gaussian rendering and neural rendering?

### Soundness
3

### Presentation
3

### Contribution
2
