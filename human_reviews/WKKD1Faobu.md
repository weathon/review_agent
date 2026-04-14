# HouseCrafter: Lifting Floorplans to 3D Scenes with 2D Diffusion Model

- Decision: Reject
- Scores: 6, 5, 6

## Abstract
We introduce HouseCrafter, a novel approach that can lift a floorplan into a complete large 3D indoor scene (e.g., a house). Our key insight is to adapt a 2D diffusion model, which is trained on web-scale images, to generate consistent multi-view color (RGB) and depth (D) images across different locations of the scene. Specifically, the RGB-D images are generated autoregressively in a batch-wise manner along sampled locations based on the floorplan, where previously generated images are used as condition to the diffusion model to produce images at nearby locations. The global floorplan and attention design in the diffusion model ensures the consistency of the generated images, from which a 3D scene can be reconstructed. Through extensive evaluation on the 3D-Front dataset, we demonstrate that HouseCraft can generate high-quality house-scale 3D scenes. Ablation studies also validate the effectiveness of different design choices. We will release our code and model weights.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an approach for generating house-scale 3D indoor scenes based on 2D floorplans.
Starting from a floorplan, camera poses are uniformly sampled throughout the scene.
Then, these camera poses are traversed, and RGB-D images are synthesized in batches in an auto-regressive manner, where
each batch is conditioned on the existing nearby RGB-D images (if they exist) and an embedding of the floorplan. 
By incorporating depth into the synthesis process, the generated RGB-D images exhibit better visual and depth consistency than generating RGB images and using a standalone monocular depth estimation. 
Experiments on the 3D-Front dataset show that this approach can generate large, consistent, high-quality scenes that accurately align with the provided floorplans.

### Strengths
- The paper addresses a compelling problem with potential applications in architectural design and game development.
- The paper covers the literature of novel-view synthesis well.
- The method is clear and well explained.
- The floorplan embedding method is novel and clever.
- The generated results are geometrically and visually consistent.

### Weaknesses
- The paper does not clarify why generating "house-scale" scenes is preferable to generating "room-scale" scenes. Why is consistency across the entire house necessary, rather than generating individual rooms and combining them? In reality, different rooms within a house may vary in style and are not necessarily consistent with one another. This needs to be clarified as the authors used it as grounds to exclude comparison against Controlroom3d and Ctrl-Room at L207.

- The authors argue at L203 that they can not compare against BlockFusion as it generates 3D meshes without texture. However, the BlockFusion paper provides some interesting results for textured room generation where Meshy was used to texturize the scene. I believe that BlockFusion is highly relevant and should be considered for comparison.

- The quality of the generated scene (textures and geometry) is relatively low compared to existing room-scale approaches mentioned in the paper. Is this caused by the use of an old reconstruction method TSDF? If so, why are more recent approaches, such as depth-aware Nerfs/Gaussian splatting, not considered?

- The improvement from generating the depth map with the RGB images has not been demonstrated! To be able to judge if this approach is beneficial over the standard RGB synthesis followed by a monocular depth estimation, additional experiments are needed! For example, how would the scene reconstruction be affected if only the RGB images were predicted and then the depth was estimated using an off-shelf monocular depth estimation model?

### Questions
- Can the proposed method used for outdoor scenes similar to BlockFusion? In that case, how can the depth range in section 3.2.1 be extended to longer distances?

- What is the motivation for the proposed layout embedding approach? Were other embedding approaches tested?

- Why do you choose to add $\text{PosEnc}(p_K)$ to $v_K$ in (4)? 

- What is the inference time for generating a complete floorplan?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a method for generating house-level 3D scenes conditioned on 2D floorplans. Given a 2D floorplan, the approach first samples camera positions and rotations within free space. Next, it adapts a large 2D diffusion model, trained on internet images, to generate RGBD scans. Finally, it applies TSDF Fusion to reconstruct a 3D mesh with textured information derived from all the generated dense RGBD point clouds. Experimental results show that the generated novel views maintain consistency with both the input view and the floorplan. Furthermore, the resulting 3D scenes exhibit improved global coherence compared to existing methods, such as CC3D and Text2Room.

### Strengths
1. Using 2D floorplans as a conditioning method requires less human intervention and manual effort, yet produces globally coherent 3D scenes. In contrast, methods like Text2Room are relatively simple but do not ensure a plausible layout in generated 3D scenes. This highlights the advantages of using 2D floorplans as a basis for 3D scene generation.
2. The proposed approach of floorplan conditioning and multi-view RGB-D conditioning may inspire new methods for extending 2D generation to 3D at the scene level.
3. The paper is well-written and easy to follow, offering a detailed overview of related work. The motivation to generate 3D scenes by creating multi-view 2D observations is also well-justified and logical.

### Weaknesses
1. The current comparisons are not entirely convincing. For example, CC3D focuses on novel view rendering at the scene level rather than geometry, and its renderings qualitatively appear superior to those in this work. Additionally, Text2Room, which generates scene layouts based on text input alone, may not provide an appropriate basis for comparison in this context.
2. Since this paper focuses on floorplan-based 3D scene generation, a more relevant comparison would be with BlockFusion [1], which operates directly in the 3D domain to generate scenes based on input scene layouts (or unconditionally), making it a closely related baseline.
3. The generated 3D scenes exhibit relatively coarse and blocky geometry; for example, most chairs and tables lack legs. This may be due to the camera sampling strategy or the resolution of the generated images.
4. Additionally, the paper should report the number of RGB views used and the time required to generate a complete 3D scene for a more thorough evaluation.

Reference:

[1] BlockFusion: Expandable 3D Scene Generation using Latent Tri-plane Extrapolation

### Questions
1. What is the bottleneck in the current pipeline for generating high-resolution 3D geometry? The paper reports that the generated 2D images have a resolution of only 256, which may be insufficient for reconstructing high-quality 3D geometry.
2. Can the pipeline support interactive editing, such as moving objects, without requiring retraining?

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
2

### Summary
This work proposes a novel approach for transforming 2D floorplans into detailed 3D indoor scenes, such as houses. The authors introduce HouseCrafter, an autoregressive pipeline that leverages a pre-trained 2D diffusion model to generate consistent multi-view RGB and depth images. This method addresses the challenges of generating large-scale scenes by sampling camera poses based on the floorplan and producing images in batches, ensuring spatial coherence across the generated views.

### Strengths
(1) This is a well-written paper.
(2) This method generates high-resolution depth images for larges-scale scene reconstruction, which is more practicala and meaningful in real-world scenerios.
(3) The proposed method is compared with various methods. The experiments are complete and convincing
(4) Some visualizations are helpful to understand.

### Weaknesses
(1) Lack of inference time comparison.

### Questions
(1) In line 1080, what do you think are the advantages and disadvantages of sampling pose and sampling point cloud.

### Soundness
3

### Presentation
3

### Contribution
3
