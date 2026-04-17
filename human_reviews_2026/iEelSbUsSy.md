# One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Generating explorable 3D scenes from a single image is a highly challenging problem in 3D vision. Existing methods struggle to support free exploration, often producing severe geometric distortions and noisy artifacts when the viewpoint moves far from the original perspective. We introduce One2Scene, an effective framework that decomposes this ill-posed problem into three tractable sub-tasks to enable immersive explorable scene generation.
We first use a panorama generator to produce anchor views from a single input image as initialization. Then, we lift these 2D anchors into an explicit 3D geometric scaffold via a generalizable, feed-forward Gaussian Splatting network. 
Instead of treating the panorama as a single image for reconstruction, we project it into multiple sparse anchor views and reformulate the reconstruction task as multi-view stereo matching, which allows us to leverage robust geometric priors learned from large-scale multi-view datasets.
A bidirectional feature fusion module is used to enforce cross-view consistency, yielding an efficient and geometrically reliable scaffold.
Finally, the scaffold serves as a strong prior for a novel view generator to produce photorealistic and geometrically accurate views at arbitrary cameras. By explicitly conditioning on a 3D-consistent scaffold to perform reconstruction, One2Scene works stably under large camera motions, supporting immersive scene exploration. Extensive experiments show that One2Scene substantially outperforms state-of-the-art methods in panorama depth estimation, feed-forward 360° reconstruction, and explorable 3D scene generation. Project page can be found at: https://one2scene5406.github.io

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript proposes a framework for generating an explorable 3DGS scene from a single perspective image. For doing that, authors decompose the problems in three independent tasks: 1. they generate a panorama 2. they select 2D anchor views from this panorama and create a 3DGS based scaffold by solving a multi-view stereo matching problem; 3. they use the scaffold for creating novel views at arbitrary camera poses.  They assess the framework in multiple ways: panoramic depth estimation, scaffold reconstruction, and explorable generation. The project website contain additional information and qualitative results.

### Strengths
1. The paper is well written and technically solid. Narrative style is convincing, and the manuscript is easy to follow.
2. The problem is really challenging and worth of exploration: 3D content generation from single images and automatic world generation has a strong impact on media creation.
3. The proposed solution appears convincing and well designed; technical details are provided, and authors indicate they will release the code for replicability.
4. The website resource contain a good variety of additional qualitative results: they look impressive and they show that the proposed framework has a strong potential.
5. Assessment: the framework is benchmarked against the most modern SOTA, and the relevant literature is generally discussed and compared. Quantitative and qualitative results are generally convincing.

### Weaknesses
1. Practical usage: even if the qualitative results appear impressive, I would expect attempts to go at different level of detail. I would like to see some examples of animations with different level of zoom and showing cluttered areas. For example, I would expect attempts to generate detailed leaves in plants, etc..  In most of cases, the generated scenes are with limited clutter. Also, I would expect to see how it works on the wild, with some casual image. Also, it is importante to clearly state what are the image resolution limitations.
2. Comparison with SOTA: in related work section, I would expect some sentences showcasing in detail how the proposed framework improve current technologies.
3. Link to panoramic scene reasoning: since the proposed framework pass through a panorama generation, and then it does all the subsequent tasks on a panoramic image with depth estimation; I would expect a comparison with methods generating 3DGS or explorable scenes from single panoramic images, like for example Pano2Room 
Pu, G., Zhao, Y., & Lian, Z. (2024, December). Pano2room: Novel view synthesis from a single indoor panorama. In SIGGRAPH Asia 2024 Conference Papers (pp. 1-11).
https://github.com/TrickyGo/Pano2Room.  Actually, the strategy presented is quite similar to Pano2Room, and I would expect at minimum a comparative discussion, and probably also a benchmark against this baseline.
4. Minor: for fairness, in qualitative results authors should use the same camera poses for baselines. 
line 57:  topicm
lines 199-201: please rephrase it. Lots of methods try to perform depth estimation from single panorama for novel view synthesis, like
Pintore, G., Bettio, F., Agus, M., & Gobbetti, E. (2023). Deep scene synthesis of Atlanta-world interiors from a single omnidirectional image. IEEE Transactions on Visualization and Computer Graphics, 29(11), 4708-4718.
Figure A7: one of them is SEVA+, which one?

### Questions
1. How the method works on the wild, with casual images? What are the limitations for usage in a game or VR scenarios? Image resolution, details in extreme zooming, etc. I would expect an extensive assessment in supplementary, and some information on the main manuscript
2. How is the method related to competing baseline, like SAVA+?  How is the related to Pano2Room? I would expect a detailed discussion on how the pipeline differs from this paper, as well as a benchmarked comparison against it.
3. Can you provide additional details about the bidirectional fuse model, since it it a key component in the proposed architecture? Some details about Cube Projection can be shortened, instead.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores how to leverage a 3D representation to condition the generation of new viewpoints given a single viewpoint. They propose a pipeline that takes an image, uses an off the shelf model to predict panoramas, then selects anchor views and from this constructs a 3D representation using a trained neuraal network. This is then rendered to the target view and, with the anchoring view, fed as a prior to the diffusion model.

The paper's point is that explicitly having an accurate 3D representation is key to obtaining good, robust, generated views as extreme viewpoints and they aim to demonstrate that their approach of doing so is SOTA.

### Strengths
The paper appears to have done comprehensive ablations:
1. Comparing sub parts of their approach (the 3D generation) and the image generation part (by using others' 3D representation), finding improvements in both cases.

2. The use of a fixed 3D representation makes a lot of sense to fix geometric mistakes / ambiguity. By creating an approximation of the whole 3D scene a priori, the authors get around issues taht exist in other works due to the accumulation of errors.

### Weaknesses
1. Why no comparison with Cat3D ? This seems like an obvious baseline which also does 1 view to multiple views ? The paper focusses on other methods that, to my understanding, seem to be mroe aimed to handle a trajectory of views (e.g. SEVA / AnySplat) and so may in general be expected to handle more complex scenes. While the author's approach is better than these in this case, they should also compare it to methods with more similar aims (e.g. Cat3D).

2. How does the model fair if you don't do a loop closure setting -- e.g. where you keep moving 'forward' by zooming in on a location? I fear in this case the panorama wouldn't be so useful for maintaining consistency and so is this method really only better than others in this one case for the camera angles ?

3. The authors evaluate their full pipeline on only one dataset -- but surely they could explore others as well ? Such as those in Table 1 in the Cat3D paper ?

### Questions
1. For Figure 1, it's quite confusing what I'm looking at -- what was input to the model, what was it tasked to do and what was the output? This is useful to put the results in context.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper One2Scene proposes a three-stage framework for generating fully explorable 3D scenes from a single image, addressing the challenges of geometric inconsistency, scale ambiguity, and semantic drift common in prior generative view synthesis methods.
The pipeline includes:
1. Panorama generation, expanding a single input image into a 360° representation using Hunyuan-Pano-DiT, providing global context.
2. Feed-forward Panorama 3D Gaussian Splatting (3DGS) network, converting the panorama into a 3D geometric scaffold by reformulating panoramic depth estimation as a multi-view stereo matching problem.
3. Scaffold-guided novel view synthesis, generating photorealistic novel views conditioned on both the 3D scaffold and anchor views through a Dual-LoRA training strategy and 3D attention fusion.
The method yields significant improvements in photorealism, geometric accuracy, and stability under large viewpoint changes, outperforming DreamScene360, WonderJourney, SEVA, and VMem across multiple metrics.

### Strengths
1. The overall pipeline is well-structured and intuitive, decomposing the single-image 3D generation problem into panorama expansion, geometric scaffold reconstruction, and scaffold-guided view synthesis. Each stage has clear motivation and contributes coherently to the final performance. And the paper is easy to follow with clear logistics.
2. The authors conducted comprehensive experiments with both quantitative and qualitative results.The sufficient evidence through extensive metrics makes the results solid.
3. The ablation studies are clear, for core components (Dual-LoRA Training, Memory Condition,  Bidirectional Fusion Module), there are sufficient experiments to show the effectiveness of them.
4. High efficiency and scalability. The proposed method achieves fast 3D reconstruction (0.5 s on H20, 0.1 s on H100) while maintaining visual quality, showing strong potential for real-world applications that require interactive generation like robot policy learning.

### Weaknesses
1. Although quantitative results are sufficient, the paper lacks explicit visualization of geometric outputs such as reconstructed point clouds or extracted scene meshes. There are many visualization with continuous frames in the website, however, geometric consistency is one of the paper’s core claims, these visualizations would provide more direct and convincing evidence of 3D structural accuracy.
2. The method has not been tested on complex, dynamic real-world environments, such as urban or outdoor settings with moving objects like pedestrians, vehicles. It remains uncertain how well the framework generalizes beyond indoor or static scenarios.
3. The pipeline depends on several large pretrained models (e.g., Hunyuan-Pano-DiT, SEVA, VGGT), which limits the degree of innovation and increases the engineering nature of the work. However, this design choice is understandable and acceptable.

### Questions
1. Geometric consistency validation:
Although quantitative results are comprehensive, the paper lacks explicit geometric visualizations such as reconstructed point clouds or extracted meshes. Given that geometric consistency is a central claim, could the authors provide additional evidence, to directly demonstrate 3D structural accuracy beyond continuous frame visualizations?
2. Generalization to dynamic and outdoor environments:
How does the proposed framework perform on more complex, dynamic real-world scenarios, such as urban or outdoor scenes containing moving objects (e.g., pedestrians, vehicles)? Have the authors tested or considered extensions for handling dynamic content or temporally varying geometry?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
One2Scene introduces a novel three-stage framework for generating geometrically consistent, explorable 3D scenes from a single image, overcoming severe distortions and artifacts that plague prior methods like DreamScene360 and WonderJourney. First, it generates a 360° panorama using Hunyuan-Pano-DiT; second, it projects this panorama into six sparse cubemap views and lifts them into an explicit 3D Gaussian Splatting scaffold via a feed-forward network that reformulates monocular depth estimation as multi-view stereo matching, leveraging robust geometric priors from large datasets through a bidirectional feature fusion module that enforces cross-view consistency; finally, it uses this scaffold as a strong geometric and appearance prior for novel view synthesis, guided by a Dual-LoRA training strategy that fuses anchor views and scaffold-rendered views via 3D-aware attention, enabling stable, photorealistic generation under large camera motions—achieving state-of-the-art results on visual fidelity, semantic consistency, and geometric stability, with a 0.5-second reconstruction time and significantly improved depth estimation accuracy over existing methods, as validated on Matterport3D and Stanford2D3D.

### Strengths
S1: The paper introduces a clever reformulation of monocular 3D scene generation by treating panoramic depth estimation as a multi-view stereo problem, which had not been done before.

S2: The feed-forward 3D Gaussian Splatting scaffold runs in under half a second and delivers unprecedented geometric stability without iterative refinement.

S3: The bidirectional fusion module and Dual-LoRA conditioning are elegant, practical innovations that significantly improve cross-view consistency and visual fidelity.

S4: By enabling immersive, long-range exploration from just one image—with results that surpass prior methods in both quality and robustness—it sets a new practical standard for 3D content creation.

### Weaknesses
W1: The method relies on a proprietary panorama generator (Hunyuan-Pano-DiT) without ablation on alternatives, limiting reproducibility.

W2: The 3D scaffold still shows minor artifacts in occluded regions under extreme rotations, suggesting room for post-processing or iterative refinement.

W3: Evaluation on stylized scenes lacks quantitative metrics for artistic fidelity—CLIP-I and NIQE may not capture stylistic coherence.

W4: No comparison to recent diffusion-based single-image 3D baselines like Bolt3D or DreamReward, which also aim for speed and realism.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
