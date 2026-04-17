# Parametric SDF for Dynamic Surface Reconstruction

- Decision: Reject
- Scores: 8, 2, 8, 4

## Abstract
Reconstructing high-fidelity, temporally coherent surfaces of dynamic scenes remains a critical challenge in computer vision. While recent methods excel at novel view synthesis, they often fail to recover accurate geometry, yielding noisy or temporally inconsistent meshes that are suboptimal for downstream applications such as simulation or editing. In this work, we introduce a new paradigm for dynamic surface reconstruction based on a parametric Signed Distance Function ({\nameshort}). Our key insight is to generalize static SDF fields—where each spatial point stores a constant value—into time-dependent parametric curves, where each curve defines a temporally evolving SDF trajectory. Such a parametric SDF modeling provides a principled way to capture complex temporal variations, naturally enforcing smoothness and continuity in shape dynamics. At each timestamp, a static SDF field can be queried from {\nameshort} and converted into an explicit surface mesh via differentiable iso-surfacing. By rendering these meshes with a physically based differentiable renderer, we optimize the underlying parametric curves end-to-end against 2D image observations. Our framework produces high-fidelity, temporally coherent surfaces and inherently disentangles geometry, material, and lighting from multi-view videos. It robustly reconstructs geometry under large-scale motions and resolves appearance ambiguities caused by challenging lighting and occlusions. Experiments on both synthetic and real-world scenes demonstrate that our method achieves state-of-the-art geometric accuracy and temporal consistency, delivering delicate meshes that surpass prior work.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a novel framework for dynamic surface reconstruction based on a parametric Signed Distance Function (p-SDF) representation. The core idea is to generalize static SDF fields—where each spatial point stores a single constant value—into time-dependent parametric curves, with each curve modeling the continuous temporal evolution of the SDF at that point. This formulation enforces temporal coherence while accommodating large deformations and topological changes. At any given timestamp, a static SDF field can be queried from the parametric function and converted into an explicit mesh through differentiable iso-surfacing. Combined with a physically based differentiable renderer and a dynamic tri-plane appearance field, the framework enables end-to-end optimization from multi-view video observations. Experiments on synthetic and real-world datasets demonstrate state-of-the-art geometric accuracy, temporal consistency, and visual fidelity. The proposed method demonstrates smooth, temporally stable reconstructions that surpass prior dynamic scene methods in both quality and robustness.

### Strengths
1. The introduction of parametric SDF trajectories is novel and well-explored in the paper. Modeling the evolution of SDF values as continuous parametric curves provides an intuitive and principled way to ensure temporal coherence while still allowing for topological flexibility.
2. The proposed pipeline combines p-SDF-based geometry, differentiable iso-surfacing, and physically based rendering in a unified system. The design practically bridges geometric modeling and appearance estimation with end-to-end differentiability.
3. The method consistently outperforms previous approaches in both quantitative metrics (PSNR, Chamfer distance, MAE) and qualitative visual quality. The results on highly dynamic and complex datasets validate the robustness of the approach.

### Weaknesses
1. Since each spatial point stores its own set of curve parameters, the memory footprint may increase rapidly with larger scenes or longer sequences. The paper would benefit from a more detailed discussion on scalability. In addition, it is recommended to include a table comparing the computational requirements and efficiency (e.g., training time and inference time) of different methods to better illustrate the practicality of the proposed approach.
2. The framework assumes synchronized multi-view inputs, which may limit its applicability in monocular or unsynchronized video settings.
3. The paper could include a detailed analysis or ablation on how the choice and balance of polynomial versus Fourier basis functions affect reconstruction quality, especially for non-periodic motion.

### Questions
1. How do the memory consumption and runtime scale with longer sequences (e.g., several thousand frames)? It would be helpful to understand the computational trade-offs and whether the method remains efficient for large-scale dynamic scenes.
2. Can the proposed p-SDF representation handle discontinuous or abrupt temporal changes, such as object appearance or disappearance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a method for reconstructing dynamic scenes from multi-view videos. The central goal of the work is to recover accurate, temporally coherent geometry, and the key idea proposed to achieve this is a temporally parameterised SDF representation. The achieved reconstruction is evaluated on multiple datasets (the newly proposed SynthoMotion-360 dataset and the established CMU Panoptic Studio) and compared with multiple dynamic novel-view synthesis and geometry reconstruction methods, achieving lower geometry and novel-view errors. In addition, the work estimates materials and lighting in a disentangled way and shows some relighting examples.

### Strengths
1. The idea of parametric SDF that traces a continuous SDF trajectory, with a basis formulation that inherently enforces temporal coherency, is well-motivated and well-formulated.  
2. The combination of polynomial and Fourier bases that respectively capture both large-scale deformations and fine-grained high-frequency motions is also a nice takeaway and well-evaluated. 
3. Generally, the paper is clearly written and structured.

### Weaknesses
Despite some interesting insights and its originality in parametric SDF representation, I think the current submission is not ready for publication. The primary reason is the misleading positioning of the proposed method with respect to the related works and its unfair comparison with the state of the art. Please see the following points for further details:

1.	While the proposed method performs reconstruction from multi-view videos, all the compared methods are designed for reconstruction from monocular videos. This list includes Dynamic-2DGS (Zhang et al., 2024), DGMesh (Liu et al., 2024), Deformable-3DGS (Yang et al., 2024), SC-GS (Huang et al., 2023), and Grid4D (Xu et al., 2024).  Such a comparison is completely unfair. Moreover, the proposed paper doesn't even mention whether these methods are evaluated using a multi-view loss. Note that the proposed method states in the limitations that it doesn’t work in a monocular setting. Then, it is highly unclear why the proposed method is better; is it the new parametric SDF representation or the additional input views? 

2.	There are works which reconstruct surfaces from multi-view video, e.g, NeuS2 [Wang et al. 2023], Chen et al. 2025, Zheng et al. 2025. Kindly make such an appropriate comparison.

[Wang et al. 2023] Wang, Yiming, et al. "Neus2: Fast learning of neural implicit surfaces for multi-view reconstruction." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[Chen et al. 2025] Chen, Decai, et al. "Adaptive and temporally consistent gaussian surfels for multi-view dynamic reconstruction." 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025.

[Zheng et al. 2025] Zheng, Chengwei, et al. "GauSTAR: Gaussian Surface Tracking and Reconstruction." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

3.	In the related works, monocular and multi-view works are not clearly differentiated in their approach to dynamics. Moreover, although the proposed work takes multi-view video as input, most discussions in the related work focus on monocular dynamic NeRF/3DGS and are written to imply that the proposed work falls into this category. 

4.	The central claim and goal of the proposed method is to achieve higher temporal coherence. I checked the supplemental video, and I do not see any better temporal consistency for the proposed method compared to the other works (eg, for cactus, the results of competing methods are pretty good for dynamic parts). So, my observation conflicts with the main claims of the paper: prior work produces noisy or flickering reconstructions under complex dynamics or large motions, but the proposed work doesn’t. In addition, temporal coherence is not quantitatively measured, as PSNR/MAE do not directly reflect that (especially in the case of multi-view reconstruction, where sufficient supervision is available per frame) 

5.	In Section 4.3, it is mentioned, “our method is the first to achieve dynamic reconstructions with precise PBR materials to support realistic dynamic relighting.” This statement seems incorrect; such a task has been addressed before, e.g. see SAFT [Stotko et al.]. Moreover, material/ lighting estimation is not discussed at all in the related works. 

[Stotko et al.] Stotko, David, and Reinhard Klein. "SAFT: Shape and Appearance of Fabrics from Template via Differentiable Physical Simulations from Monocular Video." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

Minor comments

1.	Some typos – frist at L159 and reconsturction at L168, and overviews at L200

2.	Referencing grid entry as a “node” might be clearer than calling it a vertex (e.g. L200), as a vertex could be confused with a mesh vertex (which is used during the next stage of mesh optimisation).

3.	L212 - Following prior work on temporal signal modelling, could you please add citations as to which ones?

4. I got the idea behind Fig. 2, but the location of points and SDF vs. time plots do not seem to match exactly. Please double-check.

### Questions
Please check the weakness section to find my questions and comments.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduced a parametric SDF representation for dynamic 3d scenes. My understanding is that every frame of the dynamic scene is represented by an SDF. Rendering this dynamic scene at any given timestamp, which differs from the timestamps at which the scene was captured in every frame, requires proper capture and representation of the scene's dynamics. I am not sure if this is the main motivation, but this is my assumption, based on my older experience on the topic. Here, it is clear that the authors do not propose tracking 
of the dynamics of the moving objects in the scene, but rather how to properly interpolate motion between some of the frames. Equally, they aim to evaluate how their proposed representation can be utilized for novel view synthesis of dynamic scenes. So, parametric SDF proposes representing a dynamic scene as an SDF evolving over time, where for every SDF voxel, there is a parametric function that represents the change in SDF values over time. This was used for capturing geometry. The appearance has been captured in three orthogonal planes, representing dynamic changes in the appreciation for every SDF voxel.

### Strengths
I find the two key ideas, parametric SDF and appearance tri-plane representation, quite interesting. They seem quite simple, but effective. Optimizing the decoded appearance with PBR rendering in an end-to-end fashion allows for the well-adjustment of the SDF basis function parameters to properly interpolate the geometry. Additionally, learning well, the appearance decomposed in the material properties, aligned with the actual multiple images, allows for the learning of decoupled scene appearance, enabling effective scene relighting. I find these elements as a strong contribution to this particular problem.

### Weaknesses
Although the technical part of the paper is well described and comprehensive, I still felt that the exact motivation was missing. Simply said, describe what your input is in every frame, and what you want to obtain as a result. If I haven't misunderstood, rendering the scenes in any time frame other than those where the multi-view capture occurred is the key.

### Questions
In the experimental results, it is stated that a larger number of views was used for training and a smaller number for testing, both for NVS and for geometric errors.
For NVS, I can understand that the scenes are rendered in the test views and corresponding measures have been evaluated. However, if the parametric SDF has been optimized on the larger
number of views(training), what does the testing in terms of the geometric accuracy with the smaller number of views mean?
If my understanding is correct, the learned parametric SDF serves for interpolating the dynamic scene at new intermediate timestamps. So I would expect learning from N views and 
for K frames out of the total M frames. Now, for the geometric accuracy, I would check the geometric error between the mesh produced from the parametric SDF at the M-K test frames not used for the learning. I would really love to understand this and strongly encourage you to explain it in the rebuttal

### Soundness
3

### Presentation
2

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
This paper proposes a parametric SDF for dynamic surface reconstruction. The parametric SDF is built basd on SDF and temporally varying signal modeling with basis functions. In implementation, this parametric function is modeled by a grid and the appearance is modeled by a 4D hash grid. With differentiable rendering, the parametric SDF and appearance are optimized. In experiments, the work curates a new synthetic benchmark, SynthoMotion-360. The experimental results on SynthoMotion-360, DiVa-360 and CMU Panoptic Studio show that the method achieves promoising results, with better fidelity and coherence.

### Strengths
1. The method introduces a parametric SDF for dynamic surface reconstuction, which leverages the advantages of SDF and temporlly varying signal modeling with basis functions.
2. The method distanges geometry and appearance to optimize the parametric SDF and material properties. 
3. The work curates a new synthetic benchmark to evaluate the dynamic surface reconstruction.
4. The method achieves promoising results for dynamic surface reconstruction.

### Weaknesses
1. The quantitative surface reconsutruction evaluation is only test on the synthetic benchmark. 
2. For more complex scenes, like the scenes on CMU Panoptic Studio dataset, the method fails to reconstruct high-fidelity surfaces.
3. The method cannot recover complex geometris and details.

### Questions
1. Many works focus on dynamic scene reconstruction and they evaluated their methods on some human datasets. It is better to also evaluate the proposed methods on human dataset, like ZJU-Mocap and PeopleSnapshot.
2. It seems that NeuS2 is a SOTA SDF-based method that can reconstruct dynamic scenes. It is better to compare with this method.
3. Can you discuss training efficiency for different methods, including NeuS2.
4. For the detail reconstruction, is it possible to increase the grid resolution to improve the reconstruciton?
5. Can you show some reconstruciton examples under challenging lighting conditions (Line103)?

### Soundness
3

### Presentation
3

### Contribution
3
