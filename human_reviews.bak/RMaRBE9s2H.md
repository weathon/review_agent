# GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 6

## Abstract
LiDAR novel view synthesis (NVS) has emerged as a novel task within LiDAR simulation, offering valuable simulated point cloud data from novel viewpoints to aid in autonomous driving systems. However, existing LiDAR NVS methods typically rely on neural radiance fields (NeRF) as their 3D representation, which incurs significant computational costs in both training and rendering. Moreover, NeRF and its variants are designed for symmetrical scenes, making them ill-suited for driving scenarios. To address these challenges, we propose GS-LiDAR, a novel framework for generating realistic LiDAR point clouds with panoramic Gaussian splatting. Our approach employs 2D Gaussian primitives with periodic vibration properties, allowing for precise geometric reconstruction of both static and dynamic elements in driving scenarios. We further introduce a novel panoramic rendering technique with explicit ray-splat intersection, guided by panoramic LiDAR supervision. By incorporating intensity and ray-drop spherical harmonic (SH) coefficients into the Gaussian primitives, we enhance the realism of the rendered point clouds. Extensive experiments on KITTI-360 and nuScenes demonstrate the superiority of our method in terms of quantitative metrics, visual quality, as well as training and rendering efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors present GS-LiDAR, a novel framework for generating realistic LiDAR point clouds. They employ 2D Gaussian primitives with periodic vibration properties to precisely model both static and dynamic elements in driving scenarios. Additionally, they introduce a panoramic rendering technique to obtain depth, intensity and ray-drop probability maps.

### Strengths
1)	The authors successfully take advantage of periodic vibration 2D Gaussian primitives to reconstruct dynamic point cloud from lidar measurements, and achieve state-of-the-art performance in terms of rendering speed and accuracy.

### Weaknesses
1）	Insufficient reference. The authors emphasized they employed 2D Gaussian primitives with periodic vibration properties and counted it as one of the contributions. However, periodic vibration and 2D Gaussian primitives are proposed by [1] and [2] respectively, which haven’t been credited in the introduction section.

[1] Huang, Binbin, et al. "2d gaussian splatting for geometrically accurate radiance fields." ACM SIGGRAPH 2024 Conference Papers. 2024. 
[2] Yurui Chen, et al. "Periodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering.” arXiv preprint, 2023a.

2）	Ambiguous symbol. (u, v) in Line 204 and equation 2 denotes the target pixel of the rendering image, but later it becomes the coordinates within the local tangent plane space of 2D Gaussian in Line 245-255. 

3）	Incorrect definition. The formulations of intensity map I in Line 316 and ray-drop probability map P_gs in Line 32 are identical.

### Questions
1）	The presentation of lidar rendering is not clear enough to make it difficult to follow. The authors highlight that they introduce a novel panoramic rendering technique with explicit ray-splat intersection, guided by panoramic LiDAR supervision. However, it is even unclear whether the image is rendered by ray marching, rasterizing or a hybrid approach.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel method for LiDAR novel view synthesis called GS-LiDAR. GS-LiDAR utilizes 2D Gaussian Splatting (2DGS) as a foundational representation for LiDAR reconstruction and incorporates periodic vibration for enhanced results. The reconstruction process is supervised by panoramic LiDAR data. The method is thoughtfully designed, incorporating various strategies to adapt 2DGS to LiDAR NVS scenarios, and is backed by extensive experimentation to demonstrate its performance. However, several concerns are noted below.

### Strengths
1. The paper is well-structured and easy to read.
2. The application of Gaussian Splatting to achieve LiDAR novel view synthesis is an innovative approach.
3. The method is well-conceived, integrating the unique properties of LiDAR point clouds into 2DGS.

### Weaknesses
1. The title of the paper suggests that the method is intended for “Generating” LiDAR point clouds. However, GS-LiDAR is designed for “reconstructing” LiDAR point cloud scenes. While the method does achieve novel view synthesis, the concept of reconstruction differs from generation. Generation [1][2] implies the creation of LiDAR point clouds from other user inputs, not just from an existing sequence of point clouds.
2. The rationale for using 2DGS instead of 3DGS as the primary representation is unclear. Although 2DGS has demonstrated superiority in geometric reconstruction, a comparative analysis between 2DGS and 3DGS (with the proposed strategies, rather than a standard 3DGS) would strengthen the paper.
3. The performance improvement over LiDAR4D, as shown in Tables 1 to 3, appears marginal. Additionally, the visualization results in Figure 4 do not clearly demonstrate significant performance gains. Including comparisons in video format would help substantiate the results.

[1] Ran, Haoxi, Vitor Guizilini, and Yue Wang. "Towards Realistic Scene Generation with LiDAR Diffusion Models." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

[2] Zyrianov, Vlas, et al. "LidarDM: Generative LiDAR Simulation in a Generated World." *arXiv preprint arXiv:2404.02903* (2024).

### Questions
1. Can the proposed method handle solid-state LiDAR?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper addresses the task of LIDAR novel view synthesis and introduces a framework called GS-LIDAR, which generates realistic LiDAR point clouds using panoramic Gaussian splatting. To accurately reconstruct both static and dynamic elements in driving scenarios, the authors utilize 2D Gaussian primitives with periodic vibration properties. Additionally, the paper presents a novel panoramic Gaussian splatting technique that incorporates explicit ray-splat intersection for rapid and efficient rendering of panoramic depth maps. Experimental results demonstrate the framework's state-of-the-art performance in terms of quantitative metrics, visual quality, and efficiency.

### Strengths
The paper is generally well-presented, with a system architecture that is clearly outlined and easy to follow. Additionally, the experimental results—encompassing both quantitative and qualitative evaluations—are impressive, demonstrating performance that surpasses the current state-of-the-art in this specific task. Furthermore, the ablation study is thorough and effectively substantiates the proposed approach.

### Weaknesses
1) This article appears to present an incremental improvement rather than a significant innovation. The primary contribution seems to be the replacement of NeRF in LIDAR4D with Gaussian Splatting. While this method offers advantages in training and rendering speed compared to LIDAR4D, these benefits are inherently tied to the properties of Gaussian Splatting itself.
2) Regarding the methodology and experiments, the most notable innovation appears to be the Ray-splat intersection in panoramic Gaussian Splatting. However, this aspect is not clearly articulated in the methods section. For instance, in Line 249, the physical significance of the variables u and v is not intuitively conveyed, nor is the derivation of the transformation from UV space to screen space adequately explained. The authors should provide a more detailed explanation of this section or include a schematic diagram  (i.e. a geometric illustration) to enhance reader comprehension.
3) Furthermore, the authors need to conduct a thorough analysis of how the Ray-splat intersection, as demonstrated in the ablation experiment, enhances the capability of GS-LiDAR in accurately modeling surface features of the scene and leads to significant performance improvements.

### Questions
1. The novelty of the proposed method, especially in comparison to LIDAR4D, requires clearer articulation, which should focus on  how the paper's contributions go beyond simply replacing NeRF with Gaussian Splatting.
2. The Method section includes extensive background information on existing techniques, such as ray-drop refinement and various loss functions, which do not directly contribute to the core innovations of the paper. Consider condensing these parts or moving them to the supplementary materials. What's more, a more detailed explanation and accompanying diagram (i.e. a geometric illustration) of the Ray-splat intersection would improve clarity and focus.
3. Please explain in detail how the addition of the Ray-splat intersection mechanism significantly enhances the performance of the proposed method.
4. Additional experiments and analyses, such as testing on the Waymo dataset  with different sensor configurations or environmental conditions, would further validate the effectiveness and generalization ability of the approach.
5. Typo: In Line 323, the ray-drop probability signal $\rho_{k}$ is wrongly spelled by $\lambda_{k}$.

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
4

### Summary
This paper proposes GS-LiDAR, which introduces a novel framework for LiDAR novel view synthesis (NVS). Compared to previous NeRF-based works designed with fixed resolution and symmetrical structure, this paper employs 2D Gaussian primitives with periodic vibration properties to reconstruct the dynamic driving scenarios more effectively. Furthermore, based on the panoramic LiDAR rendering technique, the authors calculate the ray-splat intersection to achieve explicit and accurate results. And the spherical harmonic (SH) of Gaussian primitives can also support intensity and ray-drop reconstruction. Experiments on KITTI-360 and nuScenes demonstrate the superiority of GS-LiDAR in reconstruction quality and rendering efficiency.

### Strengths
- Previous NeRF-based works with Hashgrid have a symmetrical structure and fixed resolution for each axis, which is not suitable and effective enough for driving scenarios. This paper first adopts GS representation for LiDAR NVS, making the reconstruction and rendering more efficient.

- 2D Gaussian primitives have higher geometric accuracy. Moreover, it incorporates the periodic vibration properties, allowing for both static and dynamic scene reconstruction.

- Based on the projection nature of LiDAR point clouds, this paper utilizes the panoramic rendering technique to accommodate Gaussian Splatting, and further proposes ray-splat intersection to obtain more accurate rendering results.

### Weaknesses
- In terms of the core methodology, this paper is more of a combination of existing methods (2D-GS and PVG) and does not substantially innovate them, which results in a limited contribution. It should include more insights and analysis of the methods and why they are well suited for LiDAR-NVS. 

- The concept of panoramic Gaussian splatting is a little bit confusing in Section 3.3. And according to the proposed ray-splat intersection, it's more like the "ray tracing" manner than the "splatting" one. If so, it needs to compute the intersection of every ray and all Gaussian primitives, which is still not efficient enough.

- Despite the efficiency gains due to the Gaussian itself, the final rendering results don't seem to be a huge improvement.

### Questions
Besides the concerns mentioned in the weaknesses, there are additional questions as follows: 

- It introduces normal consistency term in the loss function and uses depth gradient as pseudo ground-truth. However, will it produce incorrect results due to LiDAR projection, ray-drop or at object edges? Is it more plausible to use 3D point clouds for normal estimation? In addition, a comparison of this loss term is missing in the ablation study.

- How to initialize Gaussian points for a dynamic scene? If all point clouds are aggregated and sampled, this may lead to unreasonable initialization of dynamic objects (especially with long trajectories). Furthermore, are different scenes with different sequence lengths and ranges initialized using the same number of points? And how about the final number of Gaussian points? 

- As can be seen in Figure 1, the results of 3D-GS are very poor, but the paper lacks details of its implementation and the analysis of its failure. Since 3D-GS could not be directly migrated to LiDAR-NVS, relevant explanations are necessary.

- Is it reasonable to model long trajectory motion with periodic vibration properties? It would be better to visualize the motion trajectories of Gaussian primitives to prove their temporal consistency.

### Soundness
3

### Presentation
3

### Contribution
2
