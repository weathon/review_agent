# DiffTrans: Differentiable Geometry-Materials Decomposition for Reconstructing Transparent Objects

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Reconstructing transparent objects from a set of multi-view images is a challenging task due to the complicated nature and indeterminate behavior of light propagation. Typical methods are primarily tailored to specific scenarios, such as objects following a uniform topology, exhibiting ideal transparency and surface specular reflections, or with only surface materials, which substantially constrains their practical applicability in real-world settings. In this work, we propose a differentiable rendering framework for transparent objects, dubbed \emph{DiffTrans}, which allows for efficient decomposition and reconstruction of the geometry and materials of transparent objects, thereby reconstructing transparent objects accurately in intricate scenes with diverse topology and complex texture. Specifically, we first utilize FlexiCubes with dilation and smoothness regularization as the iso-surface representation to reconstruct an initial geometry efficiently from the multi-view object silhouette. Meanwhile, we employ the environment light radiance field to recover the environment of the scene. Then we devise a recursive differentiable ray tracer to further optimize the geometry, index of refraction and absorption rate simultaneously in a unified and end-to-end manner, leading to high-quality reconstruction of transparent objects in intricate scenes. A prominent advantage of the designed ray tracer is that it can be implemented in CUDA, enabling a significantly reduced computational cost. Extensive experiments on multiple benchmarks demonstrate the superior reconstruction performance of our \emph{DiffTrans} compared with other methods, especially in intricate scenes involving transparent objects with diverse topology and complex texture. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes DiffTrans, a differentiable rendering framework for reconstructing the 3D geometry and material properties (index of refraction and absorption rate) of transparent objects from multi-view images. The method operates in three stages: 1) Geometry Initialization: Using FlexiCubes and multi-view object masks to reconstruct a coarse mesh, regularized for smoothness and completeness. 2) Environment Initialization: Recovering the surrounding scene lighting using a radiance field from the out-of-mask image regions. 3) Joint Optimization: A novel, efficient, recursive differentiable ray tracer implemented in OptiX/CUDA that jointly refines the geometry, IoR, and absorption rate in an end-to-end manner. The key advantage is the ability to handle transparent objects with complex geometry, enabling high-quality reconstruction and relighting. Experimental results and ablation studies demonstrate nice results compared with the state-of-the-art methods.

### Strengths
Basically, the explicit and joint optimization of absorption rate alongside geometry and IoR is a significant step beyond prior work, which often focused only on surface properties or ideal transparency. This directly addresses a major limitation in reconstructing real-world objects.

Overall, the work combines several advanced techniques (FlexiCubes for mesh initialization, a radiance field for environment, and a custom differentiable ray tracer) into a cohesive, progressive pipeline. The integration of a mesh-based differentiable ray tracer, as opposed to a volume-based or implicit one, is a distinct choice for this problem.

The design of the recursive ray tracer and the specific regularization terms (like the tone loss L_tone) are novel contributions tailored to the ill-posed nature of reconstructing absorptive transparent objects.

### Weaknesses
First of all, reconstructing transparent objects has always been a very challenging problem in both the vision and graphics communities. The method proposed in this paper takes another step forward in this difficult area. Although the results are quite good, I still have several questions. I am open to hearing the authors’ feedback.

(1) In fact, this method can be categorized as a mesh optimization approach rather than a neural radiance one. Therefore, I am curious why it is not compared with [Lyu et al. 2020]. In other words, the comparison in Table 1 seems quite unfair, as all other methods are NeRF-based, which are inherently inferior in reconstruction quality compared to mesh optimization approaches.

(2) The initialization seems to have a significant impact. As shown in Table 1, the optimization process itself does not lead to a very noticeable improvement. I would like to see a comparative visualization, such as a per-vertex heatmap showing the difference in Chamfer Distance / Hausdorff Distance before and after optimization, which would be more intuitive to check the effectiveness of optimization.

(3) IOR optimization has always been a difficult problem. I suggest that the authors analyze the recovery of the IOR under different refractive indices for the same object. As the IOR increases, the proportion of total internal reflection rises sharply, making IOR estimation more challenging. It is unclear whether the ray tracer can correctly handle such cases.

(4) Regarding the environment map, it would be helpful to include some results showing the recovery of the environment map.

(5) The geometric details could be better illustrated using a normal map, since the current white model rendering makes it hard to perceive geometric enhancement. From Fig. 4, for example, the eyes of the Monkey model are almost not recovered at all. Another useful metric would be the MAE between the predicted and ground-truth outgoing ray directions.

(6) In recursive ray tracing, it is not mentioned how the method handles cases of internal total reflection.

(7) The training time for a single object? Any failure cases? Moreover, given that the optimization involves a large number of variables, I wonder whether the entire optimization process remains stable.

### Questions
As discussed above.

### Soundness
3

### Presentation
4

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
This paper proposes DiffTrans, a framework for jointly reconstructing geometry and performing inverse rendering from multi-view images of transparent objects. It first initializes the geometry using the silhouette from the object mask with several regularization losses. Then, it refines the geometry and optimizes the refraction parameters using differentiable ray tracing. With the inverse rendering results, it supports applications such as relighting and demonstrates superior quality to baseline SOTA methods.

### Strengths
1. The proposed framework successfully solved the highly challenging transparent object reconstruction and inverse rendering task, while previous works struggled to handle them simultaneously.
2. The proposed differentiable rendering algorithm in Sec 3.3 is physically based and technically sound, following the physical laws of light transport.
3. The experimental results are significantly superior to baseline methods.

### Weaknesses
1. Related work and citations. This paper is an inverse and differentiable rendering paper, but Section 2 does not survey related work in these research directions. I suggest adding a paragraph for a comprehensive review of existing inverse/differentiable rendering and relighting methods.
2. Paper writing. A lot of necessary details and explanations are missing in the paper, making it hard to understand for people without a background in the related physics laws. Please refer to the "Questions" part for some specific confusions when I read the paper, and the authors should include these details/explanations in the revised paper.
3. Baseline selection. The paper selects NeRO as one of its baselines, but NeRO does not support refraction objects at all, leading to an unfair comparison. It would be better to choose baselines supporting transparent objects.
4. Geometry initialization. A comparison between the initialized geometry from the first stage and the refined geometry from the last stage should be included, so that we can know how differentiable rendering helps the geometry refinement.

Several example citations in differentiable and inverse rendering (both mesh-based and NeRF-based, including but not limited to):
```
@article{zhang2021nerfactor,
  title={Nerfactor: Neural factorization of shape and reflectance under an unknown illumination},
  author={Zhang, Xiuming and Srinivasan, Pratul P and Deng, Boyang and Debevec, Paul and Freeman, William T and Barron, Jonathan T},
  journal={ACM Transactions on Graphics (ToG)},
  volume={40},
  number={6},
  pages={1--18},
  year={2021},
  publisher={ACM New York, NY, USA}
}
@inproceedings{jin2023tensoir,
  title={Tensoir: Tensorial inverse rendering},
  author={Jin, Haian and Liu, Isabella and Xu, Peijia and Zhang, Xiaoshuai and Han, Songfang and Bi, Sai and Zhou, Xiaowei and Xu, Zexiang and Su, Hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={165--174},
  year={2023}
}
@inproceedings{gao2024relightable,
  title={Relightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing},
  author={Gao, Jian and Gu, Chun and Lin, Youtian and Li, Zhihao and Zhu, Hao and Cao, Xun and Zhang, Li and Yao, Yao},
  booktitle={European Conference on Computer Vision},
  pages={73--89},
  year={2024},
  organization={Springer}
}
@inproceedings{dai2025inverse,
  title={Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling},
  author={Dai, Yuxin and Wang, Qi and Zhu, Jingsen and Xi, Dianbing and Huo, Yuchi and Qian, Chen and He, Ying},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year = {2025},
  url = {https://openreview.net/forum?id=KEXoZxTwbr}
}
@article{son2024dmesh,
  title={DMesh: A Differentiable Mesh Representation},
  author={Son, Sanghyun and Gadelha, Matheus and Zhou, Yang and Xu, Zexiang and Lin, Ming C and Zhou, Yi},
  journal={arXiv preprint arXiv:2404.13445},
  year={2024}
}
@article{binninger2025tetweave,
  title={TetWeave: Isosurface Extraction using On-The-Fly Delaunay Tetrahedral Grids for Gradient-Based Mesh Optimization},
  author={Binninger, Alexandre and Wiersma, Ruben and Herholz, Philipp and Sorkine-Hornung, Olga},
  journal={ACM Transactions on Graphics (TOG)},
  volume={44},
  number={4},
  pages={1--19},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```
I generally like the paper's proposed method, but the current writing quality needs significant improvement.

### Questions
1. In Eq. 1, what are the points $x_i$? Are they mesh vertices, points sampled from the surface of the mesh, or points sampled randomly in the space? Besides, `\mathcal` is missing for the $\mathcal L$ symbol.
2. What are the optimizable refraction material parameters in the last stage (Refine phase)? Are they the IoR $\eta$ and the absorption rate $\mu_t(\mathbf{x})$? Is $\eta$ a single scalar while $\mu_t(\mathbf{x})$ a spatially-varying field?
3. In Eqs. 5 and 6, what are $\omega^\parallel_t$ and $\omega^\perp_t$? I guess they refer to the "parallel and perpendicular part" of the direction, but I don't know their specific physical meanings, and I think this should be explained.
4. For the environment initialization, did you handle the LDR-HDR issue? Since the environment NeRF is learned from LDR images, the lighting queried in the ray tracing stage will remain LDR. Using LDR values to compute the rendering equation may lead to overdark estimation, and also be vulnerable to imaging artifacts such as over-/under-exposure.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel multi-view image-based 3D reconstruction framework tailored for transparent objects. By adopting a two-stage design, the method achieves a more accurate geometric representation.

In Stage 1, the authors model an initial outer surface of the transparent object using object masks. To ensure stable convergence, several operations—including dilation—and regularization terms are introduced to constrain the mesh shape. Additionally, non-mask regions are leveraged to learn an environment field that captures scene appearance and lighting.

In Stage 2, a custom optical model is employed for joint optimization, enabling simultaneous recovery of the object’s material properties and refinement of the geometry and appearance from Stage 1.

Experimental results demonstrate that the proposed method achieves state-of-the-art (SOTA) performance.

The main contributions of this work are:

1) A two-stage reconstruction pipeline that produces a high-quality initialization robust to the challenges of transparency.

2) A novel rendering pipeline capable of simulating light transport through transparent materials.

3) A new real-world dataset specifically captured for transparent object reconstruction.

### Strengths
1) This paper implements a ray tracer for transparent materials based on OptiX and CUDA, which serves as a valuable contribution to the community.

2) To obtain a high-quality initial mesh suitable for ray tracing, the paper proposes a flexible cube-based modeling approach.

### Weaknesses
1) The amount of real-world test data is insufficient—only a single real captured object is used for evaluation. Moreover, the rendered results exhibit inaccurate material appearance, and the paper does not provide any geometric visualization (e.g., mesh or point cloud) of this real object. This raises concerns about the method’s practicality. In contrast, methods like NU-NeRF validate their approach on multiple real transparent objects, offering stronger empirical support.

2) The proposed method relies heavily on an accurate foreground mask for Stage 1 training. In the paper, masks for real data are obtained by combining SAM with manual refinement, which undermines the method’s applicability in fully automatic settings. Notably, prior work such as [1] also leverages vision models for transparent object reconstruction but avoids human intervention entirely, making it more scalable and practical. Also, you may cite this paper.


[1] TSGS: Improving Gaussian Splatting for Transparent Surface Reconstruction via Normal and De-lighting Priors

### Questions
1) In the ablation study in Figure 10, does “other reg” refer to the original regularization losses used in FlexibleCube?

2) How does the proposed method perform on the less obvious transparent objects in the NU-NeRF dataset? Would the presence of opaque objects inside transparent ones in the NU-NeRF dataset negatively impact—or even cause failure in—the reconstruction using the proposed method? The approach appears to be designed primarily for solid transparent objects, and may not handle internal occluders or complex layered transparency effectively.

3) Many structural details of transparent objects cannot be captured by silhouette masks alone. Although the paper suggests that the two-stage optimization can recover such missing structures, could the authors provide a direct geometric comparison between the results of Stage 1 and Stage 2 to demonstrate this improvement?

4) Could the authors provide a full 360-degree video rendering of the results?

### Soundness
3

### Presentation
2

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
This paper presents DiffTrans, a differentiable rendering framework for reconstructing transparent objects with complex geometry and internal texture from multi-view images. It jointly optimizes geometry and material properties—specifically the index of refraction (IoR) and absorption rate—in an end-to-end manner. The method initializes geometry using FlexiCubes and estimates illumination via an environment radiance field, then refines both using a recursive differentiable ray tracer implemented in CUDA/OptiX. Experiments on synthetic and real data demonstrate strong reconstruction and relighting performance.

### Strengths
The work tackles a highly challenging problem in differentiable rendering, enabling reconstruction and relighting of transparent objects with complex internal geometry and refractions. The decoupling of geometry and materials of transparent objects also presents a reasonable path for modeling light refraction and absorption within an end-to-end differentiable pipeline. The method is efficiently implemented in CUDA/OptiX with strong empirical performance on both synthetic and real-world data.

### Weaknesses
The framework relies on several simplifying assumptions, uniform refractive index, purely specular surfaces, and absorption-only materials, which could limit its applicability to more realistic transparent objects such as frosted, layered, or scattering materials.


The evaluation on real-world data is limited to a single “glass flower” scene, which does not sufficiently demonstrate robustness to real capture conditions such as noise, imperfect masks, or lighting variations. More diverse real-world examples and quantitative comparisons would strengthen the experimental section.

Although the recursive differentiable ray tracer is central to the method, its computational and memory efficiency are not analyzed in detail. Providing runtime comparisons or scalability analysis would help clarify its practical feasibility.

The method’s sensitivity to initialization (e.g., mask accuracy, environment estimation quality) is not thoroughly discussed. An ablation on initialization errors or noisy inputs would clarify robustness in real applications.

The authors could better articulate their conceptual distinction from prior works such as NeRRF, Nu-NeRF, and TransparentGS, to highlight their unique contributions more clearly. The authors should also discuss the conceptual differences from [1], which also highlights decoupling geometry and radiance for reconstructing semi-transparent surfaces, and may provide quantitative comparisons.

[1] Wu, Tianhao, et al. "αsurf: Implicit surface reconstruction for semi-transparent and thin objects with decoupled geometry and opacity." 2025 International Conference on 3D Vision (3DV). IEEE, 2025.

### Questions
Please refer to the above section for questions

### Soundness
3

### Presentation
3

### Contribution
3
