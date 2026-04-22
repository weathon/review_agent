# TraceFlow: Dynamic 3D Reconstruction of Specular Scenes Driven by Ray Tracing

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 6, 2, 4

## Abstract
We present TraceFlow, a novel framework for high-fidelity rendering of dynamic specular scenes by addressing two key challenges: precise reflection direction estimation and physically accurate reflection modeling. To achieve this, we propose a Residual Material-Augmented 2D Gaussian Splatting representation that models dynamic geometry and material properties, allowing accurate reflection ray computation. Furthermore, we introduce a Dynamic Environment Gaussian representation and a hybrid rendering pipeline that decomposes rendering into diffuse and specular components, enabling physically grounded specular synthesis via rasterization and ray tracing. Finally, we devise a coarse-to-fine training strategy to improve optimization stability and promote physically meaningful decomposition. Extensive experiments on dynamic scene benchmarks demonstrate that TraceFlow outperforms prior methods both quantitatively and qualitatively, producing sharper and more realistic specular reflections in complex dynamic environments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses dynamic 3D reconstruction of specular (reflective) scenes from monocular video. TraceFlow builds on 2D Gaussian Splatting (2DGS) because it provides exact surface normals needed for accurate reflection ray computation, unlike 3DGS, which requires approximations. The method has three components: (1) Residual Material-Augmented 2DGS with time-conditioned deformation networks and learnable material properties, (2) Dynamic Environment Gaussians for modeling time-varying illumination at higher resolution than environment maps, and (3) hybrid rendering that decomposes appearance into diffuse (rasterization) and specular (ray tracing) components. The method uses temporal-consistent normal supervision and coarse-to-fine training (diffuse->specular->joint). Experiments show a +0.74 PSNR improvement over prior work, with sharper specular reflections, though contributions are primarily incremental (combining existing techniques) and yield modest gains at 2.5× the computational cost.

### Strengths
- The paper articulates a genuine challenge in 3D reconstruction: handling dynamic specular surfaces and identifies the precise technical bottlenecks (inaccurate reflection directions from approximate normals and limited environment map resolution).
- Using 2DGS instead of 3DGS specifically to obtain exact surface normals is an elegant insight. This demonstrates that representation choice matters for downstream tasks: a simple but powerful idea that 2D surface primitives naturally provide the geometric quantities (normals) needed for physically-based rendering, while 3D volumetric primitives require approximations.
- The visual quality improvements are substantial and clearly visible in Figures 1 and 3, showing sharper, more realistic specular highlights compared to all baselines. The method achieves consistent improvements across multiple scenes (7/7 on NeRF-DS).
- The explicit diffuse/specular decomposition following PBR principles, combined with material properties (specular tint) and ray tracing, provides interpretability and physical correctness.
- The paper includes proper ablation studies (Table 3, Figure 4) that show each component's contribution, compare against both dynamic and static specular methods, and evaluate across multiple datasets.
- Most prior work handles either static specular OR dynamic diffuse scenes. Enabling dynamic specular reconstruction from monocular video has clear applications in AR/VR content creation and 4D scene capture.

### Weaknesses
- The paper does not introduce fundamentally new techniques but rather combines established methods: temporal deformation networks are standard in dynamic Gaussian splatting (Yang et al., 2023; Yang et al., 2024a), Dynamic Environment Gaussians directly adopt the approach from EnvGS (Xie et al., 2024), and diffuse/specular decomposition with hybrid rendering is common in PBR. The main contribution is "making these work together for dynamic scenes," which is incremental engineering rather than a novel algorithmic or representational advance.
- In Table 1, EnvGS (Xie et al., 2024) (which TraceFlow builds heavily upon) achieves only 20.21 PSNR on average, performing worse than even general dynamic methods like Deformable 3DGS (23.43 PSNR). This is surprising since EnvGS was specifically designed for specular reconstruction using environment Gaussians. The paper does not explain: (a) how EnvGS was adapted for dynamic scenes, (b) whether per-frame independent reconstruction was used (which would be unfair), or (c) why a specular-focused method performs so poorly when temporal modeling is the primary addition.
- Compared to SpectroMotion (Fan et al., 2024), the most relevant baseline for dynamic specular reconstruction, TraceFlow achieves +0.74 PSNR improvement (24.17->24.91) but requires 2.5x longer training time (1.1->2.8 hours, Table 4). While the improvement is consistent across scenes, the magnitude is modest for the added complexity.

### Questions
- How exactly was EnvGS (Xie et al., 2024) adapted for the dynamic setting in your experiments? Was it trained independently per frame, or did you implement temporal extensions? The performance gap between EnvGS (20.21 PSNR) and your method (24.91 PSNR) seems surprisingly large, given that you build heavily on their environment Gaussian approach.
- What happens when you use alternative monocular normal estimators or no external normal supervision at all (using only L_norm)? Table 3 shows L_tc-norm contributes 0.41 PSNR improvement (20.69->21.10), which is substantial relative to your +0.74 PSNR gain over SpectroMotion.
- Can you provide quantitative evidence that the learned diffuse/specular decomposition is physically meaningful beyond image quality metrics? For example, comparisons against ground truth in synthetic scenes, or demonstrations that the method enables relighting/material editing?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces TraceFlow, a framework for dynamic 3D reconstruction of specular scenes from monocular videos. It tackles challenges in reflection direction estimation and physically based reflection modeling by proposing a Residual Material-Augmented 2D Gaussian Splatting representation that jointly models dynamic geometry and material properties. A Dynamic Environment Gaussian further captures time-varying illumination, while a hybrid rasterization and ray tracing pipeline enables accurate decomposition of diffuse and specular components. A coarse-to-fine training strategy improves stability and ensures physically meaningful learning. Experiments on dynamic benchmarks show that TraceFlow achieves state-of-the-art rendering quality.

### Strengths
1.By integrating Dynamic Environment Gaussians and a hybrid rasterization–ray tracing pipeline, the method achieves realistic and physically consistent specular rendering.

2.The coarse-to-fine training strategy effectively stabilizes optimization and promotes meaningful separation of diffuse and specular components.

3.Extensive experiments on dynamic scene benchmarks demonstrate clear quantitative and qualitative improvements over state-of-the-art baselines.

### Weaknesses
1.The method integrates existing components, 2D Gaussian Splatting, dynamic environment modeling, and ray tracing, into a unified framework, showing limited conceptual innovation.

2.The computation of the specular term largely follows EnvGS and relies on a single reflection ray, which restricts its effectiveness in scenes with complex or glossy interactions.

3.The ablation study focuses mainly on temporal and geometric aspects, while the specular and ray-tracing components, central to the paper’s contribution, lack sufficient in-depth analysis.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
TraceFlow reconstructs and renders dynamic specular scenes from monocular videos using a ``material-augmented'' 2D Gaussian Splatting (2DGS) representation. The framework integrates rasterization for diffuse components, ray tracing for specular reflections, and a Dynamic Environment Gaussian model to handle lighting variations. A coarse-to-fine training scheme is employed to stabilize optimization, yielding improvements over existing baselines.

### Strengths
1. This work effectively leverages 2DGS-based normals together with priors from foundation models to enhance surface normal accuracy, leading to improved specular rendering quality. The motivation for this design choice is clear and well-formulated.
2. The explicit modeling of temporally varying environments and reflected-ray tracing contributes to more accurate reconstruction of reflective and dynamically illuminated scenes, improving overall quality.

### Weaknesses
1. The experimental evaluation could be improved for further validation.
For example, ablation studies are conducted on a single scene, and the performance gains on HyperNeRF are modest, raising questions about the method’s generalizability. The rendered normals also appear noisy, undermining the performance and reliability of the proposed method. Additional quantitative and qualitative experiments on surface normal accuracy and specular component modelling could be included to demonstrate the effectiveness of the proposed method. Furthermore, the reported training time is roughly twice that of comparable baselines such as SpectroMotion, suggesting suboptimal efficiency.

2. The claim of “physically grounded” rendering is not well supported. The proposed reflection model accounts only for mirror-like reflections and does not capture more general specular behaviors. In Figure 2, the shown “specular” components poorly align with the actual specular regions and lack physical interpretability, making the proposed method questionable. Moreover, the reflected rays interact solely with the Dynamic Environment Gaussians rather than modeling near-field interreflections within scene geometry, which limits realism in complex dynamic scenes.

3. Some implementation details could be included for enhancing reproducibility.
The paper omits key implementation details such as the exact architectures of the scene and environment networks, the input encodings, and the training hyperparameters. It is also unclear whether point densification or pruning is used during optimization, despite being a standard component of Gaussian-splat pipelines.

### Questions
1. Please see the weakness section.
2. Various typos, such as subscripts.
3. Is the s_tint a 3-channel color or just a scalar?
4. Should the normals in Fig.4 be pseudo-GT instead of GT?

### Soundness
2

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
4

### Summary
The paper proposes TraceFlow, a novel framework for dynamic, view-dependent 3D reconstruction of specular scenes from monocular videos. It introduces a residual material-augmented 2D Gaussian splatting to model dynamic geometry and temporally evolving materials with accurate reflection ray computation, a dynamic environment Gaussian representation paired with a hybrid rendering pipeline to decompose diffuse and specular components, and a coarse-to-fine training strategy.

### Strengths
### Strengths
1. Quantitative results outperform baselines.
2. Extending the environment Gaussian representation to handle dynamic specular scenes significantly enhances quality.

### Weaknesses
### Weaknesses
1. Qualitative results show limited improvement over baselines in most cases. For instance, in the second row of Figure 3, specular reflections in highlighted regions are comparable to SpectroMotion. Even in stronger cases (first row), while more details are captured than SpectroMotion, spurious artifacts absent in the ground truth are introduced.
2. Ablation studies are conducted only on the Plate scene; validation across all four qualitative scenes is recommended to strengthen generalizability claims. The current ablations do not sufficiently demonstrate component robustness across diverse scenarios.
3. The coarse-to-fine training is presented as an independent contribution, yet no experiments validate its effectiveness.

### Questions
### Questions
What are the underlying reasons for the large gap between rendered high-frequency specular details and ground truth?

### Soundness
2

### Presentation
2

### Contribution
2
